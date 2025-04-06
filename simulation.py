"""
Simulate a user by making requests
"""

import asyncio
import time
import warnings
import aiohttp
from dataclasses import asdict, field, dataclass
from typing import List, Optional
import psutil
import os
from queue import Queue
from tqdm.asyncio import tqdm
import hashlib

from request import request
from utils import sample_gaussian_integer, set_fd_limit
from data import get_conversations
from cache import Cache
from transformers import PreTrainedTokenizerBase


@dataclass
class SimulationTurn:
    prompt: str
    prompt_len: int
    max_new_tokens: int
    cd: float


@dataclass
class SimulationTurnResult:
    completion: str
    prompt: str = ""
    prompt_len: int = 0
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0
    itl: List[float] = field(default_factory=list)
    error: str = ""


@dataclass
class SimulationResult:
    duration: float
    turn_results: List[SimulationTurnResult]


def truncate_context(
    context: Queue[int],
    system_tokens: List[int],
    max_prompt: int,
    block_size: int = 800,
) -> None:
    """
    Truncate tokens from the front in blocks of `block_size` while
    the total number of tokens (system + context) exceeds `max_prompt`.
    The system prompt tokens are never removed.
    """
    while len(system_tokens) + context.qsize() > max_prompt:
        removed_count = 0
        # Pop up to block_size tokens from the context queue
        while removed_count < block_size and not context.empty():
            context.get()
            removed_count += 1


@dataclass
class Simulation:
    model: str
    tokenizer: PreTrainedTokenizerBase
    tokenizer_id: str
    backend_kind: str
    cache: Optional[Cache]
    ccu: int = 100
    timelimit: int = 30
    min_cd: int = 7
    max_cd: int = 14
    min_completion: int = 130
    max_completion: int = 250
    max_prompt: int = 5000
    served_model_name: Optional[str] = None
    dataset_file: Optional[str] = None
    enable_warmup: bool = False

    def __post_init__(self):
        # NEW: Make cache key unique to dataset (safe even if file is missing)
        if self.dataset_file:
            base = os.path.basename(self.dataset_file)
            self.cache_key = f"simulation-turn::{base}"
        else:
            self.cache_key = "simulation-turn"
        self.turn_results: List[SimulationTurnResult] = []
        set_fd_limit()

    async def boot(self) -> SimulationResult:
        """
        By default a connector pools 100 tcp connections.
        If ccu is large, raise the FD limits to avoid queueing requests.
        """
        connector = aiohttp.TCPConnector(
            limit=32768,
            limit_per_host=0,
            force_close=False,
            enable_cleanup_closed=True,
            keepalive_timeout=30.0,
        )

        turns_list = self._prepare()
        turns_list_idx = 0
        user_tasks = set()
        self.start_time = time.perf_counter()

        async with aiohttp.ClientSession(connector=connector) as session:
            # Keep spawning new "users" until we hit timelimit or exhaust data
            while time.perf_counter() - self.start_time < self.timelimit:
                if turns_list_idx == len(turns_list):
                    warnings.warn(
                        "Ran out of conversations to simulate! Ending benchmark early."
                    )
                    break
                while len(user_tasks) < self.ccu:
                    if turns_list_idx == len(turns_list):
                        break
                    turns = turns_list[turns_list_idx]
                    task = asyncio.create_task(self._simulate_user(session, turns))
                    turns_list_idx += 1
                    user_tasks.add(task)

                done, _ = await asyncio.wait(
                    user_tasks, return_when=asyncio.FIRST_COMPLETED
                )
                user_tasks.difference_update(done)

        end_time = time.perf_counter()

        for task in user_tasks:
            if not task.done():
                task.cancel()

        await asyncio.gather(*user_tasks, return_exceptions=True)

        duration = end_time - self.start_time
        print(f"Test ran for {duration}s.")
        return SimulationResult(duration, self.turn_results)

    async def _simulate_user(self, session, turns: List[SimulationTurn]):
        """
        Loop over the prepared turns for a single conversation
        """
        for i, t in enumerate(turns):
            now = time.perf_counter()
            if (now - self.start_time) >= self.timelimit:
                print(f"Timelimit reached ({now - self.start_time:.2f}s). Ending user.")
                break

            prompt = t.prompt
            prompt_len = t.prompt_len
            max_new_tokens = t.max_new_tokens
            cd = t.cd

            # Wait for cooldown except on the first request
            if i > 0:
                await asyncio.sleep(cd)

            # NEW: Print out the prompt before sending to VLLM
            print("\n=== Prompt to VLLM ===")
            print(prompt)
            print("======================\n")

            # Make the request
            res = await request(
                session=session,
                model=self.model,
                backend_kind=self.backend_kind,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                served_model_name=self.served_model_name or self.model,
            )
            
            # Skip adding first turn result if warmup is enabled
            if not (self.enable_warmup and i == 0):
                self.turn_results.append(
                    SimulationTurnResult(
                        completion=res.text,
                        prompt=prompt,
                        prompt_len=prompt_len,
                        success=res.success,
                        latency=res.latency,
                        ttft=res.ttft,
                        itl=res.itl,
                        error=res.error,
                    )
                )

    def _prepare(self) -> List[List[SimulationTurn]]:
        """
        Build a list of conversation "turns" from the dataset, with each GPT turn
        forming a SimulationTurn to be run in _simulate_user().
        """
        import gc
        import sys
        import time

        start_time = time.time()
        print(f"[0.00s] Starting simulation preparation")

        # Print tokenizer info for debugging
        try:
            print(
                f"[{time.time() - start_time:.2f}s] Tokenizer: "
                f"{self.tokenizer.__class__.__name__}, model_max_length: {self.tokenizer.model_max_length}"
            )
        except Exception as e:
            print(f"[{time.time() - start_time:.2f}s] Failed to get tokenizer info: {str(e)}")

        # Fetch 1000 conversations from data
        print(f"[{time.time() - start_time:.2f}s] Getting conversations")
        conversations = get_conversations(1000, dataset_file=self.dataset_file)
        print(f"[{time.time() - start_time:.2f}s] Got {len(conversations)} conversations")

        processed: List[List[SimulationTurn]] = []

        # Read from cache if available
        print(f"[{time.time() - start_time:.2f}s] Checking cache")
        if self.cache:
            print(f"[{time.time() - start_time:.2f}s] Reading from cache key: {self.cache_key}")
            cached = self.cache.get(self.cache_key) or []
            cached_len = len(cached)
            print(f"[{time.time() - start_time:.2f}s] Cache read, found {cached_len} items")

            if cached_len > 0:
                # Convert cached data to SimulationTurn
                for i, convo_turns in enumerate(cached):
                    if i % 100 == 0 and i > 0:
                        print(f"[{time.time() - start_time:.2f}s] Processed {i}/{cached_len} cached convos")
                    sim_turns: List[SimulationTurn] = []
                    for turn in convo_turns:
                        try:
                            max_tokens = sample_gaussian_integer(
                                self.min_completion, self.max_completion
                            )
                            cooldown = sample_gaussian_integer(self.min_cd, self.max_cd)
                            sim_turns.append(
                                SimulationTurn(
                                    prompt=turn.prompt,
                                    prompt_len=turn.prompt_len,
                                    max_new_tokens=max_tokens,
                                    cd=cooldown,
                                )
                            )
                        except Exception as e:
                            print(
                                f"[{time.time() - start_time:.2f}s] Error in cached turn: {str(e)}"
                            )
                    processed.append(sim_turns)

                print(f"[{time.time() - start_time:.2f}s] Simulation turns cache hit! Len: {len(processed)}")

        offset = len(processed)
        print(f"[{time.time() - start_time:.2f}s] Starting to process {len(conversations) - offset} new conversations")

        # Force GC before heavy processing
        gc.collect()

        conversation_count = len(conversations[offset:])
        for i, convo in enumerate(
            tqdm(conversations[offset:], desc="Processing conversations...")
        ):
            if i % 10 == 0:
                # Show memory usage
                process = psutil.Process(os.getpid())
                mem_info = process.memory_info()
                mem_usage_mb = mem_info.rss / (1024 * 1024)
                print(
                    f"[{time.time() - start_time:.2f}s] Processing conversation {i}/{conversation_count}, "
                    f"mem usage: {mem_usage_mb:.2f} MB"
                )
                if i % 100 == 0 and i > 0:
                    gc.collect()

            try:
                # NEW/CHANGED: Keep system tokens separate so they're never truncated
                system_tokens: List[int] = []
                context: Queue[int] = Queue()
                simulation_turns: List[SimulationTurn] = []

                for j, turn in enumerate(convo):
                    # Encode the new turn
                    encode_start = time.time()
                    tokens = self.tokenizer.encode(turn.value)
                    encode_time = time.time() - encode_start

                    if encode_time > 1.0:
                        print(
                            f"[{time.time() - start_time:.2f}s] Slow encode: "
                            f"{encode_time:.2f}s for text length {len(turn.value)}"
                        )

                    # If it's the very first turn and from_ == system, treat it as "system prompt" we never drop
                    if j == 0 and turn.from_ == "system":
                        system_tokens.extend(tokens)
                        continue

                    # Otherwise, put tokens in the main context
                    for tok in tokens:
                        context.put(tok)

                    # If this is a GPT turn, do the decode => produce a SimulationTurn
                    if turn.from_ == "gpt":
                        # NEW/CHANGED: Truncate in blocks of 800 tokens, ignoring system prompt
                        truncate_context(context, system_tokens, self.max_prompt, block_size=800)

                        # Now decode combined system + context
                        queue_list = list(context.queue)
                        prompt_decode_start = time.time()
                        prompt_str = self.tokenizer.decode(system_tokens + queue_list)
                        decode_time = time.time() - prompt_decode_start

                        if decode_time > 1.0:
                            print(
                                f"[{time.time() - start_time:.2f}s] Slow decode: "
                                f"{decode_time:.2f}s for {len(system_tokens) + len(queue_list)} tokens"
                            )

                        sim_turn = SimulationTurn(
                            prompt=prompt_str,
                            prompt_len=len(system_tokens) + context.qsize(),
                            max_new_tokens=sample_gaussian_integer(
                                self.min_completion, self.max_completion
                            ),
                            cd=sample_gaussian_integer(self.min_cd, self.max_cd),
                        )
                        simulation_turns.append(sim_turn)

                processed.append(simulation_turns)

            except Exception as e:
                print(f"[{time.time() - start_time:.2f}s] Error processing conversation {i}: {str(e)}")

        print(
            f"[{time.time() - start_time:.2f}s] Processing complete, "
            f"total simulation turns: {len(processed)}"
        )

        # If new data was processed, save back to cache
        if self.cache and len(processed) > offset:
            print(f"[{time.time() - start_time:.2f}s] Saving to cache")
            try:
                self.cache.set(self.cache_key, processed)
                print(f"[{time.time() - start_time:.2f}s] Cache save complete")
            except Exception as e:
                print(f"[{time.time() - start_time:.2f}s] Failed to save to cache: {str(e)}")

        return processed