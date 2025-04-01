"""
Simulate a user by making requests
"""

import asyncio
from dataclasses import asdict, field, dataclass, astuple
from typing import List, Optional
import time
import warnings
import aiohttp
from queue import Queue
import os
import psutil
from request import request
from utils import sample_gaussian_integer, set_fd_limit
from data import get_conversations
from cache import Cache
from transformers import PreTrainedTokenizerBase
from tqdm.asyncio import tqdm


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
    max_prompt: int = 2048
    served_model_name: Optional[str] = None

    def __post_init__(self):
        self.cache_key = "simulation-turn"
        self.turn_results: List[SimulationTurnResult] = []
        set_fd_limit()

    async def boot(self) -> SimulationResult:
        """
        By default a connector pools 100 tcp connections.
        Depending on --ccu and the cooldown between each request,
        this might be too low, we'll run out of tcp connections,
        and requests will be queued, leading to inaccurate benchmarks.
        So we just set the limit to some large number to fix this.
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
        async with aiohttp.ClientSession(connector=connector) as session:
            self.start_time = time.perf_counter()

            # Constantly spawn more simulated users when we're under the limit
            while time.perf_counter() - self.start_time < self.timelimit:
                if turns_list_idx == len(turns_list):
                    warnings.warn(
                        "Ran out of converesations to simulate with! Terminating benchmark early."
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
        for i, t in enumerate(turns):
            now = time.perf_counter()
            if self.timelimit <= now - self.start_time:
                print(f"Time passed = {now - self.start_time}")
                print("Timelimit hit, returning")
                break
            prompt, prompt_len, max_new_tokens, cd = astuple(t)

            # Wait for cd before continuing except for the first request
            if i > 0:
                await asyncio.sleep(cd)
            res = await request(
                session=session,
                model=self.model,
                backend_kind=self.backend_kind,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                served_model_name=self.served_model_name or self.model,
            )
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
        pass

    def _prepare(self) -> List[List[SimulationTurn]]:
        import time
        import gc
        import sys
        
        start_time = time.time()
        print(f"[{0:.2f}s] Starting simulation preparation")
        
        # Try to print tokenizer info to debug
        try:
            print(f"[{time.time() - start_time:.2f}s] Tokenizer: {self.tokenizer.__class__.__name__}, model_max_length: {self.tokenizer.model_max_length}")
        except Exception as e:
            print(f"[{time.time() - start_time:.2f}s] Failed to get tokenizer info: {str(e)}")
            
        print(f"[{time.time() - start_time:.2f}s] Getting conversations")
        conversations = get_conversations(1000)  # Reduced from 5000 to 1000 for testing
        print(f"[{time.time() - start_time:.2f}s] Got {len(conversations)} conversations")
        
        processed: List[List[SimulationTurn]] = []
        
        print(f"[{time.time() - start_time:.2f}s] Checking cache")
        if self.cache:
            print(f"[{time.time() - start_time:.2f}s] Reading from cache key: {self.cache_key}")
            cached = self.cache.get(self.cache_key) or []
            cached_len = len(cached)
            print(f"[{time.time() - start_time:.2f}s] Cache read, found {cached_len} items")
            
            # Only use the prompt and prompt_len from the cache
            for i, convo in enumerate(cached):
                if i % 100 == 0 and i > 0:
                    print(f"[{time.time() - start_time:.2f}s] Processed {i}/{cached_len} cached conversations")
                    
                simulation_turns: List[SimulationTurn] = []
                for turn in convo:
                    try:
                        # Check if we can generate the values properly
                        max_tokens = sample_gaussian_integer(
                            self.min_completion, self.max_completion
                        )
                        cooldown = sample_gaussian_integer(self.min_cd, self.max_cd)
                        
                        simulation_turns.append(
                            SimulationTurn(
                                prompt=turn.prompt,
                                prompt_len=turn.prompt_len,
                                max_new_tokens=max_tokens,
                                cd=cooldown,
                            )
                        )
                    except Exception as e:
                        print(f"[{time.time() - start_time:.2f}s] Error with cached turn: {str(e)}")
                        
                processed.append(simulation_turns)

        if len(processed) > 0:
            print(f"[{time.time() - start_time:.2f}s] Simulation turns cache hit! Len: {len(processed)}")

        offset = len(processed)
        print(f"[{time.time() - start_time:.2f}s] Starting to process {len(conversations) - offset} new conversations")
        
        # Force garbage collection before intensive processing
        gc.collect()
        
        conversation_count = len(conversations[offset:])
        for i, convo in enumerate(tqdm(conversations[offset:], desc="Processing conversations...")):
            if i % 10 == 0:
                # Print memory usage
                process = psutil.Process(os.getpid())
                mem_info = process.memory_info()
                mem_usage_mb = mem_info.rss / (1024 * 1024)
                print(f"[{time.time() - start_time:.2f}s] Processing conversation {i}/{conversation_count}, mem usage: {mem_usage_mb:.2f} MB")
                
                # Force garbage collection periodically
                if i % 100 == 0 and i > 0:
                    gc.collect()
                
            try:
                context: Queue[int] = Queue()
                turns: List[SimulationTurn] = []
                
                for j, turn in enumerate(convo):
                    try:
                        # gpt turn, we want to get *all* the previous messages and use it as a prompt,
                        # deleting tokens from the beginning it if it's pass max_prompt
                        if turn.from_ == "gpt":
                            context_size = context.qsize()
                            
                            # Debug extremely large contexts
                            if context_size > 10000:
                                print(f"[{time.time() - start_time:.2f}s] Very large context: {context_size} tokens")
                                
                            # get rid of extra tokens at the front of the queue
                            tokens_removed = 0
                            while context.qsize() > self.max_prompt:
                                context.get()
                                tokens_removed += 1
                                
                            if tokens_removed > 0 and tokens_removed % 1000 == 0:
                                print(f"[{time.time() - start_time:.2f}s] Removed {tokens_removed} tokens from context")

                            # This is potentially slow for large contexts - convert queue to list
                            queue_list = list(context.queue)
                            
                            # Debug token decoding
                            decode_start = time.time()
                            prompt = self.tokenizer.decode(queue_list)
                            decode_time = time.time() - decode_start
                            
                            # Log slow decodes
                            if decode_time > 1.0:
                                print(f"[{time.time() - start_time:.2f}s] Slow decode: {decode_time:.2f}s for {len(queue_list)} tokens")
                            
                            sim_turn = SimulationTurn(
                                prompt=prompt,
                                prompt_len=context.qsize(),
                                max_new_tokens=sample_gaussian_integer(
                                    self.min_completion, self.max_completion
                                ),
                                cd=sample_gaussian_integer(self.min_cd, self.max_cd),
                            )
                            turns.append(sim_turn)

                        # Add new tokens to the context
                        encode_start = time.time()
                        tokens = self.tokenizer.encode(turn.value)
                        encode_time = time.time() - encode_start
                        
                        # Log slow encodes
                        if encode_time > 1.0:
                            print(f"[{time.time() - start_time:.2f}s] Slow encode: {encode_time:.2f}s for text length {len(turn.value)}")
                            
                        for tok in tokens:
                            context.put(tok)
                    except Exception as e:
                        print(f"[{time.time() - start_time:.2f}s] Error processing turn {j} in conversation {i}: {str(e)}")
                
                processed.append(turns)
                
            except Exception as e:
                print(f"[{time.time() - start_time:.2f}s] Error processing conversation {i}: {str(e)}")

        print(f"[{time.time() - start_time:.2f}s] Processing complete, total simulation turns: {len(processed)}")
        
        if self.cache and len(processed) > offset:
            print(f"[{time.time() - start_time:.2f}s] Saving to cache")
            try:
                self.cache.set(self.cache_key, processed)
                print(f"[{time.time() - start_time:.2f}s] Cache save complete")
            except Exception as e:
                print(f"[{time.time() - start_time:.2f}s] Failed to save to cache: {str(e)}")

        return processed
