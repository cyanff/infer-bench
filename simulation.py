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
        conversations = get_conversations(5000)
        processed: List[List[SimulationTurn]] = []
        if self.cache:
            cached = self.cache.get(self.cache_key) or []
            # Only use the prompt and prompt_len from the cache
            for convo in cached:
                simulation_turns: List[SimulationTurn] = []
                for turn in convo:
                    simulation_turns.append(
                        SimulationTurn(
                            prompt=turn.prompt,
                            prompt_len=turn.prompt_len,
                            max_new_tokens=sample_gaussian_integer(
                                self.min_completion, self.max_completion
                            ),
                            cd=sample_gaussian_integer(self.min_cd, self.max_cd),
                        )
                    )
                processed.append(simulation_turns)

        if len(processed) > 0:
            print(f"Simulation turns cache hit! Len: {len(processed)}")

        offset = len(processed)
        for convo in tqdm(conversations[offset:], desc="Processing conversations..."):
            context: Queue[int] = Queue()
            turns: List[SimulationTurn] = []
            for turn in convo:
                # gpt turn, we want to get *all* the previous messages and use it as a prompt,
                # deleting tokens from the beginning it if it's pass max_prompt
                if turn.from_ == "gpt":
                    # get rid of extra tokens at the front of the queue
                    while context.qsize() > self.max_prompt:
                        context.get()

                    prompt = self.tokenizer.decode(list(context.queue))
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
                for tok in self.tokenizer.encode(turn.value):
                    context.put(tok)
            processed.append(turns)

        if self.cache:
            self.cache.set(self.cache_key, processed)

        return processed
