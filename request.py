from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
import os
import time
from typing import Any, Dict, List, Optional
import warnings
from aiohttp import ClientSession
from utils import remove_prefix


@dataclass
class RequestResult:
    text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = field(default_factory=list)
    error: str = ""


class Adapter(ABC):
    @abstractmethod
    def extract_data(self, obj: Any) -> str: ...

    @abstractmethod
    def get_payload(self, prompt: str, max_new_tokens: int) -> Dict: ...

    @abstractmethod
    def get_headers(self) -> Dict: ...

    @property
    @abstractmethod
    def api_url(self) -> str: ...


# TODO: remove all adapters except openai


class SGLang(Adapter):
    def __init__(self, model: str, host: str, port: str):
        self.model = model
        self.host = host
        self.port = port

    def extract_data(self, obj: Any) -> str:
        return obj["choices"][0]["text"]

    def get_payload(self, prompt, max_new_tokens) -> Dict:
        return {
            "model": self.model,
            "prompt": prompt,
            "temperature": 0.0,
            "best_of": 1,
            "max_tokens": max_new_tokens,
            "stream": True,
            "ignore_eos": False,
        }

    def get_headers(self) -> Dict:
        return {}

    @property
    def api_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1/completions"


class OpenAI(Adapter):
    def __init__(self, model: str, host: str, port: str):
        self.model = model
        self.host = host
        self.port = port

    def extract_data(self, obj: Any) -> str:
        return obj["choices"][0]["text"]

    def get_payload(self, prompt: str, max_new_tokens: int) -> Dict:
        return {
            "model": self.model,
            "prompt": prompt,
            "temperature": 0.0,
            "best_of": 1,
            "max_tokens": max_new_tokens,
            "stream": True,
            "ignore_eos": False,
        }

    def get_headers(self) -> Dict:
        return {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            "Content-Type": "application/json",
        }

    @property
    def api_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1/completions"


async def request(
    session: ClientSession,
    prompt: str,
    max_new_tokens: int,
    backend_kind: str,
    model: str,
) -> RequestResult:
    adapter: Adapter
    match backend_kind:
        case "sglang":
            adapter = SGLang(model, "0.0.0.0", "30000")
        case "lmdeploy":
            adapter = OpenAI(model, "0.0.0.0", "8080")
        case _:
            raise Exception(f"Unknown backend kind: {backend_kind}")
    api_url = adapter.api_url
    payload = adapter.get_payload(prompt, max_new_tokens)
    headers = adapter.get_headers()

    out = RequestResult()
    st = time.perf_counter()
    ttft: Optional[float] = None
    text = ""
    last_token_time = st
    async with session.post(url=api_url, json=payload, headers=headers) as res:
        if res.status != 200:
            out.success = False
            out.error = res.reason or ""
            warnings.warn("Respone has a non 200 status.")
            return out

        latency = 0.0
        text = ""
        async for chunk in res.content:
            latency = time.perf_counter() - st

            chunk = chunk.strip()
            # Empty chunk, moving on
            if not chunk:
                continue

            chunk = remove_prefix(chunk.decode("utf-8"), "data: ")
            if chunk != "[DONE]":
                data = json.loads(chunk)

                extracted = adapter.extract_data(data)

                # NOTE: Some completion API might have a last
                # usage summary res without a token so we
                # want to check a token was generated
                if extracted:
                    now = time.perf_counter()
                    # First token
                    if ttft is None:
                        ttft = now - st
                        out.ttft = ttft

                    # Non first tokens
                    else:
                        out.itl.append(now - last_token_time)
                    text += extracted
                    last_token_time = time.perf_counter()

        out.text = text
        out.success = True
        out.latency = latency
    return out
