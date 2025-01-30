from dataclasses import dataclass


@dataclass
class BenchmarkConfig:
    model: str
    backend: str
    tokenizer: str
    ccu: int
    host: str
    port: str
    max_prompt: int
    min_completion: int
    max_completion: int
    min_cd: int
    max_cd: int
    time: int
