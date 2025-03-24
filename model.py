from dataclasses import dataclass
from typing import Optional


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
    served_model_name: Optional[str] = None
