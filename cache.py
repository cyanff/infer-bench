from dataclasses import dataclass
import os
import pickle
from typing import Any
import lmdb


@dataclass
class Cache:
    path: str = "~/.cache/hyperbench"

    def __post_init__(self):
        os.makedirs(self.path, exist_ok=True)
        self.env = lmdb.open(self.path, map_size=2**20 * 1024 * 5)

    def get(self, k: str):
        with self.env.begin(write=False) as txn:
            res = txn.get(k.encode("utf-8"))
            if res is None:
                return None
            return pickle.loads(res)

    def set(self, k: str, v: Any):
        bytes = pickle.dumps(v)
        with self.env.begin(write=True) as txn:
            txn.put(k.encode("utf-8"), bytes)
