"""
Load and process data
"""

from dataclasses import dataclass
import json
import os
from typing import List, Literal
from utils import download_and_cache_file


@dataclass
class Message:
    from_: Literal["system", "human", "gpt"]
    value: str


def get_conversations(num_convos: int) -> List[List[Message]]:
    """
    Retrieves a specified number of conversations from a cached dataset of chat logs.
    Args: num_convos (int): The number of conversations to retrieve.
    Returns: List[List[Message]]: A list of conversations, where each conversation is a list of `Message` objects.
    """

    def load_jsonl(file_path):
        with open(file_path, "r") as file:
            return [json.loads(line) for line in file]

    data_url = "https://huggingface.co/datasets/anthracite-org/c2_logs_16k_llama_v1.1/resolve/main/c2_deduped_16k_llama3_tok_deanon_dsclean.jsonl"
    dataset_path = "/tmp/c2.jsonl"
    if not os.path.isfile("/tmp/c2.jsonl"):
        dataset_path = download_and_cache_file(data_url, "/tmp/c2.jsonl")

    ds = load_jsonl(dataset_path)
    filtered: List[List[Message]] = []
    for _, d in enumerate(ds):

        def is_alternating(conversation):
            if len(conversation) < 2:
                return True

            skip_first = conversation[0]["from"] == "system"

            expected_roles = ["human", "gpt"]
            for i, turn in enumerate(conversation):
                if skip_first and i == 0:
                    continue
                if turn["from"] != expected_roles[(i - int(skip_first)) % 2]:
                    return False
            return True

        if is_alternating(d):
            convo = d["conversations"]
            messages = [Message(from_=t["from"], value=t["value"]) for t in convo]
            filtered.append(messages)
        if len(filtered) == num_convos:
            break
    return filtered
