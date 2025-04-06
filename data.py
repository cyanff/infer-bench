"""
Load and process data
"""

from dataclasses import dataclass
import json
import os
from typing import List, Literal, Optional, Union
from utils import download_and_cache_file


@dataclass
class Message:
    """
    Represents a message in a conversation with normalized role names.
    The from_ field is normalized to standard values even if the source uses variants.
    """

    from_: Literal["system", "human", "gpt"]
    value: str


def get_conversations(
    num_convos: int, dataset_file: Optional[str] = None
) -> List[List[Message]]:
    """
    Retrieves a specified number of conversations from a dataset (local or remote).
    Args:
        num_convos (int): Number of conversations to return
        dataset_file (str, optional): Custom dataset path. If not provided, use HF dataset.
    Returns:
        List[List[Message]]: List of conversations, each a list of Message objects.
    """
    import time

    start_time = time.time()
    print(f"[{0:.2f}s] Starting to get {num_convos} conversations")

    def load_jsonl(file_path: str):
        try:
            print(f"[{time.time() - start_time:.2f}s] Loading JSONL from {file_path}")
            file_size = os.path.getsize(file_path)
            print(
                f"[{time.time() - start_time:.2f}s] File size: {file_size / (1024 * 1024):.2f} MB"
            )

            lines = []
            with open(file_path, "r", encoding="utf-8") as file:
                for i, line in enumerate(file):
                    try:
                        lines.append(json.loads(line))
                        if i % 1000 == 0 and i > 0:
                            print(f"[{time.time() - start_time:.2f}s] Loaded {i} lines")
                    except json.JSONDecodeError as e:
                        print(
                            f"[{time.time() - start_time:.2f}s] JSON decode error at line {i}: {str(e)}"
                        )
            print(f"[{time.time() - start_time:.2f}s] Loaded {len(lines)} lines total")
            return lines
        except Exception as e:
            print(f"[{time.time() - start_time:.2f}s] Failed to load JSONL: {str(e)}")
            raise

    # Decide on source file
    if dataset_file and os.path.isfile(dataset_file):
        dataset_path = dataset_file
        print(
            f"[{time.time() - start_time:.2f}s] Using user-specified dataset at {dataset_path}"
        )
    else:
        data_url = "https://huggingface.co/datasets/anthracite-org/c2_logs_16k_llama_v1.1/resolve/main/c2_deduped_16k_llama3_tok_deanon_dsclean.jsonl"
        dataset_path = "/tmp/c2.jsonl"
        if not os.path.isfile(dataset_path):
            print(
                f"[{time.time() - start_time:.2f}s] Cache file not found, downloading from {data_url}"
            )
            dataset_path = download_and_cache_file(data_url, dataset_path)
        else:
            print(
                f"[{time.time() - start_time:.2f}s] Using cached dataset at {dataset_path}"
            )

    print(f"[{time.time() - start_time:.2f}s] Loading conversation dataset")
    ds = load_jsonl(dataset_path)
    print(f"[{time.time() - start_time:.2f}s] Dataset loaded, filtering conversations")

    filtered: List[List[Message]] = []
    skipped = 0

    def normalize_role(role: str) -> str:
        """
        Normalize different role names to our standard format.
        """
        if role.lower() in ["user", "human"]:
            return "human"
        elif role.lower() in ["assistant", "gpt"]:
            return "gpt"
        elif role.lower() == "system":
            return "system"
        else:
            return role  # Return as-is if unknown

    def is_alternating(conversation):
        """
        Checks if the conversation follows the expected pattern:
        optional system message followed by alternating human/gpt messages.
        Also handles variations like "user" instead of "human" and "assistant" instead of "gpt".
        """
        if len(conversation) < 2:
            return True

        # Map role variations to our standard roles
        normalized_roles = [normalize_role(turn["from"]) for turn in conversation]

        skip_first = normalized_roles[0] == "system"
        expected_roles = ["human", "gpt"]

        for i, role in enumerate(normalized_roles):
            if skip_first and i == 0:
                continue
            if role != expected_roles[(i - int(skip_first)) % 2]:
                return False
        return True

    for i, d in enumerate(ds):
        try:
            if i % 1000 == 0:
                print(
                    f"[{time.time() - start_time:.2f}s] Processing conversation {i}/{len(ds)} ({len(filtered)} accepted)"
                )

            if "conversations" not in d:
                skipped += 1
                continue

            if is_alternating(d["conversations"]):
                convo = d["conversations"]
                # Normalize role names when creating Message objects
                messages = [
                    Message(from_=normalize_role(t["from"]), value=t["value"])  # type: ignore
                    for t in convo
                ]  # type: ignore
                filtered.append(messages)
            else:
                skipped += 1

            if len(filtered) == num_convos:
                print(
                    f"[{time.time() - start_time:.2f}s] Got requested {num_convos} conversations, stopping"
                )
                break

        except Exception as e:
            print(
                f"[{time.time() - start_time:.2f}s] Error processing conversation {i}: {str(e)}"
            )
            skipped += 1

    print(
        f"[{time.time() - start_time:.2f}s] Filtering complete: {len(filtered)} conversations accepted, {skipped} skipped"
    )
    return filtered
