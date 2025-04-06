#!/usr/bin/env python3
"""
Generate a ShareGPT-style JSONL dataset from a source file containing conversations.
Each source conversation has a 'system' prompt in its first element of "conversations", e.g.:
{
  "conversations": [
    {"from": "system", "value": "...SYSTEM PROMPT TEXT..."},
    {"from": "user",   "value": "..."},
    ...
  ]
}

Key Points / Changes:
  • The default OpenRouter model is "meta-llama/llama-4-scout:groq".
  • We want to create "ShareGPT-like" lines with:
      {"conversations": [ 
         {"from": "system", "value": "<original system prompt>"},
         {"from": "user", "value": "<roleplayed user text>"},
         {"from": "assistant", "value": "<assistant text>"},
         ...
      ]}
  • The "assistant" portion uses the *outside dataset's system prompt* to guide the response
    (i.e. it uses the original system text from the dataset).
  • The "user" portion is generated with a special instruction to "emulate a user in a roleplay scenario."
  • The assistant portion responds "normally" to that user text.
  • We only keep system prompts from the dataset that have a token length in [500..1000] (using your chosen tokenizer).
  • Each conversation has `--turns_per_convo` user->assistant pairs.
  • We pick a random max token limit for each generation (within the specified range), pass it to OpenRouter,
    and also measure locally to ensure the final text is in the correct token range. (Retries up to 5.)
  • Output is JSONL lines, each line is just {"conversations": [...]} with no "id"/"model" keys.
  • Provide concurrency via ThreadPoolExecutor.
  • Provide CLI arguments for user control.

Dependencies:
  • openai
  • datasets
  • transformers

Example usage:
  python dataGen.py \
    --input_dataset /path/to/system_prompts.jsonl \
    --output_dataset /path/to/output_sharegpt.jsonl \
    --openrouter_api_key "YOUR_OPENROUTER_KEY" \
    --tokenizer_id "gpt2" \
    --max_workers 10 \
    --turns_per_convo 100

"""

import argparse
import json
import sys
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from openai import OpenAI
from datasets import load_dataset
from transformers import AutoTokenizer

# ------------------ CONFIGURATIONS ------------------ #

# OpenRouter endpoint for Chat Completions
OPENROUTER_BASE_URL = "https://api.groq.com/openai/v1"

# Default model: "meta-llama/llama-4-scout:groq"
OPENROUTER_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Ranges for user/assistant lengths
USER_MIN_TOKENS, USER_MAX_TOKENS = 8, 20
ASSIST_MIN_TOKENS, ASSIST_MAX_TOKENS = 25, 65

# Required system prompt length from dataset
MIN_SYSTEM_PROMPT_TOKENS = 500
MAX_SYSTEM_PROMPT_TOKENS = 1000

# Retry attempts if generation doesn't meet token-length constraints
MAX_RETRIES_PER_TURN = 5

# Maximum context window size in tokens
MAX_CONTEXT_TOKENS = 4000

# ------------------ FUNCTIONS ------------------ #

def openrouter_chat_completion(
    api_key: str,
    messages: list,
    max_tokens: int,
    temperature: float = 0.7
) -> str:
    """
    Make a request to OpenRouter's Chat Completions endpoint using the OpenAI library.
    Takes a full list of messages in OpenAI format and returns the assistant's textual response.
    """
    client = OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL
    )

    response = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        extra_headers={
            "HTTP-Referer": "https://github.com/", # Optional: For attribution
            "X-Title": "Data Generation Script"    # Optional: For attribution
        }
    )
    
    if not response.choices or not response.choices[0].message.content:
        raise ValueError("No completions returned or empty content")
        
    return response.choices[0].message.content


def count_tokens(tokenizer, text: str) -> int:
    """
    Count tokens in 'text' using a user-selected HF tokenizer.
    """
    return len(tokenizer.encode(text))


def extract_system_prompt(sample: dict) -> str:
    """
    Given a record like:
    {
      "conversations": [
        {"from": "system", "value": "..."},
        ...
      ]
    }
    Return the 'value' for the first 'system' item.
    """
    conv_list = sample.get("conversations", [])
    for c in conv_list:
        if c.get("from") == "system":
            return c.get("value", "")
    return ""


def prepare_context_window(tokenizer, conversation_history, system_content, max_tokens=MAX_CONTEXT_TOKENS):
    """
    Prepare a context window of messages that fits within max_tokens,
    always keeping the system prompt and trimming oldest pairs of messages if needed.
    
    Returns a list of messages in OpenAI format ready for API call.
    """
    messages = [{"role": "system", "content": system_content}]
    
    # Count system prompt tokens
    system_tokens = count_tokens(tokenizer, system_content)
    available_tokens = max_tokens - system_tokens
    
    # Extract all non-system messages from conversation history
    history_messages = []
    for item in conversation_history:
        if item["from"] == "system":
            continue
        role = "assistant" if item["from"] == "gpt" else item["from"]
        history_messages.append({"role": role, "content": item["value"]})
    
    # Start from most recent messages and work backwards
    reversed_messages = list(reversed(history_messages))
    included_messages = []
    tokens_used = 0
    
    for msg in reversed_messages:
        msg_tokens = count_tokens(tokenizer, msg["content"])
        # Add 4 tokens for message overhead (role, formatting)
        adjusted_tokens = msg_tokens + 4
        
        if tokens_used + adjusted_tokens <= available_tokens:
            included_messages.append(msg)
            tokens_used += adjusted_tokens
        else:
            # No more space, stop including messages
            break
    
    # Restore original order (oldest first)
    included_messages.reverse()
    
    # Add included messages to our context window
    messages.extend(included_messages)
    
    return messages


def generate_user_text(
    api_key: str, 
    tokenizer, 
    dataset_system_prompt: str, 
    conversation_history: list,
    min_toks: int, 
    max_toks: int
) -> str:
    """
    Generate a single user turn, with access to full conversation history.
    We'll pick a random limit within [min_toks..max_toks], pass to openrouter, measure locally,
    retry if out of range.
    """
    # Create system message with instructions
    system_content = (
        f"Original system prompt: {dataset_system_prompt}\n\n"
        "You are an imaginary user roleplaying in a scenario based on the above system prompt. "
        "You must produce exactly one user message that goes along with the conversation. "
        f"Desired token length: between {min_toks} and {max_toks}. "
        "Do not reveal you are an AI or a system. Output only the user's text."
    )
    
    # Prepare context-managed message history
    messages = prepare_context_window(tokenizer, conversation_history, system_content)
    
    # Add final instruction
    messages.append({"role": "user", "content": "Write the next user turn now."})

    for _ in range(MAX_RETRIES_PER_TURN):
        # Choose random max token limit to pass to OpenRouter
        target_max = random.randint(min_toks, max_toks)
        text = openrouter_chat_completion(
            api_key=api_key,
            messages=messages,
            max_tokens=target_max,
        )
        n_tokens = count_tokens(tokenizer, text)
        if min_toks <= n_tokens <= max_toks:
            return text

    # If not successful, truncate or pad as fallback
    tokens = tokenizer.encode(text)
    if len(tokens) < min_toks:
        while len(tokens) < min_toks:
            tokens += tokens[-1:]
    elif len(tokens) > max_toks:
        tokens = tokens[:max_toks]
    return tokenizer.decode(tokens)


def generate_assistant_text(
    api_key: str,
    tokenizer,
    dataset_system_prompt: str,
    conversation_history: list,
    min_toks: int,
    max_toks: int
) -> str:
    """
    Generate a single assistant turn, with access to full conversation history.
    We want a normal response that is between [min_toks..max_toks] tokens.
    """
    # Prepare context-managed message history
    messages = prepare_context_window(tokenizer, conversation_history, dataset_system_prompt)

    for _ in range(MAX_RETRIES_PER_TURN):
        target_max = random.randint(min_toks, max_toks)
        text = openrouter_chat_completion(
            api_key=api_key,
            messages=messages,
            max_tokens=target_max,
        )
        n_tokens = count_tokens(tokenizer, text)
        if min_toks <= n_tokens <= max_toks:
            return text

    # If not successful within retries, forcibly fix length
    tokens = tokenizer.encode(text)
    if len(tokens) < min_toks:
        while len(tokens) < min_toks:
            tokens += tokens[-1:]
    elif len(tokens) > max_toks:
        tokens = tokens[:max_toks]
    return tokenizer.decode(tokens)


def build_single_conversation(
    api_key: str,
    tokenizer,
    system_prompt: str,       # from dataset
    turns_per_convo: int
) -> dict:
    """
    Build a conversation with:
       - one "system" entry (the dataset system prompt)
       - for each turn in 1..turns_per_convo:
            "user" (roleplayed) -> "gpt" (normal response)
    Returns: {"conversations": [...]}
    """
    conversation_list = []
    # Add the dataset's system text as the system turn:
    conversation_list.append({"from": "system", "value": system_prompt})

    for turn_idx in range(turns_per_convo):
        # Generate user message with full history context
        user_text = generate_user_text(
            api_key=api_key,
            tokenizer=tokenizer,
            dataset_system_prompt=system_prompt,
            conversation_history=conversation_list.copy(),
            min_toks=USER_MIN_TOKENS,
            max_toks=USER_MAX_TOKENS,
        )
        conversation_list.append({"from": "user", "value": user_text})

        # Generate gpt response with full history context
        assist_text = generate_assistant_text(
            api_key=api_key,
            tokenizer=tokenizer,
            dataset_system_prompt=system_prompt,
            conversation_history=conversation_list.copy(),
            min_toks=ASSIST_MIN_TOKENS,
            max_toks=ASSIST_MAX_TOKENS,
        )
        conversation_list.append({"from": "gpt", "value": assist_text})
        
        # Validate conversation format after each turn
        if len(conversation_list) < 2:
            raise ValueError("Conversation missing required entries")
        if conversation_list[0]["from"] != "system":
            raise ValueError("First message must be from 'system'")
        for i in range(1, len(conversation_list)-1, 2):
            if conversation_list[i]["from"] != "user" or conversation_list[i+1]["from"] != "gpt":
                raise ValueError(f"Invalid conversation format at turn {i//2+1}: must be user->gpt")

    return {"conversations": conversation_list}


def main():
    parser = argparse.ArgumentParser(description="Generate ShareGPT-like dataset with roleplayed user and normal assistant.")
    parser.add_argument("--input_dataset", type=str, required=True,
                        help="Path to input dataset (JSON/JSONL) with 'conversations'.")
    parser.add_argument("--output_dataset", type=str, required=True,
                        help="Path to output JSONL file.")
    parser.add_argument("--openrouter_api_key", type=str, required=True,
                        help="OpenRouter API key.")
    parser.add_argument("--tokenizer_id", type=str, default="gpt2",
                        help="Hugging Face tokenizer ID (e.g. 'gpt2', 'EleutherAI/gpt-neox-20b', etc.).")
    parser.add_argument("--max_workers", type=int, default=4,
                        help="Concurrency threads.")
    parser.add_argument("--turns_per_convo", type=int, default=100,
                        help="Number of user->assistant turns per conversation.")
    parser.add_argument("--max_prompts", type=int, default=1000,
                        help="Maximum number of system prompts to use (default: 1000).")
    args = parser.parse_args()

    # 1) Load the dataset with HF
    ds = load_dataset("json", data_files=[args.input_dataset], split="train")

    # 2) Load tokenizer from HF
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)

    # 3) Filter system prompts by length [500..1000]
    valid_prompts = []
    for sample in ds:
        sprompt = extract_system_prompt(sample)
        n_toks = count_tokens(tokenizer, sprompt)
        if MIN_SYSTEM_PROMPT_TOKENS <= n_toks <= MAX_SYSTEM_PROMPT_TOKENS:
            valid_prompts.append(sprompt)

    print(f"[INFO] Found {len(valid_prompts)} system prompts in the range {MIN_SYSTEM_PROMPT_TOKENS}..{MAX_SYSTEM_PROMPT_TOKENS} tokens.")
    
    # Limit to max_prompts if there are more valid prompts than requested
    if len(valid_prompts) > args.max_prompts:
        print(f"[INFO] Randomly selecting {args.max_prompts} out of {len(valid_prompts)} valid prompts.")
        random.shuffle(valid_prompts)
        valid_prompts = valid_prompts[:args.max_prompts]
    
    if not valid_prompts:
        print("[WARNING] No valid system prompts found. Exiting.")
        sys.exit(0)

    print(f"[INFO] Will produce {args.turns_per_convo} pairs of user->gpt turns for each valid prompt.")
    print(f"[INFO] Output dataset: {args.output_dataset}")

    total_success = 0
    total_fail = 0
    progress_lock = threading.Lock()

    def worker(idx, system_prompt):
        try:
            conversation = build_single_conversation(
                api_key=args.openrouter_api_key,
                tokenizer=tokenizer,
                system_prompt=system_prompt,
                turns_per_convo=args.turns_per_convo,
            )
            
            # Final validation of the conversation structure
            convo_list = conversation["conversations"]
            if len(convo_list) != 1 + 2 * args.turns_per_convo:  # system + (user,gpt) pairs
                raise ValueError(f"Invalid conversation length: {len(convo_list)}")
            if convo_list[0]["from"] != "system":
                raise ValueError("First message must be from 'system'")
            
            # Check pattern: system, user, gpt, user, gpt...
            for i in range(1, len(convo_list), 2):
                if i+1 >= len(convo_list):
                    raise ValueError("Conversation ended with user message, missing gpt response")
                if convo_list[i]["from"] != "user" or convo_list[i+1]["from"] != "gpt":
                    raise ValueError(f"Invalid turn at position {i}: must be user->gpt")
            
            return idx, conversation
        except Exception as exc:
            return idx, f"ERROR: {exc}"

    # 4) Run concurrency
    with ThreadPoolExecutor(max_workers=args.max_workers) as exe, open(args.output_dataset, "w", encoding="utf-8") as outf:
        futures = [exe.submit(worker, i, sp) for i, sp in enumerate(valid_prompts)]
        
        for fut in as_completed(futures):
            idx, result = fut.result()
            
            with progress_lock:
                if isinstance(result, dict):
                    # Write conversation to output
                    outf.write(json.dumps(result, ensure_ascii=False) + "\n")
                    outf.flush()  # Ensure it's written immediately
                    total_success += 1
                    print(f"[PROGRESS] Conversation {idx+1}/{len(valid_prompts)} built successfully ({total_success} complete, {total_fail} failed)")
                else:
                    total_fail += 1
                    print(f"[ERROR] Failed to build conversation {idx+1}/{len(valid_prompts)}: {result}")

    # 5) Log summary
    print("=== Summary ===")
    print(f"Valid system prompts:  {len(valid_prompts)}")
    print(f"Conversations written: {total_success}")
    print(f"Failures:              {total_fail}")
    print("[DONE]")


if __name__ == "__main__":
    main()
