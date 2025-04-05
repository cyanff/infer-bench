import os
import json
import argparse
import sys
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="/tmp/c2.jsonl",
                        help="Path to the original dataset JSONL")
    parser.add_argument("--output", type=str, default="/tmp/c2_filtered.jsonl",
                        help="Where to write the filtered dataset JSONL")
    parser.add_argument("--tokenizer", type=str, default="facebook/galactica-125m",
                        help="Name/path of tokenizer used to measure system prompt length")
    parser.add_argument("--min-turns", type=int, default=8,
                        help="Conversation must have at least this many total turns")
    parser.add_argument("--max-system-tokens", type=int, default=1000,
                        help="System prompt must have fewer than this many tokens")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"Filtering from {args.input} -> {args.output}")
    pass_count = 0
    total_count = 0

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:

        for line in fin:
            total_count += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            # We expect 'conversations' to be a list of turns
            convo = item.get("conversations", [])
            if len(convo) < args.min_turns:
                # Not enough turns; skip
                continue

            # We also need a first turn from system that is < max-system-tokens
            first_turn = convo[0]
            if first_turn["from"] != "system":
                continue

            # Count tokens in system prompt
            system_prompt = first_turn["value"]
            tok_count = len(tokenizer.encode(system_prompt))

            if tok_count >= args.max_system_tokens:
                # System prompt too big; skip
                continue

            # If we reached here, this conversation meets our criteria
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            pass_count += 1

    print(f"Done. {pass_count} out of {total_count} conversations passed the filter.")

if __name__ == "__main__":
    main()
