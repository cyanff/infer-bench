#!/usr/bin/env python3

import argparse
import json
import random
import uuid

def main():
    parser = argparse.ArgumentParser(description="Augment ShareGPT JSONL with UUID prefixes.")
    parser.add_argument("--input-file", required=True, help="Path to the original ShareGPT JSONL (72 conversations).")
    parser.add_argument("--output-file", required=True, help="Path to the output JSONL file.")
    parser.add_argument("--num-total", required=True, type=int, help="Total number of conversations to generate.")
    parser.add_argument("--seed", default=None, type=int, help="Random seed for reproducible sampling.")
    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)

    # Read all input conversations into memory
    original_conversations = []
    with open(args.input_file, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            conversation = json.loads(line)
            original_conversations.append(conversation)

    if not original_conversations:
        print("No conversations found in the input file.")
        return

    # Sanity check: we expect 72 conversations typically
    print(f"Loaded {len(original_conversations)} original conversation(s).")

    # Prepare the output file
    with open(args.output_file, "w", encoding="utf-8") as fout:
        for i in range(args.num_total):
            # Randomly sample one conversation (with replacement)
            convo = random.choice(original_conversations)

            # Deep copy (since we'll modify the conversation object)
            # but shallow copy is often enough for simple structures;
            # we'll do a full JSON dump/load if we want to be safest
            convo_copy = json.loads(json.dumps(convo))

            # Prepend the UUID to the first message from="system"
            if "conversations" in convo_copy and isinstance(convo_copy["conversations"], list):
                if len(convo_copy["conversations"]) > 0:
                    first_msg = convo_copy["conversations"][0]
                    if first_msg.get("from") == "system":
                        unique_id = str(uuid.uuid4())
                        original_value = first_msg.get("value", "")
                        first_msg["value"] = f"{unique_id} {original_value}"
                    else:
                        # If for some reason the conversation doesn't start with system, do nothing
                        # or you might want to find the "system" message. But the spec says first is system.
                        pass

            # Write the modified conversation to output
            fout.write(json.dumps(convo_copy, ensure_ascii=False))
            fout.write("\n")

    print(f"Successfully wrote {args.num_total} conversations to {args.output_file}.")


if __name__ == "__main__":
    main()
