import json
import os
from pathlib import Path
from transformers import AutoTokenizer

DATA_DIR = "/opt/data/private/xrd2c_v2/xrd2c/ShareGPT-Chinese-English-90K/sharegpt_jsonl"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts.jsonl")
MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
MAX_TOKEN_LEN = 32768
MIN_TOKEN_LEN = 10

SOURCE_FILES = [
    "common_zh_70k.jsonl",
    "common_en_70k.jsonl",
    "computer_zh_26k.jsonl",
    "computer_en_26k.jsonl",
    "unknow_zh_38k.jsonl",
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)

prompts = []
skipped_short = 0
skipped_empty = 0

for filename in SOURCE_FILES:
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"Skip missing: {filename}")
        continue
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            conv = item.get("conversation", [])
            if not conv or "human" not in conv[0]:
                skipped_empty += 1
                continue
            human_text = conv[0]["human"]
            if not human_text or not human_text.strip():
                skipped_empty += 1
                continue
            token_ids = tokenizer.encode(human_text)
            if len(token_ids) < MIN_TOKEN_LEN:
                skipped_short += 1
                continue
            if len(token_ids) > MAX_TOKEN_LEN:
                token_ids = token_ids[:MAX_TOKEN_LEN]
            prompts.append({
                "token_ids": token_ids,
                "prompt_length": len(token_ids),
                "source": filename,
            })

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for p in prompts:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

print(f"Total prompts: {len(prompts)}")
print(f"Skipped (too short < {MIN_TOKEN_LEN}): {skipped_short}")
print(f"Skipped (empty): {skipped_empty}")
print(f"Saved to: {OUTPUT_PATH}")

lengths = [p["prompt_length"] for p in prompts]
print(f"Token length stats: min={min(lengths)}, max={max(lengths)}, "
      f"mean={sum(lengths)/len(lengths):.0f}, median={sorted(lengths)[len(lengths)//2]}")
