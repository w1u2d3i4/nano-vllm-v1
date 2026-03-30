"""
Speculative decoding benchmark.

Compares normal decode vs speculative decode throughput.

Usage:
    python bench_speculative.py --num_seqs 512 --qps 0
    python bench_speculative.py --num_seqs 512 --qps 0 --draft_layers 7 --spec_tokens 3
"""
import argparse
import json
import os
import time
from random import seed, sample

from nanovllm import LLM, SamplingParams


def load_prompts(path, num_seqs, max_model_len):
    all_prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            if p["prompt_length"] < max_model_len - 1:
                all_prompts.append(p)
    seed(42)
    if num_seqs < len(all_prompts):
        return sample(all_prompts, num_seqs)
    return all_prompts[:num_seqs]


def main():
    parser = argparse.ArgumentParser(description="Speculative Decoding Benchmark")
    parser.add_argument("--num_seqs", type=int, default=512)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--qps", type=float, default=0)
    parser.add_argument("--prompts_path", type=str, default="prompts.jsonl")
    parser.add_argument("--draft_layers", type=int, default=7)
    parser.add_argument("--spec_tokens", type=int, default=3)
    args = parser.parse_args()

    prompts = load_prompts(args.prompts_path, args.num_seqs, args.max_model_len)
    token_ids_list = [p["token_ids"] for p in prompts]
    sampling_params_list = [SamplingParams(temperature=0.6, max_tokens=args.max_tokens)
                            for _ in prompts]
    num_seqs = len(prompts)

    model_path = "/opt/data/private/huggingface/Qwen3-0.6B/"
    mbt = max(16384, args.max_model_len)

    print(f"[Speculative] {num_seqs} prompts, draft_layers={args.draft_layers}, "
          f"spec_tokens={args.spec_tokens}")

    llm = LLM(model_path, enforce_eager=False,
              max_model_len=args.max_model_len, max_num_batched_tokens=mbt,
              enable_speculative=True,
              num_draft_layers=args.draft_layers,
              num_speculative_tokens=args.spec_tokens)
    llm.generate(["warmup"], SamplingParams(), use_tqdm=False)

    if args.qps == 0:
        # All-at-once mode
        for tid, sp in zip(token_ids_list, sampling_params_list):
            llm.add_request(tid, sp)

        total_output_tokens = 0
        completed = 0
        t_start = time.time()

        while not llm.is_finished():
            output, num_step_tokens = llm.step()
            total_output_tokens += num_step_tokens
            completed += len(output)

        elapsed = time.time() - t_start
        print(f"[Speculative] Seqs: {completed}, Output tokens: {total_output_tokens}, "
              f"Time: {elapsed:.2f}s, Throughput: {total_output_tokens/elapsed:.2f} tok/s")
    else:
        # QPS mode
        injected = 0
        completed = 0
        total_output_tokens = 0
        t_start = time.time()

        while completed < num_seqs:
            if injected < num_seqs:
                elapsed = time.time() - t_start
                target = min(num_seqs, int(elapsed * args.qps) + 1)
                while injected < target:
                    llm.add_request(token_ids_list[injected], sampling_params_list[injected])
                    injected += 1

            if not llm.is_finished():
                output, num_step_tokens = llm.step()
                total_output_tokens += num_step_tokens
                completed += len(output)
            elif injected < num_seqs:
                time.sleep(0.001)

        elapsed = time.time() - t_start
        print(f"[Speculative] Seqs: {completed}, Output tokens: {total_output_tokens}, "
              f"Time: {elapsed:.2f}s, Throughput: {total_output_tokens/elapsed:.2f} tok/s")


if __name__ == "__main__":
    main()
