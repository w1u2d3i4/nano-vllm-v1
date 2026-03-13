"""
FCFS baseline benchmark with real prompts + optional label collection.

Usage:
    python bench_real.py                                  # FCFS benchmark (all at once)
    python bench_real.py --qps 50                         # FCFS with 50 req/s injection
    python bench_real.py --collect                        # FCFS + collect labels for LTR
    python bench_real.py --num_seqs 5000 --qps 50         # More sequences with qps
    python bench_real.py --max_model_len 4096             # Shorter context
"""
import argparse
import json
import os
import time
from random import seed, sample

from nanovllm import LLM, SamplingParams


def load_prompts(path: str, num_seqs: int, max_model_len: int):
    all_prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            if p["prompt_length"] < max_model_len - 1:
                all_prompts.append(p)
    seed(42)
    if num_seqs < len(all_prompts):
        selected = sample(all_prompts, num_seqs)
    else:
        selected = all_prompts[:num_seqs]
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true", help="Also collect labels for LTR training")
    parser.add_argument("--num_seqs", type=int, default=256)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--qps", type=float, default=0,
                        help="Queries per second. 0 = inject all at once.")
    parser.add_argument("--prompts_path", type=str, default="prompts.jsonl")
    parser.add_argument("--labels_path", type=str, default="training_data.jsonl")
    args = parser.parse_args()

    prompts = load_prompts(args.prompts_path, args.num_seqs, args.max_model_len)
    token_ids_list = [p["token_ids"] for p in prompts]
    sampling_params_list = [SamplingParams(temperature=0.6, max_tokens=args.max_tokens) for _ in prompts]
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

    num_seqs = len(prompts)
    sample_interval = max(1, num_seqs // 1000)
    print_interval = max(50, num_seqs // 20)

    print(f"[FCFS] {num_seqs} prompts, max_model_len={args.max_model_len}, "
          f"qps={args.qps or 'all-at-once'}")

    mbt = max(16384, args.max_model_len)
    llm = LLM(model_path, enforce_eager=False,
              max_model_len=args.max_model_len, max_num_batched_tokens=mbt)
    llm.generate(["warmup"], SamplingParams(), use_tqdm=False)

    inject_all = args.qps <= 0

    # All-at-once mode: use original generate() API
    if inject_all:
        t = time.time()
        outputs = llm.generate(token_ids_list, sampling_params_list)
        elapsed = time.time() - t

        total_output = sum(len(o["token_ids"]) for o in outputs)
        print(f"[FCFS] Seqs: {num_seqs}, Output tokens: {total_output}, "
              f"Time: {elapsed:.2f}s, Throughput: {total_output/elapsed:.2f} tok/s")

        if args.collect:
            save_labels(prompts, outputs, args)
        return

    # QPS mode: use step() API with gradual injection
    total_tokens = 0
    completed = 0
    injected = 0
    last_sampled = 0
    t_start = time.time()
    throughput_trace = []
    outputs_collected = []

    def is_done():
        return completed >= num_seqs

    while not is_done():
        if injected < num_seqs:
            elapsed = time.time() - t_start
            target = min(num_seqs, int(elapsed * args.qps) + 1)
            while injected < target:
                llm.add_request(token_ids_list[injected], sampling_params_list[injected])
                injected += 1

        if not llm.is_finished():
            output, num_step_tokens = llm.step()
            total_tokens += num_step_tokens
            completed += len(output)
            outputs_collected.extend(output)
        elif injected < num_seqs:
            time.sleep(0.001)
            continue

        if completed - last_sampled >= sample_interval:
            elapsed = time.time() - t_start
            if elapsed > 0:
                throughput_trace.append((elapsed, total_tokens / elapsed, completed))
            last_sampled = completed

        if completed > 0 and completed % print_interval < max(1, len(output) if not llm.is_finished() else 1):
            elapsed = time.time() - t_start
            pending = injected - completed - len(llm.scheduler.running)
            print(f"  completed {completed}/{num_seqs}, injected {injected}, "
                  f"waiting {max(0, pending)}, "
                  f"throughput={total_tokens / elapsed:.0f} tok/s")

    elapsed_total = time.time() - t_start
    print(f"\n[FCFS] Seqs: {num_seqs}, Output tokens: {total_tokens}, "
          f"Time: {elapsed_total:.2f}s, Throughput: {total_tokens/elapsed_total:.2f} tok/s")
    print(f"[FCFS] Trace points: {len(throughput_trace)}")

    if args.collect:
        save_labels(prompts, outputs_collected, args)


def save_labels(prompts, outputs, args):
    labels = []
    for prompt_info, output in zip(prompts, outputs):
        labels.append({
            "prompt_length": prompt_info["prompt_length"],
            "temperature": 0.6,
            "max_tokens": args.max_tokens,
            "output_length": len(output["token_ids"]),
        })
    with open(args.labels_path, "w", encoding="utf-8") as f:
        for label in labels:
            f.write(json.dumps(label) + "\n")
    out_lens = [l["output_length"] for l in labels]
    print(f"Labels saved to: {args.labels_path}")
    print(f"Output length stats: min={min(out_lens)}, max={max(out_lens)}, "
          f"mean={sum(out_lens)/len(out_lens):.0f}")


if __name__ == "__main__":
    main()
