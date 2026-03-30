"""
Kernel-level profiling for nano-vllm inference.

Profiles N decode steps and reports CUDA kernel time breakdown.
Exports Chrome trace for detailed visualization.

Usage:
    python profile_baseline.py                          # default 20 steps
    python profile_baseline.py --steps 50               # more steps
    python profile_baseline.py --trace baseline.json    # export Chrome trace
"""
import argparse
import os
import torch
from torch.profiler import profile, ProfilerActivity, record_function

from nanovllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description="Kernel-level profiling")
    parser.add_argument("--steps", type=int, default=20, help="Number of steps to profile")
    parser.add_argument("--num_seqs", type=int, default=64, help="Number of sequences to inject")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--trace", type=str, default="baseline_trace.json",
                        help="Chrome trace output path")
    parser.add_argument("--top", type=int, default=30, help="Top N kernels to show")
    args = parser.parse_args()

    model_path = "/opt/data/private/huggingface/Qwen3-0.6B/"
    mbt = max(16384, args.max_model_len)
    llm = LLM(model_path, enforce_eager=False,
              max_model_len=args.max_model_len, max_num_batched_tokens=mbt)

    # Warmup: inject sequences and run a few steps to enter steady decode state
    print(f"[Profile] Injecting {args.num_seqs} sequences for warmup...")
    sp = SamplingParams(temperature=0.6, max_tokens=args.max_tokens)
    for _ in range(args.num_seqs):
        llm.add_request("Hello, tell me a long story about artificial intelligence.", sp)

    # Warmup steps: run until all seqs have started decoding
    warmup_steps = 0
    while not llm.is_finished() and warmup_steps < 20:
        llm.step()
        warmup_steps += 1
    print(f"[Profile] Warmup done ({warmup_steps} steps). Running {args.num_seqs} seqs in decode.")

    # Check we still have running sequences
    if llm.is_finished():
        print("[Profile] WARNING: All sequences finished during warmup. Increase --max_tokens.")
        return

    running_count = len(llm.scheduler.running)
    print(f"[Profile] Running sequences: {running_count}")

    # ── Profile decode steps ──
    print(f"[Profile] Profiling {args.steps} decode steps...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    ) as prof:
        for i in range(args.steps):
            if llm.is_finished():
                print(f"[Profile] All finished after {i} steps")
                break
            llm.step()

    # ── Print kernel breakdown ──
    print(f"\n{'='*80}")
    print(f"  CUDA Kernel Breakdown (top {args.top}, sorted by cuda_time_total)")
    print(f"{'='*80}\n")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=args.top))

    # ── Print CPU time breakdown ──
    print(f"\n{'='*80}")
    print(f"  CPU Time Breakdown (top {args.top})")
    print(f"{'='*80}\n")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=args.top))

    # ── Export Chrome trace ──
    if args.trace:
        prof.export_chrome_trace(args.trace)
        print(f"\n[Profile] Chrome trace saved to: {args.trace}")

    # ── Summary: categorize key kernels ──
    print(f"\n{'='*80}")
    print(f"  Kernel Category Summary")
    print(f"{'='*80}\n")

    total_cuda_time = 0
    category_times = {
        "RMSNorm": 0,
        "RoPE": 0,
        "Attention (flash)": 0,
        "GEMM (linear)": 0,
        "Activation (SiLU/Mul)": 0,
        "Sampling": 0,
        "Other": 0,
    }

    for evt in prof.key_averages():
        cuda_time = evt.self_cuda_time_total
        total_cuda_time += cuda_time
        name = evt.key.lower()

        if "rmsnorm" in name or "rms_norm" in name or ("norm" in name and "layer" in name):
            category_times["RMSNorm"] += cuda_time
        elif "rope" in name or "rotary" in name:
            category_times["RoPE"] += cuda_time
        elif "flash" in name or "attention" in name or "sdpa" in name or "fmha" in name:
            category_times["Attention (flash)"] += cuda_time
        elif "gemm" in name or "cublas" in name or "mm" in name or "linear" in name:
            category_times["GEMM (linear)"] += cuda_time
        elif "silu" in name or "activation" in name or "gelu" in name:
            category_times["Activation (SiLU/Mul)"] += cuda_time
        elif "sample" in name or "topk" in name or "multinomial" in name:
            category_times["Sampling"] += cuda_time
        else:
            category_times["Other"] += cuda_time

    if total_cuda_time > 0:
        print(f"  {'Category':<25} {'CUDA Time (us)':>15} {'Percentage':>12}")
        print(f"  {'-'*52}")
        for cat, t in sorted(category_times.items(), key=lambda x: -x[1]):
            pct = t / total_cuda_time * 100
            print(f"  {cat:<25} {t:>15,.0f} {pct:>11.1f}%")
        print(f"  {'-'*52}")
        print(f"  {'TOTAL':<25} {total_cuda_time:>15,.0f} {'100.0%':>12}")
    else:
        print("  No CUDA time recorded.")

    print(f"\n[Profile] Done.")


if __name__ == "__main__":
    main()
