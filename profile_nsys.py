"""
System-level profiling script for Nsight Systems.
Runs a short decode workload for nsys to capture the full system timeline.

Usage:
    nsys profile -o report python profile_nsys.py
"""
import os
import time
import numpy as np
import torch

from nanovllm import LLM, SamplingParams


def main():
    model_path = "/opt/data/private/huggingface/Qwen3-0.6B/"
    llm = LLM(model_path, enforce_eager=False,
              max_model_len=32768, max_num_batched_tokens=32768)

    # Warmup
    llm.generate(["warmup"], SamplingParams(), use_tqdm=False)

    # Inject 64 sequences to get a realistic decode batch
    sp = SamplingParams(temperature=0.6, max_tokens=128)
    for i in range(64):
        llm.add_request(f"Tell me about topic {i} in great detail.", sp)

    # Run until prefill is done (a few steps)
    for _ in range(10):
        if llm.is_finished():
            break
        llm.step()

    # Now profile 50 pure decode steps
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("decode_profiling")
    t0 = time.time()
    total_tokens = 0
    for i in range(50):
        if llm.is_finished():
            break
        torch.cuda.nvtx.range_push(f"step_{i}")
        out, n = llm.step()
        total_tokens += n
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    elapsed = time.time() - t0

    # Also measure CPU overhead explicitly
    print(f"\n=== System-Level Metrics ===")
    print(f"Decode steps: 50")
    print(f"Total time: {elapsed*1000:.1f}ms")
    print(f"Avg step time: {elapsed/50*1000:.2f}ms")
    print(f"Running seqs: {len(llm.scheduler.running)}")

    # Measure step breakdown: CPU scheduling vs GPU forward
    times_schedule = []
    times_forward = []
    times_postprocess = []

    for i in range(20):
        if llm.is_finished():
            break

        t1 = time.perf_counter()
        seqs = llm.scheduler.schedule()
        t2 = time.perf_counter()

        token_ids, seq_logits = llm.model_runner.call("run", seqs)
        torch.cuda.synchronize()
        t3 = time.perf_counter()

        llm.scheduler.postprocess(seqs, token_ids, seq_logits)
        t4 = time.perf_counter()

        times_schedule.append((t2 - t1) * 1000)
        times_forward.append((t3 - t2) * 1000)
        times_postprocess.append((t4 - t3) * 1000)

    print(f"\n=== Step Breakdown (20 steps) ===")
    print(f"Schedule:    avg={np.mean(times_schedule):.3f}ms  p50={np.median(times_schedule):.3f}ms  p99={np.percentile(times_schedule, 99):.3f}ms")
    print(f"Forward+Sample: avg={np.mean(times_forward):.3f}ms  p50={np.median(times_forward):.3f}ms  p99={np.percentile(times_forward, 99):.3f}ms")
    print(f"Postprocess: avg={np.mean(times_postprocess):.3f}ms  p50={np.median(times_postprocess):.3f}ms  p99={np.percentile(times_postprocess, 99):.3f}ms")
    total_step = np.array(times_schedule) + np.array(times_forward) + np.array(times_postprocess)
    print(f"Total step:  avg={np.mean(total_step):.3f}ms  p50={np.median(total_step):.3f}ms")

    cpu_pct = (np.mean(times_schedule) + np.mean(times_postprocess)) / np.mean(total_step) * 100
    gpu_pct = np.mean(times_forward) / np.mean(total_step) * 100
    print(f"\nCPU overhead (schedule+postprocess): {cpu_pct:.1f}%")
    print(f"GPU forward+sample: {gpu_pct:.1f}%")

    # Memory analysis
    print(f"\n=== Memory Analysis ===")
    free, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    print(f"GPU total: {total/1e9:.1f} GB")
    print(f"Allocated: {allocated/1e9:.1f} GB")
    print(f"Reserved:  {reserved/1e9:.1f} GB")
    print(f"Free:      {free/1e9:.1f} GB")

    bm = llm.scheduler.block_manager
    print(f"KV blocks total: {len(bm.blocks)}")
    print(f"KV blocks free:  {len(bm.free_block_ids)}")
    print(f"KV blocks used:  {len(bm.used_block_ids)}")
    print(f"KV utilization:  {len(bm.used_block_ids)/len(bm.blocks)*100:.1f}%")


if __name__ == "__main__":
    main()
