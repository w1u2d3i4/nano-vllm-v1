"""
LTR scheduling benchmark with FROZEN model (no online learning).
Uses pre-trained model for ranking but disables partial_fit() updates.
This isolates the performance impact of online learning vs pure LTR ranking.

Usage:
    python bench_ltr_freeze.py --num_seqs 5000 --qps 50    # 50 req/s injection
    python bench_ltr_freeze.py --num_seqs 512               # All at once (qps=0)

Comparison:
    bench_ltr.py       → LTR with online learning (partial_fit every 100 seqs)
    bench_ltr_freeze.py → LTR with frozen model (no training overhead)
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
    parser.add_argument("--num_seqs", type=int, default=256)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--qps", type=float, default=0,
                        help="Queries per second. 0 = inject all at once.")
    parser.add_argument("--prompts_path", type=str, default="prompts.jsonl")
    parser.add_argument("--labels_path", type=str, default="training_data.jsonl")
    args = parser.parse_args()

    if not os.path.exists(args.labels_path):
        print(f"Error: {args.labels_path} not found. Run bench_real.py --collect first.")
        return

    model_file = args.labels_path + ".model"
    if not os.path.exists(model_file):
        print(f"Error: {model_file} not found. Run bench_ltr.py first to train a model.")
        return

    prompts = load_prompts(args.prompts_path, args.num_seqs, args.max_model_len)
    token_ids_list = [p["token_ids"] for p in prompts]
    sampling_params_list = [SamplingParams(temperature=0.6, max_tokens=args.max_tokens) for _ in prompts]
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

    num_seqs = len(prompts)
    sample_interval = max(1, num_seqs // 1000)
    print_interval = max(50, num_seqs // 20)

    print(f"[LTR-FREEZE] {num_seqs} prompts, max_model_len={args.max_model_len}, "
          f"qps={args.qps or 'all-at-once'}")
    print(f"[LTR-FREEZE] Loading frozen model from {model_file}")

    mbt = max(16384, args.max_model_len)
    llm = LLM(model_path, enforce_eager=False,
              max_model_len=args.max_model_len, max_num_batched_tokens=mbt,
              enable_ltr=True, ltr_data_path=args.labels_path)

    # Freeze the model: disable all online learning overhead
    scheduler = llm.scheduler
    scheduler.frozen = True
    print(f"[LTR-FREEZE] Model frozen (phase={scheduler._phase_name}, no collect/train overhead)")

    llm.generate(["warmup"], SamplingParams(), use_tqdm=False)

    inject_all = args.qps <= 0
    if inject_all:
        for prompt, sp in zip(token_ids_list, sampling_params_list):
            llm.add_request(prompt, sp)

    total_tokens = 0
    completed = 0
    injected = 0 if not inject_all else num_seqs
    last_sampled = 0
    t_start = time.time()
    throughput_trace = []

    def is_done():
        return completed >= num_seqs

    while not is_done():
        if not inject_all and injected < num_seqs:
            elapsed = time.time() - t_start
            target = min(num_seqs, int(elapsed * args.qps) + 1)
            while injected < target:
                llm.add_request(token_ids_list[injected], sampling_params_list[injected])
                injected += 1

        if not llm.is_finished():
            output, num_step_tokens = llm.step()
            total_tokens += num_step_tokens
            completed += len(output)
        elif not inject_all and injected < num_seqs:
            time.sleep(0.001)
            continue

        if completed - last_sampled >= sample_interval:
            elapsed = time.time() - t_start
            if elapsed > 0:
                throughput_trace.append((elapsed, total_tokens / elapsed, completed))
            last_sampled = completed

        if completed > 0 and completed % print_interval < max(1, len(output) if not llm.is_finished() or inject_all else 1):
            elapsed = time.time() - t_start
            pending = injected - completed - len(llm.scheduler.running)
            print(f"  completed {completed}/{num_seqs}, injected {injected}, "
                  f"waiting {max(0, pending)}, "
                  f"throughput={total_tokens / elapsed:.0f} tok/s")

    elapsed_total = time.time() - t_start
    print(f"\n[LTR-FREEZE] Seqs: {num_seqs}, Output tokens: {total_tokens}, "
          f"Time: {elapsed_total:.2f}s, Throughput: {total_tokens/elapsed_total:.2f} tok/s")
    print(f"[LTR-FREEZE] Trace points: {len(throughput_trace)}")

    # Verify no model updates occurred
    update_log = getattr(scheduler, "update_log", [])
    initial_updates = len([u for u in update_log if u["time"] < t_start])
    runtime_updates = len([u for u in update_log if u["time"] >= t_start])
    print(f"[LTR-FREEZE] Model updates during run: {runtime_updates} (expected 0)")
    if runtime_updates > 0:
        print(f"[WARN] Model was updated during frozen run! Check UPDATE_INTERVAL override.")

    plot_throughput(throughput_trace, t_start, elapsed_total, total_tokens, num_seqs)


def plot_throughput(trace, t_start, elapsed_total, total_tokens, num_seqs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[WARN] matplotlib not installed, skipping plot.")
        return

    if not trace:
        return

    times = np.array([t[0] for t in trace])
    tps = np.array([t[1] for t in trace])

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(times, tps, linewidth=0.8, color="#4CAF50", alpha=0.7, label="Throughput (tok/s)")

    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_title(f"LTR Scheduler (FROZEN)  |  {num_seqs} seqs, "
                 f"avg {total_tokens/elapsed_total:.0f} tok/s, "
                 f"no model updates")
    ax.grid(True, alpha=0.3)

    out_path = "ltr_freeze_throughput.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Saved to {out_path}")


if __name__ == "__main__":
    main()
