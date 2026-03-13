"""
LTR scheduling benchmark with real prompts.
Simulates online traffic with --qps (queries per second).
Tracks throughput over time and plots against model update events.

Usage:
    python bench_ltr.py --num_seqs 5000 --qps 50          # 50 req/s injection
    python bench_ltr.py --num_seqs 512                     # All at once (qps=0)
    python bench_ltr.py --num_seqs 50000 --qps 100         # High traffic
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

    prompts = load_prompts(args.prompts_path, args.num_seqs, args.max_model_len)
    token_ids_list = [p["token_ids"] for p in prompts]
    sampling_params_list = [SamplingParams(temperature=0.6, max_tokens=args.max_tokens) for _ in prompts]
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

    num_seqs = len(prompts)
    sample_interval = max(1, num_seqs // 1000)
    print_interval = max(50, num_seqs // 20)

    print(f"[LTR] {num_seqs} prompts, max_model_len={args.max_model_len}, "
          f"qps={args.qps or 'all-at-once'}")

    mbt = max(16384, args.max_model_len)
    llm = LLM(model_path, enforce_eager=False,
              max_model_len=args.max_model_len, max_num_batched_tokens=mbt,
              enable_ltr=True, ltr_data_path=args.labels_path)
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
    print(f"\n[LTR] Seqs: {num_seqs}, Output tokens: {total_tokens}, "
          f"Time: {elapsed_total:.2f}s, Throughput: {total_tokens/elapsed_total:.2f} tok/s")
    print(f"[LTR] Trace points: {len(throughput_trace)}")

    scheduler = llm.scheduler
    update_log = getattr(scheduler, "update_log", [])
    print(f"[LTR] Model updates: {len(update_log)}")

    plot_throughput(throughput_trace, update_log, t_start, elapsed_total, total_tokens, num_seqs)


def plot_throughput(trace, update_log, t_start, elapsed_total, total_tokens, num_seqs):
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
    ax.plot(times, tps, linewidth=0.8, color="#2196F3", alpha=0.7, label="Throughput (tok/s)")

    max_annotations = 20
    updates_to_show = update_log
    if len(update_log) > max_annotations:
        step = len(update_log) // max_annotations
        updates_to_show = update_log[::step]
        if update_log[-1] not in updates_to_show:
            updates_to_show.append(update_log[-1])

    for u in update_log:
        t_rel = u["time"] - t_start
        if 0 <= t_rel <= elapsed_total:
            ax.axvline(x=t_rel, color="#F44336", linestyle="-", linewidth=0.3, alpha=0.3)

    for i, u in enumerate(updates_to_show):
        t_rel = u["time"] - t_start
        if t_rel < 0 or t_rel > elapsed_total:
            continue
        y_pos = tps.max() * (0.98 - 0.06 * (i % 3))
        ax.annotate(f"{u['samples']}",
                    xy=(t_rel, y_pos),
                    fontsize=6, color="#F44336", ha="center", va="top",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#F44336", alpha=0.7, lw=0.5))

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#2196F3", linewidth=1, label="Throughput (tok/s)"),
        Line2D([0], [0], color="#F44336", linewidth=0.8, linestyle="-",
               label=f"Model updates ({len(update_log)} total)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_title(f"LTR Scheduler  |  {num_seqs} seqs, "
                 f"avg {total_tokens/elapsed_total:.0f} tok/s, "
                 f"{len(update_log)} model updates")
    ax.grid(True, alpha=0.3)

    out_path = "ltr_throughput.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Saved to {out_path}")


if __name__ == "__main__":
    main()
