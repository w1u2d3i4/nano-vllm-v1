#!/usr/bin/env python3
"""Run a deterministic, same-process offline throughput comparison.

Launch one engine per process. Select the physical GPU outside this script, e.g.
CUDA_VISIBLE_DEVICES=1 python benchmarks/benchmark_offline.py --engine nano ...
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import subprocess
import time
from pathlib import Path

import torch
from transformers import AutoConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=("nano", "vllm"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def make_prompts(args: argparse.Namespace) -> list[list[int]]:
    config = AutoConfig.from_pretrained(args.model)
    rng = random.Random(args.seed)
    low = 100
    high = min(config.vocab_size - 1, 100_000)
    return [
        [rng.randint(low, high) for _ in range(args.input_len)]
        for _ in range(args.num_prompts)
    ]


def build_engine(args: argparse.Namespace):
    common = dict(
        model=args.model,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    if args.engine == "nano":
        from nanovllm import LLM, SamplingParams

        return LLM(**common), SamplingParams

    from vllm import LLM, SamplingParams

    common["dtype"] = "bfloat16"
    return LLM(**common), SamplingParams


def generate(engine_name, llm, prompts, sampling_params, *, use_tqdm=False):
    if engine_name == "vllm":
        prompts = [{"prompt_token_ids": prompt} for prompt in prompts]
    return llm.generate(prompts, sampling_params, use_tqdm=use_tqdm)


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")

    prompts = make_prompts(args)
    llm, sampling_params_cls = build_engine(args)
    sampling_params = sampling_params_cls(
        temperature=1.0,
        ignore_eos=True,
        max_tokens=args.output_len,
    )

    warmup_prompts = [prompt[: min(32, args.input_len)] for prompt in prompts]
    warmup_params = sampling_params_cls(temperature=1.0, ignore_eos=True, max_tokens=8)
    # Match the measured batch size so shape-specialized sampling and decode
    # paths compile before the first recorded repetition.
    generate(args.engine, llm, warmup_prompts, warmup_params)

    durations = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        outputs = generate(args.engine, llm, prompts, sampling_params)
        durations.append(time.perf_counter() - started)
        if len(outputs) != args.num_prompts:
            raise RuntimeError(f"expected {args.num_prompts} outputs, got {len(outputs)}")

    total_output_tokens = args.num_prompts * args.output_len
    throughputs = [total_output_tokens / duration for duration in durations]
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    result = {
        "schema_version": 1,
        "engine": args.engine,
        "engine_version": getattr(
            __import__(args.engine if args.engine == "vllm" else "nanovllm"),
            "__version__",
            None,
        ),
        "git_revision": git_revision() if args.engine == "nano" else None,
        "model": str(Path(args.model).resolve()),
        "gpu": torch.cuda.get_device_name(0),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "config": {
            "num_prompts": args.num_prompts,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": args.enforce_eager,
            "seed": args.seed,
            "repeats": args.repeats,
        },
        "duration_seconds": durations,
        "output_tokens_per_second": throughputs,
        "median_output_tokens_per_second": statistics.median(throughputs),
        "mean_output_tokens_per_second": statistics.fmean(throughputs),
        "stdev_output_tokens_per_second": statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0,
        "gpu_memory_used_mib": (total_bytes - free_bytes) / 2**20,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
