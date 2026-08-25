#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics

import torch

from nanovllm.layers.sampler import _exponential_race


@torch.compile
def probability_space_sampler(
    logits: torch.Tensor, temperatures: torch.Tensor
) -> torch.Tensor:
    scores = logits.float().div_(temperatures.unsqueeze(dim=1))
    probabilities = torch.softmax(scores, dim=-1)
    exponential = torch.empty_like(probabilities).exponential_(1).clamp_min_(1e-10)
    return probabilities.div_(exponential).argmax(dim=-1)


@torch.compile
def log_space_sampler(
    logits: torch.Tensor, temperatures: torch.Tensor
) -> torch.Tensor:
    logits = logits.float()
    exponential = torch.empty_like(logits).exponential_(1)
    return _exponential_race(logits, temperatures, exponential)


def elapsed_ms(fn, logits, temperatures, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn(logits, temperatures)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--vocab-size", type=int, default=151_936)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(0)
    results = []
    for batch_size in args.batch_sizes:
        logits = torch.randn(
            batch_size,
            args.vocab_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        temperatures = torch.ones(batch_size, device="cuda", dtype=torch.float32)
        for _ in range(args.warmup):
            probability_space_sampler(logits, temperatures)
            log_space_sampler(logits, temperatures)
        torch.cuda.synchronize()

        reference_times = []
        log_space_times = []
        for repeat in range(args.repeats):
            if repeat % 2:
                ordered = (
                    ("log_space", log_space_sampler),
                    ("probability_space", probability_space_sampler),
                )
            else:
                ordered = (
                    ("probability_space", probability_space_sampler),
                    ("log_space", log_space_sampler),
                )
            for label, sampler in ordered:
                value = elapsed_ms(sampler, logits, temperatures, args.iterations)
                if label == "probability_space":
                    reference_times.append(value)
                else:
                    log_space_times.append(value)

        reference_ms = statistics.median(reference_times)
        log_space_ms = statistics.median(log_space_times)
        results.append(
            {
                "batch_size": batch_size,
                "probability_space_ms": reference_ms,
                "log_space_ms": log_space_ms,
                "probability_space_repetitions_ms": reference_times,
                "log_space_repetitions_ms": log_space_times,
                "speedup_percent": (reference_ms / log_space_ms - 1.0) * 100.0,
            }
        )

    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "vocab_size": args.vocab_size,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "repeats": args.repeats,
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
