# H100 single-GPU optimization report

Date: 2026-08-25

## Outcome

This branch turns the latest upstream nano-vLLM snapshot into a small,
reviewable performance contribution:

- upstream baseline: `bb823b3`
- branch: `resume/logspace-sampler-vllm0271`
- benchmark and regression tests: `fa915ac`
- sampler optimization: `cf587cd`
- expanded equivalence test: `03fc0cb`

The optimization replaces probability-space exponential-race sampling with an
equivalent log-space formulation. It removes softmax and probability-space
division from the sampling hot path.

## Why the transformation is equivalent

For scaled logits `x_i` and independent exponential noise `E_i > 0`, the
previous implementation selected:

```text
argmax_i softmax(x)_i / E_i
```

Because the softmax normalization is the same positive constant for every
candidate, this is equivalent to:

```text
argmax_i exp(x_i) / E_i
= argmax_i x_i - log(E_i)
```

The implementation keeps the original exponential draw and temperature
scaling, but performs the race in log space. Tests compare the old and new
argmax exactly with fixed noise, including randomized 64 x 257 inputs.

## Test coverage

`pytest -q` reports `9 passed`.

The new tests cover:

- the 256/257-token KV block allocation boundary;
- complete-block-only prefix reuse;
- hash collision protection through token equality;
- chunked prefill crossing a block boundary before decode;
- real decode preemption and finished-sequence block release;
- exact sampler equivalence for fixed and randomized inputs.

## Sampler microbenchmark

Hardware: physical GPU 1, NVIDIA H100 PCIe. Software:
PyTorch 2.9.1+cu129, CUDA 12.9, vocabulary 151,936, 20 warmups,
7 alternating repetitions of 200 iterations. Values are medians.

| Batch | Probability space | Log space | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 0.150552 ms | 0.128447 ms | 17.21% |
| 8 | 0.152336 ms | 0.130199 ms | 17.00% |
| 32 | 0.161624 ms | 0.130046 ms | 24.28% |

Raw data: [sampler-microbenchmark-h100.json](sampler-microbenchmark-h100.json).

## End-to-end nano A/B

Both sides use the same nano environment, model files, deterministic token
prompts, BF16 model dtype, engine limits, CUDA Graph setting, physical GPU 1,
one warmup batch, and five recorded repetitions. The baseline worktree is the
unmodified upstream commit `bb823b3`. The optimized measurement is commit
`cf587cd`; the later commit only adds a test.

| Model / workload | Upstream nano | Optimized nano | Delta |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B, 1 request, in 128 / out 512 | 373.19 tok/s | 376.41 tok/s | +0.86% |
| Qwen3-0.6B, 32 requests, in 128 / out 256 | 9201.65 tok/s | 9347.29 tok/s | +1.58% |
| Qwen3-4B, 1 request, in 128 / out 256 | 144.61 tok/s | 143.39 tok/s | -0.84% |

The 4B result is a boundary, not a win: sampling is a smaller fraction of
end-to-end time and the measured change is slightly negative. The defensible
claim is a 17-24% component improvement and a 0.9-1.6% end-to-end improvement
for the tested 0.6B workloads.

Raw results:

- [0.6B c1 baseline](recheck-baseline-nano-qwen3-0.6b-c1-o512.json) and
  [optimized](final-nano-qwen3-0.6b-c1-o512.json)
- [0.6B c32 baseline](recheck-baseline-nano-qwen3-0.6b-c32-o256.json) and
  [optimized](final-nano-qwen3-0.6b-c32-o256.json)
- [4B c1 baseline](recheck-baseline-nano-qwen3-4b-c1-o256.json) and
  [optimized](final-nano-qwen3-4b-c1-o256.json)

## Comparison with current vLLM

The reference is the official
[vLLM v0.27.1 release](https://github.com/vllm-project/vllm/releases/tag/v0.27.1),
using its `+cu129` wheel, PyTorch 2.13.0+cu129, Transformers 5.15.1,
FlashAttention 3, the V1 engine, and V2 Model Runner. The release wheel SHA-256
was checked against the GitHub release API.

The cross-engine test uses the same physical H100, local model, BF16 dtype,
token prompts, input/output lengths, engine limits, GPU memory utilization,
warmup shape, and five repetitions. It is an end-user engine comparison, not a
pure component A/B, because the supported software stacks differ.

| Model / workload | Optimized nano | vLLM 0.27.1 | vLLM lead |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B, 1 request, in 128 / out 512 | 376.41 tok/s | 457.97 tok/s | 21.67% |
| Qwen3-0.6B, 32 requests, in 128 / out 256 | 9347.29 tok/s | 9598.61 tok/s | 2.69% |
| Qwen3-4B, 1 request, in 128 / out 256 | 143.39 tok/s | 163.46 tok/s | 14.00% |

The 32-request vLLM repetitions range from 8489.77 to 10375.20 tok/s, so the
2.69% median gap should be described as near parity for this one workload, not
as a stable nano win.

Raw vLLM results:

- [0.6B c1](vllm0271-qwen3-0.6b-c1-o512.json)
- [0.6B c32](vllm0271-qwen3-0.6b-c32-o256.json)
- [4B c1](vllm0271-qwen3-4b-c1-o256.json)

## What is better, and what is not

nano-vLLM is better for learning, review, and a tightly scoped contribution:
the core is small enough to understand end to end, the optimization has a
short mathematical argument, and the regression/performance evidence fits in
one PR. In the tested 0.6B 32-request workload it comes within 2.7% of the
current vLLM median while retaining that simplicity.

vLLM remains better as a production engine and is faster in all three measured
workloads. Its current architecture includes a V1 KV cache manager and
automatic prefix caching, FP8 KV cache support, speculative decoding,
disaggregated prefill, broad model/hardware support, and much deeper kernel and
serving infrastructure:

- [vLLM automatic prefix caching design](https://docs.vllm.ai/en/stable/design/prefix_caching/)
- [vLLM quantized KV cache](https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache/)
- [vLLM disaggregated prefill](https://docs.vllm.ai/en/latest/features/disagg_prefill/)
- [vLLM GPU installation and wheel variants](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)

Therefore this contribution should not be presented as nano-vLLM globally
beating vLLM. Its value is a verified optimization in a readable modern
inference engine, plus an honest comparison against the current production
reference.

## Rejected experiments

Two broader changes were prototyped and then reverted:

- persistent CUDA-Graph metadata staging buffers;
- a fused FlashAttention paged-KV update replacing the separate cache store.

Pilot end-to-end runs were about 4-9% slower depending on workload. Reverting
them keeps the branch evidence-driven and avoids expanding a three-day
contribution around regressions.

## Reproduction

Run tests:

```bash
/opt/data/private/vllm/bin/python -m pytest -q
```

Run the sampler microbenchmark:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. \
  /opt/data/private/vllm/bin/python benchmarks/benchmark_sampler.py \
  --batch-sizes 1 8 32 --vocab-size 151936 \
  --warmup 20 --iterations 200 --repeats 7
```

Run one nano workload:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. \
  /opt/data/private/vllm/bin/python benchmarks/benchmark_offline.py \
  --engine nano --model /opt/data/private/huggingface/Qwen3-0.6B \
  --num-prompts 32 --input-len 128 --output-len 256 \
  --max-model-len 4096 --max-num-seqs 64 \
  --max-num-batched-tokens 8192 --gpu-memory-utilization 0.9 \
  --repeats 5
```

Run the matching current-vLLM workload:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. \
  /opt/data/private/llm_test/vllm-0.27.1-cu129-env/bin/python \
  benchmarks/benchmark_offline.py \
  --engine vllm --model /opt/data/private/huggingface/Qwen3-0.6B \
  --num-prompts 32 --input-len 128 --output-len 256 \
  --max-model-len 4096 --max-num-seqs 64 \
  --max-num-batched-tokens 8192 --gpu-memory-utilization 0.9 \
  --repeats 5
```

## Resume-ready wording

> Optimized the sampling hot path in a latest-upstream nano-vLLM fork using a
> mathematically equivalent log-space exponential race, reducing H100 sampler
> latency by 17-24% and improving Qwen3-0.6B end-to-end throughput by 0.9-1.6%.
> Added 9 regression tests and a reproducible single-GPU benchmark against
> vLLM 0.27.1; reached within 2.7% of its median throughput on the tested
> 32-request workload while documenting the 4B boundary and reverted
> regressions.
