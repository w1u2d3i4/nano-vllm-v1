# Reproducible single-GPU comparison

Run each engine in a fresh process on the same physical GPU. Keep model,
precision, prompt tokens, token budgets, CUDA Graph setting, and repeat count
identical. The benchmark deliberately reports raw repetitions instead of only
the best run.

Example:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python benchmarks/benchmark_offline.py \
  --engine nano --model /opt/data/private/huggingface/Qwen3-0.6B \
  --num-prompts 16 --input-len 128 --output-len 256 --repeats 5 \
  --output reports/nano-qwen3-0.6b-c16.json
```

Use an isolated environment containing the selected vLLM release for the
vLLM run. Do not compare against an unrecorded or older installed version.
