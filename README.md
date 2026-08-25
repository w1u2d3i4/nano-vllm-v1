<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" width="250" height="55"/></a>
</p>

# Nano-vLLM: Reproducible H100 Single-GPU Optimization

**[中文](#中文) | [English](#english) | [完整实验报告 / Full report](reports/RESULTS.md)**

This contribution is based on the latest upstream nano-vLLM architecture at
commit `bb823b3`. It optimizes the sampler hot path, adds KV-cache and scheduler
regression coverage, and provides a reproducible comparison with vLLM 0.27.1
on one NVIDIA H100 PCIe GPU.

本贡献基于上游 nano-vLLM 的最新架构提交 `bb823b3`，优化采样器热路径，补充
KV Cache 与调度器回归测试，并在单张 NVIDIA H100 PCIe 上与 vLLM 0.27.1
进行可复现实验对比。

---

## 中文

### 项目简介

Nano-vLLM 是一个从零实现的轻量级 vLLM，核心代码规模小、结构清晰，适合学习、
代码审查和验证有明确边界的推理优化，同时保留 Prefix Cache、张量并行、
Torch Compilation 和 CUDA Graph 等关键能力。

本分支 `resume/logspace-sampler-vllm0271` 包含：

- 用数学等价的对数空间指数竞争替换概率空间采样，移除采样热路径中的
  `softmax` 和概率除法；
- 采样等价性测试，包括固定噪声和随机 `64 x 257` 输入；
- KV Cache 块边界、Prefix Cache 碰撞保护、Chunked Prefill、真实 Decode
  抢占及完成序列资源释放等回归测试；
- 单卡 benchmark 工具、原始 JSON 数据和与 vLLM 0.27.1 的可复现对比。

### 优化原理

对温度缩放后的 logits `x_i` 和独立指数噪声 `E_i > 0`，原实现选择：

```text
argmax_i softmax(x)_i / E_i
```

Softmax 的归一化因子对所有候选 token 都是相同的正常数，因此可等价写成：

```text
argmax_i exp(x_i) / E_i
= argmax_i x_i - log(E_i)
```

新实现保留原有的温度缩放和指数噪声分布，但在对数空间完成竞争。测试在固定噪声下
逐项验证新旧实现的 `argmax` 完全一致。

### 实验环境与方法

| 项目 | 配置 |
| --- | --- |
| 日期 | 2026-08-25 |
| GPU | 物理 GPU 1，NVIDIA H100 PCIe |
| Nano 基线 | 未修改的上游提交 `bb823b3` |
| 优化代码 | `cf587cd`；后续提交只增加测试和文档 |
| Nano 环境 | PyTorch 2.9.1+cu129，CUDA 12.9，BF16 |
| vLLM 参考 | vLLM 0.27.1，V1 Engine，V2 Model Runner，FlashAttention 3 |
| 模型 | Qwen3-0.6B、Qwen3-4B，本地相同权重 |
| 统一条件 | 相同物理 GPU、token prompts、输入/输出长度、引擎限制、CUDA Graph 设置和预热形状 |

Nano A/B 和跨引擎结果均报告 5 次正式重复的中位数，不选取最好成绩。跨引擎实验
由于使用各自支持的软件栈，属于用户视角的引擎比较，而不是纯组件 A/B。

### 实验一：采样器微基准

词表大小 151,936；20 次预热；新旧实现交替运行 7 轮，每轮 200 次迭代。

| Batch | 概率空间实现 | 对数空间实现 | 加速 |
| ---: | ---: | ---: | ---: |
| 1 | 0.150552 ms | 0.128447 ms | **17.21%** |
| 8 | 0.152336 ms | 0.130199 ms | **17.00%** |
| 32 | 0.161624 ms | 0.130046 ms | **24.28%** |

结论：优化直接命中的采样组件延迟降低 **17%–24%**，且 batch 越大，避免大词表
Softmax 和除法带来的收益越明显。

### 实验二：Nano-vLLM 前后 A/B

| 模型与负载 | 上游 Nano | 优化后 Nano | 变化 |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B，1 请求，输入 128 / 输出 512 | 373.19 tok/s | 376.41 tok/s | **+0.86%** |
| Qwen3-0.6B，32 请求，输入 128 / 输出 256 | 9201.65 tok/s | 9347.29 tok/s | **+1.58%** |
| Qwen3-4B，1 请求，输入 128 / 输出 256 | 144.61 tok/s | 143.39 tok/s | **-0.84%** |

分析：0.6B 工作负载获得 **0.9%–1.6%** 的端到端提升。4B 单请求结果轻微下降，
说明采样在较大模型总耗时中的占比更小，当前数据不支持宣称该场景有收益。

### 实验三：与 vLLM 0.27.1 对比

| 模型与负载 | 优化后 Nano | vLLM 0.27.1 | vLLM 领先 |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B，1 请求，输入 128 / 输出 512 | 376.41 tok/s | 457.97 tok/s | 21.67% |
| Qwen3-0.6B，32 请求，输入 128 / 输出 256 | 9347.29 tok/s | 9598.61 tok/s | **2.69%** |
| Qwen3-4B，1 请求，输入 128 / 输出 256 | 143.39 tok/s | 163.46 tok/s | 14.00% |

32 请求实验中，vLLM 5 次结果范围为 8489.77–10375.20 tok/s。因此 2.69% 是该
负载下的中位数差距，应描述为“接近 vLLM”，而不是稳定超过 vLLM。

### 结果分析与项目价值

- **本优化的明确优势：** 采样热路径缩短 17%–24%，代码改动小、数学等价、测试
  可精确验证；在测试的 0.6B/32 请求负载下，Nano 吞吐量距离当前 vLLM 中位数
  仅 2.69%。
- **Nano-vLLM 的优势：** 核心实现紧凑，便于端到端理解推理引擎、审查 KV Cache
  和调度行为，并在一个小型 PR 内完成优化、回归测试和性能证据闭环。
- **不能宣称的结论：** Nano-vLLM 没有全面超过 vLLM。vLLM 在三个实测负载中
  中位数都更快，并拥有更成熟的生产级内核、模型/硬件覆盖、推测解码、FP8 KV
  Cache、解耦 Prefill 和服务基础设施。
- **负结果同样保留：** CUDA Graph 元数据持久化缓冲区和融合 Paged-KV 写入两个
  原型在试验中慢约 4%–9%，因此已撤回，没有为了扩大改动而保留回退方案。

### 测试与复现

完整测试结果为 `9 passed`。运行：

```bash
/opt/data/private/vllm/bin/python -m pytest -q
```

在物理 GPU 1 上运行采样器微基准：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. \
  /opt/data/private/vllm/bin/python benchmarks/benchmark_sampler.py \
  --batch-sizes 1 8 32 --vocab-size 151936 \
  --warmup 20 --iterations 200 --repeats 7
```

端到端复现命令、逐轮原始结果和软件版本见：

- [完整实验报告](reports/RESULTS.md)
- [Benchmark 使用说明](benchmarks/README.md)
- [采样器微基准原始数据](reports/sampler-microbenchmark-h100.json)

### 安装与快速开始

安装本贡献分支：

```bash
pip install "git+https://github.com/w1u2d3i4/nano-vllm-v1.git@resume/logspace-sampler-vllm0271"
```

```python
from nanovllm import LLM, SamplingParams

llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
outputs = llm.generate(["Hello, Nano-vLLM."], sampling_params)
print(outputs[0]["text"])
```

---

## English

### Overview

Nano-vLLM is a lightweight vLLM implementation built from scratch. Its compact,
readable core makes it useful for learning, code review, and tightly scoped
inference optimization, while retaining key features such as prefix caching,
tensor parallelism, Torch compilation, and CUDA Graphs.

The `resume/logspace-sampler-vllm0271` branch adds:

- a mathematically equivalent log-space exponential race that removes softmax
  and probability-space division from the sampling hot path;
- exact sampler-equivalence tests with fixed noise and randomized `64 x 257`
  inputs;
- regressions for KV-cache block boundaries, prefix-cache collision safety,
  chunked prefill, real decode preemption, and finished-sequence block release;
- single-GPU benchmark tools, raw JSON results, and a reproducible comparison
  against vLLM 0.27.1.

### Optimization

For temperature-scaled logits `x_i` and independent exponential noise
`E_i > 0`, the previous implementation selects:

```text
argmax_i softmax(x)_i / E_i
```

The softmax normalizer is the same positive constant for every candidate, so
the expression is equivalent to:

```text
argmax_i exp(x_i) / E_i
= argmax_i x_i - log(E_i)
```

The optimized implementation preserves temperature scaling and the original
exponential-noise distribution while running the race in log space. Tests
verify exact old/new `argmax` equality under fixed noise.

### Experimental setup

| Item | Configuration |
| --- | --- |
| Date | 2026-08-25 |
| GPU | Physical GPU 1, NVIDIA H100 PCIe |
| Nano baseline | Unmodified upstream commit `bb823b3` |
| Optimized code | `cf587cd`; later commits only add tests and documentation |
| Nano stack | PyTorch 2.9.1+cu129, CUDA 12.9, BF16 |
| vLLM reference | vLLM 0.27.1, V1 Engine, V2 Model Runner, FlashAttention 3 |
| Models | Qwen3-0.6B and Qwen3-4B with the same local weights |
| Controlled inputs | Same physical GPU, token prompts, input/output lengths, engine limits, CUDA Graph setting, and warmup shape |

Nano A/B and cross-engine tables report medians over five recorded repetitions,
not best-case runs. The cross-engine experiment is an end-user engine comparison,
not a pure component A/B, because each engine uses its supported software stack.

### Experiment 1: sampler microbenchmark

Vocabulary 151,936; 20 warmups; 7 alternating repetitions of 200 iterations.

| Batch | Probability space | Log space | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 0.150552 ms | 0.128447 ms | **17.21%** |
| 8 | 0.152336 ms | 0.130199 ms | **17.00%** |
| 32 | 0.161624 ms | 0.130046 ms | **24.28%** |

The directly targeted sampler component is **17%–24% faster**. The benefit is
larger at batch 32, where avoiding a large-vocabulary softmax and division does
more work per launch.

### Experiment 2: end-to-end Nano-vLLM A/B

| Model / workload | Upstream Nano | Optimized Nano | Delta |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B, 1 request, input 128 / output 512 | 373.19 tok/s | 376.41 tok/s | **+0.86%** |
| Qwen3-0.6B, 32 requests, input 128 / output 256 | 9201.65 tok/s | 9347.29 tok/s | **+1.58%** |
| Qwen3-4B, 1 request, input 128 / output 256 | 144.61 tok/s | 143.39 tok/s | **-0.84%** |

The tested 0.6B workloads improve by **0.9%–1.6% end to end**. The slightly
negative 4B single-request result is an important boundary: sampling accounts
for less total time with the larger model, and the data does not support a win
for this workload.

### Experiment 3: comparison with vLLM 0.27.1

| Model / workload | Optimized Nano | vLLM 0.27.1 | vLLM lead |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B, 1 request, input 128 / output 512 | 376.41 tok/s | 457.97 tok/s | 21.67% |
| Qwen3-0.6B, 32 requests, input 128 / output 256 | 9347.29 tok/s | 9598.61 tok/s | **2.69%** |
| Qwen3-4B, 1 request, input 128 / output 256 | 143.39 tok/s | 163.46 tok/s | 14.00% |

The five vLLM repetitions for the 32-request workload range from 8489.77 to
10375.20 tok/s. The 2.69% median gap therefore supports “near parity on this
workload,” not a claim that Nano consistently beats vLLM.

### Analysis and project value

- **Measured optimization:** the sampler hot path is 17%–24% faster with a
  small, mathematically equivalent, exactly tested change. Nano comes within
  2.69% of the current vLLM median on the tested 0.6B/32-request workload.
- **Where Nano-vLLM is stronger:** its compact core is easier to understand end
  to end, inspect for KV-cache and scheduling behavior, and improve with a
  reviewable PR that closes the loop from reasoning to tests and measurements.
- **What this does not prove:** Nano-vLLM does not globally outperform vLLM.
  vLLM is faster in all three measured medians and has deeper production
  kernels, model/hardware coverage, speculative decoding, FP8 KV cache,
  disaggregated prefill, and serving infrastructure.
- **Negative results are documented:** prototypes for persistent CUDA-Graph
  metadata staging and fused paged-KV writes were approximately 4%–9% slower
  and were reverted instead of expanding the contribution around regressions.

### Tests and reproduction

The complete suite reports `9 passed`:

```bash
/opt/data/private/vllm/bin/python -m pytest -q
```

Run the sampler benchmark on physical GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. \
  /opt/data/private/vllm/bin/python benchmarks/benchmark_sampler.py \
  --batch-sizes 1 8 32 --vocab-size 151936 \
  --warmup 20 --iterations 200 --repeats 7
```

End-to-end commands, per-run raw results, and software versions are available
in:

- [Full experimental report](reports/RESULTS.md)
- [Benchmark guide](benchmarks/README.md)
- [Raw sampler microbenchmark](reports/sampler-microbenchmark-h100.json)

### Installation and quick start

Install this contribution branch:

```bash
pip install "git+https://github.com/w1u2d3i4/nano-vllm-v1.git@resume/logspace-sampler-vllm0271"
```

```python
from nanovllm import LLM, SamplingParams

llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
outputs = llm.generate(["Hello, Nano-vLLM."], sampling_params)
print(outputs[0]["text"])
```

## Upstream

This work is based on [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
