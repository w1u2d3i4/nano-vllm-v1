# Nano-vLLM-v1

基于 [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 的改进版本，在保持轻量级（~1,800 行 Python）的前提下，引入多项来自 vLLM v1 和 SGLang 的核心优化。

## 性能总览

在 H100 80GB 单卡上，Qwen3-0.6B 模型的优化结果：

| 指标 | 原始 Nano-vLLM | Nano-vLLM-v1 | 提升 |
|------|---------------|-------------|------|
| **最大吞吐量** (512 seqs, all-at-once) | 11,139 tok/s | **37,430 tok/s** | **+236%** |
| **Max Goodput** (chatbot SLA) | — | **12 req/s** | — |
| **最大上下文** (单序列) | — | **655K tokens** | — |

## 分支说明

| 分支 | 用途 |
|------|------|
| `main` | 稳定版本，包含所有核心优化（Phase 1-3） |
| `ltr_test` | LTR 实验分支，在 main 基础上新增 Learning-to-Rank 调度器 |

## 改进总览

### Phase 1: 核心架构升级

#### 1. vLLM v1 统一调度器
替换原版 prefill/decode 二阶段分离调度，改为统一的 token-aware 调度逻辑。基于 `max_num_batched_tokens` 进行 token 级别的资源控制。

#### 2. Chunked Prefill（分块预填充）
将长 prompt 分块处理，避免单条长序列阻塞整个 batch，显著提升在线推理场景下的响应性。

#### 3. BlockManager 增强
新增 `get_token_layout()` 精确计算 token 在 block 内的布局。新增 `trim_blocks()` 支持 speculative decoding 的 KV cache 回滚。

#### 4. Decode 快速路径（Zero-Overhead Scheduler）
纯 decode 阶段跳过完整调度逻辑，直接为所有 running 序列分配 1 token。实测 **+37.3%** 吞吐。

### Phase 2: LTR 调度器（ltr_test 分支）

LTR 三阶段调度器：`FCFS → Heuristic → Model`，通过预测 output length 实现 SJF 排序。

### Phase 3: 系统级推理优化

基于系统级 profiling（Nsight Systems + step 分解）发现并修复 CPU 端瓶颈：

#### 1. Postprocess CUDA Tensor 修复（+49.6%）
**问题**：`scheduler.postprocess()` 中逐元素迭代 CUDA tensor，每次 `tensor[i]` 触发一次 GPU→CPU 传输。512 序列 = 512 次 `cudaMemcpy`，耗时 7.9ms。

**修复**：在循环前添加 `.tolist()` 一次性批量传输。一行代码修复。

```python
# Before: 7.928ms (512 individual GPU→CPU transfers)
for seq_index in seq_need_compute_logits:  # CUDA tensor
    ...

# After: 0.168ms (1 bulk transfer)
seq_need_compute_logits = seq_need_compute_logits.tolist()
for seq_index in seq_need_compute_logits:  # Python list
    ...
```

#### 2. Decode 快速输入准备（+9.5%）
**问题**：`prepare_model_input()` 为每个序列做 `list.extend()` + `list(range())`，Python 列表操作开销 1.4ms/step。

**修复**：为纯 decode 场景新增 `_prepare_decode_fast()` 快速路径，预分配固定大小数组，直接索引赋值。

#### 优化效果（512 seqs step 分解）

| 指标 | 优化前 | 优化后 | 变化 |
|------|--------|--------|------|
| Schedule | 0.87ms | 0.77ms | -11% |
| Forward+Sample | 10.07ms | 9.04ms | -10% |
| **Postprocess** | **8.15ms** | **0.43ms** | **-95%** |
| **Total step** | **19.09ms** | **10.24ms** | **-46%** |
| **CPU 占比** | **47.2%** | **11.7%** | **-75%** |

### Phase 3 附加：Triton Kernel & Speculative Decoding

作为优化探索，还实现了（代码保留但未启用）：

- **Triton Fused RMSNorm / RoPE**（`nanovllm/kernels/`）：手写 Triton kernel 在 eager 模式下比 `@torch.compile` 快 12.7%，但在 CUDA Graph 模式下因分配开销反而慢 5.4%。保留作为大模型参考。
- **Layer-Skip Speculative Decoding**（`nanovllm/engine/speculative.py`）：完整实现 Leviathan 2023 rejection sampling，draft=target 时 100% 接受率验证通过。但 Qwen3-0.6B 模型太小，CUDA Graph 使 normal decode 极快，speculative 无收益。
- **企业级 Benchmark 工具**：`bench_speculative.py` 和 `profile_baseline.py` / `profile_nsys.py` 用于 kernel 级和系统级 profiling。

详细的实验过程、分析和经验教训见 [conclusion_triron.md](conclusion_triron.md)。

## 项目结构

```
nanovllm/
├── config.py                    # 配置（含 speculative decoding 参数）
├── engine/
│   ├── llm_engine.py            # 引擎入口 + speculative_step
│   ├── scheduler.py             # 统一调度器 + Fast Path + postprocess 修复
│   ├── ltr_scheduler.py         # LTR 三阶段调度器
│   ├── model_runner.py          # 模型运行 + decode fast path + draft/verify
│   ├── block_manager.py         # KV cache 管理 + trim_blocks
│   ├── sequence.py              # 序列管理 + rollback
│   └── speculative.py           # Rejection sampling
├── kernels/
│   ├── fused_norm.py            # Triton Fused Add+RMSNorm
│   └── fused_rope.py            # Triton Fused RoPE
├── layers/                      # 模型层（attention, layernorm, rotary_embedding 等）
├── models/
│   └── qwen3.py                 # Qwen3 模型（+num_layers early exit）
└── utils/
    ├── context.py               # 推理上下文
    └── data_collector.py        # LTR 数据收集

bench_real.py                    # FCFS baseline benchmark
bench_ltr.py                     # LTR benchmark（在线学习）
bench_ltr_freeze.py              # LTR benchmark（冻结模型）
bench_speculative.py             # Speculative decoding benchmark
profile_baseline.py              # Kernel-level CUDA profiling
profile_nsys.py                  # System-level Nsight Systems profiling
conclusion_triron.md             # 完整技术报告
```

## Quick Start

```python
from nanovllm import LLM, SamplingParams

llm = LLM("/path/to/Qwen3-0.6B", enforce_eager=False, max_model_len=32768,
           max_num_batched_tokens=32768)
outputs = llm.generate(["Hello, Nano-vLLM."],
                        SamplingParams(temperature=0.6, max_tokens=256))
print(outputs[0]["text"])
```

## Benchmark

```bash
# 最大吞吐量测试
python bench_real.py --num_seqs 512 --qps 0 --max_model_len 32768

# QPS 在线流量测试
python bench_real.py --num_seqs 5000 --qps 50

# Speculative decoding（实验性）
python bench_speculative.py --num_seqs 512 --draft_layers 14 --spec_tokens 3
```

## 测试环境

- GPU: NVIDIA H100 80GB HBM3
- Model: Qwen3-0.6B (28 layers, hidden=896, 14 heads, 2 KV heads)
- Python: 3.12, PyTorch 2.x, flash_attn 2.8.3

## 关键经验

> **LLM 推理优化是系统问题，不是 kernel 问题。**

本项目中，31 行系统级修复（postprocess `.tolist()` + decode fast path）带来 +63.7% 吞吐提升，而 680 行 Triton kernel + Speculative Decoding 代码带来 0% 有效提升。详见 [conclusion_triron.md](conclusion_triron.md)。


