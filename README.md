# Nano-vLLM-v1

基于 [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 的改进版本，在保持轻量级（~1,500 行 Python，原版 ~1,360 行基础上仅增加 ~100 行）的前提下，引入多项来自 vLLM v1 和 SGLang 的核心优化。

## 分支说明

| 分支 | 用途 |
|------|------|
| `main` | 稳定版本，包含 Phase 1 核心架构升级（统一调度器 + Fast Path + Chunked Prefill） |
| `ltr_test` | LTR 实验分支，在 main 基础上新增 Learning-to-Rank 调度器及相关 benchmark |

### ltr_test 分支 vs main 分支

| 特性 | main | ltr_test |
|------|------|----------|
| 统一调度器 + Fast Path | ✅ | ✅ |
| Chunked Prefill | ✅ | ✅ |
| LTR 三阶段调度器 | ❌ | ✅ |
| 在线学习（partial_fit） | ❌ | ✅ |
| Freeze 模式（纯预测排序） | ❌ | ✅ |
| 真实 prompt benchmark | ❌ | ✅ |
| qps 流量注入模式 | ❌ | ✅ |

### ltr_test 新增文件

```
nanovllm/engine/ltr_scheduler.py   # LTR 三阶段调度器
nanovllm/utils/data_collector.py   # 训练数据收集（环形 buffer）
bench_real.py                      # FCFS baseline benchmark
bench_ltr.py                       # LTR benchmark（在线学习）
bench_ltr_freeze.py                # LTR benchmark（冻结模型，最终优化目标）
scripts/preprocess_sharegpt.py     # 真实 prompt 预处理脚本
scripts/push.sh                    # 一键 push 脚本
```

### ltr_test 修改文件

```
nanovllm/config.py      # +enable_ltr, +ltr_data_path 配置
nanovllm/engine/llm_engine.py  # 按 config 选择调度器，exit 时持久化
nanovllm/engine/sequence.py    # +arrival_time（防饥饿机制）
```

## 实验目的

### ltr_test 分支核心目标

**在 qps 在线流量模拟场景下，使 LTR（冻结模型）调度器的吞吐量超过 FCFS baseline。**

LTR 调度器通过预测每个请求的 output length，实现 Shortest-Job-First（SJF）排序。理论上，SJF 能减少平均等待时间，提升 GPU 利用率。实验分为三个阶段：

1. **bench_real.py**：FCFS baseline，作为对比基准
2. **bench_ltr.py**：LTR 在线学习版本，验证模型训练效果
3. **bench_ltr_freeze.py**：LTR 冻结模型，代表生产环境推理场景，是最终优化目标

### LTR 调度器三阶段

```
FCFS (< 50 样本) → Heuristic (< 200 样本, 按 prompt_length 排序) → Model (>= 200 样本, SGDRegressor 预测 output_length 做 SJF)
```

### Freeze 模式优化

冻结模型模式通过 `scheduler.frozen = True` 启用，去除所有在线学习开销：
- 跳过 `_update_phase()` 检查
- 跳过 `collector.collect()` 数据收集
- 跳过 `_maybe_train()` 模型训练
- 只保留 SJF 排序调度

## 改进总览

### Phase 1: 核心架构升级（main 分支）

#### 1. Chunked Prefill（分块预填充）
将长 prompt 分块处理，避免单条长序列阻塞整个 batch，显著提升在线推理场景下的响应性。
- 通过 `chunked_prefill=True` 启用
- 长序列按 `max_num_batched_tokens` 拆分为多个 chunk，与 decode 请求混合调度

#### 2. vLLM v1 统一调度器
替换原版 prefill/decode 二阶段分离调度，改为统一的 token-aware 调度逻辑：
- 不再区分 prefill 和 decode 阶段，统一在同一个调度循环中处理
- 基于 `max_num_batched_tokens` 进行 token 级别的资源控制
- Attention 层统一使用 `flash_attn_varlen_func` 单一路径，兼容 flash_attn 2.8.3

#### 3. BlockManager 增强
- 新增 `get_token_layout()` 方法，精确计算 token 在 block 内的布局
- 区分 `num_new_computed_tokens_in_used`（已有 block 中的新 token）和 `in_free`（需要新 block 的 token）
- 实现更精确的内存预测与按需分配

#### 4. Decode 快速路径（Zero-Overhead Scheduler）
纯 decode 阶段且无新请求时，跳过完整调度逻辑（token_budget 计算、waiting 队列遍历、preemption 检查），直接为所有 running 序列分配 1 token 并返回：
- 调度开销从 O(n_running + n_waiting) 降至 O(n_running) 的轻量检查
- 实测吞吐量提升 **+27.8%**（11,961 → 15,294 tok/s）
- 技术来源：SGLang v0.4 Zero-Overhead Batch Scheduler

### Phase 2: LTR 调度器（ltr_test 分支）

详见上方「实验目的」和「LTR 调度器三阶段」。

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Quick Start

```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Benchmark

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--num_seqs` | 请求总数 | 256 |
| `--max_tokens` | 每个请求最大生成 token 数 | 1024 |
| `--max_model_len` | 模型最大上下文长度 | 32768 |
| `--qps` | 每秒注入请求数（0=一次性注入） | 0 |
| `--prompts_path` | 真实 prompt 文件路径 | prompts.jsonl |
| `--labels_path` | 训练数据文件路径 | training_data.jsonl |
| `--collect` | （bench_real.py 专用）收集训练标签 | false |

### 运行命令

```bash
# FCFS baseline（qps 在线模式）
python bench_real.py --num_seqs 5000 --qps 50

# LTR 在线学习
python bench_ltr.py --num_seqs 50000 --qps 50

# LTR 冻结模型（最终优化目标）
python bench_ltr_freeze.py --num_seqs 5000 --qps 50

# 收集训练标签
python bench_real.py --num_seqs 5000 --qps 50 --collect
```

### 测试环境

- Hardware: NVIDIA H100 PCIe (80GB)
- Model: Qwen3-0.6B
- Python: 3.12
- flash_attn: 2.8.3
- scikit-learn: 1.8.0

### Phase 1 性能结果（bench.py, 256 seq, max_model_len=4096）

| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| Nano-vLLM（原版） | 133,966   | 12.03    | 11,138.68             |
| Nano-vLLM-v1   | 133,966     | 8.76     | 15,294.23 (**+37.3%**) |

### Phase 2 性能结果（5000 seq, qps=50, max_model_len=32768）

| Benchmark | 模式 | Output Tokens | Time (s) | Throughput (tok/s) | vs FCFS |
|-----------|------|-------------|----------|--------------------|---------|
| bench_real.py | FCFS baseline | 5,516,652 | 300.75 | **18,342.89** | — |
| bench_ltr.py | LTR 在线学习 | 5,516,652 | 309.07 | 17,849.44 | -2.7% |
| bench_ltr_freeze.py | LTR 冻结模型 | 5,516,652 | 300.95 | 18,330.86 | -0.07% |

**分析：**
- Freeze 优化有效：去除在线学习开销后，LTR Freeze 与 FCFS 基本持平
- Oracle 测试（完美预测 output_length）证实：按 output_length 做 SJF 不能提升吞吐量
- SJF 优化的是平均延迟而非吞吐量，在 continuous batching 场景下调度顺序对吞吐几乎无影响
- 下一步方向：从 output_length 预测转向 prefill-cost-aware 调度（按 prompt_length 排序，减少 prefill 占比）
