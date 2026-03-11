<p align="center">
<img width="300" src="assets/logo.png">
</p>

# Nano-vLLM-v1

基于 [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 的改进版本，在保持轻量级（~1,500 行 Python，原版 ~1,360 行基础上仅增加 ~100 行）的前提下，引入多项来自 vLLM v1 和 SGLang 的核心优化。

## 改进总览

### 已实现

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

### 规划中

#### 5. BlockManager LRU 淘汰策略
为 Prefix Cache 引入 LRU 淘汰机制，提升缓存命中率：
- 当 KV Cache 空间不足时，优先淘汰最久未使用的 cached block
- 更激进的 prefix cache 复用策略

### TBD

#### 6. Learning to Rank 调度
基于 Learning to Rank 的智能请求调度策略，根据序列特征（长度、预估生成长度、缓存命中率等）学习最优调度顺序，替代传统的 FCFS 或简单优先级策略。

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

**测试环境：**
- Hardware: NVIDIA H100 PCIe (80GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: 随机采样 100–1024 tokens
- Output Length: 随机采样 100–1024 tokens
- `enforce_eager=False`, `max_model_len=4096`

**当前版本改动（相对原版 Nano-vLLM）：**
- 统一调度器（不再区分 prefill/decode 阶段）
- Chunked Prefill 支持（通过 `chunked_prefill=True` 启用）
- BlockManager 增强（`get_token_layout`、按 token 数分配）
- Attention 统一 varlen 路径 + flash_attn 2.8.3 CUDA Graph 兼容适配
- Decode 快速路径（纯 decode 无新请求时跳过完整调度逻辑，效率提升最高）

**性能结果：**

| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| Nano-vLLM（原版） | 133,966   | 12.03    | 11,138.68             |
| Nano-vLLM-v1   | 133,966     | 8.76     | 15,294.23 (**+37.3%**) |
