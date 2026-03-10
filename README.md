<p align="center">
<img width="300" src="assets/logo.png">
</p>

# Nano-vLLM-v1

基于 [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 的改进版本，在保持轻量级（~1,600 行 Python）的前提下，引入多项来自 vLLM v1 和 SGLang 的核心优化。

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
- 为后续 Pipeline 并行提供更好的架构基础

#### 3. BlockManager 增强
- 新增 `get_token_layout()` 方法，精确计算 token 在 block 内的布局
- 区分 `num_new_computed_tokens_in_used`（已有 block 中的新 token）和 `in_free`（需要新 block 的 token）
- 实现更精确的内存预测与按需分配

### 规划中

#### 4. 快速路径优化（Zero-Overhead Scheduler）
Decode 阶段当无新请求到达时，跳过完整调度逻辑，直接复用上一轮的 running 序列集合：
- 预期调度开销从 5% 降至 1%
- 预期吞吐量提升 5-8%
- 技术来源：SGLang v0.4

#### 5. Pipeline 并行执行
通过后台线程实现调度与推理的重叠执行，消除 CPU 调度阶段对 GPU 的阻塞：
- 后台线程预先调度下一个 batch，GPU 推理时 CPU 同步工作
- 预期 GPU 利用率从 70% 提升至 85%
- 预期吞吐量提升 15-20%
- 技术来源：vLLM v1 Pipeline 架构

#### 6. BlockManager LRU 淘汰策略
为 Prefix Cache 引入 LRU 淘汰机制，提升缓存命中率：
- 当 KV Cache 空间不足时，优先淘汰最久未使用的 cached block
- 更激进的 prefix cache 复用策略

### TBD

#### 7. Learning to Rank 调度
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

| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |
| Nano-vLLM-v1 (+ Pipeline) | -  | -       | ~1,720 (预期)         |
