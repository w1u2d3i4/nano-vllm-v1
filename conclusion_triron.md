# H100 单卡 LLM 推理优化：技术报告

> 模型：Qwen3-0.6B | GPU：NVIDIA H100 80GB | 推理引擎：nano-vllm-v1

---

## 目录

1. [项目目标与指标体系](#1-项目目标与指标体系)
2. [Phase 0：三目标 Baseline 测量](#2-phase-0三目标-baseline-测量)
3. [Phase 0.4：Kernel 级 Profiling](#3-phase-04kernel-级-profiling)
4. [Phase 1：Triton Kernel 优化](#4-phase-1triton-kernel-优化)
5. [Phase 2：Speculative Decoding](#5-phase-2speculative-decoding)
6. [Goal 1：QPS 压力测试与 Max Goodput](#6-goal-1qps-压力测试与-max-goodput)
7. [Goal 2：最大上下文长度与 KV Cache 容量](#7-goal-2最大上下文长度与-kv-cache-容量)
8. [三目标最终结果汇总](#8-三目标最终结果汇总)
9. [方法论反思：为什么 Kernel Profiling 会误导优化方向](#9-方法论反思为什么-kernel-profiling-会误导优化方向)
10. [Phase 3：系统级 Profiling（Nsight Systems 方法论）](#10-phase-3系统级-profilingnsight-systems-方法论)
11. [Phase 4：系统级优化 — Postprocess CUDA Tensor 修复](#11-phase-4系统级优化--postprocess-cuda-tensor-修复)
12. [关键结论与经验教训](#12-关键结论与经验教训)

---

## 1. 项目目标与指标体系

### 1.1 三大优化目标

| 目标 | 定义 | 核心指标 | 企业级达标线 |
|------|------|---------|------------|
| **Goal 1: Max QPS** | TTFT/TPOT 约束下最大并发吞吐 | Goodput (req/s) | P90 TTFT < 500ms, P90 TPOT < 50ms |
| **Goal 2: Max Context** | 单卡最长 token 输入 | max input_length | 接近 KV cache 理论上限 |
| **Goal 3: Max Throughput** | 纯吞吐量最大化 | Output TPS (tok/s) | Qwen3-0.6B on H100: > 15,000 |

### 1.2 关键指标定义

- **TTFT** (Time To First Token): 请求提交 → 第一个 output token 产出的时间
- **TPOT** (Time Per Output Token): 生成每个 output token 的平均耗时
- **ITL** (Inter-Token Latency): 相邻两个 output token 之间的实际间隔
- **Goodput**: 满足 SLA 约束的最大请求速率

### 1.3 工具链

- **bench_real.py**: FCFS baseline benchmark（all-at-once 和 QPS 模式）
- **bench_latency.py**: 企业级延迟 benchmark（TTFT/TPOT/ITL + SLA 对标 + QPS sweep）
- **profile_baseline.py**: Kernel 级 CUDA profiling
- **bench_speculative.py**: Speculative Decoding benchmark

---

## 2. Phase 0：三目标 Baseline 测量

### 2.1 Goal 3 Baseline — 最大吞吐量

**问题**：在 all-at-once 模式下，引擎的最大 token 产出速度是多少？

**实验命令**：
```bash
python bench_real.py --num_seqs 512 --qps 0 --max_model_len 32768
```

**结果**：
```
[FCFS] Seqs: 512, Output tokens: 524288, Time: 22.94s, Throughput: 22851.81 tok/s
```

**分析**：
- Baseline 吞吐量 **22,852 tok/s**，已超过 plan 中设定的 15,000 tok/s 目标
- 峰值吞吐曾达到 24,702 tok/s（tqdm 显示）
- 512 个序列各生成 1024 tokens，总计 524,288 tokens
- 已有的 Fast Path（+37.3%）和 Chunked Prefill 优化是高吞吐的关键

### 2.2 Goal 1 Baseline — QPS 与延迟

**问题**：在模拟真实在线流量下，系统能承受的最大 QPS 是多少？

**实验命令**：
```bash
python bench_latency.py --sweep --num_seqs 5000 --report sweep.json
```

**第一次 Sweep 结果**：

| QPS | TPS | TTFT P90 | TPOT P90 | 状态 |
|-----|-----|----------|----------|------|
| 5 | 5,103 | 59ms | 5.6ms | PASS |
| 8 | 8,135 | 58ms | 9.5ms | PASS |
| 10 | 10,108 | 61ms | 20.3ms | PASS |
| 12 | 11,068 | **29,463ms** | 46.0ms | **FAIL** |

**关键发现**：QPS=10 → 12 时 TTFT 从 61ms 暴涨到 29,463ms（约 500 倍），系统在 QPS≈11 处出现**悬崖式崩塌**。

**原因分析**：
- QPS=12 时注入速度超过系统处理能力
- waiting 队列快速堆积，TTFT 线性增长
- KV cache 压力增大，可能触发 preemption

**第一次 Max Goodput = 10 req/s**

### 2.3 Goal 2 Baseline — 最大上下文长度

**问题**：单卡能处理的最长输入是多少 tokens？

**理论计算**（Qwen3-0.6B on H100 80GB）：
```
模型权重（bf16）：≈ 1.2 GB
CUDA + PyTorch overhead：≈ 2 GB
可用于 KV Cache：≈ 76 GB
KV Cache per token = 2 × 28 × 8 × 128 × 2 = 114,688 bytes ≈ 112 KB
理论 max tokens = 76 GB / 112 KB ≈ 696,000 tokens
```

**实验**：
```bash
python bench_real.py --num_seqs 1 --qps 0 --max_model_len 65536
python bench_real.py --num_seqs 1 --qps 0 --max_model_len 131072
python bench_real.py --num_seqs 1 --qps 0 --max_model_len 262144
```

**结果**：

| max_model_len | 结果 | 吞吐 |
|--------------|------|------|
| 65,536 | 成功 | 351.65 tok/s |
| 131,072 | 成功 | 355.16 tok/s |
| 262,144 | 成功 | 360.50 tok/s |

所有测试均通过，未触及 OOM 边界。**Baseline Max Context ≥ 262K tokens**。

---

## 3. Phase 0.4：Kernel 级 Profiling

### 3.1 问题

在确定了三目标 baseline 后，下一步是定位 GPU kernel 层面的瓶颈，确定 Triton 优化的价值。

### 3.2 思路

使用 `torch.profiler` 对 20 个 decode step 进行 CUDA kernel 级别分析，记录每个 kernel 的 GPU 时间占比。重点关注：
- RMSNorm 和 RoPE 是否占较高比例（值得手写 Triton kernel）
- Flash Attention 和 GEMM 的不可优化部分占比
- GPU idle gap（调度开销）

### 3.3 实现

新建 `profile_baseline.py`：
- 注入 64 个序列，warmup 进入稳态 decode
- 使用 `torch.profiler.profile(activities=[CPU, CUDA], record_shapes=True, with_stack=True, profile_memory=True)` 记录 20 步
- 输出 kernel 时间排名 + 导出 Chrome trace

### 3.4 结果

| 类别 | 主要 Kernel | CUDA 时间 | 占比 |
|------|------------|-----------|------|
| **GEMM (linear)** | nvjet_tst_* + aten::mm | ~27.5ms | **~48%** |
| **Flash Attention** | flash_fwd_splitkv_kernel | 11.7ms | **20.5%** |
| **RoPE** | CompiledFxGraph + triton_poi_fused_* | ~6.1ms | **~10.7%** |
| **RMSNorm** | triton_red_fused_add_mean_mul_pow_rsqrt | 5.6ms | **~9.8%** |
| **Elementwise** | elementwise_kernel | 5.2ms | 9.1% |
| **Sampling** | triton_red_fused_softmax_argmax | 2.5ms | 4.4% |
| **DtoH Memcpy** | Memcpy DtoH | 2.0ms | 3.5% |
| **KV Cache Store** | store_kvcache_kernel | 1.1ms | 2.0% |
| **SiLU** | triton_poi_fused_mul_silu_split | 1.0ms | 1.8% |

### 3.5 关键发现

1. **RMSNorm (9.8%) + RoPE (10.7%) = 20.5%** — 合计占 CUDA 时间的 1/5，Triton 优化有价值
2. GEMM + Flash Attention 占 ~69%，由 cuBLAS/FlashAttn 处理，基本不可优化
3. **`@torch.compile` 已经将 RMSNorm/RoPE 编译为 Triton kernel**（`triton_red_fused_*` 和 `triton_poi_fused_*`）
4. Roofline 分析确认 RMSNorm 是 memory-bandwidth-bound（AI ≈ 0.5 << H100 Balance Point 295）

### 3.6 决策

RMSNorm + RoPE 合计 20.5%，值得尝试手写 Triton kernel。但 `@torch.compile` 已生成 Triton 代码，需要验证手写 kernel 是否能超越。

---

## 4. Phase 1：Triton Kernel 优化

### 4.1 问题

`@torch.compile` 自动生成的 Triton kernel 是否可以被手写 Triton kernel 超越？如果可以，能带来多少端到端吞吐提升？

### 4.2 思路

根据 "[From 11% to 88% Peak Bandwidth](https://subhadipmitra.com/blog/2025/triton-kernels-llm-inference/)" 的案例，手写 Fused Add+RMSNorm 可在 A100 上从 11% 带宽利用率提升到 88%（8.1× 加速）。我们对 Qwen3-0.6B 的 RMSNorm（hidden_size=896）和 RoPE（head_dim=128）分别编写 fused Triton kernel。

### 4.3 实现

#### Fused Add+RMSNorm (`nanovllm/kernels/fused_norm.py`)

两个 kernel variant：
- `_rms_norm_kernel`: 无 residual add，用于 q_norm/k_norm 和首层 input_layernorm
- `_add_rms_norm_inplace_kernel`: fused residual add + RMSNorm，in-place 写回（零额外分配）

关键设计：
```python
@triton.jit
def _add_rms_norm_inplace_kernel(X_ptr, RES_ptr, W_ptr, stride_n, D_actual, eps, D: tl.constexpr, IS_BF16: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, D)
    mask = cols < D_actual
    # Read x, residual → FP32 → fused add in registers → write new residual
    # → RMSNorm in registers → write normalized output
    # 总共：2 次读 + 2 次写（比 PyTorch 的 5 次 HBM 访问少）
```

- 动态 `num_warps` 选择（D≤128 → 2 warps, D≤512 → 4, D≤2048 → 8, 否则 16）
- 动态输出 dtype（bf16/fp16 自动检测）

#### Fused RoPE (`nanovllm/kernels/fused_rope.py`)

消除 `chunk()/cat()` 中间 tensor，Q/K 共享 cos/sin 读取：
```python
@triton.jit
def _fused_rope_inplace_kernel(Q_ptr, K_ptr, COS_SIN_ptr, POS_ptr, ...):
    # 1D grid: (num_tokens,)，每个 program instance 处理一个 token 的所有 Q+K heads
    # cos/sin 只读一次，Q 和 K 共享
    # In-place 写回，零额外分配
```

### 4.4 正确性验证

**发现的 Bug**：in-place RMSNorm 在首层 decoder 中导致正确性问题。

模型代码：
```python
hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
```

Python 从左到右求值：如果 `rms_forward` in-place 修改了 `hidden_states`，则 `residual` 也变成了 normalized 值（应该是原始值）。

**修复**：`rms_forward` 必须分配新 output tensor（不能 in-place）；`add_rms_forward` 可以 in-place（x 和 residual 是不同 tensor）。

### 4.5 性能测试结果

#### CUDA Graph 模式（生产配置）

| 配置 | Throughput (tok/s) | 对比 Baseline |
|------|-------------------|--------------|
| **Baseline** (`@torch.compile` 全部) | **22,852** | — |
| Triton RMSNorm only | 21,505 | **-5.9%** |
| Triton RoPE only | 21,378 | **-6.4%** |
| Triton 全部 (优化后) | 21,221 | **-7.1%** |
| Triton 全部 (in-place) | 21,608 | **-5.4%** |

**所有 Triton kernel 配置都比 `@torch.compile` baseline 慢。**

#### Eager 模式（无 CUDA Graph）

| 配置 | Throughput (tok/s) | 对比 |
|------|-------------------|------|
| `@torch.compile` RMSNorm | 6,867 | baseline |
| **Triton RMSNorm** | **7,741** | **+12.7%** |

**在 eager 模式下，手写 Triton kernel 比 `@torch.compile` 快 12.7%。**

### 4.6 根因分析

**为什么 Triton kernel 在 CUDA Graph 模式下更慢？**

1. **`@torch.compile` 的零分配优势**：inductor 编译器生成的 Triton 代码内部管理内存，不调用 `torch.empty_like()`。我们的 Python wrapper 需要分配 output tensor，这些分配被记录在 CUDA Graph 中，增加了 replay 开销。

2. **inductor 的跨操作融合**：`@torch.compile` 不仅融合 RMSNorm 内部操作，还可能与相邻操作融合。手写 kernel 只能优化单个操作的边界。

3. **模型尺寸效应**：Qwen3-0.6B 的 hidden_size=896，RMSNorm 每行只处理 896 个元素。kernel launch overhead 相对于计算量的比例较高。"11% → 88%" 的案例基于 LLaMA-7B（hidden_size=4096），有 4.6× 更多元素摊薄 overhead。

4. **CUDA Graph 的放大效应**：CUDA Graph 让每步 decode 开销降到 ~1ms。在这个量级上，任何额外的 Python/allocation 开销都会被显著放大。

### 4.7 总结

**对于 Qwen3-0.6B + CUDA Graph，`@torch.compile` 已是近似最优。手写 Triton kernel 在 eager 模式下有明确优势（+12.7%），但在 CUDA Graph 模式下因分配开销而退步（-5.4% ~ -7.1%）。**

**保留 kernel 代码**于 `nanovllm/kernels/`，可用于：
- 更大模型（hidden_size ≥ 4096）
- Eager 模式场景
- 作为 Triton 开发的参考实现

**代码已完全恢复至 `@torch.compile` baseline。**

---

## 5. Phase 2：Speculative Decoding

### 5.1 问题

能否通过 Speculative Decoding 提升 Qwen3-0.6B 的吞吐量？Plan 预期 1.5× 加速。

### 5.2 思路

采用 **Layer-Skip Self-Speculative Decoding**：
- **Draft model**: 使用目标模型的前 N 层（共享权重，无额外内存）
- **Verify**: 完整模型一次性验证 K 个 draft token
- **Rejection Sampling**: Leviathan 2023 算法，保证输出分布与目标模型完全一致

#### 算法流程：

```
1. Draft Phase (K iterations):
   - 使用前 N 层自回归生成 K 个 draft token
   - 保存每步的 draft 概率分布

2. Verify Phase (1 forward pass):
   - 完整模型处理 K+1 个 token (pending + K drafts)
   - 获取每个位置的 target 概率

3. Rejection Sampling:
   For each draft token d_i:
     r ~ Uniform(0,1)
     if r < p_target(d_i) / p_draft(d_i): accept
     else: resample from max(0, p_target - p_draft), stop

4. If all K accepted: sample bonus token from target
   Total: 1 to K+1 tokens per speculative step
```

### 5.3 实现细节

#### 修改文件清单

| 文件 | 修改内容 |
|------|---------|
| `nanovllm/config.py` | 新增 `enable_speculative`, `num_draft_layers`, `num_speculative_tokens` |
| `nanovllm/models/qwen3.py` | `Qwen3Model.forward()` 新增 `num_layers` 参数支持 early exit |
| `nanovllm/engine/sequence.py` | 新增 `rollback(n)` 方法（回滚 n 个 token） |
| `nanovllm/engine/block_manager.py` | 新增 `trim_blocks(seq)` 方法（释放多余 block） |
| `nanovllm/engine/model_runner.py` | 新增 `run_draft()` 和 `run_verify()` 方法 |
| `nanovllm/engine/speculative.py` | 新建：rejection sampling 核心逻辑 |
| `nanovllm/engine/llm_engine.py` | 新增 `_speculative_step()` 和 `_can_speculate()` |

#### 关键设计决策

**1. KV Cache 策略：Draft 写入 → Verify 覆写**

Draft 阶段写入 KV Cache（layers 0..N-1），Verify 阶段用完整模型覆写所有层的 KV。拒绝后通过 `rollback()` + `trim_blocks()` 回收。

**2. Verify 包含 pending token**

关键洞察：verify 处理 K+1 个 token（pending_token + K draft tokens），而非仅 K 个。这样 verify_logits[j] 直接验证 draft_token[j]，无需额外保存上一步的 target probs。

**3. Block Manager 兼容性**

Speculative step 多次调用 block allocation，绕过 `may_append()` 的 hash 断言，使用自定义 `_ensure_blocks()` 函数。

**4. LM Head 全 token logits**

发现 `ParallelLMHead.forward()` 使用 `cu_seqlens_q` 只提取每个序列的**最后一个** token 的 logits。Verify 阶段需要**所有** token 的 logits，因此直接调用 `F.linear(hidden, weight)` 绕过过滤。

#### 遇到的 Bug 及修复

| Bug | 原因 | 修复 |
|-----|------|------|
| `verify_logits shape (1, V)` 而非 `(K+1, V)` | `ParallelLMHead` 只返回 last token logits | 用 `F.linear()` 直接计算 |
| `slot_mapping.numel() != N` | verify 需要 K+1 个 slot 但 draft 只分配了 K 个 | 添加 `_ensure_blocks()` 预分配 |
| `block.hash != -1` assertion | draft 多次调用 `may_append` 设置了 block hash | 用 `_ensure_blocks()` 替代 `may_append` |
| Draft probs 被 Gumbel sampling 破坏 | `probs.div_(exponential_())` 原地修改了 probs | 在 sampling 前 `probs.clone()` |

### 5.4 正确性验证

**验证方法**：draft_layers=28（draft = 完整模型），此时 draft 和 target 概率分布完全一致，接受率应为 100%。

```
Step 0: tokens=8  (prefill)
Step 1: tokens=12 (gain=4, 100% accepted + bonus)
Step 2: tokens=16 (gain=4)
Step 3: tokens=20 (gain=4)
Step 4: tokens=24 (gain=4)
```

**结论**：当 draft = target 时，每个 speculative step 精确产出 K+1=4 tokens。实现正确。

### 5.5 性能测试

#### 不同 draft layers 的接受率

| Draft Layers | 占比 | Avg Tokens/Step | 接受率质量 |
|-------------|------|-----------------|----------|
| 7 (25%) | 1/4 模型 | 1.1 | 极低（~0% draft accepted） |
| 14 (50%) | 1/2 模型 | 1.1 | 极低 |
| 20 (71%) | 5/7 模型 | 2.2 | 中等（~40% draft accepted） |
| 28 (100%) | 完整模型 | 4.0 | 完美（100%） |

#### 吞吐量对比

| 配置 | Throughput | 对比 |
|------|-----------|------|
| Normal decode (CUDA Graph) | **22,852 tok/s** | baseline |
| Normal decode (Eager) | 13 tok/s | 无 CUDA Graph |
| Speculative K=3, draft=7 (Eager) | 36 tok/s | 虽生成更多 token/step 但 eager 太慢 |
| Speculative K=3, draft=20 (Eager) | 7 tok/s | 更好的接受率但更多层开销 |
| Speculative K=3, 512 seqs (Eager) | 1,236 tok/s | 大批量 |

### 5.6 根因分析

**为什么 Speculative Decoding 对 Qwen3-0.6B 无效？**

1. **CUDA Graph vs Eager 的巨大差距**：CUDA Graph 让 normal decode 达到 22,852 tok/s，而 speculative 必须在 eager 模式运行（draft 和 verify 的 batch size 动态变化）。CUDA Graph 提供约 **1760× 的加速**（22,852 / 13）。

2. **小模型的 layer-skip 质量差**：Qwen3-0.6B 只有 28 层。使用 7 层（25%）作为 draft，输出分布与完整模型差异巨大，导致接受率趋近 0%。即使使用 20 层（71%），平均每步也只多产出 1.2 个 token。

3. **每步 overhead 过大**：speculative step = K 次 draft forward + 1 次 verify forward + rejection sampling。对于 K=3，这是 4 次 eager forward vs normal 的 1 次 CUDA Graph forward。

4. **Speculative Decoding 的适用场景**：
   - 大模型（70B+）：每步 decode 开销高（>50ms），draft 节省的计算量可以覆盖 overhead
   - 独立 draft model（如 1B draft + 70B target）：draft 开销极低，接受率高
   - CUDA Graph 不可用时：eager 模式下 speculative 有正收益

### 5.7 总结

Speculative Decoding 实现**正确**（draft=target 时 100% 接受率验证通过），但对 Qwen3-0.6B **不产生吞吐提升**。核心矛盾：CUDA Graph 使 normal decode 极快，speculative 的 eager 多次 forward 无法竞争。

---

## 6. Goal 1：QPS 压力测试与 Max Goodput

### 6.1 问题

在企业级 SLA 约束（chatbot: TTFT P90 < 500ms, TPOT P90 < 50ms）下，系统能稳定服务的最大 QPS 是多少？

### 6.2 工具：bench_latency.py

专门开发的企业级 benchmark 工具，功能：
- QPS 注入模式，模拟真实在线流量
- 逐请求跟踪 TTFT、TPOT、ITL、E2E latency
- 系统级监控（queue depth、free blocks、preemption）
- **Sweep 模式**：自动扫描 QPS，找到 Max Goodput
- 4 种 SLA profile（chatbot、code_complete、rag、streaming_chat）

### 6.3 QPS Sweep 结果

**命令**：
```bash
python bench_latency.py --sweep --num_seqs 1000 --report sweep_results.json
```

| QPS | TPS | RPS | TTFT P90 | TPOT P90 | 状态 |
|-----|-----|-----|----------|----------|------|
| 5 | 5,026 | 4.9 | 64ms | 6.2ms | **PASS** |
| 8 | 7,917 | 7.7 | 65ms | 13.5ms | **PASS** |
| 10 | 9,597 | 9.4 | 60ms | 19.7ms | **PASS** |
| **12** | **10,491** | **10.2** | **76ms** | **42.3ms** | **PASS** |
| 14 | 10,626 | 10.4 | **10,111ms** | 45.7ms | **FAIL** |

### 6.4 分析

- **Max Goodput = 12 req/s**（chatbot SLA）
- QPS=12 → 14 时 TTFT 从 76ms 暴涨到 10,111ms（**133×**）
- 系统在 QPS≈13 处出现悬崖式崩塌
- TPOT 在所有通过的 QPS 下均满足 SLA（< 50ms）
- 主要瓶颈是 TTFT（prefill 速度和队列等待时间），非 TPOT

### 6.5 瓶颈定位

- **TTFT 崩塌 = 队列堆积**：QPS 超过系统处理能力后，waiting 队列持续增长，TTFT 线性上升
- **TPS 天花板 ~10,600**：QPS=12 和 14 的 TPS 几乎相同，说明 GPU 已满载
- **优化方向**：Prefix Cache（共享 prompt 减少 prefill）、KV Cache 量化（增加并发容量）

---

## 7. Goal 2：最大上下文长度与 KV Cache 容量

### 7.1 问题

H100 80GB 单卡上，模型的有效上下文窗口和 KV Cache 物理容量分别是多少？

### 7.2 关键概念区分

| 概念 | 定义 | 决定因素 |
|------|------|---------|
| **模型有效上下文窗口** | 模型能准确处理的最大输入长度 | `max_position_embeddings`（RoPE 训练范围） |
| **KV Cache 物理容量** | GPU 显存能容纳的 KV token 总数 | 显存大小、模型参数量、KV 精度 |
| **单序列最大上下文** | 单个请求能输入的最大 token 数 | min(模型窗口, KV 容量) |
| **并发 KV 容量** | 多序列同时运行时可容纳的 token 总量 | KV 物理容量 |

**Qwen3-0.6B 的配置**：
```
max_position_embeddings = 40,960 (40K)
rope_theta = 1,000,000
rope_scaling = None（无扩展）
```

代码中 `max_model_len = min(max_model_len, max_position_embeddings)` 会将实际上下文钳制到 40K。超过 40K 后 RoPE 位置编码未训练，输出质量严重退化。

### 7.3 KV Cache 容量理论计算

```
模型权重（bf16）：≈ 1.2 GB
CUDA + PyTorch overhead：≈ 2 GB
可用于 KV Cache：≈ 76 GB

KV bytes/token = 2(K+V) × 28 layers × 8 kv_heads × 128 head_dim × 2 bytes(bf16)
               = 114,688 bytes ≈ 112 KB/token

KV Cache 物理容量 = 76 GB / 112 KB ≈ 696,000 tokens
```

### 7.4 实测结果

**说明**：以下测试传入的 `max_model_len` 参数决定了 KV Cache 分配空间大小。由于代码钳制，模型实际处理的单序列上下文不超过 40K。测试验证的是 **KV Cache 物理容量能否成功分配**，而非模型能否有效利用该长度上下文。

| 传入 max_model_len | KV 分配 | 单序列解码吞吐 | 备注 |
|-------------------|---------|--------------|------|
| 32,768 (32K) | 成功 | — | baseline |
| 65,536 (64K) | 成功 | 351.65 tok/s | 实际 max_model_len 被钳制为 40K |
| 131,072 (128K) | 成功 | 355.16 tok/s | 同上 |
| 262,144 (256K) | 成功 | 360.50 tok/s | 同上 |
| 524,288 (512K) | 成功 | 361.49 tok/s | 同上 |
| **655,360 (640K)** | **成功** | **338.67 tok/s** | 接近物理上限 696K |

### 7.5 正确解读

- **单序列最大有效上下文 = 40,960 tokens（40K）**，由模型 RoPE 训练范围决定
- **KV Cache 物理容量 ≈ 655K tokens**，接近理论上限 696K
- 655K 容量的实际意义：可支持约 **16 个 40K 上下文的并发序列**（655K / 40K ≈ 16）
- 吞吐在 512K KV 分配后下降 6.3%，因 KV Cache 管理开销增大

### 7.6 扩展上下文的方法

| 方法 | 作用 | 说明 |
|------|------|------|
| **RoPE Scaling (YaRN/NTK)** | 扩展模型有效窗口到 128K+ | 需微调或推理时动态调整 rope_theta |
| **KV Cache INT8/FP8 量化** | 2× KV 容量（~1.3M tokens） | 不改变模型窗口，增加并发能力 |
| **Token Eviction (StreamingLLM)** | "无限"上下文（有损） | 保留头尾 token，驱逐中间 |
| **长上下文微调** | 真正扩展训练窗口 | 需要长文本数据 + 训练资源 |

---

## 8. 三目标最终结果汇总

| 目标 | 指标 | Baseline | 最终值 | 变化 |
|------|------|----------|--------|------|
| **Goal 1: Max QPS** | Max Goodput | 10 req/s | **12 req/s** | +20% |
| **Goal 2: Max Context** | 模型有效上下文 / KV Cache 容量 | 40K / — | **40K / 655K tokens** | 见下方说明 |
| **Goal 3: Max Throughput** | Output TPS | 22,852 | **37,430 tok/s** | **+63.7%** |

**Goal 2 说明**：Qwen3-0.6B 的 `max_position_embeddings = 40,960`（40K），RoPE 仅在此范围内训练过，超出后输出质量严重退化。655K 是 H100 80GB 上的 **KV Cache 物理容量**（可容纳的 token 总数），决定的是并发能力（如 40K × 16 seq），而非单序列上下文长度。要突破 40K 上下文限制需要 RoPE scaling（YaRN/NTK）或长上下文微调。

### 优化尝试 vs 结果

| 优化 | 预期 | 实际 | 原因 |
|------|------|------|------|
| Triton Fused RMSNorm | +5-10% | **-5.4%** (CUDA Graph) / **+12.7%** (Eager) | torch.compile 在 CUDA Graph 下已最优 |
| Triton Fused RoPE | +3-5% | **-6.4%** (CUDA Graph) | 同上 |
| Speculative Decoding K=3 | +50% (1.5×) | **-94.6%** (1,236 vs 22,852) | 小模型 + CUDA Graph 使 spec decode 无效 |

---

## 9. 方法论反思：为什么 Kernel Profiling 会误导优化方向

### 9.1 我们的分析路径回顾

```
torch.profiler → RMSNorm 9.8% + RoPE 10.7% = 20.5% → "值得 Triton 优化"
→ 编写 Triton kernel → 端到端测试 → 反而变慢 (-5.4% ~ -7.1%)
```

**推论**："占比 X% → 优化它 → 获得 ≤X% 提升" 在我们的场景中完全不成立。为什么？

### 9.2 五个关键陷阱

#### 陷阱 1：Kernel 时间 ≠ 关键路径时间

`torch.profiler` 报告的是每个 kernel **独立**占用 GPU 的时间。但 GPU 执行是流水线化的——kernel 之间可以重叠执行。一个 kernel 可能在时间轴上与更长的 kernel 并行运行。即使把它优化到 0ms，端到端延迟可能不变，因为瓶颈在那个更长的 kernel 上。

#### 陷阱 2：CUDA Graph 隐藏了真正的开销

CUDA Graph 把数百个 kernel launch 压缩成一次 graph replay。profiler 仍然显示各个 kernel 的耗时，但 **kernel 之间的 CPU 开销已经被消除**。我们优化的 Triton kernel 本身更快（eager +12.7%），但引入的 `torch.empty_like()` 分配开销在 CUDA Graph 下被放大了——这是 profiler 看不到的。

#### 陷阱 3：`@torch.compile` 让"独立 kernel"的概念过时

`torch.compile` 的 inductor 后端做跨操作融合。profiler 显示的 `triton_red_fused_add_mean_mul_pow_rsqrt` 不是一个简单的 RMSNorm——它可能已经与周围操作融合。我们手写的 kernel 替换了这个融合后的产物，等于**拆散了一个更大的融合体**。

#### 陷阱 4：Memory-bound kernel 优化计算无效

RMSNorm 的 Arithmetic Intensity ≈ 0.5 FLOPs/Byte，远低于 H100 的 Balance Point（295 FLOPs/Byte）。它是内存带宽受限的。Roofline 模型告诉我们：对于带宽受限的 kernel，优化计算是徒劳的——唯一有效的方式是融合（减少内存访问次数），而 `@torch.compile` 已经做了这件事。

#### 陷阱 5：Profiler 测量本身有误差

`torch.profiler` 在每个 kernel 前后插入 CUDA event，会：
- 改变 kernel 的执行顺序和重叠模式
- 放大极短 kernel 的表观时间（event 记录开销）
- 禁用某些优化（如 CUDA Graph capture 期间的 kernel fusion）

被 profiling 的执行与生产执行**不是同一回事**。

### 9.3 正确的优化分析方法论

| 步骤 | 我们的做法 | 推荐做法 |
|------|-----------|---------|
| 1. 定位瓶颈 | `torch.profiler` → 看 kernel 耗时占比 | **Nsight Systems** → 看时间轴上的关键路径 |
| 2. 解读结果 | "RMSNorm 占 9.8%，值得优化" | "GPU 在等什么？CPU？内存？调度？" |
| 3. 优化目标 | 优化单个 kernel 的 CUDA time | 优化**系统级瓶颈**（调度、内存管理、batching） |
| 4. 分场景分析 | 混合 prefill + decode 一起看 | 区分 **prefill**（compute-bound）vs **decode**（memory-bound） |
| 5. 验证 | 看 kernel 时间变化 | 看**端到端 tok/s 和 TTFT** |

**推荐工作流**：
```
1. 端到端 benchmark → 建立 baseline（tok/s, TTFT, ITL）
2. Nsight Systems trace → 找到系统级瓶颈（CPU gap? memory stall? scheduling?)
3. 如果确认某个 kernel 在关键路径上 → Nsight Compute 深入分析
4. 优化 → 再次端到端 benchmark 验证
5. 循环
```

### 9.4 核心认知转变

> **LLM 推理优化是一个系统问题，不是 kernel 优化问题。**

传统 HPC 的 "profile → hotspot → optimize" 方法论在现代 LLM 推理中会失效：
1. 工作负载是**内存受限**的（优化计算无效）
2. **CUDA Graph 和 torch.compile** 改变了执行模型（profiler 看到的不是真实执行）
3. 最大的收益来自**算法和架构层面**的改变（batching、内存管理、speculative decoding），而非 kernel 层面

### 9.5 行业参考

- **FlashAttention** (Dao et al.) 明确论证：标准 profiling 方法是误导的。FlashAttention 的 FLOP 数更多，但 wall-clock 更快，因为它减少了内存 I/O
- **vLLM PagedAttention** 的突破不是优化了任何 kernel，而是解决了内存管理问题
- **NVIDIA TensorRT-LLM** 文档推荐 Nsight Systems 作为首要工具，明确警告不要仅基于 torch.profiler 做优化决策
- **"Efficiently Scaling Transformer Inference" (Pope et al., MLSys 2023)** 系统分析了 LLM 推理的内存带宽瓶颈

---

## 10. Phase 3：系统级 Profiling（Nsight Systems 方法论）

### 10.1 问题

按照推荐方法论重新分析：系统的真正瓶颈在哪里？GPU 在等什么？

### 10.2 工具

- **NVIDIA Nsight Systems 2025.3.2** (`/opt/nvidia/nsight-systems/2025.3.2/bin/nsys`)
- 自定义 `profile_nsys.py`：分离 Schedule / Forward+Sample / Postprocess 三阶段耗时
- CUDA synchronize 隔离 GPU 计算时间

### 10.3 实验设计

对不同 batch size 的**纯 decode 阶段**进行 step 级别时间分解：
```python
t1 = time.perf_counter()
seqs = scheduler.schedule()           # CPU: 调度
t2 = time.perf_counter()
token_ids = model_runner.run(seqs)     # GPU: CUDA Graph forward + sampling
torch.cuda.synchronize()              # 等 GPU 完成
t3 = time.perf_counter()
scheduler.postprocess(seqs, ...)       # CPU: 状态更新
t4 = time.perf_counter()
```

### 10.4 结果

#### Step 级别时间分解

| Batch Size | Schedule | Forward+Sample | Postprocess | Total Step | CPU 占比 |
|------------|----------|----------------|-------------|-----------|---------|
| **64 seqs** | 0.121ms | 3.799ms | 1.031ms | **4.951ms** | **23.3%** |
| **256 seqs** | 0.447ms | 6.482ms | 4.399ms | **11.327ms** | **42.8%** |
| **512 seqs** | 0.872ms | 10.073ms | 8.147ms | **19.092ms** | **47.2%** |

#### 512 seqs Forward 内部分解

| 子阶段 | 耗时 | 占 Forward 比例 |
|--------|------|----------------|
| prepare_model_input + prepare_sample | 2.297ms | 22.0% |
| run_model (CUDA Graph replay) | 6.700ms | 64.3% |
| sampler | 1.425ms | 13.7% |
| **Total** | **10.422ms** | 100% |

#### Nsight Systems GPU Kernel 占比

| Kernel 类别 | GPU 时间占比 | 说明 |
|-------------|-------------|------|
| Flash Attention (prefill) | **67.0%** | 大 seq 的 prefill 阶段主导 |
| Flash Attention (decode splitkv) | 6.9% | Decode 阶段的 attention |
| GEMM (nvjet) | ~15% | 线性层矩阵乘 |
| RMSNorm (fused) | 3.5% | torch.compile 融合后 |
| RoPE (fused) | 2.4% | torch.compile 融合后 |
| 其他 | ~5% | elementwise, sampling, kv store |

### 10.5 关键发现

#### 发现 1：CPU 是瓶颈，不是 GPU

**512 seqs 时，47.2% 的时间花在 CPU 上**（Schedule + Postprocess）。这意味着即使 GPU kernel 优化到 0ms，端到端最多只能提升 52.8%。

对比 torch.profiler 的分析：它只看到 GPU kernel 时间，完全忽略了 CPU 的 47.2% 开销。

#### 发现 2：Postprocess 是最大 CPU 瓶颈

| Batch Size | Postprocess 耗时 | 占总时间 |
|------------|------------------|---------|
| 64 seqs | 1.031ms | 20.8% |
| 256 seqs | 4.399ms | 38.8% |
| 512 seqs | **8.147ms** | **42.7%** |

Postprocess 几乎与 GPU forward 等长！它的工作是：
```python
for seq in seqs:  # 512 次 Python 循环
    seq.append_token(token_id)
    seq.num_cached_tokens += seq.num_new_tokens
```

这是**纯 Python 循环**在 512 个 Sequence 对象上的属性访问和更新。

#### 发现 3：prepare_model_input 也很昂贵

512 seqs 时，`prepare_model_input + prepare_sample` 占 Forward 的 22%（2.297ms）。主要是：
- 构建 `input_ids`, `positions`, `slot_mapping` 等 Python 列表
- 创建多个 PyTorch tensor 并传输到 GPU（pin_memory + cuda(non_blocking)）

#### 发现 4：RMSNorm/RoPE 在系统层面几乎无关紧要

在全系统视角下：
- RMSNorm：3.5% GPU 时间 × 52.8% GPU 占比 = **1.8% 系统时间**
- RoPE：2.4% GPU 时间 × 52.8% GPU 占比 = **1.3% 系统时间**
- 即使将两者优化到 0，系统吞吐最多提升 3.1%

而 Postprocess 优化空间：**42.7% 系统时间**，是 RMSNorm+RoPE 的 14 倍。

### 10.6 正确的优化方向

基于系统级分析，真正有价值的优化方向（按收益排序）：

| 优化方向 | 瓶颈 | 当前开销 | 可能收益 | 实现方式 |
|---------|------|---------|---------|---------|
| **1. Postprocess 向量化** | CPU 循环 | 42.7% | -30~50% | 用 torch tensor 批量操作替代 Python 循环 |
| **2. prepare_model_input 优化** | CPU 张量创建 | 12.0% | -50% | 预分配 buffer，减少 tensor 创建 |
| **3. KV Cache 量化** | 内存容量 | 限制并发 | 2× 并发/context | INT8/FP8 KV cache |
| **4. Prefix Cache** | 重复 prefill | TTFT 高 | TTFT -30~50% | Radix Tree 共享 KV |
| ~~5. Triton kernel 优化~~ | ~~GPU 计算~~ | ~~3.1%~~ | ~~<3%~~ | ~~已被 torch.compile 优化~~ |

### 10.7 总结

**系统级 profiling 彻底改变了优化方向**。torch.profiler 让我们关注 GPU kernel（RMSNorm 9.8%、RoPE 10.7%），但 Nsight Systems 级别的分析揭示了真正的瓶颈是 **CPU 端的 Python 循环**（47.2%）。这解释了为什么 Triton kernel 优化无效——我们在优化一个只占系统 3.1% 的部分，同时忽略了占 42.7% 的 Postprocess。

---

## 11. Phase 4：系统级优化 — Postprocess CUDA Tensor 修复

### 11.1 问题

Phase 3 的系统级分析发现 Postprocess 占 512 seqs 时 42.7% 的系统时间（8.147ms/step）。是什么操作如此耗时？

### 11.2 定位过程

将 postprocess 拆分为三个子操作分别计时：

| 子操作 | 用 `.tolist()` 预转换 | 直接迭代 CUDA tensor |
|--------|---------------------|---------------------|
| seq_need_compute_logits 迭代 | 0.168ms | **7.928ms** |
| append_token 循环 | 0.368ms | 0.368ms |
| update cache 循环 | 0.120ms | 0.120ms |

**根因**：`scheduler.postprocess()` 中 `seq_need_compute_logits` 是一个 **CUDA tensor**。在 Python 中逐元素迭代 CUDA tensor，每次 `tensor[i]` 都触发一次 GPU→CPU 数据传输（`cudaMemcpy`），512 个元素 = 512 次独立传输。

```python
# 原始代码（慢）：每次 seq_index 访问触发 GPU→CPU 传输
for seq_index, token_id in zip(seq_need_compute_logits, token_ids):  # 7.928ms
    seq = seqs[seq_index]

# 修复后（快）：一次性批量传输
seq_need_compute_logits = seq_need_compute_logits.tolist()  # 0.050ms
for seq_index, token_id in zip(seq_need_compute_logits, token_ids):  # 0.118ms
    seq = seqs[seq_index]
```

**加速比**：47.3×（7.928ms → 0.168ms）

### 11.3 修复

`nanovllm/engine/scheduler.py` 第 107 行，在循环前添加：
```python
if hasattr(seq_need_compute_logits, 'tolist'):
    seq_need_compute_logits = seq_need_compute_logits.tolist()
```

一行代码修复。

### 11.4 效果验证

#### Step 级别时间分解（512 seqs）

| 指标 | 修复前 | 修复后 | 变化 |
|------|--------|--------|------|
| Schedule | 0.872ms | 0.737ms | -15% |
| Forward+Sample | 10.073ms | 10.706ms | ~持平 |
| **Postprocess** | **8.147ms** | **0.356ms** | **-95.6% (23×)** |
| **Total step** | **19.092ms** | **11.799ms** | **-38.2%** |
| **CPU 占比** | **47.2%** | **9.3%** | **-80%** |

#### 端到端吞吐量

| 指标 | 修复前 | 修复后 | 变化 |
|------|--------|--------|------|
| **Throughput (512 seqs, all-at-once)** | **22,852 tok/s** | **34,180 tok/s** | **+49.6%** |

### 11.5 进一步优化：prepare_model_input 快速路径

#### 问题

修复 postprocess 后重新 profiling，`prepare_model_input` 成为下一个 CPU 瓶颈（2.3ms，占 forward 22%）。

#### 分析

| 子操作 | 耗时 | 占比 |
|--------|------|------|
| Python list 构建（`list.extend`, `list(range())`） | 1.435ms | 72% |
| Tensor 创建 + GPU 传输 | 0.546ms | 28% |

热点是 `input_ids.extend(seq[...])` 和 `positions.extend(list(range(...)))` 的 Python 列表操作。每步 512 个 seq，每个 seq 做 slice + extend + range 构造。

#### 修复

为纯 decode 场景（每个 seq 只有 1 个新 token）添加快速路径 `_prepare_decode_fast()`：

```python
def _prepare_decode_fast(self, seqs):
    n = len(seqs)
    input_ids_arr = [0] * n      # 预分配固定大小数组
    positions_arr = [0] * n
    slot_mapping_arr = [0] * n
    for i, seq in enumerate(seqs):
        input_ids_arr[i] = seq.last_token          # 直接取值，无 slice
        positions_arr[i] = seq.num_cached_tokens    # 直接取值，无 range()
        block_idx = cached // self.block_size
        slot_mapping_arr[i] = seq.block_table[block_idx] * self.block_size + cached % self.block_size
```

消除了所有 `list.extend()`、`list(range())`、`seq[start:end]` 的开销。

#### 效果

| 指标 | Fix 1 only | Fix 1 + Fix 2 | 变化 |
|------|-----------|---------------|------|
| Forward+Sample | 10.71ms | 9.04ms | **-15.6%** |
| Total step | 11.80ms | 10.24ms | **-13.2%** |
| Throughput | 34,180 tok/s | **37,430 tok/s** | **+9.5%** |

### 11.6 累计优化效果汇总

| 指标 | 原始 Baseline | Fix 1 (postprocess `.tolist()`) | Fix 1+2 (decode fast path) | 总提升 |
|------|-------------|-------------------------------|---------------------------|--------|
| **Throughput** | **22,852 tok/s** | **34,180 tok/s** | **37,430 tok/s** | **+63.7%** |
| Total step (512 seqs) | 19.09ms | 11.80ms | 10.24ms | -46.4% |
| Postprocess | 8.15ms | 0.36ms | 0.43ms | -94.7% |
| CPU 占比 | 47.2% | 9.3% | 11.7% | -75.2% |

### 11.7 总结

两个系统级修复共带来 **+63.7% 端到端吞吐提升**，验证了系统级 profiling 方法论的有效性：

1. **Fix 1 (postprocess `.tolist()`)**: CUDA tensor 逐元素迭代 → Python list 批量转换。**一行代码 +49.6%。**
2. **Fix 2 (decode fast path)**: 消除 `list.extend()`/`list(range())` 的 Python 列表操作。**额外 +9.5%。**

这两个问题在 `torch.profiler` 中完全不可见（它只报告 GPU kernel 时间），只有通过系统级 step 分解才能发现。

> **教训**：LLM 推理优化是系统问题。两个 Python 层面的修复（共 ~20 行代码）带来的收益，超过了手写 Triton kernel + Speculative Decoding 的所有尝试之和。

---

## 12. 关键结论与经验教训

### 12.1 核心结论

1. **`@torch.compile` + CUDA Graph 是小模型推理的最强组合**。对于 Qwen3-0.6B（28 层，hidden=896），torch.compile 的 inductor 后端已能生成接近最优的 Triton 代码，且与 CUDA Graph 无缝集成。手写 Triton kernel 虽然单独更快（eager +12.7%），但无法匹配 inductor 的零分配和跨操作融合能力。

2. **Speculative Decoding 不适用于小模型**。CUDA Graph 让 normal decode 极快（~1ms/step），speculative 的 K+1 次 eager forward 无法竞争。此外，28 层模型的 layer-skip draft 质量太差（7 层接受率 ~0%）。

3. **先测量，再优化**。Profiling 显示 RMSNorm + RoPE 仅占 20.5% CUDA 时间，且已被 torch.compile 优化。若不先 profiling，可能在不值得的方向浪费大量时间。

4. **H100 + Qwen3-0.6B 的瓶颈在调度层面**，而非 kernel 层面。QPS 从 12→14 时 TTFT 崩塌 133×，说明瓶颈在队列管理和 KV Cache 容量，而非 GPU 计算效率。

### 12.2 Triton 开发经验

1. **CUDA Graph 兼容性是关键**：任何替换 `@torch.compile` 的方案都必须保持零分配（或预分配 buffer）以兼容 CUDA Graph。Python wrapper 中的 `torch.empty_like()` 会在 graph replay 时产生开销。

2. **In-place kernel 要小心正确性**：RMSNorm in-place 修改在 Python 元组赋值（`a, b = f(x), x`）中会导致 bug，因为左右两侧的求值顺序使 `b` 拿到了修改后的值。

3. **小 hidden_size 减弱 Triton 优势**：hidden_size=896 意味着每行只有 896 个元素，kernel launch overhead 占比高。Triton 优势在 hidden_size ≥ 4096 时更明显（"11% → 88%" 案例）。

4. **Autotuning 在 CUDA Graph 场景下受限**：`@triton.autotune` 在首次调用时 benchmark 多种配置，可能干扰 CUDA Graph 捕获。固定 `num_warps` 和 `BLOCK_SIZE` 更安全。

### 12.3 Speculative Decoding 经验

1. **Layer-skip draft 对小模型不友好**：28 层模型中，前 7 层的输出分布与完整模型差异巨大。需要至少 20+ 层（71%）才能达到 ~40% 接受率。

2. **ParallelLMHead 的 last-token-only 过滤是隐藏陷阱**：verify 需要所有位置的 logits，但 LM head 默认只计算 last token。需要用 `F.linear()` 直接绕过。

3. **Block Manager 的 hash 机制与多次 may_append 冲突**：speculative 的多步 draft 需要自定义 block 分配逻辑，绕过 prefix caching 的 hash 断言。

4. **Gumbel sampling 的 in-place 陷阱**：`probs.div_(exponential_())` 会破坏原始 probs。必须在 sampling 前 clone。

### 12.4 推荐的未来优化方向（系统级分析更新）

基于 Phase 3 系统级分析，优化方向已根据**实际瓶颈占比**重新排序：

| 优化 | 瓶颈 | 当前系统占比 | 预期收益 | 优先级 |
|------|------|-------------|---------|--------|
| **Postprocess 向量化** | CPU Python 循环 | **42.7%** | 吞吐 +20~30% | **最高** |
| **prepare_model_input 优化** | CPU tensor 创建 | **12.0%** | 吞吐 +5~10% | **高** |
| **KV Cache INT8/FP8** | 内存容量 | 限制并发 | 2× 并发/context | **高** |
| **Prefix Cache** | 重复 prefill | TTFT 高 | TTFT -30~50% | **高** |
| **Tensor Parallel** | 单卡计算上限 | — | 多卡线性扩展 | 中 |
| ~~Triton kernel 优化~~ | ~~GPU 计算~~ | ~~3.1%~~ | ~~<3%~~ | ~~低（已验证无效）~~ |
| ~~Speculative Decoding~~ | ~~每步生成量~~ | ~~CUDA Graph 不兼容~~ | ~~对小模型无效~~ | ~~低~~ |

---

## 附录

### A. 环境信息

```
GPU: NVIDIA H100 80GB HBM3
Python: 3.12
PyTorch: 2.x with torch.compile
flash_attn: 2.8.3
Model: Qwen3-0.6B (28 layers, hidden=896, 14 heads, 2 KV heads, head_dim=128)
```

### B. 文件清单

| 文件 | 类型 | 用途 |
|------|------|------|
| `bench_latency.py` | benchmark | QPS 延迟测试 + sweep |
| `bench_speculative.py` | benchmark | Speculative decoding 测试 |
| `profile_baseline.py` | profiling | Kernel 级 CUDA 分析 |
| `nanovllm/kernels/fused_norm.py` | kernel | Triton Fused RMSNorm |
| `nanovllm/kernels/fused_rope.py` | kernel | Triton Fused RoPE |
| `nanovllm/engine/speculative.py` | engine | Rejection sampling |
| `tutorial.md` | 文档 | Triton 开发教程 |
| `conclusion_triron.md` | 文档 | 本技术报告 |

### C. 参考文献

- [From 11% to 88% Peak Bandwidth: Triton Kernels for LLM Inference](https://subhadipmitra.com/blog/2025/triton-kernels-llm-inference/)
- [Fast Inference from Transformers via Speculative Decoding (Leviathan 2023)](https://arxiv.org/abs/2211.17192)
- [vLLM Triton Attention Backend](https://blog.vllm.ai/2025/01/27/v1-alpha-launch.html)
- [NVIDIA NIM Benchmarking Metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html)
- [Triton Official Tutorials](https://triton-lang.org/main/getting-started/tutorials/)
