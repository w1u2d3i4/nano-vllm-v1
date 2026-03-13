# Nano-vLLM-v1 对话与修改记录

## 项目概述

基于 [nano-vllm](https://github.com/niconielsen32/nano-vllm) 原版（~1,360 行），我们逐步实现了多项改进，最终目标是加入 Learning-to-Rank (LTR) 在线学习调度器。

---

## 已完成的改进（按时间顺序）

### Phase 1: 核心架构升级（已完成）

从 `nano-vllm-v1-main`（作者的 v1 版本）移植精简后的改动到用户的 `nano-vllm-v1` 项目。

| 改动 | 文件 | 说明 |
|------|------|------|
| Chunked Prefill | `config.py`, `scheduler.py` | 长 prompt 分块处理，防止 batch 饥饿 |
| vLLM v1 统一调度器 | `scheduler.py`, `model_runner.py` | 合并 prefill/decode 为统一 token-aware 调度 |
| BlockManager 增强 | `block_manager.py` | `get_token_layout`, `can_allocate(num_tokens)`, `can_append(seq, num_new_tokens)` |
| 统一 Attention 路径 | `attention.py`, `embed_head.py`, `context.py` | 只用 `flash_attn_varlen_func`，移除 `is_prefill` 分支 |
| Decode 快速路径 | `scheduler.py` | 纯 decode 无新请求时跳过完整调度，实测 +37.3% 吞吐量 |

**关键修复：**
- FlashAttention 2.8.3 不接受 `None` 的 `cu_seqlens_q/k` → CUDA Graph 捕获时传 dummy tensor
- CUDA Graph 只在纯 decode 场景使用，混合 batch 回退 eager
- Fast Path 块分配 race condition → 改为一次性统计总需求块数

### Phase 2: LTR 调度器（已完成）

| 文件 | 状态 | 说明 |
|------|------|------|
| `sequence.py` | ✅ | 加 `arrival_time = time.time()` 用于防饥饿 |
| `config.py` | ✅ | 加 `enable_ltr: bool`, `ltr_data_path: str` |
| `utils/data_collector.py` | ✅ 新建 | 环形 buffer（maxlen=50000），收集 `(prompt_length, output_length)`，支持 `load_from_file` / `save_to_file` |
| `engine/ltr_scheduler.py` | ✅ 新建 | 三阶段调度器（见下文） |
| `engine/llm_engine.py` | ✅ | 按 `config.enable_ltr` 选择 Scheduler 类；`exit()` 时调用 `save_state()` |

**LTR 调度器三阶段：**

```
FCFS (< 50 样本)  →  Heuristic (< 200 样本, 按 prompt_length 排序)  →  Model (≥ 200 样本, SGDRegressor 预测 output_length 做 SJF)
```

**核心机制：**
- **冷启动**：从 `training_data.jsonl` 加载历史数据 → `SGDRegressor.fit()`
- **热启动**：如果存在 `.model` 文件，直接 `joblib.load()` 跳过训练
- **在线收集**：每个序列完成时自动收集 `(prompt_len, output_len)`
- **在线更新**：每 100 个新样本触发 `partial_fit()` 增量更新
- **防饥饿**：等待超过 30s 的序列强制提前
- **持久化**：退出时保存数据到 JSONL + 模型权重到 `.model` 文件
- **`_need_reorder` 优化**：只在新 seq 加入或模型更新时才重排 waiting 队列

### Phase 3: Benchmark 脚本

| 文件 | 用途 |
|------|------|
| `bench.py` | 原版 benchmark（写死参数，256 seq） |
| `bench_real.py` | FCFS baseline + `--collect` 收集 label |
| `bench_ltr.py` | LTR benchmark，支持 `--qps` 流量注入 + 吞吐量曲线图 |
| `scripts/preprocess_sharegpt.py` | 从 ShareGPT-Chinese-English-90K 提取真实 prompt → `prompts.jsonl`（181,989 条） |

---

## 性能数据

### bench.py（256 seq，随机 prompt，max_model_len=4096）

| 版本 | Output Tokens | Time | Throughput |
|------|-------------|------|------------|
| 原版 | 133,966 | 12.03s | 11,138 tok/s |
| v1 (统一调度) | 133,966 | 11.20s | 11,961 tok/s (+7.4%) |
| v1 + Fast Path | 133,966 | 8.76s | 15,294 tok/s (+37.3%) |

### bench_real.py（512 seq，真实 prompt，max_model_len=32768）

| 调度 | Output Tokens | Time | Throughput |
|------|-------------|------|------------|
| FCFS | 524,288 | 27.61s | 18,991 tok/s |
| LTR (冷启动) | 524,288 | 25.72s | 20,381 tok/s (+7.3%) |

### bench_ltr.py（5000 seq，qps=50，在线学习开启）

**第一次运行**（冷启动 → Warm start 11,390 samples）：
- Output: 5,516,652 tokens, Time: 336.84s, **Throughput: 16,377 tok/s**
- 模型更新 42 次
- 打印存在重复行问题（print_interval 逻辑 bug），但不影响最终结果
- 退出时保存 16,391 samples + model

**第二次运行**（Warm start 16,391 samples，模型更成熟）：
- Output: 5,516,652 tokens, Time: 315.11s, **Throughput: 17,507 tok/s (+6.9% vs 第一次)**
- 模型更新 49 次
- 打印正常无重复
- 退出时保存 21,392 samples + model
- 生成 `ltr_throughput.png` 吞吐量曲线图

**观察**：Warm start 第二次比第一次快了 6.9%，说明在线学习持久化生效，模型越跑越准。但 qps=50 模式下的吞吐量（16k-17k tok/s）低于 bench_real.py 静态模式的 ~20k tok/s，因为 qps 注入导致初期 GPU 未满载 + 持续排队开销。

---

## 关键文件路径

```
/opt/data/private/llm_test/
├── nano-vllm-main/           # 原版 + 分析文档
│   ├── analysis_report.md    # 原版 vs v1-main 对比分析
│   ├── plan.md               # LTR 实施计划（原始 3-4 天版）
│   └── nanovllm/             # 原版代码 + annotated 版本
├── nano-vllm-v1-main/        # 作者的 v1 版本（参考，不修改）
├── nano-vllm-v1/             # 我们的项目（所有改动在这里）
│   ├── nanovllm/
│   │   ├── config.py         # +enable_ltr, +ltr_data_path
│   │   ├── engine/
│   │   │   ├── scheduler.py      # 统一调度 + Fast Path
│   │   │   ├── ltr_scheduler.py  # LTR 三阶段调度器 ★
│   │   │   ├── model_runner.py   # 统一 prepare_model_input
│   │   │   ├── llm_engine.py     # LTR scheduler 选择 + exit 持久化
│   │   │   ├── sequence.py       # +arrival_time
│   │   │   └── block_manager.py  # v1-main 移植
│   │   ├── layers/
│   │   │   ├── attention.py      # 统一 flash_attn_varlen_func
│   │   │   └── embed_head.py     # seq_need_compute_logits
│   │   └── utils/
│   │       ├── context.py        # +cu_seqlens_q/k, +seq_need_compute_logits
│   │       └── data_collector.py # LTR 数据收集 ★
│   ├── bench.py              # 原版 benchmark
│   ├── bench_real.py         # FCFS baseline + collect
│   ├── bench_ltr.py          # LTR benchmark + qps + plot
│   ├── prompts.jsonl         # 181,989 条真实 prompt
│   ├── training_data.jsonl   # LTR 训练数据（21,392 条，持续增长）
│   ├── training_data.jsonl.model  # SGDRegressor 模型权重
│   ├── ltr_throughput.png    # 吞吐量曲线图
│   └── scripts/
│       └── preprocess_sharegpt.py
└── data/
    └── ShareGPT-Chinese-English-90K/  # 原始数据集
```

---

## 已知问题 / 待改进

1. **LTR qps 模式吞吐量偏低**：`bench_ltr.py --qps 50` 的 17,507 tok/s 低于 `bench_real.py` 的 ~20,000 tok/s。原因可能是：
   - qps 注入导致初期 GPU 未满载（ramp-up 阶段拉低平均值）
   - 在线 `_maybe_train()` 的 `partial_fit` 开销（每 100 个完成序列触发一次）
   - **待验证**：用户要求写一个「只排序不更新」的纯推理版本来隔离变量

2. **`_reorder_waiting` 性能**：已优化为只在新 seq 加入或模型更新时触发（`_need_reorder` 标记），但 waiting 队列很大时单次排序仍有开销

3. **重排次数**：当前 `add()` 每调用一次设 `_need_reorder=True`，但 `schedule()` 只会在下次调用时排序一次（不是每个 add 排一次）。所以如果 1 秒内 add 了 50 个 seq 但只 schedule 了一次，就只重排 1 次

4. **Staging Buffer 已放弃**：Python 层面的 pinned memory 预分配反而导致性能下降（11,961 → 6,826 tok/s），需要 C++/CUDA 级别实现才有效

---

## 环境信息

- GPU: H100
- Python: 3.12（`/opt/data/private/vllm/bin/python`）
- 虚拟环境: `/opt/data/private/vllm/`
- 模型: `~/huggingface/Qwen3-0.6B/`（max_position_embeddings=40960）
- flash_attn: 2.8.3
- scikit-learn: 1.8.0
- matplotlib: 3.10.8
