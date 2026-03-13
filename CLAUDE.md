# Nano-vLLM-v1 Project

## Overview
基于 nano-vllm 的 LLM 推理引擎，核心目标：实现 Learning-to-Rank (LTR) 在线学习调度器，使其吞吐量超过 FCFS baseline。

## Environment
- GPU: H100
- Python: `/opt/data/private/vllm/bin/python`
- 虚拟环境: `/opt/data/private/vllm/`
- 模型: `~/huggingface/Qwen3-0.6B/`
- flash_attn: 2.8.3

## Key Commands
```bash
# FCFS baseline benchmark
python bench_real.py --num_seqs 512 --max_model_len 32768

# LTR benchmark (with online learning)
python bench_ltr.py --num_seqs 5000 --qps 50

# LTR benchmark (frozen model, no training)
python bench_ltr_freeze.py --num_seqs 5000 --qps 50

# Collect training labels
python bench_real.py --collect

# One-command push: add + commit + push
bash scripts/push.sh <branch> "<commit_message>"
```

## Project Structure
- `nanovllm/engine/scheduler.py` — 统一调度器 + Fast Path
- `nanovllm/engine/ltr_scheduler.py` — LTR 三阶段调度器 (FCFS → Heuristic → Model)
- `nanovllm/engine/llm_engine.py` — 引擎入口，按 config 选择调度器
- `nanovllm/utils/data_collector.py` — LTR 数据收集（环形 buffer）
- `bench_real.py` — FCFS baseline benchmark
- `bench_ltr.py` — LTR benchmark (在线学习)
- `bench_ltr_freeze.py` — LTR benchmark (冻结模型)
- `prompts.jsonl` — 181,989 条真实 prompt
- `training_data.jsonl` — LTR 训练数据

## Conventions
- 所有 benchmark 对比必须使用相同的 API 路径（generate() 或 step()），确保公平
- 修改调度器时注意不要破坏 Fast Path（纯 decode 快速路径，+37.3% 吞吐）
- LTR 调度逻辑不应阻塞推理热路径
