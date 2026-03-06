# 核心代码实现

> 可运行的 PyTorch 代码，覆盖注意力、KV Cache、调度器和训练模块。

## 当前模块

- [attention/mha_gqa.py](attention/mha_gqa.py)：MHA、MQA、GQA 的最小实现。
- [attention/rope_rmsnorm.py](attention/rope_rmsnorm.py)：RoPE 与 RMSNorm。
- [attention/flash_attn_sim.py](attention/flash_attn_sim.py)：FlashAttention 思路模拟。
- [kv_cache/core.py](kv_cache/core.py)：KV Cache 核心数据结构。
- [training/lora.py](training/lora.py)：LoRA 低秩适配。
- [simulators/scheduler.py](simulators/scheduler.py)：调度器模拟。

## KV Cache 子模块

- [kv_cache/eviction/policies.py](kv_cache/eviction/policies.py)：LRU、LFU、H2O 等驱逐策略。
- [kv_cache/compression/quantizer.py](kv_cache/compression/quantizer.py)：KV 量化压缩。

## From Scratch 模块

- [from_scratch/README.md](from_scratch/README.md)：训练与对齐复现的统一骨架。