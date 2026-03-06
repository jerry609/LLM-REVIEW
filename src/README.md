# 代码实现

> 这里放的是可运行的最小实现。目录设计遵循“公式先行，代码跟进”的原则：每个核心脚本都尽量能被某一页数学推导或专题 walkthrough 直接解释。

## 推荐的阅读方式

1. 先从 `notes/` 里的 walkthrough 建立公式和张量形状。
2. 再回到这里看最小实现，把变量名和函数边界对上。
3. 最后跑 `tests/` 里的单测，确认这些公式在代码里真的成立。

## 重点源码地图

| 文件 | 你会看到什么 | 对应文档 | 快速运行 |
|------|--------------|----------|----------|
| [attention/mha_gqa.py](attention/mha_gqa.py) | MHA / MQA / GQA 的最小 NumPy 实现 | [../notes/attention/formula-to-code-walkthrough.md](../notes/attention/formula-to-code-walkthrough.md) | `python src/attention/mha_gqa.py` |
| [attention/rope_rmsnorm.py](attention/rope_rmsnorm.py) | RoPE cache、旋转、RMSNorm | [../math_dictionary/transformer-attention-math.md](../math_dictionary/transformer-attention-math.md) | `python src/attention/rope_rmsnorm.py` |
| [attention/flash_attn_sim.py](attention/flash_attn_sim.py) | FlashAttention 的在线 Softmax 模拟 | [../notes/attention/formula-to-code-walkthrough.md](../notes/attention/formula-to-code-walkthrough.md) | `python src/attention/flash_attn_sim.py` |
| [kv_cache/core.py](kv_cache/core.py) | block allocator、Paged KV Cache、Copy-on-Write | [../notes/kv-cache/formula-to-code-walkthrough.md](../notes/kv-cache/formula-to-code-walkthrough.md) | `python src/kv_cache/core.py` |
| [kv_cache/compression/quantizer.py](kv_cache/compression/quantizer.py) | KV 的对称 / 非对称 per-channel 量化 | [../notes/kv-compression/formula-to-code-walkthrough.md](../notes/kv-compression/formula-to-code-walkthrough.md) | `python src/kv_cache/compression/quantizer.py` |
| [kv_cache/compression/sparsifier.py](kv_cache/compression/sparsifier.py) | H2O / SnapKV 风格的 token 选择与压缩比 | [../notes/kv-compression/formula-to-code-walkthrough.md](../notes/kv-compression/formula-to-code-walkthrough.md) | `python src/kv_cache/compression/sparsifier.py` |
| [kv_cache/eviction/policies.py](kv_cache/eviction/policies.py) | LRU、LFU、公平配额驱逐 | [../notes/kv-eviction/formula-to-code-walkthrough.md](../notes/kv-eviction/formula-to-code-walkthrough.md) | `python src/kv_cache/eviction/policies.py` |
| [simulators/scheduler.py](simulators/scheduler.py) | continuous batching 和 decode 优先调度 | [../notes/serving/formula-to-code-walkthrough.md](../notes/serving/formula-to-code-walkthrough.md) | `python src/simulators/scheduler.py` |
| [simulators/serving_metrics.py](simulators/serving_metrics.py) | TTFT、TPOT、Goodput、batch utilization | [../notes/serving/formula-to-code-walkthrough.md](../notes/serving/formula-to-code-walkthrough.md) | `python src/simulators/serving_metrics.py` |
| [simulators/queueing_slo.py](simulators/queueing_slo.py) | Little 定律、M/M/1、Erlang C、M/G/1、SLO 反推 | [../notes/serving/queueing-slo-formula-to-code-walkthrough.md](../notes/serving/queueing-slo-formula-to-code-walkthrough.md) | `python src/simulators/queueing_slo.py` |
| [simulators/moe_routing.py](simulators/moe_routing.py) | MoE router、capacity、dispatch、drop rate | [../notes/distributed/moe-formula-to-code-walkthrough.md](../notes/distributed/moe-formula-to-code-walkthrough.md) | `python src/simulators/moe_routing.py` |
| [training/lora.py](training/lora.py) | LoRA 的最小实现 | [../math_dictionary/lora-peft-math.md](../math_dictionary/lora-peft-math.md) | 见对应测试 |

## 从专题跳回源码的最短路径

- Attention：先看 [../notes/attention/formula-to-code-walkthrough.md](../notes/attention/formula-to-code-walkthrough.md)，再读 `attention/`。
- KV Cache：先看 [../notes/kv-cache/formula-to-code-walkthrough.md](../notes/kv-cache/formula-to-code-walkthrough.md)，再读 `kv_cache/`。
- KV Compression：先看 [../notes/kv-compression/formula-to-code-walkthrough.md](../notes/kv-compression/formula-to-code-walkthrough.md)，再读 `kv_cache/compression/`。
- KV Eviction：先看 [../notes/kv-eviction/formula-to-code-walkthrough.md](../notes/kv-eviction/formula-to-code-walkthrough.md)，再读 `kv_cache/eviction/`。
- Serving：先看 [../notes/serving/formula-to-code-walkthrough.md](../notes/serving/formula-to-code-walkthrough.md) 和 [../notes/serving/queueing-slo-formula-to-code-walkthrough.md](../notes/serving/queueing-slo-formula-to-code-walkthrough.md)，再读 `simulators/`。
- MoE：先看 [../notes/distributed/moe-formula-to-code-walkthrough.md](../notes/distributed/moe-formula-to-code-walkthrough.md)，再读 `simulators/moe_routing.py`。
