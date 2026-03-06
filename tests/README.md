# 测试

> 这里的测试不是为了覆盖所有边角，而是为了把文档里的关键公式钉在可执行样例上。你可以把它理解成“公式是否真的在代码里成立”的最小回归集。

## 重点测试

- [test_attention.py](test_attention.py)：Attention、RoPE、FlashAttention 的基础行为。
- [test_kv_cache.py](test_kv_cache.py)：KV block 分配、追加、fork 和释放。
- [test_scheduler.py](test_scheduler.py)：continuous batching 的基本调度逻辑。
- [test_serving_metrics.py](test_serving_metrics.py)：TTFT、TPOT、Goodput、batch utilization、KV 步带宽。
- [test_moe_routing.py](test_moe_routing.py)：top-k router、辅助损失、capacity、dispatch、drop rate、All-to-All 字节量。
- [test_lora.py](test_lora.py)：LoRA 的形状和前向逻辑。

## 常用运行方式

```bash
python -m pytest tests -v
python -m pytest tests/test_kv_cache.py -v
python -m pytest tests/test_serving_metrics.py tests/test_moe_routing.py -v
```

## 建议的对照顺序

1. 先读 [../notes/kv-cache/formula-to-code-walkthrough.md](../notes/kv-cache/formula-to-code-walkthrough.md)，再跑 `test_kv_cache.py`。
2. 先读 [../notes/serving/formula-to-code-walkthrough.md](../notes/serving/formula-to-code-walkthrough.md)，再跑 `test_serving_metrics.py` 和 `test_scheduler.py`。
3. 先读 [../notes/distributed/moe-formula-to-code-walkthrough.md](../notes/distributed/moe-formula-to-code-walkthrough.md)，再跑 `test_moe_routing.py`。
