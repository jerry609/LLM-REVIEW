# KV Cache 面试题 003：KV Cache 压缩方案

## 题目
> 你的 LLM 推理系统需要支持 128K 上下文，但 GPU 显存有限。
> 设计 KV Cache 的压缩方案，分析精度-效率的 trade-off。

## 参考答案（限时 10 分钟）

### 1. 问题量化
```
Llama-70B, 128K context, FP16:
  KV per layer = 2 × seq_len × n_heads × head_dim × 2 bytes
              = 2 × 128K × 8 × 128 × 2 = 512 MB per layer
  Total (80 layers) = 40 GB  ← 超过单卡显存
```

### 2. 压缩方案对比

| 方案 | 压缩比 | 精度损失 | 实现难度 | 推荐场景 |
|------|--------|---------|---------|---------|
| KV INT8 | 2x | 极小 | 低 | 通用首选 |
| KV INT4 | 4x | 可接受 | 中 | 超长上下文 |
| KV 稀疏 (H₂O) | 2-4x | 小 | 中 | attention 分布集中 |
| KV 合并 (CLA) | 2x | 中 | 高 | 需要训练 |
| Offload (CPU) | ∞ | 无 | 中 | 极长上下文 |

### 3. 推荐组合方案
```
Layer 1-10:  FP16 (精度敏感层)
Layer 11-70: INT8 per-head quantization
Layer 71-80: INT4 + H₂O sparse (注意力集中)
```

### 4. INT8 KV 实现要点
```python
# Per-head quantization (比 per-tensor 精度高)
for head in range(n_heads):
    kv_slice = kv_cache[:, head, :, :]
    scale = kv_slice.abs().max() / 127
    kv_int8[:, head, :, :] = (kv_slice / scale).round().clamp(-128, 127)
    scales[head] = scale
```

### 5. H₂O (Heavy-Hitter Oracle) 稀疏
- 保留 attention score 最高的 K 个 token 的 KV
- 动态维护 top-K 集合
- 精度：保留 20% KV 可达 95%+ 的精度

### 6. 面试加分项
- **分层策略**：不同层用不同压缩（浅层保留精度）
- **在线 calibration**：根据 runtime attention pattern 动态调整
- **与 PagedAttention 结合**：block 粒度量化，减少碎片
