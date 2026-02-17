# KV 量化

## 三种量化的区别
| 类型 | 量化对象 | 时机 | 典型方法 |
|------|---------|------|---------|
| 权重量化 | 模型 weight | 离线 | GPTQ, AWQ, SmoothQuant |
| KV 量化 | K/V 缓存 | 在线（推理时） | FP8, INT8 per-channel |
| 激活量化 | 中间激活值 | 在线 | SmoothQuant (W8A8) |

## 对称 vs 非对称
- **对称**：zero_point = 0，`scale = max(|x|) / 127`
  - 简单快速，适合分布对称的 KV
- **非对称**：`scale = (max - min) / 255`，`zp = round(-min/scale)`
  - 更精确，但需要额外存储 zero_point

## Per-tensor vs Per-channel
- **Per-tensor**：整个张量一个 scale → 大误差
- **Per-channel**：每个通道/head 独立 scale → 误差小得多
- KV 量化推荐 **per-channel**（head_dim 维度）

## 精度-显存权衡
| 精度 | 每元素 bytes | 相对 bf16 | 典型 PPL 变化 |
|------|-------------|-----------|-------------|
| bf16 | 2 | 1× | baseline |
| fp8 | 1 | 0.5× | <0.1% |
| int8 | 1 | 0.5× | <0.5% |
| int4 | 0.5 | 0.25× | 1-3% |

## vLLM KV 量化
- 支持 FP8 KV Cache（`--kv-cache-dtype fp8`）
- 在线计算 scale（per-tensor 或 per-head）
- 精度损失极小，吞吐提升 ~30%（显存释放 → 更大 batch）

## 面试一句话
- "KV 量化的核心收益是显存减半 → 能放更多并发请求 → 吞吐提升。推荐 FP8 或 per-channel INT8，精度损失通常可忽略。"
