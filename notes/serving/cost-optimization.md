# 推理成本优化

## 成本公式
```
cost_per_token = GPU_cost_per_hour / tokens_per_hour
cost_per_request = cost_per_token × (input_tokens + output_tokens)
```

## 优化维度

### 1. 提高吞吐（分子不变，分母变大）
- KV 量化 (fp8) → 显存减半 → batch size ↑ → 吞吐 +30-50%
- 前缀缓存 → 重复 prefill 省掉 → 有效吞吐 ↑
- Continuous Batching → 消除空泡
- 更大 batch size（在 SLO 允许范围内）

### 2. 降低单请求成本
- 投机解码 → 1.5-3× 加速（同 GPU 时间产出更多 token）
- Prompt 压缩 → 减少输入 token 数
- 模型蒸馏 → 70B → 8B + 质量过滤

### 3. 降低硬件成本
- 混合精度部署：热模型 A100/H100，冷模型 CPU/较便宜 GPU
- 弹性伸缩：按需扩缩容
- Spot instance：可中断场景（batch 推理）

## 成本对比示例
| 方案 | 成本/1M token | 相对 baseline |
|------|-------------|-------------|
| H100 FP16 baseline | $3.00 | 1.0× |
| + FP8 KV 量化 | $2.10 | 0.7× |
| + 前缀缓存 (50% hit) | $1.50 | 0.5× |
| + 投机解码 (2×) | $0.75 | 0.25× |
| + 蒸馏 70B→8B | $0.15 | 0.05× |

*注：数值为估算，实际因场景而异*

## 面试一句话
- "推理成本优化的组合拳：量化省显存→大 batch→高吞吐，前缀缓存省重复计算，投机解码省 decode 时间，蒸馏降模型规模。"
