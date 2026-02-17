# KV 稀疏化

## 核心思想
- 注意力分布通常是**稀疏的**：少数 token 贡献了大部分注意力权重
- 只保留"重要" token 的 KV → 大幅压缩缓存 → 精度损失可控

## 主要方法

### H2O (Heavy-Hitter Oracle)
- 观察：少数 "heavy-hitter" token 在所有 head 上注意力分数都很高
- 策略：保留 top-k heavy-hitter + 最近 W 个 token
- 优点：简单有效，压缩率高
- 论文：[H2O, NeurIPS 2023](https://arxiv.org/abs/2306.14048)

### SnapKV
- 在 prefill 结束时用一个观察窗口统计每个 token 的重要性
- 只保留重要 token 的 KV → decode 阶段用压缩后的 KV
- 优点：一次性决策，decode 无额外开销
- 论文：[SnapKV, 2024](https://arxiv.org/abs/2404.14469)

### PyramidInfer
- 按层递减保留 KV 数量（浅层保留多、深层保留少）
- 观察：深层注意力更集中，浅层更分散
- 不同层的压缩率可以不同

## 稀疏 vs 量化
| 维度 | 稀疏化 | 量化 |
|------|--------|------|
| 压缩方式 | 减少 token 数 | 降低每 token 精度 |
| 压缩率 | 可达 10-50× | 2-4× |
| 精度风险 | 丢失关键 token 则灾难性 | 均匀退化 |
| 可组合 | ✅ 可叠加量化 | ✅ 可叠加稀疏 |

## 面试一句话
- "稀疏化通过只保留注意力权重高的 token 来压缩 KV，和量化正交可叠加。但必须验证长上下文任务（如 Needle-in-a-Haystack）不丢关键信息。"
