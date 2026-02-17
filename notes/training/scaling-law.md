# Scaling Laws

## Kaplan Scaling Law (OpenAI, 2020)
- `L(N,D) ∝ N^{-α_N} + D^{-α_D} + const`
- 性能随模型参数 N 和数据量 D 幂律下降
- 训练 compute C = 6ND → 固定 C 下 N 和 D 有最优比例

## Chinchilla Law (DeepMind, 2022)
- 结论：之前的大模型**训练不充分**（数据量不足）
- 最优比例：D ≈ 20×N（每个参数至少需要 20 个 token）
- 例：70B 模型应训练 1.4T token

## Chinchilla vs Kaplan 的实践影响
| 维度 | Kaplan 预测 | Chinchilla 修正 |
|------|-----------|--------------|
| N vs D 比例 | 多给 N 少给 D | N 和 D 等比增长 |
| 大模型数据 | 不够用 | 必须匹配 |
| 推理效率 | 大模型推理贵 | 训练充分的小模型更高效 |

## 推理 Scaling Law
- 推理成本 ∝ N × T（参数量 × 上下文长度）
- Chinchilla 模型推理更便宜（同质量下参数更少）
- 但训练更贵（需要更多数据和 FLOPs）

## 面试一句话
- "Chinchilla law 告诉我们每参数配 20 token 训练数据，训练充分的小模型在推理时性价比远超欠训练的大模型。"
