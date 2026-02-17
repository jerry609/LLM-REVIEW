# 投机解码 (Speculative Decoding)

## 核心思想
- Decode 是 memory-bound → GPU 算力闲置
- 用小模型（draft model）快速生成 K 个候选 token
- 用大模型（target model）一次性验证 K 个 token（并行前向）
- 接受正确的 token，拒绝错误的 → 保证与大模型分布**完全一致**

## 接受-拒绝采样
```
for each draft token x_i:
    p = target_prob(x_i)
    q = draft_prob(x_i)
    accept with probability min(1, p/q)
    if reject: resample from adjusted distribution
```
- 数学保证：最终输出分布 = target model 分布（无损）

## 加速比分析
- 设 draft 接受率 α，每次投机 K 个 token
- 期望每次接受 token 数 ≈ (1 - α^(K+1)) / (1 - α)
- α=0.8, K=5 → 期望 ~4.0 token/iteration → 加速 ~4×
- 但 draft 模型本身有开销，实际加速通常 1.5-3×

## Draft 模型选择
| 方案 | 优点 | 缺点 |
|------|------|------|
| 小同系列模型 (如 1B) | 接受率高 | 需要额外显存 |
| 自蒸馏模型 | 接受率最高 | 需要训练 |
| Medusa heads | 无需额外模型 | 接受率较低 |
| EAGLE | 高接受率 | 需要训练额外 head |

## 面试一句话
- "投机解码利用 decode 的 memory-bound 空闲算力，用小模型猜测 + 大模型验证，实现无损加速 1.5-3×。"
