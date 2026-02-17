# FlashAttention

## 核心问题
- 标准注意力需要将 T×T 的注意力矩阵写入 HBM → IO 是 O(T²)
- GPU SRAM 很快但很小（~20MB），HBM 大但慢

## 核心思路：Tiling + Online Softmax
1. 将 Q/K/V 切成小块（tile），每块放入 SRAM
2. 在 SRAM 内完成 softmax + 加权求和
3. 用 online softmax 算法跨块维护正确的归一化
4. **不需要把完整 T×T 矩阵写回 HBM**

## IO 复杂度
| 方法 | FLOPs | HBM IO |
|------|-------|--------|
| 标准注意力 | O(T² × d) | O(T²) |
| FlashAttention | O(T² × d) | O(T² × d / M_sram) |

- FLOPs 相同，但 IO 大幅减少
- 实际加速来源：减少 memory-bound 操作

## Online Softmax 关键
```
对每个 Q 块 i，遍历所有 K 块 j：
  s_ij = Q_i @ K_j^T / sqrt(d)
  m_new = max(m_old, max(s_ij))
  l_new = exp(m_old - m_new) * l_old + sum(exp(s_ij - m_new))
  O_i = (rescale old + new contribution) / l_new
```

## FlashAttention-2 改进
- 更好的 warp 级并行（沿 seq_len 维度并行）
- 减少非矩阵乘 FLOPs
- 支持 GQA native

## 面试一句话
- "FlashAttention 不减少计算量，而是通过 tiling + online softmax 减少 HBM 读写次数，把注意力从 memory-bound 变成 compute-bound。"
