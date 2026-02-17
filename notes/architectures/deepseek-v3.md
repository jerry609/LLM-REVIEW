# DeepSeek-V3 架构拆解

## 核心创新
1. **MLA (Multi-head Latent Attention)**：压缩 KV 到低维潜在空间
2. **DeepSeekMoE**：更细粒度的专家路由（更多小专家 + shared expert）
3. **FP8 训练**：全程 FP8 混合精度训练，节省训练成本

## MLA vs GQA
| 维度 | GQA | MLA |
|------|-----|-----|
| KV 表示 | n_kv_heads × head_dim | 低秩压缩（d_c 维） |
| KV Cache 大小 | 2 × n_kv_heads × head_dim | 2 × d_c（可远小于 GQA） |
| 计算 | 直接投影 | 下投影 → 缓存 → 上投影还原 |
| 质量 | 微损 | 接近 MHA |

## MLA 公式
```
# 下投影（压缩）
c_kv = x @ W_down_kv    # [B, T, d_c]，d_c << n_kv_heads * head_dim
# 缓存 c_kv（而非完整 K/V）
# 上投影（还原）
K = c_kv @ W_up_k       # [B, T, n_heads, head_dim]
V = c_kv @ W_up_v
```

## DeepSeekMoE
- 更多更小的 expert（如 256 个小 expert vs Mixtral 的 8 个大 expert）
- 1-2 个 shared expert（所有 token 都激活）+ top-k routing expert
- 效果：路由更灵活，负载更均衡

## 面试一句话
- "DeepSeek-V3 的 MLA 把 KV 压缩到低维潜在空间再缓存，比 GQA 更省显存；配合细粒度 MoE 实现了性价比最高的大模型推理。"
