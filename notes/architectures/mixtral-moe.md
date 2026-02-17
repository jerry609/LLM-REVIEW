# Mixtral (MoE) 架构拆解

## 核心思想
- Mixture of Experts：每层有多个 FFN "专家"，每个 token 只激活 top-k 个
- 总参数量大（如 8×7B = 46.7B），但每 token 激活参数少（2×7B ≈ 13B）
- 效果接近 70B dense 模型，推理成本接近 13B

## Mixtral 8x7B 规格
| 参数 | 值 |
|------|-----|
| n_layers | 32 |
| d_model | 4096 |
| n_heads | 32 |
| n_kv_heads | 8 (GQA) |
| n_experts | 8 |
| top_k | 2 |
| d_ff (per expert) | 14336 |
| 总参数量 | 46.7B |
| 激活参数量 | ~13B |

## 路由机制
```
gate_logits = x @ W_gate       # [B, T, n_experts]
top_k_idx = topk(gate_logits)  # 选 top-2 专家
weights = softmax(gate_logits[top_k_idx])
output = Σ weights[i] * Expert_i(x)
```

## 负载均衡
- 训练时加辅助损失（auxiliary load balancing loss）防止路由坍缩
- 推理时：若路由不均 → 部分 GPU 空闲，其他过载
- Expert Parallelism (EP)：每个 GPU 放部分 expert，需要 All-to-All 通信

## 推理挑战
1. **显存**：总参数大，但可以用 EP 分摊
2. **通信**：All-to-All 延迟高（每 token 需路由到对应 GPU）
3. **负载均衡**：不均匀路由 → 尾延迟
4. **KV Cache**：Attention 部分和 dense 模型一样，不受 MoE 影响

## 面试一句话
- "MoE 用稀疏激活实现了以小激活量获得大模型质量的效果。推理瓶颈是 EP 通信和路由负载均衡。"
