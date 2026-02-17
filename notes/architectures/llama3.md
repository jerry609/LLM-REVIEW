# Llama 3 架构拆解

## 模型规格（Llama 3.1-70B 为例）
| 参数 | 值 |
|------|-----|
| n_layers | 80 |
| d_model | 8192 |
| n_heads (Q) | 64 |
| n_kv_heads (K/V) | 8 (GQA, group_size=8) |
| head_dim | 128 |
| d_ff | 28672 (SwiGLU: 3.5×d_model) |
| vocab_size | 128,256 |
| max_context | 128K |
| 参数量 | ~70B |

## 核心组件
1. **GQA**：8 个 KV head 服务 64 个 Q head → KV Cache 缩小 8×
2. **RoPE**：旋转位置编码，支持长上下文扩展（θ=500,000）
3. **SwiGLU FFN**：`output = SiLU(xW_gate) ⊙ (xW_up) @ W_down`，比标准 FFN 参数多但效果更好
4. **RMSNorm**：pre-norm，比 LayerNorm 快（省去均值计算）

## 前向流程（单层）
```
x → RMSNorm → Q/K/V Proj → RoPE(Q,K) → GQA Attention → + residual
  → RMSNorm → SwiGLU FFN → + residual → next layer
```

## 权重显存
- FP16: 70B × 2 bytes = 140 GB（需 ≥2× H100 80GB）
- INT8: ~70 GB（单机 1× H100 可放下）
- INT4: ~35 GB

## 面试一句话
- "Llama 3 = GQA + RoPE + SwiGLU + RMSNorm，架构简洁但 scaling 效果好。GQA 让 KV Cache 缩小 8×，是推理友好的关键设计。"
