# MoE 训练策略深度解析

> 结构原理 → 训练挑战 → 负载均衡 → 工程实践

---

## 一、MoE 结构原理回顾

### 1.1 核心架构
```
标准 Transformer FFN:
  output = FFN(x)

MoE 替换 FFN:
  gate_logits = x @ W_gate                    # [B, T, n_experts]
  top_k_idx, top_k_weights = TopK(softmax(gate_logits), k)
  output = Σ(top_k_weights[i] × Expert_i(x))  # 只激活 top-k 个 expert
```

### 1.2 关键参数
| 参数 | 含义 | 典型值 |
|------|------|--------|
| n_experts | 总 expert 数 | 8 (Mixtral), 256 (DeepSeek-V3) |
| top_k | 每 token 激活 expert 数 | 2 (Mixtral), 8 (DeepSeek-V3) |
| d_ff (per expert) | 每个 expert 的 FFN 隐藏维度 | 14336 (Mixtral) |
| capacity_factor | expert 容量因子 | 1.0-1.5 |

### 1.3 为什么 MoE 好？
```
总参数量大（多样性）× 每 token 激活少（推理便宜）= 性能/成本最优
Mixtral 8x7B: 总 46.7B 参数，每 token 只激活 ~13B → 效果接近 70B dense
```

---

## 二、MoE 训练核心挑战

### 2.1 路由坍缩 (Routing Collapse)
```
问题：所有 token 被路由到同一个或少数 expert
      → 其他 expert 得不到训练 → 退化为 dense 模型
原因：训练初期某个 expert 偶然表现好 → 更多 token 被路由到它
      → 它得到更多训练 → 表现更好 → 正反馈循环（马太效应）
```

### 2.2 负载不均 (Load Imbalance)
```
问题：某些 expert 被过度分配，处理的 token 数远多于其他
影响：
  - 训练时：过载 expert 的 GPU 成为瓶颈 → 其他 GPU 空等
  - 推理时：All-to-All 通信不均匀 → 尾延迟高
```

### 2.3 Expert 容量限制
```
问题：每个 expert 有最大容量（capacity = batch_size × capacity_factor / n_experts）
      超出容量的 token 被丢弃（token dropping）→ 信息丢失
```

### 2.4 通信开销
```
MoE 需要 All-to-All 通信：
  每个 GPU 上的 token → 发送到对应 expert 的 GPU
  Expert 计算完 → 发送回原 GPU
通信量：与 token 数 × hidden_dim 成正比
```

---

## 三、负载均衡策略

### 3.1 Auxiliary Load Balancing Loss（经典方法）

```python
# Switch Transformer 的 z-loss
f_i = fraction_of_tokens_routed_to_expert_i  # 实际负载
P_i = mean_of_gate_probs_for_expert_i         # 平均路由概率
L_aux = α × n_experts × Σ(f_i × P_i)        # 鼓励均匀分配
```

- **原理**：惩罚负载集中的 expert
- **缺点**：额外的 loss 项可能干扰主任务训练，需要仔细调 α

### 3.2 Auxiliary-Loss-Free (DeepSeek-V3 方案) ⭐

```python
# 不用额外 loss，而是动态调整 bias
gate_logits = x @ W_gate
for i in range(n_experts):
    if load[i] > average_load:
        bias[i] -= δ  # 减少过载 expert 的路由概率
    else:
        bias[i] += δ  # 增加空闲 expert 的路由概率
adjusted_logits = gate_logits + bias
```

- **优势**：不污染主训练目标，负载更均匀
- **原理**：通过 bias 软调控路由，而非硬性 loss 约束

### 3.3 Token Dropping

```python
capacity = int(batch_size * seq_len / n_experts * capacity_factor)
for expert_i in experts:
    tokens_i = tokens_routed_to(expert_i)
    if len(tokens_i) > capacity:
        tokens_i = tokens_i[:capacity]  # 丢弃超出的 token
        # 被丢弃的 token 直接走残差路径
```

- **训练时**：capacity_factor = 1.0~1.5，允许丢弃
- **推理时**：**不丢弃**（capacity = ∞），保证输出质量

### 3.4 Expert Choice (Zhou et al.)
```
传统：每个 token 选 expert（Token Choice）
新方法：每个 expert 选自己想处理的 token（Expert Choice）
```
- 天然负载均衡（每个 expert 选固定数量的 token）
- 缺点：同一个 token 可能被多个 expert 选中或没被选中

---

## 四、MoE 训练工程实践

### 4.1 初始化策略
```
关键：每个 expert 的初始化要有足够差异
方法：
  1. 随机初始化（标准）
  2. 从 dense 模型 upcycle：
     - 先训练 dense 模型
     - 复制 FFN 到所有 expert（加小扰动）
     - 继续训练
  好处：加速收敛，避免路由坍缩
```

### 4.2 Upcycling（Dense → MoE 转换）
```
Dense 模型 (7B)
    │
    ▼
复制 FFN 为 8 个 expert
    │  加入 gate 网络（随机初始化）
    │  每个 expert = 原 FFN + 小随机扰动
    ▼
MoE 模型 (8x7B ≈ 47B)
    │
    ▼
继续预训练少量 tokens → 路由学会分工
```
- **Qwen 的策略**：从 dense Qwen2.5 upcycle 到 MoE 版本
- **优势**：节省大量预训练 compute

### 4.3 分布式训练：Expert Parallelism (EP)

```
┌────────────────────────────────────────┐
│  Expert Parallelism (EP=4)              │
│                                         │
│  GPU 0: Expert 0, 1                     │
│  GPU 1: Expert 2, 3                     │
│  GPU 2: Expert 4, 5                     │
│  GPU 3: Expert 6, 7                     │
│                                         │
│  All-to-All 通信:                       │
│  ┌──┐  ┌──┐  ┌──┐  ┌──┐               │
│  │G0│←→│G1│←→│G2│←→│G3│               │
│  └──┘  └──┘  └──┘  └──┘               │
│  Token dispatch → Expert compute → Combine│
└────────────────────────────────────────┘
```

#### EP + TP 混合并行
- 小 MoE（8 expert）：EP=8 即可
- 大 MoE（256 expert, DeepSeek-V3）：EP=32 + TP=4
- All-to-All 通信是瓶颈 → 需要高速 NVLink/IB

### 4.4 DeepSeek-V3 训练工程亮点

| 技术 | 细节 |
|------|------|
| FP8 训练 | 前向/反向 GEMM 用 FP8，显著降低显存和通信 |
| Multi-Plane 通信 | All-to-All 和 DP gradient 分别走 IB 和 NVLink |
| Auxiliary-Loss-Free | Bias 动态调整替代辅助损失 |
| 256 Expert | 每 token 激活 8 个，总 671B 参数 |
| Pipeline-Free | 用 DualPipe 避免传统 PP 的 bubble |

---

## 五、MoE 推理挑战

### 5.1 显存问题
```
Mixtral 8x7B FP16: 8 × 7B × 2 bytes ≈ 93 GB (需要 2×H100)
但每 token 只激活 13B → 计算量和 dense 13B 接近
```

### 5.2 Expert Locality
- 推理时每个 batch 可能需要不同的 expert 组合
- GPU 需要加载所有 expert 的权重到 HBM
- **解决**：Expert Offloading（不活跃的 expert 放 CPU/NVMe）

### 5.3 All-to-All 通信延迟
- 推理时 batch 小 → All-to-All 通信占比更大
- **优化**：让同一个 GPU 处理完整的 token → 减少跨 GPU 通信

---

## 六、MoE 变体

### 6.1 Shared Expert (DeepSeek-V3)
```
output = shared_expert(x) + Σ(w_i × routed_expert_i(x))
```
- 1 个 always-active 的共享 expert → 保证基础能力
- 其余 expert 通过路由选择性激活

### 6.2 Fine-grained Expert Segmentation (DeepSeek-V3)
```
传统：8 个大 expert（每个 d_ff=14336）
DeepSeek：256 个小 expert（每个 d_ff 更小）
→ 更灵活的组合（C(256,8) >> C(8,2)）
```

### 6.3 Soft MoE (Google)
```
传统：硬路由（每个 token 只去 top-k expert）
Soft MoE：软路由（每个 token 的表示是所有 expert 的加权和）
```
- 训练更稳定，但推理时失去稀疏优势

---

## 面试高频问答

**Q1：MoE 的路由坍缩是什么？怎么解决？**
> 路由坍缩是所有 token 被路由到少数 expert，其他 expert 退化。解决方法：(1) 辅助负载均衡损失；(2) DeepSeek 的 bias 动态调整；(3) Token dropping 限制单个 expert 容量；(4) 合理的初始化策略。

**Q2：DeepSeek-V3 的 Auxiliary-Loss-Free 怎么实现的？**
> 对每个 expert 维护一个 bias 值，动态调整：过载 expert 减少 bias，空闲 expert 增加 bias。bias 加在 gate logits 上影响路由概率，不需要额外的 loss 项。好处是不干扰主训练目标。

**Q3：MoE 的训练和推理各有什么挑战？**
> 训练：路由坍缩、负载不均、token dropping 信息丢失、All-to-All 通信开销大。推理：所有 expert 权重需常驻显存（即使只激活少数）、batch 小时通信占比高。

**Q4：Upcycling 是什么？有什么优势？**
> 从训练好的 dense 模型出发，复制 FFN 为多个 expert（加扰动），继续少量训练让路由学会分工。优势：节省大量预训练 compute，避免从零训练 MoE 的不稳定性。

**Q5：为什么 DeepSeek-V3 用 256 个小 expert 而不是 8 个大 expert？**
> Fine-grained segmentation：小 expert 的组合空间指数级增大（C(256,8) >> C(8,2)），模型可以更精细地为不同 token 选择最合适的 expert 组合，提升表达能力。
