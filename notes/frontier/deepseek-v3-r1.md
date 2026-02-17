# DeepSeek V3 & R1 技术深度解析

## 一、DeepSeek-V3（2024.12 发布）

### 1.1 核心架构创新

#### Multi-head Latent Attention (MLA)
- **动机**：标准 MHA 的 KV Cache 随 head 数线性增长，GQA 虽然减少但损失精度
- **核心思想**：将 KV 投影到低秩 latent space，只缓存 latent vector
- **公式**：
  - 压缩：`c_kv = W_dkv @ h`，其中 `c_kv` 维度远小于原始 KV
  - 恢复：`k = W_uk @ c_kv`，`v = W_uv @ c_kv`
  - KV Cache 只存 `c_kv`（如 512d vs 原始 8192d），**压缩比 ~16x**
- **与 GQA 对比**：
  | 方法 | KV Cache/token | 精度损失 | 实现复杂度 |
  |------|---------------|---------|-----------|
  | MHA | `2 * n_heads * d_head * L` | 无 | 低 |
  | GQA-8 | `2 * n_groups * d_head * L` | 轻微 | 低 |
  | MLA | `2 * d_latent * L` | 几乎无 | 中 |
- **面试关键**：MLA 本质是在 attention 层做了一个 autoencoder，压缩 KV 表示

#### DeepSeekMoE 架构
- **设计**：256 个 routed experts + 1 shared expert
- **Fine-grained Expert Segmentation**：将大 expert 拆成更多小 expert，提高组合灵活性
- **Shared Expert Isolation**：1 个 always-active expert 保证基础能力，避免 expert collapse
- **Auxiliary-Loss-Free Load Balancing**：
  - 传统 MoE：用 auxiliary loss 平衡负载（如 Switch Transformer 的 z-loss）
  - DeepSeek：用 **bias term** 动态调整 gating score，不需要额外 loss
  - 原理：对负载低的 expert 增加 bias，对过载的减少 bias
- **Token dropping**：训练时 drop 超出 capacity 的 token，推理时不 drop

#### Multi-Token Prediction (MTP)
- 每个位置预测未来 **2 个** token（不只是 next-token）
- 额外的预测头共享 main model 的表示
- **推理加速**：可以作为 speculative decoding 的 draft head，零额外显存

### 1.2 训练工程

- **FP8 Mixed Precision Training**：
  - 前向/反向 GEMM 用 FP8（E4M3 + E5M2）
  - Master weights / optimizer states 保持 BF16/FP32
  - 节约 ~40% 显存 + ~1.5x 训练速度
- **DualPipe**：改良的 pipeline parallelism
  - 同时执行前向和反向计算，减少 bubble
  - 前向的 micro-batch N+1 和反向的 micro-batch N 重叠

### 1.3 性能数据
- 671B 参数，37B 激活参数
- 训练成本：**$5.57M**（2048 H800 训练 ~2 个月）
- 性能：MMLU 87.1，MATH 90.2，HumanEval 65.2%

---

## 二、DeepSeek-R1（2025.01 发布）

### 2.1 核心创新：纯 RL 训练推理能力

#### 训练流程（4 阶段）

Stage 1: Cold Start
  - 用少量 long-CoT 数据 SFT，让模型学会"展开推理"格式
  - 数据量：数千条（非常少）

Stage 2: Reasoning-oriented RL（核心）
  - 算法：GRPO (Group Relative Policy Optimization)
  - Reward：规则 reward（数学/代码有标准答案）+ format reward
  - 不使用 PRM / ORM！纯规则 reward
  - 模型自发涌现：self-verification, reflection, long chain-of-thought

Stage 3: Rejection Sampling + SFT
  - 从 Stage 2 模型采样，筛选高质量推理路径
  - 混合通用 SFT 数据，恢复非推理能力

Stage 4: 全任务 RL
  - 推理 + helpfulness + safety 多目标 RL
  - reward = rule_reward (推理) + reward_model (通用)

#### GRPO 算法详解
- **动机**：PPO 需要 critic model（显存翻倍），GRPO 去掉 critic
- **核心**：
  1. 对同一 prompt 采样 G 个回答 {o1, o2, ..., oG}
  2. 计算每个回答的 reward {r1, r2, ..., rG}
  3. Advantage = (ri - mean(r)) / std(r)（组内相对优势）
  4. Policy gradient 用 clipped surrogate objective（类似 PPO-clip）
- **公式**：
  `A_i = (r_i - mean(r_{1..G})) / std(r_{1..G})`
  `L = E[min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)] - beta * KL`
- **优势**：
  - 不需要 value network（节省 ~50% 显存）
  - 组内归一化天然稳定训练
  - 适合 reward 稀疏的场景（只有最终答案正确/错误）

#### Reasoning 涌现行为
- **Self-verification**：模型自主回头检查中间步骤
- **Reflection**："Wait, let me reconsider..."
- **Aha moments**："I realize my approach was wrong. Let me try..."
- 这些行为完全从 RL 中涌现，没有在 SFT 数据中显式教

### 2.2 蒸馏系列
- R1 蒸馏到小模型：1.5B, 7B, 8B, 14B, 32B, 70B
- 蒸馏数据：从 R1 采样的 800K 推理轨迹
- **关键发现**：蒸馏效果 > 直接对小模型做 RL
  - DeepSeek-R1-Distill-Qwen-32B > OpenAI o1-mini（多个 benchmark）

### 2.3 面试高频问答

**Q: DeepSeek-R1 和 OpenAI o1 的核心区别？**
- o1：用大量 PRM (Process Reward Model) 数据监督训练
- R1：纯 RL + 规则 reward，推理行为从 RL 中涌现
- R1 的训练成本显著更低

**Q: GRPO 相比 PPO 的优劣？**
- 优：不需要 critic，显存减半，实现更简单
- 劣：需要同时采样多个回答（但可以 pipeline 化）
- 适合推理任务（reward 容易定义），不适合 open-ended 对话

**Q: MLA 的 KV Cache 压缩原理？为什么不损失精度？**
- KV 的有效信息维度远低于其表示维度（低秩假设）
- MLA 在训练时 end-to-end 学习压缩/恢复矩阵
- 类比：PCA 降维后恢复，但这里是 learned projection

**Q: DeepSeek 的训练成本为什么这么低？**
- FP8 训练（减少显存和计算）
- MoE 架构（只激活 37B/671B）
- DualPipe 减少 pipeline bubble
- Auxiliary-loss-free load balancing 提高 GPU 利用率
