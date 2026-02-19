# 后训练高级方法：RLAIF / SimPO / KTO / ORPO / IPO

> 在 RLHF 和 DPO 基础上的新一代对齐方法

---

## 一、后训练全景图

```
预训练 (CLM)
    │
    ▼
SFT (Supervised Fine-Tuning)
    │  ← 用对话数据监督微调
    ▼
偏好对齐（选择以下之一或多种）
    ├── RLHF (PPO)        ← 经典，但复杂
    ├── DPO               ← 简化 RLHF，主流
    ├── RLAIF             ← 用 AI 替代人类反馈
    ├── SimPO             ← 去掉参考模型
    ├── KTO               ← 不需要偏好对
    ├── ORPO              ← SFT + 对齐一步到位
    └── IPO               ← DPO 的理论改进
    │
    ▼
安全对齐 / Red Teaming → 部署
```

---

## 二、经典方法回顾

### 2.1 RLHF (PPO)
```
Step 1: 训练 Reward Model (RM)：从偏好对 (y_w > y_l) 学习打分
Step 2: PPO 优化：max E[R(y)] - β·KL(π||π_ref)
```
- **优点**：效果最强，灵活性高
- **缺点**：需要 4 个模型（Policy + Ref + RM + Critic），训练不稳定

### 2.2 DPO (Direct Preference Optimization)
```
L_DPO = -E[log σ(β(log π/π_ref(y_w) - log π/π_ref(y_l)))]
```
- **核心洞察**：最优 RM 可以用 policy 的 log-ratio 直接表示
- **优点**：不需要训练 RM，稳定高效
- **缺点**：仍需要参考模型 `π_ref`（显存 ×2）

---

## 三、新一代对齐方法

### 3.1 RLAIF（RL from AI Feedback）

```
传统 RLHF：人类标注偏好对 → 训练 RM → PPO
RLAIF：    AI 模型标注偏好对 → 训练 RM → PPO (或 DPO)
```

#### 核心思想
- 用强大的 AI 模型（如 GPT-4, Claude）替代人类做偏好标注
- 大幅降低标注成本，提高标注一致性

#### Constitutional AI (Anthropic)
```
1. Critique：AI 自己评价回复是否有害
2. Revision：AI 根据"宪法原则"修改回复
3. RL：用修改后的数据做 RLHF
```
- "宪法原则"是预定义的安全规则（如"回复应该无害"）
- 减少了人工干预，特别适合安全对齐

#### RLAIF 的变体
| 方法 | AI 的角色 | 对比 |
|------|----------|------|
| 直接标注 | AI 生成偏好对 (y_w, y_l) | 最简单 |
| AI 打分 | AI 给回复打分 → 排序 | 更细粒度 |
| 自我改进 | AI 生成 → 自我评价 → 筛选 | Self-Play |
| Constitutional | AI 按规则自我修正 | 更可控 |

#### 局限性
- AI 偏见被放大（"model collapse"风险）
- 对 AI 评判者的质量高度依赖
- 某些细微偏好（幽默感、文化敏感度）AI 难以判断

---

### 3.2 SimPO (Simple Preference Optimization) ⭐ 2024 热门

```
L_SimPO = -E[log σ(β/|y| · (log π(y_w) - log π(y_l)) - γ)]
```

#### 核心创新
1. **去掉参考模型**：不再需要 `π_ref`，只用当前策略 `π` 的 log-prob
2. **长度归一化**：`log π(y) / |y|`，避免偏向短回复
3. **目标 reward margin**：γ 参数确保 win/lose 之间有明确差距

#### 与 DPO 的区别
| 特性 | DPO | SimPO |
|------|-----|-------|
| 参考模型 | ✅ 需要 π_ref | **❌ 不需要** |
| 显存 | 2× policy 模型 | **1× policy 模型** |
| 长度偏好 | 倾向长回复 | **长度归一化，公平** |
| Reward margin | 无 | **有 (γ)** |
| 效果 | 强 | **更强**（多个 benchmark 超越 DPO） |

#### 为什么不需要参考模型？
- DPO 用 `log π/π_ref` 隐式定义 reward
- SimPO 直接用 `log π / |y|`（平均 log-prob）作为 reward
- 直觉：**高平均概率的序列 = 高质量的序列**

---

### 3.3 KTO (Kahneman-Tversky Optimization)

```
L_KTO = E_w[w(y_w) · (1 - σ(r_w - z_ref))]
      + E_l[w(y_l) · (1 - σ(z_ref - r_l))]
```

#### 核心创新
- **不需要偏好对**：只需要"好回复"和"坏回复"的独立标注
- 灵感来自**前景理论**（Kahneman & Tversky 的行为经济学）
  - 人类对损失更敏感（"loss aversion"）
  - KTO 给 bad 样本更大权重

#### 与 DPO 的区别
| 特性 | DPO | KTO |
|------|-----|-----|
| 数据需求 | **偏好对** (y_w, y_l) 配对 | **独立**标注（好/坏） |
| 标注成本 | 高（需要比较两个回复） | **低**（只需判断好坏） |
| 数据利用 | 一个偏好对 = 一个样本 | 每个回复独立使用 |
| 效果 | 强 | 接近 DPO |

#### 适用场景
- 有大量"好/坏"评价但缺少偏好对的场景
- 例如：用户 upvote/downvote 数据
- 比 DPO 数据收集成本低很多

---

### 3.4 ORPO (Odds Ratio Preference Optimization)

```
L_ORPO = L_SFT(y_w) + λ · log_odds_ratio(y_w, y_l)

log_odds_ratio = log(P(y_w)/(1-P(y_w))) - log(P(y_l)/(1-P(y_l)))
```

#### 核心创新
- **SFT 和对齐一步完成**：不需要先 SFT 再 DPO
- 用 Odds Ratio（几率比）代替 log probability ratio
- 无需参考模型

#### 与 DPO 的区别
| 特性 | DPO | ORPO |
|------|-----|------|
| 训练阶段 | SFT → DPO (两步) | **一步完成** |
| 参考模型 | ✅ 需要 | **❌ 不需要** |
| 基础损失 | 只有偏好损失 | **SFT loss + 偏好 loss** |
| 效率 | 两次训练 | **节省一半时间** |

---

### 3.5 IPO (Identity Preference Optimization)

```
L_IPO = E[(log π/π_ref(y_w) - log π/π_ref(y_l) - 1/(2β))²]
```

#### 核心改进
- DPO 假设 Bradley-Terry 偏好模型（可能不准确）
- IPO 使用更一般的偏好假设
- 正则化效果更好，避免 DPO 的 "over-optimization"

#### DPO 的问题
- DPO 可能让 log-ratio 无限增大（把 lose 的概率压到 0）
- 这导致"过度优化"：对训练集过拟合
- IPO 的平方损失自然限制了优化幅度

---

### 3.6 SPIN (Self-Play Fine-Tuning)

```
Round 1: 模型 M_0 生成回复 → 与人类回复对比 → DPO 训练 → M_1
Round 2: M_1 生成回复 → 与人类回复对比 → DPO 训练 → M_2
...
```
- 自博弈：每轮用上一轮模型的生成作为 "lose"，人类回复作为 "win"
- 不断逼近人类水平
- 不需要额外的偏好标注

---

## 四、总对比表

| 方法 | 需要 RM | 需要 π_ref | 需要偏好对 | 训练阶段 | 稳定性 | 效果 |
|------|---------|-----------|----------|---------|--------|------|
| RLHF (PPO) | ✅ | ✅ | ✅ | SFT→RM→PPO | 难调 | **最强** |
| DPO | ❌ | ✅ | ✅ | SFT→DPO | 稳定 | 强 |
| **SimPO** | ❌ | **❌** | ✅ | SFT→SimPO | **很稳** | **很强** |
| KTO | ❌ | ✅ | **❌** | SFT→KTO | 稳定 | 接近 DPO |
| ORPO | ❌ | **❌** | ✅ | **一步** | 很稳 | 中上 |
| IPO | ❌ | ✅ | ✅ | SFT→IPO | 很稳 | 接近 DPO |
| RLAIF | ✅/❌ | ✅ | ✅(AI标) | 同 RLHF/DPO | 同上 | 取决于 AI |

---

## 五、2024-2025 趋势

### 趋势 1：去参考模型化
- SimPO、ORPO 都去掉了参考模型 → 显存减半
- 实验表明效果不降甚至更好

### 趋势 2：RL 回归
- DeepSeek-R1 证明 RL (GRPO) 在推理能力上远超 DPO
- 复杂推理任务 → RL 优势明显
- 简单对齐任务 → DPO/SimPO 足够

### 趋势 3：过程奖励 (Process Reward)
- 不只给最终回复打分，给每个推理步骤打分
- Math/Code 任务效果显著提升
- DeepSeek-R1、OpenAI o1 都使用过程奖励

### 趋势 4：在线 vs 离线
```
离线：DPO/SimPO/KTO (用固定数据集训练)
在线：PPO/GRPO (实时生成 + 训练)
```
- 在线方法效果更好但工程复杂度高
- 离线方法简单高效但可能过拟合训练数据

---

## 面试高频问答

**Q1：DPO 和 RLHF(PPO) 的核心区别是什么？**
> DPO 证明了最优 RM 可以用 policy 的 log-ratio 隐式表示，因此不需要显式训练 RM 和 PPO 过程。只需要一个 DPO loss 就能直接从偏好数据优化策略。

**Q2：SimPO 为什么不需要参考模型？**
> SimPO 用序列的平均 log-probability（归一化 log π(y)/|y|）直接作为隐式 reward，而 DPO 需要 log π/π_ref 来定义 reward。去掉参考模型后显存减半，且长度归一化避免了长度偏好问题。

**Q3：KTO 的数据需求和 DPO 有什么区别？**
> DPO 需要配对的偏好数据 (y_w, y_l)，KTO 只需要独立标注的好/坏回复。KTO 灵感来自前景理论的 loss aversion，对"坏回复"给更大的惩罚权重。

**Q4：为什么 DeepSeek-R1 用 GRPO 而不是 DPO？**
> 推理任务需要模型学习新的思维模式（Thinking Tokens），这本质上是探索-利用问题。在线 RL (GRPO) 可以不断生成新样本并从中学习，而离线 DPO 只能从固定数据中学习，探索能力有限。

**Q5：RLAIF 的主要风险是什么？**
> AI 偏见放大（强化 AI 自身的偏好而非人类偏好）、"model collapse"（AI 评判 AI 生成的内容）、文化/价值观盲区。缓解方法：人工抽检、多模型交叉验证、设置明确的"宪法原则"。
