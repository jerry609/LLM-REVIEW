# 奖励模型训练 + RLHF 实战 Checklist

> 从 RM 训练到 PPO/DPO/GRPO 实战的完整工程指南

---

## 一、奖励模型 (Reward Model) 训练

### 1.1 RM 的本质

```
给定一个 prompt x 和回复 y，RM 输出一个标量分数：
r = RM(x, y) ∈ R

训练目标：人类偏好的回复得分更高
Loss = -E[log σ(r(x, y_w) - r(x, y_l))]   (Bradley-Terry 模型)

y_w: chosen (人类偏好的回复)
y_l: rejected (人类不偏好的回复)
```

### 1.2 RM 架构

```
方式 1 (经典)：在 LLM 末尾加一个线性 head
    LLM → 最后一个 token 的 hidden_state → Linear(d, 1) → scalar reward

方式 2 (序列级)：用特殊 token 的 hidden_state
    [prompt tokens] [response tokens] [REWARD_TOKEN]
                                          ↓
                                    Linear → scalar

方式 3 (过程奖励模型 PRM)：对每个 token/步骤打分
    适用于数学推理：每个推理步骤独立评分
```

### 1.3 RM 训练代码

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig

# 1. 模型 —— 通常用比 Policy 小一号的模型
model = AutoModelForSequenceClassification.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    num_labels=1,          # 回归任务，输出标量
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

# 2. 数据格式
# {"chosen": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
#  "rejected": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

# 3. 训练
reward_config = RewardConfig(
    output_dir="reward_model",
    per_device_train_batch_size=4,
    num_train_epochs=1,
    learning_rate=1e-5,
    bf16=True,
    max_length=2048,
    logging_steps=10,
)

trainer = RewardTrainer(
    model=model,
    args=reward_config,
    train_dataset=preference_dataset,
    processing_class=tokenizer,
)

trainer.train()
```

### 1.4 RM 评估指标

| 指标 | 公式 | 好的范围 |
|------|------|---------|
| **Accuracy** | P(r(y_w) > r(y_l)) | >70% (人类一致性约 75%) |
| **Reward Margin** | mean(r(y_w) - r(y_l)) | 持续正值且稳定 |
| **Calibration** | reward 分布合理 | 不应退化为常数 |

---

## 二、PPO 实战

### 2.1 PPO 训练流程

```
每个训练 Step:
1. Policy 生成:     y ~ π_θ(·|x)       ← 采样回复
2. Reward 评分:     r = RM(x, y)        ← 给回复打分
3. KL 惩罚:         r' = r - β·KL(π_θ||π_ref)  ← 防止偏离太远
4. GAE 优势估计:    Â_t = δ_t + (γλ)δ_{t+1} + ...  ← 每个 token 的优势
5. PPO 更新:        max L_clip(θ) - c₁·L_VF + c₂·H  ← 裁剪目标

需要同时维护 4 个模型:
├── Policy Model (π_θ)      ← 被优化
├── Reference Model (π_ref)  ← 冻结，计算 KL
├── Reward Model             ← 冻结，打分
└── Value Model (Critic)     ← 被优化，估计 V(s)
```

### 2.2 PPO 代码

```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# 1. Policy + Value Head
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.bfloat16,
    peft_config=lora_config,     # 可选 LoRA
)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.bfloat16,
)

# 2. 已训练好的 RM
reward_model = AutoModelForSequenceClassification.from_pretrained("reward_model")

# 3. PPO 配置
ppo_config = PPOConfig(
    learning_rate=1e-6,
    batch_size=16,
    mini_batch_size=4,
    gradient_accumulation_steps=4,
    ppo_epochs=4,          # 每批数据上 PPO 迭代次数
    kl_penalty="kl",
    init_kl_coef=0.2,      # β: KL 惩罚系数
    target_kl=6.0,         # 自适应 KL 目标
    cliprange=0.2,         # PPO clip 范围
    vf_coef=0.1,           # Value loss 系数
)

# 4. 训练循环
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
)

for batch in prompt_dataloader:
    # 4a. 生成回复
    query_tensors = tokenizer(batch["prompt"], return_tensors="pt", padding=True).input_ids
    response_tensors = ppo_trainer.generate(query_tensors, max_new_tokens=256)

    # 4b. 计算 reward
    texts = tokenizer.batch_decode(response_tensors)
    rewards = compute_rewards(reward_model, batch["prompt"], texts)

    # 4c. PPO 更新
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)
```

### 2.3 PPO 调参要点

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `learning_rate` | 1e-6 ~ 5e-6 | 比 SFT 低一个数量级 |
| `init_kl_coef (β)` | 0.05-0.2 | β 太大 → 不敢探索；β 太小 → 偏离太远 |
| `cliprange` | 0.2 | PPO 核心，限制策略更新幅度 |
| `ppo_epochs` | 2-4 | 每批数据的重复训练次数 |
| `target_kl` | 3-8 | KL 自适应调节目标 |
| `batch_size` | 32-128 | 越大越稳定 |

---

## 三、DPO 实战

### 3.1 DPO 优势

```
DPO = 将 RM 训练和 PPO 优化合并为一个 loss

L_DPO = -E[log σ(β(log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]

只需要:
├── Policy Model (π_θ)      ← 被优化
└── Reference Model (π_ref)  ← 冻结 (LoRA 时自动用底座)
```

### 3.2 DPO 训练 Checklist

```
数据准备:
□ 格式正确：{"prompt": "...", "chosen": "...", "rejected": "..."}
□ chosen 和 rejected 来自同一个 prompt
□ 偏好标注一致性检查（人类一致率应 > 70%）
□ chosen 不能太长（避免长度偏好）

训练配置:
□ β = 0.1 (默认值)，可尝试 0.05-0.5
□ 学习率 5e-7 ~ 5e-6 （比 SFT 更低）
□ 1 epoch 通常足够（过拟合风险高）
□ max_prompt_length 不要太大（节省显存）

训练监控:
□ reward/chosen 和 reward/rejected 的差距在增大
□ reward/margin > 0 且缓慢增长
□ reward/accuracy > 0.5 且上升
□ loss 平稳下降
□ KL 散度不要爆炸（<10）

训练后验证:
□ 人工评测 Win Rate 提升
□ 通用 Benchmark 不降（检查灾难性遗忘）
□ 检查是否过度冗长（DPO 的常见问题）
```

---

## 四、GRPO 实战

### 4.1 GRPO 核心原理

```
GRPO = PPO 的简化版本，去掉了 Critic 模型

流程:
1. 对每个 prompt 生成 G 个候选回复: {y₁, y₂, ..., y_G}
2. 对每个回复计算 reward: {r₁, r₂, ..., r_G}
3. 组内归一化: â_i = (r_i - mean(r)) / std(r)
4. 策略梯度: L = -E[min(ρ·â, clip(ρ,1±ε)·â)] + β·KL

关键: 不需要 Value Model (Critic)，用组内对比替代
```

### 4.2 GRPO 代码

```python
from trl import GRPOTrainer, GRPOConfig

# 自定义奖励函数（可以是规则/模型/混合）
def reward_fn(completions, prompts, **kwargs):
    rewards = []
    for completion, prompt in zip(completions, prompts):
        score = 0.0
        # 规则奖励示例
        if "答案" in completion and len(completion) > 50:
            score += 1.0
        if "<think>" in completion:  # 鼓励思考
            score += 0.5
        # 也可以调用 RM 模型
        # score = reward_model.score(prompt, completion)
        rewards.append(score)
    return rewards

config = GRPOConfig(
    output_dir="grpo_output",
    per_device_train_batch_size=2,
    num_generations=8,          # 每个 prompt 生成 8 个候选
    max_completion_length=512,
    num_train_epochs=1,
    learning_rate=5e-7,
    beta=0.04,                  # KL 系数
    bf16=True,
    logging_steps=5,
    gradient_accumulation_steps=4,
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    args=config,
    train_dataset=prompt_dataset,
    reward_funcs=reward_fn,
    processing_class=tokenizer,
)

trainer.train()
```

---

## 五、PPO vs DPO vs GRPO 实战对比

| 维度 | PPO | DPO | GRPO |
|------|-----|-----|------|
| **数据** | 只需 prompts | 偏好对 (w/l) | 只需 prompts |
| **模型数** | 4 个 | 2 个 | 2 个 |
| **显存** | 极高 | 中等 | 中等 |
| **工程复杂度** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **训练稳定性** | 难调 | 稳定 | 较稳定 |
| **推理任务** | 强 | 中 | **最强** |
| **通用对齐** | 强 | **最适合** | 强 |
| **代表模型** | InstructGPT | Llama-3 | **DeepSeek-R1** |

### 选型建议

```
任务需求:
├── 通用对话对齐     → DPO（简单高效，偏好数据易获取）
├── 数学/代码推理    → GRPO（在线探索，适合可验证任务）
├── 极致效果        → PPO（灵活但需要大量工程）
└── 安全对齐        → DPO + Red Teaming

资源约束:
├── 单卡/小资源     → DPO + QLoRA
├── 多卡/中资源     → GRPO
└── 大规模集群      → PPO (完整 RLHF)
```

---

## 六、常见问题与调试

### 6.1 Reward Hacking

```
症状: reward 持续上升但回复质量下降
原因: 模型找到了 RM 的漏洞（如特定短语总是高分）
解决:
  - 增大 KL 惩罚 (β)
  - 改进 RM（更多样的训练数据）
  - 对 reward 设上限 (clip reward)
  - 用过程奖励替代结果奖励
```

### 6.2 KL 散度爆炸

```
症状: KL(π_θ||π_ref) 快速增大到 >20
原因: 策略偏离参考模型太远
解决:
  - 降低学习率
  - 增大 β (KL 惩罚)
  - 开启 KL 自适应调节 (target_kl)
  - 减小 PPO cliprange
```

### 6.3 模式坍缩 (Mode Collapse)

```
症状: 所有回复变得高度相似
原因: 策略退化到只产生一种高 reward 的输出
解决:
  - 降低温度采样 → 提高温度 (temperature > 1.0)
  - 增大 KL 惩罚
  - 使用 entropy bonus (增加探索)
  - 多样性 reward (惩罚重复)
```

---

## 面试高频问答

**Q1：PPO 中 4 个模型各自的作用？**
> Policy (π_θ) 是被优化的策略模型；Reference (π_ref) 是冻结的参考模型，用于计算 KL 惩罚防止偏离太远；Reward Model 给回复打分；Value Model (Critic) 估计状态价值 V(s)，用于计算 GAE 优势。

**Q2：DPO 为什么不需要 RM？**
> DPO 论文证明，在 Bradley-Terry 偏好模型下，最优 Reward 可以用 policy 和 reference policy 的 log-ratio 隐式表示：r*(x,y) = β log(π*(y|x)/π_ref(y|x)) + C。因此偏好数据的优化可以直接通过 policy 的 log-prob 差异实现，跳过了显式 RM 训练。

**Q3：GRPO 和 PPO 的核心区别？**
> GRPO 去掉了 Critic (Value Model)，改用 Group 内的 reward 均值和方差做归一化来估计优势（â = (r - mean) / std）。这简化了工程实现（少维护一个模型），且在可验证任务（数学/代码）上效果比 PPO 更好，因为这些任务有确定性的 reward。

**Q4：什么是 Reward Hacking？如何避免？**
> 模型学会了利用 RM 的缺陷来获取高分，而不是真正提高回复质量。例如产出冗长但废话连篇的回复，或者总是加上某个高分短语。防御手段：增大 KL 惩罚、改进 RM 的鲁棒性、对 reward 设上限、使用过程奖励而非仅结果奖励。

## 面试一句话
- "RM 训练用 Bradley-Terry 模型从偏好对学习打分，PPO 需要 4 个模型工程复杂但效果最强，DPO 绕过 RM 直接优化偏好概率简单高效，GRPO 用组内归一化替代 Critic 适合推理任务。工程上最关键的是 KL 控制和防止 Reward Hacking。"
