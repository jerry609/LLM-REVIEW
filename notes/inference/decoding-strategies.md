# 解码策略全景 (Decoding Strategies)

> Greedy / Beam Search / Top-K / Top-P / Temperature / Min-P / 对比

---

## 一、核心概念

LLM 推理时，每一步从 vocabulary 上的概率分布中选择下一个 token：
```
logits = model(input_ids)[-1]       # [vocab_size]
probs  = softmax(logits / temperature)
next_token = sample(probs)          # 怎么 sample？
```

---

## 二、确定性解码

### 2.1 Greedy Decoding（贪心）
```python
next_token = argmax(probs)
```
- 每步选概率最大的 token
- **优点**：最快，确定性
- **缺点**：容易重复、生硬、陷入循环
- **适用**：代码生成、数学推理（需要确定性答案）

### 2.2 Beam Search（束搜索）
```python
# 维护 beam_width 个候选序列
# 每步扩展每个候选的 top-k token
# 保留总概率最高的 beam_width 个
```
- **beam_width=1** 退化为 greedy
- **优点**：搜索空间更大，找到更高概率的完整序列
- **缺点**：
  - 计算量 ×beam_width
  - 容易生成通用、无趣的回复（"I don't know"）
  - 长序列时概率偏向短序列 → 需要 length penalty
- **Length Penalty**：`score = log_prob / length^α`，α>1 偏好长序列
- **适用**：机器翻译、摘要（质量>多样性的场景）

---

## 三、随机采样策略

### 3.1 Temperature（温度）
```python
probs = softmax(logits / T)
```
- **T < 1**：分布更尖锐 → 更确定（接近 greedy）
- **T = 1**：原始分布
- **T > 1**：分布更平坦 → 更随机（更有创造性）
- **T → 0**：等价于 greedy
- **T → ∞**：等价于均匀分布

| Temperature | 效果 | 适用场景 |
|-------------|------|---------|
| 0.1-0.3 | 非常确定 | 数学、代码、事实性 QA |
| 0.5-0.7 | 平衡 | 通用对话 |
| 0.8-1.0 | 较随机 | 创意写作 |
| 1.0-1.5 | 很随机 | 头脑风暴 |

### 3.2 Top-K Sampling
```python
top_k_logits = topk(logits, k)
probs = softmax(top_k_logits / T)
next_token = sample(probs)
```
- 只从概率最高的 K 个 token 中采样
- **K 太小**（如 K=1 = greedy）：缺乏多样性
- **K 太大**（如 K=50000）：可能采到低质量 token
- **问题**：K 是固定的，但不同位置的概率分布形状不同
  - 有时分布很尖锐（只有 2-3 个合理 token） → K=50 太大
  - 有时分布很平坦（很多 token 都合理） → K=50 太小

### 3.3 Top-P (Nucleus) Sampling ⭐
```python
sorted_probs = sort(probs, descending=True)
cumsum = cumulative_sum(sorted_probs)
mask = cumsum <= p  # 保留累计概率 <= p 的 token
# 从 mask 内的 token 中采样
```
- 动态选择候选集大小：保留累计概率达到 P 的最小 token 集合
- **P = 0.9**：保留前 90% 概率质量的 token
- **P = 1.0**：等价于无截断（full sampling）
- **优于 Top-K**：自适应候选集大小
  - 分布尖锐时：自动只保留少数 token
  - 分布平坦时：自动保留更多 token

### 3.4 Min-P Sampling（新方法）
```python
threshold = p_min * max(probs)  # 相对阈值
mask = probs >= threshold
next_token = sample(probs[mask])
```
- 设置相对最大概率的阈值（如 min_p=0.05 → 概率 < 最大概率的 5% 的 token 被过滤）
- **优势**：比 Top-P 更好地处理极端分布
- **直觉**：如果最可能的 token 概率是 0.8，min_p=0.05 → 只保留概率 ≥ 0.04 的 token

### 3.5 Typical Sampling
```python
# 选择"信息量最典型"的 token（接近平均信息量）
entropy = -sum(p * log(p))
info = -log(p)
# 保留 |info - entropy| 最小的 token
```
- 基于信息论：选择信息量接近分布熵的 token
- 避免选择"太可预测"或"太惊讶"的 token

---

## 四、采样组合策略

### 4.1 常见组合
```python
# 典型的 LLM 采样流程
logits = model(input_ids)

# Step 1: Temperature
logits = logits / temperature

# Step 2: Top-K (可选)
logits = top_k_filter(logits, k=50)

# Step 3: Top-P
logits = top_p_filter(logits, p=0.9)

# Step 4: Sample
probs = softmax(logits)
next_token = multinomial(probs)
```

### 4.2 Repetition Penalty
```python
for token in generated_tokens:
    if logits[token] > 0:
        logits[token] /= penalty  # 降低已生成 token 的概率
    else:
        logits[token] *= penalty
```
- **repetition_penalty > 1**：抑制重复（通常 1.1-1.3）
- **frequency_penalty**：按出现频率线性惩罚
- **presence_penalty**：只要出现过就惩罚（不管频率）

---

## 五、各策略对比

| 策略 | 确定性 | 多样性 | 质量 | 速度 | 适用场景 |
|------|--------|--------|------|------|---------|
| Greedy | ✅✅✅ | ❌ | 中 | **最快** | 代码/数学 |
| Beam Search | ✅✅ | ❌ | **高** | 慢 | 翻译/摘要 |
| Temperature | 可调 | 可调 | 可调 | 快 | 通用 |
| Top-K | 中 | 中 | 中 | 快 | 通用 |
| **Top-P** | 中 | **好** | **高** | 快 | **通用推荐** |
| Min-P | 中 | 好 | 高 | 快 | 新方法 |
| Typical | 中 | 好 | 高 | 中 | 实验性 |

### 推荐配置

| 场景 | Temperature | Top-P | Top-K | Rep Penalty |
|------|-------------|-------|-------|-------------|
| 代码生成 | 0.0-0.2 | 1.0 | — | 1.0 |
| 数学推理 | 0.0 | 1.0 | — | 1.0 |
| 通用对话 | 0.7 | 0.9 | 50 | 1.1 |
| 创意写作 | 0.9-1.0 | 0.95 | — | 1.2 |
| RL Rollout | 1.0 | 1.0 | — | 1.0 |

---

## 六、Speculative Decoding（投机解码）中的采样

- Draft 模型用 `T=1.0, top_p=1.0`（完整分布采样）
- Target 模型验证时：
  ```
  accept_prob = min(1, p_target(x) / p_draft(x))
  ```
- **关键**：接受-拒绝采样保证最终分布精确等于 target 模型的分布

---

## 七、Constrained Decoding（约束解码）

### 结构化输出
- **JSON Mode**：只允许生成合法 JSON 的 token
- **Function Calling**：引导生成特定格式的函数调用
- **Grammar-based**：用 CFG/正则表达式约束生成

### 实现方式
```python
# 在每步 logits 上施加 mask
valid_tokens = grammar.get_valid_tokens(current_state)
logits[~valid_tokens] = -inf
# 然后正常采样
```

- **Outlines**：基于 FSM 的高效约束解码
- **vLLM guided decoding**：支持 JSON schema / regex 约束
- **SGLang**：内置 structured output 支持

---

## 面试高频问答

**Q1：Top-K 和 Top-P 有什么区别？为什么 Top-P 更好？**
> Top-K 固定选 K 个 token，但不同位置的概率分布形状不同，K 值难以适配所有情况。Top-P 动态选择累计概率达到 P 的最小集合，能自适应分布形状——尖锐时少选、平坦时多选。

**Q2：Temperature 的数学含义是什么？**
> Temperature 缩放 softmax 前的 logits：`softmax(logits/T)`。T<1 使分布更尖锐（更确定），T>1 使分布更平坦（更随机），T→0 等价于 argmax。

**Q3：为什么 RL 训练的 rollout 通常用 T=1.0？**
> RL 需要从策略的真实分布中采样来计算 log_prob 和 advantage，Temperature≠1 会扭曲分布导致 importance sampling ratio 偏差。

**Q4：Beam Search 为什么不适合对话生成？**
> Beam Search 优化的是序列概率，容易生成通用、安全但无趣的回复（"高概率 ≠ 高质量"）。对话需要多样性和创造性，随机采样更合适。

**Q5：如何强制 LLM 输出合法 JSON？**
> Constrained decoding：每步根据当前 JSON 状态（用 FSM 或 parser 追踪），将不合法的 token 的 logits 设为 -inf，只允许合法 token 被采样。vLLM 和 SGLang 都内置支持。
