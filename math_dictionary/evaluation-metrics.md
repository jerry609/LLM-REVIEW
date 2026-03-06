# LLM 常见评测指标数学详解

> **核心定位**：为 LLM 推理系统中涉及的所有评测指标提供严格的数学定义，覆盖语言建模（PPL）、生成质量（BLEU/ROUGE/BERTScore）、代码生成（Pass@k）、校准误差（ECE）、长上下文评测，以及安全和线上 A/B 测试的统计框架。

---

## 1. 困惑度 (Perplexity)

### 1.1 定义

$$
\boxed{\text{PPL} = \exp\!\left(-\frac{1}{N}\sum_{t=1}^{N}\log p_\theta(x_t \mid x_{<t})\right) = \exp(\text{CE})}
$$

### 1.2 性质

| PPL 值 | 含义 |
|:------:|------|
| $1$ | 完美预测（每个 token 概率为 $1$） |
| $10$ | 平均每步等价于从 $10$ 个等概率选项中选 |
| $V$ | 等于随机猜（$V$ 为词表大小） |

### 1.3 注意事项

- **不同 Tokenizer 的 PPL 不可直接比较**：Token 粒度不同（中文 Tokenizer 的 PPL 天然偏低，因为每个 token 承载更多信息）。
- **PPL 低 $\ne$ 任务准确率高**：PPL 是语言建模的全局指标，对特定下游任务不一定有好的区分度。
- **常用于**：语言建模基准、量化/压缩后的质量评估（$\Delta\text{PPL} < 0.5$ 通常可接受）。

---

## 2. 分类与抽取任务

### 2.1 Precision / Recall / F1

$$
\text{Precision} = \frac{TP}{TP + FP} \quad (\text{预测为正中实际为正的比例})
$$
$$
\text{Recall} = \frac{TP}{TP + FN} \quad (\text{实际为正中被预测为正的比例})
$$
$$
\text{F1} = \frac{2 \cdot P \cdot R}{P + R} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}
$$

F1 是 Precision 和 Recall 的**调和平均**（对极端值更敏感）。

### 2.2 Accuracy

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

**类别不平衡时 F1 比 Accuracy 更有参考价值**。

---

## 3. 代码生成 Pass@k

### 3.1 无偏估计

从 $n$ 个生成样本中，有 $c$ 个通过测试。$\text{Pass@}k$ 的无偏估计为：

$$
\boxed{\text{Pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}}
$$

当 $k = 1$：$\text{Pass@}1 = c / n$（简单经验估计）。

### 3.2 直觉

$\text{Pass@}k$ = 从 $n$ 个样本中选 $k$ 个，**至少有一个正确**的概率。

---

## 4. 生成质量指标

### 4.1 BLEU (Bilingual Evaluation Understudy)

$$
\text{BLEU} = \text{BP} \cdot \exp\!\left(\sum_{n=1}^{N} w_n \log p_n\right)
$$

| 组件 | 公式 | 含义 |
|------|------|------|
| $p_n$ | $\frac{\sum_{\text{n-gram}} \min(\text{count}_{\text{hyp}}, \text{count}_{\text{ref}})}{\sum_{\text{n-gram}} \text{count}_{\text{hyp}}}$ | 修正的 N-gram Precision |
| $w_n$ | 通常 $1/N$（均匀权重） | N-gram 权重 |
| BP | $\min\!\left(1, \exp\!\left(1 - \frac{\lvert \text{ref} \rvert}{\lvert \text{hyp} \rvert}\right)\right)$ | Brevity Penalty（短句惩罚） |

**局限**：基于 N-gram 匹配，不考虑语义相似性。

### 4.2 ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

- **ROUGE-N**：N-gram 的 **Recall**（参考摘要中的 N-gram 被生成覆盖的比例）。
- **ROUGE-L**：基于最长公共子序列（LCS）：

$$
\text{ROUGE-L} = F_1(\text{LCS}) = \frac{(1 + \beta^2) \cdot P_{\text{LCS}} \cdot R_{\text{LCS}}}{\beta^2 \cdot P_{\text{LCS}} + R_{\text{LCS}}}
$$

### 4.3 BERTScore

用预训练 BERT 的 Embedding 计算 Token 级余弦相似度：

$$
R_{\text{BERT}} = \frac{1}{|\text{ref}|}\sum_{r \in \text{ref}} \max_{h \in \text{hyp}} \cos(e_r, e_h)
$$
$$
P_{\text{BERT}} = \frac{1}{|\text{hyp}|}\sum_{h \in \text{hyp}} \max_{r \in \text{ref}} \cos(e_h, e_r)
$$
$$
F_{\text{BERT}} = \frac{2 P_{\text{BERT}} R_{\text{BERT}}}{P_{\text{BERT}} + R_{\text{BERT}}}
$$

**优势**：捕捉语义相似性（同义词替换不会被 BLEU 捕获，但 BERTScore 可以）。

---

## 5. 校准 (Calibration)

### 5.1 Expected Calibration Error (ECE)

将预测按置信度分为 $M$ 个桶（Bin），计算每桶内准确率与置信度的偏差：

$$
\boxed{\text{ECE} = \sum_{b=1}^{M} \frac{|B_b|}{N} \cdot |\text{acc}(B_b) - \text{conf}(B_b)|}
$$

- $\text{acc}(B_b)$：桶 $b$ 内的实际准确率。
- $\text{conf}(B_b)$：桶 $b$ 内的平均置信度。

### 5.2 LLM 的常见问题

大模型通常**过度自信**（$\text{conf} > \text{acc}$）：模型给出高置信度但实际经常错误。

---

## 6. 长上下文评测

### 6.1 Needle-in-a-Haystack (NIAH)

在长文本中的**随机位置**插入一段特定信息（"Needle"），测试模型能否准确检索。

评测矩阵：$\text{文本长度} \times \text{插入位置}$，生成热力图。

### 6.2 RULER Benchmark

多种长上下文能力的综合测试：
- 单 Key 检索、多 Key 检索
- Key-Value 关联
- 长文本摘要

### 6.3 与压缩/驱逐策略的关系

$$
\Delta\text{Quality} = \text{Quality}_{\text{compressed}} - \text{Quality}_{\text{baseline}}
$$

压缩后在 NIAH 上的退化直接反映了**信息丢失**的严重程度。

---

## 7. 线上评测与 A/B 测试

### 7.1 核心原则

离线质量指标 + 在线业务指标（转化率、满意度、人工评分）**必须同时看**。

### 7.2 A/B 测试样本量估计

$$
n \ge \frac{(z_{\alpha/2} + z_\beta)^2 \cdot 2\sigma^2}{\delta^2}
$$

| 符号 | 含义 | 典型值 |
|------|------|--------|
| $z_{\alpha/2}$ | 显著性水平（$\alpha = 0.05 \to 1.96$） | $1.96$ |
| $z_\beta$ | 检验功效（$1 - \beta = 0.8 \to 0.84$） | $0.84$ |
| $\sigma^2$ | 指标方差 | 从历史数据估计 |
| $\delta$ | 最小可检测效果量 | 业务需求决定 |

### 7.3 分桶评测建议

压缩/驱逐策略上线时，建议按**任务类型分桶**：

| 桶 | 代表任务 | 关注指标 |
|----|---------|---------|
| 代码 | HumanEval, MBPP | Pass@1, Pass@10 |
| 数学 | GSM8K, MATH | Accuracy |
| 对话 | MT-Bench | 人工评分 |
| 长文本 | NIAH, RULER | Retrieval Accuracy |

---

## 8. 安全评测

| 评测维度 | 工具/基准 | 指标 |
|---------|---------|------|
| **毒性** | Perspective API | Toxicity Score $\in [0, 1]$ |
| **偏见** | BBQ, WinoBias | Bias Score / Accuracy Gap |
| **对抗鲁棒性** | 对抗 Prompt 集 | 拒绝率 (Refusal Rate) |
| **隐私泄露** | Canary 检测 | Extraction Rate |

---

## 面试一句话

> "评测必须多维度：PPL 看语言建模能力，F1/Pass@k 看任务表现，ECE 看校准可信度，NIAH 看信息保留能力。线上看 Goodput + A/B 测试统计显著性，不能只看离线分。"
