# LLM 常见评测指标速查

## 1) 困惑度（Perplexity）
- `PPL = exp( - (1/N) * sum_t log p(x_t|x_{<t}) )`
- 越低通常越好，但不等价于任务准确率。
- 常用于语言建模基准、压缩/量化后质量评估。
- 注意：不同 tokenizer 的 PPL 不可直接比较（token 粒度不同）。

## 2) 分类/抽取任务
- Precision / Recall / F1：
  - `Precision = TP/(TP+FP)`（预测为正中实际为正的比例）
  - `Recall = TP/(TP+FN)`（实际为正中被预测为正的比例）
  - `F1 = 2PR/(P+R)`（Precision 和 Recall 的调和平均）
- Accuracy：`(TP+TN)/(TP+TN+FP+FN)`
- 类别不平衡时 F1 比 Accuracy 更有参考价值

## 3) 代码生成 Pass@k
- 常见估计（无放回近似）：
  `pass@k = 1 - C(n-c, k)/C(n, k)`
- `n` 为样本数，`c` 为正确样本数。
- 实际使用无偏估计，避免直接用经验比率

## 4) 生成质量指标

### BLEU（机器翻译常用）
- N-gram precision 的几何平均 + 短句惩罚
- `BLEU = BP * exp(sum_{n=1}^{N} w_n * log p_n)`
- `BP = min(1, exp(1 - ref_len/hyp_len))`（brevity penalty）
- 局限：词袋匹配，不考虑语义

### ROUGE（摘要常用）
- ROUGE-N：N-gram recall（参考摘要中 n-gram 被生成覆盖的比例）
- ROUGE-L：最长公共子序列（LCS）
  `ROUGE-L = F1(LCS_recall, LCS_precision)`
- 常用 ROUGE-1, ROUGE-2, ROUGE-L

### BERTScore
- 用 BERT embedding 计算 token 级别的余弦相似度
- 比 BLEU/ROUGE 更能捕捉语义相似性

## 5) 校准（Calibration）
- 模型输出的置信度是否与实际正确率匹配
- Expected Calibration Error（ECE）：
  `ECE = sum_b (|B_b|/N) * |acc(B_b) - conf(B_b)|`
- 将预测按置信度分桶，计算每桶内准确率与置信度的差距
- LLM 常见问题：过度自信（confidence > accuracy）

## 6) 长上下文评测
- Needle-in-a-Haystack：在长文本中插入特定信息，测试能否检索
  - 变量：文本长度 × 插入位置（生成热力图）
- RULER benchmark：多种长上下文能力的综合测试
- 评估压缩/驱逐策略时特别重要：压缩后是否丢失了关键信息

## 7) 线上评测
- 离线质量 + 在线业务指标（转化、满意度、人工评分）必须同时看。
- 压缩/驱逐策略上线时，建议做任务分桶评测（代码/数学/对话分开）。
- A/B 测试：统计显著性需要足够样本量
  - 样本量估计：`n ≈ (z_{alpha/2} + z_beta)^2 * 2*sigma^2 / delta^2`
  - `delta` 为期望检测到的最小效果量

## 8) 安全评测
- 毒性检测：Perspective API score
- 偏见评测：BBQ benchmark、WinoBias
- 对抗鲁棒性：对抗 prompt 下的拒绝率

## 面试一句话
- "评测要多维度：PPL 看语言建模能力，任务指标看实用性，校准看可信度，长上下文看信息保留。"
