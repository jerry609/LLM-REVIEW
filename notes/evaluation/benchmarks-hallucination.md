# 模型评估 Benchmark + 幻觉检测

> 面试必知的评估体系和幻觉问题

---

## 一、主流 Benchmark 全景

### 1.1 综合能力评估

| Benchmark | 评测内容 | 题目数 | 格式 | 重要度 |
|-----------|---------|--------|------|--------|
| **MMLU** | 57 学科知识（数学/历史/法律等） | 14K | 4 选 1 | ⭐⭐⭐⭐⭐ |
| **MMLU-Pro** | MMLU 升级版，10 选 1，更难 | 12K | 10 选 1 | ⭐⭐⭐⭐⭐ |
| **C-Eval** | 中文综合评测，52 学科 | 13K | 4 选 1 | ⭐⭐⭐⭐ |
| **CMMLU** | 中文多任务评测 | 11K | 4 选 1 | ⭐⭐⭐⭐ |
| **ARC** | 科学推理（小学到高中） | 7.7K | 多选 | ⭐⭐⭐ |
| **HellaSwag** | 常识推理（选择合理续写） | 10K | 4 选 1 | ⭐⭐⭐ |
| **WinoGrande** | 共指消解 | 1.7K | 2 选 1 | ⭐⭐⭐ |
| **TruthfulQA** | 真实性（抗幻觉能力） | 817 | 多选/生成 | ⭐⭐⭐⭐ |

### 1.2 数学推理

| Benchmark | 评测内容 | 难度 | 格式 |
|-----------|---------|------|------|
| **GSM8K** | 小学数学应用题 | 中 | 数值答案 |
| **MATH** | 竞赛级数学（7 类别） | 难 | 数值/表达式 |
| **AIME** | AMC/AIME 竞赛题 | 极难 | 整数答案 |
| **Minerva** | 数学/科学综合 | 难 | 自由形式 |
| **MathBench** | 分层数学评测 | 全面 | 多格式 |

### 1.3 代码能力

| Benchmark | 评测内容 | 语言 | 评测方式 |
|-----------|---------|------|---------|
| **HumanEval** | 函数级代码生成 | Python | pass@k (执行) |
| **HumanEval+** | HumanEval 增强测试用例 | Python | pass@k |
| **MBPP** | 基础编程题 | Python | pass@k |
| **LiveCodeBench** | 实时更新的竞赛题（防泄露） | 多语言 | pass@k |
| **SWE-bench** | 真实 GitHub issue 修复 | Python | 自动验证 |
| **Codeforces Rating** | 竞赛编程 ELO | 多语言 | 比赛排名 |

### 1.4 长上下文

| Benchmark | 评测内容 | 长度范围 |
|-----------|---------|---------|
| **RULER** | 多种长上下文任务（搜索/计算/QA） | 4K-128K |
| **Needle-in-a-Haystack** | 在长文本中找隐藏的信息 | 任意 |
| **InfiniteBench** | 100K+ 长文本理解 | 100K+ |
| **LongBench** | 中文长上下文评测 | 8K-128K |

### 1.5 Agent / 工具使用

| Benchmark | 评测内容 |
|-----------|---------|
| **GAIA** | 真实世界复杂任务（需要搜索/计算/推理） |
| **WebArena** | 网页操作任务 |
| **ToolBench** | API 调用能力 |
| **BFCL** | 函数调用评测 |

### 1.6 安全/对齐

| Benchmark | 评测内容 |
|-----------|---------|
| **MT-Bench** | 多轮对话质量（GPT-4 评分） |
| **AlpacaEval** | 指令跟随能力（GPT-4 评分） |
| **Arena-Hard** | 高难度对话评测 |
| **IFEval** | 指令遵循精确度 |
| **SafetyBench** | 安全性评测 |

---

## 二、评估方法论

### 2.1 评估指标

| 指标 | 含义 | 适用场景 |
|------|------|---------|
| **Perplexity (PPL)** | 模型对测试集的困惑度，越低越好 | 预训练评估 |
| **Accuracy** | 选择题正确率 | MMLU, ARC 等 |
| **pass@k** | k 次采样中至少一次通过的概率 | 代码生成 |
| **Win Rate** | 对比评测中的胜率 | MT-Bench, AlpacaEval |
| **F1 / ROUGE** | 文本生成质量 | 摘要、QA |
| **ELO Rating** | 竞赛式排名 | Chatbot Arena |

### 2.2 评估方式对比

| 方式 | 描述 | 优缺点 |
|------|------|--------|
| **固定答案** | 选择题/数值题对答案 | 客观，但覆盖面窄 |
| **代码执行** | 运行代码检查正确性 | 客观，但只适用代码 |
| **模型评分** | GPT-4/Claude 打分 | 灵活，但有偏差 |
| **人工评分** | 人类标注者评分 | 最可靠，但昂贵 |
| **Arena 对比** | 用户投票 A vs B | 最真实，但需要大量流量 |

### 2.3 Chatbot Arena（最权威）
```
用户提问 → 两个匿名模型回复 → 用户投票哪个更好 → 更新 ELO Rating
```
- **优势**：最接近真实用户偏好
- **LMSYS**：lmsys.org/chatbot-arena
- **当前 Top**：GPT-4o, Claude-3.5-Sonnet, Gemini 2.0

### 2.4 数据污染问题
```
问题：模型训练数据可能包含 benchmark 的测试集
  → benchmark 分数虚高，不反映真实能力

检测方法：
  1. N-gram 重叠检测（13-gram）
  2. Canary strings（在训练数据中放特殊标记）
  3. 时间分割（只用模型发布后的题目）
  
应对措施：
  - LiveCodeBench：持续更新新题目
  - Arena：实时用户对话，无法预知
  - 私有测试集：不公开测试题
```

---

## 三、幻觉检测 (Hallucination Detection)

### 3.1 幻觉的分类

```
┌─────────────────────────────────────────┐
│            LLM 幻觉分类                   │
├─────────────────────────────────────────┤
│ 1. 事实性幻觉 (Factual Hallucination)    │
│    - 生成与现实不符的"事实"               │
│    - 例："爱因斯坦发明了电话"             │
│                                          │
│ 2. 忠实性幻觉 (Faithfulness Hallucination)│
│    - 生成与给定上下文矛盾的内容           │
│    - 例：文档说A，模型回答B               │
│                                          │
│ 3. 指令幻觉 (Instruction Hallucination)  │
│    - 不遵循指令格式/约束                  │
│    - 例：要求 JSON 输出但生成自然语言     │
└─────────────────────────────────────────┘
```

### 3.2 幻觉检测方法

#### 方法 1：SelfCheckGPT
```
1. 让模型对同一问题生成 N 个不同回复（采样多次）
2. 计算回复之间的一致性
3. 一致性低 → 高幻觉概率（模型不确定的部分更容易胡编）
```
- **原理**：如果模型对事实有把握，不同采样结果应该一致
- **指标**：BERTScore / NLI 一致性

#### 方法 2：知识检索验证
```
1. 模型生成回复
2. 用检索系统找到相关文档
3. 用 NLI 模型判断回复是否被文档支持
   - Entailment → 非幻觉
   - Contradiction → 幻觉
   - Neutral → 不确定
```

#### 方法 3：不确定性估计
```
1. 计算 token 级别的 log-probability
2. 低概率区域 → 可能是幻觉
3. 序列级不确定性：entropy / predictive variance
```

#### 方法 4：LLM-as-Judge
```
用 GPT-4 / Claude 评判回复的事实性
prompt = """
请判断以下回复是否包含事实错误：
问题：{question}
回复：{response}
参考知识：{knowledge}
"""
```

### 3.3 幻觉缓解方法

| 方法 | 描述 | 效果 |
|------|------|------|
| **RAG** | 检索增强生成，基于真实文档回答 | ⭐⭐⭐⭐⭐ |
| **Constrained Decoding** | 约束输出格式/内容 | ⭐⭐⭐ |
| **Chain-of-Thought** | 让模型展示推理过程 | ⭐⭐⭐⭐ |
| **Self-Consistency** | 多次采样 + 投票 | ⭐⭐⭐⭐ |
| **拒绝回答** | 不确定时说"我不知道" | ⭐⭐⭐ |
| **RLHF 对齐** | 训练时惩罚幻觉 | ⭐⭐⭐⭐ |
| **Retrieval-Augmented Training** | 训练时也用检索 | ⭐⭐⭐⭐ |

### 3.4 幻觉检测 Benchmark

| Benchmark | 评测内容 |
|-----------|---------|
| **TruthfulQA** | 817 题，测试模型是否会重复常见误解 |
| **FActScore** | 人物传记的事实准确性 |
| **HaluEval** | 专门的幻觉评测数据集 |
| **FEVER** | 事实验证（Fact Extraction and VERification） |

---

## 四、主流模型评估成绩对比（2025）

### 4.1 综合排名（Chatbot Arena ELO）
```
Tier 1 (ELO 1300+):
  GPT-4o, Claude-3.5-Sonnet, Gemini 2.0 Pro

Tier 2 (ELO 1200-1300):
  DeepSeek-V3, Qwen2.5-72B, Claude-3-Opus

Tier 3 (ELO 1100-1200):
  Llama-3.1-70B, Mistral-Large, GLM-4-Plus
```

### 4.2 数学推理（MATH benchmark）
```
DeepSeek-R1:     ~97% (AIME ~80%)
GPT-o1:          ~96%
Qwen2.5-Math-72B: ~90%
Llama-3.1-70B:   ~68%
```

### 4.3 代码能力（HumanEval）
```
GPT-4o:          ~92%
DeepSeek-V3:     ~90%
Claude-3.5-Sonnet: ~92%
Qwen2.5-Coder-32B: ~90%
```

---

## 五、评估最佳实践

### 5.1 评估一个模型应该看什么？
```
1. Chatbot Arena ELO：最真实的综合能力
2. MMLU-Pro：知识面
3. GSM8K/MATH：数学推理
4. HumanEval+：代码
5. RULER/NIAH：长上下文
6. MT-Bench：对话质量
7. IFEval：指令遵循
```

### 5.2 评估陷阱
| 陷阱 | 说明 |
|------|------|
| **数据泄露** | 模型训练数据包含测试集 |
| **Prompt 敏感** | 换一种问法就答不对 |
| **过拟合 benchmark** | 针对性训练特定 benchmark |
| **Cherry-picking** | 只展示好的结果 |
| **Saturation** | 某些 benchmark 太简单（如 MMLU 已接近 90%+） |

---

## 面试高频问答

**Q1：如何评估一个 LLM 的综合能力？**
> 多维度评估：知识面（MMLU-Pro）、数学推理（MATH/GSM8K）、代码（HumanEval+）、长上下文（RULER）、对话（MT-Bench）。最权威的综合指标是 Chatbot Arena 的 ELO 评分，因为它反映真实用户偏好。

**Q2：什么是数据污染？如何检测？**
> 模型训练数据包含 benchmark 测试集导致分数虚高。检测方法：n-gram 重叠检测、canary strings、时间分割（用模型发布后的新题目）。LiveCodeBench 和 Arena 通过持续更新避免污染。

**Q3：LLM 幻觉的主要类型和检测方法？**
> 三种类型：事实性幻觉（编造事实）、忠实性幻觉（与上下文矛盾）、指令幻觉（不遵循格式）。检测方法：SelfCheckGPT（多次采样一致性）、知识检索验证（NLI）、不确定性估计（低 log-prob 区域）。

**Q4：pass@k 是什么？和 accuracy 有什么区别？**
> pass@k 是代码评估指标：生成 k 个候选中至少一个通过的概率。与 accuracy 的区别是 pass@k 允许多次尝试，更贴合实际开发（试几次就行），通常报告 pass@1 和 pass@10。

**Q5：如何缓解 LLM 幻觉？**
> 最有效的是 RAG（让模型基于真实文档回答而非凭记忆）。其他方法：Chain-of-Thought（暴露推理过程便于验证）、Self-Consistency（多次采样投票）、训练时用 RLHF 惩罚幻觉、不确定时让模型拒绝回答。
