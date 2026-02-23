# 高质量指令数据构建方法论

> SFT 数据 = 模型微调的天花板。数据质量 > 数量 > 一切

---

## 一、指令数据格式

### 1.1 标准格式

```json
{
  "instruction": "请将以下英文翻译成中文",
  "input": "The attention mechanism allows the model to focus on relevant parts.",
  "output": "注意力机制允许模型聚焦于相关部分。",
  "system": "你是一个专业的翻译助手",
  "history": []
}
```

### 1.2 多轮对话格式 (Sharegpt / ChatML)

```json
{
  "conversations": [
    {"role": "system", "content": "你是一个法律咨询助手"},
    {"role": "user", "content": "劳动合同法中关于试用期的规定是什么？"},
    {"role": "assistant", "content": "根据《劳动合同法》第十九条..."},
    {"role": "user", "content": "如果公司违反了试用期规定怎么办？"},
    {"role": "assistant", "content": "如果用人单位违反试用期规定..."}
  ]
}
```

### 1.3 ChatML 模板 (Qwen 风格)

```
<|im_start|>system
你是一个法律咨询助手<|im_end|>
<|im_start|>user
劳动合同法中关于试用期的规定是什么？<|im_end|>
<|im_start|>assistant
根据《劳动合同法》第十九条...<|im_end|>
```

---

## 二、数据质量原则

### 2.1 LIMA 原则："少而精"

> "1000 条高质量数据 > 100万条低质量数据" —— LIMA (Less Is More for Alignment)

| 原则 | 说明 | 反例 |
|------|------|------|
| **多样性** | 覆盖多种任务类型和场景 | 全是翻译任务 |
| **复杂度梯度** | 从简单到困难分层 | 全是简单 QA |
| **信息完整性** | 回答详尽、逻辑完整 | "好的" 一字回答 |
| **事实准确性** | 内容可验证 | 编造的数据/引用 |
| **格式一致性** | 统一的模板和风格 | 格式混乱 |
| **安全合规** | 无有害/偏见内容 | 包含敏感信息 |

### 2.2 数据质量分级

```
S 级 (专家标注):  人类专家精心撰写，事实核查 → 核心数据
A 级 (人工审核):  AI 生成 + 人工审核修改 → 补充数据
B 级 (自动生成):  Self-Instruct / Evol-Instruct → 扩充数据
C 级 (爬虫收集):  网上收集 + 自动清洗 → 仅用于预训练
```

---

## 三、数据构建方法

### 3.1 人工标注 —— 最高质量

**流程：**
```
制定标注规范 → 培训标注员 → 试标注 → 正式标注 → 交叉审核 → 质检
```

**标注规范要点：**
- 回答长度要求（如 200-2000 字）
- 格式要求（分点/代码块/表格）
- 引用要求（标注来源）
- 拒绝模板（敏感问题的标准拒答方式）

### 3.2 Self-Instruct (自我指令生成)

```python
SEED_TASKS = [
    {"instruction": "写一个 Python 函数计算斐波那契数列", "output": "..."},
    {"instruction": "解释量子纠缠的概念", "output": "..."},
    # ... 175 个种子任务
]

GENERATION_PROMPT = """
以下是一些任务示例：
{seed_examples}

请生成一个新的、不同的任务，包含 instruction 和 output。
要求：
1. 与已有任务不重复
2. 多样化（覆盖不同领域和难度）
3. output 需要详尽、准确
"""

def self_instruct(model, seed_tasks, n_generate=50000):
    generated = []
    for _ in range(n_generate):
        # 随机选 3 个种子作为 few-shot
        examples = random.sample(seed_tasks + generated, min(3, len(seed_tasks)))
        prompt = GENERATION_PROMPT.format(seed_examples=format_examples(examples))
        new_task = model.generate(prompt)

        # 质量过滤
        if is_valid(new_task) and not is_duplicate(new_task, generated):
            generated.append(new_task)
    return generated
```

### 3.3 Evol-Instruct (WizardLM 方法) ⭐ 推荐

```
核心：让 AI 将简单指令"进化"为更复杂的指令

进化方向：
├── 深度进化：增加约束条件、推理步骤
├── 广度进化：拓展到新领域/场景
├── 增难进化：增加干扰项、边界条件
└── 具象进化：从抽象到具体实例
```

```python
EVOLVE_PROMPT = """
请将以下指令改写为更复杂、更具挑战性的版本：

原始指令：{instruction}

改写要求（随机选择一种）：
1. 增加约束条件（如字数限制、格式要求、多步骤推理）
2. 引入更多上下文（如角色扮演、行业术语）
3. 增加边界条件和异常处理
4. 将单一问题扩展为多角度分析

改写后的指令：
"""

def evolve_instruction(model, instruction, n_rounds=3):
    """多轮进化"""
    current = instruction
    for _ in range(n_rounds):
        prompt = EVOLVE_PROMPT.format(instruction=current)
        current = model.generate(prompt)
    return current
```

### 3.4 领域数据构建 (Domain-Specific)

```
以法律领域为例：

1. 种子收集：
   - 真实案例判决书 → 提取 QA 对
   - 法律条文 → 生成解释性 QA
   - 律师论坛 → 收集高频问题

2. 模板扩充：
   - "根据《{法律名称}》第{条}条，{情况描述}应该如何处理？"
   - "A公司与B签订了{合同类型}，如果发生{违约情况}，法律后果是什么？"

3. AI 扩充 + 人工审核：
   - 用 GPT-4 / Claude 生成回答
   - 法律专家审核事实准确性
   - 标注引用条文

4. 对抗生成：
   - 故意构造容易混淆的法律概念
   - 构造边界案例（如管辖权争议）
```

---

## 四、数据质量过滤

### 4.1 自动过滤规则

```python
def quality_filter(sample: dict) -> bool:
    """多维度质量过滤"""
    instruction = sample['instruction']
    output = sample['output']

    # 1. 长度过滤
    if len(instruction) < 10 or len(output) < 50:
        return False
    if len(output) > 10000:  # 过长可能是垃圾
        return False

    # 2. 语言一致性
    if detect_lang(instruction) != detect_lang(output):
        return False

    # 3. 重复检测
    if has_excessive_repetition(output, threshold=0.3):
        return False

    # 4. 安全过滤
    if contains_harmful_content(output):
        return False

    # 5. 格式检查
    if output.strip() == "" or output.startswith("作为一个 AI"):
        return False

    return True
```

### 4.2 AI 评分过滤

```python
SCORING_PROMPT = """
请对以下 AI 回答进行评分 (1-5):

指令: {instruction}
回答: {output}

评分维度:
1. 准确性: 信息是否正确
2. 完整性: 是否全面回答了问题
3. 有用性: 对提问者是否有帮助
4. 清晰度: 表达是否清晰易懂

请输出 JSON: {{"score": 分数, "reason": "理由"}}
"""

# 使用 GPT-4 评分，过滤掉 < 4 分的样本
```

### 4.3 去重策略

| 方法 | 粒度 | 速度 | 适用场景 |
|------|------|------|---------|
| 精确去重 | 完全相同 | 快 | 基础去重 |
| N-gram 去重 | 片段重叠 | 中 | 近似重复 |
| MinHash + LSH | 语义近似 | 快 | 大规模去重 |
| Embedding 聚类 | 语义相似 | 慢 | 精细去重 |

---

## 五、数据配比与采样

### 5.1 任务类型配比

| 任务类型 | 建议占比 | 示例 |
|---------|---------|------|
| 指令跟随 | 30% | "请把这段话翻译成..." |
| 知识问答 | 20% | "量子计算的原理是什么？" |
| 代码生成 | 15% | "用 Python 实现..." |
| 推理/数学 | 15% | "如果 A>B, B>C，那么..." |
| 创意写作 | 10% | "写一首关于...的诗" |
| 多轮对话 | 10% | 上下文追问/澄清 |

### 5.2 难度分层

```python
# 按难度分层采样
difficulty_distribution = {
    "easy": 0.2,      # 简单 QA、格式转换
    "medium": 0.5,    # 多步推理、综合分析
    "hard": 0.2,      # 复杂数学、长文推理
    "expert": 0.1,    # 专家级、开放性问题
}
```

---

## 六、开源数据集参考

| 数据集 | 规模 | 语言 | 特点 |
|--------|------|------|------|
| **Alpaca** | 52K | EN | Self-Instruct 生成 |
| **ShareGPT** | 90K | 多语言 | 真实用户对话 |
| **WizardLM** | 250K | EN | Evol-Instruct 进化 |
| **BELLE** | 1.5M | ZH | 中文指令数据 |
| **Firefly** | 1.1M | ZH | 多任务中文数据 |
| **OpenAssistant** | 160K | 多语言 | 人工标注对话树 |
| **UltraChat** | 1.5M | EN | 多轮对话 |
| **Magpie** | 1M+ | EN | 从模型中提取高质量数据 |
| **OpenHermes** | 1M | EN | 多数据源融合 |

---

## 七、数据格式规范

### 7.1 Alpaca 格式
```json
{"instruction": "...", "input": "...", "output": "..."}
```

### 7.2 ShareGPT 格式
```json
{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
```

### 7.3 LlamaFactory 统一格式
```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

---

## 面试高频问答

**Q1：如何评估指令数据的质量？**
> 从 5 个维度评估：1）多样性（指令类型覆盖度）；2）复杂度分布（不能全是简单任务）；3）事实准确性（人工抽检 + AI 交叉验证）；4）回答完整性（是否详尽）；5）格式一致性。可用 GPT-4 对数据进行 1-5 分评分，过滤 <4 分的样本。

**Q2：数据量多少合适？**
> 取决于任务和基模型能力。通用 SFT：5K-100K 高质量数据足够。领域微调：1K-10K 领域数据 + 通用数据混合。关键是质量而非数量 —— LIMA 论文证明 1000 条精选数据就能让 65B 模型表现优异。

**Q3：如何构建领域数据？**
> 三步策略：1）种子数据收集（真实业务数据 + 领域文档 QA 提取）；2）AI 扩充（GPT-4 生成 + Evol-Instruct 进化）；3）专家审核（领域专家校验事实准确性）。关键是保证领域知识的准确性，宁可少但对。

## 面试一句话
- "指令数据构建的核心是 '多样性 × 复杂度梯度 × 事实准确'，Self-Instruct 和 Evol-Instruct 是主流的数据扩充方法，1K 高质量数据的效果往往超过 100K 低质量数据。"
