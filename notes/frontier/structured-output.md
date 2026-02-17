# Structured Output（结构化输出）技术解析

## 一、概述

### 1.1 什么是结构化输出
- 让 LLM 输出符合指定 schema 的格式（JSON, XML, SQL 等）
- 应用：function calling, API integration, data extraction

### 1.2 三种实现方式

#### 方式一：Prompt Engineering
简单但不可靠，模型可能输出无效 JSON

#### 方式二：Constrained Decoding（核心技术）
- 在 decode 时强制 token 满足语法约束
- **原理**：每一步只允许 valid next tokens
- **实现**：
  - 用 **CFG (Context-Free Grammar)** 或 **JSON Schema** 描述约束
  - 维护一个 **状态机**，追踪当前在语法的哪个位置
  - 每步根据状态机过滤 logits

#### 方式三：Fine-tuning
- 在格式化数据上 SFT
- Function calling models：GPT-4, Claude, Qwen 等

### 1.3 开源框架
- **Outlines**：Python, 支持 JSON Schema / Regex / CFG
- **vLLM guided decoding**：集成 outlines backend
- **SGLang**：内置 constrained decoding
- **llama.cpp GBNF**：GGML 格式的语法约束

---

## 二、面试高频问答

**Q: Constrained decoding 会影响生成质量吗？**
- 理论上：masking 改变了概率分布 -> 可能偏离最优
- 实际上：格式化输出本身就是训练目标之一 -> 影响很小
- 关键：约束越严格，对质量的潜在影响越大

**Q: Function calling 的实现原理？**
- 训练时：在 SFT 数据中加入 function call 格式
- 推理时：constrained decoding 确保输出 valid JSON
- 路由：模型先判断是否需要调用函数
