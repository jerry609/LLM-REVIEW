# LLM-REVIEW

LLM 推理系统知识库，按“数学公式 → 技术笔记 → 代码实现 → 面试表达”四层组织，适合 GitBook 连续阅读，也适合作为面试前速查手册。

## 从哪里开始

如果你的目标是面试速通，先读 [math_dictionary/README.md](math_dictionary/README.md)、[notes/README.md](notes/README.md) 和 [mock_interview/README.md](mock_interview/README.md)。

如果你的目标是系统学习，直接从 [roadmap/README.md](roadmap/README.md) 开始，再按路线进入数学、推理系统、训练与服务章节。

如果你的目标是动手复现，优先看 [notebooks/README.md](notebooks/README.md)、[src/README.md](src/README.md) 和 [src/from_scratch/README.md](src/from_scratch/README.md)。

## 核心入口

### 数学与原理

- [math_dictionary/README.md](math_dictionary/README.md)：公式索引、推荐学习顺序、面试回答模板。
- [notes/README.md](notes/README.md)：按主题组织的系统性技术笔记。
- [notes/attention/README.md](notes/attention/README.md)：注意力机制、FlashAttention、长上下文。
- [notes/llm-system/README.md](notes/llm-system/README.md)：吞吐、延迟、服务系统设计。

### 实验与代码

- [notebooks/README.md](notebooks/README.md)：Jupyter 实验与 HTML 讲解文档。
- [src/README.md](src/README.md)：当前可运行实现与模块入口。
- [src/from_scratch/README.md](src/from_scratch/README.md)：from scratch 复现规划与目录说明。
- [tests/README.md](tests/README.md)：测试入口与覆盖范围。

### 面试与规划

- [mock_interview/README.md](mock_interview/README.md)：按主题、公司、行为面试组织的问题库。
- [roadmap/README.md](roadmap/README.md)：总览、12 周计划、45 天日程。
- [benchmarks/README.md](benchmarks/README.md)：热点分析、自测记录与报告。
- [notes/bitter-lessons/README.md](notes/bitter-lessons/README.md)：复现与调试中的真实踩坑记录。

## 推荐阅读路线

### 路线一：面试前速查

1. [math_dictionary/symbols-glossary.md](math_dictionary/symbols-glossary.md)
2. [math_dictionary/tensor-shapes.md](math_dictionary/tensor-shapes.md)
3. [math_dictionary/kv-memory.md](math_dictionary/kv-memory.md)
4. [math_dictionary/serving-metrics.md](math_dictionary/serving-metrics.md)
5. [mock_interview/by-topic/README.md](mock_interview/by-topic/README.md)

### 路线二：推理系统主线

1. [notes/basics/README.md](notes/basics/README.md)
2. [notes/attention/README.md](notes/attention/README.md)
3. [notes/kv-cache/README.md](notes/kv-cache/README.md)
4. [notes/inference/README.md](notes/inference/README.md)
5. [notes/distributed/README.md](notes/distributed/README.md)
6. [notes/serving/README.md](notes/serving/README.md)

### 路线三：训练与对齐扩展

1. [notes/training/README.md](notes/training/README.md)
2. [notes/frameworks/README.md](notes/frameworks/README.md)
3. [notes/rl-infra/README.md](notes/rl-infra/README.md)
4. [notes/frontier/README.md](notes/frontier/README.md)

### 路线四：from scratch 复现

1. [src/from_scratch/README.md](src/from_scratch/README.md)
2. [notebooks/README.md](notebooks/README.md)
3. [benchmarks/README.md](benchmarks/README.md)
4. [tests/README.md](tests/README.md)

## 高价值页面

- [math_dictionary/attention-complexity.md](math_dictionary/attention-complexity.md)：复杂度、FLOPs、Roofline、FlashAttention IO。
- [math_dictionary/prefill-decode-performance.md](math_dictionary/prefill-decode-performance.md)：Prefill / Decode 性能分析。
- [notes/attention/attention-evolution-and-inference.md](notes/attention/attention-evolution-and-inference.md)：注意力机制演进与推理流程总览。
- [notes/frameworks/vllm-architecture.md](notes/frameworks/vllm-architecture.md)：vLLM 架构拆解。
- [notes/training/pretraining-pipeline.md](notes/training/pretraining-pipeline.md)：预训练完整流程。
- [notes/rl-infra/rl-training-inference-systems.md](notes/rl-infra/rl-training-inference-systems.md)：RL 系统链路优化。
- [mock_interview/by-company/README.md](mock_interview/by-company/README.md)：按公司风格整理的题目入口。
- [roadmap/00-overview.md](roadmap/00-overview.md)：全局学习地图。

## 仓库组织原则

- 数学层回答“公式是什么、量纲是什么、数量级是多少”。
- 笔记层回答“为什么这样设计、工程折中在哪里”。
- 代码层回答“如何运行、如何验证、如何复现”。
- 面试层回答“如何在有限时间内说清楚问题与取舍”。

## 快速开始

```bash
git clone https://github.com/jerry609/LLM-REVIEW.git
cd LLM-REVIEW

pip install torch numpy pytest
python -m pytest tests -v
```

如果你更偏向阅读而不是运行代码，直接从 [math_dictionary/README.md](math_dictionary/README.md) 或 [notes/README.md](notes/README.md) 开始即可。

## 后续建设

- 补全 from scratch 模块中的 tokenizer、model、training、alignment 复现内容。
- 持续把公式密集页面改成更稳定的 GitBook 数学排版。
- 为关键实验补上更清晰的入口说明与结果页。

## 参考

- [wyf3/llm_related](https://github.com/wyf3/llm_related)：from scratch 复现主题的重要参考。
- [CMU 17-445/645: ML in Production](https://mlip-cmu.github.io/f2025/)：系统化学习机器学习生产化与部署实践。
