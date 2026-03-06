# LLM-REVIEW

> 面向 LLM 推理、训练与服务系统的系统化知识库，按“公式 -> 原理 -> 代码 -> 面试表达”四层组织，适合直接放进 GitBook 连续阅读，也适合作为面试前的速查手册。

## 这次整理了什么

- 新增 GitBook 目录页：[SUMMARY.md](SUMMARY.md)
- 统一重写首页和各模块 README，导航结构更清晰
- 新增“公式到源码”对照页：[notes/attention/formula-to-code-walkthrough.md](notes/attention/formula-to-code-walkthrough.md)
- 修复多处会导致 Markdown / GitBook 公式渲染异常的表格写法
- 增加文档校验脚本：[scripts/validate_markdown_math.py](scripts/validate_markdown_math.py)

## 从哪里开始

| 你的目标 | 建议入口 | 你会得到什么 |
|----------|----------|--------------|
| 30 分钟内速查 | [math_dictionary/README.md](math_dictionary/README.md) | 公式、变量定义、张量形状、性能估算 |
| 系统学习推理链路 | [notes/README.md](notes/README.md) | 从注意力、KV Cache 到 serving 的长文笔记 |
| 对照源码理解实现 | [src/README.md](src/README.md) | NumPy / PyTorch 最小实现与运行入口 |
| 准备面试表达 | [mock_interview/README.md](mock_interview/README.md) | 按主题、公司、行为面试组织的问题库 |
| 按计划推进 | [roadmap/README.md](roadmap/README.md) | 12 周冲刺和 45 天日程安排 |

## GitBook 阅读入口

- 直接从 [SUMMARY.md](SUMMARY.md) 打开整站目录，侧边栏层级已经按 GitBook 阅读习惯重新整理。
- 如果你只想看“公式 + 张量形状 + 对应代码”，从 [notes/attention/formula-to-code-walkthrough.md](notes/attention/formula-to-code-walkthrough.md) 开始。
- 如果你正在排查公式渲染问题，先运行 `python scripts/validate_markdown_math.py`。

## 核心入口

### 数学与原理

- [math_dictionary/README.md](math_dictionary/README.md)：公式索引、推导入口、量级估算。
- [math_dictionary/transformer-attention-math.md](math_dictionary/transformer-attention-math.md)：Attention、RoPE、LayerNorm、RMSNorm、SwiGLU。
- [math_dictionary/flashattention-math.md](math_dictionary/flashattention-math.md)：在线 Softmax、分块 IO、FA-2 / FA-3。
- [math_dictionary/kv-memory.md](math_dictionary/kv-memory.md)：KV Cache 显存估算与容量规划。
- [math_dictionary/prefill-decode-performance.md](math_dictionary/prefill-decode-performance.md)：Prefill / Decode 的带宽与时延分析。

### 长文笔记

- [notes/README.md](notes/README.md)：所有主题笔记的总入口。
- [notes/attention/README.md](notes/attention/README.md)：注意力演进、长上下文、MHA / GQA / MLA / DSA / Linear。
- [notes/frameworks/vllm-architecture.md](notes/frameworks/vllm-architecture.md)：vLLM 架构拆解。
- [notes/training/pretraining-pipeline.md](notes/training/pretraining-pipeline.md)：预训练全链路。
- [notes/rl-infra/rl-training-inference-systems.md](notes/rl-infra/rl-training-inference-systems.md)：RL 训练与推理系统协同。

### 代码与实验

- [src/README.md](src/README.md)：最小实现和运行方法。
- [src/from_scratch/README.md](src/from_scratch/README.md)：from scratch 路线图。
- [notebooks/README.md](notebooks/README.md)：HTML 讲解稿和 notebook 源文件。
- [tests/README.md](tests/README.md)：测试入口与覆盖范围。

### 面试与路线

- [mock_interview/README.md](mock_interview/README.md)：按主题、公司、行为问题组织的题库。
- [roadmap/README.md](roadmap/README.md)：12 周冲刺与 45 天执行节奏。
- [benchmarks/README.md](benchmarks/README.md)：热点、复盘和自测记录。

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

### 路线三：公式到源码

1. [math_dictionary/transformer-attention-math.md](math_dictionary/transformer-attention-math.md)
2. [notes/attention/formula-to-code-walkthrough.md](notes/attention/formula-to-code-walkthrough.md)
3. [src/attention/mha_gqa.py](src/attention/mha_gqa.py)
4. [src/attention/rope_rmsnorm.py](src/attention/rope_rmsnorm.py)
5. [src/attention/flash_attn_sim.py](src/attention/flash_attn_sim.py)

## 文档维护建议

- 新增公式页时，优先用独立段落的 `$$ ... $$`，不要把带竖线的公式直接塞进表格。
- 需要在表格里写范数或 KL 时，优先使用 `\lVert \cdot \rVert`、`\parallel`、`\lvert x \rvert` 这类写法。
- 每个专题最好同时给出：公式、张量形状、工程结论、对应源码。
- 维护侧边栏时，优先同步更新 [SUMMARY.md](SUMMARY.md)。

## 快速开始

```bash
git clone https://github.com/jerry609/LLM-REVIEW.git
cd LLM-REVIEW

pip install torch numpy pytest
python scripts/validate_markdown_math.py
python -m pytest tests -v
```

如果你当前只想阅读，不想跑代码，直接从 [SUMMARY.md](SUMMARY.md) 或 [math_dictionary/README.md](math_dictionary/README.md) 开始即可。
