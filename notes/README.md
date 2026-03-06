# 深度技术笔记

> 这里是知识库的正文主体，面向“理解为什么这样设计”。如果数学字典解决的是“公式是什么”，这里解决的是“工程上为什么这么做、代价在哪里、替代方案是什么”。

## 建议阅读顺序

| 目标 | 建议起点 | 接下来读什么 |
|------|----------|--------------|
| 先把推理主线跑通 | [attention/README.md](attention/README.md) | KV Cache -> Inference -> Serving |
| 先补系统设计 | [llm-system/README.md](llm-system/README.md) | Distributed -> Frameworks -> Serving |
| 先补训练与对齐 | [training/README.md](training/README.md) | RL Infra -> Frontier |
| 先看公式到实现 | [attention/formula-to-code-walkthrough.md](attention/formula-to-code-walkthrough.md) | 再对照 `src/attention/*.py` |

## 核心专题

### 基础与核心机制

- [basics/README.md](basics/README.md)：Python、神经网络、PyTorch 的最短学习路径。
- [attention/README.md](attention/README.md)：从 MHA 到 GQA、MLA、FlashAttention、长上下文。
- [architectures/README.md](architectures/README.md)：LLaMA、Mixtral、DeepSeek、MoE 与位置编码。

### 推理系统

- [kv-cache/README.md](kv-cache/README.md)：KV Cache 的核心概念与工程权衡。
- [kv-compression/README.md](kv-compression/README.md)：量化、稀疏化、压缩比与误差。
- [kv-eviction/README.md](kv-eviction/README.md)：驱逐策略、多租户公平性。
- [inference/README.md](inference/README.md)：采样、解码、Prefill / Decode 分离。
- [distributed/README.md](distributed/README.md)：TP、PP、EP、MoE 推理优化。
- [frameworks/README.md](frameworks/README.md)：vLLM、TensorRT-LLM、SGLang。
- [serving/README.md](serving/README.md)：容量规划、成本与服务稳定性。
- [llm-system/README.md](llm-system/README.md)：吞吐、延迟、架构模式。

### 训练与前沿

- [training/README.md](training/README.md)：预训练、SFT、PEFT、RLHF、后训练。
- [rl-infra/README.md](rl-infra/README.md)：Rollout、权重同步、Slime、Multi-LoRA、GPU 瓶颈。
- [frontier/README.md](frontier/README.md)：模型演进、推理模型、结构化输出与新框架。
- [bitter-lessons/README.md](bitter-lessons/README.md)：踩坑记录、复现总结、经验复盘。

### 其他专题

- [cuda/basics.md](cuda/basics.md)
- [multimodal/vision-language-models.md](multimodal/vision-language-models.md)
- [multimodal/multimodal-training.md](multimodal/multimodal-training.md)
- [system-design/methodology.md](system-design/methodology.md)
- [source-reading/slime-sglang-verl-deep-dive.md](source-reading/slime-sglang-verl-deep-dive.md)
- [tools/profiling.md](tools/profiling.md)

## 公式深挖与源码对照

- [attention/formula-to-code-walkthrough.md](attention/formula-to-code-walkthrough.md)：把 Attention / GQA / RoPE / RMSNorm / FlashAttention 逐段映射到仓库代码。
- [kv-cache/formula-to-code-walkthrough.md](kv-cache/formula-to-code-walkthrough.md)：把 KV 容量账本、PagedAttention、量化、驱逐逐段映射到源码。
- [serving/formula-to-code-walkthrough.md](serving/formula-to-code-walkthrough.md)：把 TTFT / TPOT / Goodput / 调度策略映射到指标与调度代码。
- [distributed/moe-formula-to-code-walkthrough.md](distributed/moe-formula-to-code-walkthrough.md)：把 MoE router、capacity、drop rate、All-to-All 映射到模拟器。
- [attention/mha-vs-gqa-full-derivation.md](attention/mha-vs-gqa-full-derivation.md)：GQA 为什么能显著降低 Decode 带宽。
- [attention/mha-vs-mla-full-derivation.md](attention/mha-vs-mla-full-derivation.md)：MLA 的矩阵吸收和潜在空间压缩。
- [attention/mha-vs-dsa-full-derivation.md](attention/mha-vs-dsa-full-derivation.md)：DSA 的稀疏选择与双阶段结构。
- [attention/mha-vs-linear-attention-full-derivation.md](attention/mha-vs-linear-attention-full-derivation.md)：线性注意力如何把状态压缩成常数大小。

## GitBook 阅读建议

- 如果你使用 GitBook 侧边栏阅读，建议先从 [../SUMMARY.md](../SUMMARY.md) 进入，而不是在文件系统里跳来跳去。
- 每个专题尽量遵循“定义 -> 推导 -> 复杂度 -> 工程结论 -> 对应代码”的顺序。
- 如果你发现某一页公式变成纯文本，优先用 [../scripts/validate_markdown_math.py](../scripts/validate_markdown_math.py) 巡检该页写法。
