# 深度技术笔记

> 这部分是知识库的正文主体，适合系统学习，而不是临时速查。

## 基础与核心机制

- [基础速通](basics/README.md)：Python、神经网络、PyTorch 的最短学习路径。
- [注意力机制](attention/README.md)：从 MHA 到 GQA、MLA、FlashAttention、长上下文。
- [模型架构](architectures/README.md)：LLaMA、DeepSeek、Mixtral、MoE、位置编码与混合架构。

## 推理系统

- [KV Cache](kv-cache/README.md)：核心概念、PagedAttention 与面试问答。
- [KV 压缩](kv-compression/README.md)：量化、稀疏化与压缩权衡。
- [KV 驱逐](kv-eviction/README.md)：驱逐策略、面试问答与公平性。
- [推理与解码](inference/README.md)：采样、投机解码、Prefill-Decode 分离。
- [分布式系统](distributed/README.md)：TP、PP、EP 与 MoE 推理优化。
- [推理框架](frameworks/README.md)：vLLM、TensorRT-LLM、SGLang。
- [服务与运维](serving/README.md)：容量规划与成本优化。
- [LLM 系统设计](llm-system/README.md)：吞吐、延迟与服务系统设计。

## 训练与对齐

- [训练与对齐总览](training/README.md)：预训练、PEFT、SFT、RLHF、后训练全链路。
- [RL 系统优化](rl-infra/README.md)：Rollout、权重同步、Slime、Multi-LoRA、GPU 瓶颈分析。

## 前沿与复盘

- [前沿追踪](frontier/README.md)：模型演进、推理模型、结构化输出与前沿框架对比。
- [复现与复盘](bitter-lessons/README.md)：踩坑记录、Bitter Lessons、复现总结。

## 其他专题

- [CUDA 基础](cuda/basics.md)
- [显存层次](cuda/memory-hierarchy.md)
- [多模态总览](multimodal/vision-language-models.md)
- [多模态训练](multimodal/multimodal-training.md)
- [VLM Serving](multimodal/vlm-serving.md)
- [评测与幻觉检测](evaluation/benchmarks-hallucination.md)
- [系统设计方法论](system-design/methodology.md)
- [系统设计模式](system-design/patterns.md)
- [LLM 相关算法编程](coding/llm-related-algorithms.md)
- [生产环境排障](debugging/production-runbook.md)
- [源码阅读](source-reading/slime-sglang-verl-deep-dive.md)
- [Profiling 工具链](tools/profiling.md)
- [终极速查卡](cheatsheet-final.md)