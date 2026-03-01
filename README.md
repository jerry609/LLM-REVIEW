<div align="center">

# LLM-REVIEW

**LLM 推理系统全栈知识库**

从数学公式到系统设计，从注意力机制到生产部署

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-Personal_Study-blue)](#license)

</div>

---

本仓库以**数学公式驱动 · 代码可验证 · 面试实战导向**为核心理念，构建了一套完整的 LLM Inference 学习与备战体系。所有笔记均包含严格的公式推导、PyTorch 实现、以及带 KaTeX 渲染的交互式 HTML 文档。

## 目录导航

| 目录 | 定位 | 内容概览 |
|------|------|----------|
| [`math_dictionary/`](#-math_dictionary--数学公式速查字典) | 公式速查 | 27 篇系统性公式手册，覆盖线代 → 注意力 → KV Cache → 分布式推理 |
| [`notes/`](#-notes--深度技术笔记) | 深度笔记 | 20+ 子主题，从 Transformer 架构到前沿模型的完整知识体系 |
| [`notebooks/`](#-notebooks--交互式文档与实验) | 交互文档 | Jupyter 实验 + KaTeX 渲染的学术级 HTML 展示 |
| [`src/`](#-src--核心代码实现) | 代码实现 | 注意力机制、KV Cache、LoRA 等核心模块的 PyTorch 实现 |
| [`notes/bitter-lessons/`](#-bitter-lessons--项目复现血泪录) | 血泪教训 | 复现项目过程中的踩坑记录、Bitter Lessons 提炼 |
| [`mock_interview/`](#-mock_interview--模拟面试题库) | 面试实战 | 按主题 / 公司分类的系统设计与技术深挖题库 |
| [`roadmap/`](#-roadmap--学习路线) | 学习规划 | 12 周冲刺计划 + 45 天每日任务表 |
| [`benchmarks/`](#-benchmarks--评测与分析) | 评测分析 | 面试高频考点热力分析 + 周度自测模板 |
| [`tests/`](#-tests--单元测试) | 质量保障 | 注意力、KV Cache、LoRA、调度器的自动化测试 |

---

## <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/latex/latex-original.svg" width="20" /> `math_dictionary` · 数学公式速查字典

> 27 篇公式手册，面试前 15 分钟快速回顾核心公式。

<details>
<summary><b>基础数学</b></summary>

| 文件 | 内容 |
|------|------|
| [`linear-algebra-basics.md`](math_dictionary/linear-algebra-basics.md) | 矩阵乘法 FLOPs、SVD 分解、范数、余弦相似度 |
| [`symbols-glossary.md`](math_dictionary/symbols-glossary.md) | 符号与单位总表，含常见模型参数速查 |
| [`probability-and-sampling.md`](math_dictionary/probability-and-sampling.md) | Softmax、Top-k/p 采样、KL 散度、投机解码接受概率 |

</details>

<details>
<summary><b>Transformer 核心</b></summary>

| 文件 | 内容 |
|------|------|
| [`tensor-shapes.md`](math_dictionary/tensor-shapes.md) | MHA/GQA/MQA Tensor 形状速查、投影权重、RoPE、FFN |
| [`transformer-attention-math.md`](math_dictionary/transformer-attention-math.md) | 注意力公式、RoPE、ALiBi、LayerNorm/RMSNorm、残差连接 |
| [`rope-and-position-encoding.md`](math_dictionary/rope-and-position-encoding.md) | RoPE 及位置编码专题（PI、NTK、YaRN、ALiBi 对比） |
| [`mqa-vs-gqa.md`](math_dictionary/mqa-vs-gqa.md) | MQA/GQA 数学与工程对比（显存、带宽、质量权衡） |
| [`attention-complexity.md`](math_dictionary/attention-complexity.md) | 复杂度 & FLOPs、Roofline 模型、FlashAttention IO 分析 |
| [`flashattention-math.md`](math_dictionary/flashattention-math.md) | FlashAttention 全链路推导（Online Softmax、分块、FA-2/3） |
| [`linear-attention-math.md`](math_dictionary/linear-attention-math.md) | 线性注意力（Mamba、RWKV、RetNet） |
| [`gated-attention-math.md`](math_dictionary/gated-attention-math.md) | 门控注意力与信息流控制 |
| [`tokenizer-math.md`](math_dictionary/tokenizer-math.md) | BPE、压缩率、词表大小权衡 |

</details>

<details>
<summary><b>KV Cache 系统</b></summary>

| 文件 | 内容 |
|------|------|
| [`kv-memory.md`](math_dictionary/kv-memory.md) | KV 缓存显存估算与容量规划（PagedAttention、预算分配） |
| [`pagedattention-math.md`](math_dictionary/pagedattention-math.md) | PagedAttention 分页管理（碎片分析、Prefix Caching、CoW） |
| [`kv-eviction-math.md`](math_dictionary/kv-eviction-math.md) | KV 驱逐策略建模（LRU/LFU/注意力感知、公平性、粒度） |
| [`kv-compression-math.md`](math_dictionary/kv-compression-math.md) | KV 压缩与量化（GPTQ/AWQ/SmoothQuant、稀疏化） |

</details>

<details>
<summary><b>推理服务</b></summary>

| 文件 | 内容 |
|------|------|
| [`prefill-decode-performance.md`](math_dictionary/prefill-decode-performance.md) | Prefill/Decode 延迟与吞吐、Continuous Batching、Chunked Prefill |
| [`serving-metrics.md`](math_dictionary/serving-metrics.md) | 推理服务指标速查（Goodput、诊断规则、告警阈值） |
| [`queueing-and-slo.md`](math_dictionary/queueing-and-slo.md) | 排队论与 SLO（M/M/1、尾延迟分析、限流策略） |
| [`distributed-serving-math.md`](math_dictionary/distributed-serving-math.md) | 多机多卡推理（TP/PP/EP 通信分析、气泡率、负载均衡） |
| [`speculative-decoding-math.md`](math_dictionary/speculative-decoding-math.md) | 投机解码（接受-拒绝采样、加速比分析） |

</details>

<details>
<summary><b>训练与对齐</b></summary>

| 文件 | 内容 |
|------|------|
| [`optimization-and-scaling.md`](math_dictionary/optimization-and-scaling.md) | Adam/AdamW、Chinchilla Scaling Law、MFU |
| [`lora-peft-math.md`](math_dictionary/lora-peft-math.md) | LoRA 低秩增量、QLoRA、多 LoRA 服务 |
| [`rlhf-alignment-math.md`](math_dictionary/rlhf-alignment-math.md) | RLHF（Bradley-Terry、PPO、DPO、KTO） |

</details>

<details>
<summary><b>评测与工具</b></summary>

| 文件 | 内容 |
|------|------|
| [`evaluation-metrics.md`](math_dictionary/evaluation-metrics.md) | PPL、BLEU/ROUGE、Pass@k、校准、长上下文评测 |
| [`mental-math-cheatsheet.md`](math_dictionary/mental-math-cheatsheet.md) | 面试心算速记（2^n 表、GPU 参数、权重速算） |
| [`moe-routing-math.md`](math_dictionary/moe-routing-math.md) | MoE 路由策略数学建模 |

</details>

---

## <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/markdown/markdown-original.svg" width="20" /> `notes` · 深度技术笔记

> 20+ 子主题的系统性深度笔记，每篇均包含原理推导、代码示例、面试问答。

### 基础速通 [`notes/basics/`](notes/basics/)

| 文件 | 内容 |
|------|------|
| [`python-essentials.md`](notes/basics/python-essentials.md) | Python 核心语法（函数式编程、OOP、装饰器、NumPy/Pandas、asyncio） |
| [`neural-network-fundamentals.md`](notes/basics/neural-network-fundamentals.md) | 神经网络核心（前向/反向传播、损失函数、梯度下降、激活函数、完整训练循环） |
| [`pytorch-quickstart.md`](notes/basics/pytorch-quickstart.md) | PyTorch 速通（Tensor 操作、autograd、nn.Module、DataLoader、混合精度、HF 生态） |

### 注意力机制 [`notes/attention/`](notes/attention/)

| 文件 | 内容 |
|------|------|
| [`multi-head-divergence.md`](notes/attention/multi-head-divergence.md) | 多头注意力为何学到不同空间：几何投影、隐式偏置、Muon 优化器、Induction Heads |
| [`attention-evolution-and-inference.md`](notes/attention/attention-evolution-and-inference.md) | 注意力机制演进全景（MHA→MQA→GQA→MLA→SSM）+ 端到端推理流程 |
| [`mha-vs-mla-full-derivation.md`](notes/attention/mha-vs-mla-full-derivation.md) | **MHA vs MLA 全流程对比**：三次矩阵吸收推导、Decode 阶段带宽分析、伪代码 |
| [`mha-vs-gqa-full-derivation.md`](notes/attention/mha-vs-gqa-full-derivation.md) | **MHA vs GQA 全流程对比**：分组共享 KV、SRAM 复用、Roofline 分析、GQA vs MLA 哲学 |
| [`mha-vs-dsa-full-derivation.md`](notes/attention/mha-vs-dsa-full-derivation.md) | **MHA vs DSA 全流程对比**：闪电索引器、ReLU 替代 Softmax、序列×特征双重压缩 3640× |
| [`mha-vs-linear-attention-full-derivation.md`](notes/attention/mha-vs-linear-attention-full-derivation.md) | **MHA vs 线性注意力全流程对比**：结合律消除 L²、RNN 等价、状态压缩瓶颈、前沿变体 |
| [`attention-mechanisms-unified-comparison.md`](notes/attention/attention-mechanisms-unified-comparison.md) | **五大注意力统一对比表**：MHA/GQA/MLA/DSA/Linear 全流程并排矩阵推导（Excel 风格） |
| [`flashattention.md`](notes/attention/flashattention.md) | FlashAttention 原理与实现细节 |
| [`linear-attention.md`](notes/attention/linear-attention.md) | 线性注意力机制（Mamba、RWKV） |
| [`long-context.md`](notes/attention/long-context.md) | 长上下文处理技术 |

### 模型架构 [`notes/architectures/`](notes/architectures/)

| 文件 | 内容 |
|------|------|
| [`deepseek-v3.md`](notes/architectures/deepseek-v3.md) | DeepSeek-V3 架构拆解（MLA + MoE） |
| [`llama3.md`](notes/architectures/llama3.md) | LLaMA-3 架构分析（GQA + RoPE） |
| [`mixtral-moe.md`](notes/architectures/mixtral-moe.md) | Mixtral MoE 架构与路由机制 |
| [`moe-training.md`](notes/architectures/moe-training.md) | MoE 训练策略与负载均衡 |
| [`position-encoding.md`](notes/architectures/position-encoding.md) | 位置编码方案对比 |
| [`ssm-hybrid.md`](notes/architectures/ssm-hybrid.md) | SSM 混合架构（Mamba / Jamba） |
| [`transformer-components.md`](notes/architectures/transformer-components.md) | Transformer 核心组件详解 |

### KV Cache [`notes/kv-cache/`](notes/kv-cache/) · [`notes/kv-compression/`](notes/kv-compression/) · [`notes/kv-eviction/`](notes/kv-eviction/)

| 文件 | 内容 |
|------|------|
| [`kv-cache/concepts.md`](notes/kv-cache/concepts.md) | KV Cache 核心概念 |
| [`kv-cache/paged-attention.md`](notes/kv-cache/paged-attention.md) | PagedAttention 深度剖析 |
| [`kv-compression/quantization.md`](notes/kv-compression/quantization.md) | KV 量化（FP8/INT8/INT4/KIVI 2-bit） |
| [`kv-compression/sparsity.md`](notes/kv-compression/sparsity.md) | KV 稀疏化策略 |
| [`kv-eviction/policies.md`](notes/kv-eviction/policies.md) | 驱逐策略（H2O、StreamingLLM、SnapKV、Ada-KV、ACTA） |

### 推理与解码 [`notes/inference/`](notes/inference/)

| 文件 | 内容 |
|------|------|
| [`decoding-strategies.md`](notes/inference/decoding-strategies.md) | 解码策略全景（Greedy / Beam / Sampling） |
| [`speculative-decoding.md`](notes/inference/speculative-decoding.md) | 投机解码深度分析 |
| [`pd-disaggregation.md`](notes/inference/pd-disaggregation.md) | Prefill-Decode 分离架构 |

### 分布式系统 [`notes/distributed/`](notes/distributed/)

| 文件 | 内容 |
|------|------|
| [`tp-pp-tradeoff.md`](notes/distributed/tp-pp-tradeoff.md) | TP/PP 并行策略权衡 |
| [`moe-ep.md`](notes/distributed/moe-ep.md) | MoE Expert Parallelism |
| [`moe-inference-deep.md`](notes/distributed/moe-inference-deep.md) | MoE 推理深度优化 |

### 训练与对齐 [`notes/training/`](notes/training/)

| 文件 | 内容 |
|------|------|
| [`pretraining-data.md`](notes/training/pretraining-data.md) | 预训练数据工程（CLM/MLM/MTP 任务、数据来源、清洗、配比） |
| [`pretraining-pipeline.md`](notes/training/pretraining-pipeline.md) | **预训练实战全流程**（数据打包→Tokenizer→模型初始化→训练循环→MFU→断点续训→评估导出） |
| [`scaling-law.md`](notes/training/scaling-law.md) | Scaling Law 与训练策略（Kaplan / Chinchilla / 推理 Scaling） |
| [`peft-methods-comparison.md`](notes/training/peft-methods-comparison.md) | **PEFT 方法全景对比**（LoRA/QLoRA/DoRA/Adapter/IA³/Prefix-Tuning/P-Tuning v2 + 选型指南） |
| [`instruction-data-construction.md`](notes/training/instruction-data-construction.md) | **指令数据构建**（Self-Instruct/Evol-Instruct/领域构建/质量过滤/配比策略） |
| [`finetuning-frameworks.md`](notes/training/finetuning-frameworks.md) | **微调框架实战**（LlamaFactory YAML/WebUI + Unsloth 加速 + trl SFT/DPO/GRPO + 资源估算） |
| [`lora-rlhf.md`](notes/training/lora-rlhf.md) | LoRA + RLHF 联合训练 |
| [`alignment-pipeline.md`](notes/training/alignment-pipeline.md) | 对齐训练全流程（SFT → RLHF → DPO） |
| [`post-training-advanced.md`](notes/training/post-training-advanced.md) | 后训练高级技术（RLAIF/SimPO/KTO/ORPO/IPO/SPIN） |
| [`reward-model-and-rlhf-practice.md`](notes/training/reward-model-and-rlhf-practice.md) | **RM 训练 + RLHF 实战**（奖励模型架构/训练/PPO 代码/DPO checklist/GRPO 实战/调试指南） |

### 推理框架 [`notes/frameworks/`](notes/frameworks/)

| 文件 | 内容 |
|------|------|
| [`vllm-architecture.md`](notes/frameworks/vllm-architecture.md) | vLLM 架构深度解析 |
| [`trt-llm-sglang.md`](notes/frameworks/trt-llm-sglang.md) | TensorRT-LLM / SGLang 对比 |

### 服务与运维 [`notes/serving/`](notes/serving/) · [`notes/llm-system/`](notes/llm-system/)

| 文件 | 内容 |
|------|------|
| [`serving/capacity-planning.md`](notes/serving/capacity-planning.md) | 容量规划与资源分配 |
| [`serving/cost-optimization.md`](notes/serving/cost-optimization.md) | 推理成本优化 |
| [`llm-system/throughput-latency.md`](notes/llm-system/throughput-latency.md) | 吞吐-延迟权衡分析 |
| [`llm-system/serving.md`](notes/llm-system/serving.md) | LLM 服务系统设计 |

### 前沿追踪 [`notes/frontier/`](notes/frontier/)

| 文件 | 内容 |
|------|------|
| [`model-evolution-timeline.md`](notes/frontier/model-evolution-timeline.md) | 大模型演进时间线 |
| [`deepseek-v3-r1.md`](notes/frontier/deepseek-v3-r1.md) | DeepSeek-V3/R1 技术分析 |
| [`reasoning-models.md`](notes/frontier/reasoning-models.md) | 推理模型（o1、DeepSeek-R1、QwQ） |
| [`test-time-compute.md`](notes/frontier/test-time-compute.md) | Test-Time Compute Scaling |
| [`qwen-series.md`](notes/frontier/qwen-series.md) | Qwen 系列技术追踪 |
| [`structured-output.md`](notes/frontier/structured-output.md) | 结构化输出（JSON Schema、Grammar Decoding） |

### RL 系统优化 [`notes/rl-infra/`](notes/rl-infra/)

| 文件 | 内容 |
|------|------|
| [`rl-training-inference-systems.md`](notes/rl-infra/rl-training-inference-systems.md) | **RL 训练/推理全链路优化**：Rollout 加速、权重同步、GRPO 系统设计、vLLM/SGLang/AReaL/Slime/Megatron 框架剖析 |
| [`multi-lora-joint-training.md`](notes/rl-infra/multi-lora-joint-training.md) | **千级 Multi-LoRA 并行训练**：Batched GEMM / S-LoRA 原理、权重池管理、Joint Training 策略、资源调度系统设计 |
| [`gpu-optimization-bottleneck-analysis.md`](notes/rl-infra/gpu-optimization-bottleneck-analysis.md) | **GPU 优化与瓶颈定位**：Roofline 模型、Profiling 工具链、Compute/Memory Bound 诊断、RL 特有稳定性优化 |
| [`slime-deep-dive.md`](notes/rl-infra/slime-deep-dive.md) | **Slime 框架深度拆解**：异步流水线源码、Data Buffer、SGLang 集成、权重同步、staleness 分析、vs verl/AReaL 对比 |
| [`hands-on-experiment-log.md`](notes/rl-infra/hands-on-experiment-log.md) | **实战 Demo 清单 & 试错日志**：13 个必跑 Demo（vLLM Multi-LoRA / SGLang Prefix / Slime 训练 / nsys Profiling）+ 踩坑速查 |

### 🩸 Bitter Lessons [`notes/bitter-lessons/`](notes/bitter-lessons/)

> 每一次复现失败，都比读十遍论文学到的多。

| 文件 | 内容 |
|------|------|
| [`reproduction-log.md`](notes/bitter-lessons/reproduction-log.md) | **项目复现血泪录**：环境踩坑、训练收敛、推理部署、分布式系统中的认知颠覆与 Bitter Lessons 提炼 |

### 其他专题

| 目录 | 内容 |
|------|------|
| [`notes/cuda/`](notes/cuda/) | CUDA 编程基础与显存层次 |
| [`notes/multimodal/`](notes/multimodal/) | 多模态（[VLM Serving](notes/multimodal/vlm-serving.md) · [ViT/CLIP/BLIP/LLaVA 架构](notes/multimodal/vision-language-models.md) · [多模态训练全流程](notes/multimodal/multimodal-training.md)） |
| [`notes/evaluation/`](notes/evaluation/) | 评测基准与幻觉检测 |
| [`notes/system-design/`](notes/system-design/) | 系统设计方法论与模式 |
| [`notes/coding/`](notes/coding/) | LLM 相关算法编程 |
| [`notes/debugging/`](notes/debugging/) | 生产环境问题排查手册 |
| [`notes/source-reading/`](notes/source-reading/) | 源码阅读（SGLang / vERL / Slime） |
| [`notes/tools/`](notes/tools/) | 性能分析工具（Profiling） |

---

## <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/jupyter/jupyter-original-wordmark.svg" width="20" /> `notebooks` · 交互式文档与实验

> Jupyter Notebook 动手实验 + KaTeX 渲染的学术级 HTML 文档，可直接在浏览器中查看。

### 学术级 HTML 文档（KaTeX 公式渲染）

| 文件 | 内容 |
|------|------|
| [`multi-head-divergence.html`](notebooks/multi-head-divergence.html) | 多头注意力分化机制：几何投影 → 隐式偏置 → Muon 优化器 → Induction Heads |
| [`attention-evolution-and-inference.html`](notebooks/attention-evolution-and-inference.html) | 注意力机制演进（MHA/MQA/GQA/MLA）+ 现代推理引擎优化全景 |
| [`mha-vs-mla-full-derivation.html`](notebooks/mha-vs-mla-full-derivation.html) | **MHA vs MLA 全流程对比**：三次矩阵吸收推导 + Decode 伪代码 |
| [`mha-vs-gqa-full-derivation.html`](notebooks/mha-vs-gqa-full-derivation.html) | **MHA vs GQA 全流程对比**：分组共享 + SRAM 复用 + Roofline 性能分析 |
| [`mha-vs-dsa-full-derivation.html`](notebooks/mha-vs-dsa-full-derivation.html) | **MHA vs DSA 全流程对比**：闪电索引器 + 稀疏 MLA + 3640× 带宽压缩 |
| [`mha-vs-linear-attention-full-derivation.html`](notebooks/mha-vs-linear-attention-full-derivation.html) | **MHA vs 线性注意力全流程对比**：结合律 + RNN 等价 + 恒定 Cache |
| [`attention-mechanisms-unified-comparison.html`](notebooks/attention-mechanisms-unified-comparison.html) | **五大注意力统一对比表**：MHA/GQA/MLA/DSA/Linear 全流程并排（彩色表格） |

### Jupyter Notebook 实验

| 文件 | 内容 |
|------|------|
| [`python_nn_pytorch_fundamentals_workshop.ipynb`](notebooks/python_nn_pytorch_fundamentals_workshop.ipynb) | **Python/NN/PyTorch 基础实战**：函数式编程、手写反向传播、Dataset/DataLoader、MLP 训练闭环、最小 next-token 训练 |
| [`mini_transformer_from_scratch_workshop.ipynb`](notebooks/mini_transformer_from_scratch_workshop.ipynb) | **手写 Mini Transformer**：Causal Self-Attention、Decoder-only Block、next-token 训练、采样生成、注意力热力图 |
| [`attention_tokenizer_beamsearch.ipynb`](notebooks/attention_tokenizer_beamsearch.ipynb) | 注意力计算、Tokenizer、Beam Search 手撕实现 |
| [`llm_inference_fundamentals.ipynb`](notebooks/llm_inference_fundamentals.ipynb) | LLM 推理基础：Prefill/Decode、KV Cache 机制 |
| [`kv_cache_paged_lru_workshop.ipynb`](notebooks/kv_cache_paged_lru_workshop.ipynb) | PagedAttention 与 LRU 驱逐策略实战 |
| [`quantization_precision_experiment.ipynb`](notebooks/quantization_precision_experiment.ipynb) | 量化精度实验（FP16/INT8/INT4 对比） |
| [`speculative_decoding_simulator.ipynb`](notebooks/speculative_decoding_simulator.ipynb) | 投机解码模拟器 |
| [`distributed_inference_roofline.ipynb`](notebooks/distributed_inference_roofline.ipynb) | 分布式推理 Roofline 分析 |
| [`rl_ppo_grpo_implementation.ipynb`](notebooks/rl_ppo_grpo_implementation.ipynb) | PPO / GRPO 强化学习实现 |
| [`grpo_training.ipynb`](notebooks/grpo_training.ipynb) | GRPO 训练流程 |
| [`reasoning_models_workshop.ipynb`](notebooks/reasoning_models_workshop.ipynb) | 推理模型 Workshop |
| [`rag_prefix_caching_simulator.ipynb`](notebooks/rag_prefix_caching_simulator.ipynb) | RAG + Prefix Caching 模拟 |
| [`vllm_architecture_walkthrough.ipynb`](notebooks/vllm_architecture_walkthrough.ipynb) | vLLM 架构源码走读 |
| [`mla_latent_space_analysis.ipynb`](notebooks/mla_latent_space_analysis.ipynb) | **MLA 潜在空间分析**：SVD 衰减 / 压缩重建 / RoPE 解耦 / 瓶颈训练 / 注意力保真 |
| [`needle_in_haystack_demo.ipynb`](notebooks/needle_in_haystack_demo.ipynb) | **大海捞针评测**：5 模型模拟 / 热力图 / Lost in the Middle / MLA vs MHA 对比 |
| [`leetcode_llm_system_related.ipynb`](notebooks/leetcode_llm_system_related.ipynb) | LLM 系统相关算法题 |

---

## <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pytorch/pytorch-original.svg" width="20" /> `src` · 核心代码实现

> 可运行的 PyTorch 代码，覆盖推理系统核心模块。

```
src/
├── attention/
│   ├── mha_gqa.py          # MHA / GQA / MQA 实现
│   ├── rope_rmsnorm.py     # RoPE 位置编码 + RMSNorm
│   └── flash_attn_sim.py   # FlashAttention 模拟实现
├── kv_cache/
│   ├── core.py             # KV Cache 核心数据结构
│   ├── eviction/
│   │   └── policies.py     # 驱逐策略（LRU / LFU / H2O）
│   └── compression/
│       └── quantizer.py    # KV 量化压缩
├── training/
│   └── lora.py             # LoRA 低秩适配实现
├── simulators/
│   └── scheduler.py        # 推理调度器模拟
└── cuda/
    └── simulation.py       # CUDA 计算模拟
```

---

## <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/readthedocs/readthedocs-original.svg" width="20" /> `mock_interview` · 模拟面试题库

> 系统性整理的技术深挖与系统设计题库，按维度分类，附评分维度与追问链。

### 按主题 [`mock_interview/by-topic/`](mock_interview/by-topic/)

- **KV Cache 系统设计**（4 道）：驱逐策略深挖、压缩量化、Prefill-Decode-PagedAttention
- **Serving 系统设计**（3 道）：高并发 LLM 服务、长上下文、多租户隔离
- **前沿技术**（3 道）：推理模型对比、架构演进、长上下文方案
- **RAG 与成本优化**（2 道）

### 按风格 [`mock_interview/by-company/`](mock_interview/by-company/)

> 不同团队的技术面试侧重点各异，此处按典型面试风格归类。

| 文件 | 面试风格 |
|------|----------|
| [`bytedance.md`](mock_interview/by-company/bytedance.md) | 偏重工程落地与极致性能优化 |
| [`alibaba.md`](mock_interview/by-company/alibaba.md) | 偏重大规模分布式系统设计 |
| [`tencent.md`](mock_interview/by-company/tencent.md) | 偏重全链路架构与稳定性保障 |

### 行为面试 [`mock_interview/behavior/`](mock_interview/behavior/)

| 文件 | 内容 |
|------|------|
| [`self-introduction.md`](mock_interview/behavior/self-introduction.md) | 自我介绍框架与模板 |
| [`star-stories.md`](mock_interview/behavior/star-stories.md) | STAR 方法论故事库 |
| [`star-kv-cache-serving.md`](mock_interview/behavior/star-kv-cache-serving.md) | 推理系统项目 STAR 案例 |
| [`project-one-pager.md`](mock_interview/behavior/project-one-pager.md) | 项目一页纸模板 |

---

## <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" width="20" /> `roadmap` · 学习路线

| 文件 | 内容 |
|------|------|
| [`00-overview.md`](roadmap/00-overview.md) | 总体概览与知识图谱 |
| [`01-12week-plan.md`](roadmap/01-12week-plan.md) | 12 周系统冲刺计划 |
| [`02-45day-daily-plan.md`](roadmap/02-45day-daily-plan.md) | 45 天每日任务清单 |

---

## <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/grafana/grafana-original.svg" width="20" /> `benchmarks` · 评测与分析

| 文件 | 内容 |
|------|------|
| [`reports/kv_interview_hotspots_2026-02-16.md`](benchmarks/reports/kv_interview_hotspots_2026-02-16.md) | KV Cache 面试热点分析（15 个权威来源交叉验证） |
| [`runs/week_template.md`](benchmarks/runs/week_template.md) | 周度自测模板 |

---

## <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pytest/pytest-original.svg" width="20" /> `tests` · 单元测试

```bash
# 运行全部测试
python -m pytest tests/ -v

# 单独运行
python -m pytest tests/test_attention.py   # 注意力机制
python -m pytest tests/test_kv_cache.py    # KV Cache
python -m pytest tests/test_lora.py        # LoRA
python -m pytest tests/test_scheduler.py   # 调度器
```

---

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/jerry609/LLM-REVIEW.git
cd LLM-REVIEW

# 浏览学术文档（浏览器直接打开）
open notebooks/multi-head-divergence.html
open notebooks/attention-evolution-and-inference.html

# 运行代码实验
pip install torch numpy
python -c "from src.attention.mha_gqa import *; print('Ready!')"

# 运行测试
python -m pytest tests/ -v
```

---

## 项目结构总览

```
LLM-REVIEW/
├── README.md                    ← 你在这里
├── math_dictionary/             ← 27 篇公式速查手册
├── notes/                       ← 20+ 主题深度技术笔记
│   ├── attention/               ←   注意力机制（MHA 分化、演进、FlashAttention）
│   ├── architectures/           ←   模型架构（LLaMA-3、DeepSeek-V3、Mixtral）
│   ├── kv-cache/                ←   KV Cache 核心 + PagedAttention
│   ├── kv-compression/          ←   KV 压缩与量化
│   ├── kv-eviction/             ←   KV 驱逐策略
│   ├── inference/               ←   推理与解码策略
│   ├── distributed/             ←   分布式推理（TP/PP/EP）
│   ├── basics/                  ←   基础速通（Python、神经网络、PyTorch）
│   ├── training/                ←   训练全栈（预训练 / PEFT / 指令数据 / 微调框架 / RLHF）
│   ├── rl-infra/                ←   RL 系统优化（Slime / Multi-LoRA / GPU 优化）
│   ├── bitter-lessons/          ←   项目复现血泪录（Bitter Lessons）
│   ├── frameworks/              ←   推理框架（vLLM、TRT-LLM、SGLang）
│   ├── serving/                 ←   服务与运维
│   ├── frontier/                ←   前沿模型追踪
│   ├── multimodal/              ←   多模态（ViT/CLIP/LLaVA 架构、训练、VLM 推理）
│   └── ...                      ←   CUDA、评测、系统设计、源码阅读等
├── notebooks/                   ← Jupyter 实验 + HTML 学术文档
├── src/                         ← PyTorch 核心实现
├── mock_interview/              ← 面试题库（主题 / 公司 / 行为）
├── roadmap/                     ← 学习路线与计划
├── benchmarks/                  ← 考点分析与自测
└── tests/                       ← 单元测试
```

---

## Roadmap

> 本仓库正在向更专业的形态演进。

- **LaTeX 重写**：所有核心笔记将迁移至 LaTeX 排版，输出可引用的 PDF 文档（公式、表格、算法伪代码使用 `algorithm2e` / `booktabs` 等学术级排版）。
- **独立在线站点**：计划基于 [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) 或 [VitePress](https://vitepress.dev/) 构建可全文搜索的静态文档站，支持 KaTeX 实时渲染、暗色模式、版本化发布，域名与部署方案待定。
- **动画演示关键流程**：为以下核心机制制作可交互的动画 / 动图，嵌入在线站点与 Notebook：

  | 动画主题 | 演示内容 | 技术方案 |
  |----------|----------|----------|
  | **KV Cache 生命周期** | Prefill 填充 → Decode 逐步追加 → PagedAttention 分页 → 驱逐回收 | Manim / D3.js |
  | **MHA vs MLA 矩阵吸收** | Q/K/V 投影 → 潜空间压缩 → 吸收矩阵合并 → Decode 读取对比 | Manim |
  | **Attention 热力图动态** | Token 逐步生成时 causal mask 扩展、注意力权重分布实时变化 | Matplotlib animation / Plotly |
  | **LoRA 权重注入** | 原始权重冻结 → 低秩分解 A·B 旁路注入 → 推理时合并 | Manim |
  | **GRPO 训练循环** | Rollout 采样 → Group 内打分 → 优势计算 → 策略更新 → 权重同步 | D3.js / React Flow |
  | **Continuous Batching** | 请求到达 → 动态插入 batch → Prefill/Decode 交错 → 请求完成释放 slot | D3.js |
  | **分布式并行策略** | TP 切分 Attention 头 → PP 流水线气泡 → EP 专家路由 All-to-All | Manim / Excalidraw animation |

- **Notebook → Colab**：所有 `.ipynb` 将附带一键 Open in Colab 按钮，降低环境配置门槛。

---

## 参考资源

| 资源 | 说明 |
|------|------|
| [wyf3/llm_related](https://github.com/wyf3/llm_related) | 复现大模型相关算法及学习记录（GRPO/PPO/DAPO from scratch、知识蒸馏、MoE 训练、多模态等） |
| [CMU 17-445/645: ML in Production](https://mlip-cmu.github.io/f2025/) | CMU「机器学习生产化」课程——涵盖 MLOps、数据基础设施、模型部署与监控、负责任 AI、系统质量保证等全生命周期主题，附带百万用户级推荐系统大作业，[教材](https://mlip-cmu.github.io/book/) 与 [课件](https://github.com/mlip-cmu) 全部开源 |

---

<div align="center">

**Built with** &nbsp;
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pytorch/pytorch-original.svg" width="16" /> PyTorch &nbsp;·&nbsp;
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="16" /> Python &nbsp;·&nbsp;
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/jupyter/jupyter-original-wordmark.svg" width="16" /> Jupyter &nbsp;·&nbsp;
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg" width="16" /> NumPy

</div>
