# LLM-REVIEW

> **LLM 推理系统全栈知识库** — 从数学公式到系统设计，从注意力机制到生产部署，系统性覆盖大模型推理工程师所需的全部知识图谱。

本仓库以**数学公式驱动 + 代码可验证 + 面试实战导向**为核心理念，构建了一套完整的 LLM Inference 学习与备战体系。所有笔记均包含严格的公式推导、PyTorch 实现、以及带 KaTeX 渲染的交互式 HTML 文档。

---

## 目录导航

| 目录 | 定位 | 内容概览 |
|------|------|----------|
| [`math_dictionary/`](#-math_dictionary数学公式速查字典) | 公式速查 | 27 篇系统性公式手册，覆盖线代 → 注意力 → KV Cache → 分布式推理 |
| [`notes/`](#-notes深度技术笔记) | 深度笔记 | 20+ 子主题，从 Transformer 架构到前沿模型的完整知识体系 |
| [`notebooks/`](#-notebooks交互式文档与实验) | 交互文档 | Jupyter 实验 + KaTeX 渲染的学术级 HTML 展示 |
| [`src/`](#-src核心代码实现) | 代码实现 | 注意力机制、KV Cache、LoRA 等核心模块的 PyTorch 实现 |
| [`mock_interview/`](#-mock_interview模拟面试题库) | 面试实战 | 按主题 / 公司分类的系统设计与技术深挖题库 |
| [`roadmap/`](#-roadmap学习路线) | 学习规划 | 12 周冲刺计划 + 45 天每日任务表 |
| [`benchmarks/`](#-benchmarks评测与分析) | 评测分析 | 面试高频考点热力分析 + 周度自测模板 |
| [`tests/`](#-tests单元测试) | 质量保障 | 注意力、KV Cache、LoRA、调度器的自动化测试 |

---

## 📐 `math_dictionary/`：数学公式速查字典

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

## 📝 `notes/`：深度技术笔记

> 20+ 子主题的系统性深度笔记，每篇均包含原理推导、代码示例、面试问答。

### 注意力机制 [`notes/attention/`](notes/attention/)

| 文件 | 内容 |
|------|------|
| [`multi-head-divergence.md`](notes/attention/multi-head-divergence.md) | 多头注意力为何学到不同空间：几何投影、隐式偏置、Muon 优化器、Induction Heads |
| [`attention-evolution-and-inference.md`](notes/attention/attention-evolution-and-inference.md) | 注意力机制演进全景（MHA→MQA→GQA→MLA→SSM）+ 端到端推理流程 |
| [`mha-vs-mla-full-derivation.md`](notes/attention/mha-vs-mla-full-derivation.md) | **MHA vs MLA 全流程对比**：三次矩阵吸收推导、Decode 阶段带宽分析、伪代码 |
| [`mha-vs-gqa-full-derivation.md`](notes/attention/mha-vs-gqa-full-derivation.md) | **MHA vs GQA 全流程对比**：分组共享 KV、SRAM 复用、Roofline 分析、GQA vs MLA 哲学 |
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
| [`alignment-pipeline.md`](notes/training/alignment-pipeline.md) | 对齐训练全流程（SFT → RLHF → DPO） |
| [`lora-rlhf.md`](notes/training/lora-rlhf.md) | LoRA + RLHF 联合训练 |
| [`post-training-advanced.md`](notes/training/post-training-advanced.md) | 后训练高级技术 |
| [`scaling-law.md`](notes/training/scaling-law.md) | Scaling Law 与训练策略 |
| [`pretraining-data.md`](notes/training/pretraining-data.md) | 预训练数据工程 |

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

### 其他专题

| 目录 | 内容 |
|------|------|
| [`notes/cuda/`](notes/cuda/) | CUDA 编程基础与显存层次 |
| [`notes/multimodal/`](notes/multimodal/) | 多模态模型推理（VLM Serving） |
| [`notes/evaluation/`](notes/evaluation/) | 评测基准与幻觉检测 |
| [`notes/system-design/`](notes/system-design/) | 系统设计方法论与模式 |
| [`notes/coding/`](notes/coding/) | LLM 相关算法编程 |
| [`notes/debugging/`](notes/debugging/) | 生产环境问题排查手册 |
| [`notes/source-reading/`](notes/source-reading/) | 源码阅读（SGLang / vERL / Slime） |
| [`notes/tools/`](notes/tools/) | 性能分析工具（Profiling） |

---

## 📊 `notebooks/`：交互式文档与实验

> Jupyter Notebook 动手实验 + KaTeX 渲染的学术级 HTML 文档，可直接在浏览器中查看。

### 学术级 HTML 文档（KaTeX 公式渲染）

| 文件 | 内容 |
|------|------|
| [`multi-head-divergence.html`](notebooks/multi-head-divergence.html) | 多头注意力分化机制：几何投影 → 隐式偏置 → Muon 优化器 → Induction Heads |
| [`attention-evolution-and-inference.html`](notebooks/attention-evolution-and-inference.html) | 注意力机制演进（MHA/MQA/GQA/MLA）+ 现代推理引擎优化全景 |
| [`mha-vs-mla-full-derivation.html`](notebooks/mha-vs-mla-full-derivation.html) | **MHA vs MLA 全流程对比**：三次矩阵吸收推导 + Decode 伪代码 |
| [`mha-vs-gqa-full-derivation.html`](notebooks/mha-vs-gqa-full-derivation.html) | **MHA vs GQA 全流程对比**：分组共享 + SRAM 复用 + Roofline 性能分析 |
| [`kv_cache_pipeline.html`](notebooks/kv_cache_pipeline.html) | KV Cache 压缩流水线：H2O / StreamingLLM / SnapKV / Ada-KV / ACTA |

### Jupyter Notebook 实验

| 文件 | 内容 |
|------|------|
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

## 💻 `src/`：核心代码实现

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

## 🎯 `mock_interview/`：模拟面试题库

> 按主题 / 公司 / 行为面试分类，含评分标准与追问链。

### 按主题 [`mock_interview/by-topic/`](mock_interview/by-topic/)

- **KV Cache 系统设计**（4 道）：驱逐策略深挖、压缩量化、Prefill-Decode-PagedAttention
- **Serving 系统设计**（3 道）：100 QPS LLM 服务、长上下文、多租户
- **前沿技术**（3 道）：推理模型对比、模型比较、长上下文方案
- **RAG 与成本优化**（2 道）

### 按公司 [`mock_interview/by-company/`](mock_interview/by-company/)

| 文件 | 公司 |
|------|------|
| [`bytedance.md`](mock_interview/by-company/bytedance.md) | 字节跳动 |
| [`alibaba.md`](mock_interview/by-company/alibaba.md) | 阿里巴巴 |
| [`tencent.md`](mock_interview/by-company/tencent.md) | 腾讯 |

### 行为面试 [`mock_interview/behavior/`](mock_interview/behavior/)

| 文件 | 内容 |
|------|------|
| [`self-introduction.md`](mock_interview/behavior/self-introduction.md) | 自我介绍模板 |
| [`star-stories.md`](mock_interview/behavior/star-stories.md) | STAR 故事库 |
| [`star-kv-cache-serving.md`](mock_interview/behavior/star-kv-cache-serving.md) | KV Cache 项目 STAR 案例 |
| [`project-one-pager.md`](mock_interview/behavior/project-one-pager.md) | 项目一页纸 |

---

## 🗺️ `roadmap/`：学习路线

| 文件 | 内容 |
|------|------|
| [`00-overview.md`](roadmap/00-overview.md) | 总体概览与知识图谱 |
| [`01-12week-plan.md`](roadmap/01-12week-plan.md) | 12 周系统冲刺计划 |
| [`02-45day-daily-plan.md`](roadmap/02-45day-daily-plan.md) | 45 天每日任务清单 |

---

## 📈 `benchmarks/`：评测与分析

| 文件 | 内容 |
|------|------|
| [`reports/kv_interview_hotspots_2026-02-16.md`](benchmarks/reports/kv_interview_hotspots_2026-02-16.md) | KV Cache 面试热点分析（15 个权威来源交叉验证） |
| [`runs/week_template.md`](benchmarks/runs/week_template.md) | 周度自测模板 |

---

## 🧪 `tests/`：单元测试

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
open notebooks/kv_cache_pipeline.html

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
│   ├── training/                ←   训练与对齐（LoRA、RLHF、Scaling Law）
│   ├── frameworks/              ←   推理框架（vLLM、TRT-LLM、SGLang）
│   ├── serving/                 ←   服务与运维
│   ├── frontier/                ←   前沿模型追踪
│   └── ...                      ←   CUDA、多模态、评测等
├── notebooks/                   ← Jupyter 实验 + HTML 学术文档
├── src/                         ← PyTorch 核心实现
├── mock_interview/              ← 面试题库（主题/公司/行为）
├── roadmap/                     ← 学习路线与计划
├── benchmarks/                  ← 考点分析与自测
└── tests/                       ← 单元测试
```

---

## License

This project is for personal study and interview preparation purposes.
