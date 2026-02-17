# 12 周 LLM Inference Engineer 面试冲刺计划

> 前置说明：
> - ✏️ = 写笔记/公式卡  🧑‍💻 = 写代码  🎤 = 口述练习  📄 = 论文精读  🧪 = 模拟面试
> - 每个小节标注了对应的仓库文件路径，学完即填充
> - 「可选」标记的内容可根据时间取舍，核心内容不可跳过

---

## Phase 1: 地基加固（W1-W3）

### Week 1: Transformer 架构精讲

**目标**：能在白板上画出完整的 Transformer decoder 前向流程，口述每一步的 tensor shape 变化。

| 天 | 任务 | 类型 | 产出 / 文件 |
|----|------|------|-------------|
| D1 | 复习 `math_dictionary/tensor-shapes.md` + `transformer-attention-math.md` | ✏️ | 手画 shape 流转图 |
| D2 | 详细拆解 Llama 3 架构：GQA + RoPE + SwiGLU + RMSNorm | ✏️ | → `notes/architectures/llama3.md`（新建） |
| D3 | 对比 Llama 3 vs Mixtral（MoE）：路由、EP、负载均衡 | ✏️ | → `notes/architectures/mixtral-moe.md`（新建） |
| D4 | PyTorch 手写 Multi-Head Attention（含 GQA） | 🧑‍💻 | → `src/attention/mha_gqa.py`（新建） |
| D5 | PyTorch 手写 RoPE + RMSNorm | 🧑‍💻 | → `src/attention/rope_rmsnorm.py`（新建） |
| D6 | 口述练习：「请解释 GQA 相比 MHA 的优势」（限时 5 分钟） | 🎤 | 录音 + 自评 |
| D7 | 通读 `math_dictionary/mqa-vs-gqa.md`，补充 DeepSeek-V3 的 MLA | ✏️ | → `notes/architectures/deepseek-v3.md`（新建） |

**本周面试题（自测）**：
1. 画出 Llama3-70B 的完整前向流程，标注每层的 tensor shape
2. GQA 的 KV head 数量如何影响显存和质量？给出公式
3. SwiGLU 和标准 FFN 的区别是什么？FLOPs 差多少？

---

### Week 2: 位置编码 + 注意力变体

**目标**：深入理解 RoPE 及其扩展，掌握线性注意力和门控注意力的原理。

| 天 | 任务 | 类型 | 产出 / 文件 |
|----|------|------|-------------|
| D1 | 精读 `rope-and-position-encoding.md`，推导 RoPE 旋转矩阵 | ✏️ | 手推公式验证 |
| D2 | RoPE 扩展：PI、NTK-Aware、YaRN 的数学原理 | ✏️ | 更新 `notes/architectures/` 相关笔记 |
| D3 | 精读 `flashattention-math.md` + FlashAttention-2 论文核心部分 | 📄 | → `notes/attention/flashattention.md`（新建） |
| D4 | 精读 `linear-attention-math.md`：Mamba/RWKV/RetNet 对比 | 📄 | → `notes/attention/linear-attention.md`（新建） |
| D5 | 实现简化版 FlashAttention 的 tiling 逻辑（Python 模拟） | 🧑‍💻 | → `src/attention/flash_attn_sim.py`（新建） |
| D6 | 口述：「FlashAttention 为什么快？它减少了什么？」 | 🎤 | 限时 3 分钟 |
| D7 | 总复习 W1-W2，整理薄弱点 | ✏️ | 更新 `benchmarks/` 自评记录 |

**本周面试题**：
1. RoPE 如何实现相对位置编码？为什么不需要额外参数？
2. FlashAttention 的 IO 复杂度是多少？和标准注意力比优化了什么？
3. Mamba 和标准 Transformer 的本质区别是什么？各自的优劣？

---

### Week 3: Tokenizer + 训练基础 + Scaling Law

**目标**：理解从 tokenizer 到 loss 的完整训练流程，掌握 Scaling Law 的直觉。

| 天 | 任务 | 类型 | 产出 / 文件 |
|----|------|------|-------------|
| D1 | 复习 `tokenizer-math.md` + `probability-and-sampling.md` | ✏️ | — |
| D2 | 精读 `optimization-and-scaling.md`：AdamW + Chinchilla + MFU | ✏️ | → `notes/training/scaling-law.md`（新建） |
| D3 | 精读 `lora-peft-math.md` + `rlhf-alignment-math.md` | ✏️ | → `notes/training/lora-rlhf.md`（新建） |
| D4 | PyTorch 实现 LoRA 层（简化版） | 🧑‍💻 | → `src/training/lora.py`（新建） |
| D5 | 理解预训练 → SFT → RLHF/DPO 完整链路 | ✏️ | → `notes/training/alignment-pipeline.md`（新建） |
| D6 | 口述：「解释 Chinchilla Scaling Law 及其对推理的影响」 | 🎤 | 限时 5 分钟 |
| D7 | Phase 1 全面复习 + 阶段自测 | 🧪 | 3 道综合题模拟 |

**Phase 1 结束检查清单**：
- [ ] 能画出 Llama3 完整架构图（含参数量标注）
- [ ] 能手写 GQA attention 的 PyTorch 实现
- [ ] 能口述 RoPE、FlashAttention、GQA 的核心优势
- [ ] 能解释 Scaling Law 对模型大小和数据量的指导意义

---

## Phase 2: 推理核心（W4-W6）

### Week 4: KV Cache 深度掌握

**目标**：成为 KV Cache 领域的专家，能设计完整的 KV 缓存管理系统。

| 天 | 任务 | 类型 | 产出 / 文件 |
|----|------|------|-------------|
| D1 | 精读 `kv-memory.md` + `pagedattention-math.md` | ✏️ | → 填充 `notes/kv-cache/concepts.md` |
| D2 | 精读 PagedAttention 论文（vLLM 原始论文） | 📄 | → 填充 `notes/kv-cache/paged-attention.md` |
| D3 | 实现简化版 Paged KV Cache（块分配 + 释放 + 查找） | 🧑‍💻 | → 填充 `src/kv_cache/core.py` |
| D4 | 精读 `kv-eviction-math.md` + H2O / SnapKV 论文 | 📄 | → 填充 `notes/kv-eviction/policies.md` |
| D5 | 实现 LRU + 注意力感知驱逐策略 | 🧑‍💻 | → `src/kv_cache/eviction/` 下实现 |
| D6 | 做 `mock_interview/by-topic/kv-cache-q001` 完整模拟（限时 45min） | 🧪 | 录音 + 评分 |
| D7 | 复盘 Q001 表现，补强薄弱点 | ✏️ | → 填充 `notes/kv-cache/interview-qa.md` |

---

### Week 5: 量化 + 压缩 + 投机解码

**目标**：掌握推理加速的核心技术栈，能做精度-效率权衡分析。

| 天 | 任务 | 类型 | 产出 / 文件 |
|----|------|------|-------------|
| D1 | 精读 `kv-compression-math.md`：GPTQ/AWQ/SmoothQuant | ✏️ | → 填充 `notes/kv-compression/quantization.md` |
| D2 | 理解权重量化 vs KV 量化 vs 激活量化的区别 | ✏️ | → 填充 `notes/kv-compression/interview-qa.md` |
| D3 | 实现简化版量化器（per-channel INT8 + 反量化） | 🧑‍💻 | → `src/kv_cache/compression/` 下实现 |
| D4 | 精读 `speculative-decoding-math.md` + Medusa/EAGLE 论文 | 📄 | → `notes/inference/speculative-decoding.md`（新建） |
| D5 | 精读 `kv-compression-math.md` 稀疏化部分 | ✏️ | → 填充 `notes/kv-compression/sparsity.md` |
| D6 | 做 `mock_interview/by-topic/kv-cache-q003` 模拟 | 🧪 | 录音 + 评分 |
| D7 | 综合复习：KV 全链路（分配 → 使用 → 压缩 → 驱逐 → 回收） | ✏️ | 画完整生命周期图 |

---

### Week 6: Prefill/Decode 优化 + Continuous Batching

**目标**：理解推理服务的核心调度机制，能分析性能瓶颈。

| 天 | 任务 | 类型 | 产出 / 文件 |
|----|------|------|-------------|
| D1 | 精读 `prefill-decode-performance.md` + `attention-complexity.md` | ✏️ | → 填充 `notes/llm-system/throughput-latency.md` |
| D2 | Continuous Batching + Chunked Prefill 深度理解 | ✏️ | → 填充 `notes/llm-system/serving.md` |
| D3 | 精读 `serving-metrics.md` + `queueing-and-slo.md` | ✏️ | → 填充 `notes/llm-system/interview-qa.md` |
| D4 | 实现简化版 Continuous Batching 调度器 | 🧑‍💻 | → `src/simulators/scheduler.py`（新建） |
| D5 | Prefill-Decode 分离架构（Splitwise / DistServe 思路） | ✏️📄 | → `notes/inference/pd-disaggregation.md`（新建） |
| D6 | 做 `mock_interview/by-topic/kv-cache-q004` 模拟 | 🧪 | 录音 + 评分 |
| D7 | Phase 2 全面复习 + 阶段自测 | 🧪 | 综合系统设计题 |

**Phase 2 结束检查清单**：
- [ ] 能手写 Paged KV Cache 的核心数据结构和分配逻辑
- [ ] 能对比 3 种以上驱逐策略的优劣，含公式推导
- [ ] 能解释 GPTQ vs AWQ vs SmoothQuant 的原理和适用场景
- [ ] 能分析 Continuous Batching 在不同负载下的行为
- [ ] 能口述投机解码的接受-拒绝采样过程
- [ ] 4 道 KV Cache 面试题均能在 45min 内完成（≥7 分）

---

## Phase 3: 系统实战（W7-W9）

### Week 7: 推理框架源码精读

**目标**：理解 vLLM / TensorRT-LLM 的核心设计，能在面试中引用源码。

| 天 | 任务 | 类型 | 产出 / 文件 |
|----|------|------|-------------|
| D1 | vLLM 架构总览：Engine → Worker → ModelRunner → Scheduler | 📄 | → `notes/frameworks/vllm-architecture.md`（新建） |
| D2 | vLLM 源码：BlockManager + PagedAttention 实现 | 🧑‍💻 | 阅读源码 + 写注释笔记 |
| D3 | vLLM 源码：Scheduler 调度逻辑（Prefill/Decode 优先级） | 🧑‍💻 | 补充到框架笔记 |
| D4 | TensorRT-LLM / SGLang 架构对比 | ✏️ | → `notes/frameworks/trt-llm-sglang.md`（新建） |
| D5 | vLLM 的 Prefix Caching + Chunked Prefill 实现 | 🧑‍💻 | 补充到框架笔记 |
| D6 | 口述：「vLLM 的调度器是怎么工作的？」 | 🎤 | 限时 5 分钟 |
| D7 | 输出一篇 vLLM 核心流程的精简笔记 | ✏️ | 确保可以面试时快速回忆 |

---

### Week 8: 分布式推理 + CUDA 基础

**目标**：理解多卡推理的通信瓶颈，具备 CUDA 基础认知。

| 天 | 任务 | 类型 | 产出 / 文件 |
|----|------|------|-------------|
| D1 | 精读 `distributed-serving-math.md` | ✏️ | 重新推导每个公式 |
| D2 | TP vs PP 在推理中的延迟分析：什么时候用 TP，什么时候用 PP？ | ✏️ | → `notes/distributed/tp-pp-tradeoff.md`（新建） |
| D3 | MoE Expert Parallelism + All-to-All 通信 | ✏️ | → `notes/distributed/moe-ep.md`（新建） |
| D4 | CUDA 编程基础：thread/block/grid、共享内存、warp | ✏️🧑‍💻 | → `notes/cuda/basics.md`（新建） |
| D5 | 理解 GPU 内存层次：HBM → L2 → SRAM，FlashAttention 为什么用 tiling | ✏️ | → `notes/cuda/memory-hierarchy.md`（新建） |
| D6 | （可选）写一个简单 CUDA kernel：向量加法或矩阵乘法 | 🧑‍💻 | → `src/cuda/` |
| D7 | 口述：「如何为 70B 模型选择并行策略？」 | 🎤 | 限时 5 分钟 |

---

### Week 9: 性能分析 + 容量规划 + 成本优化

**目标**：掌握 Roofline 分析、容量规划和成本效率计算，形成"工程直觉"。

| 天 | 任务 | 类型 | 产出 / 文件 |
|----|------|------|-------------|
| D1 | Roofline 模型实战：A100/H100 上分析 Prefill vs Decode | ✏️ | 用 `attention-complexity.md` 的公式做真实计算 |
| D2 | 容量规划练习：给定 QPS 和 SLO，需要多少卡？ | ✏️ | → `notes/serving/capacity-planning.md`（新建） |
| D3 | 成本分析：GPU 小时成本、$/token、$/query | ✏️ | → `notes/serving/cost-optimization.md`（新建） |
| D4 | 用 `mental-math-cheatsheet.md` 做心算训练（计时） | 🎤 | 目标：30 秒内算出 7B/70B 模型的显存/KV占用 |
| D5 | 性能 Profiling 工具：nsight、torch.profiler 使用 | 🧑‍💻 | → `notes/tools/profiling.md`（新建） |
| D6 | 新增面试题：推理服务系统设计（非 KV 主题） | 🧪 | → `mock_interview/by-topic/serving-q001-system-design.md`（新建） |
| D7 | Phase 3 全面复习 + 阶段自测 | 🧪 | 综合系统设计 |

**Phase 3 结束检查清单**：
- [ ] 能描述 vLLM 从请求到响应的完整流程
- [ ] 能分析 TP/PP/EP 的通信代价并做选择
- [ ] 能在 30 秒内心算模型显存和 KV 缓存占用
- [ ] 能做一个完整的容量规划（从 QPS → GPU 数量）
- [ ] 理解 CUDA 内存层次和 FlashAttention 的 tiling 原理

---

## Phase 4: 综合冲刺（W10-W12）

### Week 10: 系统设计专项

**目标**：能在 45 分钟内完成一道完整的 LLM 推理系统设计题。

| 天 | 任务 | 类型 | 产出 / 文件 |
|----|------|------|-------------|
| D1 | 系统设计方法论：需求分析 → 架构 → 深入 → 权衡 → 监控 | ✏️ | → `notes/system-design/methodology.md`（新建） |
| D2 | 题目：设计一个支持 100K+ context 的在线推理服务 | 🧪 | → `mock_interview/by-topic/serving-q002-long-context.md` |
| D3 | 题目：设计一个多租户 LLM 推理平台（资源隔离 + 公平调度） | 🧪 | → `mock_interview/by-topic/serving-q003-multi-tenant.md` |
| D4 | 题目：设计一个 RAG 系统的推理后端（含缓存策略） | 🧪 | → `mock_interview/by-topic/rag-q001-serving.md` |
| D5 | 题目：如何将推理成本降低 50%？（量化 + 蒸馏 + 调度综合） | 🧪 | → `mock_interview/by-topic/cost-q001-optimization.md` |
| D6 | 回顾所有系统设计题，提炼共性模式 | ✏️ | → `notes/system-design/patterns.md`（新建） |
| D7 | 薄弱环节查漏补缺 | ✏️ | — |

---

### Week 11: 行为面试 + 公司定向

**目标**：准备行为面试 STAR 故事，研究目标公司的技术栈。

| 天 | 任务 | 类型 | 产出 / 文件 |
|----|------|------|-------------|
| D1 | 准备 5 个核心 STAR 故事 | ✏️ | → `mock_interview/behavior/star-stories.md`（新建） |
|    | ① 最有挑战的技术项目 | | |
|    | ② 一次重大技术决策（权衡与结果） | | |
|    | ③ 团队合作/冲突解决 | | |
|    | ④ 在压力/紧急情况下的表现 | | |
|    | ⑤ 一次失败及学到的教训 | | |
| D2 | 研究目标公司 #1（技术博客、开源项目、面经） | ✏️ | → `mock_interview/by-company/company1.md` |
| D3 | 研究目标公司 #2 | ✏️ | → `mock_interview/by-company/company2.md` |
| D4 | 研究目标公司 #3 | ✏️ | → `mock_interview/by-company/company3.md` |
| D5 | 准备「Why this company?」和「项目经历 Deep Dive」回答 | 🎤 | 口述练习 |
| D6 | 完整模拟面试（技术 + 行为，60 分钟） | 🧪 | 录音 + 评分 |
| D7 | 复盘 + 调整最后冲刺策略 | ✏️ | — |

**STAR 故事模板**：
```
Situation: 项目背景、业务约束（1-2 句）
Task:      你的职责、面临的挑战（1-2 句）
Action:    你做了什么（技术决策 + 具体操作，3-5 句，重点）
Result:    量化结果（延迟降低 X%、成本节省 Y%、上线后效果）
```

---

### Week 12: 全真模拟 + 最终打磨

**目标**：以实战状态迎接面试。

| 天 | 任务 | 类型 | 产出 / 文件 |
|----|------|------|-------------|
| D1 | 全真模拟 #1：系统设计（45min） | 🧪 | 评分记录到 `benchmarks/` |
| D2 | 全真模拟 #2：技术深度题（30min × 2） | 🧪 | 评分记录 |
| D3 | 全真模拟 #3：行为面试（30min） | 🧪 | 评分记录 |
| D4 | 薄弱环节集中突破 | ✏️🧑‍💻 | — |
| D5 | 精简所有笔记为面试当天速查版 | ✏️ | → `notes/cheatsheet-final.md`（新建） |
| D6 | 最后一轮全真模拟（找朋友/AI 模拟面试官） | 🧪 | — |
| D7 | 休息 + 心态调整 | 🧘 | — |

---

## 附录 A：新增文件清单（12 周内需创建）

### 笔记 `notes/`
```
notes/
├── architectures/          ← 新建目录
│   ├── llama3.md
│   ├── mixtral-moe.md
│   └── deepseek-v3.md
├── attention/              ← 新建目录
│   ├── flashattention.md
│   └── linear-attention.md
├── inference/              ← 新建目录
│   ├── speculative-decoding.md
│   └── pd-disaggregation.md
├── training/               ← 新建目录
│   ├── scaling-law.md
│   ├── lora-rlhf.md
│   └── alignment-pipeline.md
├── frameworks/             ← 新建目录
│   ├── vllm-architecture.md
│   └── trt-llm-sglang.md
├── distributed/            ← 新建目录
│   ├── tp-pp-tradeoff.md
│   └── moe-ep.md
├── cuda/                   ← 新建目录
│   ├── basics.md
│   └── memory-hierarchy.md
├── serving/                ← 新建目录
│   ├── capacity-planning.md
│   └── cost-optimization.md
├── system-design/          ← 新建目录
│   ├── methodology.md
│   └── patterns.md
├── tools/                  ← 新建目录
│   └── profiling.md
├── kv-cache/               ← 已有，填充内容
├── kv-compression/         ← 已有，填充内容
├── kv-eviction/            ← 已有，填充内容
├── llm-system/             ← 已有，填充内容
└── cheatsheet-final.md     ← W12 最终速查
```

### 代码 `src/`
```
src/
├── attention/              ← 新建目录
│   ├── mha_gqa.py          ← W1: 手写 Multi-Head / GQA Attention
│   ├── rope_rmsnorm.py     ← W1: RoPE + RMSNorm 实现
│   └── flash_attn_sim.py   ← W2: FlashAttention tiling 模拟
├── kv_cache/               ← 已有
│   ├── core.py             ← W4: Paged KV Cache 核心
│   ├── eviction/           ← W4: 驱逐策略实现
│   └── compression/        ← W5: 量化/压缩实现
├── training/               ← 新建目录
│   └── lora.py             ← W3: 简化版 LoRA
├── simulators/             ← 已有
│   └── scheduler.py        ← W6: Continuous Batching 调度器
└── cuda/                   ← 新建目录（可选）
    └── ...
```

### 面试题 `mock_interview/`
```
mock_interview/
├── by-topic/
│   ├── kv-cache-q001~q004  ← 已有
│   ├── serving-q001-system-design.md      ← W9
│   ├── serving-q002-long-context.md       ← W10
│   ├── serving-q003-multi-tenant.md       ← W10
│   ├── rag-q001-serving.md                ← W10
│   └── cost-q001-optimization.md          ← W10
├── behavior/
│   └── star-stories.md                    ← W11
└── by-company/
    ├── company1.md                        ← W11
    ├── company2.md                        ← W11
    └── company3.md                        ← W11
```

---

## 附录 B：推荐阅读清单

### 必读论文（按优先级排序）
| 优先级 | 论文 | 对应周 | 核心要点 |
|--------|------|--------|---------|
| P0 | Attention Is All You Need | W1 | Transformer 基础 |
| P0 | vLLM (PagedAttention) | W4 | 推理内存管理 |
| P0 | FlashAttention 1 & 2 | W2 | IO 感知注意力 |
| P0 | GQA (Ainslie et al.) | W1 | 多查询分组注意力 |
| P1 | RoFormer (RoPE) | W2 | 旋转位置编码 |
| P1 | Chinchilla Scaling Law | W3 | 训练最优配置 |
| P1 | LoRA | W3 | 参数高效微调 |
| P1 | DPO | W3 | 对齐方法 |
| P1 | H2O / SnapKV | W4 | KV 驱逐策略 |
| P1 | GPTQ / AWQ | W5 | 权重量化 |
| P1 | Speculative Decoding (Leviathan et al.) | W5 | 投机解码 |
| P2 | Splitwise / DistServe | W6 | P/D 分离 |
| P2 | Mamba | W2 | 状态空间模型 |
| P2 | Mixtral of Experts | W1 | MoE 路由 |
| P2 | DeepSeek-V3 | W1 | MLA + MoE |

### 推荐博客 / 资料
- [vLLM 官方文档](https://docs.vllm.ai/)
- [NVIDIA TensorRT-LLM 文档](https://nvidia.github.io/TensorRT-LLM/)
- [Jay Alammar - The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Lilian Weng - Large Transformer Model Inference Optimization](https://lilianweng.github.io/)
- [Chip Huyen - Designing Machine Learning Systems](https://huyenchip.com/)

---

## 附录 C：自评追踪模板

建议在 `benchmarks/runs/` 下每周记录：

```markdown
# Week X 自评（日期）

## 模拟面试评分
| 题目 | 得分(1-10) | 薄弱环节 | 改进计划 |
|------|-----------|---------|---------|
| ... | ... | ... | ... |

## 本周完成
- [ ] ...

## 下周重点
- ...

## 心算测试（30 秒内）
- 7B 模型 FP16 权重大小？ →  ___GB（正确：14GB）
- 70B 模型 128K context KV cache (GQA, bf16)？ → ___GB
- H100 bf16 峰值算力？ → ___TFLOPS（正确：990）
```

---

> **最后提醒**：计划的价值不在于完美执行，而在于给你方向感。如果某一周花了更多时间，不要焦虑——深度学通一个主题比赶进度更重要。每周日晚花 15 分钟调整下周计划。加油 💪
