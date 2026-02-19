# 🗓️ 45 天 LLM 面试冲刺每日计划

> **时间跨度**：6.5 周（45 天）
> **每日投入**：工作日 3-4h / 周末 5-6h
> **节奏**：📖 读笔记 → 🧑‍💻 跑 Notebook → 🎤 口述练习 → 🧪 模拟面试
> **核心原则**：每个知识点必须走完 **「理解 → 实现 → 讲解」** 闭环

---

## 📊 总览：6 个阶段

```
阶段         天数        主题                             核心产出
─────────────────────────────────────────────────────────────────
Phase 1     D1-D8      Transformer 架构 + 数学基础       手画架构图，口述 shape 变化
Phase 2     D9-D17     KV Cache + 推理优化               手写 Paged KV Cache，量化实验
Phase 3     D18-D25    Serving 系统 + 框架源码           vLLM 流程讲解，系统设计方法论
Phase 4     D26-D33    分布式 + CUDA + 训练 + 源码       分布式选型分析，源码深读
Phase 5     D34-D40    前沿技术 + RL + 算法编程          PPO/GRPO 实现，LeetCode 刷题
Phase 6     D41-D45    全真模拟 + 终极冲刺               3 场全真模拟，速查表背诵
─────────────────────────────────────────────────────────────────
```

---

## ═══════════════════════════════════════
## Phase 1: Transformer 架构精讲 (D1-D8)
## ═══════════════════════════════════════

> 🎯 目标：能白板画出 Llama3/Mixtral/DeepSeek-V3 完整前向流程，口述每一步 tensor shape

### Day 1 — 数学基础 + Transformer 核心
| 时间 | 任务 | 类型 | 对应文件 |
|------|------|------|---------|
| 30min | 通读 `symbols-glossary.md` + `linear-algebra-basics.md` | 📖 | `math_dictionary/` |
| 60min | 精读 `tensor-shapes.md` + `transformer-attention-math.md` | 📖 | `math_dictionary/` |
| 60min | 精读 `mqa-vs-gqa.md`，手画 MHA→MQA→GQA 对比图 | 📖✏️ | `math_dictionary/` |
| 40min | 🆕 精读 Transformer 组件专题：FFN/残差连接/归一化/稀疏注意力 | 📖 | `notes/architectures/transformer-components.md` |
| 30min | **口述练习**：「请解释 GQA 相比 MHA 的优势」（限时 5min，录音） | 🎤 | — |

**✅ 今日检查**：能默写 Attention 的 shape 流转 + 能画出 Pre-Norm vs Post-Norm 的区别

---

### Day 2 — Llama3 架构拆解 + 位置编码全景
| 时间 | 任务 | 类型 | 对应文件 |
|------|------|------|---------|
| 60min | 精读 Llama3 架构笔记：GQA + RoPE + SwiGLU + RMSNorm | 📖 | `notes/architectures/llama3.md` |
| 40min | 精读 RoPE 数学推导 | 📖 | `math_dictionary/rope-and-position-encoding.md` |
| 50min | 🆕 精读位置编码全景：Sinusoidal / RoPE / **ALiBi** / CoPE 对比 | 📖 | `notes/architectures/position-encoding.md` |
| 20min | 精读 RoPE 扩展：PI、NTK-Aware、YaRN | 📖 | `notes/attention/long-context.md` |
| 30min | **口述练习**：「RoPE 和 ALiBi 的根本区别？为什么 Llama 选 RoPE？」 | 🎤 | — |

**✅ 今日检查**：能手画 Llama3 单层 decoder block 完整流程 + 能对比 RoPE vs ALiBi

---

### Day 3 — Mixtral MoE + DeepSeek-V3 架构
| 时间 | 任务 | 类型 | 对应文件 |
|------|------|------|---------|
| 60min | 精读 Mixtral MoE 笔记：路由、Top-K、负载均衡 | 📖 | `notes/architectures/mixtral-moe.md` |
| 30min | 精读 MoE 路由数学 | 📖 | `math_dictionary/moe-routing-math.md` |
| 60min | 精读 DeepSeek-V3 笔记：MLA + MoE + 辅助损失 | 📖 | `notes/architectures/deepseek-v3.md` |
| 30min | **口述练习**：「对比 Llama3 vs Mixtral vs DeepSeek-V3 三种架构」（5min） | 🎤 | — |

**✅ 今日检查**：能默写 MoE 路由公式 `g(x) = TopK(softmax(W_g · x))`

---

### Day 4 — 手写 Attention + RoPE (Notebook 实战)
| 时间 | 任务 | 类型 | 对应文件 |
|------|------|------|---------|
| 90min | **运行 + 精读** LLM 推理基础 Notebook（MHA/GQA 部分） | 🧑‍💻 | `notebooks/llm_inference_fundamentals.ipynb` |
| 60min | 精读对应源码实现 | 🧑‍💻 | `src/attention/mha_gqa.py` + `src/attention/rope_rmsnorm.py` |
| 30min | 尝试**不看代码**重写 GQA forward pass（限时 20min） | 🧑‍💻 | 白纸/新 notebook |

**✅ 今日检查**：Notebook 全部 cell 运行通过，理解每行代码

---

### Day 5 — FlashAttention + 注意力变体
| 时间 | 任务 | 类型 | 对应文件 |
|------|------|------|---------|
| 60min | 精读 FlashAttention 数学 + 笔记 | 📖 | `math_dictionary/flashattention-math.md` + `notes/attention/flashattention.md` |
| 60min | 精读线性注意力 + 门控注意力 | 📖 | `notes/attention/linear-attention.md` + `math_dictionary/linear-attention-math.md` + `math_dictionary/gated-attention-math.md` |
| 30min | 运行 FlashAttention 模拟代码 | 🧑‍💻 | `src/attention/flash_attn_sim.py` |
| 30min | **口述练习**：「FlashAttention 为什么快？减少了什么？IO 复杂度是多少？」 | 🎤 | — |

**✅ 今日检查**：能画出 FlashAttention 的 tiling 过程示意图

---

### Day 6 — Tokenizer + 解码策略 + LoRA
| 时间 | 任务 | 类型 | 对应文件 |
|------|------|------|---------|
| 30min | 精读 Tokenizer 数学 | 📖 | `math_dictionary/tokenizer-math.md` |
| 30min | 精读概率与采样 | 📖 | `math_dictionary/probability-and-sampling.md` |
| 50min | 🆕 精读解码策略全景：**Top-K / Top-P / Temperature / Beam Search** | 📖 | `notes/inference/decoding-strategies.md` |
| 40min | 精读 LoRA/PEFT 数学 + 笔记 | 📖 | `math_dictionary/lora-peft-math.md` + `notes/training/lora-rlhf.md` |
| 30min | 运行 LoRA + Continuous Batching 部分 | 🧑‍💻 | `notebooks/llm_inference_fundamentals.ipynb`（后半部分） |

**✅ 今日检查**：能解释 Top-K vs Top-P 区别 + LoRA 参数量计算

---

### Day 7 — Attention / Tokenizer / Beam Search 手写强化
| 时间 | 任务 | 类型 | 对应文件 |
|------|------|------|---------|
| 120min | **逐 cell 运行** Attention + BPE Tokenizer + Beam Search Notebook | 🧑‍💻 | `notebooks/attention_tokenizer_beamsearch.ipynb` |
| 60min | **限时练习**：不看代码手写 Scaled Dot-Product Attention（15min 限时） | 🧑‍💻 | 白纸 |
| 30min | **限时练习**：不看代码手写 BPE merge 逻辑（15min 限时） | 🧑‍💻 | 白纸 |

**✅ 今日检查**：3 个手写练习均能在限时内完成

---

### Day 8 — Phase 1 复盘 + 预训练 + Scaling Law + 阶段自测
| 时间 | 任务 | 类型 | 对应文件 |
|------|------|------|---------|
| 40min | 精读 Scaling Law + 优化 | 📖 | `math_dictionary/optimization-and-scaling.md` + `notes/training/scaling-law.md` |
| 50min | 🆕 精读**预训练专题**：CLM/MLM/数据筛选/配比/合成数据 | 📖 | `notes/training/pretraining-data.md` |
| 30min | 精读对齐流程 | 📖 | `notes/training/alignment-pipeline.md` |
| 20min | 复习 D1-D7 全部口述笔记，查漏补缺 | 📖 | 回顾笔记 |
| 60min | **Phase 1 阶段自测**（3 道综合题，限时 45min）| 🧪 | |

> **自测题**：
> 1. 画出 Llama3-70B 完整前向流程，标注每层 tensor shape（15min）
> 2. GQA 的 KV head 数量如何影响显存和质量？给出公式（10min）
> 3. 对比 FlashAttention vs 标准 Attention 的 IO 复杂度（10min）

---

## ═══════════════════════════════════════
## Phase 2: KV Cache + 推理优化 (D9-D17)
## ═══════════════════════════════════════

> 🎯 目标：成为 KV Cache 专家，能设计完整的缓存管理系统

### Day 9 — KV Cache 基础 + PagedAttention
| 时间 | 任务 | 类型 | 对应文件 |
|------|------|------|---------|
| 60min | 精读 KV Cache 概念 + 数学 | 📖 | `math_dictionary/kv-memory.md` + `notes/kv-cache/concepts.md` |
| 60min | 精读 PagedAttention 数学 + 笔记 | 📖 | `math_dictionary/pagedattention-math.md` + `notes/kv-cache/paged-attention.md` |
| 30min | 精读 KV Cache 面试 Q&A | 📖 | `notes/kv-cache/interview-qa.md` |
| 30min | **口述练习**：「PagedAttention 解决了什么问题？核心设计是什么？」 | 🎤 | — |

**✅ 今日检查**：能画出 PagedAttention 的物理块 → 逻辑块映射图

---

### Day 10 — KV Cache 驱逐策略
| 时间 | 任务 | 类型 | 对应文件 |
|------|------|------|---------|
| 60min | 精读 KV 驱逐数学 + 策略笔记 | 📖 | `math_dictionary/kv-eviction-math.md` + `notes/kv-eviction/policies.md` |
| 30min | 精读驱逐面试 Q&A | 📖 | `notes/kv-eviction/interview-qa.md` |
| 60min | 精读源码：LRU / LFU / Fair 驱逐策略实现 | 🧑‍💻 | `src/kv_cache/eviction/policies.py` + `src/kv_cache/core.py` |
| 30min | **口述练习**：「对比 LRU vs LFU vs H2O 三种驱逐策略的优劣」 | 🎤 | — |

**✅ 今日检查**：能手写 LRU 驱逐的核心逻辑（数据结构 + evict 方法）

---

### Day 11 — KV Cache Workshop Notebook (上半)
| 时间 | 任务 | 类型 | 对应文件 |
|------|------|------|---------|
| 120min | **逐 cell 运行** KV Cache Workshop 前半部分 | 🧑‍💻 | `notebooks/kv_cache_paged_lru_workshop.ipynb` |
|        | — Paged 分配实验 | | |
|        | — LRU vs LFU 对比实验 | | |
|        | — 命中率统计 + 可视化 | | |
| 60min | 理解 Adaptive 策略的切换逻辑 | 🧑‍💻 | notebook 后半部分 |

**✅ 今日检查**：能解释 Adaptive 策略在什么条件下从 LRU 切换到 LFU

---

### Day 12 — KV Cache Workshop (下半) + 多租户公平驱逐
| 时间 | 任务 | 类型 | 对应文件 |
|------|------|------|---------|
| 90min | **运行** KV Cache Workshop 下半部分 | 🧑‍💻 | `notebooks/kv_cache_paged_lru_workshop.ipynb` |
|        | — 多租户公平驱逐实验 | | |
|        | — Jain's Fairness Index 计算 | | |
|        | — 容量敏感性实验 | | |
| 30min | **面试模拟**：KV Cache 驱逐系统设计 | 🧪 | `mock_interview/by-topic/kv-cache-q002-eviction-design.md` |
| 60min | 精读面试题解 + 对照自己的答案查漏 | 📖 | `mock_interview/by-topic/kv-cache-q002-eviction-deep-dive.md` |

**✅ 今日检查**：能画出多租户场景下的 quota-aware 驱逐流程图

---

### Day 13 — 量化技术
| 时间 | 任务 | 类型 | 对应文件 |
|------|------|------|---------|
| 60min | 精读 KV 压缩数学 + 量化笔记 | 📖 | `math_dictionary/kv-compression-math.md` + `notes/kv-compression/quantization.md` |
| 30min | 精读稀疏化笔记 | 📖 | `notes/kv-compression/sparsity.md` |
| 30min | 精读压缩面试 Q&A | 📖 | `notes/kv-compression/interview-qa.md` |
| 60min | **逐 cell 运行**量化精度实验 Notebook | 🧑‍💻 | `notebooks/quantization_precision_experiment.ipynb` |
| 30min | **口述练习**：「对比 GPTQ vs AWQ vs SmoothQuant 的原理和适用场景」 | 🎤 | — |

**✅ 今日检查**：能解释 per-channel vs per-tensor 量化的误差差异

---

### Day 14 — 投机解码
| 时间 | 任务 | 类型 | 对应文件 |
|------|------|------|---------|
| 60min | 精读投机解码数学 + 笔记 | 📖 | `math_dictionary/speculative-decoding-math.md` + `notes/inference/speculative-decoding.md` |
| 90min | **逐 cell 运行**投机解码模拟器 Notebook | 🧑‍💻 | `notebooks/speculative_decoding_simulator.ipynb` |
| 30min | **口述练习**：「投机解码的接受-拒绝采样如何保证无损？」 | 🎤 | — |

**✅ 今日检查**：能推导 speculative decoding 的 acceptance probability 公式

---

### Day 15 — Prefill/Decode 优化 + Continuous Batching
| 时间 | 任务 | 类型 | 对应文件 |
|------|------|------|---------|
| 60min | 精读 Prefill-Decode 性能分析 | 📖 | `math_dictionary/prefill-decode-performance.md` + `math_dictionary/attention-complexity.md` |
| 40min | 精读吞吐 vs 延迟笔记 | 📖 | `notes/llm-system/throughput-latency.md` |
| 40min | 精读 Serving 笔记 + 面试 Q&A | 📖 | `notes/llm-system/serving.md` + `notes/llm-system/interview-qa.md` |
| 30min | 精读 P/D 分离架构 | 📖 | `notes/inference/pd-disaggregation.md` |
| 30min | 运行 scheduler 源码 | 🧑‍💻 | `src/simulators/scheduler.py` |

**✅ 今日检查**：能分析 prefill-bound vs decode-bound 的场景差异

---

### Day 16 — KV Cache 面试模拟 (2 道)
| 时间 | 任务 | 类型 | 对应文件 |
|------|------|------|---------|
| 45min | **面试模拟 #1**：KV Cache 压缩量化题（限时 45min） | 🧪 | `mock_interview/by-topic/kv-cache-q003-compression.md` |
| 15min | 对照参考答案查漏 | 📖 | `mock_interview/by-topic/kv-cache-q003-compression-quantization.md` |
| 45min | **面试模拟 #2**：KV Cache 系统设计题（限时 45min） | 🧪 | `mock_interview/by-topic/kv-cache-q004-system-design.md` |
| 15min | 对照参考答案查漏 | 📖 | `mock_interview/by-topic/kv-cache-q004-prefill-decode-pagedattention.md` |
| 30min | 总结错误模式，记录到复盘笔记 | ✏️ | `benchmarks/runs/` |

**✅ 今日检查**：两道题均达 7/10 分以上

---

### Day 17 — Phase 2 复盘 + RAG/Prefix Cache
| 时间 | 任务 | 类型 | 对应文件 |
|------|------|------|---------|
| 90min | **逐 cell 运行** RAG + Prefix Caching 模拟器 | 🧑‍💻 | `notebooks/rag_prefix_caching_simulator.ipynb` |
| 30min | 复习 D9-D16 口述笔记，查漏补缺 | 📖 | — |
| 60min | **Phase 2 阶段自测**（限时 30min） | 🧪 | |

> **自测题**：
> 1. 手写 Paged KV Cache 的 allocate/free 逻辑（10min）
> 2. 画出 KV Cache 全生命周期：分配 → 使用 → 压缩 → 驱逐 → 回收（10min）
> 3. 口述投机解码的完整流程（10min）

---

## ═══════════════════════════════════════
## Phase 3: Serving 系统 + 框架源码 (D18-D25)
## ═══════════════════════════════════════

> 🎯 目标：理解 vLLM/SGLang 核心设计，能做系统设计题

### Day 18 — vLLM 架构总览
| 时间 | 任务 | 类型 | 对应文件 |
|------|------|------|---------|
| 90min | 精读 vLLM 架构笔记（Engine → Worker → Scheduler → BlockManager） | 📖 | `notes/frameworks/vllm-architecture.md` |
| 60min | 精读 TRT-LLM / SGLang 对比笔记 | 📖 | `notes/frameworks/trt-llm-sglang.md` |
| 30min | **口述练习**：「vLLM 的调度器是怎么工作的？从请求到响应的完整流程」 | 🎤 | — |

**✅ 今日检查**：能画出 vLLM 的 Engine → Scheduler → Worker → ModelRunner 调用链

---

### Day 19 — vLLM 核心流程 Notebook
| 时间 | 任务 | 类型 | 对应文件 |
|------|------|------|---------|
| 120min | **逐 cell 运行** vLLM Architecture Walkthrough | 🧑‍💻 | `notebooks/vllm_architecture_walkthrough.ipynb` |
| 30min | 对照笔记，补充理解 preempt/swap 机制 | 📖 | `notes/frameworks/vllm-architecture.md` |
| 30min | **口述练习**：「vLLM 如何处理显存不足时的 preempt？」 | 🎤 | — |

**✅ 今日检查**：能解释 vLLM 的 swap/recompute 两种 preempt 策略的 trade-off

---

### Day 20 — Serving 指标 + 排队论
| 时间 | 任务 | 类型 | 对应文件 |
|------|------|------|---------|
| 40min | 精读 Serving 指标数学 | 📖 | `math_dictionary/serving-metrics.md` |
| 40min | 精读排队论与 SLO | 📖 | `math_dictionary/queueing-and-slo.md` |
| 40min | 精读容量规划笔记 | 📖 | `notes/serving/capacity-planning.md` |
| 40min | 精读成本优化笔记 | 📖 | `notes/serving/cost-optimization.md` |
| 30min | 精读心算速查表（计时训练：30 秒内算出 7B/70B 显存） | 🎤 | `math_dictionary/mental-math-cheatsheet.md` |

**✅ 今日检查**：能在 30 秒内心算 70B 模型 FP16 的显存占用（140GB）和 128K KV 占用

---

### Day 21 — 系统设计方法论 + 模式
| 时间 | 任务 | 类型 | 对应文件 |
|------|------|------|---------|
| 60min | 精读系统设计方法论 | 📖 | `notes/system-design/methodology.md` |
| 60min | 精读系统设计模式 | 📖 | `notes/system-design/patterns.md` |
| 30min | 精读 Profiling 工具笔记 | 📖 | `notes/tools/profiling.md` |
| 30min | **口述练习**：用 5 步法分析一个系统设计题（需求→架构→深入→权衡→监控） | 🎤 | — |

**✅ 今日检查**：能默写系统设计的 5 步方法论框架

---

### Day 22 — Serving 系统设计面试模拟 (2 道)
| 时间 | 任务 | 类型 | 对应文件 |
|------|------|------|---------|
| 45min | **面试模拟 #1**：设计 LLM 推理系统 | 🧪 | `mock_interview/by-topic/serving-q001-system-design.md` |
| 45min | **面试模拟 #2**：支持 100K+ context 的在线推理 | 🧪 | `mock_interview/by-topic/serving-q002-long-context.md` |
| 30min | 精读 100 QPS LLM Serving 系统设计 mock 脚本 | 📖 | `mock_interview/by-topic/system-design-100qps-llm-serving.md` |
| 30min | 总结系统设计中的共性错误模式 | ✏️ | `benchmarks/runs/` |

**✅ 今日检查**：两道题的系统架构图画得完整且有层次

---

### Day 23 — 更多系统设计 + RAG/多租户
| 时间 | 任务 | 类型 | 对应文件 |
|------|------|------|---------|
| 45min | **面试模拟 #3**：多租户 LLM 推理平台 | 🧪 | `mock_interview/by-topic/serving-q003-multi-tenant.md` |
| 45min | **面试模拟 #4**：RAG 推理后端设计 | 🧪 | `mock_interview/by-topic/rag-q001-serving.md` |
| 45min | **面试模拟 #5**：成本优化 50% | 🧪 | `mock_interview/by-topic/cost-q001-optimization.md` |
| 30min | 复盘 5 道系统设计题，提炼共性模式 | ✏️ | — |

**✅ 今日检查**：能在 45min 内完成一道完整系统设计题且评分 ≥ 7/10

---

### Day 24 — KV Cache 系统设计面试模拟
| 时间 | 任务 | 类型 | 对应文件 |
|------|------|------|---------|
| 45min | **面试模拟 #6**：KV Cache 系统设计 | 🧪 | `mock_interview/by-topic/kv-cache-q001-system-design.md` |
| 60min | 精读 Slime vs verl 笔记 | 📖 | `notes/frontier/slime-vs-verl.md` |
| 60min | **口述练习**：「为什么 Slime 选 SGLang 而不是 vLLM？」 | 🎤 | — |

**✅ 今日检查**：能清晰解释 RadixAttention 在 RL 场景下 prefix cache 效率更高的原因

---

### Day 25 — Phase 3 复盘 + 评估指标
| 时间 | 任务 | 类型 | 对应文件 |
|------|------|------|---------|
| 40min | 精读评估指标数学 | 📖 | `math_dictionary/evaluation-metrics.md` |
| 30min | 复习 D18-D24 全部口述笔记 | 📖 | — |
| 60min | **Phase 3 阶段自测**（限时 40min） | 🧪 | |
| 30min | 记录分数到 benchmarks，制定薄弱点攻克计划 | ✏️ | `benchmarks/runs/` |

> **自测题**：
> 1. 描述 vLLM 从请求到响应的完整流程（10min）
> 2. 设计一个支持 100 并发的 LLM Serving 系统——画架构图 + 列 trade-off（20min）
> 3. 心算：70B 模型需要多少张 A100 80GB？（2min）

---

## ═══════════════════════════════════════
## Phase 4: 分布式 + CUDA + 训练 + 源码 (D26-D33)
## ═══════════════════════════════════════

> 🎯 目标：理解多卡推理通信瓶颈，掌握 CUDA 基础认知，深入源码

### Day 26 — 分布式推理：TP / PP / EP
| 时间 | 任务 | 类型 | 对应文件 |
|------|------|------|---------|
| 60min | 精读分布式 Serving 数学 | 📖 | `math_dictionary/distributed-serving-math.md` |
| 60min | 精读 TP vs PP trade-off 笔记 | 📖 | `notes/distributed/tp-pp-tradeoff.md` |
| 40min | 精读 MoE EP 笔记 | 📖 | `notes/distributed/moe-ep.md` |
| 30min | **口述练习**：「如何为 70B 模型选择并行策略？TP vs PP 的 trade-off？」 | 🎤 | — |

**✅ 今日检查**：能推导 TP 的通信量公式并解释什么时候通信成为瓶颈

---

### Day 27 — 分布式推理 Notebook + MoE 深度
| 时间 | 任务 | 类型 | 对应文件 |
|------|------|------|---------|
| 90min | **逐 cell 运行**分布式推理 Roofline Notebook | 🧑‍💻 | `notebooks/distributed_inference_roofline.ipynb` |
| 60min | 精读 MoE 推理深度笔记 | 📖 | `notes/distributed/moe-inference-deep.md` |
| 30min | **口述练习**：「Roofline 模型怎么分析 Prefill vs Decode？」 | 🎤 | — |

**✅ 今日检查**：能画 Roofline 图并标出 prefill 和 decode 的位置

---

### Day 28 — CUDA 基础 + 内存层次
| 时间 | 任务 | 类型 | 对应文件 |
|------|------|------|---------|
| 60min | 精读 CUDA 基础笔记 | 📖 | `notes/cuda/basics.md` |
| 60min | 精读 GPU 内存层次笔记 | 📖 | `notes/cuda/memory-hierarchy.md` |
| 40min | 阅读 CUDA kernel 模拟代码 | 🧑‍💻 | `src/cuda/simulation.py` |
| 30min | **口述练习**：「GPU 内存层次是什么？FlashAttention 为什么用 tiling？」 | 🎤 | — |

**✅ 今日检查**：能画出 HBM → L2 → SRAM 的层次图并标注延迟和带宽

---

### Day 29 — 源码深读 Level 1：架构概览
| 时间 | 任务 | 类型 | 对应文件 |
|------|------|------|---------|
| 90min | 精读源码深入理解路线图（L1-L4 全部） | 📖 | `notes/source-reading/slime-sglang-verl-deep-dive.md` |
| 40min | 精读 Slime vs verl 对比笔记（含 SGLang vs vLLM 详解） | 📖 | `notes/frontier/slime-vs-verl.md` |
| 40min | 手写 RL Training Loop 伪代码（对标 nano-L1） | 🧑‍💻 | 白纸 |
| 20min | 列出 SGLang RadixAttention 的 4 个核心操作 | ✏️ | — |

**✅ 今日检查**：能画出 Slime 三大模块交互图 + verl HybridFlow 时序图

---

### Day 30 — 源码深读 Level 2：核心模块
| 时间 | 任务 | 类型 | 对应文件 |
|------|------|------|---------|
| 60min | 深度理解 RadixAttention 内部结构（TreeNode, match_prefix, insert, evict） | 📖 | `notes/source-reading/slime-sglang-verl-deep-dive.md`（L2 部分） |
| 60min | 深度理解 verl Rollout Worker + PPO Trainer 源码结构 | 📖 | 同上 |
| 60min | 模拟实现简化版 RadixTree（对标 nano-L2a） | 🧑‍💻 | 白纸/notebook |

**✅ 今日检查**：能手写 RadixTree 的 match_prefix + insert 两个核心方法

---

### Day 31 — 源码深读 Level 3：分布式与异步
| 时间 | 任务 | 类型 | 对应文件 |
|------|------|------|---------|
| 60min | 深度理解 Slime 异步 RL 数据流 + staleness 问题 | 📖 | `notes/source-reading/slime-sglang-verl-deep-dive.md`（L3 部分） |
| 60min | 深度理解 verl HybridFlow 模型 reshard 机制 | 📖 | 同上 |
| 40min | 做显存估算练习（7B, G=16, batch=100） | ✏️ | 同上（L4 部分） |
| 30min | **口述练习**：「同步 vs 异步 RL 训练的 trade-off 是什么？」 | 🎤 | — |

**✅ 今日检查**：能计算有/无 RadixAttention 时 KV Cache 显存差异（~47% 节省）

---

### Day 32 — Nano 项目路线图 + 训练
| 时间 | 任务 | 类型 | 对应文件 |
|------|------|------|---------|
| 60min | 通读 Nano 项目路线图，理解面试覆盖矩阵 | 📖 | `notes/nano-projects/roadmap.md` |
| 40min | 精读 RLHF/对齐数学 | 📖 | `math_dictionary/rlhf-alignment-math.md` |
| 40min | 复习训练相关笔记 | 📖 | `notes/training/alignment-pipeline.md` |
| 30min | **口述练习**：面试"三段式"展示策略排练 | 🎤 | — |
| 30min | 运行单元测试确保所有 src 代码正常 | 🧑‍💻 | `tests/` |

**✅ 今日检查**：能用 Level 1→2→3 递进法展示一个技术点

---

### Day 33 — Phase 4 复盘 + 阶段自测
| 时间 | 任务 | 类型 | 对应文件 |
|------|------|------|---------|
| 30min | 复习 D26-D32 全部口述笔记 | 📖 | — |
| 60min | **Phase 4 阶段自测**（限时 40min） | 🧪 | |
| 30min | 记录分数，识别最薄弱的 2-3 个点 | ✏️ | `benchmarks/runs/` |
| 60min | 针对薄弱点查漏补缺（重读对应笔记/Notebook） | 📖🧑‍💻 | — |

> **自测题**：
> 1. 分析 TP/PP/EP 的通信代价并做选择——给定 8×H100，部署 70B MoE（15min）
> 2. 画出 Slime 异步 RL 的完整数据流图（10min）
> 3. 用 Radix Tree 演示同一 prompt 生成 G=16 个 response 的 cache 复用过程（10min）

---

## ═══════════════════════════════════════
## Phase 5: 前沿技术 + RL + 算法编程 (D34-D40)
## ═══════════════════════════════════════

> 🎯 目标：覆盖前沿热点，PPO/GRPO 实现，LeetCode 高频题

### Day 34 — 前沿：Reasoning Models + Test-Time Compute
| 时间 | 任务 | 类型 | 对应文件 |
|------|------|------|---------|
| 60min | 精读 Reasoning Models 笔记（o1/DeepSeek-R1, Thinking Tokens） | 📖 | `notes/frontier/reasoning-models.md` |
| 40min | 精读 Test-Time Compute Scaling | 📖 | `notes/frontier/test-time-compute.md` |
| 60min | **逐 cell 运行** Reasoning Models Workshop | 🧑‍💻 | `notebooks/reasoning_models_workshop.ipynb` |
| 30min | **口述练习**：「DeepSeek-R1 的 GRPO 是怎么工作的？和 PPO 有什么区别？」 | 🎤 | — |

**✅ 今日检查**：能解释 GRPO 不需要 Critic 的原因及 group-relative advantage 的计算

---

### Day 35 — 前沿：DeepSeek + Qwen + Kimi + 发展脉络
| 时间 | 任务 | 类型 | 对应文件 |
|------|------|------|---------|
| 40min | 精读 DeepSeek-V3/R1 笔记 | 📖 | `notes/frontier/deepseek-v3-r1.md` |
| 30min | 精读 Qwen 系列笔记 | 📖 | `notes/frontier/qwen-series.md` |
| 30min | 精读 Kimi/Moonshot 笔记 | 📖 | `notes/frontier/kimi-moonshot.md` |
| 20min | 精读 GLM/智谱笔记 | 📖 | `notes/frontier/glm-zhipu.md` |
| 20min | 精读 MiniMax-01 笔记 | 📖 | `notes/frontier/minimax-01.md` |
| 30min | 🆕 精读**主流大模型发展脉络**：技术路线 + 核心优化点对比 | 📖 | `notes/frontier/model-evolution-timeline.md` |
| 20min | **口述练习**：「对比 DeepSeek-V3 vs Qwen2.5 vs MiniMax-01 的架构差异」 | 🎤 | — |

**✅ 今日检查**：能列出每家公司的差异化技术（各 3 个关键词）

---

### Day 36 — 后训练高级 + 评估 + SSM/Hybrid + 面试模拟
| 时间 | 任务 | 类型 | 对应文件 |
|------|------|------|---------|
| 40min | 🆕 精读**后训练高级方法**：RLAIF/SimPO/KTO/ORPO 对比 | 📖 | `notes/training/post-training-advanced.md` |
| 40min | 🆕 精读**模型评估 + 幻觉检测**：Benchmark/评估方法/幻觉缓解 | 📖 | `notes/evaluation/benchmarks-hallucination.md` |
| 40min | 🆕 精读 **SSM + Hybrid 架构**：Mamba-2/RWKV-6/Jamba/MiniMax | 📖 | `notes/architectures/ssm-hybrid.md` |
| 20min | 🆕 精读 **MoE 训练策略**：负载均衡/Upcycling/EP | 📖 | `notes/architectures/moe-training.md` |
| 30min | 精读结构化输出 + VLM Serving | 📖 | `notes/frontier/structured-output.md` + `notes/multimodal/vlm-serving.md` |
| 30min | **面试模拟**：Reasoning Models 题 | 🧪 | `mock_interview/by-topic/frontier-q001-reasoning-models.md` |

**✅ 今日检查**：能对比 SimPO vs DPO；能解释 Mamba 的选择性机制

---

### Day 37 — PPO / GRPO 手写实现
| 时间 | 任务 | 类型 | 对应文件 |
|------|------|------|---------|
| 120min | **逐 cell 运行** PPO/GRPO Notebook（完整训练循环） | 🧑‍💻 | `notebooks/rl_ppo_grpo_implementation.ipynb` |
| 60min | **限时手写**：不看代码重写 PPO loss 计算（20min 限时） | 🧑‍💻 | 白纸 |
| 30min | **口述练习**：「PPO 的 clipped surrogate objective 是什么？为什么要 clip？」 | 🎤 | — |

**✅ 今日检查**：能默写 PPO loss 公式 `L = min(r·A, clip(r, 1-ε, 1+ε)·A)`

---

### Day 38 — LeetCode 高频题 (上)
| 时间 | 任务 | 类型 | 对应文件 |
|------|------|------|---------|
| 150min | **逐 cell 运行** LeetCode Notebook（前半部分） | 🧑‍💻 | `notebooks/leetcode_llm_system_related.ipynb` |
|         | — LRU Cache (O(1) 实现) | | |
|         | — Top-K 频率统计（堆） | | |
|         | — 滑动窗口最大值 | | |
|         | — 前缀树 (Trie) | | |
| 30min | 精读算法编程笔记 | 📖 | `notes/coding/llm-related-algorithms.md` |

**✅ 今日检查**：LRU Cache 和 Trie 能在 15min 内手写完成

---

### Day 39 — LeetCode 高频题 (下) + Beam Search
| 时间 | 任务 | 类型 | 对应文件 |
|------|------|------|---------|
| 90min | **逐 cell 运行** LeetCode Notebook（后半部分） | 🧑‍💻 | `notebooks/leetcode_llm_system_related.ipynb` |
|         | — 一致性哈希 | | |
|         | — 生产者-消费者队列 | | |
|         | — 令牌桶限流 | | |
| 60min | **限时练习**：手写 Beam Search（不看代码，20min 限时） | 🧑‍💻 | 回顾 `notebooks/attention_tokenizer_beamsearch.ipynb` |
| 30min | **限时练习**：手写 Top-K Sampling（15min 限时） | 🧑‍💻 | — |

**✅ 今日检查**：所有 LeetCode 题目 Notebook 运行通过且理解核心思路

---

### Day 40 — Phase 5 复盘 + 心算训练
| 时间 | 任务 | 类型 | 对应文件 |
|------|------|------|---------|
| 40min | 精读速查表（最终版） | 📖 | `notes/cheatsheet-final.md` |
| 40min | 心算训练（30 秒挑战 × 20 题） | 🎤 | `math_dictionary/mental-math-cheatsheet.md` |
| 40min | 复习 D34-D39 口述笔记 | 📖 | — |
| 60min | **Phase 5 阶段自测**（限时 30min） | 🧪 | |

> **自测题**：
> 1. 手写 LRU Cache 的 get/put（限时 10min）
> 2. 解释 GRPO 和 PPO 的区别（口述 3min）
> 3. 列出 DeepSeek-V3 的 3 个核心创新点（2min）

---

## ═══════════════════════════════════════
## Phase 6: 全真模拟 + 终极冲刺 (D41-D45)
## ═══════════════════════════════════════

> 🎯 目标：以实战状态迎接面试

### Day 41 — 行为面试准备
| 时间 | 任务 | 类型 | 对应文件 |
|------|------|------|---------|
| 40min | 精读自我介绍 | 📖🎤 | `mock_interview/behavior/self-introduction.md` |
| 60min | 精读 + 口述 STAR 故事（5 个核心故事） | 📖🎤 | `mock_interview/behavior/star-stories.md` |
| 40min | 精读 + 口述 KV Cache/Serving 专项 STAR | 📖🎤 | `mock_interview/behavior/star-kv-cache-serving.md` |
| 40min | 精读项目一页纸（1/3/5 分钟版本） | 📖🎤 | `mock_interview/behavior/project-one-pager.md` |
| 30min | 练习：分别用 1min / 3min / 5min 讲述项目（录音） | 🎤 | — |

**✅ 今日检查**：自我介绍 ≤ 2min，每个 STAR 故事 ≤ 3min，流畅无卡顿

---

### Day 42 — 公司定向准备
| 时间 | 任务 | 类型 | 对应文件 |
|------|------|------|---------|
| 60min | 精读字节跳动面试准备 | 📖 | `mock_interview/by-company/bytedance.md` |
| 60min | 精读腾讯面试准备 | 📖 | `mock_interview/by-company/tencent.md` |
| 60min | 精读阿里巴巴面试准备 | 📖 | `mock_interview/by-company/alibaba.md` |
| 30min | 针对目标公司的「Why this company?」口述练习 | 🎤 | — |

**✅ 今日检查**：能针对每家公司说出 3 个技术匹配点 + 1 个业务理解点

---

### Day 43 — 全真模拟 #1：技术深度 + 系统设计（60min）
| 时间 | 任务 | 类型 | 对应文件 |
|------|------|------|---------|
| 15min | **模拟第 1 轮**：自我介绍 + 项目 Deep Dive | 🧪 | 录音 |
| 45min | **模拟第 2 轮**：系统设计——设计支持 100 并发的 LLM Serving | 🧪 | `mock_interview/by-topic/system-design-100qps-llm-serving.md` |
| 30min | **模拟第 3 轮**：技术深度追问（随机 3 题） | 🧪 | 随机选取 by-topic |
| 30min | 复盘：评分（1-10），记录薄弱点 | ✏️ | `benchmarks/runs/` |
| 60min | **薄弱点集中突破**（重读对应笔记） | 📖 | — |

---

### Day 44 — 全真模拟 #2：算法 + 行为面试（60min）
| 时间 | 任务 | 类型 | 对应文件 |
|------|------|------|---------|
| 30min | **模拟第 1 轮**：手撕算法（LRU Cache + Top-K，限时 30min） | 🧪 | — |
| 30min | **模拟第 2 轮**：行为面试 STAR（随机 3 个故事） | 🧪 | 录音 |
| 30min | **模拟第 3 轮**：口述 Attention 原理 + 手写 Scaled Dot-Product | 🧪 | — |
| 30min | 复盘：评分，最后查漏补缺 | ✏️ | `benchmarks/runs/` |
| 60min | 精简所有笔记为面试当天速查版（最后通读） | 📖 | `notes/cheatsheet-final.md` |

---

### Day 45 — 终极检查 + 休息调整
| 时间 | 任务 | 类型 | 对应文件 |
|------|------|------|---------|
| 60min | **终极速查**：快速过一遍所有关键公式/架构图 | 📖 | `notes/cheatsheet-final.md` + `math_dictionary/mental-math-cheatsheet.md` |
| 30min | **终极口述**：随机抽 5 个知识点，每个 3min 口述 | 🎤 | — |
| 30min | 检查 Notebook 清单（确保每个都能快速打开演示） | 🧑‍💻 | `notebooks/` |
| — | 🧘 **休息 + 心态调整** | 🧘 | — |

---

## 📋 附录 A：每日学习材料完整映射

### 📓 Notebooks 学习时间线

| 天 | Notebook | 预计时间 |
|----|----------|---------|
| D4 | `llm_inference_fundamentals.ipynb` | 90min |
| D7 | `attention_tokenizer_beamsearch.ipynb` | 120min |
| D11-12 | `kv_cache_paged_lru_workshop.ipynb` | 180min |
| D13 | `quantization_precision_experiment.ipynb` | 60min |
| D14 | `speculative_decoding_simulator.ipynb` | 90min |
| D17 | `rag_prefix_caching_simulator.ipynb` | 90min |
| D19 | `vllm_architecture_walkthrough.ipynb` | 120min |
| D27 | `distributed_inference_roofline.ipynb` | 90min |
| D34 | `reasoning_models_workshop.ipynb` | 60min |
| D37 | `rl_ppo_grpo_implementation.ipynb` | 120min |
| D38-39 | `leetcode_llm_system_related.ipynb` | 240min |

### 📖 Notes 学习时间线

| 天 | 笔记文件夹/文件 |
|----|----------------|
| D1 | 🆕 `notes/architectures/transformer-components.md` (FFN/残差/LN/稀疏注意力) |
| D2 | 🆕 `notes/architectures/position-encoding.md` (RoPE/ALiBi/Sinusoidal 全景) |
| D2-3 | `notes/architectures/` (llama3, mixtral-moe, deepseek-v3) |
| D2,5 | `notes/attention/` (flashattention, linear-attention, long-context) |
| D6 | 🆕 `notes/inference/decoding-strategies.md` (Top-K/P/Temperature/Beam) |
| D6,8 | `notes/training/` (scaling-law, lora-rlhf, alignment-pipeline) |
| D8 | 🆕 `notes/training/pretraining-data.md` (预训练任务/数据筛选/合成数据) |
| D9-10 | `notes/kv-cache/`, `notes/kv-eviction/` |
| D13 | `notes/kv-compression/` |
| D15 | `notes/llm-system/`, `notes/inference/` |
| D18 | `notes/frameworks/` |
| D20-21 | `notes/serving/`, `notes/system-design/`, `notes/tools/` |
| D26-28 | `notes/distributed/`, `notes/cuda/` |
| D29-31 | `notes/source-reading/`, `notes/frontier/slime-vs-verl.md` |
| D34-36 | `notes/frontier/` (全部 10 篇), `notes/multimodal/`, `notes/debugging/` |
| D36 | 🆕 `notes/training/post-training-advanced.md` (RLAIF/SimPO/KTO/ORPO) |
| D36 | 🆕 `notes/evaluation/benchmarks-hallucination.md` (评估+幻觉检测) |
| D36 | 🆕 `notes/architectures/ssm-hybrid.md` (Mamba-2/RWKV-6/Hybrid 深度) |
| D36 | 🆕 `notes/architectures/moe-training.md` (MoE 训练策略深度) |
| D35 | 🆕 `notes/frontier/model-evolution-timeline.md` (主流大模型发展脉络) |
| D38 | `notes/coding/` |

### 🧪 面试模拟时间线

| 天 | 模拟题目 | 来源 |
|----|---------|------|
| D12 | KV Cache 驱逐设计 | `kv-cache-q002-*` |
| D16 | KV Cache 压缩 + 系统设计 (2道) | `kv-cache-q003/q004-*` |
| D22 | Serving 系统设计 + 长上下文 (2道) | `serving-q001/q002-*` |
| D23 | 多租户 + RAG + 成本优化 (3道) | `serving-q003, rag-q001, cost-q001` |
| D24 | KV Cache 系统设计 | `kv-cache-q001-*` |
| D36 | 前沿技术 (3道) | `frontier-q001/q002/q003-*` |
| D43 | 全真模拟 #1：系统设计全流程 | 综合 |
| D44 | 全真模拟 #2：算法 + 行为 | 综合 |

### 📐 数学字典学习时间线

| 天 | 文件 |
|----|------|
| D1 | `symbols-glossary`, `linear-algebra-basics`, `tensor-shapes`, `transformer-attention-math`, `mqa-vs-gqa` |
| D2 | `rope-and-position-encoding` |
| D3 | `moe-routing-math` |
| D5 | `flashattention-math`, `linear-attention-math`, `gated-attention-math` |
| D6 | `tokenizer-math`, `probability-and-sampling`, `lora-peft-math` |
| D8 | `optimization-and-scaling` |
| D9 | `kv-memory`, `pagedattention-math` |
| D10 | `kv-eviction-math` |
| D13 | `kv-compression-math` |
| D14 | `speculative-decoding-math` |
| D15 | `prefill-decode-performance`, `attention-complexity` |
| D20 | `serving-metrics`, `queueing-and-slo`, `mental-math-cheatsheet` |
| D25 | `evaluation-metrics` |
| D26 | `distributed-serving-math` |
| D32 | `rlhf-alignment-math` |

---

## 📋 附录 B：阶段检查清单

### Phase 1 结束 (D8) ✅
- [ ] 能画出 Llama3 完整架构图（含参数量标注）
- [ ] 能手写 GQA Attention 的 PyTorch 实现
- [ ] 能口述 RoPE、FlashAttention、GQA 的核心优势
- [ ] 能解释 Scaling Law 对模型大小和数据量的指导意义
- [ ] BPE Tokenizer 手写通过

### Phase 2 结束 (D17) ✅
- [ ] 能手写 Paged KV Cache 核心数据结构和分配逻辑
- [ ] 能对比 3 种以上驱逐策略的优劣，含公式
- [ ] 能解释 GPTQ vs AWQ vs SmoothQuant 的原理
- [ ] 能口述投机解码的接受-拒绝采样过程
- [ ] 4 道 KV Cache 面试题均 ≥ 7/10 分

### Phase 3 结束 (D25) ✅
- [ ] 能描述 vLLM 从请求到响应的完整流程
- [ ] 能在 30 秒内心算模型显存和 KV 占用
- [ ] 能在 45min 内完成完整系统设计题
- [ ] 5 道 Serving 面试题均 ≥ 7/10 分

### Phase 4 结束 (D33) ✅
- [ ] 能分析 TP/PP/EP 的通信代价并做选择
- [ ] 理解 CUDA 内存层次和 tiling 原理
- [ ] 能画出 Slime 和 verl 的完整架构图
- [ ] 能解释 RadixAttention 在 RL 场景下的优势
- [ ] 能用"三段式"展示一个技术点

### Phase 5 结束 (D40) ✅
- [ ] 能解释 DeepSeek-R1 / Qwen / Kimi 的技术差异
- [ ] PPO loss 公式能默写
- [ ] LRU Cache 15min 内手写完成
- [ ] 前沿面试题 ≥ 7/10 分

### Phase 6 结束 (D45) ✅
- [ ] 自我介绍 ≤ 2min 流畅
- [ ] STAR 故事 ≤ 3min 流畅
- [ ] 全真模拟 ≥ 7/10 分
- [ ] 所有 Notebook 能快速打开演示

---

## 📋 附录 C：心算速查 20 题（D40 + D45 使用）

| # | 题目 | 答案 | 限时 |
|---|------|------|------|
| 1 | 7B 模型 FP16 权重大小？ | 14 GB | 5s |
| 2 | 70B 模型 FP16 权重大小？ | 140 GB | 5s |
| 3 | 70B 模型需要几张 A100-80GB？ | 2 张 | 5s |
| 4 | 70B 模型 INT4 需要几张？ | 1 张（35GB） | 10s |
| 5 | H100 BF16 峰值算力？ | 990 TFLOPS | 5s |
| 6 | A100 HBM 带宽？ | 2 TB/s | 5s |
| 7 | H100 HBM 带宽？ | 3.35 TB/s | 5s |
| 8 | Llama3-70B GQA KV head 数？ | 8 | 5s |
| 9 | 70B 32层 128K context KV cache (GQA, bf16)？ | ~40 GB | 15s |
| 10 | FlashAttention IO 复杂度？ | O(N²d/M) | 5s |
| 11 | 标准 Attention IO 复杂度？ | O(N² + Nd) | 5s |
| 12 | RoPE 需要额外参数吗？ | 不需要 | 3s |
| 13 | LoRA rank=16, d=4096 的额外参数量？ | 2×16×4096 = 131K | 10s |
| 14 | Mixtral 的 MoE 有几个 expert？ | 8个，Top-2 路由 | 5s |
| 15 | DeepSeek-V3 总参数量？ | 671B (37B 激活) | 5s |
| 16 | 1M token 对话的 KV cache 大概多大？ | ~百 GB 级 | 10s |
| 17 | Speculative Decoding 的 draft 模型通常多大？ | 原模型 1/10~1/5 | 10s |
| 18 | INT8 量化相比 FP16 理论加速？ | 2× | 5s |
| 19 | TP=4 时 AllReduce 通信量？ | 2(N-1)/N × data_size | 10s |
| 20 | PPO clip ratio ε 的典型值？ | 0.2 | 5s |

---

## 📋 附录 D：每日时间模板

### 工作日（3-4h）
```
┌─────────────────────────────────────────┐
│ 🌅 早上 30min：复习昨日公式 + 口述练习  │
│    （用 mental-math-cheatsheet 热身）     │
│                                          │
│ 🌆 晚上 Session 1 (60-90min)：           │
│    主题深度学习                           │
│    （读笔记 → 跑 Notebook → 写代码）     │
│                                          │
│ 🌃 晚上 Session 2 (60min)：              │
│    面试题模拟 / 口述练习 / 录音回听       │
│                                          │
│ 🛏️ 睡前 15min：回顾今日要点             │
└─────────────────────────────────────────┘
```

### 周末（5-6h/天）
```
┌─────────────────────────────────────────┐
│ 🌅 上午 (2-3h)：                         │
│    论文/源码精读 + Notebook 实战          │
│                                          │
│ 🌆 下午 (2-3h)：                         │
│    系统设计模拟 + 口述练习               │
│    回顾薄弱环节                          │
└─────────────────────────────────────────┘
```

---

> 💪 **最后提醒**：
> - 计划的价值不在于完美执行，而在于给你方向感
> - 如果某一天学得更深入花了更多时间，不要焦虑——**深度学通 > 赶进度**
> - 每个 Phase 结束时花 20min 调整下一阶段计划
> - 口述练习和面试模拟是最容易被跳过的，但**它们是面试通过率最关键的变量**
> - 遇到不懂的知识点，**先搜索本仓库**——大概率已经有对应笔记
> - 祝你 🏆 **OFFER GET!**
