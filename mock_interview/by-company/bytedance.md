# 字节跳动（ByteDance）— LLM 推理工程师面试定向准备

> 目标岗位：LLM Inference / Serving Engineer
> 相关团队：AML（Applied Machine Learning）、Seed、豆包大模型团队、火山引擎

---

## 一、公司技术栈与核心方向

### 1. 核心产品
| 产品 | 技术侧重 |
|------|----------|
| **豆包（Doubao）** | 多模态大模型推理，日活亿级，对延迟和成本极度敏感 |
| **火山引擎·方舟** | 大模型推理云服务（MaaS），多租户、弹性扩缩容 |
| **Coze** | Agent 平台，大量多轮对话 + 工具调用场景 |
| **抖音内嵌 AI** | 推荐系统中的 LLM 应用，QPS 极高 |

### 2. 推理技术栈
- **MegaScale**：字节自研大规模训练/推理基础设施
- **FlashDecoding**：Decode 阶段的 CUDA kernel 优化（基于 FlashAttention 理念）
- **连续批处理（Continuous Batching）**：类 vLLM/Orca 的调度框架
- **多模态推理**：Vision-Language Model 的 KV Cache 管理（图像 token 很长）
- **LoRA 推理**：在同一个 base model 上高效切换多个 LoRA adapter
- **投机解码（Speculative Decoding）**：小模型 draft + 大模型 verify

### 3. 核心技术关注点
1. **极致成本优化**：字节体量大 → 每个百分点的成本降低 = 数百万美元
2. **长上下文推理**：豆包支持 128K+ context，KV Cache 管理是关键
3. **多模态 KV Cache**：图像 encoder 产生的 KV 如何与 text KV 共存
4. **LoRA 推理效率**：多 adapter 场景下 base weight + adapter 的内存管理
5. **大规模分布式推理**：MoE 模型（如 DeepSeek-V3 类）的 EP + TP 策略

---

## 二、高频面试题（8 道）

### Q1: 系统设计 — 设计豆包的推理后端
**题目**：设计一个支持 10 万 QPS 的多模态大模型推理服务（文本 + 图片输入）

**考察要点**：
- 请求路由与负载均衡
- Prefill / Decode 分离架构（Splitwise/DistServe 思路）
- KV Cache 内存管理（Paged Allocation + 驱逐）
- 多模态 token 的 KV Cache 处理（图像 token 通常很长但不需要驱逐）
- 弹性扩缩容策略

**回答框架**：
```
1. 需求分析：QPS、SLO（TTFT < 500ms, TPOT < 50ms）、多模态比例
2. 架构：Router → Prefill Worker Pool → Decode Worker Pool → KV Transfer
3. KV 管理：PagedAttention + Prefix Caching + 图像 KV 固定不驱逐
4. 扩缩容：基于 pending queue length 的 HPA
5. 监控：per-request latency percentiles + GPU utilization + cache hit rate
```

### Q2: 深度题 — KV Cache 内存管理
**题目**：当 128K context 的请求和 2K context 的请求混合时，如何管理 KV Cache？

**考察要点**：
- 长短请求混排对内存碎片的影响
- Paged Allocation 如何缓解碎片
- 是否需要 preemption（抢占）机制
- 如何设计 admission control

**参考回答要点**：
- Paged 分配天然支持混合长度，不需要预留最大长度
- 当总 block 不足时，优先驱逐 non-running 请求的 KV（vLLM 的 swap/recompute 策略）
- 设置 admission control：若 pending 请求所需 block 超过剩余容量 × 安全系数，延迟 prefill
- 监控 block utilization 和 preemption rate

### Q3: 深度题 — FlashDecoding 原理
**题目**：解释 FlashDecoding 和 FlashAttention 的区别，为什么 Decode 需要单独优化？

**考察要点**：
- Prefill: sequence parallelism，Q/K/V 都很长 → compute-bound
- Decode: Q 只有 1 个 token，K/V 是整个历史 → memory-bound
- FlashDecoding: 在 KV 的 sequence 维度上做并行（split-K），每个 thread block 处理一段 K/V
- 最后做 reduction 合并各段的 softmax 结果（online softmax trick）

### Q4: 深度题 — LoRA 推理优化
**题目**：在一个 base model 上需要同时服务 100+ 个 LoRA adapter，如何设计？

**考察要点**：
- Base weight 共享，仅加载 adapter 的 A/B 矩阵
- Batching 策略：同 adapter 的请求 batch 到一起（S-LoRA 思路）
- 内存管理：adapter 按需加载/卸载，LRU 管理 adapter 缓存
- Unified paging：adapter weight 也用 paged 方式管理

### Q5: 算法题 — Continuous Batching 调度
**题目**：实现一个简化版的 Continuous Batching 调度器

**考察要点**：
- 核心数据结构：waiting queue + running batch
- 每步逻辑：先做 decode（running batch），再填充 prefill（from waiting）
- 抢占策略：当 KV 不够时，swap out 最近最少用的请求
- Chunked Prefill：大 prompt 分片 prefill，避免一次性占满 GPU

### Q6: 成本优化
**题目**：如何把一个 70B 模型的推理成本降低 50%？

**考察要点**：
- 量化：INT8 权重 + INT8 KV Cache → 显存减半
- 投机解码：合适场景 2-3x 加速
- Prefix Caching：高复用场景（RAG/多轮对话）减少 prefill
- 混合精度 KV：热门 layer 用 FP16，冷门 layer 用 INT4
- 硬件选型：A100 vs H100 的 $/token 对比

### Q7: 分布式推理
**题目**：为 MoE 模型（如 DeepSeek-V3，671B）选择并行策略

**考察要点**：
- Attention 用 TP（需要 AllReduce）
- Expert 用 EP（需要 All-to-All）
- 通信量分析：TP 的 AllReduce vs EP 的 All-to-All
- NVLink 内 TP + 节点间 EP 的混合策略
- Expert 负载均衡 + 容量因子

### Q8: 长上下文
**题目**：如何支持 1M token 的推理？

**考察要点**：
- KV Cache 显存计算：1M × 2 × n_layers × d_model × 2 bytes
- Ring Attention / Striped Attention 跨设备分布 KV
- KV Cache 压缩：量化 + 稀疏化（H2O/SnapKV）
- 分层存储：GPU → CPU → NVMe
- Chunked Prefill 避免 OOM

---

## 三、字节特色追问

1. **"你怎么看 MegaScale 的设计？"** → 了解字节的训练基础设施论文
2. **"多模态推理的 compute profile 和纯文本有什么不同？"** → Vision encoder + cross-attention 的额外开销
3. **"如何做 A/B 实验评估推理优化的效果？"** → 在线 latency/throughput + 离线 quality（PPL, accuracy）
4. **"你在之前的项目中做过什么推理优化？"** → 准备 STAR 故事

---

## 四、面试流程（典型）

| 轮次 | 内容 | 时长 |
|------|------|------|
| 一面 | 算法/数据结构 + LLM 基础 | 60 min |
| 二面 | 系统设计 + 推理深度题 | 60 min |
| 三面 | 项目 Deep Dive + 行为面试 | 45 min |
| HR 面 | 薪资/团队匹配 | 30 min |

### 一面准备清单
- [ ] LeetCode Medium 级别的 coding（Python/C++）
- [ ] Transformer 架构细节（GQA, RoPE, RMSNorm 的公式和作用）
- [ ] KV Cache 基本概念（为什么需要、显存计算、PagedAttention）

### 二面准备清单
- [ ] 完整系统设计框架（需求分析 → 架构 → 深入 → 权衡）
- [ ] 分布式推理 TP/PP/EP 的通信分析
- [ ] 推理优化技术对比（量化/投机解码/Cache 复用/PD 分离）

### 三面准备清单
- [ ] 3-5 个 STAR 故事（见 `mock_interview/behavior/star-stories.md`）
- [ ] "为什么选择字节？" → 体量大 + 推理挑战多 + MegaScale 基础设施
- [ ] "你最有挑战性的项目" → 量化结果（延迟降低 X%，成本节省 Y%）

---

## 五、推荐阅读

| 资料 | 重点关注 |
|------|---------|
| MegaScale 论文（NSDI 2024） | 大规模训练基础设施设计 |
| FlashDecoding blog（Tri Dao） | Decode 阶段 GPU 并行策略 |
| S-LoRA 论文 | 多 LoRA adapter 推理优化 |
| vLLM 源码 | PagedAttention + Scheduler |
| DeepSeek-V3 技术报告 | MoE + MLA 推理架构 |

---

## 六、心算练习（字节面试常考）

```
快速回答（30 秒内）：
1. Llama3-70B FP16 权重大小？ → 140 GB
2. 70B 模型 128K context 的 KV Cache（GQA 8 heads, bf16）？
   → 2 × 80 layers × 8 × 128 × 128K × 2 bytes ≈ 40 GB
3. H100 bf16 峰值算力？ → 990 TFLOPS
4. H100 HBM 带宽？ → 3.35 TB/s
5. 一次 AllReduce (ring) 的通信量？ → 2(N-1)/N × data_size
```
