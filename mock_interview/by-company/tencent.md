# 腾讯（Tencent）— LLM 推理工程师面试定向准备

> 目标岗位：LLM Inference / GPU 推理优化工程师
> 相关团队：TEG（技术工程事业群）、混元大模型、腾讯云智能

---

## 一、公司技术栈与核心方向

### 1. 核心产品与平台
| 产品/平台 | 技术侧重 |
|-----------|----------|
| **混元大模型** | 自研 MoE 架构，Hunyuan-Large（389B MoE），支持多模态 |
| **腾讯云 TI 平台** | 大模型推理云服务，多租户，弹性资源调度 |
| **微信 AI** | 对话/搜索/推荐中的 LLM 应用，低延迟要求极高 |
| **游戏 AI** | NPC 对话生成，实时性要求 |

### 2. 推理技术栈
- **vLLM 深度定制**：腾讯在 vLLM 基础上做了大量定制优化
  - `kv_cache.py` 核心重构：Paged KVCache 2.0
  - 分层缓存设计（GPU HBM → CPU DRAM → NVMe SSD）
  - 动态内存管理
- **TensorRT-LLM 集成**：NVIDIA 生态的深度使用
- **自研 CUDA Kernel**：FlashAttention 变体、融合算子
- **混元 MoE 推理**：Expert Parallelism + 动态路由优化

### 3. 核心技术关注点
1. **Paged KVCache 2.0 架构**：在 vLLM PagedAttention 基础上的改进
   - Block 大小自适应（根据请求长度动态调整）
   - 跨请求的 Block 共享（CoW - Copy on Write）
   - 分层存储（热 blocks 在 HBM，温 blocks 在 DRAM，冷 blocks 在 SSD）
2. **混合缓存（Hybrid Cache）**：
   - GPU/CPU/SSD 三级缓存架构
   - 异步预取（prefetch）机制
   - 基于访问频率的分层策略
3. **MoE 推理优化**：
   - 混元 MoE 的 Expert 调度与负载均衡
   - Expert 缓存：不活跃 expert 卸载到 CPU
   - 动态 expert 并行度调整
4. **多租户资源隔离**：
   - 腾讯云场景需要严格的 SLO 保障
   - 基于 GPU MPS/MIG 的硬隔离 vs 软隔离方案

---

## 二、高频面试题（8 道）

### Q1: 系统设计 — 设计腾讯云的大模型推理平台
**题目**：设计一个多租户大模型推理平台，支持 100+ 租户共享 GPU 集群

**考察要点**：
- 资源调度：GPU 分配、抢占、弹性扩缩容
- 隔离性：不同租户的 SLO 保障
- KV Cache 管理：多租户共享 vs 隔离
- 成本分摊与计费模型

**回答框架**：
```
1. 需求：100 租户，每租户不同模型/SLO，总 GPU = 1000 卡
2. 架构层次：
   - 控制面：Tenant Manager → Model Registry → GPU Scheduler
   - 数据面：Router → Inference Engine (vLLM) → KV Store
3. KV Cache 策略：
   - 同模型共享 Prefix Cache（跨租户）
   - 租户级配额：Fair Eviction（你 notebook 中实现的）
4. SLO 保障：
   - 优先级队列（priority = f(SLO_headroom, tenant_weight)）
   - 过载保护：admission control + request shedding
5. 监控：per-tenant latency SLO compliance rate
```

### Q2: 深度题 — Paged KVCache 实现细节
**题目**：详细描述 PagedAttention 的内存管理，如何处理碎片？

**考察要点**：
- Block Table：logical block → physical block 的映射
- Free Block 管理：类似 OS 的 buddy system / free list
- Copy-on-Write：fork 时共享 block，修改时复制
- Block 大小选择的 trade-off（大 block：碎片少 / 小 block：灵活）

**参考回答要点**：
```
PagedAttention 核心思想：将 KV Cache 看作虚拟内存页。

数据结构：
- block_table[seq_id][logical_idx] → physical_block_id
- free_block_list: 空闲物理 block 栈
- ref_count[physical_block_id]: 引用计数（用于 CoW）

分配流程：
1. 新请求到来 → 计算所需 block 数 = ceil(seq_len / block_size)
2. 从 free_list pop 所需 block → 建立映射
3. decode 每生成 block_size 个 token → 分配新 block

驱逐流程（内存不足时）：
1. 选择 victim（LRU/LFU/Fair）
2. 如果 victim 可 swap → 异步写到 CPU
3. 如果不可 swap → 标记为 recompute
4. 释放 victim 的所有 physical block → 回到 free_list

碎片分析：
- 内部碎片 = 最后一个 block 的未填满部分，最多浪费 block_size-1 tokens
- 外部碎片 = 0（paged 天然无外部碎片）
```

### Q3: 深度题 — 分层缓存（Hybrid Cache）
**题目**：如何设计 GPU → CPU → SSD 的三级 KV Cache？

**考察要点**：
- 哪些 KV 放 GPU？哪些放 CPU？哪些放 SSD？
- 异步预取（prefetch）策略
- 每层的带宽和延迟对比
- 一致性问题

**参考回答要点**：
```
层级设计：
- L1 (GPU HBM): 当前 running batch 的 KV + 高频复用前缀
  容量: 40-60GB, 带宽: 3.35 TB/s (H100)
- L2 (CPU DRAM): 被 swap out 但可能被复用的 KV
  容量: 256-512GB, 带宽: ~50 GB/s (PCIe 5.0)
- L3 (NVMe SSD): 长期保存的 prefix cache（RAG 场景）
  容量: 数 TB, 带宽: ~7 GB/s (PCIe Gen4 x4)

预取策略：
- 基于 prefix tree 预测下一步可能访问的 KV
- 多轮对话场景：在上一轮 decode 完成时，异步预取下一轮需要的 KV

关键挑战：
- GPU ↔ CPU 传输是瓶颈（PCIe 限制）
- 需要 overlap compute 和 transfer：在做 layer N attention 时，预取 layer N+1 的 KV
- 使用 CUDA streams + pinned memory 实现异步传输
```

### Q4: 深度题 — MoE 推理优化
**题目**：混元大模型是 MoE 架构，如何优化推理？

**考察要点**：
- Expert Parallelism (EP) 的通信模式：All-to-All
- Expert 负载不均衡问题
- Expert 缓存策略（不活跃 expert 卸载）
- 与 TP 的混合策略

**参考回答要点**：
```
MoE 推理流程：
1. Router 计算每个 token 的 top-k expert 分配
2. All-to-All 通信：将 token 发送到对应 expert 所在的 GPU
3. Expert 计算
4. All-to-All 通信：将结果发回原 GPU

优化策略：
1. Expert 缓存：
   - 并非所有 expert 都常被选中
   - 热门 expert 常驻 GPU，冷门 expert 放 CPU
   - 基于历史路由统计预测 expert 访问模式

2. 负载均衡：
   - Auxiliary loss 在训练时鼓励均匀路由
   - 推理时动态 capacity factor 控制
   - Token dropping：过载 expert 丢弃低权重 token

3. 通信优化：
   - All-to-All 与 attention 计算 overlap
   - 分组 All-to-All：将多个 expert 的通信合并
```

### Q5: 深度题 — vLLM Scheduler 源码
**题目**：描述 vLLM 的 Scheduler 核心逻辑

**考察要点**：
- SequenceGroup 和 SequenceGroupMetadata 的关系
- _schedule_prefills() 和 _schedule_running() 的优先级
- preemption 策略：swap vs recompute 的选择
- budget 管理：max_num_seqs + max_num_batched_tokens

**参考回答要点**：
```
vLLM Scheduler 核心流程（每个 step）：
1. _schedule_running(): 处理已在运行的 decode 请求
   - 检查每个 seq 是否有足够的 block 继续 decode
   - 如果 block 不足 → preempt（swap to CPU 或 recompute）
   
2. _schedule_swapped(): 尝试恢复被 swap 的请求
   - 检查 GPU block 是否够恢复
   
3. _schedule_prefills(): 从 waiting queue 取新请求
   - 受 budget 约束（max_num_seqs, max_num_batched_tokens）
   - 支持 chunked prefill（大 prompt 分片处理）

关键设计决策：
- Decode-first：优先保证已运行请求的进度，避免 starvation
- Preemption 策略：短序列 → recompute（开销小），长序列 → swap（避免重计算）
```

### Q6: 算法题 — Block Allocator
**题目**：实现一个 Block Allocator，支持 allocate/free/CoW

```python
class BlockAllocator:
    def __init__(self, num_blocks: int): ...
    def allocate(self) -> int: ...
    def free(self, block_id: int): ...
    def cow(self, block_id: int) -> int: ...  # copy-on-write
    def ref_count(self, block_id: int) -> int: ...
```

### Q7: 性能分析
**题目**：一个 H100 GPU 上跑 Llama3-70B（TP=4），decode 吞吐量理论上限是多少？

**考察要点**：
- Decode 是 memory-bound → 吞吐受 HBM 带宽限制
- 每 token decode 需要读取的参数量：70B / 4 (TP) × 2 bytes = 35 GB
- H100 HBM 带宽 3.35 TB/s
- 理论上限 ≈ 3350 / 35 ≈ 96 tokens/s（单请求）
- Batch size 增加 → 共摊参数读取 → 吞吐线性增长直到 compute-bound

### Q8: 多租户公平性
**题目**：多租户场景下，如何保证小租户不被大租户 "饿死"？

**考察要点**：
- 你在 notebook 中实现的 Fair Eviction + Jain Fairness
- 请求调度：Weighted Fair Queuing
- Block 配额：per-tenant block budget
- 监控：per-tenant hit rate / latency SLO compliance

---

## 三、腾讯特色追问

1. **"vLLM 的 BlockManager V1 和 V2 有什么区别？"** → V2 支持 sliding window, prefix caching 等更复杂策略
2. **"混元的 MoE 和 Mixtral 有什么不同？"** → 架构细节（expert 数、routing 方式、MLA 等）
3. **"腾讯云如何做 GPU 资源调度？"** → 类似 Kubernetes + GPU Operator + 自研调度器
4. **"你对 CUDA 了解多少？"** → 腾讯重视 kernel 优化能力
5. **"如何评估一个推理优化的 ROI？"** → 延迟改善 vs 开发成本 vs 通用性

---

## 四、面试流程（典型）

| 轮次 | 内容 | 时长 |
|------|------|------|
| 一面 | 算法题（2 道）+ LLM 基础 | 60 min |
| 二面 | 系统设计 + 项目 Deep Dive | 60 min |
| 三面 | 总监面：技术 Vision + 行为 | 45 min |
| GM 面 | 团队匹配 + 职业规划 | 30 min |

### 一面准备清单
- [ ] LeetCode Medium-Hard（重点：链表、树、图、动态规划）
- [ ] Transformer 架构全流程（能手写代码）
- [ ] KV Cache / PagedAttention 原理
- [ ] GPU 内存模型基础（HBM, SRAM, L2）

### 二面准备清单
- [ ] 完整系统设计（推理平台 or 长上下文服务）
- [ ] vLLM 源码核心流程（Engine → Scheduler → Worker → BlockManager）
- [ ] 项目中的量化数据（延迟降低 X%，GPU 利用率提升 Y%）

### 三面准备清单
- [ ] STAR 故事（见 `mock_interview/behavior/star-stories.md`）
- [ ] "你怎么看未来 3 年 LLM Serving 的技术趋势？"
- [ ] "你对腾讯混元大模型有什么了解？"

---

## 五、推荐阅读

| 资料 | 重点关注 |
|------|---------|
| vLLM 源码（block_manager.py, scheduler.py） | Paged 内存管理 + 调度策略 |
| 腾讯云 vLLM KV Cache 博客 | Paged KVCache 2.0 设计 |
| Hunyuan-Large 技术报告 | 混元 MoE 架构 |
| FlashAttention 1 & 2 论文 | CUDA kernel 优化 |
| NVIDIA GPU Memory Hierarchy 文档 | 内存带宽分析 |

---

## 六、心算练习

```
快速回答（30 秒内）：
1. Hunyuan-Large (389B MoE, top-2/16 experts) 每 token 实际计算量？
   → 激活参数 ≈ 389B × 2/16 ≈ 49B（attention 层全激活，仅 FFN 是 MoE）
2. 一个 block_size=16 的 Paged KVCache，存 100K tokens 需要多少 block？
   → ceil(100000 / 16) = 6250 blocks
3. PCIe Gen5 x16 带宽？ → 64 GB/s（双向）
4. H100 NVLink 带宽？ → 900 GB/s（双向）
5. 8 卡 TP 一次 AllReduce 通信量（fp16, hidden=8192）？
   → 2 × (8-1)/8 × 8192 × 2 bytes ≈ 28 KB per token
```
