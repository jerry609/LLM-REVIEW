# 系统设计 Mock：设计一个支持 100 QPS 的 LLM Serving 系统

> 面试时间：45 分钟。先确认需求，再画架构，最后深入关键组件。

---

## 第一步：需求澄清 (5分钟)

### 我会主动问面试官的问题

| 问题 | 假设答案 |
|------|---------|
| 模型大小？ | Llama-3.1-70B |
| 目标 QPS？ | 100 QPS（峰值 150） |
| 延迟 SLA？ | TTFT < 500ms, TPOT < 50ms (P99) |
| 平均输入/输出长度？ | input 1024 tokens, output 256 tokens |
| 多租户？ | 是，20+ 租户，需要公平性保证 |
| 可用性要求？ | 99.95% |
| 预算？ | 按需，但需要成本优化建议 |

### 关键计算

```
单请求处理时间 ≈ TTFT + output_tokens × TPOT
                = 500ms + 256 × 50ms = 13.3s

单 GPU 70B 吞吐（Tensor Parallel=8, 1 node）:
  - Continuous batching, batch_size ≈ 32
  - 吞吐 ≈ 32 / 13.3s ≈ 2.4 req/s/node (8×A100)

支持 100 QPS 需要:
  - 100 / 2.4 ≈ 42 nodes → 约 336 张 A100
  - 考虑 burst 和冗余: ~50 nodes (400 张 A100)
```

---

## 第二步：高层架构 (10分钟)

```
                    ┌─────────────┐
                    │   Gateway   │ ← Rate Limit / Auth / Routing
                    │  (Nginx/K8s)│
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Router /   │ ← 一致性哈希 (prefix-aware)
                    │  Load Bal.  │   + 权重路由 (SLA-based)
                    └──────┬──────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   ┌──────────┐     ┌──────────┐     ┌──────────┐
   │  Engine   │     │  Engine   │     │  Engine   │
   │  Node 1   │     │  Node 2   │     │  Node N   │
   │ (8×A100)  │     │ (8×A100)  │     │ (8×A100)  │
   └──────────┘     └──────────┘     └──────────┘
        │                │                │
        └────────────────┼────────────────┘
                         │
                  ┌──────▼──────┐
                  │  Metrics /   │ ← Prometheus + Grafana
                  │  Monitoring  │   Jain fairness, SLO
                  └─────────────┘
```

### 每个 Engine Node 内部

```
┌─────────────────────────────────────────┐
│                 Engine Node              │
│                                          │
│  ┌──────────┐  ┌───────────┐  ┌───────┐ │
│  │ Scheduler │→│   Worker   │→│ Block  │ │
│  │(Cont.Batch)│ │(TP across │  │Manager│ │
│  │           │  │ 8 GPUs)   │  │(Paged)│ │
│  └──────────┘  └───────────┘  └───────┘ │
│       ↕              ↕            ↕      │
│  ┌──────────┐  ┌───────────┐  ┌───────┐ │
│  │  Request  │  │ KV Cache  │  │ Swap  │ │
│  │  Queue    │  │ (GPU HBM) │  │ (CPU) │ │
│  └──────────┘  └───────────┘  └───────┘ │
└─────────────────────────────────────────┘
```

---

## 第三步：关键组件深入 (20分钟)

### 3.1 Scheduler（最核心）

**Continuous Batching + Decode-first + Chunked Prefill**

```python
# 伪代码
class Scheduler:
    def step(self):
        # 1. Decode 优先：已在 running 的请求先出 1 token
        decode_batch = self.running_requests
        decode_tokens = len(decode_batch)  # 每个 1 token
        
        # 2. Prefill 填充：剩余 budget 给新请求
        budget = MAX_BATCH_TOKENS - decode_tokens
        while self.waiting and budget > 0:
            req = self.waiting.peek()
            chunk = min(req.remaining_prefill, budget, CHUNK_SIZE)
            self.schedule_prefill_chunk(req, chunk)
            budget -= chunk
        
        # 3. Preemption：如果显存不足
        while self.gpu_memory_pressure():
            victim = self.select_victim()  # LRU among lowest priority
            self.swap_out(victim)          # KV Cache → CPU
```

**关键参数选择**:
- `MAX_BATCH_TOKENS = 8192`（A100 80GB，70B TP=8）
- `CHUNK_SIZE = 512`（平衡 TTFT 和 decode 干扰）
- Decode-first 保证已在 running 的请求不被饿死

### 3.2 KV Cache Management（PagedAttention）

**Block 配置**:
- `block_size = 16` tokens（经验最优）
- 70B model, 80 layers, GQA (n_kv_heads=8), d_head=128
- 每 block KV 大小 = 2 × 80 × 8 × 128 × 16 × 2B(FP16) = 5.24MB
- A100 80GB, TP=8: 每 GPU 10GB 给 KV Cache → ~240 blocks/GPU → 总 1920 blocks
- 最大并发序列 ≈ 1920 blocks / (avg_seq_len/16) ≈ 1920 / 80 ≈ 24 seqs/node

**优化**:
- FP8 KV 量化 → blocks 翻倍 → 48 seqs/node
- Prefix sharing → 公共 system prompt 只存 1 份

### 3.3 Router / Load Balancer

**两级路由策略**:

1. **Prefix-aware Consistent Hashing** (SGLang style):
   - 对请求的 system prompt hash → 路由到相同 engine
   - 好处：prefix KV Cache 复用率最大化
   - 实现：consistent hash ring，每个 engine 100 个虚拟节点

2. **SLA-weighted Routing**:
   - 高 SLA 租户 → 优先路由到负载低的 engine
   - 超载保护：单 engine pending > threshold → 拒绝并重路由

```python
def route(request):
    # 1. 计算 prefix hash
    prefix_hash = hash(request.system_prompt)
    
    # 2. Consistent hash 选候选 engine
    candidates = consistent_hash.get_nodes(prefix_hash, n=3)
    
    # 3. 在候选中选负载最低的
    best = min(candidates, key=lambda e: e.pending_count)
    
    # 4. 检查租户配额
    if best.tenant_usage[request.tenant_id] > quota:
        return fallback_engine()
    
    return best
```

### 3.4 Multi-Tenancy & Fairness

**三层保障**:

| 层 | 机制 | 作用 |
|----|------|------|
| Gateway | Rate Limiting | 防止单租户突发流量 |
| Router | Weighted Fair Queuing | 按 SLA 权重调度 |
| Engine | Quota-Aware Eviction | KV Cache 按配额分配 |

**监控指标**:
- Per-tenant TTFT / TPOT P50/P99
- Jain's fairness index (全局)
- Cache hit rate per tenant
- 告警：Jain index < 0.85 → 自动调整配额权重

### 3.5 高可用 & 容错

| 故障场景 | 应对 |
|---------|------|
| 单 GPU 故障 | 自动摘除，TP 重分布 |
| 单 Node 故障 | Router 健康检查摘除，流量重路由 |
| 模型更新 | 滚动更新 + 金丝雀发布 |
| 突发流量 | 自动扩容 + 降级到小模型 |

---

## 第四步：成本优化 (5分钟)

| 策略 | 节省 | 风险 |
|------|------|------|
| FP8 KV 量化 | GPU -40% | 质量微降 |
| Prefix Caching | GPU -20% | 冷启动延迟 |
| 投机解码 | GPU -30% | 增加系统复杂度 |
| 混合模型 (蒸馏 7B + 70B) | GPU -50% | 需要路由逻辑 |
| Spot Instance | 费用 -60% | 可能被回收 |

**最佳组合**：FP8 + Prefix Caching + 场景化蒸馏 → 总成本降 55%

---

## 第五步：回答追问 (5分钟)

### Q: "如果延迟 SLA 更严格（TPOT < 20ms）怎么办？"
- Speculative Decoding：2× 加速
- 更多 TP：TP=16（2 nodes），减少每 GPU 计算量
- 更小模型 + 蒸馏

### Q: "如何从 100 QPS 扩展到 1000 QPS？"
- 水平扩展 Engine Nodes（10×）
- 引入 Disaggregated Prefill-Decode：
  - Prefill Worker pool：专门处理长 prompt（大 batch，compute-bound）
  - Decode Worker pool：专门做 token generation（小 batch，memory-bound）
  - KV Cache 通过 RDMA 在两个 pool 间传输

### Q: "如何保证模型更新不影响服务？"
- 金丝雀发布：5% 流量 → 观察 30 分钟 → 全量
- 自动 benchmark：CI 流水线测试性能和质量
- 一键回滚：保留上一版本的 model artifact

### Q: "多模型共存（70B + 7B + Embedding Model）怎么管理？"
- GPU 池化：统一 block manager
- 按负载自动分配 GPU 给不同模型
- 轻量模型（7B）用 DP，大模型（70B）用 TP
- 共享 prefix cache（如果 system prompt 一样）

---

## 板书清单

面试时需要在白板上画的内容（按顺序）：

1. **高层架构图**：Gateway → Router → Engine Nodes → Monitoring
2. **Engine 内部**：Scheduler → Worker → Block Manager → KV Cache
3. **数据流**：request → prefill → decode loop → response
4. **关键计算**：单节点吞吐、总节点数、KV Cache 大小
5. **时序图**：continuous batching 中 prefill 和 decode 的交替
