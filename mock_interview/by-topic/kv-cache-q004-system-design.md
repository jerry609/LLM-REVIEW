# KV Cache 面试题 004：多机 KV Cache 系统设计

## 题目
> 设计一个支持 100+ 台 GPU 的分布式 KV Cache 系统。
> 需要支持 prefix sharing、多租户隔离、故障恢复。

## 参考答案（限时 15 分钟）

### 1. 需求分析
- **规模**：100+ GPU，每台 80GB 显存，总 KV Cache 池 ~5TB
- **功能**：prefix sharing（RAG/多轮对话）、tenant isolation、fault tolerance
- **SLA**：P99 TTFT < 200ms，cache hit 场景 < 50ms

### 2. 整体架构

```
┌─────────────────────────────────┐
│         Load Balancer           │
│   (一致性哈希 + 热点感知)         │
└──────────┬──────────────────────┘
           │
┌──────────▼──────────────────────┐
│      Metadata Service           │
│  (etcd/ZooKeeper)               │
│  - prefix → GPU mapping         │
│  - tenant quotas                │
│  - block allocation table       │
└──────────┬──────────────────────┘
           │
┌──────────▼──────────────────────┐
│     Distributed Cache Layer      │
│  ┌────┐ ┌────┐ ┌────┐ ┌────┐   │
│  │GPU0│ │GPU1│ │GPU2│ │... │   │
│  │Pool│ │Pool│ │Pool│ │    │   │
│  └────┘ └────┘ └────┘ └────┘   │
│  (每台维护 PagedAttention)       │
└──────────┬──────────────────────┘
           │
┌──────────▼──────────────────────┐
│      CPU/SSD Overflow Pool       │
│  (冷 KV offload)                │
└─────────────────────────────────┘
```

### 3. 关键设计

#### 3.1 Prefix Sharing
- Radix Tree 全局索引，存在 Metadata Service 中
- 一致性哈希路由：相同 prefix hash → 同一 GPU
- prefix 分级：system prompt (L1, pinned) > document chunk (L2, LRU) > user query (L3, evictable)

#### 3.2 多租户隔离
```python
tenant_config = {
    "tenant_A": {"quota_blocks": 1000, "priority": "high", "max_seq_len": 128000},
    "tenant_B": {"quota_blocks": 500, "priority": "normal", "max_seq_len": 32000},
}
# 驱逐策略：先驱逐超配租户的冷 entry
```

#### 3.3 故障恢复
- **Block 级复制**：热门 prefix 复制到 2 台 GPU（2-replica）
- **Recompute fallback**：cache miss → 退化为完整 prefill（不影响正确性）
- **元数据持久化**：etcd 保证 metadata 高可用

### 4. 性能优化
- **RDMA/GPUDirect**：跨 GPU KV 传输 < 1ms/block
- **预取**：根据 routing prediction 提前加载 KV block
- **分层缓存**：GPU (hot) → CPU (warm) → SSD (cold)

### 5. 评估指标
```
- Global hit rate: 目标 > 70%
- Per-tenant hit rate variance: Jain > 0.8
- P99 cache lookup latency: < 5ms
- Recovery time: < 30s (single GPU failure)
```

### 6. 面试加分项
- 实际系统参考：SGLang RadixAttention, Mooncake, MemServe
- 一致性保证：同一 session 的请求必须看到之前的 KV
- 成本分析：cache 投入 vs prefill 节省的 GPU 时间
