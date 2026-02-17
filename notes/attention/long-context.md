# 长上下文推理技术

## 1. 核心挑战

标准 Self-Attention 的复杂度为 O(n²)，当 context length 从 4K → 128K → 1M 时：
- **计算量**：二次增长，prefill 时间急剧上升
- **内存**：KV Cache 线性增长，显存成为瓶颈
- **精度**：注意力权重稀释，远距离信息丢失

## 2. Ring Attention

### 原理
- 将 KV 序列分成 P 份，分布在 P 个设备上
- 每个设备持有完整的 Q，但只有 1/P 的 KV
- 通过 **环形通信**：每个设备依次把自己的 KV block 传给下一个设备
- P 轮通信后，每个设备都计算了完整的 attention

### 关键公式
```
Device i, round r:
  local_kv = KV_block[(i + r) % P]
  partial_attn[r] = softmax(Q @ local_kv.T / √d) @ local_kv
  
最终: attn_output = online_softmax_combine(partial_attn[0:P])
```

### 优势
- 通信与计算 **完全重叠**（pipeline）
- 理论上可处理 **无限长** 序列（受设备数限制）
- 无近似，精确等价于标准 attention

### 限制
- 需要 causal mask 的特殊处理
- P 轮通信延迟 = P × KV_block 传输时间
- 要求高速互联（NVLink / InfiniBand）

## 3. Striped Attention

### 原理
- Ring Attention 的改进：将序列按 **striped pattern** 分布
- Token i 分配到设备 i % P
- 好处：每个设备的 KV block 包含均匀分布的 token，负载更均衡

### vs Ring Attention
| 特性 | Ring Attention | Striped Attention |
|------|---------------|-------------------|
| 分布方式 | 连续块 | 交错分布 |
| Causal mask | 需要特殊处理（半空 block） | 每个 block 都有有效 token |
| 负载均衡 | 可能不均（最后 round 计算少） | 天然均衡 |
| 实现复杂度 | 较低 | 略高 |

## 4. Inf-LLM

### 核心思想
- 不要求所有 KV 都在 GPU 上
- **Offload**：旧的 KV 移到 CPU，需要时再取回
- **选择策略**：保留 initial tokens + recent window + attention score 高的 tokens

### 架构
```
GPU: [initial_tokens (固定)] + [recent_window (滑动)] + [important_tokens (动态)]
CPU: [全部历史 KV]
```

### Eviction/Recall 策略
1. 每步 decode 时，计算 Q 与 CPU 上所有 KV 的近似 attention score
2. Top-K 高分 KV block 换回 GPU
3. 最低分的 GPU KV block 换出到 CPU

### 适用场景
- 超长文档摘要（100K+ tokens）
- 价格敏感场景（H100 显存有限）

## 5. 其他方法

### 5.1 Sliding Window Attention (Mistral)
- 固定窗口 W（如 4096），每个 token 只看最近 W 个 token
- O(n × W) 复杂度
- 多层叠加后，信息可以跨层传播到更远

### 5.2 Sparse Attention (LongFormer, BigBird)
- 局部窗口 + 全局 token + 随机连接
- 理论 O(n) 但实际 kernel 优化困难

### 5.3 线性 Attention (RWKV, Mamba)
- 将 softmax attention 替换为线性递推
- O(n) 复杂度，常数 memory
- 精度略低于 Transformer（尤其是 recall 任务）

### 5.4 YaRN / NTK-aware Scaling
- 对 RoPE 位置编码进行频率缩放
- 在不微调的情况下外推到更长上下文
- NTK-aware: 高频外推 + 低频内插

## 6. 面试回答模板

> "长上下文推理的核心瓶颈是 attention 的 O(n²) 复杂度和 KV Cache 的线性内存增长。
> 解决方案分三类：
> 1. **精确分布式**：Ring/Striped Attention - 将 KV 分布到多设备，环形通信 + 计算重叠
> 2. **近似/稀疏**：Sliding Window + Sparse Attention - 限制 attention 范围
> 3. **Offload**：Inf-LLM 风格 - 热 KV 在 GPU，冷 KV 在 CPU，按需换入
>
> 工程上，vLLM 的 Chunked Prefill + PagedAttention 是基础设施。
> 在此之上，根据精度要求选择精确（Ring）还是近似（Sliding Window）方案。"
