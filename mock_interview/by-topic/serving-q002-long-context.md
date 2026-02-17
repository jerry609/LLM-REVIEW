# 面试题：如何支持 128K 长上下文推理？

## 题目
"你们的用户需要 128K context window，但目前系统最大只支持 8K。你怎么扩展？"

## 答题思路

### 挑战分析
| 问题 | 8K → 128K 放大倍数 |
|------|------------------|
| KV Cache 大小 | 16× |
| Prefill 计算量 | 256×（O(T²)） |
| TTFT | 大幅增加 |
| 并发能力 | 降到 1/16 |

### 方案分层

#### 1. 显存管理
- **KV 量化**：bf16→fp8 省一半 → 8× KV
- **稀疏化**：H2O/SnapKV 保留 20% token → 再省 5×
- **分页管理**：PagedAttention 避免碎片
- **CPU Offload**：冷 KV 迁移到 CPU，decode 时预取

#### 2. 计算优化
- **FlashAttention-2**：减少 HBM IO，支持长序列
- **Ring Attention**：跨 GPU 分布式注意力（序列并行）
- **Chunked Prefill**：切分 prefill 避免显存峰值

#### 3. 架构优化
- **P/D 分离**：长 prefill 不阻塞短 decode
- **投机解码**：加速 decode 阶段
- **更大 TP 度**：TP=16 分摊 KV 显存

### 质量保证
- Needle-in-a-Haystack 测试：确保长上下文不丢关键信息
- PPL 对比：量化/稀疏后的质量退化可接受
- 逐级降级：超长时自动截断 + 告知用户

## 面试一句话
- "128K 的核心矛盾是显存和计算的 16×/256× 增长。分层解法：量化→稀疏→分页→Ring Attention→P/D 分离。"
