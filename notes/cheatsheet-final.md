# LLM 推理面试终极速查卡

## 🧠 模型架构
| 模型 | 核心特点 | KV 影响 |
|------|---------|---------|
| Llama3 | GQA + RoPE + SwiGLU | KV 缩小 8× |
| Mixtral | MoE 8×7B | Attention 部分 KV 不变 |
| DeepSeek-V3 | MLA + 细粒度 MoE | KV 压缩到低维潜在空间 |

## ⚡ 注意力机制
| 方法 | 复杂度 | 核心 |
|------|--------|------|
| MHA | O(T²d) | 所有 head 独立 K/V |
| GQA | O(T²d) 但 KV 少 | 多 Q head 共享 KV head |
| FlashAttention | O(T²d) FLOPs, 低 IO | Tiling + Online Softmax |
| Linear Attention | O(Td²) | φ(Q)(φ(K)^TV) |

## 💾 KV Cache 管理
| 技术 | 效果 |
|------|------|
| PagedAttention | 碎片率 → ~0% |
| 前缀缓存 | TTFT ↓ (热前缀跳过 prefill) |
| FP8/INT8 量化 | 显存省 50%, 吞吐 +30% |
| H2O/SnapKV 稀疏化 | KV 压缩 5-50× |
| LRU/LFU/Fair 驱逐 | 管理有限缓存 |

## 🔧 推理优化技术
| 技术 | 加速比 | 有损？ |
|------|--------|--------|
| Continuous Batching | 2-10× 吞吐 | 无损 |
| Chunked Prefill | 控制 P99 TPOT | 无损 |
| 投机解码 | 1.5-3× | 无损 |
| KV 量化 | 1.3× 吞吐 | 微损 |
| 模型蒸馏 | 5-10× 成本 | 有损 |

## 🌐 分布式推理
| 并行方式 | 切分维度 | 通信 | 适用 |
|---------|---------|------|------|
| TP | 权重（头/FFN 列） | AllReduce | 机内 NVLink |
| PP | 层 | 逐级传递 | 跨机 |
| EP | Expert | All-to-All | MoE |

## 📊 关键公式
```
KV bytes/token = 2 × n_layers × n_kv_heads × head_dim × dtype_bytes
算术强度 = FLOPs / Bytes
Jain = (Σx)² / (n × Σx²)
TP 通信/层 = 2 × B × T × d × bytes
投机解码期望 = (1 - α^(K+1)) / (1 - α)
```

## 🎯 系统设计四步走
1. **需求**：QPS, SLO (TTFT/TPOT), 模型, 上下文
2. **架构**：Router → Scheduler → Workers → KV Manager
3. **深入**：KV 管理 / 调度策略 / 容量规划
4. **权衡**：延迟 vs 吞吐 / 精度 vs 成本 / 扩展方向

## ❓ 万能答题模板
> "这个问题的核心矛盾是 [X vs Y]。
> 我的方案是 [方案]，核心思路是 [一句话]。
> 实现上分 [N] 步：[步骤]。
> 预期效果是 [量化指标]。
> 权衡是 [代价]，可以通过 [缓解措施] 控制。"
