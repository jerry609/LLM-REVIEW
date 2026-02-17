# KV Cache 核心概念

## 为什么需要 KV Cache？
- 自回归生成：每步只预测 1 个 token，但需要对**所有历史 token** 做注意力
- 不缓存：第 t 步需重算 O(t) 的 K/V → 生成 T 个 token 累计 O(T²) 的冗余计算
- 缓存后：历史 K/V 复用，每步只需计算新 token 的 Q·K^T → 单步 O(T_cache)

## KV Cache 存什么？
- 每层存 K 和 V 各一份：`[batch, n_kv_heads, seq_len, head_dim]`
- **不存 Q**：Q 只需当前 token，用完即弃
- GQA 下 n_kv_heads < n_heads，显著减少缓存量

## 每 token KV 大小公式
```
bytes_per_token = 2 × n_layers × n_kv_heads × head_dim × bytes_per_elem
```
- 例：Llama3-70B（GQA, n_layers=80, n_kv_heads=8, head_dim=128, bf16=2B）
  → 2 × 80 × 8 × 128 × 2 = 327,680 B ≈ **320 KB/token**
- 128K context → **~40 GB/会话** → 必须做分页、压缩、驱逐

## Prefill vs Decode 阶段
| 阶段 | 处理 token 数 | 瓶颈 | KV 行为 |
|------|-------------|------|---------|
| Prefill | T_input（一次性） | Compute-bound | 一次性写入 KV |
| Decode | 每步 1 token | Memory-bound | 读全量 KV + 追加 1 行 |

## KV Cache 生命周期
```
请求到达 → 分配块 → Prefill 填充 → Decode 追加 → 请求结束 → 释放块
                                                    ↑ 前缀可能被缓存复用
```

## 面试一句话
- "KV Cache 把自回归推理的 O(T²) 冗余降到 O(T)，代价是显存占用随上下文线性增长，必须通过分页、量化、驱逐来管理。"
