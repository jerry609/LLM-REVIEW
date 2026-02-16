# MQA / GQA 专题对比（数学与工程）

## 1) 定义
- MHA（Multi-Head Attention）：`H_kv = H`（每个 head 独立 KV）
- MQA（Multi-Query Attention）：`H_kv = 1`（所有 head 共享 1 组 KV）
- GQA（Grouped-Query Attention）：`1 < H_kv < H`（每组 query head 共享 1 组 KV）
- 组大小：`group_size = H / H_kv`

## 2) KV 显存关系
- 每 token KV 字节：`2 * L * H_kv * d_head * s`
- 在 `L,d_head,s` 相同下，KV 占用与 `H_kv` 线性相关。

### 具体数字对比（以 7B 模型为例，L=32, d_head=128, s=2）
| 变体 | H_kv | bytes/token | 32 并发 4K token 总 KV |
|------|------|-------------|----------------------|
| MHA  | 32   | 512 KB      | 64 GB               |
| GQA-8| 8    | 128 KB      | 16 GB               |
| GQA-4| 4    | 64 KB       | 8 GB                |
| MQA  | 1    | 16 KB       | 2 GB                |

## 3) 带宽节省（Decode 阶段）
- Decode 每步需读取全部 KV cache
- 带宽需求 ∝ `H_kv`
- GQA-8 vs MHA：带宽需求降 4×
- 直接影响 TPOT：读取越快，每步 decode 越快

## 4) 计算量不变
- Q 的 head 数 `H` 不变 → QK^T 计算量不变
- 区别仅在存储和读取 KV 的成本
- GQA 通过广播实现：`K_expanded[:, h, :, :] = K[:, h // group_size, :, :]`

## 5) 质量影响
- MQA：显存最省、带宽压力最低，但精度风险更高（共享过度）。
- GQA：在显存和质量间折中，是工程上常见选择。
- MHA：表达能力最好但 KV 成本最高。
- 实验发现：GQA-8 通常与 MHA 质量接近，MQA 在某些任务有明显退化。

## 6) 从 MHA 转换到 GQA
- 方法：将 MHA 的 KV head 按组平均（mean pooling）
  `K_gqa[:, g, :] = mean(K_mha[:, g*group_size:(g+1)*group_size, :])`
- 可通过少量继续训练恢复质量

## 7) 面试常见结论
- "是否用 GQA，本质是用少量注意力表达能力换可服务并发与上下文长度。"
- "GQA 不改变模型能力上限（Q 侧不变），只压缩 KV 侧的冗余。"

## 8) 选型建议
- 先用 GQA（`H_kv=8` 或 `H/4`）作为默认。
- 对极长上下文（>100K）可考虑更小的 `H_kv`。
- 结合 KV 压缩/驱逐联动看端到端收益，而非只看单点指标。
- MQA 适用于对延迟极敏感、质量要求相对低的场景（如草稿模型）。
