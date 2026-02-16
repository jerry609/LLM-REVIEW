# PagedAttention 数学与工程速查

## 1) 核心思想
- 逻辑上序列连续，物理上 KV 按固定大小 block/page 存放。
- 通过页表把"逻辑 token 位置 → 物理块地址"映射起来。
- 类比操作系统虚拟内存：逻辑地址连续，物理页分散。

## 2) 基本参数
- `b`：每个 block 的 token 数（常见 16 或 32）
- `T`：序列长度
- block 数：`N_block = ceil(T / b)`
- 每个 block 的大小：`block_bytes = 2 * L * H_kv * d_head * s * b`

## 3) 内部碎片
- 单序列末块浪费 token：`w = (b - (T mod b)) mod b`
- 最坏浪费 `< b`；平均浪费约 `b/2` 个 token
- 总碎片字节近似：`Frag_bytes = w * bytes_per_token`
- 碎片率：`frag_rate = avg_waste / (avg_waste + avg_used) ≈ (b/2) / (avg_T + b/2)`
  - b=16, avg_T=2048 → 碎片率 < 0.4%（非常低）

## 4) 与连续分配的对比
| 特性 | 连续分配 | PagedAttention |
|------|---------|---------------|
| 碎片 | 外部碎片严重 | 仅 block 内碎片 |
| 扩展 | 需预分配或重新分配 | 按需增加 block |
| 并发 | 受限于最大预分配 | 灵活利用空闲块 |
| 共享 | 困难 | 可共享物理块 |

## 5) Prefix Caching（前缀共享）
- 多个请求共享相同前缀（如系统 prompt）
- 共享物理块 + 引用计数：`ref_count[block_id]++`
- 当 `ref_count == 0` 时回收块
- 节省量：`shared_tokens * bytes_per_token * (num_requests - 1)`
  - 例：2K system prompt × 128 KB/token × 63 请求 ≈ 15.75 GB 节省

## 6) Copy-on-Write
- 共享前缀的请求开始各自生成后，分叉点创建新 block
- 仅复制分叉所在的 block，其余继续共享
- 复制开销：`block_bytes`（单个 block 的数据）

## 7) Block 分配器
- Free list：维护空闲块链表
- 分配：O(1)（从 free list 头部取）
- 回收：O(1)（归还到 free list）
- 当 free list 为空 → 触发驱逐或拒绝新请求

## 8) 吞吐收益来源
- 固定块分配/回收减少大块连续内存需求与碎片累积。
- 块级迁移/驱逐更灵活，提升高并发下稳定性。
- 实测（vLLM 论文）：相比连续分配，batch 吞吐提升 2-4×

## 面试一句话
- "PagedAttention 本质是把 KV 从连续大数组改成分页管理，用页表换取低碎片和高弹性，是 vLLM 高吞吐的核心。"
