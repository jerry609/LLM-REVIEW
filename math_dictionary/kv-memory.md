# KV 缓存显存估算与容量规划

## 1) 每 token 占用
- 每层每 token 元素数：`2 * H_kv * d_head`
- 每层每 token 字节数：`2 * H_kv * d_head * s`
- 全层每 token 字节数：
  `bytes_per_token = 2 * L * H_kv * d_head * s`

## 2) 总占用
- 单序列：`KV_bytes = bytes_per_token * T_cache`
- 多并发：`KV_total = bytes_per_token * sum(T_i)`
- 实际占用还需加上内存碎片和对齐开销

## 3) GQA 的收益
- 与 MHA 相比，KV 内存近似按 `H_kv/H` 成比例下降。
- 当 `H_kv << H` 时，长上下文能力显著提升。

## 4) 快速代入示例

### 70B 模型（GQA, H_kv=8）
- 假设：`L=80, H_kv=8, d_head=128, s=2(BF16)`
- `bytes_per_token = 2*80*8*128*2 = 327,680 B ≈ 320 KB`
- `128K` token 约 `40 GB`（仅 KV，不含权重与激活）

### 7B 模型（GQA, H_kv=8）
- 假设：`L=32, H_kv=8, d_head=128, s=2(BF16)`
- `bytes_per_token = 2*32*8*128*2 = 131,072 B = 128 KB`
- `4K` token 单序列约 `0.5 GB`
- `4K` token × 64 并发约 `32 GB`

### 7B 模型（MHA, H_kv=32）
- 同 7B 但 MHA：`bytes_per_token = 2*32*32*128*2 = 524,288 B = 512 KB`
- 对比 GQA 版本膨胀 4×

## 5) 压缩后容量
- 压缩比 `r = s_old/s_new`
- 新容量近似：`KV_new = KV_old / r`
- 例如 BF16 -> INT8，`r=2`；BF16 -> INT4，`r=4`

## 6) PagedAttention 块管理
- 将 KV cache 分为固定大小的块（block），如每块 16 token
- 块大小 `block_size`，每块占用：`2 * L * H_kv * d_head * s * block_size`
- 用页表（page table）将逻辑 token 位置映射到物理块
- 显存利用率提升：消除碎片，支持动态增长
- 块共享：多个请求若有相同 prefix，可共享同一物理块（prefix caching）

## 7) 显存预算分配
```
GPU 显存 = 模型权重 + KV cache + 激活缓冲 + 系统预留
```
- 模型权重（BF16）：`≈ N * 2 B`（如 7B → ~14 GB，70B → ~140 GB）
- 激活缓冲：与 batch size 和序列长度相关，通常数 GB
- 系统预留：CUDA context + 碎片安全边际，通常 1-3 GB
- **可用 KV 容量 = 总显存 - 权重 - 激活 - 预留**

## 8) 规划流程
1. 先算静态模型占用（权重、运行缓冲）。
2. 预留安全边际（通常 10%-20%）。
3. 剩余显存给 KV，反推最大 `sum(T_i)`。
4. 再决定"压缩优先"还是"驱逐优先"。
5. 用 `max_concurrent = KV_budget / (bytes_per_token * avg_seq_len)` 估算最大并发。

## 面试一句话
- "KV 容量规划是线性账本：token 总数乘每 token 成本，再叠加精度字节数与安全边际。"
