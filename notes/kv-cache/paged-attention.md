# PagedAttention 与块管理

## 核心思想
- 借鉴 OS 虚拟内存的分页机制：**逻辑连续的 KV 序列 → 物理不连续的固定大小 block**
- 每个 block 存 `block_size` 个 token 的 K/V
- 页表（block_table）维护逻辑块号 → 物理块号映射

## 为什么需要分页？
| 问题 | 不分页 | 分页后 |
|------|--------|--------|
| 内存碎片 | 连续分配，不同长度请求导致大量外部碎片 | 固定块大小，只有末尾块有内部碎片 |
| 最大序列长度 | 预分配最大长度，浪费严重 | 按需分配，用多少分多少 |
| 前缀共享 | 每个请求独立拷贝 | Copy-on-Write，引用同一物理块 |

## 碎片分析
- **内部碎片**：最后一个块可能未填满 → `waste = block_size - (seq_len % block_size)`
- **碎片率**：`internal_frag = avg_waste / block_size`
- block_size 越大 → 内部碎片越大，但页表越小
- vLLM 默认 block_size = 16

## Prefix Caching
- 相同 system prompt 或 few-shot 前缀 → 对应的 KV 块可以跨请求复用
- 索引方式：hash(prefix_tokens) → block_table
- vLLM 的 Automatic Prefix Caching：用 radix tree 做前缀匹配
- 命中时跳过 prefill → **TTFT 大幅降低**

## Copy-on-Write
- fork 时新序列共享源序列的物理块（ref_count +1）
- 只有写入（decode 追加）时才真正复制 → 节省分配开销
- 适合 beam search、并行采样等场景

## 关键参数
| 参数 | 含义 | 典型值 |
|------|------|--------|
| block_size | 每块 token 数 | 16 |
| num_gpu_blocks | GPU 上的块数 | 按显存自动计算 |
| num_cpu_blocks | CPU offload 块数 | 0 或按需 |

## 面试一句话
- "PagedAttention 通过虚拟内存分页消除了 KV 缓存的外部碎片，使显存利用率接近 100%，同时支持前缀缓存和 CoW 共享。"
