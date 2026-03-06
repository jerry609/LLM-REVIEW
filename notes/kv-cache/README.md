# KV Cache

> 这一组笔记把 KV Cache 拆成三层来讲：先算清楚容量账本，再理解 PagedAttention 的块管理，最后再看量化压缩和驱逐策略。阅读顺序已经按 GitBook 的连续阅读方式重排：先总览，再推导，再源码。

## 最推荐的阅读顺序

1. [formula-to-code-walkthrough.md](formula-to-code-walkthrough.md)：先建立“容量公式 -> block 分配 -> 压缩/驱逐 -> 源码”的整条主线。
2. [concepts.md](concepts.md)：补齐 KV Cache 的生命周期、Prefill / Decode 分工和容量直觉。
3. [paged-attention.md](paged-attention.md)：把 block table、Prefix Caching、Copy-on-Write 放到同一张工程图里看。
4. [../../math_dictionary/kv-memory.md](../../math_dictionary/kv-memory.md)：需要精确算显存和并发时回到数学页。
5. [interview-qa.md](interview-qa.md)：最后用问答形式复盘。

## 这一组专题覆盖什么

- 容量估算：`bytes/token`、序列总 KV 显存、最大并发、block 数量。
- PagedAttention：逻辑连续、物理离散的 block 映射，内部碎片率、Prefix Caching、Copy-on-Write。
- 压缩：INT8 per-channel 量化、反量化、误差指标、压缩比。
- 驱逐：LRU、LFU、多租户公平配额。

## 对应源码

- [../../src/kv_cache/core.py](../../src/kv_cache/core.py)：`BlockAllocator`、`PagedKVCacheManager`、`fork()`、`fragmentation()`。
- [../../src/kv_cache/compression/quantizer.py](../../src/kv_cache/compression/quantizer.py)：对称 / 非对称 per-channel 量化。
- [../../src/kv_cache/eviction/policies.py](../../src/kv_cache/eviction/policies.py)：LRU、LFU、Fair quota 驱逐。

## 如果你只剩 20 分钟

- 先读 [formula-to-code-walkthrough.md](formula-to-code-walkthrough.md)
- 再看 [../../math_dictionary/kv-memory.md](../../math_dictionary/kv-memory.md) 里的 `bytes_per_token` 推导
- 最后对照 [../../src/kv_cache/core.py](../../src/kv_cache/core.py) 里的 `_blocks_needed()`、`append_tokens()`、`fork()`
