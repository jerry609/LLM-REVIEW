# KV 驱逐

> 这一组笔记关注“KV 预算不够时该淘汰谁”。从最基础的 LRU / LFU，到多租户公平驱逐，再到注意力感知策略，核心都在于把“保留价值”写成一个明确的评分函数。

## 最推荐的阅读顺序

1. [formula-to-code-walkthrough.md](formula-to-code-walkthrough.md)：先把 LRU / LFU / Fair quota 的目标函数和源码对上。
2. [../../math_dictionary/kv-eviction-math.md](../../math_dictionary/kv-eviction-math.md)：再看更完整的命中率、重算代价和注意力感知策略。
3. [policies.md](policies.md)：把工程里常见策略和适用场景扫一遍。
4. [interview-qa.md](interview-qa.md)：最后用问答形式复盘。

## 这一组专题覆盖什么

- 基线策略：LRU、LFU、它们的 tie-break 规则。
- 多租户公平：按权重分配预算、超配租户优先回收。
- 系统含义：命中率、回填率、重算成本、质量退化。

## 对应源码

- [../../src/kv_cache/eviction/policies.py](../../src/kv_cache/eviction/policies.py)：`LRUPolicy`、`LFUPolicy`、`FairPolicy`。
- [../../src/kv_cache/core.py](../../src/kv_cache/core.py)：驱逐依赖的 `last_access_step`、`use_count`、`num_blocks()` 等元数据来源。

## 如果你只剩 20 分钟

- 先读 [formula-to-code-walkthrough.md](formula-to-code-walkthrough.md)
- 再看 [../../math_dictionary/kv-eviction-math.md](../../math_dictionary/kv-eviction-math.md) 里的评分函数和公平配额公式
- 最后对照 [../../src/kv_cache/eviction/policies.py](../../src/kv_cache/eviction/policies.py)
