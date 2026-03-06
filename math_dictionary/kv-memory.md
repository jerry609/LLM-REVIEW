# KV Cache 显存规划：从 `bytes/token` 到块分配与并发预算

> 这页不再试图把 KV Cache 写成一整章教材，而是把真正会落到工程预算里的三件事串起来：先算单个 token 的 KV 开销，再把它扩展到序列与并发预算，最后落到分页分块、碎片率和前缀共享。读完这页，你应该能直接回答“这张卡能撑多长上下文、多少并发、为什么会碎片化”。

## 1. 先统一最小符号集

| 记号 | 含义 |
|------|------|
| `L_layers` | Transformer 层数 |
| `n_heads` | Query head 数 |
| `n_kv_heads` | KV head 数 |
| `d_head` | 单个 head 的维度 |
| `s_bytes` | 单个元素的字节数，例如 BF16 是 2，INT8 是 1 |
| `T_cache` | 当前序列缓存的 token 数 |
| `B_active` | 同时活跃的序列数 |
| `block_size` | 单个 KV block 可容纳的 token 数 |
| `N_blocks` | GPU 上可用的物理 block 总数 |
| `M_budget` | 分给 KV Cache 的显存预算 |

## 2. 从单层单 token 开始推导

对任意一层，只要还是显式存 `K` 和 `V`，每个 token 的存储量就是两份张量：

$$
\text{bytes}_{\text{layer, token}} = 2 \times n_{\mathrm{kv\_heads}} \times d_{\mathrm{head}} \times s_{\mathrm{bytes}}
$$

这条式子里每个因子都很直白：

- 前面的 `2` 来自 `K` 和 `V` 两份缓存。
- `n_kv_heads` 决定每层到底要存多少组 KV 头。
- `d_head` 决定每个头的宽度。
- `s_bytes` 决定精度对应的字节成本。

把所有层叠起来，就得到整个模型的单 token KV 开销：

$$
\text{bytes}_{\text{token}} = 2 \times L_{\mathrm{layers}} \times n_{\mathrm{kv\_heads}} \times d_{\mathrm{head}} \times s_{\mathrm{bytes}}
$$

这就是最核心的 `bytes_per_token` 公式。后面的容量规划、并发估算、块数量估算，本质上都只是围绕它做代数改写。

## 3. 不同注意力结构到底改了哪一项

从 KV 预算角度看，MHA、GQA、MQA、MLA 的差异，本质上是“每个 token 要存多少维的历史表示”。

| 结构 | 单 token 显存主式 | 真正被改变的量 | 工程含义 |
|------|-------------------|----------------|----------|
| MHA | `2 x L_layers x n_heads x d_head x s_bytes` | `n_kv_heads = n_heads` | KV 最完整，也最贵 |
| GQA | `2 x L_layers x n_kv_heads x d_head x s_bytes` | `n_kv_heads < n_heads` | 通过共享 KV 头显著降显存和 decode 带宽 |
| MQA | `2 x L_layers x 1 x d_head x s_bytes` | `n_kv_heads = 1` | 显存最省，但共享更激进 |
| MLA | 近似写成 `L_layers x (d_c + d_r) x s_bytes` | 把历史表示改写成共享潜变量与位置部分 | 不是简单少几个 head，而是整个缓存表征变了 |

如果只比较 MHA 与 GQA，压缩比最容易看：

$$
\text{saving ratio}_{\mathrm{GQA\ vs\ MHA}} = \frac{n_{\mathrm{kv\_heads}}}{n_{\mathrm{heads}}}
$$

例如 `n_heads = 32`、`n_kv_heads = 8`，那么 KV 显存和 decode 阶段的历史读取量都会降到原来的四分之一。

## 4. 从单序列扩展到总预算

单条序列的 KV 显存：

$$
M_{\mathrm{seq}} = T_{\mathrm{cache}} \times \text{bytes}_{\text{token}}
$$

如果系统里同时有多条活跃序列，总 KV 显存就是所有序列长度的加总：

$$
M_{\mathrm{total}} = \sum_{i=1}^{B_{\mathrm{active}}} T_i \times \text{bytes}_{\text{token}}
$$

当你只想做粗估时，可以把每条序列近似成同样的平均长度：

$$
M_{\mathrm{total}} \approx B_{\mathrm{active}} \times \bar{T}_{\mathrm{cache}} \times \text{bytes}_{\text{token}}
$$

这就是容量规划里最常用的并发估算式。

## 5. 从 GPU 显存反推最大上下文和最大并发

真正能分给 KV Cache 的显存，不等于整张卡的物理容量。更稳妥的预算写法是：

$$
M_{\mathrm{budget}} = M_{\mathrm{GPU}} - M_{\mathrm{weights}} - M_{\mathrm{activations}} - M_{\mathrm{workspace}} - M_{\mathrm{safety}}
$$

于是，总可缓存 token 数上限可以直接反推：

$$
T_{\mathrm{max,total}} = \left\lfloor \frac{M_{\mathrm{budget}}}{\text{bytes}_{\text{token}}} \right\rfloor
$$

如果你预期平均上下文长度是 `bar_T_cache`，最大并发就近似为：

$$
B_{\mathrm{max}} \approx \left\lfloor \frac{T_{\mathrm{max,total}}}{\bar{T}_{\mathrm{cache}}} \right\rfloor
$$

### 5.1 一个最常用的代入例子

以 LLaMA-3 8B 常见配置为例：

- `L_layers = 32`
- `n_kv_heads = 8`
- `d_head = 128`
- `s_bytes = 2`，也就是 BF16

代入后得到：

$$
\text{bytes}_{\text{token}} = 2 \times 32 \times 8 \times 128 \times 2 = 131072\ \text{bytes}
$$

也就是每个 token 约 `128 KiB` 的 KV 开销。

如果单条序列的上下文长度是 `8192`，那么这条序列的 KV 显存大约是：

$$
8192 \times 131072 = 1073741824\ \text{bytes}
$$

也就是约 `1 GiB`。这也是为什么大家一做长上下文就会立刻感受到 KV Cache 的压力：不是模型权重先爆，而是“历史 token 的存储账单”先开始失控。

## 6. Paged KV：显存不是按 token 分配，而是按 block 分配

真实系统通常不会一边来一个 token、一边精确地为一个 token 单独申请显存，而是把 KV Cache 分成固定大小的 block。这样一来，分配问题会从“多少 token”转成“多少块”。

一个序列需要的 block 数量是：

$$
N_{\mathrm{blocks}}(T_{\mathrm{cache}}) = \left\lceil \frac{T_{\mathrm{cache}}}{\text{block\_size}} \right\rceil
$$

这条序列真正占掉的已分配容量则变成：

$$
M_{\mathrm{alloc, seq}} = N_{\mathrm{blocks}}(T_{\mathrm{cache}}) \times \text{block\_size} \times \text{bytes}_{\text{token}}
$$

于是，内部碎片率可以直接写成：

$$
\text{fragmentation}(T_{\mathrm{cache}}) = 1 - \frac{T_{\mathrm{cache}}}{N_{\mathrm{blocks}}(T_{\mathrm{cache}}) \times \text{block\_size}}
$$

这条式子解释了一个很工程化的现象：当 `block_size` 过大、而请求长度分布又很离散时，最后一个 block 往往塞不满，碎片率就会上去。

## 7. 前缀共享与 Copy-on-Write 为什么能省很多

如果两条请求共享长前缀，例如同一段 system prompt 或者 prompt cache 命中，那么第二条请求一开始并不需要真的复制全部 KV 数据。更合理的写法是：

$$
\Delta M_{\mathrm{fork, initial}} \approx M_{\mathrm{metadata}}
$$

只有在后续写入新 token，或者发生需要独占修改的场景时，才会通过 Copy-on-Write 新分配 block。此时增量开销更接近：

$$
\Delta M_{\mathrm{append}} = \Delta N_{\mathrm{blocks}} \times \text{block\_size} \times \text{bytes}_{\text{token}}
$$

这也是 prefix caching 好用的根本原因：共享前缀时，真正被放大的通常不是 KV 数据本身，而是很轻量的 block table 元数据。

## 8. 公式如何落到仓库源码

这页最值得对照的不是抽象论文，而是仓库里的最小实现：

- `../src/kv_cache/core.py` 里的 `PagedKVCacheManager._blocks_needed()`，就是上面 `ceil(T_cache / block_size)` 的直接实现。
- `../src/kv_cache/core.py` 里的 `allocate_for_sequence()`，体现的是“先按需要块数分配，再登记 block table”。
- `../src/kv_cache/core.py` 里的 `append_tokens()`，体现的是 token 追加时按需补块，而不是每次整段重建。
- `../src/kv_cache/core.py` 里的 `fragmentation()`，对应的正是“已分配块里有多少空槽位被浪费”。
- `../src/kv_cache/core.py` 里的 `fork()`，对应的正是前缀共享和 Copy-on-Write 的元数据复用逻辑。

如果你想把公式和测试一起对上，可以继续看：

- `../tests/test_kv_cache.py`
- `../notes/kv-cache/formula-to-code-walkthrough.md`
- `pagedattention-math.md`

## 9. 这页真正应该记住什么

- KV 预算最先要抓的不是“模型多少参数”，而是 `bytes_per_token`。
- GQA 和 MQA 的收益，首先体现在 `n_kv_heads` 下降；MLA 则是把“要存什么”这件事改写掉。
- 真正的线上容量规划，不是只算 token 数，还要算 block 粒度带来的碎片。
- 只要系统支持前缀共享，就应该把“共享前缀命中率”当成和显存预算同等重要的指标。

## 10. 继续深入的阅读顺序

1. 先回到 `../notes/kv-cache/formula-to-code-walkthrough.md`，把 block allocator、fork、append 这些接口走一遍。
2. 再看 `pagedattention-math.md`，理解为什么逻辑连续和物理离散可以同时成立。
3. 如果你接下来要优化显存，再接 `kv-compression-math.md` 和 `kv-eviction-math.md`。
