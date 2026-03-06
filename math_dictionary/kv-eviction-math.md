# KV Eviction：从 LRU / LFU 到公平配额

> 这页关心的不是“缓存策略大全”，而是线上真正会触发的那个瞬间：新 token 还要继续写入，但 block 预算已经不够了，系统到底该回收谁、为什么先回收它、回收之后又会付出什么代价。读完这页，你应该能把驱逐问题直接翻译成可执行的评分函数和配额规则。

## 1. 先把驱逐问题写成预算约束

当新请求进入，或者已有序列继续追加 token 时，系统需要先看 block 预算是否会溢出。最常用的写法是：

$$
\text{overflow\_blocks} = \max\!\left(0, \text{used\_blocks} + \Delta b - \text{budget\_blocks}\right)
$$

只要这项大于零，系统就必须做一件事：从当前缓存里选出若干 victim，把它们的 block 释放掉。

这一步和压缩不同。压缩是在“尽量继续保留信息”的前提下，把表示变便宜；驱逐则是在预算已经顶到墙的时候，明确地决定“谁要先被请出去”。

## 2. 一个更接近工程现实的目标函数

如果把驱逐策略抽象成优化问题，更有用的写法不是某个单一指标，而是三项折中：

$$
\min\ \alpha \cdot \text{miss\_rate} + \beta \cdot \text{recompute\_cost} + \gamma \cdot \text{unfairness}
$$

这三项分别对应：

- `miss_rate`：下次需要它时已经不在缓存里。
- `recompute_cost`：被驱逐后若要恢复，需要付出的 prefill 或回填代价。
- `unfairness`：多租户场景下，某个租户是不是长期吃掉了超额预算。

不同系统的区别，不在于有没有这三项，而在于各项权重偏向哪里。

## 3. LRU：把“最近有没有用过”写成分数

LRU 的核心非常直接：越久没被访问，越应该优先驱逐。分数可写成：

$$
\text{score}_{\mathrm{LRU}}(i) = t_{\mathrm{now}} - t_{\mathrm{last\_access}}(i)
$$

于是 victim 就是年龄最大的条目：

$$
\text{victim}_{\mathrm{LRU}} = \arg\max_i\ \text{score}_{\mathrm{LRU}}(i)
$$

它的优点是简单、稳定、实现代价低；它的缺点是完全不关心“这个条目是不是虽然不常访问，但一旦命中就很重要”。

仓库里的对应实现是 `../src/kv_cache/eviction/policies.py` 的 `LRUPolicy.select_victim()`。它直接按 `last_access_step` 最小的序列做选择。

## 4. LFU：把“历史上用过多少次”也算进去

如果系统里存在稳定热点，只看最近一次访问就不够了。LFU 更自然的写法是先比较访问次数，再用 LRU 做 tie-break：

$$
\text{victim}_{\mathrm{LFU}} = \arg\min_i\ \left(\text{use\_count}(i),\ t_{\mathrm{last\_access}}(i)\right)
$$

这条式子的意思是：

- 优先驱逐总访问次数更少的对象。
- 如果访问次数一样，再驱逐更久没被用过的对象。

仓库里的对应实现是 `../src/kv_cache/eviction/policies.py` 的 `LFUPolicy.select_victim()`，排序键正是 `use_count`、`last_access_step` 和序列标识。

## 5. Fair Quota：先看谁超配，再在超配租户里挑 victim

多租户系统里，最怕的不是局部 miss，而是某个大租户把 block 几乎全吃掉。此时更合适的写法是先给每个租户一份预算。

如果租户 `t` 的权重是 `w_t`，总 block 预算是 `B_total`，那么它的配额可以写成：

$$
\text{quota}_t = \frac{w_t}{\sum_{t'} w_{t'}} \times B_{\mathrm{total}}
$$

而租户当前实际占用为：

$$
\text{usage}_t = \sum_{i \in \text{tenant}(t)} \text{blocks}_i
$$

先找出超配最多的租户：

$$
 t^* = \arg\max_t\ \left(\text{usage}_t - \text{quota}_t\right)
$$

再在这个租户内部按 LRU 选 victim：

$$
\text{victim}_{\mathrm{fair}} = \arg\min_{i \in \text{tenant}(t^*)}\ t_{\mathrm{last\_access}}(i)
$$

这样做的重点不是“绝对公平”，而是先防止一个租户持续挤压其他租户的生存空间。

仓库里的对应实现是：

- `../src/kv_cache/eviction/policies.py` 的 `FairPolicy._tenant_of()`
- `../src/kv_cache/eviction/policies.py` 的 `FairPolicy._quotas()`
- `../src/kv_cache/eviction/policies.py` 的 `FairPolicy.select_victim()`

## 6. 驱逐成本为什么不能只看命中率

同样是一次驱逐，不同 victim 的代价可能完全不同。最粗但很实用的近似，是把重算成本和被驱逐 token 数直接挂钩：

$$
\text{recompute\_tokens}(i) \approx \text{num\_blocks}(i) \times \text{block\_size}
$$

如果 prefill 吞吐近似稳定，还可以继续改写成时间成本：

$$
\text{recompute\_time}(i) \approx \frac{\text{recompute\_tokens}(i)}{\text{prefill\_throughput}}
$$

这就解释了一个线上常见现象：

- 驱逐一个很旧但很长的序列，未必比驱逐几个短小的边缘序列更便宜。
- 如果回填非常慢，那么“命中率看起来还行”不等于端到端时延真的还行。

## 7. 这些公式如何落到仓库数据结构

驱逐策略要成立，底层元数据必须先存在。这个仓库里最值得对照的是两层：

### 7.1 元数据从哪里来

- `../src/kv_cache/core.py` 的 `SequenceKVCache.num_blocks()` 提供每条序列的 block 占用。
- `../src/kv_cache/core.py` 里的 `last_access_step` 和 `use_count` 是 LRU / LFU 的核心输入。
- `../src/kv_cache/core.py` 的 `release()` 负责真正把 victim 占用的 block 归还给分配器。

### 7.2 策略如何消费这些元数据

- `LRUPolicy` 只看“最后访问时间”。
- `LFUPolicy` 看“访问次数 + 最后访问时间”。
- `FairPolicy` 先按租户聚合 `num_blocks()`，再根据配额与超配量做二段式选择。

如果你想把驱逐和测试、分页、压缩串起来看，建议继续读：

- `../notes/kv-eviction/formula-to-code-walkthrough.md`
- `kv-memory.md`
- `kv-compression-math.md`

## 8. 什么时候该用哪一类策略

| 场景 | 更适合的策略 | 原因 |
|------|--------------|------|
| 单租户、稳定业务、实现要尽量简单 | LRU | 成本最低，行为可预测 |
| 热点非常稳定、重复访问明显 | LFU | 能保住高频热点 |
| 多租户共享集群 | Fair quota | 先防止单租户吃满预算 |
| 强依赖历史重要性、需要内容感知 | 驱逐前加压缩或打分 | 只靠 recency / frequency 不够 |

## 9. 驱逐和压缩不要混成一回事

这两件事很容易在讨论里混淆，但职责并不一样：

- 压缩解决的是“同样的信息能不能更便宜地存”。
- 驱逐解决的是“预算不够时必须先舍弃谁”。
- 一个成熟系统通常会先做分页和压缩，再在溢出时用驱逐策略兜底。

所以真正的顺序往往是：

1. 先用 `kv-memory.md` 算清预算。
2. 再用 `kv-compression-math.md` 尽量把每个 token 变便宜。
3. 最后才在预算仍然不足时启用 `kv eviction`。

## 10. 这页真正该记住什么

- LRU、LFU、Fair 的差别，本质上是评分函数和优先级规则不同。
- 多租户环境里，不显式写出配额公式，就很难真正谈公平。
- 线上策略评估不能只看命中率，还要看重算成本和尾延迟外溢。
- 驱逐是最后一道闸门，不应该替代容量规划和压缩本身。
