# 从公式到源码：KV Cache / PagedAttention / 压缩 / 驱逐对照手册

> 这页专门解决“知道 KV Cache 很重要，但一到 block allocator、量化器、驱逐器就连不起来”的问题。阅读顺序固定为：先算容量账本，再看 block 分配，再看共享与压缩，最后看驱逐策略。

## 这页覆盖哪些源码

- [../../src/kv_cache/core.py](../../src/kv_cache/core.py)：`BlockAllocator`、`PagedKVCacheManager`、`fragmentation()`、`fork()`。
- [../../src/kv_cache/compression/quantizer.py](../../src/kv_cache/compression/quantizer.py)：对称 / 非对称 per-channel 量化与误差统计。
- [../../src/kv_cache/eviction/policies.py](../../src/kv_cache/eviction/policies.py)：LRU、LFU、多租户公平驱逐。

## 1. 先把 KV Cache 的容量账本写清楚

### 1.1 单 token 到底要占多少字节

对一层 attention 来说，每个 token 都要保留一份 Key 和一份 Value。若 KV 头数为 $H_{\text{KV}}$，每个头维度为 $d_{\text{head}}$，每个元素字节数为 $s$，则

$$
\text{bytes/token/layer} = 2 \times H_{\text{KV}} \times d_{\text{head}} \times s
$$

乘上层数 $L$，得到全模型每个 token 的 KV 占用：

$$
\text{bytes\_per\_token} = 2 \times L \times H_{\text{KV}} \times d_{\text{head}} \times s
$$

于是长度为 $T$ 的单条序列，KV 总占用为

$$
M_{\text{KV}}(T) = T \times \text{bytes\_per\_token}
$$

这就是为什么 GQA 能直接省显存：它不改 $L$、不改 $d_{\text{head}}$，而是把 $H_{\text{KV}}$ 变小。

### 1.2 为什么 block 数一定是向上取整

PagedAttention 不按“整条连续序列”分配，而是把序列切成固定大小的 block。若每个 block 能放 $B_{\text{block}}$ 个 token，则长度为 $T$ 的序列需要的块数是

$$
n_{\text{blocks}} = \left\lceil \frac{T}{B_{\text{block}}} \right\rceil
$$

推导很直接：前 $n-1$ 个块都被填满，最后一个块允许不满，但仍然需要真实存在，所以必须向上取整。

源码对应的是 [../../src/kv_cache/core.py](../../src/kv_cache/core.py) 里的 `_blocks_needed()`：

```python
@staticmethod
def _blocks_needed(num_tokens: int, block_size: int) -> int:
    if num_tokens <= 0:
        return 0
    return int(math.ceil(num_tokens / block_size))
```

这段实现正对应上面的向上取整公式，没有额外魔法。

### 1.3 尾块浪费和内部碎片率怎么定义

固定块大小的直接代价是尾块通常填不满。单条序列的尾块浪费可以写成

$$
\text{tail\_waste}(T) = \left(B_{\text{block}} - (T \bmod B_{\text{block}})\right) \bmod B_{\text{block}}
$$

整个系统里，更常用的是“已分配块中空闲 slot 的比例”，也就是内部碎片率：

$$
\text{fragmentation} = \frac{\sum_b \text{free\_slots}(b)}{\sum_b \text{capacity}(b)}
$$

源码里对应 `BlockAllocator.fragmentation()`：

```python
def fragmentation(self) -> float:
    used_blocks = [b for b in self._blocks if b.ref_count > 0]
    if not used_blocks:
        return 0.0
    total_slots = sum(b.capacity for b in used_blocks)
    wasted = sum(b.free_slots for b in used_blocks)
    return wasted / total_slots
```

注意这里统计的是“已使用物理块”的空闲比例，而不是整个 block pool 的空闲比例。也就是说，它量化的是分页后的内部浪费，而不是 allocator 总体利用率。

## 2. 从生命周期看 Paged KV Cache 的代码长什么样

### 2.1 新请求到来时：先算块数，再逐块填充

对初始输入长度 $T_{\text{init}}$，先算

$$
n_{\text{init}} = \left\lceil \frac{T_{\text{init}}}{B_{\text{block}}} \right\rceil
$$

然后按顺序把 token 填进这 $n_{\text{init}}$ 个物理块。对应源码：

```python
def allocate_for_sequence(self, seq_id: str, num_tokens: int) -> SequenceKVCache:
    self._step += 1
    n = self._blocks_needed(num_tokens, self.block_size)
    blocks = self.allocator.allocate_n(n)

    remaining = num_tokens
    for blk in blocks:
        fill = min(remaining, blk.capacity)
        blk.filled = fill
        remaining -= fill

    seq = SequenceKVCache(
        seq_id=seq_id,
        block_table=blocks,
        num_tokens=num_tokens,
        last_access_step=self._step,
        use_count=1,
    )
    self.sequences[seq_id] = seq
    return seq
```

这段代码最值得对照的点有两个：

- `block_table` 是逻辑视图，保存“这条序列用了哪些物理块”。
- `filled` 只记录块内写了多少 token，因此尾块天然允许不满。

### 2.2 Decode 追加 token 时：先填尾块，再申请新块

设当前序列最后一个块还有 $f_{\text{last}}$ 个空位，本轮 decode 需要追加 $\Delta T$ 个 token，则新增块数是

$$
\Delta n_{\text{blocks}} = \left\lceil \frac{\max(0, \Delta T - f_{\text{last}})}{B_{\text{block}}} \right\rceil
$$

它的含义是：先把最后一个未满块吃满，只有剩下的 token 才需要申请新块。

源码里完全按这个顺序执行：

```python
def append_tokens(self, seq: SequenceKVCache, num_new_tokens: int) -> None:
    self._step += 1
    seq.last_access_step = self._step
    seq.use_count += 1

    remaining = num_new_tokens
    if seq.block_table:
        last_blk = seq.block_table[-1]
        fill = min(remaining, last_blk.free_slots)
        last_blk.filled += fill
        remaining -= fill

    while remaining > 0:
        blk = self.allocator.allocate()
        if blk is None:
            raise RuntimeError("OOM: no free blocks")
        fill = min(remaining, blk.capacity)
        blk.filled = fill
        remaining -= fill
        seq.block_table.append(blk)

    seq.num_tokens += num_new_tokens
```

所以这段实现不是“每次 decode 都重算整条序列”，而是增量地修改尾块和 block table。这正是 PagedAttention 的工程意义：把连续扩展序列变成局部块操作。

### 2.3 释放时为什么只需要遍历 block table

对序列 $s$，释放的工作量和它拥有的块数成正比：

$$
\text{release\_cost}(s) = O(n_{\text{blocks}}(s))
$$

因为系统不需要在一大片连续地址中做 compaction，只需要遍历这条序列的 `block_table`，逐块减 `ref_count`，必要时归还 free pool。

这也是分页设计在服务系统里比“连续大块分配”更稳的原因：释放路径短，而且不需要搬迁别的序列。

## 3. Prefix 共享与 Copy-on-Write 为什么能省显存

### 3.1 共享前缀的收益怎么写成公式

设一段公共前缀长度为 $T_{\text{prefix}}$，有 $k$ 条分支都复用它。若不共享，则前缀部分的显存是

$$
M_{\text{no-share}} = k \times M_{\text{KV}}(T_{\text{prefix}})
$$

若共享同一份物理块，则前缀部分只保留一份：

$$
M_{\text{share}} = M_{\text{KV}}(T_{\text{prefix}})
$$

仅前缀这一段的节省量就是

$$
\Delta M = (k - 1) \times M_{\text{KV}}(T_{\text{prefix}})
$$

这正是 Prefix Caching、beam search 分叉、speculative decoding 共享前缀的核心收益来源。

### 3.2 `fork()` 为什么只加引用计数

[../../src/kv_cache/core.py](../../src/kv_cache/core.py) 的 `fork()` 没有复制物理块，只是递增 `ref_count` 并浅拷贝 `block_table`：

```python
def fork(self, src: SequenceKVCache, new_seq_id: str) -> SequenceKVCache:
    self._step += 1
    for blk in src.block_table:
        blk.ref_count += 1

    new_seq = SequenceKVCache(
        seq_id=new_seq_id,
        block_table=list(src.block_table),
        num_tokens=src.num_tokens,
        last_access_step=self._step,
        use_count=1,
    )
    self.sequences[new_seq_id] = new_seq
    return new_seq
```

也就是说：

- 共享发生在“物理块层面”。
- 隔离发生在“逻辑 block table 层面”。
- 真正需要复制时，应该等某个共享块被写入，这就是 Copy-on-Write 的思想。

虽然这份最小实现没有把“写时复制”完全展开成额外函数，但 `ref_count` 已经把最关键的共享语义表达清楚了。

## 4. KV 压缩：量化公式如何落到代码

### 4.1 对称 per-channel 量化

对第 $c$ 个通道，若采用对称量化，则

$$
\text{scale}_c = \frac{\max |x_c|}{q_{\max}}
$$

$$
q_c = \operatorname{clip}\left(\operatorname{round}\left(\frac{x_c}{\text{scale}_c}\right), -q_{\max}, q_{\max}\right)
$$

这和 [../../src/kv_cache/compression/quantizer.py](../../src/kv_cache/compression/quantizer.py) 的实现一一对应：

```python
qmax = (1 << (bits - 1)) - 1
abs_max = np.max(np.abs(flat), axis=0)
abs_max = np.clip(abs_max, a_min=1e-8, a_max=None)
scale = abs_max / qmax

quantized = np.round(flat / scale).astype(np.int8)
quantized = np.clip(quantized, -qmax, qmax).astype(np.int8)
```

这里先 `moveaxis(..., axis, -1)` 再 `reshape(-1, C)`，本质上是在把“最后一维当作通道维”来做 per-channel 量化。

### 4.2 非对称 per-channel 量化

若数据分布不以 0 为中心，则更常见的写法是

$$
\text{scale}_c = \frac{\max(x_c) - \min(x_c)}{q_{\max} - q_{\min}}
$$

$$
\text{zero\_point}_c = \operatorname{round}\left(q_{\min} - \frac{\min(x_c)}{\text{scale}_c}\right)
$$

$$
q_c = \operatorname{clip}\left(\operatorname{round}\left(\frac{x_c}{\text{scale}_c}\right) + \text{zero\_point}_c, q_{\min}, q_{\max}\right)
$$

源码对应：

```python
c_min = np.min(flat, axis=0)
c_max = np.max(flat, axis=0)
c_range = np.clip(c_max - c_min, a_min=1e-8, a_max=None)
scale = c_range / (qmax - qmin)
zero_point = np.round(qmin - c_min / scale).astype(np.int8)

quantized = np.round(flat / scale).astype(np.int32) + zero_point.astype(np.int32)
quantized = np.clip(quantized, qmin, qmax).astype(np.int8)
```

### 4.3 反量化和误差指标

反量化公式是

$$
\hat{x}_c = (q_c - \text{zero\_point}_c) \times \text{scale}_c
$$

源码里的 `dequantize()` 正是这一步：

```python
zp = qt.zero_point.astype(np.float32)
result = (flat - zp) * qt.scale
```

而 `quantization_error()` 进一步把误差整理成 `max_abs_error`、`mean_abs_error`、`rmse` 和 `compression_ratio`，相当于把“精度损失”和“显存收益”放在同一张表里对比。

## 5. 驱逐策略：公式和选择规则怎么对应

### 5.1 LRU：按最近访问时间最小者淘汰

LRU 的规则可以写成

$$
\operatorname{victim}_{\text{LRU}} = \arg\min_s \; \text{last\_access\_step}(s)
$$

源码：

```python
return min(sequences.keys(), key=lambda k: (sequences[k].last_access_step, k))
```

### 5.2 LFU：先看访问次数，再用 LRU 打平

LFU 可以写成

$$
\operatorname{victim}_{\text{LFU}} = \arg\min_s \; \big(\text{use\_count}(s), \text{last\_access\_step}(s)\big)
$$

源码：

```python
return min(
    sequences.keys(),
    key=lambda k: (sequences[k].use_count, sequences[k].last_access_step, k),
)
```

### 5.3 Fair policy：先找超配租户，再在租户内做 LRU

设租户 $t$ 的权重为 $w_t$，总 block 预算为 $B_{\text{total}}$，则它的配额为

$$
\text{quota}_t = \frac{w_t}{\sum_j w_j} \times B_{\text{total}}
$$

若租户当前占用为 `usage_t`，那么只在超配租户集合

$$
\mathcal{O} = \{ t : \text{usage}_t > \text{quota}_t \}
$$

中选受害者，并优先淘汰超配最多的租户内最老序列。

源码可以直接读成这个逻辑：

```python
quotas = self._quotas(sequences)
over = [t for t in usage if usage[t] > quotas.get(t, 0.0)]

if over:
    victim_t = max(over, key=lambda t: (usage[t] - quotas.get(t, 0.0), t))
    candidates = [sid for sid in sequences if self._tenant_of(sid) == victim_t]
    return min(candidates, key=lambda k: (sequences[k].last_access_step, k))

return min(sequences.keys(), key=lambda k: (sequences[k].last_access_step, k))
```

所以 Fair policy 不是“全局最旧”，而是“先公平，再最旧”。这在多租户 serving 里比纯 LRU 更符合资源隔离目标。

## 6. 建议的源码阅读顺序

1. 先读 [../../math_dictionary/kv-memory.md](../../math_dictionary/kv-memory.md)，把 `bytes_per_token` 和并发公式算熟。
2. 再读 [../../src/kv_cache/core.py](../../src/kv_cache/core.py) 的 `_blocks_needed()`、`allocate_for_sequence()`、`append_tokens()`、`fork()`。
3. 接着读 [../../src/kv_cache/compression/quantizer.py](../../src/kv_cache/compression/quantizer.py)，把量化公式和 `scale` / `zero_point` 变量对上。
4. 最后读 [../../src/kv_cache/eviction/policies.py](../../src/kv_cache/eviction/policies.py)，把 LRU / LFU / Fair 对照成三种不同的目标函数。

## 这一页记住一句话

> KV Cache 本质上是一笔线性容量账：先用 $2 L H_{\text{KV}} d_{\text{head}} s$ 算清 bytes/token，再用 block 把线性空间切成可管理的页，接着用 Copy-on-Write 复用前缀、用量化压缩字节、用驱逐策略守住预算。
