# 注意力机制总览：入口页，而不是百科全书

> 这页不再试图把 MHA、GQA、MLA、DSA、Linear Attention 的所有公式塞进一张超大表里。它现在只做两件事：第一，告诉你每条技术线到底在解决什么问题；第二，把你送到对应的深入页、源码页和性能页。

## 这页怎么用

- 你想先抓主线：先看这一页，再跳到对应专题深挖页。
- 你想准备面试：先看“什么时候该提哪种机制”，再看每页最后的“一句话结论”。
- 你想对源码：先读 [formula-to-code-walkthrough.md](formula-to-code-walkthrough.md)，再回到这页选专题。
- 你想排查长上下文瓶颈：优先看 GQA、MLA、DSA，再接 [../serving/formula-to-code-walkthrough.md](../serving/formula-to-code-walkthrough.md)。

## 五条技术线先一句话讲清楚

| 机制 | 它主要解决什么 | 它到底改了哪里 | 最典型的收益 | 最该继续读哪一页 |
|------|----------------|----------------|--------------|------------------|
| MHA | 作为基线，表达力强 | 每个 head 都有独立的 Q / K / V | 最完整、最好讲清楚 | [multi-head-divergence.md](multi-head-divergence.md) |
| GQA | KV Cache 太大、decode 带宽太高 | 保留很多 query head，但把 K / V 按组共享 | 显存和带宽明显下降 | [mha-vs-gqa-full-derivation.md](mha-vs-gqa-full-derivation.md) |
| MLA | GQA 还不够省，想继续压缩 KV 表示 | 把 KV 存成共享潜变量，再按 head 重建 | 进一步压缩缓存，适合长上下文 | [mha-vs-mla-full-derivation.md](mha-vs-mla-full-derivation.md) |
| DSA | 长上下文里“不是所有 token 都值得看” | 先用索引器筛 token，再做稀疏注意力 | 把扫描成本从全量改成候选集 | [mha-vs-dsa-full-derivation.md](mha-vs-dsa-full-derivation.md) |
| Linear Attention | 二次复杂度太贵 | 用核技巧改写注意力，避免显式全矩阵 | 复杂度更低，但表达形式变了 | [mha-vs-linear-attention-full-derivation.md](mha-vs-linear-attention-full-derivation.md) |

## 先按问题选，而不是按论文名选

### 1. 如果你在想“为什么 decode 越跑越慢”

优先看：

1. [mha-vs-gqa-full-derivation.md](mha-vs-gqa-full-derivation.md)
2. [mha-vs-mla-full-derivation.md](mha-vs-mla-full-derivation.md)
3. [../serving/formula-to-code-walkthrough.md](../serving/formula-to-code-walkthrough.md)

这一组主要回答：KV Cache 占用、KV 扫描字节数、memory-bound decode、TPOT 为什么恶化。

### 2. 如果你在想“为什么多头不是越多越浪费”

优先看：

1. [multi-head-divergence.md](multi-head-divergence.md)
2. [mha-vs-gqa-full-derivation.md](mha-vs-gqa-full-derivation.md)
3. [formula-to-code-walkthrough.md](formula-to-code-walkthrough.md)

这一组主要回答：head 为什么会分化、共享 KV 为什么还能保住质量、什么时候共享过头会伤表达力。

### 3. 如果你在想“长上下文到底该靠压缩，还是靠稀疏”

优先看：

1. [mha-vs-mla-full-derivation.md](mha-vs-mla-full-derivation.md)
2. [mha-vs-dsa-full-derivation.md](mha-vs-dsa-full-derivation.md)
3. [long-context.md](long-context.md)

这一组主要回答：MLA 是“把每个 token 变轻”，DSA 是“减少需要看的 token 数量”，两者不是一回事。

### 4. 如果你在想“FlashAttention、Linear、DSA 到底是不是一类东西”

不是。

- FlashAttention：还是 dense attention，只是把 IO 做对。
- GQA / MLA：还是 dense attention，只是把 KV 表示做轻。
- DSA：减少真正参与注意力计算的 token 集合。
- Linear Attention：连注意力的代数形式都改掉了。

所以它们不是互斥路线，更像不同层面的优化旋钮。

## 统一比较时，真正该比哪几项

这页以后统一只建议比下面 6 项，不再堆大而全公式表：

| 比较维度 | 你真正该问什么 |
|----------|----------------|
| KV 表示 | 每个 token 要存多少东西 |
| 访问范围 | 每步 decode 要看全部历史，还是只看候选集 |
| 计算形态 | 还是 dense softmax，还是换了代数形式 |
| 主要收益 | 省显存、省带宽、省算力，还是降低尾延迟 |
| 主要代价 | 精度风险、重建开销、索引误判，还是训练不稳定 |
| 工程位置 | 改模型结构、改缓存结构、改调度，还是改 kernel |

## 源码和文档怎么对起来

- 基线实现： [../../src/attention/mha_gqa.py](../../src/attention/mha_gqa.py)
- RoPE 与归一化： [../../src/attention/rope_rmsnorm.py](../../src/attention/rope_rmsnorm.py)
- dense attention 的 IO 视角： [../../src/attention/flash_attn_sim.py](../../src/attention/flash_attn_sim.py)
- 公式到源码总导读： [formula-to-code-walkthrough.md](formula-to-code-walkthrough.md)
- 长上下文与系统侧后果： [../serving/formula-to-code-walkthrough.md](../serving/formula-to-code-walkthrough.md)

## 最推荐的分专题阅读顺序

1. 先读 [multi-head-divergence.md](multi-head-divergence.md)，建立“为什么需要多头”的直觉。
2. 再读 [mha-vs-gqa-full-derivation.md](mha-vs-gqa-full-derivation.md)，理解第一层 KV 压缩。
3. 接着读 [mha-vs-mla-full-derivation.md](mha-vs-mla-full-derivation.md)，理解更激进的潜空间压缩。
4. 然后读 [mha-vs-dsa-full-derivation.md](mha-vs-dsa-full-derivation.md)，理解“减少访问集合”这条路线。
5. 最后读 [mha-vs-linear-attention-full-derivation.md](mha-vs-linear-attention-full-derivation.md)，看完全不同的代数改写路线。

## 这一页记住一句话

> 注意力机制的优化不是一条单线演化史，而是三条主线并行推进：一条在优化 KV 表示，一条在优化访问集合，一条在优化 kernel 和 IO。先分清楚你在动哪一层，再谈公式、实现和收益，文档就不会重新写成一本教材。
