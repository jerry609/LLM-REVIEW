# MHA vs GQA：从 KV Cache 到 decode 带宽账本

> 这页不再把 GQA 写成一篇“从训练史到论文史”的长综述，而是只回答四个问题：GQA 到底共享了什么，为什么它能大幅节省 KV Cache，为什么质量通常还能接受，以及它和源码里的最小实现怎么对上。

## 1. 问题先讲清楚

MHA 的基线做法是：每个 query head 都有自己独立的 key 和 value。这样表达力最强，但 KV Cache 也最贵。

GQA 的核心变化只有一句话：

- query 头仍然很多；
- key 和 value 按组共享；
- 一个 KV 组服务多个 query 头。

如果你只记一句工程结论，就是：GQA 主要是在保留 query 头数量的前提下，压缩 KV 表示。

## 2. MHA 的基线公式

设输入 hidden states 为：

$$
X \in \mathbb{R}^{B \times T \times d_{\mathrm{model}}}
$$

对第 `i` 个 head，MHA 的投影是：

$$
Q_i = X W_i^Q,
\qquad
K_i = X W_i^K,
\qquad
V_i = X W_i^V
$$

单头输出写成：

$$
O_i = \operatorname{Softmax}\left(\frac{Q_i K_i^\top}{\sqrt{d_{\mathrm{head}}}}\right)V_i
$$

全部 head 拼接后再经过输出投影：

$$
O = \operatorname{Concat}(O_1, \dots, O_{n_{\mathrm{heads}}}) W^O
$$

## 3. GQA 的核心改写

设总 query head 数为 `n_heads`，KV 组数为 `n_kv_heads`。对第 `i` 个 query head，记它所属的 KV 组为 `g(i)`，则 GQA 的投影改成：

$$
Q_i = X W_i^Q
$$

$$
K_g = X W_g^K,
\qquad
V_g = X W_g^V,
\qquad g = 1, \dots, n_{\mathrm{kv\_heads}}
$$

于是第 `i` 个 query head 实际访问的是所属组的 K 和 V：

$$
O_i = \operatorname{Softmax}\left(\frac{Q_i K_{g(i)}^\top}{\sqrt{d_{\mathrm{head}}}}\right)V_{g(i)}
$$

这里真正共享的是 `K` 和 `V`，不是 `Q`。这也是为什么 GQA 在工程上常被看成“对 decode 友好”的折中，而不是彻底砍掉多头表达力。

## 4. KV Cache 为什么能直接省下来

在 decode 阶段，最关键的账本不是 FLOPs，而是每个 token 要缓存多少 K 和 V。

MHA 的 KV 表示规模可以写成：

$$
M_{\mathrm{KV}}^{\mathrm{MHA}} \propto 2 \times n_{\mathrm{heads}} \times d_{\mathrm{head}} \times T
$$

GQA 的 KV 表示规模则变成：

$$
M_{\mathrm{KV}}^{\mathrm{GQA}} \propto 2 \times n_{\mathrm{kv\_heads}} \times d_{\mathrm{head}} \times T
$$

因此两者的比值是：

$$
\frac{M_{\mathrm{KV}}^{\mathrm{GQA}}}{M_{\mathrm{KV}}^{\mathrm{MHA}}}
= \frac{n_{\mathrm{kv\_heads}}}{n_{\mathrm{heads}}}
$$

如果每个 KV 组服务 `G` 个 query head，也就是：

$$
G = \frac{n_{\mathrm{heads}}}{n_{\mathrm{kv\_heads}}}
$$

那么又可以写成：

$$
\frac{M_{\mathrm{KV}}^{\mathrm{GQA}}}{M_{\mathrm{KV}}^{\mathrm{MHA}}} = \frac{1}{G}
$$

这就是 GQA 最重要的结论：KV Cache 规模按组共享比例直接缩小。

## 5. 为什么它更像“省带宽”，而不是“重写注意力”

GQA 带来的收益主要集中在 decode 阶段：

- 缓存里存的 K / V 更少；
- 每步要扫的 KV 字节数更少；
- 因而 TPOT 更容易下降。

但它并没有把注意力机制改成另一种代数形式。query 头还是要逐头计算注意力，只是它们访问的是共享后的 K / V。

所以你可以把 GQA 理解成：

- 不是在改 softmax 本身；
- 不是在减少 query 头数；
- 而是在保留多头 query 的前提下，压缩 decode 的缓存和带宽成本。

## 6. 为什么质量通常还能保住

GQA 的质量没有像 MQA 那样更容易掉，关键在于它没有把所有 head 都压成一套共享 KV，而是只在组内共享。

直观上看：

- MHA：每个 head 都完全独立；
- MQA：所有 head 共用一套 K / V；
- GQA：介于两者之间，既共享，又保留组间差异。

因此 GQA 的核心折中不是“有没有共享”，而是“共享到什么粒度”。组太大，表达力会更受限；组适中，通常能在质量和成本之间取得更稳的平衡。

## 7. 对应到源码该看什么

这条主线最适合对照：

- [../../src/attention/mha_gqa.py](../../src/attention/mha_gqa.py)
- [formula-to-code-walkthrough.md](formula-to-code-walkthrough.md)
- [../../tests/test_attention.py](../../tests/test_attention.py)

在 [../../src/attention/mha_gqa.py](../../src/attention/mha_gqa.py) 里，最关键的不是一个复杂公式，而是“怎么把 KV 头扩回 query 头需要的形状”。你真正要看懂的是：

- `group_size` 的含义；
- 为什么需要 `np.repeat` 或等价广播；
- query 头数和 KV 头数不同的时候，张量是怎么对齐的。

## 8. 从 GQA 再往后该接哪条线

- 如果你想继续压缩 KV 表示：接 [mha-vs-mla-full-derivation.md](mha-vs-mla-full-derivation.md)
- 如果你想看服务系统影响：接 [../serving/formula-to-code-walkthrough.md](../serving/formula-to-code-walkthrough.md)
- 如果你想理解多头为什么还能有区分度：接 [multi-head-divergence.md](multi-head-divergence.md)

## 这一页记住一句话

> GQA 的本质不是“减少多头”，而是“减少 KV 的独立份数”。它把 query 头保留下来，把 key / value 按组共享，因此真正省下来的主要是 KV Cache 容量和 decode 带宽。
