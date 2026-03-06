# MHA vs DSA：从全量扫描到先检索再注意力

> 这页只保留 DSA 最值得讲清楚的主线：dense attention 的问题不是每个 token 太重，而是历史太长时“要看的 token 太多”。DSA 的核心动作不是继续压缩 K / V，而是先筛，再算。

## 1. 先把问题抽象对

标准 MHA 在长上下文里的主要问题是：对当前位置做注意力时，要把历史里几乎所有 token 都参与打分。于是 Prefill 和 Decode 的计算范围都随上下文长度增长。

如果当前长度记为 `T`，dense self-attention 的典型复杂度量级是：

$$
\mathcal{O}(T^2)
$$

对于 decode 场景，如果一次只生成一个 token，那么每一步仍要扫过全部历史，成本量级近似是：

$$
\mathcal{O}(T)
$$

DSA 想改的正是这一点：不是“把每个历史 token 存得更轻”，而是“不要每次都看全部历史 token”。

## 2. DSA 的基本结构：两阶段，而不是一步到位

DSA 最适合用“两阶段”理解：

1. 先用一个轻量索引器给历史 token 打相关性分；
2. 再只对分数最高的候选集做真正的注意力。

如果把第 `t` 个位置要保留的候选集合记作 `S_t`，集合大小记作 `k`，那么 DSA 的关键动作可以写成：

$$
S_t = \operatorname{TopK}(I_{t,:}, k)
$$

其中 `I_t` 是索引器给所有历史位置打出的分数向量。

在候选集合上再做真正的注意力：

$$
O_t = \operatorname{Softmax}\left(\frac{Q_t K_{S_t}^\top}{\sqrt{d_{\mathrm{head}}}}\right)V_{S_t}
$$

这就是 DSA 的核心：先检索，再注意力。

## 3. 为什么它和 MLA 不是同一类优化

MLA 的动作是“压缩每个 token 的表示”。

DSA 的动作是“减少要访问的 token 数量”。

所以两者分别在优化不同账本：

- MLA：降低单 token 的缓存成本和带宽成本；
- DSA：降低每一步真正参与计算的 token 数量；
- FlashAttention：降低 dense attention 的 IO 成本，但不改变参与集合。

如果你把这三者混成一类，就很容易把文档重新写成一张巨大的教材总表。

## 4. 复杂度为什么会从全量变成候选集

dense attention 的一个直观量级是：

$$
\mathcal{O}(T^2)
$$

如果每个位置只保留 `k` 个候选，且 `k` 远小于 `T`，那么稀疏阶段的主复杂度会更接近：

$$
\mathcal{O}(Tk)
$$

对 decode 来说，单步扫描量也更像：

$$
\mathcal{O}(k)
$$

当然，这不是免费午餐，因为你还要为索引器本身付费。更完整的账本是：

$$
\text{Total Cost} = \text{Indexer Cost} + \text{Sparse Attention Cost}
$$

DSA 真正成立的前提是：索引器足够轻，而且筛出的候选集足够准。

## 5. DSA 成功与否，关键不是公式，而是召回

DSA 最怕的问题不是“公式写错”，而是“索引器没把该看的 token 召回”。

如果真实重要位置不在 `TopK` 里，那么后面的稀疏注意力再精确也没用。于是 DSA 的核心风险变成：

- 候选集太小，召回不够；
- 索引器太弱，打分失真；
- 稀疏性一上来，质量先掉。

也正因为这个原因，DSA 往往不是一个“只靠闭式公式就能说清楚”的方法，而是强依赖索引器设计和实际数据分布。

## 6. 为什么它特别适合长上下文专题

随着上下文继续增长，dense attention 的问题会越来越明显：

- 扫描范围更大；
- KV 带宽压力更大；
- TPOT 更容易随上下文长度恶化。

DSA 选择的路线是：不再默认所有历史都重要，而是把“重要位置识别”单独抽成一个模块。这个思路和检索增强、稀疏激活、本地窗口加全局 token 的路线都很接近。

## 7. 和 GQA / MLA / FlashAttention 分别怎么配合

- 与 GQA 配合：一边减少 KV 的独立份数，一边减少访问集合。
- 与 MLA 配合：一边压缩单 token 表示，一边减少访问 token 数量。
- 与 FlashAttention 配合：即使候选集已经变小，块化和 IO 优化仍然有价值。

因此 DSA 更像“访问集合优化层”，而不是一切注意力优化的替代品。

## 8. 在仓库里怎么读这条线

仓库里没有完整 DSA 内核实现，所以更推荐这样读：

- dense 基线和张量直觉： [../../src/attention/mha_gqa.py](../../src/attention/mha_gqa.py)
- IO 和 dense kernel 视角： [../../src/attention/flash_attn_sim.py](../../src/attention/flash_attn_sim.py)
- 长上下文和系统后果： [long-context.md](long-context.md)
- serving 里的 decode / KV 带宽账本： [../serving/formula-to-code-walkthrough.md](../serving/formula-to-code-walkthrough.md)

你真正要理解的是“先筛，再算”这条结构，而不是在这里找一份不存在的完整实现。

## 9. 面试时最值得说的三句话

- DSA 不只是稀疏 attention，而是把相关性检索单独前置成一个索引阶段。
- 它解决的是“访问集合太大”，不是“每个 token 太重”。
- 它的主要风险是召回，而不是单纯的计算复杂度公式。

## 这一页记住一句话

> DSA 的本质是把 dense attention 的“全量扫描”改成“候选召回 + 稀疏精算”。如果 MLA 在省每个 token 的成本，那么 DSA 就是在省每一步真正要看的 token 数量。
