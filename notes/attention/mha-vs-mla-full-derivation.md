# MHA vs MLA：从显存账本到潜空间重建

> 这页把 MLA 收紧成一条很清楚的主线：MHA 和 GQA 仍然在缓存显式 K / V，而 MLA 试图把每个 token 的缓存表示继续压到一个共享潜空间里。你真正需要看懂的是：它压缩了什么，又把什么计算挪回了运行时。

## 1. 先说结论

MLA 的核心不是“减少 query 头”，也不是“让所有 head 完全一样”，而是：

- 把每个 token 的 KV 信息先压到一个更小的潜变量；
- 真正做注意力时，再按 head 从潜变量里重建出需要的 K / V；
- RoPE 通常还会单独处理，避免位置编码和潜空间压缩彼此打架。

所以 MLA 的关键词是“共享潜变量 + 按 head 重建”。

## 2. MHA 的基线缓存为什么还不够轻

MHA 的缓存本质上是在为每个 head 显式保存 K 和 V：

$$
K_i = X W_i^K,
\qquad
V_i = X W_i^V
$$

因此每个 token 的缓存规模正比于：

$$
2 \times n_{\mathrm{heads}} \times d_{\mathrm{head}}
$$

GQA 已经把这个规模压到和 `n_kv_heads` 成正比，但如果你还希望更进一步压缩，就需要改变“缓存里到底存什么”。这正是 MLA 的出发点。

## 3. MLA 的第一步：先把 KV 压到潜空间

设共享潜空间维度为 `d_c`，则 MLA 先对输入做一次下投影：

$$
C^{KV} = X W^{DKV}
$$

其中：

$$
C^{KV} \in \mathbb{R}^{B \times T \times d_c}
$$

在缓存层面，系统保存的不再是每个 head 各自的显式 K / V，而是这个共享潜变量 `C_KV`。

## 4. MLA 的第二步：按 head 把 K / V 重建回来

真正做注意力时，再针对每个 head 做上投影：

$$
K_i^C = C^{KV} W_i^{UK}
$$

$$
V_i^C = C^{KV} W_i^{UV}
$$

这里的关键直觉是：

- 缓存层共享；
- 使用层分头；
- 节省的是存储，付出的代价是重建计算。

因此 MLA 不是白拿收益，而是在“省缓存”和“增重建”之间做交换。

## 5. 为什么 MLA 往往还要把位置编码解耦

如果把位置编码完全揉进共享潜空间，位置相关信息和内容信息会一起被压缩，重建时更容易互相干扰。于是很多 MLA 方案会把位置相关部分单独留出来。

一个常见写法是把 query 和 key 分成内容分支与位置分支：

$$
Q_i = [Q_i^C ; Q_i^R]
$$

$$
K_i = [K_i^C ; K^R]
$$

其中：

- `Q_i^C` 和 `K_i^C` 来自共享潜空间重建；
- `Q_i^R` 和 `K^R` 专门承载位置相关分量。

这样做的好处是：缓存压缩这件事，尽量只作用在内容表示上；RoPE 或其它位置编码机制则保持单独路径。

## 6. MLA 到底省了多少缓存

如果忽略实现细节，只看最核心的缓存账本，那么 MHA 每个 token 需要保存的量级是：

$$
M_{\mathrm{KV}}^{\mathrm{MHA}} \propto 2 \times n_{\mathrm{heads}} \times d_{\mathrm{head}}
$$

而 MLA 的一个常见近似账本是：

$$
M_{\mathrm{KV}}^{\mathrm{MLA}} \propto d_c + d_r
$$

其中 `d_r` 表示为位置分支额外保留的维度量级。于是缓存比值近似为：

$$
\frac{M_{\mathrm{KV}}^{\mathrm{MLA}}}{M_{\mathrm{KV}}^{\mathrm{MHA}}}
\approx
\frac{d_c + d_r}{2 \times n_{\mathrm{heads}} \times d_{\mathrm{head}}}
$$

这就是 MLA 的核心吸引力：只要 `d_c + d_r` 远小于显式 K / V 的总维度，长上下文缓存就能明显缩小。

## 7. MLA 真正多出来的成本是什么

MLA 省掉的是缓存和带宽，但多出来的是重建开销：

- 每次使用 K / V 时，需要从共享潜空间做上投影；
- 实现上要更小心处理内容分支和位置分支；
- kernel、缓存格式、张量布局通常都比 GQA 更复杂。

所以你不能把 MLA 理解成“更强的 GQA”。更准确的说法是：

- GQA 共享的是显式 K / V；
- MLA 共享的是更小的潜变量；
- MLA 把一部分成本从缓存阶段搬到了重建阶段。

## 8. 它和 GQA、DSA 分别是什么关系

- 相对 GQA：MLA 在“每个 token 存什么”这件事上走得更远。
- 相对 DSA：MLA 仍然会看全部历史，只是把每个历史 token 变轻；DSA 则在“看哪些 token”这件事上动刀。
- 相对 FlashAttention：MLA 不是 kernel 优化，而是表示层优化。

所以它们是不同层面的旋钮，可以叠加，而不是二选一。

## 9. 源码和系统页该怎么接

仓库里没有完整 MLA 内核实现，但你可以这样对照：

- 基线张量与 attention 直觉： [../../src/attention/mha_gqa.py](../../src/attention/mha_gqa.py)
- RoPE 路径直觉： [../../src/attention/rope_rmsnorm.py](../../src/attention/rope_rmsnorm.py)
- 缓存和 decode 预算： [../serving/formula-to-code-walkthrough.md](../serving/formula-to-code-walkthrough.md)
- 注意力总导读： [formula-to-code-walkthrough.md](formula-to-code-walkthrough.md)

最关键的不是找一段现成 MLA 代码，而是把“潜变量共享”和“按 head 重建”这两个动作先在脑子里拆开。

## 10. 继续读哪一页

- 想看第一层 KV 压缩：回到 [mha-vs-gqa-full-derivation.md](mha-vs-gqa-full-derivation.md)
- 想看减少访问集合：去 [mha-vs-dsa-full-derivation.md](mha-vs-dsa-full-derivation.md)
- 想看系统侧 TPOT / 带宽影响：去 [../serving/formula-to-code-walkthrough.md](../serving/formula-to-code-walkthrough.md)

## 这一页记住一句话

> MLA 的本质不是“共享 KV 头”，而是“共享更小的 KV 潜变量”。它靠压缩每个 token 的缓存表示来省显存和带宽，再用按 head 的重建把表达力补回来。
