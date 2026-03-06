# 注意力机制

> 这里聚焦注意力机制的演进、工程优化和典型架构对比。目录已经按 GitBook 连续阅读习惯重排：先总览，再推导，再对照源码。

## 最推荐的阅读顺序

1. [formula-to-code-walkthrough.md](formula-to-code-walkthrough.md)：最快建立“公式 -> 张量形状 -> 源码”映射。
2. [attention-evolution-and-inference.md](attention-evolution-and-inference.md)：理解 MHA / MQA / GQA / MLA / SSM 的总图。
3. [attention-mechanisms-unified-comparison.md](attention-mechanisms-unified-comparison.md)：先走专题入口，再按问题跳到对应深挖页。
4. [mha-vs-gqa-full-derivation.md](mha-vs-gqa-full-derivation.md)：看清 GQA 如何压缩 KV Cache。
5. [mha-vs-mla-full-derivation.md](mha-vs-mla-full-derivation.md)：看清 MLA 为什么能进一步压缩。

## 总览与入门

- [attention-evolution-and-inference.md](attention-evolution-and-inference.md)：MHA、MQA、GQA、MLA、SSM 的整体地图。
- [multi-head-divergence.md](multi-head-divergence.md)：为什么不同 head 会学到不同子空间。
- [flashattention.md](flashattention.md)：在线 Softmax、分块与 IO 优化。
- [long-context.md](long-context.md)：长上下文的技术路线与权衡。

## 深度推导

- [mha-vs-gqa-full-derivation.md](mha-vs-gqa-full-derivation.md)：分组共享 KV、Decode 带宽、Roofline。
- [mha-vs-mla-full-derivation.md](mha-vs-mla-full-derivation.md)：矩阵吸收、解耦 RoPE、缓存压缩。
- [mha-vs-dsa-full-derivation.md](mha-vs-dsa-full-derivation.md)：稀疏选择、双阶段计算、索引器成本。
- [mha-vs-linear-attention-full-derivation.md](mha-vs-linear-attention-full-derivation.md)：结合律、RNN 等价、状态机视角。
- [attention-mechanisms-unified-comparison.md](attention-mechanisms-unified-comparison.md)：作为总览入口页，负责把你送到 GQA / MLA / DSA / Linear 等专题页。

## 对应源码

- [formula-to-code-walkthrough.md](formula-to-code-walkthrough.md)：本目录新增的源码导读页。
- [../../src/attention/mha_gqa.py](../../src/attention/mha_gqa.py)：MHA / GQA 最小 NumPy 实现。
- [../../src/attention/rope_rmsnorm.py](../../src/attention/rope_rmsnorm.py)：RoPE cache、旋转与 RMSNorm。
- [../../src/attention/flash_attn_sim.py](../../src/attention/flash_attn_sim.py)：FlashAttention 在线 Softmax 模拟器。

## 如果你只剩 20 分钟

- 先读 [formula-to-code-walkthrough.md](formula-to-code-walkthrough.md)
- 再扫一遍 [mha-vs-gqa-full-derivation.md](mha-vs-gqa-full-derivation.md)
- 最后对照 [../../src/attention/mha_gqa.py](../../src/attention/mha_gqa.py) 看 `group_size` 和 `np.repeat`
