# 注意力机制

> 这一组笔记覆盖注意力演进、工程优化和典型架构对比。

- [注意力机制演进与推理流程](attention-evolution-and-inference.md)：MHA、MQA、GQA、MLA、SSM 的整体地图。
- [多头注意力分化](multi-head-divergence.md)：为什么不同 head 学到不同子空间。
- [FlashAttention](flashattention.md)：在线 softmax、分块与 IO 优化。
- [长上下文](long-context.md)：长上下文技术路线与关键权衡。
- [五大注意力统一对比](attention-mechanisms-unified-comparison.md)：MHA、GQA、MLA、DSA、Linear 并排对比。
- [MHA vs MLA](mha-vs-mla-full-derivation.md)：矩阵吸收、Decode 带宽与伪代码。
- [MHA vs GQA](mha-vs-gqa-full-derivation.md)：分组共享 KV 与 SRAM 复用。
- [MHA vs DSA](mha-vs-dsa-full-derivation.md)：DSA 的压缩思路与实现哲学。
- [MHA vs 线性注意力](mha-vs-linear-attention-full-derivation.md)：结合律、RNN 等价与状态压缩。