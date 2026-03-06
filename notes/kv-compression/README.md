# KV 压缩

> 这一组笔记把 KV Compression 拆成两条互补主线：一条是“每个 token 占多少字节”的量化压缩，另一条是“保留哪些 token”的稀疏压缩。前者在数值精度上做折中，后者在上下文保留上做选择。

## 最推荐的阅读顺序

1. [formula-to-code-walkthrough.md](formula-to-code-walkthrough.md)：先把量化、误差、H2O / SnapKV 选择规则和源码对上。
2. [../../math_dictionary/kv-compression-math.md](../../math_dictionary/kv-compression-math.md)：再看完整压缩比、误差传播和保留策略推导。
3. [quantization.md](quantization.md)：补齐对称 / 非对称、per-tensor / per-channel 的工程判断。
4. [sparsity.md](sparsity.md)：再看 Heavy-Hitter、观测窗口、层间差异等稀疏思路。
5. [interview-qa.md](interview-qa.md)：最后用问答形式复盘。

## 这一组专题覆盖什么

- 量化：对称 / 非对称、per-channel、反量化、误差指标、压缩比。
- 稀疏化：H2O、SnapKV、保留最近窗口、保留高分 token。
- 工程权衡：显存收益、额外元数据开销、重算风险、对长上下文质量的影响。

## 对应源码

- [../../src/kv_cache/compression/quantizer.py](../../src/kv_cache/compression/quantizer.py)：对称 / 非对称 per-channel 量化与误差统计。
- [../../src/kv_cache/compression/sparsifier.py](../../src/kv_cache/compression/sparsifier.py)：H2O 风格的累积注意力打分、SnapKV 风格的一次性 token 选择。

## 如果你只剩 20 分钟

- 先读 [formula-to-code-walkthrough.md](formula-to-code-walkthrough.md)
- 再看 [../../math_dictionary/kv-compression-math.md](../../math_dictionary/kv-compression-math.md) 里的量化误差和保留预算公式
- 最后对照 [../../src/kv_cache/compression/quantizer.py](../../src/kv_cache/compression/quantizer.py) 和 [../../src/kv_cache/compression/sparsifier.py](../../src/kv_cache/compression/sparsifier.py)
