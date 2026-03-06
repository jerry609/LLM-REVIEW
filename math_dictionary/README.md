# 数学查找字典

> 用来在最短时间内定位公式、变量定义、张量形状和量级估算。这个目录现在按 GitBook 连续阅读习惯重新整理，建议优先配合 [SUMMARY.md](../SUMMARY.md) 使用。

## 这部分适合什么时候看

- 面试前 15 分钟：先看符号表、张量形状、KV 显存、服务指标。
- 系统学习前：先统一符号和量纲，再进入长文笔记。
- 写代码前：先把张量形状和复杂度过一遍，避免实现时维度混乱。
- 做性能分析时：直接查 Prefill / Decode、Roofline、SLO、队列模型。

## 先看这 6 页

1. [symbols-glossary.md](symbols-glossary.md)：统一符号和单位。
2. [tensor-shapes.md](tensor-shapes.md)：Q / K / V、GQA、MoE 的形状速查。
3. [transformer-attention-math.md](transformer-attention-math.md)：Attention、RoPE、Norm、FFN 的核心公式。
4. [flashattention-math.md](flashattention-math.md)：在线 Softmax 与 IO 优化的详细推导。
5. [kv-memory.md](kv-memory.md)：KV Cache 显存估算与容量规划。
6. [serving-metrics.md](serving-metrics.md)：TTFT、TPOT、吞吐、P99、Goodput。

## 按主题阅读

### Transformer 核心

- [linear-algebra-basics.md](linear-algebra-basics.md)：矩阵乘法、范数、SVD、Jacobian。
- [probability-and-sampling.md](probability-and-sampling.md)：Softmax、Top-k、Top-p、KL。
- [transformer-attention-math.md](transformer-attention-math.md)：缩放点积注意力、RoPE、LayerNorm、RMSNorm、SwiGLU。
- [rope-and-position-encoding.md](rope-and-position-encoding.md)：RoPE、ALiBi、YaRN、PI、NTK scaling。
- [mqa-vs-gqa.md](mqa-vs-gqa.md)：MHA / MQA / GQA 的数学与工程差异。

### KV Cache 与推理性能

- [kv-memory.md](kv-memory.md)：容量、层数、上下文长度和精度如何影响显存。
- [pagedattention-math.md](pagedattention-math.md)：分页、碎片与 prefix caching。
- [kv-eviction-math.md](kv-eviction-math.md)：LRU、LFU、注意力感知驱逐。
- [kv-compression-math.md](kv-compression-math.md)：量化、压缩误差与收益。
- [prefill-decode-performance.md](prefill-decode-performance.md)：Prefill / Decode 的瓶颈差异。
- [attention-complexity.md](attention-complexity.md)：复杂度、FLOPs、Roofline。

### 服务与系统

- [serving-metrics.md](serving-metrics.md)：TTFT、TPOT、吞吐、Goodput 的定义和联系。
- [queueing-and-slo.md](queueing-and-slo.md)：排队论、尾延迟、SLO 与限流。
- [distributed-serving-math.md](distributed-serving-math.md)：TP、PP、EP、气泡率与负载均衡。
- [speculative-decoding-math.md](speculative-decoding-math.md)：投机解码的接受率和加速比。
- [moe-routing-math.md](moe-routing-math.md)：MoE 路由与负载均衡损失。

### 训练与对齐

- [optimization-and-scaling.md](optimization-and-scaling.md)：优化器、Scaling Law、MFU。
- [lora-peft-math.md](lora-peft-math.md)：LoRA、QLoRA、参数高效微调。
- [rlhf-alignment-math.md](rlhf-alignment-math.md)：Bradley-Terry、PPO、DPO、KTO。
- [evaluation-metrics.md](evaluation-metrics.md)：PPL、BLEU、ROUGE、win rate、统计显著性。

## 公式和源码怎么联动

- 先看 [transformer-attention-math.md](transformer-attention-math.md) 和 [flashattention-math.md](flashattention-math.md) 把公式捋顺。
- 再看 [../notes/attention/formula-to-code-walkthrough.md](../notes/attention/formula-to-code-walkthrough.md) 把公式、张量形状和源码逐段对齐。
- 最后对照实现：
  - [../src/attention/mha_gqa.py](../src/attention/mha_gqa.py)
  - [../src/attention/rope_rmsnorm.py](../src/attention/rope_rmsnorm.py)
  - [../src/attention/flash_attn_sim.py](../src/attention/flash_attn_sim.py)

## GitBook 友好写法约定

- 显示公式优先使用独立的 `$$ ... $$` 块，避免把长公式塞进表格。
- 在表格里出现绝对值、范数、KL 时，优先写成 `\lvert x \rvert`、`\lVert x \rVert`、`\parallel`。
- 若某一页需要同时讲公式和实现，优先补一节“对应源码”，并从本目录回链过去。
