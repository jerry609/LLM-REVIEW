# 数学查找字典

> 用途：面试前快速回顾公式、变量定义和数量级估算。默认按“基础 → 核心 → 系统 → 训练 → 评测”顺序学习。

## 如何使用

- 面试前 15 分钟：优先读符号表、张量形状、KV 显存估算、服务指标和心算速查。
- 系统学习：按下面的推荐顺序顺读，每一组都先看定义，再看复杂度，再看工程含义。
- 回答问题：优先给公式，再代入典型模型和典型上下文长度，最后补工程结论与风险。

## 推荐学习顺序

### 第一阶段：基础数学与符号

1. [symbols-glossary.md](symbols-glossary.md)：符号与单位总表。
2. [linear-algebra-basics.md](linear-algebra-basics.md)：矩阵乘法、SVD、范数、余弦相似度。
3. [probability-and-sampling.md](probability-and-sampling.md)：Softmax、Top-k、Top-p、KL 散度。
4. [tokenizer-math.md](tokenizer-math.md)：BPE、压缩率与词表大小权衡。

### 第二阶段：Transformer 核心

5. [tensor-shapes.md](tensor-shapes.md)：Q、K、V 与 GQA 的张量形状速查。
6. [transformer-attention-math.md](transformer-attention-math.md)：注意力核心公式、RoPE、LayerNorm、RMSNorm。
7. [rope-and-position-encoding.md](rope-and-position-encoding.md)：PI、NTK、YaRN、ALiBi。
8. [mqa-vs-gqa.md](mqa-vs-gqa.md)：MQA 与 GQA 的数学和工程对比。
9. [flashattention-math.md](flashattention-math.md)：FlashAttention 的 IO 优化与分块策略。
10. [linear-attention-math.md](linear-attention-math.md)：线性注意力、Mamba、RWKV、RetNet。
11. [gated-attention-math.md](gated-attention-math.md)：门控注意力与信息流控制。

### 第三阶段：KV Cache 系统

12. [kv-memory.md](kv-memory.md)：KV 显存估算与容量规划。
13. [pagedattention-math.md](pagedattention-math.md)：分页、碎片与 Prefix Caching。
14. [kv-eviction-math.md](kv-eviction-math.md)：LRU、LFU、注意力感知与多租户公平性。
15. [kv-compression-math.md](kv-compression-math.md)：量化、压缩误差与收益。
16. [attention-complexity.md](attention-complexity.md)：复杂度、FLOPs 与 Roofline 模型。

### 第四阶段：推理服务

17. [prefill-decode-performance.md](prefill-decode-performance.md)：Prefill / Decode 延迟与吞吐。
18. [serving-metrics.md](serving-metrics.md)：TTFT、TPOT、吞吐、P99、Goodput。
19. [queueing-and-slo.md](queueing-and-slo.md)：排队论、尾延迟、SLO 与限流。
20. [distributed-serving-math.md](distributed-serving-math.md)：TP、PP、EP、气泡率与负载均衡。
21. [speculative-decoding-math.md](speculative-decoding-math.md)：投机解码与接受概率。
22. [moe-routing-math.md](moe-routing-math.md)：MoE 路由与负载均衡。

### 第五阶段：训练与对齐

23. [optimization-and-scaling.md](optimization-and-scaling.md)：优化器、Scaling Law、MFU。
24. [lora-peft-math.md](lora-peft-math.md)：LoRA、QLoRA 与参数高效微调。
25. [rlhf-alignment-math.md](rlhf-alignment-math.md)：Bradley-Terry、PPO、DPO、KTO。

### 第六阶段：评测与心算

26. [evaluation-metrics.md](evaluation-metrics.md)：PPL、BLEU、ROUGE、Pass@k、校准。
27. [mental-math-cheatsheet.md](mental-math-cheatsheet.md)：2 的幂、GPU 参数、显存与权重速算。

## 面试回答模板

1. 先定义变量和边界条件。
2. 再写核心公式。
3. 再代入一个数量级例子，例如 7B、70B、8k、32k。
4. 最后说工程结论、瓶颈位置和风险控制。
