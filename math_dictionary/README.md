# 数学查找字典（Math Dictionary）

> 用途：LLM 岗位面试的公式速查与口述提纲。默认按"基础 → 核心 → 系统 → 训练 → 评测"顺序学习。

## 学习顺序（建议）

### 第一阶段：基础数学与符号
1. `symbols-glossary.md` — 符号与单位总表
2. `linear-algebra-basics.md` — 线性代数基础（矩阵乘法、SVD、范数）
3. `probability-and-sampling.md` — 概率与采样（softmax、Top-k/p、KL 散度）
4. `tokenizer-math.md` — Tokenizer 与词表（BPE、压缩率）

### 第二阶段：Transformer 核心
5. `tensor-shapes.md` — Q/K/V 与 GQA 的形状速查
6. `transformer-attention-math.md` — 注意力核心公式（含 RoPE、LayerNorm/RMSNorm）
7. `rope-and-position-encoding.md` — RoPE 与位置编码（PI、NTK、YaRN、ALiBi）
8. `mqa-vs-gqa.md` — MQA/GQA 数学与工程对比
9. `flashattention-math.md` — FlashAttention IO 优化（online softmax、分块策略）
10. `linear-attention-math.md` — 线性注意力与高效注意力（Mamba、RWKV、RetNet）
11. `gated-attention-math.md` — 门控注意力与信息流控制

### 第三阶段：KV Cache 系统
12. `kv-memory.md` — KV 显存估算与容量规划（PagedAttention 块管理）
13. `pagedattention-math.md` — PagedAttention 分页与碎片估算（Prefix Caching）
14. `kv-eviction-math.md` — 驱逐策略的数学建模（LRU/LFU/注意力感知、多租户）
15. `kv-compression-math.md` — 量化/压缩误差与收益（GPTQ/AWQ/SmoothQuant、稀疏化）
16. `attention-complexity.md` — 复杂度、FLOPs 与 Roofline 模型

### 第四阶段：推理服务
17. `prefill-decode-performance.md` — Prefill/Decode 延迟与吞吐（Continuous Batching、Chunked Prefill）
18. `serving-metrics.md` — TTFT/TPOT/吞吐/P99/Goodput 指标
19. `queueing-and-slo.md` — 排队论与 SLO（M/M/1、M/G/1、尾延迟、限流）
20. `distributed-serving-math.md` — 多机多卡通信（TP/PP/EP、气泡率、负载均衡）
21. `speculative-decoding-math.md` — 投机解码（接受-拒绝采样、加速比分析）
22. `moe-routing-math.md` — MoE 路由与负载均衡（门控函数、EP 通信）

### 第五阶段：训练与对齐
23. `optimization-and-scaling.md` — 优化器与 Scaling Law（Adam/AdamW、Chinchilla、MFU）
24. `lora-peft-math.md` — LoRA 与参数高效微调（低秩增量、QLoRA）
25. `rlhf-alignment-math.md` — RLHF 与对齐（Bradley-Terry、PPO、DPO、KTO）

### 第六阶段：评测与心算
26. `evaluation-metrics.md` — 评测指标（PPL、BLEU/ROUGE、Pass@k、校准）
27. `mental-math-cheatsheet.md` — 面试心算速记（2^n 表、GPU 参数、模型权重速算）

## 面试回答模板
1. **先给公式**：定义变量、假设边界。
2. **再做代入**：给出一个数量级例子（7B/70B）。
3. **再说结论**：资源瓶颈和工程策略。
4. **最后讲风险**：误差、尾延迟、公平性、回滚条件。
