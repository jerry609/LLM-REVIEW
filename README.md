# LLM-REVIEW

## 目录

### `math_dictionary/`：数学查找字典（公式与指标速查）

**基础数学**
- `linear-algebra-basics.md`：线性代数基础（矩阵乘法 FLOPs、SVD、范数、余弦相似度）
- `symbols-glossary.md`：符号与单位总表（含常见模型参数速查表）
- `probability-and-sampling.md`：概率与采样（softmax、Top-k/p、KL 散度、投机解码接受概率）

**Transformer 核心**
- `tensor-shapes.md`：Tensor 形状速查（MHA/GQA/MQA、投影权重、RoPE、FFN）
- `transformer-attention-math.md`：注意力与核心组件公式（RoPE、ALiBi、LayerNorm/RMSNorm、残差连接）
- `rope-and-position-encoding.md`：RoPE 与位置编码专题（PI、NTK、YaRN、ALiBi 对比）
- `mqa-vs-gqa.md`：MQA/GQA 数学与工程对比（显存、带宽、质量权衡）
- `attention-complexity.md`：复杂度、FLOPs 与算力瓶颈（Roofline 模型、FlashAttention IO 分析）
- `flashattention-math.md`：FlashAttention IO 优化（online softmax、分块策略、FA-2/3）
- `linear-attention-math.md`：线性注意力与高效注意力（Mamba、RWKV、RetNet）
- `gated-attention-math.md`：门控注意力与信息流控制
- `tokenizer-math.md`：Tokenizer 与词表（BPE、压缩率、词表大小权衡）

**KV Cache 系统**
- `kv-memory.md`：KV 缓存显存估算与容量规划（PagedAttention、显存预算分配）
- `pagedattention-math.md`：PagedAttention 分页管理（碎片分析、Prefix Caching、Copy-on-Write）
- `kv-eviction-math.md`：KV 驱逐策略数学建模（LRU/LFU/注意力感知、公平性、粒度）
- `kv-compression-math.md`：KV 压缩与量化（分组量化、GPTQ/AWQ/SmoothQuant、稀疏化）

**推理服务**
- `prefill-decode-performance.md`：Prefill/Decode 延迟与吞吐（Continuous Batching、Chunked Prefill、P/D 分离）
- `serving-metrics.md`：推理服务指标速查（Goodput、诊断规则、告警阈值）
- `queueing-and-slo.md`：排队论与 SLO（M/M/1、M/G/1、尾延迟分析、限流策略）
- `distributed-serving-math.md`：多机多卡推理（TP/PP/EP 通信分析、气泡率、负载均衡）
- `speculative-decoding-math.md`：投机解码（接受-拒绝采样、加速比分析、draft 模型选择）

**训练与对齐**
- `optimization-and-scaling.md`：优化器与 Scaling Law（Adam/AdamW、Chinchilla、MFU、推理 Scaling）
- `lora-peft-math.md`：LoRA 与参数高效微调（低秩增量、QLoRA、多 LoRA 服务）
- `rlhf-alignment-math.md`：RLHF 与对齐（Bradley-Terry、PPO、DPO、KTO）

**评测与工具**
- `evaluation-metrics.md`：评测指标（PPL、BLEU/ROUGE、Pass@k、校准、长上下文评测）
- `mental-math-cheatsheet.md`：面试心算速记（2^n 表、GPU 参数、模型权重速算）
