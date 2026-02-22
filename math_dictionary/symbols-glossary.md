# 符号与单位总表

> **核心定位**：为整个 LLM 推理知识库提供统一的符号约定、单位换算和常见模型参数速查表。所有数学公式中的符号遵循本文档的定义。

---

## 1. 模型结构符号

| 符号 | 含义 | 别名 | 说明 |
|------|------|------|------|
| $B$ | Batch Size | — | 并发请求数或活跃序列数 |
| $T$ | 序列长度 | $\text{seq\_len}$ | token 数；$T_q$ = Query 长度，$T_k$ = Key 长度 |
| $L$ | 模型层数 | $\text{n\_layers}$ | — |
| $H$ | Attention Head 总数 | $\text{n\_heads}$ | Q 的 Head 数 |
| $H_{\text{KV}}$ | KV Head 数 | $\text{n\_kv\_heads}$ | GQA: $H_{\text{KV}} < H$；MQA: $H_{\text{KV}} = 1$ |
| $g$ | GQA 组大小 | $\text{group\_size}$ | $g = H / H_{\text{KV}}$ |
| $d$ | 隐藏维度 | $d_{\text{model}}$, $d_{\text{hidden}}$ | — |
| $d_h$ | 单头维度 | $d_{\text{head}}$ | $d_h = d / H$，通常 $128$ |
| $d_{\text{ff}}$ | FFN 中间维度 | — | 标准: $4d$；SwiGLU: $\approx 8d/3$ |
| $V$ | 词表大小 | $\text{vocab\_size}$ | 如 $32$K, $128$K |
| $N$ | 模型参数量 | — | 如 $7 \times 10^9$（7B） |
| $E$ | MoE 专家总数 | — | — |
| $E_{\text{active}}$ | 每 token 激活专家数 | — | 如 Mixtral: $E_{\text{active}} = 2$ |
| $r$ | LoRA 秩 | — | 通常 $4$–$64$ |

---

## 2. 精度与内存符号

| 符号 | 含义 | 说明 |
|------|------|------|
| $s$ | 每元素字节数 | BF16/FP16: $2$, FP8/INT8: $1$, INT4: $0.5$, FP32: $4$ |
| $M$ | GPU 显存总量 | 如 A100: $80$ GB |
| $\text{BW}$ | 显存带宽 | A100: $2$ TB/s, H100: $3.35$ TB/s |
| $M_{\text{KV}}$ | KV Cache 显存 | $= 2 L H_{\text{KV}} d_h s \times \sum_i T_i$ |

---

## 3. 推理服务符号

| 符号 | 含义 | 单位 |
|------|------|------|
| $\lambda$ | 请求到达率 | req/s |
| $\mu$ | 服务速率 | req/s |
| $\rho$ | 系统利用率 | $\lambda / \mu$，要求 $< 1$ |
| $\tau$ | 采样温度 | 无量纲 |
| $\text{TTFT}$ | Time To First Token | ms |
| $\text{TPOT}$ | Time Per Output Token | ms/token |
| $\text{ITL}$ | Inter-Token Latency | ms/token |
| $\text{P99}$ | 99th percentile latency | ms |

---

## 4. 训练相关符号

| 符号 | 含义 | 说明 |
|------|------|------|
| $\eta$ / $\text{lr}$ | 学习率 | — |
| $D$ | 训练数据量 | token 数 |
| $C$ | 训练 FLOPs | $C \approx 6ND$ |
| $\text{MFU}$ | Model FLOPs Utilization | $\text{实际 FLOPs} / \text{理论峰值}$ |
| $\beta_1, \beta_2$ | Adam 的 EMA 系数 | 通常 $0.9, 0.95$ |
| $\epsilon$ | Adam 的数值稳定项 | $10^{-8}$ |

---

## 5. 数学运算符号

| 符号 | 含义 |
|------|------|
| $\odot$ | 逐元素乘法（Hadamard Product） |
| $\otimes$ | 外积（Outer Product） |
| $\|\cdot\|_F$ | Frobenius 范数 |
| $\|\cdot\|_*$ | 核范数（Nuclear Norm） |
| $\sigma(\cdot)$ | Sigmoid 函数 |
| $\text{KL}(p \| q)$ | KL 散度 |
| $H(p)$ | Shannon 熵 |
| $\mathbb{E}[\cdot]$ | 期望 |
| $\mathbb{1}[\cdot]$ | 指示函数 |
| $\lceil \cdot \rceil$ / $\lfloor \cdot \rfloor$ | 上/下取整 |

---

## 6. 单位换算

### 6.1 数据量

$$
1 \text{ KB} = 1024 \text{ B}, \quad 1 \text{ MB} = 1024 \text{ KB}, \quad 1 \text{ GB} = 1024 \text{ MB}, \quad 1 \text{ TB} = 1024 \text{ GB}
$$

### 6.2 算力

$$
1 \text{ TFLOPS} = 10^{12} \text{ FLOPS}, \quad 1 \text{ PFLOPS} = 10^{15} \text{ FLOPS}
$$

### 6.3 常用 2 的幂次

| $2^n$ | 值 | 近似 | LLM 中的用途 |
|:-----:|:--:|:----:|:----------:|
| $2^7$ | $128$ | — | 常见 $d_h$ |
| $2^{10}$ | $1{,}024$ | $\approx 1$K | — |
| $2^{12}$ | $4{,}096$ | $\approx 4$K | 常见 $d_{\text{model}}$ |
| $2^{15}$ | $32{,}768$ | $\approx 32$K | 常见 $V$ |
| $2^{17}$ | $131{,}072$ | $\approx 128$K | 长上下文长度 |
| $2^{20}$ | $1{,}048{,}576$ | $\approx 1$M | — |
| $2^{30}$ | — | $\approx 1$G | — |

---

## 7. 常见模型参数速查

### 7.1 LLaMA 系列

| 规模 | $L$ | $d$ | $H$ | $H_{\text{KV}}$ | $d_h$ | $d_{\text{ff}}$ | BF16 权重 |
|:----:|:---:|:----:|:---:|:---------------:|:-----:|:---------------:|:---------:|
| **7B** | $32$ | $4096$ | $32$ | $32$ (MHA) | $128$ | $11008$ | $\sim 14$ GB |
| **7B** (Llama 2/3) | $32$ | $4096$ | $32$ | $8$ (GQA) | $128$ | $11008$ | $\sim 14$ GB |
| **13B** | $40$ | $5120$ | $40$ | $40$ | $128$ | $13824$ | $\sim 26$ GB |
| **34B** | $48$ | $6656$ | $52$ | $8$ | $128$ | $17920$ | $\sim 68$ GB |
| **70B** | $80$ | $8192$ | $64$ | $8$ | $128$ | $28672$ | $\sim 140$ GB |
| **405B** | $126$ | $16384$ | $128$ | $8$ | $128$ | $53248$ | $\sim 810$ GB |

### 7.2 GPU 参数速查

| GPU | HBM | BW | BF16 TFLOPS | Roofline 拐点 AI |
|:---:|:---:|:--:|:-----------:|:---------------:|
| **A100 80GB** | $80$ GB | $2$ TB/s | $312$ | $\sim 156$ |
| **H100 80GB** | $80$ GB | $3.35$ TB/s | $990$ | $\sim 295$ |
| **H200 141GB** | $141$ GB | $4.8$ TB/s | $990$ | $\sim 206$ |
| **B200** | $192$ GB | $8$ TB/s | $2250$ | $\sim 281$ |

---

## 8. 常见 KV 显存速算

$$
\text{bytes/token} = 2 L H_{\text{KV}} d_h s
$$

| 模型 | 精度 | bytes/token | 4K token 单序列 | 128K token 单序列 |
|------|:----:|:-----------:|:--------------:|:----------------:|
| 7B GQA ($H_{\text{KV}}=8$) | BF16 | $128$ KB | $512$ MB | $16$ GB |
| 7B MHA ($H_{\text{KV}}=32$) | BF16 | $512$ KB | $2$ GB | $64$ GB |
| 70B GQA ($H_{\text{KV}}=8$) | BF16 | $320$ KB | $1.25$ GB | $40$ GB |
| 7B GQA | INT8 | $64$ KB | $256$ MB | $8$ GB |
| 7B GQA | INT4 | $32$ KB | $128$ MB | $4$ GB |

---

## 9. 面试易错点

| 易错点 | 正确理解 |
|--------|---------|
| "$H$ 和 $H_{\text{KV}}$ 一样" | GQA/MQA 下 $H_{\text{KV}} \ll H$，显著影响 KV 显存 |
| "吞吐就是 tokens/s" | 必须说明口径：tokens/s vs req/s，裸吞吐 vs Goodput |
| "延迟看平均值" | 必须带分位数（P95/P99），平均值掩盖尾延迟 |
| "$d_{\text{ff}} = 4d$" | SwiGLU 架构中 $d_{\text{ff}} \approx 8d/3$（3 个权重矩阵） |
| "参数量 = 层数 × 隐藏维度²" | 粗估公式 $N \approx 12 L d^2$（不含 Embedding 和 Head） |