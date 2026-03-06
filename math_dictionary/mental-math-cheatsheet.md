# 面试心算速记

> **核心定位**：提供 LLM 推理面试中最常用的速算公式、常数表和估算技巧。目标是在面试中 30 秒内给出"量级正确"的答案。

---

## 1. 核心速算公式

### 1.1 模型权重显存

$$
\boxed{M_{\text{weights}} = N \times s}
$$

| 模型 | $N$ | BF16 ($s=2$) | INT8 ($s=1$) | INT4 ($s=0.5$) |
|:----:|:---:|:-----------:|:----------:|:-----------:|
| 7B | $7 \times 10^9$ | $14$ GB | $7$ GB | $3.5$ GB |
| 13B | $1.3 \times 10^{10}$ | $26$ GB | $13$ GB | $6.5$ GB |
| 34B | $3.4 \times 10^{10}$ | $68$ GB | $34$ GB | $17$ GB |
| 70B | $7 \times 10^{10}$ | $140$ GB | $70$ GB | $35$ GB |
| 405B | $4.05 \times 10^{11}$ | $810$ GB | $405$ GB | $202.5$ GB |

### 1.2 KV Cache 每 Token 显存

$$
\boxed{\text{bytes/token} = 2 L H_{\text{KV}} d_h s}
$$

**三个必记数**：
- 7B GQA ($H_{\text{KV}}=8$, BF16)：**$128$ KB/token**
- 7B MHA ($H_{\text{KV}}=32$, BF16)：**$512$ KB/token**
- 70B GQA ($H_{\text{KV}}=8$, BF16)：**$320$ KB/token**

### 1.3 训练 FLOPs

$$
\boxed{C = 6 N D}
$$

### 1.4 最大并发

$$
\boxed{\text{max\_concurrent} = \frac{M_{\text{GPU}} - M_{\text{weights}} - M_{\text{overhead}}}{\text{bytes/token} \times \bar{T}}}
$$

---

## 2. 常用常数表

### 2.1 精度字节数

| 精度 | $s$ (Bytes) |
|:----:|:-----------:|
| FP32 | $4$ |
| BF16 / FP16 | $2$ |
| FP8 / INT8 | $1$ |
| INT4 | $0.5$ |
| INT2 | $0.25$ |

### 2.2 GPU 参数

| GPU | HBM | BW | BF16 TFLOPS | 拐点 AI |
|:---:|:---:|:--:|:-----------:|:------:|
| A100 | $80$ GB | $2$ TB/s | $312$ | $\sim 156$ |
| H100 | $80$ GB | $3.35$ TB/s | $990$ | $\sim 295$ |
| H200 | $141$ GB | $4.8$ TB/s | $990$ | $\sim 206$ |

### 2.3 2 的幂次速记

| $2^n$ | 值 | 近似 | 用途 |
|:-----:|:--:|:----:|:----:|
| $2^7$ | $128$ | — | $d_h$ |
| $2^{10}$ | $1024$ | $1$K | — |
| $2^{12}$ | $4096$ | $4$K | $d_{\text{model}}$ |
| $2^{15}$ | $32768$ | $32$K | $V$ |
| $2^{17}$ | $131072$ | $128$K | 长上下文 |
| $2^{20}$ | — | $1$M | — |
| $2^{30}$ | — | $1$G | — |

---

## 3. 速算示例

### 3.1 "7B 模型在 A100 上最多跑多少并发？"

**思路**：权重 + 开销 → 剩余给 KV → 反推并发。

$$
\text{KV budget} = 80 - 14 - 2 = 64 \text{ GB}
$$
$$
\text{max\_concurrent} = \frac{64 \text{ GB}}{128 \text{ KB} \times 2048} = \frac{64 \text{ GB}}{256 \text{ MB}} = 250
$$

**答**："大约 $250$ 并发（假设 2K 平均序列长度）"。

### 3.2 "70B BF16 需要几张 A100？"

$$
M_{\text{weights}} = 70 \times 10^9 \times 2 = 140 \text{ GB}
$$
$$
\text{卡数} = \lceil 140 / 80 \rceil = 2 \text{（仅放权重）}
$$

但要留 KV Cache 空间 → 实际需要 **$4$ 张 A100**（TP=4）。

### 3.3 "训练 7B 模型 1T token 需要多少 A100-days？"

$$
C = 6 \times 7 \times 10^9 \times 10^{12} = 4.2 \times 10^{22} \text{ FLOPs}
$$
$$
\text{GPU-seconds} = \frac{4.2 \times 10^{22}}{312 \times 10^{12} \times 0.5} = \frac{4.2 \times 10^{22}}{1.56 \times 10^{14}} \approx 2.7 \times 10^8 \text{ s}
$$
$$
\text{GPU-days} = \frac{2.7 \times 10^8}{86400} \approx 3{,}125
$$

**答**："约 $3{,}000$ A100-days（MFU=50%）"。

### 3.4 "TPOT 大约多少？"（单请求 7B BF16 on A100）

$$
\text{TPOT} \approx \frac{M_{\text{weights}}}{\text{BW}} = \frac{14 \text{ GB}}{2 \text{ TB/s}} = 7 \text{ ms}
$$

**答**："约 $7$ ms/token（单请求 Memory-bound 下限）"。

### 3.5 "Prefill 4K token 需要多久？"（7B on A100）

$$
T_{\text{prefill}} = \frac{2 \times 7 \times 10^9 \times 4096}{312 \times 10^{12} \times 0.5} \approx 370 \text{ ms}
$$

**答**："约 $370$ ms"。

---

## 4. 通用估算技巧

### 4.1 有效数字法则

- 先保留 $1$–$2$ 位有效数字，再做数量级换算。
- 结果先报"约等于"，再给误差范围（$\pm 10\%$）。

### 4.2 乘法拆分

$$
70\text{B} \times 2 = 140 \text{ GB} \quad (\text{参数量} \times \text{字节数} = \text{权重显存})
$$

### 4.3 压缩比速算

$$
\text{量化节省} = 1 - \frac{s_{\text{new}}}{s_{\text{old}}}
$$

BF16 → INT8：节省 $50\%$。BF16 → INT4：节省 $75\%$。

### 4.4 KV vs 权重的相对大小

$$
\frac{M_{\text{KV}}}{M_{\text{weights}}} = \frac{2LH_{\text{KV}}d_h s \times \sum T_i}{N \times s} \approx \frac{2LH_{\text{KV}}d_h}{N} \times \sum T_i
$$

7B GQA：$2 \times 32 \times 8 \times 128 / (7 \times 10^9) \approx 9.4 \times 10^{-6}$ per token。

当 $\sum T_i > 10^5$（如 $50$ 并发 × $2$K token）时，KV 超过权重显存。

---

## 5. 面试口述模板

| 场景 | 口述 |
|------|------|
| 估算开始 | "我先给线性估算，再补碎片和调度开销作为修正项。" |
| 精度声明 | "数量级对了就行，面试更看重思路清晰，不追求小数点后精确。" |
| 显存规划 | "先算权重占多少，剩余给 KV，反推并发上限。" |
| 延迟估算 | "Prefill 看 FLOPs/TFLOPS，Decode 看 Bytes/BW。" |
| 训练估算 | "$C = 6ND$，除以 GPU 峰值 × MFU × 86400 = GPU-days。" |
