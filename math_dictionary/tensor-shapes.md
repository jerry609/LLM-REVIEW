# Tensor 形状速查（MHA / GQA / MQA / MLA）

> **核心定位**：以精确的张量形状标注为核心，系统性地梳理 Transformer 推理过程中每一步操作的输入/输出形状、KV Cache 的存储形状、GQA 广播机制、FFN（SwiGLU）形状，以及 RoPE 作用形状。所有形状都遵循统一的符号约定。

---

## 1. 符号约定

| 符号 | 含义 | 典型值 |
|------|------|--------|
| $B$ | Batch Size | $1$–$256$ |
| $T$ | 序列长度 | $1$–$131072$ |
| $T_q$ | Query 长度（Prefill 时 $= T$，Decode 时 $= 1$） | — |
| $T_k$ | KV Cache 长度（$\le T$） | — |
| $H$ | Query Head 总数 | $32$–$128$ |
| $H_{\text{KV}}$ | KV Head 数 | MHA: $H$, GQA: $H/g$, MQA: $1$ |
| $d$ | $d_{\text{model}}$（隐藏维度） | $4096$–$8192$ |
| $d_h$ | $d_{\text{head}} = d / H$ | $128$ |
| $d_{\text{ff}}$ | FFN 中间维度 | $\sim 8d/3$ |
| $g$ | GQA 组大小 $= H / H_{\text{KV}}$ | $4$–$8$ |

---

## 2. 输入与线性投影

### 2.1 输入隐藏态

$$
X \in \mathbb{R}^{B \times T_q \times d}
$$

### 2.2 QKV 投影权重

$$
W_Q \in \mathbb{R}^{d \times (H \cdot d_h)} = \mathbb{R}^{d \times d}
$$
$$
W_K \in \mathbb{R}^{d \times (H_{\text{KV}} \cdot d_h)}
$$
$$
W_V \in \mathbb{R}^{d \times (H_{\text{KV}} \cdot d_h)}
$$
$$
W_O \in \mathbb{R}^{(H \cdot d_h) \times d} = \mathbb{R}^{d \times d}
$$

### 2.3 投影后形状

$$
Q = X W_Q \in \mathbb{R}^{B \times T_q \times H \times d_h}
$$
$$
K = X W_K \in \mathbb{R}^{B \times T_q \times H_{\text{KV}} \times d_h}
$$
$$
V = X W_V \in \mathbb{R}^{B \times T_q \times H_{\text{KV}} \times d_h}
$$

---

## 3. 注意力计算形状流

### 3.1 注意力分数

$$
S = Q K^\top \in \mathbb{R}^{B \times H \times T_q \times T_k}
$$

**GQA 广播**：$K$ 的 $H_{\text{KV}}$ 维度广播到 $H$ 维度（见 §6）。

### 3.2 缩放

$$
S_{\text{scaled}} = \frac{S}{\sqrt{d_h}} \in \mathbb{R}^{B \times H \times T_q \times T_k}
$$

### 3.3 因果掩码

$$
S_{\text{masked}}[b, h, i, j] = \begin{cases}
S_{\text{scaled}}[b, h, i, j] & j \le i \\
-\infty & j > i
\end{cases}
$$

### 3.4 Softmax

$$
P = \text{softmax}(S_{\text{masked}}) \in \mathbb{R}^{B \times H \times T_q \times T_k}
$$

### 3.5 加权求和

$$
O_{\text{heads}} = P V \in \mathbb{R}^{B \times H \times T_q \times d_h}
$$

### 3.6 拼接 + 输出投影

$$
O_{\text{concat}} = \text{Reshape}(O_{\text{heads}}) \in \mathbb{R}^{B \times T_q \times (H \cdot d_h)}
$$
$$
\text{Output} = O_{\text{concat}} W_O \in \mathbb{R}^{B \times T_q \times d}
$$

---

## 4. KV Cache 存储形状

### 4.1 每层 Cache

$$
K_{\text{cache}}, V_{\text{cache}} \in \mathbb{R}^{B \times T_{\text{cache}} \times H_{\text{KV}} \times d_h}
$$

### 4.2 每 Token 元素数

$$
\text{Elements/token/layer} = 2 \times H_{\text{KV}} \times d_h
$$

$$
\text{Bytes/token} = 2 \times L \times H_{\text{KV}} \times d_h \times s
$$

### 4.3 Decode 时的 Append 操作

新 token 的 $K, V$ 被 append 到 Cache 末尾：
$$
K_{\text{cache}}[:, T_{\text{cache}}, :, :] = K_{\text{new}} \in \mathbb{R}^{B \times 1 \times H_{\text{KV}} \times d_h}
$$
$$
T_{\text{cache}} \mathrel{+}= 1
$$

---

## 5. FFN 形状（SwiGLU 变体）

### 5.1 权重

$$
W_{\text{gate}} \in \mathbb{R}^{d \times d_{\text{ff}}}, \quad W_{\text{up}} \in \mathbb{R}^{d \times d_{\text{ff}}}, \quad W_{\text{down}} \in \mathbb{R}^{d_{\text{ff}} \times d}
$$

**注意**：SwiGLU 有 **3 个**权重矩阵（标准 FFN 只有 2 个），因此 $d_{\text{ff}} \approx 8d/3$（而非 $4d$）以保持参数量平衡。

### 5.2 计算流

$$
\text{gate} = X W_{\text{gate}} \in \mathbb{R}^{B \times T \times d_{\text{ff}}}
$$
$$
\text{up} = X W_{\text{up}} \in \mathbb{R}^{B \times T \times d_{\text{ff}}}
$$
$$
\text{hidden} = \text{SiLU}(\text{gate}) \odot \text{up} \in \mathbb{R}^{B \times T \times d_{\text{ff}}}
$$
$$
\text{output} = \text{hidden} \cdot W_{\text{down}} \in \mathbb{R}^{B \times T \times d}
$$

---

## 6. GQA 广播机制详解

设 $H = 32$，$H_{\text{KV}} = 8$，则组大小 $g = 32 / 8 = 4$。每 4 个 Query Head 共享 1 组 KV Head。

### 6.1 广播公式

$$
K_{\text{expanded}}[:, :, h, :] = K[:, :, \lfloor h / g \rfloor, :]
$$

例如 Query Head $0, 1, 2, 3$ 共享 KV Head $0$；Query Head $4, 5, 6, 7$ 共享 KV Head $1$；以此类推。

### 6.2 实现方式

广播**不实际复制数据**，通过 `expand` 或 `repeat_interleave` 实现逻辑重复：

```python
# K: (B, T, H_kv, d_h) -> (B, T, H, d_h)
K_expanded = K.repeat_interleave(g, dim=2)
```

---

## 7. RoPE 位置编码作用形状

### 7.1 作用对象

RoPE 作用于 **Q 和 K**（不作用于 V）的每个 Head：

$$
q_{\text{rot}}, k_{\text{rot}} \in \mathbb{R}^{d_h}
$$

### 7.2 旋转操作

将 $d_h$ 维向量两两配对为 $d_h / 2$ 个 2D 子空间，每个子空间做旋转：

$$
\begin{pmatrix} q_{2i}' \\ q_{2i+1}' \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}
$$

其中 $m$ 是 token 位置，$\theta_i = 10000^{-2i/d_h}$ 是频率。

**不引入额外参数**，仅依赖位置 $m$。

---

## 8. 各架构形状对比总表

| 张量 | MHA ($H_{\text{KV}}=H$) | GQA ($H_{\text{KV}}=H/g$) | MQA ($H_{\text{KV}}=1$) |
|------|:----------------------:|:-------------------------:|:----------------------:|
| $Q$ | $B \times T \times H \times d_h$ | 同左 | 同左 |
| $K, V$ | $B \times T \times H \times d_h$ | $B \times T \times H/g \times d_h$ | $B \times T \times 1 \times d_h$ |
| KV/token/layer | $2Hd_h$ | $2(H/g)d_h$ | $2d_h$ |
| KV/token 总量 | $2LHd_h s$ | $2L(H/g)d_h s$ | $2Ld_h s$ |

---

## 面试一句话

> "$H$ 和 $H_{\text{KV}}$ 的区别是理解 MHA/GQA/MQA 的关键：$H$ 决定 Q 的并行度（计算量），$H_{\text{KV}}$ 决定 KV Cache 大小（显存）。GQA 通过广播让多个 Q Head 共享一组 KV，在 $< 5\%$ 质量损失下节省 $g$ 倍 KV 显存。"