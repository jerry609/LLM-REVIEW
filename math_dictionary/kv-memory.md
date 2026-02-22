# KV Cache 显存估算与容量规划数学详解

> **核心定位**：从第一性原理出发，精确推导 KV Cache 在 MHA / GQA / MQA / MLA 各架构下的显存消耗公式，给出实际模型的代入计算，并建立完整的 GPU 显存预算分配框架与最大并发估算方法。

---

## 1. KV Cache 的精确显存公式

### 1.1 单 Token 显存占用

对于每个 token，每层需要存储 **Key** 和 **Value** 两个张量，每个张量的形状为 $\mathbb{R}^{H_{\text{KV}} \times d_{\text{head}}}$。

$$
\text{Bytes/token/layer} = 2 \times H_{\text{KV}} \times d_{\text{head}} \times s
$$

其中 $s$ 是每个元素的字节数（BF16: $s=2$，FP8/INT8: $s=1$，INT4: $s=0.5$）。

全模型的单 Token 显存占用：

$$
\boxed{\text{bytes\_per\_token} = 2 \times L \times H_{\text{KV}} \times d_{\text{head}} \times s}
$$

### 1.2 公式推导说明

这个公式中每个因子的含义：
- $2$：K 和 V 各一份
- $L$：模型总层数（每层独立存储 KV）
- $H_{\text{KV}}$：KV 头的数目（MHA 时 $H_{\text{KV}} = H$，GQA 时 $H_{\text{KV}} < H$，MQA 时 $H_{\text{KV}} = 1$）
- $d_{\text{head}}$：每个头的维度（通常 $d_{\text{head}} = d_{\text{model}} / H$）
- $s$：精度字节数

---

## 2. 典型模型代入计算

### 2.1 Llama-2-7B（GQA，$H_{\text{KV}} = 8$）

| 参数 | 值 |
|------|-----|
| $L$ | $32$ |
| $H$ | $32$ |
| $H_{\text{KV}}$ | $8$ |
| $d_{\text{head}}$ | $128$ |
| $s$ (BF16) | $2$ |

$$
\text{bytes\_per\_token} = 2 \times 32 \times 8 \times 128 \times 2 = 131{,}072 \text{ B} = 128 \text{ KB}
$$

| 序列长度 | 单序列 KV | 64 并发 KV |
|---------|----------|-----------|
| $4{,}096$ | $128 \text{ KB} \times 4096 = 512 \text{ MB}$ | $32 \text{ GB}$ |
| $32{,}768$ | $4 \text{ GB}$ | $256 \text{ GB}$ (需多卡) |
| $131{,}072$ | $16 \text{ GB}$ | $1 \text{ TB}$ (需集群) |

### 2.2 Llama-2-7B（MHA，$H_{\text{KV}} = 32$）

$$
\text{bytes\_per\_token} = 2 \times 32 \times 32 \times 128 \times 2 = 524{,}288 \text{ B} = 512 \text{ KB}
$$

相比 GQA 版本**膨胀 $4\times$**。

### 2.3 Llama-2-70B（GQA，$H_{\text{KV}} = 8$）

$$
\text{bytes\_per\_token} = 2 \times 80 \times 8 \times 128 \times 2 = 327{,}680 \text{ B} = 320 \text{ KB}
$$

| 序列长度 | 单序列 KV |
|---------|----------|
| $4{,}096$ | $1.25 \text{ GB}$ |
| $128{,}000$ | $\approx 39 \text{ GB}$ |

### 2.4 各架构 KV 比较总表

以 $d_{\text{model}} = 4096$，$H = 32$，$L = 32$，BF16 为基准：

| 架构 | $H_{\text{KV}}$ | bytes/token | 相对 MHA |
|------|:---------------:|:-----------:|:--------:|
| **MHA** | $32$ | $512 \text{ KB}$ | $1\times$ |
| **GQA** ($H_{\text{KV}}=8$) | $8$ | $128 \text{ KB}$ | $0.25\times$ |
| **GQA** ($H_{\text{KV}}=4$) | $4$ | $64 \text{ KB}$ | $0.125\times$ |
| **MQA** | $1$ | $16 \text{ KB}$ | $0.03\times$ |

---

## 3. 量化压缩后的显存

### 3.1 压缩比计算

$$
\text{compression\_ratio} = \frac{s_{\text{new}}}{s_{\text{old}}}
$$

$$
\text{bytes\_per\_token}_{\text{new}} = \text{bytes\_per\_token}_{\text{old}} \times \frac{s_{\text{new}}}{s_{\text{old}}}
$$

| 量化精度 | $s$ | 压缩比 (vs BF16) | 节省率 |
|---------|-----|:-----------------:|:------:|
| BF16 | $2$ | $1\times$ | $0\%$ |
| FP8 / INT8 | $1$ | $0.5\times$ | $50\%$ |
| INT4 | $0.5$ | $0.25\times$ | $75\%$ |
| INT2 (KIVI) | $0.25$ | $0.125\times$ | $87.5\%$ |

### 3.2 分组量化的实际压缩比

分组量化（Group Quantization）需要额外存储每组的 Scale（和可选的 Zero Point）。实际压缩比为：

$$
\text{effective\_ratio} = \frac{n_{\text{elem}} \times s_{\text{new}} + n_{\text{groups}} \times s_{\text{scale}}}{n_{\text{elem}} \times s_{\text{old}}}
$$

以 INT4 Group=128 为例：
$$
\text{effective\_ratio} = \frac{0.5 + 2/128}{2} = \frac{0.5156}{2} \approx 25.8\%
$$

接近理论 $25\%$，额外开销约 $3\%$。

---

## 4. GPU 显存预算分配模型

完整的 GPU 显存分配方程：

$$
\boxed{M_{\text{GPU}} = M_{\text{weights}} + M_{\text{KV}} + M_{\text{activations}} + M_{\text{overhead}}}
$$

| 组件 | 公式 | 典型值 (7B BF16) |
|------|------|:----------------:|
| **模型权重** $M_{\text{weights}}$ | $N \times s$ | $7 \times 10^9 \times 2 = 14 \text{ GB}$ |
| **KV Cache** $M_{\text{KV}}$ | $\text{bytes\_per\_token} \times \sum_i T_i$ | 取决于并发和序列长度 |
| **激活缓冲** $M_{\text{act}}$ | 与 $B \times T$ 相关 | $1$–$4 \text{ GB}$ |
| **系统预留** $M_{\text{overhead}}$ | CUDA Context + 碎片 | $1$–$3 \text{ GB}$ |

### 4.1 最大并发估算

$$
M_{\text{KV\_budget}} = M_{\text{GPU}} - M_{\text{weights}} - M_{\text{act}} - M_{\text{overhead}}
$$

$$
\boxed{\text{max\_concurrent} = \left\lfloor \frac{M_{\text{KV\_budget}}}{\text{bytes\_per\_token} \times \bar{T}} \right\rfloor}
$$

**代入示例**（7B GQA on A100 80GB）：
$$
M_{\text{KV\_budget}} = 80 - 14 - 2 - 2 = 62 \text{ GB}
$$
$$
\text{max\_concurrent} = \left\lfloor \frac{62 \text{ GB}}{128 \text{ KB} \times 2048} \right\rfloor = \left\lfloor \frac{62 \text{ GB}}{256 \text{ MB}} \right\rfloor = 248
$$

---

## 5. PagedAttention 对显存效率的影响

传统连续分配需按 $T_{\max}$ 预分配，PagedAttention 按实际使用分配：

$$
\text{有效显存放大率} = \frac{T_{\max}}{\bar{T}} = \frac{8192}{2048} = 4\times
$$

即 PagedAttention 可以让最大并发提升约 **4 倍**（在平均序列长度远小于最大长度时）。

---

## 6. 规划流程（工程 Checklist）

1. **计算静态占用**：$M_{\text{weights}} = N \times s$，加上激活缓冲和系统预留。
2. **确定 KV 预算**：$M_{\text{KV\_budget}} = M_{\text{GPU}} - M_{\text{static}}$。
3. **选择精度策略**：根据质量门槛确定 KV Cache 精度 ($s$)。
4. **反推并发能力**：$\text{max\_concurrent} = M_{\text{KV\_budget}} / (\text{bytes\_per\_token} \times \bar{T})$。
5. **决定压缩 vs 驱逐**：若并发不够 → 先量化（50%–75% 压缩），再考虑驱逐策略。
6. **设置安全边际**：预留 $10\%$–$20\%$ 应对突发长序列。

---

## 面试一句话

> "KV 容量规划是线性账本：$2 \times L \times H_{\text{KV}} \times d_{\text{head}} \times s \times \sum T_i$。GQA 省 $H/H_{\text{KV}}$ 倍，量化再省 $s_{\text{old}}/s_{\text{new}}$ 倍，最后用 PagedAttention 消除碎片。"