# KV Cache 压缩与量化数学详解

> **核心定位**：从量化理论的第一性原理出发，严格推导线性量化、分组量化的误差界，深入剖析 KV Cache 专用量化方法（KIVI、Per-token / Per-channel），以及权重量化的核心算法（GPTQ、AWQ、SmoothQuant）的数学本质。

---

## 1. 线性量化的数学基础

### 1.1 对称量化 (Symmetric Quantization)

将浮点张量 $x$ 映射到 $b$-bit 整数域 $\{-2^{b-1}+1, \dots, 2^{b-1}-1\}$：

$$
\text{scale} = \frac{\max(|x|)}{2^{b-1} - 1}
$$
$$
q = \text{round}\!\left(\frac{x}{\text{scale}}\right), \quad \hat{x} = \text{scale} \cdot q
$$

- **Zero Point** $= 0$（对称，原点不偏移）。
- **适用场景**：权重（通常近似对称分布）。

### 1.2 非对称量化 (Asymmetric Quantization)

当数据分布不对称时（如 ReLU 后的激活），使用非对称量化：

$$
\text{scale} = \frac{\max(x) - \min(x)}{2^b - 1}
$$
$$
\text{zp} = \text{round}\!\left(-\frac{\min(x)}{\text{scale}}\right)
$$
$$
q = \text{round}\!\left(\frac{x}{\text{scale}}\right) + \text{zp}, \quad \hat{x} = \text{scale} \cdot (q - \text{zp})
$$

### 1.3 量化误差分析

单元素量化误差的上界：
$$
|x - \hat{x}| \le \frac{\text{scale}}{2} = \frac{\text{range}}{2^{b+1} - 2}
$$

均方误差（MSE）的期望（假设 round 误差为均匀分布 $U[-\Delta/2, \Delta/2]$，$\Delta = \text{scale}$）：
$$
\mathbb{E}\left[(x - \hat{x})^2\right] = \frac{\Delta^2}{12} = \frac{\text{scale}^2}{12}
$$

信噪比（SNR）：
$$
\text{SNR} = 10 \log_{10} \frac{\|x\|^2}{\|x - \hat{x}\|^2} \approx 10 \log_{10} \frac{12 \, \text{Var}(x)}{\text{scale}^2} \quad \text{(dB)}
$$

---

## 2. 分组量化 (Group Quantization)

### 2.1 核心动机

全局量化的问题：如果张量中存在**离群值（Outlier）**，单个极大值会拉大 $\text{scale}$，导致其他正常值的量化精度严重退化。

**解决方案**：将通道分为多组（Group Size $g$，典型值 $64$ 或 $128$），每组独立计算 scale 和 zp。

### 2.2 误差改善的数学直觉

假设组内的动态范围为 $R_g$，全局动态范围为 $R$，且 $R_g \ll R$（组内离群值概率低）：

$$
\frac{\text{MSE}_{\text{group}}}{\text{MSE}_{\text{global}}} \approx \left(\frac{R_g}{R}\right)^2 \ll 1
$$

### 2.3 额外存储开销

每组需要存储 $\text{scale}$（FP16，2 bytes）和可选的 $\text{zp}$（INT8，1 byte）。对于 $n$ 个元素、组大小 $g$ 的张量：

$$
\text{overhead\_ratio} = \frac{(n/g) \times s_{\text{param}}}{n \times s_{\text{quant}}} = \frac{s_{\text{param}}}{g \times s_{\text{quant}}}
$$

**代入**：INT4 ($s_{\text{quant}} = 0.5$ B)，$g = 128$，FP16 scale ($s_{\text{param}} = 2$ B)：
$$
\text{overhead\_ratio} = \frac{2}{128 \times 0.5} = 3.1\%
$$

---

## 3. KV Cache 专用量化

### 3.1 Per-token vs Per-channel 量化

KV Cache 张量形状为 $\mathbb{R}^{T \times d}$。量化维度的选择至关重要：

| 策略 | 量化粒度 | Scale 数量 | 适用对象 | 原因 |
|------|---------|-----------|---------|------|
| **Per-token** | 每行独立 | $T$ 个 | **Value Cache** | 不同 token 的 $V$ 值域差异大 |
| **Per-channel** | 每列独立 | $d$ 个 | **Key Cache** | Key 的不同通道存在离群维度 |

### 3.2 KIVI (2-bit KV Cache)

> **出处**：Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache", 2024

KIVI 实现了极端的 **2-bit 量化**，其核心设计：

1. **Key Cache → Per-channel INT2**：Key 的特定通道存在离群值（与 $W_K$ 的权重分布有关），per-channel 可以对每个通道独立处理离群范围。
2. **Value Cache → Per-token INT2**：Value 的不同 token 行间方差大，per-token 更合适。
3. **残差补偿（Residual）**：保留最近 $w$（如 $128$）个 token 的 KV 为 FP16 全精度，作为"滑动窗口残差"。

$$
\text{KV}_{\text{total}} = \underbrace{\text{INT2}(K_{\text{old}}, V_{\text{old}})}_{\text{历史}} + \underbrace{\text{FP16}(K_{\text{recent}}, V_{\text{recent}})}_{\text{窗口内}}
$$

显存降至原始 BF16 的约 $\frac{1}{8}$（2-bit vs 16-bit），配合窗口残差几乎不损失质量。

---

## 4. 权重量化方法（面试重点）

### 4.1 GPTQ

> **出处**：Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers", 2022

基于 **Optimal Brain Surgeon (OBS)** 的逐列量化框架：

**目标**：量化权重 $W$ 为 $\hat{W}$，最小化输出误差：
$$
\min_{\hat{W}} \| W X - \hat{W} X \|_F^2
$$

**逐列更新规则**：当第 $c$ 列被量化时，量化误差通过 Hessian 矩阵的逆 $H^{-1}$ 补偿到尚未量化的其他列：

$$
\delta_{W_{:,c}} = -\frac{w_{q,c} - w_c}{[H^{-1}]_{cc}} \cdot H^{-1}_{:,c}
$$

其中 $w_{q,c}$ 是第 $c$ 列量化后的值，$H = 2 X X^\top$ 是 Hessian 矩阵。

### 4.2 AWQ (Activation-Aware Weight Quantization)

> **出处**：Lin et al., "AWQ: Activation-Aware Weight Quantization for LLM Compression and Acceleration", 2023

**核心洞察**：不是所有权重同等重要。对应**激活值大**的通道的权重，量化误差会被放大。

AWQ 对每个通道 $i$ 计算一个保护缩放因子：
$$
s_i = \left(\max_{\text{batch}} |X_i|\right)^\alpha, \quad \alpha \in [0, 1]
$$

量化前对权重做缩放：$W' = W \cdot \text{diag}(s)$，量化后反缩放：$\hat{W}_{\text{eff}} = \hat{W}' \cdot \text{diag}(s)^{-1}$。

等价效果：重要通道的权重被放大后量化，其**相对量化误差**变小。

### 4.3 SmoothQuant

> **出处**：Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models", 2022

**核心问题**：激活值中存在极端离群值（Outlier），使得激活难以量化。

**解决方案**：将激活的离群值"迁移"到权重中：
$$
Y = (X \, \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \, W) = \tilde{X} \cdot \tilde{W}
$$

选择 $s_i = \max|X_i|^\alpha / \max|W_i|^{1-\alpha}$（$\alpha = 0.5$ 时为几何平均），使得 $\tilde{X}$ 和 $\tilde{W}$ 的量化难度均衡。

---

## 5. 分层温度分级策略

对于 KV Cache 的不同"热度"，采用不同精度：

| 温度等级 | 精度 | 典型对象 | 回迁延迟 |
|---------|------|---------|---------|
| **热 (Hot)** | BF16 / FP16 | 最近 $w$ 个 token，高注意力权重 token | $0$ |
| **温 (Warm)** | FP8 / INT8 | 中等活跃区间 | 低 |
| **冷 (Cold)** | INT4 或 CPU Offload | 远距离低注意力 token | 高 |

回迁（反量化）延迟：
$$
T_{\text{dequant}} \approx \frac{n_{\text{tokens}} \times d \times s_{\text{compressed}}}{\text{dequant\_throughput}}
$$

工程上需要设置**每步回迁预算**（$\text{budget\_tokens\_per\_step}$），防止 decode 延迟突增。

---

## 6. 面试实战追问

**Q1：量化 KV Cache 和量化模型权重有什么区别？**
> 权重量化在离线完成（static），KV Cache 量化需要**在线实时进行**（每个新 token 生成时立即量化）。因此 KV 量化对延迟更敏感，不适合用 GPTQ 这种需要校准集的方法，更适合简单的 per-token / per-channel 均匀量化。

**Q2：INT4 量化 KV Cache 的质量损失如何评估？**
> 标准做法：在 Perplexity、MMLU、HumanEval 等基准上对比 BF16 baseline。关键看 $\Delta \text{PPL} = \text{PPL}_{\text{quant}} - \text{PPL}_{\text{baseline}}$。一般 $\Delta \text{PPL} < 0.5$ 认为质量可接受。配合 Group Quantization（$g = 128$）和 Residual Window，INT4 通常能达到 $\Delta \text{PPL} < 0.3$。

---

## 7. 对应源码与阅读顺序

- 先读 [../notes/kv-compression/formula-to-code-walkthrough.md](../notes/kv-compression/formula-to-code-walkthrough.md)，把“量化减少每个 token 的字节数”和“稀疏化减少需要保留的 token 数量”两条主线串起来，并对照 H2O / SnapKV 的选择逻辑。
- 再看 [../src/kv_cache/compression/quantizer.py](../src/kv_cache/compression/quantizer.py)，重点对应 `quantize_per_channel_symmetric()`、`quantize_per_channel_asymmetric()`、`dequantize()`、`quantization_error()`，把 scale、zero point、反量化误差落到实现。
- 接着看 [../src/kv_cache/compression/sparsifier.py](../src/kv_cache/compression/sparsifier.py)，对应 `cumulative_attention_scores()`、`keep_recent_and_heavy_hitters()`、`snapkv_select()`、`compression_ratio()`，理解“重点击中 + 最近窗口”如何映射成最终保留集合。
- 最后跑 `python -m pytest tests/test_kv_compression.py -v`，验证量化误差、保留 token 数量和压缩比计算是否与公式一致。
