# MQA 与 GQA 数学与工程剖析

> **核心定位**：从注意力 KV 头的分配机制出发，深度解析 MHA、MQA、GQA 的显存占用、显存带宽需求以及在不同推理阶段（Prefill vs Decode）的性能差异。全篇包含精确的公式推导。

---

## 1. 架构定义与张量维度

在标准的多头注意力（MHA）中，Query、Key、Value 的头数（Head Number）是相同的。MQA 和 GQA 通过减少 Key 和 Value 的头数来大幅降低显存和内存带宽。

假设：
- $d_{\text{model}}$：模型隐藏层维度
- $H$：Query 的头数（Query Heads）
- $H_{\text{KV}}$：Key 和 Value 的头数（KV Heads）
- $d_{\text{head}} = \frac{d_{\text{model}}}{H}$：每个头的特征维度
- $G = \frac{H}{H_{\text{KV}}}$：每个 KV 头服务的 Query 头数量（组大小 Group Size）

### 1.1 三种架构的严格划分

| 变体名称 | KV 头数 ($H_{\text{KV}}$) | 分组大小 ($G$) | 共享机制 |
|----------|-------------------------|--------------|---------|
| **MHA** (Multi-Head) | $H_{\text{KV}} = H$ | $1$ | 每个 Query 头拥有独立的 KV 头 |
| **GQA** (Grouped-Query) | $1 < H_{\text{KV}} < H$ | $\frac{H}{H_{\text{KV}}}$ | 每 $G$ 个 Query 头共享一个 KV 头 |
| **MQA** (Multi-Query) | $H_{\text{KV}} = 1$ | $H$ | 所有 $H$ 个 Query 头共享唯一一个 KV 头 |

### 1.2 注意力计算公式 (以 GQA 为例)

在 GQA 中，对于第 $h$ 个 Query 头（$0 \le h < H$），它所对应的 KV 头索引为 $g = \lfloor \frac{h}{G} \rfloor$。计算公式为：

$$
\text{head}_h = \text{softmax}\left(\frac{Q_h K_g^\top}{\sqrt{d_{\text{head}}}}\right) V_g
$$

其中：
- $Q_h \in \mathbb{R}^{T \times d_{\text{head}}}$
- $K_g, V_g \in \mathbb{R}^{T_{\text{kv}} \times d_{\text{head}}}$ （$T_{\text{kv}}$ 为 KV Cache 的长度）

---

## 2. 显存容量与带宽分析

> MQA 和 GQA 并不减少计算量（FLOPs），它们的唯一目的是**拯救 Memory-bound 的 Decode 阶段**。

### 2.1 KV Cache 显存容量计算

假设序列总长度为 $L$，批次大小为 $B$，使用 16-bit 浮点数（2 Bytes/参数）。

单层单 Token 的 KV Cache 占用：
$$
\text{Bytes per Token} = 2 \text{ (K和V)} \times H_{\text{KV}} \times d_{\text{head}} \times 2 \text{ Bytes} = 4 \cdot H_{\text{KV}} \cdot d_{\text{head}}
$$

**以 LLaMA-2-70B 为例：**
- $H = 64, d_{\text{head}} = 128$。使用 GQA 时 $H_{\text{KV}} = 8$。
- **MHA 占用**：$4 \times 64 \times 128 = 32,768 \text{ Bytes/token} \approx 32 \text{ KB}$
- **GQA-8 占用**：$4 \times 8 \times 128 = 4,096 \text{ Bytes/token} \approx 4 \text{ KB}$ （显存节省 8 倍）
- 若上下文长 $8192$，Batch 为 $128$，$80$ 层：MHA 需 $320 \text{ GB}$（OOM），GQA 仅需 $40 \text{ GB}$。

### 2.2 内存带宽需求 (Memory Bandwidth)

在自回归的 **Decode 阶段**，每生成一个新 token，必须将前面所有的 KV Cache 从 HBM 读取到 SRAM 中算内积。

$$
\text{Bandwidth Requirement} \propto \text{Total KV Cache Size} \propto H_{\text{KV}}
$$

- MHA 由于过大的 KV Cache，极易触碰硬件的 Memory Wall，导致算力闲置。
- GQA 将访存量降低了 $G$ 倍，使得算术强度（Arithmetic Intensity）成比例提升，极大提高了大 Batch Size 下的吞吐量（Throughput）。

### 2.3 计算量 (FLOPs) 为何不变？

在生成注意力分数 $Q_h K_g^\top$ 时，虽然 $K_g$ 是被多个 Query 头共享的，但**每个 Query 头 $Q_h$ 依然是独立且不同的**。
因此，必须进行 $H$ 次独立的向量-矩阵乘法。
总 FLOPs 仍严格正比于 Query 头数 $H$，并未减少：
$$
\text{Decode FLOPs per Step} \approx 2 \cdot B \cdot H \cdot T_{\text{kv}} \cdot d_{\text{head}}
$$

---

## 3. 从 MHA 转换为 GQA 的数学方法 (Uptraining)

如果已经训练好了一个庞大的 MHA 模型（如 LLaMA-1），想要极低成本地转换为 GQA/MQA 架构，通常采用**平均池化（Mean Pooling）**后进行微调（Uptraining）。

假设原 MHA 模型有 $H$ 个 KV 头，现在要压缩为 $H_{\text{KV}}$ 个头，组大小为 $G$。
对于第 $g$ 组新的 KV 权重矩阵 $\hat{W}_g^K$ 和 $\hat{W}_g^V$：

$$
\hat{W}_g^K = \frac{1}{G} \sum_{i = g \cdot G}^{(g+1) \cdot G - 1} W_i^K
$$
$$
\hat{W}_g^V = \frac{1}{G} \sum_{i = g \cdot G}^{(g+1) \cdot G - 1} W_i^V
$$

**工程验证**：
1. 直接 Mean Pooling 会导致初始 PPL 上升。
2. 只需要在总训练数据量的 $5\%$ 左右进行继续预训练（Continual Pre-training），模型性能就能恢复到接近原始 MHA 的水平。

---

## 4. 面试实战总结

**Q1：为什么 GQA/MQA 在 Prefill 阶段没有明显的加速效果？**
> 答：Prefill 阶段是计算密集的（Compute-bound），其算术强度很高。在此阶段，系统瓶颈在于 Tensor Core 的矩阵乘法算力。由于 GQA/MQA 并不减少 $Q K^\top$ 的计算量（FLOPs 不变），所以 Prefill 速度基本和 MHA 一致。它们的加速主战场在严重访存受限（Memory-bound）的 Decode 阶段。

**Q2：MQA 和 GQA 如何选择？**
> 答：**MQA (H_kv=1)** 达到了显存压缩的极限，但在处理复杂推理、代码生成或长上下文任务时，往往会出现严重的表达能力退化（"共享过度"导致注意力坍塌）。
> **GQA (通常 H_kv=4 或 8)** 是一种完美的帕累托最优（Pareto Optimal）：它用极小的质量损失（通常 < 1%），换取了接近 MQA 的推理速度和显存容量，是目前业界标准（LLaMA-3, Qwen-2 等）。

**Q3：在代码实现上，GQA 是怎么让算子识别的？**
> 答：在 PyTorch 的具体实现中，不需要显式写 for 循环。通常使用 `repeat_interleave` 将 $K$ 和 $V$ 在 head 维度复制 $G$ 次，在逻辑上伪装成 MHA 的形状，然后直接调用 `F.scaled_dot_product_attention`。在底层 FlashAttention 内核中，则原生支持 GQA 模式，直接通过步长（Stride）映射读取共享的内存地址。
