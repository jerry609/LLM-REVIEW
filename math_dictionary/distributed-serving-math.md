# 多机多卡推理的数学模型详解

> **核心定位**：系统性推导 Tensor Parallelism (TP)、Pipeline Parallelism (PP)、Data Parallelism (DP)、Expert Parallelism (EP) 在推理场景下的通信代价、显存分摊和延迟模型。基于 α-β 通信模型给出精确的公式推导，并分析各策略的适用场景与组合方案。

---

## 1. 并行策略总览

| 策略 | 切分维度 | 通信模式 | 每层通信次数 | 适用场景 |
|------|---------|---------|:----------:|---------|
| **TP** (Tensor Parallel) | 层内 Weight 切分 | All-Reduce | $2$ | 单机多卡，延迟敏感 |
| **PP** (Pipeline Parallel) | 层间切分 | Point-to-Point | $1$（层间） | 多机，带宽有限 |
| **DP** (Data Parallel) | Batch 切分 | 无（推理时） | $0$ | 多副本扩容 |
| **EP** (Expert Parallel) | MoE Expert 切分 | All-to-All | $2$ | MoE 模型 |
| **SP** (Sequence Parallel) | 序列维度切分 | All-Gather / RS | $2$ | 超长序列 |

---

## 2. α-β 通信模型

所有通信操作都可以用**延迟-带宽模型**描述：

$$
\boxed{T_{\text{comm}} = \alpha + \beta \times n_{\text{bytes}}}
$$

| 参数 | 含义 | NVLink (同机) | IB (跨机) | PCIe |
|------|------|:------------:|:---------:|:----:|
| $\alpha$ | 启动延迟 | $\sim 1 \mu$s | $\sim 10$–$50 \mu$s | $\sim 5 \mu$s |
| $\beta$ | 每字节传输时间 | $\sim 0.0011 \mu$s/B (900 GB/s) | $\sim 0.02 \mu$s/B (50 GB/s) | $\sim 0.03 \mu$s/B |

**关键结论**：
- **小消息**（Decode，$T=1$）：受 $\alpha$ 主导，多卡反而可能更慢。
- **大消息**（Prefill，$T$ 大）：受 $\beta$ 主导，多卡能有效分摊。

---

## 3. Tensor Parallelism (TP) 详解

### 3.1 切分策略

将每层的权重矩阵沿列（或行）切分到 $P$ 张卡上：

**Attention 层**：$W_Q, W_K, W_V$ 按**列切分**（各卡计算不同 Head），$W_O$ 按**行切分**。
**FFN 层**（SwiGLU）：$W_{\text{gate}}, W_{\text{up}}$ 按**列切分**，$W_{\text{down}}$ 按**行切分**。

每层需要 **2 次 All-Reduce**（Attention 后一次，FFN 后一次）。

### 3.2 All-Reduce 通信量

Ring All-Reduce 的通信量：
$$
T_{\text{AR}} \approx 2 \cdot \frac{P-1}{P} \cdot (\alpha + \beta \cdot n_{\text{bytes}})
$$

每次 All-Reduce 的数据量：
$$
n_{\text{bytes}} = B \times T \times d_{\text{model}} \times s
$$

### 3.3 每层总通信量

$$
\text{Comm}_{\text{per\_layer}} = 2 \times n_{\text{bytes}} = 2 B T d_{\text{model}} s \quad \text{(2 次 All-Reduce)}
$$

全模型：
$$
\text{Comm}_{\text{total}} = 2L \times 2 B T d_{\text{model}} s = 4 L B T d_{\text{model}} s
$$

### 3.4 显存分摊

每张卡的权重显存：
$$
M_{\text{weights\_per\_card}} \approx \frac{N \times s}{P}
$$

KV Cache 按 Head 自然分配到各卡（GQA 下每卡 $H_{\text{KV}} / P$ 个 KV Head）。

### 3.5 TP 的加速极限

Decode 阶段（$T = 1$）的 All-Reduce 延迟由 $\alpha$ 主导。假设单次 All-Reduce 延迟为 $T_{\text{AR}} \approx 5 \mu$s（NVLink），则：

$$
T_{\text{comm\_overhead}} = 2L \times T_{\text{AR}}
$$

以 $L = 80$（70B 模型）：$T_{\text{comm\_overhead}} = 160 \times 5 \mu$s $= 0.8$ ms。

与 TPOT $\approx 5$–$10$ ms 相比，通信开销约 $8$–$16\%$——**可接受但不可忽略**。

---

## 4. Pipeline Parallelism (PP) 详解

### 4.1 切分策略

将 $L$ 层分为 $P_{\text{PP}}$ 个 Stage，每个 Stage 包含 $L / P_{\text{PP}}$ 层。

### 4.2 通信量

Stage 之间传递激活值：
$$
n_{\text{bytes\_PP}} = B \times T \times d_{\text{model}} \times s
$$

**通信频率低**（仅在 Stage 边界，而非每层），但受限于跨机带宽。

### 4.3 气泡率（Bubble Ratio）

推理时（自回归 Decode），每步只有 1 个 token，Pipeline 中只有一个 Stage 在工作，其余空闲：

$$
\text{Bubble}_{\text{decode}} = \frac{P_{\text{PP}} - 1}{P_{\text{PP}}}
$$

以 $P_{\text{PP}} = 4$：$\text{Bubble} = 75\%$——**极其浪费**！

**结论**：PP 在 Decode 阶段效率极差，应尽量用 TP 替代。PP 更适合 Prefill 阶段（可通过 Micro-batch 填充 Pipeline）：

$$
\text{Bubble}_{\text{prefill}} = \frac{P_{\text{PP}} - 1}{P_{\text{PP}} - 1 + N_{\text{micro}}}
$$

---

## 5. Expert Parallelism (EP) 详解

### 5.1 切分策略

将 $E$ 个 Expert 分配到 $P_{\text{EP}}$ 张卡上，每张卡 $E / P_{\text{EP}}$ 个 Expert。

### 5.2 All-to-All 通信

每个 MoE 层需要 **2 次 All-to-All**：
1. **Dispatch**：每张卡将 token 路由到持有目标 Expert 的卡。
2. **Combine**：Expert 计算完成后将结果返回原始卡。

每次 All-to-All 的数据量（最坏情况）：
$$
n_{\text{bytes\_A2A}} = B \times T \times d_{\text{model}} \times s
$$

### 5.3 负载不均衡

若路由不均匀，部分卡过载、部分卡空闲。**等效延迟取决于最慢的卡**：

$$
T_{\text{EP\_step}} = \max_{j=1}^{P_{\text{EP}}} \left( T_{\text{compute}}(j) + T_{\text{comm}}(j) \right)
$$

训练时通过辅助损失 $\mathcal{L}_{\text{balance}}$ 缓解，推理时需要 Token Dropping 或 Capacity Factor 控制。

---

## 6. 跨卡 KV Cache 迁移分析

在 Prefill-Decode 分离或请求迁移场景下，需要传输 KV Cache：

$$
M_{\text{migrate}} = \text{bytes\_per\_token} \times T_{\text{cache}}
$$

**迁移决策准则**：
$$
\text{Benefit}_{\text{reuse}} > T_{\text{comm}} + T_{\text{reindex}}
$$

其中 $\text{Benefit}_{\text{reuse}}$ 是避免重算的收益，$T_{\text{comm}}$ 是传输延迟，$T_{\text{reindex}}$ 是页表重映射开销。

对短序列请求，**迁移通常不划算**（传输成本 > 重算成本）。

---

## 7. 典型部署配置

| 模型 | GPU 配置 | TP | PP | EP | 说明 |
|------|---------|:--:|:--:|:--:|------|
| 7B | 1× A100 | 1 | 1 | — | 单卡可放下 |
| 70B (BF16) | 4× A100 | 4 | 1 | — | 权重 140 GB，4 卡 TP |
| 70B (INT4) | 1× A100 | 1 | 1 | — | 权重 35 GB，单卡 |
| Mixtral 8×7B | 2× A100 | 2 | 1 | 2 | TP 切 Attention，EP 切 Expert |
| 405B | 8× H100 | 8 | 1 | — | 全 TP |

---

## 8. 多副本负载均衡

### 8.1 Join-the-Shortest-Queue (JSQ)

$$
\text{Route request to } j^* = \arg\min_j \text{QueueDepth}(j)
$$

### 8.2 预测负载

考虑已有请求的剩余生成长度：
$$
\text{EstimatedLoad}(j) = \sum_{\text{req} \in j} \text{RemainingTokens}_{\text{req}} \times \text{TPOT}
$$

将新请求分配到 $\text{EstimatedLoad}$ 最小的副本。

---

## 面试一句话

> "分布式推理的上限由通信决定，不是纯算力。TP 减延迟但每层需要 All-Reduce，Decode 时 $\alpha$ 主导；PP 省通信但有气泡，Decode 时气泡率高达 $(P-1)/P$。实际部署以 TP 为主，跨机再加 PP/EP。"