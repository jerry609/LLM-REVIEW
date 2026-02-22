# Prefill / Decode 延迟与吞吐模型数学详解

> **核心定位**：从 Roofline Model 出发，严格推导 Prefill（Compute-bound）和 Decode（Memory-bound）的延迟模型，深入剖析 Continuous Batching、Chunked Prefill、Prefill-Decode 分离（Disaggregation）的数学原理与工程权衡。

---

## 1. 延迟分解公式

### 1.1 核心延迟指标

$$
\boxed{\text{TTFT} = T_{\text{queue}} + T_{\text{prefill}} + T_{\text{first\_decode}}}
$$

$$
\boxed{\text{E2E Latency} = \text{TTFT} + N_{\text{out}} \times \text{TPOT}}
$$

| 指标 | 全称 | 含义 |
|------|------|------|
| **TTFT** | Time To First Token | 从请求到达到返回第一个 token 的时间 |
| **TPOT** | Time Per Output Token | 每生成一个 token 的平均时间 |
| **E2E** | End-to-End Latency | 完整请求的总延迟 |

### 1.2 Prefill 延迟（Compute-bound 近似）

Prefill 阶段需要对整个输入序列做一次完整的 Transformer 前向传播。FLOPs 与输入长度 $T_{\text{in}}$ 成正比：

$$
\text{FLOPs}_{\text{prefill}} \approx 2 \times N \times T_{\text{in}}
$$

其中 $N$ 为模型参数量。则：

$$
T_{\text{prefill}} = \frac{\text{FLOPs}_{\text{prefill}}}{\text{GPU Peak FLOPS} \times \text{MFU}}
$$

**代入示例**（7B 模型，$T_{\text{in}} = 4096$，A100 312 TFLOPS BF16，MFU $= 50\%$）：
$$
T_{\text{prefill}} = \frac{2 \times 7 \times 10^9 \times 4096}{312 \times 10^{12} \times 0.5} = \frac{5.73 \times 10^{13}}{1.56 \times 10^{14}} \approx 0.37 \text{ s}
$$

### 1.3 Decode 延迟（Memory-bound 近似）

Decode 阶段每步仅处理 1 个 token，但需要加载模型全部权重和 KV Cache。算术强度极低（$\text{AI} \approx B$），远低于 GPU 的 Roofline 拐点。

$$
\text{TPOT} \approx \frac{M_{\text{weights}} + M_{\text{KV\_accessed}}}{\text{Memory Bandwidth}}
$$

**代入示例**（7B BF16 权重 14 GB，忽略 KV，A100 BW 2 TB/s，batch=1）：
$$
\text{TPOT} \approx \frac{14 \text{ GB}}{2 \text{ TB/s}} = 7 \text{ ms/token}
$$

Batch Size 增大时，多个请求共享一次权重加载，TPOT 均摊：
$$
\text{TPOT}_{\text{batched}} \approx \frac{M_{\text{weights}}}{B \times \text{BW}} + \frac{M_{\text{KV\_per\_req}}}{\text{BW}}
$$

---

## 2. Roofline 模型与 Prefill / Decode 的分野

**算术强度 (Arithmetic Intensity)**：
$$
\text{AI} = \frac{\text{FLOPs}}{\text{Bytes Accessed}}
$$

**Roofline 拐点**：
$$
\text{AI}_{\text{ridge}} = \frac{\text{Peak FLOPS}}{\text{Peak BW}}
$$

| GPU | Peak FLOPS (BF16) | Peak BW | 拐点 AI |
|-----|:------------------:|:-------:|:-------:|
| A100 | 312 TFLOPS | 2 TB/s | $\sim 156$ |
| H100 | 990 TFLOPS | 3.35 TB/s | $\sim 295$ |

| 阶段 | AI 近似 | 性质 | 优化策略 |
|------|---------|------|---------|
| **Prefill** | $\text{AI} \propto B \times T_{\text{in}} \gg \text{AI}_{\text{ridge}}$ | **Compute-bound** | 提升 FLOPS 利用率 |
| **Decode** | $\text{AI} \propto B \ll \text{AI}_{\text{ridge}}$ | **Memory-bound** | 提升 Bandwidth 利用率 / 增大 Batch |

---

## 3. Continuous Batching（连续批处理）

### 3.1 传统 Static Batching 的问题

一个 Batch 中所有请求必须等**最长的那个**完成才能接收新请求。

$$
T_{\text{idle}} = \sum_{i \in \text{batch}} (T_{\max} - T_i) \times \text{TPOT}
$$

当输出长度方差大时，GPU 空闲时间严重浪费。

### 3.2 Continuous Batching 机制

**Iteration-level Scheduling**：每个 Decode step 独立调度。

每步检查：
1. 有请求输出 EOS 或达长度上限 → 移出 Running Pool。
2. Waiting Queue 有请求且显存（KV Cache）足够 → 加入 Running Pool。

$$
\text{Throughput}_{\text{continuous}} \approx \frac{\bar{B}_{\text{active}}}{\text{TPOT}}
$$

对比 Static Batching，吞吐提升可达 **2–5×**（取决于输出长度分布的方差）。

---

## 4. Chunked Prefill（分块预填充）

### 4.1 动机

长 Prompt 的 Prefill 可能需要数百毫秒，期间会**阻塞所有 Decode 请求**，导致 TPOT 出现巨大毛刺。

### 4.2 机制

将长 Prompt 切分为固定大小的 Chunk（如 $512$ token），每个 Chunk 作为一个"迭代"与 Decode 请求**交替执行**：

$$
T_{\text{chunk}} = \frac{2 N \times C_{\text{size}}}{\text{FLOPS} \times \text{MFU}}
$$

每个 Chunk 完成后，调度器有机会插入 Decode 迭代。

| Chunk Size | Prefill 效率 | Decode 抖动 |
|:----------:|:----------:|:-----------:|
| 大 | 高（更接近纯 Prefill 效率） | 大（阻塞时间长） |
| 小 | 低（Kernel launch 开销占比高） | 小（可及时服务 Decode） |

经验值：$C_{\text{size}} = 256$–$512$ 是较好的平衡点。

---

## 5. Prefill-Decode 分离 (Disaggregation)

### 5.1 核心思想

Prefill 是 **Compute-bound**，Decode 是 **Memory-bound**。两者对硬件的需求完全不同，混在一起必然导致资源利用率不佳。

$$
\begin{cases}
\text{Prefill Instance}: & \text{需要高 FLOPS（如 H100）} \\
\text{Decode Instance}: & \text{需要高 Bandwidth、大 HBM（如 H200）}
\end{cases}
$$

### 5.2 KV Cache 传输开销

Prefill 实例计算完成后，需要将 KV Cache 传输到 Decode 实例：

$$
M_{\text{transfer}} = \text{bytes\_per\_token} \times T_{\text{in}}
$$

**代入**（7B GQA，$T_{\text{in}} = 4096$）：
$$
M_{\text{transfer}} = 128 \text{ KB} \times 4096 = 512 \text{ MB}
$$

要求在可接受时间内完成传输（如 $< 100$ ms）：
$$
\text{Required BW} = \frac{512 \text{ MB}}{0.1 \text{ s}} = 5.12 \text{ GB/s}
$$

InfiniBand / RoCE（$100$–$400$ Gbps）可以满足，普通以太网可能成为瓶颈。

### 5.3 适用场景

| 场景 | Prefill/Decode 比 | 是否适合分离 |
|------|:-----------------:|:----------:|
| RAG（长 Prompt + 短回答） | 高 | ✅ 非常适合 |
| 聊天（短 Prompt + 长回答） | 低 | ❌ 不划算 |
| 代码补全（中等 Prompt + 中等输出） | 中 | ⚠️ 取决于请求量 |

---

## 6. 吞吐与延迟的权衡

### 6.1 Latency-Throughput 曲线

$$
\text{Throughput} \uparrow \quad \Leftrightarrow \quad \text{Batch Size} \uparrow \quad \Rightarrow \quad \text{TPOT} \uparrow \text{ (slightly)} + \text{TTFT} \uparrow \text{ (queue delay)}
$$

存在一个最优 Batch Size 使得 **Goodput**（满足 SLO 的有效吞吐）最大：
$$
\text{Goodput} = \frac{\text{SLO-satisfied requests}}{\text{Total time}}
$$

### 6.2 压测建议

| 维度 | 建议 |
|------|------|
| 输入长度 | 短（$<512$）/ 中（$512$–$4K$）/ 长（$>4K$）分桶 |
| 观测指标 | TTFT / TPOT / P95 / P99 / Throughput / OOM Rate / Goodput |
| 加压方式 | 逐步增加并发，绘制 Latency-Throughput 曲线 |
| SLO 设定 | TTFT $< 2$ s，TPOT $< 50$ ms（示例） |

---

## 面试一句话

> "Prefill 是 Compute-bound（优化方向：FLOPs 利用率），Decode 是 Memory-bound（优化方向：Bandwidth 利用率 + 增大 Batch）。Continuous Batching 消除等待浪费，Chunked Prefill 消除阻塞抖动，Disaggregation 让硬件各司其职。"