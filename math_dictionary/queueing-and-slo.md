# 排队论与 SLO 数学详解

> **核心定位**：从排队论的基本定理（Little 定律、M/M/1、M/G/1）出发，严格推导 LLM 推理服务中的延迟分布、尾延迟爆炸机制和利用率-延迟的非线性关系，并给出 SLO 预算分解、自适应限流和优先级调度的数学框架。

---

## 1. Little 定律 (Little's Law)

$$
\boxed{L = \lambda \cdot W}
$$

| 符号 | 含义 |
|------|------|
| $L$ | 系统内平均请求数（含排队 + 服务中） |
| $\lambda$ | 请求到达率（req/s） |
| $W$ | 平均逗留时间（排队 + 服务） |

**普适性**：对任何稳态排队系统成立，**不需要假设到达分布或服务时间分布**。

**LLM 推理中的应用**：
$$
\bar{B}_{\text{active}} = \lambda \cdot \overline{E2E}
$$

---

## 2. M/M/1 模型（单服务器基础）

### 2.1 假设

- 到达过程：泊松过程，速率 $\lambda$
- 服务时间：指数分布，速率 $\mu$
- 单服务器（单 GPU 实例）

### 2.2 核心公式

$$
\rho = \frac{\lambda}{\mu} \quad (\text{利用率，要求 } \rho < 1)
$$

$$
\bar{W} = \frac{1}{\mu - \lambda} = \frac{1}{\mu(1 - \rho)} \quad (\text{平均逗留时间})
$$

$$
\bar{W}_q = \frac{\rho}{\mu - \lambda} = \frac{\rho}{\mu(1 - \rho)} \quad (\text{平均排队时间})
$$

$$
\bar{L} = \frac{\rho}{1 - \rho} \quad (\text{系统内平均请求数})
$$

### 2.3 利用率-延迟的非线性关系

$$
\bar{W} = \frac{1}{\mu(1-\rho)} \quad \Rightarrow \quad \text{当 } \rho \to 1 \text{ 时，} \bar{W} \to \infty
$$

| $\rho$ | $\bar{W} / (1/\mu)$（延迟倍数） |
|:------:|:--------------------------:|
| $0.5$ | $2\times$ |
| $0.7$ | $3.3\times$ |
| $0.8$ | $5\times$ |
| $0.9$ | $10\times$ |
| $0.95$ | $20\times$ |

**经验法则**：$\rho < 0.7$ 延迟可控，$\rho > 0.85$ 进入危险区。

---

## 3. M/M/c 模型（多服务器）

$c$ 台服务器并行处理，利用率 $\rho = \lambda / (c \mu)$。

**Erlang C 公式**给出请求需要排队的概率：

$$
P_{\text{wait}} = \frac{\frac{(c\rho)^c}{c!(1-\rho)}}{\sum_{k=0}^{c-1} \frac{(c\rho)^k}{k!} + \frac{(c\rho)^c}{c!(1-\rho)}}
$$

$$
\bar{W}_q = P_{\text{wait}} \cdot \frac{1}{c\mu(1-\rho)}
$$

**工程含义**：多 GPU/多实例 ≈ 多服务器。但负载均衡质量决定实际是否接近理论值——不均衡的负载等效于减少 $c$。

---

## 4. M/G/1 模型（通用服务时间）

### 4.1 Pollaczek-Khinchine 公式

当服务时间不再是指数分布（LLM 输出长度差异大！），使用 M/G/1 模型：

$$
\boxed{\bar{W}_q = \frac{\rho}{2(1-\rho)} \cdot \frac{1 + C_s^2}{\mu}}
$$

其中 **服务时间变异系数**：
$$
C_s = \frac{\sigma_s}{\mathbb{E}[S]} = \frac{\text{Std}(\text{服务时间})}{\text{Mean}(\text{服务时间})}
$$

### 4.2 关键洞察

$$
\bar{W}_q \propto (1 + C_s^2)
$$

**服务时间方差越大，排队越严重**。

- 指数分布：$C_s = 1$，$\bar{W}_q \propto 2$。
- 确定性服务：$C_s = 0$，$\bar{W}_q \propto 1$（排队减半！）。
- LLM 场景：输出长度可以从 1 到 $> 4096$ token，$C_s \gg 1$，排队**极其严重**。

**这就是为什么**：
1. 输出长度预测很重要（减小 $C_s$）。
2. Continuous Batching 比 Static Batching 好（将长请求的剩余服务时间从批次整体中解耦）。
3. Shortest-Job-First 调度能显著降低平均延迟。

---

## 5. 尾延迟分析

### 5.1 M/M/1 的尾延迟

逗留时间的互补 CDF（CCDF）：
$$
\Pr[W > t] = \rho \cdot e^{-(\mu - \lambda) t}
$$

**P99 延迟**（$\Pr[W > t_{99}] = 0.01$）：
$$
t_{99} = \frac{-\ln(0.01 / \rho)}{\mu - \lambda} = \frac{\ln(100\rho)}{\mu(1-\rho)}
$$

### 5.2 P99 随利用率的爆炸

| $\rho$ | $t_{99} / (1/\mu)$ |
|:------:|:------------------:|
| $0.5$ | $\sim 7.8$ |
| $0.7$ | $\sim 13.4$ |
| $0.9$ | $\sim 46.1$ |

**$\rho$ 从 $0.5 \to 0.9$，P99 膨胀 $\sim 6\times$**。

实际系统的尾延迟通常比理论模型更差（因为还有 GC、通信抖动、KV Cache 驱逐等额外延迟源）。

---

## 6. SLO 预算分解

将端到端 SLO（如 P99 $< 2$ s）拆解为各子组件的预算：

$$
\text{SLO}_{\text{E2E}} = \underbrace{\text{SLO}_{\text{queue}}}_{\text{排队}} + \underbrace{\text{SLO}_{\text{prefill}}}_{\text{Prefill}} + \underbrace{\text{SLO}_{\text{decode}} + \text{SLO}_{\text{network}}}_{\text{Decode + 网络}}
$$

**示例**：

| 子项 | P99 预算 |
|------|---------|
| 排队 | $300$ ms |
| Prefill | $700$ ms |
| Decode ($N_{\text{out}} = 200$ tokens × $\text{TPOT}$) | $800$ ms |
| 网络传输 | $200$ ms |
| **合计** | **$2000$ ms** |

任何子项超预算 → 触发**降级**（减少输出长度上限）或**限流**（拒绝新请求）。

---

## 7. 自适应限流策略

### 7.1 Token Bucket

$$
\text{tokens}(t) = \min(\text{tokens}(t-\Delta t) + r \cdot \Delta t, \; B_{\text{burst}})
$$

- $r$：令牌补充速率（控制平均到达率）。
- $B_{\text{burst}}$：桶容量（控制突发流量上限）。

### 7.2 多信号联合限流

$$
\text{Admission Decision} = \begin{cases}
\text{Accept} & \text{if } Q_{\text{depth}} < \theta_Q \text{ AND } \rho_{\text{KV}} < \theta_{\text{KV}} \text{ AND } \text{P99} < 0.8 \cdot \text{SLO} \\
\text{Reject / Queue} & \text{otherwise}
\end{cases}
$$

| 信号 | 阈值示例 | 含义 |
|------|---------|------|
| 队列深度 $Q_{\text{depth}}$ | $> 50$ | 排队请求过多 |
| KV 利用率 $\rho_{\text{KV}}$ | $> 90\%$ | 显存即将耗尽 |
| 实时 P99 | $> 0.8 \times \text{SLO}$ | 接近 SLO 上限 |

---

## 8. 优先级调度

### 8.1 多级优先级队列

VIP 请求优先调度。使用 **Aging 机制**防止普通请求饥饿：

$$
\text{EffectivePriority}(i) = \text{BasePriority}(i) + \text{WaitTime}(i) \times \alpha_{\text{aging}}
$$

### 8.2 Shortest-Job-First (SJF) 近似

预测输出长度 $\hat{N}_{\text{out}}$，短请求先服务：

$$
\text{ScheduleOrder} = \arg\min_i \hat{N}_{\text{out}}(i)
$$

**理论优势**：SJF 在 M/G/1 模型下最小化平均等待时间（$\bar{W}_q$）。

**实践问题**：长请求可能被无限推迟。结合 Aging 使用：$\text{SortKey}(i) = \hat{N}_{\text{out}}(i) - \alpha \cdot \text{WaitTime}(i)$。

---

## 面试一句话

> "LLM 推理的排队抖动常被低估：输出长度方差高 $\to$ $C_s$ 大 $\to$ M/G/1 的 P-K 公式告诉我们排队时间正比于 $(1 + C_s^2)$ $\to$ 尾延迟爆炸。优化手段：降 $\rho$（扩容）、降 $C_s$（SJF + 长度预测）、硬限流（Token Bucket + KV 利用率门控）。"