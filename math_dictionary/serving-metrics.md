# 推理服务指标体系数学详解

> **核心定位**：建立 LLM 推理服务的完整监控指标体系，给出每个指标的精确数学定义、计算公式、指标间的推导关系，以及异常诊断的决策树。

---

## 1. 核心延迟指标

### 1.1 精确定义

$$
\boxed{\text{TTFT} = t_{\text{first\_token\_out}} - t_{\text{request\_arrive}}}
$$

$$
\boxed{\text{TPOT} = \frac{t_{\text{last\_token\_out}} - t_{\text{first\_token\_out}}}{N_{\text{out}} - 1}}
$$

$$
\boxed{\text{E2E Latency} = t_{\text{last\_token\_out}} - t_{\text{request\_arrive}} = \text{TTFT} + (N_{\text{out}} - 1) \times \text{TPOT}}
$$

$$
\text{ITL}_i = t_{\text{token}_i} - t_{\text{token}_{i-1}} \quad (\text{Inter-Token Latency，逐 token 粒度})
$$

### 1.2 TTFT 的分解

$$
\text{TTFT} = \underbrace{T_{\text{queue}}}_{\text{排队等待}} + \underbrace{T_{\text{prefill}}}_{\text{Prefill 计算}} + \underbrace{T_{\text{scheduling}}}_{\text{调度开销}}
$$

- **无 Chunked Prefill 时**：$T_{\text{prefill}} \approx 2NT_{\text{in}} / (\text{FLOPS} \times \text{MFU})$。
- **有 Chunked Prefill 时**：$T_{\text{prefill}} = \lceil T_{\text{in}} / C_{\text{size}} \rceil \times T_{\text{chunk}} + T_{\text{decode\_interleave}}$。

### 1.3 TPOT 的分解

$$
\text{TPOT} \approx \frac{M_{\text{weights}} + M_{\text{KV\_step}}}{\text{BW}} + T_{\text{comm}} + T_{\text{scheduling}}
$$

其中 $M_{\text{KV\_step}} = B_{\text{active}} \times \text{bytes\_per\_token} \times T_{\text{avg\_cache}}$ 是每步需要读取的 KV Cache 量。

---

## 2. 吞吐指标

### 2.1 Token 吞吐

$$
\text{Throughput}_{\text{tok}} = \frac{\sum_{\text{requests}} N_{\text{out}}(i)}{\text{Total Time}} \quad (\text{tokens/s})
$$

与 Batch Size 的关系：
$$
\text{Throughput}_{\text{tok}} \approx \frac{B_{\text{active}}}{\text{TPOT}}
$$

### 2.2 请求吞吐

$$
\text{Throughput}_{\text{req}} = \frac{N_{\text{completed}}}{\text{Total Time}} \quad (\text{req/s})
$$

**注意**：请求吞吐与输出长度分布强相关，不宜用单一值概括。

### 2.3 有效吞吐（Goodput）

$$
\boxed{\text{Goodput} = \frac{|\{i : \text{TTFT}(i) < \text{SLO}_{\text{TTFT}} \wedge \text{TPOT}(i) < \text{SLO}_{\text{TPOT}}\}|}{\text{Total Time}}}
$$

**只统计满足 SLO 约束的请求**。裸吞吐高但大量违反 SLO 是无意义的。

---

## 3. 资源利用率指标

### 3.1 GPU 计算利用率

$$
\text{GPU Utilization} = \frac{T_{\text{active\_compute}}}{T_{\text{total}}}
$$

### 3.2 KV Cache 利用率

$$
\rho_{\text{KV}} = \frac{N_{\text{used\_blocks}}}{N_{\text{total\_blocks}}}
$$

### 3.3 Batch 利用率

$$
\text{Batch Utilization} = \frac{\bar{B}_{\text{active}}}{B_{\max}}
$$

---

## 4. 缓存与驱逐指标

### 4.1 KV 命中率（Prefix Caching 场景）

$$
\text{KV Hit Rate} = \frac{N_{\text{prefix\_hits}}}{N_{\text{prefix\_lookups}}}
$$

### 4.2 回填率（Refill Rate）

$$
\text{Refill Rate} = \frac{N_{\text{evicted\_then\_recomputed}}}{N_{\text{total\_evictions}}}
$$

Refill Rate 高说明**驱逐策略存在严重误判**——被驱逐的 token 很快又被需要。

### 4.3 其他运维指标

$$
\text{OOM Rate} = \frac{N_{\text{OOM}}}{N_{\text{total}}}
$$
$$
\text{Preemption Rate} = \frac{N_{\text{preempted}}}{N_{\text{total}}}
$$

---

## 5. 指标间的推导关系

### 5.1 最大 Batch Size 估算

$$
B_{\max} \approx \frac{M_{\text{KV\_budget}}}{\text{bytes\_per\_token} \times \bar{T}_{\text{cache}}}
$$

### 5.2 吞吐与延迟的关系

由 Little 定律：
$$
\bar{B}_{\text{active}} = \text{Throughput}_{\text{req}} \times \overline{\text{E2E}}
$$

### 5.3 Goodput 与 SLO 的关系

$$
\text{Goodput} \le \text{Throughput}_{\text{req}}
$$

等号当且仅当所有请求都满足 SLO 时成立。

---

## 6. 异常诊断决策树

| 现象 | 首查指标 | 可能原因 | 建议操作 |
|------|---------|---------|---------|
| TPOT 升高 | Batch Util / KV Util | Batch 过大 → Memory-bound 恶化 | 减小 Batch 或量化 KV |
| TTFT 升高 | Queue Depth | Prefill 被 Decode 挤压 | 启用 Chunked Prefill |
| 吞吐高但 TTFT 差 | Prefill/Decode 比例 | Decode Batch 过大占满资源 | 设定 Prefill 准入配额 |
| 命中率升但 TPOT 差 | Refill Rate / Dequant | 回迁/反量化抖动 | 增加回迁预算 / 减少驱逐频率 |
| P99 剧烈抖动 | KV Eviction Spike / Comm | 驱逐风暴或跨卡通信峰值 | 预留 KV Buffer / 优化通信 |
| GPU Util 低但 TPOT 高 | AI (Arithmetic Intensity) | Memory-bound | 增大 Batch / 量化权重 |
| Queue Depth 持续增长 | $\rho$ (利用率) | 服务容量不足 | 扩容或限流 |

---

## 7. 告警阈值参考

| 指标 | 预警阈值 | 紧急阈值 | 说明 |
|------|---------|---------|------|
| $\rho_{\text{KV}}$ | $> 80\%$ | $> 90\%$ | 准备触发驱逐 |
| P99 TTFT | $> 0.7 \times \text{SLO}$ | $> 0.9 \times \text{SLO}$ | 接近 SLO 上限 |
| OOM Rate | $> 0.01\%$ | $> 0.1\%$ | 调整 Batch 策略 |
| Refill Rate | $> 5\%$ | $> 15\%$ | 优化驱逐策略 |
| Queue Depth | $> 2 \times B_{\max}$ | 持续增长 | 扩容或限流 |

---

## 面试一句话

> "推理指标必须分层看：延迟看 TTFT + TPOT + P99，吞吐看 Goodput（不是裸吞吐），资源看 KV 利用率和 Batch 利用率，诊断看 Refill Rate 和 Queue Depth。单一指标好不代表系统好，指标之间的推导关系（Little 定律、Roofline）才是面试核心。"

---

## 对应源码与阅读顺序

- 先读 [../notes/serving/formula-to-code-walkthrough.md](../notes/serving/formula-to-code-walkthrough.md)，把 TTFT、TPOT、Goodput、batch utilization 的定义和调度器行为对上。
- 再对照 [../src/simulators/serving_metrics.py](../src/simulators/serving_metrics.py) 的 `ttft()`、`tpot()`、`goodput()`、`batch_utilization()`、`kv_step_bytes()`，确认每个指标都是逐公式实现。
- 然后读 [../src/simulators/scheduler.py](../src/simulators/scheduler.py) 的 `Request.stage()` 和 `step()`，理解 decode 优先为何会同时影响 TTFT 和 TPOT。
- 最后跑 `python -m pytest tests/test_serving_metrics.py tests/test_scheduler.py -v`，把指标和调度逻辑一起验证。
