# 从公式到源码：Serving 指标 / 调度 / Goodput 对照手册

> 这一页把推理服务里最容易混掉的几件事串成一条主线：请求先经历排队和 prefill，随后进入 decode；吞吐和 Goodput 不是一回事；continuous batching 的调度规则会同时改写 TTFT、TPOT 和 KV 带宽压力。读完之后，你应该能把指标、调度、容量和源码放进同一张账本里理解。

## 这页覆盖哪些源码

- [../../src/simulators/serving_metrics.py](../../src/simulators/serving_metrics.py)：TTFT、TPOT、E2E、Goodput、服务需求、batch utilization、KV 步带宽下界。
- [../../src/simulators/scheduler.py](../../src/simulators/scheduler.py)：请求状态机、decode 优先、prefill chunking。
- [../../tests/test_serving_metrics.py](../../tests/test_serving_metrics.py)：把关键闭式公式钉在最小可执行样例上。
- [queueing-slo-formula-to-code-walkthrough.md](queueing-slo-formula-to-code-walkthrough.md)：继续往 Little 定律、M/M/1、Erlang C 延伸。

## 1. 先把一次请求的服务预算拆开

### 1.1 TTFT / TPOT / E2E 的定义

对单个请求，记请求到达时间为 $t_{\text{arrive}}$，首 token 输出时间为 $t_{\text{first}}$，最后一个 token 输出时间为 $t_{\text{last}}$，输出 token 数为 $N_{\text{out}}$，则

$$
\text{TTFT} = t_{\text{first}} - t_{\text{arrive}}
$$

$$
\text{TPOT} =
\begin{cases}
0, & N_{\text{out}} \le 1 \\
\frac{t_{\text{last}} - t_{\text{first}}}{N_{\text{out}} - 1}, & N_{\text{out}} > 1
\end{cases}
$$

$$
\text{E2E} = t_{\text{last}} - t_{\text{arrive}}
$$

在 [../../src/simulators/serving_metrics.py](../../src/simulators/serving_metrics.py) 里，这三条式子直接落成了 `ttft()`、`tpot()` 和 `e2e_latency()`。

### 1.2 TTFT 其实是三段时间之和

如果把首 token 之前的等待过程拆开，最常见的近似写法是

$$
\text{TTFT} \approx W_{\text{queue}} + T_{\text{prefill}} + T_{\text{first-decode}}
$$

其中：

- $W_{\text{queue}}$：请求在 admission、队列或 batch 等待中的时间。
- $T_{\text{prefill}}$：把输入 prompt 编码成首轮 KV 的时间。
- $T_{\text{first-decode}}$：生成首个输出 token 的时间。

这条分解式解释了一个非常常见的现象：当线上 TTFT 变差时，不代表 decode 一定变慢，也可能只是 prefill 被长 prompt 或 decode 优先调度挤压了。

### 1.3 E2E 可以由 TTFT 和 TPOT 反推

把定义联立起来，可以得到

$$
\text{E2E} = \text{TTFT} + (N_{\text{out}} - 1) \times \text{TPOT}
$$

这条关系现在有了显式源码对应：

```python
def e2e_from_ttft_tpot(ttft_value: float, tpot_value: float, output_tokens: int) -> float:
    return ttft_value + max(output_tokens - 1, 0) * tpot_value
```

以及把三者一次性打包的：

```python
def request_metrics(
    request_id: str,
    request_arrive: float,
    first_token_out: float,
    last_token_out: float,
    output_tokens: int,
) -> RequestMetrics:
    return RequestMetrics(
        request_id=request_id,
        output_tokens=output_tokens,
        ttft=ttft(request_arrive, first_token_out),
        tpot=tpot(first_token_out, last_token_out, output_tokens),
        e2e_latency=e2e_latency(request_arrive, last_token_out),
    )
```

所以线上一旦看到 E2E 抖动，第一步不该是盯着总时延发愁，而是立刻反问：是 $W_{\text{queue}}$ 变大了，还是 TPOT 被 decode 路径拖慢了？

## 2. 吞吐和 Goodput 不在同一层

### 2.1 裸吞吐只回答“产出多少”

若总时长为 $T_{\text{total}}$，系统总共完成了 $N_{\text{req}}$ 个请求并生成了 $\sum_i N_{\text{out}}^{(i)}$ 个 token，则

$$
\text{Request Throughput} = \frac{N_{\text{req}}}{T_{\text{total}}}
$$

$$
\text{Token Throughput} = \frac{\sum_i N_{\text{out}}^{(i)}}{T_{\text{total}}}
$$

源码里分别对应 `request_throughput()` 和 `token_throughput()`。

### 2.2 Goodput 关心“满足 SLO 的吞吐”

若 TTFT 和 TPOT 的 SLO 阈值分别是 $\tau_{\text{TTFT}}$ 和 $\tau_{\text{TPOT}}$，则满足条件的请求指标可以写成

$$
I_i = \mathbb{1}\left\{\text{TTFT}_i \le \tau_{\text{TTFT}},\ \text{TPOT}_i \le \tau_{\text{TPOT}}\right\}
$$

于是 Goodput 为

$$
\text{Goodput} = \frac{1}{T_{\text{total}}} \sum_i I_i
$$

在实现上，我把它拆成了两段：先算满足率，再乘请求吞吐。

```python
def goodput_ratio(metrics: Iterable[RequestMetrics], ttft_slo: float, tpot_slo: float) -> float:
    metrics = list(metrics)
    satisfied = sum(m.ttft <= ttft_slo and m.tpot <= tpot_slo for m in metrics)
    return satisfied / len(metrics)


def goodput(
    metrics: Iterable[RequestMetrics],
    total_time: float,
    ttft_slo: float,
    tpot_slo: float,
) -> float:
    metrics = list(metrics)
    return request_throughput(len(metrics), total_time) * goodput_ratio(metrics, ttft_slo, tpot_slo)
```

这对应着一个很实用的推导：

$$
\text{Goodput} = \text{Request Throughput} \times \text{SLO Attainment Ratio}
$$

所以“吞吐上去了，但 Goodput 下来了”并不矛盾，它通常意味着请求处理得更多了，但满足 SLO 的比例在下降。

### 2.3 一个两秒窗口的直观例子

假设 2 秒内完成 2 个请求：

- 裸请求吞吐是 $2 / 2 = 1$ req/s。
- 若只有 1 个请求同时满足 TTFT 和 TPOT 的 SLO，则满足率是 $1 / 2 = 0.5$。
- 因而 Goodput 是 $1 \times 0.5 = 0.5$ req/s。

这正是 [../../tests/test_serving_metrics.py](../../tests/test_serving_metrics.py) 里 `test_throughput_and_goodput()` 在验证的事情。

## 3. 从请求长度到服务需求

### 3.1 为什么输入长度和输出长度要分开记账

Serving 里一个请求的负载并不是单一的“时长”，而是两部分：

- 输入 token 决定 prefill 成本。
- 输出 token 决定 decode 成本。

因此最常用的近似服务需求写法是

$$
D_{\text{req}} \approx N_{\text{in}} \cdot c_{\text{prefill}} + N_{\text{out}} \cdot c_{\text{decode}}
$$

其中 $c_{\text{prefill}}$ 是每个输入 token 的 prefill 平均代价，$c_{\text{decode}}$ 是每个输出 token 的 decode 平均代价。

### 3.2 对应源码

```python
def request_service_demand(
    input_tokens: int,
    output_tokens: int,
    prefill_seconds_per_token: float,
    decode_seconds_per_token: float,
) -> float:
    return input_tokens * prefill_seconds_per_token + output_tokens * decode_seconds_per_token
```

它的意义不是“精准建模 GPU 执行时间”，而是提供一个够用的账本：你至少能区分一个请求到底是被长 prompt 拖慢，还是被长输出拖慢。

### 3.3 为什么这一步能接到排队论

如果平均服务需求大致是 $\mathbb{E}[D_{\text{req}}]$，那么单副本的平均服务率近似满足

$$
\mu_{\text{req}} \approx \frac{1}{\mathbb{E}[D_{\text{req}}]}
$$

接下来再把它带到 [queueing-slo-formula-to-code-walkthrough.md](queueing-slo-formula-to-code-walkthrough.md) 的 $\rho = \lambda / \mu$、M/M/1 和 M/G/1 里，就能把“请求长度分布”接到“系统是否会炸”上。

## 4. 调度器如何改写 TTFT 和 TPOT

### 4.1 最小状态机：prefill -> decode -> done

在 [../../src/simulators/scheduler.py](../../src/simulators/scheduler.py) 中，一个请求只经过三种状态：

$$
\text{prefill} \rightarrow \text{decode} \rightarrow \text{done}
$$

对应的判断逻辑是：

```python
def stage(self) -> str:
    if self.finished:
        return "done"
    if self.prefilled < self.input_tokens:
        return "prefill"
    return "decode"
```

这个状态机很小，但它已经捕捉到 serving 的主矛盾：prefill 是大块写入和大算子，decode 是小步迭代但对交互延迟更敏感。

### 4.2 decode first 的分配规则

记第 $t$ 个调度步的最大 batch 容量为 $B_{\max}$，活跃 decode 请求数为 $B_t^{\text{decode}}$，则 prefill 能用的剩余位置是

$$
B_t^{\text{prefill}} = B_{\max} - B_t^{\text{decode}}
$$

而源码正是按这条规则写的：

```python
decode_reqs = [r for r in active if r.stage() == "decode"]
prefill_reqs = [r for r in active if r.stage() == "prefill"]

selected_decode = decode_reqs[: self.max_batch_size]
remaining_slots = self.max_batch_size - len(selected_decode)
selected_prefill = prefill_reqs[:remaining_slots]
```

这意味着：

- decode 越多，prefill 可用配额越少。
- decode first 往往有利于 TPOT，但容易先伤到 TTFT。
- 交互业务一多，长 prompt 请求通常先感受到排队和 chunking 代价。

### 4.3 chunked prefill 的离散更新式

对某个被选中的 prefill 请求，设输入长度为 $T_{\text{in}}^{(i)}$，当前已完成 prefill 的 token 数为 $p_t^{(i)}$，每步最多推进 $C_{\text{prefill}}$ 个 token，则

$$
p_{t+1}^{(i)} = \min\left(T_{\text{in}}^{(i)},\; p_t^{(i)} + C_{\text{prefill}}\right)
$$

源码里对应 `prefill_chunk`：

```python
req.prefilled = min(req.input_tokens, req.prefilled + self.prefill_chunk)
```

同理，decode 每次只推进 1 个输出 token：

$$
d_{t+1}^{(i)} = d_t^{(i)} + 1
$$

在满足

$$
d_{t+1}^{(i)} \ge N_{\text{out}}^{(i)}
$$

时进入 `done`。

### 4.4 这套调度为什么会产生此消彼长

把上面的更新式放在一起看，你就能解释常见的指标现象：

- `prefill_chunk` 调大，单次 prefill 推进更快，长 prompt 的 TTFT 可能改善；但它也会更像“重型任务”，更容易挤占 decode 的调度机会。
- decode 请求数一多，`remaining_slots` 变小，prefill 更容易排队，TTFT 会先恶化。
- 即使裸 token throughput 还在上涨，用户感知也可能已经开始变差。

## 5. batch utilization 不是越高越好

### 5.1 数学定义

若第 $t$ 个调度步的活跃 batch 大小为 $b_t$，共观察 $T$ 个调度步，则平均 batch 利用率为

$$
U_{\text{batch}} = \frac{1}{T \times B_{\max}} \sum_{t=1}^{T} b_t
$$

源码实现为：

```python
def batch_utilization(active_batch_history: Sequence[int], max_batch_size: int) -> float:
    return sum(active_batch_history) / (len(active_batch_history) * max_batch_size)
```

### 5.2 为什么高利用率不一定是好事

若请求吞吐为 $\lambda_{\text{req}}$，平均端到端时延为 $\overline{\text{E2E}}$，则由 Little 定律可近似得到

$$
\bar{B}_{\text{active}} \approx \lambda_{\text{req}} \times \overline{\text{E2E}}
$$

所以你在图上看到 batch utilization 变高时，既可能意味着 GPU 更吃满，也可能意味着队列和 E2E 一起涨了。单独追求高利用率，往往会把服务系统从“忙但稳定”推向“忙且失控”。

## 6. KV 带宽如何把 decode 推向 memory-bound

### 6.1 每个 decode 步要扫多少 KV

设当前活跃 batch 为 $B_{\text{active}}$，每个 token 的 KV 开销是 `bytes_per_token`，平均缓存长度为 $\bar{T}_{\text{cache}}$，则一个 decode 步需要读取的 KV 数据量近似是

$$
\text{KV\_bytes\_per\_step} = B_{\text{active}} \times \text{bytes\_per\_token} \times \bar{T}_{\text{cache}}
$$

对应实现：

```python
def kv_step_bytes(active_batch: int, bytes_per_token: int, avg_cache_tokens: int) -> int:
    return active_batch * bytes_per_token * avg_cache_tokens
```

### 6.2 带宽下界给出 TPOT 的硬约束

如果显存可提供的有效带宽是 $BW_{\text{mem}}$，那么仅从搬运 KV 的角度看，每一步 decode 时间都满足下界

$$
T_{\text{decode-step}} \ge \frac{\text{KV\_bytes\_per\_step}}{BW_{\text{mem}}}
$$

源码对应：

```python
def kv_step_time_lower_bound(
    active_batch: int,
    bytes_per_token: int,
    avg_cache_tokens: int,
    memory_bandwidth_bytes_per_s: float,
) -> float:
    return kv_step_bytes(active_batch, bytes_per_token, avg_cache_tokens) / memory_bandwidth_bytes_per_s
```

这条式子非常重要，因为它说明：上下文越长、batch 越大、KV 表示越胖，decode 就越容易被内存带宽锁死，而不是被算力锁死。

### 6.3 它和 TPOT 的关系

若一次 decode 迭代的总时长近似是“算子时间”和“搬运时间”的较大者，则可以写成

$$
\text{TPOT} \approx \max\left(T_{\text{compute}},\; T_{\text{decode-step}}\right)
$$

因此一旦你看到 TTFT 还算稳定，但 TPOT 随上下文长度明显变差，优先怀疑 KV 访问和 decode 调度，而不是先怀疑 admission 层。

## 7. 观测指标后，怎么反推问题落点

### 7.1 TTFT 高、TPOT 稳

优先怀疑：

- admission 太紧或排队过长；
- prefill 被 decode first 长时间挤压；
- prompt 偏长，导致 $T_{\text{prefill}}$ 占比升高。

### 7.2 TTFT 稳、TPOT 高

优先怀疑：

- KV 扫描量过大；
- decode 路径 memory-bound；
- 调度抖动让 decode 请求拿不到稳定配额。

### 7.3 吞吐高、Goodput 低

通常意味着：

- 系统仍在拼命出活，但 SLO 违约率上升；
- batch 利用率提高了，但用户体验没有同步改善；
- 这时更该去看 `goodput_ratio()`，而不是只看 `token_throughput()`。

### 7.4 batch utilization 高、E2E 也高

这不一定是“调得好”，也可能是 Little 定律在提醒你：系统里积压的活跃请求更多了。此时应继续接到 [queueing-slo-formula-to-code-walkthrough.md](queueing-slo-formula-to-code-walkthrough.md)，用 $\rho$、M/M/1 或 M/G/1 判断是否已接近过载。

## 8. 建议的源码阅读顺序

1. 先读 [../../math_dictionary/serving-metrics.md](../../math_dictionary/serving-metrics.md)，把 TTFT、TPOT、Goodput 和服务预算串起来。
2. 再读 [../../src/simulators/serving_metrics.py](../../src/simulators/serving_metrics.py)，对照 `e2e_from_ttft_tpot()`、`goodput_ratio()`、`request_service_demand()`、`kv_step_time_lower_bound()`。
3. 接着读 [../../src/simulators/scheduler.py](../../src/simulators/scheduler.py)，把 `prefill -> decode -> done` 状态机、decode first 和 `prefill_chunk` 的更新逻辑看懂。
4. 然后跑 [../../tests/test_serving_metrics.py](../../tests/test_serving_metrics.py) 和 [../../tests/test_scheduler.py](../../tests/test_scheduler.py)，确认公式和最小调度行为在代码里是一致的。
5. 最后接上 [queueing-slo-formula-to-code-walkthrough.md](queueing-slo-formula-to-code-walkthrough.md)、[capacity-planning.md](capacity-planning.md) 和 [cost-optimization.md](cost-optimization.md)，把指标、排队、容量和成本闭环起来。

## 这一页记住一句话

> Serving 优化不是单纯把 token 吞吐堆高，而是把“排队 + prefill + decode + KV 带宽”这整条链路压到 SLO 预算之内。TTFT 告诉你首 token 之前卡在哪，TPOT 告诉你后续流式生成卡在哪，Goodput 则告诉你这些优化到底有没有转化成真实可交付的服务能力。
