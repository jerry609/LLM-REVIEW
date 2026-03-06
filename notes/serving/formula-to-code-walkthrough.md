# 从公式到源码：Serving 指标 / 调度 / Goodput 对照手册

> 这一页把推理服务里最容易混掉的三件事拆开：请求延迟怎么定义，吞吐和 Goodput 怎么区分，continuous batching 为什么会让 decode 和 prefill 彼此争资源。读完之后，文档里的公式可以直接映射到 `scheduler.py` 和 `serving_metrics.py`。

## 这页覆盖哪些源码

- [../../src/simulators/scheduler.py](../../src/simulators/scheduler.py)：请求状态机、decode 优先、prefill chunking。
- [../../src/simulators/serving_metrics.py](../../src/simulators/serving_metrics.py)：TTFT、TPOT、E2E、吞吐、Goodput、batch utilization、KV 步带宽。

## 1. 延迟指标先分清楚：TTFT / TPOT / E2E

### 1.1 数学定义

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

把这三个式子联立起来，可以直接得到

$$
\text{E2E} = \text{TTFT} + (N_{\text{out}} - 1) \times \text{TPOT}
$$

这条关系很重要，因为它告诉你：E2E 不是一个独立世界，它总能拆回“首 token 等待 + 后续 token 平均生成时间”。

### 1.2 对应源码

[../../src/simulators/serving_metrics.py](../../src/simulators/serving_metrics.py) 里的实现几乎是逐公式翻译：

```python
def ttft(request_arrive: float, first_token_out: float) -> float:
    return first_token_out - request_arrive


def tpot(first_token_out: float, last_token_out: float, output_tokens: int) -> float:
    if output_tokens <= 1:
        return 0.0
    return (last_token_out - first_token_out) / (output_tokens - 1)


def e2e_latency(request_arrive: float, last_token_out: float) -> float:
    return last_token_out - request_arrive
```

如果你看到线上监控里 E2E 抖动，下一步应该先拆是 TTFT 抖，还是 TPOT 抖，而不是只盯一个总数。

### 1.3 为什么还要有 `request_metrics()`

文档里我们经常按单个公式讨论，但工程里通常更关心“对单个请求一次性算全”。所以 `request_metrics()` 把三个指标打包成 `RequestMetrics`：

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

这一步的意义不是“省三行代码”，而是把后续 Goodput 判断、SLO 检查、聚合统计都统一到了一个结构体上。

## 2. 吞吐和 Goodput 不是一回事

### 2.1 裸吞吐的定义

若系统在总时间 $T_{\text{total}}$ 内一共产生了 $\sum_i N_{\text{out}}^{(i)}$ 个输出 token，完成了 $N_{\text{done}}$ 个请求，则

$$
\text{Throughput}_{\text{tok}} = \frac{\sum_i N_{\text{out}}^{(i)}}{T_{\text{total}}}
$$

$$
\text{Throughput}_{\text{req}} = \frac{N_{\text{done}}}{T_{\text{total}}}
$$

源码对应：

```python
def token_throughput(output_tokens: Sequence[int], total_time: float) -> float:
    if total_time <= 0:
        raise ValueError("total_time must be positive")
    return sum(output_tokens) / total_time


def request_throughput(num_completed: int, total_time: float) -> float:
    if total_time <= 0:
        raise ValueError("total_time must be positive")
    return num_completed / total_time
```

### 2.2 Goodput 的定义

真正从服务视角看，只有满足 SLO 的请求才算“有效产出”。若 TTFT 阈值为 $S_{\text{ttft}}$，TPOT 阈值为 $S_{\text{tpot}}$，则

$$
\text{Goodput} = \frac{|\{ i : \text{TTFT}_i \le S_{\text{ttft}} \land \text{TPOT}_i \le S_{\text{tpot}} \}|}{T_{\text{total}}}
$$

进一步把它拆一下：

$$
\text{Goodput} = \text{Throughput}_{\text{req}} \times \frac{N_{\text{SLO-ok}}}{N_{\text{done}}}
$$

也就是说，Goodput 等于“完成请求速率”乘上“满足 SLO 的比例”。这就是为什么堆大 batch 可能让裸吞吐更高，却让 Goodput 更差。

源码里的 `goodput()` 正是这个定义：

```python
def goodput(
    metrics: Iterable[RequestMetrics],
    total_time: float,
    ttft_slo: float,
    tpot_slo: float,
) -> float:
    if total_time <= 0:
        raise ValueError("total_time must be positive")
    metrics = list(metrics)
    satisfied = sum(m.ttft <= ttft_slo and m.tpot <= tpot_slo for m in metrics)
    return satisfied / total_time
```

## 3. continuous batching 为何天然偏向 decode

### 3.1 请求状态机

在最小实现里，请求只分三种状态：

$$
\text{prefill} \rightarrow \text{decode} \rightarrow \text{done}
$$

状态判断在 [../../src/simulators/scheduler.py](../../src/simulators/scheduler.py) 的 `Request.stage()` 中完成：

```python
def stage(self) -> str:
    if self.finished:
        return "done"
    if self.prefilled < self.input_tokens:
        return "prefill"
    return "decode"
```

这个状态机虽然简单，但已经抓住了 serving 的核心矛盾：prefill 是大块写入和大算子，decode 是小步迭代但延迟敏感。

### 3.2 调度规则为什么是 decode first

设第 $t$ 个调度步的最大 batch 容量为 $B_{\max}$。若有 $B_t^{\text{decode}}$ 个 decode 请求活跃，那么本步最多只剩

$$
B_t^{\text{prefill}} = B_{\max} - B_t^{\text{decode}}
$$

个位置给 prefill。当前实现就按这个规则写：

```python
def step(self) -> None:
    self.time_step += 1
    active = self._active()
    if not active:
        return

    decode_reqs = [r for r in active if r.stage() == "decode"]
    prefill_reqs = [r for r in active if r.stage() == "prefill"]

    selected_decode = decode_reqs[: self.max_batch_size]
    for req in selected_decode:
        req.decoded += 1
        if req.decoded >= req.output_tokens:
            req.finished = True

    remaining_slots = self.max_batch_size - len(selected_decode)
    if remaining_slots <= 0:
        return

    selected_prefill = prefill_reqs[:remaining_slots]
    for req in selected_prefill:
        req.prefilled = min(req.input_tokens, req.prefilled + self.prefill_chunk)
```

这段代码表达了两个非常典型的 serving 结论：

- decode 优先，因为在线系统通常更怕 TPOT 拉长。
- prefill 只能吃剩余配额，所以请求一多，TTFT 很容易先变差。

### 3.3 prefill 和 decode 的离散时间更新式

如果把上面的代码写成离散时间方程，某个被选中的 prefill 请求满足

$$
p_{t+1}^{(i)} = \min\left(T_{\text{in}}^{(i)},\; p_t^{(i)} + C_{\text{prefill}}\right)
$$

其中 $C_{\text{prefill}}$ 对应代码里的 `prefill_chunk`。

某个被选中的 decode 请求满足

$$
d_{t+1}^{(i)} = d_t^{(i)} + 1
$$

并在满足

$$
d_{t+1}^{(i)} \ge N_{\text{out}}^{(i)}
$$

时进入 `done`。

所以这个调度器虽然小，但已经能拿来解释 TTFT 和 TPOT 为什么会出现此消彼长。

## 4. Batch utilization 和 Little 定律怎么连到一起

### 4.1 batch utilization 的公式

若第 $t$ 个调度步的活跃 batch 大小为 $b_t$，共有 $T$ 个调度步，则平均 batch 利用率是

$$
U_{\text{batch}} = \frac{1}{T \times B_{\max}} \sum_{t=1}^{T} b_t
$$

源码对应：

```python
def batch_utilization(active_batch_history: Sequence[int], max_batch_size: int) -> float:
    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive")
    if not active_batch_history:
        return 0.0
    return sum(active_batch_history) / (len(active_batch_history) * max_batch_size)
```

### 4.2 Little 定律给出的系统视角

若请求吞吐为 $\lambda_{\text{req}}$，平均端到端时延为 $\overline{\text{E2E}}$，则平均活跃请求数近似满足

$$
\bar{B}_{\text{active}} \approx \lambda_{\text{req}} \times \overline{\text{E2E}}
$$

所以你在图表上看到“batch utilization 变高”的时候，不一定是纯好事。它既可能意味着机器更吃满，也可能意味着队列和 E2E 一起涨了。

## 5. KV 带宽压力为什么会把 decode 推向 memory-bound

### 5.1 每个调度步要扫多少 KV 字节

设当前活跃 batch 为 $B_{\text{active}}$，每个 token 的 KV 开销是 `bytes_per_token`，平均缓存长度为 $\bar{T}_{\text{cache}}$，则一个 decode 步大致要处理的 KV 数据量可写成

$$
\text{KV\_bytes\_per\_step} = B_{\text{active}} \times \text{bytes\_per\_token} \times \bar{T}_{\text{cache}}
$$

源码对应函数非常直接：

```python
def kv_step_bytes(active_batch: int, bytes_per_token: int, avg_cache_tokens: int) -> int:
    if active_batch < 0 or bytes_per_token < 0 or avg_cache_tokens < 0:
        raise ValueError("inputs must be non-negative")
    return active_batch * bytes_per_token * avg_cache_tokens
```

这条式子是理解“为什么 decode 往往 memory-bound”的入口：随着上下文变长，扫描量线性增长；随着 batch 变大，扫描量再乘一个系数。

### 5.2 为什么高吞吐不一定意味着高 Goodput

当 batch 继续增大时：

- `token_throughput` 可能还在上升。
- TTFT 可能因为 prefill 被挤压而恶化。
- TPOT 可能因为 KV 扫描量增大而恶化。
- `goodput()` 反而可能下降。

这就是 serving 里最常见的误区：把“硬件吞吐变高”误读成“用户体验变好”。

## 6. 建议的源码阅读顺序

1. 先读 [../../math_dictionary/serving-metrics.md](../../math_dictionary/serving-metrics.md)，把 TTFT、TPOT、Goodput、Little 定律串起来。
2. 再读 [../../src/simulators/serving_metrics.py](../../src/simulators/serving_metrics.py)，确认每个指标都是逐公式实现。
3. 接着读 [../../src/simulators/scheduler.py](../../src/simulators/scheduler.py)，把 `prefill -> decode -> done` 状态机和 decode 优先调度看懂。
4. 最后把 [capacity-planning.md](capacity-planning.md) 和 [cost-optimization.md](cost-optimization.md) 接上，思考指标如何反过来影响 batch、GPU 数量和成本。

## 这一页记住一句话

> 服务系统里最重要的不是“吞吐大不大”，而是“在 SLO 约束下的有效吞吐大不大”。TTFT 决定首 token 体验，TPOT 决定后续流式体验，Goodput 则把二者和调度策略一起收敛成最终目标。
