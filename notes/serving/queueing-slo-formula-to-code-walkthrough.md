# 从公式到源码：Queueing / SLO / Erlang C 对照手册

> 这一页把服务系统里最容易“只会背结论”的公式直接落到可运行代码：Little 定律负责做一阶容量估算，M/M/1 负责解释单机过载，Erlang C 负责解释多实例排队，M/G/1 负责解释为什么长尾输出会打爆 P99。

## 这页覆盖哪些源码

- [../../src/simulators/queueing_slo.py](../../src/simulators/queueing_slo.py)：Little 定律、M/M/1、Erlang C、M/G/1、SLO 反推。
- [../../src/simulators/serving_metrics.py](../../src/simulators/serving_metrics.py)：Goodput、batch utilization 等服务指标。
- [../../src/simulators/scheduler.py](../../src/simulators/scheduler.py)：请求状态机和 decode 优先调度。

## 1. Little 定律：为什么并发数等于到达率乘时延

### 1.1 公式

Little 定律对稳态系统成立：

$$
L = \lambda W
$$

映射到推理服务里，可以写成

$$
\bar{B}_{\text{active}} = \lambda_{\text{req}} \times \bar{T}_{\text{E2E}}
$$

这条式子的工程含义非常直接：如果到达率不变，而 E2E 变长，那么系统里同时活着的请求数必然上升。

### 1.2 对应源码

[../../src/simulators/queueing_slo.py](../../src/simulators/queueing_slo.py) 中的 `little_law_concurrency()`：

```python
def little_law_concurrency(arrival_rate: float, avg_latency: float) -> float:
    if arrival_rate < 0 or avg_latency < 0:
        raise ValueError("arrival_rate and avg_latency must be non-negative")
    return arrival_rate * avg_latency
```

所以 Little 定律不是一个“只在论文里出现的结论”，而是最朴素的一阶容量乘法。

## 2. M/M/1：为什么利用率接近 1 时延会爆炸

### 2.1 利用率和平均时延

对单服务台模型，利用率定义为

$$
\rho = \frac{\lambda}{\mu}, \qquad \rho < 1
$$

平均系统时延和平均排队时延分别为

$$
\bar{W} = \frac{1}{\mu - \lambda}
$$

$$
\bar{W}_q = \frac{\rho}{\mu - \lambda}
$$

平均系统内请求数和排队长度则由 Little 定律直接给出

$$
\bar{L} = \lambda \bar{W}, \qquad \bar{L}_q = \lambda \bar{W}_q
$$

### 2.2 对应源码

`mm1_stats()` 把这些量一次性算出来：

```python
def mm1_stats(arrival_rate: float, service_rate: float) -> MM1Stats:
    rho = utilization(arrival_rate, service_rate, servers=1)
    if rho >= 1.0:
        raise ValueError("MM1 requires arrival_rate < service_rate")

    avg_response_time = 1.0 / (service_rate - arrival_rate)
    avg_queue_wait = rho / (service_rate - arrival_rate)
    avg_in_system = arrival_rate * avg_response_time
    avg_queue_length = arrival_rate * avg_queue_wait
```

最重要的面试点在这里：不是背出公式本身，而是能说出“`mu - lambda` 越小，时延就会以非线性方式发散”。

## 3. Erlang C：多实例为什么也会排队

### 3.1 多服务台利用率

当系统有 `c` 个并行服务台时，利用率变成

$$
\rho = \frac{\lambda}{c\mu}
$$

但即便 `rho < 1`，请求仍然可能需要等待。Erlang C 给出等待概率：

$$
P_{\text{wait}} = \frac{\frac{(\lambda / \mu)^c}{c!(1-\rho)}}{\sum_{k=0}^{c-1} \frac{(\lambda / \mu)^k}{k!} + \frac{(\lambda / \mu)^c}{c!(1-\rho)}}
$$

平均排队等待时间为

$$
\bar{W}_q = \frac{P_{\text{wait}}}{c\mu - \lambda}
$$

### 3.2 对应源码

源码分成两步实现：

```python
def erlang_c_wait_probability(arrival_rate: float, service_rate: float, servers: int) -> float:
    rho = utilization(arrival_rate, service_rate, servers=servers)
    if rho >= 1.0:
        raise ValueError("MMC requires utilization < 1")

    traffic = arrival_rate / service_rate
    numerator = (traffic ** servers / math.factorial(servers)) * (1.0 / (1.0 - rho))
    denominator = sum((traffic ** k) / math.factorial(k) for k in range(servers)) + numerator
    return numerator / denominator


def mmc_avg_queue_wait(arrival_rate: float, service_rate: float, servers: int) -> float:
    wait_prob = erlang_c_wait_probability(arrival_rate, service_rate, servers)
    return wait_prob / (servers * service_rate - arrival_rate)
```

Erlang C 的关键工程含义是：加副本会缓解排队，但不会让排队概率凭空消失，尤其在高利用率下仍然会显著等待。

## 4. M/G/1：为什么服务时间方差会拖垮尾延迟

### 4.1 Pollaczek-Khinchine 公式

若服务时间不再是指数分布，而是一般分布，平均服务时间记为 `E[S]`，二阶矩记为 `E[S^2]`，则

$$
\bar{W}_q = \frac{\lambda \mathbb{E}[S^2]}{2(1-\rho)}, \qquad \rho = \lambda \mathbb{E}[S]
$$

把变异系数 `C_s` 写进去，可得

$$
\mathbb{E}[S^2] = (1 + C_s^2) \mathbb{E}[S]^2
$$

于是当输出长度波动变大、`C_s` 上升时，排队等待也会显著恶化。

### 4.2 对应源码

```python
def mg1_queue_wait(arrival_rate: float, mean_service_time: float, service_time_cv: float = 1.0) -> float:
    rho = arrival_rate * mean_service_time
    if rho >= 1.0:
        raise ValueError("MG1 requires utilization < 1")

    second_moment = (1.0 + service_time_cv ** 2) * (mean_service_time ** 2)
    return arrival_rate * second_moment / (2.0 * (1.0 - rho))
```

而 `mg1_response_time()` 只是把服务时间再加回去：

```python
def mg1_response_time(arrival_rate: float, mean_service_time: float, service_time_cv: float = 1.0) -> float:
    return mean_service_time + mg1_queue_wait(arrival_rate, mean_service_time, service_time_cv)
```

## 5. SLO 反推：已知目标时延，最少需要多大服务能力

若在 M/M/1 近似下，你希望平均响应时间满足

$$
\bar{W} \le W_{\text{target}}
$$

又因为

$$
\bar{W} = \frac{1}{\mu - \lambda}
$$

所以只要反解即可得到

$$
\mu_{\text{required}} \ge \lambda + \frac{1}{W_{\text{target}}}
$$

对应源码 `required_mm1_service_rate()`：

```python
def required_mm1_service_rate(arrival_rate: float, target_response_time: float) -> float:
    if arrival_rate < 0 or target_response_time <= 0:
        raise ValueError("arrival_rate must be non-negative and target_response_time positive")
    return arrival_rate + 1.0 / target_response_time
```

这条反推特别适合面试时回答“给定目标 QPS 和目标平均时延，需要多大吞吐能力”。

## 6. 队列公式怎样和服务指标、调度器连起来

队列模型给出的是“系统为什么会堵”，而 [../../src/simulators/serving_metrics.py](../../src/simulators/serving_metrics.py) 给出的是“堵了以后用户会看到什么”：

- TTFT 变差：prefill 等待增加。
- TPOT 变差：decode 被 KV 带宽和调度抖动拖慢。
- Goodput 下降：满足 SLO 的请求比例下滑。

[../../src/simulators/scheduler.py](../../src/simulators/scheduler.py) 的 decode 优先策略，则解释了为什么 TTFT 和 TPOT 往往会此消彼长：

$$
\text{decode first} \Rightarrow \text{TPOT 更稳，但 TTFT 更容易受压}
$$

## 7. 建议的源码阅读顺序

1. 先读 [../../math_dictionary/queueing-and-slo.md](../../math_dictionary/queueing-and-slo.md)，把 Little 定律、M/M/1、M/G/1、Erlang C 串起来。
2. 再读 [../../src/simulators/queueing_slo.py](../../src/simulators/queueing_slo.py)，确认这些公式如何落成最小函数。
3. 接着读 [../../src/simulators/serving_metrics.py](../../src/simulators/serving_metrics.py) 和 [../../src/simulators/scheduler.py](../../src/simulators/scheduler.py)，把队列结论接到 TTFT / TPOT / Goodput 和调度器行为。
4. 最后跑 [../../tests/test_queueing_slo.py](../../tests/test_queueing_slo.py)，确认几个经典闭式公式在代码里是自洽的。

## 这一页记住一句话

> 服务系统的排队问题，本质上是在算“流入速度”和“处理速度”之间的余量。Little 定律告诉你并发会怎么涨，M/M/1 告诉你过载会怎么炸，M/G/1 告诉你长尾为什么可怕，Erlang C 则告诉你多副本也不是万能药。
