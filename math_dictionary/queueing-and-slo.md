# 排队与 SLO：从系统直觉到可执行预算

> 这页的目标不是把排队论写成一本课本，而是把 serving 场景里真正有用的几条公式整理成一条主线：先看 Little 定律，再看 M/M/1 和 M/M/c，接着看 M/G/1 的长尾放大，最后把结论落到 SLO 预算、限流和 admission 上。

## 1. 先统一最小符号集

| 记号 | 含义 |
|------|------|
| `lambda` | 到达率，单位通常是 req/s |
| `mu` | 单服务台服务率，单位通常是 req/s |
| `c` | 并行服务台数量，也可以理解成副本数或实例数 |
| `rho` | 利用率 |
| `W` | 平均逗留时间，包含排队与服务 |
| `W_q` | 平均排队等待时间 |
| `L` | 系统中的平均请求数 |
| `L_q` | 队列中的平均请求数 |
| `C_s` | 服务时间变异系数 |
| `P99` | 99 分位尾延迟 |

## 2. Little 定律先给出第一层直觉

Little 定律写成最熟悉的形式就是：

$$
L = \lambda W
$$

在 LLM serving 语境里，最常用的改写是：

$$
\bar{B}_{\mathrm{active}} \approx \lambda_{\mathrm{req}} \times \bar{T}_{\mathrm{E2E}}
$$

这条式子的价值不是“考试会不会考”，而是它能直接解释下面这些现象：

- 平均到达率不变，但 E2E 变长，系统里的活跃请求数就会涨。
- batch utilization 变高，不一定是纯好事，也可能只是请求在系统里待得更久。
- 只看 GPU 利用率，不看队列和 E2E，很容易误判系统还在健康区间。

## 3. M/M/1：单服务台下最常用的闭式公式

### 3.1 基本假设

- 到达过程近似泊松。
- 服务时间近似指数分布。
- 单服务台，也就是单条服务通道的抽象。

### 3.2 核心公式

利用率定义为：

$$
\rho = \frac{\lambda}{\mu}
$$

平均逗留时间：

$$
W = \frac{1}{\mu - \lambda}
$$

平均排队等待时间：

$$
W_q = \frac{\rho}{\mu - \lambda} = \frac{\lambda}{\mu(\mu - \lambda)}
$$

系统中的平均请求数：

$$
L = \lambda W = \frac{\rho}{1 - \rho}
$$

队列中的平均请求数：

$$
L_q = \lambda W_q = \frac{\rho^2}{1 - \rho}
$$

### 3.3 工程解读

这组公式共同说明一件事：只要 `rho` 接近 1，等待时间和队列长度都会非线性爆炸。也就是说，系统最怕的不是“慢一点”，而是“接近打满之后开始没有余量”。

| `rho` | 平均时延相对单次服务时间的放大量 | 工程含义 |
|--------|-----------------------------------|----------|
| 0.5 | 约 2 倍 | 还有明显余量 |
| 0.7 | 约 3.3 倍 | 常被视为比较稳的上限 |
| 0.8 | 约 5 倍 | 抖动开始明显放大 |
| 0.9 | 约 10 倍 | 已进入危险区 |
| 0.95 | 约 20 倍 | 很容易出现尾延迟雪崩 |

## 4. M/M/c 和 Erlang C：多副本不是万能药

当系统有 `c` 个并行服务台时，利用率改写为：

$$
\rho = \frac{\lambda}{c\mu}
$$

此时即便 `rho` 小于 1，仍然可能发生排队。等待概率由 Erlang C 给出：

$$
P_{\mathrm{wait}} = \frac{\frac{(c\rho)^c}{c!(1-\rho)}}{\sum_{n=0}^{c-1} \frac{(c\rho)^n}{n!} + \frac{(c\rho)^c}{c!(1-\rho)}}
$$

平均排队等待时间可写成：

$$
W_q = \frac{P_{\mathrm{wait}}}{c\mu - \lambda}
$$

平均响应时间则是：

$$
W = W_q + \frac{1}{\mu}
$$

### 4.1 工程解读

- 扩副本当然能降等待，但收益会递减。
- 如果负载分配不均，系统的有效 `c` 会比你部署的副本数更小。
- 如果单请求服务时间方差很大，多副本也只能缓解，不能根治长尾。

## 5. M/G/1：为什么服务时间方差会把队列放大

真实的 LLM 请求很少满足指数分布，因为 prompt 长度和输出长度差异都很大。此时更有用的公式是 Pollaczek-Khinchine：

$$
W_q = \frac{\lambda \mathbb{E}[S^2]}{2(1-\rho)}
$$

其中：

$$
\rho = \lambda \mathbb{E}[S]
$$

如果把服务时间变异系数写成 `C_s`，则二阶矩可以改写为：

$$
\mathbb{E}[S^2] = (1 + C_s^2) \mathbb{E}[S]^2
$$

代回去之后，平均排队等待时间变成：

$$
W_q = \frac{\lambda (1 + C_s^2) \mathbb{E}[S]^2}{2(1-\rho)}
$$

### 5.1 工程解读

- 同样的平均服务时间下，`C_s` 越大，排队越糟。
- 输出长度长尾、prefill 大小差异大、调度抖动大，都会抬高 `C_s`。
- continuous batching、长度分桶、SJF 风格调度，本质上都在尝试降低服务时间方差。

## 6. P99 为什么比平均时延更早暴露问题

在 M/M/1 里，等待时间的尾部分布可以写成：

$$
\Pr[W > t] = \rho e^{-(\mu - \lambda)t}
$$

把尾概率设成 0.01，就得到一个常用的 P99 近似：

$$
t_{99} = \frac{-\ln(0.01 / \rho)}{\mu - \lambda}
$$

也可以写成：

$$
t_{99} = \frac{\ln(100\rho)}{\mu(1-\rho)}
$$

| `rho` | P99 相对单次服务时间的量级 | 直觉解释 |
|--------|----------------------------|----------|
| 0.5 | 约 7.8 倍 | 尾延迟已经显著大于平均值 |
| 0.7 | 约 13.4 倍 | 队列波动会明显传到用户侧 |
| 0.9 | 约 46.1 倍 | 轻微抖动也会被放大 |

所以线上排障时，P99 经常会先坏，平均值还没坏到离谱。

## 7. 把端到端 SLO 拆成预算，而不是只盯一个总数

如果目标是端到端 P99，一个很实用的预算写法是：

$$
T_{\mathrm{E2E}}^{99} = T_{\mathrm{queue}}^{99} + T_{\mathrm{prefill}}^{99} + T_{\mathrm{decode}}^{99} + T_{\mathrm{network}}^{99}
$$

这张预算表最好写成非公式表，而不是把公式塞进表格：

| 子项 | 常见含义 |
|------|----------|
| queue | admission、排队、batch 等待 |
| prefill | prompt 编码、首轮 KV 构建 |
| decode | 后续 token 生成、KV 扫描 |
| network | RPC、网关、流式返回 |

### 7.1 超预算时优先动什么

- queue 先爆：先看扩容、限流、admission、调度。
- prefill 先爆：先看 prompt 长度、chunked prefill、prefill 算子效率。
- decode 先爆：先看 KV 带宽、输出长度、batch 策略、压缩与驱逐。
- network 先爆：再回到网关、代理层和跨机通信。

## 8. 限流和 admission 是 SLO 的执行层

一个常见的 admission 规则可以写成：

$$
\operatorname{Admit} = \mathbb{1}\left\{Q_{\mathrm{depth}} < \theta_Q,\ \rho_{\mathrm{KV}} < \theta_{\mathrm{KV}},\ P99 < 0.8 \operatorname{SLO}\right\}
$$

这个规则背后的意思是：

- `Q_depth` 太深，说明队列已经开始堆积。
- `rho_KV` 太高，说明显存快没有余量了。
- `P99` 接近红线，说明系统已经不能只看平均值。

Token Bucket 也是同类执行层工具：

$$
\operatorname{tokens}(t) = \min\left(\operatorname{tokens}(t-\Delta t) + r\Delta t, B_{\mathrm{burst}}\right)
$$

它的作用不是让系统变快，而是阻止突发流量把系统瞬间压垮。

## 9. 和源码怎么对起来

- 队列公式和闭式解： [../src/simulators/queueing_slo.py](../src/simulators/queueing_slo.py)
- 服务指标与预算： [../src/simulators/serving_metrics.py](../src/simulators/serving_metrics.py)
- 调度器最小实现： [../src/simulators/scheduler.py](../src/simulators/scheduler.py)
- 对照手册： [../notes/serving/queueing-slo-formula-to-code-walkthrough.md](../notes/serving/queueing-slo-formula-to-code-walkthrough.md)
- 服务主线： [../notes/serving/formula-to-code-walkthrough.md](../notes/serving/formula-to-code-walkthrough.md)

## 10. 最推荐的阅读顺序

1. 先读 Little 定律和 M/M/1，建立“系统为什么会炸”的第一直觉。
2. 再读 M/G/1，理解为什么长尾请求会把队列放大。
3. 接着看 P99 和 SLO 预算，把排队论接到真实服务指标。
4. 最后再看 admission、限流和源码，把理论落到执行层。

## 这一页记住一句话

> 排队论在 LLM serving 里最重要的不是背定义，而是知道该先看 `rho`、再看服务时间方差、最后看 SLO 预算。只要这三件事能接起来，你就不会再把所有延迟问题都粗暴归因成“模型太慢”。
