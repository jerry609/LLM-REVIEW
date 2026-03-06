# 服务与成本

> 这一组笔记关注推理服务的真实运行指标：请求是怎么排队的，TTFT / TPOT / Goodput 是怎么互相推出来的，continuous batching 为什么会让系统在高吞吐和低延迟之间拉扯。

## 最推荐的阅读顺序

1. [formula-to-code-walkthrough.md](formula-to-code-walkthrough.md)：先把 TTFT / TPOT / E2E / Goodput 和调度代码对上。
2. [../../math_dictionary/serving-metrics.md](../../math_dictionary/serving-metrics.md)：再回到完整指标体系和推导关系。
3. [capacity-planning.md](capacity-planning.md)：从指标落到 GPU 容量规划和 batch 选择。
4. [cost-optimization.md](cost-optimization.md)：再看成本、缓存命中率和投机解码等优化手段。
5. [../../math_dictionary/queueing-and-slo.md](../../math_dictionary/queueing-and-slo.md)：最后把排队论和 SLO 结合起来看。

## 这一组专题覆盖什么

- 延迟分解：TTFT、TPOT、E2E 以及它们之间的线性关系。
- 吞吐与 Goodput：裸吞吐不等于有效吞吐，SLO 约束才是服务视角的核心。
- 调度：continuous batching、decode 优先、chunked prefill。
- 容量与成本：batch 利用率、KV 带宽压力、扩容与限流。

## 对应源码

- [../../src/simulators/scheduler.py](../../src/simulators/scheduler.py)：请求状态机、decode 优先、prefill chunking。
- [../../src/simulators/serving_metrics.py](../../src/simulators/serving_metrics.py)：TTFT、TPOT、E2E、Goodput、batch utilization、KV 步带宽。

## 如果你只剩 20 分钟

- 先读 [formula-to-code-walkthrough.md](formula-to-code-walkthrough.md)
- 再看 [../../math_dictionary/serving-metrics.md](../../math_dictionary/serving-metrics.md) 里的 Goodput 和 Little 定律
- 最后对照 [../../src/simulators/scheduler.py](../../src/simulators/scheduler.py) 的 `step()` 和 [../../src/simulators/serving_metrics.py](../../src/simulators/serving_metrics.py) 的 `goodput()`
