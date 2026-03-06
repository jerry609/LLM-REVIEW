# 服务与成本

> 这一组笔记关注推理服务的真实运行指标：请求如何排队、TTFT / TPOT / Goodput 如何互相推导、continuous batching 为什么会在高吞吐和低延迟之间拉扯，以及这些指标如何反过来约束 SLO 和扩容。

## 最推荐的阅读顺序

1. [formula-to-code-walkthrough.md](formula-to-code-walkthrough.md)：先把 TTFT / TPOT / Goodput、服务预算、KV 带宽和调度代码对上。
2. [queueing-slo-formula-to-code-walkthrough.md](queueing-slo-formula-to-code-walkthrough.md)：再把 Little 定律、M/M/1、M/G/1、Erlang C 和源码对上。
3. [../../math_dictionary/serving-metrics.md](../../math_dictionary/serving-metrics.md)：回到完整指标体系和诊断逻辑。
4. [../../math_dictionary/queueing-and-slo.md](../../math_dictionary/queueing-and-slo.md)：从排队模型理解 P99、限流和容量余量。
5. [capacity-planning.md](capacity-planning.md)：把指标落到 GPU 容量规划和 batch 选择。
6. [cost-optimization.md](cost-optimization.md)：最后看成本、缓存命中率和投机解码等优化手段。

## 这一组专题覆盖什么

- 服务预算：TTFT、TPOT、E2E，以及 queue / prefill / decode 的拆分账本。
- 吞吐与 Goodput：裸吞吐不等于有效吞吐，SLO 约束才是服务视角的核心。
- 队列与 SLO：Little 定律、M/M/1、M/M/c、M/G/1、Erlang C。
- 调度与规划：continuous batching、decode 优先、chunked prefill、扩容与限流。
- 带宽与显存：KV 步扫描量、memory-bound decode、长上下文成本。

## 对应源码

- [../../src/simulators/scheduler.py](../../src/simulators/scheduler.py)：continuous batching、decode 优先、prefill chunking。
- [../../src/simulators/serving_metrics.py](../../src/simulators/serving_metrics.py)：TTFT、TPOT、E2E、Goodput、服务需求、batch utilization、KV 步带宽下界。
- [../../src/simulators/queueing_slo.py](../../src/simulators/queueing_slo.py)：Little 定律、M/M/1、Erlang C、M/G/1、SLO 反推。

## 如果你只剩 20 分钟

- 先读 [formula-to-code-walkthrough.md](formula-to-code-walkthrough.md)
- 再读 [queueing-slo-formula-to-code-walkthrough.md](queueing-slo-formula-to-code-walkthrough.md)
- 最后对照 [../../src/simulators/serving_metrics.py](../../src/simulators/serving_metrics.py)、[../../src/simulators/queueing_slo.py](../../src/simulators/queueing_slo.py) 和 [../../src/simulators/scheduler.py](../../src/simulators/scheduler.py)
