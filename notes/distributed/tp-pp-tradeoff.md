# Tensor Parallel vs Pipeline Parallel

## Tensor Parallel (TP)
- 将每层的权重矩阵沿列/行切分到多个 GPU
- 每个 GPU 算部分结果 → AllReduce 汇总
- 通信：每层 2 次 AllReduce（一次 attention，一次 FFN）

### TP 切分方式
```
Attention: Q/K/V 按 head 维度切分，每个 GPU 处理部分 head
FFN: W_gate/W_up 按列切分，W_down 按行切分
AllReduce 后得到完整输出
```

### TP 通信量
- 每层通信量 = 2 × batch × seq_len × d_model × bytes
- TP=8 时每层 AllReduce 量级 ~MB → 需要 NVLink 带宽

## Pipeline Parallel (PP)
- 将模型不同层分到不同 GPU
- GPU 0: layer 0-9, GPU 1: layer 10-19, ...
- 前向传播像流水线一样逐级传递

### PP 通信量
- 每个 micro-batch 只传一次中间激活 = batch × seq_len × d_model × bytes
- 通信量远小于 TP，但有 pipeline bubble

### Pipeline Bubble
- 效率 = 1 - (p-1)/m，其中 p=PP 度，m=micro-batch 数
- 需要足够多 micro-batch 才能填满流水线

## 选择策略
| 场景 | 推荐 | 原因 |
|------|------|------|
| 单机多卡 (NVLink) | TP | NVLink 带宽高，AllReduce 快 |
| 多机 | TP(机内) + PP(跨机) | 跨机带宽有限用 PP 减少通信 |
| 延迟敏感 | TP | TP 每层并行，延迟低 |
| 吞吐优先 | PP + 大 batch | micro-batch 多则 bubble 小 |

## TP + PP 组合
- 典型：8 GPU/node → TP=8 (机内), PP=2+ (跨机)
- vLLM 默认：`--tensor-parallel-size=8 --pipeline-parallel-size=2`

## 面试一句话
- "TP 切权重、每层通信，适合 NVLink 互联；PP 切层数、逐级传递，适合跨机。实践中机内 TP + 跨机 PP 混合使用。"
