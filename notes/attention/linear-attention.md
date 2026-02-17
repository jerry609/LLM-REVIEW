# 线性注意力与高效注意力

## 标准注意力的问题
- O(T²) 的计算和内存 → 长上下文不可扩展
- 即使有 FlashAttention，FLOPs 仍然是 O(T²)

## 线性注意力核心思想
- 将 softmax(QK^T)V 改写为 φ(Q)(φ(K)^T V)
- 先算 φ(K)^T V → O(d² × T)，再乘 φ(Q) → O(d² × T)
- 总复杂度 O(T × d²)，当 T >> d 时远优于 O(T² × d)

## 主要方法

### Mamba (S6)
- 基于状态空间模型（SSM），不是注意力
- 隐状态递推：h_t = A·h_{t-1} + B·x_t，y_t = C·h_t
- 选择性机制：A/B/C 依赖于输入（input-dependent）
- 推理：O(1) 每步（只维护固定大小隐状态）
- 缺点：长距离精确检索能力弱于 Transformer

### RWKV
- 类 RNN 结构，支持并行训练 + 线性推理
- WKV 机制：加权键值聚合，带指数衰减

### RetNet
- 多尺度指数衰减注意力
- 支持三种计算模式：并行（训练）、递推（推理）、分块（混合）

## Hybrid 架构趋势
- Jamba (AI21): Mamba + Transformer 混合层
- 浅层用 Mamba（处理局部依赖），深层用 Transformer（处理全局依赖）

## 面试一句话
- "线性注意力把 O(T²) 降到 O(T)，但精确长距离检索能力不如标准注意力。当前趋势是 hybrid：大部分层用线性注意力/SSM，少数层用标准注意力。"
