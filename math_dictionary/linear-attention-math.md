# 线性注意力（Linear Attention）与高效注意力速查

## 1) 目标
- 将标准注意力对序列长度的 `O(T^2)` 开销，降低为 `O(T)` 或 `O(T * log T)`

## 2) 线性注意力核心形式
- 用特征映射 `phi(.)` 近似 softmax kernel：
  `Attn(Q,K,V) ≈ (phi(Q) (phi(K)^T V)) / (phi(Q) (phi(K)^T 1))`
- 关键点：先聚合 `phi(K)^T V`（形状 `d x d`），再与 `phi(Q)` 结合
- 避免显式构建 `T x T` 注意力矩阵
- 时间复杂度：`O(T * d^2)`（`d` 为特征维度）

## 3) RNN 视角（递推形式）
- 线性注意力可写成递推：
  ```
  S_t = S_{t-1} + phi(k_t) v_t^T    # 状态更新
  o_t = phi(q_t) S_t / (phi(q_t) z_t)  # 输出
  z_t = z_{t-1} + phi(k_t)            # 归一化项
  ```
- 状态 `S_t in R^{d x d}`：固定大小，不随序列增长
- 这解释了为什么线性注意力也被称为"隐式 RNN"

## 4) 常见变体

### Mamba（Structured State Space）
- 选择性状态空间模型（S6）
- 输入依赖的 A, B, C 矩阵（与 token 相关）
- 硬件友好的 scan 操作（并行前缀求和）
- 推理时 O(1) 复杂度 per token（固定状态大小）

### RWKV
- 基于 WKV（Weighted Key-Value）机制
- 时间混合 + 通道混合
- 递推形式推理，训练时可并行

### RetNet（Retentive Network）
- 保留（retention）机制替代注意力
- 三种等价形式：并行（训练）、递推（推理）、分块（混合）
- 状态衰减：`gamma^{m-n}` 引入位置相关的衰减

## 5) 复杂度对比
| 方法 | 训练 | 推理 per token | 状态大小 |
|------|------|---------------|---------|
| Transformer | O(T^2 d) | O(T d)（KV cache） | O(T d) 递增 |
| Linear Attention | O(T d^2) | O(d^2) | O(d^2) 固定 |
| Mamba | O(T d N) | O(d N) | O(d N) 固定 |

（N 为 SSM 状态维度，通常 16-64）

## 6) 与 KV cache 的关系
- 传统 Transformer：KV cache 随序列增长，是显存瓶颈
- 线性注意力/SSM：固定状态大小，无 KV cache 问题
- 但信息容量有限：长序列中可能"遗忘"早期信息
- 混合架构（如 Jamba）：部分层用注意力 + 部分层用 Mamba

## 7) 风险与权衡
- 近似而非严格等价，长程依赖与精度可能受影响。
- 关联检索（associative recall）能力通常弱于标准注意力。
- 需要任务级验证，不能只看理论复杂度。
- 在 needle-in-a-haystack 等测试中表现通常不如 Transformer。

## 面试一句话
- "线性注意力/SSM 用固定状态替代递增的 KV cache，换来 O(1) 推理但牺牲了精确的长程检索能力；混合架构试图兼顾两者。"
