# FlashAttention IO 优化速查

## 1) 核心目标
- 减少注意力计算中的 HBM 读写，做 IO-aware 优化。
- 通过 tile/block 化，把更多中间结果留在片上 SRAM/shared memory。

## 2) 数学不变
- 仍计算 `softmax(QK^T / sqrt(d_head))V`，结果等价于标准注意力。
- 改变的是计算顺序与数值稳定实现，不是目标函数。

## 3) 标准注意力的 IO 问题
- 标准流程：
  1. 计算 `S = QK^T` → 写入 HBM（`O(T^2)` 元素）
  2. 计算 `P = softmax(S)` → 读/写 HBM（`O(T^2)` 元素）
  3. 计算 `O = PV` → 读/写 HBM
- 总 HBM IO：`O(T^2 * d_head + T^2)` → 对长序列 IO 成为瓶颈

## 4) FlashAttention 的分块策略
- 将 Q, K, V 分成 `T_block` 大小的块
- 外循环遍历 K/V 块，内循环遍历 Q 块
- 每块在 SRAM 内完成：`S_block = Q_block K_block^T` → `P_block = softmax(S_block)` → 累加 `O += P_block V_block`
- 不需要将完整 `T x T` 的 S 矩阵写入 HBM

## 5) Online Softmax（关键技巧）
- 普通 softmax 需要全量 S 才能算 max 和 sum
- Online softmax 流式更新：
  ```
  处理新块后：
  m_new = max(m_old, m_block)          // 更新全局 max
  l_new = l_old * exp(m_old - m_new) + l_block * exp(m_block - m_new)  // 更新全局 sum
  O_new = O_old * (l_old/l_new) * exp(m_old - m_new) + P_block V_block * exp(m_block - m_new) / l_new
  ```
- 数学上严格等价于全量 softmax（浮点误差范围内）

## 6) IO 复杂度分析
- 标准注意力 HBM 访问：`O(T^2 * d_head)` + 中间矩阵 `O(T^2)`
- FlashAttention HBM 访问：`O(T^2 * d_head^2 / M_sram)`
  - `M_sram`：SRAM 大小（A100 约 192 KB per SM，总 ~20 MB）
  - 当 `d_head << M_sram` 时，IO 降低显著
- FLOPs 不变：仍为 `O(T^2 * d_head)`

## 7) FlashAttention-2 改进
- 更好的 work partitioning：在 warp 间更均匀分配工作
- 减少非矩阵乘法运算（softmax 等）的同步开销
- 达到理论峰值 FLOPS 的 50-73%（对比 FA-1 的 25-40%）

## 8) FlashAttention-3（H100 优化）
- 利用 H100 的 Tensor Memory Accelerator (TMA) 异步加载
- 低精度：支持 FP8 注意力计算
- Warp specialization：producer-consumer 模式流水线化

## 9) 与 KV cache 的关系
- Decode 阶段：Q 只有 1 个 token，K/V 来自 cache
  - FlashAttention 对 decode 的加速相对有限（本身 IO 就少）
  - 更大收益在 prefill 阶段（长序列）
- FlashDecoding：针对 decode 优化，在序列维度并行

## 10) 实际加速数据（参考）
- 长序列（>2K token）：2-4× 加速
- 显存节省：从 `O(T^2)` → `O(T)`（不存中间注意力矩阵）
- 使得在不降低精度的情况下支持更长上下文

## 面试一句话
- "FlashAttention 不是换模型，而是换内核执行路径：分块 + online softmax 避免了 O(T^2) 中间矩阵的 HBM 读写，FLOPs 不变但 IO 大幅降低。"
