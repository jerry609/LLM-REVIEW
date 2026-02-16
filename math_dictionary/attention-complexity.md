# 复杂度、FLOPs 与算力瓶颈分析

## 1) 不使用 KV cache 的自回归生成
- 第 `t` 步需对长度为 `t` 的序列做完整注意力
- 生成 `T` 个 token 的累积注意力 FLOPs ∝ `sum_{t=1}^{T} t = O(T^2)`
- 加上 FFN 等线性部分，总开销极大

## 2) 使用 KV cache 的自回归生成
- 历史 `K,V` 复用，新步只计算 `q_t` 与历史 `K` 的点积
- 单步注意力 FLOPs ∝ `O(T_cache)` — 对上下文长度线性
- 单步 FFN FLOPs 固定（`O(d_model * d_ff)`）
- 瓶颈从 compute 转向 memory bandwidth

## 3) Prefill vs Decode 瓶颈
- **Prefill**：处理 `T_input` 个 token，注意力 FLOPs ∝ `O(T_input^2 * d_head)`
  - 通常 compute-bound（算力受限）
  - 算术强度高，能充分利用 GPU 计算单元
- **Decode**：每步 1 token，但要读取全部 KV cache
  - 通常 memory-bound（带宽受限）
  - 算术强度低：大量读取、少量计算

## 4) FLOPs 精确计算

### 单层注意力（Prefill，序列长度 T）
- QKV 投影：`6 * B * T * d_model * d_head * H`（三次矩阵乘）
  - GQA 下 KV 投影更小：`2 * B * T * d_model * d_head * H + 4 * B * T * d_model * d_head * H_kv`
- 注意力分数：`2 * B * H * T * T * d_head`
- 注意力加权：`2 * B * H * T * T * d_head`
- 输出投影：`2 * B * T * H * d_head * d_model`

### 单层 FFN（SwiGLU）
- gate + up 投影：`2 * 2 * B * T * d_model * d_ff`（两个上投影矩阵）
- down 投影：`2 * B * T * d_ff * d_model`
- 总计：`6 * B * T * d_model * d_ff`

### 全模型 Prefill 粗估
- `FLOPs ≈ L * (12 * B * T * d_model * d_head * H + 4 * B * H * T^2 * d_head + 6 * B * T * d_model * d_ff)`
- 简化口径：`≈ 2 * N * B * T`（`N` 为参数量，乘 2 因为乘加各算 1 次）

### 单步 Decode
- 注意力相关 FLOPs 与 `T_cache` 成正比
- FFN FLOPs 固定

## 5) 算术强度（Arithmetic Intensity）
- 定义：`AI = FLOPs / Bytes_moved`
- 单位：FLOPs/Byte
- Prefill（大矩阵乘）：AI 高 → compute-bound
- Decode（读 KV cache，算少量点积）：AI 低 → memory-bound

## 6) Roofline 模型
```
实际性能 = min(峰值算力, 峰值带宽 * AI)
```
- A100 峰值：~312 TFLOPS（BF16 Tensor Core），~2 TB/s HBM 带宽
- 拐点 AI：`312e12 / 2e12 ≈ 156 FLOPs/Byte`
- Prefill 通常在拐点右侧（compute-bound）
- Decode 通常在拐点左侧（memory-bound）

### H100 参考
- 峰值：~990 TFLOPS（BF16），~3.35 TB/s HBM 带宽
- 拐点 AI：`990e12 / 3.35e12 ≈ 295 FLOPs/Byte`

## 7) FlashAttention 的 IO 复杂度
- 标准注意力 IO：`O(T^2)` — 需要将完整注意力矩阵写入/读出 HBM
- FlashAttention IO：`O(T^2 * d_head / M_sram)`
  - 其中 `M_sram` 为 SRAM（shared memory）大小
  - 分块（tiling）计算，每块在 SRAM 内完成 softmax（online softmax 算法）
  - 不需要存储完整的 `T x T` 注意力矩阵到 HBM
- FLOPs 不变（仍是 `O(T^2 * d_head)`），但减少了 HBM 读写次数
- 实际加速来源：减少了 memory-bound 操作

## 工程结论
- Prefill 优化方向：kernel 效率、张量并行、FlashAttention
- Decode 优化方向：KV 布局（连续内存）、访存优化、调度、驱逐/回迁抖动控制
- 增大 batch → Decode 的 AI 提升 → 更接近 compute-bound → 吞吐提升
- 但增大 batch → KV 占用增加 → 可能触发驱逐/OOM
