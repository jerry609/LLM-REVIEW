# KV 压缩与量化数学速查

## 1) 线性量化（Uniform Quantization）
- 对张量 `x`：
  `x_hat = scale * (q - zp)`
- 其中 `q` 是离散整数（INT8/INT4），`scale` 与 `zp` 为量化参数。
- 对称量化（`zp=0`）：`scale = max(|x|) / (2^{b-1} - 1)`
- 非对称量化：`scale = (max(x) - min(x)) / (2^b - 1)`，`zp = round(-min(x)/scale)`

## 2) 分组量化（Group Quantization）
- 将通道分为组（group_size 常为 64 或 128）
- 每组有独立的 `scale` 和 `zp`
- 精度更高：减少组内离群值对整体的影响
- 开销：每组多存 `scale`（和可选 `zp`），约增加 `2/group_size` 比例的额外存储

## 3) 误差度量
- `MSE = E[(x - x_hat)^2]`
- `RelativeError = ||x-x_hat|| / ||x||`
- `SNR = 10 * log10(||x||^2 / ||x-x_hat||^2)` dB
- 实战中看任务指标变化（准确率/困惑度）而非只看 MSE。

## 4) 压缩收益
- `saving = 1 - s_new / s_old`
- BF16→FP8/INT8：约 50% 节省
- BF16→INT4：约 75% 节省
- 考虑分组量化额外参数后的实际压缩比：
  `effective_ratio = (n_elements * s_new + n_groups * s_param) / (n_elements * s_old)`

## 5) KV-specific 量化方法

### KV cache 量化（Per-token / Per-channel）
- Per-token：每个 token 的 KV 向量独立量化
- Per-channel：跨 token 按 channel 维度量化
- Per-token 更常见：token 间值域差异大

### KIVI（2-bit KV cache）
- Key cache 按 channel 量化，Value cache 按 token 量化
- 配合残差（residual）保留部分高精度信息
- 可做到 2-bit 量化，KV 显存降至 1/8

## 6) 权重量化方法（面试参考）

### GPTQ
- 基于 OBS（Optimal Brain Surgeon）的逐层量化
- 最小化量化误差：`min ||Wx - W_hat x||^2`
- 利用 Hessian 矩阵逆校准：`delta_W = -w_q_err * H^{-1}[:,col] / H[col,col]`
- 不需要训练，纯校准

### AWQ（Activation-Aware Weight Quantization）
- 核心思想：对激活值大的通道的权重保护（乘以 scale 后再量化）
- `s_i = (max(|X_i|))^alpha`，`alpha` 在 [0,1] 间搜索
- 等价于减少重要通道的量化误差

### SmoothQuant
- 将激活的离群值"平滑"转移到权重中：
  `Y = (X diag(s)^{-1}) (diag(s) W) = X_smooth W_smooth`
- 使 `X_smooth` 和 `W_smooth` 都更易量化

## 7) 分层策略（温度分级）
- 热块：BF16/FP16（不压或轻压）
- 温块：FP8/INT8
- 冷块：INT4 或 CPU offload
- 分级判断信号：最近访问时间、注意力权重、请求活跃度

## 8) 回迁抖动控制
- 每 step 限制反量化预算：`budget_tokens_per_step`
- 异步预取下一批高概率命中块。
- 回迁延迟：`T_dequant ≈ tokens * d * s_compressed / dequant_throughput`
- 预算超额时排队等待下一 step，避免 decode 延迟突增。

## 9) 稀疏化压缩
- 注意力稀疏：许多 attention weight 接近 0，可跳过
  - Top-k 稀疏：只保留 top-k 个注意力权重
  - 节省计算但可能影响长距离依赖
- KV 稀疏：只保留"重要" token 的 KV
  - 与驱逐策略有重叠，区别是稀疏化在计算层面，驱逐在存储层面
- 结构化稀疏 vs 非结构化稀疏：
  - 结构化（如整行/整列置零）→ 硬件友好
  - 非结构化（任意位置置零）→ 精度好但加速难

## 口述模板
- "先给压缩比，再给质量门槛，再给延迟抖动兜底（预算+回滚）。"
- "量化不是免费午餐：离群值处理是关键，分组量化和 scale 搜索是工程核心。"
