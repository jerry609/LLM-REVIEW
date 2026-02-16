# 多机多卡推理的数学模型

## 1) 并行策略总览
| 策略 | 切分维度 | 通信模式 | 适用场景 |
|------|---------|---------|---------|
| 张量并行 TP | 层内 weight 切分 | All-Reduce（每层 2 次） | 单机多卡 |
| 流水线并行 PP | 层间切分 | 点对点（前向/反向） | 多机 |
| 数据并行 DP | batch 切分 | 无需通信（推理时） | 多副本 |
| 专家并行 EP | MoE expert 切分 | All-to-All | MoE 模型 |
| 序列并行 SP | 序列维度切分 | All-Gather / Reduce-Scatter | 超长序列 |

## 2) 显存分摊
- 张量并行（TP）下权重近似按 `1/TP` 分摊。
- KV cache 通常随会话分布，不一定等比例分摊；需要额外路由策略。
- PP 下每个 stage 只存自己负责的层的权重和 KV cache。

## 3) 通信代价（alpha-beta 模型）
- `T_comm = alpha + beta * n_bytes`
- `alpha`：固定时延（启动开销），NVLink ~1μs，跨机 ~10-50μs
- `beta`：每字节传输时间
- 结论：小消息受 `alpha` 主导，大消息受 `beta` 主导。

## 4) All-Reduce 近似
- Ring All-Reduce：`T_ar ≈ 2*(P-1)/P * (alpha + beta * n_bytes/P)`
- 简化：`T_ar ≈ 2*alpha*P + 2*beta*n_bytes`（小 P 时近似）
- `P` 为并行参与数。
- 卡数增加不一定线性提速，通信会吃掉收益。

## 5) TP 的通信量分析
- 每层 Attention + FFN 各需一次 All-Reduce
- 每次 All-Reduce 数据量：`2 * B * T * d_model * s` 字节
- 每层通信量：`4 * B * T * d_model * s`（两次 All-Reduce）
- 总通信量：`4 * L * B * T * d_model * s`
- Decode 时 T=1，通信量小（但 latency 受 alpha 主导）

## 6) PP（流水线并行）
- 将 `L` 层分为 `PP` 段，每段 `L/PP` 层
- 通信：段间传递激活值，数据量 `B * T * d_model * s`
- **气泡率（Bubble Ratio）**：
  - 简单调度：`bubble = (PP - 1) / (PP - 1 + num_microbatches)`
  - 微批次越多气泡越小
  - 推理时因为是自回归，PP 气泡问题更严重（每步只有 1 个 token）
  - 补救：用 TP 替代 PP（延迟敏感场景），或做 PP + TP 混合

## 7) EP（专家并行）
- MoE 中每个 GPU 放置部分 expert
- All-to-All 通信：每个 token 路由到对应 expert 的 GPU
- 通信量 ∝ `B * T * d_model * s`（每 token 的隐藏态）
- 负载均衡：若路由不均匀，部分 GPU 空闲，其他 GPU 过载
  - 辅助损失（auxiliary load balancing loss）训练时使用

## 8) 跨卡 KV 迁移权衡
- 迁移收益 > 迁移成本 才值得：
  `Benefit_reuse > T_comm + T_reindex`
- 对短序列请求，迁移常常不划算。
- 迁移数据量：`bytes_per_token * T_cache` per sequence

## 9) 多副本负载均衡
- 理想情况：均匀分配请求到各副本
- 实际挑战：请求长度差异大 → 部分副本先完成、部分阻塞
- Join-the-Shortest-Queue（JSQ）：新请求分配给队列最短的副本
- 预测负载：考虑已有请求的剩余生成长度
  `estimated_load_j = sum_{req in j} remaining_tokens_req * TPOT`

## 10) Prefill-Decode 分离的网络要求
- 传输一个请求的 KV cache：`bytes_per_token * T_input` 字节
- 例如 7B 模型 4K token：`128 KB/token * 4096 = 512 MB`
- 要求网络带宽：若传输需在 100ms 内完成 → `512 MB / 0.1s = 5.12 GB/s`
- InfiniBand / RoCE 通常满足，普通以太网可能瓶颈

## 面试一句话
- "分布式推理的上限常由通信决定，不是纯算力决定。TP 减延迟但通信频繁，PP 减通信但有气泡。"
