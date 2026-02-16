# 投机解码（Speculative Decoding）数学速查

## 1) 核心思想
- 用小模型（draft model）快速生成 K 个候选 token
- 用大模型（target model）**一次前向**并行验证这 K 个 token
- 被接受的 token 直接输出，被拒绝的位置重新采样
- 最终输出分布与 target model 完全一致（无损）

## 2) 接受-拒绝采样
- 对 draft model 提出的 token `x`：
  `P_accept(x) = min(1, p_target(x) / p_draft(x))`
- 若被拒绝，从修正分布重新采样：
  `p_resample(x) = max(0, p_target(x) - p_draft(x)) / Z`
  其中 `Z = sum_x max(0, p_target(x) - p_draft(x))`
- 这保证了最终分布 = target model 分布（数学上严格无损）

## 3) 期望接受长度
- 设每步接受概率为 `alpha`（对分布求期望）
- 期望接受的 token 数：
  `E[accepted] = sum_{i=1}^{K} alpha^i ≈ alpha / (1 - alpha)`（当 K 足够大时）
- 实际中 K 取 3-8，视 draft model 质量而定

## 4) 加速比分析
- 无投机：每 token 需 1 次 target model 前向（latency = `T_target`）
- 有投机：每 K+1 步需 1 次 target model 验证 + K 次 draft model 前向
  ```
  latency_per_token ≈ (K * T_draft + T_target) / (1 + E[accepted])
  speedup ≈ T_target / ((K * T_draft + T_target) / (1 + E[accepted]))
  ```
- 当 `T_draft << T_target` 且 `alpha` 高时，加速比可达 2-3×

## 5) 最优 K 的选择
- K 太小：验证频率高，摊销不够
- K 太大：后续 token 的接受概率 `alpha^K` 衰减快，浪费 draft 计算
- 最优 K 取决于：
  - draft/target 分布匹配度（`alpha`）
  - draft/target 速度比（`T_draft/T_target`）
- 可动态调整：跟踪近期接受率，自适应调节 K

## 6) Draft Model 选择
- 独立小模型：如用 1B 模型给 70B 模型做 draft
  - 优点：灵活、容易部署
  - 缺点：分布可能差异大，接受率低
- Self-draft（Medusa, EAGLE）：
  - 在 target model 上加轻量 head 预测多个未来 token
  - 共享 backbone，无需额外模型
  - 内存开销小
- N-gram 匹配：利用 prompt 中的 n-gram 模式（无需额外模型）
  - 适用于有大量重复模式的场景（如代码补全）

## 7) 树形投机（Tree Speculation）
- 每步生成多个候选分支（而非单条链）
- 用 tree attention 一次验证整棵候选树
- 提高验证通过的期望 token 数
- 需要 tree attention mask 支持（特殊 causal mask）

## 8) 与 KV Cache 的交互
- draft model 需要自己的 KV cache（如果不是 self-draft）
- 验证时 target model 的 KV cache 需要回滚被拒绝的 token
- PagedAttention 的 fork/rollback 操作对投机解码友好

## 面试一句话
- "投机解码本质是用 draft model 的低成本 token 换 target model 的高成本前向传播次数，数学上保证无损。"
