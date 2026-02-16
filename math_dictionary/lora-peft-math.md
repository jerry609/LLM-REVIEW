# LoRA 与参数高效微调数学速查

## 1) LoRA（Low-Rank Adaptation）核心公式
- 原始线性层：`Y = X W`，`W in R^{d_in x d_out}`
- LoRA 修改：`Y = X (W + delta_W) = X W + X B A`
  - `B in R^{d_in x r}`，`A in R^{r x d_out}`
  - `delta_W = B A`，秩为 `r`（通常 r = 4, 8, 16, 64）
- 训练时冻结 `W`，只更新 `A` 和 `B`

## 2) 参数量节省
- 原始参数：`d_in * d_out`
- LoRA 参数：`d_in * r + r * d_out = r * (d_in + d_out)`
- 压缩比：`r * (d_in + d_out) / (d_in * d_out)`
- 例：`d_in = d_out = 4096, r = 16`
  - 原始：16.8M，LoRA：131K，压缩比 ≈ 0.78%

## 3) 初始化
- `A`：随机高斯初始化（`N(0, sigma^2)`）
- `B`：零初始化
- 保证训练开始时 `delta_W = BA = 0`（不改变预训练权重的行为）

## 4) 缩放因子
- 实际公式：`Y = X W + (alpha/r) * X B A`
- `alpha`：LoRA 缩放超参（常设为 `r` 或 `2*r`）
- `alpha/r` 控制 LoRA 更新的学习率缩放
- 使得调节 `r` 时不需要同步调节学习率

## 5) 应用位置选择
- 通常应用于 Attention 的 QKV 和 O 投影：`W_Q, W_K, W_V, W_O`
- 也可应用于 FFN 的上下投影
- 研究表明同时 adapt Q 和 V 效果最好（原始 LoRA 论文）
- 全量 LoRA（所有线性层都加）效果更优但参数量增加

## 6) QLoRA（量化 + LoRA）
- 将基座模型量化到 4-bit（NF4 格式），LoRA adapter 保持 BF16
- 显存节省：基座 ~1/4 原始大小，只有 adapter 占 BF16 显存
- Double quantization：对量化参数本身再量化
- 7B 模型 QLoRA 微调可在单张 24GB GPU 上运行

## 7) 多 LoRA 服务
- 同一基座模型可加载多个 LoRA adapter（多租户场景）
- 基座权重共享，每个请求按 adapter 路由
- adapter 显存开销：每个 adapter 约几十 MB（取决于 r 和应用层数）
- 切换 adapter 开销：修改 KV cache 中的投影（或分别缓存）

## 8) 数学直觉：为什么低秩有效
- 权重更新矩阵 `delta_W` 的秩通常远小于 `min(d_in, d_out)`
  - 微调只是在预训练特征空间中做小幅调整
  - 内在维度（intrinsic dimensionality）远低于参数空间维度
- SVD 视角：`delta_W ≈ U_r Sigma_r V_r^T`（只保留前 r 个奇异值）
  - LoRA 等价于学习这个低秩分解

## 9) 其他 PEFT 方法对比
| 方法 | 参数量 | 推理开销 | 核心思想 |
|------|--------|---------|---------|
| LoRA | r*(d_in+d_out) | 可合并，0 额外开销 | 低秩增量 |
| Prefix Tuning | L*2*d_model*k | 每层加 k 个虚拟 token | 学习前缀 KV |
| Adapter | L*2*d_model*d_adapter | 每层多两个小 FFN | 瓶颈层 |
| Prompt Tuning | k*d_model | 仅修改 embedding | 学习软 prompt |

## 面试一句话
- "LoRA 利用微调增量的低秩特性，用 <1% 的参数逼近全量微调效果，且推理时可合并回原权重零开销。"
