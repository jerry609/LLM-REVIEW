# 符号与单位总表

## 模型结构符号
- `B`：batch size（并发请求数或活跃序列数）
- `T`：序列长度（token 数）；`T_q` 为 query 长度，`T_k` 为 key 长度
- `L`：模型层数（`n_layers`）
- `H`：attention head 总数（`n_heads`）
- `H_kv`：KV head 数（`n_kv_heads`，GQA/MQA 常用）
- `d_model`：隐藏维度（又称 `d_hidden`）
- `d_head`：单头维度（通常 `d_model / H`）
- `d_ff`：FFN 中间维度（通常 `4 * d_model`，SwiGLU 架构常为 `8/3 * d_model` 对齐后）
- `V`：词表大小（vocabulary size）
- `N`：模型参数量（如 7B = 7×10^9）
- `E`：MoE 专家总数；`E_active`：每 token 激活专家数

## 精度与内存符号
- `s`：每元素字节数（BF16/FP16=2，FP8/INT8=1，INT4=0.5）
- `C`：KV 缓存容量（字节）
- `M`：GPU 显存总量（如 80 GB for A100）
- `BW`：显存带宽（如 A100 HBM: 2 TB/s）

## 推理服务符号
- `lambda`：请求到达率（req/s）
- `mu`：服务速率（req/s）
- `rho`：系统利用率（`lambda/mu`，要求 `rho < 1`）
- `tau`：采样温度

## 训练相关符号
- `eta` / `lr`：学习率
- `D`：训练数据量（token 数）
- `F`：浮点运算次数（FLOPs）

## 单位换算
- `1 KB = 1024 B`，`1 MB = 1024 KB`，`1 GB = 1024 MB`，`1 TB = 1024 GB`
- `1 TFLOPS = 10^12 FLOPS`，`1 PFLOPS = 10^15 FLOPS`

## 常见模型参数速查
| 模型规模 | 典型 L | 典型 d_model | 典型 H | 典型 d_head | 典型 d_ff |
|---------|--------|-------------|--------|------------|----------|
| 7B      | 32     | 4096        | 32     | 128        | 11008    |
| 13B     | 40     | 5120        | 40     | 128        | 13824    |
| 34B     | 48     | 6656        | 52     | 128        | 17920    |
| 70B     | 80     | 8192        | 64     | 128        | 28672    |

（以 LLaMA 系列为参考，其他架构可能不同。）

## 面试易错点
- `H` 和 `H_kv` 不同：GQA 显著降低 KV 内存，但不影响 Q 的 head 数。
- "吞吐"要说明口径：tokens/s 还是 req/s。
- 延迟必须带分位数：平均值无法代表线上体验。
- `d_ff` 不等于 `4 * d_model`：LLaMA 用 SwiGLU，中间维度是 `8/3 * d_model` 向上对齐。
- 参数量估算：`N ≈ 12 * L * d_model^2`（粗估，不含 embedding 和 head）。
