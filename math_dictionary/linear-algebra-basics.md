# 线性代数基础速查（LLM 相关）

## 1) 矩阵乘法 FLOPs
- `C = A B`，`A in R^{m x k}`，`B in R^{k x n}`
- FLOPs = `2 * m * k * n`（每个输出元素需 k 次乘法 + k-1 次加法，约 2k）
- 这是 Transformer 中最核心的操作：注意力投影和 FFN 都是矩阵乘法

## 2) 转置与点积
- `A^T`：转置，`(A^T)_{ij} = A_{ji}`
- 向量点积：`a · b = sum_i a_i * b_i = ||a|| ||b|| cos(theta)`
- 注意力分数 `QK^T` 本质是 query 和 key 的点积相似度

## 3) Softmax 作为概率分布
- 将实数向量映射为概率分布：`p_i = exp(z_i) / sum_j exp(z_j)`
- 性质：`sum_i p_i = 1`，`p_i > 0`
- 梯度：`dp_i/dz_j = p_i(delta_{ij} - p_j)`

## 4) 范数（Norm）
- L2 范数：`||x||_2 = sqrt(sum_i x_i^2)`
- L1 范数：`||x||_1 = sum_i |x_i|`
- Frobenius 范数（矩阵）：`||A||_F = sqrt(sum_{ij} A_{ij}^2)`
- 梯度裁剪常用：`if ||g|| > max_norm: g = g * max_norm / ||g||`

## 5) SVD（奇异值分解）
- `A = U Sigma V^T`
  - `U in R^{m x m}`：左奇异向量（正交）
  - `Sigma in R^{m x n}`：奇异值（对角，从大到小）
  - `V in R^{n x n}`：右奇异向量（正交）
- 低秩近似：`A_r = U_r Sigma_r V_r^T`（保留前 r 个奇异值）
  - 这是 LoRA 的理论基础：权重增量是低秩的
- Eckart-Young 定理：最优的秩-r 近似就是截断 SVD

## 6) 特征值与特征向量
- `A v = lambda v`
- 对称矩阵的特征值全为实数
- PCA（主成分分析）本质是对协方差矩阵做特征分解
- 在 LLM 中：激活值的协方差矩阵的特征值分布影响量化效果
  - 离群维度 = 大特征值对应的方向

## 7) 余弦相似度
- `cos_sim(a, b) = a · b / (||a|| * ||b||)`
- 取值范围：[-1, 1]
- 用途：embedding 相似度比较、BERTScore 计算
- 与 L2 距离的关系：`||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b`

## 8) 广播（Broadcasting）
- GQA 中 K 广播到多个 query head 就是张量广播
- 规则：从尾部维度对齐，维度为 1 的可扩展
- 不实际复制数据，只是逻辑上重复

## 9) 逐元素操作
- Hadamard 积（⊙）：`(A ⊙ B)_{ij} = A_{ij} * B_{ij}`
- SwiGLU 中的 `SiLU(gate) ⊙ up` 就是逐元素乘
- GELU 激活：`GELU(x) ≈ 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3)))`
- SiLU/Swish：`SiLU(x) = x * sigmoid(x)`

## 面试一句话
- "Transformer 的计算核心是矩阵乘法（投影和注意力）和逐元素操作（激活和归一化），理解形状和 FLOPs 是估算资源的基础。"
