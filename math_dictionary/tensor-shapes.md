# Tensor 形状速查（MHA/GQA/MQA）

## 常见张量
- 输入隐藏态：`X in R^{B x T x d_model}`
- 查询：`Q in R^{B x T x H x d_head}`
- 键值（MHA）：`K,V in R^{B x T x H x d_head}`
- 键值（GQA）：`K,V in R^{B x T x H_kv x d_head}`
- 键值（MQA）：`K,V in R^{B x T x 1 x d_head}`（即 `H_kv=1`）

## 线性投影权重
- `W_Q in R^{d_model x (H * d_head)}`
- `W_K in R^{d_model x (H_kv * d_head)}`
- `W_V in R^{d_model x (H_kv * d_head)}`
- `W_O in R^{(H * d_head) x d_model}`

## 注意力计算
- 打分：`QK^T -> R^{B x H x T_q x T_k}`（GQA 下 K 广播到对应的 query head 组）
- 缩放：`S = QK^T / sqrt(d_head)`
- 掩码：`S_masked = S + mask`（因果 mask 中未来位置填 `-inf`）
- 概率：`P = softmax(S_masked)`
- 输出：`O = PV`，形状 `R^{B x H x T_q x d_head}`
- 拼接投影：`Output = Concat(O_1,...,O_H) W_O`，形状 `R^{B x T x d_model}`

## FFN 形状（SwiGLU 变体）
- 门控：`gate = X W_gate`，`W_gate in R^{d_model x d_ff}`
- 上投影：`up = X W_up`，`W_up in R^{d_model x d_ff}`
- 激活：`hidden = SiLU(gate) ⊙ up`（逐元素乘）
- 下投影：`output = hidden W_down`，`W_down in R^{d_ff x d_model}`

## KV cache 存储形状（按层）
- `K_cache, V_cache in R^{B x T_cache x H_kv x d_head}`
- 每层每 token 元素数：`2 * H_kv * d_head`
- 新增一个 token 时只 append 一行，不重算历史

## RoPE 位置编码形状
- 旋转矩阵作用于 Q 和 K 的每个 head：`q_rot, k_rot in R^{d_head}`
- 将 `d_head` 维两两配对（cos/sin 旋转），不引入额外参数
- RoPE 作用于 Q 和 K（不作用于 V）

## GQA 广播细节
- 若 `H=32, H_kv=8`，则每 4 个 query head 共享 1 组 KV head
- 组大小：`group_size = H / H_kv`
- 广播方式：`K_expanded[:,h,:,:] = K[:,h // group_size,:,:]`

## 工程结论
- 增大 `H` 不一定增加 KV 占用；关键是 `H_kv`。
- 在同等容量下，GQA 比 MHA 能容纳更长上下文/更高并发。
- MQA（`H_kv=1`）KV 最省但质量可能下降；GQA 是主流折中。
- SwiGLU FFN 有 3 个权重矩阵（而非标准 FFN 的 2 个），参数量略多。
