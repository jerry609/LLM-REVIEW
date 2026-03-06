# 符号速查与专题入口

> 这页不再追求“把全仓所有符号一次性列完”。它现在只保留跨专题最常用、最容易混掉的记号，并把你引到对应的深入页。用法很简单：先在这里统一读法，再去专题页看完整推导。

## 先记住这套读法约定

- `B`：batch size，也可以理解成同一时刻的活跃请求数或活跃序列数。
- `T` 或 `L`：序列长度；在 serving 里常对应上下文长度。
- `d_model`：模型隐藏维度。
- `n_heads`：query head 数。
- `d_head`：单个 head 的维度。
- `n_kv_heads`：KV head 数；在 GQA 里通常小于 `n_heads`。
- `rho`：利用率；在排队论里通常表示系统是否接近满载。
- `lambda`：到达率。
- `mu`：服务率。
- `P99`：99 分位时延或尾延迟。

如果某页里出现了同一字母在不同语境下复用，优先看该页开头的“符号约定”段，而不是死记这里的字典。

## 一、Attention / 张量形状高频符号

| 记号 | 含义 | 最常出现在哪 |
|------|------|--------------|
| `X` | 输入 hidden states | Transformer 核心公式、Attention 推导 |
| `Q` | query 投影结果 | Attention、GQA、MLA、DSA |
| `K` | key 投影结果 | Attention、GQA、MLA、DSA |
| `V` | value 投影结果 | Attention、GQA、MLA、DSA |
| `B` | batch size | 张量形状、serving、训练 |
| `L` 或 `T` | 序列长度 | Attention、长上下文、KV Cache |
| `d_model` | 模型主维度 | Transformer 核心公式 |
| `n_heads` | query head 数 | MHA、GQA、MLA |
| `d_head` | 单头维度 | Attention、RoPE、FlashAttention |

继续深入：

- [tensor-shapes.md](tensor-shapes.md)
- [transformer-attention-math.md](transformer-attention-math.md)
- [../notes/attention/formula-to-code-walkthrough.md](../notes/attention/formula-to-code-walkthrough.md)

## 二、KV Cache / 长上下文高频符号

| 记号 | 含义 | 最常出现在哪 |
|------|------|--------------|
| `n_kv_heads` | KV head 数 | GQA、MQA、KV Cache |
| `group_size` | 一个 KV 组服务多少 query head | GQA |
| `d_c` | MLA 里共享潜变量的维度 | MLA |
| `d_r` | 与 RoPE 相关的独立维度 | MLA |
| `T_cache` | 当前缓存中的 token 长度 | KV Cache、serving |
| `bytes_per_token` | 单 token 的 KV 开销 | KV 显存估算、serving |
| `M_KV` | KV Cache 总占用 | KV 显存估算 |
| `k` | 稀疏注意力保留的候选 token 数 | DSA、稀疏化 |

继续深入：

- [kv-memory.md](kv-memory.md)
- [kv-compression-math.md](kv-compression-math.md)
- [kv-eviction-math.md](kv-eviction-math.md)
- [../notes/attention/mha-vs-gqa-full-derivation.md](../notes/attention/mha-vs-gqa-full-derivation.md)
- [../notes/attention/mha-vs-mla-full-derivation.md](../notes/attention/mha-vs-mla-full-derivation.md)
- [../notes/attention/mha-vs-dsa-full-derivation.md](../notes/attention/mha-vs-dsa-full-derivation.md)

## 三、Serving / Queueing 高频符号

| 记号 | 含义 | 最常出现在哪 |
|------|------|--------------|
| `TTFT` | 首 token 时延 | 服务指标、SLO |
| `TPOT` | 每输出 token 平均时延 | 服务指标、decode 性能 |
| `E2E` | 端到端时延 | serving、排队与 SLO |
| `lambda` | 到达率 | 排队与 SLO |
| `mu` | 服务率 | 排队与 SLO |
| `rho` | 利用率 | 排队与 SLO、admission |
| `W_q` | 平均排队等待时间 | M/M/1、M/G/1 |
| `L_q` | 平均队列长度 | Little 定律、M/M/1 |
| `Goodput` | 满足 SLO 的有效吞吐 | serving 指标 |

继续深入：

- [serving-metrics.md](serving-metrics.md)
- [queueing-and-slo.md](queueing-and-slo.md)
- [../notes/serving/formula-to-code-walkthrough.md](../notes/serving/formula-to-code-walkthrough.md)
- [../notes/serving/queueing-slo-formula-to-code-walkthrough.md](../notes/serving/queueing-slo-formula-to-code-walkthrough.md)

## 四、MoE / 训练高频符号

| 记号 | 含义 | 最常出现在哪 |
|------|------|--------------|
| `n_experts` | expert 数量 | MoE 路由 |
| `top_k` | 每个 token 选几个 expert | MoE 路由 |
| `capacity` | 单 expert 可接纳 token 上限 | MoE serving |
| `p_i` | 路由器给第 `i` 个 expert 的概率 | MoE 数学 |
| `A`, `B` | LoRA 的低秩矩阵 | LoRA / PEFT |
| `r` | LoRA rank，或 Token Bucket 速率；语境很重要 | 训练、serving |
| `beta` | 各类正则或偏好优化的权重 | RLHF、DPO、KTO |

继续深入：

- [moe-routing-math.md](moe-routing-math.md)
- [lora-peft-math.md](lora-peft-math.md)
- [rlhf-alignment-math.md](rlhf-alignment-math.md)
- [../notes/distributed/moe-formula-to-code-walkthrough.md](../notes/distributed/moe-formula-to-code-walkthrough.md)

## 五、最容易混掉的“同名不同义”

| 记号 | 在 Attention 里 | 在 Serving / 训练里 |
|------|-----------------|---------------------|
| `B` | batch size | 活跃请求数、活动序列数 |
| `T` | 序列长度 | 总观测时间、调度步数 |
| `r` | 低秩 rank 之类的局部记号 | Token Bucket 的速率 |
| `k` | top-k 选择数 | DSA 的候选 token 数、MoE 的 expert 数 |
| `C` | 常表示压缩后的 latent cache | 也可能表示 capacity |

遇到这类冲突，最佳做法不是硬背，而是先看“它在描述模型结构、缓存结构，还是服务系统”。

## 六、按专题跳转，而不是按字母跳转

- 想统一 Attention 记号：去 [transformer-attention-math.md](transformer-attention-math.md)
- 想统一 KV / 长上下文记号：去 [kv-memory.md](kv-memory.md) 和 [../notes/attention/mha-vs-gqa-full-derivation.md](../notes/attention/mha-vs-gqa-full-derivation.md)
- 想统一 Serving 记号：去 [serving-metrics.md](serving-metrics.md) 和 [queueing-and-slo.md](queueing-and-slo.md)
- 想统一 MoE / 训练记号：去 [moe-routing-math.md](moe-routing-math.md) 和 [lora-peft-math.md](lora-peft-math.md)

## 七、如果你只剩 5 分钟

1. 先把这一页里的 `B`、`L`、`d_model`、`n_heads`、`rho`、`lambda`、`mu` 读熟。
2. 再去看 [tensor-shapes.md](tensor-shapes.md) 和 [serving-metrics.md](serving-metrics.md)。
3. 如果你准备的是推理系统面试，再补 [queueing-and-slo.md](queueing-and-slo.md)。

## 这一页记住一句话

> 符号表不该是一张越堆越厚的百科目录，而应该是一个“把读法统一之后，立刻把人送去专题页”的入口。真正的推导放在专题里，真正的速查只保留跨专题最常用的记号。
