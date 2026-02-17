# MoE 推理优化深入

## 1. MoE 基础回顾

### 结构
- 每层 FFN 替换为 N 个 Expert（如 Mixtral: 8 experts, top-2 gating）
- Gate/Router 网络决定每个 token 发给哪些 experts
- 只有被选中的 expert 参与计算 → 稀疏激活

### 参数量 vs 计算量
| 模型 | 总参数 | 激活参数 | Experts | Top-K |
|------|--------|---------|---------|-------|
| Mixtral-8x7B | 46.7B | ~12.9B | 8 | 2 |
| DeepSeek-V3 | 671B | ~37B | 256 | 8 |
| Qwen-MoE | ~14.3B | ~2.7B | 60 | 4 |

## 2. Expert Parallelism (EP)

### 原理
- N 个 expert 分布在 N 个 GPU 上（或 N/k 个 GPU，每个放 k 个 expert）
- All-to-All 通信：
  1. 每个 GPU 根据 routing 结果，把 token 发给对应 expert 所在的 GPU
  2. Expert 计算
  3. 结果 All-to-All 送回

### 通信模式
```
Step 1: All-to-All dispatch (每个 GPU 发 tokens 给目标 expert GPU)
Step 2: Expert compute (本地)
Step 3: All-to-All combine (结果送回)
```

### 通信量分析
- 每个 token 的 hidden state: `h` floats
- 每个 GPU 平均发出 `batch_size * top_k / num_gpus` 个 token
- 总通信量 ≈ `2 × batch_size × top_k × h × sizeof(float)`

## 3. 负载均衡

### 问题
- 如果所有 token 都选同一个 expert → 该 expert 成为瓶颈
- 其他 expert 空闲 → GPU 利用率低

### 解决方案

#### 3.1 辅助损失（Auxiliary Loss）
```python
# Load balancing loss
# f_i = 分配给 expert i 的 token 比例
# P_i = routing 概率的平均值
# 目标: f_i ≈ 1/N (均匀分布)

aux_loss = N * sum(f_i * P_i for i in range(N))
total_loss = task_loss + alpha * aux_loss  # alpha ~ 0.01
```

#### 3.2 Capacity Factor
```python
capacity = int(capacity_factor * batch_size * top_k / num_experts)
# capacity_factor = 1.0 ~ 1.5
# 超过 capacity 的 token → dropped（训练）或 fallback（推理）
```

#### 3.3 Token Dropping
- **训练时**：超出 capacity 的 token 直接丢弃（通过 auxiliary loss 确保稳定）
- **推理时**：不能丢弃！必须用 fallback（如发给 shared expert 或溢出到其他设备）

### DeepSeek-V3 的改进
- **Device-level load balancing**：在 device 粒度做均衡，而非 expert 粒度
- **无需 auxiliary loss**：通过 bias term 动态调节，推理时移除 bias

## 4. 推理优化技巧

### 4.1 Expert Buffering
- 预计算下一步的 routing → 提前加载对应 expert 到 GPU
- 减少 expert 切换延迟

### 4.2 Expert Quantization
- 不同 expert 可以用不同的量化精度
- 热门 expert (高频使用): FP16
- 冷门 expert (低频使用): INT4

### 4.3 Expert Pruning
- 统计 expert 使用频率
- 移除使用率 < 1% 的 expert
- 实测对精度影响很小

### 4.4 Shared Expert + Routed Expert
```
output = shared_expert(x) + sum(gate_i * routed_expert_i(x) for i in top_k)
```
- DeepSeek-V2/V3 使用此结构
- Shared expert 处理通用知识
- Routed experts 处理专业知识

## 5. EP + TP/PP 组合

### 典型配置（以 8×H100 为例）
- **Mixtral-8x7B**: EP=8（每个 GPU 1 expert）
- **更大模型**: EP=8 + TP=2（intra-node TP，inter-node EP）
- **超大模型**: EP=32 + PP=4 + TP=2

### All-to-All 优化
- **Hierarchy**: intra-node 用 NVLink，inter-node 用 InfiniBand
- **Overlap**: All-to-All 与 shared expert 计算重叠
- **Compression**: 通信时用 FP8 压缩 hidden states

## 6. 面试回答模板

> "MoE 推理的核心挑战是 Expert Parallelism 的通信开销和负载均衡：
>
> 1. **EP**: 每个 GPU 放 N/k 个 expert，通过 All-to-All 通信交换 token
> 2. **负载均衡**: auxiliary loss + capacity factor 确保 token 均匀分布
> 3. **Token dropping**: 训练可以 drop，推理必须 fallback（shared expert）
> 4. **优化**: All-to-All overlap、expert 量化、shared expert 并行
>
> DeepSeek-V3 的创新：256 expert + device-level balancing + no auxiliary loss"

### 追问准备
1. **EP vs TP？** → EP 按 expert 切分，TP 按 tensor 切分；EP 通信是 All-to-All，TP 是 AllReduce
2. **capacity factor 怎么选？** → 训练 1.2-1.5，推理用足够大或动态扩展
3. **为什么 DeepSeek-V3 不用 aux loss？** → 用 bias term 动态调节，更稳定
4. **MoE 的显存优势？** → 推理时只需加载 top-k expert，但所有参数都要存
