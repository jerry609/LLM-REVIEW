# RL 训练/推理系统优化：从单 LoRA 到千级并行

> 面向 JD：**提升 RL 系统效率与可扩展性，让算法以更快周期演进。**

---

## 一、RLHF/RL 训练链路全景

### 1.1 经典 RLHF 训练链路

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RLHF 训练一次迭代                            │
│                                                                     │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐   │
│  │  Policy   │───▶│ Generate  │───▶│  Reward   │───▶│  Update   │   │
│  │  (Actor)  │    │ Rollout   │    │  Model    │    │  (PPO/    │   │
│  │           │    │ (推理)    │    │  (RM)     │    │   GRPO)   │   │
│  └───────────┘    └───────────┘    └───────────┘    └───────────┘   │
│       ▲                                                   │         │
│       └───────────────────────────────────────────────────┘         │
│                      参数更新后回到 Policy                           │
└─────────────────────────────────────────────────────────────────────┘
```

**链路瓶颈分析**：

| 阶段 | 计算类型 | 瓶颈 | 典型耗时占比 |
|------|----------|------|-------------|
| Generate Rollout | 自回归推理 | **Memory Bandwidth Bound** (KV Cache 读写) | 40-60% |
| Reward Scoring | Forward-only | Compute Bound (可 batch) | 10-20% |
| PPO/GRPO Update | 训练 (前向+反向) | Compute Bound | 20-30% |
| 数据传输 | GPU↔CPU / 跨节点 | Network / PCIe | 5-15% |

> **核心洞察**：Rollout 生成是最大瓶颈 — 它是自回归推理，无法像训练那样大 batch 并行。优化 RL 系统的第一优先级是加速 Rollout。

### 1.2 GRPO vs PPO 的系统差异

| 维度 | PPO | GRPO |
|------|-----|------|
| 额外模型 | Critic (Value) + Reference | 仅 Reference |
| GPU 显存 | 需 4 份模型权重 (Actor, Critic, RM, Ref) | 需 3 份 (Actor, RM, Ref) |
| Rollout 要求 | 每条 prompt 生成 1 条 | 每条 prompt 生成 **G 条** (group) |
| 优势估计 | GAE (需 Critic 前向) | 组内标准化 (无需 Critic) |
| 系统优势 | 单条生成即可 | 省去 Critic 但 Rollout 量翻 G 倍 |

```python
# GRPO 优势估计 — 无需 Critic
def grpo_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """rewards: (batch, group_size)"""
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True).clamp(min=1e-8)
    return (rewards - mean) / std
```

---

## 二、RL 推理链路优化

### 2.1 Rollout 生成加速

Rollout 是 RL 训练中推理吞吐的最大瓶颈。关键优化：

#### (a) vLLM 集成 Rollout

```python
# 将 Actor 权重同步到 vLLM 推理引擎
# AReaL / OpenRLHF 的典型做法
class VLLMRolloutWorker:
    def __init__(self, model_path, tp_size=4):
        from vllm import LLM, SamplingParams
        self.llm = LLM(model_path, tensor_parallel_size=tp_size,
                        enforce_eager=True,   # RL 场景权重频繁更新
                        enable_prefix_caching=False)

    def generate(self, prompts, sampling_params):
        outputs = self.llm.generate(prompts, sampling_params)
        return [o.outputs[0].text for o in outputs]

    def update_weights(self, new_state_dict):
        """RL 每步更新后，把新权重同步到 vLLM"""
        # 方案1: 直接 load_state_dict（简单但慢）
        # 方案2: NCCL broadcast（AReaL 方案，快）
        # 方案3: 共享显存（Slime 方案，零拷贝）
        pass
```

#### (b) 关键推理优化技术

| 技术 | 原理 | 对 RL 的意义 |
|------|------|-------------|
| **Continuous Batching** | 动态组 batch，完成即出 | GRPO G 条并发生成时利用率提升 |
| **PagedAttention** | KV Cache 分页管理 | 长 rollout 不 OOM |
| **Chunked Prefill** | Prefill 分片不阻塞 decode | prompt 长时不卡推理 |
| **Prefix Caching** | 相同 prompt 复用 KV Cache | GRPO 同 prompt 生成 G 条时可直接复用 |
| **Speculative Decoding** | 小模型猜 + 大模型验 | 减少 Rollout 延迟 |
| **FP8 推理** | 量化推理 | 推理吞吐翻倍，RL 对推理精度不敏感 |

#### (c) Rollout 与训练的并行调度

```
时间线 ──────────────────────────────────────────────────▶

传统串行:
  [====== Rollout ======][=== Reward ===][==== Train ====]

Rollout-Train 流水线 (AReaL):
  [====== Rollout Batch 1 ======]
                        [=== Reward 1 ===][==== Train 1 ====]
                        [====== Rollout Batch 2 ======]
                                          [=== Reward 2 ===][==== Train 2 ====]

异步 Rollout (Slime):
  [==== Rollout (连续) ============================]
            [=== Reward ===]   [=== Reward ===]
                  [=== Train ===]    [=== Train ===]
```

### 2.2 权重同步策略

RL 训练中 Actor 权重持续更新，需要高效同步到推理引擎：

| 策略 | 延迟 | 适用场景 |
|------|------|---------|
| **全量 load_state_dict** | 秒级 | 单机小模型 |
| **NCCL Broadcast** | 百毫秒 | 多机分布式，AReaL 默认方案 |
| **共享显存 (Zero-Copy)** | 接近零 | 同机训练+推理，Slime 方案 |
| **异步更新 + Stale Policy** | 零等待 | 允许轻微 off-policy，吞吐最高 |

```python
# NCCL Broadcast 权重同步示例
def broadcast_weights(actor_model, vllm_engine, src_rank=0):
    """训练完成后，将 actor 权重广播到所有推理 worker"""
    for name, param in actor_model.named_parameters():
        torch.distributed.broadcast(param.data, src=src_rank)
    # vLLM worker 侧接收后直接更新本地模型
```

---

## 三、千级 LoRA-RL 并行训练

### 3.1 为什么要千级 LoRA 并行

| 场景 | 说明 |
|------|------|
| **多奖励信号探索** | 同时训练 1000 个 LoRA，每个对应不同奖励函数/偏好 |
| **NAS 式搜索** | 不同 LoRA rank / target module 组合的超参搜索 |
| **群体进化** | Population-based Training (PBT)，LoRA 之间淘汰与突变 |
| **多任务对齐** | 每个 LoRA 专注一个任务（代码/数学/对话/安全），最后合并 |

### 3.2 Multi-LoRA 训练架构

```
┌──────────────────────────────────────────────────────────────┐
│                    共享 Base Model (冻结)                      │
│                    e.g. Qwen-72B, FP8                        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────┐  ┌──────┐  ┌──────┐       ┌──────┐               │
│  │LoRA 1│  │LoRA 2│  │LoRA 3│ ...   │LoRA N│               │
│  │r=16  │  │r=32  │  │r=16  │       │r=64  │               │
│  │task_A│  │task_B│  │task_C│       │task_N│               │
│  └──┬───┘  └──┬───┘  └──┬───┘       └──┬───┘               │
│     │         │         │              │                     │
│  ┌──▼───┐  ┌──▼───┐  ┌──▼───┐       ┌──▼───┐               │
│  │Optim1│  │Optim2│  │Optim3│ ...   │OptimN│               │
│  │State │  │State │  │State │       │State │               │
│  └──────┘  └──────┘  └──────┘       └──────┘               │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 3.3 显存估算与可行性

| 组件 | 单 LoRA (r=16, 7B 模型) | 1000 个 LoRA |
|------|------------------------|-------------|
| Base Model (FP8) | 7 GB | 7 GB (共享) |
| LoRA 权重 (A+B) | ~27 MB | ~27 GB |
| 优化器状态 (AdamW, FP32) | ~108 MB | ~108 GB |
| 梯度 (FP16) | ~54 MB | ~54 GB |
| KV Cache (推理) | 取决于 batch | 共享 Base 的 KV |
| **合计** | ~7.2 GB | ~196 GB |

> 结论：1000 个 r=16 的 LoRA 在 **4×A100-80G** 上理论可行（权重+优化器）。挑战在于如何高效调度计算。

### 3.4 计算调度方案

#### 方案 A：串行逐 LoRA (Naive)

```python
for lora_id in range(N):
    activate_lora(lora_id)
    loss = forward(base_model, lora[lora_id], batch[lora_id])
    loss.backward()
    optimizer[lora_id].step()
```

- 优点：实现最简单
- 缺点：无法利用 batch 并行，GPU 利用率极低

#### 方案 B：Batched LoRA Forward (S-LoRA 思路)

```python
# 核心思想：一次 forward 同时处理多个 LoRA 的请求
# Base model 的 compute 只做一次，LoRA 的增量用 batched GEMM

def batched_lora_forward(x, base_weight, lora_A_batch, lora_B_batch, lora_indices):
    """
    x: (total_tokens, d)
    base_weight: (d, d_out) — 共享
    lora_A_batch: (num_active_loras, d, r)
    lora_B_batch: (num_active_loras, r, d_out)
    lora_indices: (total_tokens,) — 每个 token 属于哪个 LoRA
    """
    # 1. Base 计算 (共享，只算一次)
    base_out = x @ base_weight                         # (total_tokens, d_out)

    # 2. LoRA 增量 (使用 CUDA Custom Kernel 或 Triton)
    #    按 lora_indices gather 对应的 A, B 矩阵
    #    等效于: delta = x @ A[lora_indices] @ B[lora_indices]
    delta = batched_lora_gemm(x, lora_A_batch, lora_B_batch, lora_indices)

    return base_out + delta
```

- S-LoRA 论文证明：用 **CUDA Custom Kernel** 做 segmented gather-GEMM，可以在一次 forward 中高效处理上千个 LoRA
- GPU 利用率接近单模型训练

#### 方案 C：LoRA 分组 + 数据并行

```
GPU 0: Base Model Shard (TP=0) + LoRA 1-250
GPU 1: Base Model Shard (TP=1) + LoRA 1-250
GPU 2: Base Model Shard (TP=0) + LoRA 251-500
GPU 3: Base Model Shard (TP=1) + LoRA 251-500
...
```

- 每组 GPU 负责一批 LoRA 的训练
- 组间不需要通信（各自独立的 LoRA 参数）
- Base Model 用 TP 切分在组内共享

### 3.5 Multi-LoRA Rollout 优化

RL 训练中每个 LoRA 都要生成 Rollout，是吞吐瓶颈：

```
┌─────────────────────────────────────────────┐
│         vLLM Multi-LoRA Serving             │
│                                             │
│   Shared Base KV Cache                      │
│   ┌─────────────────────────────────┐       │
│   │    PagedAttention 物理块池       │       │
│   └─────────────────────────────────┘       │
│                                             │
│   Request Queue:                            │
│   [prompt_1, lora_id=3]                     │
│   [prompt_2, lora_id=7]                     │
│   [prompt_3, lora_id=3]  ← 复用 LoRA 3     │
│   [prompt_4, lora_id=12]                    │
│                                             │
│   Scheduler: 按 LoRA ID 分组 batch          │
│   → 最小化 LoRA 权重切换次数                 │
└─────────────────────────────────────────────┘
```

**关键优化点**：
1. **LoRA-Aware Scheduling**：调度器按 LoRA ID 聚合请求，减少权重切换
2. **LoRA Weight Prefetching**：预取下一批 LoRA 权重到 GPU
3. **Shared KV Cache**：Base Model 部分的 KV Cache 可复用
4. **异步权重更新**：训练更新 LoRA 权重时不阻塞推理

---

## 四、Multi-LoRA Joint Training

### 4.1 核心思想

不是简单地并行训练多个独立 LoRA，而是让多个 LoRA 之间**共享信息、协同进化**。

### 4.2 Joint Training 策略

#### (a) 共享梯度信号

```python
# 不同 LoRA 之间共享部分梯度信息
# 类似 Multi-Task Learning 的思路
def joint_training_step(base_model, loras, batches, alpha=0.1):
    all_grads = {}

    for lora_id, (lora, batch) in enumerate(zip(loras, batches)):
        loss = compute_loss(base_model, lora, batch)
        grads = torch.autograd.grad(loss, lora.parameters())
        all_grads[lora_id] = grads

    # 共享梯度：每个 LoRA 的更新 = 自己的梯度 + alpha * 其他 LoRA 梯度均值
    for lora_id, lora in enumerate(loras):
        own_grad = all_grads[lora_id]
        other_grads_mean = mean_of([all_grads[j] for j in range(len(loras)) if j != lora_id])

        for p, g_own, g_shared in zip(lora.parameters(), own_grad, other_grads_mean):
            p.grad = g_own + alpha * g_shared

        optimizers[lora_id].step()
```

#### (b) LoRA 合并与分裂 (进化策略)

```python
# Population-Based Training (PBT) for LoRA
def pbt_step(loras, rewards, mutation_rate=0.1):
    """
    1. 评估所有 LoRA 的奖励
    2. 淘汰表现差的 LoRA
    3. 复制表现好的 LoRA 并加入突变
    """
    sorted_indices = torch.argsort(rewards, descending=True)

    # 淘汰后 25%，用前 25% 替换
    n_replace = len(loras) // 4
    for i in range(n_replace):
        src = sorted_indices[i]           # 好的
        dst = sorted_indices[-(i+1)]      # 差的

        # 复制权重
        loras[dst].load_state_dict(loras[src].state_dict())

        # 突变：加噪声
        with torch.no_grad():
            for p in loras[dst].parameters():
                p.add_(torch.randn_like(p) * mutation_rate)
```

#### (c) LoRA 融合 (Merging)

```python
# 训练完成后，多个 LoRA 按任务权重融合
def merge_loras(loras, weights):
    """
    Task Arithmetic: ΔW = Σ w_i * (A_i @ B_i)
    """
    merged_A = sum(w * lora.A.weight for w, lora in zip(weights, loras))
    merged_B = sum(w * lora.B.weight for w, lora in zip(weights, loras))
    return merged_A, merged_B
```

### 4.3 资源调度

#### 调度器设计

```python
@dataclass
class LoRAJob:
    lora_id: int
    priority: float          # 基于当前 reward 动态调整
    gpu_memory_mb: float     # 预估显存需求
    compute_flops: float     # 预估计算量
    last_update_step: int    # 上次更新步数

class LoRAScheduler:
    """千级 LoRA 的资源调度器"""

    def __init__(self, gpu_pool: List[GPU], max_concurrent: int = 64):
        self.gpu_pool = gpu_pool
        self.max_concurrent = max_concurrent
        self.job_queue = PriorityQueue()

    def schedule(self, jobs: List[LoRAJob]) -> Dict[int, List[LoRAJob]]:
        """
        调度策略:
        1. 按 priority 排序
        2. Bin-packing 到 GPU（类似 Kubernetes 调度）
        3. 同 GPU 上的 LoRA 共享 Base Model 计算
        """
        # 按优先级排序
        sorted_jobs = sorted(jobs, key=lambda j: j.priority, reverse=True)

        gpu_assignments = {gpu.id: [] for gpu in self.gpu_pool}
        gpu_mem_used = {gpu.id: 0 for gpu in self.gpu_pool}

        for job in sorted_jobs[:self.max_concurrent]:
            # First-Fit Decreasing bin packing
            best_gpu = min(self.gpu_pool, key=lambda g: gpu_mem_used[g.id])
            if gpu_mem_used[best_gpu.id] + job.gpu_memory_mb <= best_gpu.total_memory_mb:
                gpu_assignments[best_gpu.id].append(job)
                gpu_mem_used[best_gpu.id] += job.gpu_memory_mb

        return gpu_assignments
```

---

## 五、框架深度剖析

### 5.1 框架定位对比

| 框架 | 定位 | RL 相关能力 | 核心优势 |
|------|------|-----------|---------|
| **vLLM** | 通用 LLM 推理引擎 | Rollout 生成 | PagedAttention, Continuous Batching |
| **SGLang** | 结构化生成 + 推理 | Rollout + 约束解码 | RadixAttention, FSM 约束 |
| **AReaL** | 字节跳动 RL 训练框架 | 端到端 RLHF | Rollout-Train Pipeline, NCCL 同步 |
| **Slime** | RL 训练加速 | 同机零拷贝 | 共享显存, 异步调度 |
| **Megatron-LM** | 大规模分布式训练 | 预训练/SFT | 3D 并行, 极致 MFU |

### 5.2 vLLM 在 RL 中的应用

```python
# vLLM 作为 Rollout Engine 的典型集成
# OpenRLHF 风格
class VLLMRolloutEngine:
    def __init__(self, model_path, tp_size):
        self.engine = AsyncLLMEngine.from_engine_args(
            EngineArgs(
                model=model_path,
                tensor_parallel_size=tp_size,
                max_model_len=4096,
                enable_prefix_caching=True,   # GRPO 同 prompt 多次采样复用
                disable_log_stats=True,
            )
        )

    async def generate_rollouts(self, prompts, n_samples=8):
        """GRPO: 每个 prompt 生成 n_samples 条"""
        params = SamplingParams(
            temperature=1.0,
            top_p=0.95,
            max_tokens=512,
            n=n_samples,  # 一次请求生成多条
        )
        results = []
        async for output in self.engine.generate(prompts, params):
            results.append(output)
        return results
```

**vLLM 源码关键路径**（面试高频）：

```
vllm/
├── engine/
│   ├── async_llm_engine.py    ← RL 场景用 AsyncEngine
│   └── llm_engine.py          ← 核心调度循环
├── core/
│   ├── scheduler.py           ← Continuous Batching 调度
│   └── block_manager_v2.py    ← PagedAttention 块管理
├── worker/
│   └── model_runner.py        ← 模型执行 (forward + sample)
├── model_executor/
│   └── layers/
│       └── lora/              ← Multi-LoRA 推理支持
│           ├── layers.py      ← LoRA linear 层实现
│           └── worker_manager.py ← LoRA 权重热加载
└── attention/
    └── backends/
        └── flash_attn.py      ← FlashAttention 后端
```

### 5.3 SGLang 核心机制

```python
# SGLang 的 RadixAttention 在 RL 中的价值
# 同一个 system prompt 被大量 RL rollout 复用

# 传统 vLLM: 每次 rollout 都重新算 system prompt 的 KV
# SGLang:    自动识别公共前缀，KV Cache 命中率极高

import sglang as sgl

@sgl.function
def rl_rollout(s, system_prompt, user_prompt):
    s += sgl.system(system_prompt)  # 自动前缀缓存
    s += sgl.user(user_prompt)
    s += sgl.assistant(sgl.gen("response", max_tokens=512, temperature=1.0))
```

**RadixAttention vs vLLM Prefix Caching**：

| 维度 | vLLM Prefix Caching | SGLang RadixAttention |
|------|---------------------|----------------------|
| 匹配粒度 | Block 级 (16 tokens) | Token 级 |
| 数据结构 | Hash Table | Radix Tree |
| 部分匹配 | 支持 | 更高效（LPM） |
| RL 场景 | 可用 | 更优（细粒度复用） |

### 5.4 AReaL 架构

AReaL (Autonomous REinforcement Learning) 是字节跳动的 RL 训练框架：

```
┌──────────────────────────────────────────────────────┐
│                    AReaL Controller                   │
│               (任务编排 + 资源调度)                     │
├──────────────┬──────────────┬────────────────────────┤
│  Actor Pool  │  Critic Pool │  Reference Pool        │
│  (训练+推理)  │  (训练)      │  (推理, 冻结)          │
│              │              │                        │
│  vLLM 推理 ──┤              │                        │
│  Megatron 训练┤              │                        │
└──────────────┴──────────────┴────────────────────────┘
```

核心设计：
1. **训练推理分离**：Actor 的训练和推理用不同的 GPU 组
2. **NCCL 权重同步**：训练完一步后通过 NCCL broadcast 更新推理引擎
3. **流水线调度**：Rollout 和 Training 交替执行，最大化 GPU 利用率
4. **弹性扩缩容**：根据 Rollout 压力动态调整推理 GPU 数量

### 5.5 Slime 框架特色

Slime 的核心创新是**同机训练推理共享显存**：

```
┌──────────────────────────────────────────────┐
│              单机 8×A100/H100                 │
│                                              │
│  GPU 0-3: Training (Megatron TP=4)           │
│     ▲                                        │
│     │ 共享显存 (Zero-Copy)                    │
│     ▼                                        │
│  GPU 4-7: Inference (vLLM TP=4)             │
│                                              │
│  权重同步: CUDA IPC / cudaMemcpy peer-to-peer│
│  延迟: < 1ms (vs NCCL broadcast ~100ms)     │
└──────────────────────────────────────────────┘
```

核心优势：
- **零拷贝权重同步**：训练完直接让推理引擎看到新权重
- **消除序列化开销**：无需 state_dict → CPU → GPU 的搬运
- **异步流水线**：训练和推理可以真正并行

### 5.6 Megatron-LM 在 RL 中的角色

Megatron 通常作为 RL 系统中的**训练后端**：

```python
# Megatron 3D 并行配置 (RL 训练后端)
# TP: 机内切权重 (NVLink)
# PP: 跨机切层 (IB)
# DP: 数据并行 (多副本)
#
# 8 机 × 8 卡 = 64 GPU
# TP=8, PP=4, DP=2
# → 每条流水线 32 GPU, 2 路数据并行

megatron_args = {
    "tensor_model_parallel_size": 8,
    "pipeline_model_parallel_size": 4,
    "data_parallel_size": 2,
    "micro_batch_size": 1,
    "global_batch_size": 64,     # RL 通常用较小 batch
    "gradient_accumulation_steps": 32,
    "fp16": True,
    "use_flash_attn": True,
}
```

---

## 六、系统优化实战

### 6.1 瓶颈定位方法论

```bash
# Step 1: 整体 profiling — 找到时间分布
nsys profile -o rl_trace python train_rl.py

# Step 2: GPU 利用率监控
nvidia-smi dmon -s u -d 1 -f gpu_util.log

# Step 3: 通信瓶颈分析
NCCL_DEBUG=INFO python train_rl.py 2>&1 | grep "NCCL"

# Step 4: 内存瓶颈分析
torch.cuda.memory_summary(device=0, abbreviated=True)
```

### 6.2 常见瓶颈与优化方案

| 瓶颈现象 | 诊断方法 | 优化方案 |
|----------|---------|---------|
| GPU 利用率低 (<50%) | `nvidia-smi` | 增大 batch / 合并 LoRA forward |
| Rollout 太慢 | 计时各阶段 | 换 vLLM 推理 / 开 prefix caching |
| 权重同步延迟大 | NCCL profile | 改用共享显存或异步更新 |
| OOM | `memory_summary()` | 梯度检查点 / 减 batch / FP8 |
| LoRA 切换频繁 | 调度日志 | LoRA-aware scheduling / 预取 |
| 多机通信慢 | `NCCL_DEBUG` | TP 机内 + PP 跨机 / 压缩通信 |

### 6.3 GRPO 全链路优化示例

```python
class OptimizedGRPOTrainer:
    """千级 LoRA GRPO 训练的工程优化版本"""

    def __init__(self, base_model, loras, vllm_engine, reward_fn):
        self.base_model = base_model
        self.loras = loras
        self.vllm_engine = vllm_engine
        self.reward_fn = reward_fn

    def train_step(self, prompts):
        # 1. 批量 Rollout（利用 vLLM 的 Continuous Batching）
        #    所有 LoRA 的 rollout 请求混合提交
        all_requests = []
        for lora_id, lora in enumerate(self.loras):
            for prompt in prompts[lora_id]:
                all_requests.append({
                    "prompt": prompt,
                    "lora_id": lora_id,
                    "n": 8,  # GRPO group size
                })

        # vLLM 按 lora_id 聚合 batch，最小化切换
        responses = self.vllm_engine.batch_generate(all_requests)

        # 2. 批量 Reward（所有 LoRA 的 response 一起打分）
        rewards = self.reward_fn.batch_score(responses)

        # 3. 分组更新（可并行）
        #    使用 batched LoRA forward 一次计算多个 LoRA 的 loss
        losses = batched_grpo_loss(
            self.base_model, self.loras,
            responses, rewards,
        )

        # 4. 梯度更新（各 LoRA 独立，可并行）
        for lora_id, (lora, loss) in enumerate(zip(self.loras, losses)):
            loss.backward()
            self.optimizers[lora_id].step()
            self.optimizers[lora_id].zero_grad()

        # 5. 异步权重同步到 vLLM
        self.vllm_engine.async_update_lora_weights(self.loras)
```

---

## 七、面试高频问题

### Q1: RL 训练中，Rollout 生成是最大的瓶颈，你会怎么优化？

**参考答案**：
1. 使用 vLLM 或 SGLang 替代朴素的 HuggingFace generate（Continuous Batching + PagedAttention，吞吐提升 5-10x）。
2. GRPO 场景下开启 Prefix Caching（同 prompt 生成多条时复用 KV Cache）。
3. 训练和推理流水线化（不等 rollout 全部完成就开始训练）。
4. 权重同步用 NCCL broadcast 或共享显存，避免 state_dict 序列化。
5. 推理侧使用 FP8 量化（RL 对推理精度不敏感）。

### Q2: 如何实现千级 LoRA 的并行训练？

**参考答案**：
1. Base Model 冻结并共享，只训练 LoRA 参数（显存可控）。
2. 使用 S-LoRA 的 batched GEMM 技术，一次 forward 处理多个 LoRA。
3. 资源调度：按 LoRA 优先级做 bin-packing，类似 K8s Pod 调度。
4. Rollout 阶段用 vLLM Multi-LoRA serving，按 LoRA ID 聚合 batch。
5. 可选 PBT 策略：定期淘汰表现差的 LoRA，复制好的并加突变。

### Q3: vLLM 的 PagedAttention 在 RL 训练中有什么特殊价值？

**参考答案**：
RL 训练中 Rollout 生成的 sequence length 不确定（可能很长），传统连续显存分配会导致碎片化和 OOM。PagedAttention 用分页管理 KV Cache，按需分配物理块，显存利用率从 ~50% 提升到 >95%。对于 GRPO 的 group sampling（同 prompt 多条），还可以用 CoW (Copy-on-Write) 共享前缀的 KV 物理块。

### Q4: AReaL 和 Slime 在权重同步策略上有什么区别？

**参考答案**：
- AReaL 用 **NCCL Broadcast**：训练完一步后，通过 NCCL 将 Actor 权重广播到推理 worker。延迟约百毫秒级，适合多机场景。
- Slime 用 **共享显存 (Zero-Copy)**：训练和推理在同一台机器上，通过 CUDA IPC 直接共享 GPU 显存，延迟 <1ms。适合单机多卡场景。
- 权衡：AReaL 可扩展到多机但有同步开销；Slime 同步极快但限制在单机内。

### Q5: 如何设计 Multi-LoRA 的调度系统？

**参考答案**：
核心思路类似 Kubernetes 的资源调度：
1. **优先级**：基于每个 LoRA 当前的奖励表现动态调整。
2. **Bin Packing**：将 LoRA 按显存需求打包到 GPU，最大化利用率。
3. **亲和性**：相同 LoRA 的训练和推理尽量放同一机器（减少权重同步）。
4. **弹性伸缩**：Rollout 阶段动态扩推理 GPU，训练阶段收回。
5. **公平性**：避免高优先级 LoRA 饿死低优先级的。

---

## 八、延伸阅读

| 资源 | 说明 |
|------|------|
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | 开源 RLHF 框架，Ray + vLLM 集成 |
| [S-LoRA](https://arxiv.org/abs/2311.03285) | 千级 LoRA 高效服务 |
| [vLLM Multi-LoRA](https://docs.vllm.ai/en/latest/models/lora.html) | vLLM 官方 LoRA 支持 |
| [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | NVIDIA 分布式训练框架 |
| [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-chat) | 微软 RLHF 训练方案 |
| [TRL](https://github.com/huggingface/trl) | HuggingFace 的 RL 训练库 |
