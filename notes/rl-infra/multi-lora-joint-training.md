# Multi-LoRA Joint Training：实现与资源调度

> 核心问题：**如何在共享 Base Model 上同时训练/推理上千个 LoRA，并让它们协同进化？**

---

## 一、单 LoRA 推理基线（回顾）

### 1.1 标准 LoRA 推理流程

对模型中每个 target module（如 `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`）：

$$
h = xW + x \cdot \underbrace{A \cdot B}_{\Delta W} \cdot \frac{\alpha}{r}
$$

- $W \in \mathbb{R}^{d_{in} \times d_{out}}$：冻结的 Base 权重
- $A \in \mathbb{R}^{d_{in} \times r}$：LoRA 下投影（初始化为 Kaiming Uniform）
- $B \in \mathbb{R}^{r \times d_{out}}$：LoRA 上投影（初始化为 0）
- $r$：LoRA 秩（rank），典型 8-64
- $\alpha$：缩放因子，典型等于 $r$ 或 $2r$

### 1.2 推理时合并 vs 动态计算

| 模式 | 计算方式 | 切换成本 | 适用场景 |
|------|---------|---------|---------|
| **权重合并** | $W' = W + \frac{\alpha}{r} AB$ | 高（需要重新合并/还原） | 单 LoRA 长期服务 |
| **动态计算** | $h = xW + x \cdot A \cdot B \cdot s$ | 低（只切 A, B 指针） | Multi-LoRA serving |

> Multi-LoRA 场景必须用动态计算模式，否则每次切换都要修改 Base 权重。

---

## 二、Multi-LoRA 推理：从 S-LoRA 到 Batched GEMM

### 2.1 朴素方法的困境

```python
# 朴素实现：逐个 LoRA 串行处理
for request in batch:
    lora = load_lora(request.lora_id)
    activate_lora(base_model, lora)
    output = base_model(request.input)  # 单个请求 forward

# 问题：
# 1. GPU 利用率极低（batch_size=1 的 forward）
# 2. LoRA 切换开销（加载权重到 GPU）
# 3. 无法利用 batching 优势
```

### 2.2 S-LoRA 的 Batched GEMM

核心：**一次 forward 中混合处理属于不同 LoRA 的 token**。

```
输入:  [tok_1(LoRA_A), tok_2(LoRA_A), tok_3(LoRA_B), tok_4(LoRA_C), tok_5(LoRA_B)]
索引:  [      0,              0,              1,              2,              1     ]

Base 计算 (一次 GEMM):
  base_out = X @ W                    # (5, d_out) — 所有 token 共享

LoRA 增量 (Segmented GEMM):
  对每个 LoRA ID 分组:
    Group 0 (LoRA_A): [tok_1, tok_2] @ A_0 @ B_0
    Group 1 (LoRA_B): [tok_3, tok_5] @ A_1 @ B_1
    Group 2 (LoRA_C): [tok_4]        @ A_2 @ B_2

最终:  output = base_out + scatter(lora_deltas, indices)
```

### 2.3 CUDA Kernel 实现要点

```cpp
// Triton 伪代码：Segmented Batched LoRA GEMM
@triton.jit
def batched_lora_kernel(
    X_ptr, A_ptr, B_ptr,        // 输入, LoRA A 矩阵池, LoRA B 矩阵池
    lora_indices_ptr,            // 每个 token 对应的 LoRA ID
    segment_starts_ptr,          // 每个 LoRA 组的起始位置
    output_ptr,                  // 输出
    d_in, r, d_out, num_tokens,
):
    // 1. 每个 thread block 处理一个 LoRA 组的一个 tile
    pid = tl.program_id(0)
    lora_id = get_lora_id(pid)

    // 2. 加载该 LoRA 的 A, B 矩阵到 shared memory
    A = load_tile(A_ptr + lora_id * d_in * r, ...)
    B = load_tile(B_ptr + lora_id * r * d_out, ...)

    // 3. 加载该组的输入 token
    segment_start = tl.load(segment_starts_ptr + lora_id)
    x_tile = load_tile(X_ptr + segment_start * d_in, ...)

    // 4. 计算 x @ A @ B
    tmp = tl.dot(x_tile, A)    // (tile, r)
    out = tl.dot(tmp, B)       // (tile, d_out)

    // 5. scatter 回对应位置
    store_tile(output_ptr + segment_start * d_out, out)
```

### 2.4 显存管理：LoRA 权重池

```python
class LoRAWeightPool:
    """
    在 GPU 上预分配一个大池子，存放所有活跃 LoRA 的权重。
    类似 PagedAttention 的思想，但管理的是 LoRA 权重而非 KV Cache。
    """
    def __init__(self, max_loras: int, d_in: int, d_out: int, r: int, dtype=torch.float16):
        self.max_loras = max_loras
        # 预分配连续显存
        self.A_pool = torch.zeros(max_loras, d_in, r, dtype=dtype, device="cuda")
        self.B_pool = torch.zeros(max_loras, r, d_out, dtype=dtype, device="cuda")
        self.active_set = set()
        self.lru_cache = OrderedDict()  # LRU 驱逐策略

    def load_lora(self, lora_id: int, A: torch.Tensor, B: torch.Tensor) -> int:
        """加载 LoRA 到池中，返回 slot index"""
        if len(self.active_set) >= self.max_loras:
            evict_id = self.lru_cache.popitem(last=False)[0]  # LRU evict
            self.active_set.discard(evict_id)

        slot = self._find_free_slot()
        self.A_pool[slot].copy_(A)
        self.B_pool[slot].copy_(B)
        self.active_set.add(lora_id)
        self.lru_cache[lora_id] = slot
        return slot
```

---

## 三、Multi-LoRA 训练

### 3.1 训练 vs 推理的差异

| 维度 | Multi-LoRA 推理 | Multi-LoRA 训练 |
|------|----------------|----------------|
| 权重更新 | 不需要 | 需要每个 LoRA 独立更新 |
| 优化器状态 | 不需要 | 每个 LoRA 独立的 Adam 状态 |
| 梯度 | 不需要 | 每个 LoRA 需要独立的梯度 |
| 数据 | 任意混合 | 每个 LoRA 对应特定任务/奖励 |
| 显存 | 权重 only | 权重 + 优化器 + 梯度 + 激活 |
| Base Model | 纯 forward | forward + 需要保存激活 (for backward) |

### 3.2 梯度计算：链式法则下的 LoRA 训练

```
Forward:  h = x W + x (A B) s
                ↑       ↑ ↑
              冻结    可训练

Backward (对 A):
  ∂L/∂A = (x^T · ∂L/∂h) · B^T · s

Backward (对 B):
  ∂L/∂B = A^T · (x^T · ∂L/∂h) · s

关键：x 和 ∂L/∂h 都来自 Base Model 的前向/反向传播，是所有 LoRA 共享的。
```

### 3.3 高效 Multi-LoRA 训练实现

```python
class MultiLoRATrainer:
    """高效多 LoRA 训练器"""

    def __init__(self, base_model, num_loras, r=16, target_modules=None):
        self.base_model = base_model
        for p in self.base_model.parameters():
            p.requires_grad = False  # 冻结 Base

        # 为每个 LoRA 创建独立的参数组和优化器
        self.lora_params = nn.ModuleList([
            LoRAAdapter(base_model, r=r, target_modules=target_modules)
            for _ in range(num_loras)
        ])

        self.optimizers = [
            torch.optim.AdamW(self.lora_params[i].parameters(), lr=3e-4)
            for i in range(num_loras)
        ]

    def train_step_shared_forward(self, batches):
        """
        优化思路：Base Model 的 forward 只做一次，多个 LoRA 复用中间激活。

        限制：只在 LoRA 挂在相同层且输入相同时有效。
        RL 场景下各 LoRA 的 prompt 不同，通常无法复用。
        """
        all_losses = []

        for lora_id, (lora, batch, optimizer) in enumerate(
            zip(self.lora_params, batches, self.optimizers)
        ):
            # 激活当前 LoRA
            self.activate(lora_id)

            # Forward + Loss
            output = self.base_model(batch["input_ids"])
            loss = compute_loss(output, batch["labels"])

            # Backward (只更新当前 LoRA 的梯度)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora.parameters(), 1.0)
            optimizer.step()

            all_losses.append(loss.item())

        return all_losses

    def train_step_batched(self, batches):
        """
        更激进的优化：使用 Batched LoRA Forward
        将所有 LoRA 的数据合并成一个大 batch，
        Base 部分一次 GEMM，LoRA 部分用 segmented GEMM。
        """
        # 合并所有 LoRA 的输入
        combined_input, lora_indices = self.merge_batches(batches)

        # 一次 forward（内部使用 batched LoRA kernel）
        combined_output = self.batched_forward(combined_input, lora_indices)

        # 按 LoRA ID 切分输出，分别计算 loss 和反向传播
        for lora_id in range(len(self.lora_params)):
            mask = lora_indices == lora_id
            output_i = combined_output[mask]
            loss_i = compute_loss(output_i, batches[lora_id]["labels"])

            self.optimizers[lora_id].zero_grad()
            loss_i.backward(retain_graph=(lora_id < len(self.lora_params) - 1))
            self.optimizers[lora_id].step()
```

### 3.4 显存优化技巧

| 技术 | 节省 | 代价 |
|------|------|------|
| **Base Model FP8** | 权重减半 | 精度微降 |
| **Optimizer Offload** | 优化器状态到 CPU | 同步延迟 |
| **梯度检查点** | 激活显存减少 ~70% | 重计算开销 ~30% |
| **LoRA 分批训练** | 同一时刻只激活 K 个 LoRA | 训练总时间增加 |
| **共享 Optimizer** | 用同一个优化器交替更新 | 实现复杂 |

---

## 四、资源调度系统设计

### 4.1 调度问题形式化

给定：
- $N$ 个 LoRA 任务，每个有优先级 $p_i$、显存需求 $m_i$、计算需求 $c_i$
- $G$ 个 GPU，每个有总显存 $M_g$、算力 $C_g$
- Base Model 需要占用 $M_{base}$ 显存（在每个 GPU 上）

目标：最大化 $\sum_i p_i \cdot \text{progress}_i$（加权进度总和）

约束：
- $\forall g: \sum_{i \in \text{GPU}_g} m_i + M_{base} \leq M_g$（显存约束）
- 同一 GPU 上的 LoRA 共享 Base Model 计算（亲和性）

### 4.2 调度算法

```python
from enum import Enum
from heapq import heappush, heappop

class SchedulingPolicy(Enum):
    PRIORITY = "priority"           # 纯优先级调度
    FAIR_SHARE = "fair_share"       # 公平共享
    REWARD_AWARE = "reward_aware"   # 基于 RL reward 动态调整

class MultiLoRAScheduler:
    def __init__(self, gpus, base_mem_mb, policy=SchedulingPolicy.REWARD_AWARE):
        self.gpus = gpus
        self.base_mem_mb = base_mem_mb
        self.policy = policy

    def schedule_round(self, lora_jobs):
        """一轮调度：决定哪些 LoRA 上 GPU，哪些等待"""

        if self.policy == SchedulingPolicy.REWARD_AWARE:
            return self._reward_aware_schedule(lora_jobs)

    def _reward_aware_schedule(self, jobs):
        """
        基于 reward 的调度策略：
        1. reward 上升快的 LoRA 获得更多 GPU 时间
        2. reward 停滞的 LoRA 降低优先级或淘汰
        3. 新 LoRA 获得保护期（explore）
        """
        for job in jobs:
            # 动态优先级 = 基础优先级 × reward 增长率
            reward_slope = job.recent_reward_slope()
            explore_bonus = max(0, 100 - job.total_steps) * 0.01  # 探索奖励
            job.dynamic_priority = job.base_priority * (1 + reward_slope) + explore_bonus

        # 按动态优先级排序
        sorted_jobs = sorted(jobs, key=lambda j: j.dynamic_priority, reverse=True)

        # Bin-packing
        assignments = {gpu.id: [] for gpu in self.gpus}
        gpu_avail_mem = {gpu.id: gpu.total_mem - self.base_mem_mb for gpu in self.gpus}

        scheduled, waiting = [], []
        for job in sorted_jobs:
            placed = False
            # Best-fit decreasing
            best_gpu = None
            best_remaining = float("inf")
            for gpu in self.gpus:
                remaining = gpu_avail_mem[gpu.id] - job.mem_mb
                if remaining >= 0 and remaining < best_remaining:
                    best_gpu = gpu
                    best_remaining = remaining

            if best_gpu:
                assignments[best_gpu.id].append(job)
                gpu_avail_mem[best_gpu.id] -= job.mem_mb
                scheduled.append(job)
            else:
                waiting.append(job)

        return assignments, scheduled, waiting
```

### 4.3 弹性伸缩

```python
class ElasticLoRAManager:
    """根据训练阶段动态调整 Rollout / Train 的 GPU 比例"""

    def __init__(self, total_gpus: int):
        self.total_gpus = total_gpus
        self.rollout_gpus = total_gpus // 2
        self.train_gpus = total_gpus // 2

    def rebalance(self, rollout_queue_depth: int, train_queue_depth: int):
        """
        动态调整:
        - Rollout 队列深 → 多分配推理 GPU
        - Train 队列深 → 多分配训练 GPU
        """
        total = rollout_queue_depth + train_queue_depth + 1
        rollout_ratio = rollout_queue_depth / total

        new_rollout = max(1, int(self.total_gpus * rollout_ratio))
        new_train = self.total_gpus - new_rollout

        if new_rollout != self.rollout_gpus:
            self._migrate_gpus(
                from_pool="train" if new_rollout > self.rollout_gpus else "rollout",
                count=abs(new_rollout - self.rollout_gpus)
            )
            self.rollout_gpus = new_rollout
            self.train_gpus = new_train
```

---

## 五、端到端系统架构参考

```
┌────────────────────────────────────────────────────────────────────────┐
│                        RL Training Controller                         │
│               (任务分发 / 调度 / 监控 / 弹性伸缩)                      │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌─────────────────┐     ┌─────────────────┐     ┌────────────────┐   │
│  │  LoRA Registry   │     │  Reward Service  │     │  Metrics &     │   │
│  │  (权重/配置存储)  │     │  (批量打分)      │     │  Dashboard     │   │
│  └────────┬────────┘     └────────┬────────┘     └────────────────┘   │
│           │                       │                                    │
│  ┌────────▼────────────────────────▼──────────────────────────────┐    │
│  │                     GPU Pool (N × 8 GPUs)                      │    │
│  │                                                                │    │
│  │  ┌─────────────────────┐    ┌─────────────────────┐           │    │
│  │  │   Rollout Workers    │    │   Training Workers   │           │    │
│  │  │   (vLLM / SGLang)   │    │   (Megatron / DS)    │           │    │
│  │  │                     │    │                       │           │    │
│  │  │  Multi-LoRA Serving │◄──►│  Batched LoRA Train  │           │    │
│  │  │  PagedAttention     │    │  Gradient Checkpoint  │           │    │
│  │  │  Prefix Caching     │    │  FP8 Base Model      │           │    │
│  │  │                     │    │                       │           │    │
│  │  │  Active LoRA Pool:  │    │  Active LoRA Pool:    │           │    │
│  │  │  [L1,L3,L7,L12,...] │    │  [L1,L3,L7,L12,...]  │           │    │
│  │  └─────────────────────┘    └─────────────────────┘           │    │
│  │                                                                │    │
│  │  Weight Sync: NCCL / Shared Memory / Async                    │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 六、对标框架实现参考

### 6.1 OpenRLHF 的 Multi-LoRA 路径

```python
# OpenRLHF 通过 Ray 做资源编排
# 每个 LoRA 任务是一个 Ray Actor

@ray.remote(num_gpus=1)
class LoRATrainWorker:
    def __init__(self, base_model_path, lora_config, reward_fn):
        self.model = load_model(base_model_path)
        self.lora = init_lora(self.model, lora_config)
        self.reward_fn = reward_fn

    def train_step(self, prompts):
        responses = self.rollout(prompts)
        rewards = self.reward_fn(responses)
        loss = grpo_loss(responses, rewards)
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item(), "mean_reward": rewards.mean().item()}

# 启动 N 个 LoRA 训练 worker
workers = [LoRATrainWorker.remote(base_path, cfg_i, reward_i)
           for cfg_i, reward_i in zip(lora_configs, reward_fns)]

# 并行训练
futures = [w.train_step.remote(prompts) for w in workers]
results = ray.get(futures)
```

### 6.2 vLLM 的 LoRA 热加载源码路径

```
vllm/lora/
├── models.py              ← LoRAModel 定义
├── layers.py              ← LoRA 层替换逻辑
│   ├── ColumnParallelLinearWithLoRA
│   ├── RowParallelLinearWithLoRA
│   └── VocabParallelEmbeddingWithLoRA
├── worker_manager.py      ← LoRA 权重管理
│   ├── create_lora_manager()
│   ├── add_lora()         ← 热加载新 LoRA
│   ├── remove_lora()      ← 卸载 LoRA
│   └── set_active_loras() ← 设置当前 batch 活跃的 LoRA
└── punica.py              ← Punica/BGMV kernel 调用
    └── bgmv()             ← Batched Gather Matrix-Vector（核心算子）
```

---

## 七、面试关键点速查

| 问题 | 一句话回答 |
|------|----------|
| Multi-LoRA 的核心挑战？ | Base Model 共享计算 + LoRA 增量的高效 batched GEMM |
| S-LoRA 怎么做到千级？ | 预分配 LoRA 权重池 + segmented batched GEMM kernel + LRU 驱逐 |
| RL 训练中 Multi-LoRA 的特殊性？ | 每个 LoRA 需要独立 Rollout + 独立 reward + 独立梯度更新 |
| 调度器的核心目标？ | 最大化 GPU 利用率同时保证公平性和探索性 |
| 权重同步的最优方案？ | 同机用共享显存 (Slime)；跨机用 NCCL broadcast (AReaL) |
| 如何判断一个 LoRA 该淘汰？ | reward 连续 K 步无增长 + 梯度范数趋近于 0 |
