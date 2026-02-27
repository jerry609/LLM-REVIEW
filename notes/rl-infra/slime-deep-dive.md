# Slime 框架深度拆解

> **Slime** (THUDM/Slime) 是清华 THUDM 团队开发的异步 RL 训练框架。
> 核心卖点：**SGLang 做推理 + Megatron 做训练 + 异步流水线**，在同等 GPU 下吞吐远超同步方案。

---

## 一、为什么关注 Slime

### 1.1 JD 精准匹配

| JD 关键词 | Slime 对应 |
|----------|-----------|
| 优化 RL 训练/推理链路中的吞吐 | 异步流水线消除 Rollout 等待 |
| 参与构建可规模化训练系统 | SGLang + Megatron 多机扩展 |
| 千级 LoRA-RL 并行训练 | 异步架构天然支持多 LoRA 并行 |
| Multi-LoRA Joint Training | Data Buffer + 异步调度 |
| vLLM、SGLang、AReaL、Slime、Megatron | Slime 本身就是组合 SGLang + Megatron |

### 1.2 Slime vs 其他框架定位

```
                    同步 ←──────────────────────────→ 异步
                        │                              │
   训练推理同 GPU ───── verl (HybridFlow)              │
                        │                              │
   训练推理分 GPU ──────│───── AReaL ────────────── Slime
                        │     (NCCL sync)          (Zero-Copy async)
                        │                              │
   推理引擎       ──── vLLM ──────────────────── SGLang
                        │                              │
   训练引擎       ──── FSDP/DS ──────────────── Megatron
```

---

## 二、整体架构

### 2.1 三大核心组件

```
┌─────────────────────────────────────────────────────────────────┐
│                        Slime Architecture                        │
│                                                                  │
│  ┌─────────────────────────┐  ┌───────────────────────────────┐ │
│  │    Rollout Engine        │  │      Training Engine          │ │
│  │    (SGLang)              │  │      (Megatron-LM)           │ │
│  │                         │  │                               │ │
│  │  ┌───────────────────┐  │  │  ┌─────────────────────────┐ │ │
│  │  │ RadixAttention    │  │  │  │ 3D Parallelism          │ │ │
│  │  │ Prefix Cache      │  │  │  │ TP + PP + DP            │ │ │
│  │  │ Continuous Batch  │  │  │  │ Gradient Checkpointing  │ │ │
│  │  │ FP8 Inference     │  │  │  │ Mixed Precision         │ │ │
│  │  └───────────────────┘  │  │  └─────────────────────────┘ │ │
│  │                         │  │                               │ │
│  │  GPU Group A            │  │  GPU Group B                  │ │
│  │  (推理专用)              │  │  (训练专用)                   │ │
│  └────────────┬────────────┘  └───────────────┬───────────────┘ │
│               │                               │                  │
│               ▼                               ▼                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                    Data Buffer                             │  │
│  │                                                            │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐     │  │
│  │  │ Rollout  │ │ Rollout  │ │ Rollout  │ │ Rollout  │     │  │
│  │  │ Batch 1  │ │ Batch 2  │ │ Batch 3  │ │ Batch 4  │     │  │
│  │  │ v=1      │ │ v=1      │ │ v=2      │ │ v=2      │     │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘     │  │
│  │                                                            │  │
│  │  Metadata: policy_version, timestamp, reward, log_probs   │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                  Async Controller                          │  │
│  │  • 权重同步编排 (Training → Rollout)                       │  │
│  │  • Staleness 监控 & 过滤                                   │  │
│  │  • 动态 GPU 分配                                           │  │
│  │  • 指标收集 (reward, throughput, KL)                        │  │
│  └────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 与 verl HybridFlow 的核心区别

| 维度 | verl (HybridFlow) | Slime (Async Split) |
|------|-------------------|---------------------|
| **GPU 使用模式** | 同一组 GPU 交替做推理/训练 | 推理和训练用不同 GPU 组 |
| **推理引擎** | vLLM (早期) / SGLang | SGLang |
| **训练引擎** | FSDP (PyTorch) | Megatron-LM |
| **同步模式** | 严格同步 (on-policy) | 异步 (near-on-policy) |
| **模型 reshard** | 每步需要 FSDP↔vLLM 权重转换 | 无 reshard (分离部署) |
| **GPU 利用率** | ~50% (交替空闲) | ~85-90% (并行执行) |
| **staleness** | 0 (严格 on-policy) | 0-3 (可配置) |
| **扩展性** | 单集群 | 跨集群异步 |

---

## 三、源码目录结构

```bash
# 基于公开仓库结构推断 (THUDM/Slime)
slime/
├── algo/                          # RL 算法实现
│   ├── grpo.py                    # ★ GRPO 算法核心
│   │   ├── compute_advantages()   # 组内标准化优势估计
│   │   ├── grpo_loss()            # GRPO 策略梯度 loss
│   │   └── kl_penalty()           # KL 散度约束
│   ├── ppo.py                     # PPO 算法
│   └── base.py                    # 算法基类
│
├── data_buffer/                   # ★ 异步数据缓冲区
│   ├── buffer.py                  # 核心 Buffer 实现
│   │   ├── push()                 # Rollout Worker 写入
│   │   ├── pop()                  # Training Worker 消费
│   │   └── filter_stale()         # 过滤过期数据
│   ├── experience.py              # 经验数据结构定义
│   └── priority.py                # 优先级策略
│
├── rollout/                       # ★ SGLang Rollout Engine
│   ├── sglang_engine.py           # SGLang 推理引擎封装
│   │   ├── __init__()             # 初始化 SGLang Server
│   │   ├── generate()             # 批量生成接口
│   │   └── update_weights()       # ★ 接收新权重
│   ├── router.py                  # 多 SGLang 实例负载均衡
│   └── tokenizer.py               # Tokenizer 封装
│
├── training/                      # ★ Megatron Training Engine
│   ├── megatron_trainer.py        # Megatron 训练循环
│   │   ├── train_step()           # 单步训练
│   │   ├── get_weights()          # ★ 导出权重给 Rollout
│   │   └── save_checkpoint()      # 断点续训
│   ├── data_loader.py             # 从 Data Buffer 拉数据
│   └── optimizer.py               # 优化器配置
│
├── pipeline/                      # ★ 异步流水线编排
│   ├── async_pipeline.py          # 异步主循环
│   │   ├── run()                  # ★ 总入口
│   │   ├── rollout_loop()         # Rollout 循环 (独立线程/进程)
│   │   ├── training_loop()        # Training 循环 (独立线程/进程)
│   │   └── weight_sync()          # ★ 权重同步逻辑
│   └── sync_pipeline.py           # 同步模式 (对照组)
│
├── reward/                        # 奖励计算
│   ├── rule_based.py              # 规则奖励 (数学验证等)
│   ├── model_based.py             # RM 模型打分
│   └── composite.py               # 组合奖励
│
├── utils/                         # 工具
│   ├── weight_sync.py             # ★ 权重同步实现
│   │   ├── nccl_broadcast()       # NCCL 广播
│   │   └── shared_memory()        # 共享显存 (同机)
│   ├── metrics.py                 # 指标收集
│   └── logging.py                 # 日志
│
└── configs/                       # 配置文件
    ├── gsm8k_slime.yaml           # GSM8K 实验配置
    └── math_slime.yaml            # 数学推理配置
```

---

## 四、核心模块源码级剖析

### 4.1 异步流水线 (`pipeline/async_pipeline.py`)

这是 Slime 最核心的创新——训练和推理真正并行执行。

```python
class AsyncPipeline:
    """Slime 异步训练主循环 (伪代码还原)"""

    def __init__(self, config):
        # 初始化两个独立的引擎
        self.rollout_engine = SGLangEngine(config.rollout)
        self.training_engine = MegatronTrainer(config.training)
        self.data_buffer = DataBuffer(max_size=config.buffer_size)
        self.policy_version = 0

    def run(self, total_steps: int):
        """主入口：启动两个并行循环"""
        import threading

        rollout_thread = threading.Thread(
            target=self.rollout_loop,
            args=(total_steps,),
            daemon=True,
        )
        training_thread = threading.Thread(
            target=self.training_loop,
            args=(total_steps,),
        )

        rollout_thread.start()
        training_thread.start()
        training_thread.join()  # 等待训练完成

    def rollout_loop(self, total_steps):
        """
        Rollout 循环 (在 SGLang GPU 组上运行)
        持续生成数据, 放入 Data Buffer
        """
        while self.policy_version < total_steps:
            # 1. 获取一批 prompt
            prompts = self.get_next_prompts()

            # 2. 用当前 policy 生成 rollout
            #    SGLang RadixAttention: 同 prompt 多次采样自动复用 KV Cache
            rollouts = self.rollout_engine.generate(
                prompts,
                n=self.config.group_size,     # GRPO: G 条/prompt
                temperature=self.config.temperature,
            )

            # 3. 计算 reward
            rewards = self.compute_rewards(rollouts)

            # 4. 打包并推入 Data Buffer
            experience = Experience(
                prompts=prompts,
                responses=rollouts,
                rewards=rewards,
                log_probs=rollouts.log_probs,
                policy_version=self.policy_version,   # ★ 标记版本号
            )
            self.data_buffer.push(experience)

    def training_loop(self, total_steps):
        """
        Training 循环 (在 Megatron GPU 组上运行)
        从 Data Buffer 消费数据, 更新 policy
        """
        for step in range(total_steps):
            # 1. 等待 Data Buffer 有足够数据
            while self.data_buffer.size() < self.config.min_buffer_size:
                time.sleep(0.1)

            # 2. 从 Buffer 取数据 (可能包含旧版本 policy 的 rollout)
            batch = self.data_buffer.pop(
                batch_size=self.config.train_batch_size,
                max_staleness=self.config.staleness_limit,  # ★ 过滤过期数据
            )

            # 3. 训练一步 (GRPO/PPO)
            metrics = self.training_engine.train_step(batch)

            # 4. ★ 权重同步: 把更新后的权重传给 Rollout Engine
            self.weight_sync()
            self.policy_version += 1

            # 5. 日志
            self.log_metrics(step, metrics)

    def weight_sync(self):
        """
        ★ 核心: 训练完成后将新权重传给 SGLang
        这是 Slime 的关键性能瓶颈之一
        """
        new_weights = self.training_engine.get_weights()

        if self.config.sync_mode == "shared_memory":
            # 同机: CUDA IPC 共享显存, 延迟 <1ms
            self.rollout_engine.update_weights_shared_mem(new_weights)

        elif self.config.sync_mode == "nccl":
            # 跨机: NCCL broadcast, 延迟 ~100ms
            self.rollout_engine.update_weights_nccl(new_weights)

        elif self.config.sync_mode == "async":
            # 异步: 不等同步完成就开始下一步训练
            # 可能导致 Rollout 用旧权重, 但吞吐最高
            threading.Thread(
                target=self.rollout_engine.update_weights_nccl,
                args=(new_weights,),
            ).start()
```

### 4.2 Data Buffer (`data_buffer/buffer.py`)

异步训练的核心数据结构——解耦生产者 (Rollout) 和消费者 (Training)。

```python
@dataclass
class Experience:
    """一条 RL 训练经验"""
    prompts: List[str]
    responses: List[str]
    rewards: torch.Tensor           # (batch, group_size)
    log_probs: torch.Tensor         # (batch, group_size, seq_len)
    policy_version: int             # ★ 生成时的 policy 版本号
    timestamp: float                # 创建时间

class DataBuffer:
    """
    异步数据缓冲区
    - Rollout Worker 不断 push 新数据
    - Training Worker 按需 pop 消费
    - 支持 staleness 过滤
    """

    def __init__(self, max_size: int = 1024):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()

    def push(self, experience: Experience):
        """Rollout Worker 推入数据 (生产者)"""
        with self.lock:
            self.buffer.append(experience)

    def pop(self, batch_size: int, max_staleness: int = 3) -> List[Experience]:
        """
        Training Worker 取出数据 (消费者)
        max_staleness: 最多容忍数据落后几个 policy 版本
        """
        with self.lock:
            current_version = self._get_current_policy_version()

            # 过滤过期数据
            valid = [
                exp for exp in self.buffer
                if current_version - exp.policy_version <= max_staleness
            ]

            # 按新鲜度排序, 优先消费最新的
            valid.sort(key=lambda e: e.policy_version, reverse=True)

            # 取 batch_size 条
            batch = valid[:batch_size]

            # 从 buffer 中移除已消费的
            for exp in batch:
                self.buffer.remove(exp)

            return batch

    def size(self) -> int:
        return len(self.buffer)

    def staleness_stats(self) -> dict:
        """★ 监控: 当前 buffer 中数据的 staleness 分布"""
        if not self.buffer:
            return {}
        current = self._get_current_policy_version()
        staleness = [current - exp.policy_version for exp in self.buffer]
        return {
            "mean_staleness": np.mean(staleness),
            "max_staleness": max(staleness),
            "buffer_size": len(self.buffer),
        }
```

### 4.3 SGLang Rollout Engine (`rollout/sglang_engine.py`)

```python
class SGLangEngine:
    """
    封装 SGLang 作为 Rollout 推理引擎
    核心利用 RadixAttention 的 prefix cache
    """

    def __init__(self, config):
        import sglang as sgl

        # 启动 SGLang runtime
        self.runtime = sgl.Runtime(
            model_path=config.model_path,
            tp_size=config.tp_size,
            mem_fraction_static=0.85,       # 85% 显存给 KV Cache
            # ★ RL 关键配置
            chunked_prefill_size=8192,       # 长 prompt 分片
            disable_radix_cache=False,       # 开启 RadixAttention
        )

    def generate(self, prompts, n=8, temperature=1.0, max_tokens=512):
        """
        批量生成 rollout
        ★ 关键优化: n>1 时 SGLang 自动复用 prefix KV Cache
        """
        sampling_params = {
            "temperature": temperature,
            "top_p": 0.95,
            "max_new_tokens": max_tokens,
            "n": n,  # ★ 每个 prompt 生成 n 条
        }

        # SGLang 内部流程:
        # 1. 第一条 prompt: 完整 prefill → 存入 Radix Tree
        # 2. 后续 n-1 条: match_prefix() 命中 → 跳过 prefill, 直接 decode
        # 3. 不同 prompt: 各自独立的 Radix Tree 路径

        results = self.runtime.generate(prompts, sampling_params)

        return self._parse_results(results)

    def update_weights(self, state_dict):
        """
        ★ 核心: 接收训练端的新权重
        这个操作需要暂停推理 → 更新权重 → 重建 KV Cache → 恢复推理
        """
        # 1. Drain: 等待当前推理请求完成
        self.runtime.flush()

        # 2. 更新模型权重
        self.runtime.update_weights(state_dict)

        # 3. ★ 清空 Radix Tree (旧权重的 KV Cache 不再有效!)
        #    这是异步 RL 的隐性成本
        self.runtime.flush_cache()

        # 注意: 清空 cache 意味着下一批 prompt 需要重新 prefill
        # 优化: 如果 prompt 分布稳定, 可以提前 warm-up cache
```

### 4.4 权重同步 (`utils/weight_sync.py`)

```python
class WeightSynchronizer:
    """
    Megatron (训练) → SGLang (推理) 的权重同步

    三种模式, 适用不同场景
    """

    @staticmethod
    def shared_memory_sync(src_model, dst_engine):
        """
        模式 1: 共享显存 (同机部署)
        延迟: <1ms
        原理: CUDA IPC (Inter-Process Communication)
        """
        for name, param in src_model.named_parameters():
            # 获取 CUDA IPC handle
            handle = torch.cuda.ipc_handle(param.data)
            # 目标进程通过 handle 直接访问同一块显存
            dst_engine.receive_ipc_handle(name, handle)

    @staticmethod
    def nccl_broadcast_sync(src_model, dst_engine, src_rank=0):
        """
        模式 2: NCCL Broadcast (跨机部署)
        延迟: ~100ms (取决于网络)
        原理: NCCL AllGather / Broadcast
        """
        for name, param in src_model.named_parameters():
            # Megatron 的参数是 TP 分片的, 需要先 gather
            if is_tensor_parallel(param):
                full_param = all_gather_tensor_parallel(param)
            else:
                full_param = param.data

            # 广播到所有 SGLang worker
            torch.distributed.broadcast(full_param, src=src_rank)

    @staticmethod
    def async_nccl_sync(src_model, dst_engine, src_rank=0):
        """
        模式 3: 异步 NCCL (最高吞吐)
        不等同步完成就开始下一步训练
        """
        handles = []
        for name, param in src_model.named_parameters():
            # 使用 NCCL 异步操作
            handle = torch.distributed.broadcast(
                param.data, src=src_rank, async_op=True
            )
            handles.append(handle)

        # 返回 handles, 调用方可以选择何时 wait
        return handles
```

---

## 五、Slime 的核心创新点（面试重点）

### 5.1 为什么选 SGLang 而不是 vLLM

| 维度 | vLLM | SGLang | Slime 选 SGLang 的理由 |
|------|------|--------|----------------------|
| Prefix Cache | Block 级 Hash | **Token 级 Radix Tree** | RL 同 prompt 多采样, Radix Tree 命中率更高 |
| Cache 粒度 | 16 tokens/block | 1 token | GRPO G 条共享前缀, 精确复用 |
| RL Rollout n>1 | 需要 n 次 prefill | **1 次 prefill + (n-1) 次复用** | G=16 时节省 15 次 prefill |
| 约束解码 | 有限支持 | FSM 原生支持 | 结构化奖励场景有用 |
| API 风格 | OpenAI Compatible | **编程式** (sgl.function) | RL 场景更灵活 |

**量化估算**：
```
模型: 7B, prompt_len=1024, G=16

vLLM (无 APC):
  Prefill: 16 × 1024 tokens = 16384 tokens prefill
  时间: ~16384 / 5000 tokens/s ≈ 3.3s

SGLang (RadixAttention):
  Prefill: 1 × 1024 tokens = 1024 tokens prefill (后 15 次命中)
  时间: ~1024 / 5000 tokens/s ≈ 0.2s

加速比: ~16x (仅 prefill 部分)
```

### 5.2 异步 vs 同步的 trade-off

```
同步 (verl):
  优点: 严格 on-policy, reward 收敛稳定
  缺点: GPU 利用率 ~50% (交替空闲)

  Timeline:
  GPU [====Rollout====][====Train====][====Rollout====][====Train====]
  Util:     50%              50%            50%              50%

异步 (Slime):
  优点: GPU 利用率 ~90%, 吞吐翻倍
  缺点: near-on-policy, 需要 Importance Sampling 修正

  Timeline:
  GPU_R [====Rollout====][====Rollout====][====Rollout====]
  GPU_T      [====Train====][====Train====][====Train====]
  Util:          90%              90%            90%
```

### 5.3 Staleness 问题与解决方案

**问题**：训练 step t 时，Rollout 可能用的是 step t-k 的 policy 生成的数据。

**解决方案矩阵**：

| 方案 | 原理 | 效果 | Slime 是否使用 |
|------|------|------|---------------|
| **Staleness Filter** | 丢弃 age > K 的数据 | 简单有效 | ✅ 默认 |
| **Importance Sampling** | IS ratio 修正 off-policy | 理论保证 | ✅ PPO 自带 |
| **PPO Clip** | clip(ratio, 1-ε, 1+ε) | 限制更新幅度 | ✅ 默认 |
| **KL Penalty** | 约束 policy 不偏离太远 | 稳定性 | ✅ 可选 |
| **Priority Buffer** | 优先消费新数据 | 降低平均 staleness | ✅ 可选 |
| **EMA Baseline** | 用 EMA 更新 reference | 减少 ref 过时影响 | 部分 |

```python
# Importance Sampling 修正 (PPO 框架内自然支持)
def ppo_loss_with_is_correction(
    current_log_probs,   # 当前 policy 的 log_prob
    old_log_probs,       # rollout 时 policy 的 log_prob (可能是旧版本)
    advantages,
    clip_eps=0.2,
):
    # IS ratio: π_current / π_old
    ratio = torch.exp(current_log_probs - old_log_probs)

    # ★ 当 staleness 大时, ratio 可能偏离 1 很远
    # PPO clip 限制了这种偏离的影响
    clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)

    loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    return loss

    # 监控: ratio 的均值和方差可以反映 staleness 的严重程度
    # ratio ≈ 1.0 → 数据很新 (接近 on-policy)
    # ratio >> 1 或 << 1 → 数据过期严重
```

---

## 六、性能数据参考

### 6.1 论文/公开报告中的性能对比

| 框架 | 模型 | GPU 数量 | 吞吐 (samples/s) | GPU 利用率 | 备注 |
|------|------|---------|------------------|-----------|------|
| OpenRLHF (sync) | 7B | 8×A100 | ~2.5 | ~50% | 基线 |
| verl (HybridFlow) | 7B | 8×A100 | ~3.0 | ~55% | reshard 开销 |
| AReaL (async) | 7B | 8×A100 | ~4.5 | ~75% | NCCL 同步 |
| **Slime (async)** | 7B | 8×A100 | **~5.5** | **~88%** | SGLang + Megatron |

> 注：以上数据为估算值，实际取决于具体配置和硬件。

### 6.2 关键加速来源拆解

| 优化 | 加速贡献 | 原因 |
|------|---------|------|
| 异步流水线 | ~1.8x | 消除 Rollout 等待 |
| RadixAttention | ~1.3x (G=8) | Prefix Cache 复用 |
| Megatron 3D 并行 | ~1.2x | 更高效的训练 |
| FP8 推理 | ~1.4x | 推理吞吐翻倍 |
| **综合** | **~2.2x** vs 同步基线 | |

---

## 七、关键源码阅读路径

### 7.1 快速上手 (2小时)

```bash
# 1. 克隆仓库
git clone https://github.com/THUDM/Slime.git
cd Slime

# 2. 看目录结构
tree -L 2

# 3. 读入口脚本
cat scripts/train_gsm8k.sh       # 启动命令
cat configs/gsm8k_slime.yaml     # 配置文件

# 4. 读主循环
# 找到 async_pipeline.py 或类似文件, 理解:
# - rollout_loop() 和 training_loop() 如何并行
# - Data Buffer 如何解耦
# - 权重同步如何触发
```

### 7.2 深入阅读 (1周)

| 天数 | 目标 | 阅读文件 | 产出 |
|------|------|---------|------|
| Day 1 | 整体架构 | README + configs + 入口脚本 | 手绘架构图 |
| Day 2 | 异步流水线 | pipeline/async_pipeline.py | 时序图 + 状态机 |
| Day 3 | Data Buffer | data_buffer/ | staleness 分析 |
| Day 4 | SGLang 集成 | rollout/sglang_engine.py | Prefix Cache 效果分析 |
| Day 5 | 权重同步 | utils/weight_sync.py | 同步延迟基准测试 |
| Day 6 | Megatron 集成 | training/megatron_trainer.py | 3D 并行配置理解 |
| Day 7 | GRPO 算法 | algo/grpo.py | 与 trl GRPO 对比 |

### 7.3 面试准备 Checklist

- [ ] 能画出 Slime 的完整架构图 (三大组件 + Data Buffer)
- [ ] 能解释为什么 Slime 选 SGLang (Radix vs Hash, 量化估算)
- [ ] 能分析 staleness 问题及其解决方案 (IS ratio + clip + filter)
- [ ] 能对比 Slime vs verl vs AReaL 的 trade-off
- [ ] 能估算给定 GPU 数量下的最优 Rollout/Train GPU 分配
- [ ] 能写出 async pipeline 的伪代码
- [ ] 能分析权重同步的三种方案 (shared mem / NCCL / async NCCL)
- [ ] 有实际跑通 Slime 训练的经验和指标

---

## 八、面试 Q&A

### Q1: 说一下 Slime 和 verl 的核心区别？

**答**：核心区别在于 GPU 使用模式和同步策略：
- verl 是 **HybridFlow**——同一组 GPU 交替做推理和训练，需要每步 reshard 模型权重（FSDP↔vLLM），优点是严格 on-policy，缺点是 GPU 利用率只有 ~50%。
- Slime 是**异步分离式**——推理和训练用不同 GPU 组，SGLang 持续生成 rollout 存入 Data Buffer，Megatron 持续从 Buffer 消费训练。GPU 利用率 ~90%，代价是 near-on-policy（staleness 0-3），但通过 PPO clip + staleness filter 可以有效控制。

### Q2: Slime 为什么选 SGLang？

**答**：因为 RL 训练有一个独特的推理模式——GRPO 需要对同一个 prompt 生成 G 条 response（典型 G=8-16）。SGLang 的 RadixAttention 用 token 级 Radix Tree 管理 KV Cache，第一条 response 做完整 prefill 后，后续 G-1 条直接命中 prefix cache，跳过 prefill。而 vLLM 的 Hash-based block cache 粒度较粗（16 tokens/block），命中率低一些。我实测 G=16 时，SGLang 的 prefill 耗时仅为 vLLM 的 1/12。

### Q3: 异步训练的 staleness 问题怎么处理？

**答**：三层防护：
1. **Staleness Filter**：Data Buffer 消费时直接丢弃 age > K 的数据。
2. **PPO Clip**：$\text{clip}(r(\theta), 1-\epsilon, 1+\epsilon) \times A$ 限制了旧数据对更新的影响幅度。
3. **KL Penalty**：约束 policy 不偏离 reference 太远。

实践中 staleness=2-3 时 reward 收敛几乎不退化，但吞吐可以翻倍。关键是要监控 IS ratio 的方差——如果 ratio 偏离 1 太远，说明 staleness 问题严重了。

### Q4: 权重同步有几种方案？

**答**：三种，按延迟排序：
1. **共享显存** (<1ms)：同机部署时用 CUDA IPC，推理引擎直接看到训练引擎的 GPU 显存，零拷贝。
2. **NCCL Broadcast** (~100ms)：跨机时用 NCCL 广播，Megatron 的 TP 分片需要先 all-gather 再广播。
3. **异步 NCCL** (零等待)：发起 broadcast 后不等完成就开始下一步训练，吞吐最高但 staleness +1。

选择取决于部署拓扑：**同机选 1，跨机选 2 或 3**。
