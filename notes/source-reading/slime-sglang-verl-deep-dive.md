# 源码深入理解路线图：Slime + SGLang + verl

> 目标：通过源码阅读建立 LLM RL 训练系统的**完整心智模型**，面试时能讲清楚每一层的设计决策。

---

## 一、总体学习策略

```
                      ┌─────────────────────┐
                      │  L4: 系统设计综合     │  ← 面试系统设计题
                      │  "设计一个支持 100    │
                      │   并发的 RL 训练系统" │
                      ├─────────────────────┤
                      │  L3: 分布式与性能     │  ← FSDP/Megatron + 通信优化
                      │  Slime 异步, verl    │
                      │  HybridFlow          │
                      ├─────────────────────┤
                      │  L2: 核心模块源码     │  ← Rollout/PPO/Reward 实现
                      │  SGLang RadixAttn    │
                      │  verl Single Controller│
                      ├─────────────────────┤
                      │  L1: 架构概览        │  ← Paper + README + 目录结构
                      │  Slime vs verl       │
                      │  基本概念对齐         │
                      └─────────────────────┘
```

**原则**：每个 Level 都有对应的 **nano 项目**作为"学完了"的验证标准。

---

## 二、Level 1 — 架构概览（2 天）

### 2.1 必读材料

| 材料 | 时间 | 产出 |
|------|------|------|
| Slime README + 论文 | 2h | 手绘架构图 |
| verl HybridFlow 论文 (NeurIPS 2024 Workshop) | 2h | 对比表格 |
| SGLang 论文 "Efficiently Programming Large Language Models using SGLang" | 2h | RadixAttention 原理笔记 |
| THUDM/slime `tree -L 2` 目录结构分析 | 1h | 模块依赖图 |
| volcengine/verl `tree -L 2` 目录结构分析 | 1h | 模块依赖图 |

### 2.2 核心理解目标

- [ ] 能画出 Slime 的三大模块交互图（Training / Rollout / Data Buffer）
- [ ] 能画出 verl 的 HybridFlow 时序图（Generation → Training → Generation）
- [ ] 能解释 SGLang RadixAttention 的 Radix Tree 数据结构
- [ ] 能回答：为什么 Slime 选 SGLang？为什么 verl 选 vLLM？

### 2.3 验证 nano 项目

**nano-L1: RL Training Loop Pseudocode**（30 行伪代码）

```python
# 手写一个 RL 训练循环的伪代码框架
class RLTrainer:
    def __init__(self, policy, value_fn, reward_fn, gen_engine):
        self.policy = policy       # Actor model
        self.value_fn = value_fn   # Critic model
        self.reward_fn = reward_fn # Reward function
        self.engine = gen_engine   # SGLang or vLLM

    def train_step(self, prompts):
        # 1. Rollout: 每个 prompt 生成 G 个 response
        responses = []
        for prompt in prompts:
            rs = self.engine.generate(prompt, n=G, temperature=1.0)
            # SGLang: RadixAttention 自动复用 prompt 的 KV Cache
            # vLLM:  需要 APC 或重复 prefill
            responses.append(rs)

        # 2. Reward: 计算每个 response 的奖励
        rewards = self.reward_fn.score(prompts, responses)

        # 3. Advantage: 计算 GAE 或 group-relative advantage
        advantages = compute_advantage(rewards, self.value_fn)

        # 4. PPO Update: 策略梯度 + clipping
        loss = ppo_loss(self.policy, responses, advantages)
        loss.backward()
        self.policy.step()
```

---

## 三、Level 2 — 核心模块源码（1 周）

### 3.1 SGLang RadixAttention 源码走读

**目标文件**: `sglang/srt/managers/schedule_batch.py`, `sglang/srt/mem_cache/radix_cache.py`

```
SGLang 内部 Radix Tree 结构:
┌──────────────────────────────────────────────┐
│  class RadixCache:                            │
│    root: TreeNode                             │
│    evictable_size: int                        │
│                                               │
│  class TreeNode:                              │
│    children: Dict[token_id, TreeNode]         │
│    parent: TreeNode                           │
│    value: KVCacheRef  # 指向物理 KV block     │
│    ref_count: int     # 引用计数              │
│    last_access: float # LRU 时间戳            │
│    lock: bool         # 防止 eviction         │
│                                               │
│  核心操作:                                     │
│  1. match_prefix(tokens) → (node, match_len) │
│     沿树往下匹配，返回最长匹配节点             │
│  2. insert(tokens, kv_cache)                  │
│     不匹配的部分创建新节点                     │
│  3. evict()                                   │
│     LRU evict ref_count=0 的叶子节点          │
│  4. inc_ref(node) / dec_ref(node)             │
│     管理引用计数，防止活跃 prefix 被 evict     │
└──────────────────────────────────────────────┘
```

**关键代码段阅读清单**:

| 文件 | 关键函数 | 理解目标 |
|------|---------|---------|
| `radix_cache.py` | `match_prefix()` | 如何沿 Radix Tree 查找最长匹配 |
| `radix_cache.py` | `insert()` | 新 token 如何分裂已有节点 |
| `radix_cache.py` | `evict()` | LRU eviction 如何在树上执行 |
| `schedule_batch.py` | `add_request()` | 新请求如何触发 prefix match |
| `schedule_batch.py` | `_prefill_batch()` | 如何只 prefill 未匹配的 suffix |

**nano-L2a: Radix Tree Simulator**（~200 行 Python）

```python
# 手写一个简化版 Radix Tree，模拟 RL 场景
class RadixNode:
    def __init__(self):
        self.children = {}
        self.kv_cache = None
        self.ref_count = 0
        self.last_access = 0

class RadixTree:
    def __init__(self, max_nodes):
        self.root = RadixNode()
        self.max_nodes = max_nodes

    def match_prefix(self, tokens):
        """返回最长匹配长度和对应节点"""
        node = self.root
        matched = 0
        for t in tokens:
            if t in node.children:
                node = node.children[t]
                matched += 1
            else:
                break
        return node, matched

    def insert(self, tokens, kv_cache):
        """插入 token 序列和对应 KV Cache"""
        node, matched = self.match_prefix(tokens)
        for t in tokens[matched:]:
            child = RadixNode()
            node.children[t] = child
            node = child
        node.kv_cache = kv_cache

    def simulate_rl_rollout(self, prompt_tokens, G):
        """模拟 RL rollout: 同一 prompt 生成 G 个 response"""
        # 第一次: prefill 全部 prompt
        node, matched = self.match_prefix(prompt_tokens)
        prefill_tokens = len(prompt_tokens) - matched
        self.insert(prompt_tokens, "kv_cache_placeholder")

        stats = {"prefill_tokens": prefill_tokens, "cache_hits": 0}

        # 后续 G-1 次: 全部命中 prefix cache
        for i in range(1, G):
            node, matched = self.match_prefix(prompt_tokens)
            assert matched == len(prompt_tokens)  # 100% cache hit!
            stats["cache_hits"] += 1

        return stats
```

### 3.2 verl Rollout Worker 源码走读

**目标文件**: `verl/workers/rollout/sglang_rollout/sglang_rollout.py`

| 关键类/函数 | 理解目标 |
|------------|---------|
| `SGLangRollout.__init__()` | SGLang engine 初始化，模型加载 |
| `SGLangRollout.generate_sequences()` | 异步生成接口，批量 prompt 处理 |
| `_build_sampling_params()` | temperature/top_p/n 参数构造 |
| `_post_process_outputs()` | 输出解析，log_prob 提取 |

**nano-L2b: Mini Rollout Worker**（~300 行）
- 实现一个简化版 Rollout Worker
- 支持 `generate(prompts, n=G)` 接口
- 内置 prefix cache 统计（命中率、显存节省）

### 3.3 verl PPO Trainer 源码走读

**目标文件**: `verl/trainer/main_ppo.py`, `verl/trainer/ppo/`

| 关键逻辑 | 代码位置 | 理解目标 |
|---------|---------|---------|
| PPO 训练主循环 | `main_ppo.py:training_step()` | Rollout → Reward → Advantage → Update |
| GAE 计算 | `ppo/core.py:compute_gae()` | λ-return, advantage normalization |
| Policy Loss | `ppo/core.py:ppo_loss()` | Clipped surrogate objective |
| KL 散度 | `ppo/core.py:kl_penalty()` | KL divergence 约束 |
| GRPO 变体 | `ppo/grpo.py` | Group-relative advantage |

**nano-L2c: PPO from Scratch**（~500 行 PyTorch）
- 已有 `notebooks/rl_ppo_grpo_implementation.ipynb`，在此基础上扩展
- 增加 GAE 的完整实现（不仅是简化版）
- 增加 KL penalty / dual-clip 等高级特性

---

## 四、Level 3 — 分布式与性能优化（1 周）

### 4.1 Slime 异步架构深入

**核心挑战**：Training 和 Generation 异步执行，如何保证数据一致性？

```
┌─────────────────────────────────────────────────────────┐
│  Slime 异步 RL 数据流                                     │
│                                                           │
│  Generator (SGLang):                                      │
│    policy_v1 → [prompt_1: G responses] → Data Buffer     │
│    policy_v1 → [prompt_2: G responses] → Data Buffer     │
│    policy_v2 → [prompt_3: G responses] → Data Buffer     │
│                                                           │
│  Trainer (Megatron):                                      │
│    从 Data Buffer 取数据 → 可能取到 policy_v1 的数据       │
│    但当前 policy 已是 v3 → off-policy 问题！              │
│                                                           │
│  解决方案:                                                 │
│  1. Importance Sampling: r(θ) = π_θ(a|s) / π_θ_old(a|s) │
│  2. PPO Clip: clip(r(θ), 1-ε, 1+ε) × A                 │
│  3. Staleness Filter: 丢弃 age > K 的 rollout 数据       │
│  4. Priority Queue: 优先使用最新 policy 的 rollout        │
└─────────────────────────────────────────────────────────┘
```

**关键源码**:

| 文件 | 理解目标 |
|------|---------|
| `slime/data_buffer/` | Data Buffer 的 FIFO/Priority Queue 设计 |
| `slime/train_async.py` | 异步循环主入口，Generator-Trainer 协调 |
| `slime/rollout/router.py` | 请求路由，多 SGLang 实例负载均衡 |
| `slime/training/megatron_trainer.py` | Megatron 3D 并行训练循环 |

### 4.2 verl HybridFlow 深入

**核心挑战**：同一组 GPU 交替做 inference 和 training，如何最小化切换开销？

```
┌─────────────────────────────────────────────────────────┐
│  verl HybridFlow 时序图                                   │
│                                                           │
│  Time →                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐│
│  │ Gen Phase │  │Train Phase│  │ Gen Phase │  │Train Phase││
│  │           │  │           │  │           │  │           ││
│  │ Actor:    │→│ Actor:     │→│ Actor:     │→│ Actor:     ││
│  │ vLLM infer│  │ FSDP train│  │ vLLM infer│  │ FSDP train││
│  │           │  │           │  │           │  │           ││
│  │ 模型权重: │  │ 模型权重:  │  │ 模型权重:  │  │ 模型权重:  ││
│  │ vLLM 格式 │→│ FSDP 分片  │→│ vLLM 格式  │→│ FSDP 分片  ││
│  │           │  │           │  │           │  │           ││
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘│
│       ↕              ↕              ↕              ↕      │
│   模型 reshard   模型 reshard   模型 reshard   模型 reshard │
│  (FSDP→vLLM)   (vLLM→FSDP)   (FSDP→vLLM)   (vLLM→FSDP) │
│                                                           │
│  ⚡ 核心优化: 3D-HybridEngine                             │
│  - 减少 reshard 通信量                                     │
│  - 用 NCCL overlap 掩盖 reshard 延迟                      │
│  - 预分配显存避免碎片化                                    │
└─────────────────────────────────────────────────────────┘
```

**关键源码**:

| 文件 | 理解目标 |
|------|---------|
| `verl/workers/sharding_manager/fsdp_sglang.py` | FSDP ↔ SGLang 模型切换 |
| `verl/workers/sharding_manager/megatron_vllm.py` | Megatron ↔ vLLM 模型切换 |
| `verl/single_controller/ray/` | Ray-based worker 编排 |
| `verl/trainer/main_ppo.py:fit()` | HybridFlow 主循环 |

**nano-L3: Async RL Simulator**（~400 行）
- 模拟异步 RL 训练中的 staleness 问题
- 可视化 on-policy vs near-on-policy 的训练曲线差异
- 实验：不同 staleness K 值对收敛速度的影响

---

## 五、Level 4 — 系统设计综合（3 天）

### 5.1 面试系统设计题："设计一个支持 100 并发 RL 训练的系统"

```
需求分析:
- 模型: 7B/70B/MoE 模型
- Prompt: 1024 tokens
- G = 16 (每个 prompt 生成 16 个 response)
- 并发: 100 prompts/batch
- 总生成量: 100 × 16 = 1600 responses/step
- 目标: 最大化训练吞吐，保证 on-policy 质量

架构设计:
┌─────────────────────────────────────────────────────┐
│  RL Training System (100 QPS × G=16)                 │
│                                                       │
│  Layer 1: Request Scheduler                           │
│  ├─ 批量收集 100 prompts                              │
│  ├─ 按 prompt hash 分组（提高 prefix cache 命中率）    │
│  └─ 动态 batch_size 调整                              │
│                                                       │
│  Layer 2: Generation Engine (SGLang × N instances)    │
│  ├─ RadixAttention prefix tree (per instance)         │
│  ├─ 异步生成 1600 responses                           │
│  ├─ 自动 prefix cache 管理                            │
│  └─ 负载均衡: consistent hashing by prompt            │
│                                                       │
│  Layer 3: Reward Engine                               │
│  ├─ Rule-based reward (code execution, math verify)   │
│  ├─ Model-based reward (RM inference)                 │
│  └─ 异步/并行计算                                     │
│                                                       │
│  Layer 4: Training Engine (Megatron/FSDP)             │
│  ├─ PPO/GRPO 梯度计算                                │
│  ├─ 3D 并行 (TP=4, PP=2, DP=8)                       │
│  └─ 参数同步到 Generation Engine                      │
│                                                       │
│  Layer 5: Data Buffer + Orchestration                 │
│  ├─ Priority Queue (按 policy version 排序)           │
│  ├─ Staleness filter (丢弃 age > K 的数据)            │
│  └─ 监控: reward/throughput/KL 仪表盘                 │
└─────────────────────────────────────────────────────┘
```

### 5.2 关键设计决策和 trade-off

| 决策点 | 选项 A | 选项 B | 推荐 |
|-------|--------|--------|------|
| 同步 vs 异步 | verl 同步 | Slime 异步 | >100 GPU 选异步 |
| 推理引擎 | vLLM | SGLang | RL 场景选 SGLang |
| 训练引擎 | FSDP | Megatron | >70B 选 Megatron |
| RL 算法 | PPO | GRPO | GRPO 更简单高效 |
| Prefix Cache | Hash-based | Radix Tree | RL 选 Radix Tree |

### 5.3 显存估算

```
7B 模型, FP16, seq_len=2048, G=16, batch=100:

模型显存:
  7B × 2 bytes = 14 GB (per replica)
  FSDP 4-way: 14/4 = 3.5 GB/GPU

KV Cache 显存 (无 prefix cache):
  batch × G × seq × layers × heads × dim × 2(K+V) × 2(bytes)
  = 100 × 16 × 2048 × 32 × 32 × 128 × 2 × 2
  = ~800 GB  ← 不可行！

KV Cache 显存 (有 RadixAttention):
  prefix 部分: 100 × 1 × 1024 × ... = ~25 GB (共享)
  decode 部分: 100 × 16 × 1024 × ... = ~400 GB
  总计: ~425 GB  ← 节省 ~47%!

→ 这就是为什么 Slime 选 SGLang 的 RadixAttention
```

---

## 六、阅读进度追踪

### Slime 源码阅读清单

- [ ] `README.md` — 架构概览
- [ ] `slime/train_async.py` — 异步训练主入口
- [ ] `slime/data_buffer/` — Data Buffer 设计
- [ ] `slime/rollout/` — SGLang Rollout 逻辑
- [ ] `slime/training/` — Megatron 训练循环
- [ ] `slime_plugins/` — 插件系统
- [ ] `configs/` — 配置文件示例

### verl 源码阅读清单

- [ ] `README.md` — HybridFlow 概览
- [ ] `verl/trainer/main_ppo.py` — PPO 训练主循环
- [ ] `verl/single_controller/` — 单控制器编排
- [ ] `verl/workers/rollout/sglang_rollout/` — SGLang Rollout Worker
- [ ] `verl/workers/sharding_manager/` — 模型切换管理
- [ ] `verl/workers/actor/` — Actor Worker
- [ ] `verl/workers/critic/` — Critic Worker
- [ ] `verl/trainer/ppo/core.py` — PPO 核心算法

### SGLang 源码阅读清单

- [ ] `sglang/srt/mem_cache/radix_cache.py` — RadixAttention 核心
- [ ] `sglang/srt/managers/schedule_batch.py` — 调度器
- [ ] `sglang/srt/model_executor/` — 模型推理执行
- [ ] `sglang/srt/server.py` — HTTP API 服务

---

## 七、学习时间规划

| 周次 | Level | 产出 | 验证标准 |
|------|-------|------|---------|
| Week 1 | L1 架构 + L2 模块 | 3 篇源码笔记 + nano-L1 伪代码 | 能画完整架构图 |
| Week 2 | L2 深入 + L3 开始 | nano-L2a/b/c 代码 | 能讲清 RadixAttention 原理 |
| Week 3 | L3 分布式 | nano-L3 异步模拟 | 能分析 staleness 影响 |
| Week 4 | L4 系统设计 | 完整系统设计文档 | 能在 45 min 内讲完整系统 |
