# Slime vs verl —— LLM RL 训练框架对比

> 面试高频话题：2025-2026 最热的 post-training 基础设施

---

## 一、Slime（清华 THUDM）

- **GitHub**: [github.com/THUDM/slime](https://github.com/THUDM/slime)
- **License**: Apache-2.0 | 4.2k ⭐ | 544 forks | 119 contributors
- **最新版本**: v0.2.2 (2026.01.18)
- **定位**: LLM post-training framework for **RL Scaling**
- **背后模型**: **GLM-4.5 / GLM-4.6 / GLM-4.7**（智谱 AI Z.ai 系列）

### 1.1 核心架构

```
┌─────────────────────────────────────────────────┐
│                slime Architecture                 │
│                                                   │
│  ┌───────────┐   ┌─────────────┐   ┌───────────┐│
│  │  Training  │   │   Rollout    │   │   Data    ││
│  │ (Megatron) │◄─►│(SGLang+Router)│◄─►│  Buffer   ││
│  │            │   │             │   │           ││
│  └───────────┘   └─────────────┘   └───────────┘│
│       ▲                ▲                          │
│  梯度更新 +        异步数据生成 +                    │
│  参数同步          奖励/验证输出                     │
└─────────────────────────────────────────────────┘
```

三大模块：
1. **Training (Megatron-LM)**: 读取 Data Buffer 中的数据做梯度更新，训练后同步参数到 Rollout
2. **Rollout (SGLang + Router)**: 用 SGLang 做高速生成，产出 new data（包含 reward/verifier 输出）
3. **Data Buffer**: 桥接模块，管理 prompt 初始化、自定义数据、rollout 生成方法

### 1.2 关键特性

| 特性 | 说明 |
|------|------|
| **异步训练** | `train_async.py` 原生支持，Generation ↔ Training 解耦并行 |
| **SGLang 生成** | RadixAttention 对 RL 场景（大量相同 prompt）的 prefix caching 更友好 |
| **Megatron 训练** | 生产级 3D 并行 (TP/PP/DP)，支持超大 MoE 模型 |
| **插件系统** | `slime_plugins/` 可扩展自定义 RL 算法和数据生成流程 |
| **多模型支持** | Qwen3 系列、DeepSeek V3/R1、Llama 3 |

### 1.3 基于 Slime 构建的项目

| 项目 | 说明 |
|------|------|
| **P1** | 物理奥赛推理模型，多阶段 RL + 自适应难度调整 |
| **RLVE** | 400 个可验证环境联合训练，动态调整难度分布 |
| **TritonForge** | SFT + RL 训练 LLM 生成优化 GPU Kernel |
| **APRIL** | 加速 rollout（over-provisioning + 主动管理 partial rollout） |
| **qqr** | ArenaRL 算法 + MCP 协议，开放式 Agent 进化 |

---

## 二、verl（字节跳动/火山引擎）

- **GitHub**: github.com/volcengine/verl
- **定位**: Flexible and efficient RL training framework for LLMs
- **背后**: 字节跳动内部 LLM 后训练
- **论文**: *"VERL: An Extensible and Efficient RL Training Framework for LLMs"* (NeurIPS 2024 Workshop)

### 2.1 核心架构 — HybridFlow

```
┌─────────────────────────────────────────┐
│            verl HybridFlow               │
│                                          │
│  ┌──────────────────────────────────┐   │
│  │        Single Controller          │   │
│  │  (Ray-based orchestration)        │   │
│  └──────────┬───────────────────────┘   │
│             │                            │
│  ┌──────────▼───────────────────────┐   │
│  │   同一组 GPU 交替执行:             │   │
│  │                                    │   │
│  │  Generation Phase    Training Phase │  │
│  │  (vLLM inference) ←→ (FSDP/Megatron)│ │
│  │                                    │   │
│  └──────────────────────────────────┘   │
│                                          │
│  Actor / Critic / Reference / Reward     │
│  Workers 在同一组 GPU 上角色切换          │
└─────────────────────────────────────────┘
```

### 2.2 关键特性

| 特性 | 说明 |
|------|------|
| **同步训练** | HybridFlow: generation → training 交替，保证 on-policy |
| **GPU 复用** | 同一组 GPU 上做 inference 和 training，无需额外资源 |
| **vLLM 生成** | 利用 vLLM 的 PagedAttention 做高速采样 |
| **多 RL 算法** | PPO / GRPO / REINFORCE++ |
| **弹性调度** | 基于 Ray 的弹性资源管理 |

---

## 三、核心对比

| 维度 | **Slime** (THUDM/智谱) | **verl** (字节/火山引擎) |
|------|------------------------|------------------------|
| **训练模式** | **异步** (Generation ↔ Training 解耦) | **同步** (HybridFlow 角色切换) |
| **训练引擎** | Megatron-LM | FSDP / Megatron |
| **生成引擎** | **SGLang** (RadixAttention) | **vLLM** (PagedAttention) |
| **GPU 使用** | 分离式：不同 GPU 池做 gen/train | 复用式：同一 GPU 切换角色 |
| **GPU 利用率** | ✅ 更高（异步无等待） | 角色切换有间隙 |
| **数据新鲜度** | Near-on-policy（有滞后，需修正） | ✅ On-policy（每次最新 policy） |
| **训练稳定性** | 需要处理 stale data | ✅ 天然稳定 |
| **MoE 支持** | ✅ DeepSeek V3 / Qwen3MoE | ✅ |
| **RL 算法** | PPO / GRPO / 自定义 | PPO / GRPO / REINFORCE++ |
| **资源需求** | 需要更多 GPU（gen/train 分离） | 更少 GPU（复用同一组） |
| **Prefix Cache** | ✅ SGLang RadixAttention 更优 | vLLM prefix cache |
| **背后模型** | GLM-4.5/4.6/4.7 | 字节内部 |

### 为什么 Slime 选 SGLang 而不是 vLLM？

RL 训练中 rollout 阶段有大量**相同 prompt**（同一 prompt 生成 G 个 response），
SGLang 的 RadixAttention（Radix Tree 管理 prefix）天然适合这种场景，
prefix cache 复用率远高于 vLLM 的 hash-based 方案。

### 异步 vs 同步的核心 trade-off

```
同步 (verl):
  generate(policy_v1) → train → policy_v2 → generate(policy_v2) → train → ...
  ✅ 数据永远 on-policy
  ❌ generate 和 train 不能并行，GPU 有空闲

异步 (slime):
  GPU_A: generate(policy_v1) → generate(policy_v2) → generate(policy_v3) → ...
  GPU_B:                train(data_v1) → train(data_v2) → ...
  ✅ GPU 利用率最大化
  ❌ train 时用的数据可能来自旧 policy（off-policy）
  → 需要 importance sampling ratio 修正
```

---

## 四、其他 RL 训练框架对比

| 框架 | 来源 | 核心特点 |
|------|------|---------|
| **Slime** | 清华 THUDM | 异步, Megatron+SGLang |
| **verl** | 字节跳动 | 同步 HybridFlow, vLLM |
| **OpenRLHF** | 社区 | Ray+vLLM, 轻量级 |
| **TRL** | HuggingFace | 简单易用, 单机优先 |
| **NeMo-Aligner** | NVIDIA | NeMo 生态, 企业级 |

---

## 五、面试回答模板

### Q: "Slime 和 verl 有什么区别？"

> "两者都是 LLM 后训练的 RL 框架，但设计理念不同。
>
> **Slime** 来自清华 THUDM，是 GLM 系列的训练框架。
> 它的核心是**异步训练**：用 SGLang 做 rollout 生成，Megatron 做梯度更新，
> 两者用 Data Buffer 解耦，可以并行跑。好处是 GPU 利用率最大化，
> 代价是需要处理 off-policy 数据问题。
> 另外它选择 SGLang 而不是 vLLM 做生成引擎，因为 SGLang 的 RadixAttention
> 对 RL 场景中大量相同 prompt 的 prefix caching 更友好。
>
> **verl** 来自字节火山引擎，用 HybridFlow 在同一组 GPU 上交替做 generation（vLLM）
> 和 training（FSDP），是**同步模式**。好处是数据永远 on-policy，训练更稳定，
> 而且不需要额外 GPU 资源。
>
> 选择取决于场景：大规模训练（数百 GPU）优先 Slime 的异步方案提高利用率；
> 中等规模或对稳定性要求高的场景优先 verl 的同步方案。"

### Q: "为什么异步 RL 需要 importance sampling？"

> "异步模式下，training 用的 rollout 数据可能来自 policy_v1，但当前 policy 已经更新到 v3。
> 这是 off-policy 问题。需要用 importance sampling ratio r = π_new / π_old 来修正梯度估计。
> PPO 的 clip 机制天然限制了 ratio 的范围（1-ε, 1+ε），所以对轻度 off-policy 是鲁棒的。
> 但如果滞后太多（>3-5 个版本），修正会不准确，需要丢弃旧数据。"
