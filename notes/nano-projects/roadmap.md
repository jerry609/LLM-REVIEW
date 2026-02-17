# Nano 项目路线图 —— 走完 LLM 面试全模块

> "Nano 项目" = 500 行以内、可独立运行、面试中可当场演示的最小完整项目

---

## 一、经典 Nano 项目全景（开源社区）

### 🔥 Andrej Karpathy 系列（必看）

| 项目 | GitHub | 代码量 | 覆盖面试模块 |
|------|--------|--------|-------------|
| **nanoGPT** | [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | ~600行 | Transformer 训练全流程、Scaling Law |
| **build-nanogpt** | [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt) | ~1000行 | 从零手写 GPT-2、分布式训练 |
| **minbpe** | [karpathy/minbpe](https://github.com/karpathy/minbpe) | ~400行 | BPE Tokenizer 原理与实现 |
| **llm.c** | [karpathy/llm.c](https://github.com/karpathy/llm.c) | ~2000行C | CUDA/C 级别的 GPT-2 训练，极致性能 |
| **micrograd** | [karpathy/micrograd](https://github.com/karpathy/micrograd) | ~150行 | 自动求导引擎（面试手撕） |
| **makemore** | [karpathy/makemore](https://github.com/karpathy/makemore) | ~500行 | 字符级语言模型，多种架构对比 |

### 🛠️ LLM Inference/Serving 相关

| 项目 | GitHub | 代码量 | 覆盖面试模块 |
|------|--------|--------|-------------|
| **llama2.c** | [karpathy/llama2.c](https://github.com/karpathy/llama2.c) | ~700行C | 纯 C 推理，理解 Transformer 推理流程 |
| **llama3-from-scratch** | [naklecha/llama3-from-scratch](https://github.com/naklecha/llama3-from-scratch) | ~200行 | RoPE/GQA/RMSNorm 手写实现 |
| **litgpt** | [Lightning-AI/litgpt](https://github.com/Lightning-AI/litgpt) | 精简版 | 多模型支持、量化推理 |
| **TinyLlama** | [jzhang38/TinyLlama](https://github.com/jzhang38/TinyLlama) | 训练框架 | 小模型预训练最佳实践 |

### 🧠 RL/Alignment 相关

| 项目 | GitHub | 代码量 | 覆盖面试模块 |
|------|--------|--------|-------------|
| **trl** | [huggingface/trl](https://github.com/huggingface/trl) | 核心模块 | PPO/DPO/GRPO 实现（最轻量） |
| **OpenRLHF** | [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | Ray+vLLM | 分布式 RLHF（对标 verl） |
| **verl examples** | [volcengine/verl/examples/](https://github.com/volcengine/verl) | 示例脚本 | PPO/GRPO 在 GSM8K 上的训练 |
| **Slime** | [THUDM/slime](https://github.com/THUDM/slime) | 异步 RL | SGLang + Megatron 异步训练 |
| **DeepSpeed-Chat** | [microsoft/DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples) | RLHF | 三阶段 RLHF pipeline |

### ⚡ CUDA/Kernel 相关

| 项目 | GitHub | 代码量 | 覆盖面试模块 |
|------|--------|--------|-------------|
| **flash-attention** | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | CUDA | Tiling + IO-aware attention |
| **Triton tutorials** | [openai/triton](https://github.com/openai/triton) | Triton | Fused kernel 编写入门 |
| **CUDA-MODE lectures** | [cuda-mode/lectures](https://github.com/cuda-mode/lectures) | 教学 | GPU 编程系统学习 |

---

## 二、自研 Nano 项目清单（配合你的仓库）

### 📊 对照表：面试模块 ↔ Nano 项目 ↔ 已有 Notebook

```
面试模块                    Nano 项目                      已有 Notebook
─────────────────────────────────────────────────────────────────────────
Attention 原理          → nano-attention                → llm_inference_fundamentals.ipynb ✅
KV Cache 管理           → nano-kv-cache                 → kv_cache_paged_lru_workshop.ipynb ✅
Speculative Decoding    → nano-spec-decode              → speculative_decoding_simulator.ipynb ✅
量化                    → nano-quant                    → quantization_precision_experiment.ipynb ✅
RAG + Prefix Cache      → nano-prefix-cache             → rag_prefix_caching_simulator.ipynb ✅
vLLM 架构              → nano-vllm-walkthrough          → vllm_architecture_walkthrough.ipynb ✅
分布式推理              → nano-distributed               → distributed_inference_roofline.ipynb ✅
Reasoning Models        → nano-reasoning                → reasoning_models_workshop.ipynb ✅
PPO/GRPO               → nano-ppo-grpo                 → rl_ppo_grpo_implementation.ipynb ✅
LeetCode               → nano-leetcode                 → leetcode_llm_system_related.ipynb ✅
Attention/BPE/BeamSearch→ nano-core-components          → attention_tokenizer_beamsearch.ipynb ✅

────────────── 以下是建议新增的 Nano 项目 ──────────────

RadixAttention 模拟     → nano-radix-attention          → 🆕 待创建
异步 RL Rollout 模拟     → nano-async-rl-rollout         → 🆕 待创建
SGLang vs vLLM 对比     → nano-sglang-vs-vllm           → 🆕 待创建
HybridFlow 模拟         → nano-hybridflow               → 🆕 待创建
Continuous Batching     → nano-continuous-batching      → 🆕 待创建（从 llm_inference 扩展）
```

---

## 三、🆕 建议新增的 5 个 Nano 项目详细设计

### Nano-1: RadixAttention Prefix Cache 模拟器

**文件**: `notebooks/nano_radix_attention.ipynb`
**代码量**: ~300 行
**面试价值**: ⭐⭐⭐⭐⭐（直接回答"为什么 Slime 选 SGLang"）

```
核心内容:
1. 手写 Radix Tree 数据结构
   - insert / match_prefix / evict 三大操作
2. 模拟 RL 场景: 同一 prompt → G 个 response
   - 对比 Hash-based (vLLM) vs Radix Tree (SGLang) 的 cache 命中率
3. 可视化:
   - 前缀树的增长过程（动画）
   - 不同 G 值下的 prefill 节省比例
   - 显存占用对比柱状图
4. Benchmark:
   - prompt_len × G 的二维热力图
   - 端到端延迟预估
```

### Nano-2: 异步 RL Rollout 模拟器

**文件**: `notebooks/nano_async_rl_rollout.ipynb`
**代码量**: ~350 行
**面试价值**: ⭐⭐⭐⭐⭐（Slime 核心架构理解）

```
核心内容:
1. 同步 vs 异步 RL 训练循环的 Python 模拟
   - 同步 (verl): generate → train → generate → ...
   - 异步 (slime): generate 和 train 并行
2. Staleness 问题模拟:
   - 训练时使用的 rollout 数据落后几个版本？
   - 不同 staleness 下的 reward 收敛曲线
3. Importance Sampling 修正:
   - 可视化 IS ratio 的分布
   - PPO clip 如何限制 ratio 范围
4. GPU 利用率对比:
   - 甘特图展示同步 vs 异步的 GPU 占用
   - 计算理论加速比
```

### Nano-3: SGLang vs vLLM Prefix Cache 对比

**文件**: `notebooks/nano_sglang_vs_vllm.ipynb`
**代码量**: ~250 行
**面试价值**: ⭐⭐⭐⭐（工程选型能力展示）

```
核心内容:
1. 模拟两种 prefix cache 策略:
   - Hash-based (vLLM): block 级 hash → LRU eviction
   - Radix Tree (SGLang): token 级树匹配 → tree LRU eviction
2. 工作负载模拟:
   - RL 场景: 100 prompts × G=16, Zipf 分布
   - Serving 场景: 1000 不同 prompts, 随机访问
   - Multi-turn 场景: 对话历史共享
3. 对比指标:
   - Cache hit rate
   - Prefill tokens saved
   - Memory efficiency
   - Lookup latency
4. 结论:
   - RL 场景 → SGLang 大幅领先
   - Random serving → 差距缩小
   - Multi-turn → SGLang 仍有优势
```

### Nano-4: HybridFlow 资源调度模拟

**文件**: `notebooks/nano_hybridflow_simulator.ipynb`
**代码量**: ~250 行
**面试价值**: ⭐⭐⭐⭐（verl 架构理解）

```
核心内容:
1. 模拟 verl 的 HybridFlow:
   - 同一 GPU 上 Generation ↔ Training 切换
   - 模型 reshard 开销 (FSDP ↔ vLLM 权重格式转换)
2. 对比 Slime 的分离式:
   - 不同 GPU 池做 gen/train
   - 无 reshard 但需要更多 GPU
3. 资源效率分析:
   - 给定 N 张 GPU, 哪种模式吞吐更高？
   - 甘特图展示 GPU timeline
4. Scaling 分析:
   - 8 GPU / 32 GPU / 128 GPU / 512 GPU 下的最优策略
```

### Nano-5: Continuous Batching + Prefix Cache 综合

**文件**: `notebooks/nano_continuous_batching_prefix.ipynb`
**代码量**: ~300 行
**面试价值**: ⭐⭐⭐⭐（LLM Serving 核心机制）

```
核心内容:
1. 手写 Continuous Batching Scheduler:
   - 动态 batch size 调整
   - Prefill vs Decode 分离调度
   - 请求优先级管理
2. 集成 Prefix Cache:
   - 调度器感知 prefix cache 命中情况
   - 命中请求跳过 prefill，直接 decode
   - 重排调度队列提高 cache 复用
3. RL 场景优化:
   - 同 prompt G 个 response 组成一个 batch
   - 只做 1 次 prefill + G 次 decode
   - 对比逐个生成 vs 批量生成
4. 可视化:
   - 调度器时间线（类甘特图）
   - 吞吐量 vs batch_size 曲线
   - Prefix cache 对 TTFT 的影响
```

---

## 四、Nano 项目实施时间表

### 已完成 ✅

| 项目 | Notebook | 状态 |
|------|----------|------|
| Attention MHA/GQA | `llm_inference_fundamentals.ipynb` | ✅ |
| KV Cache Paged/LRU | `kv_cache_paged_lru_workshop.ipynb` | ✅ |
| Speculative Decoding | `speculative_decoding_simulator.ipynb` | ✅ |
| Quantization | `quantization_precision_experiment.ipynb` | ✅ |
| RAG + Prefix Cache | `rag_prefix_caching_simulator.ipynb` | ✅ |
| vLLM Architecture | `vllm_architecture_walkthrough.ipynb` | ✅ |
| Distributed Inference | `distributed_inference_roofline.ipynb` | ✅ |
| Reasoning Models | `reasoning_models_workshop.ipynb` | ✅ |
| PPO/GRPO | `rl_ppo_grpo_implementation.ipynb` | ✅ |
| LeetCode LLM Related | `leetcode_llm_system_related.ipynb` | ✅ |
| Attention/BPE/BeamSearch | `attention_tokenizer_beamsearch.ipynb` | ✅ |

### 建议新增 🆕

| 优先级 | 项目 | Notebook | 预计时间 |
|--------|------|----------|---------|
| P0 | RadixAttention 模拟器 | `nano_radix_attention.ipynb` | 3h |
| P0 | 异步 RL Rollout 模拟 | `nano_async_rl_rollout.ipynb` | 4h |
| P1 | SGLang vs vLLM 对比 | `nano_sglang_vs_vllm.ipynb` | 3h |
| P1 | HybridFlow 模拟 | `nano_hybridflow_simulator.ipynb` | 3h |
| P2 | Continuous Batching 综合 | `nano_continuous_batching_prefix.ipynb` | 4h |

---

## 五、面试模块完整覆盖矩阵

```
                                    已有  新增  总覆盖
────────────────────────────────────────────────────
🧠 Attention/Transformer 原理       ✅    -    100%
📦 KV Cache 管理                    ✅    -    100%
🚀 推理优化 (Spec Decode/Quant)     ✅    -    100%
🌳 Prefix Caching                  ✅   🆕    100% ← RadixAttention
🏗️ Serving 架构 (vLLM/SGLang)      ✅   🆕    100% ← SGLang vs vLLM
📊 分布式推理                       ✅    -    100%
🤖 Reasoning Models                ✅    -    100%
🎯 RL 训练 (PPO/GRPO)              ✅   🆕    100% ← Async RL
🔧 系统设计                        ✅   🆕    100% ← HybridFlow
💻 算法编程                         ✅    -    100%
🎤 项目表达 (STAR/一页纸)           ✅    -    100%
────────────────────────────────────────────────────
```

---

## 六、面试"三段式"展示策略

面试时用 Nano 项目做"由浅入深"的展示：

```
Level 1 "我会用" (30秒):
  "我在项目中使用了 vLLM/SGLang 做 LLM Serving..."

Level 2 "我理解原理" (2分钟):
  打开 nano_radix_attention.ipynb
  "这是我手写的 RadixAttention 模拟器，
   可以看到 RL 场景下 prefix cache 的命中率..."

Level 3 "我能设计系统" (5分钟):
  打开 nano_async_rl_rollout.ipynb + nano_hybridflow_simulator.ipynb
  "这是 Slime 和 verl 两种架构的完整模拟，
   我分析了在不同 GPU 数量下的最优策略..."
```

这三层递进式展示，能让面试官感受到你不仅是"用过"，而是"深入理解了设计决策"。
