# RL 系统面试备战：实战 Demo 清单 & 试错日志

> 目标：**跑通每个 Demo → 记录踩坑 → 形成面试可展示的"我实际做过"的证据**

---

## 一、必跑 Demo 清单（按优先级排序）

### 🔴 P0：面试核心考点，必须跑通

| # | Demo | 框架 | 硬件要求 | 预计耗时 | 面试价值 |
|---|------|------|---------|---------|---------|
| 1 | **vLLM Multi-LoRA Serving** | vLLM | 1×A100/4090 | 2h | 千级 LoRA 调度 |
| 2 | **SGLang Prefix Caching 基准测试** | SGLang | 1×A100/4090 | 2h | RadixAttention vs Hash |
| 3 | **OpenRLHF GRPO 训练 (Qwen2.5-0.5B)** | OpenRLHF + vLLM | 1×A100 | 4h | RL 训练全链路 |
| 4 | **Slime 异步训练 (GSM8K)** | Slime + SGLang | 4×A100 | 6h | JD 核心框架 |
| 5 | **trl DPO/GRPO 单卡训练** | trl + peft | 1×4090 | 2h | 快速验证 RL 算法 |

### 🟡 P1：加分项，有时间就跑

| # | Demo | 框架 | 硬件要求 | 预计耗时 | 面试价值 |
|---|------|------|---------|---------|---------|
| 6 | **verl HybridFlow GSM8K** | verl + vLLM | 4×A100 | 6h | 对比 Slime |
| 7 | **Megatron-LM LoRA 训练** | Megatron | 8×A100 | 8h | 分布式训练 |
| 8 | **S-LoRA 千级 LoRA 基准** | S-LoRA | 1×A100 | 3h | Multi-LoRA 性能 |
| 9 | **vLLM FP8 量化推理对比** | vLLM | 1×H100 | 2h | 量化加速 |
| 10 | **nsys Profiling 全链路分析** | nsys + PyTorch | 1×A100 | 3h | 瓶颈定位能力 |

### 🟢 P2：锦上添花

| # | Demo | 框架 | 硬件要求 | 预计耗时 | 面试价值 |
|---|------|------|---------|---------|---------|
| 11 | **Triton Fused LoRA Kernel** | Triton | 1×GPU | 4h | GPU 编程 |
| 12 | **DeepSpeed-Chat 3阶段 RLHF** | DeepSpeed | 4×A100 | 8h | 老牌方案理解 |
| 13 | **SGLang 约束解码 (JSON Schema)** | SGLang | 1×4090 | 1h | Agent 场景 |

---

## 二、详细实验指南

### Demo 1: vLLM Multi-LoRA Serving

**目的**：理解 vLLM 如何同时服务多个 LoRA，观察 LoRA 切换开销和吞吐变化。

```bash
# === 环境准备 ===
pip install vllm>=0.6.0

# === 步骤 1: 启动 Multi-LoRA Server ===
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.2-1B \
    --enable-lora \
    --lora-modules \
        lora1=path/to/lora1 \
        lora2=path/to/lora2 \
    --max-loras 4 \
    --max-lora-rank 16 \
    --max-model-len 2048 \
    --port 8000

# === 步骤 2: 请求不同 LoRA ===
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "lora1", "prompt": "hello", "max_tokens": 50}'

curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "lora2", "prompt": "hello", "max_tokens": 50}'

# === 步骤 3: 压测不同 LoRA 数量下的吞吐 ===
python benchmark_multi_lora.py --num-loras 1 2 4 8 16 --concurrent 32
```

**关键观察指标**：
- [ ] 不同 LoRA 数量下的吞吐 (tokens/s)
- [ ] LoRA 切换延迟 (通过日志观察)
- [ ] 显存占用随 LoRA 数量的增长
- [ ] 同 LoRA 请求聚合 batch 的效果

**预期踩坑**：
```
| 问题 | 原因 | 解决方案 |
|------|------|---------|
| LoRA 加载报错 | LoRA 的 target_modules 与 base 不匹配 | 检查 adapter_config.json |
| 推理结果异常 | LoRA rank/alpha 不对 | 确保 adapter_config 参数一致 |
| 显存 OOM | max-loras 太大 | 减少 max-loras 或用更小的模型 |
| 吞吐抖动 | LoRA 切换频繁 | 增加 batch 聚合窗口 |
```

---

### Demo 2: SGLang Prefix Caching 基准测试

**目的**：验证 RadixAttention 在 RL rollout (同 prompt 多次采样) 场景下的加速效果。

```bash
# === 环境准备 ===
pip install sglang[all]

# === 步骤 1: 启动 SGLang Server ===
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.2-1B \
    --port 30000

# === 步骤 2: RL Rollout 模拟 (同 prompt 多次采样) ===
import sglang as sgl
import time

@sgl.function
def rl_rollout(s, prompt):
    s += prompt
    s += sgl.gen("response", max_tokens=256, temperature=1.0)

# 同一个 prompt, 生成 G=16 条
prompt = "Solve the following math problem step by step: What is 123 * 456?"

# 第一次: 需要 prefill
t0 = time.time()
state = rl_rollout.run(prompt=prompt)
t1 = time.time()
print(f"第 1 次 (cold): {t1-t0:.3f}s")

# 第 2-16 次: prefix cache 命中
for i in range(2, 17):
    t0 = time.time()
    state = rl_rollout.run(prompt=prompt)
    t1 = time.time()
    print(f"第{i:2d}次 (cached): {t1-t0:.3f}s")

# === 步骤 3: 对比 prefix cache 关闭 ===
# 重启 server 加 --disable-radix-cache
# 重复上面的测试
```

**关键观察指标**：
- [ ] 首次 vs 后续请求的 TTFT (Time To First Token)
- [ ] Cache 命中率 (通过 /get_model_info 接口)
- [ ] 不同 prompt 长度 (512/1024/2048) 的加速比
- [ ] 不同 G 值 (4/8/16/32) 的总耗时对比

**预期结果**：
```
G=16, prompt_len=1024:
- 无 Prefix Cache: 16 × (prefill + decode) = ~16x 成本
- 有 Prefix Cache: 1 × prefill + 16 × decode = ~1x + 少量开销
- 加速比: ~10-15x (prefill 占比越高，加速越显著)
```

---

### Demo 3: OpenRLHF GRPO 训练

**目的**：端到端跑通 RL 训练，理解 Rollout→Reward→Update 全链路。

```bash
# === 环境准备 ===
git clone https://github.com/OpenRLHF/OpenRLHF.git
cd OpenRLHF
pip install -e .

# === 步骤 1: 启动 GRPO 训练 (单 GPU 版) ===
# 使用 GSM8K 数学推理任务
python examples/train_grpo.py \
    --pretrain Qwen/Qwen2.5-0.5B-Instruct \
    --dataset openai/gsm8k \
    --output_dir ./outputs/grpo_gsm8k \
    --max_epochs 1 \
    --num_episodes 512 \
    --rollout_batch_size 64 \
    --micro_rollout_batch_size 16 \
    --group_size 8 \
    --lr 5e-6 \
    --kl_coef 0.01 \
    --max_len 512 \
    --logging_steps 5 \
    --bf16

# === 步骤 2: 用 vLLM 加速 Rollout ===
python examples/train_grpo.py \
    --pretrain Qwen/Qwen2.5-0.5B-Instruct \
    --dataset openai/gsm8k \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    ... (其余参数同上)

# === 步骤 3: 记录关键指标 ===
# TensorBoard 观察:
# - reward/mean
# - policy_loss
# - kl_divergence
# - throughput (samples/s)
```

**关键观察指标**：
- [ ] Rollout 占总训练时间的比例 (目标: 了解瓶颈)
- [ ] 有/无 vLLM 的吞吐对比
- [ ] reward 收敛曲线
- [ ] KL 散度变化趋势

---

### Demo 4: Slime 异步训练 ⭐ (JD 核心)

**目的**：跑通 Slime，理解异步 RL 的工作方式，重点看 SGLang + Megatron 的协作。

```bash
# === 环境准备 ===
git clone https://github.com/THUDM/Slime.git
cd Slime
pip install -e .

# === 步骤 1: 理解 Slime 目录结构 ===
tree -L 2
# 重点关注:
# slime/
# ├── algo/          ← RL 算法 (GRPO/PPO)
# ├── data_buffer/   ← 异步数据缓冲区
# ├── rollout/       ← SGLang Rollout Engine
# ├── training/      ← Megatron Training Engine
# └── pipeline/      ← 训推流水线编排

# === 步骤 2: 阅读配置文件 ===
cat configs/gsm8k_slime.yaml
# 关注:
# - rollout.engine: sglang
# - training.engine: megatron
# - pipeline.mode: async  ← 异步模式
# - pipeline.staleness_limit: 3  ← 允许的数据过期步数

# === 步骤 3: 启动训练 ===
# (需要 4×A100, 根据实际调整)
bash scripts/train_gsm8k.sh

# === 步骤 4: 观察异步行为 ===
# 1. Rollout Worker 和 Training Worker 的 GPU 利用率
watch -n 1 nvidia-smi
# 2. Data Buffer 的填充和消费速度
# 3. 日志中的 policy_version 字段 (检查 staleness)
```

**关键观察指标**：
- [ ] Training 和 Rollout 是否真正并行执行
- [ ] Data Buffer 中数据的 staleness 分布
- [ ] 与同步训练 (verl) 的吞吐对比
- [ ] 异步训练的 reward 收敛是否有退化

**深入分析点**（面试加分）：
```
1. 权重同步延迟:
   - 观察 Training 更新权重后, Rollout 多久拿到新权重
   - 记录日志中的 weight_version 字段

2. GPU 利用率:
   - 同步模式: Training 时 Rollout GPU 空闲 → 利用率 ~50%
   - 异步模式: 两组 GPU 持续工作 → 利用率 ~90%

3. staleness 影响:
   - staleness=0 (同步): reward 收敛最稳定
   - staleness=1-3: reward 略有波动但吞吐翻倍
   - staleness>5: reward 可能不收敛
```

---

### Demo 5: trl DPO/GRPO 单卡训练

**目的**：最轻量的 RL 训练 demo，适合快速验证和理解算法。

```bash
# === 环境准备 ===
pip install trl peft transformers datasets

# === GRPO 训练 (最简版) ===
python -c "
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')

def reward_fn(completions, **kwargs):
    return [float(len(c) < 200) for c in completions]  # 奖励短回复

config = GRPOConfig(
    output_dir='./grpo_output',
    num_train_epochs=1,
    per_device_train_batch_size=2,
    num_generations=4,
    max_completion_length=128,
    logging_steps=5,
)

trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    reward_funcs=reward_fn,
    config=config,
    train_dataset=...,  # 准备 prompt 数据集
)
trainer.train()
"
```

---

## 三、试错日志模板

每个实验用以下格式记录：

```markdown
### 实验 [编号]: [名称]

**日期**: YYYY-MM-DD
**环境**: GPU型号 × 数量, CUDA版本, Python版本, 框架版本
**目标**: 一句话描述

#### 运行命令
```bash
实际运行的命令
```

#### 遇到的问题

| # | 问题描述 | 错误信息 | 根因分析 | 解决方案 | 耗时 |
|---|---------|---------|---------|---------|------|
| 1 | ... | ... | ... | ... | ... |

#### 关键指标

| 指标 | 值 | 备注 |
|------|---|------|
| 吞吐 (tokens/s) | | |
| 延迟 (ms/token) | | |
| 显存占用 (GB) | | |
| GPU 利用率 (%) | | |

#### 核心发现 (面试可讲)
1. ...
2. ...

#### 截图/图表
(附上 loss 曲线、GPU 利用率截图等)
```

---

## 四、常见问题速查表 (Troubleshooting)

### 4.1 环境问题

| 问题 | 错误信息 | 解决方案 |
|------|---------|---------|
| CUDA 版本不匹配 | `CUDA error: no kernel image` | `nvcc --version` 与 `torch.version.cuda` 对齐 |
| Flash Attention 编译失败 | `No such file: flash_attn_2_cuda.so` | `pip install flash-attn --no-build-isolation` |
| vLLM 启动报错 | `Cannot load model` | 检查模型路径和 `trust_remote_code=True` |
| SGLang import 失败 | `No module named 'sglang'` | `pip install "sglang[all]"` 注意引号 |
| Megatron 找不到 NCCL | `NCCL not found` | `export NCCL_HOME=/usr/lib/x86_64-linux-gnu` |
| Ray 集群连接失败 | `ConnectionError` | `ray stop && ray start --head` 重启 |

### 4.2 训练问题

| 问题 | 症状 | 解决方案 |
|------|------|---------|
| Loss NaN | 训练几步后 loss 变 NaN | 降低 lr (1e-6)，开 loss scaling，检查数据 |
| Reward 不涨 | reward 长期平坦 | 检查 reward function 是否正确返回信号 |
| KL 爆炸 | KL divergence 快速增长 | 增大 kl_coef (0.01→0.1)，减小 lr |
| OOM | `CUDA out of memory` | 梯度检查点 / 减 batch / 减 max_len / 量化 |
| 生成退化 | 生成重复文本 | 检查 temperature，增加 entropy bonus |
| GRPO 不收敛 | reward 方差大 | 增大 group_size (4→8→16) |

### 4.3 推理/部署问题

| 问题 | 症状 | 解决方案 |
|------|------|---------|
| vLLM 吞吐低 | tokens/s 不及预期 | 增大 `max-num-seqs`，检查 batch 是否打满 |
| Prefix Cache 不生效 | TTFT 无变化 | 确认 `--enable-prefix-caching`，检查 prompt 是否完全一致 |
| Multi-LoRA 切换慢 | 延迟抖动 | 增加 `max-loras` 预加载数量 |
| SGLang 生成截断 | 输出不完整 | 增大 `max_tokens`，检查 stop tokens |
| TP 通信瓶颈 | 多卡但吞吐不线性 | 检查 NVLink 是否启用: `nvidia-smi topo -m` |

### 4.4 Slime 特有问题

| 问题 | 症状 | 解决方案 |
|------|------|---------|
| 异步数据过期 | reward 抖动大 | 减小 `staleness_limit` |
| SGLang↔Megatron 权重同步失败 | 训练卡住 | 检查 NCCL 端口冲突 |
| Data Buffer 溢出 | OOM | 减小 buffer_size 或加速 training 消费 |
| Rollout 和 Training 速度不匹配 | 一方空闲 | 调整 GPU 分配比例 |

---

## 五、面试展示策略

### 5.1 Demo 选择矩阵

| 面试问题类型 | 展示的 Demo | 展示要点 |
|-------------|-----------|---------|
| "介绍一下 vLLM" | Demo 1: Multi-LoRA | "我实测了 N 个 LoRA 的吞吐衰减曲线" |
| "如何优化 RL 训练" | Demo 3+4: OpenRLHF→Slime | "我对比了同步和异步的 GPU 利用率" |
| "Prefix Cache 原理" | Demo 2: SGLang 基准 | "我实测了 G=16 时 RadixAttention 加速 10x" |
| "你对哪个框架最了解" | Demo 4: Slime | "我读了源码并分析了 staleness 影响" |
| "系统优化经验" | Demo 10: nsys | "我用 nsys 定位了 XX 是瓶颈，优化后提升 XX%" |

### 5.2 一句话话术

- **Demo 1**: "我实测 vLLM Multi-LoRA serving，4 个 LoRA 同时服务，吞吐仅下降 8%，LoRA-aware scheduling 是关键。"
- **Demo 2**: "我对比了 SGLang RadixAttention vs vLLM Hash-based prefix cache，RL rollout (G=16) 场景下 SGLang TTFT 加速 12x。"
- **Demo 3**: "我用 OpenRLHF 跑 GRPO on GSM8K，vLLM 作 rollout 引擎后训练吞吐提升 5x，rollout 从 70% 降到 30% 时间占比。"
- **Demo 4**: "我跑通了 Slime 异步训练，staleness=2 时 reward 收敛几乎不退化但 GPU 利用率从 50% 提到 88%。"

---

## 六、实验记录区

> 以下为实际实验记录，按时间倒序排列。

### 实验 0: 环境搭建

**日期**: 待填写
**环境**: 待填写

```bash
# 基础环境
conda create -n rl-infra python=3.10 -y
conda activate rl-infra

# PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 推理框架
pip install vllm>=0.6.0
pip install "sglang[all]"

# RL 训练框架
pip install trl peft transformers datasets accelerate
pip install deepspeed

# OpenRLHF
git clone https://github.com/OpenRLHF/OpenRLHF.git
cd OpenRLHF && pip install -e .

# Slime
git clone https://github.com/THUDM/Slime.git
cd Slime && pip install -e .

# Profiling
pip install py-spy
```

#### 环境问题记录

| # | 问题描述 | 错误信息 | 根因分析 | 解决方案 | 耗时 |
|---|---------|---------|---------|---------|------|
| | (待填写) | | | | |

---

*(更多实验记录在实际运行后追加)*
