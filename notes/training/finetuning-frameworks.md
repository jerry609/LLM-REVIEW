# 微调框架实战指南：LlamaFactory / Unsloth / trl

> 三大主流微调框架的使用方式、适用场景与对比

---

## 一、框架全景对比

| 特性 | **LlamaFactory** | **Unsloth** | **trl (HF)** |
|------|-----------------|-------------|-------------|
| 定位 | 一站式微调平台 | 极速训练加速 | 官方 RL/SFT 库 |
| 上手难度 | ⭐ 最低 (YAML/WebUI) | ⭐⭐ 低 | ⭐⭐⭐ 中等 |
| 训练速度 | 标准 | **2-5x 加速** | 标准 |
| 支持模型 | 100+ 模型 | 30+ 主流模型 | HF 全生态 |
| SFT | ✅ | ✅ | ✅ SFTTrainer |
| DPO/RLHF | ✅ | ✅ | ✅ DPOTrainer/PPOTrainer |
| GRPO | ✅ | 部分 | ✅ GRPOTrainer |
| 多机多卡 | ✅ DeepSpeed | ❌ 仅单机 | ✅ Accelerate |
| 量化训练 | ✅ QLoRA | ✅ 4bit 原生 | ✅ BitsAndBytes |
| WebUI | ✅ 可视化界面 | ❌ | ❌ |
| 灵活度 | 中 (配置化) | 中 | **高 (代码级)** |

---

## 二、LlamaFactory 实战

### 2.1 安装与启动

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"

# 启动 WebUI
llamafactory-cli webui
```

### 2.2 YAML 配置文件方式

```yaml
# examples/train_lora/qwen2_lora_sft.yaml
### model
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
quantization_bit: 4          # QLoRA 4bit

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all              # 所有线性层加 LoRA
lora_rank: 16
lora_alpha: 32

### dataset
dataset: legal_qa             # data/ 目录下的数据集名
template: qwen                # 对话模板
cutoff_len: 2048
max_samples: 50000

### train
output_dir: saves/qwen2-7b-lora-legal
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
num_train_epochs: 3
learning_rate: 2.0e-5
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
logging_steps: 10
save_steps: 500

### eval
per_device_eval_batch_size: 2
eval_strategy: steps
eval_steps: 500
```

```bash
# 启动训练
llamafactory-cli train examples/train_lora/qwen2_lora_sft.yaml

# 推理测试
llamafactory-cli chat examples/inference/qwen2_lora_sft.yaml

# 导出合并模型
llamafactory-cli export examples/merge_lora/qwen2_lora_sft.yaml
```

### 2.3 自定义数据集注册

```json
// data/dataset_info.json 添加
{
  "legal_qa": {
    "file_name": "legal_train.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant",
      "system_tag": "system"
    }
  }
}
```

---

## 三、Unsloth 实战

### 3.1 安装

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
```

### 3.2 核心训练代码

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# 1. 加载模型（自动 4bit 量化 + LoRA 注入）
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    max_seq_length=2048,
    dtype=None,           # 自动检测
    load_in_4bit=True,
)

# 2. 添加 LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",  # 节省 30% 显存
)

# 3. 数据准备
def formatting_prompts(examples):
    texts = []
    for conv in examples["conversations"]:
        text = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return {"text": texts}

dataset = load_dataset("json", data_files="train.jsonl")["train"]
dataset = dataset.map(formatting_prompts, batched=True)

# 4. 训练
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    packing=True,                # 序列打包，提升 GPU 利用率
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        num_train_epochs=3,
        learning_rate=2e-5,
        bf16=True,
        logging_steps=10,
        output_dir="outputs",
        optim="adamw_8bit",      # 8bit 优化器节省显存
    ),
)

trainer.train()

# 5. 保存
model.save_pretrained("lora_model")       # 保存 LoRA adapter
model.save_pretrained_merged("merged_model", tokenizer)  # 合并保存
```

### 3.3 Unsloth 加速原理

| 优化 | 效果 |
|------|------|
| 手写 Triton 内核 | QKV/FFN 投影 2x 加速 |
| RoPE 优化 | 融合计算，减少内存访问 |
| 交叉熵内核 | 不实例化 logits 矩阵，节省大量显存 |
| 梯度检查点优化 | 比 HF 默认节省 30% 显存 |
| 智能序列打包 | 减少 padding 浪费 |

---

## 四、trl 库实战

### 4.1 SFTTrainer

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# 1. 模型 + LoRA
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# 2. 数据集
dataset = load_dataset("json", data_files="sft_data.jsonl")["train"]

# 3. 训练
training_args = SFTConfig(
    output_dir="sft_output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    bf16=True,
    logging_steps=10,
    save_steps=500,
    max_seq_length=2048,
    packing=True,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model("sft_output/final")
```

### 4.2 DPOTrainer

```python
from trl import DPOTrainer, DPOConfig

# DPO 数据格式: {"prompt": "...", "chosen": "...", "rejected": "..."}
dpo_dataset = load_dataset("json", data_files="dpo_data.jsonl")["train"]

dpo_config = DPOConfig(
    output_dir="dpo_output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=5e-6,    # DPO 学习率要更小
    beta=0.1,              # KL 散度系数
    bf16=True,
    max_length=2048,
    max_prompt_length=512,
)

# DPO 需要 reference model
trainer = DPOTrainer(
    model=model,
    ref_model=None,        # 如果用 LoRA，自动用冻结底座作 ref
    args=dpo_config,
    train_dataset=dpo_dataset,
    processing_class=tokenizer,
)

trainer.train()
```

### 4.3 GRPOTrainer

```python
from trl import GRPOTrainer, GRPOConfig

def reward_function(completions, prompts):
    """自定义奖励函数"""
    rewards = []
    for completion in completions:
        # 示例：长度奖励 + 格式奖励
        score = 0
        if len(completion) > 100:
            score += 1.0
        if "```" in completion:  # 包含代码块
            score += 0.5
        rewards.append(score)
    return rewards

grpo_config = GRPOConfig(
    output_dir="grpo_output",
    per_device_train_batch_size=1,
    num_generations=4,       # 每个 prompt 生成 4 个候选
    num_train_epochs=1,
    learning_rate=1e-6,
    bf16=True,
)

trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=prompt_dataset,
    reward_funcs=reward_function,
    processing_class=tokenizer,
)

trainer.train()
```

---

## 五、训练监控与评估

### 5.1 关键监控指标

```python
from transformers import TrainerCallback

class MonitorCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # 核心指标
            print(f"Step {state.global_step}:")
            print(f"  loss: {logs.get('loss', 'N/A'):.4f}")
            print(f"  lr: {logs.get('learning_rate', 'N/A'):.2e}")
            print(f"  grad_norm: {logs.get('grad_norm', 'N/A'):.2f}")
            print(f"  epoch: {logs.get('epoch', 'N/A'):.2f}")
```

### 5.2 评估 Checklist

```
训练前:
□ 数据格式正确（用 tokenizer 编码后检查）
□ Label 的 masking 正确（用户部分 = -100）
□ 特殊 token 正确（BOS/EOS/PAD）
□ LoRA target modules 覆盖合理

训练中:
□ Loss 持续下降
□ 梯度范数正常（0.1-10）
□ 学习率按 schedule 变化
□ GPU 利用率 >80%

训练后:
□ 验证集 loss 不回升（无过拟合）
□ 人工检查 10-20 个样本的输出质量
□ 对比微调前后的 Benchmark 分数
□ 检查是否有灾难性遗忘（通用能力下降）
```

---

## 六、训练资源估算

### 6.1 显存估算公式

```
全量微调显存 ≈ 16 × N (参数量，单位 Bytes)
  = 2N (权重 BF16) + 2N (梯度 BF16) + 12N (Adam 优化器状态)

QLoRA 显存 ≈ 0.5N + 16 × N_lora
  = 0.5N (4bit 底座) + 16 × N_lora (LoRA 参数的优化器状态)

示例 (7B 模型):
  全量微调:  16 × 7B = 112 GB → 需要 2×A100 80GB
  QLoRA:     0.5 × 7 + 16 × 0.013 ≈ 3.7 GB → 单卡 RTX 3090 可跑
```

### 6.2 训练时间估算

```
训练时间 ≈ (3 × N_params × N_tokens) / (GPU_FLOPS × MFU × N_GPUs)

示例:
  7B 模型，100K 样本 × 2048 tokens/样本 = 200M tokens
  单 A100 (312 TFLOPS BF16)，MFU=40%
  时间 ≈ 3 × 7e9 × 2e8 / (312e12 × 0.4 × 1) ≈ 3.4 小时

  QLoRA (仅训练 0.2% 参数):
  时间 ≈ 3.4 × 0.3 ≈ 1 小时 (实际前向仍然全量，约 2-3 小时)
```

---

## 面试高频问答

**Q1：LlamaFactory 和 trl 怎么选？**
> LlamaFactory 适合快速实验和不太需要自定义训练逻辑的场景，配置驱动、WebUI 友好。trl 适合需要深度定制训练流程（如自定义 reward function、特殊 loss）的场景。工业生产中，先用 LlamaFactory 跑 baseline，再用 trl 做精细优化。

**Q2：Unsloth 的加速效果为什么这么好？**
> Unsloth 用手写的 Triton 内核替代了 PyTorch 默认的矩阵乘法和交叉熵实现，减少了大量中间张量的实例化和内存访问。特别是交叉熵内核——标准实现需要实例化 (B, L, V) 的完整 logits 矩阵（V=150K 时极其庞大），Unsloth 的 chunked 实现避免了这一点。

**Q3：LoRA 微调时学习率怎么设？**
> 通常 1e-5 到 5e-5，比预训练低一个数量级。DPO 更低（5e-7 到 5e-6）。经验法则：数据量少 → 学习率更小（避免过拟合）；LoRA rank 高 → 学习率可以稍大。

## 面试一句话
- "LlamaFactory 是配置驱动的一站式平台适合快速实验，Unsloth 用 Triton 内核实现 2-5x 训练加速，trl 是 HF 官方库提供 SFT/DPO/GRPO Trainer 适合深度定制。QLoRA 让 7B 模型在单卡 3090 上就能微调。"
