# 多模态大模型训练全流程

> 从数据准备到指令微调的多模态模型全链路 —— 对齐阶段 + 指令微调 + Q-Former + 实战指南

---

## 一、多模态训练全景图

```
                     多模态模型训练路线
                          │
    ┌─────────────────────┼─────────────────────────┐
    │                     │                         │
  从零训练             冻结组件训练                 微调已有 VLM
 (成本极高)           (BLIP-2/LLaVA)             (最实用)
                          │
              ┌───────────┼───────────┐
              │                       │
        Stage 1:                Stage 2:
        视觉-语言对齐          视觉指令微调
        (冻结 ViT+LLM)        (冻结 ViT, 训 LLM)
```

---

## 二、Stage 1：视觉-语言对齐 (Feature Alignment)

### 2.1 目标

> 让视觉编码器输出的 visual tokens 能被 LLM "听懂"

```
训练前:
  ViT 输出 → random noise (对 LLM 而言)

训练后:
  ViT 输出 → meaningful tokens (LLM 可以理解的语义向量)
```

### 2.2 训练配置

```python
# LLaVA Stage-1 配置
stage1_config = {
    # 冻结策略
    "freeze_vision_encoder": True,   # ViT 完全冻结
    "freeze_llm": True,              # LLM 完全冻结
    "trainable": "projector",        # 仅训练 MLP Projector

    # 数据
    "dataset": "LLaVA-CC3M-Pretrain-595K",  # 图文对
    "task": "image_captioning",       # 看图说话

    # 超参
    "learning_rate": 1e-3,            # 较大 (只训练 Projector)
    "batch_size": 256,
    "epochs": 1,
    "warmup_ratio": 0.03,
    "max_length": 2048,

    # 资源
    "training_time": "5.5h on 8×A100",
    "projector_params": "~20M",
}
```

### 2.3 对齐数据构建

```python
# 对齐数据格式 (图文对)
{
    "image": "path/to/image.jpg",
    "conversations": [
        {"role": "user", "content": "<image>\nDescribe this image briefly."},
        {"role": "assistant", "content": "A golden retriever playing in the park."}
    ]
}

# 数据来源
# 1. CC3M / CC12M：网络图文对，自动生成 caption
# 2. LAION-5B 子集：大规模图文数据
# 3. ShareGPT4V：GPT-4V 生成的高质量描述
```

---

## 三、Stage 2：视觉指令微调 (Visual Instruction Tuning)

### 3.1 目标

> 让模型不仅"看懂"图片，还能根据指令"回答问题"

### 3.2 训练配置

```python
# LLaVA Stage-2 配置
stage2_config = {
    # 冻结策略
    "freeze_vision_encoder": True,    # ViT 仍冻结
    "freeze_llm": False,              # LLM 解冻 (全量或 LoRA)
    "trainable": "projector + llm",   # 同时训练

    # 数据
    "dataset": "LLaVA-Instruct-150K",  # 多模态指令数据
    "tasks": ["vqa", "reasoning", "description", "conversation"],

    # 超参
    "learning_rate": 2e-5,             # 比 Stage-1 小 (全量微调 LLM)
    "batch_size": 128,
    "epochs": 1,
    "warmup_ratio": 0.03,
    "max_length": 2048,

    # LoRA 微调 (可选，节省显存)
    "use_lora": True,
    "lora_r": 128,
    "lora_alpha": 256,
    "lora_target": ["q_proj", "k_proj", "v_proj", "o_proj"],
}
```

### 3.3 多模态指令数据类型

| 类型 | 比例 | 示例 | 来源 |
|------|------|------|------|
| **详细描述** | 30% | "详细描述这张图片" | GPT-4V 生成 |
| **视觉问答** | 30% | "图中有几个人？" | VQAv2, GQA |
| **复杂推理** | 20% | "图中的场景暗示了什么？" | GPT-4V 生成 |
| **多轮对话** | 15% | 围绕图片的追问 | 合成 |
| **OCR/文档** | 5% | "读出图中的文字" | DocVQA, TextVQA |

### 3.4 指令数据生成方法

```python
# 使用 GPT-4V 生成多模态指令数据 (Self-Instruct for VLM)
GENERATION_PROMPT = """
Given the image with the following caption and bounding boxes:
Caption: {caption}
Objects: {objects}

Please generate 3 different types of question-answer pairs:
1. A simple visual question (直接观察)
2. A reasoning question (需要推理)
3. A detailed description request (详细描述)

Each QA pair should be in JSON format.
"""

# 质量过滤
def filter_vqa(qa_pair, image_info):
    # 检查答案是否可由图片支撑
    # 检查问题是否有歧义
    # 检查答案长度和质量
    pass
```

---

## 四、Q-Former 训练详解 (BLIP-2 方式)

### 4.1 Q-Former 架构

```
               Learnable Queries (32 个)
                      │
                      ▼
              ┌──────────────┐
              │  Q-Former    │  ← 轻量 Transformer
              │              │
              │  Self-Attn   │  ← queries 互相关注
              │      +       │
              │  Cross-Attn  │  ← queries 关注 visual tokens
              │      +       │
              │  FFN         │
              └──────────────┘
                      │
            32 个 output query embeddings
                      │
              ┌───────┼───────┐
              │               │
        Stage 1:         Stage 2:
        对比/匹配/生成    投影到 LLM 空间
```

### 4.2 Stage 1: 视觉-语言表示学习

```python
# 三个联合训练目标
class QFormerStage1:
    def forward(self, image_embeds, text_ids):
        # 1. ITC (Image-Text Contrastive)
        query_output = self.qformer(query_embeds, image_embeds)
        image_feats = F.normalize(self.vision_proj(query_output), dim=-1)
        text_feats = F.normalize(self.text_proj(text_output), dim=-1)
        loss_itc = contrastive_loss(image_feats, text_feats)

        # 2. ITM (Image-Text Matching)
        # 用 [CLS] token 做二分类：这对图文是否匹配
        itm_output = self.qformer(query_embeds, image_embeds, text_ids)
        loss_itm = F.cross_entropy(self.itm_head(itm_output), labels)

        # 3. ITG (Image-grounded Text Generation)
        # 给定图片，自回归生成文本
        loss_itg = self.generate_loss(query_embeds, image_embeds, text_ids)

        return loss_itc + loss_itm + loss_itg
```

### 4.3 Stage 2: 视觉-语言生成学习

```python
class QFormerStage2:
    def __init__(self):
        self.fc = nn.Linear(qformer_dim, llm_dim)  # 投影层

    def forward(self, image, text):
        # 1. ViT 编码 (冻结)
        image_embeds = self.vit(image)

        # 2. Q-Former 压缩 (冻结/微调)
        query_output = self.qformer(query_embeds, image_embeds)  # (B, 32, d_q)

        # 3. 投影到 LLM 空间
        llm_inputs = self.fc(query_output)  # (B, 32, d_llm)

        # 4. 拼接文本 token 后送入 LLM (冻结)
        text_embeds = self.llm.embed(text_ids)
        combined = torch.cat([llm_inputs, text_embeds], dim=1)
        output = self.llm(inputs_embeds=combined)

        return output
```

---

## 五、多模态微调实战 (Qwen2-VL)

### 5.1 数据准备

```json
// 训练数据格式
{
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "path/to/image.jpg"},
                {"type": "text", "text": "这张图片展示了什么？请详细描述。"}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "这张图片展示了一个城市公园..."}
            ]
        }
    ]
}
```

### 5.2 使用 LlamaFactory 微调 Qwen2-VL

```yaml
# qwen2vl_lora_sft.yaml
### model
model_name_or_path: Qwen/Qwen2-VL-7B-Instruct
quantization_bit: 4

### method
stage: sft
finetuning_type: lora
lora_target: all
lora_rank: 64

### dataset
dataset: my_vl_dataset
template: qwen2_vl
cutoff_len: 4096

### train
output_dir: saves/qwen2vl-lora
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
num_train_epochs: 3
learning_rate: 1e-5
bf16: true

### vision
image_resolution: 512
video_resolution: 128
video_fps: 2.0
```

### 5.3 使用 transformers 直接微调

```python
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# 1. 加载模型
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# 2. LoRA 配置 (只训练 LLM 部分)
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# 3. 冻结视觉编码器
for name, param in model.named_parameters():
    if "visual" in name:
        param.requires_grad = False

# 4. 数据处理
def preprocess(examples):
    messages = examples["messages"]
    texts = [processor.apply_chat_template(m, tokenize=False) for m in messages]
    # 处理图片
    images = [load_images(m) for m in messages]
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    batch["labels"] = batch["input_ids"].clone()
    return batch

# 5. 训练
training_args = SFTConfig(
    output_dir="vl_output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=1e-5,
    bf16=True,
    remove_unused_columns=False,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=preprocess,
    processing_class=processor,
)
trainer.train()
```

---

## 六、训练技巧与常见问题

### 6.1 显存优化

| 技巧 | 节省量 | 说明 |
|------|--------|------|
| 冻结 ViT | ~30% | Stage-2 标配 |
| LoRA 微调 LLM | ~60% | 替代全量微调 |
| QLoRA (4bit LLM) | ~80% | 单卡训练 7B VLM |
| 动态分辨率 | 可变 | 限制 max_pixels |
| 梯度检查点 | ~40% | 用计算换显存 |
| Flash Attention 2 | ~20% | 注意力计算优化 |

### 6.2 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 模型只描述不回答 | Stage-2 数据缺少 QA | 增加 VQA 数据比例 |
| 幻觉（描述不存在的物体） | 对齐不充分 | 增加 Stage-1 数据量 + 对比学习 |
| 忽略图片只看文字 | Projector 没学好 | Stage-1 训练更久 / 增大 lr |
| 空间理解差 | ViT 空间信息损失 | 用更高分辨率 / 保留位置编码 |
| OCR 能力差 | 训练数据缺少文字图片 | 混入 DocVQA/TextVQA 数据 |

### 6.3 评估 Benchmark

| Benchmark | 评测能力 | 数据类型 |
|-----------|---------|---------|
| **MMBench** | 综合视觉理解 | 多选题 |
| **MMMU** | 学科知识 + 视觉 | 大学考试题 |
| **MME** | 感知 + 认知 | 二选一 |
| **TextVQA** | OCR + 视觉问答 | 图中文字 |
| **DocVQA** | 文档理解 | 文档图片 |
| **MathVista** | 视觉数学推理 | 图表/几何 |
| **SEED-Bench** | 生成评估 | 多类型 |

---

## 七、训练资源估算

```
Qwen2-VL-7B LoRA 微调:
  显存: ~16GB (QLoRA 4bit) / ~28GB (BF16 LoRA)
  数据: 10K-100K 多模态指令数据
  时间: ~5-10 小时 (单 A100, 50K 数据)

LLaVA-7B 从零训练:
  Stage-1: 8×A100 × 5.5h (对齐 595K 数据)
  Stage-2: 8×A100 × 8h   (微调 665K 数据)
  总计: ~28 A100-hours

InternVL2-26B 微调:
  显存: ~40GB (LoRA) / 4×A100 (全量)
  数据: 50K+ 高质量数据
```

---

## 面试高频问答

**Q1：为什么 LLaVA 分两个阶段训练？**
> Stage-1 的目的是让 Projector 学会将 ViT 的视觉特征映射到 LLM 理解的语义空间（模态对齐）。如果直接从 Stage-2 开始，LLM 收到的是"噪声"visual tokens，训练信号混乱，容易导致 LLM 的语言能力退化。先对齐再微调是"先学走再学跑"的思路。

**Q2：Q-Former 和 MLP Projector 的核心区别？**
> Q-Former 通过 32 个 learnable queries 做 cross-attention 来"提炼"视觉信息，将任意数量的 visual tokens 压缩为固定 32 个 tokens。MLP Projector 是逐 token 映射，保留所有 visual tokens。Q-Former 更高效但可能丢失空间细节，MLP 保留更多信息但 token 数多。

**Q3：多模态微调时为什么要冻结 ViT？**
> ViT（尤其是 CLIP ViT）已在数亿图文对上预训练，具有强大的视觉特征提取能力。微调 ViT 有两个风险：1）小规模数据容易过拟合导致视觉能力退化；2）ViT 的参数量大（300M-2B），微调成本高。通常只有在视觉特征确实不够好时才解冻 ViT 的最后几层。

**Q4：如何解决多模态幻觉问题？**
> 1）提高对齐质量：Stage-1 用更高质量的图文对，确保 visual tokens 准确映射；2）数据增强：加入 negative samples（"图中没有大象"）；3）RLHF/DPO 对齐：用 AI 反馈纠正幻觉回复；4）训练时加入对比学习目标：增强图文一致性。

## 面试一句话
- "多模态训练分对齐和微调两个阶段：对齐让 ViT 输出被 LLM 理解，微调让模型学会执行视觉指令。Q-Former 压缩 visual tokens 高效但丢细节，MLP 保留全部但 token 多。冻结 ViT 避免退化是标准做法，核心挑战是幻觉控制和高分辨率处理。"
