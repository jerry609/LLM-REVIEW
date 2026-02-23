# PyTorch 框架速通

> 从 Tensor 操作到完整模型训练，覆盖 LLM 开发所需的 PyTorch 核心 API

---

## 一、Tensor 基础操作

### 1.1 创建 Tensor

```python
import torch

# 常用创建方式
x = torch.randn(4, 128, 512)          # 正态分布随机张量 (B, L, d)
x = torch.zeros(4, 128, 512)          # 全零
x = torch.ones(4, 128, 512)           # 全一
x = torch.arange(0, 128)              # [0, 1, 2, ..., 127]
x = torch.tensor([1.0, 2.0, 3.0])    # 从列表创建
x = torch.eye(128)                     # 128×128 单位矩阵

# 指定设备和精度
x = torch.randn(4, 128, 512, device='cuda', dtype=torch.bfloat16)
```

### 1.2 形状操作 —— LLM 开发最频繁的操作

```python
x = torch.randn(4, 128, 512)  # (B, L, d_model)
n_heads = 8
d_head = 64

# view / reshape：改变形状（不复制数据）
x_heads = x.view(4, 128, n_heads, d_head)      # (B, L, H, d_h)

# transpose / permute：交换维度
x_heads = x_heads.transpose(1, 2)               # (B, H, L, d_h)
x_heads = x_heads.permute(0, 2, 1, 3)           # 等价写法

# contiguous：确保内存连续（transpose 后常需要）
x_flat = x_heads.transpose(1, 2).contiguous().view(4, 128, 512)

# squeeze / unsqueeze：增减维度
mask = torch.ones(128, 128).unsqueeze(0).unsqueeze(0)  # (1, 1, L, L) 用于广播

# expand / repeat：广播扩展
mask = mask.expand(4, n_heads, -1, -1)  # (B, H, L, L) 不复制内存
```

### 1.3 索引与切片

```python
x = torch.randn(4, 128, 512)

# 基础切片
first_token = x[:, 0, :]             # (B, d) 第一个 token
last_tokens = x[:, -10:, :]          # (B, 10, d) 最后 10 个 token

# gather —— KV Cache 场景中按 index 取值
indices = torch.tensor([0, 5, 10, 15])
selected = x[:, indices, :]           # (B, 4, d) 选择特定位置的 token

# masked_fill —— Causal Mask 的标准实现
scores = torch.randn(4, 8, 128, 128)
causal_mask = torch.triu(torch.ones(128, 128, dtype=torch.bool), diagonal=1)
scores = scores.masked_fill(causal_mask, float('-inf'))
```

### 1.4 矩阵运算

```python
Q = torch.randn(4, 8, 128, 64)  # (B, H, L, d_h)
K = torch.randn(4, 8, 128, 64)
V = torch.randn(4, 8, 128, 64)

# @ 是 matmul 的简写，支持 batch 维度
attn_scores = Q @ K.transpose(-2, -1) / (64 ** 0.5)  # (B, H, L, L)
attn_weights = torch.softmax(attn_scores, dim=-1)
attn_output = attn_weights @ V                         # (B, H, L, d_h)

# einsum —— 灵活的张量运算
attn_scores = torch.einsum('bhld,bhmd->bhlm', Q, K) / (64 ** 0.5)
```

---

## 二、自动求导 (Autograd)

### 2.1 基础用法

```python
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x ** 2 + 3 * x
loss = y.sum()

# 反向传播 —— 自动计算 ∂loss/∂x
loss.backward()
print(x.grad)  # tensor([7., 9.]) = 2*x + 3

# 清零梯度（每个 step 前必须做）
x.grad.zero_()
```

### 2.2 计算图与梯度控制

```python
# 不追踪梯度 —— 推理时使用，节省显存
with torch.no_grad():
    logits = model(input_ids)

# 等价装饰器
@torch.inference_mode()
def generate(model, input_ids):
    return model(input_ids)

# 梯度累积 —— 小显存模拟大 batch
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 梯度检查点 —— 用计算换显存（节省 ~60% 显存）
from torch.utils.checkpoint import checkpoint
output = checkpoint(self.layer, hidden_states, use_reentrant=False)
```

### 2.3 混合精度训练

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    # 前向用 FP16/BF16
    with autocast(device_type='cuda', dtype=torch.bfloat16):
        logits = model(batch['input_ids'])
        loss = criterion(logits, batch['labels'])

    # 反向用 FP32 梯度（GradScaler 自动处理）
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

---

## 三、模型搭建 (nn.Module)

### 3.1 基础模型结构

```python
import torch.nn as nn
import torch.nn.functional as F

class SimpleLLM(nn.Module):
    def __init__(self, vocab_size=32000, d_model=512, n_layers=6, n_heads=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 权重共享（Embedding 和 LM Head）
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids, labels=None):
        h = self.embedding(input_ids)            # (B, L) → (B, L, d)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        logits = self.lm_head(h)                 # (B, L, V)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100
            )
        return {"logits": logits, "loss": loss}
```

### 3.2 常用模块速查

| 模块 | 用途 | LLM 场景 |
|------|------|---------|
| `nn.Embedding` | 词嵌入 | token → vector |
| `nn.Linear` | 全连接层 | QKV 投影、FFN、LM Head |
| `nn.LayerNorm` | 层归一化 | BERT/GPT 标准 |
| `nn.RMSNorm` | 均方根归一化 | Llama/Qwen/DeepSeek |
| `nn.Dropout` | 随机丢弃 | 训练正则化 |
| `nn.ModuleList` | 模块列表 | 堆叠多个 Transformer Block |
| `nn.Parameter` | 可学习参数 | 自定义权重（如 RoPE freq） |

### 3.3 模型参数操作

```python
model = SimpleLLM()

# 参数统计
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total_params/1e6:.1f}M, Trainable: {trainable_params/1e6:.1f}M")

# 冻结参数（LoRA 微调时冻结底座）
for param in model.parameters():
    param.requires_grad = False

# 仅解冻特定层
for name, param in model.named_parameters():
    if "lora" in name:
        param.requires_grad = True

# 保存 / 加载模型
torch.save(model.state_dict(), "checkpoint.pt")
model.load_state_dict(torch.load("checkpoint.pt"))

# 保存完整检查点（含优化器状态）
torch.save({
    'step': global_step,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss.item(),
}, "full_checkpoint.pt")
```

---

## 四、数据加载 (DataLoader)

### 4.1 自定义 Dataset

```python
from torch.utils.data import Dataset, DataLoader
import json

class SFTDataset(Dataset):
    """SFT 训练数据集"""
    def __init__(self, data_path, tokenizer, max_length=2048):
        with open(data_path, 'r') as f:
            self.data = [json.loads(line) for line in f]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = f"<|im_start|>user\n{item['instruction']}<|im_end|>\n" \
               f"<|im_start|>assistant\n{item['output']}<|im_end|>"
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = encodings['input_ids'].squeeze()
        labels = input_ids.clone()
        # 用户部分不计算 loss
        user_len = len(self.tokenizer.encode(f"<|im_start|>user\n{item['instruction']}<|im_end|>\n<|im_start|>assistant\n"))
        labels[:user_len] = -100
        return {'input_ids': input_ids, 'labels': labels, 'attention_mask': encodings['attention_mask'].squeeze()}
```

### 4.2 DataLoader 高效配置

```python
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,                # 训练时打乱
    num_workers=4,               # 多进程加载（CPU 密集型预处理）
    pin_memory=True,             # 加速 CPU→GPU 传输
    drop_last=True,              # 丢弃最后不完整 batch（分布式训练需要）
    prefetch_factor=2,           # 预取 batch 数
)

# 分布式数据加载
from torch.utils.data.distributed import DistributedSampler
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
dataloader = DataLoader(dataset, batch_size=4, sampler=sampler)
```

---

## 五、GPU 操作与多卡

### 5.1 设备管理

```python
# 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 指定 GPU
model = model.to('cuda:0')

# 显存监控
print(f"已分配: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"已缓存: {torch.cuda.memory_reserved()/1e9:.2f} GB")

# 显存清理
torch.cuda.empty_cache()
```

### 5.2 Hugging Face 模型加载

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型 + tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.bfloat16,    # BF16 精度
    device_map="auto",              # 自动分配多卡
    attn_implementation="flash_attention_2",  # FlashAttention
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# 推理
inputs = tokenizer("Hello, world!", return_tensors="pt").to(model.device)
with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 六、关键 API 速查表

| 类别 | API | 说明 |
|------|-----|------|
| **张量** | `torch.randn / zeros / ones` | 创建张量 |
| | `.view() / .reshape()` | 改变形状 |
| | `.transpose() / .permute()` | 交换维度 |
| | `.contiguous()` | 确保内存连续 |
| | `@ / torch.matmul` | 矩阵乘法 |
| | `.masked_fill()` | 条件填充（因果 mask） |
| **求导** | `loss.backward()` | 反向传播 |
| | `optimizer.zero_grad()` | 清零梯度 |
| | `torch.no_grad()` | 关闭梯度追踪 |
| | `torch.inference_mode()` | 推理模式 |
| **模型** | `nn.Module` | 模型基类 |
| | `nn.Linear / Embedding` | 核心层 |
| | `model.parameters()` | 参数迭代器 |
| | `model.state_dict()` | 参数字典 |
| **数据** | `Dataset / DataLoader` | 数据管道 |
| | `pin_memory / num_workers` | 加载优化 |
| **精度** | `torch.bfloat16 / float16` | 混合精度 |
| | `autocast / GradScaler` | AMP 训练 |

---

## 面试一句话
- "PyTorch 的自动求导通过计算图实现链式法则，nn.Module 是所有模型的基类，DataLoader 配合 pin_memory 和多 workers 实现高效数据流水线，混合精度训练用 BF16 前向 + FP32 梯度平衡速度和精度。"
