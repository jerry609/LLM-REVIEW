# 预训练实战全流程：从零到一训练 nano-LLM

> 端到端预训练 Pipeline —— 环境配置 → 数据处理 → 模型定义 → 训练循环 → 监控 → 评估

---

## 一、预训练全景图

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌───────────────┐
│  数据准备    │ ──→ │  Tokenizer   │ ──→ │  模型定义     │ ──→ │   训练循环     │
│ 清洗/去重    │     │ 训练/加载     │     │ Config 配置   │     │ 优化/监控/存储  │
│ 配比/打包    │     │ 词表构建      │     │ 参数初始化    │     │ 断点续训       │
└─────────────┘     └──────────────┘     └──────────────┘     └───────────────┘
                                                                       │
                                                                       ▼
                                                              ┌───────────────┐
                                                              │   评估与导出   │
                                                              │ PPL/Benchmark  │
                                                              │ HF 格式导出    │
                                                              └───────────────┘
```

---

## 二、数据准备 Pipeline

### 2.1 数据来源与配比

| 数据类型 | 占比(典型) | 来源 | 作用 |
|---------|-----------|------|------|
| 网页文本 | 60-70% | CommonCrawl / FineWeb | 通用语言能力 |
| 代码 | 10-15% | GitHub / StarCoder 数据 | 推理 + 代码能力 |
| 书籍/论文 | 5-10% | Books3 / arXiv | 长文理解 + 知识 |
| 数学 | 3-5% | OpenWebMath / Proof-Pile | 数学推理 |
| 多语言 | 5-10% | CulturaX / CC-100 | 多语言能力 |
| 对话/指令 | 2-5% | 预训练末期混入 | 提前适配对话格式 |

### 2.2 数据清洗流水线

```python
# 标准清洗流程
def clean_pipeline(text: str) -> str | None:
    # 1. 去除 HTML 标签
    text = remove_html(text)

    # 2. 语言检测（只保留目标语言）
    if detect_lang(text) not in ['zh', 'en']:
        return None

    # 3. 质量过滤
    if len(text) < 50:                    # 太短
        return None
    if text.count('\n') / len(text) > 0.3: # 格式异常
        return None
    if perplexity_filter(text) > 10000:    # KenLM 过滤低质量
        return None

    # 4. 去重（MinHash / SimHash）
    if is_duplicate(text):
        return None

    # 5. PII 脱敏
    text = remove_pii(text)

    return text
```

### 2.3 数据打包 (Packing)

```python
# 将短文本拼接到固定长度，避免 padding 浪费
def pack_sequences(tokenized_texts, max_length=4096, eos_token_id=2):
    """文档拼接打包 —— 预训练标准做法"""
    buffer = []
    for tokens in tokenized_texts:
        buffer.extend(tokens + [eos_token_id])
        while len(buffer) >= max_length:
            yield buffer[:max_length]
            buffer = buffer[max_length:]
```

---

## 三、Tokenizer 训练

### 3.1 BPE Tokenizer 训练

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# 训练 BPE tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

trainer = trainers.BpeTrainer(
    vocab_size=32000,
    special_tokens=["<|begin_of_text|>", "<|end_of_text|>", "<|pad|>"],
    min_frequency=2,
)

tokenizer.train(files=["train_corpus.txt"], trainer=trainer)
tokenizer.save("tokenizer.json")
```

### 3.2 Tokenizer 评估指标

| 指标 | 公式 | 好的范围 |
|------|------|---------|
| 压缩率 | 字符数 / token 数 | 中文 1.5-2.5，英文 3.5-4.5 |
| 词表覆盖率 | 已知 token 数 / 总 token 数 | >99.5% |
| UNK 率 | UNK 数 / 总 token 数 | <0.1% |

---

## 四、模型配置与初始化

### 4.1 模型配置 (Config)

```python
from dataclasses import dataclass

@dataclass
class NanoLLMConfig:
    # 模型架构
    vocab_size: int = 32000
    d_model: int = 2048
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 4        # GQA: 4 KV groups
    d_ff: int = 5504           # SwiGLU: d_ff ≈ 2.7 * d_model
    max_seq_len: int = 4096
    rope_theta: float = 500000.0
    norm_eps: float = 1e-5

    # 训练超参
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 2000
    total_steps: int = 100000
    batch_size: int = 4         # per GPU
    gradient_accumulation: int = 8  # 等效 batch = 4*8*n_gpu
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    dtype: str = "bfloat16"
```

### 4.2 参数初始化策略

```python
def init_weights(module):
    """标准初始化 —— 参考 GPT-2 / Llama"""
    if isinstance(module, nn.Linear):
        # Xavier 或正态初始化
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

# 残差层缩放初始化 (GPT-2 风格)
# 每个残差连接的输出层权重 × 1/√(2*n_layers)
for name, param in model.named_parameters():
    if "w2.weight" in name or "wo.weight" in name:  # FFN/Attn 输出
        torch.nn.init.normal_(param, mean=0.0, std=0.02 / (2 * n_layers)**0.5)
```

### 4.3 参数量估算

```
参数量 ≈ 12 × n_layers × d_model²   (粗略公式)

详细分解 (Llama 风格):
- Embedding:      V × d = 32000 × 2048 = 65.5M
- 每层 Attention:  4 × d² = 4 × 2048² = 16.8M  (Q/K/V/O)
- 每层 FFN:        3 × d × d_ff ≈ 3 × 2048 × 5504 = 33.8M  (SwiGLU 有 3 个矩阵)
- 每层 Norm:       2 × d = 4K (可忽略)
- 每层合计:        ≈ 50.6M
- 24 层:           ≈ 1.21B
- LM Head:         共享 Embedding = 0
- 总计:            ≈ 1.28B 参数
```

---

## 五、训练循环核心

### 5.1 完整训练脚本骨架

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def train(config: NanoLLMConfig):
    # 1. 分布式初始化
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")

    # 2. 模型
    model = NanoLLM(config).to(device)
    model = DDP(model, device_ids=[rank])

    # 3. 优化器
    # 区分需要 weight_decay 和不需要的参数
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ], lr=config.max_lr, betas=(0.9, 0.95), eps=1e-8)

    # 4. 训练循环
    for step in range(config.total_steps):
        lr = get_cosine_lr(step, config)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # 梯度累积
        optimizer.zero_grad()
        total_loss = 0
        for micro_step in range(config.gradient_accumulation):
            batch = next(data_iter)
            input_ids = batch['input_ids'].to(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, config.vocab_size),
                    input_ids[:, 1:].reshape(-1)
                ) / config.gradient_accumulation

            loss.backward()
            total_loss += loss.item()

        # 梯度裁剪 + 优化
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        # 5. 日志与监控
        if step % 10 == 0 and rank == 0:
            ppl = math.exp(total_loss)
            grad_norm = get_grad_norm(model)
            log(f"step={step} loss={total_loss:.4f} ppl={ppl:.1f} "
                f"lr={lr:.2e} grad_norm={grad_norm:.2f}")

        # 6. 定期保存 checkpoint
        if step % 1000 == 0:
            save_checkpoint(model, optimizer, step)
```

### 5.2 训练监控指标

| 指标 | 含义 | 正常范围 | 异常处理 |
|------|------|---------|---------|
| **loss** | 交叉熵损失 | 持续下降 | 突升 → 检查数据/lr |
| **PPL** | exp(loss) | 下降趋势 | 回升 → 过拟合 |
| **grad_norm** | 梯度 L2 范数 | 0.1-10 | >100 → 学习率过大 |
| **lr** | 学习率 | 按 schedule 变化 | — |
| **tokens/sec** | 训练吞吐 | 稳定 | 骤降 → GPU 利用率问题 |
| **GPU 利用率** | nvidia-smi | >80% | <50% → 数据加载瓶颈 |
| **MFU** | 模型 FLOPs 利用率 | 30-55% | <20% → 通信/IO 瓶颈 |

### 5.3 MFU (Model FLOPs Utilization) 计算

```python
def estimate_mfu(model_params, batch_tokens_per_sec, gpu_flops_bfloat16):
    """
    MFU = 实际 FLOPs / 理论峰值 FLOPs
    每个 token 的 FLOPs ≈ 6N (N=参数量, 2 前向 + 4 反向)
    """
    flops_per_token = 6 * model_params
    actual_flops = flops_per_token * batch_tokens_per_sec
    mfu = actual_flops / gpu_flops_bfloat16
    return mfu

# 示例：1.3B 模型，A100 80GB
# batch_tokens_per_sec = 50000
# A100 BF16 峰值 = 312 TFLOPS
# MFU = 6 * 1.3e9 * 50000 / 312e12 ≈ 0.125 → 12.5%（需要优化）
```

---

## 六、断点续训 (Resume Training)

```python
def save_checkpoint(model, optimizer, step, path="checkpoints"):
    """保存完整训练状态"""
    ckpt = {
        'step': step,
        'model_state_dict': model.module.state_dict(),  # DDP 需要 .module
        'optimizer_state_dict': optimizer.state_dict(),
        'rng_state': torch.cuda.get_rng_state(),
    }
    torch.save(ckpt, f"{path}/step_{step}.pt")

def load_checkpoint(model, optimizer, path):
    """恢复训练状态"""
    ckpt = torch.load(path, map_location='cpu')
    model.module.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    torch.cuda.set_rng_state(ckpt['rng_state'])
    return ckpt['step']
```

---

## 七、评估与导出

### 7.1 评估方法

```python
@torch.inference_mode()
def evaluate(model, eval_dataloader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    for batch in eval_dataloader:
        input_ids = batch['input_ids'].to(device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                input_ids[:, 1:].reshape(-1),
                reduction='sum'
            )
        total_loss += loss.item()
        total_tokens += (input_ids[:, 1:] != -100).sum().item()
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return {'loss': avg_loss, 'ppl': ppl}
```

### 7.2 导出为 Hugging Face 格式

```python
from transformers import AutoConfig, AutoModelForCausalLM

def export_to_hf(model, tokenizer, output_dir):
    """导出为 HF 格式，可直接用 transformers 加载"""
    # 1. 保存模型权重
    model.save_pretrained(output_dir)

    # 2. 保存 tokenizer
    tokenizer.save_pretrained(output_dir)

    # 3. 保存配置
    config = AutoConfig.from_pretrained(output_dir)
    config.save_pretrained(output_dir)

    # 验证
    loaded = AutoModelForCausalLM.from_pretrained(output_dir)
    print(f"Exported to {output_dir}, params: {sum(p.numel() for p in loaded.parameters())/1e9:.2f}B")
```

---

## 八、Scaling Law 实践指导

### 8.1 资源规划

```
给定计算预算 C (FLOPs):
- 最优参数量: N_opt ∝ C^0.5   (Chinchilla)
- 最优数据量: D_opt ∝ C^0.5
- 最优 tokens: D = 20 × N     (每参数 20 tokens)

示例：
- 1B 模型: 需要 20B tokens, 约 3.6e19 FLOPs, 1×A100 ≈ 32 小时
- 7B 模型: 需要 140B tokens, 约 5e21 FLOPs, 8×A100 ≈ 7 天
- 70B 模型: 需要 1.4T tokens, 约 5e23 FLOPs, 256×H100 ≈ 2 周
```

### 8.2 超参 Scaling

| 参数量 | 学习率 | Batch Size (tokens) | 上下文长度 |
|--------|--------|-------------------|-----------|
| 100M-500M | 6e-4 ~ 1e-3 | 256K ~ 512K | 2048 |
| 1B-3B | 3e-4 ~ 6e-4 | 512K ~ 1M | 4096 |
| 7B-13B | 1e-4 ~ 3e-4 | 1M ~ 4M | 4096-8192 |
| 30B-70B | 5e-5 ~ 1.5e-4 | 4M ~ 16M | 8192+ |

---

## 面试高频问答

**Q1：预训练数据清洗最关键的步骤是什么？**
> 去重（MinHash 近似去重 + 精确去重）和质量过滤（KenLM PPL 过滤 + 规则过滤）。去重不充分会导致模型记忆而非学习，质量过滤不足会引入噪声。Llama-3 在数据工程上投入巨大，最终用了 15T tokens。

**Q2：为什么需要 WSD (Warmup-Stable-Decay) 或 Cosine 学习率调度？**
> Warmup 避免训练初期梯度不稳定导致的参数偏移，Cosine/Decay 阶段逐步降低学习率让模型收敛到更好的局部最优。Llama-3 在最后 5% 步用线性 decay 到 0，DeepSeek-V3 用 WSD 在 stable 阶段持续训练。

**Q3：数据配比 (Data Mix) 如何确定？**
> 经验法则 + Scaling Law 实验。先用小模型（100M-1B）在不同配比下训练，观察下游 Benchmark 变化，选出最优配比后再用大模型全量训练。代码数据通常占 10-20%，但对推理能力提升显著。

**Q4：MFU 低的常见原因？**
> 1）通信瓶颈（TP 跨节点 AllReduce 太频繁）；2）数据加载慢（预处理没做好/IO 瓶颈）；3）batch size 太小导致 GPU 利用率低；4）梯度累积步数设置不当。A100 上好的 MFU 约 40-55%。

## 面试一句话
- "预训练的核心是数据质量 > 数据数量，Chinchilla Law 指导 N-D 配比，Cosine 学习率 + BF16 混合精度 + 梯度累积是标准训练配方，MFU 是衡量训练效率的核心指标。"
