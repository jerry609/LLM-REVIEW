# PEFT 方法全景对比

> LoRA / QLoRA / Adapter / IA³ / Prefix-Tuning / Prompt-Tuning / P-Tuning v1/v2 —— 参数高效微调方法论

---

## 一、PEFT 全景图

```
                          PEFT (Parameter-Efficient Fine-Tuning)
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         │                             │                             │
    加法类 (Additive)              重参数化类                    提示类 (Prompt-based)
    在模型中插入新参数             修改现有参数表示               在输入/隐空间加前缀
         │                             │                             │
    ├─ Adapter                    ├─ LoRA                      ├─ Prompt Tuning
    ├─ AdapterFusion              ├─ QLoRA                     ├─ Prefix Tuning
    └─ IA³                        ├─ DoRA                      ├─ P-Tuning v1
                                  └─ LoRA+                     └─ P-Tuning v2
```

---

## 二、各方法详解

### 2.1 LoRA (Low-Rank Adaptation) ⭐ 当前主流

```
原始权重:  W ∈ R^{d×d}  (冻结)
增量矩阵:  ΔW = B·A，其中 A ∈ R^{d×r}, B ∈ R^{r×d}
前向:      h = (W + α/r · BA) x
推理合并:  W' = W + α/r · BA → 无额外延迟
```

**关键参数：**
| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `r` (rank) | 低秩维度 | 8-64（任务简单→8，复杂→64） |
| `alpha` | 缩放因子 | 通常 = r 或 2r |
| `target_modules` | 插入层 | `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` |
| `dropout` | LoRA 层 dropout | 0.05-0.1 |

**初始化：**
- A：Kaiming 正态分布
- B：全零 → 训练开始时 ΔW = 0，等于原模型

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# trainable params: 13.6M || all params: 7.6B || trainable%: 0.18%
```

### 2.2 QLoRA (Quantized LoRA) ⭐ 单卡微调利器

```
底座量化:  W → 4-bit NF4 量化 (仅存储，计算时反量化)
LoRA层:    保持 BF16/FP16
梯度计算:  反量化 → 前向 → 反向 → 仅更新 LoRA 参数
```

**核心创新：**
1. **NF4 量化**：正态分布最优的 4-bit 量化方案
2. **双重量化**：对量化常数再做一次量化
3. **分页优化器**：利用 CPU 内存处理梯度检查点的峰值显存

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  # 双重量化
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B", quantization_config=bnb_config
)
# 7B 模型显存: FP16=14GB → QLoRA=4.5GB
```

### 2.3 DoRA (Weight-Decomposed Low-Rank Adaptation)

```
将权重分解为 方向 + 幅度：
W = m · (W₀ + BA) / ||W₀ + BA||

m: 可学习的幅度向量 (magnitude)
BA: LoRA 的低秩增量 (direction)
```

- 比 LoRA 效果更好，接近全量微调
- 参数量仅比 LoRA 多一个向量 `m`
- 缺点：训练略慢（多一次归一化计算）

### 2.4 Adapter

```
原始层输出:  h = Attention(x) 或 FFN(x)
Adapter:     h' = h + f(h W_down) W_up

W_down ∈ R^{d×r}  (降维)
W_up ∈ R^{r×d}    (升维)
f: 非线性激活 (ReLU / GELU)
```

- 在每个 Transformer 子层后插入 bottleneck
- 训练参数量 ≈ 2 × n_layers × d × r
- **缺点**：增加推理延迟（序列化瓶颈）

```
                 ┌──────────────────┐
            x ──→│   Attention      │──→ + ──→ LayerNorm ──→ ...
                 └──────────────────┘    ↑
                                         │
                 ┌──────────────────┐    │
            h ──→│  Adapter         │────┘
                 │  Down → Act → Up │
                 └──────────────────┘
```

### 2.5 IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)

```
修改:  K' = l_k ⊙ K    (Key 激活缩放)
       V' = l_v ⊙ V    (Value 激活缩放)
       FFN': l_ff ⊙ FFN(x)  (FFN 输出缩放)

l_k, l_v, l_ff ∈ R^d  (可学习的向量)
```

- 只训练 3 个向量/层 → 极少参数（0.01% 级别）
- 适合小数据、快速适配
- 效果不如 LoRA，适合极端资源约束场景

### 2.6 Prefix Tuning

```
给每个注意力层的 K, V 前面拼接 可学习的虚拟 token：
K' = [Prefix_K ; K]    # (p+L, d_h)
V' = [Prefix_V ; V]    # (p+L, d_h)

p: 前缀长度 (prefix length, 通常 10-200)
```

- 每层独立的 prefix → 表达力强于 Prompt Tuning
- 通过 MLP 重参数化前缀（训练稳定性）
- **缺点**：占用上下文窗口，推理时等效缩短了可用序列长度

```python
from peft import PrefixTuningConfig

prefix_config = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,        # 前缀长度
    prefix_projection=True,        # 使用 MLP 重参数化
    encoder_hidden_size=1024,
)
```

### 2.7 Prompt Tuning

```
在输入 Embedding 前拼接 可学习的软提示向量：
input' = [Soft_Prompt ; x_embed]

Soft_Prompt ∈ R^{p×d}  (p 个虚拟 token 的 embedding)
```

- 只在 Embedding 层加入 → 参数最少
- 随模型规模增大效果接近全量微调（>10B 时）
- 对小模型效果差

### 2.8 P-Tuning v1

```
用 LSTM/MLP 生成连续提示向量：
prompt_embed = LSTM(learnable_tokens)
input' = [prompt_embed ; x_embed]
```

- 比 Prompt Tuning 多了生成器网络 → 更强表达力
- 但仍只作用于输入层

### 2.9 P-Tuning v2 ⭐

```
在每一层都加入可学习前缀 (等价于 Prefix Tuning)：
每层的 K' = [Prefix_K_i ; K]
每层的 V' = [Prefix_V_i ; V]
```

- P-Tuning v1 的升级版，从 "仅输入层" → "每层都加"
- 效果接近全量微调
- 本质上 = Prefix Tuning 的重新实验

---

## 三、全方法对比表

| 方法 | 可训练参数 | 位置 | 推理延迟 | 效果 | 显存需求 |
|------|-----------|------|---------|------|---------|
| **全量微调** | 100% | 所有层 | 无额外 | **最好** | 极高 |
| **LoRA** ⭐ | 0.1-1% | QKV/FFN 投影 | **无额外** | 接近全量 | 低 |
| **QLoRA** ⭐ | 0.1-1% | 同上 + 4bit 底座 | **无额外** | 接近 LoRA | **极低** |
| **DoRA** | 0.1-1% + d | 同 LoRA | **无额外** | > LoRA | 低 |
| **Adapter** | 0.5-3% | 每层后插入 | **有延迟** | 较好 | 低 |
| **IA³** | 0.01% | KVF 缩放向量 | 无额外 | 中等 | 极低 |
| **Prefix Tuning** | 0.1% | 每层 KV 前缀 | 占上下文 | 较好 | 低 |
| **Prompt Tuning** | 0.01% | 仅输入层 | 占上下文 | 大模型好 | 极低 |
| **P-Tuning v2** | 0.1% | 每层前缀 | 占上下文 | 较好 | 低 |

---

## 四、选型指南

### 4.1 场景推荐

| 场景 | 推荐方法 | 理由 |
|------|---------|------|
| **单卡微调 7B-70B** | QLoRA | 4-bit 量化 + LoRA，显存极低 |
| **追求最佳效果** | LoRA (r=64) / DoRA | 覆盖所有投影层 |
| **多任务快速切换** | LoRA | 不同任务不同 LoRA adapter，按需加载 |
| **极端资源约束** | IA³ / Prompt Tuning | 参数量最小 |
| **代码/数学任务** | LoRA + 全 target | 需要修改模型内部表示 |
| **分类/NER 任务** | P-Tuning v2 | 轻量且适合判别式任务 |
| **多 LoRA 在线服务** | LoRA | vLLM/SGLang 原生支持多 LoRA 切换 |

### 4.2 关键超参选择

```
LoRA rank 选择:
├── r=8:   简单任务（情感分类、意图识别）
├── r=16:  中等任务（指令跟随、对话）
├── r=32:  复杂任务（代码、数学、推理）
└── r=64:  追求极限效果（接近全量微调）

Target Modules 选择:
├── 最小: ["q_proj", "v_proj"]          → 最省显存
├── 标准: ["q_proj", "k_proj", "v_proj", "o_proj"]  → 推荐
└── 全覆盖: + ["gate_proj", "up_proj", "down_proj"]  → 效果最好
```

---

## 五、多 LoRA 推理服务

```
基础模型 (冻结) → 共享权重
           │
    ┌──────┼──────┐
    │      │      │
 LoRA_A  LoRA_B  LoRA_C    ← 不同任务/客户的适配器
 (法律)   (医疗)   (金融)
```

- vLLM 原生支持：一个 GPU 加载基础模型 + 多个 LoRA
- 请求级路由：根据请求标签选择对应 LoRA
- 显存开销：每个 LoRA adapter 仅几十 MB

---

## 面试高频问答

**Q1：LoRA 为什么 B 初始化为全零？**
> 训练开始时 ΔW = BA = 0，模型输出与原始预训练模型完全一致。这保证了微调是从预训练检查点开始的"平滑过渡"，而不是从随机初始化开始。

**Q2：QLoRA 的 NF4 和普通 INT4 有什么区别？**
> NF4 (NormalFloat4) 是基于正态分布设计的 4-bit 数据类型。预训练权重近似服从正态分布，NF4 的量化桶在分布密集的中心区域更细、在稀疏的尾部更粗，因此量化误差更小。

**Q3：Adapter 和 LoRA 最大的区别是什么？**
> Adapter 是串行插入的 bottleneck（经过原层 → 再经过 Adapter → 加回），增加了推理延迟。LoRA 是并行的低秩增量（与原层权重相加），推理时可以合并到原权重中，零额外延迟。

**Q4：什么时候用 Prefix Tuning 而不是 LoRA？**
> Prefix Tuning 适合判别式 NLU 任务（分类、NER）和资源极度受限的场景。LoRA 在生成式任务和大模型微调上优势明显。目前工业界 90%+ 的微调使用 LoRA/QLoRA。

## 面试一句话
- "LoRA 通过低秩分解冻结底座只训练增量，QLoRA 进一步 4-bit 量化底座实现单卡微调 70B，推理时 LoRA 可合并到原权重无额外延迟。Adapter 是串行的会增加延迟，Prefix/Prompt Tuning 占用上下文窗口。"
