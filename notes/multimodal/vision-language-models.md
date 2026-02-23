# ViT / CLIP / BLIP / LLaVA 架构详解

> 从视觉编码到多模态对齐 —— 视觉语言模型的核心架构演进

---

## 一、多模态 LLM 架构全景图

```
                    视觉语言模型 (VLM) 演进
                          │
    ┌─────────────────────┼─────────────────────────┐
    │                     │                         │
 ViT (2020)          CLIP (2021)              BLIP-2 (2023)
 纯视觉编码          对比学习对齐             Q-Former 桥接
    │                     │                         │
    │              ┌──────┴──────┐                   │
    │              │             │                   │
    └──────────────┤  LLaVA (2023)  ├────────────────┘
                   │  简洁统一架构   │
                   │  ViT + Projector + LLM  │
                   └─────────────┘
                          │
              ┌───────────┼───────────┐
              │           │           │
         Qwen2-VL    InternVL    LLaVA-NeXT
         动态分辨率   强视觉编码   高清图理解
```

---

## 二、ViT (Vision Transformer)

### 2.1 核心思想

> "一张图片 = 一段文字"：将图像切成 patch，当作 token 序列送入 Transformer

```
输入图像: 224×224×3
    ↓  切分为 16×16 patch
Patch 序列: 196 个 patch (14×14)，每个 patch = 16×16×3 = 768 维
    ↓  线性投影
Token 序列: [CLS] + 196 个 patch token，每个 ∈ R^d
    ↓  + Position Embedding (可学习的 1D 位置)
    ↓  N 层 Transformer Encoder (标准 Self-Attention + FFN)
    ↓
输出: [CLS] token 作为图像全局表示 / 所有 token 作为空间特征
```

### 2.2 数学公式

```
Patch Embedding:
  x_patches = Split(Image, patch_size=P)    → N个 patch
  z_0 = [x_cls; x_1^p E; x_2^p E; ...; x_N^p E] + E_pos

  E ∈ R^{P²·C × D}  (线性投影矩阵)
  E_pos ∈ R^{(N+1) × D}  (位置编码)

Transformer Encoder (L层):
  z'_l = MSA(LN(z_{l-1})) + z_{l-1}
  z_l = FFN(LN(z'_l)) + z'_l

输出:
  y = LN(z_L^0)  → [CLS] token 的最终表示
```

### 2.3 关键参数

| 模型 | Patch Size | Layers | d_model | Heads | 参数量 | 输入分辨率 | Tokens 数 |
|------|-----------|--------|---------|-------|--------|----------|----------|
| ViT-B/16 | 16 | 12 | 768 | 12 | 86M | 224 | 196 |
| ViT-L/14 | 14 | 24 | 1024 | 16 | 307M | 224 | 256 |
| ViT-G/14 | 14 | 40 | 1408 | 16 | 1.8B | 224 | 256 |
| ViT-bigG/14 | 14 | 48 | 1664 | 16 | 2.5B | 224 | 256 |

---

## 三、CLIP (Contrastive Language-Image Pre-training)

### 3.1 核心思想

> 对比学习：让匹配的图文对在嵌入空间中靠近，不匹配的远离

```
         图像                  文本
          │                     │
      ViT Encoder          Text Encoder (Transformer)
          │                     │
    Image Embedding        Text Embedding
          │                     │
          └────── 余弦相似度 ────┘
              对比学习 (InfoNCE Loss)
```

### 3.2 训练目标

```
给定 batch of N 对 (image_i, text_i):

相似度矩阵 S ∈ R^{N×N}:
  S[i][j] = cosine_sim(image_emb_i, text_emb_j) / τ

损失函数 (对称 InfoNCE):
  L_i2t = -1/N ∑_i log(exp(S[i][i]) / ∑_j exp(S[i][j]))   ← 图→文
  L_t2i = -1/N ∑_i log(exp(S[i][i]) / ∑_j exp(S[j][i]))   ← 文→图
  L_CLIP = (L_i2t + L_t2i) / 2
```

### 3.3 CLIP 的价值

| 能力 | 说明 |
|------|------|
| **零样本分类** | 构造文本 "a photo of a {class}" 即可分类 |
| **图文检索** | 图搜文、文搜图 |
| **视觉编码器** | 提供强大的视觉特征 → 被 LLaVA 等后续模型复用 |
| **开放词汇检测** | 不受固定类别限制 |

### 3.4 CLIP 的局限

- 对比学习只做全局匹配 → 缺乏细粒度区域理解
- 不能做生成任务（只能匹配，不能生成文本描述）
- 对空间关系理解较弱（"左边的猫" vs "右边的猫"）

---

## 四、BLIP / BLIP-2

### 4.1 BLIP (Bootstrapping Language-Image Pre-training)

```
三个训练目标联合优化:
1. ITC (Image-Text Contrastive):  同 CLIP，对比学习
2. ITM (Image-Text Matching):     二分类，判断图文是否匹配
3. LM  (Language Modeling):       给定图像，生成文本描述 (Caption)

CapFilt (Captioning + Filtering):
  用训练好的模型给网络图片生成 caption → 过滤低质量图文对
```

### 4.2 BLIP-2 ⭐ Q-Former 架构

> 关键创新：用轻量级的 Q-Former 桥接冻结的视觉编码器和冻结的 LLM

```
                冻结的 ViT          Q-Former          冻结的 LLM
                   │               (可训练)              │
Image → [ViT] → visual tokens → [Q-Former] → query tokens → [LLM] → text
         ↑                          ↑                         ↑
       冻结                    32 个可学习                   冻结
       1.3B                    queries                     >7B
                              仅训练 188M
```

### 4.3 Q-Former 详解

```
Q-Former = 轻量 Transformer (32 个 learnable query tokens)

两阶段训练:

Stage 1: 视觉-语言表示学习（冻结 ViT，训练 Q-Former）
  ├── ITC：query tokens 和 text tokens 对比学习
  ├── ITM：判断图文匹配
  └── ITG：Image-grounded Text Generation

Stage 2: 视觉-语言生成学习（冻结 ViT + LLM，训练 Q-Former + FC）
  └── query tokens → FC 投影 → LLM 的 input embedding space
      → LLM 生成文本回答
```

**Q-Former 的核心价值：**
- 将任意分辨率的 visual tokens (196-2000+) 压缩为固定 32 个 query tokens
- 只训练 188M 参数即可桥接 ViT 和 LLM
- 高效利用预训练的视觉和语言模型

---

## 五、LLaVA (Large Language and Vision Assistant) ⭐

### 5.1 架构

> "简单才是终极的复杂" —— 最简洁的 VLM 架构

```
Image → [CLIP ViT-L/14] → visual features → [Linear Projector] → visual tokens
                                                                        ↓
Text  → [Tokenizer]     → text tokens      ─────────────────────→ [Concat]
                                                                        ↓
                                                              [LLM (Vicuna/Llama)]
                                                                        ↓
                                                                   Output text
```

### 5.2 两阶段训练

```
Stage 1: 特征对齐预训练 (Feature Alignment)
  - 数据: 558K 图文对 (CC3M 筛选)
  - 冻结: ViT ❄️ + LLM ❄️
  - 训练: 仅 Projector (MLP)
  - 目标: 让 visual tokens 对齐到 LLM 的 embedding space
  - 训练时间: ~5.5 小时 / 8×A100

Stage 2: 视觉指令微调 (Visual Instruction Tuning)
  - 数据: 150K 多模态指令数据 (GPT-4 生成)
  - 冻结: ViT ❄️
  - 训练: Projector 🔥 + LLM 🔥 (或 LoRA)
  - 目标: 让模型理解并执行视觉相关的指令
  - 数据类型: 图像描述、视觉问答、复杂推理
```

### 5.3 LLaVA 的演进

| 版本 | 视觉编码器 | Projector | LLM | 分辨率 | 关键改进 |
|------|-----------|-----------|-----|--------|---------|
| LLaVA-1.0 | CLIP ViT-L/14 | Linear | Vicuna-13B | 224 | 首次提出 |
| LLaVA-1.5 | CLIP ViT-L/14-336 | 2层 MLP | Vicuna-7/13B | 336 | 更好的 projector |
| LLaVA-NeXT | CLIP ViT-L/14-336 | 2层 MLP | Llama-3/Qwen | 动态 | 高清 AnyRes |
| LLaVA-OneVision | SigLIP | 2层 MLP | Qwen2 | 动态 | 统一图/视频 |

### 5.4 Projector 设计对比

| Projector | 结构 | 效果 | 使用者 |
|-----------|------|------|--------|
| Linear | 单层线性 | 基准 | LLaVA-1.0 |
| **2层 MLP** ⭐ | Linear → GELU → Linear | **最佳** | LLaVA-1.5+ |
| Cross-Attention | 类 Q-Former | 好但复杂 | BLIP-2, Flamingo |
| C-Abstractor | CNN + Attention | 较好 | Honeybee |
| Perceiver Resampler | 可学习 query | 好 | Flamingo |

---

## 六、现代 VLM 架构对比

### 6.1 主流模型对比

| 模型 | 视觉编码器 | 桥接方式 | LLM | 分辨率 | 特点 |
|------|-----------|---------|-----|--------|------|
| **Qwen2-VL** | ViT (自训) | 2D-RoPE | Qwen2 | 动态 | 原生多分辨率、视频 |
| **InternVL2** | InternViT-6B | MLP | InternLM2 | 动态 | 超大视觉编码器 |
| **LLaVA-NeXT** | CLIP ViT | MLP | 多种 LLM | AnyRes | 高清切片 |
| **GPT-4V** | 未公开 | 未公开 | GPT-4 | 高清 | 最强闭源 |
| **Claude 3.5** | 未公开 | 未公开 | Claude | 高清 | 长文档理解 |

### 6.2 动态分辨率方案

```
传统方案: 固定 224×224 → 信息损失大（小字看不清）

AnyRes (LLaVA-NeXT):
  高清图 (1344×896)
       ↓
  切割为多个 336×336 的 tile
       ↓
  每个 tile 独立送入 ViT → 每个 tile 得到 576 tokens
       ↓
  Concat 所有 tile tokens + 全局缩略图 tokens
       ↓
  总 visual tokens 可能 = 2880+ (5 tiles × 576)

Qwen2-VL 方案:
  不切割，直接将任意大小图片送入 ViT
  用 2D-RoPE 编码空间位置
  更加优雅，但需要自训 ViT
```

---

## 七、关键数值直觉

| 指标 | 典型值 | 说明 |
|------|--------|------|
| ViT-L/14 参数量 | 307M | 视觉编码器 |
| CLIP 训练数据 | 400M 图文对 | WIT-400M |
| LLaVA Stage-1 数据 | 558K | 对齐数据 |
| LLaVA Stage-2 数据 | 150K-665K | 指令数据 |
| 224×224 图片 visual tokens | 196-256 | (224/16)² = 196 |
| 高清图 visual tokens | 2000-5000+ | 多 tile 切片后 |
| Visual tokens 占 KV Cache | 50-80% | 推理服务的核心瓶颈 |

---

## 面试高频问答

**Q1：为什么 LLaVA 用简单的 MLP 而不用 Q-Former？**
> LLaVA 论文实验表明，2 层 MLP projector 效果与 Q-Former 接近甚至更好。Q-Former 在压缩 visual tokens（从 256→32）上有优势，但也丢失了空间细节。LLaVA 选择保留所有 visual tokens，让 LLM 自己决定关注哪些区域。简单架构也意味着更容易训练和扩展。

**Q2：CLIP 和 BLIP 的核心区别？**
> CLIP 只做对比学习（ITC），学到图文全局匹配能力。BLIP 同时做对比学习 + 图文匹配 + 文本生成三个任务，因此既能检索也能生成。BLIP-2 进一步引入 Q-Former 桥接冻结的 ViT 和 LLM，极大降低了训练成本。

**Q3：LLaVA 两阶段训练各自的作用？**
> Stage-1（对齐）：只训练 Projector，让 visual tokens 映射到 LLM 的语义空间。类比"教模型看懂图片"。Stage-2（指令微调）：训练 Projector + LLM，让模型学会根据图片回答问题、执行指令。类比"教模型用图片知识解题"。

**Q4：高清图的 visual tokens 爆炸问题怎么解决？**
> 1）Token Merging：合并相似的 visual tokens（ToMe）；2）动态分辨率：按图片复杂度自适应切片（Qwen2-VL）；3）Visual Token 压缩：用 Perceiver/Q-Former 压缩到固定数量；4）Serving 优化：visual KV Cache offloading 到 CPU。

## 面试一句话
- "ViT 把图片切成 patch 当 token 处理，CLIP 用对比学习对齐图文空间，BLIP-2 用 Q-Former 桥接冻结的 ViT 和 LLM，LLaVA 用最简洁的 MLP Projector + 两阶段训练实现了强大的视觉问答能力，现代 VLM 的核心挑战是高清图的 visual tokens 爆炸和动态分辨率处理。"
