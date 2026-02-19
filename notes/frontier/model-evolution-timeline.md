# 主流大模型发展脉络与核心优化点

> Qwen / DeepSeek / GLM / Kimi / MiniMax — 发展脉络 + 技术路线对比

---

## 一、时间线总览 (2023-2025)

```
2023.02  ───── LLaMA (Meta)          ← 开源 LLM 时代开始
2023.07  ───── Llama 2               ← 开放商用
2023.07  ───── ChatGLM2-6B (智谱)    ← 国产开源
2023.09  ───── Qwen-7B/14B (阿里)    ← 阿里入局
2023.10  ───── Mistral 7B            ← Sliding Window Attention
2023.10  ───── Kimi (Moonshot)       ← 200K 长上下文
2023.12  ───── Mixtral 8x7B          ← 开源 MoE
2023.12  ───── Mamba                 ← SSM 挑战 Transformer

2024.01  ───── DeepSeek-V2           ← MLA + DeepSeekMoE
2024.01  ───── MiniMax-abab6.5       ← Lightning Attention
2024.04  ───── Llama 3               ← 15T tokens, 128K context
2024.06  ───── Qwen2                 ← 7T tokens, 全面升级
2024.07  ───── GLM-4                 ← 转向 Causal LM
2024.07  ───── Llama 3.1             ← 405B, 128K native
2024.09  ───── Qwen2.5              ← 18T tokens, 1M context (Turbo)
2024.10  ───── Kimi k1.5             ← 思维链 + RL
2024.12  ───── DeepSeek-V3           ← MLA+MoE 671B, FP8 训练

2025.01  ───── MiniMax-01            ← 4M context, Lightning Attn
2025.01  ───── DeepSeek-R1           ← 纯 RL 涌现推理能力
2025.02  ───── Qwen2.5-Max           ← MoE 旗舰
2025.02  ───── Kimi k1.5 Long Thinking ← 超长思维链
```

---

## 二、各家技术路线详解

### 2.1 DeepSeek（深度求索）

#### 发展路径
```
DeepSeek-V1 (67B dense)
    → DeepSeek-V2: MLA + DeepSeekMoE (236B, 21B active)
    → DeepSeek-V3: 升级版 MLA+MoE (671B, 37B active) + FP8 训练
    → DeepSeek-R1: 纯 RL 推理能力 (基于 V3 + GRPO)
```

#### 核心优化点
| 创新 | 版本 | 面试价值 |
|------|------|---------|
| **MLA (Multi-head Latent Attention)** | V2/V3 | ⭐⭐⭐⭐⭐ |
| **Auxiliary-Loss-Free Load Balancing** | V3 | ⭐⭐⭐⭐⭐ |
| **FP8 Mixed Precision Training** | V3 | ⭐⭐⭐⭐ |
| **Multi-Token Prediction (MTP)** | V3 | ⭐⭐⭐⭐ |
| **GRPO (Group Relative Policy Optimization)** | R1 | ⭐⭐⭐⭐⭐ |
| **纯 RL 涌现推理** | R1 | ⭐⭐⭐⭐⭐ |

#### 技术特色
- **效率至上**：MLA 压缩 KV Cache 16×，MoE 激活参数仅 37B
- **训练成本低**：V3 仅花费 ~$5.5M（对比 Llama 3.1 405B 估计 >$100M）
- **RL 创新**：R1 证明纯 RL 可以涌现 Thinking Tokens，不需要蒸馏

---

### 2.2 Qwen（通义千问，阿里）

#### 发展路径
```
Qwen-7B/14B (2023.09)
    → Qwen1.5: 多语言增强
    → Qwen2: 7T tokens, GQA+RoPE+SwiGLU 统一
    → Qwen2.5: 18T tokens, 128K→1M (Turbo)
    → Qwen2.5-Max: MoE 旗舰
    → QwQ: 推理增强 (思维链)
```

#### 核心优化点
| 创新 | 版本 | 面试价值 |
|------|------|---------|
| **YaRN 长度扩展** | 2.5 | ⭐⭐⭐⭐ |
| **超大规模数据 (18T tokens)** | 2.5 | ⭐⭐⭐⭐ |
| **合成数据增强** | 2.5 | ⭐⭐⭐⭐ |
| **退火配比策略** | 2.5 | ⭐⭐⭐ |
| **完整模型矩阵 (0.5B-72B)** | 2.5 | ⭐⭐⭐ |
| **1M context (Turbo)** | 2.5 | ⭐⭐⭐⭐ |

#### 技术特色
- **全面性**：模型矩阵最完整（0.5B-72B + Coder + Math + VL）
- **数据为王**：18T tokens 是公开最大训练数据量
- **开源友好**：Apache 2.0 许可证，社区生态最活跃

---

### 2.3 GLM / ChatGLM（智谱 AI）

#### 发展路径
```
GLM (Prefix LM, 2022)
    → ChatGLM-6B (2023.03)
    → ChatGLM2-6B: 转向标准 decoder
    → GLM-4-9B: GQA+RoPE+SwiGLU
    → GLM-4-Plus: 闭源旗舰
    → CogVLM2: 视觉理解
    → CogVideoX: 视频生成
    → Slime: 异步 RL 训练框架
```

#### 核心优化点
| 创新 | 版本 | 面试价值 |
|------|------|---------|
| **Prefix LM → Causal LM 演进** | GLM→GLM-4 | ⭐⭐⭐ |
| **多模态矩阵 (CogVLM/CogVideo)** | CogVLM2 | ⭐⭐⭐⭐ |
| **Slime 异步 RL 框架** | 2025 | ⭐⭐⭐⭐⭐ |
| **SGLang + Megatron 组合** | Slime | ⭐⭐⭐⭐ |
| **GLM-4-Long (1M context)** | GLM-4 | ⭐⭐⭐ |

#### 技术特色
- **学术底蕴**：清华系，重视理论创新
- **多模态全面**：文本 + 视觉 + 视频
- **RL 框架**：Slime 是首个公开的异步 RL 训练框架

---

### 2.4 Kimi（月之暗面 / Moonshot AI）

#### 发展路径
```
Kimi Chat (2023.10): 200K context ← 国内最早长上下文
    → Kimi (2024): 持续优化
    → Kimi k1.5: RL 推理 + Long-CoT
    → Kimi k1.5 Long Thinking (2025)
```

#### 核心优化点
| 创新 | 版本 | 面试价值 |
|------|------|---------|
| **超长上下文 (200K → 更长)** | 初版 | ⭐⭐⭐⭐⭐ |
| **长上下文推理优化** | k1.5 | ⭐⭐⭐⭐ |
| **Long2Short 蒸馏** | k1.5 | ⭐⭐⭐⭐ |
| **RL + 长思维链** | k1.5 | ⭐⭐⭐⭐ |

#### 技术特色
- **长上下文先行者**：国内最早将超长上下文产品化
- **decode-heavy 场景优化**：长上下文 → decode 阶段更重要
- **Long2Short**：长 CoT 模型蒸馏到短 CoT，兼顾质量和效率

---

### 2.5 MiniMax

#### 发展路径
```
MiniMax-abab 系列 (2023)
    → MiniMax-abab6.5: Lightning Attention 初版
    → MiniMax-01: 4560B MoE, 4M context, 开源
```

#### 核心优化点
| 创新 | 版本 | 面试价值 |
|------|------|---------|
| **Lightning Attention** | abab6.5/01 | ⭐⭐⭐⭐⭐ |
| **Linear + Softmax Hybrid** | 01 | ⭐⭐⭐⭐⭐ |
| **4M token 上下文** | 01 | ⭐⭐⭐⭐ |
| **MoE (32 experts)** | 01 | ⭐⭐⭐ |

#### 技术特色
- **Hybrid 架构先驱**：最早将 Linear Attention 规模化用于生产
- **超长上下文**：4M tokens 是公开模型最长
- **工程创新**：Lightning Attention 的 IO 优化

---

## 三、核心技术路线对比

### 3.1 架构选择
| 维度 | DeepSeek | Qwen | GLM | Kimi | MiniMax |
|------|---------|------|-----|------|---------|
| Attention | **MLA** | GQA | GQA | 未公开 | **Lightning+Softmax** |
| FFN | SwiGLU+MoE | SwiGLU | SwiGLU | 未公开 | SwiGLU+MoE |
| 位置编码 | RoPE | RoPE+YaRN | RoPE | RoPE | RoPE |
| 归一化 | RMSNorm | RMSNorm | RMSNorm | RMSNorm | RMSNorm |

### 3.2 关键差异化技术
```
DeepSeek → MLA（KV 压缩）+ GRPO（RL 推理）
Qwen     → 超大数据（18T）+ 模型矩阵全（0.5B-72B）
GLM      → 多模态（CogVLM/Video）+ RL 框架（Slime）
Kimi     → 超长上下文 + Long-CoT
MiniMax  → Hybrid 架构（Lightning Attention）+ 4M context
```

### 3.3 训练策略对比
| 维度 | DeepSeek | Qwen | GLM | MiniMax |
|------|---------|------|-----|---------|
| 预训练数据 | 14.8T | **18T** | 未公开 | 未公开 |
| 训练精度 | **FP8** | BF16 | BF16 | BF16 |
| MoE 负载均衡 | **Aux-Loss-Free** | Aux Loss | — | Aux Loss |
| RL 对齐 | **GRPO** | DPO + RL | RL (Slime) | DPO |
| 合成数据 | ✅ 数学+代码 | ✅ 大量 | ✅ | ✅ |

### 3.4 推理能力路线
```
传统路线：
  SFT → DPO → 发布
  代表：Qwen2.5, GLM-4

RL 推理路线（2025 新趋势）：
  SFT → RL (PPO/GRPO) → Thinking Tokens → 发布
  代表：DeepSeek-R1, Kimi k1.5, QwQ

蒸馏路线：
  大模型 RL → 生成 CoT 数据 → SFT 小模型
  代表：DeepSeek-R1-Distill, 各种 distill 版本
```

---

## 四、面试中如何谈论这些模型

### 4.1 总结框架
```
"关于 XX 模型，我的理解有三个层面：
1. 架构创新点：[具体创新]
2. 工程亮点：[训练/推理优化]
3. 我的思考：[为什么这个设计有效 / trade-off 是什么]"
```

### 4.2 示例

#### 谈 DeepSeek-V3
> "DeepSeek-V3 的核心创新是 MLA + 辅助损失无关的 MoE 负载均衡。MLA 把 KV 投影到低秩空间，KV Cache 压缩 16 倍，这比 GQA 更激进但效果不降。负载均衡用 bias 动态调整而非额外 loss，不干扰主训练目标。训练用 FP8，成本仅 $5.5M。我觉得这三个创新都指向一个核心理念：用更聪明的方式而非更大的模型来达到好效果。"

#### 谈 MiniMax-01
> "MiniMax-01 最有意思的是 Hybrid 架构：底层 2/3 用 Lightning Attention（Linear Attention 变体），顶层 1/3 用 Softmax Attention。这个分层设计基于一个洞察：底层主要处理局部模式，不需要精确的全局注意力，用 Linear Attention 就够了；顶层需要全局推理，才用 Softmax。这让它支持 4M tokens，是公开模型最长的。"

---

## 面试高频问答

**Q1：对比 DeepSeek-V3 和 Qwen2.5-72B 的技术路线差异？**
> DeepSeek 走 MoE + MLA 路线，671B 总参数只激活 37B，靠架构创新降低推理成本。Qwen 走 dense + 大数据路线，72B dense 模型配 18T tokens 训练。DeepSeek 架构更创新但更复杂，Qwen 更稳健但参数效率低。

**Q2：为什么 2025 年 RL 推理（o1/R1）成为热点？**
> DeepSeek-R1 证明纯 RL（GRPO）可以让模型自发涌现 Thinking Tokens 和推理链，在数学/代码上大幅超越 SFT+DPO。关键发现是：推理能力需要 RL 的探索-利用机制来激发，单靠 SFT 模仿不够。

**Q3：Kimi 在长上下文方面有什么独到之处？**
> Kimi 最早将 200K+ 上下文产品化，核心挑战是 decode-heavy 场景的优化（长上下文 prefill 占比小，decode 占比大）。k1.5 版本引入 Long-CoT + Long2Short 蒸馏，让长思维链的推理质量也能传递给短输出模型。

**Q4：如果要选一个开源模型做基座，你怎么选？**
> 看场景：通用对话选 Qwen2.5（生态最完整、多语言好）；推理增强选 DeepSeek-R1-Distill（推理能力强）；超长上下文选 MiniMax-01（4M context）；资源受限选 Llama-3.1-8B（轻量 + 社区大）。
