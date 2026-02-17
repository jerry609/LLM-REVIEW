# GLM / ChatGLM (智谱 AI) 技术深度解析

## 一、GLM-4 系列（2024-2025）

### 1.1 模型矩阵
| 模型 | 参数量 | 特点 |
|------|--------|------|
| GLM-4-9B | 9B | 开源基座 |
| GLM-4-Plus | 未公开 | 闭源旗舰 |
| GLM-4V | 多模态 | 视觉理解 |
| GLM-4-Long | - | 1M 上下文 |
| CogVLM2 | 19B | 视觉专项 |
| CogVideoX | - | 视频生成 |

### 1.2 GLM 架构特点

#### 原始 GLM 架构
- **Prefix LM** 设计（vs 标准 causal LM）：
  - Prefix 部分：双向 attention（可以看到前后）
  - Generation 部分：causal attention（只看过去）
  - 适合"给定上下文生成回答"场景
- **2D Positional Encoding**：
  - Position id 1：inter-span position
  - Position id 2：intra-span position

#### GLM-4 的演进
- 转向标准 **Causal LM**（与 LLaMA/Qwen 对齐）
- GQA + RoPE + SwiGLU + RMSNorm
- 更好的多语言 tokenizer

### 1.3 GLM-4V / CogVLM2
- **CogVLM** 系列：
  - Visual Expert：在每个 transformer layer 加独立的 visual FFN
  - Visual tokens 过 visual expert，text tokens 过 language expert
  - 优势：visual 和 text 分支互不干扰
- **CogVideoX**：
  - 3D VAE + DiT (Diffusion Transformer)
  - Expert Adaptive LayerNorm
  - 支持 6s 视频生成

---

## 二、智谱技术栈特色

### 2.1 CodeGeeX（代码助手）
- 基于 GLM 的代码补全/生成
- 支持 VS Code / JetBrains 插件
- **FIM (Fill-in-the-Middle) 训练**：
  - 输入：prefix + suffix -> 生成 middle
  - 比纯 left-to-right 更适合代码补全场景

### 2.2 WebGLM / SearchGLM
- 联网搜索 + 生成
- 类 RAG pipeline：query -> search -> extract -> generate
- 支持引用溯源

### 2.3 对齐技术
- **Self-Contrast**：用模型自身生成正负样本对
- **ChatGLM-RLHF**：
  - PPO 训练
  - Reward model 侧重中文偏好
  - 安全性对齐（敏感话题处理）

---

## 三、面试常见考点

**Q: GLM 的 Prefix LM 和 Causal LM 有什么区别？优劣？**
- Prefix LM：prefix 部分双向 attention -> 更好地理解上下文
- Causal LM：完全自回归 -> KV Cache 友好、推理高效
- GLM-4 转向 Causal LM 说明：推理效率 > 模型表达力（在大模型尺度下）

**Q: CogVLM 的 Visual Expert 和 LLaVA 的区别？**
- LLaVA：visual tokens 和 text tokens 共享同一套 FFN
- CogVLM：visual tokens 走独立的 FFN（Visual Expert）
- CogVLM 方案：参数量更大但视觉理解更强

**Q: 为什么智谱在中文场景有优势？**
- Tokenizer 针对中文优化（字粒度 + 词粒度混合）
- 预训练数据中文比例更高
- 对齐数据（RLHF）也侧重中文场景
