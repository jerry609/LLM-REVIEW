# Qwen 系列技术深度解析

## 一、Qwen2.5（2024.09 发布）

### 1.1 模型矩阵
| 模型 | 参数量 | 上下文 | 特点 |
|------|--------|--------|------|
| Qwen2.5-0.5B/1.5B/3B | 小 | 32K | 端侧部署 |
| Qwen2.5-7B/14B/32B | 中 | 128K | 主力推理 |
| Qwen2.5-72B | 大 | 128K | 旗舰 |
| Qwen2.5-Turbo | - | 1M | 超长上下文 |

### 1.2 核心技术改进

#### 训练数据
- 预训练：**18T tokens**（Qwen2 的 7T -> 18T）
- 数据质量：合成数据占比提升，math/code 数据增强

#### 架构改进
- **GQA** (Grouped Query Attention)：所有 size 统一用 GQA
- **RoPE**：base 从 10000 提升到 1000000（更好的长度外推）
- **SwiGLU** 激活函数
- **RMSNorm** + Pre-Norm

#### 长上下文
- YARN (Yet Another RoPE ExtensioN) 扩展：
  - 训练时用 32K，推理时扩展到 128K
  - NTK-aware interpolation：对不同频率分量用不同缩放
- Qwen2.5-Turbo 达到 **1M tokens**：
  - Sparse attention pattern
  - 动态 NTK 插值

### 1.3 Qwen2.5-Coder
- **代码专项**：Fill-in-the-Middle (FIM) 训练
- HumanEval: 65.9% (7B), 86.2% (32B)
- **Repo-level 补全**：支持跨文件上下文

### 1.4 Qwen2.5-Math
- 数学推理专项：Tool-Integrated Reasoning (TIR)
- 可调用 Python interpreter 验证中间结果
- MATH benchmark: 83.1% (72B)

---

## 二、QwQ / QwQ-32B（2024.11 发布）

### 2.1 定位
- 类 o1 的推理模型（Qwen with Questions）
- 基于 Qwen2.5-32B 微调
- 支持 **32K 上下文**

### 2.2 技术细节
- **训练方法**：
  - Stage 1：Long-CoT SFT（学习推理格式）
  - Stage 2：RL 优化推理质量（类似 R1）
- **推理特点**：
  - 产生详细的思考过程（thinking tokens）
  - 自我质疑和修正
  - 数学/编程任务显著提升

### 2.3 与 R1 对比
| 维度 | QwQ-32B | DeepSeek-R1 |
|------|---------|-------------|
| 基座模型 | Qwen2.5-32B (Dense) | DeepSeek-V3 (MoE 671B) |
| 训练方法 | SFT + RL | 4-stage (cold start -> RL -> SFT -> RL) |
| 开源程度 | 权重开放 | 权重 + 训练细节全开 |
| 推理成本 | 较低 (32B Dense) | 较高 (37B activated) |
| MATH 性能 | 90.6% | 97.3% |

---

## 三、Qwen2.5-VL（2025.01 发布）

### 3.1 核心创新

#### 动态分辨率 (Dynamic Resolution)
- **Naive ViT**：固定分辨率裁剪/resize -> 信息损失
- **Qwen2.5-VL**：
  - 将图片按 28x28 patch 切分
  - 不同分辨率图片产生不同数量的 visual tokens
  - 范围：4 tokens (极小图) -> 16384 tokens (高清图)
  - Token 数 = ceil(H/28) * ceil(W/28)

#### Multimodal RoPE (M-RoPE)
- **问题**：如何给 visual tokens 编码位置？
- **方案**：3D RoPE = temporal_id + height_id + width_id
  - 图片：temporal=0, height/width=patch 网格坐标
  - 视频：temporal=frame_id, height/width=patch 坐标
  - 文本：temporal/height/width 全部等于 token position

#### 视频理解
- 支持 **20min+ 视频**理解
- 动态帧采样：根据视频复杂度调整帧数
- 时间定位：可以回答"第几秒发生了什么"

### 3.2 VLM Serving 挑战
- **Prefill 瓶颈**：高清图 -> 数千 visual tokens -> prefill 非常慢
- **KV Cache 压缩**：visual tokens 的 KV 通常可压缩率更高
- **混合 batch**：文本请求 + 多模态请求混排的调度策略

---

## 四、Qwen3（2025.04 发布）

### 4.1 核心创新

#### Thinking + Non-thinking 双模式
- **背景**：推理模型（长 CoT 如 o1/R1）和对话模型（快速响应）需求并存
- **方案**：同一模型同时支持两种模式
  - `/think`：启用长 Chain-of-Thought 推理，适合数学/代码/逻辑
  - `/no_think`：快速响应模式，适合日常对话
- **Why**：避免维护两套模型，用户根据场景按需切换

#### 全系列 MoE + Dense 矩阵
| 模型 | 类型 | 参数量 | 激活参数 | Experts |
|------|------|--------|----------|---------|
| Qwen3-0.6B/1.7B/4B/8B | Dense | 小-中 | 全量 | - |
| Qwen3-14B/32B | Dense | 中-大 | 全量 | - |
| Qwen3-30B-A3B | MoE | 30B | 3B | 128 |
| Qwen3-235B-A22B | MoE | ~235B | 22B | 128 |

#### 训练 4 阶段
1. **预训练**：36T+ tokens，数据规模继续扩大（Qwen2.5 的 18T → 36T+）
2. **Long CoT Cold Start**：用少量高质量推理数据微调，激活思考能力
3. **RL 强化**：GRPO 等方法提升推理质量
4. **双模式融合**：Thinking 和 Non-thinking 数据混合训练，实现模式统一

#### 其他亮点
- **119 种语言支持**（Qwen2.5 的 29 → 119）
- **技术报告**：arXiv:2504.07491
- **子系列**：Qwen3-Coder / Qwen3-Math（2025.06）

### 4.2 面试要点
- Qwen3 的双模式本质上是一种 **mode-switching inference**，同一个权重支持两种行为
- MoE 架构（128 experts, top-8）借鉴了 DeepSeek-V3 的经验，但规模更小
- 4 阶段训练流程是目前后训练的 best practice：预训练 → CoT 激活 → RL 强化 → 融合

---

## 五、Qwen3.5（2026.02.17 发布）⭐ 最新

### 5.1 定位
- **AI Agent 原生模型**：标志着阿里从纯模型能力竞争转向 Agent 应用层竞争
- CNBC 报道："Alibaba unveils Qwen3.5 as China's chatbot race shifts to AI agents"
- 在农历新年前发布，多家中国 AI 公司同时发布新模型

### 5.2 核心方向
- **Agent 框架协同**：深度集成工具调用、多步推理+行动
- **多步任务执行**：从"回答问题"转向"执行任务"
- **生态布局**：配合阿里云的 Agent 开发平台

### 5.3 行业意义
- 中国 AI 竞争重心从"模型 benchmark 比拼"转向"Agent 落地能力"
- Qwen 系列迭代节奏：1.0→1.5→2→2.5→3→3.5，约半年一代
- 整个系列从开源社区生态 → Agent 生态的战略转变

---

## 六、面试高频问答

**Q: Qwen2.5 和 LLaMA 3.1 的架构差异？**
- 相同：Pre-Norm, GQA, SwiGLU, RoPE
- Qwen 用更大的 RoPE base（1M vs 500K）
- Qwen 的 tokenizer 支持更好的中文（15万+ vocab）
- LLaMA 的 attention 用 GQA-8，Qwen 用不同 group 数

**Q: YARN 长度外推的原理？**
- NTK-aware：不是简单线性插值
- 低频分量（管长距离依赖）用较大缩放
- 高频分量（管局部依赖）基本不变
- 效果：4K 训练 -> 128K 推理，损失很小

**Q: VLM 的 visual token 数量如何影响推理性能？**
- Prefill latency 正比于 (text_tokens + visual_tokens)^2
- KV Cache memory 正比于 total_tokens
- 优化：visual token merging / compression after attention
