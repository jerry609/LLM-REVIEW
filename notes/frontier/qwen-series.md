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

## 四、面试高频问答

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
