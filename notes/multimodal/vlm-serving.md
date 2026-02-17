# VLM (Vision-Language Model) Serving 架构

## 一、VLM 推理流程

### 1.1 标准 VLM 架构
Image -> Vision Encoder (ViT) -> Visual Tokens -> Projector -> LLM
Text  -> Tokenizer           -> Text Tokens  ->           -> LLM

### 1.2 推理阶段分析

#### Prefill 阶段
1. **图像编码**：ViT 处理图片 -> visual tokens
   - 224x224 图片 -> 196 tokens (14x14 patches)
   - 高清图可能 -> 2000+ tokens
2. **Projector**：将 visual tokens 映射到 LLM embedding space
3. **LLM Prefill**：处理 [visual_tokens + text_tokens]
   - **瓶颈**：高清图的 visual tokens 非常多 -> prefill 极慢

#### Decode 阶段
- 与纯文本模型基本一致
- KV Cache 包含 visual tokens 的 KV（占大量显存）

### 1.3 KV Cache 特点
- Visual token 的 KV Cache 通常占总 KV Cache 的 50-80%
- 但 visual KV 在 decode 阶段是**静态的**（不会改变）
- 优化机会：visual KV compression / offloading

---

## 二、Serving 挑战与优化

### 2.1 Visual Token 压缩
- **Token Merging**：合并相似的 visual tokens
  - 如 ToMe (Token Merging)：bipartite matching 合并
  - 2000 tokens -> 500 tokens（4x 压缩）
- **Adaptive Resolution**：
  - Qwen2.5-VL：动态分辨率，按需生成 visual tokens
  - 小图用少 token，大图用多 token

### 2.2 Prefill 优化
- **图像预处理 pipeline 化**：
  - ViT 编码和 LLM prefill 可以 overlap
  - 图片 -> (GPU-1: ViT) -> visual tokens -> (GPU-2: LLM prefill)
- **Batch 优化**：
  - 同 batch 内图片大小可能不同 -> padding overhead
  - 解决：按图片大小分桶 batch

### 2.3 混合调度
- **问题**：batch 中同时有文本请求和多模态请求
- 文本请求 prefill 快（几百 tokens），多模态 prefill 慢（几千 tokens）
- **方案**：
  - 将多模态请求的 prefill 分 chunk（chunked prefill）
  - 在 chunk 间隙插入文本请求的 decode
  - 类似 continuous batching 的思路

### 2.4 多图 / 视频请求
- 多图：每张图独立编码 -> concat visual tokens
- 视频：采样关键帧 -> 每帧编码 -> concat
- **显存爆炸风险**：10 张高清图 x 2000 tokens = 20K visual tokens
- **限制策略**：max_visual_tokens budget per request

---

## 三、主流 VLM Serving 框架

| 框架 | VLM 支持 | 特点 |
|------|---------|------|
| vLLM | 支持 (v0.4+) | 支持 LLaVA, Qwen-VL, InternVL |
| SGLang | 支持 | RadixAttention 对 VLM prefix 友好 |
| TensorRT-LLM | 支持 | 高性能但配置复杂 |
| lmdeploy | 支持 | 国产框架，支持 InternVL |

---

## 四、面试高频问答

**Q: VLM 的 TTFT (Time to First Token) 为什么比纯文本慢很多？**
- ViT 编码时间 + 更多 tokens 的 prefill
- 1024 visual tokens 的 prefill 约等于 1024 text tokens
- 高清图可能有 4000+ visual tokens

**Q: 如何优化 VLM 的 KV Cache 占用？**
- Visual KV 是静态的 -> 可以量化到 INT4（几乎不影响质量）
- Visual token merging 减少 token 数
- Prefix caching：同一图片的 visual KV 可以跨请求共享

**Q: VLM batch 调度和纯文本有什么区别？**
- 需要考虑图片编码的 GPU 占用
- Visual tokens 数量不均 -> batch 内 padding
- ViT 和 LLM 可以异步 pipeline
