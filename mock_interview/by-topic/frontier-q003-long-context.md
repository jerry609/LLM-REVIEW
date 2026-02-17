# 面试题：超长上下文推理优化

## 题目
"Kimi 支持 200 万 token 上下文，MiniMax-01 支持 400 万 token。请分析实现超长上下文的关键技术挑战和解决方案。"

## 参考答案

### 关键挑战

**挑战 1：Attention 的 O(n^2) 复杂度**
- 2M tokens 的 full attention 不可行
- 解决方案：
  1. Sliding Window Attention
  2. Sparse Attention
  3. Linear Attention（MiniMax Lightning Attention）
  4. Ring Attention（序列并行）

**挑战 2：KV Cache 显存爆炸**
- 2M tokens, 32层, GQA-8, d_head=128, FP16 约 131GB
- 解决方案：
  1. 分层存储：HBM -> DRAM -> SSD
  2. KV Cache 量化：FP16 -> INT4（4x 压缩）
  3. MLA 低秩压缩（~16x）
  4. Token eviction

**挑战 3：Prefill 延迟**
- 2M tokens 的 prefill 需要数十秒
- 解决方案：Chunk Prefill + Prefix Caching + Pipeline

**挑战 4：训练时的长度泛化**
- 训练 4K-32K，推理到 2M
- 解决方案：YARN / NTK-aware RoPE 插值 + 渐进式长度扩展

### 面试延伸

**Q: 为什么不全用 Linear Attention？**
- Linear Attention 的 "Needle in a Haystack" 检索能力弱
- 是"模糊匹配"，没有 Softmax 的 sharp attention
- 所以 MiniMax 顶层保留 Softmax 做精确推理

## 加分点
- 能具体计算 KV Cache 的显存占用
- 知道 Ring Attention 的序列并行原理
