# MiniMax-01 技术深度解析

## 一、MiniMax-01（2025.01 发布）

### 1.1 核心定位
- 4560 亿参数 MoE 模型（45.9B 激活）
- 支持 **4M tokens** 上下文（目前公开模型最长）
- 开源权重 + 技术报告

### 1.2 核心技术创新

#### Lightning Attention（最大亮点）
- **动机**：标准 Softmax Attention 是 O(n^2)，无法处理 4M tokens
- **方案**：用 **Linear Attention** 替代部分层的 Softmax Attention
- **具体设计**：
  - 底层（1-27层）：Lightning Attention（Linear）
  - 顶层（28-80层）：Softmax Attention（标准）
  - 混合比例：约 1/3 Softmax + 2/3 Linear

#### Linear Attention 原理
- 标准 Attention：O = softmax(QK^T/sqrt(d)) V
  - 需要 n*n 的 attention matrix -> O(n^2)
- Linear Attention：O = phi(Q)(phi(K)^T V)
  - 先算 phi(K)^T V -> d*d matrix
  - 再算 phi(Q) @ (d*d) -> O(n * d^2)
  - 当 n >> d 时，大幅减少计算
- **Lightning Attention 的改进**：
  - 用 **cumsum** 技巧实现 causal linear attention
  - I/O 优化：类似 FlashAttention 的 tiling
  - 数值稳定性：normalization trick

#### Softmax Attention + Linear Attention 混合的直觉
- Linear Attention：擅长全局信息聚合（O(n)），但精度稍低
- Softmax Attention：局部/精细注意力更强，但 O(n^2)
- 底层用 Linear 做粗粒度特征提取
- 顶层用 Softmax 做精细推理

### 1.3 MoE 配置
- **32 experts**，每次激活 **2 experts**
- Top-K routing with load balancing loss
- 与 DeepSeek-V3 (256 experts, 8 activated) 的对比：
  - MiniMax 用更少更大的 experts
  - DeepSeek 用更多更细粒度的 experts

### 1.4 长上下文工程
- **4M tokens** 支持：
  - RoPE base: 10M（极大）
  - 训练序列长度逐步扩展：4K -> 32K -> 512K -> 4M
  - 最后阶段用少量长序列数据 fine-tune
- **推理优化**：
  - KV Cache offloading（GPU -> CPU -> SSD）
  - Chunk prefill + pipeline
  - Linear Attention 层不需要存完整 KV Cache

---

## 二、Lightning Attention 详细解析

### 2.1 标准实现 vs Lightning

Standard Softmax Attention: O(n^2)
  attn = softmax(Q @ K.T / sqrt(d)) @ V

Linear Attention: O(n * d^2)
  S = cumsum(K.T @ V, dim=seq)  # running state
  O_i = Q_i @ S_i               # query state at each position

### 2.2 训练技巧
- **Intra-chunk**：chunk 内用矩阵乘法并行
- **Inter-chunk**：chunk 间用 cumulative state 传递
- **混合精度**：Linear Attention 对精度更敏感，用 BF16

### 2.3 为什么没有全用 Linear Attention？
- Linear Attention 的 **retrieval 能力弱**：
  - "大海捞针"测试中 Linear Attention 明显不如 Softmax
  - 因为 Linear Attention 是"模糊匹配"，Softmax 是"精确匹配"
- 顶层 Softmax 保证**精确检索和推理**
- 底层 Linear 做**高效的全局信息压缩**

---

## 三、面试高频问答

**Q: Linear Attention 为什么快？代价是什么？**
- 快：将 n^2 降为 n*d^2，长序列场景显著提速
- 代价：attention 权重不再是 sharp（没有 softmax 的 peaky 分布）
  -> 检索/精确匹配能力下降
  -> 所以需要混合使用

**Q: MiniMax-01 的 4M tokens 是怎么实现的？**
- Linear Attention 层：O(n)，天然支持长序列
- Softmax Attention 层：sliding window 或 sparse pattern
- KV Cache：分层存储 + offloading
- 训练：渐进式长度扩展（4K -> 4M）

**Q: MiniMax vs DeepSeek-V3 的 MoE 策略对比？**
- MiniMax：32 experts, 2 active -> 粗粒度，路由简单
- DeepSeek：256 experts, 8 active -> 细粒度，组合丰富
- DeepSeek 还有 shared expert，MiniMax 没有
- DeepSeek 用 bias 做 load balancing（无 auxiliary loss）
