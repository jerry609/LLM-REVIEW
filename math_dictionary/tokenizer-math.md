# Tokenizer 与词表数学速查

## 1) BPE（Byte Pair Encoding）核心算法
- 初始化：每个字节（byte）为一个 token
- 迭代：统计相邻 token 对频率，合并最频繁的对为新 token
- 重复直到词表大小达到 `V`（如 32K, 64K, 128K）
- 结果：高频词 → 单一 token，低频词 → 多个子 token

## 2) 词表大小的影响
- **Embedding 层参数**：`V * d_model`
- **输出层参数（LM head）**：`d_model * V`
- 总计：`2 * V * d_model`
- 例：V=32K, d_model=4096 → embedding + head ≈ 2 * 32768 * 4096 * 2B ≈ 512 MB

## 3) 压缩率（Compression Ratio）
- `compression_ratio = total_characters / total_tokens`
- 英文：通常 3.5-4.5 字符/token
- 中文：通常 1.5-2.5 字符/token（取决于词表中的中文覆盖率）
- 压缩率越高 → 相同上下文窗口能容纳更多信息

## 4) 词表大小选择的权衡
- **V 越大**：
  - 优势：压缩率更高，相同 token 长度编码更多信息
  - 劣势：embedding 参数增加、稀有 token 训练不充分、softmax 计算量 `O(V)`
- **V 越小**：
  - 优势：每个 token 训练更充分，模型更紧凑
  - 劣势：同等文本需要更多 token，推理步数增加
- 常见取值：32K（LLaMA 1）、32K-128K（LLaMA 2/3）、100K（GPT-4）

## 5) Token 数量与推理成本
- 推理成本 ∝ token 数量（每个 token 需一次 decode 步）
- 对同一段文本：
  - 词表大、压缩率高 → token 少 → 推理步数少 → 更快
  - 但每步 softmax 计算 `O(V)` → 单步略慢
- 整体通常是压缩率高更有利（减少的步数 > 单步增加的开销）

## 6) 特殊 Token
- `<bos>`：序列起始
- `<eos>`：序列结束
- `<pad>`：填充（batch 对齐用）
- `<unk>`：未知 token（BPE 通常不需要，byte-level 可覆盖所有输入）
- Chat template tokens：`<|im_start|>`, `<|im_end|>` 等

## 7) Byte-level BPE
- 以字节（256 种）而非字符为基本单位
- 优势：可处理任何语言和二进制输入，无 `<unk>`
- 劣势：非 ASCII 字符可能被拆成多个字节 token
- GPT-2/3/4、LLaMA 等均使用 byte-level BPE

## 8) SentencePiece
- 将文本视为原始字节流，不做预分词（language-agnostic）
- 支持 BPE 和 Unigram 两种算法
- Unigram 模型：
  - 从大词表开始，迭代剪枝
  - 每步移除使 likelihood 下降最小的 token
  - 天然支持概率采样多种分词方式

## 面试一句话
- "Tokenizer 影响的是模型看到信息的粒度：词表太小浪费步数，词表太大浪费参数，BPE 是数据驱动的折中。"
