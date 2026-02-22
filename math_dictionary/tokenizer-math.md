# Tokenizer 与词表数学详解

> **核心定位**：从信息论和数据压缩的视角，系统性推导 BPE 算法的数学原理、词表大小对模型参数量和推理成本的精确影响公式、压缩率分析，以及 SentencePiece / Unigram 模型的概率框架。

---

## 1. BPE (Byte Pair Encoding) 算法推导

### 1.1 算法步骤

1. **初始化**：将文本拆分为字节序列（$256$ 种基本 token）。
2. **统计**：统计所有**相邻 token 对**的出现频率。
3. **合并**：将频率最高的 token 对合并为一个新 token，加入词表。
4. **迭代**：重复步骤 2-3，直到词表大小达到目标 $V$。

### 1.2 数学形式化

设当前词表为 $\mathcal{V}$，语料为 $\mathcal{C}$。每步选择：

$$
(a^*, b^*) = \arg\max_{(a,b) \in \mathcal{V}^2} \text{freq}(a, b \mid \mathcal{C})
$$

创建新 token $c = \text{merge}(a^*, b^*)$，更新 $\mathcal{V} \leftarrow \mathcal{V} \cup \{c\}$。

### 1.3 BPE 的信息论直觉

BPE 隐式地在做**基于频率的数据压缩**：高频 token 对被合并为单个 token，减少了序列长度。这与 Huffman 编码的思想类似——高频模式用更短的编码。

---

## 2. 词表大小对模型的影响

### 2.1 参数量影响

Embedding 层和 LM Head（输出层）的参数量直接与 $V$ 成正比：

$$
\text{Embedding 参数} = V \times d
$$
$$
\text{LM Head 参数} = d \times V
$$
$$
\boxed{\text{总计} = 2 V d}
$$

**代入示例**：

| $V$ | $d$ | Embedding + Head 参数 | BF16 显存 |
|:---:|:---:|:--------------------:|:---------:|
| $32$K | $4096$ | $268$M | $512$ MB |
| $128$K | $4096$ | $1.07$B | $2$ GB |
| $128$K | $8192$ | $2.15$B | $4$ GB |

对于 7B 模型，$V = 128$K 的 Embedding 就占总参数的 $\sim 15\%$！

### 2.2 Softmax 计算量

输出层的 Softmax 计算量：
$$
\text{FLOPs}_{\text{Softmax}} = \mathcal{O}(B \times T \times V)
$$

$V$ 越大，每步 Decode 的 Softmax 计算越慢（但 $T = 1$ 时影响可控）。

---

## 3. 压缩率 (Compression Ratio)

### 3.1 定义

$$
\text{CR} = \frac{\text{Total Characters}}{\text{Total Tokens}}
$$

| 语言 | 典型 CR | 说明 |
|------|:-------:|------|
| 英文 | $3.5$–$4.5$ 字符/token | 常见词被合并为单 token |
| 中文 | $1.5$–$2.5$ 字符/token | 取决于词表中的中文覆盖率 |
| 代码 | $2.5$–$3.5$ 字符/token | 保留字和缩进 |

### 3.2 压缩率对推理成本的影响

推理步数（Decode steps）$\propto$ Token 数。对同一段文本：

$$
\text{推理步数} = \frac{|\text{text}|}{\text{CR}}
$$

**CR 越高 → Token 越少 → 推理越快**。但每步的 Softmax 计算量 $\propto V$（略有增加）。

整体通常是压缩率高更有利：
$$
\text{Total Decode FLOPs} \propto \frac{|\text{text}|}{\text{CR}} \times (2N + V \cdot d)
$$

---

## 4. 词表大小选择的权衡

$$
V \uparrow \quad \Rightarrow \quad \begin{cases}
\text{CR} \uparrow & \text{（优势：同上下文编码更多信息）} \\
\text{Embedding 参数} \uparrow & \text{（劣势：显存和计算增加）} \\
\text{稀有 token 训练不充分} & \text{（劣势：低频 token 的 Embedding 欠拟合）} \\
\text{Softmax } \mathcal{O}(V) & \text{（劣势：单步略慢）}
\end{cases}
$$

| 模型 | $V$ |
|------|:---:|
| LLaMA 1 | $32{,}000$ |
| LLaMA 2 | $32{,}000$ |
| LLaMA 3 | $128{,}256$ |
| GPT-4 | $\sim 100{,}000$ |
| Qwen 2.5 | $151{,}936$ |

---

## 5. 特殊 Token

| Token | 含义 | 用途 |
|-------|------|------|
| `<bos>` | Beginning of Sequence | 序列起始标志 |
| `<eos>` | End of Sequence | 序列结束（触发停止） |
| `<pad>` | Padding | Batch 内对齐（不参与 Loss 计算） |
| `<unk>` | Unknown | Byte-level BPE 通常不需要 |
| `<\|im_start\|>` / `<\|im_end\|>` | Chat Template | 多轮对话格式标记 |

---

## 6. Byte-level BPE

以**字节**（$256$ 种）而非字符为基本单位。

**优势**：
- 可处理**任何语言**和二进制输入
- 永远不会产生 `<unk>`（所有输入都可以表示为字节序列）

**劣势**：
- 非 ASCII 字符可能被拆成多个字节 token（降低压缩率）

GPT-2/3/4、LLaMA 等均使用 Byte-level BPE。

---

## 7. SentencePiece 与 Unigram 模型

### 7.1 SentencePiece

将文本视为**原始字节流**，不做预分词（Language-agnostic）。支持 BPE 和 Unigram 两种算法。

### 7.2 Unigram 语言模型

**与 BPE 相反**：从大词表开始，迭代**剪枝**。

似然函数：
$$
\mathcal{L}(\mathcal{V}) = \sum_{\text{sentence}} \log P(S \mid \mathcal{V}) = \sum_{\text{sentence}} \log \sum_{\mathbf{x} \in \text{Seg}(S)} \prod_{x_i \in \mathbf{x}} p(x_i)
$$

每步移除使 $\mathcal{L}$ 下降最小的 token：

$$
t^* = \arg\min_{t \in \mathcal{V}} \left(\mathcal{L}(\mathcal{V}) - \mathcal{L}(\mathcal{V} \setminus \{t\})\right)
$$

**优势**：天然支持**概率采样多种分词方式**（正则化效果），而 BPE 的分词结果是确定性的。

---

## 8. Tokenizer 对推理系统的影响

| 方面 | 影响 |
|------|------|
| **上下文窗口** | 模型的 $T_{\max}$ 是 Token 数上限，CR 越高能容纳的文本越多 |
| **KV Cache 显存** | $\propto$ Token 数（CR 高 → Token 少 → KV 省） |
| **Prefix Caching** | Token 级匹配（不同 Tokenizer 的 prefix 长度可能不同） |
| **成本计费** | 商业 API 按 Token 计费，CR 高 → 成本低 |

---

## 面试一句话

> "Tokenizer 影响的是模型看到信息的粒度：词表太小浪费推理步数，词表太大浪费 Embedding 参数和 Softmax 计算。BPE 是基于频率的数据压缩，$V \times d$ 的 Embedding 参数在大词表下不可忽略（7B 模型中占 $\sim 15\%$）。"