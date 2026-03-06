# Tokenizer 与词表数学详解

> **核心定位**：从 BPE 的 merge 过程出发，解释 tokenizer 为什么本质上在做压缩，再把词表大小、压缩率和推理成本连到一起。重点不是记算法步骤，而是理解 tokenizer 如何影响 embedding 参数、decode 步数、KV Cache 和计费。

---

## 1. BPE 如何把文本压成更短的 token 序列

### 1.1 初始化先决定基本单位

BPE 的第一步是先选定最小可用 token。现代 LLM 更常见的是 byte-level BPE，也就是把文本先拆成字节序列，再在此基础上学习高频合并规则。这样做的直接好处是词表起点固定，任何输入都能被编码。

### 1.2 Merge loop 是 BPE 的核心

设当前词表为 $\mathcal{V}$，语料为 $\mathcal{C}$。每一轮 BPE 都会选择出现频率最高的相邻 token 对：

$$
(a^*, b^*) = \arg\max_{(a,b) \in \mathcal{V}^2} \operatorname{freq}(a, b \mid \mathcal{C})
$$

然后创建新 token：

$$
c = \operatorname{merge}(a^*, b^*), \quad \mathcal{V} \leftarrow \mathcal{V} \cup \{c\}
$$

不断重复这个过程，直到词表大小达到目标 $V$。

### 1.3 BPE 的信息论直觉

BPE 隐式地在做基于频率的数据压缩。高频片段被合并成单个 token，序列长度就会缩短。它和 Huffman 编码不完全相同，但共享同一种核心直觉：高频模式应该占用更短的表示。

### 1.4 Byte-level 与 Char-level 的差别

| 方案 | 基本单位 | 优势 | 代价 |
|------|----------|------|------|
| Byte-level BPE | 字节 | 无 `<unk>`，语言无关 | 非 ASCII 文本可能被拆成更多 token |
| Char-level BPE | 字符 | 对单语种更直观 | 多语言与异常字符处理更脆弱 |

GPT-2、GPT-4、LLaMA 等主流模型都更偏向 byte-level BPE，因为鲁棒性通常比“最优的人类可读切分”更重要。

---

## 2. 词表大小为什么会直接变成模型成本

词表大小 $V$ 不只是一个 tokenizer 超参数，它会直接进入 embedding 和输出层，因此会体现在显存、参数量和 decode 单步代价里。

### 2.1 Embedding 与 LM Head 的参数量

若模型宽度为 $d$，则：

$$
\operatorname{EmbeddingParams} = Vd
$$

$$
\operatorname{HeadParams} = dV
$$

$$
\boxed{\operatorname{TotalParams} = 2Vd}
$$

代入几个常见量级：

| $V$ | $d$ | Embedding + Head 参数 | BF16 显存 |
|:---:|:---:|:--------------------:|:---------:|
| $32$K | $4096$ | $268$M | $512$ MB |
| $128$K | $4096$ | $1.07$B | $2$ GB |
| $128$K | $8192$ | $2.15$B | $4$ GB |

对于 7B 量级模型，若词表扩到 $128$K，embedding 与输出头往往会变成不可忽略的大块参数。

### 2.2 Decode 阶段的 Softmax 代价

输出层 softmax 的复杂度直接依赖词表大小：

$$
\operatorname{SoftmaxFLOPs} = \mathcal{O}(BTV)
$$

训练或 prefill 时，$T$ 可以很大；decode 时常见的是 $T = 1$，这会让 softmax 代价看起来没有注意力那样显眼，但 $V$ 仍然会持续出现在每一步生成里。

### 2.3 大词表带来的收益与代价

| 变化 | 收益 | 代价 |
|------|------|------|
| $V$ 变大 | 压缩率通常上升，同样上下文能编码更多文本 | embedding 参数增大，输出层代价上升 |
| token 更长 | decode 步数减少 | 低频 token 更难训练充分 |
| 多语言覆盖更强 | 跨语言与代码场景更稳 | 词表维护与 special token 设计更复杂 |

### 2.4 主流模型的词表量级

| 模型 | $V$ | 工程取舍 |
|------|:---:|----------|
| LLaMA 1 | $32{,}000$ | 倾向保守词表，控制 embedding 成本 |
| LLaMA 2 | $32{,}000$ | 延续较小词表策略 |
| LLaMA 3 | $128{,}256$ | 更重视多语言与压缩率 |
| GPT-4 | $\sim 100{,}000$ | 大词表换更好的实际编码效率 |
| Qwen 2.5 | $151{,}936$ | 更强调多语言和代码覆盖 |

---

## 3. 压缩率如何影响推理成本

### 3.1 压缩率的定义

记字符数为 $N_{\mathrm{chars}}$，token 数为 $N_{\mathrm{tokens}}$，则压缩率定义为：

$$
\operatorname{CR} = \frac{N_{\mathrm{chars}}}{N_{\mathrm{tokens}}}
$$

| 语言 | 典型 $\operatorname{CR}$ | 说明 |
|------|:-----------------------:|------|
| 英文 | $3.5$ 到 $4.5$ 字符/token | 高频单词容易被合并 |
| 中文 | $1.5$ 到 $2.5$ 字符/token | 取决于词表中的中文覆盖率 |
| 代码 | $2.5$ 到 $3.5$ 字符/token | 关键字和缩进模式影响较大 |

### 3.2 压缩率如何变成 decode 步数

对同一段文本，生成步数近似正比于 token 数，因此：

$$
N_{\mathrm{decode}} = \frac{N_{\mathrm{chars}}}{\operatorname{CR}}
$$

若把每一步生成的主干计算近似记成 $F_{\mathrm{body}}$，则总 decode 代价可以写成：

$$
\operatorname{DecodeFLOPs} \propto \frac{N_{\mathrm{chars}}}{\operatorname{CR}}\left(F_{\mathrm{body}} + Vd\right)
$$

这个式子清楚说明了 tokenizer 的核心权衡：提高 $\operatorname{CR}$ 能减少步数，但增大 $V$ 又会抬高每步输出层成本。

### 3.3 工程上通常怎么判断划算

对大多数生成场景来说，较高的压缩率通常是划算的，因为 decode 步数减少会同时降低时延和 KV Cache 消耗。真正需要警惕的是词表过大带来的 embedding 冗余和低频 token 欠训练问题。

---

## 4. 特殊 Token 与 Chat Template

| Token | 含义 | 用途 |
|-------|------|------|
| `<bos>` | Beginning of Sequence | 序列起始标志 |
| `<eos>` | End of Sequence | 序列结束，触发停止 |
| `<pad>` | Padding | batch 对齐，通常不计入 loss |
| `<unk>` | Unknown | byte-level BPE 往往不需要 |
| `<\|im_start\|>` / `<\|im_end\|>` | Chat Template | 多轮对话与角色分隔 |

special token 的设计会直接影响 instruction tuning、tool call 格式以及 prefix caching 的稳定性，所以它不是附属细节，而是 tokenizer 设计的一部分。

---

## 5. 为什么仅靠 BPE 不够

### 5.1 SentencePiece 的出发点

纯 BPE 很适合高频 merge，但它默认你已经决定了基本切分方式。SentencePiece 的出发点是把文本当成原始字符串或字节流，不依赖外部分词器，因此对多语言系统更稳，也更适合统一训练流程。

### 5.2 Unigram 模型用概率视角选择切分

Unigram 与 BPE 的思路相反：它通常从一个较大的候选词表开始，再逐步剪枝。其目标函数可写成：

$$
\mathcal{L}(\mathcal{V}) = \sum_S \log \sum_{\mathbf{x} \in \operatorname{Seg}(S)} \prod_{x_i \in \mathbf{x}} p(x_i)
$$

每一步移除使似然下降最小的 token：

$$
t^* = \arg\min_{t \in \mathcal{V}} \left(\mathcal{L}(\mathcal{V}) - \mathcal{L}(\mathcal{V} \setminus \{t\})\right)
$$

这让 Unigram 天然支持对多种分词方式做概率采样，因此也更适合把 tokenizer 当成一种正则化手段。

### 5.3 BPE、SentencePiece 与 Unigram 的关系

| 方案 | 关键想法 | 优势 | 代价 |
|------|----------|------|------|
| BPE | 反复合并最高频 token 对 | 简单、稳定、工业界最常见 | 分词路径通常是确定性的 |
| SentencePiece | 不依赖外部分词器的训练框架 | 多语言和端到端训练更统一 | 仍需选择 BPE 或 Unigram 等具体算法 |
| Unigram | 用概率模型保留多种切分候选 | 采样式分词更自然 | 训练和解释都更复杂 |

---

## 6. Tokenizer 对推理系统的端到端影响

| 方面 | 影响 |
|------|------|
| 上下文窗口 | 模型的 $T_{\max}$ 是 token 数上限，$\operatorname{CR}$ 越高，同样窗口能承载的文本越多 |
| KV Cache 显存 | 与 token 数近似成正比，压缩率更高通常意味着更省 KV |
| Prefix Caching | 匹配发生在 token 级，不同 tokenizer 会改变 prefix 粒度 |
| 计费与吞吐 | 商业 API 常按 token 计费，token 少意味着成本与时延都更低 |
| 评测一致性 | tokenizer 变化会影响 perplexity、pass@k 乃至 prompt 模板兼容性 |

### 6.1 选 tokenizer 时至少要检查的四个量

- 词表大小 $V$ 是否把 embedding 和 LM head 推到不可接受的比例。
- 压缩率 $\operatorname{CR}$ 在英文、中文、代码三类主要数据上是否稳定。
- special token 与 chat template 是否和训练、推理框架一致。
- prefix caching、多语言覆盖和代码场景下的切分是否足够稳定。

---

## 7. 面试一句话

> Tokenizer 决定的是模型看到信息的粒度。词表太小会浪费 decode 步数，词表太大会抬高 embedding 和 softmax 成本。BPE 本质上是在做基于频率的数据压缩，而压缩率、词表大小和推理成本之间存在直接的工程权衡。
