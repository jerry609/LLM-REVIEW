# 主流大模型发展脉络与核心优化点（详细版）

> Qwen / DeepSeek / GLM / Kimi / MiniMax + Llama / Mistral — 发展脉络 + 横向深度对比
> 本文重点是**跨公司技术路线对比**和**每一代升级的 Why**，各家详细技术参见单独笔记

---

## 一、全景时间线 (2023-2025)

```
2023 ──────────────────────────────────────────────────────────────────
 02  LLaMA 7B/13B/33B/65B (Meta)     ← 开源LLM时代开始, 模型泄露引爆社区
 03  ChatGLM-6B (智谱)                ← Prefix LM, 国产首个好用开源对话模型
 03  Alpaca / Vicuna / …              ← Self-Instruct 合成数据微调热潮
 07  Llama 2 7B/13B/70B (Meta)        ← 正式开放商用, RLHF 对齐
 07  ChatGLM2-6B (智谱)               ← 转向标准 Decoder, 性能翻倍
 08  Qwen-7B (阿里, 内测)             ← 首发, 尚未开源
 09  Qwen-7B/14B 正式开源              ← 较好的中文能力, Apache 2.0
 10  Kimi Chat 上线 (月之暗面)          ← 产品形态, 200K 超长上下文引爆市场
 10  Mistral 7B (Mistral AI)           ← Sliding Window Attention, 小而强
 11  DeepSeek-67B (深度求索)            ← 首发 dense 模型, 对标 LLaMA 70B
 12  Mixtral 8x7B (Mistral)            ← 开源 MoE, 性价比炸裂
 12  Mamba (Tri Dao)                   ← SSM 挑战 Transformer, 引爆讨论

2024 ──────────────────────────────────────────────────────────────────
 01  Qwen1.5 系列                       ← 多语言增强, 修复 1.0 短板
 01  MiniMax-abab6.5                    ← Lightning Attention 初版试水
 03  DeepSeek-V2 (236B, 21B active)     ← ⭐ MLA + DeepSeekMoE, KV Cache 压缩16×
 04  Llama 3 8B/70B (Meta)              ← 15T tokens, 128K context, GQA 标配
 05  Qwen2 0.5B-72B                     ← 7T tokens, 全系列 GQA+RoPE+SwiGLU 统一
 05  GLM-4-9B / GLM-4-Plus             ← 放弃 Prefix LM 转向 Causal LM
 06  DeepSeek-Coder-V2 (236B MoE)      ← 代码 MoE, FIM 训练
 07  Llama 3.1 8B/70B/405B             ← 405B 开源最大 dense, 128K native
 07  Mistral Large 2                    ← 123B, function calling
 09  Qwen2.5 0.5B-72B                  ← ⭐ 18T tokens, YaRN 128K→1M, 合成数据增强
 09  Qwen2.5-Coder / Math             ← 代码/数学专项子系列
 10  CogVideoX (智谱)                  ← 3D VAE + DiT 视频生成
 10  Kimi k1.5 技术报告                 ← Long-context RL + Long2Short
 11  QwQ-32B (阿里)                    ← 类 o1 推理模型, 基于 Qwen2.5-32B
 12  DeepSeek-V3 (671B, 37B active)    ← ⭐⭐ MLA+MoE+MTP+FP8, $5.5M 训练

2025 ──────────────────────────────────────────────────────────────────
 01  MiniMax-01 (4560B, 45.9B active)  ← ⭐ Lightning+Softmax Hybrid, 4M context
 01  DeepSeek-R1 + R1-Zero              ← ⭐⭐⭐ 纯 RL 涌现推理, GRPO, 全开源, 6蒸馏版
 01  Qwen2.5-VL                         ← 动态分辨率 + M-RoPE, 20min+ 视频
 01  Kimi k1.5                          ← Long-context RL(128K) + Long2Short + 多模态推理
 01  o3-mini (OpenAI)                   ← 推理轻量版, low/medium/high 三档
 01  DeepSeek Janus-Pro                 ← 统一理解+生成 多模态, Decoupled Visual Encoder
 01  Mistral Small 3 (24B)              ← 开源最强 24B, Apache 2.0
 02  Qwen2.5-Max                        ← MoE 旗舰, 追赶 DeepSeek-V3
 02  Kimi k1.5 Long Thinking            ← 超长思维链 + 128K context RL
 02  Slime (智谱/THUDM)                 ← 异步 RL 训练框架, SGLang+Megatron
 02  GLM-Z1 / Z1-Air (智谱)             ← 推理模型, Deep Thinking + Web Search
 02  GPT-4.5 / Orion (OpenAI)           ← 最大 Dense 模型, 世界知识+低幻觉
 03  QwQ-32B 正式版 (阿里)               ← Apache 2.0 开源, AIME 79.5%, LiveCodeBench 63.4%
 03  DeepSeek-V3-0324                   ← V3 能力刷新, 推理/代码/指令遵循增强
 03  Gemini 2.5 Pro (Google)            ← 内置 Thinking, 代码/数学/推理全面领先
 04  Llama 4 Scout/Maverick (Meta)      ← Meta 首次 MoE, Scout 109B active, 10M context
 04  Qwen3 (0.6B-235B)                 ← Thinking+Non-thinking 双模式, MoE 235B (22B active)
 05  Claude 3.5 Opus (Anthropic)           ← Anthropic 最强模型, 复杂推理能力大幅提升
 05  Grok-2 (xAI)                        ← Elon Musk 的 xAI, 多模态 + 实时联网
 06  Nemotron-4 340B (NVIDIA)             ← NVIDIA 发布大模型, 专注合成数据生成
 06  Yi-Lightning (零一万物)               ← API 定价极低, 引发价格战
 07  Mistral Large 2 (123B)               ← 对标 GPT-4, function calling 增强
 08  OpenAI o1-preview                    ← 推理模型首次公开预览
 09  Llama 3.2 (1B/3B + Vision)           ← 端侧 + 多模态
 10  Claude 3.5 Haiku + Computer Use      ← Agent 操作电脑, SWE-bench 49%
 11  QwQ-32B-Preview                      ← 类 o1 推理, 开源
 12  o1 正式版 + Gemini 2.0 Flash         ← 推理竞赛白热化
 05  DeepSeek-Prover-V2 (2025.05)         ← 形式化数学证明, IMO 级别能力
 06  Qwen3-Coder / Qwen3-Math            ← Qwen3 代码/数学子系列
 07  Claude 4 (Anthropic)                 ← 新一代旗舰, 编程+推理大幅提升
 07  o3 正式版 (OpenAI)                   ← 推理模型旗舰, ARC-AGI 突破
 08  Llama 4 Behemoth (Meta)              ← 2T+ MoE, 训练中
 09  Gemini 2.5 Ultra (Google)            ← Google 最强推理模型
 10  DeepSeek-V4 预告                     ← 新一代架构, 更大规模 MoE
 11  MiniMax-02 (预期)                    ← Hybrid 架构 v2, 更长 context
 12  各家持续迭代中

2026 ──────────────────────────────────────────────────────────────────
 01  RWKV-7 稳定版 + RWKV-8 实验           ← 广义 Delta Rule, 无自注意力 SoTA
 01  各家 Agent 框架成熟                    ← MCP 协议统一, Claude Computer Use 2.0
 02  Qwen3.5 (阿里, 2026.02.17)           ← ⭐ 最新！聚焦 AI Agent, 中国 AI 竞争重心转向 Agent
 02  当前状态 (2026.02.19)                 ← 推理模型普及, MoE 标配, Agent 框架爆发, Qwen3.5 刚发布
```

---

## 二、各家逐代演进详解——不只是 What，更是 Why

### 2.1 DeepSeek（深度求索）

```
V1 (67B dense, 2023.11)
│   问题：dense 模型推理贵, 参数效率低
│   思路：能不能只激活一部分参数?
▼
V2 (236B MoE, 21B active, 2024.03) ⭐ 转折点
│   核心：MLA + DeepSeekMoE
│   · MLA: KV 投影到低秩空间, KV Cache 压缩 16×
│     → Why: GQA 虽然减少 KV head 数, 但压缩比有限;
│       MLA 用 learned projection 实现更激进的压缩
│   · 细粒度 MoE: 160 routed + 2 shared experts
│     → Why: 小 expert 组合灵活性远大于大 expert
│   · 效果: 21B 激活参数 ≈ 70B dense 性能
│   影响：证明 MoE + 架构创新 > 堆参数
▼
V3 (671B MoE, 37B active, 2024.12) ⭐⭐
│   三大升级 + Why：
│   ① Auxiliary-Loss-Free Load Balancing
│      → Why V2 的 aux loss 会干扰主任务梯度;
│        bias 动态调整不改变 loss landscape
│   ② FP8 Mixed Precision Training
│      → Why 2048×H800 的通信瓶颈;
│        FP8 让 GEMM 和通信都减半, 总成本 $5.5M
│   ③ Multi-Token Prediction (MTP)
│      → Why 提供更密集的训练信号;
│        推理时可作为 speculative decoding 的 draft head
│   工程意义: 性能追平 GPT-4o, 训练成本低 20×
▼
R1 (基于 V3 + GRPO, 2025.01) ⭐⭐⭐
│   核心: 纯 RL 涌现推理能力 (不依赖 PRM)
│   → Why: SFT 只能模仿推理格式, 真正的推理能力需要
│     RL 的探索-利用机制来激发
│   4 阶段: cold start SFT → RL (GRPO) → rejection sampling + SFT → 全任务 RL
│   关键发现:
│   · 推理行为从 RL 中涌现 (self-verification, reflection)
│   · GRPO 去掉 critic → 显存减半, 训练更简单
│   · 蒸馏 > 小模型直接 RL (R1-Distill-32B > o1-mini)
│   R1-Zero: 纯 RL 零 SFT 启动, 验证 RL 涌现推理的可能性
│   蒸馏版: 6 个模型 (Qwen2.5-1.5B/7B/14B/32B, Llama3.1-8B/70B)
▼
Janus-Pro (2025.01) 多模态
│   统一理解+生成, Decoupled Visual Encoder
▼
V3-0324 (2025.03) 持续迭代
    V3 能力刷新, 推理/代码/指令遵循增强
```

**DeepSeek 技术理念**：用架构创新（MLA、细粒度 MoE）和训练工程（FP8、DualPipe）降低成本，而非简单堆资源。

---

### 2.2 Qwen（通义千问，阿里）

```
Qwen-7B/14B (2023.09)
│   定位：首发开源中文模型, 对标 Llama
│   问题：数据量不足(~3T), 多语言能力弱
▼
Qwen1.5 (2024.01)
│   升级：多语言增强, 数据质量提升
│   → Why: Qwen1 中文好但英文/多语言差, 社区反馈强烈
│   问题：架构不统一(部分用 MHA, 部分用 GQA)
▼
Qwen2 (2024.05)
│   三大升级 + Why：
│   ① 全系列统一 GQA + RoPE + SwiGLU + RMSNorm
│      → Why: 与 Llama 3 对齐, 降低社区迁移成本
│   ② 数据量 3T → 7T tokens
│      → Why: Chinchilla 过训练路线, 小模型吃更多数据
│   ③ 完整模型矩阵 0.5B-72B
│      → Why: 覆盖端侧→云端全场景, 抢占生态位
▼
Qwen2.5 (2024.09) ⭐ 全面升级
│   四大升级 + Why：
│   ① 数据量 7T → 18T tokens (公开最大)
│      → Why: 数据 scaling 仍是最稳健的提升手段
│   ② 合成数据大规模引入
│      → Why: 真实高质量数据枯竭, 数学/代码靠合成
│   ③ YaRN 128K → Turbo 1M context
│      → Why: 长上下文是刚需, Kimi 的成功验证了市场
│   ④ 退火配比: 后期增加代码/数学比例
│      → Why: 基础能力先到位, 再针对性强化推理
│   子系列：Coder (FIM), Math (TIR + Python interpreter)
▼
QwQ-32B (2024.11)
│   定位：类 o1 推理模型
│   → Why: DeepSeek-R1 证明了 RL 推理路线的价值
│   方法：基于 Qwen2.5-32B, Long-CoT SFT + RL
▼
Qwen2.5-VL (2025.01)
│   定位：多模态视觉语言模型
│   创新：动态分辨率 + M-RoPE (3D 位置编码)
│   → Why: 固定分辨率 ViT 会损失图片信息;
│     M-RoPE 统一图片/视频/文本的位置编码
▼
Qwen2.5-Max (2025.02)
│   定位：MoE 旗舰, 追赶 DeepSeek-V3
│   → Why: Dense 72B 的参数效率瓶颈, MoE 是必然方向
▼
Qwen3 (2025.04) ⭐ 新一代
│   全系列: 0.6B-235B (Dense + MoE)
│   核心创新: Thinking + Non-thinking 双模式统一
│   → Why: 推理模型 (长 CoT) 和对话模型 (快速响应) 需求并存;
│     /think 和 /no_think 切换, 一个模型覆盖两种场景
│   MoE 旗舰: 235B (22B active), 128 experts
│   训练: 4 阶段 (预训练 → 长 CoT cold start → RL → 双模式融合)
│   119 种语言支持
│   技术报告: arXiv:2504.07491
▼
Qwen3-Coder / Qwen3-Math (2025.06)
│   代码/数学专项子系列, 延续 Qwen3 双模式能力
▼
Qwen3.5 (2026.02.17) ⭐⭐ 最新发布！
    定位：AI Agent 原生模型, 标志中国 AI 竞争从对话向 Agent 转变
    → Why: 单纯对话/推理能力已趋同质化, Agent 能力成为下一个差异化点
    → CNBC 报道: "Alibaba unveils Qwen3.5 as China's chatbot race shifts to AI agents"
    核心方向: Agent 框架协同, 工具调用增强, 多步推理+行动
    → 阿里整个 Qwen 系列从 1.0→1.5→2→2.5→3→3.5, 每代半年+迭代
```

**Qwen 技术理念**：「数据为王 + 全系列覆盖 + Agent 生态」—— 用最大规模数据训练全覆盖的模型矩阵，配合最友好的开源协议抢占社区生态；3.5 代起全面拥抱 AI Agent。

---

### 2.3 GLM / ChatGLM（智谱 AI）

```
GLM (2022, 学术论文)
│   创新：Prefix LM + 2D Positional Encoding
│   → Why: 既要理解（双向）又要生成（单向）
│   问题：工程复杂, KV Cache 不友好
▼
ChatGLM-6B (2023.03) ← 国产开源第一枪
│   意义：第一个好用的中文开源对话模型
│   但仍用 Prefix LM 架构
▼
ChatGLM2-6B (2023.07)
│   关键转向：开始引入 GQA, 部分转向 Causal
│   → Why: Prefix LM 在推理时效率低,
│     且 Llama 生态证明 Causal LM 才是主流
▼
GLM-4-9B / GLM-4-Plus (2024)
│   ① 完全转向 Causal LM (GQA + RoPE + SwiGLU)
│      → Why: 社区工具链(vLLM/TGI)都针对 Causal LM 优化,
│        Prefix LM 享受不到这些工程红利
│   ② GLM-4-Long: 1M context
│   ③ 闭源 GLM-4-Plus 在中文 benchmark 领先
│
├── CogVLM2 (2024): Visual Expert 架构
│   → Why: 普通 VLM 让视觉和文本共享 FFN 会互相干扰;
│     独立 Visual Expert 解决模态冲突
│
├── CogVideoX (2024): 3D VAE + DiT 视频生成
│   → Why: 多模态全面布局, 视频生成是下一个竞争焦点
▼
GLM-Z1 / Z1-Air (2025.02) ⭐ 推理模型
│   Deep Thinking 模式 + Web Search 联网检索增强
│   → Why: 推理模型趋势下, 智谱也推出自己的推理方案
│   → 闭源 API, Z1-Air 为轻量版
▼
Slime (2025.02) ← RL 框架
    创新: 异步 RL 训练, SGLang + Megatron-LM 组合
    → Why: 
    · 同步 RL (VERL) 在 prompt/response 长度不一时 GPU 空闲;
    · SGLang 的 RadixAttention 对 RL 中重复 prefix 缓存效率高;
    · Megatron-LM 的 3D 并行适合大模型训练
    开源: https://github.com/THUDM/slime
```

**智谱技术理念**：「学术创新 + 多模态全栈」—— 清华系学术底蕴，从架构创新（Prefix LM / Visual Expert）到框架创新（Slime），同时在文本/视觉/视频全面布局。

---

### 2.4 Kimi（月之暗面 / Moonshot AI）

```
Kimi Chat 上线 (2023.10) ← 超长上下文先行者
│   定位：200K context, 国内第一个产品化长上下文
│   → Why: 长文档分析是高价值场景 (法律/金融/研报)
│   技术栈：
│   · Sliding Window + Global Attention 混合
│   · KV Cache 分层存储 (HBM → DRAM → SSD)
│   · Chunk Prefill + 流水线化
│   市场影响：引爆长上下文需求, 各家跟进
▼
Kimi (2024 持续迭代)
│   问题：长上下文虽长但推理能力一般
│   用户需求：不仅要"能读"还要"能推理"
▼
Kimi k1.5 (2025.01) ⭐
│   三大创新 + Why：
│   ① Long Context Scaling for RL
│      → Why: 推理模型的 CoT 越长质量越高 (test-time compute scaling),
│        但传统 RL 训练 context 只有 8K-16K;
│        Kimi 逐步扩展到 128K, 让模型学会更长推理链
│   ② Long2Short 蒸馏
│      → Why: Long-CoT 模型推理成本高;
│        蒸馏到短 CoT 模型, 保留质量降低成本
│   ③ 多模态推理
│      → Why: 图表/几何等视觉数学题是硬需求
│   独特优势: 业界首个在 RL 中跑到 128K context 的方案
▼
k0-math (2025.01)
│   数学推理专项模型
▼
Kimi k1.5 Long Thinking (2025.02)
    进一步: 超长思维链, 思考步骤可达数千步
    → Why: 复杂问题需要更深的推理链,
      128K RL context 让模型有空间展开长推理
```

**Kimi 技术理念**：「长上下文 = 核心壁垒」—— 从产品形态到推理框架到 RL 训练，一切围绕超长上下文优化。独到的 Long-context RL + Long2Short 蒸馏形成完整的长思维链闭环。

---

### 2.5 MiniMax

```
MiniMax-abab 系列 (2023)
│   定位：对标 GPT-3.5, 闭源 API 服务
│   问题：标准 Transformer 的 O(n²) 限制了上下文长度
│   思路：能不能用 Linear Attention 突破?
▼
MiniMax-abab6.5 (2024.01)
│   创新：Lightning Attention 初版
│   → Why: 
│   · 当时 Linear Attention 被认为效果不如 Softmax;
│   · MiniMax 发现关键: 不是全用 Linear, 而是 **混合**;
│   · 底层 Linear + 顶层 Softmax = 效率 + 精度
│   验证：小规模实验证明 Hybrid 可行
▼
MiniMax-01 (2025.01) ⭐ Hybrid 架构里程碑
    技术栈：
    ① 4560B MoE (45.9B active, 32 experts, top-2)
    ② Lightning Attention Hybrid
       · Layer 1-27: Lightning Attention (Linear) ← 2/3 的层
       · Layer 28-80: Softmax Attention           ← 1/3 的层
       → Why 这个分层比例?
         底层: 主要做局部特征提取, 不需要精确全局 attention
         顶层: 需要精确推理和 retrieval → 必须 Softmax
    ③ 4M token context (公开模型最长)
       → Why Linear Attention 层不需要 O(n²):
         KV 信息压缩在固定大小的 running state 中
    ④ 渐进式长度扩展: 4K → 32K → 512K → 4M
       → Why: 一步到位训太贵, 渐进式节省 90%+ compute
    开源: 权重 + 技术报告, 社区可复现
```

**MiniMax 技术理念**：「架构突破 > 数据堆叠」—— 用 Hybrid Attention（Linear + Softmax）从根本上解决长上下文的 O(n²) 瓶颈，4M context 是所有公开模型最长。

---

### 2.6 RWKV（RNN 复兴者）

```
RWKV-4 (2023.05, EMNLP 2023)
│   定位：首个实用的 "纯 RNN" 语言模型, 可并行训练
│   创新：WKV (Weighted Key-Value) 替代 Self-Attention
│   → Why: Transformer 的 O(n²) 在长序列上太贵
│   → 但仍有表达力限制
▼
RWKV-5 (Eagle, 2024 Q1)
│   升级：Multi-headed Linear Attention 变体
│   → Why: RWKV-4 表达力弱于同规模 Transformer
▼
RWKV-6 (Finch, 2024.04)
│   升级：矩阵值状态 + 动态递归
│   → Why: 标量状态压缩太狠, 矩阵值状态保留更多信息
│   → 在 1.5B/3B 规模上接近 Transformer 性能
▼
RWKV-7 (Goose, 2025.03) ⭐ 重大突破
│   论文: arXiv:2503.14456
│   核心: 广义 Delta Rule (Generalized Delta Rule)
│   → Why: 传统 RNN 的状态更新太"机械";
│     Delta Rule 让状态更新有选择性, 类似 attention
│   → 2 层即可实现 NC¹ 复杂度的 S₅ 状态跟踪
│   → 4 层可识别所有正则语言 (超越 Transformer 的 TC⁰)
│   性能: RWKV-7-World 3B 达开源 SoTA 语言建模 ⭐
│         (训练数据远低于 Qwen2.5/Llama3.2)
│   优势: 恒定显存占用, 恒定推理速度, "无限" context
│         100% 不含自注意力, 但可并行训练
▼
RWKV-8 (实验中, 2026)
    新特性:
    · DeepEmbed: 端侧友好的稀疏 MoE 设计
    · DeepEmbedAttention: 精简 KV 缓存 (适配 Hybrid)
    · ROSA (Online Suffix Automaton): 新型记忆机制
    → Why: RWKV-7 虽强但缺少 MoE 和 Hybrid 能力;
      RWKV-8 探索 SSM + Sparse + Attention 融合
```

**RWKV 技术理念**：「RNN 不是过去式」—— 用恒定内存和 O(n) 计算实现 Transformer 级别性能，最适合端侧部署和超长上下文场景。

---

### 2.7 国际参照：Llama / Mistral / OpenAI / Google / Anthropic

```
Llama 系列 (Meta) ────────────────────────────
Llama 1 (2023.02): 开源的号角, 7B-65B
│   → 泄露后引爆社区, 证明开源可以追闭源
▼
Llama 2 (2023.07): 7B/13B/70B, 开放商用
│   首次公开 RLHF 流程细节
│   Ghost Attention 保持多轮指令遵循
▼
Llama 3 (2024.04): 8B/70B ⭐
│   → 15T tokens (3× Llama 2), 128K context
│   → GQA 标配, SwiGLU 标配
│   → 关键: 数据 scaling > 架构创新
▼
Llama 3.1 (2024.07): 8B/70B/405B
│   → 405B 开源最大 dense, 128K native
│   → 4D 并行 (TP+PP+CP+DP)
│   → 意义: 证明 dense 模型可以到 400B 级别
▼
Llama 3.2 (2024.09): 1B/3B 端侧 + 11B/90B Vision
▼
Llama 3.3 70B (2024.12): 性能追平 405B, 推理成本大幅下降
▼
Llama 4 Scout/Maverick (2025.04): ⭐ Meta 首次 MoE
    Scout: 109B active / ~400B total, 10M context
    Maverick: 更大规模
    → Why: Dense scaling 到 405B 后成本难以为继, MoE 是必然

Mistral 系列 ─────────────────────────────────
Mistral 7B (2023.10): Sliding Window Attention
│   → 小而强, 引入 SWA 概念到主流
▼
Mixtral 8x7B (2023.12): 开源 MoE 先驱
│   → 证明 MoE 在开源社区可行
▼
Mistral Large 2 (2024.07): 123B
│   → 闭源对标 GPT-4
▼
Mistral Small 3 (2025.01): 24B, Apache 2.0
    → 开源最强 24B 级模型

OpenAI 系列 ──────────────────────────────────
GPT-4 (2023.03) → GPT-4 Turbo (2023.11) → GPT-4o (2024.05)
│   → 多模态原生化 + 速度提升 + 价格降低
▼
o1-preview (2024.09) → o1 (2024.12): ⭐ 推理模型开创者
│   → 内部 Chain-of-Thought, AIME 96.4%
▼
o3-mini (2025.01): 推理轻量版
│   → low/medium/high 三档, 性价比优于 o1
▼
GPT-4.5 / Orion (2025.02): 最大 Dense 模型
    → 世界知识 + 低幻觉 + EQ

Google Gemini 系列 ───────────────────────────
Gemini 1.0 (2023.12) → 1.5 Pro (2024.02, 1M ctx) → 2.0 Flash (2024.12)
▼
Gemini 2.5 Pro (2025.03): ⭐ 内置 Thinking
    → 代码/数学/推理全面领先

Anthropic Claude 系列 ─────────────────────────
Claude 2 (2023.07) → Claude 3 (2024.03) → 3.5 Sonnet (2024.06) ⭐
│   → 编程最强, 200K context, Computer Use (Agent)
▼
Claude 3.5 Sonnet v2 (2024.10): SWE-bench 49%
```

---

## 三、横向技术路线深度对比

### 3.1 Attention 机制演进对比

```
                    DeepSeek        Qwen           MiniMax         Kimi          GLM
2023              MHA              MHA             Softmax         混合*          Prefix LM
2024 H1           MLA (V2) ⭐     GQA (Qwen2)    Lightning 初版  Sliding+Global GQA (GLM-4)
2024 H2           MLA (V3)        GQA (Qwen2.5)  Lightning       持续优化       GQA
2025              MLA              GQA+MoE        Hybrid 成熟 ⭐   长 context RL  GQA + Slime RL
```

**关键洞察**：
- **MLA vs GQA**：都是压缩 KV Cache，MLA 更激进（16× vs 8×），但实现复杂度更高
- **Lightning Attention**：唯一从根本改变 attention 计算方式的路线
- **所有人都收敛到**：RoPE + SwiGLU + RMSNorm + Pre-Norm（"Llama 架构"成为标准）

### 3.2 KV Cache 压缩策略对比

| 方法 | 压缩方式 | 压缩比 | 代表 | 精度损失 |
|------|---------|--------|------|---------|
| **MHA** | 无压缩 | 1× | 原始 Transformer | 无 |
| **GQA-8** | 减少 KV head | ~8× | Llama 3, Qwen2.5 | 极小 |
| **MLA** | 低秩投影 | **~16×** | DeepSeek V2/V3 | 几乎无 |
| **Lightning** | Linear Attention 层无 KV | **部分层 ∞** | MiniMax-01 | 底层略降 |
| **KV Quant** | FP8/INT8 KV | 2-4× | vLLM, TRT-LLM | 极小 |
| **Offloading** | 分层存储 | 不压缩 | Kimi, MiniMax | 无(增加延迟) |

### 3.3 数据策略对比

| 维度 | DeepSeek | Qwen | Llama 3 | GLM | MiniMax |
|------|---------|------|---------|-----|---------|
| **预训练规模** | 14.8T | **18T** ⭐ | 15T | 未公开 | 未公开 |
| **合成数据** | ✅ 数学+代码 | ✅ **大量** ⭐ | ✅ 代码 | ✅ | ✅ |
| **数据筛选** | 质量分类器 | **Llama 式 bootstrap** | 3 重去重 | 未公开 | 未公开 |
| **配比策略** | 领域增强 | **退火配比** ⭐ | 退火配比 | 未公开 | 未公开 |
| **代码训练** | FIM + 执行验证 | FIM + TIR | FIM | FIM | 未公开 |

**关键洞察**：
- Qwen 和 Llama 都走「数据为王」路线：更多更好的数据 > 架构创新
- DeepSeek 走「架构 + 工程」路线：用 MoE + FP8 降低计算，而非堆数据
- 退火配比（训练后期增加代码/数学）被 Qwen 和 Llama 3 独立验证有效

### 3.4 长上下文技术路线对比

| 维度 | Kimi | MiniMax | Qwen | DeepSeek | Llama |
|------|------|---------|------|---------|-------|
| **最长 context** | **2M** | **4M** ⭐ | 1M (Turbo) | 128K | 128K |
| **扩展方法** | 工程 + Offload | **Hybrid Attention** | YaRN | RoPE ABF | 原生训练 |
| **KV 存储** | 分层 HBM→DRAM→SSD | Linear层无需KV | 标准 | MLA 压缩 | 标准 |
| **训练方式** | 未公开 | **渐进扩展** (4K→4M) | 短训长推 | 128K 原生 | 128K 原生 |
| **代价** | 推理慢(Offload) | 底层精度略降 | YaRN 需微调 | 128K 已够 | 128K 已够 |

**关键洞察**：
- **Kimi**：用工程手段（Offloading + 分层存储）暴力支持长上下文
- **MiniMax**：用架构手段（Linear Attention）从根本上解决 O(n²)
- **Qwen**：用数学手段（YaRN 频率插值）扩展 RoPE
- 三条路线各有 trade-off，没有绝对赢家

### 3.5 MoE 设计对比

| 维度 | DeepSeek-V3 | MiniMax-01 | Mixtral 8x7B | Qwen2.5-Max |
|------|------------|-----------|-------------|-------------|
| **总参数** | 671B | 4560B | 46.7B | 未公开 |
| **激活参数** | 37B | 45.9B | 13B | 未公开 |
| **Expert 数** | **256** | 32 | 8 | 未公开 |
| **Top-K** | 8 | 2 | 2 | 未公开 |
| **Shared Expert** | ✅ 1 个 | ❌ | ❌ | 未公开 |
| **负载均衡** | **Bias 动态调整** | Aux Loss | Aux Loss | 未公开 |
| **Expert 粒度** | ⭐ 极细 (256 小) | 粗 (32 大) | 中 (8 大) | — |

**关键洞察**：
- DeepSeek 的「多小 expert」vs MiniMax 的「少大 expert」是两条路线
- 细粒度 expert 组合空间大（C(256,8) >> C(32,2)），但路由通信更重
- Shared Expert 是 DeepSeek 独创：保底 + 路由 的组合

### 3.6 对齐/后训练路线对比

| 维度 | DeepSeek-R1 | Qwen/QwQ | Kimi k1.5 | GLM/Slime |
|------|------------|---------|-----------|-----------|
| **方法** | **GRPO** (纯 RL) | SFT + RL | **Long-context RL** | **异步 RL** |
| **Reward** | 规则 reward | Rule + Model | Rule + Model + Length | 未公开 |
| **Critic** | ❌ 不需要 | 需要 | 需要 | 未公开 |
| **RL Context** | 标准 (8-16K) | 标准 | **128K** ⭐ | 异步长短混合 |
| **创新点** | 纯 RL 涌现推理 | 快速跟进 | Long2Short 蒸馏 | SGLang RadixAttention |
| **工程框架** | 自研 | 自研 | 自研 | **Slime** (开源) |

**关键洞察**：
- R1 证明了 GRPO > PPO（不需要 critic，更简单更有效）
- Kimi 的独特性在 Long-context RL：128K context 让模型学习超长推理链
- Slime 的异步 RL 避免了同步 RL 的 GPU 空闲问题

### 3.7 训练成本/效率对比

| 模型 | 参数量 | 训练成本 | GPU | 训练时间 | 核心省钱手段 |
|------|--------|---------|-----|---------|------------|
| **DeepSeek-V3** | 671B | **$5.5M** ⭐ | 2048 H800 | ~2 月 | FP8 + MoE + DualPipe |
| Llama 3.1 405B | 405B | ~$100M+ | 16384 H100 | ~3 月 | 纯暴力堆资源 |
| Qwen2.5-72B | 72B | 未公开(估~$10M+) | 未公开 | 未公开 | 数据工程 |
| MiniMax-01 | 4560B | 未公开(估$20-50M) | 未公开 | 未公开 | Hybrid Attention |

**DeepSeek 成本低 20× 的秘密**：
1. MoE：每 token 只激活 37B/671B（5.5%）
2. FP8：GEMM + 通信都减半
3. DualPipe：减少 pipeline bubble
4. Auxiliary-Loss-Free：更高 GPU 利用率

---

## 四、关键行业拐点（面试必知）

### 拐点 1：开源 vs 闭源（2023）
```
LLaMA 泄露 → 社区爆发 → Llama 2 正式开源
→ 证明: 开源模型可以接近闭源水平
→ 影响: 各家纷纷开源 (Qwen, DeepSeek, GLM, MiniMax-01)
```

### 拐点 2：MoE 成为主流（2024）
```
Mixtral 8x7B → DeepSeek-V2 → DeepSeek-V3
→ 证明: 稀疏激活 >> 密集模型的性价比
→ 影响: Qwen2.5-Max, MiniMax-01 都转向 MoE
```

### 拐点 3：RL 推理涌现（2025.01）
```
OpenAI o1 (闭源) → DeepSeek-R1 (全开源)
→ 证明: 纯 RL 可以涌现推理能力，不需要 PRM
→ 影响: QwQ, Kimi k1.5, Slime 全部跟进 RL 路线
→ 余震: SFT+DPO 对推理任务不够，RL 回归
```

### 拐点 4：Hybrid 架构验证（2025.01）
```
MiniMax-01: Linear + Softmax Hybrid → 4M context
→ 证明: 纯 Transformer 不是唯一答案
→ 影响: Jamba, Zamba 等 Hybrid 架构陆续出现
→ 启示: 未来可能 SSM + Attention + MoE 三位一体
```

---

## 五、架构趋同与分化分析

### 5.1 已趋同的技术（面试不用对比，大家都一样）
| 组件 | 收敛方案 | 说明 |
|------|---------|------|
| FFN | SwiGLU | 比 ReLU/GELU 好，已无争议 |
| 归一化 | RMSNorm + Pre-Norm | 训练更稳定，速度更快 |
| 位置编码 | RoPE | ALiBi 已被边缘化 |
| 训练精度 | BF16 (→ FP8 趋势) | DeepSeek 领先用 FP8 |
| 预训练任务 | Causal LM | Prefix LM、MLM 已退出 LLM 赛道 |
| 对齐基线 | SFT → DPO/RLHF | 至少需要 SFT + 某种偏好优化 |

### 5.2 仍在分化的技术（面试重点对比）
| 维度 | 竞争方案 | 核心 trade-off |
|------|---------|--------------|
| **Attention** | MLA vs GQA vs Hybrid | 压缩比 vs 实现复杂度 |
| **MoE** | 细粒度(256) vs 粗粒度(8-32) | 灵活性 vs 通信开销 |
| **长上下文** | 架构(Linear) vs 工程(Offload) vs 数学(YaRN) | 根本解决 vs 快速可用 |
| **后训练** | GRPO vs PPO vs DPO | 效果 vs 工程复杂度 |
| **推理模式** | 长 CoT (R1) vs 短 CoT (蒸馏) | 质量 vs 延迟 |
| **Dense vs MoE** | Dense 72B vs MoE 671B(37B active) | 简单 vs 性价比 |

---

## 六、开源生态对比

| 维度 | Qwen | DeepSeek | Llama | GLM | MiniMax |
|------|------|---------|-------|-----|---------|
| **许可证** | Apache 2.0 ⭐ | MIT ⭐ | Llama License | GLM License | 开放权重 |
| **模型矩阵** | ⭐⭐⭐⭐⭐ (0.5B-72B+子系列) | ⭐⭐⭐ (V3+R1+蒸馏) | ⭐⭐⭐⭐ (8B-405B) | ⭐⭐ (9B) | ⭐⭐ (01) |
| **技术报告** | 详细 | **极详细** ⭐ | 详细 | 部分公开 | 详细 |
| **HuggingFace 下载** | 最高 | 极高 | 最高 | 中 | 中 |
| **社区生态** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **适合场景** | 通用基座 | 推理/代码 | 通用基座 | 中文场景 | 超长上下文 |

---

## 七、面试中如何谈论这些模型

### 7.1 总结框架（万能公式）
```
"关于 XX 模型，我的理解有三个层面：
 1. 它解决了什么问题（Why）
 2. 核心技术创新是什么（What + How）
 3. 和竞品的差异化在哪（对比思维）"
```

### 7.2 各家一句话定位（面试开场用）
| 模型 | 一句话 |
|------|--------|
| DeepSeek | "用架构创新（MLA+MoE）和训练工程（FP8）实现以少胜多，V3 性能追平 GPT-4o 但成本低 20×" |
| Qwen | "全系列覆盖（0.5B-72B）+ 超大数据（18T tokens）+ 最友好开源协议，是社区生态最完整的中文模型" |
| GLM | "清华系学术创新驱动，从 Prefix LM 到 Visual Expert 到 Slime RL 框架，多模态全栈布局" |
| Kimi | "超长上下文的先行者和深耕者，从 200K 产品化到 128K context RL，Long2Short 蒸馏解决长思维链的成本问题" |
| MiniMax | "唯一从架构层面解决 O(n²) 问题的玩家，Lightning Attention Hybrid 实现 4M context，公开模型最长" |

### 7.3 深度回答示例

#### 示例 1：「对比 DeepSeek-V3 和 Qwen2.5 的技术路线差异」
> **架构层面**：DeepSeek 走 MoE+MLA 路线，671B 总参数只激活 37B，靠 MLA 把 KV Cache 压缩 16×；Qwen 走 Dense+数据路线，72B 全量模型配 18T tokens 数据。
>
> **训练层面**：DeepSeek 用 FP8 训练省计算+省通信，Auxiliary-Loss-Free 省调参；Qwen 用标准 BF16 但数据量是 DeepSeek 的 1.2 倍。DeepSeek 花 $5.5M，Qwen 估计 $10M+。
>
> **哲学层面**：DeepSeek 信奉「聪明的架构 > 暴力堆数据」，Qwen 信奉「数据 scaling 最稳健」。两条路线目前效果接近，但 DeepSeek 的路线更适合进一步 scaling（MoE 可以继续增加 expert），Qwen 可能面临高质量数据枯竭的问题。

#### 示例 2：「MiniMax-01 的 Hybrid 架构设计思路」
> **问题**：4M tokens 的 Attention 需要 4M×4M 的 attention matrix，标准方法不可能。
>
> **洞察**：不是所有层都需要 O(n²) 的精确 attention。底层主要做局部特征提取（类似 CNN 的低层），用 O(n) 的 Linear Attention 足够；顶层需要全局推理和精确检索，才需要 Softmax Attention。
>
> **验证**：Needle-in-a-Haystack 测试表明，纯 Linear Attention 检索能力不足，但只要顶层 1/3 用 Softmax，就能恢复精确检索。
>
> **代价**：Linear Attention 层的表达能力弱于 Softmax，所以 MiniMax-01 用了更大的总参数（4560B）来补偿。这也是为什么它是所有模型中参数量最大的。

#### 示例 3：「为什么 DeepSeek-R1 的 RL 训练能涌现推理能力」
> **传统路线的局限**：SFT 只能让模型模仿推理格式（pattern matching），DPO 只能在现有数据的偏好空间内优化。但真正的推理需要模型探索新的推理路径。
>
> **RL 的独特价值**：GRPO 让模型对同一问题生成多个回答，用规则 reward（数学答案对错）打分，然后强化好的回答、抑制差的。这个过程中模型会自发地：
> 1. 学会 self-verification（回头检查）
> 2. 学会 reflection（"等等，我换个思路"）
> 3. 产生更长的 CoT（因为更多推理步骤 → 更高 reward）
>
> **关键发现**：这些行为没有在训练数据中显式教过，完全是 RL 的探索机制涌现的。这就是为什么 SFT 模仿不了——你不能模仿一个你没见过的推理模式。

#### 示例 4：「如果让你选开源模型做基座，怎么选？」
> **通用对话/中文场景** → Qwen2.5-72B：生态最完整、社区最活跃、Apache 2.0 无限制
> **推理增强/数学代码** → DeepSeek-R1-Distill-32B：推理能力 > o1-mini，性价比极高
> **超长上下文应用** → MiniMax-01：4M context 架构级解决，不是工程 hack
> **资源受限/端侧** → Qwen2.5-3B 或 Llama-3.2-3B：轻量但能力不错
> **视觉多模态** → Qwen2.5-VL：动态分辨率 + M-RoPE，视觉理解最强开源

---

## 面试高频问答（扩展版）

**Q1：对比 DeepSeek-V3 和 Qwen2.5-72B 的技术路线差异？**
> （见上方示例 1，能说 3 层：架构/训练/哲学）

**Q2：为什么 2025 年 RL 推理成为热点？SFT+DPO 不够吗？**
> R1 证明推理能力不能靠模仿（SFT），需要 RL 的探索-利用机制来涌现。核心区别：SFT 只能学到训练数据中已有的推理模式，RL 可以发现新的推理路径。GRPO 去掉 critic 让工程更简单，规则 reward 让推理任务训练成本很低。

**Q3：各家长上下文方案的 trade-off？**
> Kimi 用工程（Offloading+分层存储）—— 灵活但增加推理延迟；MiniMax 用架构（Linear Attention）—— 根本解决但底层精度略降；Qwen 用数学（YaRN 频率插值）—— 低成本但有外推上限。没有银弹，选什么取决于具体需求。

**Q4：MoE 的细粒度 expert（DeepSeek 256个）和粗粒度 expert（MiniMax 32个）有什么区别？**
> 细粒度 expert 的组合空间指数级增大（C(256,8) >> C(32,2)），模型可以为不同 token 精细选择最合适的 expert 组合。代价是 All-to-All 通信更多（256 个 expert 分布在更多 GPU 上）。DeepSeek 通过 Shared Expert 保底解决了部分 expert 退化的问题。

**Q5：如何看待 Hybrid 架构（Linear + Softmax）的未来？**
> MiniMax-01 证明了 Hybrid 可行且有显著优势（4M context）。趋势是未来可能 SSM + Attention + MoE 三位一体：SSM 层做高效全局压缩，Attention 层做精确推理，MoE 增加参数多样性。Jamba（AI21）和 Zamba 也在走这个方向。但目前 Hybrid 的生态（编译器、推理框架）还不如纯 Transformer 成熟。

**Q6：从技术报告看，哪家公司的技术透明度最高？**
> DeepSeek > Qwen ≈ MiniMax > Llama > GLM > Kimi。DeepSeek 的 V3 和 R1 技术报告几乎公开了所有训练细节（包括 loss 曲线、超参、成本），是业界标杆。这也是为什么 DeepSeek 的影响力远超其公司规模。

**Q7：如果面试时被问"你怎么看国产大模型的发展"？**
> "三个趋势：(1) 架构创新取代暴力 scaling（DeepSeek MLA、MiniMax Hybrid）；(2) RL 后训练成为标配（R1 → QwQ → Kimi k1.5）；(3) 专项子系列分化（Coder/Math/VL/Long）。核心竞争力已经从'谁有更多 GPU'转向'谁有更好的架构和数据工程'。"
