# 阿里巴巴（Alibaba）— LLM 推理工程师面试定向准备

> 目标岗位：LLM Inference / Serving Engineer
> 相关团队：PAI（平台 AI）、通义千问、达摩院、阿里云智能

---

## 一、公司技术栈与核心方向

### 1. 核心产品与平台
| 产品/平台 | 技术侧重 |
|-----------|----------|
| **通义千问（Qwen）** | 开源大模型系列，Qwen2.5 / QwQ 推理模型 |
| **PAI-EAS** | 弹性推理服务平台（Elastic Algorithm Service） |
| **RTP-LLM** | 自研高性能 LLM 推理引擎 |
| **百炼平台** | MaaS 平台，支持模型微调 + 推理部署 |
| **淘宝问问** | 淘宝 App 内嵌 AI 助手，多轮对话场景 |

### 2. 推理技术栈（RTP-LLM 为核心）
- **RTP-LLM**：阿里自研推理引擎，核心特性：
  - PagedAttention + Continuous Batching
  - **多轮对话 KV Cache 复用**：哈希路由 + Prefix Cache
  - **投机采样（Speculative Decoding）**：小模型 draft + 大模型 verify
  - 支持 WeightOnly 动态量化（INT8/INT4）
  - 支持多模态（文本 + 图像 + 音频）
  - 支持 LoRA / P-Tuning 在线切换
- **PAI-Blade**：自研编译优化工具（类似 TorchScript + 算子融合）
- **CUDA 算子库**：自研 FlashAttention 变体 + 量化 kernel

### 3. 核心技术关注点

#### (1) 多轮对话 KV Cache 复用（阿里独特优势）
**场景**：淘宝问问、钉钉 AI 等多轮对话应用
```
Round 1: [System Prompt + User Q1] → [Answer 1]
Round 2: [System Prompt + User Q1 + Answer 1 + User Q2] → [Answer 2]
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
         这段 prefix 的 KV 可以从 Round 1 复用！
```

**技术实现**：
1. **前缀树（Radix Tree）索引**：以 token hash 为 key，快速匹配最长公共前缀
2. **一致性哈希路由**：在转发层用 session_id 做哈希，保证同一用户的多轮请求路由到同一台推理机器
3. **LRU 驱逐**：显存不够时按 last_access_time 淘汰
4. **异步预热**：预测性地将可能复用的 KV 从 CPU 预取到 GPU

**效果**：
- First Token Time（FTT）降低 50-80%（长历史场景）
- 整体 GPU 利用率提升（减少重复 prefill）

#### (2) 投机采样（Speculative Decoding）
**实现要点**：
1. Draft model：小模型（如 Qwen-1.8B）自回归生成 N 个候选 token
2. Target model：大模型（如 Qwen-72B）一次 forward 验证 N 个 token
3. 接受/拒绝采样：保证最终输出分布与仅用大模型时完全一致（数学等价）
4. **与其他优化正交**：可以和 KV Cache 复用、量化、TP 等同时使用

#### (3) Qwen 系列模型推理特性
- **GQA**：Qwen2.5 使用 GQA 减少 KV Cache
- **SlidingWindow Attention**：部分 layer 使用滑动窗口
- **YaRN 长度扩展**：支持 128K+ context
- **MoE 版本**：Qwen-MoE 的 Expert 管理

---

## 二、高频面试题（8 道）

### Q1: 系统设计 — 设计淘宝问问的推理后端
**题目**：设计支持千万 DAU 的多轮对话推理服务

**考察要点**：
- 多轮对话的 KV Cache 复用策略
- 一致性哈希路由保证 session 亲和性
- 负载均衡 vs session 亲和性的冲突处理
- 流式输出（SSE/WebSocket）

**回答框架**：
```
1. 需求分析：
   - DAU 千万，峰值 QPS ~10 万
   - 多轮对话（平均 5 轮），SLO: TTFT < 300ms, TPOT < 40ms
   
2. 架构：
   - L7 Router（一致性哈希 by session_id）
     → Inference Pool（每台机器跑 RTP-LLM）
     → KV Cache（GPU HBM + CPU DRAM 两级）
   
3. KV Cache 复用：
   - 每台机器维护 Radix Tree 索引 prefix cache
   - 新请求先查 cache → hit: 跳过 prefix 的 prefill
   - miss: 完整 prefill → 结果写入 cache
   
4. 容错：
   - 机器故障 → router 重新哈希 → 新机器 full prefill（可接受的一次性开销）
   - Cache eviction → 退化为 full prefill（不影响正确性）
   
5. 扩缩容：
   - 基于 pending queue + GPU utilization 的 HPA
   - 扩容时一致性哈希最小迁移
```

### Q2: 深度题 — KV Cache 复用的一致性哈希
**题目**：为什么用一致性哈希？普通哈希取模有什么问题？

**考察要点**：
- 普通哈希：机器数变化时几乎所有 key 需要重新映射 → cache 全失效
- 一致性哈希：机器数变化时只有约 1/N 的 key 需要迁移
- 虚拟节点：解决哈希不均匀问题
- session 亲和性与负载均衡的平衡

**参考回答**：
```
一致性哈希核心：
- 将机器和 key 都映射到同一个哈希环（0 ~ 2^32-1）
- key 顺时针找到的第一个机器节点即为目标
- 每台物理机映射 K 个虚拟节点（K=100~200）提高均匀性

扩容场景：
- 新增 1 台机器（从 N 变为 N+1）
- 只有新机器顺时针方向到上一个虚拟节点之间的 key 需要迁移
- 影响比例 ≈ 1/(N+1)，远小于取模的 (N-1)/N

淘宝问问的实际考量：
- session_id 做 key（不是 user_id，因为同用户可能有多个会话）
- 权重一致性哈希：新机器初始权重低，渐进增加（预热 cache）
- fallback: 如果目标机器过载，允许 overflow 到邻近机器（牺牲 cache 复用换取低延迟）
```

### Q3: 深度题 — 投机解码（Speculative Decoding）
**题目**：详细解释投机解码的数学原理，为什么是无损的？

**考察要点**：
- 接受概率公式：`min(1, p_target(x) / p_draft(x))`
- 拒绝时的修正采样
- draft 长度 N 的选择
- draft model 质量的影响

**参考回答**：
```
算法流程：
1. Draft model 生成 N 个 token: x1, x2, ..., xN
   - 记录每个 token 的 draft 概率: q(x1), q(x2), ..., q(xN)

2. Target model 一次 forward 计算:
   - p(x1|context), p(x2|context,x1), ..., p(xN|context,...,xN-1)
   - 以及 p(xN+1|context,...,xN)

3. 从左到右验证：
   - 对 xi, 采样 r ~ Uniform(0,1)
   - 如果 r < min(1, p(xi) / q(xi)) → 接受
   - 否则 → 拒绝，从修正分布采样替代 token:
     p_adj(x) = max(0, p(x) - q(x)) / Z
   - 后续 token 全部丢弃

数学证明（无损性）：
- 对于被接受的 token xi:
  P(accept xi) × q(xi) = min(q(xi), p(xi))
- 加上拒绝时的修正采样:
  P(reject) × p_adj(xi) = (1 - min(1, p/q)) × max(0, p-q)/Z
- 总概率 = p(xi)（与直接用 target model 完全一致）

工程考量：
- N 太大：draft 准确率下降，拒绝增多，浪费 compute
- N 太小：加速比低
- 经验值：N=3~5，具体取决于 draft/target 的质量差异
- 最佳 N 可以动态调整（根据接受率）
```

### Q4: 深度题 — Qwen 架构细节
**题目**：Qwen2.5-72B 和 Llama3-70B 的架构有什么不同？对推理有什么影响？

**考察要点**：
- GQA head 数：Qwen 和 Llama 的 KV head 配置不同
- 位置编码：都是 RoPE，但 base frequency 不同
- 激活函数：Qwen 用 SwiGLU
- 长度支持：Qwen 支持 128K（YaRN 扩展）
- KV Cache 大小差异对显存的影响

### Q5: 深度题 — Prefix Caching 的 Radix Tree
**题目**：解释 Radix Tree 如何用于 Prefix Caching？和简单 Hash 有什么区别？

**考察要点**：
```
Radix Tree（基数树）用于 Prefix Cache：

结构：
root
├── [sys_prompt_tokens] → KV blocks for system prompt
│   ├── [user_q1_tokens] → KV blocks for Q1
│   │   └── [answer1_tokens] → KV blocks for A1
│   │       └── [user_q2_tokens] → KV blocks for Q2 (Round 2 查到这里)
│   └── [user_q3_tokens] → KV blocks for Q3 (不同会话)
└── [other_sys_prompt] → ...

查找流程：
1. 新请求 tokens: [sys_prompt, q1, a1, q2]
2. 从 root 开始逐段匹配
3. 匹配到的最长前缀 → 复用其 KV blocks
4. 未匹配部分 → prefill 并插入树中

vs 简单 Hash：
- Hash: 只能精确匹配完整 prefix → miss rate 高
- Radix Tree: 支持最长前缀匹配 → 即使部分重叠也能复用
- 例：Round 2 的 prefix 是 Round 1 prefix 的超集 → 增量 prefill 即可
```

### Q6: 系统设计 — PAI-EAS 弹性推理服务
**题目**：设计一个支持自动扩缩容的推理服务，应对潮汐流量

**考察要点**：
- 扩容指标：pending queue length / GPU utilization / P99 latency
- 缩容安全：确保正在 decode 的请求完成
- 冷启动优化：预加载模型权重 + 预热 KV Cache
- 成本考量：Reserved vs Spot 实例混合

### Q7: 量化推理
**题目**：RTP-LLM 中 WeightOnly INT8 量化如何实现？精度损失怎么控制？

**考察要点**：
- Per-channel symmetric quantization
- Activation 保持 FP16/BF16，只量化 weight
- 反量化在 GEMM kernel 内做（W8A16 kernel）
- 精度评估：PPL on calibration set
- SmoothQuant 处理 activation outlier

### Q8: 工程实践
**题目**：一个推理服务上线后，P99 延迟突然增大 3x，如何排查？

**考察要点**：
```
排查清单：
1. 监控面板：
   - GPU utilization 是否打满？ → compute-bound 还是 memory-bound
   - KV Cache hit rate 是否下降？ → 可能有大量 cache miss
   - Batch size 是否突变？ → 流量变化
   - 是否有 OOM / swap / recompute 事件？

2. 流量分析：
   - 请求长度分布是否变化？（某些用户发了超长文本）
   - QPS 是否突增？（需要扩容）
   - 是否有恶意请求？（限流策略）

3. 系统层：
   - GPU 温度是否过高导致降频？
   - PCIe 带宽是否被占满？（多租户争抢）
   - CUDA context switching overhead？

4. 解决方案：
   - 短期：扩容 + 限流 + 拒绝超长请求
   - 中期：优化 cache 策略 / 增加 Prefix Caching
   - 长期：PD 分离 / 更好的调度算法
```

---

## 三、阿里特色追问

1. **"你了解 RTP-LLM 吗？和 vLLM 有什么区别？"** → 自研引擎，深度定制多轮对话优化
2. **"Qwen 的开源策略对推理工程有什么影响？"** → 社区反馈驱动优化方向
3. **"PAI-EAS 和 K8s 的关系？"** → PAI-EAS 基于 K8s 但有自研调度器
4. **"如何做推理服务的 chaos engineering？"** → 模拟 GPU 故障、网络抖动、流量突增
5. **"你如何衡量推理优化的业务价值？"** → latency → conversion rate, cost → $/query

---

## 四、面试流程（典型）

| 轮次 | 内容 | 时长 |
|------|------|------|
| 一面（P6/P7） | 算法题 + LLM 基础知识 | 60 min |
| 二面（P7/P8） | 系统设计 + 项目深度 | 60 min |
| 三面（P8+） | 架构讨论 + 行为面试 | 45 min |
| HR 面 | 薪资/级别/团队 | 30 min |
| 交叉面（可能） | 其他团队技术负责人 | 45 min |

### 一面准备清单
- [ ] 算法题：LeetCode Medium（重点：字符串/数组/哈希/树）
- [ ] 手写 Attention 前向传播（含 GQA）
- [ ] KV Cache 显存计算公式
- [ ] 投机解码的接受概率公式

### 二面准备清单
- [ ] 完整系统设计：多轮对话推理服务
- [ ] KV Cache 复用的技术方案（Radix Tree + 一致性哈希）
- [ ] 量化方案对比（INT8 vs INT4 vs FP8）
- [ ] 项目经验中的量化数据

### 三面准备清单
- [ ] STAR 故事（见 `mock_interview/behavior/star-stories.md`）
- [ ] "你对通义千问 / Qwen 系列了解多少？"
- [ ] "推理服务未来 3 年的技术趋势？"
- [ ] "如何平衡开源和商业化？"

---

## 五、推荐阅读

| 资料 | 重点关注 |
|------|---------|
| 阿里技术博客：KV Cache 复用与投机采样 | 多轮对话优化实践 |
| RTP-LLM GitHub | 源码架构理解 |
| Qwen2.5 技术报告 | 模型架构细节 |
| Speculative Decoding 论文（Leviathan et al.） | 数学证明 |
| Radix Attention（SGLang 论文） | Prefix Caching 的前缀树方案 |
| PAI-EAS 官方文档 | 推理服务平台架构 |

---

## 六、心算练习

```
快速回答（30 秒内）：
1. Qwen2.5-72B GQA (kv_heads=8) 的 KV Cache per token？
   → 2 × 80 layers × 8 heads × 128 dim × 2 bytes = 327,680 bytes ≈ 320 KB/token
2. 128K context 的 KV Cache 总量？
   → 320 KB × 128K = 40 GB
3. 投机解码 draft_len=5, 接受率 70% 时的平均加速比？
   → 期望接受 token 数 = 1/(1-0.7) ≈ 3.33 → 加速比 ≈ 3.33/1 ≈ 3.3x
   （更精确：E[accepted] = Σ(k=0..5) k × C(5,k) × 0.7^k × 0.3^(5-k)... 简化计算约 3x）
4. Qwen-72B INT8 权重大小？ → 72B × 1 byte = 72 GB
5. A10 GPU（24GB）能放下 Qwen-72B INT4 吗？
   → INT4: 72B × 0.5 byte = 36 GB > 24 GB → 不行，至少需要 2 卡
```

---

## 七、阿里面试特别提示

### 1. 阿里喜欢考"业务感"
- 不仅问技术方案，还会问：这个优化对用户体验的影响是什么？
- 例：KV Cache 复用 → FTT 降低 → 用户感知的"第一句话响应快了"

### 2. 阿里重视"数据驱动"
- 面试时最好能给出具体数字
- 例："这个优化在 A100 上把 FTT 从 800ms 降到 200ms"

### 3. 阿里面试链路长
- 通常 4-5 面，可能有交叉面
- 每一面都可能问项目细节，准备好至少 3 个可深入讨论的项目

### 4. 级别对标
| 阿里级别 | 对应经验 | 考察侧重 |
|----------|---------|----------|
| P6 | 3-5 年 | 扎实基础 + 能独立做模块级优化 |
| P7 | 5-8 年 | 系统设计 + 端到端项目 Owner |
| P8 | 8+ 年 | 架构决策 + 技术 Vision + 团队影响力 |
