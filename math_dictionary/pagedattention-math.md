# PagedAttention 分页管理数学与工程详解

> **核心定位**：从操作系统虚拟内存的类比出发，系统性推导 PagedAttention 的分页策略、碎片分析、Prefix Caching 的引用计数与 Copy-on-Write 机制，并给出精确的显存节省计算公式。

---

## 1. 为什么需要 PagedAttention？

在没有 PagedAttention 的传统系统中，每个请求的 KV Cache 必须分配一块**连续的**显存。由于每个请求的输出长度在生成前无法确定，系统只能按**最大可能长度**预分配。

假设系统最大序列长度为 $L_{\max} = 8192$，但平均实际使用长度为 $\bar{L} = 2048$。

$$
\text{显存浪费率} = 1 - \frac{\bar{L}}{L_{\max}} = 1 - \frac{2048}{8192} = 75\%
$$

这种**外部碎片（External Fragmentation）**使得 GPU 显存的有效利用率极低，严重限制了服务的并发能力。

---

## 2. 分页机制核心设计

### 2.1 基本参数定义

PagedAttention 将 KV Cache 切分为固定大小的**物理块（Block / Page）**：

- $b$：每个块包含的 token 数（Block Size），典型值为 $16$ 或 $32$。
- $T$：当前序列的实际长度。
- 所需块数：$N_{\text{block}} = \lceil T / b \rceil$

每个块的物理大小（Bytes）：
$$
\text{Block}_{\text{bytes}} = 2 \times L \times H_{\text{KV}} \times d_{\text{head}} \times s \times b
$$

其中因子 $2$ 表示 K 和 V 各一份，$L$ 为模型层数，$s$ 为每元素字节数。

### 2.2 逻辑→物理映射（页表）

每个请求维护一个**页表 (Page Table)**，将逻辑 token 位置映射到物理块地址：

$$
\text{PhysicalBlock}(\text{token}_i) = \text{PageTable}\!\left[\left\lfloor \frac{i}{b} \right\rfloor\right]
$$
$$
\text{Offset}(\text{token}_i) = i \bmod b
$$

**逻辑连续，物理分散**：页表解耦了逻辑序列与物理存储的绑定关系，使得显存管理变得极其灵活。

---

## 3. 碎片分析

PagedAttention 将**外部碎片**完全消除，仅剩**内部碎片（Internal Fragmentation）**——即每个序列的最后一个块可能未被填满。

### 3.1 单序列碎片

对于长度为 $T$ 的序列，最后一个块的浪费 token 数为：
$$
w = (b - T \bmod b) \bmod b
$$
- **最坏情况**：$w = b - 1$（序列长度恰好比整块多 1 个 token）。
- **平均浪费**：$\mathbb{E}[w] = \frac{b-1}{2} \approx \frac{b}{2}$（假设 $T$ 均匀分布）。

### 3.2 碎片率

$$
\text{FragRate} = \frac{\mathbb{E}[w]}{\mathbb{E}[T] + \mathbb{E}[w]} = \frac{b/2}{\bar{T} + b/2}
$$

**代入数值**：$b = 16$，$\bar{T} = 2048$：
$$
\text{FragRate} = \frac{8}{2048 + 8} \approx 0.39\%
$$

对比传统连续分配的 75% 浪费率，PagedAttention 将碎片率降低了 **约 200 倍**。

---

## 4. 块分配器 (Block Allocator)

系统维护一个**空闲块链表 (Free List)**：

| 操作 | 时间复杂度 | 说明 |
|------|----------|------|
| **分配 (Allocate)** | $\mathcal{O}(1)$ | 从 Free List 头部取出一个空闲块 |
| **回收 (Free)** | $\mathcal{O}(1)$ | 将块归还到 Free List |
| **耗尽处理** | — | Free List 为空时：触发驱逐策略或拒绝新请求 |

可用 KV 容量的估算：
$$
N_{\text{total\_blocks}} = \left\lfloor \frac{M_{\text{GPU}} - M_{\text{weights}} - M_{\text{overhead}}}{\text{Block}_{\text{bytes}}} \right\rfloor
$$
$$
\text{Max Concurrent Tokens} = N_{\text{total\_blocks}} \times b
$$

---

## 5. Prefix Caching（前缀共享）

在实际服务中，大量请求共享相同的系统提示词（System Prompt）。Prefix Caching 允许这些请求**共享同一组物理块**。

### 5.1 引用计数 (Reference Counting)

每个物理块维护一个引用计数 $\text{ref\_count}$：
- 新请求匹配到已有前缀 → $\text{ref\_count} \mathrel{+}= 1$
- 请求完成 → $\text{ref\_count} \mathrel{-}= 1$
- 当 $\text{ref\_count} = 0$ 时，块被回收至 Free List

### 5.2 显存节省计算

假设系统提示词长度为 $T_{\text{prefix}}$，共 $R$ 个并发请求共享此前缀：

$$
\text{Savings} = (R - 1) \times T_{\text{prefix}} \times \text{bytes\_per\_token}
$$

**代入数值**：$T_{\text{prefix}} = 2048$，$R = 64$，$\text{bytes\_per\_token} = 128 \text{ KB}$：
$$
\text{Savings} = 63 \times 2048 \times 128 \text{ KB} \approx 15.75 \text{ GB}
$$

---

## 6. Copy-on-Write (CoW)

当共享前缀的请求开始各自生成不同的输出时，在**分叉点**所在的块需要执行 Copy-on-Write：

1. 检测到当前块的 $\text{ref\_count} > 1$。
2. 从 Free List 分配一个新块，复制原始块的数据。
3. 修改当前请求的页表指向新块。
4. 原始块 $\text{ref\_count} \mathrel{-}= 1$。

$$
\text{CoW 开销} = \text{Block}_{\text{bytes}} \quad (\text{仅复制 1 个块的数据量})
$$

分叉之前的所有公共块继续共享，**无需复制**。

---

## 7. 与连续分配的全面对比

| 特性 | 连续分配 (Naive) | PagedAttention |
|------|:---------------:|:--------------:|
| 外部碎片 | 严重（预分配浪费） | **完全消除** |
| 内部碎片 | 无 | $< 0.5\%$ |
| 动态扩展 | 需预分配或重分配 + 数据拷贝 | 按需追加新块 |
| 前缀共享 | 困难（需额外复制） | 原生支持（引用计数） |
| 分叉操作 | 全量拷贝 | Copy-on-Write（单块） |
| 并发能力 | 受限于最大预分配 | 灵活利用所有空闲块 |

**实测 (vLLM 论文)**：在相同硬件条件下，PagedAttention 相比连续分配方案，Batch 吞吐量提升 **2–4×**。

---

## 8. 面试实战追问

**Q1：PagedAttention 的块大小 $b$ 怎么选？**
> $b$ 越大 → 内部碎片越大，但块管理开销越小（页表更短）。
> $b$ 越小 → 碎片越低，但页表条目更多，且每块的 GPU kernel launch 开销增加。
> 经验值：$b = 16$ 是碎片率 ($< 0.5\%$) 与管理效率的最优平衡点。

**Q2：PagedAttention 对 FlashAttention 有影响吗？**
> 有。标准 FlashAttention 假设 $K, V$ 在内存中是连续的。PagedAttention 的 KV 物理上分散在不同块中。因此需要修改 FlashAttention 的 kernel，在加载 $K_j, V_j$ 块时通过页表做间接寻址。vLLM 和 SGLang 都实现了 Paged FlashAttention kernel。
