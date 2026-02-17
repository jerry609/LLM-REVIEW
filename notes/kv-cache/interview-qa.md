# KV Cache 面试问答

## Q1: 为什么需要 KV Cache？没有它会怎样？
**A**: 自回归生成每步都要对所有历史 token 做注意力。不缓存 K/V 的话，第 t 步需要重算前 t-1 步的投影，生成 T 个 token 总计 O(T²) 的冗余。KV Cache 把历史 K/V 存下来，每步只计算新 token 的 Q·K^T，把冗余降到 O(T)。

## Q2: KV Cache 的显存占用怎么算？
**A**: `bytes_per_token = 2 × n_layers × n_kv_heads × head_dim × dtype_bytes`。例如 Llama3-70B (GQA, bf16)：2×80×8×128×2 ≈ 320KB/token。128K context 约 40GB。

## Q3: GQA 相比 MHA 对 KV Cache 有什么影响？
**A**: GQA 把 KV head 数从 n_heads 降到 n_kv_heads（如 32→8），KV Cache 大小线性缩小到 n_kv_heads/n_heads。质量损失通常很小（<1% PPL），是现在主流模型的标配。

## Q4: PagedAttention 解决了什么问题？
**A**: 传统连续分配有两个问题：①外部碎片（不同长度请求释放后留下空洞）；②需要预分配最大长度。PagedAttention 用固定大小块做分页，按需分配，碎片率从 60-80% 降到接近 0。

## Q5: 前缀缓存什么时候有效？什么时候没用？
**A**: 有效：多请求共享相同 system prompt / few-shot 前缀。无效：每个请求前缀都不同（如纯聊天）。关键指标是前缀命中率，低于 20% 时缓存维护开销可能不值得。

## Q6: KV Cache 占用过大怎么办？（至少说 3 种方案）
**A**:
1. **GQA/MQA**：减少 KV head 数（架构级）
2. **量化**：bf16→fp8/int8→int4，每级省一半显存
3. **分页驱逐**：LRU/LFU/注意力感知驱逐冷数据
4. **前缀缓存**：共享重复前缀，避免重复存储
5. **CPU offload**：冷块迁移到 CPU 内存
6. **稀疏化**：只保留注意力权重高的 token（如 H2O/SnapKV）

## Q7: 前缀缓存命中时，TTFT 和 TPOT 哪个受益？
**A**: **TTFT 受益**。命中后跳过 prefill 阶段（或只 prefill 未命中部分），首 token 更快返回。TPOT 不受影响，因为 decode 阶段仍然要读完整 KV。

## Q8: 如果缓存命中率很高但 TPOT 变差，你先查哪里？
**A**: 高命中率说明前缀复用好，但 TPOT 变差可能是：①并发请求增多导致 batch 更大，decode 读 KV 的总带宽增大；②驱逐/回迁抖动导致 GPU stall；③量化反量化的额外开销。先看 GPU 利用率和内存带宽是否饱和。
