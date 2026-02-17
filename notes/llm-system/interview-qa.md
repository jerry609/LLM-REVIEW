# LLM 推理系统面试问答

## Q1: Continuous Batching 和 Static Batching 的核心区别？
**A**: Static batching 等所有请求完成才处理下一批，短请求被长请求卡住。Continuous batching 在每个 decode step 检查是否有请求完成，完成即释放 slot 并填入新请求，吞吐可提升 2-10×。

## Q2: TTFT 和 TPOT 分别由什么决定？
**A**:
- **TTFT** = 排队时间 + prefill 时间。受 prefill 计算量（O(T_input²)）和排队深度影响。
- **TPOT** = 单步 decode 时间。受 KV 读取带宽（memory-bound）和 batch size 影响。

## Q3: 为什么 decode 是 memory-bound？
**A**: decode 每步只算 1 个新 token 的 Q·K^T，但要读取整个 KV Cache。计算量 O(T_cache × d_head)，数据读取也是 O(T_cache × d_head × bytes)。算术强度很低（~1 FLOP/byte），远低于 GPU 的 roofline 拐点。

## Q4: Chunked Prefill 解决什么问题？
**A**: 长 prefill（如 32K token）会占用 GPU 数百毫秒，期间所有 decode 请求停滞 → TPOT 尾延迟飙升。Chunked Prefill 把 prefill 切成小块（如 512 token），每块之间插入 decode step，把 TPOT P99 控制在可接受范围。

## Q5: P/D 分离的优缺点？
**A**:
- **优点**：prefill 和 decode 各自优化（不同 batch size、不同并行策略）
- **缺点**：① KV 迁移带宽高（7B 模型 4K token ≈ 512MB）；② 系统复杂度增加；③ 需要快速网络（InfiniBand/RoCE）

## Q6: 怎么做容量规划？
**A**: 步骤：
1. 估算单请求 KV 显存：`bytes_per_token × avg_context_length`
2. 估算每 GPU 可服务并发数：`(GPU显存 - 模型权重) / 单请求KV`
3. 估算单 GPU 吞吐：`并发数 × (1/avg_TPOT)`
4. 需要 GPU 数 = `target_QPS / 单GPU吞吐`
5. 加 30% buffer 应对峰值

## Q7: 如何降低推理成本 50%？
**A**: 组合拳：① KV 量化 fp8（省显存 → 更大 batch → 吞吐+30%）；② 前缀缓存（节省重复 prefill 计算）；③ 投机解码（2-3× 加速）；④ 合适的并行策略减少通信开销；⑤ 模型蒸馏（如 70B→8B + quality guard）。
