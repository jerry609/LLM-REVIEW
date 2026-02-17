# LLM 推理服务生产故障排查手册

## 场景一：Latency Spike（延迟尖刺）

### 症状
- P99 latency 突然从 200ms 升到 2000ms
- QPS 没有明显变化

### 排查步骤
1. 检查 GPU 利用率（nvidia-smi -l 1）
   - 如果 GPU util 100% -> 计算瓶颈
   - 如果 GPU util < 50% -> I/O 或调度瓶颈
2. 检查显存使用
   - 接近上限 -> KV Cache 可能触发 swap/preemption
   - vLLM 日志搜索 "preempt" 或 "swap"
3. 检查请求分布
   - 是否有长序列请求（>4K tokens）？
   - 长请求的 prefill 会 block 整个 batch
4. 检查 batch size
   - running batch size 是否异常大？

### 常见根因
| 根因 | 表现 | 解决 |
|------|------|------|
| 长序列请求 | prefill 阶段占用 GPU 时间长 | chunked prefill |
| KV Cache 满 | 频繁 preemption/swap | 增加 GPU 或限制并发 |
| GC/Python | 周期性卡顿 | 检查 GC 频率 |
| NCCL timeout | 分布式推理超时 | 检查网络/RDMA |

---

## 场景二：OOM (Out of Memory)

### 症状
- CUDA out of memory 错误
- 服务崩溃重启

### 排查步骤
1. 确认 OOM 来源（KV Cache 还是 model weights？）
2. 分析显存分布：
   - Model weights: ~14GB (7B fp16)
   - KV Cache: 剩余显存 * gpu_memory_utilization
   - Activation: 运行时临时分配
   - CUDA context: ~1GB
3. 计算 KV Cache 容量：
   - 单 token KV = 2 * n_layers * n_kv_heads * d_head * 2 bytes
   - 7B model: 2 * 32 * 8 * 128 * 2 = 128KB/token
   - 最大 tokens = (GPU_mem - model - overhead) / 128KB

### 常见根因
| 根因 | 解决方案 |
|------|---------|
| gpu_memory_utilization 过高 | 降到 0.85-0.9 |
| 并发请求过多 | 限制 max_num_seqs |
| 长序列占用 | 设置 max_model_len |
| 内存碎片 | vLLM PagedAttention 自动处理 |

---

## 场景三：Throughput 下降

### 症状
- 吞吐量从 1000 tokens/s 降到 300 tokens/s
- 没有硬件变化

### 排查步骤
1. 检查 batch utilization（是否 batch 填不满）
2. 检查 prefill/decode 比例
3. 检查 scheduling 策略
4. 检查 KV Cache 命中率（如有 prefix caching）

### 优化建议
- 开启 prefix caching（相同 system prompt 共享）
- 调整 max_num_batched_tokens
- 使用 speculative decoding 提升 decode 速度

---

## 场景四：输出质量下降

### 症状
- 模型输出变得重复、不相关

### 排查步骤
1. 检查量化是否有问题（INT4 在某些任务上质量下降）
2. 检查 KV Cache 精度（FP8 长序列可能累积误差）
3. 检查 temperature / sampling 参数
4. 检查 tokenizer 版本

---

## 场景五：分布式推理故障

### 症状
- 多 GPU/多机推理时部分 worker 卡死

### 排查步骤
1. 检查 NCCL 连接（NCCL_DEBUG=INFO）
2. 检查 GPU 拓扑（nvidia-smi topo -m）
3. 检查各 worker 显存
4. 检查超时设置（NCCL_TIMEOUT_MS）

---

## 通用排查工具

- GPU 状态：nvidia-smi -l 1
- PyTorch profiler
- vLLM 日志：VLLM_LOGGING_LEVEL=DEBUG
- CUDA profiling：nsys profile
- 网络诊断：NCCL_DEBUG=INFO, ib_write_bw
