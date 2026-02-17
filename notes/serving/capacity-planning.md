# 推理容量规划

## 核心公式

### 1. 单请求显存
```
KV_mem = 2 × n_layers × n_kv_heads × head_dim × dtype_bytes × seq_len
Model_mem = N_params × dtype_bytes
```

### 2. 单 GPU 最大并发
```
max_concurrent = (GPU_mem - Model_mem - OS_overhead) / KV_mem_per_request
```
- 例：H100 80GB, Llama3-70B INT8 (70GB 权重)
- 可用 KV 空间 ≈ 80 - 70 - 2 = 8 GB
- 每请求 4K context ≈ 320KB/token × 4096 ≈ 1.3 GB
- max_concurrent ≈ 6（非常少！→ 需要 TP/量化）

### 3. 吞吐估算
```
throughput = batch_size / TPOT = batch_size × output_tokens_per_second
```

### 4. GPU 数量
```
N_gpu = target_QPS × avg_latency / batch_size
      = target_QPS / per_gpu_throughput
```

## 示例：1000 QPS 的 Llama3-70B 服务
| 步骤 | 计算 |
|------|------|
| 模型部署 | TP=8 (1 node), 每 node 服务 ~128 并发 |
| 吞吐 | ~50 tok/s/request → 128×50/avg_output_len |
| 假设 avg output 200 tok | 128×50/200 = 32 QPS/node |
| 需要节点数 | 1000/32 ≈ 32 nodes = 256 GPU |
| 加 30% buffer | ~330 GPU |

## 容量规划清单
- [ ] 确定模型 + 精度 + TP 度
- [ ] 测量实际 TTFT 和 TPOT（不同 batch size）
- [ ] 确定 SLO（P99 TTFT, P99 TPOT）
- [ ] 估算峰值 QPS → 计算 GPU 数
- [ ] 考虑前缀缓存命中率对 TTFT 的改善
- [ ] 预留 30%+ buffer

## 面试一句话
- "容量规划的核心是算清三个数：每 GPU 能放多少并发（显存瓶颈）、每 GPU 吞吐多少（计算瓶颈）、需要满足多少 QPS（业务需求）。"
