# vLLM 架构

## 核心组件
```
Engine (调度/资源管理)
├── Scheduler (调度器：continuous batching + 优先级)
├── BlockManager (块管理器：PagedAttention 物理块分配)
├── CacheEngine (KV Cache：GPU/CPU 块池)
└── Worker (执行器：model forward + sampling)
```

## 关键设计
1. **PagedAttention**：虚拟内存分页 → 碎片接近 0
2. **Continuous Batching**：Scheduler 每 step 动态组 batch
3. **Prefix Caching**：Radix Tree 前缀匹配 → 自动复用
4. **Chunked Prefill**：长 prefill 分片，不阻塞 decode

## 调度流程（每 step）
1. `schedule()` → 决定本 step 运行哪些请求（prefill / decode）
2. `allocate_blocks()` → 为新请求分配物理块
3. `execute_model()` → 调用 model forward
4. `sample()` → 采样 next token
5. `free_finished()` → 释放已完成请求的块

## 并行支持
- **TP (Tensor Parallel)**：自动切分 attention 和 FFN
- **PP (Pipeline Parallel)**：多 GPU 流水线
- **EP**：MoE 专家并行（实验性）

## 启动参数
| 参数 | 含义 |
|------|------|
| `--tensor-parallel-size` | TP 度 |
| `--max-model-len` | 最大上下文长度 |
| `--kv-cache-dtype fp8` | KV 量化 |
| `--enable-prefix-caching` | 开启前缀缓存 |
| `--max-num-seqs` | 最大并发请求数 |

## 面试一句话
- "vLLM = PagedAttention + Continuous Batching + Prefix Caching，是最主流的高吞吐推理引擎。"
