# LLM 推理性能分析工具

## NVIDIA 工具链

### Nsight Systems
- **用途**：端到端 timeline 分析（CPU/GPU 活动、kernel 调度、通信）
- **关键看**：GPU idle time、kernel 排队延迟、NCCL 通信占比
- 命令：`nsys profile python3 server.py`

### Nsight Compute
- **用途**：单个 kernel 级别的详细分析
- **关键看**：occupancy、memory throughput、arithmetic intensity
- 命令：`ncu --target-processes all python3 benchmark.py`

## PyTorch 工具

### torch.profiler
```python
with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    with_stack=True,
) as prof:
    model(input)
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### torch.cuda.memory_stats()
- 跟踪显存分配/释放/碎片
- `torch.cuda.max_memory_allocated()` 看峰值

## vLLM 内置指标
| 指标 | 含义 |
|------|------|
| `vllm:num_requests_running` | 运行中请求数 |
| `vllm:num_requests_waiting` | 排队请求数 |
| `vllm:gpu_cache_usage_perc` | KV Cache 使用率 |
| `vllm:avg_prompt_throughput_toks_per_s` | Prefill 吞吐 |
| `vllm:avg_generation_throughput_toks_per_s` | Decode 吞吐 |

## 性能问题诊断流程
```
延迟高？
├── TTFT 高 → 排队深？prefill 慢？
│   ├── 排队深 → admission control / scale out
│   └── prefill 慢 → chunked prefill / 前缀缓存
└── TPOT 高 → batch 太大？KV 读取带宽饱和？
    ├── batch 太大 → 限制 max_num_seqs
    └── 带宽饱和 → KV 量化 / 更大 TP 度
```

## 面试一句话
- "性能分析先分 TTFT/TPOT 定位瓶颈阶段，再用 Nsight 看 GPU 利用率和内存带宽，最后对症下药。"
