# CUDA 基础（面试够用版）

## 执行模型
```
Grid → Block → Thread
    └── 每个 block 内 thread 可以共享 shared memory
    └── 同一 warp（32 threads）SIMT 执行
```

## 核心概念
| 概念 | 含义 | 面试关键 |
|------|------|---------|
| Warp | 32 个 thread 一组执行 | GPU 调度最小单位 |
| SM | Streaming Multiprocessor | 物理执行单元 |
| Occupancy | SM 上活跃 warp / 最大 warp | 越高越能隐藏延迟 |
| Coalesced Access | 相邻 thread 访问相邻地址 | 对齐访问带宽最大化 |

## Kernel Launch
```cpp
kernel<<<grid_dim, block_dim, shared_mem_size, stream>>>(args...);
```
- grid_dim: 多少个 block
- block_dim: 每个 block 多少个 thread（通常 128/256）
- 异步执行，需要 `cudaStreamSynchronize` 等待

## 常用优化手段
1. **Coalesced memory access**：确保连续 thread 访问连续地址
2. **Shared memory**：用作手动 L1 缓存
3. **Memory padding**：避免 bank conflict
4. **Loop unrolling**：减少循环开销
5. **Warp-level primitives**：`__shfl_sync` 等做 warp 内规约

## 面试一句话
- "GPU 编程核心是让尽可能多的 warp 保持忙碌（occupancy），同时保证内存访问合并（coalesced access）。"
