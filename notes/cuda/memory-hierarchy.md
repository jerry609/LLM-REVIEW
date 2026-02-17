# GPU 内存层次

## 层次结构
```
Register (最快, ~TB/s, KB级/thread)
  ↓
Shared Memory / L1 Cache (SRAM, ~19 TB/s, ~128-228 KB/SM)
  ↓
L2 Cache (~8-12 TB/s, 数十 MB)
  ↓
HBM (Global Memory) (~2-3.35 TB/s, 40-80 GB)
  ↓
CPU Memory (PCIe ~64 GB/s)
```

## H100 规格
| 层级 | 容量 | 带宽 |
|------|------|------|
| Register | 256KB/SM | ~TB/s |
| L1/Shared | 228KB/SM | ~33 TB/s 聚合 |
| L2 | 50 MB | ~12 TB/s |
| HBM3 | 80 GB | 3.35 TB/s |

## Roofline 模型
```
performance = min(peak_compute, peak_bandwidth × arithmetic_intensity)
arithmetic_intensity = FLOPs / Bytes
```
- 拐点 = peak_compute / peak_bandwidth
- H100: ~990 TFLOPS(FP8) / 3.35 TB/s ≈ **295 FLOP/Byte**
- Attention decode: ~1-2 FLOP/Byte → 远低于拐点 → **memory-bound**
- Prefill matmul: ~100+ FLOP/Byte → 接近拐点 → **compute-bound**

## FlashAttention 的内存视角
- 标准 attention 需要把 T×T 矩阵写到 HBM → IO-bound
- FlashAttention 把计算保持在 SRAM → 减少 HBM 读写
- 等效于把 attention 从 memory-bound 变成 compute-bound

## 面试一句话
- "GPU 优化的核心是把数据尽量保持在高层级内存（register/shared memory），减少 HBM 访问。FlashAttention 是这一思想的经典应用。"
