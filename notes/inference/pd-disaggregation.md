# Prefill-Decode 分离 (P/D Disaggregation)

## 动机
- Prefill: compute-bound，大矩阵乘，算术强度高
- Decode: memory-bound，读 KV Cache，算术强度低
- 混合在同一 GPU → 互相干扰（prefill 占用算力 → decode 延迟飙升）

## 方案
- **Prefill GPU**：专做 prefill，追求高 MFU 和大 batch
- **Decode GPU**：专做 decode，追求低延迟和高带宽利用
- **KV 传输**：prefill 完成后将 KV Cache 通过网络传给 decode GPU

## KV 传输带宽需求
```
transfer_bytes = bytes_per_token × T_input
例：7B 模型 4K token ≈ 128KB/token × 4096 = 512 MB
若需 100ms 内完成 → 需要 5.12 GB/s 带宽
```
- InfiniBand / RoCE 通常满足，普通以太网可能瓶颈

## 相关工作
| 系统 | 核心方案 |
|------|---------|
| Splitwise (UW) | P/D 分离 + 负载感知调度 |
| DistServe (PKU) | P/D 分离 + 按 SLO 优化资源分配 |
| TetriInfer | 动态 P/D 比例调整 |

## 权衡
- **优点**：各阶段独立优化，TTFT 和 TPOT 都能改善
- **缺点**：① 系统复杂度高；② KV 传输延迟和带宽；③ 需要更多 GPU（专用化）

## 面试一句话
- "P/D 分离让 prefill 和 decode 各自在最适合的硬件配置上运行，核心挑战是 KV 传输带宽和调度复杂度。"
