# 面试心算速记

## 1) 快速估算套路
- 先保留 1-2 位有效数字，再做数量级换算。
- 结果先报"约等于"，再给误差范围（如 ±10%）。
- 乘法拆分：`70B * 2 = 140 GB`（参数量 × 字节数 = 权重显存）

## 2) 2 的幂次速记表
| 2^n | 值 | 近似 |
|-----|------|------|
| 2^10 | 1,024 | ≈ 1K |
| 2^20 | 1,048,576 | ≈ 1M |
| 2^30 | 1,073,741,824 | ≈ 1G |
| 2^40 | | ≈ 1T |
| 2^16 | 65,536 | ≈ 64K（常见词表大小）|
| 2^17 | 131,072 | ≈ 128K（长上下文长度）|
| 2^12 | 4,096 | 常见 d_model |
| 2^7 | 128 | 常见 d_head |

## 3) 常见字节速记
- BF16/FP16：2B per element
- FP8/INT8：1B per element
- INT4：0.5B per element
- FP32：4B per element

## 4) 模型权重显存速算
| 模型 | 参数量 | BF16 权重 | INT8 权重 | INT4 权重 |
|------|--------|----------|----------|----------|
| 7B   | 7×10^9 | ~14 GB   | ~7 GB    | ~3.5 GB  |
| 13B  | 13×10^9| ~26 GB   | ~13 GB   | ~6.5 GB  |
| 34B  | 34×10^9| ~68 GB   | ~34 GB   | ~17 GB   |
| 70B  | 70×10^9| ~140 GB  | ~70 GB   | ~35 GB   |

公式：`weight_mem = N * s`

## 5) KV 一步心算
- 先算：`2 * L * H_kv * d_head`（每 token 元素数）
- 再乘字节数 `s`
- 最后乘 token 总数
- 快速代入：
  - 7B GQA：`2*32*8*128*2 = 128 KB/token`
  - 70B GQA：`2*80*8*128*2 = 320 KB/token`
  - 7B MHA：`2*32*32*128*2 = 512 KB/token`

## 6) GPU 参数速记
| GPU | HBM | BW | BF16 TFLOPS | 拐点 AI |
|-----|-----|------|-------------|---------|
| A100 80GB | 80 GB | 2 TB/s | 312 | ~156 |
| H100 80GB | 80 GB | 3.35 TB/s | 990 | ~295 |
| H200 141GB | 141 GB | 4.8 TB/s | 990 | ~206 |

## 7) 延迟预算心算
- `E2E ≈ TTFT + N_out * TPOT`
- 若 `TPOT` 增加 2ms，`N_out=300` 时总时长约多 0.6s。
- prefill 速度估算：`T_prefill ≈ 2*N*T_input / (TFLOPS * MFU)`

## 8) 训练 FLOPs 心算
- `C ≈ 6 * N * D`
- 7B × 1T token ≈ 4.2e22 FLOPs
- 70B × 1.4T token ≈ 5.9e23 FLOPs
- GPU-days = `C / (peak_flops * MFU * 86400)`

## 9) batch size 与吞吐心算
- `max_batch ≈ (GPU_mem - weight_mem - overhead) / (bytes_per_token * avg_seq_len)`
- `throughput_tok ≈ batch_size / TPOT`
- 例：80 GB GPU，14 GB 权重，2 GB 开销 → 64 GB KV
  - 128 KB/token，avg 2K → `64 GB / (128 KB * 2K) = 64 GB / 256 MB ≈ 250` 并发

## 10) 常用口述
- "我先给线性估算，再补上碎片和调度开销作为修正项。"
- "数量级对了就行，面试更看重思路清晰，不追求小数点后精确。"
- "先算权重占多少，剩余给 KV，反推并发上限。"
