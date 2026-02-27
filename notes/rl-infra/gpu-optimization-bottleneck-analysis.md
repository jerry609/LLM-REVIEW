# GPU 优化与瓶颈定位实战

> JD 关键词：**系统优化意识，能定位瓶颈并给出工程化方案**

---

## 一、GPU 性能模型

### 1.1 Roofline 模型

任何 GPU 计算都可以用两个维度衡量：

$$
\text{Performance} = \min\left(\text{Peak Compute},\; \text{Bandwidth} \times \text{Arithmetic Intensity}\right)
$$

| 指标 | 定义 | A100-80G SXM | H100 SXM |
|------|------|-------------|----------|
| Peak FP16 TFLOPS | 理论峰值算力 | 312 | 990 |
| Peak FP8 TFLOPS | FP8 峰值 | — | 1979 |
| HBM 带宽 | 显存带宽 | 2.0 TB/s | 3.35 TB/s |
| SRAM | 每 SM 共享内存 | 192 KB | 228 KB |
| NVLink 带宽 | 卡间通信 | 600 GB/s | 900 GB/s |

### 1.2 Compute Bound vs Memory Bound

```
                     Roofline
                        ╱
            Compute ───╱─── Bound ────────────
                     ╱ │
                   ╱   │
                 ╱     │
               ╱       │ Memory Bound
             ╱         │
           ╱           │
         ╱─────────────┤
        0       Arithmetic Intensity (FLOPs/Byte)
                        ↑
                    临界点 = Peak FLOPS / BW
                    A100: 312T / 2T = 156 FLOPs/Byte
                    H100: 990T / 3.35T = 295 FLOPs/Byte
```

### 1.3 LLM 各阶段的 Bound 分析

| 阶段 | Arithmetic Intensity | Bound 类型 | 原因 |
|------|---------------------|-----------|------|
| **Prefill** (大 batch) | 高 (~200) | Compute Bound | 大矩阵乘法 |
| **Decode** (小 batch) | 低 (~1-10) | **Memory Bound** | 每步只处理 1 token |
| **RL Rollout** | 极低 | **Memory Bound** | 自回归逐 token 生成 |
| **LoRA Forward** | 中 | 混合 | Base GEMM + LoRA 小矩阵 |
| **LoRA Backward** | 高 | Compute Bound | 梯度计算是大矩阵乘法 |

---

## 二、Profiling 工具链

### 2.1 工具选择指南

| 工具 | 粒度 | 适合场景 | 学习成本 |
|------|------|---------|---------|
| `nvidia-smi` | GPU 级 | 快速看利用率/显存 | 低 |
| `nvidia-smi dmon` | GPU 级 (连续) | 持续监控 | 低 |
| `torch.profiler` | 算子级 | PyTorch 模型分析 | 中 |
| `nsys` (Nsight Systems) | 时间线级 | 端到端系统分析 | 中 |
| `ncu` (Nsight Compute) | Kernel 级 | 单个 CUDA kernel 优化 | 高 |
| `py-spy` | Python 级 | Python 代码热点 | 低 |

### 2.2 实用命令速查

```bash
# === 快速诊断 ===

# GPU 利用率 + 显存 (1 秒刷新)
nvidia-smi -l 1

# 详细监控 (利用率、显存带宽、温度、功耗)
nvidia-smi dmon -s umt -d 1

# 看进程级 GPU 占用
nvidia-smi pmon -s u -d 1

# === PyTorch Profiler ===
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(dataloader):
        train_step(batch)
        prof.step()

# === Nsight Systems ===
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=rl_profile \
    --force-overwrite=true \
    python train_rl.py

# === 显存快照 ===
torch.cuda.memory._record_memory_history()
# ... run code ...
torch.cuda.memory._dump_snapshot("mem_snapshot.pickle")
# 可视化: https://pytorch.org/memory_viz
```

### 2.3 关键指标解读

| 指标 | 正常范围 | 异常诊断 |
|------|---------|---------|
| SM 利用率 (GPU Util) | >80% | <50%: batch 太小 / 数据加载慢 / CPU 瓶颈 |
| 显存占用 | <90% | >95%: 有 OOM 风险 |
| HBM 带宽利用率 | 取决于 bound | Memory Bound 下应接近峰值 |
| PCIe 带宽 | <20 GB/s | 过高说明频繁 CPU-GPU 搬运 |
| NVLink 利用率 | TP 场景 >50% | 过低说明通信被遮掩不好 |

---

## 三、常见瓶颈与优化方案

### 3.1 瓶颈 1：Rollout 吞吐低

**症状**：RL 训练中 >60% 时间花在生成 rollout 上。

**诊断**：

```python
import time
t0 = time.time()
responses = rollout_engine.generate(prompts)
t1 = time.time()
tokens_per_sec = total_tokens / (t1 - t0)
print(f"Rollout throughput: {tokens_per_sec:.0f} tokens/s")
# A100 上 7B 模型应该 >2000 tokens/s (FP16)
```

**优化方案**：

| 等级 | 方案 | 预期提升 |
|------|------|---------|
| L1 | 用 vLLM 替代 HF generate | 5-10× |
| L2 | 开 Continuous Batching | 2-3× |
| L3 | GRPO 开 Prefix Caching | 1.5-2× (同 prompt) |
| L4 | FP8 推理 | 1.5-2× |
| L5 | Speculative Decoding | 1.5-3× |
| L6 | Rollout-Train Pipeline | 隐藏等待时间 |

### 3.2 瓶颈 2：GPU 利用率低

**症状**：`nvidia-smi` 显示 GPU Util <50%，但训练在跑。

**诊断树**：

```
GPU Util 低
├── Data Loading 慢？
│   └── 诊断: num_workers=0? prefetch 关了?
│   └── 方案: num_workers=4+, pin_memory=True, persistent_workers=True
├── CPU 计算瓶颈？
│   └── 诊断: py-spy 看 Python 热点
│   └── 方案: 数据预处理放 GPU / 用 C++ DataLoader
├── 小 batch？
│   └── 诊断: batch_size 除以 GPU 数
│   └── 方案: 增大 batch / 梯度累积
├── 等待通信？
│   └── 诊断: nsys 看 NCCL 占比
│   └── 方案: 通信计算重叠 / 减少通信频率
└── LoRA 切换开销？
    └── 诊断: 统计切换次数
    └── 方案: LoRA-aware scheduling / 预取
```

### 3.3 瓶颈 3：OOM (Out of Memory)

**诊断**：

```python
# 显存组成分析
print(torch.cuda.memory_summary(device=0, abbreviated=True))

# 典型显存占用分解 (7B 模型 LoRA 训练)
# Model weights (FP16):     14 GB
# LoRA weights (FP16):       0.03 GB
# Optimizer states (FP32):   0.12 GB (LoRA only)
# Gradients (FP16):          0.06 GB (LoRA only)
# Activations:               2-10 GB (取决于 batch × seq_len)
# KV Cache (推理):           1-4 GB
# CUDA 碎片:                 1-3 GB
# 合计:                      ~20-30 GB
```

**优化方案**：

| 技术 | 节省量 | 适用 |
|------|--------|------|
| 梯度检查点 (Gradient Checkpointing) | 激活减少 ~5× | 所有场景 |
| Base Model FP8 量化 | 权重减半 ~7 GB | H100 |
| DeepSpeed ZeRO-Offload | 优化器到 CPU | 显存极紧时 |
| Flash Attention | 注意力显存从 $O(L^2)$ 到 $O(L)$ | 长序列 |
| LoRA 分批激活 | 同时只训练 K 个 LoRA | 千级 LoRA |

### 3.4 瓶颈 4：多机通信

**诊断**：

```bash
# 查看 NCCL 通信拓扑
NCCL_DEBUG=INFO python -c "import torch.distributed; torch.distributed.init_process_group('nccl')"

# AllReduce 带宽测试
# 期望: NVLink ~500 GB/s, IB ~100 GB/s, TCP ~10 GB/s
python -c "
import torch, torch.distributed as dist
dist.init_process_group('nccl')
t = torch.randn(1024, 1024, device='cuda')
import time
for _ in range(10):  # warmup
    dist.all_reduce(t)
torch.cuda.synchronize()
s = time.time()
for _ in range(100):
    dist.all_reduce(t)
torch.cuda.synchronize()
e = time.time()
bw = t.numel() * t.element_size() * 2 * 100 / (e - s) / 1e9
print(f'AllReduce BW: {bw:.1f} GB/s')
"
```

**优化方案**：

| 方案 | 场景 |
|------|------|
| TP 限制在 NVLink 域内 | 机内 |
| PP 做跨机（通信量小） | 跨机 |
| 梯度压缩 (FP16→FP8) | 带宽受限 |
| 通信-计算重叠 | 所有场景 |
| 减少 AllReduce 频率（梯度累积） | DP 场景 |

---

## 四、RL 系统特有的优化

### 4.1 Rollout 长度动态调整

```python
class AdaptiveRolloutManager:
    """根据 GPU 显存和 reward 信号动态调整 rollout 长度"""

    def __init__(self, max_tokens=2048, min_tokens=128):
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.current_max = max_tokens

    def adjust(self, gpu_mem_used_pct: float, mean_response_len: int):
        """
        - 显存 >85%: 缩短 rollout 避免 OOM
        - 大部分 response 很短: 减少 max_tokens 节省显存
        """
        if gpu_mem_used_pct > 0.85:
            self.current_max = max(self.min_tokens, int(self.current_max * 0.8))
        elif gpu_mem_used_pct < 0.6:
            self.current_max = min(self.max_tokens, int(self.current_max * 1.2))

        # 根据实际生成长度调整
        self.current_max = max(self.min_tokens, min(self.max_tokens,
                                                     mean_response_len * 2))
        return self.current_max
```

### 4.2 训练稳定性

| 问题 | 症状 | 方案 |
|------|------|------|
| Loss spike | 突然飙升 | 梯度裁剪 + KL penalty 上界 |
| Reward hacking | reward 高但质量差 | 多维度 reward + 人工评审 |
| Mode collapse | 生成单一模式 | 提高 temperature + entropy bonus |
| Reference 漂移 | ref 模型过时 | 定期更新 ref 或用 EMA |
| NaN 梯度 | loss 变 NaN | 混合精度的 loss scaling + grad clip |

### 4.3 工程化稳定性保障

```python
class RLTrainingGuard:
    """RL 训练的工程化防护机制"""

    def __init__(self, max_grad_norm=1.0, max_kl=0.2, loss_spike_threshold=3.0):
        self.max_grad_norm = max_grad_norm
        self.max_kl = max_kl
        self.loss_spike_threshold = loss_spike_threshold
        self.loss_ema = None
        self.alpha = 0.99

    def check_and_clip(self, model, loss, kl_div):
        """训练一步前的安全检查"""
        # 1. 梯度裁剪
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), self.max_grad_norm
        )

        # 2. Loss spike 检测
        if self.loss_ema is None:
            self.loss_ema = loss.item()
        else:
            if loss.item() > self.loss_ema * self.loss_spike_threshold:
                print(f"⚠️  Loss spike: {loss.item():.4f} >> EMA {self.loss_ema:.4f}")
                return False  # 跳过这步更新
            self.loss_ema = self.alpha * self.loss_ema + (1 - self.alpha) * loss.item()

        # 3. KL 散度过大检测
        if kl_div > self.max_kl:
            print(f"⚠️  KL too large: {kl_div:.4f} > {self.max_kl}")
            return False

        # 4. NaN 检测
        if torch.isnan(loss) or torch.isinf(loss):
            print("⚠️  NaN/Inf loss detected")
            return False

        return True
```

---

## 五、性能优化 Checklist

### 推理 (Rollout) 优化

- [ ] 使用 vLLM / SGLang 作为推理引擎
- [ ] 开启 Continuous Batching
- [ ] 开启 Prefix Caching（GRPO 场景必开）
- [ ] 使用 FP8 量化推理（H100）
- [ ] Flash Attention 开启
- [ ] 调整 `max_num_seqs` 匹配显存
- [ ] Rollout 和 Training 流水线化

### 训练优化

- [ ] 梯度检查点 (activation checkpointing)
- [ ] 混合精度 (FP16/BF16)
- [ ] 梯度裁剪 (max_norm=1.0)
- [ ] AdamW 优化器 + Cosine LR
- [ ] LoRA target 包含所有线性层
- [ ] 梯度累积到合理 effective batch size

### 系统优化

- [ ] 数据加载: `num_workers>=4`, `pin_memory=True`
- [ ] TP 限制在 NVLink 域内
- [ ] 通信计算重叠
- [ ] LoRA 权重预取 (prefetching)
- [ ] 动态 Rollout 长度调整
- [ ] 训练稳定性监控 (loss/KL/grad_norm)

---

## 六、面试一句话总结

- **瓶颈定位**："先 nvidia-smi 看利用率，再 nsys 看时间线，最后 ncu 看 kernel — 三步定位到底是 compute bound 还是 memory bound。"
- **RL 系统优化**："RL 训练 60% 时间在 rollout，用 vLLM + prefix caching + 流水线化可以把这部分时间减半以上。"
- **千级 LoRA**："共享 Base Model 冻结权重，LoRA 增量用 batched GEMM kernel 一次 forward 处理，配合 LRU 权重池和 reward-aware 调度。"
