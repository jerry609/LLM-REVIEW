"""
KV Cache 量化器（简化版 INT8 per-channel 量化 + 反量化）。

面试要点：
- 权重量化 vs KV 量化 vs 激活量化是三件不同的事
- per-channel 比 per-tensor 精度更高（每个通道独立 scale/zero）
- 对称量化：zero_point = 0，scale = max(|x|) / 127
- 非对称量化：zero_point != 0，能更好表示不均匀分布
- KV 量化通常用对称 INT8（精度损失小、反量化快）
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class QuantizedTensor:
    """量化后的整数张量 + 元数据。"""
    data: np.ndarray       # int8
    scale: np.ndarray      # float32, per-channel
    zero_point: np.ndarray  # int8, per-channel (对称时为 0)
    original_shape: Tuple[int, ...]
    symmetric: bool


def quantize_per_channel_symmetric(
    tensor: np.ndarray,
    axis: int = -1,
    bits: int = 8,
) -> QuantizedTensor:
    """
    对称 per-channel INT8 量化。

    scale_c = max(|tensor_c|) / qmax
    quantized_c = round(tensor_c / scale_c)
    """
    qmax = (1 << (bits - 1)) - 1  # 127 for INT8

    # 将 axis 移到最后
    t = np.moveaxis(tensor, axis, -1).astype(np.float32)
    shape = t.shape
    flat = t.reshape(-1, shape[-1])  # [N, C]

    abs_max = np.max(np.abs(flat), axis=0)  # [C]
    abs_max = np.clip(abs_max, a_min=1e-8, a_max=None)
    scale = abs_max / qmax  # [C]

    quantized = np.round(flat / scale).astype(np.int8)
    quantized = np.clip(quantized, -qmax, qmax).astype(np.int8)

    # 还原形状
    quantized = quantized.reshape(shape)
    quantized = np.moveaxis(quantized, -1, axis)

    return QuantizedTensor(
        data=quantized,
        scale=scale.astype(np.float32),
        zero_point=np.zeros_like(scale, dtype=np.int8),
        original_shape=tensor.shape,
        symmetric=True,
    )


def quantize_per_channel_asymmetric(
    tensor: np.ndarray,
    axis: int = -1,
    bits: int = 8,
) -> QuantizedTensor:
    """
    非对称 per-channel INT8 量化。

    scale_c = (max_c - min_c) / (qmax - qmin)
    zero_point_c = round(qmin - min_c / scale_c)
    quantized_c = round(tensor_c / scale_c) + zero_point_c
    """
    qmin, qmax = -(1 << (bits - 1)), (1 << (bits - 1)) - 1  # -128, 127

    t = np.moveaxis(tensor, axis, -1).astype(np.float32)
    shape = t.shape
    flat = t.reshape(-1, shape[-1])

    c_min = np.min(flat, axis=0)
    c_max = np.max(flat, axis=0)
    c_range = np.clip(c_max - c_min, a_min=1e-8, a_max=None)
    scale = c_range / (qmax - qmin)

    zero_point = np.round(qmin - c_min / scale).astype(np.int8)

    quantized = np.round(flat / scale).astype(np.int32) + zero_point.astype(np.int32)
    quantized = np.clip(quantized, qmin, qmax).astype(np.int8)

    quantized = quantized.reshape(shape)
    quantized = np.moveaxis(quantized, -1, axis)

    return QuantizedTensor(
        data=quantized,
        scale=scale.astype(np.float32),
        zero_point=zero_point.astype(np.int8),
        original_shape=tensor.shape,
        symmetric=False,
    )


def dequantize(qt: QuantizedTensor, axis: int = -1) -> np.ndarray:
    """反量化：float = (int - zero_point) * scale。"""
    t = np.moveaxis(qt.data.astype(np.float32), axis, -1)
    shape = t.shape
    flat = t.reshape(-1, shape[-1])

    zp = qt.zero_point.astype(np.float32)
    result = (flat - zp) * qt.scale

    result = result.reshape(shape)
    return np.moveaxis(result, -1, axis)


def quantization_error(original: np.ndarray, qt: QuantizedTensor, axis: int = -1) -> dict:
    """计算量化误差指标。"""
    reconstructed = dequantize(qt, axis=axis)
    diff = original.astype(np.float32) - reconstructed
    return {
        "max_abs_error": float(np.max(np.abs(diff))),
        "mean_abs_error": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "compression_ratio": original.nbytes / (qt.data.nbytes + qt.scale.nbytes + qt.zero_point.nbytes),
    }


# ============ Demo ============
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # 模拟一个 KV tensor: [num_layers, seq_len, head_dim]
    kv = rng.normal(size=(4, 128, 64)).astype(np.float32)

    # 对称量化
    qt_sym = quantize_per_channel_symmetric(kv, axis=-1)
    err_sym = quantization_error(kv, qt_sym, axis=-1)
    print("Symmetric INT8:")
    for k, v in err_sym.items():
        print(f"  {k}: {v:.6f}")

    # 非对称量化
    qt_asym = quantize_per_channel_asymmetric(kv, axis=-1)
    err_asym = quantization_error(kv, qt_asym, axis=-1)
    print("\nAsymmetric INT8:")
    for k, v in err_asym.items():
        print(f"  {k}: {v:.6f}")
