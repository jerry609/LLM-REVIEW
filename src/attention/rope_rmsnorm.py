"""RoPE + RMSNorm utilities for interview practice."""

from __future__ import annotations

import numpy as np


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """RMSNorm over the last dimension."""
    if x.shape[-1] != weight.shape[0]:
        raise ValueError("x last dim must match weight dim")
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def _rotate_half(x: np.ndarray) -> np.ndarray:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    out = np.empty_like(x)
    out[..., ::2] = -x2
    out[..., 1::2] = x1
    return out


def build_rope_cache(
    seqlen: int,
    head_dim: int,
    theta: float = 10000.0,
) -> tuple[np.ndarray, np.ndarray]:
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")
    idx = np.arange(0, head_dim, 2, dtype=np.float32)
    inv_freq = 1.0 / (theta ** (idx / head_dim))
    pos = np.arange(seqlen, dtype=np.float32)
    freqs = np.outer(pos, inv_freq)  # [T, Dh/2]
    cos = np.repeat(np.cos(freqs), 2, axis=-1)  # [T, Dh]
    sin = np.repeat(np.sin(freqs), 2, axis=-1)  # [T, Dh]
    return cos, sin


def apply_rope(
    q: np.ndarray,
    k: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply RoPE to Q/K.

    q, k: [B, H, T, Dh]
    cos, sin: [T, Dh]
    """
    if q.shape != k.shape:
        raise ValueError("q and k must have the same shape")
    _, _, seqlen, head_dim = q.shape
    if cos.shape != (seqlen, head_dim) or sin.shape != (seqlen, head_dim):
        raise ValueError("cos/sin shapes must be [T, Dh] and match q/k")
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q_out = q * cos + _rotate_half(q) * sin
    k_out = k * cos + _rotate_half(k) * sin
    return q_out, k_out


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    bsz, nheads, seqlen, head_dim = 2, 4, 16, 8
    q = rng.normal(size=(bsz, nheads, seqlen, head_dim)).astype(np.float32)
    k = rng.normal(size=(bsz, nheads, seqlen, head_dim)).astype(np.float32)
    cos, sin = build_rope_cache(seqlen, head_dim)
    q2, k2 = apply_rope(q, k, cos, sin)
    print("rope q/k:", q2.shape, k2.shape)

    x = rng.normal(size=(bsz, seqlen, nheads * head_dim)).astype(np.float32)
    w = np.ones((nheads * head_dim,), dtype=np.float32)
    y = rms_norm(x, w)
    print("rmsnorm:", y.shape)
