"""Minimal NumPy implementation of MHA/GQA forward pass.

This file is designed for interview practice and shape tracing. It does not
cover training-time concerns like dropout, fused kernels, or KV cache.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x - x_max)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def _split_heads(x: np.ndarray, num_heads: int) -> np.ndarray:
    # [B, T, D] -> [B, H, T, Dh]
    bsz, seqlen, dim = x.shape
    if dim % num_heads != 0:
        raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
    head_dim = dim // num_heads
    return x.reshape(bsz, seqlen, num_heads, head_dim).transpose(0, 2, 1, 3)


def _merge_heads(x: np.ndarray) -> np.ndarray:
    # [B, H, T, Dh] -> [B, T, D]
    bsz, nheads, seqlen, head_dim = x.shape
    return x.transpose(0, 2, 1, 3).reshape(bsz, seqlen, nheads * head_dim)


def _scaled_dot_product_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    # q/k/v: [B, H, T, Dh]
    head_dim = q.shape[-1]
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) / np.sqrt(head_dim)
    if mask is not None:
        # mask expected broadcastable to [B, H, T, T]
        scores = np.where(mask, scores, -1e30)
    probs = softmax(scores, axis=-1)
    return np.matmul(probs, v)


def mha_gqa_forward(
    x: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    num_heads: int,
    num_kv_heads: Optional[int] = None,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Forward for MHA/GQA.

    Args:
        x: [B, T, D]
        w_q: [D, D]
        w_k: [D, D_kv]
        w_v: [D, D_kv]
        w_o: [D, D]
        num_heads: number of Q heads.
        num_kv_heads: number of K/V heads. If None, same as num_heads (MHA).
        mask: optional attention mask.
    """
    if num_kv_heads is None:
        num_kv_heads = num_heads
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")

    q = x @ w_q  # [B, T, D]
    k = x @ w_k  # [B, T, D_kv]
    v = x @ w_v  # [B, T, D_kv]

    qh = _split_heads(q, num_heads)  # [B, Hq, T, Dh]
    kh = _split_heads(k, num_kv_heads)  # [B, Hkv, T, Dh]
    vh = _split_heads(v, num_kv_heads)  # [B, Hkv, T, Dh]

    # GQA: repeat K/V heads so each KV head serves a group of Q heads.
    group_size = num_heads // num_kv_heads
    if group_size > 1:
        kh = np.repeat(kh, repeats=group_size, axis=1)
        vh = np.repeat(vh, repeats=group_size, axis=1)

    out = _scaled_dot_product_attention(qh, kh, vh, mask=mask)
    out = _merge_heads(out)
    return out @ w_o


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    bsz, seqlen, dim = 2, 8, 32
    hq, hkv = 8, 2
    x = rng.normal(size=(bsz, seqlen, dim)).astype(np.float32)
    w_q = rng.normal(size=(dim, dim)).astype(np.float32) * 0.02
    w_k = rng.normal(size=(dim, dim // (hq // hkv))).astype(np.float32) * 0.02
    w_v = rng.normal(size=(dim, dim // (hq // hkv))).astype(np.float32) * 0.02
    w_o = rng.normal(size=(dim, dim)).astype(np.float32) * 0.02
    y = mha_gqa_forward(x, w_q, w_k, w_v, w_o, num_heads=hq, num_kv_heads=hkv)
    print("output shape:", y.shape)
