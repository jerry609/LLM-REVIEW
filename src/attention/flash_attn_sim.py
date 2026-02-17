"""A small FlashAttention-style tiled attention simulator in NumPy.

Focus: demonstrate online-softmax accumulation with block processing.
"""

from __future__ import annotations

import numpy as np


def flash_attention_tiled(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    block_size: int = 64,
) -> np.ndarray:
    """Compute attention(Q, K, V) using tiled online softmax.

    q, k, v: [T, Dh]
    returns: [T, Dh]
    """
    if not (q.ndim == k.ndim == v.ndim == 2):
        raise ValueError("q, k, v must be rank-2 arrays [T, Dh]")
    if q.shape != k.shape or k.shape != v.shape:
        raise ValueError("q, k, v must have the same shape")

    seqlen, head_dim = q.shape
    scale = 1.0 / np.sqrt(head_dim)
    out = np.zeros_like(q)

    for i in range(0, seqlen, block_size):
        i_end = min(i + block_size, seqlen)
        q_blk = q[i:i_end]  # [Bi, Dh]

        m_i = np.full((q_blk.shape[0],), -np.inf, dtype=q.dtype)
        l_i = np.zeros((q_blk.shape[0],), dtype=q.dtype)
        o_i = np.zeros((q_blk.shape[0], head_dim), dtype=q.dtype)

        for j in range(0, seqlen, block_size):
            j_end = min(j + block_size, seqlen)
            k_blk = k[j:j_end]  # [Bj, Dh]
            v_blk = v[j:j_end]  # [Bj, Dh]

            scores = (q_blk @ k_blk.T) * scale  # [Bi, Bj]
            m_ij = np.max(scores, axis=1)  # [Bi]
            p = np.exp(scores - m_ij[:, None])  # [Bi, Bj]
            l_ij = np.sum(p, axis=1)  # [Bi]

            # Online softmax update.
            m_new = np.maximum(m_i, m_ij)
            alpha = np.exp(m_i - m_new)
            beta = np.exp(m_ij - m_new)
            l_new = alpha * l_i + beta * l_ij

            o_i = (alpha[:, None] * l_i[:, None] * o_i + (beta[:, None] * (p @ v_blk))) / l_new[:, None]
            m_i, l_i = m_new, l_new

        out[i:i_end] = o_i

    return out


def reference_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    scale = 1.0 / np.sqrt(q.shape[-1])
    scores = (q @ k.T) * scale
    scores -= np.max(scores, axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= np.sum(probs, axis=-1, keepdims=True)
    return probs @ v


if __name__ == "__main__":
    rng = np.random.default_rng(123)
    t, d = 128, 64
    q = rng.normal(size=(t, d)).astype(np.float32)
    k = rng.normal(size=(t, d)).astype(np.float32)
    v = rng.normal(size=(t, d)).astype(np.float32)

    y_ref = reference_attention(q, k, v)
    y_tile = flash_attention_tiled(q, k, v, block_size=32)
    max_err = np.max(np.abs(y_ref - y_tile))
    print("shape:", y_tile.shape, "max_abs_err:", float(max_err))
