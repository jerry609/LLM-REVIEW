"""Tests for attention modules"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.attention.mha_gqa import _split_heads, _merge_heads, _scaled_dot_product_attention, mha_gqa_forward
from src.attention.rope_rmsnorm import rms_norm, build_rope_cache, apply_rope


def test_split_merge_heads():
    B, S, D, H = 2, 10, 64, 4
    x = np.random.randn(B, S, D).astype(np.float32)
    split = _split_heads(x, H)
    assert split.shape == (B, H, S, D // H)
    merged = _merge_heads(split)
    assert merged.shape == (B, S, D)
    np.testing.assert_allclose(merged, x, atol=1e-6)


def test_sdpa_output_shape():
    B, H, S, D = 2, 4, 10, 16
    Q = np.random.randn(B, H, S, D).astype(np.float32)
    K = np.random.randn(B, H, S, D).astype(np.float32)
    V = np.random.randn(B, H, S, D).astype(np.float32)
    out = _scaled_dot_product_attention(Q, K, V)
    assert out.shape == (B, H, S, D)


def test_mha_gqa_forward_mha():
    B, S, D, H = 2, 8, 64, 4
    x = np.random.randn(B, S, D).astype(np.float32)
    Wq = np.random.randn(D, D).astype(np.float32) * 0.02
    Wk = np.random.randn(D, D).astype(np.float32) * 0.02
    Wv = np.random.randn(D, D).astype(np.float32) * 0.02
    Wo = np.random.randn(D, D).astype(np.float32) * 0.02
    out = mha_gqa_forward(x, Wq, Wk, Wv, Wo, H, H)
    assert out.shape == (B, S, D)


def test_rms_norm():
    x = np.random.randn(2, 5, 64).astype(np.float32)
    g = np.ones(64, dtype=np.float32)
    out = rms_norm(x, g)
    assert out.shape == x.shape
    # 归一化后 RMS 应接近 1
    rms_val = np.sqrt(np.mean(out ** 2, axis=-1))
    np.testing.assert_allclose(rms_val, np.ones_like(rms_val), atol=0.1)


def test_rope_cache():
    # build_rope_cache(seqlen, head_dim) -> (cos[seqlen, head_dim], sin[seqlen, head_dim])
    seqlen, head_dim = 128, 64
    cos, sin = build_rope_cache(seqlen, head_dim)
    assert cos.shape == (seqlen, head_dim), f"Expected ({seqlen}, {head_dim}), got {cos.shape}"
    assert sin.shape == (seqlen, head_dim)


def test_apply_rope():
    B, H, S, D = 2, 4, 16, 64
    q = np.random.randn(B, H, S, D).astype(np.float32)
    k = np.random.randn(B, H, S, D).astype(np.float32)
    cos, sin = build_rope_cache(S, D)
    q_out, k_out = apply_rope(q, k, cos, sin)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape
    # RoPE 不应改变向量的模长（太多）
    orig_norm = np.linalg.norm(q, axis=-1)
    new_norm = np.linalg.norm(q_out, axis=-1)
    np.testing.assert_allclose(orig_norm, new_norm, rtol=0.05)


if __name__ == "__main__":
    test_split_merge_heads()
    test_sdpa_output_shape()
    test_mha_gqa_forward_mha()
    test_rms_norm()
    test_rope_cache()
    test_apply_rope()
    print("All attention tests passed ✓")
