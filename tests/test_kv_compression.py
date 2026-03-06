"""Tests for KV compression helpers."""
import sys, os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.kv_cache.compression.quantizer import (
    dequantize,
    quantization_error,
    quantize_per_channel_symmetric,
)
from src.kv_cache.compression.sparsifier import (
    compression_ratio,
    cumulative_attention_scores,
    keep_recent_and_heavy_hitters,
    snapkv_select,
)



def test_symmetric_quantization_roundtrip():
    tensor = np.array(
        [[1.0, -2.0, 0.5], [0.25, -0.75, 1.5]],
        dtype=np.float32,
    )
    qt = quantize_per_channel_symmetric(tensor, axis=-1, bits=8)
    reconstructed = dequantize(qt, axis=-1)
    assert reconstructed.shape == tensor.shape

    stats = quantization_error(tensor, qt, axis=-1)
    assert stats["compression_ratio"] > 1.0
    assert stats["rmse"] >= 0.0



def test_h2o_style_selection_keeps_recent_and_heavy_hitters():
    attention = np.array(
        [
            [0.60, 0.20, 0.10, 0.10],
            [0.70, 0.10, 0.10, 0.10],
            [0.05, 0.05, 0.10, 0.80],
        ],
        dtype=np.float32,
    )
    scores = cumulative_attention_scores(attention)
    np.testing.assert_allclose(scores, np.array([1.35, 0.35, 0.30, 1.00], dtype=np.float32), atol=1e-6)

    selection = keep_recent_and_heavy_hitters(attention, budget=3, recent_window=1)
    np.testing.assert_array_equal(selection.kept_indices, np.array([0, 1, 3], dtype=np.int32))
    np.testing.assert_array_equal(selection.dropped_indices, np.array([2], dtype=np.int32))



def test_snapkv_style_selection_and_ratio():
    observation_scores = np.array([0.9, 0.1, 0.2, 0.05, 0.8], dtype=np.float32)
    selection = snapkv_select(observation_scores, budget=3, recent_window=1)
    np.testing.assert_array_equal(selection.kept_indices, np.array([0, 2, 4], dtype=np.int32))
    assert compression_ratio(total_tokens=16, kept_tokens=4) == pytest.approx(4.0)
