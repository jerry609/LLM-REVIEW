"""Tests for LoRA module"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.training.lora import LoRALinear


def test_lora_shapes():
    lora = LoRALinear(in_features=64, out_features=32, rank=4, alpha=8)
    assert lora.weight.shape == (64, 32)
    assert lora.lora_a.shape == (64, 4)
    assert lora.lora_b.shape == (4, 32)


def test_lora_delta_weight():
    lora = LoRALinear(in_features=64, out_features=32, rank=4, alpha=8)
    dw = lora.delta_weight()
    assert dw.shape == (64, 32)


def test_lora_forward():
    lora = LoRALinear(in_features=64, out_features=32, rank=4, alpha=8)
    x = np.random.randn(2, 10, 64).astype(np.float32)
    out = lora.forward(x)
    assert out.shape == (2, 10, 32)


def test_lora_merged_vs_forward():
    lora = LoRALinear(in_features=64, out_features=32, rank=4, alpha=8)
    x = np.random.randn(2, 5, 64).astype(np.float32)
    out_forward = lora.forward(x)
    W_merged = lora.merged_weight()
    out_merged = x @ W_merged
    np.testing.assert_allclose(out_forward, out_merged, atol=1e-5)


if __name__ == "__main__":
    test_lora_shapes()
    test_lora_delta_weight()
    test_lora_forward()
    test_lora_merged_vs_forward()
    print("All LoRA tests passed ✓")
