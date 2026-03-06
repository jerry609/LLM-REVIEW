"""Tests for MoE routing simulator"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.simulators.moe_routing import (
    all_to_all_bytes,
    dispatch_to_experts,
    drop_rate,
    expert_capacity,
    load_balancing_loss,
    topk_route,
)


def test_topk_route_shapes_and_weights():
    logits = np.array([
        [4.0, 1.0, 0.0],
        [0.5, 3.0, 1.5],
        [1.0, 0.2, 2.5],
    ], dtype=np.float32)
    routing = topk_route(logits, top_k=2)
    assert routing.topk_indices.shape == (3, 2)
    assert routing.topk_weights.shape == (3, 2)
    np.testing.assert_allclose(routing.topk_weights.sum(axis=-1), np.ones(3), atol=1e-6)


def test_load_balancing_loss_balanced_vs_skewed():
    probs_balanced = np.full((4, 4), 0.25, dtype=np.float32)
    topk_balanced = np.array([[0], [1], [2], [3]], dtype=np.int32)
    loss_balanced, actual_freq, mean_prob = load_balancing_loss(probs_balanced, topk_balanced)
    np.testing.assert_allclose(actual_freq, np.full(4, 0.25, dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(mean_prob, np.full(4, 0.25, dtype=np.float32), atol=1e-6)
    assert abs(loss_balanced - 1.0) < 1e-6

    probs_skewed = np.full((4, 4), 0.01, dtype=np.float32)
    probs_skewed[:, 0] = 0.97
    topk_skewed = np.zeros((4, 1), dtype=np.int32)
    loss_skewed, _, _ = load_balancing_loss(probs_skewed, topk_skewed)
    assert loss_skewed > loss_balanced


def test_capacity_and_dispatch():
    logits = np.array([
        [5.0, 1.0],
        [4.0, 1.0],
        [3.0, 1.0],
        [0.5, 2.5],
    ], dtype=np.float32)
    hidden = np.random.randn(4, 8).astype(np.float32)
    routing = topk_route(logits, top_k=1)
    batches, dropped = dispatch_to_experts(hidden, routing, capacity=1)
    accepted = sum(len(batch.token_indices) for batch in batches)
    assert accepted == 2
    assert dropped.sum() == 2
    assert abs(drop_rate(dropped) - 0.5) < 1e-6
    assert expert_capacity(num_tokens=4, num_experts=2, capacity_factor=1.0, top_k=1) == 2


def test_all_to_all_bytes():
    assert all_to_all_bytes(num_tokens=128, model_dim=4096, top_k=2, bytes_per_elem=2) == 2 * 128 * 2 * 4096 * 2


if __name__ == "__main__":
    test_topk_route_shapes_and_weights()
    test_load_balancing_loss_balanced_vs_skewed()
    test_capacity_and_dispatch()
    test_all_to_all_bytes()
    print("All MoE routing tests passed")
