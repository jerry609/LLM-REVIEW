"""Minimal MoE routing simulator for interview practice."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class RoutingResult:
    router_probs: np.ndarray
    topk_indices: np.ndarray
    topk_weights: np.ndarray
    expert_load: np.ndarray


@dataclass(frozen=True)
class ExpertBatch:
    expert_id: int
    token_indices: np.ndarray
    weights: np.ndarray


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def topk_route(router_logits: np.ndarray, top_k: int = 2) -> RoutingResult:
    if router_logits.ndim != 2:
        raise ValueError("router_logits must be rank-2 [num_tokens, num_experts]")

    num_tokens, num_experts = router_logits.shape
    if not 1 <= top_k <= num_experts:
        raise ValueError("top_k must be in [1, num_experts]")

    router_probs = softmax(router_logits.astype(np.float32), axis=-1)
    topk_indices = np.argpartition(router_probs, -top_k, axis=-1)[:, -top_k:]

    row_ids = np.arange(num_tokens)[:, None]
    topk_scores = router_probs[row_ids, topk_indices]
    order = np.argsort(-topk_scores, axis=-1)
    topk_indices = np.take_along_axis(topk_indices, order, axis=-1)
    topk_scores = np.take_along_axis(topk_scores, order, axis=-1)
    topk_weights = topk_scores / np.sum(topk_scores, axis=-1, keepdims=True)
    expert_load = np.bincount(topk_indices.reshape(-1), minlength=num_experts).astype(np.int32)

    return RoutingResult(
        router_probs=router_probs,
        topk_indices=topk_indices,
        topk_weights=topk_weights,
        expert_load=expert_load,
    )


def load_balancing_loss(router_probs: np.ndarray, topk_indices: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    if router_probs.ndim != 2 or topk_indices.ndim != 2:
        raise ValueError("router_probs and topk_indices must both be rank-2")
    if router_probs.shape[0] != topk_indices.shape[0]:
        raise ValueError("token dimension must match")

    num_tokens, num_experts = router_probs.shape
    selection = np.zeros((num_tokens, num_experts), dtype=np.float32)
    selection[np.arange(num_tokens)[:, None], topk_indices] = 1.0

    actual_freq = np.sum(selection, axis=0) / num_tokens
    mean_prob = np.mean(router_probs, axis=0)
    loss = num_experts * np.sum(actual_freq * mean_prob)
    return float(loss), actual_freq, mean_prob


def expert_capacity(num_tokens: int, num_experts: int, capacity_factor: float = 1.25, top_k: int = 1) -> int:
    if num_tokens < 0 or num_experts <= 0 or capacity_factor <= 0 or top_k <= 0:
        raise ValueError("invalid capacity inputs")
    return int(np.ceil(capacity_factor * num_tokens * top_k / num_experts))


def dispatch_to_experts(
    hidden_states: np.ndarray,
    routing: RoutingResult,
    capacity: int,
) -> tuple[List[ExpertBatch], np.ndarray]:
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must be rank-2 [num_tokens, d_model]")
    if hidden_states.shape[0] != routing.topk_indices.shape[0]:
        raise ValueError("token dimension must match routing result")
    if capacity <= 0:
        raise ValueError("capacity must be positive")

    num_experts = routing.router_probs.shape[1]
    top_k = routing.topk_indices.shape[1]
    accepted_indices: List[List[int]] = [[] for _ in range(num_experts)]
    accepted_weights: List[List[float]] = [[] for _ in range(num_experts)]
    dropped = np.zeros((hidden_states.shape[0], top_k), dtype=bool)

    for token_id in range(hidden_states.shape[0]):
        for slot in range(top_k):
            expert_id = int(routing.topk_indices[token_id, slot])
            if len(accepted_indices[expert_id]) < capacity:
                accepted_indices[expert_id].append(token_id)
                accepted_weights[expert_id].append(float(routing.topk_weights[token_id, slot]))
            else:
                dropped[token_id, slot] = True

    expert_batches: List[ExpertBatch] = []
    for expert_id in range(num_experts):
        expert_batches.append(
            ExpertBatch(
                expert_id=expert_id,
                token_indices=np.asarray(accepted_indices[expert_id], dtype=np.int32),
                weights=np.asarray(accepted_weights[expert_id], dtype=np.float32),
            )
        )
    return expert_batches, dropped


def drop_rate(dropped_mask: np.ndarray) -> float:
    if dropped_mask.size == 0:
        return 0.0
    return float(np.mean(dropped_mask))


def all_to_all_bytes(num_tokens: int, model_dim: int, top_k: int = 1, bytes_per_elem: int = 2) -> int:
    if min(num_tokens, model_dim, top_k, bytes_per_elem) < 0:
        raise ValueError("all_to_all inputs must be non-negative")
    return 2 * num_tokens * top_k * model_dim * bytes_per_elem


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    router_logits = rng.normal(size=(8, 4)).astype(np.float32)
    hidden_states = rng.normal(size=(8, 16)).astype(np.float32)

    routing = topk_route(router_logits, top_k=2)
    loss, actual_freq, mean_prob = load_balancing_loss(routing.router_probs, routing.topk_indices)
    capacity = expert_capacity(num_tokens=hidden_states.shape[0], num_experts=4, capacity_factor=1.25, top_k=2)
    batches, dropped = dispatch_to_experts(hidden_states, routing, capacity=capacity)

    print("balance loss:", round(loss, 4))
    print("actual freq:", actual_freq)
    print("mean prob:", mean_prob)
    print("drop rate:", round(drop_rate(dropped), 4))
    print("all-to-all bytes:", all_to_all_bytes(num_tokens=8, model_dim=16, top_k=2, bytes_per_elem=2))
    for batch in batches:
        print(batch.expert_id, batch.token_indices.tolist(), batch.weights.tolist())
