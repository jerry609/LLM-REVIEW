"""Minimal KV sparsification helpers for interview practice."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TokenSelection:
    kept_indices: np.ndarray
    dropped_indices: np.ndarray
    keep_mask: np.ndarray


def cumulative_attention_scores(attention_history: np.ndarray) -> np.ndarray:
    if attention_history.ndim != 2:
        raise ValueError("attention_history must be rank-2 [num_steps, num_tokens]")
    return np.sum(attention_history.astype(np.float32), axis=0)


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.empty(0, dtype=np.int32)
    k = min(k, scores.shape[0])
    topk = np.argpartition(scores, -k)[-k:]
    ordered = topk[np.argsort(-scores[topk])]
    return np.sort(ordered.astype(np.int32))


def _build_selection(total_tokens: int, keep_indices: np.ndarray) -> TokenSelection:
    keep_mask = np.zeros(total_tokens, dtype=bool)
    keep_mask[keep_indices] = True
    dropped_indices = np.flatnonzero(~keep_mask).astype(np.int32)
    return TokenSelection(
        kept_indices=np.sort(keep_indices.astype(np.int32)),
        dropped_indices=dropped_indices,
        keep_mask=keep_mask,
    )


def keep_recent_and_heavy_hitters(
    attention_history: np.ndarray,
    budget: int,
    recent_window: int,
) -> TokenSelection:
    if attention_history.ndim != 2:
        raise ValueError("attention_history must be rank-2 [num_steps, num_tokens]")

    _, total_tokens = attention_history.shape
    if budget <= 0 or budget > total_tokens:
        raise ValueError("budget must be in [1, num_tokens]")
    if recent_window < 0:
        raise ValueError("recent_window must be non-negative")

    recent_start = max(0, total_tokens - recent_window)
    recent = np.arange(recent_start, total_tokens, dtype=np.int32)

    prefix_scores = cumulative_attention_scores(attention_history)
    prefix_scores[recent_start:] = -np.inf

    heavy_budget = max(0, budget - recent.shape[0])
    heavy = _topk_indices(prefix_scores, heavy_budget)
    keep_indices = np.unique(np.concatenate([recent, heavy])).astype(np.int32)

    if keep_indices.shape[0] > budget:
        keep_indices = keep_indices[-budget:]

    return _build_selection(total_tokens, keep_indices)


def snapkv_select(
    observation_scores: np.ndarray,
    budget: int,
    recent_window: int = 0,
) -> TokenSelection:
    if observation_scores.ndim != 1:
        raise ValueError("observation_scores must be rank-1 [num_tokens]")

    total_tokens = observation_scores.shape[0]
    if budget <= 0 or budget > total_tokens:
        raise ValueError("budget must be in [1, num_tokens]")
    if recent_window < 0:
        raise ValueError("recent_window must be non-negative")

    recent_start = max(0, total_tokens - recent_window)
    recent = np.arange(recent_start, total_tokens, dtype=np.int32)

    prefix_scores = observation_scores.astype(np.float32).copy()
    prefix_scores[recent_start:] = -np.inf

    score_budget = max(0, budget - recent.shape[0])
    selected = _topk_indices(prefix_scores, score_budget)
    keep_indices = np.unique(np.concatenate([recent, selected])).astype(np.int32)

    if keep_indices.shape[0] > budget:
        keep_indices = keep_indices[-budget:]

    return _build_selection(total_tokens, keep_indices)


def compression_ratio(total_tokens: int, kept_tokens: int) -> float:
    if total_tokens <= 0 or kept_tokens <= 0 or kept_tokens > total_tokens:
        raise ValueError("require 0 < kept_tokens <= total_tokens")
    return total_tokens / kept_tokens


if __name__ == "__main__":
    attention = np.array(
        [
            [0.60, 0.20, 0.10, 0.10],
            [0.70, 0.10, 0.10, 0.10],
            [0.05, 0.05, 0.10, 0.80],
        ],
        dtype=np.float32,
    )

    h2o = keep_recent_and_heavy_hitters(attention, budget=3, recent_window=1)
    snapkv = snapkv_select(np.array([0.9, 0.1, 0.2, 0.05, 0.8], dtype=np.float32), budget=3, recent_window=1)

    print("h2o kept:", h2o.kept_indices.tolist())
    print("snapkv kept:", snapkv.kept_indices.tolist())
    print("compression ratio:", compression_ratio(total_tokens=16, kept_tokens=4))
