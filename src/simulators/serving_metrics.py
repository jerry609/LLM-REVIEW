"""Serving metric helpers for interview practice."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class RequestMetrics:
    request_id: str
    output_tokens: int
    ttft: float
    tpot: float
    e2e_latency: float


def ttft(request_arrive: float, first_token_out: float) -> float:
    return first_token_out - request_arrive


def tpot(first_token_out: float, last_token_out: float, output_tokens: int) -> float:
    if output_tokens <= 1:
        return 0.0
    return (last_token_out - first_token_out) / (output_tokens - 1)


def e2e_latency(request_arrive: float, last_token_out: float) -> float:
    return last_token_out - request_arrive


def request_metrics(
    request_id: str,
    request_arrive: float,
    first_token_out: float,
    last_token_out: float,
    output_tokens: int,
) -> RequestMetrics:
    return RequestMetrics(
        request_id=request_id,
        output_tokens=output_tokens,
        ttft=ttft(request_arrive, first_token_out),
        tpot=tpot(first_token_out, last_token_out, output_tokens),
        e2e_latency=e2e_latency(request_arrive, last_token_out),
    )


def token_throughput(output_tokens: Sequence[int], total_time: float) -> float:
    if total_time <= 0:
        raise ValueError("total_time must be positive")
    return sum(output_tokens) / total_time


def request_throughput(num_completed: int, total_time: float) -> float:
    if total_time <= 0:
        raise ValueError("total_time must be positive")
    return num_completed / total_time


def goodput(
    metrics: Iterable[RequestMetrics],
    total_time: float,
    ttft_slo: float,
    tpot_slo: float,
) -> float:
    if total_time <= 0:
        raise ValueError("total_time must be positive")
    metrics = list(metrics)
    satisfied = sum(m.ttft <= ttft_slo and m.tpot <= tpot_slo for m in metrics)
    return satisfied / total_time


def batch_utilization(active_batch_history: Sequence[int], max_batch_size: int) -> float:
    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive")
    if not active_batch_history:
        return 0.0
    return sum(active_batch_history) / (len(active_batch_history) * max_batch_size)


def kv_step_bytes(active_batch: int, bytes_per_token: int, avg_cache_tokens: int) -> int:
    if active_batch < 0 or bytes_per_token < 0 or avg_cache_tokens < 0:
        raise ValueError("inputs must be non-negative")
    return active_batch * bytes_per_token * avg_cache_tokens


if __name__ == "__main__":
    metrics = [
        request_metrics("r1", request_arrive=0.0, first_token_out=0.25, last_token_out=0.65, output_tokens=9),
        request_metrics("r2", request_arrive=0.0, first_token_out=0.40, last_token_out=0.88, output_tokens=13),
    ]

    print("token throughput:", token_throughput([m.output_tokens for m in metrics], total_time=1.0))
    print("request throughput:", request_throughput(len(metrics), total_time=1.0))
    print("goodput:", goodput(metrics, total_time=1.0, ttft_slo=0.5, tpot_slo=0.06))
