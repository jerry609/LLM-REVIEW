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


def e2e_from_ttft_tpot(ttft_value: float, tpot_value: float, output_tokens: int) -> float:
    if ttft_value < 0 or tpot_value < 0:
        raise ValueError('ttft_value and tpot_value must be non-negative')
    if output_tokens <= 0:
        raise ValueError('output_tokens must be positive')
    return ttft_value + max(output_tokens - 1, 0) * tpot_value


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
        raise ValueError('total_time must be positive')
    return sum(output_tokens) / total_time


def request_throughput(num_completed: int, total_time: float) -> float:
    if total_time <= 0:
        raise ValueError('total_time must be positive')
    return num_completed / total_time


def goodput_ratio(
    metrics: Iterable[RequestMetrics],
    ttft_slo: float,
    tpot_slo: float,
) -> float:
    if ttft_slo < 0 or tpot_slo < 0:
        raise ValueError('SLO thresholds must be non-negative')
    metrics = list(metrics)
    if not metrics:
        return 0.0
    satisfied = sum(m.ttft <= ttft_slo and m.tpot <= tpot_slo for m in metrics)
    return satisfied / len(metrics)


def goodput(
    metrics: Iterable[RequestMetrics],
    total_time: float,
    ttft_slo: float,
    tpot_slo: float,
) -> float:
    if total_time <= 0:
        raise ValueError('total_time must be positive')
    metrics = list(metrics)
    return request_throughput(len(metrics), total_time) * goodput_ratio(metrics, ttft_slo, tpot_slo)


def batch_utilization(active_batch_history: Sequence[int], max_batch_size: int) -> float:
    if max_batch_size <= 0:
        raise ValueError('max_batch_size must be positive')
    if not active_batch_history:
        return 0.0
    return sum(active_batch_history) / (len(active_batch_history) * max_batch_size)


def kv_step_bytes(active_batch: int, bytes_per_token: int, avg_cache_tokens: int) -> int:
    if active_batch < 0 or bytes_per_token < 0 or avg_cache_tokens < 0:
        raise ValueError('inputs must be non-negative')
    return active_batch * bytes_per_token * avg_cache_tokens


def kv_step_time_lower_bound(
    active_batch: int,
    bytes_per_token: int,
    avg_cache_tokens: int,
    memory_bandwidth_bytes_per_s: float,
) -> float:
    if memory_bandwidth_bytes_per_s <= 0:
        raise ValueError('memory_bandwidth_bytes_per_s must be positive')
    return kv_step_bytes(active_batch, bytes_per_token, avg_cache_tokens) / memory_bandwidth_bytes_per_s


def request_service_demand(
    input_tokens: int,
    output_tokens: int,
    prefill_seconds_per_token: float,
    decode_seconds_per_token: float,
) -> float:
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError('token counts must be non-negative')
    if prefill_seconds_per_token < 0 or decode_seconds_per_token < 0:
        raise ValueError('per-token costs must be non-negative')
    return input_tokens * prefill_seconds_per_token + output_tokens * decode_seconds_per_token


if __name__ == "__main__":
    metrics = [
        request_metrics('r1', request_arrive=0.0, first_token_out=0.25, last_token_out=0.65, output_tokens=9),
        request_metrics('r2', request_arrive=0.0, first_token_out=0.40, last_token_out=0.88, output_tokens=13),
    ]

    print('token throughput:', token_throughput([m.output_tokens for m in metrics], total_time=1.0))
    print('request throughput:', request_throughput(len(metrics), total_time=1.0))
    print('goodput ratio:', goodput_ratio(metrics, ttft_slo=0.5, tpot_slo=0.06))
    print('goodput:', goodput(metrics, total_time=1.0, ttft_slo=0.5, tpot_slo=0.06))
    print('e2e from ttft+tpot:', e2e_from_ttft_tpot(ttft_value=0.25, tpot_value=0.05, output_tokens=9))
    print(
        'service demand:',
        request_service_demand(
            input_tokens=1024,
            output_tokens=128,
            prefill_seconds_per_token=0.0004,
            decode_seconds_per_token=0.008,
        ),
    )
