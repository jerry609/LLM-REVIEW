"""Tests for serving metrics"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.simulators.serving_metrics import (
    batch_utilization,
    e2e_latency,
    goodput,
    kv_step_bytes,
    request_metrics,
    request_throughput,
    token_throughput,
    tpot,
    ttft,
)


def test_latency_metrics():
    assert ttft(0.0, 0.25) == 0.25
    assert e2e_latency(0.0, 1.25) == 1.25
    assert tpot(0.25, 1.25, output_tokens=6) == 0.2


def test_request_metrics_bundle():
    metrics = request_metrics("r1", 0.0, 0.4, 1.0, output_tokens=4)
    assert metrics.ttft == pytest.approx(0.4)
    assert metrics.tpot == pytest.approx(0.2)
    assert metrics.e2e_latency == pytest.approx(1.0)


def test_throughput_and_goodput():
    metrics = [
        request_metrics("r1", 0.0, 0.3, 0.9, output_tokens=7),
        request_metrics("r2", 0.0, 0.8, 1.6, output_tokens=5),
    ]
    assert token_throughput([m.output_tokens for m in metrics], total_time=2.0) == 6.0
    assert request_throughput(num_completed=2, total_time=2.0) == 1.0
    assert goodput(metrics, total_time=2.0, ttft_slo=0.5, tpot_slo=0.12) == 0.5


def test_batch_utilization_and_kv_bytes():
    assert batch_utilization([2, 4, 4], max_batch_size=4) == (2 + 4 + 4) / 12
    assert kv_step_bytes(active_batch=8, bytes_per_token=131072, avg_cache_tokens=4096) == 8 * 131072 * 4096


if __name__ == "__main__":
    test_latency_metrics()
    test_request_metrics_bundle()
    test_throughput_and_goodput()
    test_batch_utilization_and_kv_bytes()
    print("All serving metric tests passed")
