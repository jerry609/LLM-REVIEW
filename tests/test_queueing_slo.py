"""Tests for queueing and SLO helpers."""
import sys, os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.simulators.queueing_slo import (
    erlang_c_wait_probability,
    little_law_concurrency,
    mg1_queue_wait,
    mg1_response_time,
    mm1_stats,
    mmc_avg_queue_wait,
    required_mm1_service_rate,
)



def test_little_law_concurrency():
    assert little_law_concurrency(arrival_rate=20.0, avg_latency=0.5) == pytest.approx(10.0)



def test_mm1_stats_match_closed_form():
    stats = mm1_stats(arrival_rate=4.0, service_rate=10.0)
    assert stats.utilization == pytest.approx(0.4)
    assert stats.avg_response_time == pytest.approx(1.0 / 6.0)
    assert stats.avg_queue_wait == pytest.approx(1.0 / 15.0)
    assert stats.avg_in_system == pytest.approx(2.0 / 3.0)
    assert stats.avg_queue_length == pytest.approx(4.0 / 15.0)



def test_erlang_c_wait_probability_and_mmc_wait():
    wait_prob = erlang_c_wait_probability(arrival_rate=8.0, service_rate=5.0, servers=2)
    assert 0.0 < wait_prob < 1.0

    wait_2 = mmc_avg_queue_wait(arrival_rate=8.0, service_rate=5.0, servers=2)
    wait_3 = mmc_avg_queue_wait(arrival_rate=8.0, service_rate=5.0, servers=3)
    assert wait_3 < wait_2



def test_mg1_matches_mm1_when_service_is_exponential():
    arrival_rate = 4.0
    service_rate = 10.0
    mean_service_time = 1.0 / service_rate

    mm1 = mm1_stats(arrival_rate=arrival_rate, service_rate=service_rate)
    assert mg1_queue_wait(arrival_rate, mean_service_time, service_time_cv=1.0) == pytest.approx(mm1.avg_queue_wait)
    assert mg1_response_time(arrival_rate, mean_service_time, service_time_cv=1.0) == pytest.approx(mm1.avg_response_time)



def test_required_service_rate_for_target_response_time():
    assert required_mm1_service_rate(arrival_rate=4.0, target_response_time=0.25) == pytest.approx(8.0)
