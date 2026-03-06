"""Queueing and SLO helpers for serving system reasoning."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MM1Stats:
    arrival_rate: float
    service_rate: float
    utilization: float
    avg_response_time: float
    avg_queue_wait: float
    avg_in_system: float
    avg_queue_length: float


def little_law_concurrency(arrival_rate: float, avg_latency: float) -> float:
    if arrival_rate < 0 or avg_latency < 0:
        raise ValueError("arrival_rate and avg_latency must be non-negative")
    return arrival_rate * avg_latency



def utilization(arrival_rate: float, service_rate: float, servers: int = 1) -> float:
    if arrival_rate < 0 or service_rate <= 0 or servers <= 0:
        raise ValueError("invalid utilization inputs")
    return arrival_rate / (servers * service_rate)



def mm1_stats(arrival_rate: float, service_rate: float) -> MM1Stats:
    rho = utilization(arrival_rate, service_rate, servers=1)
    if rho >= 1.0:
        raise ValueError("MM1 requires arrival_rate < service_rate")

    avg_response_time = 1.0 / (service_rate - arrival_rate)
    avg_queue_wait = rho / (service_rate - arrival_rate)
    avg_in_system = arrival_rate * avg_response_time
    avg_queue_length = arrival_rate * avg_queue_wait

    return MM1Stats(
        arrival_rate=arrival_rate,
        service_rate=service_rate,
        utilization=rho,
        avg_response_time=avg_response_time,
        avg_queue_wait=avg_queue_wait,
        avg_in_system=avg_in_system,
        avg_queue_length=avg_queue_length,
    )



def erlang_c_wait_probability(arrival_rate: float, service_rate: float, servers: int) -> float:
    rho = utilization(arrival_rate, service_rate, servers=servers)
    if rho >= 1.0:
        raise ValueError("MMC requires utilization < 1")

    traffic = arrival_rate / service_rate
    numerator = (traffic ** servers / math.factorial(servers)) * (1.0 / (1.0 - rho))
    denominator = sum((traffic ** k) / math.factorial(k) for k in range(servers)) + numerator
    return numerator / denominator



def mmc_avg_queue_wait(arrival_rate: float, service_rate: float, servers: int) -> float:
    wait_prob = erlang_c_wait_probability(arrival_rate, service_rate, servers)
    return wait_prob / (servers * service_rate - arrival_rate)



def mg1_queue_wait(arrival_rate: float, mean_service_time: float, service_time_cv: float = 1.0) -> float:
    if arrival_rate < 0 or mean_service_time <= 0 or service_time_cv < 0:
        raise ValueError("invalid MG1 inputs")

    rho = arrival_rate * mean_service_time
    if rho >= 1.0:
        raise ValueError("MG1 requires utilization < 1")

    second_moment = (1.0 + service_time_cv ** 2) * (mean_service_time ** 2)
    return arrival_rate * second_moment / (2.0 * (1.0 - rho))



def mg1_response_time(arrival_rate: float, mean_service_time: float, service_time_cv: float = 1.0) -> float:
    return mean_service_time + mg1_queue_wait(arrival_rate, mean_service_time, service_time_cv)



def required_mm1_service_rate(arrival_rate: float, target_response_time: float) -> float:
    if arrival_rate < 0 or target_response_time <= 0:
        raise ValueError("arrival_rate must be non-negative and target_response_time positive")
    return arrival_rate + 1.0 / target_response_time


if __name__ == "__main__":
    stats = mm1_stats(arrival_rate=4.0, service_rate=10.0)
    print("little law concurrency:", little_law_concurrency(arrival_rate=20.0, avg_latency=0.5))
    print("mm1 stats:", stats)
    print("erlang-c wait prob:", round(erlang_c_wait_probability(arrival_rate=8.0, service_rate=5.0, servers=2), 4))
    print("mg1 response:", round(mg1_response_time(arrival_rate=4.0, mean_service_time=0.1, service_time_cv=1.0), 4))
    print("required service rate:", required_mm1_service_rate(arrival_rate=4.0, target_response_time=0.25))
