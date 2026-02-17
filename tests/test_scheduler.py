"""Tests for Continuous Batching scheduler"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.simulators.scheduler import Request, ContinuousBatchScheduler


def test_submit_and_step():
    sched = ContinuousBatchScheduler(max_batch_size=4, prefill_chunk=128)
    r = Request("r0", input_tokens=50, output_tokens=5)
    sched.submit(r)
    sched.step()
    # After one step, should have started prefill or decode
    assert sched.time_step == 1


def test_run_until_done():
    sched = ContinuousBatchScheduler(max_batch_size=4, prefill_chunk=128)
    for i in range(3):
        sched.submit(Request(f"r{i}", input_tokens=30, output_tokens=10))
    steps = sched.run_until_done()
    assert steps > 0
    assert all(r.finished for r in sched.queue)


def test_max_batch_limit():
    sched = ContinuousBatchScheduler(max_batch_size=2, prefill_chunk=512)
    for i in range(5):
        sched.submit(Request(f"r{i}", input_tokens=20, output_tokens=5))
    # Run a few steps
    for _ in range(3):
        sched.step()
    # All should eventually finish
    sched.run_until_done()
    assert all(r.finished for r in sched.queue)


if __name__ == "__main__":
    test_submit_and_step()
    test_run_until_done()
    test_max_batch_limit()
    print("All scheduler tests passed ✓")
