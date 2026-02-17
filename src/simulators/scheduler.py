"""A simplified continuous batching scheduler simulator.

This is intentionally minimal: it helps reason about request lifecycle and
prefill/decode scheduling decisions in interviews.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Request:
    req_id: str
    input_tokens: int
    output_tokens: int
    prefilled: int = 0
    decoded: int = 0
    finished: bool = False

    def stage(self) -> str:
        if self.finished:
            return "done"
        if self.prefilled < self.input_tokens:
            return "prefill"
        return "decode"


class ContinuousBatchScheduler:
    def __init__(self, max_batch_size: int = 8, prefill_chunk: int = 128) -> None:
        self.max_batch_size = max_batch_size
        self.prefill_chunk = prefill_chunk
        self.queue: List[Request] = []
        self.time_step = 0

    def submit(self, req: Request) -> None:
        self.queue.append(req)

    def _active(self) -> List[Request]:
        return [r for r in self.queue if not r.finished]

    def step(self) -> None:
        """One scheduling tick:
        1) prioritize decode requests (latency sensitive),
        2) use spare slots for prefill chunking.
        """
        self.time_step += 1
        active = self._active()
        if not active:
            return

        decode_reqs = [r for r in active if r.stage() == "decode"]
        prefill_reqs = [r for r in active if r.stage() == "prefill"]

        # Decode first: one token per request each tick.
        selected_decode = decode_reqs[: self.max_batch_size]
        for req in selected_decode:
            req.decoded += 1
            if req.decoded >= req.output_tokens:
                req.finished = True

        remaining_slots = self.max_batch_size - len(selected_decode)
        if remaining_slots <= 0:
            return

        selected_prefill = prefill_reqs[:remaining_slots]
        for req in selected_prefill:
            req.prefilled = min(req.input_tokens, req.prefilled + self.prefill_chunk)

    def run_until_done(self, max_steps: int = 10_000) -> int:
        steps = 0
        while self._active() and steps < max_steps:
            self.step()
            steps += 1
        return steps


if __name__ == "__main__":
    scheduler = ContinuousBatchScheduler(max_batch_size=4, prefill_chunk=64)
    scheduler.submit(Request("r1", input_tokens=512, output_tokens=32))
    scheduler.submit(Request("r2", input_tokens=128, output_tokens=64))
    scheduler.submit(Request("r3", input_tokens=1024, output_tokens=16))

    n_steps = scheduler.run_until_done()
    print("finished in steps:", n_steps)
    for r in scheduler.queue:
        print(r.req_id, r.prefilled, r.decoded, r.finished)
