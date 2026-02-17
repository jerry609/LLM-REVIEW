"""
KV Cache 驱逐策略集合。

面试要点：
- LRU: 最久未访问者优先淘汰，简单稳健
- LFU: 访问次数最少者优先，适合稳定热点
- Fair: 超配租户优先驱逐，保护小租户
- 所有策略共享 EvictionPolicy 接口，便于对比和组合
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core import SequenceKVCache


class EvictionPolicy(ABC):
    """驱逐策略接口。"""

    @abstractmethod
    def select_victim(self, sequences: Dict[str, SequenceKVCache]) -> Optional[str]:
        """选出一个 victim seq_id。若无可驱逐对象返回 None。"""
        ...


class LRUPolicy(EvictionPolicy):
    """Least Recently Used：最久未访问者优先。"""

    def select_victim(self, sequences: Dict[str, SequenceKVCache]) -> Optional[str]:
        if not sequences:
            return None
        return min(sequences.keys(), key=lambda k: (sequences[k].last_access_step, k))


class LFUPolicy(EvictionPolicy):
    """Least Frequently Used：访问次数最少者优先（LRU 做 tie-break）。"""

    def select_victim(self, sequences: Dict[str, SequenceKVCache]) -> Optional[str]:
        if not sequences:
            return None
        return min(
            sequences.keys(),
            key=lambda k: (sequences[k].use_count, sequences[k].last_access_step, k),
        )


class FairPolicy(EvictionPolicy):
    """
    配额公平驱逐：超配租户中选最旧条目。

    tenant 通过 seq_id 的 "tenant::" 前缀识别。
    """

    def __init__(self, tenant_weights: Dict[str, float] = None, total_blocks: int = 512):
        self.tenant_weights = tenant_weights or {}
        self.total_blocks = total_blocks

    @staticmethod
    def _tenant_of(seq_id: str) -> str:
        return seq_id.split("::", 1)[0] if "::" in seq_id else "default"

    def _quotas(self, sequences: Dict[str, SequenceKVCache]) -> Dict[str, float]:
        tenants = set(self.tenant_weights.keys())
        for sid in sequences:
            tenants.add(self._tenant_of(sid))
        weights = {t: float(self.tenant_weights.get(t, 1.0)) for t in tenants}
        total_w = sum(weights.values())
        return {t: (w / total_w) * self.total_blocks for t, w in weights.items()}

    def select_victim(self, sequences: Dict[str, SequenceKVCache]) -> Optional[str]:
        if not sequences:
            return None

        usage: Dict[str, int] = {}
        for sid, seq in sequences.items():
            t = self._tenant_of(sid)
            usage[t] = usage.get(t, 0) + seq.num_blocks()

        quotas = self._quotas(sequences)
        over = [t for t in usage if usage[t] > quotas.get(t, 0.0)]

        if over:
            victim_t = max(over, key=lambda t: (usage[t] - quotas.get(t, 0.0), t))
            candidates = [sid for sid in sequences if self._tenant_of(sid) == victim_t]
            return min(candidates, key=lambda k: (sequences[k].last_access_step, k))

        # fallback LRU
        return min(sequences.keys(), key=lambda k: (sequences[k].last_access_step, k))


# ============ Demo ============
if __name__ == "__main__":
    # 构造几个 mock sequence
    seqs = {
        "A::s1": SequenceKVCache(seq_id="A::s1", num_tokens=100, last_access_step=1, use_count=5),
        "A::s2": SequenceKVCache(seq_id="A::s2", num_tokens=200, last_access_step=3, use_count=2),
        "B::s3": SequenceKVCache(seq_id="B::s3", num_tokens=50, last_access_step=2, use_count=1),
    }

    for name, policy in [("LRU", LRUPolicy()), ("LFU", LFUPolicy())]:
        v = policy.select_victim(seqs)
        print(f"{name} victim: {v}")

    fair = FairPolicy(tenant_weights={"A": 1, "B": 1}, total_blocks=64)
    print(f"Fair victim: {fair.select_victim(seqs)}")
