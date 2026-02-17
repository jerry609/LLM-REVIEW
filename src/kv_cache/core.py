"""
Paged KV Cache 核心数据结构与块管理器。

面试要点：
- 固定大小 block 池，类似 OS 虚拟内存分页
- 逻辑 token 位置 → 物理 block 映射（页表）
- 支持 Copy-on-Write（前缀共享时延迟复制）
- 碎片率 = 1 - 实际有效 token / (已分配 block 数 * block_size)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class KVBlock:
    """一个物理 KV 块，存放固定数量 token 的 K/V 张量（此处用 token 计数模拟）。"""
    block_id: int
    capacity: int  # 每块可容纳的 token 数
    filled: int = 0  # 已填充 token 数
    ref_count: int = 0  # 引用计数（用于 CoW）

    @property
    def is_full(self) -> bool:
        return self.filled >= self.capacity

    @property
    def free_slots(self) -> int:
        return self.capacity - self.filled


class BlockAllocator:
    """
    物理块分配器。

    职责：
    1. 管理 free pool
    2. 分配 / 释放 block
    3. 统计碎片率
    """

    def __init__(self, num_blocks: int, block_size: int):
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size

        self._blocks: List[KVBlock] = [
            KVBlock(block_id=i, capacity=block_size) for i in range(num_blocks)
        ]
        self._free_ids: List[int] = list(range(num_blocks))

    # ---------- 查询 ----------
    @property
    def num_free(self) -> int:
        return len(self._free_ids)

    @property
    def num_used(self) -> int:
        return self.num_blocks - self.num_free

    def get_block(self, block_id: int) -> KVBlock:
        return self._blocks[block_id]

    def utilization(self) -> float:
        return self.num_used / self.num_blocks

    def fragmentation(self) -> float:
        """内部碎片率：已分配块中空闲 slot 占比。"""
        used_blocks = [b for b in self._blocks if b.ref_count > 0]
        if not used_blocks:
            return 0.0
        total_slots = sum(b.capacity for b in used_blocks)
        wasted = sum(b.free_slots for b in used_blocks)
        return wasted / total_slots

    # ---------- 分配 / 释放 ----------
    def allocate(self) -> Optional[KVBlock]:
        if not self._free_ids:
            return None
        bid = self._free_ids.pop()
        blk = self._blocks[bid]
        blk.ref_count = 1
        blk.filled = 0
        return blk

    def allocate_n(self, n: int) -> List[KVBlock]:
        blocks = []
        for _ in range(n):
            blk = self.allocate()
            if blk is None:
                # 回滚已分配的
                for b in blocks:
                    self.free(b)
                raise RuntimeError(f"Cannot allocate {n} blocks, only {self.num_free + len(blocks)} free")
            blocks.append(blk)
        return blocks

    def free(self, block: KVBlock) -> None:
        block.ref_count -= 1
        if block.ref_count <= 0:
            block.ref_count = 0
            block.filled = 0
            self._free_ids.append(block.block_id)

    def free_all(self, blocks: List[KVBlock]) -> None:
        for b in blocks:
            self.free(b)


@dataclass
class SequenceKVCache:
    """
    单个序列（请求）的 KV Cache 元数据。

    - block_table: 逻辑 block 索引 → 物理 KVBlock
    - num_tokens: 当前缓存的总 token 数
    """
    seq_id: str
    block_table: List[KVBlock] = field(default_factory=list)
    num_tokens: int = 0
    last_access_step: int = 0
    use_count: int = 0

    def num_blocks(self) -> int:
        return len(self.block_table)


class PagedKVCacheManager:
    """
    Paged KV Cache 管理器（简化版 vLLM BlockManager）。

    核心 API：
    - allocate_for_sequence(seq_id, num_tokens) → SequenceKVCache
    - append_tokens(seq, num_new_tokens) → 追加 token，按需分配新块
    - release(seq) → 释放序列所有块
    - fork(seq) → Copy-on-Write 复制（前缀共享）
    """

    def __init__(self, num_blocks: int = 512, block_size: int = 16):
        self.allocator = BlockAllocator(num_blocks, block_size)
        self.block_size = block_size
        self.sequences: Dict[str, SequenceKVCache] = {}
        self._step = 0

    @staticmethod
    def _blocks_needed(num_tokens: int, block_size: int) -> int:
        if num_tokens <= 0:
            return 0
        return int(math.ceil(num_tokens / block_size))

    def allocate_for_sequence(self, seq_id: str, num_tokens: int) -> SequenceKVCache:
        """为新序列分配初始 KV 块。"""
        self._step += 1
        n = self._blocks_needed(num_tokens, self.block_size)
        blocks = self.allocator.allocate_n(n)

        # 模拟填充
        remaining = num_tokens
        for blk in blocks:
            fill = min(remaining, blk.capacity)
            blk.filled = fill
            remaining -= fill

        seq = SequenceKVCache(
            seq_id=seq_id,
            block_table=blocks,
            num_tokens=num_tokens,
            last_access_step=self._step,
            use_count=1,
        )
        self.sequences[seq_id] = seq
        return seq

    def append_tokens(self, seq: SequenceKVCache, num_new_tokens: int) -> None:
        """追加 token 到序列的 KV Cache，按需分配新块。"""
        self._step += 1
        seq.last_access_step = self._step
        seq.use_count += 1

        remaining = num_new_tokens
        # 先填充最后一个块的空余
        if seq.block_table:
            last_blk = seq.block_table[-1]
            fill = min(remaining, last_blk.free_slots)
            last_blk.filled += fill
            remaining -= fill

        # 分配新块
        while remaining > 0:
            blk = self.allocator.allocate()
            if blk is None:
                raise RuntimeError("OOM: no free blocks")
            fill = min(remaining, blk.capacity)
            blk.filled = fill
            remaining -= fill
            seq.block_table.append(blk)

        seq.num_tokens += num_new_tokens

    def release(self, seq: SequenceKVCache) -> int:
        """释放序列的所有块，返回释放的块数。"""
        n = len(seq.block_table)
        self.allocator.free_all(seq.block_table)
        seq.block_table.clear()
        seq.num_tokens = 0
        self.sequences.pop(seq.seq_id, None)
        return n

    def fork(self, src: SequenceKVCache, new_seq_id: str) -> SequenceKVCache:
        """Copy-on-Write 分叉：新序列共享源序列的物理块（ref_count +1）。"""
        self._step += 1
        for blk in src.block_table:
            blk.ref_count += 1

        new_seq = SequenceKVCache(
            seq_id=new_seq_id,
            block_table=list(src.block_table),  # 浅拷贝
            num_tokens=src.num_tokens,
            last_access_step=self._step,
            use_count=1,
        )
        self.sequences[new_seq_id] = new_seq
        return new_seq

    def summary(self) -> Dict[str, float]:
        return {
            "total_blocks": self.allocator.num_blocks,
            "used_blocks": self.allocator.num_used,
            "free_blocks": self.allocator.num_free,
            "utilization": self.allocator.utilization(),
            "fragmentation": self.allocator.fragmentation(),
            "active_sequences": len(self.sequences),
        }


# ============ Demo ============
if __name__ == "__main__":
    mgr = PagedKVCacheManager(num_blocks=64, block_size=16)

    # 分配两个序列
    s1 = mgr.allocate_for_sequence("req-001", num_tokens=100)
    s2 = mgr.allocate_for_sequence("req-002", num_tokens=200)
    print("After alloc:", mgr.summary())

    # 追加 decode token
    mgr.append_tokens(s1, 50)
    print("After append 50 to s1:", mgr.summary())

    # CoW fork
    s3 = mgr.fork(s1, "req-003-fork")
    print("After fork s1->s3:", mgr.summary())

    # 释放
    mgr.release(s2)
    print("After release s2:", mgr.summary())

    # 碎片率
    print(f"Fragmentation: {mgr.allocator.fragmentation():.2%}")
