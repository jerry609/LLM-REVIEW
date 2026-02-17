"""Tests for KV Cache modules"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.kv_cache.core import KVBlock, BlockAllocator, SequenceKVCache, PagedKVCacheManager


def test_kv_block():
    b = KVBlock(block_id=0, capacity=16)
    assert b.free_slots == 16
    assert not b.is_full
    b.filled = 16
    assert b.is_full
    assert b.free_slots == 0


def test_block_allocator_alloc_free():
    alloc = BlockAllocator(num_blocks=32, block_size=16)
    assert alloc.num_free == 32
    blk = alloc.allocate()
    assert blk is not None
    assert alloc.num_free == 31
    alloc.free(blk)
    assert alloc.num_free == 32


def test_block_allocator_oom():
    alloc = BlockAllocator(num_blocks=2, block_size=16)
    alloc.allocate()
    alloc.allocate()
    blk = alloc.allocate()
    assert blk is None, "Should return None when OOM"


def test_paged_kv_cache_manager_alloc():
    mgr = PagedKVCacheManager(num_blocks=16, block_size=4)
    seq = mgr.allocate_for_sequence(seq_id="s0", num_tokens=10)
    assert seq.num_tokens == 10
    assert len(seq.block_table) >= 3  # 10 tokens / 4 per block = 3 blocks


def test_paged_kv_cache_manager_append():
    mgr = PagedKVCacheManager(num_blocks=16, block_size=4)
    seq = mgr.allocate_for_sequence(seq_id="s0", num_tokens=4)
    assert seq.num_tokens == 4
    mgr.append_tokens(seq, 5)
    assert seq.num_tokens == 9


def test_paged_kv_cache_manager_release():
    mgr = PagedKVCacheManager(num_blocks=16, block_size=4)
    seq = mgr.allocate_for_sequence(seq_id="s0", num_tokens=10)
    free_before = mgr.allocator.num_free
    mgr.release(seq)
    assert mgr.allocator.num_free > free_before


def test_paged_kv_cache_fork():
    mgr = PagedKVCacheManager(num_blocks=32, block_size=4)
    seq1 = mgr.allocate_for_sequence(seq_id="s0", num_tokens=10)
    seq2 = mgr.fork(seq1, "s1")
    assert seq2.num_tokens == seq1.num_tokens
    # blocks are shared (ref_count > 1)
    for blk in seq1.block_table:
        assert blk.ref_count == 2


if __name__ == "__main__":
    test_kv_block()
    test_block_allocator_alloc_free()
    test_block_allocator_oom()
    test_paged_kv_cache_manager_alloc()
    test_paged_kv_cache_manager_append()
    test_paged_kv_cache_manager_release()
    test_paged_kv_cache_fork()
    print("All KV cache tests passed ✓")
