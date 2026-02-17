"""
CUDA 概念 Python 模拟

用纯 Python/NumPy 模拟 CUDA 的核心概念：
1. Thread/Block/Grid 映射
2. Shared Memory tiling for GEMM
3. Memory coalescing 模拟
4. Warp divergence 影响

面试用：理解 GPU 编程模型的核心抽象
"""
import numpy as np
from typing import Tuple


# ============================================================
# 1. Thread-Block-Grid 映射
# ============================================================
def thread_block_mapping(data_size: int, block_size: int = 256) -> dict:
    """模拟 1D kernel launch 的 thread 到数据的映射"""
    grid_size = (data_size + block_size - 1) // block_size
    mapping = []
    for block_id in range(grid_size):
        for thread_id in range(block_size):
            global_id = block_id * block_size + thread_id
            if global_id < data_size:
                mapping.append({
                    "block": block_id,
                    "thread": thread_id,
                    "global_id": global_id,
                })
    return {
        "grid_size": grid_size,
        "block_size": block_size,
        "total_threads": grid_size * block_size,
        "active_threads": data_size,
        "mapping_sample": mapping[:8],
    }


# ============================================================
# 2. Vector Add kernel simulation
# ============================================================
def vector_add_kernel(a: np.ndarray, b: np.ndarray, block_size: int = 256) -> np.ndarray:
    """模拟 CUDA vector add kernel"""
    n = len(a)
    c = np.zeros_like(a)
    grid_size = (n + block_size - 1) // block_size

    # 模拟每个 thread 的工作
    for blockIdx in range(grid_size):
        for threadIdx in range(block_size):
            i = blockIdx * block_size + threadIdx
            if i < n:
                c[i] = a[i] + b[i]
    return c


# ============================================================
# 3. Tiled GEMM (shared memory simulation)
# ============================================================
def tiled_gemm(A: np.ndarray, B: np.ndarray, tile_size: int = 16) -> np.ndarray:
    """
    模拟 shared memory tiling 的矩阵乘法。
    
    标准 GEMM: C[i,j] = sum_k A[i,k] * B[k,j]
    Tiling: 将 A, B 分成 tile_size × tile_size 的小块，
           每次只加载一个 tile 到 shared memory。
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = np.zeros((M, N), dtype=A.dtype)

    # 模拟 thread block: 每个 block 计算 C 的一个 tile
    for block_row in range(0, M, tile_size):
        for block_col in range(0, N, tile_size):
            # 此 block 负责 C[block_row:block_row+T, block_col:block_col+T]
            c_tile = np.zeros((tile_size, tile_size), dtype=A.dtype)

            for k_tile in range(0, K, tile_size):
                # "Load to shared memory" - 每个 thread 加载一个元素
                a_shared = np.zeros((tile_size, tile_size), dtype=A.dtype)
                b_shared = np.zeros((tile_size, tile_size), dtype=A.dtype)

                for ti in range(tile_size):
                    for tj in range(tile_size):
                        ai, aj = block_row + ti, k_tile + tj
                        if ai < M and aj < K:
                            a_shared[ti, tj] = A[ai, aj]

                        bi, bj = k_tile + ti, block_col + tj
                        if bi < K and bj < N:
                            b_shared[ti, tj] = B[bi, bj]

                # __syncthreads() 模拟
                # "Compute using shared memory"
                c_tile += a_shared @ b_shared

            # Write back to global memory
            for ti in range(tile_size):
                for tj in range(tile_size):
                    ci, cj = block_row + ti, block_col + tj
                    if ci < M and cj < N:
                        C[ci, cj] = c_tile[ti, tj]

    return C


# ============================================================
# 4. Memory coalescing analysis
# ============================================================
def analyze_coalescing(access_pattern: str, n: int = 1024, warp_size: int = 32) -> dict:
    """
    分析不同内存访问模式的 coalescing 效率。
    
    Patterns:
    - "coalesced": thread i 访问 data[i] (连续)
    - "strided": thread i 访问 data[i * stride]
    - "random": thread i 访问 data[random_index]
    """
    rng = np.random.default_rng(42)
    n_warps = n // warp_size

    if access_pattern == "coalesced":
        addresses = np.arange(n)
    elif access_pattern == "strided":
        stride = 32  # worst case: bank conflict
        addresses = (np.arange(n) * stride) % n
    elif access_pattern == "random":
        addresses = rng.permutation(n)
    else:
        raise ValueError(f"Unknown pattern: {access_pattern}")

    # 分析每个 warp 的 cache line 利用率
    cache_line_size = 128  # bytes, 32 × 4 bytes
    elements_per_line = cache_line_size // 4  # assuming float32

    total_lines_accessed = 0
    for warp_id in range(n_warps):
        start = warp_id * warp_size
        warp_addrs = addresses[start:start + warp_size]
        # 每个地址属于哪个 cache line
        lines = set(addr // elements_per_line for addr in warp_addrs)
        total_lines_accessed += len(lines)

    # 理想情况：每个 warp 只访问 1 个 cache line (32 threads × 4B = 128B = 1 line)
    ideal_lines = n_warps * 1
    efficiency = ideal_lines / total_lines_accessed

    return {
        "pattern": access_pattern,
        "total_cache_lines": total_lines_accessed,
        "ideal_cache_lines": ideal_lines,
        "efficiency": efficiency,
        "transactions_per_warp": total_lines_accessed / n_warps,
    }


# ============================================================
# 5. Warp divergence
# ============================================================
def simulate_warp_divergence(data: np.ndarray, warp_size: int = 32) -> dict:
    """
    模拟 warp divergence 的影响。
    
    场景：if (data[tid] > threshold) { path_A } else { path_B }
    如果 warp 内的线程走不同路径 → 串行执行两个路径
    """
    threshold = np.median(data)
    n = len(data)
    n_warps = n // warp_size

    total_cycles_no_diverge = 0  # 假设没有 divergence
    total_cycles_with_diverge = 0  # 实际有 divergence

    cost_A = 10  # path A 的周期
    cost_B = 5   # path B 的周期

    for warp_id in range(n_warps):
        start = warp_id * warp_size
        warp_data = data[start:start + warp_size]
        n_A = np.sum(warp_data > threshold)
        n_B = warp_size - n_A

        # 无 divergence: 只执行一个路径
        total_cycles_no_diverge += max(cost_A, cost_B)

        # 有 divergence: 如果两个路径都有线程，则串行执行两个
        if n_A > 0 and n_B > 0:
            total_cycles_with_diverge += cost_A + cost_B  # 串行！
        elif n_A > 0:
            total_cycles_with_diverge += cost_A
        else:
            total_cycles_with_diverge += cost_B

    return {
        "n_warps": n_warps,
        "cycles_ideal": total_cycles_no_diverge,
        "cycles_actual": total_cycles_with_diverge,
        "overhead": total_cycles_with_diverge / max(total_cycles_no_diverge, 1),
    }


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=== 1. Thread-Block Mapping ===")
    m = thread_block_mapping(1000, block_size=256)
    print(f"  Grid: {m['grid_size']} blocks × {m['block_size']} threads = {m['total_threads']} total")
    print(f"  Active: {m['active_threads']}, Wasted: {m['total_threads'] - m['active_threads']}")

    print("\n=== 2. Vector Add ===")
    a = np.random.randn(1024).astype(np.float32)
    b = np.random.randn(1024).astype(np.float32)
    c = vector_add_kernel(a, b)
    np.testing.assert_allclose(c, a + b, atol=1e-6)
    print("  Correctness: PASS ✓")

    print("\n=== 3. Tiled GEMM ===")
    A = np.random.randn(64, 48).astype(np.float32)
    B = np.random.randn(48, 32).astype(np.float32)
    C = tiled_gemm(A, B, tile_size=16)
    C_ref = A @ B
    err = np.max(np.abs(C - C_ref))
    print(f"  Max error vs np.matmul: {err:.2e}")
    print(f"  Correctness: {'PASS ✓' if err < 1e-4 else 'FAIL'}")

    print("\n=== 4. Memory Coalescing ===")
    for pat in ["coalesced", "strided", "random"]:
        r = analyze_coalescing(pat)
        print(f"  {pat:>10}: efficiency={r['efficiency']:.1%}, lines/warp={r['transactions_per_warp']:.1f}")

    print("\n=== 5. Warp Divergence ===")
    # 均匀数据 → 高 divergence
    data_uniform = np.random.randn(1024).astype(np.float32)
    r1 = simulate_warp_divergence(data_uniform)
    # 排序数据 → 低 divergence
    data_sorted = np.sort(data_uniform)
    r2 = simulate_warp_divergence(data_sorted)
    print(f"  Uniform data: overhead={r1['overhead']:.2f}x")
    print(f"  Sorted data:  overhead={r2['overhead']:.2f}x")
    print("  → 排序后 warp 内线程走同一路径，减少 divergence")
