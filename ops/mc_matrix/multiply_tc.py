"""TF32 Tensor Core GEMM (Ampere/Hopper SM80+).

计算 C = A @ B，A[M,K] row-major, B[K,N] row-major, C[M,N] row-major，均为 Float32。
内部把 A/B 截断为 TF32 喂给 warp-level MMA（mma.sync m16n8k8 f32.tf32.tf32.f32），
累加为 Float32，精度与 torch 的 TF32 matmul 一致。

先追求“正确 + 用上 Tensor Core”，暂假设 M/N/K 分别是 BLOCK_M/BLOCK_N/BLOCK_K 的整数倍。
"""

import cutlass
import cutlass.cute as cute
from cutlass.utils import SmemAllocator
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.nvgpu import warp as warp_mma


# CTA tile
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32

# MMA atom instruction shape (TF32): m16 n8 k8
MMA_M = 16
MMA_N = 8
MMA_K = 8

# atom layout: 2x2x1 warps -> 4 warps -> 128 threads
ATOM_M = 2
ATOM_N = 2
NUM_THREADS = ATOM_M * ATOM_N * 1 * 32  # 128

STAGES = 3

PAD = 4  # smem padding to avoid bank conflict on ldmatrix-free path


@cute.kernel
def gemm_kernel(
    mA: cute.Tensor,   # [M, K] f32, row-major
    mB: cute.Tensor,   # [K, N] f32, row-major
    mC: cute.Tensor,   # [M, N] f32, row-major
    sA_layout: cute.Layout,
    sB_layout: cute.Layout,
    tiled_copy_A: cute.TiledCopy,
    tiled_copy_B: cute.TiledCopy,
    tiled_mma: cute.TiledMma,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    # CTA tiles. A tile [BLOCK_M, BLOCK_K, k], B tile [BLOCK_N, BLOCK_K, k], C tile [BLOCK_M, BLOCK_N]
    # mB is [K, N]; we view it as [N, K] via transpose for tiling convenience.
    mB_nk = cute.make_tensor(mB.iterator, cute.select(mB.layout, mode=[1, 0]))  # [N, K]

    gA = cute.local_tile(mA, (BLOCK_M, BLOCK_K), (bidx, None))       # [BLOCK_M, BLOCK_K, k]
    gB = cute.local_tile(mB_nk, (BLOCK_N, BLOCK_K), (bidy, None))    # [BLOCK_N, BLOCK_K, k]
    gC = cute.local_tile(mC, (BLOCK_M, BLOCK_N), (bidx, bidy))       # [BLOCK_M, BLOCK_N]

    num_k = cute.size(gA, mode=[2])

    smem = SmemAllocator()
    sA = smem.allocate_tensor(cutlass.Float32, sA_layout, byte_alignment=16)  # [BLOCK_M, BLOCK_K, STAGES]
    sB = smem.allocate_tensor(cutlass.Float32, sB_layout, byte_alignment=16)  # [BLOCK_N, BLOCK_K, STAGES]

    thr_copy_A = tiled_copy_A.get_slice(tidx)
    thr_copy_B = tiled_copy_B.get_slice(tidx)
    tAgA = thr_copy_A.partition_S(gA)   # [CPY, CPY_M, CPY_K, k]
    tAsA = thr_copy_A.partition_D(sA)   # [CPY, CPY_M, CPY_K, STAGES]
    tBgB = thr_copy_B.partition_S(gB)
    tBsB = thr_copy_B.partition_D(sB)

    thr_mma = tiled_mma.get_slice(tidx)
    tCsA = thr_mma.partition_A(sA)      # [MMA, MMA_M, MMA_K, STAGES]
    tCsB = thr_mma.partition_B(sB)
    tCgC = thr_mma.partition_C(gC)      # [MMA, MMA_M, MMA_N]

    num_mma_k = cute.size(tCsA[None, None, None, 0], mode=[2])  # MMA_K 子块数

    # 双缓冲寄存器 fragment：额外一维 [2]，交替存放当前/下一个 k-block。
    acc = tiled_mma.make_fragment_C(tCgC)
    acc.fill(0.0)
    tCrA = tiled_mma.make_fragment_A(tCsA[None, None, 0, 0])  # 单个 k-block 大小
    tCrB = tiled_mma.make_fragment_B(tCsB[None, None, 0, 0])
    rA = [tCrA, tiled_mma.make_fragment_A(tCsA[None, None, 0, 0])]
    rB = [tCrB, tiled_mma.make_fragment_B(tCsB[None, None, 0, 0])]

    # ---- prologue: issue STAGES-1 async gmem->smem loads ----
    for s in cutlass.range_constexpr(STAGES - 1):
        if s < num_k:
            cute.copy(tiled_copy_A, tAgA[None, None, None, s], tAsA[None, None, None, s])
            cute.copy(tiled_copy_B, tBgB[None, None, None, s], tBsB[None, None, None, s])
        cute.arch.cp_async_commit_group()

    # 等第一个 tile 到位，预取第一个 k-block 的 smem->reg
    cute.arch.cp_async_wait_group(STAGES - 2)
    cute.arch.sync_threads()
    cute.autovec_copy(tCsA[None, None, 0, 0], rA[0])
    cute.autovec_copy(tCsB[None, None, 0, 0], rB[0])

    # ---- mainloop: 双重流水 (gmem->smem 的 STAGES 级 + smem->reg 的双缓冲) ----
    for k_tile in range(num_k):
        read_stage = k_tile % STAGES
        next_tile = k_tile + STAGES - 1
        write_stage = next_tile % STAGES

        nxt_stage = (k_tile + 1) % STAGES  # 下一个 gmem tile 所在 smem stage
        for mk in cutlass.range_constexpr(num_mma_k):
            cur = mk % 2
            nxt = (mk + 1) % 2

            # 本 tile 最后一个 k-block：切到下一个 gmem tile（需等其就绪）
            is_last_kblock = (mk == num_mma_k - 1)
            if is_last_kblock:
                if next_tile < num_k:
                    cute.arch.cp_async_wait_group(STAGES - 2)
                else:
                    cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()

            # 预取下一个 k-block 的 smem->reg（静态决定来源 stage / mk）
            nxt_mk = 0 if is_last_kblock else (mk + 1)
            src_stage = nxt_stage if is_last_kblock else read_stage
            if (not is_last_kblock) or (k_tile + 1 < num_k):
                cute.autovec_copy(tCsA[None, None, nxt_mk, src_stage], rA[nxt])
                cute.autovec_copy(tCsB[None, None, nxt_mk, src_stage], rB[nxt])

            # 本 tile 第一个 k-block 时发射下一个 gmem->smem 预取
            if mk == 0:
                if next_tile < num_k:
                    cute.copy(tiled_copy_A, tAgA[None, None, None, next_tile], tAsA[None, None, None, write_stage])
                    cute.copy(tiled_copy_B, tBgB[None, None, None, next_tile], tBsB[None, None, None, write_stage])
                cute.arch.cp_async_commit_group()

            cute.gemm(tiled_mma, acc, rA[cur], rB[cur], acc)

    # ---- epilogue: write acc -> gC ----
    cute.autovec_copy(acc, tCgC)


@cute.jit
def matrix_mul(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    M = c.shape[0]
    N = c.shape[1]
    K = a.shape[1]

    # 该原型不做边界 predication，要求尺寸整除 tile，否则最后一个 tile 会越界读写（静默错误）。
    assert M % BLOCK_M == 0, "M must be a multiple of BLOCK_M (128)"
    assert N % BLOCK_N == 0, "N must be a multiple of BLOCK_N (128)"
    assert K % BLOCK_K == 0, "K must be a multiple of BLOCK_K (32)"

    # smem layouts: [BLOCK_M, BLOCK_K, STAGES] with K contiguous + N/M padding
    sA_layout = cute.make_layout(
        (BLOCK_M, BLOCK_K, STAGES),
        stride=(BLOCK_K + PAD, 1, BLOCK_M * (BLOCK_K + PAD)),
    )
    # sB 逻辑 [BLOCK_N, BLOCK_K]，但 N 连续(stride=1)以匹配沿 N 的 128-bit 加载；
    # K 维 stride 带 padding 破 bank conflict。
    sB_layout = cute.make_layout(
        (BLOCK_N, BLOCK_K, STAGES),
        stride=(1, BLOCK_N + PAD, BLOCK_K * (BLOCK_N + PAD)),
    )

    # gmem async copy atom, 128-bit
    atom_async = cute.make_copy_atom(
        cpasync.CopyG2SOp(),
        cutlass.Float32,
        num_bits_per_copy=128,
    )
    # A[M,K] row-major: K 连续 -> 128-bit 沿 K，value=(1,4)，thread 沿 K 分 BLOCK_K/4 组
    tA_threads = cute.make_layout(
        (NUM_THREADS // (BLOCK_K // 4), BLOCK_K // 4),
        stride=(BLOCK_K // 4, 1),
    )
    tA_vals = cute.make_layout((1, 4))
    tiled_copy_A = cute.make_tiled_copy_tv(atom_async, tA_threads, tA_vals)

    # B 视图为 [N,K]，其中 N 是原始 [K,N] 的连续维(stride=1) -> col-major：
    # 128-bit 沿 N，value=(4,1)，thread 沿 N 分 BLOCK_N/4 组
    tB_threads = cute.make_layout(
        (BLOCK_N // 4, NUM_THREADS // (BLOCK_N // 4)),
        stride=(1, BLOCK_N // 4),
    )
    tB_vals = cute.make_layout((4, 1))
    tiled_copy_B = cute.make_tiled_copy_tv(atom_async, tB_threads, tB_vals)

    # tiled MMA (TF32)
    op = warp_mma.MmaTF32Op((MMA_M, MMA_N, MMA_K))
    atom_layout = cute.make_layout((ATOM_M, ATOM_N, 1))
    tiled_mma = cute.make_tiled_mma(op, atom_layout)

    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    gemm_kernel(
        a, b, c,
        sA_layout, sB_layout,
        tiled_copy_A, tiled_copy_B,
        tiled_mma,
    ).launch(
        grid=(grid_m, grid_n, 1),
        block=(NUM_THREADS, 1, 1),
    )
