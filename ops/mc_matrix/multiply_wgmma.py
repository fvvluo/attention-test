"""Hopper (SM90) BF16 WGMMA + TMA GEMM.

计算 C = A @ B，A[M,K]/B[K,N]/C[M,N] row-major。
输入在 gmem 以 BF16 存储（host 侧把 FP32 截断为 BF16），WGMMA 以 FP32 累加。
精度与 TF32 相当（BF16 尾数 7bit vs TF32 10bit），test.py 的 1e-2 阈值可过。

用 TMA(cp.async.bulk.tensor) 做 gmem->smem，PipelineTmaAsync 管理多级 mbarrier 流水；
warpgroup MMA(wgmma.mma_async) 直接从 smem 取操作数（descriptor），不占大量寄存器，
突破 warp-MMA 版本 255 寄存器 / 12% 占用率的瓶颈。

结构参照 NVIDIA CUTLASS CuTeDSL hopper dense_gemm 官方范例。
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import warpgroup, cpasync, OperandMajorMode
from cutlass.utils import SmemAllocator
import cutlass.utils.hopper_helpers as hh
from cutlass.utils.layout import LayoutEnum
from cutlass import pipeline


# CTA tile。WGMMA 指令 M 固定 64；单 warpgroup 起步。
BLOCK_M = 64
BLOCK_N = 128
MMA_INST_TILE_K = 1          # 一个 CTA k-tile 内的 wgmma-k 子块数（K = 16 * 1 = 16）
STAGES = 3
NUM_WG_M = 1
NUM_THREADS = NUM_WG_M * 128

ACC_DTYPE = cutlass.Float32
AB_DTYPE = cutlass.BFloat16


@cute.kernel
def gemm_kernel(
    tma_atom_a: cute.CopyAtom,
    tma_atom_b: cute.CopyAtom,
    mA: cute.Tensor,     # TMA coord tensor for A [M,K]
    mB: cute.Tensor,     # TMA coord tensor for B [N,K]
    mC: cute.Tensor,     # [M, N] f32
    sA_layout: cute.ComposedLayout,
    sB_layout: cute.ComposedLayout,
    tiled_mma: cute.TiledMma,
    tile_mnk: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)

    tile_m, tile_n, tile_k = tile_mnk

    # smem A/B：把 swizzle 移到指针，tensor 布局保持 affine（WGMMA fragment 要求）。
    smem = SmemAllocator()
    sA = smem.allocate_tensor(
        AB_DTYPE, cute.get_nonswizzle_portion(sA_layout), byte_alignment=1024,
        swizzle=cute.get_swizzle_portion(sA_layout))
    sB = smem.allocate_tensor(
        AB_DTYPE, cute.get_nonswizzle_portion(sB_layout), byte_alignment=1024,
        swizzle=cute.get_swizzle_portion(sB_layout))

    # ---- pipeline ----
    tma_bytes = cute.size_in_bytes(AB_DTYPE, cute.slice_(sA_layout, (None, None, 0))) + \
        cute.size_in_bytes(AB_DTYPE, cute.slice_(sB_layout, (None, None, 0)))
    mainloop_pipeline = pipeline.PipelineTmaAsync.create(
        num_stages=STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, NUM_THREADS // 32),
        tx_count=tma_bytes,
        tidx=tidx,
    )

    # gmem tiles per CTA
    gA = cute.local_tile(mA, (tile_m, tile_k), (bidx, None))   # [M,K,k]
    gB = cute.local_tile(mB, (tile_n, tile_k), (bidy, None))   # [N,K,k]
    gC = cute.local_tile(mC, (tile_m, tile_n), (bidx, bidy))

    # TMA partition (single CTA, no cluster)
    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a, 0, cute.make_layout(1),
        cute.group_modes(sA, 0, 2), cute.group_modes(gA, 0, 2))
    tBsB, tBgB = cpasync.tma_partition(
        tma_atom_b, 0, cute.make_layout(1),
        cute.group_modes(sB, 0, 2), cute.group_modes(gB, 0, 2))

    k_tile_cnt = cute.size(gA, mode=[2])

    thr_mma = tiled_mma.get_slice(tidx)
    tCsA = thr_mma.partition_A(sA)   # [MMA, MMA_M, MMA_K, STAGES]
    tCsB = thr_mma.partition_B(sB)
    tCgC = thr_mma.partition_C(gC)
    tCrA = thr_mma.make_fragment_A(tCsA)   # smem descriptor
    tCrB = thr_mma.make_fragment_B(tCsB)
    accumulators = thr_mma.make_fragment_C(tCgC)

    num_k_blocks = cute.size(tCrA, mode=[2])

    # ---- prologue: producer 预取 ----
    producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, STAGES)
    if warp_idx == 0:
        for _ in cutlass.range(STAGES - 1, unroll=1):
            if producer_state.count < k_tile_cnt:
                mainloop_pipeline.producer_acquire(producer_state)
                bar = mainloop_pipeline.producer_get_barrier(producer_state)
                cute.copy(tma_atom_a, tAgA[None, producer_state.count], tAsA[None, producer_state.index], tma_bar_ptr=bar)
                cute.copy(tma_atom_b, tBgB[None, producer_state.count], tBsB[None, producer_state.index], tma_bar_ptr=bar)
                mainloop_pipeline.producer_commit(producer_state)
                producer_state.advance()

    # ---- prologue MMAs (k_pipe_mmas = 1) ----
    k_pipe_mmas = 1
    read_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, STAGES)
    release_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, STAGES)

    tiled_mma.set(warpgroup.Field.ACCUMULATE, False)
    for _ in cutlass.range_constexpr(k_pipe_mmas):
        mainloop_pipeline.consumer_wait(read_state)
        cute.nvgpu.warpgroup.fence()
        for kb in cutlass.range(num_k_blocks, unroll_full=True):
            cute.gemm(
                tiled_mma, accumulators,
                tCrA[None, None, kb, read_state.index],
                tCrB[None, None, kb, read_state.index],
                accumulators,
            )
            tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
        cute.nvgpu.warpgroup.commit_group()
        read_state.advance()

    # ---- mainloop ----
    for k_tile in cutlass.range(k_pipe_mmas, k_tile_cnt, 1, unroll=1):
        mainloop_pipeline.consumer_wait(read_state)
        cute.nvgpu.warpgroup.fence()
        for kb in cutlass.range(num_k_blocks, unroll_full=True):
            cute.gemm(
                tiled_mma, accumulators,
                tCrA[None, None, kb, read_state.index],
                tCrB[None, None, kb, read_state.index],
                accumulators,
            )
        cute.nvgpu.warpgroup.commit_group()
        cute.nvgpu.warpgroup.wait_group(k_pipe_mmas)

        mainloop_pipeline.consumer_release(release_state)
        read_state.advance()
        release_state.advance()

        # producer 发下一个 tile
        if warp_idx == 0 and producer_state.count < k_tile_cnt:
            mainloop_pipeline.producer_acquire(producer_state)
            bar = mainloop_pipeline.producer_get_barrier(producer_state)
            cute.copy(tma_atom_a, tAgA[None, producer_state.count], tAsA[None, producer_state.index], tma_bar_ptr=bar)
            cute.copy(tma_atom_b, tBgB[None, producer_state.count], tBsB[None, producer_state.index], tma_bar_ptr=bar)
            mainloop_pipeline.producer_commit(producer_state)
            producer_state.advance()

    # flush 剩余 wgmma
    cute.nvgpu.warpgroup.wait_group(0)

    # ---- epilogue ----
    cute.autovec_copy(accumulators, tCgC)


@cute.jit
def matrix_mul(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    """a,b: BF16 [M,K]/[K,N] row-major; c: FP32 [M,N]."""
    # WGMMA 指令 M 固定 64
    # A[M,K] row-major → K 连续 → K-major。
    # B[K,N] row-major，viewed as [N,K] 后 N 连续 (stride=1)、K 带 stride=N → MN-major。
    tiled_mma = hh.make_trivial_tiled_mma(
        AB_DTYPE, AB_DTYPE,
        OperandMajorMode.K, OperandMajorMode.MN,
        ACC_DTYPE,
        (NUM_WG_M, 1, 1),
        (BLOCK_M // NUM_WG_M, BLOCK_N),
    )
    mma_inst_k = cute.size(tiled_mma.shape_mnk, mode=[2])   # 16
    tile_k = mma_inst_k * MMA_INST_TILE_K                   # 64
    mma_tiler = (BLOCK_M, BLOCK_N, tile_k)

    # B viewed as [N,K]
    b_nk = cute.make_tensor(b.iterator, cute.select(b.layout, mode=[1, 0]))

    sA_layout = hh.make_smem_layout_a(LayoutEnum.ROW_MAJOR, mma_tiler, AB_DTYPE, STAGES)
    # B viewed as [N,K] 里 N 连续 (stride=1) ⇒ MN-major，用 COL_MAJOR（其 is_k_major_b()
    # 为 False，构造 MN-major smem atom），与 tiled_mma 的 b_major_mode=MN 一致。
    sB_layout = hh.make_smem_layout_b(LayoutEnum.COL_MAJOR, mma_tiler, AB_DTYPE, STAGES)

    # TMA atom：传入 slice 掉 stage 的单级 smem 布局 + (tile_m/n, tile_k)
    tma_atom_a, mA_tma = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), a,
        cute.slice_(sA_layout, (None, None, 0)),
        (BLOCK_M, tile_k), num_multicast=1)
    tma_atom_b, mB_tma = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), b_nk,
        cute.slice_(sB_layout, (None, None, 0)),
        (BLOCK_N, tile_k), num_multicast=1)

    M = c.shape[0]
    N = c.shape[1]
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    gemm_kernel(
        tma_atom_a, tma_atom_b, mA_tma, mB_tma, c,
        sA_layout, sB_layout, tiled_mma,
        (BLOCK_M, BLOCK_N, tile_k),
    ).launch(
        grid=(grid_m, grid_n, 1),
        block=(NUM_THREADS, 1, 1),
    )
