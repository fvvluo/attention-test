"""Tensor-Core 版 FlashAttention：QK^T 与 P@V 都走 warp 级 bf16 MMA，多 warp/CTA。

并行划分：
  - 一个 CTA = NUM_WARPS 个 warp，沿 query 行方向切：warp w 负责第 w 组 BLOCK_M 行 query。
    一个 CTA 共覆盖 CTA_M = NUM_WARPS * BLOCK_M 行 query 的一个 (batch,head) 切片。
  - K/V 每个 key tile 只加载一份到 smem，被 CTA 内所有 warp 复用（沿 head_dim/BLOCK_N 协作载入）。
  - 每个 warp 独立跑自己的 QK^T (TC) -> online-softmax -> P@V (TC)，
    S/P/O/running-stat 按 warp 分区（smem 加一维 [NUM_WARPS, ...]）。
  - grid = (ceil(q_len / CTA_M), BH)，block = NUM_WARPS * 32 线程。

数学与标量 tiled 版一致，仅把两次 matmul 换成 tensor core，并提升占用率。
张量约定：3D (BH, seq, head_dim)，输入 bf16，输出 fp32。head_dim 需为 MMA_K=16 的整数倍。
"""

import cutlass
import cutlass.cute as cute
from cutlass.utils import SmemAllocator
import cutlass.cute.nvgpu.warp as warp_mma


WARP_SIZE = 32
NUM_WARPS = 8
THREADS = NUM_WARPS * WARP_SIZE

BLOCK_M = 16          # 每个 warp 负责的 query 行数（= MMA_M）
BLOCK_N = 32          # 每个 key tile 的 key 数（= P@V 的累加维 K）
CTA_M = NUM_WARPS * BLOCK_M

MMA_M = 16
MMA_N = 8
MMA_K = 16

MAX_HEAD_DIM_TC = 256


@cute.kernel
def flash_attention_tc_kernel(
    q: cute.Tensor,          # [BH, seq, head_dim] bf16
    k: cute.Tensor,
    v: cute.Tensor,
    output: cute.Tensor,     # [BH, seq, head_dim] f32
    scale: cutlass.Float32,
    causal_offset: cutlass.Int32,
    sQ_layout: cute.Layout,   # [NUM_WARPS, BLOCK_M, head_dim] bf16
    sK_layout: cute.Layout,   # [BLOCK_N, head_dim] bf16 (共享)
    sVt_layout: cute.Layout,  # [head_dim, BLOCK_N] bf16 (共享, 转置)
    sS_layout: cute.Layout,   # [NUM_WARPS, BLOCK_M, BLOCK_N] f32
    sP_layout: cute.Layout,   # [NUM_WARPS, BLOCK_M, BLOCK_N] bf16
    sO_layout: cute.Layout,   # [NUM_WARPS, BLOCK_M, head_dim] f32 (仅 epilogue)
    sVec_layout: cute.Layout, # [NUM_WARPS, BLOCK_M] f32
    mma_qk: cute.TiledMma,
    mma_pv: cute.TiledMma,
):
    block_m, bh_idx, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    lane = tidx % WARP_SIZE
    warp_idx = tidx // WARP_SIZE

    num_queries = output.shape[1]
    num_keys = k.shape[1]
    head_dim = q.shape[2]

    cta_row_base = block_m * CTA_M
    warp_row_base = cta_row_base + warp_idx * BLOCK_M

    smem = SmemAllocator()
    sQ = smem.allocate_tensor(cutlass.BFloat16, sQ_layout, byte_alignment=16)
    sK = smem.allocate_tensor(cutlass.BFloat16, sK_layout, byte_alignment=16)
    sVt = smem.allocate_tensor(cutlass.BFloat16, sVt_layout, byte_alignment=16)
    sS = smem.allocate_tensor(cutlass.Float32, sS_layout, byte_alignment=16)
    sP = smem.allocate_tensor(cutlass.BFloat16, sP_layout, byte_alignment=16)
    # sOshape 只作 partition_C 的 layout 载体（[BLOCK_M, head_dim]），不真正读写，全 CTA 共用一份。
    sOshape = smem.allocate_tensor(cutlass.Float32, sO_layout, byte_alignment=16)
    sM = smem.allocate_tensor(cutlass.Float32, sVec_layout, byte_alignment=16)
    sL = smem.allocate_tensor(cutlass.Float32, sVec_layout, byte_alignment=16)
    sCorr = smem.allocate_tensor(cutlass.Float32, sVec_layout, byte_alignment=16)

    # 本 warp 的 smem 子视图
    sQ_w = sQ[warp_idx, None, None]
    sS_w = sS[warp_idx, None, None]
    sP_w = sP[warp_idx, None, None]

    # ---- 载入 Q（每 warp 载自己的 BLOCK_M 行）+ 初始化 running 统计量 ----
    for i in range(lane, BLOCK_M * head_dim, WARP_SIZE):
        r = i // head_dim
        c = i % head_dim
        qr = warp_row_base + r
        if qr < num_queries:
            sQ_w[r, c] = q[bh_idx, qr, c]
        else:
            sQ_w[r, c] = cutlass.BFloat16(0.0)
    for r in range(lane, BLOCK_M, WARP_SIZE):
        sM[warp_idx, r] = cutlass.Float32(float("-inf"))
        sL[warp_idx, r] = cutlass.Float32(0.0)
    cute.arch.sync_threads()

    # MMA 分区（每 warp 用 local lane，对自己的 smem 子块）
    thr_qk = mma_qk.get_slice(lane)
    tAsQ = thr_qk.partition_A(sQ_w)
    tCsS = thr_qk.partition_C(sS_w)
    num_k_qk = cute.size(tAsQ, mode=[2])
    rQ = mma_qk.make_fragment_A(tAsQ[None, None, 0])
    rK = mma_qk.make_fragment_B(mma_qk.get_slice(lane).partition_B(sK)[None, None, 0])

    thr_pv = mma_pv.get_slice(lane)
    tAsP = thr_pv.partition_A(sP_w)
    # 输出累加器常驻寄存器：partition_C 只用来定 fragment 形状（sOshape 仅作 layout 载体）
    tCacc = thr_pv.partition_C(sOshape)
    num_k_pv = cute.size(tAsP, mode=[2])
    rP = mma_pv.make_fragment_A(tAsP[None, None, 0])
    rV = mma_pv.make_fragment_B(thr_pv.partition_B(sVt)[None, None, 0])

    # 常驻寄存器的输出累加器 acc_o，跨 key tile 保持（避免 sO/sDO 两块大 smem 往返）
    acc_o = mma_pv.make_fragment_C(tCacc)
    acc_o.fill(0.0)
    n_acc = cute.size(acc_o)  # 每 lane 持有的 acc 元素数 = (head_dim/8)*4

    # m16n8 MMA C fragment 的行映射（探针实测）：
    #   元素 e 的组内序号 le=e%4，row = (lane//4) + 8*(le//2)
    # 每 lane 只涉及两行：row_lo = lane//4，row_hi = lane//4 + 8
    row_lo = lane // 4
    row_hi = row_lo + 8

    # K/B 分区在循环外取一次（sK/sVt 内容每 tile 变，但分区结构不变）
    tBsK = thr_qk.partition_B(sK)
    tBsV = thr_pv.partition_B(sVt)

    # causal 早退（CTA 内最大 query 行）
    max_q_row = cta_row_base + (CTA_M - 1)
    key_limit = max_q_row + causal_offset + 1
    if key_limit > num_keys:
        key_limit = num_keys
    num_key_tiles = (key_limit + BLOCK_N - 1) // BLOCK_N

    for tile in range(num_key_tiles):
        key_base = tile * BLOCK_N

        # 全 CTA 协作载入 K（正常）与 V（转置）到共享 smem
        for i in range(tidx, BLOCK_N * head_dim, THREADS):
            r = i // head_dim
            c = i % head_dim
            kr = key_base + r
            if kr < num_keys:
                sK[r, c] = k[bh_idx, kr, c]
                sVt[c, r] = v[bh_idx, kr, c]
            else:
                sK[r, c] = cutlass.BFloat16(0.0)
                sVt[c, r] = cutlass.BFloat16(0.0)
        cute.arch.sync_threads()

        # ---- S = Q @ K^T (TC) ----
        acc_s = mma_qk.make_fragment_C(tCsS)
        acc_s.fill(0.0)
        for mk in cutlass.range_constexpr(num_k_qk):
            cute.autovec_copy(tAsQ[None, None, mk], rQ)
            cute.autovec_copy(tBsK[None, None, mk], rK)
            cute.gemm(mma_qk, acc_s, rQ, rK, acc_s)
        cute.autovec_copy(acc_s, tCsS)
        cute.arch.sync_threads()

        # ---- online-softmax（本 warp 的行），lane 分担 BLOCK_M 行 ----
        for r in range(lane, BLOCK_M, WARP_SIZE):
            qr = warp_row_base + r
            q_abs = qr + causal_offset
            tile_max = cutlass.Float32(float("-inf"))
            for c in cutlass.range_constexpr(BLOCK_N):
                kr = key_base + c
                if (kr < num_keys) and (kr <= q_abs) and (qr < num_queries):
                    sc = sS_w[r, c] * scale
                    tile_max = cute.arch.fmax(tile_max, sc)

            old_m = sM[warp_idx, r]
            new_m = cute.arch.fmax(old_m, tile_max)
            correction = cute.exp(old_m - new_m, fastmath=False)
            run_l = sL[warp_idx, r] * correction
            for c in cutlass.range_constexpr(BLOCK_N):
                kr = key_base + c
                if (kr < num_keys) and (kr <= q_abs) and (qr < num_queries):
                    sc = sS_w[r, c] * scale
                    p = cute.exp(sc - new_m, fastmath=False)
                    run_l += p
                    sP_w[r, c] = cutlass.BFloat16(p)
                else:
                    sP_w[r, c] = cutlass.BFloat16(0.0)
            sM[warp_idx, r] = new_m
            sL[warp_idx, r] = run_l
            sCorr[warp_idx, r] = correction
        cute.arch.sync_threads()

        # ---- 先按行 correction 重缩放常驻寄存器 acc_o，再让 P@V 直接累加进去 ----
        corr_lo = sCorr[warp_idx, row_lo]
        corr_hi = sCorr[warp_idx, row_hi]
        for e in cutlass.range_constexpr(n_acc):
            le = e % 4
            if le < 2:
                acc_o[e] = acc_o[e] * corr_lo
            else:
                acc_o[e] = acc_o[e] * corr_hi

        # deltaO = P @ V (TC)，直接累加到 acc_o（C = acc_o + P@V）
        for mk in cutlass.range_constexpr(num_k_pv):
            cute.autovec_copy(tAsP[None, None, mk], rP)
            cute.autovec_copy(tBsV[None, None, mk], rV)
            cute.gemm(mma_pv, acc_o, rP, rV, acc_o)
        cute.arch.sync_threads()

    # ---- 归一化 + 直接把寄存器 acc_o 按 fragment 映射写回 global（无 smem 中转）----
    # m16n8 C fragment：元素 e 组内序号 le=e%4，所在 N-block nb=e//4；
    #   row = (lane//4) + 8*(le//2)， col = nb*8 + (lane%4)*2 + (le%2)
    inv_lo = cutlass.Float32(1.0) / sL[warp_idx, row_lo]
    inv_hi = cutlass.Float32(1.0) / sL[warp_idx, row_hi]
    col_base = (lane % 4) * 2
    for e in cutlass.range_constexpr(n_acc):
        le = e % 4          # Python int（constexpr 循环）
        nb = e // 4         # Python int
        col = nb * 8 + col_base + (le % 2)
        # le<2 / le>=2 在 trace 期即为 Python 常量，用它选择运行期标量（不进 staged 分支）
        row = row_lo if le < 2 else row_hi
        inv = inv_lo if le < 2 else inv_hi
        qr = warp_row_base + row
        val = acc_o[e] * inv
        if qr < num_queries:
            output[bh_idx, qr, col] = val


@cute.jit
def flash_attention_tc(q, k, v, output, scale, causal_offset):
    head_dim = q.shape[2]

    sQ_layout = cute.make_layout((NUM_WARPS, BLOCK_M, head_dim),
                                 stride=(BLOCK_M * head_dim, head_dim, 1))
    sK_layout = cute.make_layout((BLOCK_N, head_dim), stride=(head_dim, 1))
    sVt_layout = cute.make_layout((head_dim, BLOCK_N), stride=(BLOCK_N, 1))
    sS_layout = cute.make_layout((NUM_WARPS, BLOCK_M, BLOCK_N),
                                 stride=(BLOCK_M * BLOCK_N, BLOCK_N, 1))
    sP_layout = cute.make_layout((NUM_WARPS, BLOCK_M, BLOCK_N),
                                 stride=(BLOCK_M * BLOCK_N, BLOCK_N, 1))
    # sO 只需一份 [BLOCK_M, head_dim]，作 partition_C 的 layout 载体
    sO_layout = cute.make_layout((BLOCK_M, head_dim), stride=(head_dim, 1))
    sVec_layout = cute.make_layout((NUM_WARPS, BLOCK_M), stride=(BLOCK_M, 1))

    op = warp_mma.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (MMA_M, MMA_N, MMA_K))
    mma_qk = cute.make_tiled_mma(op, cute.make_layout((1, 1, 1)))
    mma_pv = cute.make_tiled_mma(op, cute.make_layout((1, 1, 1)))

    block_num = (output.shape[1] + CTA_M - 1) // CTA_M
    batch_heads = output.shape[0]

    flash_attention_tc_kernel(
        q, k, v, output, scale, causal_offset,
        sQ_layout, sK_layout, sVt_layout, sS_layout, sP_layout,
        sO_layout, sVec_layout,
        mma_qk, mma_pv,
    ).launch(
        block=(THREADS, 1, 1),
        grid=(block_num, batch_heads, 1),
    )
