# Copyright (c) 2026. FlashAttention decode-optimized kernel in CuTe DSL.
# SPDX-License-Identifier: BSD-3-Clause
#
# A *decode*-specialized FlashAttention forward kernel written in the CUTLASS
# CuTe DSL, targeting NVIDIA Hopper (sm90 / H20).
#
# ---------------------------------------------------------------------------
# Why decode is different from prefill
# ---------------------------------------------------------------------------
# In autoregressive decoding, each step processes exactly ONE new query token
# per (batch, head).  The Q tensor is therefore tiny (seqlen_q == 1) while the
# KV cache is large (kv_len can be tens of thousands).  The arithmetic
# intensity collapses: for every element of K/V read from HBM we do only O(1)
# FLOPs.  Decode is thus *memory-bandwidth bound* — the figure of merit is how
# fast we can stream the KV cache out of HBM.  On H20 the HBM3 peak is
# ~4.0 TB/s, so the 3.5 TB/s target is ~87% of peak.
#
# A naive FlashAttention-2 kernel launches a grid of only (batch * num_head_kv)
# CTAs.  For a single decode step that is far too few blocks to fill 78 SMs, so
# the GPU sits idle and bandwidth is wasted.  The fix is the well-known
# "Flash-Decoding" split-KV strategy (Dao et al., 2023):
#
#   Pass 1 (this file: `_flash_decode_split`):
#       Partition the KV sequence into `num_splits` chunks.  Grid becomes
#       (num_splits, num_head_kv, batch) so every SM stays busy.  Each CTA
#       reads its KV chunk exactly once, runs a *local* online softmax, and
#       writes a partial output O_partial (fp32) plus its local log-sum-exp.
#
#   Pass 2 (this file: `_flash_decode_combine`):
#       A cheap reduction kernel that rescales and sums the `num_splits`
#       partials per (batch, head) using the LSE values, producing the final O.
#
# GQA head-packing:
#   Modern models use grouped-query attention: `q_per_kv` query heads share one
#   KV head.  We pack that whole query group into the M dimension of the MMA
#   (M = q_per_kv, padded to the MMA atom).  This means ONE stream of the KV
#   cache feeds `q_per_kv` query heads at once — the KV read cost is amortised
#   across the group, keeping the kernel firmly memory-bound and the tensor
#   cores usefully occupied.
#
# The KV cache is read exactly once end-to-end, so achieved bandwidth
# = (K bytes + V bytes) / kernel_time, which is what we report.
# ---------------------------------------------------------------------------

import argparse
import math
from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import from_dlpack

LOG2_E = 1.4426950408889634074


class FlashAttentionDecode:
    """Decode-optimized FlashAttention (Flash-Decoding, split-KV) for sm90.

    Tensor layouts (all row-major / contiguous in head_dim):
      Q : (batch, num_head_q,          head_dim)          seqlen_q == 1
      K : (batch, num_head_kv, kv_len, head_dim)
      V : (batch, num_head_kv, kv_len, head_dim)
      O : (batch, num_head_q,          head_dim)

    Scratch (allocated by host wrapper):
      O_partial : (batch, num_head_q, num_splits, head_dim)  fp32
      LSE       : (batch, num_head_q, num_splits)            fp32
    """

    def __init__(
        self,
        head_dim: int,
        q_per_kv: int,
        n_block_size: int = 64,
        num_threads: int = 128,
        num_stages: int = 3,
    ):
        self._head_dim = head_dim
        # Pad head_dim to a multiple of 32 for the MMA K dimension.
        self._head_dim_padded = (head_dim + 31) // 32 * 32
        self._q_per_kv = q_per_kv
        # M tile: the whole GQA query group, padded up to the 16-row MMA atom.
        self._m_block_size = max(16, (q_per_kv + 15) // 16 * 16)
        self._n_block_size = n_block_size
        self._num_threads = num_threads
        # Software-pipeline depth for K/V async loads.
        self._num_stages = num_stages
        # min CTAs/SM hint (nvvm.minctasm) caps registers/thread so the kernel
        # is not register-limited (128 regs -> 16-block register limit; the
        # binding limit becomes shared memory at 12 blocks). Empirically 16 is
        # the sweet spot; higher (18+) forces register spilling and collapses.
        import os as _os
        self._min_blocks_per_mp = int(_os.environ.get("DECODE_MIN_CTA", "0"))
        # Swizzle atom width for K/V smem. 64 (128-bit swizzle) avoids all
        # ldmatrix bank conflicts (empirically faster than the smaller-smem
        # 32-wide swizzle, whose bank conflicts outweigh the occupancy gain).
        self._smem_k_block_size = 64 if self._head_dim_padded % 64 == 0 else 32
        self.cta_sync_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=1, num_threads=num_threads
        )

    @staticmethod
    def can_implement(dtype, head_dim, num_threads) -> bool:
        if dtype not in (cutlass.Float16, cutlass.BFloat16):
            return False
        if head_dim % 32 != 0:
            return False
        if num_threads % 32 != 0:
            return False
        return True

    # =====================================================================
    # Host entry: configure smem/copy/mma and launch both passes.
    # =====================================================================
    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,          # (b, hq, d)
        mK: cute.Tensor,          # (b, hkv, kv, d)
        mV: cute.Tensor,          # (b, hkv, kv, d)
        mO: cute.Tensor,          # (b, hq, d)
        mOpartial: cute.Tensor,   # (b, hq, splits, d)  fp32
        mLSE: cute.Tensor,        # (b, hq, splits)      fp32
        softmax_scale: cutlass.Float32,
        num_splits: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        self._dtype = mQ.element_type

        # ---------------- shared memory layouts (swizzled) ----------------
        # Allow overriding the swizzle atom width to trade a little ldmatrix
        # bank-conflict tolerance for lower smem footprint (higher occupancy),
        # which matters for this bandwidth-bound decode kernel.
        smem_k_block_size = self._smem_k_block_size
        swizzle_bits = 3 if smem_k_block_size == 64 else 2
        s_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3),
            0,
            cute.make_layout((8, smem_k_block_size), stride=(smem_k_block_size, 1)),
        )
        # Q is loaded once and re-read every KV block via ldmatrix; it uses the
        # same swizzled atom as K/V so the ldmatrix path is conflict-free.
        sQ_layout = cute.tile_to_shape(
            s_layout_atom, (self._m_block_size, self._head_dim_padded), (0, 1)
        )
        # Multi-stage (pipelined) K/V smem, PER WARP. Each warp owns num_stages
        # buffers so it can independently software-pipeline the async loads of
        # its own KV sub-slice. Loads for future stages are issued ahead of
        # compute so HBM traffic overlaps MMA/softmax -> saturates bandwidth.
        # Layout modes: (n_block, head_dim, stage, warp).
        num_stages = cutlass.const_expr(self._num_stages)
        num_warps_layout = cutlass.const_expr(self._num_threads // 32)
        sKV_layout = cute.tile_to_shape(
            s_layout_atom,
            (
                self._n_block_size,
                self._head_dim_padded,
                num_stages * num_warps_layout,
            ),
            (0, 1, 2),
        )

        # Cross-warp reduction scratch. Because the MMA tiles warps over the N
        # (KV) dimension, each warp performs an independent online softmax over
        # a DISJOINT subset of KV columns. Their partial (max, sum, O) results
        # are combined at the end via smem, using the same LSE-combine math as
        # the split-combine pass. Layout: per warp we store m_block partial
        # maxes, m_block partial sums, and m_block*head_dim partial O values.
        num_warps_c = cutlass.const_expr(self._num_threads // 32)
        # For a single warp there is no cross-warp reduction, so keep the scratch
        # buffers at 1 element (avoids ~8KB of wasted smem that would otherwise
        # cut occupancy from ~11 to 7 blocks/SM).
        red_o_elems = cutlass.const_expr(
            (num_warps_c * self._m_block_size * self._head_dim_padded)
            if num_warps_c > 1 else 1
        )
        red_ms_elems = cutlass.const_expr(
            (num_warps_c * self._m_block_size) if num_warps_c > 1 else 1
        )

        @cute.struct
        class SharedStorage:
            # NOTE: no dedicated sQ buffer. Q (16 x head_dim) has the SAME
            # swizzled layout as one K stage, so we transiently load Q into
            # sK stage 0, ldmatrix it into registers once, then let the K
            # pipeline overwrite that stage. This frees ~4KB of smem, letting
            # more CTAs co-reside per SM (occupancy-bound, memory-bound kernel).
            sK: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024
            ]
            sRedO: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, red_o_elems], 128
            ]
            sRedMax: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, red_ms_elems], 128
            ]
            sRedSum: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, red_ms_elems], 128
            ]

        # ---------------- gmem tiled copies ----------------
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self._dtype.width
        # GLOBAL cache mode is fastest here: adjacent CTAs (splits/heads) share
        # some L2 residency, so evict-first STREAMING actually hurt (~2.75 vs
        # 3.37 TB/s). Keep normal cached loads.
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self._dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        # Per-warp (32-thread) tiled copy: each warp's 32 lanes cooperatively
        # load one (n_block, head_dim) KV tile into its own smem region.
        tQKV_shape_dim_1 = s_layout_atom.outer.shape[1] // async_copy_elems
        tQKV_layout = cute.make_layout(
            (32 // tQKV_shape_dim_1, tQKV_shape_dim_1),
            stride=(tQKV_shape_dim_1, 1),
        )
        vQKV_layout = cute.make_layout((1, async_copy_elems))
        gmem_tiled_copy_QKV = cute.make_tiled_copy_tv(
            atom_async_copy, tQKV_layout, vQKV_layout
        )

        # ---------------- tiled mma (16x8x16 fp16/bf16) ----------------
        # DECODE-CRITICAL: use a SINGLE-warp MMA atom (1,1,1). Each of the CTA's
        # `num_warps` warps then runs a fully independent online-softmax over a
        # DISJOINT slice of this split's KV blocks (warp-parallel sub-splits),
        # producing its own (O, max, sum). This keeps every warp's math identical
        # to the verified single-warp kernel (no cross-warp GEMM reduction), and
        # multiplies achieved occupancy by num_warps -> the memory pipeline gets
        # enough warps to hide HBM latency. The num_warps partials are merged at
        # the end via smem using the same LSE-combine as the split-combine pass.
        num_warps = cutlass.const_expr(self._num_threads // 32)
        # Single-warp MMA: 32 lanes, M=16, N=16, K=16. Each warp runs this on its
        # own KV sub-slice completely independently.
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self._dtype, cutlass.Float32, (16, 8, 16)),
            (1, 1, 1),
            permutation_mnk=(16, 16, 16),
        )

        # ---------------- Pass 1 grid: (split, head_kv, batch) ----------------
        num_head_kv = cute.size(mK.shape[1])
        batch = cute.size(mK.shape[0])
        grid_split = (num_splits, num_head_kv, batch)

        softmax_scale_log2 = softmax_scale * LOG2_E

        self._split_kernel(
            mQ, mK, mV, mOpartial, mLSE,
            softmax_scale_log2,
            sQ_layout, sKV_layout,
            gmem_tiled_copy_QKV, tiled_mma,
            SharedStorage, num_splits,
        ).launch(
            grid=grid_split, block=[self._num_threads, 1, 1], stream=stream,
            # Force the register allocator to fit >= min_blocks_per_mp CTAs per
            # SM (nvvm.minctasm). This caps registers/thread so occupancy is not
            # register-limited -> more resident warps -> higher HBM parallelism.
            min_blocks_per_mp=self._min_blocks_per_mp,
        )

        # ---------------- Pass 2 grid: one CTA per (batch, head_q) ----------------
        num_head_q = cute.size(mQ.shape[1])
        grid_combine = (num_head_q, batch, 1)
        self._combine_kernel(
            mOpartial, mLSE, mO, num_splits
        ).launch(
            grid=grid_combine, block=[self._head_dim_padded, 1, 1], stream=stream
        )

    # =====================================================================
    # Pass 1: split-KV local attention -> partial O + LSE
    # =====================================================================
    @cute.kernel
    def _split_kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mOpartial: cute.Tensor,
        mLSE: cute.Tensor,
        softmax_scale_log2: cutlass.Float32,
        sQ_layout: cute.ComposedLayout,
        sKV_layout: cute.ComposedLayout,
        gmem_tiled_copy_QKV: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
        num_splits: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        split_idx, head_kv, batch = cute.arch.block_idx()

        kv_len = mK.shape[2]
        q_per_kv = cutlass.const_expr(self._q_per_kv)

        # ---- KV range for this split ----
        n_block_total = cute.ceil_div(kv_len, self._n_block_size)
        # ceil-divide blocks across splits so each split owns a contiguous range
        blocks_per_split = cute.ceil_div(n_block_total, num_splits)
        n_block_start = split_idx * blocks_per_split
        n_block_end = cutlass.min(n_block_start + blocks_per_split, n_block_total)

        # first query head of this GQA group
        head_q_base = head_kv * q_per_kv

        # ---- global tiles ----
        # Q group is loaded element-wise into smem by _load_q_group (rows map to
        # heads head_q_base + r). K/V are tiled over the KV-block dimension.
        gK = cute.local_tile(
            mK[batch, head_kv, None, None],
            (self._n_block_size, self._head_dim_padded),
            (None, 0),
        )  # (n_block_size, d, n_block)
        gV = cute.local_tile(
            mV[batch, head_kv, None, None],
            (self._n_block_size, self._head_dim_padded),
            (None, 0),
        )

        num_warps = cutlass.const_expr(self._num_threads // 32)
        num_stages = cutlass.const_expr(self._num_stages)
        warp_id = tidx // 32
        lane = tidx % 32

        # ---- smem ----
        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sK_all = storage.sK.get_tensor(sKV_layout)   # (n_block, d, stage*warp)
        sV_all = storage.sV.get_tensor(sKV_layout)
        # Q shares stage 0 of sK's smem (same swizzled layout). Q is loaded here,
        # ldmatrix'd to registers once, then the K pipeline overwrites stage 0.
        sQ = sK_all[None, None, 0]
        # cross-warp reduction scratch (flat fp32 memranges); 1 elem if 1 warp
        _ro = cutlass.const_expr(
            (num_warps * self._m_block_size * self._head_dim_padded)
            if num_warps > 1 else 1
        )
        _rm = cutlass.const_expr(
            (num_warps * self._m_block_size) if num_warps > 1 else 1
        )
        sRedO = storage.sRedO.get_tensor(cute.make_layout(_ro))
        sRedMax = storage.sRedMax.get_tensor(cute.make_layout(_rm))
        sRedSum = storage.sRedSum.get_tensor(cute.make_layout(_rm))

        # This warp's private K/V staging region (num_stages consecutive stages).
        stage_base = warp_id * num_stages
        # Transposed V view (d, n_block, stage*warp) for the O = P@V mma.
        sVt_all = cute.composition(
            sV_all,
            cute.make_layout(
                (
                    self._head_dim_padded,
                    self._n_block_size,
                    num_stages * num_warps,
                ),
                stride=(
                    self._n_block_size,
                    1,
                    self._n_block_size * self._head_dim_padded,
                ),
            ),
        )

        # ---- load Q group into smem (shared by all warps) ----
        self._load_q_group(mQ, sQ, batch, head_q_base, q_per_kv, tidx)

        # per-warp gmem/smem copy partitions (32-thread tiled copy, indexed by lane)
        gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_slice(lane)
        tKgK = gmem_thr_copy_QKV.partition_S(gK)     # (CPY,N,K, n_block)
        tVgV = gmem_thr_copy_QKV.partition_S(gV)
        tKsK = gmem_thr_copy_QKV.partition_D(sK_all) # (CPY,N,K, stage*warp)
        tVsV = gmem_thr_copy_QKV.partition_D(sV_all)

        # ---- mma partitions (single-warp, indexed by lane) ----
        thr_mma = tiled_mma.get_slice(lane)
        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
        tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK_all[None, None, 0]))
        tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt_all[None, None, 0]))
        acc_shape_O = thr_mma.partition_shape_C(
            (self._m_block_size, self._head_dim_padded)
        )
        acc_O = cute.make_rmem_tensor(acc_shape_O, cutlass.Float32)
        acc_O.fill(0.0)

        # ---- smem->rmem copy atoms ----
        smem_copy_atom_Q = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self._dtype
        )
        smem_copy_atom_V = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self._dtype
        )
        smem_tiled_copy_Q = cute.make_tiled_copy_A(smem_copy_atom_Q, tiled_mma)
        smem_tiled_copy_K = cute.make_tiled_copy_B(smem_copy_atom_Q, tiled_mma)
        smem_tiled_copy_V = cute.make_tiled_copy_B(smem_copy_atom_V, tiled_mma)
        smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(lane)
        smem_thr_copy_K = smem_tiled_copy_K.get_slice(lane)
        smem_thr_copy_V = smem_tiled_copy_V.get_slice(lane)
        tSsQ = smem_thr_copy_Q.partition_S(sQ)
        tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
        tSsK = smem_thr_copy_K.partition_S(sK_all)   # (CPY,M,K, stage*warp)
        tSrK_copy_view = smem_thr_copy_K.retile(tSrK)
        tOsVt = smem_thr_copy_V.partition_S(sVt_all)
        tOrVt_copy_view = smem_thr_copy_V.retile(tOrVt)

        # Q is constant across all KV blocks -> load the whole A-fragment from
        # smem into registers ONCE here, so the hot KV loop skips the per-block
        # Q ldmatrix. Q lives in sK stage 0; barrier ensures the cooperative
        # Q store (all lanes) is visible before we ldmatrix it.
        self.cta_sync_barrier.arrive_and_wait()
        for k in cutlass.range_constexpr(cute.size(tSsQ.shape[2])):
            cute.copy(smem_tiled_copy_Q, tSsQ[None, None, k],
                      tSrQ_copy_view[None, None, k])
        # Barrier so every lane has finished reading Q out of sK[0] before the
        # K pipeline (below) overwrites stage 0 with the first K block.
        self.cta_sync_barrier.arrive_and_wait()

        # ---- softmax running stats (per row owned by this lane) ----
        row_max = cute.make_rmem_tensor(
            (acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32
        )
        row_sum = cute.make_rmem_tensor(
            (acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32
        )
        row_max.fill(-cutlass.Float32.inf)
        row_sum.fill(0.0)

        # ---- per-warp KV range: interleave blocks across warps for balance ----
        # warp w processes blocks n_block_start + w, + w+num_warps, ...
        has_work = (n_block_start + warp_id) < n_block_end

        if has_work:
            # local iteration space for this warp
            # blocks: [n_block_start+warp_id : n_block_end : num_warps]
            # ---- prologue: issue first (num_stages-1) loads for this warp ----
            for s in cutlass.range_constexpr(num_stages - 1):
                blk = n_block_start + warp_id + s * num_warps
                if blk < n_block_end:
                    dst = stage_base + s
                    cute.copy(
                        gmem_tiled_copy_QKV,
                        tKgK[None, None, None, blk], tKsK[None, None, None, dst],
                    )
                    cute.copy(
                        gmem_tiled_copy_QKV,
                        tVgV[None, None, None, blk], tVsV[None, None, None, dst],
                    )
                cute.arch.cp_async_commit_group()

            read_stage = 0
            write_stage = (num_stages - 1) % num_stages
            cur_block = n_block_start + warp_id
            for i in cutlass.range(0, n_block_end, 1):
                if cur_block < n_block_end:
                    cute.arch.cp_async_wait_group(
                        num_stages - 2 if num_stages > 1 else 0
                    )
                    cute.arch.sync_warp()

                    prefetch_block = cur_block + (num_stages - 1) * num_warps
                    if prefetch_block < n_block_end:
                        dst = stage_base + write_stage
                        cute.copy(
                            gmem_tiled_copy_QKV,
                            tKgK[None, None, None, prefetch_block],
                            tKsK[None, None, None, dst],
                        )
                        cute.copy(
                            gmem_tiled_copy_QKV,
                            tVgV[None, None, None, prefetch_block],
                            tVsV[None, None, None, dst],
                        )
                    cute.arch.cp_async_commit_group()

                    self._compute_one_n_block(
                        thr_mma, tiled_mma,
                        tSrQ, tSrK, tOrVt, acc_O,
                        smem_tiled_copy_Q, smem_tiled_copy_K, smem_tiled_copy_V,
                        tSsQ, tSrQ_copy_view, tSsK, tSrK_copy_view,
                        tOsVt, tOrVt_copy_view,
                        row_max, row_sum, softmax_scale_log2,
                        cur_block, n_block_end, kv_len,
                        stage_base + read_stage,
                    )
                    read_stage = (read_stage + 1) % num_stages
                    write_stage = (write_stage + 1) % num_stages
                    cur_block = cur_block + num_warps

        # ---- write partial O + LSE for this split ----
        self._write_partials(
            acc_O, row_max, row_sum, softmax_scale_log2,
            mOpartial, mLSE, batch, head_q_base, split_idx, q_per_kv,
            thr_mma, tidx, has_work,
            sRedO, sRedMax, sRedSum,
        )

    # ------- cooperative element-wise Q group load into smem -------
    @cute.jit
    def _load_q_group(self, mQ, sQ, batch, head_q_base, q_per_kv, tidx):
        # sQ is (m_block_size, head_dim_padded). Fill rows [0, q_per_kv) from
        # mQ[batch, head_q_base + r, :], rest with 0.
        m = self._m_block_size
        d = self._head_dim_padded
        hd = mQ.shape[2]
        total = m * d
        i = tidx
        while i < total:
            r = i // d
            c = i % d
            in_bounds = r < q_per_kv and c < hd
            r_safe = r if r < q_per_kv else 0
            c_safe = c if c < hd else 0
            loaded = mQ[batch, head_q_base + r_safe, c_safe]
            zero = self._dtype(0.0)
            val = loaded if in_bounds else zero
            sQ[r, c] = val
            i += self._num_threads

    # =====================================================================
    # One KV block: S=QK^T, online softmax, O += P@V
    # =====================================================================
    @cute.jit
    def _compute_one_n_block(
        self,
        thr_mma, tiled_mma,
        tSrQ, tSrK, tOrVt, acc_O,
        smem_tiled_copy_Q, smem_tiled_copy_K, smem_tiled_copy_V,
        tSsQ, tSrQ_copy_view, tSsK, tSrK_copy_view,
        tOsVt, tOrVt_copy_view,
        row_max, row_sum, softmax_scale_log2,
        n_block, n_block_end, kv_len, read_stage,
    ):
        # Preconditions (guaranteed by the per-warp pipeline driver):
        #   * K and V for this block are resident in this warp's smem stage
        #     `read_stage` (cp.async wait + warp sync happened before the call),
        #     so we consume them directly with no further waits.
        acc_shape_S = thr_mma.partition_shape_C(
            (self._m_block_size, self._n_block_size)
        )
        acc_S = cute.make_rmem_tensor(acc_shape_S, cutlass.Float32)
        acc_S.fill(0.0)

        # stage-local smem views
        tSsK_s = tSsK[None, None, None, read_stage]
        tOsVt_s = tOsVt[None, None, None, read_stage]

        # S = Q @ K^T  (Q already resident in tSrQ registers; load only K)
        cute.copy(smem_tiled_copy_K, tSsK_s[None, None, 0], tSrK_copy_view[None, None, 0])
        for k in cutlass.range_constexpr(cute.size(tSsK_s.shape[2])):
            k_next = (k + 1) % cute.size(tSsK_s.shape[2])
            cute.copy(
                smem_tiled_copy_K, tSsK_s[None, None, k_next],
                tSrK_copy_view[None, None, k_next],
            )
            cute.gemm(tiled_mma, acc_S, tSrQ[None, None, k], tSrK[None, None, k], acc_S)

        # ---- mask KV columns beyond kv_len (last block) via coordinate tensor ----
        acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
        is_last = n_block == n_block_end - 1
        if is_last:
            cS = cute.make_identity_tensor(
                (self._m_block_size, self._n_block_size)
            )
            tScS = thr_mma.partition_C(cS)
            tScS_mn = self._make_acc_tensor_mn_view(tScS)
            n_base = n_block * self._n_block_size
            for c in cutlass.range_constexpr(cute.size(acc_S_mn.shape[1])):
                col_g = n_base + tScS_mn[0, c][1]
                if col_g >= kv_len:
                    for r in cutlass.range_constexpr(cute.size(acc_S_mn.shape[0])):
                        acc_S_mn[r, c] = -cutlass.Float32.inf

        # ---- online softmax ----
        self._softmax_rescale(
            acc_S, acc_O, row_max, row_sum, softmax_scale_log2
        )

        # P (fp16) for the O = P@V gemm
        rP = cute.make_fragment_like(acc_S, self._dtype)
        rP.store(acc_S.load().to(self._dtype))
        rP_layout_divided = cute.logical_divide(rP.layout, (None, None, 2))
        rP_mma_view = cute.make_layout(
            (
                (rP_layout_divided.shape[0], rP_layout_divided.shape[2][0]),
                rP_layout_divided.shape[1],
                rP_layout_divided.shape[2][1],
            ),
            stride=(
                (rP_layout_divided.stride[0], rP_layout_divided.stride[2][0]),
                rP_layout_divided.stride[1],
                rP_layout_divided.stride[2][1],
            ),
        )
        tOrS = cute.make_tensor(rP.iterator, rP_mma_view)

        cute.copy(smem_tiled_copy_V, tOsVt_s[None, None, 0], tOrVt_copy_view[None, None, 0])
        for k in cutlass.range_constexpr(cute.size(tOrS.shape[2])):
            k_next = (k + 1) % cute.size(tOrS.shape[2])
            cute.copy(
                smem_tiled_copy_V, tOsVt_s[None, None, k_next],
                tOrVt_copy_view[None, None, k_next],
            )
            cute.gemm(tiled_mma, acc_O, tOrS[None, None, k], tOrVt[None, None, k], acc_O)

    # =====================================================================
    # Online softmax rescale of acc_O and update of row_max / row_sum
    # =====================================================================
    @cute.jit
    def _softmax_rescale(self, acc_S, acc_O, row_max, row_sum, scale_log2):
        acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
        acc_O_mn = self._make_acc_tensor_mn_view(acc_O)

        row_max_prev = cute.make_fragment_like(row_max, cutlass.Float32)
        cute.basic_copy(row_max, row_max_prev)

        for r in cutlass.range_constexpr(cute.size(row_max)):
            acc_S_row = acc_S_mn[r, None].load()
            cur = acc_S_row.reduce(cute.ReductionOp.MAX, -cutlass.Float32.inf, 0)
            cur = self._threadquad_reduce_max(cur)
            prev = row_max_prev[r]
            new_max = cute.arch.fmax(prev, cur)

            exp_row = cute.math.exp2(
                acc_S_row * scale_log2 - new_max * scale_log2, fastmath=True
            )
            s = exp_row.reduce(cute.ReductionOp.ADD, cutlass.Float32.zero, 0)
            correction = cute.math.exp2(
                prev * scale_log2 - new_max * scale_log2, fastmath=True
            )
            # if prev == -inf, correction -> exp2(-inf) = 0, which is correct.
            row_sum[r] = row_sum[r] * correction + s
            acc_O_mn[r, None] = acc_O_mn[r, None].load() * correction
            row_max[r] = new_max
            acc_S_mn[r, None] = exp_row

    # =====================================================================
    # Write per-split partial O (fp32, unnormalized) and LSE.
    #   LSE_split = row_max * scale + log(row_sum)  (in natural log)
    #   O_partial = acc_O / row_sum   (already divided so combine just weights)
    # We store normalized O_partial and the LSE; combine re-weights by
    # exp(LSE_split - LSE_global).
    # =====================================================================
    @cute.jit
    def _write_partials(
        self, acc_O, row_max, row_sum, scale_log2,
        mOpartial, mLSE, batch, head_q_base, split_idx, q_per_kv,
        thr_mma, tidx, has_work,
        sRedO, sRedMax, sRedSum,
    ):
        # ---- finalize this warp's LOCAL softmax (intra-warp quad reduction) ----
        acc_O_mn = self._make_acc_tensor_mn_view(acc_O)
        for r in cutlass.range_constexpr(cute.size(row_sum)):
            row_sum[r] = self._threadquad_reduce_sum(row_sum[r])
            # row_max was already quad-reduced inside _softmax_rescale.

        # ---- cross-warp combine over the N-tiling ----
        # Each warp owns a disjoint KV subset; combine partials (max,sum,O) via
        # smem. Warp layout: thread's warp id = tidx // 32.
        num_warps = cutlass.const_expr(self._num_threads // 32)
        warp_id = tidx // 32
        hd = mOpartial.shape[3]
        m_block = cutlass.const_expr(self._m_block_size)
        d_pad = cutlass.const_expr(self._head_dim_padded)

        # coordinate tensor to map each owned acc element to (row, col)
        cO = cute.make_identity_tensor((m_block, d_pad))
        tOcO = thr_mma.partition_C(cO)
        tOcO_mn = self._make_acc_tensor_mn_view(tOcO)

        n_rows = cutlass.const_expr(cute.size(acc_O_mn.shape[0]))
        n_cols = cutlass.const_expr(cute.size(acc_O_mn.shape[1]))

        if cutlass.const_expr(num_warps > 1):
            # Write this warp's partial max/sum (one writer per row: quad leader)
            for r in cutlass.range_constexpr(n_rows):
                row = tOcO_mn[r, 0][0]
                if cute.arch.lane_idx() % 4 == 0:
                    sRedMax[warp_id * m_block + row] = row_max[r]
                    sRedSum[warp_id * m_block + row] = row_sum[r]
                for c in cutlass.range_constexpr(n_cols):
                    col = tOcO_mn[r, c][1]
                    sRedO[(warp_id * m_block + row) * d_pad + col] = acc_O_mn[r, c]
            self.cta_sync_barrier.arrive_and_wait()

            # Every thread recomputes the combined result for the rows/cols it
            # owns by reading all warps' partials. (Redundant across the quad but
            # cheap and avoids extra synchronization.)
            for r in cutlass.range_constexpr(n_rows):
                row = tOcO_mn[r, 0][0]
                gmax = -cutlass.Float32.inf
                for w in cutlass.range_constexpr(num_warps):
                    gmax = cute.arch.fmax(gmax, sRedMax[w * m_block + row])
                gsum = cutlass.Float32.zero
                for w in cutlass.range_constexpr(num_warps):
                    sc = cute.math.exp2(
                        (sRedMax[w * m_block + row] - gmax) * scale_log2,
                        fastmath=True,
                    )
                    gsum += sc * sRedSum[w * m_block + row]
                inv = 1.0 if (gsum == 0.0 or gsum != gsum) else cute.arch.rcp_approx(gsum)
                for c in cutlass.range_constexpr(n_cols):
                    col = tOcO_mn[r, c][1]
                    acc = cutlass.Float32.zero
                    for w in cutlass.range_constexpr(num_warps):
                        sc = cute.math.exp2(
                            (sRedMax[w * m_block + row] - gmax) * scale_log2,
                            fastmath=True,
                        )
                        acc += sc * sRedO[(w * m_block + row) * d_pad + col]
                    acc_O_mn[r, c] = acc * inv
                # carry combined lse and max into local buffers for the write
                row_max[r] = gmax
                lse = (gmax * scale_log2 + cute.math.log2(
                    gsum if gsum > 0.0 else 1.0, fastmath=True
                )) * (1.0 / LOG2_E)
                if gsum == 0.0:
                    lse = -cutlass.Float32.inf
                if not has_work:
                    lse = -cutlass.Float32.inf
                row_sum[r] = lse
        else:
            for r in cutlass.range_constexpr(n_rows):
                rs = row_sum[r]
                inv = 1.0 if (rs == 0.0 or rs != rs) else cute.arch.rcp_approx(rs)
                acc_O_mn[r, None] = acc_O_mn[r, None].load() * inv
                lse = (row_max[r] * scale_log2 + cute.math.log2(
                    rs if rs > 0.0 else 1.0, fastmath=True
                )) * (1.0 / LOG2_E)
                if rs == 0.0:
                    lse = -cutlass.Float32.inf
                if not has_work:
                    lse = -cutlass.Float32.inf
                row_sum[r] = lse

        # ---- store O_partial and LSE (only real heads) ----
        for r in cutlass.range_constexpr(n_rows):
            row = tOcO_mn[r, 0][0]
            if cute.arch.lane_idx() % 4 == 0:
                if row < q_per_kv:
                    mLSE[batch, head_q_base + row, split_idx] = row_sum[r]
            for c in cutlass.range_constexpr(n_cols):
                col = tOcO_mn[r, c][1]
                if row < q_per_kv and col < hd:
                    mOpartial[batch, head_q_base + row, split_idx, col] = acc_O_mn[r, c]

    # =====================================================================
    # Pass 2: combine partials across splits (LSE reduction).
    # One CTA per (batch, head_q). blockDim.x == head_dim_padded.
    # Each thread owns one output channel; loops over splits.
    # =====================================================================
    @cute.kernel
    def _combine_kernel(
        self,
        mOpartial: cute.Tensor,   # (b, hq, splits, d) fp32
        mLSE: cute.Tensor,        # (b, hq, splits)     fp32
        mO: cute.Tensor,          # (b, hq, d)
        num_splits: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        head_q, batch, _ = cute.arch.block_idx()

        hd = mO.shape[2]

        # smem to hold per-split LSE and the running max
        smem = utils.SmemAllocator()
        sLSE = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout(num_splits), byte_alignment=16
        )

        # thread 0..num_splits-1 load LSE
        if tidx < num_splits:
            sLSE[tidx] = mLSE[batch, head_q, tidx]
        cute.arch.barrier()

        # compute global max LSE
        gmax = -cutlass.Float32.inf
        for s in cutlass.range_constexpr(num_splits):
            gmax = cute.arch.fmax(gmax, sLSE[s])

        # denominator = sum_s exp(lse_s - gmax)
        denom = cutlass.Float32.zero
        for s in cutlass.range_constexpr(num_splits):
            denom += cute.math.exp(sLSE[s] - gmax, fastmath=True)
        inv_denom = 1.0 if (denom == 0.0) else cute.arch.rcp_approx(denom)

        # each thread accumulates its own output channel across splits
        if tidx < hd:
            acc = cutlass.Float32.zero
            for s in cutlass.range_constexpr(num_splits):
                w = cute.math.exp(sLSE[s] - gmax, fastmath=True) * inv_denom
                acc += w * mOpartial[batch, head_q, s, tidx]
            mO[batch, head_q, tidx] = acc.to(mO.element_type)

    # =====================================================================
    # helpers
    # =====================================================================
    def _make_acc_tensor_mn_view(self, acc: cute.Tensor) -> cute.Tensor:
        acc_layout_col_major = cute.make_layout(acc.layout.shape)
        acc_layout_mn = cute.make_layout(
            (
                (acc_layout_col_major.shape[0][1], acc_layout_col_major.shape[1]),
                (acc_layout_col_major.shape[0][0], acc_layout_col_major.shape[2]),
            ),
            stride=(
                (acc_layout_col_major.stride[0][1], acc_layout_col_major.stride[1]),
                (acc_layout_col_major.stride[0][0], acc_layout_col_major.stride[2]),
            ),
        )
        acc_layout_mn = cute.composition(acc.layout, acc_layout_mn)
        return cute.make_tensor(acc.iterator, acc_layout_mn)

    def _threadquad_reduce(self, val, op):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=2, mask=-1, mask_and_clamp=31))
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1, mask=-1, mask_and_clamp=31))
        return val

    def _threadquad_reduce_max(self, val):
        return self._threadquad_reduce(val, lambda x, y: cute.arch.fmax(x, y))

    def _threadquad_reduce_sum(self, val):
        return self._threadquad_reduce(val, lambda x, y: x + y)
