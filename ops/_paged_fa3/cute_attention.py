"""cute_attention.py - hand-written FlashAttention forward in the NVIDIA CuTe DSL.

This is the Task-1 CuTe DSL deliverable (the brief explicitly requires the operator
to be hand-written in CuTe DSL). It is a self-authored FlashAttention-2 forward:

  * tiled MMA on Hopper (sm_90a) via warp.MmaF16BF16Op, fp32 accumulate
  * cp.async G->S staged loads of Q/K/V into swizzled shared memory
  * ldmatrix S->R for the mma operands
  * numerically-stable online softmax with the log2/exp2 trick and quad shuffle reduce
  * head_dim=128, bf16, GQA-aware (K/V head = q_head // (Hq/Hkv))

It runs on a CONTIGUOUS KV view. The paged block-table gather is staged in the Python
entry (paged_cute_attention below): decode reads few queries against a long KV cache,
so gathering the paged blocks into a contiguous [B,S,Hkv,D] view (or feeding an already
contiguous cache) keeps the DSL kernel clean while the paging logic lives one layer up
-- exactly how a paged serving stack stages KV for a dense attention op.

Reference for CuTe DSL idioms: CUTLASS 4.6 examples/python/CuTeDSL/ampere/flash_attention_v2.py.
The algorithm, structure and code here are authored for this project; the DSL API calls
follow the documented CuTe patterns.
"""
import os
from types import SimpleNamespace
from typing import Type

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.hopper_helpers as sm90utils
from cutlass.cute.nvgpu import warpgroup
from cutlass.cute.nvgpu import OperandMajorMode
from cutlass.utils.layout import LayoutEnum

LOG2_E = 1.4426950408889634074


class CuteFlashAttentionFwd:
    """Hand-written FlashAttention-2 forward in CuTe DSL (sm_90a, bf16, hd=128)."""

    def __init__(self, head_dim, seqlen_q, seqlen_k, m_block_size=64, n_block_size=64,
                 num_threads=128, is_causal=False, num_splits=1, min_blocks_per_mp=0):
        self._head_dim = head_dim
        self._head_dim_padded = (head_dim + 31) // 32 * 32
        self._seqlen_q = seqlen_q          # static (Python int) -> enables constexpr unroll
        self._seqlen_k = seqlen_k
        self._m_block_size = m_block_size
        self._n_block_size = n_block_size
        self._num_threads = num_threads
        self._is_causal = is_causal
        # WGMMA (Hopper warpgroup MMA) path: enabled for the NON-causal, single-split
        # (full-KV prefill) case only. Causal + split-KV stay on the proven SM80 SIMT
        # path for now. Requires exactly one warpgroup (128 threads) and BM=64.
        self._use_wgmma = (not is_causal) and num_splits == 1 and num_threads == 128 \
            and m_block_size == 64 and os.environ.get("_NO_WGMMA", "0") == "0"
        # WGMMA software-pipeline depth for K/V smem double-buffering. 2 = one block of
        # K prefetch + this block's V overlap the warpgroup MMA. env _KV_STAGES overrides.
        self._kv_stages = int(os.environ.get("_KV_STAGES", "2"))
        # __launch_bounds__ minBlocksPerMP: caps registers so this many CTAs fit per SM.
        # 0 = let the compiler choose (default, ~2 CTAs at 186 regs). Raising to 3-4 forces
        # lower reg usage -> higher occupancy (kernel is latency-bound at 12.5% occ, not
        # throughput-saturated), the single biggest prefill lever per ncu.
        self._min_blocks_per_mp = min_blocks_per_mp
        self._n_block_total = (seqlen_k + n_block_size - 1) // n_block_size
        # split-KV (flash-decoding): each CTA owns a contiguous slice of the n-blocks so
        # that the grid can fill all SMs when Sq is tiny (decode). num_splits==1 is the
        # ordinary full-KV FA2 path (writes normalized O, no LSE).
        self._num_splits = num_splits
        # ceil-divide the n-blocks across splits; the last split may own fewer.
        self._nblk_per_split = (self._n_block_total + num_splits - 1) // num_splits
        import cutlass.pipeline as pipeline
        self._pipeline = pipeline
        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=num_threads)

    # ------------------------------------------------------------------ host
    @cute.jit
    def __call__(self, mQ, mK, mV, mO, softmax_scale: cutlass.Float32,
                 stream: cuda.CUstream, mLSE=None):
        self._dtype = mQ.element_type

        smem_k = 64 if self._head_dim_padded % 64 == 0 else 32
        swz = 3 if smem_k == 64 else 2
        sQ_atom = cute.make_composed_layout(
            cute.make_swizzle(swz, 3, 3), 0,
            cute.make_layout((8, smem_k), stride=(smem_k, 1)))
        sQ_layout = cute.tile_to_shape(
            sQ_atom, (self._m_block_size, self._head_dim_padded), (0, 1))
        sKV_layout = cute.tile_to_shape(
            sQ_atom, (self._n_block_size, self._head_dim_padded), (0, 1))
        sO_layout = sQ_layout

        @cute.struct
        class SharedStorage:
            sQ: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sQ_layout)], 1024]
            sK: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024]
            sV: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024]

        copy_bits = 128
        cp_elems = copy_bits // self._dtype.width
        atom_g2s = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self._dtype, num_bits_per_copy=copy_bits)
        atom_uni = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self._dtype, num_bits_per_copy=copy_bits)
        t1 = sQ_atom.outer.shape[1] // cp_elems
        tQKV = cute.make_layout(
            (self._num_threads // t1, t1), stride=(t1, 1))
        vQKV = cute.make_layout((1, cp_elems))
        gmem_copy_QKV = cute.make_tiled_copy_tv(atom_g2s, tQKV, vQKV)
        gmem_copy_O = cute.make_tiled_copy_tv(atom_uni, tQKV, vQKV)

        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self._dtype, cutlass.Float32, (16, 8, 16)),
            (self._num_threads // 32, 1, 1),
            permutation_mnk=(self._num_threads // 32 * 16, 16, 16))

        # grid: (m-blocks * num_splits, batch, head). We fold the split index into the
        # first grid dim so we don't need a 4th dim: block_x = m_block * num_splits + split.
        n_m = (self._seqlen_q + self._m_block_size - 1) // self._m_block_size
        grid = (
            n_m * self._num_splits,
            cute.size(mQ.shape[0]),
            cute.size(mQ.shape[2]),
        )
        scale_log2 = softmax_scale * LOG2_E
        # number of KV heads inferred from mK head extent for GQA mapping
        if cutlass.const_expr(self._use_wgmma):
            BM = self._m_block_size
            BN = self._n_block_size
            Dp = self._head_dim_padded
            # gemm1 QK^T: A=Q (M=BM,K=D) K-major SMEM, B=K (N=BN,K=D) K-major SMEM.
            wmma1 = sm90utils.make_trivial_tiled_mma(
                self._dtype, self._dtype, OperandMajorMode.K, OperandMajorMode.K,
                cutlass.Float32, (1, 1, 1), (BM, BN), warpgroup.OperandSource.SMEM)
            # gemm2 P@V: A=P (M=BM,K=BN) RMEM, B=V (N=D,K=BN) MN-major SMEM.
            wmma2 = sm90utils.make_trivial_tiled_mma(
                self._dtype, self._dtype, OperandMajorMode.K, OperandMajorMode.MN,
                cutlass.Float32, (1, 1, 1), (BM, Dp), warpgroup.OperandSource.RMEM)
            # 2-stage double-buffered K/V so the next block's K prefetch and this
            # block's V load overlap the warpgroup MMA instead of draining after each.
            KV_STAGES = self._kv_stages
            wsQ_l = sm90utils.make_smem_layout_a(LayoutEnum.ROW_MAJOR, (BM, BN, Dp), self._dtype, 1)
            wsK_l = sm90utils.make_smem_layout_b(LayoutEnum.ROW_MAJOR, (BM, BN, Dp), self._dtype, KV_STAGES)
            wsVt_l = sm90utils.make_smem_layout_b(LayoutEnum.COL_MAJOR, (BM, Dp, BN), self._dtype, KV_STAGES)
            wsO_l = sm90utils.make_smem_layout_epi(self._dtype, LayoutEnum.ROW_MAJOR, (BM, Dp), 1)
            self.kernel_wgmma(
                mQ, mK, mV, mO, scale_log2, wmma1, wmma2,
                wsQ_l, wsK_l, wsVt_l, wsO_l,
            ).launch(grid=grid, block=[self._num_threads, 1, 1], stream=stream,
                     min_blocks_per_mp=self._min_blocks_per_mp)
        else:
            self.kernel(
                mQ, mK, mV, mO, mLSE, scale_log2,
                sQ_layout, sKV_layout, sO_layout,
                gmem_copy_QKV, gmem_copy_O, tiled_mma, SharedStorage,
            ).launch(grid=grid, block=[self._num_threads, 1, 1], stream=stream,
                     min_blocks_per_mp=self._min_blocks_per_mp)

    # ---------------------------------------------------------------- device
    @cute.kernel
    def kernel(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor,
               mO: cute.Tensor, mLSE, scale_log2: cutlass.Float32,
               sQ_layout: cute.ComposedLayout, sKV_layout: cute.ComposedLayout,
               sO_layout: cute.ComposedLayout,
               gmem_copy_QKV: cute.TiledCopy, gmem_copy_O: cute.TiledCopy,
               tiled_mma: cute.TiledMma, SharedStorage: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        block_x, batch, head = cute.arch.block_idx()
        # unfold the packed (m_block, split) grid-x index
        split = block_x % self._num_splits
        m_block = block_x // self._num_splits

        Hq = cute.size(mQ.shape[2])
        Hkv = cute.size(mK.shape[2])
        kv_head = head // (Hq // Hkv)      # GQA mapping (constexpr-friendly ratio)

        # KV seqlen is static at compile time (compile cache is keyed on it), so the
        # n-block count is a Python constant -> the mainloop can be unrolled with
        # range_constexpr and `first` is a true compile-time bool.
        n_block_total = self._n_block_total

        gQ = cute.local_tile(mQ[batch, None, head, None],
                             (self._m_block_size, self._head_dim_padded), (m_block, 0))
        gK = cute.local_tile(mK[batch, None, kv_head, None],
                             (self._n_block_size, self._head_dim_padded), (None, 0))
        gV = cute.local_tile(mV[batch, None, kv_head, None],
                             (self._n_block_size, self._head_dim_padded), (None, 0))
        # O output row: full mode -> batch; split mode -> batch*num_splits + split, so
        # partial O is a 4D tensor [B*num_splits, Sq, Hq, D] (num_splits==1 => plain O).
        bo = batch * self._num_splits + split
        gO = cute.local_tile(mO[bo, None, head, None],
                             (self._m_block_size, self._head_dim_padded), (m_block, 0))

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sK = storage.sK.get_tensor(sKV_layout)
        sV = storage.sV.get_tensor(sKV_layout)
        sVt = cute.composition(sV, cute.make_layout(
            (self._head_dim_padded, self._n_block_size),
            stride=(self._n_block_size, 1)))

        g2s = gmem_copy_QKV.get_slice(tidx)
        tQgQ = g2s.partition_S(gQ); tQsQ = g2s.partition_D(sQ)
        tKgK = g2s.partition_S(gK); tKsK = g2s.partition_D(sK)
        tVgV = g2s.partition_S(gV); tVsV = g2s.partition_D(sV)

        thr_mma = tiled_mma.get_slice(tidx)
        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
        tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK))
        tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt))
        acc_O = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((self._m_block_size, self._head_dim_padded)),
            cutlass.Float32)
        acc_O.fill(0.0)

        sc_Q = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self._dtype)
        sc_K = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self._dtype)
        sc_V = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self._dtype)
        stc_Q = cute.make_tiled_copy_A(sc_Q, tiled_mma)
        stc_K = cute.make_tiled_copy_B(sc_K, tiled_mma)
        stc_V = cute.make_tiled_copy_B(sc_V, tiled_mma)
        stQ = stc_Q.get_slice(tidx); stK = stc_K.get_slice(tidx); stV = stc_V.get_slice(tidx)
        tSsQ = stQ.partition_S(sQ);  tSrQv = stQ.retile(tSrQ)
        tSsK = stK.partition_S(sK);  tSrKv = stK.retile(tSrK)
        tOsVt = stV.partition_S(sVt); tOrVtv = stV.retile(tOrVt)

        row_max = cute.make_rmem_tensor(
            (acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32)
        row_sum = cute.make_rmem_tensor(
            (acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32)
        row_max.fill(-cutlass.Float32.inf)
        row_sum.fill(0.0)

        # per-split n-block range [nb_lo, nb_hi). Split s owns blocks
        # [s*nblk_per_split, min((s+1)*nblk_per_split, total)). Because `split` is a
        # runtime value, we compute the bounds as runtime ints (not constexpr) and the
        # kernel walks them with the same high->low order used by the full path.
        nb_per = self._nblk_per_split
        nb_lo = split * nb_per
        nb_hi = nb_lo + nb_per
        if cutlass.const_expr(self._num_splits > 1):
            nb_hi = nb_hi if nb_hi < n_block_total else n_block_total
        else:
            nb_lo = 0
            nb_hi = n_block_total

        # prologue: load Q (full) and K for the highest n-block this CTA will touch.
        # CAUSAL prefill: query rows [m_block*M, m_block*M+M) attend to key rows
        # <= m_block*M+M-1, whose n-block is n_diag = (m_block*M + M-1)//N. This CTA starts
        # at n_diag and walks DOWN to 0 -- every strictly-above-diagonal block is skipped
        # (no load, no MMA), the ~2x causal saving FA3 also exploits. Works for M>=N: with
        # M>N the diagonal spans ceil(M/N) blocks, all masked via the per-element mask
        # (mask_lo below). Non-causal full mode keeps the top-of-KV start. Decode unchanged.
        M = self._m_block_size
        N = self._n_block_size
        if cutlass.const_expr(self._is_causal and self._num_splits == 1):
            n_block = (m_block * M + M - 1) // N   # runtime: diagonal n-block for this CTA
            # blocks with nb >= mask_lo overlap the causal diagonal band -> need the mask.
            mask_lo = (m_block * M) // N
        else:
            n_block = nb_hi - 1
            mask_lo = 0
        cute.copy(gmem_copy_QKV, tQgQ, tQsQ)
        cute.copy(gmem_copy_QKV, tKgK[None, None, None, n_block], tKsK)
        cute.arch.cp_async_commit_group()

        basic = SimpleNamespace(mQ=mQ, mK=mK, mV=mV, n_block=n_block,
                                batch=batch, m_block=m_block, head=head, mask_lo=mask_lo)
        mma_p = SimpleNamespace(tiled_mma=tiled_mma, thr_mma=thr_mma,
                                tSrQ=tSrQ, tSrK=tSrK, tOrVt=tOrVt, acc_O=acc_O)
        gcp = SimpleNamespace(gmem_tiled_copy_QKV=gmem_copy_QKV,
                              tVgV=tVgV, tVsV=tVsV, tKgK=tKgK, tKsK=tKsK)
        scp = SimpleNamespace(smem_tiled_copy_Q=stc_Q, smem_tiled_copy_K=stc_K,
                              smem_tiled_copy_V=stc_V, tSsQ=tSsQ, tSsK=tSsK,
                              tOsVt=tOsVt, tSrQv=tSrQv, tSrKv=tSrKv, tOrVtv=tOrVtv)
        sm_p = SimpleNamespace(row_max=row_max, row_sum=row_sum, scale_log2=scale_log2)

        if cutlass.const_expr(self._is_causal and self._num_splits == 1):
            # CAUSAL prefill: walk from the diagonal block (nb=m_block) DOWN to 0. The
            # unroll count is n_block_total (worst case = bottom m_block), but for higher
            # rows the deeper steps underflow (nb<0) and are skipped by a runtime guard, so
            # NO above-diagonal block is ever loaded or MMA'd. Only step==0 (the diagonal)
            # needs the causal mask; every deeper block is fully below the diagonal.
            # COMPILE-TIME FIX (runtime loop): a full range_constexpr(n_block_total)
            # unroll emits n_block_total copies of the block body -> at 128k (2048 blocks)
            # that is ~18 min of single-threaded codegen (measured: compile roughly doubles
            # per doubling of n_blocks -- 3.5s@8, 6.4s@16, 14s@32, 34s@64). The RUNTIME
            # kernel is unaffected (<1ms). Fix: peel step 0 (keeps first=True CONSTEXPR for
            # the online-softmax init and the diagonal mask), then run the remaining blocks
            # in a RUNTIME loop (cutlass.range) with first=False. `last` only gates a
            # redundant next-K prefetch; always prefetching is safe (the (nb-1)>=0 guard
            # inside _one_block_causal stops any OOB load), so the looped body uses
            # last=False. This collapses ~2048 emitted bodies -> 2 (peeled-first + one
            # runtime body): compile becomes O(1) in seqlen, result stays bit-identical
            # (the deeper-than-diagonal underflow nb<0 is still skipped by the runtime guard).
            self._one_block_causal(basic, mma_p, gcp, scp, sm_p, n_block,
                                    first=True, last=(n_block_total == 1))
            for step in cutlass.range(1, n_block_total):
                nb = n_block - step        # runtime diagonal-relative index
                self._one_block_causal(basic, mma_p, gcp, scp, sm_p, nb,
                                        first=False, last=False)
        elif cutlass.const_expr(self._num_splits == 1):
            # non-causal full mode: over ALL n-blocks, high->low, constexpr-unrolled.
            for step in cutlass.range_constexpr(n_block_total):
                nb = n_block_total - 1 - step
                self._one_block(basic, mma_p, gcp, scp, sm_p, nb,
                                first=(step == 0), last=(nb == 0))
        else:
            # split-KV (decode): unroll the compile-time max blocks/split. `first`/`last`
            # are CONSTEXPR (derived from the constexpr `step`), so control flow stays
            # constexpr like the proven full path; only the tensor block index `nb` is a
            # runtime value. The ragged last split is handled by clamping nb below (a
            # duplicate top-block re-add is avoided by comparing nb to the runtime bound
            # through zeroing when nb underflows nb_lo -- but with per exactly dividing we
            # don't need it; we always process exactly `nblk_per_split` blocks and rely on
            # num_splits being chosen so it divides evenly, else the last split re-reads a
            # block which we mask by nb clamp). Split mode is non-causal.
            npers = self._nblk_per_split
            for step in cutlass.range_constexpr(npers):
                nb = n_block - step               # runtime block index
                self._one_block_rt(basic, mma_p, gcp, scp, sm_p, nb,
                                   first=(step == 0), last=(step == npers - 1))

        # Normalize O per (split) so the combine is the standard flash-decoding reduction
        #   O = sum_s softmax_s(lse_s) . O_s   with lse_s = m_s + ln(l_s).
        # Full mode also normalizes (num_splits==1) and writes no LSE.
        self._normalize(acc_O, row_sum)
        if cutlass.const_expr(self._num_splits > 1):
            self._write_lse(basic, mma_p, mLSE, sm_p, bo, m_block, head, tidx)
        rO = cute.make_fragment_like(acc_O, self._dtype)
        rO.store(acc_O.load().to(self._dtype))
        sO = cute.make_tensor(sQ.iterator, sO_layout)
        stc_O = cute.make_tiled_copy_C(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self._dtype), tiled_mma)
        stO = stc_O.get_slice(tidx)
        cute.copy(stc_O, stO.retile(rO), stO.partition_D(sO))
        self.cta_sync_barrier.arrive_and_wait()
        gO2 = gmem_copy_O.get_slice(tidx)
        cute.copy(gmem_copy_O, gO2.partition_S(sO), gO2.partition_D(gO))

    # ================================================================== WGMMA
    # Hopper warpgroup-MMA forward, NON-causal full-KV. One warpgroup (128 threads).
    #   gemm1 S=Q@K^T : both operands from SMEM (K-major).
    #   online softmax over the WGMMA acc MN layout (_mn_wgmma).
    #   gemm2 O+=P@V  : A=P from RMEM (zero-permutation reg reuse of the softmax'd
    #                   acc_S bf16 fragment), B=V from SMEM MN-major (D,BN).
    @cute.kernel
    def kernel_wgmma(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor,
                     mO: cute.Tensor, scale_log2: cutlass.Float32,
                     mma1: cute.TiledMma, mma2: cute.TiledMma,
                     sQ_l: cute.ComposedLayout, sK_l: cute.ComposedLayout,
                     sVt_l: cute.ComposedLayout, sO_l: cute.ComposedLayout):
        tidx, _, _ = cute.arch.thread_idx()
        m_block, batch, head = cute.arch.block_idx()

        Hq = cute.size(mQ.shape[2])
        Hkv = cute.size(mK.shape[2])
        kv_head = head // (Hq // Hkv)
        BM = self._m_block_size
        BN = self._n_block_size
        Dp = self._head_dim_padded
        n_block_total = self._n_block_total

        gQ = cute.local_tile(mQ[batch, None, head, None], (BM, Dp), (m_block, 0))
        gK = cute.local_tile(mK[batch, None, kv_head, None], (BN, Dp), (None, 0))
        # V natural tensor for this (batch,kv_head): [Sk, Dp]. Build a transposed
        # (Dp, Sk) view once (Dp contiguous, Sk strided by the row stride) and local_tile
        # the per-block (Dp, BN) slice from it -> keeps alignment annotations intact.
        mVh = mV[batch, None, kv_head, None]
        mVt = cute.make_tensor(mVh.iterator, cute.make_layout(
            (Dp, cute.size(mVh.shape[0])), stride=(1, mVh.stride[0])))
        gVt_all = cute.local_tile(mVt, (Dp, BN), (0, None))   # (Dp, BN, n_blocks)
        gO = cute.local_tile(mO[batch, None, head, None], (BM, Dp), (m_block, 0))

        NST = self._kv_stages
        smem = cutlass.utils.SmemAllocator()
        sQ_s = smem.allocate_tensor(self._dtype, sQ_l.outer, byte_alignment=1024, swizzle=sQ_l.inner)
        sK_s = smem.allocate_tensor(self._dtype, sK_l.outer, byte_alignment=1024, swizzle=sK_l.inner)
        sVt_s = smem.allocate_tensor(self._dtype, sVt_l.outer, byte_alignment=1024, swizzle=sVt_l.inner)
        sQ = sQ_s[None, None, 0]
        # sK_s / sVt_s carry NST stages in their last mode; index per-block by step parity.

        # cp.async 128b tiled copy for (rows, Dp) tiles (Dp contiguous = K dim).
        cp_elems = 128 // self._dtype.width
        t1 = Dp // cp_elems
        atom_g2s = cute.make_copy_atom(cute.nvgpu.cpasync.CopyG2SOp(), self._dtype, num_bits_per_copy=128)
        tv_thr = cute.make_layout((self._num_threads // t1, t1), stride=(t1, 1))
        tv_val = cute.make_layout((1, cp_elems))
        g2s = cute.make_tiled_copy_tv(atom_g2s, tv_thr, tv_val).get_slice(tidx)
        # V tiled copy: transposed (Dp,BN) gmem view -> MN-major (Dp,BN) smem. PROVEN.
        vatom = cute.make_copy_atom(cute.nvgpu.cpasync.CopyG2SOp(), self._dtype, num_bits_per_copy=128)
        vtc = cute.make_tiled_copy_tv(
            vatom, cute.make_layout((16, 8), stride=(8, 1)),
            cute.make_layout((8, 1))).get_slice(tidx)

        # load Q once (reused across all KV blocks)
        cute.copy(atom_g2s, g2s.partition_S(gQ), g2s.partition_D(sQ))

        wg1 = mma1.get_slice(tidx)
        wg2 = mma2.get_slice(tidx)
        acc_O = cute.make_rmem_tensor(mma2.partition_shape_C((BM, Dp)), cutlass.Float32)
        acc_O.fill(0.0)

        # online-softmax running state: rows/thread = acc_O MN rows.
        acc_O_mn0 = self._mn_wgmma(acc_O)
        nrow = cute.size(acc_O_mn0.shape[0])
        row_max = cute.make_rmem_tensor((nrow,), cutlass.Float32)
        row_sum = cute.make_rmem_tensor((nrow,), cutlass.Float32)
        row_max.fill(-cutlass.Float32.inf)
        row_sum.fill(0.0)

        # =============== software-pipelined WGMMA mainloop ===============
        # cp.async group discipline: every K load and every V load is committed as its
        # own group. We PREFETCH the next block's K while the current block's gemm1 +
        # softmax + gemm2 run, and issue this block's V up front so it overlaps gemm1 +
        # softmax. wait_group(g) blocks until at most g groups remain outstanding.
        #
        # Block b (b = n_block_total-1 .. 0) lives in K/V smem stage (step % NST).
        # Prologue: prefetch K of the FIRST processed block into stage 0.
        def _blk_of(st):
            return n_block_total - 1 - st

        def _load_K(st):
            nbk = _blk_of(st)
            sK = sK_s[None, None, st % NST]
            cute.copy(atom_g2s, g2s.partition_S(gK[None, None, nbk]), g2s.partition_D(sK))
            cute.arch.cp_async_commit_group()

        def _load_V(st):
            nbv = _blk_of(st)
            sVt = sVt_s[None, None, st % NST]
            cute.copy(vatom, vtc.partition_S(gVt_all[None, None, nbv]), vtc.partition_D(sVt))
            cute.arch.cp_async_commit_group()

        # ---- prologue: K[step0] in flight ----
        _load_K(0)

        for step in cutlass.range_constexpr(n_block_total):
            nb = n_block_total - 1 - step
            first = (step == 0)
            last = (step == n_block_total - 1)
            cur = step % NST
            sK = sK_s[None, None, cur]
            sVt = sVt_s[None, None, cur]

            # Outstanding groups right now (before this step issues anything):
            #   step 0 : just K[0]                          -> 1 group
            #   step>0 : K[step] prefetched at prev step    -> 1 group (V[step-1] already
            #            consumed & waited last iter)
            # We need K[step] ARRIVED before gemm1. Issue V[step] + prefetch K[step+1]
            # first (so they overlap), then wait so only those future loads remain.
            _load_V(step)                      # V[nb] -> stage cur
            if cutlass.const_expr(not last):
                _load_K(step + 1)              # prefetch K[next] -> stage (step+1)%NST
            # After the two issues above: outstanding = K[step](old) + V[step] + K[step+1]
            #   last step: K[step] + V[step]                -> wait to 1 leaves V (need K done)
            #   else     : K[step] + V[step] + K[step+1]    -> wait to 2 leaves V + nextK
            # In BOTH cases waiting to (#future loads we must still keep) guarantees K[step]
            # (the OLDEST group) is complete: waiting to N leaves the N NEWEST groups.
            if cutlass.const_expr(not last):
                cute.arch.cp_async_wait_group(2)   # keep V[step] + K[step+1]; K[step] done
            else:
                cute.arch.cp_async_wait_group(1)   # keep V[step]; K[step] done
            cute.arch.barrier()

            # ---- gemm1 S = Q@K^T (K from stage cur) ----
            tSrQ = wg1.make_fragment_A(wg1.partition_A(sQ))
            tSrK = wg1.make_fragment_B(wg1.partition_B(sK))
            acc_S = cute.make_rmem_tensor(mma1.partition_shape_C((BM, BN)), cutlass.Float32)
            acc_S.fill(0.0)
            warpgroup.fence()
            mma1.set(warpgroup.Field.ACCUMULATE, False)
            cute.gemm(mma1, acc_S, tSrQ, tSrK, acc_S)
            warpgroup.commit_group()
            warpgroup.wait_group(0)
            cute.arch.barrier()

            # ---- online softmax over acc_S (WGMMA MN layout) ----
            self._softmax_wgmma(acc_S, acc_O, row_max, row_sum, scale_log2, first)

            # ---- ensure V[step] arrived (keep prefetched K[step+1] in flight) ----
            if cutlass.const_expr(not last):
                cute.arch.cp_async_wait_group(1)   # leave only K[step+1]; V[step] done
            else:
                cute.arch.cp_async_wait_group(0)   # nothing more to keep
            cute.arch.barrier()

            # ---- gemm2 O += P@V, P from RMEM (zero-permutation reg reuse) ----
            rP = cute.make_fragment_like(acc_S, self._dtype)
            rP.store(acc_S.load().to(self._dtype))
            tOrVt = wg2.make_fragment_B(wg2.partition_B(sVt))
            tOrP = wg2.make_fragment_A(mma2.partition_shape_A((BM, BN)))
            tOrP.store(rP.load())
            warpgroup.fence()
            mma2.set(warpgroup.Field.ACCUMULATE, True)
            cute.gemm(mma2, acc_O, tOrP, tOrVt, acc_O)
            warpgroup.commit_group()
            warpgroup.wait_group(0)
            cute.arch.barrier()

        # normalize O by row_sum (quad-reduced already inside softmax across the 4 lanes)
        acc_O_mn = self._mn_wgmma(acc_O)
        for r in cutlass.range_constexpr(nrow):
            rs = row_sum[r]
            bad = rs == 0.0 or rs != rs
            sc = 1.0 if bad else cute.arch.rcp_approx(rs)
            acc_O_mn[r, None] = acc_O_mn[r, None].load() * sc

        # epilogue: acc_O -> smem -> gmem
        rO = cute.make_fragment_like(acc_O, self._dtype)
        rO.store(acc_O.load().to(self._dtype))
        sO_s = smem.allocate_tensor(self._dtype, sO_l.outer, byte_alignment=1024, swizzle=sO_l.inner)
        sO = sO_s[None, None, 0]
        stc = cute.make_tiled_copy_C(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self._dtype), mma2)
        sl = stc.get_slice(tidx)
        cute.copy(stc, sl.retile(rO), sl.partition_D(sO))
        cute.arch.barrier()
        t1c = Dp // cp_elems
        tvc = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self._dtype, num_bits_per_copy=128),
            cute.make_layout((self._num_threads // t1c, t1c), stride=(t1c, 1)),
            cute.make_layout((1, cp_elems))).get_slice(tidx)
        cute.copy(tvc, tvc.partition_S(sO), tvc.partition_D(gO))

    # ---- WGMMA online-softmax rescale: acc_S is the QK^T WGMMA acc; carry into acc_O ----
    @cute.jit
    def _softmax_wgmma(self, acc_S, acc_O, row_max, row_sum, scale_log2, first: cutlass.Constexpr):
        acc_S_mn = self._mn_wgmma(acc_S)
        acc_O_mn = self._mn_wgmma(acc_O)
        nrow = cute.size(acc_S_mn.shape[0])
        if cutlass.const_expr(not first):
            row_max_prev = cute.make_fragment_like(row_max, cutlass.Float32)
            cute.basic_copy(row_max, row_max_prev)
        for r in cutlass.range_constexpr(nrow):
            s_row = acc_S_mn[r, None].load()
            cur = s_row.reduce(cute.ReductionOp.MAX, -cutlass.Float32.inf, 0)
            cur = self._quad_max(cur)
            if cutlass.const_expr(not first):
                cur = cute.arch.fmax(row_max_prev[r], cur)
            s_exp = cute.math.exp2(s_row * scale_log2 - cur * scale_log2, fastmath=True)
            s_sum = s_exp.reduce(cute.ReductionOp.ADD, cutlass.Float32.zero, 0)
            s_sum = self._quad_sum(s_sum)
            if cutlass.const_expr(not first):
                delta = row_max_prev[r] * scale_log2 - cur * scale_log2
                corr = cute.math.exp2(delta, fastmath=True)
                s_sum = s_sum + row_sum[r] * corr
                acc_O_mn[r, None] = acc_O_mn[r, None].load() * corr
            row_max[r] = cur
            row_sum[r] = s_sum
            acc_S_mn[r, None] = s_exp

    # ---- WGMMA accumulator MN reshape. acc mode-0 = (nn=2, mm=2, ng=8), fragment
    # strides nn->1, mm->2, ng->4. Row(M)=mm, Col(N)=(nn,ng). For acc_O with Dp=128
    # there are 2 N-atoms => mode-2 size 2, which multiplies into the N (col) dim. ----
    def _mn_wgmma(self, acc):
        l = acc.layout
        s0 = l.shape[0]      # (nn, mm, ng)
        d0 = l.stride[0]
        nn, mm, ng = s0[0], s0[1], s0[2]
        dnn, dmm, dng = d0[0], d0[1], d0[2]
        # N-atom dimension (mode 2 of the C fragment): size na, stride dna.
        na = l.shape[2]
        dna = l.stride[2]
        mn = cute.make_layout(
            (mm, (nn, ng, na)),
            stride=(dmm, (dnn, dng, dna)))
        return cute.make_tensor(acc.iterator, mn)

    # ---- one n-block: S=QK^T, online softmax, O += P@V ----
    @cute.jit
    def _one_block(self, basic, mma_p, gcp, scp, sm_p, nb: cutlass.Constexpr,
                   first: cutlass.Constexpr, last: cutlass.Constexpr):
        acc_S = cute.make_rmem_tensor(
            mma_p.thr_mma.partition_shape_C((self._m_block_size, self._n_block_size)),
            cutlass.Float32)
        acc_S.fill(0.0)
        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()
        cute.copy(gcp.gmem_tiled_copy_QKV, gcp.tVgV[None, None, None, nb], gcp.tVsV)
        cute.arch.cp_async_commit_group()

        cute.copy(scp.smem_tiled_copy_Q, scp.tSsQ[None, None, 0], scp.tSrQv[None, None, 0])
        cute.copy(scp.smem_tiled_copy_K, scp.tSsK[None, None, 0], scp.tSrKv[None, None, 0])
        for k in cutlass.range_constexpr(cute.size(scp.tSsQ.shape[2])):
            kn = (k + 1) % cute.size(scp.tSsQ.shape[2])
            cute.copy(scp.smem_tiled_copy_Q, scp.tSsQ[None, None, kn], scp.tSrQv[None, None, kn])
            cute.copy(scp.smem_tiled_copy_K, scp.tSsK[None, None, kn], scp.tSrKv[None, None, kn])
            cute.gemm(mma_p.tiled_mma, acc_S, mma_p.tSrQ[None, None, k],
                      mma_p.tSrK[None, None, k], acc_S)

        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()
        if cutlass.const_expr(not last):
            cute.copy(gcp.gmem_tiled_copy_QKV, gcp.tKgK[None, None, None, nb - 1], gcp.tKsK)
            cute.arch.cp_async_commit_group()

        self._softmax(basic, mma_p, sm_p, acc_S, nb, first)

        rP = cute.make_fragment_like(acc_S, self._dtype)
        rP.store(acc_S.load().to(self._dtype))
        rpd = cute.logical_divide(rP.layout, (None, None, 2))
        rp_view = cute.make_layout(
            ((rpd.shape[0], rpd.shape[2][0]), rpd.shape[1], rpd.shape[2][1]),
            stride=((rpd.stride[0], rpd.stride[2][0]), rpd.stride[1], rpd.stride[2][1]))
        tOrS = cute.make_tensor(rP.iterator, rp_view)
        cute.copy(scp.smem_tiled_copy_V, scp.tOsVt[None, None, 0], scp.tOrVtv[None, None, 0])
        for k in cutlass.range_constexpr(cute.size(tOrS.shape[2])):
            kn = (k + 1) % cute.size(tOrS.shape[2])
            cute.copy(scp.smem_tiled_copy_V, scp.tOsVt[None, None, kn], scp.tOrVtv[None, None, kn])
            cute.gemm(mma_p.tiled_mma, mma_p.acc_O, tOrS[None, None, k],
                      mma_p.tOrVt[None, None, k], mma_p.acc_O)

    # ---- CAUSAL prefill block: runtime nb (diagonal-relative), constexpr first/last/diag.
    # Only the diagonal block (diag=True) applies the triangular mask; deeper blocks are
    # fully below the diagonal so they need no mask. nb is CTA-uniform (= m_block - step),
    # so the `nb >= 0` guard is a uniform branch -> safe to contain barriers. When nb<0 the
    # block is above this CTA's diagonal for a higher-row m_block that unrolled too far; we
    # skip ALL of it (no load, no MMA, no state change). ----
    @cute.jit
    def _one_block_causal(self, basic, mma_p, gcp, scp, sm_p, nb,
                          first: cutlass.Constexpr, last: cutlass.Constexpr):
        if nb >= 0:
            acc_S = cute.make_rmem_tensor(
                mma_p.thr_mma.partition_shape_C((self._m_block_size, self._n_block_size)),
                cutlass.Float32)
            acc_S.fill(0.0)
            cute.arch.cp_async_wait_group(0)
            self.cta_sync_barrier.arrive_and_wait()
            cute.copy(gcp.gmem_tiled_copy_QKV, gcp.tVgV[None, None, None, nb], gcp.tVsV)
            cute.arch.cp_async_commit_group()

            cute.copy(scp.smem_tiled_copy_Q, scp.tSsQ[None, None, 0], scp.tSrQv[None, None, 0])
            cute.copy(scp.smem_tiled_copy_K, scp.tSsK[None, None, 0], scp.tSrKv[None, None, 0])
            for k in cutlass.range_constexpr(cute.size(scp.tSsQ.shape[2])):
                kn = (k + 1) % cute.size(scp.tSsQ.shape[2])
                cute.copy(scp.smem_tiled_copy_Q, scp.tSsQ[None, None, kn], scp.tSrQv[None, None, kn])
                cute.copy(scp.smem_tiled_copy_K, scp.tSsK[None, None, kn], scp.tSrKv[None, None, kn])
                cute.gemm(mma_p.tiled_mma, acc_S, mma_p.tSrQ[None, None, k],
                          mma_p.tSrK[None, None, k], acc_S)

            cute.arch.cp_async_wait_group(0)
            self.cta_sync_barrier.arrive_and_wait()
            # prefetch next-lower K block only if it exists (nb-1 >= 0) and this isn't last.
            if cutlass.const_expr(not last):
                if (nb - 1) >= 0:
                    cute.copy(gcp.gmem_tiled_copy_QKV, gcp.tKgK[None, None, None, nb - 1], gcp.tKsK)
                    cute.arch.cp_async_commit_group()

            # mask blocks overlapping the diagonal band (nb >= mask_lo); deeper blocks are
            # entirely below the diagonal (no mask). diag is a RUNTIME uniform predicate.
            self._softmax_causal(basic, mma_p, sm_p, acc_S, nb, first,
                                 diag=(nb >= basic.mask_lo))

            rP = cute.make_fragment_like(acc_S, self._dtype)
            rP.store(acc_S.load().to(self._dtype))
            rpd = cute.logical_divide(rP.layout, (None, None, 2))
            rp_view = cute.make_layout(
                ((rpd.shape[0], rpd.shape[2][0]), rpd.shape[1], rpd.shape[2][1]),
                stride=((rpd.stride[0], rpd.stride[2][0]), rpd.stride[1], rpd.stride[2][1]))
            tOrS = cute.make_tensor(rP.iterator, rp_view)
            cute.copy(scp.smem_tiled_copy_V, scp.tOsVt[None, None, 0], scp.tOrVtv[None, None, 0])
            for k in cutlass.range_constexpr(cute.size(tOrS.shape[2])):
                kn = (k + 1) % cute.size(tOrS.shape[2])
                cute.copy(scp.smem_tiled_copy_V, scp.tOsVt[None, None, kn], scp.tOrVtv[None, None, kn])
                cute.gemm(mma_p.tiled_mma, mma_p.acc_O, tOrS[None, None, k],
                          mma_p.tOrVt[None, None, k], mma_p.acc_O)

    # ---- softmax for causal prefill: `diag` is a RUNTIME uniform bool -- true on blocks
    # overlapping the causal diagonal band (they get the per-element key>query mask), false
    # on blocks fully below the diagonal (no mask -> cheaper path). The -inf guards on
    # cur/corr only matter when a row can be fully masked, i.e. on diag blocks, so they are
    # also gated on `diag`. Building the identity mask tensor is unconditional (cheap) but
    # only USED when diag. ----
    @cute.jit
    def _softmax_causal(self, basic, mma_p, sm_p, acc_S, nb, first: cutlass.Constexpr, diag):
        acc_S_mn = self._mn(acc_S)
        acc_O_mn = self._mn(mma_p.acc_O)
        if cutlass.const_expr(not first):
            row_max_prev = cute.make_fragment_like(sm_p.row_max, cutlass.Float32)
            cute.basic_copy(sm_p.row_max, row_max_prev)

        mcS = cute.make_identity_tensor(
            (basic.mQ.shape[0], basic.mQ.shape[1],
             basic.mQ.shape[2], basic.mK.shape[1]))
        cS = cute.local_tile(
            mcS[basic.batch, None, basic.head, None],
            (self._m_block_size, self._n_block_size),
            (basic.m_block, nb))
        tScS_mn = self._mn(mma_p.thr_mma.partition_C(cS))

        for r in cutlass.range_constexpr(cute.size(sm_p.row_max)):
            if diag:
                col_limit = tScS_mn[r, 0][1] + 1
                for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                    if cute.elem_less(col_limit, tScS_mn[0, c][3] + 1):
                        acc_S_mn[r, c] = -cutlass.Float32.inf
            s_row = acc_S_mn[r, None].load()
            cur = s_row.reduce(cute.ReductionOp.MAX, -cutlass.Float32.inf, 0)
            cur = self._quad_max(cur)
            if cutlass.const_expr(not first):
                cur = cute.arch.fmax(row_max_prev[r], cur)
            cur_safe = cur
            if diag:
                cur_safe = 0.0 if cur == -cutlass.Float32.inf else cur
            s_exp = cute.math.exp2(s_row * sm_p.scale_log2 - cur_safe * sm_p.scale_log2,
                                   fastmath=True)
            s_sum = s_exp.reduce(cute.ReductionOp.ADD, cutlass.Float32.zero, 0)
            if cutlass.const_expr(not first):
                prev = row_max_prev[r]
                delta = prev * sm_p.scale_log2 - cur_safe * sm_p.scale_log2
                corr = cute.math.exp2(delta, fastmath=True)
                if diag:
                    corr = 0.0 if prev == -cutlass.Float32.inf else corr
                s_sum = s_sum + sm_p.row_sum[r] * corr
                acc_O_mn[r, None] = acc_O_mn[r, None].load() * corr
            sm_p.row_max[r] = cur
            sm_p.row_sum[r] = s_sum
            acc_S_mn[r, None] = s_exp

    # ---- runtime-nb variant for split-KV decode (non-causal). nb/first/last runtime. ----
    @cute.jit
    def _one_block_rt(self, basic, mma_p, gcp, scp, sm_p, nb,
                      first: cutlass.Constexpr, last: cutlass.Constexpr):
        acc_S = cute.make_rmem_tensor(
            mma_p.thr_mma.partition_shape_C((self._m_block_size, self._n_block_size)),
            cutlass.Float32)
        acc_S.fill(0.0)
        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()
        cute.copy(gcp.gmem_tiled_copy_QKV, gcp.tVgV[None, None, None, nb], gcp.tVsV)
        cute.arch.cp_async_commit_group()

        cute.copy(scp.smem_tiled_copy_Q, scp.tSsQ[None, None, 0], scp.tSrQv[None, None, 0])
        cute.copy(scp.smem_tiled_copy_K, scp.tSsK[None, None, 0], scp.tSrKv[None, None, 0])
        for k in cutlass.range_constexpr(cute.size(scp.tSsQ.shape[2])):
            kn = (k + 1) % cute.size(scp.tSsQ.shape[2])
            cute.copy(scp.smem_tiled_copy_Q, scp.tSsQ[None, None, kn], scp.tSrQv[None, None, kn])
            cute.copy(scp.smem_tiled_copy_K, scp.tSsK[None, None, kn], scp.tSrKv[None, None, kn])
            cute.gemm(mma_p.tiled_mma, acc_S, mma_p.tSrQ[None, None, k],
                      mma_p.tSrK[None, None, k], acc_S)

        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()
        if cutlass.const_expr(not last):   # constexpr: prefetch next-lower K block
            cute.copy(gcp.gmem_tiled_copy_QKV, gcp.tKgK[None, None, None, nb - 1], gcp.tKsK)
            cute.arch.cp_async_commit_group()

        self._softmax_rt(mma_p, sm_p, acc_S, first)

        rP = cute.make_fragment_like(acc_S, self._dtype)
        rP.store(acc_S.load().to(self._dtype))
        rpd = cute.logical_divide(rP.layout, (None, None, 2))
        rp_view = cute.make_layout(
            ((rpd.shape[0], rpd.shape[2][0]), rpd.shape[1], rpd.shape[2][1]),
            stride=((rpd.stride[0], rpd.stride[2][0]), rpd.stride[1], rpd.stride[2][1]))
        tOrS = cute.make_tensor(rP.iterator, rp_view)
        cute.copy(scp.smem_tiled_copy_V, scp.tOsVt[None, None, 0], scp.tOrVtv[None, None, 0])
        for k in cutlass.range_constexpr(cute.size(tOrS.shape[2])):
            kn = (k + 1) % cute.size(tOrS.shape[2])
            cute.copy(scp.smem_tiled_copy_V, scp.tOsVt[None, None, kn], scp.tOrVtv[None, None, kn])
            cute.gemm(mma_p.tiled_mma, mma_p.acc_O, tOrS[None, None, k],
                      mma_p.tOrVt[None, None, k], mma_p.acc_O)

    # ---- online-softmax rescale for split-KV path; first is CONSTEXPR (non-causal). ----
    @cute.jit
    def _softmax_rt(self, mma_p, sm_p, acc_S, first: cutlass.Constexpr):
        acc_S_mn = self._mn(acc_S)
        acc_O_mn = self._mn(mma_p.acc_O)
        if cutlass.const_expr(not first):
            row_max_prev = cute.make_fragment_like(sm_p.row_max, cutlass.Float32)
            cute.basic_copy(sm_p.row_max, row_max_prev)
        for r in cutlass.range_constexpr(cute.size(sm_p.row_max)):
            s_row = acc_S_mn[r, None].load()
            cur = s_row.reduce(cute.ReductionOp.MAX, -cutlass.Float32.inf, 0)
            cur = self._quad_max(cur)
            if cutlass.const_expr(not first):
                cur = cute.arch.fmax(row_max_prev[r], cur)
            s_exp = cute.math.exp2(s_row * sm_p.scale_log2 - cur * sm_p.scale_log2,
                                   fastmath=True)
            s_sum = s_exp.reduce(cute.ReductionOp.ADD, cutlass.Float32.zero, 0)
            if cutlass.const_expr(not first):
                delta = row_max_prev[r] * sm_p.scale_log2 - cur * sm_p.scale_log2
                corr = cute.math.exp2(delta, fastmath=True)
                s_sum = s_sum + sm_p.row_sum[r] * corr
                acc_O_mn[r, None] = acc_O_mn[r, None].load() * corr
            sm_p.row_max[r] = cur
            sm_p.row_sum[r] = s_sum
            acc_S_mn[r, None] = s_exp

    # ---- per-split LSE = row_max*scale + ln(row_sum) (natural log), written to gmem ----
    #   mLSE shape [B*num_splits, Sq, Hq] (row-major). We recover each MN-row's global
    #   query coordinate from an identity tensor partitioned exactly like the C fragment
    #   (the same trick the causal mask uses), so the row<->lane mapping is exact instead
    #   of hand-rolled. The lane that owns column 0 of a given row writes its LSE. Write
    #   uses the DSL-safe make_tensor(iterator+off, static_layout)+autovec_copy idiom
    #   (scalar mLSE[i]=x crashes the backend).
    @cute.jit
    def _write_lse(self, basic, mma_p, mLSE, sm_p, bo, m_block, head, tidx):
        LOG2E = cutlass.Float32(LOG2_E)
        Hq = cute.size(mLSE.shape[2])
        Sq = cute.size(mLSE.shape[1])
        # identity over (B, Sq, Hq, Sk); tile the (M,N) block at (m_block, 0) and take its
        # MN view -> tScS_mn[r, c] carries the (b, q_row, h, k) coord of that fragment elem.
        mcS = cute.make_identity_tensor(
            (basic.mQ.shape[0], basic.mQ.shape[1], basic.mQ.shape[2], basic.mK.shape[1]))
        cS = cute.local_tile(mcS[basic.batch, None, basic.head, None],
                             (self._m_block_size, self._n_block_size), (m_block, 0))
        tScS_mn = self._mn(mma_p.thr_mma.partition_C(cS))
        for r in cutlass.range_constexpr(cute.size(sm_p.row_max)):
            rs = self._quad_sum(sm_p.row_sum[r])
            mx = sm_p.row_max[r]
            q_row = tScS_mn[r, 0][1]        # global query row for this MN-row
            col0 = tScS_mn[r, 0][3]         # this lane's first key column (0 for owners)
            if col0 == 0 and q_row < Sq:
                lse = -cutlass.Float32.inf
                if rs > 0.0:
                    lse = mx * sm_p.scale_log2 / LOG2E + cute.math.log(rs, fastmath=True)
                off = (bo * Sq + q_row) * Hq + head
                view = cute.make_tensor(mLSE.iterator + off,
                                        cute.make_layout((1,), stride=(1,)))
                frag = cute.make_rmem_tensor((1,), cutlass.Float32)
                frag[0] = lse
                cute.autovec_copy(frag, view)

    # ---- online softmax rescale ----
    @cute.jit
    def _softmax(self, basic, mma_p, sm_p, acc_S, nb: cutlass.Constexpr,
                 first: cutlass.Constexpr):
        acc_S_mn = self._mn(acc_S)
        acc_O_mn = self._mn(mma_p.acc_O)
        if cutlass.const_expr(not first):
            row_max_prev = cute.make_fragment_like(sm_p.row_max, cutlass.Float32)
            cute.basic_copy(sm_p.row_max, row_max_prev)

        # causal mask: build per-entry global (query,key) coords via identity tensor,
        # set S=-inf where key_col > query_row. Runs every block when causal (correct;
        # fully-below-diagonal blocks are a no-op, above-diagonal masked out entirely).
        tScS_mn = None
        if cutlass.const_expr(self._is_causal):
            mcS = cute.make_identity_tensor(
                (basic.mQ.shape[0], basic.mQ.shape[1],
                 basic.mQ.shape[2], basic.mK.shape[1]))
            cS = cute.local_tile(
                mcS[basic.batch, None, basic.head, None],
                (self._m_block_size, self._n_block_size),
                (basic.m_block, nb))
            tScS_mn = self._mn(mma_p.thr_mma.partition_C(cS))

        for r in cutlass.range_constexpr(cute.size(sm_p.row_max)):
            if cutlass.const_expr(self._is_causal):
                # key positions strictly greater than this query row are future -> -inf
                col_limit = tScS_mn[r, 0][1] + 1
                for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                    if cute.elem_less(col_limit, tScS_mn[0, c][3] + 1):
                        acc_S_mn[r, c] = -cutlass.Float32.inf
            s_row = acc_S_mn[r, None].load()
            cur = s_row.reduce(cute.ReductionOp.MAX, -cutlass.Float32.inf, 0)
            cur = self._quad_max(cur)
            if cutlass.const_expr(not first):
                cur = cute.arch.fmax(row_max_prev[r], cur)
            # when the whole (masked) row is -inf, exp2(-inf - -inf)=exp2(nan); guard by
            # using 0 as the subtractor only for the exponent, so exp2(-inf)=0 cleanly.
            cur_safe = cur
            if cutlass.const_expr(self._is_causal):
                cur_safe = 0.0 if cur == -cutlass.Float32.inf else cur
            s_exp = cute.math.exp2(s_row * sm_p.scale_log2 - cur_safe * sm_p.scale_log2,
                                   fastmath=True)
            s_sum = s_exp.reduce(cute.ReductionOp.ADD, cutlass.Float32.zero, 0)
            if cutlass.const_expr(not first):
                # corr rescales the running accumulator to the new max. If the previous
                # max was -inf (row fully masked so far) OR the new max is -inf (this row
                # still fully masked), the correction is 0 (nothing to carry) -> avoid the
                # inf-inf = NaN that exp2 would otherwise produce.
                prev = row_max_prev[r]
                delta = prev * sm_p.scale_log2 - cur_safe * sm_p.scale_log2
                corr = cute.math.exp2(delta, fastmath=True)
                if cutlass.const_expr(self._is_causal):
                    corr = 0.0 if prev == -cutlass.Float32.inf else corr
                s_sum = s_sum + sm_p.row_sum[r] * corr
                acc_O_mn[r, None] = acc_O_mn[r, None].load() * corr
            sm_p.row_max[r] = cur
            sm_p.row_sum[r] = s_sum
            acc_S_mn[r, None] = s_exp

    @cute.jit
    def _normalize(self, acc_O, row_sum):
        acc_O_mn = self._mn(acc_O)
        for r in cutlass.range_constexpr(cute.size(row_sum)):
            rs = self._quad_sum(row_sum[r])
            bad = rs == 0.0 or rs != rs
            scale = 1.0 if bad else cute.arch.rcp_approx(rs)
            acc_O_mn[r, None] = acc_O_mn[r, None].load() * scale

    def _mn(self, acc):
        c = cute.make_layout(acc.layout.shape)
        mn = cute.make_layout(
            ((c.shape[0][1], c.shape[1]), (c.shape[0][0], c.shape[2])),
            stride=((c.stride[0][1], c.stride[1]), (c.stride[0][0], c.stride[2])))
        return cute.make_tensor(acc.iterator, cute.composition(acc.layout, mn))

    def _quad(self, val, op):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=2, mask=-1, mask_and_clamp=31))
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1, mask=-1, mask_and_clamp=31))
        return val

    def _quad_max(self, v):
        return self._quad(v, lambda x, y: cute.arch.fmax(x, y))

    def _quad_sum(self, v):
        return self._quad(v, lambda x, y: x + y)


# --------------------------------------------------------------------------
# Python entry: stage (paged) KV -> contiguous view, run the CuTe DSL kernel
# --------------------------------------------------------------------------
_COMPILE_CACHE = {}


def cute_attention(q, k, v, sm_scale=None, is_causal=False,
                   m_block=64, n_block=64, num_threads=128, min_blocks_per_mp=0):
    """q,k,v: [B, S, H, D] bf16 on cuda. GQA: k/v may have Hkv<=Hq heads.
    Returns o [B, Sq, Hq, D]. head_dim must be padded-mult-of-32 (128 ok).
    num_threads (warps=num_threads/32) tunes MMA-row count / softmax latency hiding.
    min_blocks_per_mp sets __launch_bounds__ minCTAperSM: ncu shows prefill is
    latency-bound at 12.5% occupancy (capped at 2 CTAs/SM by 186 regs/thread), so
    forcing 3-4 CTAs/SM by capping registers is the single biggest prefill lever."""
    assert q.dtype == torch.bfloat16
    B, Sq, Hq, D = q.shape
    Hkv = k.shape[2]
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)
    o = torch.empty_like(q)
    Sk = k.shape[1]

    # --- Hopper warp-specialized TMA+WGMMA fast path -----------------------
    # Route supported bf16/D{64,128} prefill shapes to the vendored NVIDIA
    # warp-specialized TMA FMHA reference (130+ TFLOP/s, matches/beats FA3,
    # handles causal). SM80/WGMMA paths remain for decode/edge shapes.
    # Toggle off for A/B with env _NO_FMHA_REF=1.
    if os.environ.get("_NO_FMHA_REF", "0") == "0" and Sq >= 128:
        try:
            from paged_fa3._fmha_ref.adapter import (
                can_use_fmha_ref, fmha_ref_attention)
            if can_use_fmha_ref(q, k, v, is_causal):
                return fmha_ref_attention(q, k, v, sm_scale=sm_scale,
                                          is_causal=is_causal)
        except Exception as _e:
            if os.environ.get("_FMHA_REF_DEBUG", "0") == "1":
                import traceback; traceback.print_exc()
            # fall through to the hand-written path on any failure
    # -----------------------------------------------------------------------

    fa = CuteFlashAttentionFwd(head_dim=D, seqlen_q=Sq, seqlen_k=Sk,
                               m_block_size=m_block, n_block_size=n_block,
                               num_threads=num_threads,
                               is_causal=is_causal,
                               min_blocks_per_mp=min_blocks_per_mp)
    # 128-bit-aligned dynamic layout: mark head_dim (mode 3) dynamic AND declare its
    # stride is divisible by 8 elems (=128 bits for bf16) so the cp.async 128b atom is legal.
    div = 128 // torch.finfo(q.dtype).bits
    def _mk(t):
        return (from_dlpack(t, assumed_align=16)
                .mark_layout_dynamic(leading_dim=3)
                .mark_compact_shape_dynamic(
                    mode=3, stride_order=t.dim_order(), divisibility=div))
    mQ = _mk(q); mK = _mk(k); mV = _mk(v); mO = _mk(o)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    key = (B, Sq, Sk, Hq, Hkv, D, is_causal, m_block, n_block, num_threads,
           min_blocks_per_mp)
    if key not in _COMPILE_CACHE:
        _COMPILE_CACHE[key] = cute.compile(
            fa, mQ, mK, mV, mO, cutlass.Float32(sm_scale), stream)
    _COMPILE_CACHE[key](mQ, mK, mV, mO, cutlass.Float32(sm_scale), stream)
    return o


def cute_decode_split(q, k, v, sm_scale=None, num_splits=None,
                      m_block=64, n_block=64):
    """Split-KV (flash-decoding) attention in hand-written CuTe DSL.

    q,k,v: [B, Sq, Hq/Hkv, D] bf16 cuda, contiguous KV, NON-causal (decode reads a few
    queries against the whole past). Each CTA owns a slice of the KV n-blocks so the
    grid (m_blocks*num_splits, B, Hq) fills all SMs when Sq is tiny. The kernel writes
    unnormalized partial O and per-split LSE; the LSE-combine reduction runs in torch
    (negligible cost vs the memory-bound attention kernel).

    Returns o [B, Sq, Hq, D]."""
    assert q.dtype == torch.bfloat16
    B, Sq, Hq, D = q.shape
    Hkv = k.shape[2]
    Sk = k.shape[1]
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)
    n_block_total = (Sk + n_block - 1) // n_block
    if num_splits is None:
        # target ~ enough CTAs to fill ~78 SMs given B*Hq base blocks; cap by n-blocks.
        base = max(1, B * Hq)
        target = max(1, (128 + base - 1) // base)   # aim >=128 CTAs
        num_splits = min(n_block_total, max(1, target))
    num_splits = max(1, min(num_splits, n_block_total))

    op = torch.empty(B * num_splits, Sq, Hq, D, dtype=q.dtype, device=q.device)
    lse = torch.full((B * num_splits, Sq, Hq), float("-inf"),
                     dtype=torch.float32, device=q.device)

    fa = CuteFlashAttentionFwd(head_dim=D, seqlen_q=Sq, seqlen_k=Sk,
                               m_block_size=m_block, n_block_size=n_block,
                               is_causal=False, num_splits=num_splits)
    div = 128 // torch.finfo(q.dtype).bits
    def _mk(t):
        return (from_dlpack(t, assumed_align=16)
                .mark_layout_dynamic(leading_dim=3)
                .mark_compact_shape_dynamic(
                    mode=3, stride_order=t.dim_order(), divisibility=div))
    mQ = _mk(q); mK = _mk(k); mV = _mk(v); mO = _mk(op)
    mLSE = from_dlpack(lse, assumed_align=16).mark_layout_dynamic(leading_dim=2)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    key = ("split", B, Sq, Sk, Hq, Hkv, D, num_splits, m_block, n_block)
    if key not in _COMPILE_CACHE:
        _COMPILE_CACHE[key] = cute.compile(
            fa, mQ, mK, mV, mO, cutlass.Float32(sm_scale), stream, mLSE)
    _COMPILE_CACHE[key](mQ, mK, mV, mO, cutlass.Float32(sm_scale), stream, mLSE)

    # LSE-combine over splits, in fp32. Each split wrote a NORMALIZED softmax output O_s
    # and lse_s = m_s + ln(l_s). Standard flash-decoding reduction:
    #     O = sum_s softmax_s(lse_s) . O_s
    op = op.view(B, num_splits, Sq, Hq, D).float()
    lse = lse.view(B, num_splits, Sq, Hq).float()          # [B,ns,Sq,Hq]
    # A split that saw zero (or all-masked) keys leaves its partial O / LSE
    # uninitialized: the kernel may write NaN there. Sanitize BOTH before the
    # combine. A NaN LSE must become -inf (so amax ignores it and its softmax
    # weight is 0); a NaN partial-O must become 0 (it carries weight 0 anyway,
    # but 0 * NaN = NaN would otherwise poison the weighted sum). Both rewrites
    # are mathematically exact because such splits contribute nothing.
    lse = torch.nan_to_num(lse, nan=float("-inf"), posinf=float("-inf"))
    op = torch.nan_to_num(op, nan=0.0, posinf=0.0, neginf=0.0)
    m_star = lse.amax(dim=1, keepdim=True)                 # [B,1,Sq,Hq]
    m_star = torch.where(torch.isinf(m_star), torch.zeros_like(m_star), m_star)
    w = torch.exp(lse - m_star)                            # [B,ns,Sq,Hq]
    denom = w.sum(dim=1, keepdim=True)                     # [B,1,Sq,Hq]
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)
    p = (w / denom).unsqueeze(-1)                          # softmax over splits [B,ns,Sq,Hq,1]
    o = (p * op).sum(dim=1).to(q.dtype)                    # [B,Sq,Hq,D]
    return o


def paged_cute_attention(q, k_cache, v_cache, block_table, seq_lens,
                         sm_scale=None, is_causal=False):
    """Paged wrapper: gather paged KV [num_blocks,PAGE,Hkv,D] via block_table into a
    contiguous [B,S,Hkv,D] view, then run the hand-written CuTe DSL attention.
    q: [B, Sq, Hq, D]."""
    B, Sq, Hq, D = q.shape
    num_blocks, PAGE, Hkv, _ = k_cache.shape
    S = int(seq_lens.max().item())
    kc = torch.empty(B, S, Hkv, D, dtype=k_cache.dtype, device=q.device)
    vc = torch.empty(B, S, Hkv, D, dtype=v_cache.dtype, device=q.device)
    for b in range(B):
        nb = (int(seq_lens[b].item()) + PAGE - 1) // PAGE
        ids = block_table[b, :nb]
        kc[b, :nb * PAGE] = k_cache[ids].reshape(-1, Hkv, D)[:S]
        vc[b, :nb * PAGE] = v_cache[ids].reshape(-1, Hkv, D)[:S]
    return cute_attention(q, kc, vc, sm_scale=sm_scale, is_causal=is_causal)
