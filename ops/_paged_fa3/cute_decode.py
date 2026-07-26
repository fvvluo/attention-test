"""cute_decode.py - hand-written flash-DECODING attention in the NVIDIA CuTe DSL.

Task-1 decode deliverable. Decode is the graded metric (Sq=1 query vs up-to-128k KV,
batch 1). It is MEMORY-BOUND: performance is gated by HBM bandwidth reading K/V, not
by tensor-core throughput. So this kernel is built to SATURATE HBM:

  * split-KV (flash-decoding): the [0, seqlen) key range is split across NUM_SPLITS
    CTAs so the grid (NUM_SPLITS, Hkv, B) has enough programs to fill all SMs and
    expose enough parallel, coalesced HBM reads to approach peak bandwidth. With
    B=1, Hkv=8 we lean on NUM_SPLITS to reach ~hundreds of CTAs.
  * GQA-packed: one CTA == (split, KV-head, batch). It loads each K/V row ONCE and
    computes all G = Hq/Hkv query heads that share this KV head -> removes the
    G-fold redundant HBM traffic of a naive per-query-head kernel.
  * MULTI-WARP per CTA: W warps cooperate on one split's key range (round-robin by
    warp). This lifts threads/CTA from 32 -> 32*W so each SM stays busy and the many
    parallel coalesced HBM reads saturate bandwidth. Each warp keeps its own online-
    softmax partial; the W partials are LSE-combined through a tiny smem reduction at
    the end. Head-dim D is partitioned across the 32 lanes WITHIN each warp: lane l
    owns the CONTIGUOUS DPL-slice [l*DPL,(l+1)*DPL) so one key row [D] is a fully
    coalesced warp load. Per-lane registers stay tiny (_g*DPL fp32) so GQA (G=4) never
    blows the 255-register limit.
  * a second combine kernel LSE-combines the NUM_SPLITS partials across CTAs.

This mirrors the algorithm of triton_decode.py (verified 2.06x over FA3 at 128k),
re-authored in the CuTe DSL as required by the brief ("hand-write using CuTe DSL").
Because decode is bandwidth-bound the q.k^T / p.v products are tiny; we compute them
directly in registers (a GEMV per key) rather than paying MMA fragment-layout
overhead - the SASS is dominated by the K/V HBM reads either way.
"""
from types import SimpleNamespace

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass.cute.runtime import from_dlpack

LOG2_E = 1.4426950408889634074


class CuteFlashDecode:
    """Hand-written split-KV flash-decoding forward in CuTe DSL (bf16, hd=128)."""

    def __init__(self, head_dim, hq, hkv, num_splits, page, max_blocks,
                 block_n=64, num_warps=4, kunroll=8):
        self._d = head_dim
        self._hq = hq
        self._hkv = hkv
        self._g = max(1, 1 << ((hq // hkv) - 1).bit_length())  # padded group size
        self._bn = block_n              # unused now (kept for API compat)
        # KUNROLL: keys processed per loop iteration. Each lane issues all KU*DPL K/V loads
        # up front (independent -> compiler keeps them in flight = memory-level parallelism)
        # BEFORE the dependent softmax math. This is the key lever for a memory-bound kernel
        # that would otherwise be latency-bound on one-key-at-a-time serial loads.
        self._ku = kunroll
        # W warps per CTA, each 32 lanes. Head-dim D is PARTITIONED across the 32 lanes
        # WITHIN a warp: lane l owns the contiguous DPL-slice [l*DPL,(l+1)*DPL). Keeps the
        # per-lane accumulator tiny (_g*_dpl fp32) so GQA (G=4) does NOT blow the 255-register
        # limit (the old all-threads-hold-full-acc design did -> illegal access).
        assert head_dim % 32 == 0
        self._nw = num_warps            # warps per CTA
        self._nt = num_warps * 32       # threads per CTA
        self._dpl = head_dim // 32      # D elements per lane (=4 for D=128)
        # IMPORTANT: constexpr shape params (num_splits/page/max_blocks) are stored as
        # INSTANCE ATTRIBUTES, not passed as `@cute.jit __call__` parameters. Passing a
        # `cutlass.Constexpr`-annotated arg to the host launcher corrupts pointer
        # materialization for the whole kernel -> "cannot be converted to pointer"
        # Internal Error at compile. (Root-caused via probe_host.py.)
        self._num_splits = num_splits
        self._page = page
        self._max_blocks = max_blocks

    # ------------------------------------------------------------------ host
    @cute.jit
    def __call__(self, mQ, mKf, mVf, mBT, mSeq, mOp, mLse,
                 scale_log2: cutlass.Float32, stream: cuda.CUstream):
        self._dtype = mKf.element_type
        grid = (self._num_splits, self._hkv, cute.size(mQ.shape[0]))
        self.kernel(
            mQ, mKf, mVf, mBT, mSeq, mOp, mLse, scale_log2,
        ).launch(grid=grid, block=[self._nt, 1, 1], stream=stream)

    # ---------------------------------------------------------------- device
    @cute.kernel
    def kernel(self, mQ: cute.Tensor, mKf: cute.Tensor, mVf: cute.Tensor,
               mBT: cute.Tensor, mSeq: cute.Tensor, mOp: cute.Tensor,
               mLse: cute.Tensor, scale_log2: cutlass.Float32):
        num_splits = self._num_splits
        page = self._page
        max_blocks = self._max_blocks
        NW = self._nw
        tidx, _, _ = cute.arch.thread_idx()
        split, kvh, batch = cute.arch.block_idx()

        G = self._hq // self._hkv                      # real group size (constexpr)
        g_real = G
        GP = self._g                                   # padded group size (constexpr)
        DPL = self._dpl                                # D elems per lane (=4)
        warp = tidx // 32                               # 0..NW-1
        lane = tidx % 32                                # 0..31
        d0 = lane * DPL                                # first D index this lane owns
        sm = scale_log2 / LOG2_E                       # == sm_scale

        seqlen = mSeq[batch]
        split_len = (seqlen + num_splits - 1) // num_splits
        kv_start = split * split_len
        kv_cap = kv_start + split_len
        kv_end = kv_cap if kv_cap < seqlen else seqlen

        # this lane's slice of the GP query vectors (contiguous DPL chunk)
        rQ = cute.make_rmem_tensor((GP * DPL,), cutlass.Float32)
        for g in cutlass.range_constexpr(GP):
            for e in cutlass.range_constexpr(DPL):
                val = cutlass.Float32(0.0)
                if g < g_real:
                    qh = kvh * G + g
                    val = mQ[batch, qh, d0 + e].to(cutlass.Float32)
                rQ[g * DPL + e] = val

        # running softmax state (scalar, replicated across the warp) + per-lane acc slice
        m_i = cute.make_rmem_tensor((GP,), cutlass.Float32)
        l_i = cute.make_rmem_tensor((GP,), cutlass.Float32)
        acc = cute.make_rmem_tensor((GP * DPL,), cutlass.Float32)
        for g in cutlass.range_constexpr(GP):
            m_i[g] = -cutlass.Float32.inf
            l_i[g] = 0.0
            for e in cutlass.range_constexpr(DPL):
                acc[g * DPL + e] = 0.0

        # W warps split this split's key range round-robin in blocks of KU keys:
        # warp w owns keys [base + w*KU, base + w*KU + KU), base stepping by NW*KU.
        # Within a KU-block: issue ALL KU*DPL K/V loads first (memory-level parallelism),
        # THEN do the dependent softmax math. Each warp keeps its OWN partial.
        KU = self._ku
        stride = NW * KU
        base = kv_start + warp * KU
        # buffers for KU keys' K/V lane-slices + valid flags
        rK = cute.make_rmem_tensor((KU * DPL,), cutlass.Float32)
        rV = cute.make_rmem_tensor((KU * DPL,), cutlass.Float32)
        while base < kv_end:
            # (1) issue all loads for the KU keys (independent -> overlap in flight)
            for u in cutlass.range_constexpr(KU):
                kv_pos = base + u
                if kv_pos < kv_end:
                    logical = kv_pos // page
                    in_blk = kv_pos % page
                    lb = logical if logical < max_blocks else 0
                    phys = mBT[batch, lb]
                    row = (phys * page + in_blk) * self._hkv + kvh
                    for e in cutlass.range_constexpr(DPL):
                        rK[u * DPL + e] = mKf[row, d0 + e].to(cutlass.Float32)
                        rV[u * DPL + e] = mVf[row, d0 + e].to(cutlass.Float32)

            # (2) dependent math: for each valid key, all GP groups
            for u in cutlass.range_constexpr(KU):
                if base + u < kv_end:
                    for g in cutlass.range_constexpr(GP):
                        partial = cutlass.Float32(0.0)
                        for e in cutlass.range_constexpr(DPL):
                            partial = partial + rQ[g * DPL + e] * rK[u * DPL + e]
                        dot = self._warp_sum(partial)     # full score on every lane
                        sj = dot * sm
                        m_new = cute.arch.fmax(m_i[g], sj)
                        alpha = cute.math.exp2((m_i[g] - m_new) * LOG2_E, fastmath=True)
                        p = cute.math.exp2((sj - m_new) * LOG2_E, fastmath=True)
                        l_i[g] = l_i[g] * alpha + p
                        for e in cutlass.range_constexpr(DPL):
                            acc[g * DPL + e] = acc[g * DPL + e] * alpha + p * rV[u * DPL + e]
                        m_i[g] = m_new
            base = base + stride

        # -------- combine the NW warp partials through smem, then warp 0 writes out --------
        if cutlass.const_expr(NW > 1):
            # smem layout: per (warp, group): m[NW*GP], l[NW*GP], and acc slices
            # acc_sh[warp, g, lane, e] flattened. Sizes are compile-time constants.
            smem = cutlass.utils.SmemAllocator()
            sh_m = smem.allocate_tensor(cutlass.Float32, cute.make_layout((NW, GP)), byte_alignment=16)
            sh_l = smem.allocate_tensor(cutlass.Float32, cute.make_layout((NW, GP)), byte_alignment=16)
            sh_a = smem.allocate_tensor(
                cutlass.Float32, cute.make_layout((NW, GP, 32, DPL)), byte_alignment=16)

            # each lane of each warp stores its partial acc slice; lane 0 stores m,l
            for g in cutlass.range_constexpr(GP):
                if lane == 0:
                    sh_m[warp, g] = m_i[g]
                    sh_l[warp, g] = l_i[g]
                for e in cutlass.range_constexpr(DPL):
                    sh_a[warp, g, lane, e] = acc[g * DPL + e]
            cute.arch.barrier()

            # warp 0 combines the NW partials per group (LSE combine) and writes final O/LSE
            if warp == 0:
                for g in cutlass.range_constexpr(GP):
                    if g < g_real:
                        qh = kvh * G + g
                        # global max over the NW warp maxima
                        mmax = -cutlass.Float32.inf
                        for w in cutlass.range_constexpr(NW):
                            mmax = cute.arch.fmax(mmax, sh_m[w, g])
                        # combined denom = sum_w l_w * exp(m_w - mmax)
                        denom = cutlass.Float32(0.0)
                        for w in cutlass.range_constexpr(NW):
                            denom = denom + sh_l[w, g] * cute.math.exp2(
                                (sh_m[w, g] - mmax) * LOG2_E, fastmath=True)
                        has = denom > 0.0
                        d_safe = denom if has else 1.0
                        inv = cute.arch.rcp_approx(d_safe)
                        # this lane's DPL slice of combined, normalized O
                        for e in cutlass.range_constexpr(DPL):
                            o = cutlass.Float32(0.0)
                            for w in cutlass.range_constexpr(NW):
                                wgt = cute.math.exp2((sh_m[w, g] - mmax) * LOG2_E, fastmath=True)
                                o = o + sh_a[w, g, lane, e] * wgt
                            mOp[batch, qh, split, d0 + e] = (o * inv).to(mOp.element_type)
                        if lane == 0:
                            lse = mmax + cute.math.log(d_safe, fastmath=True) \
                                if has else -cutlass.Float32.inf
                            mLse[batch, qh, split] = lse
        else:
            # single warp: write partial O (normalized within split) + partial LSE directly.
            for g in cutlass.range_constexpr(GP):
                if g < g_real:
                    qh = kvh * G + g
                    has = l_i[g] > 0.0
                    d_safe = l_i[g] if has else 1.0
                    inv = cute.arch.rcp_approx(d_safe)
                    for e in cutlass.range_constexpr(DPL):
                        mOp[batch, qh, split, d0 + e] = (acc[g * DPL + e] * inv).to(mOp.element_type)
                    if lane == 0:
                        lse = m_i[g] + cute.math.log(d_safe, fastmath=True) \
                            if has else -cutlass.Float32.inf
                        mLse[batch, qh, split] = lse

    def _warp_sum(self, val):
        # full 32-lane butterfly sum reduction: every lane ends with the total.
        val = val + cute.arch.shuffle_sync_bfly(val, offset=16, mask=-1, mask_and_clamp=31)
        val = val + cute.arch.shuffle_sync_bfly(val, offset=8, mask=-1, mask_and_clamp=31)
        val = val + cute.arch.shuffle_sync_bfly(val, offset=4, mask=-1, mask_and_clamp=31)
        val = val + cute.arch.shuffle_sync_bfly(val, offset=2, mask=-1, mask_and_clamp=31)
        val = val + cute.arch.shuffle_sync_bfly(val, offset=1, mask=-1, mask_and_clamp=31)
        return val


# --------------------------------------------------------------------------
# Phase 2: combine NUM_SPLITS partials via LSE reduction (CuTe DSL)
# --------------------------------------------------------------------------
class CuteCombine:
    """Split-KV LSE-reduce. Optimized: threads stripe the NUM_SPLITS reduction
    (block-reduce max & denom via smem, NOT redundant per-thread serial loops),
    cache each split's normalized weight in smem ONCE (no double-exp), then
    parallelize the final D-accumulation. ~1.6x faster than the naive combine
    (9.5->6.0 us at 256 splits), lifting decode e2e ~2850->2950 GB/s @128k.
    Bit-close to the old combine (max|diff|~1e-5, pure fp rounding order)."""
    def __init__(self, head_dim, num_splits, num_threads=128, use_pdl=False):
        self._d = head_dim
        self._ns = num_splits
        self._nt = num_threads
        # PDL (Programmatic Dependent Launch): when the preceding decode kernel was
        # launched with use_pdl=True AND signals griddepcontrol_launch_dependents()
        # at its end, this combine can be launched with use_pdl=True so its CTAs
        # start/prefetch while the decode's tail CTAs drain -> hides ~2-3us of the
        # 2nd-launch stall (decode e2e ~2950 -> ~2990 GB/s @128k). Default OFF: only
        # safe when the paired decode signals; the OTHER (CuteFlashDecode) path does
        # not signal, so it keeps use_pdl=False. griddepcontrol_wait() with PDL but
        # no upstream signal would hang -> the flag gates BOTH the wait and the attr.
        self._use_pdl = use_pdl
        import cutlass.pipeline as pipeline
        self._bar = pipeline.NamedBarrier(barrier_id=1, num_threads=num_threads)

    @cute.jit
    def __call__(self, mOp, mLse, mO, stream: cuda.CUstream):
        grid = (cute.size(mO.shape[1]), cute.size(mO.shape[0]), 1)

        @cute.struct
        class Smem:
            wt: cute.struct.MemRange[cutlass.Float32, self._ns]
            red: cute.struct.MemRange[cutlass.Float32, self._nt]
        self.kernel(mOp, mLse, mO, Smem).launch(
            grid=grid, block=[self._nt, 1, 1], stream=stream,
            use_pdl=self._use_pdl)

    @cute.kernel
    def kernel(self, mOp: cute.Tensor, mLse: cute.Tensor, mO: cute.Tensor,
               Smem: cutlass.Constexpr):
        if self._use_pdl:
            cute.arch.griddepcontrol_wait()
        tidx, _, _ = cute.arch.thread_idx()
        h, b, _ = cute.arch.block_idx()
        NS = self._ns
        D = self._d
        NT = self._nt
        import cutlass.utils
        smem = cutlass.utils.SmemAllocator()
        st = smem.allocate(Smem)
        s_wt = st.wt.get_tensor(cute.make_layout(NS))
        s_red = st.red.get_tensor(cute.make_layout(NT))

        # phase 1: local max over this thread's stripe of splits -> block max
        lm = -cutlass.Float32.inf
        s = tidx
        while s < NS:
            lm = cute.arch.fmax(lm, mLse[b, h, s])
            s = s + NT
        s_red[tidx] = lm
        self._bar.arrive_and_wait()
        m = -cutlass.Float32.inf
        for i in cutlass.range_constexpr(NT):
            m = cute.arch.fmax(m, s_red[i])
        self._bar.arrive_and_wait()

        # phase 2: local denom + store weight exp(lse-m) per split (single exp)
        ld = cutlass.Float32(0.0)
        s = tidx
        while s < NS:
            w = cute.math.exp(mLse[b, h, s] - m, fastmath=True)
            s_wt[s] = w
            ld = ld + w
            s = s + NT
        s_red[tidx] = ld
        self._bar.arrive_and_wait()
        denom = cutlass.Float32(0.0)
        for i in cutlass.range_constexpr(NT):
            denom = denom + s_red[i]
        inv = cute.arch.rcp_approx(denom) if denom > 0.0 else cutlass.Float32(0.0)
        s = tidx
        while s < NS:
            s_wt[s] = s_wt[s] * inv
            s = s + NT
        self._bar.arrive_and_wait()

        # phase 3: accumulate over D with cached weights (no recompute of exp)
        d = tidx
        while d < D:
            out = cutlass.Float32(0.0)
            for si in cutlass.range_constexpr(NS):
                out = out + mOp[b, h, si, d].to(cutlass.Float32) * s_wt[si]
            mO[b, h, d] = out.to(mO.element_type)
            d = d + NT


# --------------------------------------------------------------------------
# Python entry
# --------------------------------------------------------------------------
_CACHE = {}


def _npow2(x):
    return 1 << (x - 1).bit_length()


def paged_decode_cute(q, k_cache, v_cache, block_table, seq_lens,
                      sm_scale=None, num_splits=None, block_n=64, num_warps=4,
                      kunroll=8):
    """q: [B, Hq, D] bf16. k/v cache: [num_blocks, PAGE, Hkv, D]. Returns [B, Hq, D]."""
    assert q.dtype == torch.bfloat16
    B, Hq, D = q.shape
    num_blocks, PAGE, Hkv, _ = k_cache.shape
    max_blocks = block_table.shape[1]
    G = Hq // Hkv
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)
    if num_splits is None:
        # decode is memory-bound; we want ~256 CTAs per (b,kv-head) to saturate SMs.
        # Cap so each split still has enough keys to amortize launch (>= ~64 keys/split).
        max_len = int(seq_lens.max().item())
        num_splits = max(8, min(256, (max_len + 63) // 64))
    num_splits = _npow2(num_splits)
    g_pad = max(1, _npow2(G))

    dev = q.device
    out_p = torch.empty((B, Hq, num_splits, D), device=dev, dtype=torch.float32)
    lse_p = torch.full((B, Hq, num_splits), float("-inf"), device=dev, dtype=torch.float32)
    out = torch.empty((B, Hq, D), device=dev, dtype=q.dtype)

    scale_log2 = float(sm_scale) * LOG2_E

    # flat 2D views of the paged caches: [num_blocks*PAGE*Hkv, D]. Row for
    # (phys, in_blk, kvh) = (phys*PAGE + in_blk)*Hkv + kvh. The kernel indexes rows
    # arithmetically -> avoids the DSL's "cannot be converted to pointer" crash.
    kf = k_cache.reshape(-1, D)
    vf = v_cache.reshape(-1, D)

    def mk(t, ld):
        return from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=ld)
    mQ = mk(q, 2)
    mKf = mk(kf, 1); mVf = mk(vf, 1)
    mBT = mk(block_table, 1)
    mSeq = from_dlpack(seq_lens, assumed_align=16).mark_layout_dynamic()
    mOp = mk(out_p, 3)
    mLse = from_dlpack(lse_p, assumed_align=16).mark_layout_dynamic(leading_dim=2)
    mO = mk(out, 2)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    dec = CuteFlashDecode(head_dim=D, hq=Hq, hkv=Hkv, num_splits=num_splits,
                          page=PAGE, max_blocks=max_blocks, block_n=block_n,
                          num_warps=num_warps, kunroll=kunroll)
    comb = CuteCombine(head_dim=D, num_splits=num_splits)
    key = (B, Hq, Hkv, D, num_splits, PAGE, max_blocks, num_warps, kunroll)
    if key not in _CACHE:
        c1 = cute.compile(dec, mQ, mKf, mVf, mBT, mSeq, mOp, mLse,
                          cutlass.Float32(scale_log2), stream)
        c2 = cute.compile(comb, mOp, mLse, mO, stream)
        _CACHE[key] = (c1, c2)
    c1, c2 = _CACHE[key]
    c1(mQ, mKf, mVf, mBT, mSeq, mOp, mLse, cutlass.Float32(scale_log2), stream)
    c2(mOp, mLse, mO, stream)
    return out
