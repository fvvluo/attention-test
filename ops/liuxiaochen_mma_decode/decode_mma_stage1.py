#!/usr/bin/env python3
"""B5: warp-level Tensor-Core (m16n8k16) GQA Split-KV decode Stage-1.

Independent implementation authored by Liu Xiaochen.

Design reference (idea only, no code copied): the M16 Pack-GQA mapping — putting
a KV head's 8 query rows into the first 8 rows of a 16-row warp-MMA tile — was
learned from reading quanbofeng's decode during the code-study phase. The kernel
structure, layouts, softmax bookkeeping, split scheduling, workspace format and
combine here are written from scratch using the public CUTLASS CuTe DSL
primitives (make_tiled_mma / make_fragment_A|B / make_tiled_copy_A|B / ldmatrix
atoms / cute.gemm). This first version uses a SYNCHRONOUS shared-memory load
(no cp.async double-buffering); async prefetch is deferred to a later phase.

Fixed target: B=1, Hq=64, Hkv=8, q_len=1, D=128, BF16, causal=False,
kv_len % N_BLOCK == 0.

Per CTA (32 threads = 1 warp): one (split, kv_head, batch).
  - The 8 query heads of this kv_head occupy rows 0..7 of an M16 MMA tile;
    rows 8..15 are zero padding (never read out-of-bounds Q, never written).
  - QK: MMA A = Q[16,128], B = K[64,128] (K-major) -> scores[16,64] FP32.
  - PV: MMA A = P[16,64] (BF16), B = V^T[128,64] -> O[16,128] FP32.
Both QK and PV use the Tensor Core; there is no SIMT dot/FMA main path.

Stage-1 writes normalized-O (BF16) + LSE (FP32) partials per split.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import warp

LOG2E = 1.4426950408889634


class LiuXiaochenMmaDecodeStage1B5:
    """M16 GQA-packed warp-MMA split-KV decode Stage-1 (synchronous load)."""

    def __init__(self, head_dim, q_heads, kv_heads, seqlen_k, num_splits, n_block=64):
        self.head_dim = head_dim
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.heads_per_kv = q_heads // kv_heads       # 8
        self.mma_m = 16
        self.n_block = n_block                         # 64
        self.seqlen_k = seqlen_k
        self.num_splits = num_splits
        self.num_n_blocks = seqlen_k // n_block
        if self.num_n_blocks % num_splits != 0:
            raise ValueError("num_n_blocks must be divisible by num_splits")
        self.blocks_per_split = self.num_n_blocks // num_splits
        self.num_threads = 32

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,        # [B, Hq, D]        (q_len squeezed)
        mK: cute.Tensor,        # [B, Sk, Hkv, D]   (BSHD view)
        mV: cute.Tensor,        # [B, Sk, Hkv, D]
        mPartial: cute.Tensor,  # [B, Hq, split, D] FP32
        mLSE: cute.Tensor,      # [B, Hq, split]    FP32
        scale_log2: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        self.dtype = mQ.element_type
        D = self.head_dim
        NB = self.n_block

        # Shared layouts (row-major, swizzled for ldmatrix). smem_k=64 covers D=128 in 2 chunks.
        smem_k = 64
        smem_atom = cute.make_composed_layout(
            cute.make_swizzle(3, 3, 3),
            0,
            cute.make_layout((8, smem_k), stride=(smem_k, 1)),
        )
        q_smem_layout = cute.tile_to_shape(smem_atom, (self.mma_m, D), (0, 1))
        kv_smem_layout = cute.tile_to_shape(smem_atom, (NB, D), (0, 1))

        @cute.struct
        class Smem:
            sQ: cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(q_smem_layout)], 1024]
            sK: cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(kv_smem_layout)], 1024]
            sV: cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(kv_smem_layout)], 1024]

        # 128-bit cooperative G2S copy atoms (synchronous universal copy).
        copy_bits = 128
        copy_elems = copy_bits // self.dtype.width
        gmem_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self.dtype, num_bits_per_copy=copy_bits
        )
        threads_minor = smem_k // copy_elems
        thr_layout = cute.make_layout(
            (self.num_threads // threads_minor, threads_minor), stride=(threads_minor, 1)
        )
        val_layout = cute.make_layout((1, copy_elems))
        kv_copy = cute.make_tiled_copy_tv(gmem_atom, thr_layout, val_layout)
        q_copy = cute.make_tiled_copy_tv(
            gmem_atom, cute.make_layout((8, 4), stride=(4, 1)), val_layout
        )

        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, cutlass.Float32, (16, 8, 16)),
            (1, 1, 1),
            permutation_mnk=(16, 16, 16),
        )

        self.kernel(
            mQ, mK, mV, mPartial, mLSE, scale_log2,
            q_smem_layout, kv_smem_layout, q_copy, kv_copy, tiled_mma, Smem,
        ).launch(
            grid=[self.num_splits, self.kv_heads, mQ.shape[0]],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mPartial: cute.Tensor,
        mLSE: cute.Tensor,
        scale_log2: cutlass.Float32,
        q_smem_layout: cute.ComposedLayout,
        kv_smem_layout: cute.ComposedLayout,
        q_copy: cute.TiledCopy,
        kv_copy: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        Smem: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        split, kv_head, batch = cute.arch.block_idx()
        D = self.head_dim
        NB = self.n_block
        HPK = self.heads_per_kv

        # ---- Global tensors for this CTA's kv_head ----
        # Q: gather the 8 q_heads of this kv_head -> logical [HPK, D].
        q_ptr = cute.make_ptr(
            self.dtype,
            (mQ.iterator + cute.crd2idx((batch, kv_head * HPK, 0), mQ.layout)).toint(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        gQ = cute.make_tensor(q_ptr, cute.make_layout((HPK, D), stride=(D, 1)))

        # K/V: [n_block, D, num_n_blocks] view for this kv_head (BSHD).
        k_ptr = cute.make_ptr(
            self.dtype,
            (mK.iterator + cute.crd2idx((batch, 0, kv_head, 0), mK.layout)).toint(),
            cute.AddressSpace.gmem, assumed_align=16,
        )
        v_ptr = cute.make_ptr(
            self.dtype,
            (mV.iterator + cute.crd2idx((batch, 0, kv_head, 0), mV.layout)).toint(),
            cute.AddressSpace.gmem, assumed_align=16,
        )
        block_layout = cute.make_layout(
            (NB, D, self.num_n_blocks), stride=(D, 1, NB * D)
        )
        gK = cute.make_tensor(k_ptr, block_layout)
        gV = cute.make_tensor(v_ptr, block_layout)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(Smem)
        sQ = storage.sQ.get_tensor(q_smem_layout)
        sK = storage.sK.get_tensor(kv_smem_layout)
        sV = storage.sV.get_tensor(kv_smem_layout)
        # V^T view for PV (B operand is V with N=head_dim, K=n_block).
        sVt = cute.composition(sV, cute.make_layout((D, NB), stride=(NB, 1)))

        # ---- Load Q once (invariant across all blocks of this split) ----
        sQ_valid = cute.local_tile(sQ, (HPK, D), (0, 0))
        q2s = q_copy.get_slice(tidx)
        cute.copy(q_copy, q2s.partition_S(gQ), q2s.partition_D(sQ_valid))
        # zero the padding rows HPK..mma_m-1
        pad_elems = (self.mma_m - HPK) * D
        for i in cutlass.range_constexpr((pad_elems + self.num_threads - 1) // self.num_threads):
            lin = i * self.num_threads + tidx
            if lin < pad_elems:
                row = HPK + lin // D
                col = lin % D
                sQ[row, col] = cutlass.Float32(0.0).to(self.dtype)
        cute.arch.sync_warp()

        thr_mma = tiled_mma.get_slice(tidx)
        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
        tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK))
        tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt))
        acc_o = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((self.mma_m, D)), cutlass.Float32
        )
        acc_o.fill(0.0)

        # ldmatrix atoms.
        ldm_q = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype)
        ldm_k = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype)
        ldm_v = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype)
        tiled_q = cute.make_tiled_copy_A(ldm_q, tiled_mma)
        tiled_k = cute.make_tiled_copy_B(ldm_k, tiled_mma)
        tiled_v = cute.make_tiled_copy_B(ldm_v, tiled_mma)
        thr_q, thr_k, thr_v = (t.get_slice(tidx) for t in (tiled_q, tiled_k, tiled_v))
        tSsQ = thr_q.partition_S(sQ); tSrQv = thr_q.retile(tSrQ)
        tSsK = thr_k.partition_S(sK); tSrKv = thr_k.retile(tSrK)
        tOsVt = thr_v.partition_S(sVt); tOrVtv = thr_v.retile(tOrVt)

        # Q -> register fragment once.
        for kk in cutlass.range_constexpr(cute.size(tSsQ.shape[2])):
            cute.copy(tiled_q, tSsQ[None, None, kk], tSrQv[None, None, kk])

        # per-row softmax state (rows held by this thread's C fragment).
        row_count = acc_o.shape[0][0] * acc_o.shape[1]
        row_max = cute.make_rmem_tensor((row_count,), cutlass.Float32)
        row_sum = cute.make_rmem_tensor((row_count,), cutlass.Float32)
        row_max.fill(-cutlass.Float32.inf)
        row_sum.fill(0.0)

        block_begin = split * self.blocks_per_split
        for step in cutlass.range_constexpr(self.blocks_per_split):
            block = block_begin + step
            self._one_block(
                tiled_mma, thr_mma, tSrQ, tSrK, tOrVt, acc_o,
                kv_copy, tidx, gK, gV, sK, sV,
                tiled_k, tiled_v, tSsK, tSrKv, tOsVt, tOrVtv,
                row_max, row_sum, scale_log2, block, first=(step == 0),
            )

        self._epilogue(thr_mma, acc_o, row_max, row_sum, mPartial, mLSE,
                       batch, kv_head, split, scale_log2)

    @cute.jit
    def _one_block(self, tiled_mma, thr_mma, tSrQ, tSrK, tOrVt, acc_o,
                   kv_copy, tidx, gK, gV, sK, sV,
                   tiled_k, tiled_v, tSsK, tSrKv, tOsVt, tOrVtv,
                   row_max, row_sum, scale_log2, block, first: cutlass.Constexpr):
        # Synchronous load K,V tile for this block.
        g2s = kv_copy.get_slice(tidx)
        cute.copy(kv_copy, g2s.partition_S(gK[None, None, block]), g2s.partition_D(sK))
        cute.copy(kv_copy, g2s.partition_S(gV[None, None, block]), g2s.partition_D(sV))
        cute.arch.sync_warp()

        acc_s = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((self.mma_m, self.n_block)), cutlass.Float32
        )
        acc_s.fill(0.0)
        # QK: load K fragment, gemm.
        for kk in cutlass.range_constexpr(cute.size(tSsK.shape[2])):
            cute.copy(tiled_k, tSsK[None, None, kk], tSrKv[None, None, kk])
        for kk in cutlass.range_constexpr(cute.size(tSrQ.shape[2])):
            cute.gemm(tiled_mma, acc_s, tSrQ[None, None, kk], tSrK[None, None, kk], acc_s)

        self._softmax(thr_mma, acc_o, acc_s, row_max, row_sum, scale_log2, first)

        # P (BF16) -> PV. The QK accumulator fragment (C layout) must be
        # reinterpreted as the PV A operand: split the n-fragment mode into
        # (2, k-slices) so each MMA k-step consumes 2 contiguous score columns,
        # matching the m16n8k16 A fragment expected by the PV gemm.
        probs = cute.make_fragment_like(acc_s, self.dtype)
        probs.store(acc_s.load().to(self.dtype))
        divided = cute.logical_divide(probs.layout, (None, None, 2))
        pv_a_layout = cute.make_layout(
            (
                (divided.shape[0], divided.shape[2][0]),
                divided.shape[1],
                divided.shape[2][1],
            ),
            stride=(
                (divided.stride[0], divided.stride[2][0]),
                divided.stride[1],
                divided.stride[2][1],
            ),
        )
        pv_a = cute.make_tensor(probs.iterator, pv_a_layout)
        for kk in cutlass.range_constexpr(cute.size(tOsVt.shape[2])):
            cute.copy(tiled_v, tOsVt[None, None, kk], tOrVtv[None, None, kk])
        for kk in cutlass.range_constexpr(cute.size(pv_a.shape[2])):
            cute.gemm(tiled_mma, acc_o, pv_a[None, None, kk], tOrVt[None, None, kk], acc_o)

    @cute.jit
    def _softmax(self, thr_mma, acc_o, acc_s, row_max, row_sum, scale_log2, first: cutlass.Constexpr):
        scores_mn = self._mn(acc_s)
        out_mn = self._mn(acc_o)
        prev_max = cute.make_fragment_like(row_max, cutlass.Float32)
        if cutlass.const_expr(not first):
            cute.basic_copy(row_max, prev_max)
        for row in cutlass.range_constexpr(cute.size(row_max)):
            sc = scores_mn[row, None].load()
            cur = sc.reduce(cute.ReductionOp.MAX, -cutlass.Float32.inf, 0)
            cur = self._quad_max(cur)
            if cutlass.const_expr(not first):
                cur = cute.arch.fmax(prev_max[row], cur)
            p = cute.math.exp2(sc * scale_log2 - cur * scale_log2, fastmath=True)
            s = p.reduce(cute.ReductionOp.ADD, cutlass.Float32.zero, 0)
            if cutlass.const_expr(not first):
                corr = cute.math.exp2((prev_max[row] - cur) * scale_log2, fastmath=True)
                s += row_sum[row] * corr
                out_mn[row, None] = out_mn[row, None].load() * corr
            row_max[row] = cur
            row_sum[row] = s
            scores_mn[row, None] = p

    @cute.jit
    def _epilogue(self, thr_mma, acc_o, row_max, row_sum, mPartial, mLSE,
                  batch, kv_head, split, scale_log2):
        out_mn = self._mn(acc_o)
        ident = cute.make_identity_tensor((self.mma_m, self.head_dim))
        coord = self._mn(thr_mma.partition_C(ident))
        for row in cutlass.range_constexpr(cute.size(row_max)):
            denom = self._quad_sum(row_sum[row])
            out_mn[row, None] = out_mn[row, None].load() * cute.arch.rcp_approx(denom)
        for row in cutlass.range_constexpr(cute.size(row_max)):
            packed = coord[row, 0][0]
            fd = coord[row, 0][1]
            denom = self._quad_sum(row_sum[row])
            if fd == 0 and packed < self.heads_per_kv:
                q_head = kv_head * self.heads_per_kv + packed
                scale = scale_log2 / cutlass.Float32(LOG2E)
                mLSE[batch, q_head, split] = row_max[row] * scale + cute.math.log(denom, fastmath=True)
        cols = cute.size(coord.shape[1])
        for row in cutlass.range_constexpr(cute.size(row_max)):
            packed = coord[row, 0][0]
            if packed < self.heads_per_kv:
                q_head = kv_head * self.heads_per_kv + packed
                for col in cutlass.range_constexpr(cols):
                    dim = coord[row, col][1]
                    if dim < self.head_dim:
                        mPartial[batch, q_head, split, dim] = out_mn[row, col].to(mPartial.element_type)

    def _mn(self, acc):
        c = cute.make_layout(acc.layout.shape)
        mn = cute.make_layout(
            ((c.shape[0][1], c.shape[1]), (c.shape[0][0], c.shape[2])),
            stride=((c.stride[0][1], c.stride[1]), (c.stride[0][0], c.stride[2])),
        )
        return cute.make_tensor(acc.iterator, cute.composition(acc.layout, mn))

    def _quad(self, value, op):
        value = op(value, cute.arch.shuffle_sync_bfly(value, offset=2, mask=-1, mask_and_clamp=31))
        value = op(value, cute.arch.shuffle_sync_bfly(value, offset=1, mask=-1, mask_and_clamp=31))
        return value

    def _quad_max(self, v):
        return self._quad(v, lambda x, y: cute.arch.fmax(x, y))

    def _quad_sum(self, v):
        return self._quad(v, lambda x, y: x + y)
