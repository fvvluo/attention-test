# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause
"""Warp-MMA split-KV decode for the SM90 BF16 GQA8 target.

One 32-thread CTA owns (split, KV head, batch).  The eight query heads sharing
one KV head occupy the first eight rows of an M16 MMA tile; the other rows are
zero padding.  K/V N64 tiles are staged with cp.async and consumed by
16x8x16 warp MMA.  Stage 1 writes normalized BF16 partial O and FP32 LSE;
stage 2 performs a self-authored LSE-weighted split reduction.
"""

from types import SimpleNamespace

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils
from cutlass.cute.nvgpu import cpasync, warp


LOG2E = 1.4426950408889634


class GQADecodeMma:
    """M16 GQA-packed contiguous split-KV decode."""

    def __init__(
        self,
        head_dim: int,
        q_heads: int,
        kv_heads: int,
        seqlen_k: int,
        num_splits: int,
        n_block: int = 64,
        cache_mode=cpasync.LoadCacheMode.GLOBAL,
    ):
        self.head_dim = head_dim
        self.head_dim_padded = (head_dim + 31) // 32 * 32
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.heads_per_kv = q_heads // kv_heads
        self.mma_m = max(16, 1 << (self.heads_per_kv - 1).bit_length())
        self.seqlen_k = seqlen_k
        self.num_splits = num_splits
        self.n_block = n_block
        self.num_threads = 32
        self.num_n_blocks = seqlen_k // n_block
        if self.num_n_blocks % num_splits != 0:
            raise ValueError(
                "MMA decode requires num_n_blocks divisible by num_splits"
            )
        self.blocks_per_split = self.num_n_blocks // num_splits
        self.cache_mode = cache_mode
        self.cta_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.num_threads
        )

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,       # [B, Hq, D]
        mK: cute.Tensor,       # [B, Sk, Hkv, D]
        mV: cute.Tensor,       # [B, Sk, Hkv, D]
        mPartial: cute.Tensor, # [B, Hq, split, D]
        mLSE: cute.Tensor,     # [B, Hq, split]
        scale_log2: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        self.dtype = mQ.element_type
        smem_k = 64 if self.head_dim_padded % 64 == 0 else 32
        swizzle = 3 if smem_k == 64 else 2
        smem_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle, 3, 3),
            0,
            cute.make_layout((8, smem_k), stride=(smem_k, 1)),
        )
        q_smem_layout = cute.tile_to_shape(
            smem_atom, (self.mma_m, self.head_dim_padded), (0, 1)
        )
        kv_smem_layout = cute.tile_to_shape(
            smem_atom, (self.n_block, self.head_dim_padded), (0, 1)
        )

        @cute.struct
        class SharedStorage:
            sK: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(kv_smem_layout)],
                1024,
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(kv_smem_layout)],
                1024,
            ]

        copy_bits = 128
        copy_elems = copy_bits // self.dtype.width
        gmem_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=self.cache_mode),
            self.dtype,
            num_bits_per_copy=copy_bits,
        )
        threads_minor = smem_atom.outer.shape[1] // copy_elems
        thread_layout = cute.make_layout(
            (self.num_threads // threads_minor, threads_minor),
            stride=(threads_minor, 1),
        )
        value_layout = cute.make_layout((1, copy_elems))
        gmem_copy = cute.make_tiled_copy_tv(
            gmem_atom, thread_layout, value_layout
        )
        q_gmem_copy = cute.make_tiled_copy_tv(
            gmem_atom,
            cute.make_layout((8, 4), stride=(4, 1)),
            value_layout,
        )

        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, cutlass.Float32, (16, 8, 16)),
            (1, 1, 1),
            permutation_mnk=(16, 16, 16),
        )
        self.kernel(
            mQ,
            mK,
            mV,
            mPartial,
            mLSE,
            scale_log2,
            q_smem_layout,
            kv_smem_layout,
            gmem_copy,
            q_gmem_copy,
            tiled_mma,
            SharedStorage,
        ).launch(
            grid=[self.num_splits, self.kv_heads, mQ.shape[0]],
            block=[self.num_threads, 1, 1],
            stream=stream,
            use_pdl=True,
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
        gmem_copy: cute.TiledCopy,
        q_gmem_copy: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        split, kv_head, batch = cute.arch.block_idx()

        q_ptr = cute.make_ptr(
            self.dtype,
            (
                mQ.iterator
                + cute.crd2idx(
                    (batch, kv_head * self.heads_per_kv, 0), mQ.layout
                )
            ).toint(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        gQ = cute.make_tensor(
            q_ptr,
            cute.make_layout(
                (self.heads_per_kv, self.head_dim_padded),
                stride=(self.head_dim_padded, 1),
            ),
        )

        # Re-establish pointer alignment after the runtime batch/head offset.
        # For this exact target every offset is a multiple of 16 bytes, but the
        # generic pointer analysis cannot prove that through the strided view.
        k_ptr = cute.make_ptr(
            self.dtype,
            (
                mK.iterator
                + cute.crd2idx((batch, 0, kv_head, 0), mK.layout)
            ).toint(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        v_ptr = cute.make_ptr(
            self.dtype,
            (
                mV.iterator
                + cute.crd2idx((batch, 0, kv_head, 0), mV.layout)
            ).toint(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        block_layout = cute.make_layout(
            (self.n_block, self.head_dim_padded, self.num_n_blocks),
            stride=(self.head_dim_padded, 1, self.n_block * self.head_dim_padded),
        )
        gK = cute.make_tensor(k_ptr, block_layout)
        gV = cute.make_tensor(v_ptr, block_layout)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sK = storage.sK.get_tensor(kv_smem_layout)
        # Q is consumed before the first K copy, so alias it onto the first
        # rows of sK and keep only the loaded register fragment afterwards.
        sQ = storage.sK.get_tensor(q_smem_layout)
        sV = storage.sV.get_tensor(kv_smem_layout)
        sVt = cute.composition(
            sV,
            cute.make_layout(
                (self.head_dim_padded, self.n_block),
                stride=(self.n_block, 1),
            ),
        )

        sQ_real = cute.local_tile(
            sQ,
            (self.heads_per_kv, self.head_dim_padded),
            (0, 0),
        )
        q2s = q_gmem_copy.get_slice(tidx)
        tQgQ = q2s.partition_S(gQ)
        tQsQ = q2s.partition_D(sQ_real)
        g2s = gmem_copy.get_slice(tidx)
        tKgK = g2s.partition_S(gK)
        tKsK = g2s.partition_D(sK)
        tVgV = g2s.partition_S(gV)
        tVsV = g2s.partition_D(sV)

        thr_mma = tiled_mma.get_slice(tidx)
        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
        tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK))
        tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt))
        acc_o = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((self.mma_m, self.head_dim_padded)),
            cutlass.Float32,
        )
        acc_o.fill(0.0)

        copy_q = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self.dtype,
        )
        copy_k = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self.dtype,
        )
        copy_v = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
            self.dtype,
        )
        tiled_q = cute.make_tiled_copy_A(copy_q, tiled_mma)
        tiled_k = cute.make_tiled_copy_B(copy_k, tiled_mma)
        tiled_v = cute.make_tiled_copy_B(copy_v, tiled_mma)
        thread_q = tiled_q.get_slice(tidx)
        thread_k = tiled_k.get_slice(tidx)
        thread_v = tiled_v.get_slice(tidx)
        tSsQ = thread_q.partition_S(sQ)
        tSrQv = thread_q.retile(tSrQ)
        tSsK = thread_k.partition_S(sK)
        tSrKv = thread_k.retile(tSrK)
        tOsVt = thread_v.partition_S(sVt)
        tOrVtv = thread_v.retile(tOrVt)

        row_count = acc_o.shape[0][0] * acc_o.shape[1]
        row_max = cute.make_rmem_tensor((row_count,), cutlass.Float32)
        row_sum = cute.make_rmem_tensor((row_count,), cutlass.Float32)
        row_max.fill(-cutlass.Float32.inf)
        row_sum.fill(0.0)

        cute.copy(q_gmem_copy, tQgQ, tQsQ)
        cute.arch.cp_async_commit_group()
        padding_elements = (
            (self.mma_m - self.heads_per_kv) * self.head_dim_padded
        )
        for i in cutlass.range_constexpr(
            (padding_elements + self.num_threads - 1) // self.num_threads
        ):
            linear = i * self.num_threads + tidx
            if linear < padding_elements:
                row = self.heads_per_kv + linear // self.head_dim_padded
                dim = linear % self.head_dim_padded
                sQ[row, dim] = cutlass.Float32(0.0).to(self.dtype)
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_warp()
        # Q is invariant across all N blocks in this split.  Keep the complete
        # MMA A fragment resident instead of issuing ldmatrix loads per block.
        for k in cutlass.range_constexpr(cute.size(tSsQ.shape[2])):
            cute.copy(
                tiled_q,
                tSsQ[None, None, k],
                tSrQv[None, None, k],
            )

        block_begin = split * self.blocks_per_split
        cute.copy(
            gmem_copy,
            tKgK[None, None, None, block_begin],
            tKsK,
        )
        cute.arch.cp_async_commit_group()

        mma_state = SimpleNamespace(
            tiled_mma=tiled_mma,
            thr_mma=thr_mma,
            tSrQ=tSrQ,
            tSrK=tSrK,
            tOrVt=tOrVt,
            acc_o=acc_o,
        )
        global_copy = SimpleNamespace(
            gmem_copy=gmem_copy,
            tVgV=tVgV,
            tVsV=tVsV,
            tKgK=tKgK,
            tKsK=tKsK,
        )
        smem_copy = SimpleNamespace(
            tiled_q=tiled_q,
            tiled_k=tiled_k,
            tiled_v=tiled_v,
            tSsQ=tSsQ,
            tSsK=tSsK,
            tOsVt=tOsVt,
            tSrQv=tSrQv,
            tSrKv=tSrKv,
            tOrVtv=tOrVtv,
        )
        softmax_state = SimpleNamespace(
            row_max=row_max,
            row_sum=row_sum,
            scale_log2=scale_log2,
        )

        for step in cutlass.range_constexpr(self.blocks_per_split):
            block = block_begin + step
            self._one_block(
                mma_state,
                global_copy,
                smem_copy,
                softmax_state,
                block,
                first=(step == 0),
                last=(step == self.blocks_per_split - 1),
            )

        # Let the dependent combine grid begin launching while this CTA writes
        # its final partial.  The consumer waits before reading any producer data.
        cute.arch.griddepcontrol_launch_dependents()
        self._epilogue(
            mma_state,
            softmax_state,
            mPartial,
            mLSE,
            batch,
            kv_head,
            split,
        )

    @cute.jit
    def _one_block(
        self,
        mma_state,
        global_copy,
        smem_copy,
        softmax_state,
        block,
        first: cutlass.Constexpr,
        last: cutlass.Constexpr,
    ):
        acc_s = cute.make_rmem_tensor(
            mma_state.thr_mma.partition_shape_C(
                (self.mma_m, self.n_block)
            ),
            cutlass.Float32,
        )
        acc_s.fill(0.0)
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_warp()
        cute.copy(
            global_copy.gmem_copy,
            global_copy.tVgV[None, None, None, block],
            global_copy.tVsV,
        )
        cute.arch.cp_async_commit_group()

        cute.copy(
            smem_copy.tiled_k,
            smem_copy.tSsK[None, None, 0],
            smem_copy.tSrKv[None, None, 0],
        )
        for k in cutlass.range_constexpr(
            cute.size(smem_copy.tSsQ.shape[2])
        ):
            next_k = (k + 1) % cute.size(smem_copy.tSsQ.shape[2])
            cute.copy(
                smem_copy.tiled_k,
                smem_copy.tSsK[None, None, next_k],
                smem_copy.tSrKv[None, None, next_k],
            )
            cute.gemm(
                mma_state.tiled_mma,
                acc_s,
                mma_state.tSrQ[None, None, k],
                mma_state.tSrK[None, None, k],
                acc_s,
            )

        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_warp()
        if cutlass.const_expr(not last):
            cute.copy(
                global_copy.gmem_copy,
                global_copy.tKgK[None, None, None, block + 1],
                global_copy.tKsK,
            )
            cute.arch.cp_async_commit_group()

        self._softmax(mma_state, softmax_state, acc_s, first)

        probabilities = cute.make_fragment_like(acc_s, self.dtype)
        probabilities.store(acc_s.load().to(self.dtype))
        divided = cute.logical_divide(
            probabilities.layout, (None, None, 2)
        )
        probability_layout = cute.make_layout(
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
        tOrS = cute.make_tensor(probabilities.iterator, probability_layout)
        cute.copy(
            smem_copy.tiled_v,
            smem_copy.tOsVt[None, None, 0],
            smem_copy.tOrVtv[None, None, 0],
        )
        for k in cutlass.range_constexpr(cute.size(tOrS.shape[2])):
            next_k = (k + 1) % cute.size(tOrS.shape[2])
            cute.copy(
                smem_copy.tiled_v,
                smem_copy.tOsVt[None, None, next_k],
                smem_copy.tOrVtv[None, None, next_k],
            )
            cute.gemm(
                mma_state.tiled_mma,
                mma_state.acc_o,
                tOrS[None, None, k],
                mma_state.tOrVt[None, None, k],
                mma_state.acc_o,
            )

    @cute.jit
    def _softmax(
        self,
        mma_state,
        softmax_state,
        acc_s,
        first: cutlass.Constexpr,
    ):
        scores_mn = self._mn(acc_s)
        output_mn = self._mn(mma_state.acc_o)
        if cutlass.const_expr(not first):
            previous_max = cute.make_fragment_like(
                softmax_state.row_max, cutlass.Float32
            )
            cute.basic_copy(softmax_state.row_max, previous_max)
        for row in cutlass.range_constexpr(
            cute.size(softmax_state.row_max)
        ):
            scores = scores_mn[row, None].load()
            current_max = scores.reduce(
                cute.ReductionOp.MAX, -cutlass.Float32.inf, 0
            )
            current_max = self._quad_max(current_max)
            if cutlass.const_expr(not first):
                current_max = cute.arch.fmax(
                    previous_max[row], current_max
                )
            probabilities = cute.math.exp2(
                scores * softmax_state.scale_log2
                - current_max * softmax_state.scale_log2,
                fastmath=True,
            )
            current_sum = probabilities.reduce(
                cute.ReductionOp.ADD, cutlass.Float32.zero, 0
            )
            if cutlass.const_expr(not first):
                correction = cute.math.exp2(
                    (previous_max[row] - current_max)
                    * softmax_state.scale_log2,
                    fastmath=True,
                )
                current_sum += softmax_state.row_sum[row] * correction
                output_mn[row, None] = (
                    output_mn[row, None].load() * correction
                )
            softmax_state.row_max[row] = current_max
            softmax_state.row_sum[row] = current_sum
            scores_mn[row, None] = probabilities

    @cute.jit
    def _epilogue(
        self,
        mma_state,
        softmax_state,
        mPartial,
        mLSE,
        batch,
        kv_head,
        split,
    ):
        output_mn = self._mn(mma_state.acc_o)
        identity = cute.make_identity_tensor(
            (self.mma_m, self.head_dim_padded)
        )
        coord_mn = self._mn(mma_state.thr_mma.partition_C(identity))

        for row in cutlass.range_constexpr(
            cute.size(softmax_state.row_max)
        ):
            denominator = self._quad_sum(softmax_state.row_sum[row])
            output_mn[row, None] = (
                output_mn[row, None].load()
                * cute.arch.rcp_approx(denominator)
            )

        for row in cutlass.range_constexpr(
            cute.size(softmax_state.row_max)
        ):
            packed_row = coord_mn[row, 0][0]
            first_dim = coord_mn[row, 0][1]
            denominator = self._quad_sum(softmax_state.row_sum[row])
            if first_dim == 0 and packed_row < self.heads_per_kv:
                q_head = kv_head * self.heads_per_kv + packed_row
                scale = softmax_state.scale_log2 / cutlass.Float32(LOG2E)
                mLSE[batch, q_head, split] = (
                    softmax_state.row_max[row] * scale
                    + cute.math.log(denominator, fastmath=True)
                )

        columns = cute.size(coord_mn.shape[1])
        for row in cutlass.range_constexpr(
            cute.size(softmax_state.row_max)
        ):
            packed_row = coord_mn[row, 0][0]
            if packed_row < self.heads_per_kv:
                q_head = kv_head * self.heads_per_kv + packed_row
                for column in cutlass.range_constexpr(columns):
                    dim = coord_mn[row, column][1]
                    if dim < self.head_dim:
                        mPartial[batch, q_head, split, dim] = (
                            output_mn[row, column].to(mPartial.element_type)
                        )

    def _mn(self, accumulator):
        canonical = cute.make_layout(accumulator.layout.shape)
        mn = cute.make_layout(
            (
                (canonical.shape[0][1], canonical.shape[1]),
                (canonical.shape[0][0], canonical.shape[2]),
            ),
            stride=(
                (canonical.stride[0][1], canonical.stride[1]),
                (canonical.stride[0][0], canonical.stride[2]),
            ),
        )
        return cute.make_tensor(
            accumulator.iterator, cute.composition(accumulator.layout, mn)
        )

    def _quad(self, value, op):
        value = op(
            value,
            cute.arch.shuffle_sync_bfly(
                value, offset=2, mask=-1, mask_and_clamp=31
            ),
        )
        value = op(
            value,
            cute.arch.shuffle_sync_bfly(
                value, offset=1, mask=-1, mask_and_clamp=31
            ),
        )
        return value

    def _quad_max(self, value):
        return self._quad(value, lambda x, y: cute.arch.fmax(x, y))

    def _quad_sum(self, value):
        return self._quad(value, lambda x, y: x + y)


class GQADecodeCombine:
    """LSE-weighted split reduction with one 128-thread CTA per query head."""

    def __init__(self, head_dim: int, num_splits: int):
        self.head_dim = head_dim
        self.num_splits = num_splits
        self.num_threads = 128
        self.barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.num_threads
        )

    @cute.jit
    def __call__(
        self,
        mPartial: cute.Tensor,
        mLSE: cute.Tensor,
        mO: cute.Tensor,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class SharedStorage:
            weights: cute.struct.MemRange[
                cutlass.Float32, self.num_splits
            ]
            reduction: cute.struct.MemRange[
                cutlass.Float32, self.num_threads
            ]

        self.kernel(mPartial, mLSE, mO, SharedStorage        ).launch(
            grid=[mO.shape[1], mO.shape[0], 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        mPartial: cute.Tensor,
        mLSE: cute.Tensor,
        mO: cute.Tensor,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        q_head, batch, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        weights = storage.weights.get_tensor(
            cute.make_layout(self.num_splits)
        )
        reduction = storage.reduction.get_tensor(
            cute.make_layout(self.num_threads)
        )

        cute.arch.griddepcontrol_wait()
        local_max = -cutlass.Float32.inf
        split = tidx
        while split < self.num_splits:
            local_max = cute.arch.fmax(
                local_max, mLSE[batch, q_head, split]
            )
            split += self.num_threads
        reduction[tidx] = local_max
        self.barrier.arrive_and_wait()

        global_max = -cutlass.Float32.inf
        for i in cutlass.range_constexpr(self.num_threads):
            global_max = cute.arch.fmax(global_max, reduction[i])
        self.barrier.arrive_and_wait()

        local_sum = cutlass.Float32(0.0)
        split = tidx
        while split < self.num_splits:
            weight = cute.math.exp(
                mLSE[batch, q_head, split] - global_max,
                fastmath=True,
            )
            weights[split] = weight
            local_sum += weight
            split += self.num_threads
        reduction[tidx] = local_sum
        self.barrier.arrive_and_wait()

        denominator = cutlass.Float32(0.0)
        for i in cutlass.range_constexpr(self.num_threads):
            denominator += reduction[i]
        inv_denominator = cute.arch.rcp_approx(denominator)
        split = tidx
        while split < self.num_splits:
            weights[split] *= inv_denominator
            split += self.num_threads
        self.barrier.arrive_and_wait()

        dim = tidx
        while dim < self.head_dim:
            value = cutlass.Float32(0.0)
            for split_idx in cutlass.range_constexpr(self.num_splits):
                value += (
                    mPartial[batch, q_head, split_idx, dim].to(
                        cutlass.Float32
                    )
                    * weights[split_idx]
                )
            mO[batch, q_head, dim] = value.to(mO.element_type)
            dim += self.num_threads
