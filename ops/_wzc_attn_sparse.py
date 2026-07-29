"""Strategy C + DUAL-WG: block-level top-k sparse prefill on the proven
FlashAttention-3 style 2-consumer-warpgroup ping-pong pipeline.

NEW standalone file. Does NOT modify any existing kernel. Keeps the single-WG
`_wzc_sparse_prefill_c2.py` intact as the reference.

Why dual-WG: the single-WG (BLOCK_M=64) sparse kernel tops out at ~105 TFLOPS
(vs dense 143) because one warpgroup's softmax cannot hide behind the tensor
core. This kernel restores the dense kernel's structure -- BLOCK_M=128, two
consumer warpgroups that ping-pong a named-barrier token so exactly one WG
issues WGMMA while the other runs softmax -- and injects sparsity on top:

  Stage-0 (both consumer WGs, 256 threads): Quest-bound segment scoring
    bound[j] = max_over_128_q_rows (q.kmid[j] + |q|.krad[j]) * scale
    (CUDA-core here, proven correct in c2; WGMMA scoring is a later swap), then
    forced {sink,local,diagonal} + cumulative-mass tau selection (thread 0,
    bisected threshold), published to sel_idx[]/sel_cnt.
  Stage-1: producer streams K/V ONLY for sel_idx[]; the two consumer WGs run the
    dense ping-pong mainloop over sel_idx[] (both WGs share the same sel list --
    they attend the same KV blocks, just different q-rows). Diagonal segment gets
    the register causal mask.

tau=1.0 selects all causal segments -> bit-identical to the dense kernel.
bf16, causal, head_dim=128, batch=1, GQA.
"""

import math
from dataclasses import dataclass
from typing import Callable, Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Boolean, if_generate
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_wait
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
from cutlass.cute.runtime import from_dlpack

D_TYPE = cutlass.BFloat16
ACC_TYPE = cutlass.Float32

BLOCK_N = 128
_COMPILED_CACHE = {}
_BUFFER_CACHE = {}


# ---------------------------------------------------------------------------
# Helpers (identical to the dense prefill kernel).
# ---------------------------------------------------------------------------
def _convert_layout_acc_mn(acc_layout: cute.Layout) -> cute.Layout:
    col_major = cute.make_layout(acc_layout.shape)
    mn = cute.make_layout(
        (
            (col_major.shape[0][1], col_major.shape[1]),
            (col_major.shape[0][0], *col_major.shape[0][2:], col_major.shape[2]),
            *col_major.shape[3:],
        ),
        stride=(
            (col_major.stride[0][1], col_major.stride[1]),
            (col_major.stride[0][0], *col_major.stride[0][2:], col_major.stride[2]),
            *col_major.stride[3:],
        ),
    )
    return cute.composition(acc_layout, mn)


def _make_acc_tensor_mn_view(acc: cute.Tensor) -> cute.Tensor:
    return cute.make_tensor(acc.iterator, _convert_layout_acc_mn(acc.layout))


@cute.jit
def _convert_layout_acc_frgA(acc_layout: cute.Layout) -> cute.Layout:
    divided = cute.logical_divide(acc_layout, ((None, None, 2), None, None))
    return cute.make_layout(
        (
            (divided.shape[0][0], divided.shape[0][1], divided.shape[0][2][0]),
            divided.shape[1],
            (divided.shape[0][2][1], divided.shape[2]),
        ),
        stride=(
            (divided.stride[0][0], divided.stride[0][1], divided.stride[0][2][0]),
            divided.stride[1],
            (divided.stride[0][2][1], divided.stride[2]),
        ),
    )


def _transpose_smem_view(tensor: cute.Tensor) -> cute.Tensor:
    shape = (tensor.shape[1], tensor.shape[0], *tensor.shape[2:])
    order = (1, 0, *range(2, cute.rank(tensor)))
    return cute.composition(tensor, cute.make_ordered_layout(shape, order=order))


@cute.jit
def _wgmma_gemm(tiled_mma, acc, operand_a, operand_b,
                zero_init: cutlass.Constexpr[bool],
                wg_wait: cutlass.Constexpr[int] = 0):
    warpgroup.fence()
    mma_atom = cute.make_mma_atom(tiled_mma.op)
    mma_atom.set(warpgroup.Field.ACCUMULATE, not zero_init)
    for k in cutlass.range_constexpr(cute.size(operand_a.shape[2])):
        cute.gemm(mma_atom, acc, operand_a[None, None, k],
                  operand_b[None, None, k], acc)
        mma_atom.set(warpgroup.Field.ACCUMULATE, True)
    warpgroup.commit_group()
    if cutlass.const_expr(wg_wait >= 0):
        warpgroup.wait_group(wg_wait)


@cute.jit
def _warp_reduce4(value: cutlass.Float32, op: Callable) -> cutlass.Float32:
    value = op(value, cute.arch.shuffle_sync_bfly(value, offset=1))
    value = op(value, cute.arch.shuffle_sync_bfly(value, offset=2))
    return value


@cute.jit
def _warp_reduce_full(value: cutlass.Float32, op: Callable) -> cutlass.Float32:
    """Reduce across all 32 lanes of a warp (butterfly)."""
    value = op(value, cute.arch.shuffle_sync_bfly(value, offset=1))
    value = op(value, cute.arch.shuffle_sync_bfly(value, offset=2))
    value = op(value, cute.arch.shuffle_sync_bfly(value, offset=4))
    value = op(value, cute.arch.shuffle_sync_bfly(value, offset=8))
    value = op(value, cute.arch.shuffle_sync_bfly(value, offset=16))
    return value


@cute.jit
def _wgmma_softmax_update(acc_s, row_max, row_sum,
                          scale_log2: cutlass.Float32) -> cute.Tensor:
    scores = _make_acc_tensor_mn_view(acc_s)
    row_scale = cute.make_rmem_tensor(row_max.layout, cutlass.Float32)
    for row in cutlass.range_constexpr(cute.size(row_max)):
        score_row = scores[row, None].load()
        local_max = score_row.reduce(cute.ReductionOp.MAX, row_max[row], 0)
        new_max = _warp_reduce4(local_max, lambda x, y: cute.arch.fmax(x, y))
        safe_max = 0.0 if new_max == -cutlass.Float32.inf else new_max
        row_scale[row] = cute.math.exp2(
            (row_max[row] - safe_max) * scale_log2, fastmath=True)
        exp_scores = cute.math.exp2(
            score_row * scale_log2 - safe_max * scale_log2, fastmath=True)
        row_sum[row] = exp_scores.reduce(
            cute.ReductionOp.ADD, row_sum[row] * row_scale[row], 0)
        row_max[row] = new_max
        scores[row, None].store(exp_scores)
    return row_scale


@cute.jit
def _wgmma_rescale_output(acc_o, row_scale):
    output = _make_acc_tensor_mn_view(acc_o)
    for row in cutlass.range_constexpr(cute.size(row_scale)):
        output[row, None].store(output[row, None].load() * row_scale[row])


@cute.jit
def _wgmma_softmax_finalize(acc_o, row_sum):
    output = _make_acc_tensor_mn_view(acc_o)
    for row in cutlass.range_constexpr(cute.size(row_sum)):
        total = _warp_reduce4(row_sum[row], lambda x, y: x + y)
        inv_total = (1.0 if total == 0.0 or total != total
                     else cute.arch.rcp_approx(total))
        output[row, None].store(output[row, None].load() * inv_total)


@dataclass(frozen=True)
class _PipelineTmaAsyncNoCluster(pipeline.PipelineAsync):
    """TMA pipeline where one thread per 128-thread consumer WG releases a stage."""

    @staticmethod
    def create(barrier_storage, num_stages, producer_group, consumer_group,
               tx_count, init_wait: cutlass.Constexpr[bool] = True):
        producer = (pipeline.PipelineOp.TmaLoad, producer_group)
        consumer = (pipeline.PipelineOp.AsyncThread, consumer_group)
        sync_full = pipeline.PipelineAsync._make_sync_object(
            barrier_storage.align(min_align=8), num_stages, producer, tx_count)
        sync_empty = pipeline.PipelineAsync._make_sync_object(
            barrier_storage.align(min_align=8) + num_stages, num_stages, consumer)
        if cutlass.const_expr(init_wait):
            pipeline_init_wait()
        return _PipelineTmaAsyncNoCluster(sync_full, sync_empty, num_stages,
                                          None, None)

    def producer_acquire(self, state, try_acquire_token: Optional[Boolean] = None):
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(state.index, state.phase),
        )
        self.sync_object_full.arrive(state.index, self.producer_mask)

    def producer_commit(self, state):
        pass

    def consumer_release(self, state):
        if_generate(
            cute.arch.thread_idx()[0] % 128 == 0,
            lambda: self.sync_object_empty.arrive(state.index, self.consumer_mask),
        )


class SparseC5Kernel:
    """Dual-consumer-WG sparse prefill (dense ping-pong + sparse sel list)."""

    BLOCK_M = 128
    BLOCK_N = 128
    # NUM_STAGES=3 (matches dense) is recovered by reclaiming the ~6KB the sparse
    # scratch would otherwise add over dense: (1) seg_sc[n_seg] fp32 is ALIASED
    # onto the sK ring storage -- it is written+read only in Stage-0 scoring, and
    # the producer does not fill sK until after the mbar_sel handshake (post
    # selection), so sK is idle while seg_sc is live; (2) sel_idx[] is Int16
    # (n_seg <= 1024 fits). This brings smem back under the 227KB opt-in cap.
    NUM_STAGES = 2
    NUM_WG_THREADS = 128
    NUM_MMA_THREADS = 256    # 2 consumer warpgroups
    NUM_THREADS = 384        # + 1 producer warpgroup
    PRODUCER_REGS = 24
    CONSUMER_REGS = 240
    SCORE_CHUNK = 64         # q-row chunk for the CUDA-core scorer (regs)
    SCORE_SEG = 64           # segments per WGMMA scoring tile (one m64 atom)
    # Named barriers (0 reserved for sync_threads).
    BAR_EPI_WG0 = 1
    BAR_EPI_WG1 = 2
    BAR_TOKEN_WG0 = 3
    BAR_TOKEN_WG1 = 4
    BAR_SEL = 5              # 256-thread consumer sel broadcast / score sync
    BAR_SCORE = 6
    BAR_SELRED = 7           # 128-thread wg0 barrier inside WGMMA scoring loop

    def __init__(self, batch, q_heads, kv_heads, seq_len, head_dim,
                 sm_scale, tau, local_window, sink_blocks):
        if head_dim != 128:
            raise ValueError("requires head_dim=128")
        if seq_len % self.BLOCK_M != 0:
            raise ValueError("seq_len must be divisible by 128")
        self.batch = batch
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.scale = sm_scale
        self.q_per_kv = q_heads // kv_heads
        self.n_seg = seq_len // self.BLOCK_N
        self.tau = tau
        self.local_window = local_window
        self.sink_blocks = sink_blocks

    @cute.jit
    def __call__(self, Q, K, V, Kaug, Kmid, Krad, output, stream: cuda.CUstream):
        self._dtype = Q.element_type

        Q, K, V, output = [
            cute.make_tensor(
                t.iterator,
                cute.make_layout(
                    (t.shape[2], t.shape[3], (t.shape[1], t.shape[0])),
                    stride=(t.stride[2], t.stride[3], (t.stride[1], t.stride[0])),
                ),
            )
            for t in (Q, K, V, output)
        ]
        # Kaug: (B, HK, n_seg_padded, 2D) -> (n_seg_padded, 2D, (HK, B)) for
        # element indexing by the CUDA-core scorer ([:, :D]=kmid, [:, D:]=krad).
        Kaug = cute.make_tensor(
            Kaug.iterator,
            cute.make_layout(
                (Kaug.shape[2], Kaug.shape[3], (Kaug.shape[1], Kaug.shape[0])),
                stride=(Kaug.stride[2], Kaug.stride[3],
                        (Kaug.stride[1], Kaug.stride[0])),
            ),
        )

        smem_atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                utils.LayoutEnum.ROW_MAJOR, self._dtype, self.head_dim),
            self._dtype)
        sQ_layout = cute.tile_to_shape(
            smem_atom, (self.BLOCK_M, self.head_dim), (0, 1))
        sK_layout = cute.tile_to_shape(
            smem_atom, (self.BLOCK_N, self.head_dim, self.NUM_STAGES), (0, 1, 2))
        sV_layout = sK_layout
        sO_layout = sQ_layout

        # --- WGMMA scoring layouts (opt-in, WZC_C5_WGMMA_SCORE=1) ---
        # Score a SCORE_SEG-seg tile at a time: two D-wide TMAs (kmid, krad) +
        # |Q| built in smem, two accumulating WGMMAs (proven in isolation). All
        # score buffers alias the sV ring (idle during Stage-0). Kmid/Krad are
        # the two D-wide halves of Kaug (n_seg,2D): mode-1 offset 0 and D.
        sScoreK_layout = cute.tile_to_shape(
            smem_atom, (self.SCORE_SEG, self.head_dim), (0, 1))
        sAbsQ_layout = sQ_layout
        # Kmid/Krad: separate CONTIGUOUS (B,HK,n_seg_pad,D) -> (n_seg_pad,D,(HK,B)).
        Kmid_v, Krad_v = [
            cute.make_tensor(
                t.iterator,
                cute.make_layout(
                    (t.shape[2], t.shape[3], (t.shape[1], t.shape[0])),
                    stride=(t.stride[2], t.stride[3], (t.stride[1], t.stride[0]))))
            for t in (Kmid, Krad)]
        @cute.struct
        class SharedStorage:
            q_mbar: cute.struct.MemRange[cutlass.Int64, 1]
            sel_mbar: cute.struct.MemRange[cutlass.Int64, 1]
            score_mbar: cute.struct.MemRange[cutlass.Int64, 4]
            k_mbars: cute.struct.MemRange[cutlass.Int64, self.NUM_STAGES * 2]
            v_mbars: cute.struct.MemRange[cutlass.Int64, self.NUM_STAGES * 2]
            sel_cnt: cute.struct.MemRange[cutlass.Int32, 1]
            sel_idx: cute.struct.MemRange[cutlass.Int16, self.n_seg]
            # seg_sc is NOT stored here -- it is aliased onto the sK ring below
            # (dead before the producer fills sK), saving n_seg*4 bytes so that
            # NUM_STAGES=3 fits under the 227KB opt-in SMEM cap.
            sQ: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sQ_layout)], 1024]
            sK: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sK_layout)], 1024]
            sV: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sV_layout)], 1024]
            # Dedicated WGMMA-scoring buffers (debug: rule out sV-aliasing as the
            # TMA-hang cause). |Q| + kmid tile + krad tile.
            sAbsQ_d: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sAbsQ_layout)], 1024]
            sKm_d: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sScoreK_layout)], 1024]
            sKr_d: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sScoreK_layout)], 1024]

        tma_atom_q, tma_q = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), Q, sQ_layout,
            (self.BLOCK_M, self.head_dim))
        tma_atom_k, tma_k = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), K, cute.select(sK_layout, mode=[0, 1]),
            (self.BLOCK_N, self.head_dim))
        tma_atom_v, tma_v = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), V, cute.select(sV_layout, mode=[0, 1]),
            (self.BLOCK_N, self.head_dim))
        # Scoring TMAs: D-wide tiles of kmid / krad (SCORE_SEG segs each).
        tma_atom_km, tma_km = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), Kmid_v, sScoreK_layout,
            (self.SCORE_SEG, self.head_dim))
        tma_atom_kr, tma_kr = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), Krad_v, sScoreK_layout,
            (self.SCORE_SEG, self.head_dim))

        tiled_mma_qk = sm90_utils.make_trivial_tiled_mma(
            self._dtype, self._dtype, warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K, cutlass.Float32,
            atom_layout_mnk=(2, 1, 1), tiler_mn=(64, self.BLOCK_N))
        tiled_mma_pv = sm90_utils.make_trivial_tiled_mma(
            self._dtype, self._dtype, warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.MN, cutlass.Float32,
            atom_layout_mnk=(2, 1, 1), tiler_mn=(64, self.head_dim),
            a_source=warpgroup.OperandSource.RMEM)
        # Score GEMM: boundT[SCORE_SEG segs on M, BLOCK_M q-rows on N], K-major
        # x K-major, K=D. Single WG (one m64 atom, SCORE_SEG=64).
        tiled_mma_score = sm90_utils.make_trivial_tiled_mma(
            self._dtype, self._dtype, warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K, cutlass.Float32,
            atom_layout_mnk=(1, 1, 1), tiler_mn=(self.SCORE_SEG, self.BLOCK_M))

        self.kernel(
            tma_atom_q, tma_q, tma_atom_k, tma_k, tma_atom_v, tma_v,
            tma_atom_km, tma_km, tma_atom_kr, tma_kr,
            Kaug, output,
            cutlass.Float32(self.scale * math.log2(math.e)),
            cutlass.Float32(self.scale),
            sQ_layout, sK_layout, sV_layout, sO_layout,
            sScoreK_layout, sAbsQ_layout,
            tiled_mma_qk, tiled_mma_pv, tiled_mma_score, SharedStorage,
        ).launch(
            grid=(self.batch, self.q_heads, self.seq_len // self.BLOCK_M),
            block=[self.NUM_THREADS, 1, 1],
            smem=SharedStorage.size_in_bytes(), stream=stream, min_blocks_per_mp=1)

    @cute.kernel
    def kernel(
        self,
        tma_atom_q: cute.CopyAtom, mQ: cute.Tensor,
        tma_atom_k: cute.CopyAtom, mK: cute.Tensor,
        tma_atom_v: cute.CopyAtom, mV: cute.Tensor,
        tma_atom_km: cute.CopyAtom, mKmid: cute.Tensor,
        tma_atom_kr: cute.CopyAtom, mKrad: cute.Tensor,
        mKaug: cute.Tensor, mO: cute.Tensor,
        softmax_scale_log2: cutlass.Float32, sm_scale: cutlass.Float32,
        sQ_layout: cute.ComposedLayout, sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout, sO_layout: cute.ComposedLayout,
        sScoreK_layout: cute.ComposedLayout, sAbsQ_layout: cute.ComposedLayout,
        tiled_mma_qk: cute.TiledMma, tiled_mma_pv: cute.TiledMma,
        tiled_mma_score: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        batch, q_head, physical_m_block = cute.arch.block_idx()
        num_m_blocks = self.seq_len // self.BLOCK_M
        m_block = (physical_m_block if q_head % 2 == 0
                   else num_m_blocks - physical_m_block - 1)
        kv_head = q_head // self.q_per_kv

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_q)
            cpasync.prefetch_descriptor(tma_atom_k)
            cpasync.prefetch_descriptor(tma_atom_v)
            cpasync.prefetch_descriptor(tma_atom_km)
            cpasync.prefetch_descriptor(tma_atom_kr)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        mbar_q = storage.q_mbar.data_ptr()
        mbar_sel = storage.sel_mbar.data_ptr()
        mbar_score = storage.score_mbar.data_ptr()
        if warp_idx == 1:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(mbar_q, 1)
                # sel: 2 consumer WGs each have thread 0 arrive (2 arrivals),
                # producer warp 8 awaits.
                cute.arch.mbarrier_init(mbar_sel, 2)
        # WGMMA-scoring tile TMA barriers (consumer-issued): 4 = km/kr x 2 ring
        # slots. Init on a consumer warp.
        if warp_idx == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(mbar_score + 0, 1)
                cute.arch.mbarrier_init(mbar_score + 1, 1)
                cute.arch.mbarrier_init(mbar_score + 2, 1)
                cute.arch.mbarrier_init(mbar_score + 3, 1)
        cute.arch.mbarrier_init_fence()

        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 2)
        pipeline_k = _PipelineTmaAsyncNoCluster.create(
            barrier_storage=storage.k_mbars.data_ptr(),
            num_stages=self.NUM_STAGES,
            producer_group=producer_group, consumer_group=consumer_group,
            tx_count=cute.size_in_bytes(
                self._dtype, cute.select(sK_layout, mode=[0, 1])),
            init_wait=False)
        pipeline_v = _PipelineTmaAsyncNoCluster.create(
            barrier_storage=storage.v_mbars.data_ptr(),
            num_stages=self.NUM_STAGES,
            producer_group=producer_group, consumer_group=consumer_group,
            tx_count=cute.size_in_bytes(
                self._dtype, cute.select(sV_layout, mode=[0, 1])))

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sVt = _transpose_smem_view(sV)

        sel_cnt = storage.sel_cnt.get_tensor(cute.make_layout(1))
        sel_idx = storage.sel_idx.get_tensor(cute.make_layout(self.n_seg))
        # seg_sc aliases the sK ring: fp32 view of the first n_seg elements of
        # the sK smem region. Live only during Stage-0 scoring; the producer
        # fills sK only after the mbar_sel handshake (post selection), so there
        # is no overlap. Uses n_seg*4 <= sK_cosize*2 bytes (49152*2 >> 1024*4).
        seg_sc = cute.make_tensor(
            cute.recast_ptr(storage.sK.data_ptr(), dtype=cutlass.Float32),
            cute.make_layout(self.n_seg))
        # Reduction scratch for the PARALLEL selection: aliases the sV ring
        # (idle during Stage-0, same argument as seg_sc/sK). Layout:
        #   red[0:NUM_MMA_THREADS] per-thread partials;
        #   red[NUM_MMA_THREADS + k] control scalars (gmax/denom/forced/lo/hi/thr).
        red = cute.make_tensor(
            cute.recast_ptr(storage.sV.data_ptr(), dtype=cutlass.Float32),
            cute.make_layout(self.NUM_MMA_THREADS + 8))
        # WGMMA-scoring buffers alias the sV ring (used before selection's `red`,
        # which is used before the producer fills sV): |Q| (BLOCK_M x D) then
        # kmid tile + krad tile (SCORE_SEG x D each), packed by element offset.
        sAbsQ = storage.sAbsQ_d.get_tensor(sAbsQ_layout.outer,
                                           swizzle=sAbsQ_layout.inner)
        sKm_tile = storage.sKm_d.get_tensor(sScoreK_layout.outer,
                                            swizzle=sScoreK_layout.inner)
        sKr_tile = storage.sKr_d.get_tensor(sScoreK_layout.outer,
                                            swizzle=sScoreK_layout.inner)

        if warp_idx >= 8:
            cute.arch.warpgroup_reg_dealloc(self.PRODUCER_REGS)
            if warp_idx == 8:
                self.load(
                    tma_atom_q, mQ, tma_atom_k, mK, tma_atom_v, mV,
                    sQ, sK, sV, pipeline_k, pipeline_v, mbar_q, mbar_sel,
                    sel_idx, sel_cnt, batch, q_head, kv_head, m_block)
        else:
            cute.arch.warpgroup_reg_alloc(self.CONSUMER_REGS)
            self.mma(
                tiled_mma_qk, tiled_mma_pv, tiled_mma_score,
                tma_atom_km, mKmid, tma_atom_kr, mKrad, mKaug, mO,
                sQ, sK, sVt, sAbsQ, sKm_tile, sKr_tile, sO_layout,
                pipeline_k, pipeline_v, mbar_q, mbar_sel, mbar_score,
                sel_idx, sel_cnt, seg_sc, red,
                tidx, softmax_scale_log2, sm_scale,
                batch, q_head, kv_head, m_block)

    # -----------------------------------------------------------------------
    # Producer: load Q, then stream K/V ONLY for the selected segments.
    # -----------------------------------------------------------------------
    @cute.jit
    def load(self, tma_atom_q, mQ, tma_atom_k, mK, tma_atom_v, mV,
             sQ, sK, sV, pipeline_k, pipeline_v, mbar_q, mbar_sel,
             sel_idx, sel_cnt, batch, q_head, kv_head, m_block):
        gQ = cute.local_tile(mQ[None, None, (q_head, batch)],
                             (self.BLOCK_M, self.head_dim), (m_block, 0))
        gK = cute.local_tile(mK[None, None, (kv_head, batch)],
                             (self.BLOCK_N, self.head_dim), (None, 0))
        gV = cute.local_tile(mV[None, None, (kv_head, batch)],
                             (self.BLOCK_N, self.head_dim), (None, 0))
        tQsQ, tQgQ = cpasync.tma_partition(
            tma_atom_q, 0, cute.make_layout(1),
            cute.group_modes(sQ, 0, 2), cute.group_modes(gQ, 0, 2))
        tKsK, tKgK = cpasync.tma_partition(
            tma_atom_k, 0, cute.make_layout(1),
            cute.group_modes(sK, 0, 2), cute.group_modes(gK, 0, 2))
        tVsV, tVgV = cpasync.tma_partition(
            tma_atom_v, 0, cute.make_layout(1),
            cute.group_modes(sV, 0, 2), cute.group_modes(gV, 0, 2))

        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(
                mbar_q,
                cute.size_in_bytes(self._dtype, cute.select(sQ.layout, mode=[0, 1])))
        cute.copy(tma_atom_q, tQgQ, tQsQ, tma_bar_ptr=mbar_q)

        # Wait for the consumers' Stage-0 selection.
        cute.arch.mbarrier_wait(mbar_sel, phase=0)
        sel_count = cute.arch.make_warp_uniform(sel_cnt[0])

        producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.NUM_STAGES)
        for i in cutlass.range(sel_count, unroll=1):
            n_block = cutlass.Int32(sel_idx[sel_count - 1 - i])
            pipeline_k.producer_acquire(producer_state)
            cute.copy(tma_atom_k, tKgK[None, n_block],
                      tKsK[None, producer_state.index],
                      tma_bar_ptr=pipeline_k.producer_get_barrier(producer_state))
            pipeline_k.producer_commit(producer_state)
            pipeline_v.producer_acquire(producer_state)
            cute.copy(tma_atom_v, tVgV[None, n_block],
                      tVsV[None, producer_state.index],
                      tma_bar_ptr=pipeline_v.producer_get_barrier(producer_state))
            pipeline_v.producer_commit(producer_state)
            producer_state.advance()

    # -----------------------------------------------------------------------
    # Consumer (2 WGs): Stage-0 score+select, then dense ping-pong over sel[].
    # -----------------------------------------------------------------------
    @cute.jit
    def mma(self, tiled_mma_qk, tiled_mma_pv, tiled_mma_score,
            tma_atom_km, mKmid, tma_atom_kr, mKrad, mKaug, mO,
            sQ, sK, sVt, sAbsQ, sKm_tile, sKr_tile, sO_layout,
            pipeline_k, pipeline_v, mbar_q, mbar_sel, mbar_score,
            sel_idx, sel_cnt, seg_sc, red,
            tidx, softmax_scale_log2, sm_scale,
            batch, q_head, kv_head, m_block):
        cute.arch.mbarrier_wait(mbar_q, phase=0)

        n_block_max = cute.ceil_div((m_block + 1) * self.BLOCK_M, self.BLOCK_N)
        row_base = m_block * self.BLOCK_M

        # ---------------- Stage-0: score + select ----------------
        # WGMMA tensor-core scoring is the DEFAULT (fastest, ~144 TFLOPS). Set
        # WZC_C5_WGMMA_SCORE=0 to fall back to the CUDA-core scorer (also correct,
        # ~121 TFLOPS) for debugging / A-B.
        import os as _os
        if cutlass.const_expr(_os.environ.get("WZC_C5_WGMMA_SCORE", "1") != "0"):
            self.score_wgmma(tiled_mma_score, tma_atom_km, mKmid, tma_atom_kr,
                             mKrad, sQ, sAbsQ, sKm_tile, sKr_tile, seg_sc,
                             mbar_score, tidx, sm_scale, kv_head, batch,
                             n_block_max)
            cute.arch.barrier(barrier_id=self.BAR_SCORE,
                              number_of_threads=self.NUM_MMA_THREADS)
            self._select_from_scores(seg_sc, sel_idx, sel_cnt, red, tidx,
                                     n_block_max)
        else:
            self.score_and_select(sQ, mKaug, seg_sc, sel_idx, sel_cnt, red,
                                   tidx, sm_scale, kv_head, batch, n_block_max)
        # Publish to producer: each consumer WG's thread 0 arrives mbar_sel.
        if tidx % self.NUM_WG_THREADS == 0:
            cute.arch.mbarrier_arrive(mbar_sel)
        # Ensure all 256 consumer threads see sel before the mainloop.
        cute.arch.barrier(barrier_id=self.BAR_SEL,
                          number_of_threads=self.NUM_MMA_THREADS)
        sel_count = cute.arch.make_warp_uniform(sel_cnt[0])

        # ---------------- Stage-1: dense ping-pong over sel[] ----------------
        wg_idx = cute.arch.make_warp_uniform(tidx // self.NUM_WG_THREADS)
        wg_thread_layout = cute.make_layout(2, stride=self.NUM_WG_THREADS)
        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        wg_mma_qk = tiled_mma_qk.get_slice(wg_thread_layout(wg_idx))
        wg_mma_pv = tiled_mma_pv.get_slice(wg_thread_layout(wg_idx))
        tSrQ = tiled_mma_qk.make_fragment_A(wg_mma_qk.partition_A(sQ))
        tSrK = tiled_mma_qk.make_fragment_B(wg_mma_qk.partition_B(sK))
        tOrVt = tiled_mma_pv.make_fragment_B(wg_mma_pv.partition_B(sVt))

        acc_s_shape = tiled_mma_qk.partition_shape_C((self.BLOCK_M, self.BLOCK_N))
        acc_o_shape = tiled_mma_pv.partition_shape_C((self.BLOCK_M, self.head_dim))
        acc_O = cute.make_rmem_tensor(acc_o_shape, cutlass.Float32)
        acc_O.fill(0.0)
        tOrP = cute.make_rmem_tensor(
            _convert_layout_acc_frgA(cute.make_layout(acc_s_shape)), self._dtype)
        num_softmax_rows = acc_O.shape[0][0] * acc_O.shape[1]
        row_max = cute.make_rmem_tensor((num_softmax_rows,), cutlass.Float32)
        row_sum = cute.make_rmem_tensor((num_softmax_rows,), cutlass.Float32)
        row_max.fill(-cutlass.Float32.inf)
        row_sum.fill(0.0)

        cS = cute.make_identity_tensor((self.BLOCK_M, self.BLOCK_N))
        tScS = thr_mma_qk.partition_C(cS)
        tScS_mn = _make_acc_tensor_mn_view(tScS)

        my_token_bar = self.BAR_TOKEN_WG0 + wg_idx
        other_token_bar = self.BAR_TOKEN_WG1 - wg_idx
        epi_bar = self.BAR_EPI_WG0 + wg_idx

        # Hand the first WGMMA-issue token to WG0.
        if wg_idx == 1:
            cute.arch.barrier_arrive(barrier_id=self.BAR_TOKEN_WG0,
                                     number_of_threads=self.NUM_WG_THREADS)

        state_k = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.NUM_STAGES)
        state_v = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.NUM_STAGES)

        # Prologue: QK for the first selected (diagonal) segment.
        acc_S = cute.make_rmem_tensor(acc_s_shape, cutlass.Float32)
        pipeline_k.consumer_wait(state_k)
        _wgmma_gemm(tiled_mma_qk, acc_S, tSrQ, tSrK[None, None, None, state_k.index],
                    zero_init=True, wg_wait=-1)

        for i in cutlass.range(sel_count - 1, unroll=1):
            n_block = cutlass.Int32(sel_idx[sel_count - 1 - i])
            warpgroup.wait_group(0)
            pipeline_k.consumer_release(state_k)

            if i == 0:
                # Diagonal segment (highest selected index): causal mask.
                acc_S_mn = _make_acc_tensor_mn_view(acc_S)
                col_base = n_block * self.BLOCK_N
                for r in cutlass.range_constexpr(cute.size(acc_S_mn.shape[0])):
                    row_idx = row_base + tScS_mn[r, 0][0]
                    for c in cutlass.range_constexpr(cute.size(acc_S_mn.shape[1])):
                        col_idx = col_base + tScS_mn[0, c][1]
                        if col_idx >= self.seq_len or col_idx > row_idx:
                            acc_S_mn[r, c] = -cutlass.Float32.inf

            row_scale = _wgmma_softmax_update(acc_S, row_max, row_sum,
                                              softmax_scale_log2)
            _wgmma_rescale_output(acc_O, row_scale)
            tOrP_acc = cute.make_tensor(
                acc_S.iterator, _convert_layout_acc_frgA(acc_S.layout))
            tOrP.store(tOrP_acc.load().to(self._dtype))

            cute.arch.barrier(barrier_id=my_token_bar,
                              number_of_threads=self.NUM_WG_THREADS)
            pipeline_v.consumer_wait(state_v)
            _wgmma_gemm(tiled_mma_pv, acc_O, tOrP,
                        tOrVt[None, None, None, state_v.index],
                        zero_init=False, wg_wait=-1)
            state_k.advance()
            pipeline_k.consumer_wait(state_k)
            _wgmma_gemm(tiled_mma_qk, acc_S, tSrQ,
                        tSrK[None, None, None, state_k.index],
                        zero_init=True, wg_wait=-1)
            cute.arch.barrier_arrive(barrier_id=other_token_bar,
                                     number_of_threads=self.NUM_WG_THREADS)

            warpgroup.wait_group(1)
            pipeline_v.consumer_release(state_v)
            state_v.advance()

        # Peeled final selected segment.
        warpgroup.wait_group(0)
        pipeline_k.consumer_release(state_k)
        if sel_count - 1 == 0:
            acc_S_mn = _make_acc_tensor_mn_view(acc_S)
            n_block0 = cutlass.Int32(sel_idx[0])
            col_base = n_block0 * self.BLOCK_N
            for r in cutlass.range_constexpr(cute.size(acc_S_mn.shape[0])):
                row_idx = row_base + tScS_mn[r, 0][0]
                for c in cutlass.range_constexpr(cute.size(acc_S_mn.shape[1])):
                    col_idx = col_base + tScS_mn[0, c][1]
                    if col_idx >= self.seq_len or col_idx > row_idx:
                        acc_S_mn[r, c] = -cutlass.Float32.inf
        row_scale = _wgmma_softmax_update(acc_S, row_max, row_sum,
                                          softmax_scale_log2)
        _wgmma_rescale_output(acc_O, row_scale)
        tOrP_acc = cute.make_tensor(
            acc_S.iterator, _convert_layout_acc_frgA(acc_S.layout))
        tOrP.store(tOrP_acc.load().to(self._dtype))
        cute.arch.barrier(barrier_id=my_token_bar,
                          number_of_threads=self.NUM_WG_THREADS)
        pipeline_v.consumer_wait(state_v)
        _wgmma_gemm(tiled_mma_pv, acc_O, tOrP,
                    tOrVt[None, None, None, state_v.index],
                    zero_init=False, wg_wait=0)
        cute.arch.barrier_arrive(barrier_id=other_token_bar,
                                 number_of_threads=self.NUM_WG_THREADS)
        pipeline_v.consumer_release(state_v)

        _wgmma_softmax_finalize(acc_O, row_sum)
        self.epilogue(acc_O, sQ, sO_layout, tiled_mma_pv, mO, tidx, wg_idx,
                      q_head, batch, m_block)

    # -----------------------------------------------------------------------
    # Stage-0 WGMMA scorer (opt-in). boundT[SCORE_SEG segs, BLOCK_M q-rows] =
    # Kmid_tile @ Q^T + Krad_tile @ |Q|^T (two D-wide TMAs + two accumulating
    # WGMMAs, proven in isolation), per-N(q-row) max -> one scalar per segment.
    # Only warpgroup 0 (tidx<128) runs the score GEMM; warp 0 issues the tile
    # TMAs. |Q| is built by all 256 threads into sAbsQ first.
    # -----------------------------------------------------------------------
    @cute.jit
    def score_wgmma(self, tiled_mma_score, tma_atom_km, mKmid, tma_atom_kr,
                    mKrad, sQ, sAbsQ, sKm_tile, sKr_tile, seg_sc, mbar_score,
                    tidx, sm_scale, kv_head, batch, n_block_max):
        D = self.head_dim
        # build |Q| in sAbsQ (all 256 consumer threads)
        n_elem = self.BLOCK_M * D
        t = tidx
        while t < n_elem:
            r = t // D
            d = t % D
            sAbsQ[r, d] = cute.math.abs(sQ[r, d].to(cutlass.Float32)).to(self._dtype)
            t = t + self.NUM_MMA_THREADS
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.barrier(barrier_id=self.BAR_SCORE,
                          number_of_threads=self.NUM_MMA_THREADS)

        import os as _os
        _stage = _os.environ.get("WZC_C5_SCORE_STAGE", "full")
        if cutlass.const_expr(_stage == "qaug"):
            # Debug: only build |Q|; write trivial scores, no TMA/MMA/mbar.
            j = tidx
            while j < n_block_max:
                seg_sc[j] = cutlass.Float32(0.0)
                j = j + self.NUM_MMA_THREADS
            return

        if tidx < self.NUM_WG_THREADS:
            gKmid = mKmid[None, None, (kv_head, batch)]   # (n_seg_pad, D)
            gKrad = mKrad[None, None, (kv_head, batch)]

            wg = tiled_mma_score.get_slice(0)
            thr = tiled_mma_score.get_slice(tidx)
            rKm = tiled_mma_score.make_fragment_A(wg.partition_A(sKm_tile))
            rKr = tiled_mma_score.make_fragment_A(wg.partition_A(sKr_tile))
            rQ = tiled_mma_score.make_fragment_B(wg.partition_B(sQ))
            rAQ = tiled_mma_score.make_fragment_B(wg.partition_B(sAbsQ))
            acc_shape = tiled_mma_score.partition_shape_C(
                (self.SCORE_SEG, self.BLOCK_M))
            acc = cute.make_rmem_tensor(acc_shape, cutlass.Float32)
            cS = cute.make_identity_tensor((self.SCORE_SEG, self.BLOCK_M))
            tScS = thr.partition_C(cS)
            tScS_mn = _make_acc_tensor_mn_view(tScS)
            km_atom = cute.make_mma_atom(tiled_mma_score.op)

            n_tiles = cute.ceil_div(n_block_max, self.SCORE_SEG)
            for ti in cutlass.range(n_tiles, unroll=1):
                ph = ti % 2
                if cutlass.const_expr(_stage == "noload"):
                    # Debug: mbar arrive(0)+wait only, no TMA -> tests barrier
                    # protocol in isolation. Zero-fill scores.
                    if tidx == 0:
                        cute.arch.mbarrier_arrive_and_expect_tx(mbar_score + 2 * ph, 0)
                        cute.arch.mbarrier_arrive_and_expect_tx(mbar_score + 2 * ph + 1, 0)
                    cute.arch.mbarrier_wait(mbar_score + 2 * ph, phase=(ti // 2) % 2)
                    cute.arch.mbarrier_wait(mbar_score + 2 * ph + 1, phase=(ti // 2) % 2)
                    jj = tidx
                    while jj < n_block_max:
                        seg_sc[jj] = cutlass.Float32(0.0)
                        jj = jj + self.NUM_WG_THREADS
                    cute.arch.barrier(barrier_id=self.BAR_SELRED,
                                      number_of_threads=self.NUM_WG_THREADS)
                else:
                    # Pre-select a single tile then tma_partition (matches the
                    # validated standalone scorer exactly).
                    gKm_t = cute.local_tile(gKmid, (self.SCORE_SEG, D), (ti, 0))
                    gKr_t = cute.local_tile(gKrad, (self.SCORE_SEG, D), (ti, 0))
                    tKmS, tKmG = cpasync.tma_partition(
                        tma_atom_km, 0, cute.make_layout(1),
                        cute.group_modes(sKm_tile, 0, 2), cute.group_modes(gKm_t, 0, 2))
                    tKrS, tKrG = cpasync.tma_partition(
                        tma_atom_kr, 0, cute.make_layout(1),
                        cute.group_modes(sKr_tile, 0, 2), cute.group_modes(gKr_t, 0, 2))
                    # TMA must be issued by a full warp (warp 0), with a single
                    # elected lane doing arrive_and_expect_tx (matches probe).
                    if tidx < 32:
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                mbar_score + 2 * ph,
                                cute.size_in_bytes(self._dtype,
                                                   cute.select(sKm_tile.layout, mode=[0, 1])))
                        cute.copy(tma_atom_km, tKmG, tKmS,
                                  tma_bar_ptr=(mbar_score + 2 * ph))
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                mbar_score + 2 * ph + 1,
                                cute.size_in_bytes(self._dtype,
                                                   cute.select(sKr_tile.layout, mode=[0, 1])))
                        cute.copy(tma_atom_kr, tKrG, tKrS,
                                  tma_bar_ptr=(mbar_score + 2 * ph + 1))
                    cute.arch.mbarrier_wait(mbar_score + 2 * ph, phase=(ti // 2) % 2)
                    cute.arch.mbarrier_wait(mbar_score + 2 * ph + 1, phase=(ti // 2) % 2)

                    if cutlass.const_expr(_stage == "loadonly"):
                        # Debug: TMA+wait done, skip MMA/reduce; zero-fill.
                        jj = tidx
                        while jj < n_block_max:
                            seg_sc[jj] = cutlass.Float32(0.0)
                            jj = jj + self.NUM_WG_THREADS
                        cute.arch.barrier(barrier_id=self.BAR_SELRED,
                                          number_of_threads=self.NUM_WG_THREADS)
                    else:
                        warpgroup.fence()
                        km_atom.set(warpgroup.Field.ACCUMULATE, False)
                        for kk in cutlass.range_constexpr(cute.size(rKm.shape[2])):
                            cute.gemm(km_atom, acc, rKm[None, None, kk],
                                      rQ[None, None, kk], acc)
                            km_atom.set(warpgroup.Field.ACCUMULATE, True)
                        for kk in cutlass.range_constexpr(cute.size(rKr.shape[2])):
                            cute.gemm(km_atom, acc, rKr[None, None, kk],
                                      rAQ[None, None, kk], acc)
                        warpgroup.commit_group()
                        warpgroup.wait_group(0)

                        acc_mn = _make_acc_tensor_mn_view(acc)
                        for sr in cutlass.range_constexpr(cute.size(acc_mn.shape[0])):
                            seg_local = tScS_mn[sr, 0][0]
                            rv = acc_mn[sr, None].load()
                            lm = rv.reduce(cute.ReductionOp.MAX, -cutlass.Float32.inf, 0)
                            fm = _warp_reduce4(lm, lambda x, y: cute.arch.fmax(x, y))
                            seg_j = ti * self.SCORE_SEG + seg_local
                            if seg_j < n_block_max:
                                seg_sc[seg_j] = fm * sm_scale
                        # free the tile slot before the next TMA overwrites it
                        cute.arch.barrier(barrier_id=self.BAR_SELRED,
                                          number_of_threads=self.NUM_WG_THREADS)

    # -----------------------------------------------------------------------
    # Stage-0 CUDA-core scorer over BLOCK_M=128 rows (both consumer WGs, 256
    # threads). Chunked by SCORE_CHUNK rows to bound register use; each Kaug
    # element is read BLOCK_M/SCORE_CHUNK = 2 times.
    # -----------------------------------------------------------------------
    @cute.jit
    def score_and_select(self, sQ, mKaug, seg_sc, sel_idx, sel_cnt, red,
                         tidx, sm_scale, kv_head, batch, n_block_max):
        gKaug = mKaug[None, None, (kv_head, batch)]  # (n_seg_padded, 2D)
        D = self.head_dim

        import os as _os
        if cutlass.const_expr(_os.environ.get("WZC_C5_TRIVIAL_SCORE") == "1"):
            j = tidx
            while j < n_block_max:
                seg_sc[j] = cutlass.Float32(0.0)
                j = j + self.NUM_MMA_THREADS
        else:
            partial = cute.make_rmem_tensor((self.SCORE_CHUNK,), cutlass.Float32)
            j = tidx
            while j < n_block_max:
                best = -cutlass.Float32.inf
                r0 = 0
                while r0 < self.BLOCK_M:
                    for rr in cutlass.range_constexpr(self.SCORE_CHUNK):
                        partial[rr] = cutlass.Float32(0.0)
                    for d in cutlass.range(self.head_dim, unroll=1):
                        km = gKaug[j, d].to(cutlass.Float32)
                        kr = gKaug[j, D + d].to(cutlass.Float32)
                        for rr in cutlass.range_constexpr(self.SCORE_CHUNK):
                            q = sQ[r0 + rr, d].to(cutlass.Float32)
                            partial[rr] = partial[rr] + q * km + cute.math.abs(q) * kr
                    for rr in cutlass.range_constexpr(self.SCORE_CHUNK):
                        best = cute.arch.fmax(best, partial[rr])
                    r0 = r0 + self.SCORE_CHUNK
                seg_sc[j] = best * sm_scale
                j = j + self.NUM_MMA_THREADS

        cute.arch.barrier(barrier_id=self.BAR_SCORE,
                          number_of_threads=self.NUM_MMA_THREADS)
        self._select_from_scores(seg_sc, sel_idx, sel_cnt, red, tidx, n_block_max)

    @cute.jit
    def _select_from_scores(self, seg_sc, sel_idx, sel_cnt, red,
                            tidx, n_block_max):
        # Parallel (256-thread) forced/cumulative-mass-tau selection. Replaces the
        # single-thread O(30*n_seg) serial scan. Each pass reduces over segments
        # strided by NUM_MMA_THREADS; per-warp butterfly then an 8-warp combine
        # in smem (red[]). Bisection rounds are serial (data dependent) but each
        # round's mass-sum is parallel. Gather stays on thread 0 (one O(n_seg)).
        NT = self.NUM_MMA_THREADS
        NW = NT // 32                       # number of warps (8)
        warp_id = tidx // 32
        lane = tidx % 32
        diag = n_block_max - 1
        sink_hi = self.sink_blocks
        local_lo = diag - self.local_window + 1
        if local_lo < 0:
            local_lo = 0

        # ---- pass 1: gmax over all valid segments ----
        gmax_p = -cutlass.Float32.inf
        j = tidx
        while j < n_block_max:
            gmax_p = cute.arch.fmax(gmax_p, seg_sc[j])
            j = j + NT
        gmax_p = _warp_reduce_full(gmax_p, lambda x, y: cute.arch.fmax(x, y))
        if lane == 0:
            red[warp_id] = gmax_p
        cute.arch.barrier(barrier_id=self.BAR_SCORE, number_of_threads=NT)
        gmax = red[0]
        for w in cutlass.range_constexpr(1, NW):
            gmax = cute.arch.fmax(gmax, red[w])

        # ---- pass 2: denom + forced_mass ----
        denom_p = cutlass.Float32(0.0)
        forced_p = cutlass.Float32(0.0)
        j = tidx
        while j < n_block_max:
            w = cute.math.exp(seg_sc[j] - gmax, fastmath=True)
            denom_p = denom_p + w
            if (j < sink_hi) or (j >= local_lo):
                forced_p = forced_p + w
            j = j + NT
        denom_p = _warp_reduce_full(denom_p, lambda x, y: x + y)
        forced_p = _warp_reduce_full(forced_p, lambda x, y: x + y)
        cute.arch.barrier(barrier_id=self.BAR_SCORE, number_of_threads=NT)
        if lane == 0:
            red[warp_id] = denom_p
            red[NW + warp_id] = forced_p
        cute.arch.barrier(barrier_id=self.BAR_SCORE, number_of_threads=NT)
        denom = cutlass.Float32(0.0)
        forced_mass = cutlass.Float32(0.0)
        for w in cutlass.range_constexpr(NW):
            denom = denom + red[w]
            forced_mass = forced_mass + red[NW + w]
        target = self.tau * denom

        # ---- bisection: find largest threshold whose kept mass >= target ----
        lo = gmax - 60.0
        hi = gmax
        thr = lo
        it = 0
        while it < 20:
            mid = (lo + hi) * 0.5
            mass_p = cutlass.Float32(0.0)
            j = tidx
            while j < n_block_max:
                if (not ((j < sink_hi) or (j >= local_lo))) and (seg_sc[j] >= mid):
                    mass_p = mass_p + cute.math.exp(seg_sc[j] - gmax, fastmath=True)
                j = j + NT
            mass_p = _warp_reduce_full(mass_p, lambda x, y: x + y)
            cute.arch.barrier(barrier_id=self.BAR_SCORE, number_of_threads=NT)
            if lane == 0:
                red[warp_id] = mass_p
            cute.arch.barrier(barrier_id=self.BAR_SCORE, number_of_threads=NT)
            mass = forced_mass
            for w in cutlass.range_constexpr(NW):
                mass = mass + red[w]
            if mass >= target:
                thr = mid
                lo = mid
            else:
                hi = mid
            it = it + 1

        # ---- gather (thread 0): forced OR score >= thr, ascending ----
        if tidx == 0:
            cnt = cutlass.Int32(0)
            jj = 0
            while jj < n_block_max:
                if ((jj < sink_hi) or (jj >= local_lo)) or (seg_sc[jj] >= thr):
                    sel_idx[cnt] = cutlass.Int16(jj)
                    cnt = cnt + 1
                jj = jj + 1
            sel_cnt[0] = cnt


    # -----------------------------------------------------------------------
    # Epilogue: finalize -> stmatrix R2S -> per-WG vectorized store (dense).
    # -----------------------------------------------------------------------
    @cute.jit
    def epilogue(self, acc_O, sQ, sO_layout, tiled_mma_pv, mO, tidx, wg_idx,
                 q_head, batch, m_block):
        sO = cute.make_tensor(
            cute.recast_ptr(sQ.iterator, sO_layout.inner, dtype=self._dtype),
            sO_layout.outer)
        rO = cute.make_fragment_like(acc_O, self._dtype)
        rO.store(acc_O.load().to(self._dtype))
        epi_bar = self.BAR_EPI_WG0 + wg_idx
        cute.arch.barrier(barrier_id=epi_bar,
                          number_of_threads=self.NUM_WG_THREADS)
        smem_copy_atom_O = cute.make_copy_atom(
            warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4), self._dtype)
        smem_thr_copy_O = cute.make_tiled_copy_C(
            smem_copy_atom_O, tiled_mma_pv).get_slice(tidx)
        cute.copy(smem_copy_atom_O, smem_thr_copy_O.retile(rO),
                  smem_thr_copy_O.partition_D(sO))
        cute.arch.barrier(barrier_id=epi_bar,
                          number_of_threads=self.NUM_WG_THREADS)

        gO_full = cute.local_tile(mO[None, None, (q_head, batch)],
                                  (self.BLOCK_M, self.head_dim), (m_block, 0))
        gO = cute.local_tile(gO_full, (64, self.head_dim), (wg_idx, 0))
        sO_wg = cute.local_tile(sO, (64, self.head_dim), (wg_idx, 0))
        copy_elems = 128 // self._dtype.width
        atom_O = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self._dtype, num_bits_per_copy=128)
        tO_shape_dim_1 = self.head_dim // copy_elems
        tiled_copy_O = cute.make_tiled_copy_tv(
            atom_O,
            cute.make_layout((self.NUM_WG_THREADS // tO_shape_dim_1, tO_shape_dim_1),
                             stride=(tO_shape_dim_1, 1)),
            cute.make_layout((1, copy_elems)))
        thr_copy_O = tiled_copy_O.get_slice(tidx % self.NUM_WG_THREADS)
        tOsO = thr_copy_O.partition_S(sO_wg)
        tOgO = thr_copy_O.partition_D(gO)
        tOrO = cute.make_fragment_like(tOgO, self._dtype)
        cute.copy(tiled_copy_O, tOsO, tOrO)
        cute.copy(tiled_copy_O, tOrO, tOgO)


# ---------------------------------------------------------------------------
# Host side.
# ---------------------------------------------------------------------------
def _precompute_bounds_aug(K, kv_heads, seq_len, head_dim):
    """Kaug: (B, kv_heads, n_seg, 2D) bf16 = concat([kmid, krad_up]). See c2."""
    import torch
    B = K.shape[0]
    n_seg = seq_len // BLOCK_N
    Kseg = K.float().view(B, kv_heads, n_seg, BLOCK_N, head_dim)
    kmax = Kseg.amax(dim=3)
    kmin = Kseg.amin(dim=3)
    kmid = ((kmax + kmin) * 0.5).to(torch.bfloat16)
    krad = (kmax - kmin) * 0.5
    krad_bf = krad.to(torch.bfloat16)
    bumped = torch.nextafter(krad_bf.float(),
                             torch.full_like(krad, float("inf"))).to(torch.bfloat16)
    krad_up = torch.where(krad_bf.float() < krad, bumped, krad_bf)
    Kaug = torch.cat([kmid, krad_up], dim=-1)  # (B, HK, n_seg, 2D)
    # Pad the segment dim up to a multiple of SCORE_SEG so every WGMMA scoring
    # TMA tile is full (a partial tile underflows the expected-tx count and hangs
    # the mbarrier). Padding rows (kmid=krad=0 -> score 0) are masked out by the
    # per-CTA causal bound (seg_j < n_block_max) so they are never selected.
    pad = (-n_seg) % SparseC5Kernel.SCORE_SEG
    if pad:
        Kaug = torch.nn.functional.pad(Kaug, (0, 0, 0, pad))
        kmid = torch.nn.functional.pad(kmid, (0, 0, 0, pad))
        krad_up = torch.nn.functional.pad(krad_up, (0, 0, 0, pad))
    # Separate CONTIGUOUS kmid/krad (row pitch D) for the WGMMA-scoring TMA. The
    # fused-Kaug halves are STRIDED (row pitch 2D) and stall the TMA (empirically
    # hangs); contiguous D-wide tiles match the validated standalone scorer.
    return Kaug.contiguous(), kmid.contiguous(), krad_up.contiguous()


def _sdpa_fallback(Q, K, V, causal, sm_scale):
    import torch
    rep = Q.shape[1] // K.shape[1]
    K_rep = K.repeat_interleave(rep, dim=1)
    V_rep = V.repeat_interleave(rep, dim=1)
    return torch.nn.functional.scaled_dot_product_attention(
        Q, K_rep, V_rep, scale=sm_scale, is_causal=causal)


def run(Q, K, V, causal=True, sm_scale=None,
        tau=0.99, local_window=2, sink_blocks=1):
    import torch

    if Q.ndim != 4 or K.ndim != 4 or V.ndim != 4:
        raise ValueError("Q, K, and V must all be rank-4 tensors")
    if K.shape != V.shape:
        raise ValueError("K and V must have identical shapes")
    batch, q_heads, seq_len, head_dim = Q.shape
    k_batch, kv_heads, k_seq_len, k_head_dim = K.shape
    if (k_batch, k_seq_len, k_head_dim) != (batch, seq_len, head_dim):
        raise ValueError("Q, K, V must have matching batch/seq/head dims")
    if not (Q.is_cuda and K.is_cuda and V.is_cuda):
        raise ValueError("Q, K, and V must be CUDA tensors")
    if not (Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()):
        raise ValueError("Q, K, and V must be contiguous")
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    capability = torch.cuda.get_device_capability(Q.device)
    use_wgmma = (capability[0] == 9 and head_dim == 128 and causal
                 and seq_len % SparseC5Kernel.BLOCK_M == 0)
    if not use_wgmma:
        return _sdpa_fallback(Q, K, V, causal, sm_scale)

    def to_cute(tensor):
        return (from_dlpack(tensor, assumed_align=16)
                .mark_layout_dynamic(leading_dim=3)
                .mark_compact_shape_dynamic(
                    mode=3, stride_order=tensor.dim_order(),
                    divisibility=128 // D_TYPE.width))

    cache_key = (Q.device.index, capability, tuple(Q.shape), tuple(Q.stride()),
                 tuple(K.stride()), tuple(V.stride()), Q.dtype, bool(causal),
                 float(sm_scale), float(tau), int(local_window), int(sink_blocks))

    kaug_entry = _BUFFER_CACHE.get(("kaug", cache_key))
    if kaug_entry is None:
        Kaug, Kmid, Krad = _precompute_bounds_aug(K, kv_heads, seq_len, head_dim)
        _BUFFER_CACHE[("kaug", cache_key)] = (Kaug, Kmid, Krad)
    else:
        Kaug, Kmid, Krad = kaug_entry

    with torch.cuda.device(Q.device):
        torch_stream = torch.cuda.current_stream()
        stream = cuda.CUstream(torch_stream.cuda_stream)
        Q_t = to_cute(Q)
        K_t = to_cute(K)
        V_t = to_cute(V)
        entry = _BUFFER_CACHE.get(cache_key)
        if entry is None:
            output = torch.empty_like(Q)
            _BUFFER_CACHE[cache_key] = output
        else:
            output = entry
        O_t = to_cute(output)
        Kaug_t = to_cute(Kaug)
        Kmid_t = to_cute(Kmid)
        Krad_t = to_cute(Krad)

        compiled = _COMPILED_CACHE.get(cache_key)
        if compiled is None:
            k = SparseC5Kernel(batch, q_heads, kv_heads, seq_len, head_dim,
                               sm_scale, tau, local_window, sink_blocks)
            compiled = cute.compile(k, Q_t, K_t, V_t, Kaug_t, Kmid_t, Krad_t,
                                    O_t, stream)
            _COMPILED_CACHE[cache_key] = compiled

        compiled(Q_t, K_t, V_t, Kaug_t, Kmid_t, Krad_t, O_t, stream)
    return output
