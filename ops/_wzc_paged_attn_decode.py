"""Flash-Decoding attention decode kernel for H20 (sm_90a) in CuTe DSL.

Target shape: batch=1, q_heads=64, kv_heads=8, head_dim=128, q_seqlen=1,
kv_seqlen=131072. Memory-bound: KV cache = 512MB bf16, HBM ~4.0TB/s ->
floor ~0.128ms, goal ~0.14ms.

Design (see h20_attention_decode_design.md):
  * Split-KV (Flash-Decoding): grid = (num_splits, kv_heads, batch).
  * GQA group-packing: one CTA handles the GROUP=8 q_heads that share one
    kv_head, so every KV byte is read from HBM exactly once.
  * Transposed WGMMA dataflow so the tiny per-group M=8 never lands on the
    tensor-core M dimension:
      GEMM1: S^T[BLOCK_N, 8] = K_tile[BLOCK_N, D] @ Q^T[D, 8]   (M = kv rows)
      GEMM2: O^T[D, 8]        = V^T[D, BLOCK_N] @ P^T[BLOCK_N, 8] (M = head_dim)
    Both run at full WGMMA M=128 efficiency (compute ~29us << memory ~128us).
    Softmax reductions become per-column (per q_head) ops on the
    [kv, head] fragments; P^T round-trips through a tiny smem tile.
  * TMA producer/consumer warp-specialization with a multi-stage smem ring.
  * Partial kernel writes O_partial (fp32, normalized) + LSE (log2 domain);
    a small combine kernel reduces across splits.
"""

import math
from dataclasses import dataclass
from typing import Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Boolean, if_generate
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
from cutlass.cute.runtime import from_dlpack

import os as _os

D_TYPE = cutlass.BFloat16
ACC_TYPE = cutlass.Float32

GROUP_M = 8        # q_heads per kv_head for the target shape
HEAD_DIM = 128
BLOCK_N = 128      # kv rows per mainloop tile
NUM_STAGES = 3     # TMA pipeline stages (K and V each)
NUM_THREADS = 256  # 1 producer warpgroup + 1 consumer warpgroup
NUM_SPLITS = 6     # 6 splits x 8 kv_heads = 48 items, 1 item/SM (48 of 78 SMs).
                   # Same tuning as the contiguous decode kernel: on the 128K target
                   # 48 CTAs already saturate HBM, while fewer/longer splits minimize
                   # the combine reduction size (6 vs 39, fully unrolled), cut empty
                   # splits at short seq_len, and give perfect 1-item/SM balance.
                   # Measured ~0.158ms vs ~0.164ms at splits=39 (same GPU, A/B).

# Debug bisect switches (compile-time constants read from env).
_DBG_SKIP_QCOPY = _os.environ.get("AD_SKIP_QCOPY") == "1"
_DBG_SKIP_KVLOOP = _os.environ.get("AD_SKIP_KVLOOP") == "1"
_DBG_SKIP_GEMM1 = _os.environ.get("AD_SKIP_GEMM1") == "1"
_DBG_SKIP_SOFTMAX = _os.environ.get("AD_SKIP_SOFTMAX") == "1"
_DBG_SKIP_GEMM2 = _os.environ.get("AD_SKIP_GEMM2") == "1"
_DBG_SKIP_EPI = _os.environ.get("AD_SKIP_EPI") == "1"
_DBG_SKIP_COMBINE = _os.environ.get("AD_SKIP_COMBINE") == "1"


# ---------------------------------------------------------------------------
# WGMMA accumulator helpers (layout plumbing borrowed from the prefill kernel)
# ---------------------------------------------------------------------------

def _convert_layout_acc_mn(acc_layout: cute.Layout) -> cute.Layout:
    """View an SM90 accumulator as logical (M, N)."""
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
def _wgmma_gemm(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    operand_a: cute.Tensor,
    operand_b: cute.Tensor,
    zero_init: cutlass.Constexpr[bool],
):
    """acc (MMA, MMA_M, MMA_N) += a (MMA, MMA_M, MMA_K) @ b (MMA, MMA_N, MMA_K)."""
    warpgroup.fence()
    mma_atom = cute.make_mma_atom(tiled_mma.op)
    mma_atom.set(warpgroup.Field.ACCUMULATE, not zero_init)
    for k in cutlass.range_constexpr(cute.size(operand_a.shape[2])):
        cute.gemm(
            mma_atom,
            acc,
            operand_a[None, None, k],
            operand_b[None, None, k],
            acc,
        )
        mma_atom.set(warpgroup.Field.ACCUMULATE, True)
    warpgroup.commit_group()
    warpgroup.wait_group(0)


def _transpose_smem_view(tensor: cute.Tensor) -> cute.Tensor:
    shape = (tensor.shape[1], tensor.shape[0], *tensor.shape[2:])
    order = (1, 0, *range(2, cute.rank(tensor)))
    return cute.composition(tensor, cute.make_ordered_layout(shape, order=order))


@dataclass(frozen=True)
class _PipelineTmaAsyncNoCluster(pipeline.PipelineAsync):
    """TMA pipeline where one thread per 128-thread consumer WG releases a stage."""

    @staticmethod
    def create(
        barrier_storage: cute.Pointer,
        num_stages: int,
        producer_group: pipeline.CooperativeGroup,
        consumer_group: pipeline.CooperativeGroup,
        tx_count: int,
        init_wait: cutlass.Constexpr[bool] = True,
    ):
        producer = (pipeline.PipelineOp.TmaLoad, producer_group)
        consumer = (pipeline.PipelineOp.AsyncThread, consumer_group)
        sync_full = pipeline.PipelineAsync._make_sync_object(
            barrier_storage.align(min_align=8), num_stages, producer, tx_count
        )
        sync_empty = pipeline.PipelineAsync._make_sync_object(
            barrier_storage.align(min_align=8) + num_stages,
            num_stages,
            consumer,
        )
        if cutlass.const_expr(init_wait):
            pipeline_init_wait()
        return _PipelineTmaAsyncNoCluster(
            sync_full, sync_empty, num_stages, None, None
        )

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


# ---------------------------------------------------------------------------
# Flash-decoding kernel: partial + combine
# ---------------------------------------------------------------------------

class FlashDecodeKernel:
    GROUP_M = GROUP_M
    HEAD_DIM = HEAD_DIM

    def __init__(
        self,
        q_heads: int,
        kv_heads: int,
        max_pages: int,
        batch: int,
        num_splits: int = NUM_SPLITS,
        block_n: int = BLOCK_N,
        num_stages: int = NUM_STAGES,
        num_workers: int = 78,
        sm_scale: float = 1.0 / math.sqrt(HEAD_DIM),
    ):
        if kv_heads * self.GROUP_M != q_heads:
            raise ValueError("kernel expects q_heads == kv_heads * 8")
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.max_pages = max_pages   # block_table row length; mBT[max_pages] holds seq_len
        self.batch = batch
        self.num_splits = num_splits
        self.num_workers = num_workers
        self.BLOCK_N = block_n
        self.NUM_STAGES = num_stages
        self.NUM_THREADS = NUM_THREADS
        self.scale_log2 = sm_scale * math.log2(math.e)

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,      # (H, 1, D) bf16   (single sequence: batch folded out)
        mK: cute.Tensor,      # (num_pages, HK, page_size, D) bf16  paged pool
        mV: cute.Tensor,      # (num_pages, HK, page_size, D) bf16  paged pool
        mBT: cute.Tensor,     # (max_pages,) int32  block_table row for this seq
        mOpart: cute.Tensor,  # (S_splits, HK, G, D) bf16
        mLSE: cute.Tensor,    # (S_splits, HK, G) fp32 (log2 domain)
        mO: cute.Tensor,      # (H, 1, D) bf16
        stream: cuda.CUstream,
    ):
        self._dtype = mK.element_type

        # (H, D) view of Q/O (q_len == 1 squeezed).
        Q = cute.make_tensor(
            mQ.iterator,
            cute.make_layout((mQ.shape[0], mQ.shape[2]),
                             stride=(mQ.stride[0], mQ.stride[2])),
        )
        O = cute.make_tensor(
            mO.iterator,
            cute.make_layout((mO.shape[0], mO.shape[2]),
                             stride=(mO.stride[0], mO.stride[2])),
        )
        # Paged pool (num_pages, HK, page_size, D) -> (page_size, D, (HK, num_pages)):
        # logically identical to the continuous kernel's (S, D, (HK, B)) view, so all
        # TMA-atom / tma_partition / WGMMA code below is unchanged. The producer just
        # indexes the num_pages mode with block_table[tile] instead of a linear tile.
        K, V = [
            cute.make_tensor(
                t.iterator,
                cute.make_layout(
                    (t.shape[2], t.shape[3], (t.shape[1], t.shape[0])),
                    stride=(t.stride[2], t.stride[3], (t.stride[1], t.stride[0])),
                ),
            )
            for t in (mK, mV)
        ]

        smem_atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                utils.LayoutEnum.ROW_MAJOR, self._dtype, self.HEAD_DIM
            ),
            self._dtype,
        )
        sK_layout = cute.tile_to_shape(
            smem_atom, (self.BLOCK_N, self.HEAD_DIM, self.NUM_STAGES), (0, 1, 2)
        )
        sV_layout = sK_layout
        sQ_layout = cute.tile_to_shape(smem_atom, (self.GROUP_M, self.HEAD_DIM), (0, 1))
        sP_layout = cute.tile_to_shape(smem_atom, (self.GROUP_M, self.BLOCK_N), (0, 1))

        @cute.struct
        class SharedStorage:
            k_mbars: cute.struct.MemRange[cutlass.Int64, self.NUM_STAGES * 2]
            v_mbars: cute.struct.MemRange[cutlass.Int64, self.NUM_STAGES * 2]
            red: cute.struct.MemRange[cutlass.Float32, 2 * 4 * self.GROUP_M]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sQ_layout)], 1024
            ]
            sP: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sP_layout)], 1024
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sK_layout)], 1024
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sV_layout)], 1024
            ]

        tma_atom_k, tma_tensor_k = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            K,
            cute.select(sK_layout, mode=[0, 1]),
            (self.BLOCK_N, self.HEAD_DIM),
        )
        tma_atom_v, tma_tensor_v = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            V,
            cute.select(sV_layout, mode=[0, 1]),
            (self.BLOCK_N, self.HEAD_DIM),
        )

        # GEMM1: S^T[N=kv, 8] = K_tile @ Q^T   (A: K-major smem, B: K-major smem)
        tiled_mma_qk = sm90_utils.make_trivial_tiled_mma(
            self._dtype,
            self._dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            ACC_TYPE,
            atom_layout_mnk=(1, 1, 1),
            tiler_mn=(64, self.GROUP_M),
        )
        # GEMM2: O^T[D, 8] = V^T @ P^T         (A: MN-major smem, B: K-major smem)
        tiled_mma_pv = sm90_utils.make_trivial_tiled_mma(
            self._dtype,
            self._dtype,
            warpgroup.OperandMajorMode.MN,
            warpgroup.OperandMajorMode.K,
            ACC_TYPE,
            atom_layout_mnk=(1, 1, 1),
            tiler_mn=(64, self.GROUP_M),
        )

        self.partial_kernel(
            Q,
            tma_atom_k,
            tma_tensor_k,
            tma_atom_v,
            tma_tensor_v,
            mBT,
            mOpart,
            mLSE,
            cutlass.Float32(self.scale_log2),
            sQ_layout,
            sK_layout,
            sV_layout,
            sP_layout,
            tiled_mma_qk,
            tiled_mma_pv,
            SharedStorage,
        ).launch(
            grid=(self.num_workers, 1, 1),
            block=[self.NUM_THREADS, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

        self.combine_kernel(
            mOpart,
            mLSE,
            O,
        ).launch(
            grid=(self.q_heads, self.batch, 1),
            block=[128, 1, 1],
            smem=self.num_splits * 4,
            stream=stream,
        )



    # ------------------------------------------------------------------
    # Partial kernel
    # ------------------------------------------------------------------
    @cute.kernel
    def partial_kernel(
        self,
        mQ: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        mK: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        mV: cute.Tensor,
        mBT: cute.Tensor,
        mOpart: cute.Tensor,
        mLSE: cute.Tensor,
        scale_log2: cutlass.Float32,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        worker, _, _ = cute.arch.block_idx()

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_k)
            cpasync.prefetch_descriptor(tma_atom_v)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        pipeline_k = _PipelineTmaAsyncNoCluster.create(
            barrier_storage=storage.k_mbars.data_ptr(),
            num_stages=self.NUM_STAGES,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=cute.size_in_bytes(
                self._dtype, cute.select(sK_layout, mode=[0, 1])
            ),
            init_wait=False,
        )
        pipeline_v = _PipelineTmaAsyncNoCluster.create(
            barrier_storage=storage.v_mbars.data_ptr(),
            num_stages=self.NUM_STAGES,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=cute.size_in_bytes(
                self._dtype, cute.select(sV_layout, mode=[0, 1])
            ),
        )

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sVt = _transpose_smem_view(sV)

        # Dynamic per-seq tile count from block_table (seq_len passed via mBT[max_pages]:
        # last int32 slot holds seq_len; pages occupy [0, n_tiles)).
        seq_len = mBT[self.max_pages]
        tiles_total = (seq_len + self.BLOCK_N - 1) // self.BLOCK_N

        num_items = self.num_splits * self.kv_heads * self.batch

        if warp_idx < 4:
            # ------------------------- producer (warp 0 only) -------------------------
            if warp_idx == 0:
                producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.NUM_STAGES
                )
                for item in cutlass.range(worker, num_items, self.num_workers, unroll=1):
                    kv_head = (item // self.batch) % self.kv_heads
                    split = item // (self.batch * self.kv_heads)
                    tile_beg = split * tiles_total // self.num_splits
                    tile_end = (split + 1) * tiles_total // self.num_splits
                    n_tiles = tile_end - tile_beg
                    for i in cutlass.range(0 if _DBG_SKIP_KVLOOP else n_tiles, unroll=1):
                        # paged: logical tile -> physical page via block_table
                        page = mBT[tile_beg + i]
                        gK = cute.local_tile(
                            mK[None, None, (kv_head, page)],
                            (self.BLOCK_N, self.HEAD_DIM),
                            (None, 0),
                        )
                        gV = cute.local_tile(
                            mV[None, None, (kv_head, page)],
                            (self.BLOCK_N, self.HEAD_DIM),
                            (None, 0),
                        )
                        tKsK, tKgK = cpasync.tma_partition(
                            tma_atom_k, 0, cute.make_layout(1),
                            cute.group_modes(sK, 0, 2),
                            cute.group_modes(gK, 0, 2),
                        )
                        tVsV, tVgV = cpasync.tma_partition(
                            tma_atom_v, 0, cute.make_layout(1),
                            cute.group_modes(sV, 0, 2),
                            cute.group_modes(gV, 0, 2),
                        )
                        pipeline_k.producer_acquire(producer_state)
                        cute.copy(
                            tma_atom_k,
                            tKgK[None, 0],
                            tKsK[None, producer_state.index],
                            tma_bar_ptr=pipeline_k.producer_get_barrier(producer_state),
                        )
                        pipeline_k.producer_commit(producer_state)
                        pipeline_v.producer_acquire(producer_state)
                        cute.copy(
                            tma_atom_v,
                            tVgV[None, 0],
                            tVsV[None, producer_state.index],
                            tma_bar_ptr=pipeline_v.producer_get_barrier(producer_state),
                        )
                        pipeline_v.producer_commit(producer_state)
                        producer_state.advance()
        else:
            # ------------------------- consumer -------------------------
            tidx2 = tidx - 128
            lane = tidx2 % 32
            warp_in_wg = tidx2 // 32

            wg_mma_qk = tiled_mma_qk.get_slice(0)
            wg_mma_pv = tiled_mma_pv.get_slice(0)
            thr_mma_qk = tiled_mma_qk.get_slice(tidx2)
            thr_mma_pv = tiled_mma_pv.get_slice(tidx2)

            tSrK = tiled_mma_qk.make_fragment_A(wg_mma_qk.partition_A(sK))
            tSrQ = tiled_mma_qk.make_fragment_B(wg_mma_qk.partition_B(sQ))
            tOsVt = tiled_mma_pv.make_fragment_A(wg_mma_pv.partition_A(sVt))
            tPsP = tiled_mma_pv.make_fragment_B(wg_mma_pv.partition_B(sP))

            acc_s_shape = tiled_mma_qk.partition_shape_C((self.BLOCK_N, self.GROUP_M))
            acc_o_shape = tiled_mma_pv.partition_shape_C((self.HEAD_DIM, self.GROUP_M))
            acc_O = cute.make_rmem_tensor(acc_o_shape, ACC_TYPE)
            acc_O_mn = _make_acc_tensor_mn_view(acc_O)

            cS = cute.make_identity_tensor((self.BLOCK_N, self.GROUP_M))
            tScS = thr_mma_qk.partition_C(cS)
            tScS_mn = _make_acc_tensor_mn_view(tScS)
            cO = cute.make_identity_tensor((self.HEAD_DIM, self.GROUP_M))
            tOcO = thr_mma_pv.partition_C(cO)
            tOcO_mn = _make_acc_tensor_mn_view(tOcO)

            NR = cute.size(tScS_mn.shape[0])
            NC = cute.size(tScS_mn.shape[1])
            row_max = cute.make_rmem_tensor((NC,), ACC_TYPE)
            row_sum = cute.make_rmem_tensor((NC,), ACC_TYPE)

            red = storage.red.data_ptr()

            q_copy_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self._dtype, num_bits_per_copy=128
            )
            q_tiled_copy = cute.make_tiled_copy_tv(
                q_copy_atom,
                cute.make_layout(
                    (self.GROUP_M, 128 // self.GROUP_M), stride=(128 // self.GROUP_M, 1)
                ),
                cute.make_layout((1, 128 // (128 // self.GROUP_M))),
            )
            q_thr_copy = q_tiled_copy.get_slice(tidx2)

            consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.NUM_STAGES
            )
            for item in cutlass.range(worker, num_items, self.num_workers, unroll=1):
                kv_head = (item // self.batch) % self.kv_heads
                split = item // (self.batch * self.kv_heads)
                tile_beg = split * tiles_total // self.num_splits
                tile_end = (split + 1) * tiles_total // self.num_splits
                n_tiles = tile_end - tile_beg

                if not _DBG_SKIP_QCOPY:
                    gQ = cute.local_tile(
                        mQ, (self.GROUP_M, self.HEAD_DIM), (kv_head, 0)
                    )
                    cute.copy(
                        q_tiled_copy,
                        q_thr_copy.partition_S(gQ),
                        q_thr_copy.partition_D(sQ),
                    )
                    cute.arch.fence_proxy("async.shared", space="cta")
                    cute.arch.barrier(barrier_id=1, number_of_threads=128)

                acc_O.fill(0.0)
                row_max.fill(-cutlass.Float32.inf)
                row_sum.fill(0.0)

                for i in cutlass.range(0 if _DBG_SKIP_KVLOOP else n_tiles, unroll=1):
                    acc_S = cute.make_rmem_tensor(acc_s_shape, ACC_TYPE)
                    pipeline_k.consumer_wait(consumer_state)
                    if not _DBG_SKIP_GEMM1:
                        _wgmma_gemm(
                            tiled_mma_qk,
                            acc_S,
                            tSrK[None, None, None, consumer_state.index],
                            tSrQ,
                            zero_init=True,
                        )
                    pipeline_k.consumer_release(consumer_state)

                    acc_S_mn = _make_acc_tensor_mn_view(acc_S)

                    # ---- last-page mask (paged): only the global-last tile, and only
                    # when seq_len not a multiple of BLOCK_N. kv rows are the ROW(M)
                    # dim of S^T; set out-of-range rows to -inf BEFORE the column max.
                    if not _DBG_SKIP_GEMM1:
                        tile_base = (tile_beg + i) * self.BLOCK_N
                        if tile_base + self.BLOCK_N > seq_len:
                            for c in cutlass.range_constexpr(NC):
                                for r in cutlass.range_constexpr(NR):
                                    if tile_base + tScS_mn[r, c][0] >= seq_len:
                                        acc_S_mn[r, c] = -cutlass.Float32.inf

                    # ---- online softmax over kv (columns of the fragment) ----
                    if _DBG_SKIP_SOFTMAX:
                        for c in cutlass.range_constexpr(NC):
                            for r in cutlass.range_constexpr(NR):
                                acc_S_mn[r, c] = acc_S_mn[r, c] * scale_log2
                    if not _DBG_SKIP_SOFTMAX:
                        # scale into log2 domain.
                        for c in cutlass.range_constexpr(NC):
                            for r in cutlass.range_constexpr(NR):
                                acc_S_mn[r, c] = acc_S_mn[r, c] * scale_log2
                        # per-warp column max: thread-local rows + butterfly (no smem).
                        warp_max = cute.make_rmem_tensor((NC,), ACC_TYPE)
                        for c in cutlass.range_constexpr(NC):
                            v = acc_S_mn[0, c]
                            for r in cutlass.range_constexpr(1, NR):
                                v = cute.arch.fmax(v, acc_S_mn[r, c])
                            v = cute.arch.fmax(v, cute.arch.shuffle_sync_bfly(v, offset=4))
                            v = cute.arch.fmax(v, cute.arch.shuffle_sync_bfly(v, offset=8))
                            v = cute.arch.fmax(v, cute.arch.shuffle_sync_bfly(v, offset=16))
                            # safe max: if this warp's kv rows are all masked (-inf),
                            # use 0 so exp2(-inf - 0)=0 instead of exp2(-inf+inf)=NaN.
                            warp_max[c] = 0.0 if v == -cutlass.Float32.inf else v
                        # p with warp-local max (always <= 1, no overflow); per-warp sums.
                        warp_sum = cute.make_rmem_tensor((NC,), ACC_TYPE)
                        for c in cutlass.range_constexpr(NC):
                            p = cute.math.exp2(acc_S_mn[0, c] - warp_max[c], fastmath=True)
                            acc_S_mn[0, c] = p
                            tsum = p
                            for r in cutlass.range_constexpr(1, NR):
                                p = cute.math.exp2(acc_S_mn[r, c] - warp_max[c], fastmath=True)
                                acc_S_mn[r, c] = p
                                tsum += p
                            tsum += cute.arch.shuffle_sync_bfly(tsum, offset=4)
                            tsum += cute.arch.shuffle_sync_bfly(tsum, offset=8)
                            tsum += cute.arch.shuffle_sync_bfly(tsum, offset=16)
                            warp_sum[c] = tsum
                        # single cross-warp exchange of (max, sum) through smem
                        if lane < 4:
                            for c in cutlass.range_constexpr(NC):
                                red[warp_in_wg * self.GROUP_M + tScS_mn[0, c][1]] = warp_max[c]
                                red[4 * self.GROUP_M + warp_in_wg * self.GROUP_M + tScS_mn[0, c][1]] = warp_sum[c]
                        cute.arch.barrier(barrier_id=1, number_of_threads=128)
                        new_max = cute.make_rmem_tensor((NC,), ACC_TYPE)
                        alpha = cute.make_rmem_tensor((NC,), ACC_TYPE)
                        for c in cutlass.range_constexpr(NC):
                            col = tScS_mn[0, c][1]
                            mt = red[0 * self.GROUP_M + col]
                            for w in cutlass.range_constexpr(1, 4):
                                mt = cute.arch.fmax(mt, red[w * self.GROUP_M + col])
                            new_max[c] = cute.arch.fmax(row_max[c], mt)
                            alpha[c] = cute.math.exp2(row_max[c] - new_max[c], fastmath=True)
                            row_max[c] = new_max[c]
                            corr = cute.math.exp2(warp_max[c] - new_max[c], fastmath=True)
                            tsum = cutlass.Float32(0.0)
                            for w in cutlass.range_constexpr(4):
                                tsum += red[4 * self.GROUP_M + w * self.GROUP_M + col] * cute.math.exp2(
                                    red[w * self.GROUP_M + col] - new_max[c], fastmath=True
                                )
                            row_sum[c] = row_sum[c] * alpha[c] + tsum
                            # rescale O accumulator and P (warp-local -> global max correction)
                            for r in cutlass.range_constexpr(cute.size(acc_O_mn.shape[0])):
                                acc_O_mn[r, c] = acc_O_mn[r, c] * alpha[c]
                            for r in cutlass.range_constexpr(NR):
                                acc_S_mn[r, c] = acc_S_mn[r, c] * corr
                        # store P^T (bf16) to smem for the second WGMMA
                        for c in cutlass.range_constexpr(NC):
                            col = tScS_mn[0, c][1]
                            for r in cutlass.range_constexpr(NR):
                                sP[col, tScS_mn[r, c][0]] = cutlass.BFloat16(acc_S_mn[r, c])
                        cute.arch.fence_proxy("async.shared", space="cta")
                        cute.arch.barrier(barrier_id=1, number_of_threads=128)

                    pipeline_v.consumer_wait(consumer_state)
                    if not _DBG_SKIP_GEMM2:
                        _wgmma_gemm(
                            tiled_mma_pv,
                            acc_O,
                            tOsVt[None, None, None, consumer_state.index],
                            tPsP,
                            zero_init=False,
                        )
                    pipeline_v.consumer_release(consumer_state)
                    consumer_state.advance()

                # ------------------------- epilogue -------------------------
                if _DBG_SKIP_EPI:
                    pass
                if not _DBG_SKIP_EPI:
                    for c in cutlass.range_constexpr(NC):
                        total = row_sum[c]
                        inv = (
                            0.0
                            if total == 0.0 or total != total
                            else cute.arch.rcp_approx(total)
                        )
                        for r in cutlass.range_constexpr(cute.size(acc_O_mn.shape[0])):
                            acc_O_mn[r, c] = acc_O_mn[r, c] * inv
                        lse = (
                            -cutlass.Float32.inf
                            if total == 0.0 or total != total
                            else row_max[c] + cute.math.log2(total, fastmath=True)
                        )
                        if warp_in_wg == 0 and lane < 4:
                            mLSE[split, kv_head, tOcO_mn[0, c][1], 0] = lse
                    for c in cutlass.range_constexpr(NC):
                        col = tOcO_mn[0, c][1]
                        for r in cutlass.range_constexpr(cute.size(acc_O_mn.shape[0])):
                            mOpart[split, kv_head, col, tOcO_mn[r, c][0], 0] = cutlass.BFloat16(
                                acc_O_mn[r, c]
                            )


    # ------------------------------------------------------------------
    # Combine kernel: log-sum-exp reduction across splits
    # ------------------------------------------------------------------
    @cute.kernel
    def combine_kernel(
        self,
        mOpart: cute.Tensor,  # (S, HK, G, D, B) bf16
        mLSE: cute.Tensor,    # (S, HK, G, B) fp32 (log2 domain)
        mO: cute.Tensor,      # (H, D) bf16
    ):
        q_head, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        kv_head = q_head // self.GROUP_M
        g = q_head % self.GROUP_M

        smem = cutlass.utils.SmemAllocator()
        lse_buf = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout(self.num_splits), 16
        )
        for idx in cutlass.range(tidx, self.num_splits, 128):
            lse_buf[idx] = mLSE[idx, kv_head, g, 0]
        cute.arch.barrier()

        # num_splits is a compile-time constant: fully unroll for ILP.
        ws = cute.make_rmem_tensor((self.num_splits,), ACC_TYPE)
        lse_max = -cutlass.Float32.inf
        for s in cutlass.range_constexpr(self.num_splits):
            lse_max = cute.arch.fmax(lse_max, lse_buf[s])
        denom = cutlass.Float32(0.0)
        for s in cutlass.range_constexpr(self.num_splits):
            ws[s] = cute.math.exp2(lse_buf[s] - lse_max, fastmath=True)
            denom += ws[s]
        acc = cutlass.Float32(0.0)
        for s in cutlass.range_constexpr(self.num_splits):
            acc += ws[s] * cutlass.Float32(mOpart[s, kv_head, g, tidx, 0])
        inv = 0.0 if denom == 0.0 or denom != denom else cute.arch.rcp_approx(denom)
        mO[q_head, tidx] = cutlass.BFloat16(acc * inv)


# ---------------------------------------------------------------------------
# Host-side wrapper with compile/buffer caching
# ---------------------------------------------------------------------------

_COMPILED_CACHE = {}
_BUFFER_CACHE = {}

# ---------------------------------------------------------------------------
# Host-side paged decoder
# ---------------------------------------------------------------------------

_COMPILED_CACHE = {}
_BTROW_CACHE = {}


def _bt_to_cute(bt_row):
    # (max_pages+1,) int32 contiguous
    return from_dlpack(bt_row, assumed_align=16)


def _kv_to_cute(t):
    # paged pool (num_pages, kv_heads, page_size, head_dim) bf16, contiguous.
    # head_dim (mode 3) is the leading contiguous dim.
    return (
        from_dlpack(t, assumed_align=16)
        .mark_layout_dynamic(leading_dim=3)
        .mark_compact_shape_dynamic(
            mode=3, stride_order=t.dim_order(),
            divisibility=128 // D_TYPE.width,
        )
    )


def _qo_to_cute(t):
    # (q_heads, 1, head_dim) bf16 contiguous
    return (
        from_dlpack(t, assumed_align=16)
        .mark_layout_dynamic(leading_dim=2)
        .mark_compact_shape_dynamic(
            mode=2, stride_order=t.dim_order(),
            divisibility=128 // D_TYPE.width,
        )
    )


class PagedKVDecoder:
    """Stateless paged decode entry expected by paged_attn_benchmark.py.

    Physical layout (kv_head-major, evrey (page,kv_head) tile is contiguous 32KB):
        kv_cache_k/v[num_pages, kv_heads, page_size=128, head_dim=128]  bf16
        block_table[max_seqs, max_pages_per_seq]                        int32
        seq_len[max_seqs]                                               int32
    """

    @staticmethod
    def decode(q_new, kv_cache_k, kv_cache_v, block_table, seq_len,
               seq_id, sm_scale=None, num_splits=NUM_SPLITS, page_size=BLOCK_N,
               num_workers=None):
        import torch

        num_pages, kv_heads, ps, head_dim = kv_cache_k.shape
        q_heads = q_new.shape[0]
        assert ps == page_size == BLOCK_N and head_dim == HEAD_DIM
        assert q_heads == kv_heads * GROUP_M
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(head_dim)
        max_pages = block_table.shape[1]
        dev = q_new.device

        # Per-seq block-table row + seq_len packed into one int32 vector on-device
        #   bt_row[0:max_pages] = block_table[seq_id];  bt_row[max_pages] = seq_len.
        # Built with device ops only (no .item() sync) and cached per (bt, sl, seq).
        bt_key = (id(block_table), id(seq_len), int(seq_id))
        bt_row = _BTROW_CACHE.get(bt_key)
        if bt_row is None:
            bt_row = torch.empty(max_pages + 1, dtype=torch.int32, device=dev)
            _BTROW_CACHE[bt_key] = bt_row
        bt_row[:max_pages].copy_(block_table[seq_id])
        bt_row[max_pages].copy_(seq_len[seq_id])

        if num_workers is None:
            # Use exactly as many persistent CTAs as there are work items
            # (num_splits * kv_heads * batch), capped at the SM count. With the
            # tuned num_splits=6 this is 48 items -> 48 CTAs, avoiding 30 empty
            # CTAs that would otherwise launch/sync for no work. Measured
            # ~0.159 vs ~0.161 ms at num_workers=SM_count (same GPU, A/B).
            sm_count = torch.cuda.get_device_properties(dev).multi_processor_count
            num_items = num_splits * kv_heads  # batch == 1 for this decoder
            num_workers = min(sm_count, num_items)
        # Cache the compiled callable AND all stable device buffers + cute views,
        # keyed on the exact input tensor identities. bench_attention reuses the
        # same tensors every iter, so the timed loop then does only: copy q_new
        # into the persistent q buffer + launch (all per-call host work removed).
        key = (id(kv_cache_k), id(kv_cache_v), id(block_table), id(seq_len),
               int(seq_id), q_heads, num_splits, num_workers, float(sm_scale))

        with torch.cuda.device(dev):
            stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            entry = _COMPILED_CACHE.get(key)
            if entry is None:
                kernel = FlashDecodeKernel(
                    q_heads, kv_heads, max_pages, 1,
                    num_splits=num_splits, num_workers=num_workers, sm_scale=sm_scale,
                )
                q_buf = torch.empty(q_heads, 1, head_dim, dtype=torch.bfloat16, device=dev)
                o_buf = torch.empty(q_heads, 1, head_dim, dtype=torch.bfloat16, device=dev)
                o_part = torch.empty(
                    (num_splits, kv_heads, GROUP_M, head_dim, 1),
                    dtype=torch.bfloat16, device=dev)
                lse_part = torch.empty(
                    (num_splits, kv_heads, GROUP_M, 1),
                    dtype=torch.float32, device=dev)
                cQ = _qo_to_cute(q_buf)
                cK = _kv_to_cute(kv_cache_k)
                cV = _kv_to_cute(kv_cache_v)
                cBT = _bt_to_cute(bt_row)
                cOp = from_dlpack(o_part, assumed_align=16)
                cLSE = from_dlpack(lse_part, assumed_align=16)
                cO = _qo_to_cute(o_buf)
                compiled = cute.compile(kernel, cQ, cK, cV, cBT, cOp, cLSE, cO, stream)
                entry = (compiled, q_buf, o_buf, cQ, cK, cV, cBT, cOp, cLSE, cO)
                _COMPILED_CACHE[key] = entry
            compiled, q_buf, o_buf, cQ, cK, cV, cBT, cOp, cLSE, cO = entry
            q_buf.view(q_heads, head_dim).copy_(q_new)
            compiled(cQ, cK, cV, cBT, cOp, cLSE, cO, stream)
        return o_buf.view(q_heads, head_dim)
