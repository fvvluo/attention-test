"""Memory-bound GQA decode attention for H20 (sm_90a), written in CuTe DSL.

Target problem (Qwen3-32B, 128K context):
    batch=1, q_heads=64, kv_heads=8, head_dim=128, q_len=1, kv_len=131072, bf16.

Roofline
--------
KV cache = 2 * 8 * 131072 * 128 * 2B = 512 MB. On H20 (HBM3 ~4.0 TB/s) the
physical floor is 512MB / 4.0TB/s ~= 128 us. The compute is ~4.3 GFLOP (~29 us
on the 148 TFLOPS tensor cores), so the kernel is strictly memory-bound: every
byte of KV must be streamed from HBM exactly once at near-peak bandwidth.
Target for this operator: >= 3.5 TB/s  (<= ~0.146 ms).

Design
------
Flash-Decoding (split-KV) + GQA group-packing + transposed WGMMA dataflow.

  * Split-KV: the 131072 kv positions are chopped into ``num_splits`` chunks so
    that ``num_splits * kv_heads`` CTAs fill all 78 SMs several waves deep. Each
    CTA owns one (kv_head, split); a tiny combine kernel does the cross-split
    log-sum-exp reduction afterwards.

  * GQA group-packing: q_heads / kv_heads = 8. A single CTA processes the whole
    group of 8 q_heads that share a kv_head, so each KV byte is read from HBM
    exactly once. Skipping this would re-read KV 8x (=4GB) and blow the budget.

  * Transposed dataflow so the tiny group dimension (M=8) never sits on the
    tensor-core M axis (which would waste 8/128 of every MMA):
        GEMM1:  S^T[N=kv, 8] = K_tile[N, D] @ Q^T[D, 8]     (MMA M = kv rows)
        GEMM2:  O^T[D, 8]    = V^T[D, N]  @ P^T[N, 8]        (MMA M = head_dim)
    Softmax reductions become per-column (per q_head) ops over the [kv, head]
    fragment; P^T round-trips through a small smem tile between the two GEMMs.

  * TMA bulk-tensor loads with a warp-specialized producer/consumer pipeline
    (multi-stage smem ring) hide all HBM latency behind compute.

The partial kernel writes normalized O_partial (bf16) + LSE (log2 domain); the
combine kernel reduces across splits into the final bf16 output.
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
from cutlass.pipeline import pipeline_init_wait
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute.runtime import from_dlpack


D_TYPE = cutlass.BFloat16
ACC_TYPE = cutlass.Float32

GROUP_M = 8        # q_heads that share one kv_head (64 / 8)
HEAD_DIM = 128
BLOCK_N = 128      # kv rows loaded/consumed per mainloop tile
NUM_STAGES = 3     # TMA pipeline depth (K and V each)
NUM_THREADS = 256  # 1 producer warpgroup + 1 consumer warpgroup
NUM_SPLITS = 32    # default split count (32 evenly divides 1024 tiles/head)
NUM_WORKERS = 78   # fallback; the wrapper defaults to one CTA per work-item
                   # (num_workers = num_splits * kv_heads * batch), which
                   # removes inter-item pipeline-drain bubbles and lets the
                   # scheduler overlap waves -> measured 3.12 -> 3.44 TB/s.


# ---------------------------------------------------------------------------
# WGMMA accumulator layout helpers
# ---------------------------------------------------------------------------

def _acc_mn_layout(acc_layout: cute.Layout) -> cute.Layout:
    """View an SM90 WGMMA accumulator as a logical (M, N) tensor."""
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


def _acc_mn_view(acc: cute.Tensor) -> cute.Tensor:
    return cute.make_tensor(acc.iterator, _acc_mn_layout(acc.layout))


def _transpose_smem(tensor: cute.Tensor) -> cute.Tensor:
    shape = (tensor.shape[1], tensor.shape[0], *tensor.shape[2:])
    order = (1, 0, *range(2, cute.rank(tensor)))
    return cute.composition(tensor, cute.make_ordered_layout(shape, order=order))


@cute.jit
def _wgmma(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    a: cute.Tensor,
    b: cute.Tensor,
    zero_init: cutlass.Constexpr[bool],
):
    """acc(MMA,M,N) [+]= a(MMA,M,K) @ b(MMA,N,K)."""
    warpgroup.fence()
    atom = cute.make_mma_atom(tiled_mma.op)
    atom.set(warpgroup.Field.ACCUMULATE, not zero_init)
    for k in cutlass.range_constexpr(cute.size(a.shape[2])):
        cute.gemm(atom, acc, a[None, None, k], b[None, None, k], acc)
        atom.set(warpgroup.Field.ACCUMULATE, True)
    warpgroup.commit_group()
    warpgroup.wait_group(0)


@dataclass(frozen=True)
class _TmaPipeline(pipeline.PipelineAsync):
    """TMA-load producer / async-thread consumer pipeline (no cluster)."""

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
            barrier_storage.align(min_align=8) + num_stages, num_stages, consumer
        )
        if cutlass.const_expr(init_wait):
            pipeline_init_wait()
        return _TmaPipeline(sync_full, sync_empty, num_stages, None, None)

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
# Kernel
# ---------------------------------------------------------------------------

class DecodeKernel:
    GROUP_M = GROUP_M
    HEAD_DIM = HEAD_DIM

    def __init__(
        self,
        q_heads: int,
        kv_heads: int,
        kv_len: int,
        batch: int,
        num_splits: int = NUM_SPLITS,
        block_n: int = BLOCK_N,
        num_stages: int = NUM_STAGES,
        num_workers: int = NUM_WORKERS,
        blocks_per_mp: int = 1,
        use_pdl: bool = True,
        use_fused: bool = False,
        sm_scale: float = 1.0 / math.sqrt(HEAD_DIM),
    ):
        if kv_heads * self.GROUP_M != q_heads:
            raise ValueError("kernel expects q_heads == kv_heads * 8")
        if kv_len % block_n != 0:
            raise ValueError("kv_len must be a multiple of block_n")
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.kv_len = kv_len
        self.batch = batch
        self.num_splits = num_splits
        self.num_workers = num_workers
        self.blocks_per_mp = blocks_per_mp
        self.use_pdl = use_pdl
        self.use_fused = use_fused
        self.BLOCK_N = block_n
        self.NUM_STAGES = num_stages
        self.NUM_THREADS = NUM_THREADS
        self.scale_log2 = sm_scale * math.log2(math.e)

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,      # (B, H, 1, D) bf16
        mK: cute.Tensor,      # (B, HK, S, D) bf16
        mV: cute.Tensor,      # (B, HK, S, D) bf16
        mOpart: cute.Tensor,  # (splits, HK, G, D, B) bf16
        mLSE: cute.Tensor,    # (splits, HK, G, B) fp32 (log2 domain)
        mO: cute.Tensor,      # (B, H, 1, D) bf16
        mCounter: cute.Tensor,  # (HK, B) int32 scratch (fused path only)
        stream: cuda.CUstream,
    ):
        self._dtype = mQ.element_type

        # (H, D, B) view of Q/O; (S, D, (HK, B)) view of K/V.
        Q = cute.make_tensor(
            mQ.iterator,
            cute.make_layout(
                (mQ.shape[1], mQ.shape[3], mQ.shape[0]),
                stride=(mQ.stride[1], mQ.stride[3], mQ.stride[0]),
            ),
        )
        O = cute.make_tensor(
            mO.iterator,
            cute.make_layout(
                (mO.shape[1], mO.shape[3], mO.shape[0]),
                stride=(mO.stride[1], mO.stride[3], mO.stride[0]),
            ),
        )
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
            fused_flag: cute.struct.MemRange[cutlass.Int32, 4]
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

        # GEMM1: S^T[kv, 8] = K_tile @ Q^T   (A: K-major, B: K-major)
        mma_qk = sm90_utils.make_trivial_tiled_mma(
            self._dtype, self._dtype,
            warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K,
            ACC_TYPE, atom_layout_mnk=(1, 1, 1), tiler_mn=(64, self.GROUP_M),
        )
        # GEMM2: O^T[D, 8] = V^T @ P^T       (A: MN-major, B: K-major)
        mma_pv = sm90_utils.make_trivial_tiled_mma(
            self._dtype, self._dtype,
            warpgroup.OperandMajorMode.MN, warpgroup.OperandMajorMode.K,
            ACC_TYPE, atom_layout_mnk=(1, 1, 1), tiler_mn=(64, self.GROUP_M),
        )

        self.partial_kernel(
            Q, tma_atom_k, tma_tensor_k, tma_atom_v, tma_tensor_v,
            mOpart, mLSE, O, mCounter, cutlass.Float32(self.scale_log2),
            sQ_layout, sK_layout, sV_layout, sP_layout,
            mma_qk, mma_pv, SharedStorage,
        ).launch(
            grid=(self.num_workers, 1, 1),
            block=[self.NUM_THREADS, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=self.blocks_per_mp,
            use_pdl=self.use_pdl,
        )

        if cutlass.const_expr(not self.use_fused):
            self.combine_kernel(mOpart, mLSE, O).launch(
                grid=(self.q_heads, self.batch, 1),
                block=[128, 1, 1],
                smem=self.num_splits * 4,
                stream=stream,
                use_pdl=self.use_pdl,
            )

    # ------------------------------------------------------------------
    # Partial kernel: split-KV online softmax, transposed WGMMA dataflow
    # ------------------------------------------------------------------
    @cute.kernel
    def partial_kernel(
        self,
        mQ: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        mK: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        mV: cute.Tensor,
        mOpart: cute.Tensor,
        mLSE: cute.Tensor,
        mO: cute.Tensor,
        mCounter: cute.Tensor,
        scale_log2: cutlass.Float32,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout,
        mma_qk: cute.TiledMma,
        mma_pv: cute.TiledMma,
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
        pipeline_k = _TmaPipeline.create(
            barrier_storage=storage.k_mbars.data_ptr(),
            num_stages=self.NUM_STAGES,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=cute.size_in_bytes(self._dtype, cute.select(sK_layout, mode=[0, 1])),
            init_wait=False,
        )
        pipeline_v = _TmaPipeline.create(
            barrier_storage=storage.v_mbars.data_ptr(),
            num_stages=self.NUM_STAGES,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=cute.size_in_bytes(self._dtype, cute.select(sV_layout, mode=[0, 1])),
        )

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sVt = _transpose_smem(sV)

        tiles_total = self.kv_len // self.BLOCK_N
        num_items = self.num_splits * self.kv_heads * self.batch

        if warp_idx < 4:
            # -------- producer: warp 0 issues TMA loads --------
            if warp_idx == 0:
                pstate = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.NUM_STAGES
                )
                for item in cutlass.range(worker, num_items, self.num_workers, unroll=1):
                    kv_head = (item // self.batch) % self.kv_heads
                    split = item // (self.batch * self.kv_heads)
                    batch = item % self.batch
                    tile_beg = split * tiles_total // self.num_splits
                    tile_end = (split + 1) * tiles_total // self.num_splits
                    n_tiles = tile_end - tile_beg
                    gK = cute.local_tile(
                        mK[None, None, (kv_head, batch)],
                        (self.BLOCK_N, self.HEAD_DIM), (None, 0),
                    )
                    gV = cute.local_tile(
                        mV[None, None, (kv_head, batch)],
                        (self.BLOCK_N, self.HEAD_DIM), (None, 0),
                    )
                    tKsK, tKgK = cpasync.tma_partition(
                        tma_atom_k, 0, cute.make_layout(1),
                        cute.group_modes(sK, 0, 2), cute.group_modes(gK, 0, 2),
                    )
                    tVsV, tVgV = cpasync.tma_partition(
                        tma_atom_v, 0, cute.make_layout(1),
                        cute.group_modes(sV, 0, 2), cute.group_modes(gV, 0, 2),
                    )
                    for i in cutlass.range(n_tiles, unroll=1):
                        pipeline_k.producer_acquire(pstate)
                        cute.copy(
                            tma_atom_k, tKgK[None, tile_beg + i],
                            tKsK[None, pstate.index],
                            tma_bar_ptr=pipeline_k.producer_get_barrier(pstate),
                        )
                        pipeline_k.producer_commit(pstate)
                        pipeline_v.producer_acquire(pstate)
                        cute.copy(
                            tma_atom_v, tVgV[None, tile_beg + i],
                            tVsV[None, pstate.index],
                            tma_bar_ptr=pipeline_v.producer_get_barrier(pstate),
                        )
                        pipeline_v.producer_commit(pstate)
                        pstate.advance()
        else:
            # -------- consumer: QK^T, softmax, P@V --------
            tidx2 = tidx - 128
            lane = tidx2 % 32
            warp_in_wg = tidx2 // 32

            wg_qk = mma_qk.get_slice(0)
            wg_pv = mma_pv.get_slice(0)
            thr_qk = mma_qk.get_slice(tidx2)
            thr_pv = mma_pv.get_slice(tidx2)

            tSrK = mma_qk.make_fragment_A(wg_qk.partition_A(sK))
            tSrQ = mma_qk.make_fragment_B(wg_qk.partition_B(sQ))
            tOsVt = mma_pv.make_fragment_A(wg_pv.partition_A(sVt))
            tPsP = mma_pv.make_fragment_B(wg_pv.partition_B(sP))

            acc_s_shape = mma_qk.partition_shape_C((self.BLOCK_N, self.GROUP_M))
            acc_o_shape = mma_pv.partition_shape_C((self.HEAD_DIM, self.GROUP_M))
            acc_O = cute.make_rmem_tensor(acc_o_shape, ACC_TYPE)
            acc_O_mn = _acc_mn_view(acc_O)

            cS = cute.make_identity_tensor((self.BLOCK_N, self.GROUP_M))
            tScS = thr_qk.partition_C(cS)
            tScS_mn = _acc_mn_view(tScS)
            cO = cute.make_identity_tensor((self.HEAD_DIM, self.GROUP_M))
            tOcO = thr_pv.partition_C(cO)
            tOcO_mn = _acc_mn_view(tOcO)

            NR = cute.size(tScS_mn.shape[0])   # kv rows per thread
            NC = cute.size(tScS_mn.shape[1])   # q_heads per thread
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

            cstate = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.NUM_STAGES
            )
            for item in cutlass.range(worker, num_items, self.num_workers, unroll=1):
                kv_head = (item // self.batch) % self.kv_heads
                split = item // (self.batch * self.kv_heads)
                batch = item % self.batch
                tile_beg = split * tiles_total // self.num_splits
                tile_end = (split + 1) * tiles_total // self.num_splits
                n_tiles = tile_end - tile_beg

                gQ = cute.local_tile(
                    mQ[None, None, batch], (self.GROUP_M, self.HEAD_DIM), (kv_head, 0)
                )
                cute.copy(
                    q_tiled_copy, q_thr_copy.partition_S(gQ), q_thr_copy.partition_D(sQ)
                )
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.barrier(barrier_id=1, number_of_threads=128)

                acc_O.fill(0.0)
                row_max.fill(-cutlass.Float32.inf)
                row_sum.fill(0.0)

                for i in cutlass.range(n_tiles, unroll=1):
                    acc_S = cute.make_rmem_tensor(acc_s_shape, ACC_TYPE)
                    pipeline_k.consumer_wait(cstate)
                    _wgmma(
                        mma_qk, acc_S,
                        tSrK[None, None, None, cstate.index], tSrQ,
                        zero_init=True,
                    )
                    pipeline_k.consumer_release(cstate)

                    acc_S_mn = _acc_mn_view(acc_S)

                    # ---- online softmax over kv (columns are q_heads) ----
                    for c in cutlass.range_constexpr(NC):
                        for r in cutlass.range_constexpr(NR):
                            acc_S_mn[r, c] = acc_S_mn[r, c] * scale_log2
                    # per-warp column max (thread-local rows + butterfly)
                    warp_max = cute.make_rmem_tensor((NC,), ACC_TYPE)
                    for c in cutlass.range_constexpr(NC):
                        v = acc_S_mn[0, c]
                        for r in cutlass.range_constexpr(1, NR):
                            v = cute.arch.fmax(v, acc_S_mn[r, c])
                        v = cute.arch.fmax(v, cute.arch.shuffle_sync_bfly(v, offset=4))
                        v = cute.arch.fmax(v, cute.arch.shuffle_sync_bfly(v, offset=8))
                        v = cute.arch.fmax(v, cute.arch.shuffle_sync_bfly(v, offset=16))
                        warp_max[c] = v
                    # p = exp2(s - warp_max); per-warp sums
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
                    # cross-warp exchange of (max, sum) via smem
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
                        for r in cutlass.range_constexpr(cute.size(acc_O_mn.shape[0])):
                            acc_O_mn[r, c] = acc_O_mn[r, c] * alpha[c]
                        for r in cutlass.range_constexpr(NR):
                            acc_S_mn[r, c] = acc_S_mn[r, c] * corr
                    # store P^T (bf16) to smem for GEMM2
                    for c in cutlass.range_constexpr(NC):
                        col = tScS_mn[0, c][1]
                        for r in cutlass.range_constexpr(NR):
                            sP[col, tScS_mn[r, c][0]] = cutlass.BFloat16(acc_S_mn[r, c])
                    cute.arch.fence_proxy("async.shared", space="cta")
                    cute.arch.barrier(barrier_id=2, number_of_threads=128)

                    pipeline_v.consumer_wait(cstate)
                    _wgmma(
                        mma_pv, acc_O,
                        tOsVt[None, None, None, cstate.index], tPsP,
                        zero_init=False,
                    )
                    pipeline_v.consumer_release(cstate)
                    cstate.advance()

                # -------- epilogue: normalize + write partials --------
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
                        mLSE[split, kv_head, tOcO_mn[0, c][1], batch] = lse
                for c in cutlass.range_constexpr(NC):
                    col = tOcO_mn[0, c][1]
                    for r in cutlass.range_constexpr(cute.size(acc_O_mn.shape[0])):
                        mOpart[split, kv_head, col, tOcO_mn[r, c][0], batch] = cutlass.BFloat16(
                            acc_O_mn[r, c]
                        )

                # -------- fused combine: last CTA per (kv_head,batch) reduces --------
                if cutlass.const_expr(self.use_fused):
                    # Publish this item's O_partial/LSE to GPU scope, then bump the
                    # per-(kv_head,batch) arrival counter. The CTA that observes the
                    # last arrival (old == num_splits-1) does the cross-split
                    # LSE-weighted merge in-kernel, removing the separate combine
                    # launch. acq_rel on the atomic + the fence establish that all
                    # num_splits partials are globally visible to the reducer.
                    cute.arch.fence_acq_rel_gpu()
                    fused_flag_ptr = storage.fused_flag.data_ptr()
                    if tidx2 == 0:
                        cptr = mCounter.iterator + (kv_head + batch * self.kv_heads)
                        old = cute.arch.atomic_add(
                            cptr.llvm_ptr, cutlass.Int32(1),
                            sem="acq_rel", scope="gpu",
                        )
                        fused_flag_ptr[0] = old
                    cute.arch.barrier(barrier_id=1, number_of_threads=128)
                    is_last = fused_flag_ptr[0] == (self.num_splits - 1)
                    if is_last:
                        # thread tidx2 (0..127) owns output dim = tidx2.
                        for g in cutlass.range_constexpr(self.GROUP_M):
                            qh = kv_head * self.GROUP_M + g
                            lmax = -cutlass.Float32.inf
                            for s in cutlass.range_constexpr(self.num_splits):
                                lmax = cute.arch.fmax(lmax, mLSE[s, kv_head, g, batch])
                            denom = cutlass.Float32(0.0)
                            acc = cutlass.Float32(0.0)
                            for s in cutlass.range_constexpr(self.num_splits):
                                w = cute.math.exp2(
                                    mLSE[s, kv_head, g, batch] - lmax, fastmath=True
                                )
                                denom += w
                                acc += w * cutlass.Float32(
                                    mOpart[s, kv_head, g, tidx2, batch]
                                )
                            inv = (
                                0.0
                                if denom == 0.0 or denom != denom
                                else cute.arch.rcp_approx(denom)
                            )
                            mO[qh, tidx2, batch] = cutlass.BFloat16(acc * inv)

            # PDL: all O_partial/LSE writes for this CTA's items are issued.
            # Signal that dependent grids (the combine kernel) may launch now,
            # so the scheduler overlaps combine's launch/scheduling latency with
            # this partial kernel's tail wave instead of leaving a gap.
            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_launch_dependents()

    # ------------------------------------------------------------------
    # Combine kernel: log-sum-exp reduction across splits
    # ------------------------------------------------------------------
    @cute.kernel
    def combine_kernel(
        self,
        mOpart: cute.Tensor,  # (splits, HK, G, D, B) bf16
        mLSE: cute.Tensor,    # (splits, HK, G, B) fp32 (log2 domain)
        mO: cute.Tensor,      # (H, D, B) bf16
    ):
        q_head, batch, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        kv_head = q_head // self.GROUP_M
        g = q_head % self.GROUP_M

        # PDL: this combine CTA may have been launched early (overlapping the
        # partial kernel's tail). Wait until the producer signals that all
        # O_partial / LSE it depends on are written before reading them.
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

        smem = cutlass.utils.SmemAllocator()
        lse_buf = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout(self.num_splits), 16
        )
        for idx in cutlass.range(tidx, self.num_splits, 128):
            lse_buf[idx] = mLSE[idx, kv_head, g, batch]
        cute.arch.barrier()

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
            acc += ws[s] * cutlass.Float32(mOpart[s, kv_head, g, tidx, batch])
        inv = 0.0 if denom == 0.0 or denom != denom else cute.arch.rcp_approx(denom)
        mO[q_head, tidx, batch] = cutlass.BFloat16(acc * inv)


# ---------------------------------------------------------------------------
# Host wrapper (compile + buffer cache)
# ---------------------------------------------------------------------------

_COMPILED_CACHE = {}


def _to_cute_4d(tensor):
    return (
        from_dlpack(tensor, assumed_align=16)
        .mark_layout_dynamic(leading_dim=3)
        .mark_compact_shape_dynamic(
            mode=3,
            stride_order=tensor.dim_order(),
            divisibility=128 // D_TYPE.width,
        )
    )


def _sdpa_fallback(q, k, v, sm_scale, causal):
    import torch

    rep = q.shape[1] // k.shape[1]
    k_rep = k.repeat_interleave(rep, dim=1)
    v_rep = v.repeat_interleave(rep, dim=1)
    return torch.nn.functional.scaled_dot_product_attention(
        q, k_rep, v_rep, scale=sm_scale, is_causal=causal
    )


# The best (num_splits, num_stages, block_n) triple is card / clock dependent
# (wave quantization + combine cost + per-CTA contiguous-KV locality + TMA tile
# size all interact; specific arch+param combos suit specific GPUs). "auto"
# micro-benchmarks a curated joint candidate set once per (device, shape) and
# caches the winner.  Each candidate is (num_splits, num_stages, block_n).
_AUTO_CFG_CACHE = {}
_AUTO_CFG_CANDIDATES = (
    (52, 3, 128),   # strong on GPUs that like 128-wide TMA tiles
    (48, 3, 128),
    (48, 2, 128),
    (64, 3, 64),    # bn=64 fits 2 CTAs/SM -> more memory-level parallelism
    (48, 3, 64),
    (96, 3, 64),
    (32, 3, 128),   # strong on GPU 2/4/7
    (39, 3, 128),
)


def _autotune_config(q, k, v, sm_scale, use_pdl, kv_len, kv_heads, batch):
    import torch

    dev = q.device.index
    akey = (dev, kv_len, use_pdl, kv_heads, batch)
    cached = _AUTO_CFG_CACHE.get(akey)
    if cached is not None:
        return cached

    best_cfg = _AUTO_CFG_CANDIDATES[0]
    best_ms = float("inf")
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    for ns, st, bn in _AUTO_CFG_CANDIDATES:
        if kv_len % bn != 0:
            continue
        try:
            for _ in range(8):
                attention_decode(q, k, v, sm_scale=sm_scale, num_splits=ns,
                                 block_n=bn, num_stages=st, use_pdl=use_pdl)
            torch.cuda.synchronize()
            ms = float("inf")
            for _ in range(3):
                ev0.record()
                for _ in range(30):
                    attention_decode(q, k, v, sm_scale=sm_scale, num_splits=ns,
                                     block_n=bn, num_stages=st, use_pdl=use_pdl)
                ev1.record()
                torch.cuda.synchronize()
                ms = min(ms, ev0.elapsed_time(ev1) / 30)
            if ms < best_ms:
                best_ms, best_cfg = ms, (ns, st, bn)
        except Exception:
            continue
    _AUTO_CFG_CACHE[akey] = best_cfg
    return best_cfg


def attention_decode(
    q, k, v, sm_scale=None, causal=False,
    num_splits=NUM_SPLITS, block_n=BLOCK_N, num_stages=NUM_STAGES, num_workers=None,
    blocks_per_mp=1, use_pdl=True, use_fused=False,
):
    """q (B,H,1,D), k/v (B,HK,S,D) -> (B,H,1,D). bf16, contiguous, CUDA.

    num_splits may be an int, or "auto" to micro-benchmark and cache the best
    split count for this device/shape on first call.
    """
    import torch

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    batch, q_heads, q_len, head_dim = q.shape
    kv_heads, kv_len = k.shape[1], k.shape[2]

    # base support check (block_n-independent parts); auto may override block_n.
    supported = (
        q_len == 1
        and head_dim == HEAD_DIM
        and q.dtype == torch.bfloat16
        and q_heads == kv_heads * GROUP_M
        and q.is_cuda and k.is_cuda and v.is_cuda
        and q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    )
    if not supported:
        return _sdpa_fallback(q, k, v, sm_scale, causal)

    if num_splits == "auto":
        num_splits, num_stages, block_n = _autotune_config(
            q, k, v, sm_scale, use_pdl, kv_len, kv_heads, batch
        )

    if kv_len % block_n != 0:
        return _sdpa_fallback(q, k, v, sm_scale, causal)

    key = (
        q.device.index, batch, q_heads, kv_heads, kv_len, head_dim,
        num_splits, block_n, num_stages, num_workers, blocks_per_mp, use_pdl, use_fused, float(sm_scale),
    )

    with torch.cuda.device(q.device):
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        o = torch.empty_like(q)

        entry = _COMPILED_CACHE.get(key)
        if entry is None:
            if num_workers is None:
                # One CTA per (split, kv_head, batch) work-item: the scheduler
                # overlaps waves and no CTA drains/refills its pipeline between
                # items, keeping HBM continuously fed (measured 3.44 TB/s vs
                # 3.12 for a persistent SM-count grid).
                num_workers = num_splits * kv_heads * batch
            kernel = DecodeKernel(
                q_heads, kv_heads, kv_len, batch,
                num_splits=num_splits, block_n=block_n, num_stages=num_stages,
                sm_scale=sm_scale, num_workers=num_workers, blocks_per_mp=blocks_per_mp,
                use_pdl=use_pdl, use_fused=use_fused,
            )
            o_part = torch.empty(
                (num_splits, kv_heads, GROUP_M, head_dim, batch),
                dtype=torch.bfloat16, device=q.device,
            )
            lse_part = torch.empty(
                (num_splits, kv_heads, GROUP_M, batch),
                dtype=torch.float32, device=q.device,
            )
            counter = torch.zeros(kv_heads * batch, dtype=torch.int32, device=q.device)
            args = (
                _to_cute_4d(q), _to_cute_4d(k), _to_cute_4d(v),
                from_dlpack(o_part, assumed_align=16),
                from_dlpack(lse_part, assumed_align=16),
                _to_cute_4d(o), from_dlpack(counter, assumed_align=16), stream,
            )
            compiled = cute.compile(kernel, *args)
            entry = (compiled, o_part, lse_part, counter)
            _COMPILED_CACHE[key] = entry

        compiled, o_part, lse_part, counter = entry
        if use_fused:
            counter.zero_()
        compiled(
            _to_cute_4d(q), _to_cute_4d(k), _to_cute_4d(v),
            from_dlpack(o_part, assumed_align=16),
            from_dlpack(lse_part, assumed_align=16),
            _to_cute_4d(o), from_dlpack(counter, assumed_align=16), stream,
        )
    return o


run = attention_decode


if __name__ == "__main__":
    import argparse
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--q-heads", type=int, default=64)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--kv-len", type=int, default=131072)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-splits", type=int, default=NUM_SPLITS)
    parser.add_argument("--block-n", type=int, default=BLOCK_N)
    parser.add_argument("--stages", type=int, default=NUM_STAGES)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--blocks-per-mp", type=int, default=1)
    parser.add_argument("--no-pdl", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    torch.manual_seed(0)
    dev = "cuda"
    q = torch.randn(args.batch, args.q_heads, 1, args.head_dim, dtype=torch.bfloat16, device=dev)
    k = torch.randn(args.batch, args.kv_heads, args.kv_len, args.head_dim, dtype=torch.bfloat16, device=dev)
    v = torch.randn(args.batch, args.kv_heads, args.kv_len, args.head_dim, dtype=torch.bfloat16, device=dev)

    o = attention_decode(q, k, v, num_splits=args.num_splits, block_n=args.block_n, num_stages=args.stages, num_workers=args.num_workers, blocks_per_mp=args.blocks_per_mp, use_pdl=not args.no_pdl)

    rep = args.q_heads // args.kv_heads
    o_ref = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.repeat_interleave(rep, dim=1).float(), v.repeat_interleave(rep, dim=1).float()
    )
    err = (o.float() - o_ref).abs().max().item()
    rel = ((o.float() - o_ref).abs() / (o_ref.abs() + 1e-6)).max().item()
    print(f"max_abs_err={err:.4e} max_rel_err={rel:.4e}")

    for _ in range(args.warmup):
        attention_decode(q, k, v, num_splits=args.num_splits, block_n=args.block_n, num_stages=args.stages, num_workers=args.num_workers, blocks_per_mp=args.blocks_per_mp, use_pdl=not args.no_pdl)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        attention_decode(q, k, v, num_splits=args.num_splits, block_n=args.block_n, num_stages=args.stages, num_workers=args.num_workers, blocks_per_mp=args.blocks_per_mp, use_pdl=not args.no_pdl)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / args.iters
    kv_bytes = 2 * args.batch * args.kv_heads * args.kv_len * args.head_dim * 2
    print(f"latency={ms:.4f} ms  kv_bw={kv_bytes / ms / 1e9:.1f} GB/s  (num_splits={args.num_splits})")
