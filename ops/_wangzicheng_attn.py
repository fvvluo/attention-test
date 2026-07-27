import argparse
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Optional, Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Boolean, if_generate
import cutlass.cute.testing as testing
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
from cutlass.cute.runtime import from_dlpack

D_TYPE = cutlass.BFloat16
ACC_TYPE = cutlass.Float32

# Each CTA computes BLOCK_M query rows for one (batch, q_head).
BLOCK_M = 32
# Each mainloop iteration consumes BLOCK_N key/value rows. A wider KV tile
# amortizes online-softmax and TMA pipeline synchronization over more MMA work.
BLOCK_N = 64
# 2 warps (64 threads) so that the m16n8k16 warp MMA atom, tiled (2,1,1), covers
# exactly BLOCK_M=32 rows with no padding.
NUM_THREADS = 64
# Number of KV smem stages used for the TMA producer/consumer pipeline.
KV_STAGES = 2

# cute.compile performs substantial Python/DSL frontend work even when its
# lower-level compilation cache hits. Keep compiled callables by specialization
# so the benchmark hot path contains only argument wrapping, output allocation,
# and the kernel launch.
_COMPILED_ATTN_CACHE = {}


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
def _wgmma_gemm(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    operand_a: cute.Tensor,
    operand_b: cute.Tensor,
    zero_init: cutlass.Constexpr[bool],
    wg_wait: cutlass.Constexpr[int] = 0,
):
    """Issue a WGMMA k-loop and commit it as one group.

    wg_wait=-1 leaves the group in flight (async); otherwise waits until at most
    wg_wait groups are still pending.
    """
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
    if cutlass.const_expr(wg_wait >= 0):
        warpgroup.wait_group(wg_wait)


@cute.jit
def _warp_reduce4(value: cutlass.Float32, op: Callable) -> cutlass.Float32:
    value = op(value, cute.arch.shuffle_sync_bfly(value, offset=1))
    value = op(value, cute.arch.shuffle_sync_bfly(value, offset=2))
    return value


@cute.jit
def _wgmma_softmax_update(
    acc_s: cute.Tensor,
    row_max: cute.Tensor,
    row_sum: cute.Tensor,
    scale_log2: cutlass.Float32,
) -> cute.Tensor:
    scores = _make_acc_tensor_mn_view(acc_s)
    row_scale = cute.make_rmem_tensor(row_max.layout, cutlass.Float32)
    for row in cutlass.range_constexpr(cute.size(row_max)):
        score_row = scores[row, None].load()
        local_max = score_row.reduce(cute.ReductionOp.MAX, row_max[row], 0)
        new_max = _warp_reduce4(local_max, lambda x, y: cute.arch.fmax(x, y))
        safe_max = 0.0 if new_max == -cutlass.Float32.inf else new_max
        row_scale[row] = cute.math.exp2(
            (row_max[row] - safe_max) * scale_log2, fastmath=True
        )
        exp_scores = cute.math.exp2(
            score_row * scale_log2 - safe_max * scale_log2, fastmath=True
        )
        row_sum[row] = exp_scores.reduce(
            cute.ReductionOp.ADD, row_sum[row] * row_scale[row], 0
        )
        row_max[row] = new_max
        scores[row, None].store(exp_scores)
    return row_scale


@cute.jit
def _wgmma_rescale_output(acc_o: cute.Tensor, row_scale: cute.Tensor):
    output = _make_acc_tensor_mn_view(acc_o)
    for row in cutlass.range_constexpr(cute.size(row_scale)):
        output[row, None].store(output[row, None].load() * row_scale[row])


@cute.jit
def _wgmma_softmax_finalize(acc_o: cute.Tensor, row_sum: cute.Tensor):
    output = _make_acc_tensor_mn_view(acc_o)
    for row in cutlass.range_constexpr(cute.size(row_sum)):
        total = _warp_reduce4(row_sum[row], lambda x, y: x + y)
        inv_total = (
            1.0 if total == 0.0 or total != total else cute.arch.rcp_approx(total)
        )
        output[row, None].store(output[row, None].load() * inv_total)


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


class HopperWGMMAKernel:
    """SM90 long-sequence forward: 1 TMA producer WG + 2 consumer WGs (ping-pong).

    Each CTA owns a (batch, q_head, BLOCK_M=128 q-rows) tile. The two consumer
    warpgroups each compute 64 rows with m64n128k16 WGMMA and pass a named-barrier
    token so that exactly one warpgroup issues WGMMA while the other runs its
    softmax on CUDA cores -- the tensor core never sees a softmax bubble.
    """

    BLOCK_M = 128            # 64 rows per consumer warpgroup
    BLOCK_N = 128
    NUM_STAGES = 2           # K/V smem ring depth
    NUM_WG_THREADS = 128
    NUM_MMA_THREADS = 256    # 2 consumer warpgroups
    NUM_THREADS = 384        # + 1 producer warpgroup
    PRODUCER_REGS = 24
    CONSUMER_REGS = 240
    # Named barriers (0 is reserved for sync_threads).
    BAR_EPI_WG0 = 1          # per-consumer-WG epilogue barrier
    BAR_EPI_WG1 = 2
    BAR_TOKEN_WG0 = 3        # ping-pong token: WG0's turn to issue WGMMA
    BAR_TOKEN_WG1 = 4

    def __init__(
        self,
        batch: int,
        q_heads: int,
        kv_heads: int,
        seq_len: int,
        head_dim: int,
        causal: bool,
        sm_scale: float,
    ):
        if head_dim != 128:
            raise ValueError("WGMMA path requires head_dim=128")
        if seq_len % self.BLOCK_M != 0:
            raise ValueError("WGMMA path requires seq_len divisible by 128")
        self.batch = batch
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.causal = causal
        self.scale = sm_scale
        self.q_per_kv = q_heads // kv_heads

    @cute.jit
    def __call__(
        self,
        Q: cute.Tensor,
        K: cute.Tensor,
        V: cute.Tensor,
        output: cute.Tensor,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(
            not (Q.element_type == K.element_type == V.element_type == output.element_type)
        ):
            raise TypeError("All tensors must have the same data type")
        self._dtype = Q.element_type

        # (B,H,S,D) -> (S,D,(H,B)); the head/batch mode is outside the TMA tile.
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

        smem_atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                utils.LayoutEnum.ROW_MAJOR, self._dtype, self.head_dim
            ),
            self._dtype,
        )
        sQ_layout = cute.tile_to_shape(
            smem_atom, (self.BLOCK_M, self.head_dim), (0, 1)
        )
        sK_layout = cute.tile_to_shape(
            smem_atom,
            (self.BLOCK_N, self.head_dim, self.NUM_STAGES),
            (0, 1, 2),
        )
        sV_layout = sK_layout
        sO_layout = sQ_layout

        @cute.struct
        class SharedStorage:
            q_mbar: cute.struct.MemRange[cutlass.Int64, 1]
            k_mbars: cute.struct.MemRange[cutlass.Int64, self.NUM_STAGES * 2]
            v_mbars: cute.struct.MemRange[cutlass.Int64, self.NUM_STAGES * 2]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sQ_layout)], 1024
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sK_layout)], 1024
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sV_layout)], 1024
            ]

        tma_atom_q, tma_tensor_q = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            Q,
            sQ_layout,
            (self.BLOCK_M, self.head_dim),
        )
        tma_atom_k, tma_tensor_k = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            K,
            cute.select(sK_layout, mode=[0, 1]),
            (self.BLOCK_N, self.head_dim),
        )
        tma_atom_v, tma_tensor_v = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            V,
            cute.select(sV_layout, mode=[0, 1]),
            (self.BLOCK_N, self.head_dim),
        )

        # Two m64n128k16 WGMMA warpgroups along M: per-WG 64x128 accumulator.
        # tiler_mn is the single-atom tile; atom_layout_mnk=(2,1,1) places one
        # atom per consumer warpgroup, covering BLOCK_M=128 rows in total.
        tiled_mma_qk = sm90_utils.make_trivial_tiled_mma(
            self._dtype,
            self._dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            cutlass.Float32,
            atom_layout_mnk=(2, 1, 1),
            tiler_mn=(64, self.BLOCK_N),
        )
        tiled_mma_pv = sm90_utils.make_trivial_tiled_mma(
            self._dtype,
            self._dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.MN,
            cutlass.Float32,
            atom_layout_mnk=(2, 1, 1),
            tiler_mn=(64, self.head_dim),
            a_source=warpgroup.OperandSource.RMEM,
        )

        self.kernel(
            tma_atom_q,
            tma_tensor_q,
            tma_atom_k,
            tma_tensor_k,
            tma_atom_v,
            tma_tensor_v,
            output,
            cutlass.Float32(self.scale * math.log2(math.e)),
            sQ_layout,
            sK_layout,
            sV_layout,
            sO_layout,
            tiled_mma_qk,
            tiled_mma_pv,
            SharedStorage,
        ).launch(
            grid=(self.batch, self.q_heads, self.seq_len // self.BLOCK_M),
            block=[self.NUM_THREADS, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        tma_atom_q: cute.CopyAtom,
        mQ: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        mK: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        mV: cute.Tensor,
        mO: cute.Tensor,
        softmax_scale_log2: cutlass.Float32,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        batch, q_head, physical_m_block = cute.arch.block_idx()
        # Balance the causal triangle across each wave: adjacent heads traverse
        # query blocks in opposite directions while still covering every row.
        num_m_blocks = self.seq_len // self.BLOCK_M
        m_block = (
            physical_m_block
            if q_head % 2 == 0
            else num_m_blocks - physical_m_block - 1
        )
        kv_head = q_head // self.q_per_kv

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_q)
            cpasync.prefetch_descriptor(tma_atom_k)
            cpasync.prefetch_descriptor(tma_atom_v)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        mbar_q = storage.q_mbar.data_ptr()
        if warp_idx == 1:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(mbar_q, 1)

        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        # Two consumer warpgroups, one signaling thread each.
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 2)
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
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sVt = _transpose_smem_view(sV)

        if warp_idx >= 8:
            # Producer warpgroup: only warp 8 issues TMA; warps 9-11 idle.
            cute.arch.warpgroup_reg_dealloc(self.PRODUCER_REGS)
            if warp_idx == 8:
                self.load(
                    tma_atom_q,
                    mQ,
                    tma_atom_k,
                    mK,
                    tma_atom_v,
                    mV,
                    sQ,
                    sK,
                    sV,
                    pipeline_k,
                    pipeline_v,
                    mbar_q,
                    batch,
                    q_head,
                    kv_head,
                    m_block,
                )
        else:
            cute.arch.warpgroup_reg_alloc(self.CONSUMER_REGS)
            self.mma(
                tiled_mma_qk,
                tiled_mma_pv,
                mO,
                sQ,
                sK,
                sVt,
                sO_layout,
                pipeline_k,
                pipeline_v,
                mbar_q,
                tidx,
                softmax_scale_log2,
                batch,
                q_head,
                m_block,
            )

    @cute.jit
    def load(
        self,
        tma_atom_q,
        mQ,
        tma_atom_k,
        mK,
        tma_atom_v,
        mV,
        sQ,
        sK,
        sV,
        pipeline_k,
        pipeline_v,
        mbar_q,
        batch,
        q_head,
        kv_head,
        m_block,
    ):
        gQ = cute.local_tile(
            mQ[None, None, (q_head, batch)],
            (self.BLOCK_M, self.head_dim),
            (m_block, 0),
        )
        gK = cute.local_tile(
            mK[None, None, (kv_head, batch)],
            (self.BLOCK_N, self.head_dim),
            (None, 0),
        )
        gV = cute.local_tile(
            mV[None, None, (kv_head, batch)],
            (self.BLOCK_N, self.head_dim),
            (None, 0),
        )
        tQsQ, tQgQ = cpasync.tma_partition(
            tma_atom_q,
            0,
            cute.make_layout(1),
            cute.group_modes(sQ, 0, 2),
            cute.group_modes(gQ, 0, 2),
        )
        tKsK, tKgK = cpasync.tma_partition(
            tma_atom_k,
            0,
            cute.make_layout(1),
            cute.group_modes(sK, 0, 2),
            cute.group_modes(gK, 0, 2),
        )
        tVsV, tVgV = cpasync.tma_partition(
            tma_atom_v,
            0,
            cute.make_layout(1),
            cute.group_modes(sV, 0, 2),
            cute.group_modes(gV, 0, 2),
        )

        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(
                mbar_q,
                cute.size_in_bytes(self._dtype, cute.select(sQ.layout, mode=[0, 1])),
            )
        cute.copy(tma_atom_q, tQgQ, tQsQ, tma_bar_ptr=mbar_q)

        n_block_max = cute.ceil_div(
            (m_block + 1) * self.BLOCK_M, self.BLOCK_N
        )
        producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.NUM_STAGES
        )
        for i in cutlass.range(n_block_max, unroll=1):
            n_block = n_block_max - i - 1
            pipeline_k.producer_acquire(producer_state)
            cute.copy(
                tma_atom_k,
                tKgK[None, n_block],
                tKsK[None, producer_state.index],
                tma_bar_ptr=pipeline_k.producer_get_barrier(producer_state),
            )
            pipeline_k.producer_commit(producer_state)
            pipeline_v.producer_acquire(producer_state)
            cute.copy(
                tma_atom_v,
                tVgV[None, n_block],
                tVsV[None, producer_state.index],
                tma_bar_ptr=pipeline_v.producer_get_barrier(producer_state),
            )
            pipeline_v.producer_commit(producer_state)
            producer_state.advance()

    @cute.jit
    def mma(
        self,
        tiled_mma_qk,
        tiled_mma_pv,
        mO,
        sQ,
        sK,
        sVt,
        sO_layout,
        pipeline_k,
        pipeline_v,
        mbar_q,
        tidx,
        softmax_scale_log2,
        batch,
        q_head,
        m_block,
    ):
        wg_idx = cute.arch.make_warp_uniform(tidx // self.NUM_WG_THREADS)
        wg_thread_layout = cute.make_layout(2, stride=self.NUM_WG_THREADS)
        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        wg_mma_qk = tiled_mma_qk.get_slice(wg_thread_layout(wg_idx))
        wg_mma_pv = tiled_mma_pv.get_slice(wg_thread_layout(wg_idx))
        tSrQ = tiled_mma_qk.make_fragment_A(wg_mma_qk.partition_A(sQ))
        tSrK = tiled_mma_qk.make_fragment_B(wg_mma_qk.partition_B(sK))
        tOrVt = tiled_mma_pv.make_fragment_B(wg_mma_pv.partition_B(sVt))

        acc_s_shape = tiled_mma_qk.partition_shape_C(
            (self.BLOCK_M, self.BLOCK_N)
        )
        acc_o_shape = tiled_mma_pv.partition_shape_C(
            (self.BLOCK_M, self.head_dim)
        )
        acc_O = cute.make_rmem_tensor(acc_o_shape, cutlass.Float32)
        acc_O.fill(0.0)
        tOrP = cute.make_rmem_tensor(
            _convert_layout_acc_frgA(cute.make_layout(acc_s_shape)),
            self._dtype,
        )
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

        cute.arch.mbarrier_wait(mbar_q, phase=0)
        # Hand the first WGMMA-issue token to WG0: WG1 arrives WG0's barrier.
        if wg_idx == 1:
            cute.arch.barrier_arrive(
                barrier_id=self.BAR_TOKEN_WG0,
                number_of_threads=self.NUM_WG_THREADS,
            )

        state_k = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.NUM_STAGES
        )
        state_v = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.NUM_STAGES
        )
        n_block_max = cute.ceil_div(
            (m_block + 1) * self.BLOCK_M, self.BLOCK_N
        )
        row_base = m_block * self.BLOCK_M

        # Prologue: issue QK for the first (diagonal) KV block, asynchronously.
        acc_S = cute.make_rmem_tensor(acc_s_shape, cutlass.Float32)
        pipeline_k.consumer_wait(state_k)
        _wgmma_gemm(
            tiled_mma_qk,
            acc_S,
            tSrQ,
            tSrK[None, None, None, state_k.index],
            zero_init=True,
            wg_wait=-1,
        )

        # Software-pipelined mainloop over KV blocks n = n_block_max-1 .. 1.
        # Invariant on entry: QK[n] is in flight; state_k/state_v index stage(n).
        # The final block (n = 0) is peeled to keep the loop body branch-free.
        for i in cutlass.range(n_block_max - 1, unroll=1):
            n_block = n_block_max - 1 - i
            # QK[n] done (also PV[n+1], waited last iteration); K stage is free.
            warpgroup.wait_group(0)
            pipeline_k.consumer_release(state_k)

            if i == 0:
                # Diagonal block: apply the causal mask in registers.
                acc_S_mn = _make_acc_tensor_mn_view(acc_S)
                col_base = n_block * self.BLOCK_N
                for r in cutlass.range_constexpr(cute.size(acc_S_mn.shape[0])):
                    row_idx = row_base + tScS_mn[r, 0][0]
                    for c in cutlass.range_constexpr(cute.size(acc_S_mn.shape[1])):
                        col_idx = col_base + tScS_mn[0, c][1]
                        if col_idx >= self.seq_len or col_idx > row_idx:
                            acc_S_mn[r, c] = -cutlass.Float32.inf

            # Softmax on CUDA cores while the other warpgroup owns the tensor core.
            row_scale = _wgmma_softmax_update(
                acc_S, row_max, row_sum, softmax_scale_log2
            )
            _wgmma_rescale_output(acc_O, row_scale)
            tOrP_acc = cute.make_tensor(
                acc_S.iterator, _convert_layout_acc_frgA(acc_S.layout)
            )
            tOrP.store(tOrP_acc.load().to(self._dtype))

            # Take the WGMMA token: PV[n] then QK[n-1], both async.
            cute.arch.barrier(
                barrier_id=my_token_bar, number_of_threads=self.NUM_WG_THREADS
            )
            pipeline_v.consumer_wait(state_v)
            _wgmma_gemm(
                tiled_mma_pv,
                acc_O,
                tOrP,
                tOrVt[None, None, None, state_v.index],
                zero_init=False,
                wg_wait=-1,
            )
            state_k.advance()
            pipeline_k.consumer_wait(state_k)
            _wgmma_gemm(
                tiled_mma_qk,
                acc_S,
                tSrQ,
                tSrK[None, None, None, state_k.index],
                zero_init=True,
                wg_wait=-1,
            )
            cute.arch.barrier_arrive(
                barrier_id=other_token_bar,
                number_of_threads=self.NUM_WG_THREADS,
            )

            # Wait for PV[n] only; QK[n-1] stays in flight for the next iteration.
            warpgroup.wait_group(1)
            pipeline_v.consumer_release(state_v)
            state_v.advance()

        # Peeled final iteration (n = 0): no next QK to issue.
        warpgroup.wait_group(0)
        pipeline_k.consumer_release(state_k)
        if n_block_max - 1 == 0:
            # Single-block tile: the diagonal block is block 0, mask it here.
            acc_S_mn = _make_acc_tensor_mn_view(acc_S)
            for r in cutlass.range_constexpr(cute.size(acc_S_mn.shape[0])):
                row_idx = row_base + tScS_mn[r, 0][0]
                for c in cutlass.range_constexpr(cute.size(acc_S_mn.shape[1])):
                    col_idx = tScS_mn[0, c][1]
                    if col_idx >= self.seq_len or col_idx > row_idx:
                        acc_S_mn[r, c] = -cutlass.Float32.inf
        row_scale = _wgmma_softmax_update(
            acc_S, row_max, row_sum, softmax_scale_log2
        )
        _wgmma_rescale_output(acc_O, row_scale)
        tOrP_acc = cute.make_tensor(
            acc_S.iterator, _convert_layout_acc_frgA(acc_S.layout)
        )
        tOrP.store(tOrP_acc.load().to(self._dtype))
        cute.arch.barrier(
            barrier_id=my_token_bar, number_of_threads=self.NUM_WG_THREADS
        )
        pipeline_v.consumer_wait(state_v)
        _wgmma_gemm(
            tiled_mma_pv,
            acc_O,
            tOrP,
            tOrVt[None, None, None, state_v.index],
            zero_init=False,
            wg_wait=0,
        )
        cute.arch.barrier_arrive(
            barrier_id=other_token_bar,
            number_of_threads=self.NUM_WG_THREADS,
        )
        pipeline_v.consumer_release(state_v)

        _wgmma_softmax_finalize(acc_O, row_sum)

        # R2S with stmatrix, then a vectorized per-warpgroup store to global.
        sO = cute.make_tensor(
            cute.recast_ptr(sQ.iterator, sO_layout.inner, dtype=self._dtype),
            sO_layout.outer,
        )
        rO = cute.make_fragment_like(acc_O, self._dtype)
        rO.store(acc_O.load().to(self._dtype))
        cute.arch.barrier(
            barrier_id=epi_bar, number_of_threads=self.NUM_WG_THREADS
        )
        smem_copy_atom_O = cute.make_copy_atom(
            warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self._dtype,
        )
        smem_thr_copy_O = cute.make_tiled_copy_C(
            smem_copy_atom_O, tiled_mma_pv
        ).get_slice(tidx)
        cute.copy(
            smem_copy_atom_O,
            smem_thr_copy_O.retile(rO),
            smem_thr_copy_O.partition_D(sO),
        )
        cute.arch.barrier(
            barrier_id=epi_bar, number_of_threads=self.NUM_WG_THREADS
        )

        gO_full = cute.local_tile(
            mO[None, None, (q_head, batch)],
            (self.BLOCK_M, self.head_dim),
            (m_block, 0),
        )
        gO = cute.local_tile(gO_full, (64, self.head_dim), (wg_idx, 0))
        sO_wg = cute.local_tile(sO, (64, self.head_dim), (wg_idx, 0))
        copy_elems = 128 // self._dtype.width
        atom_O = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self._dtype,
            num_bits_per_copy=128,
        )
        tO_shape_dim_1 = self.head_dim // copy_elems
        tiled_copy_O = cute.make_tiled_copy_tv(
            atom_O,
            cute.make_layout(
                (self.NUM_WG_THREADS // tO_shape_dim_1, tO_shape_dim_1),
                stride=(tO_shape_dim_1, 1),
            ),
            cute.make_layout((1, copy_elems)),
        )
        thr_copy_O = tiled_copy_O.get_slice(tidx % self.NUM_WG_THREADS)
        tOsO = thr_copy_O.partition_S(sO_wg)
        tOgO = thr_copy_O.partition_D(gO)
        tOrO = cute.make_fragment_like(tOgO, self._dtype)
        cute.copy(tiled_copy_O, tOsO, tOrO)
        cute.copy(tiled_copy_O, tOrO, tOgO)


def _sdpa_fallback(Q, K, V, causal, sm_scale):
    """Torch SDPA fallback for shapes the WGMMA kernel does not cover."""
    import torch

    rep = Q.shape[1] // K.shape[1]
    K_rep = K.repeat_interleave(rep, dim=1)
    V_rep = V.repeat_interleave(rep, dim=1)
    return torch.nn.functional.scaled_dot_product_attention(
        Q, K_rep, V_rep, scale=sm_scale, is_causal=causal
    )


def run(
    Q: "torch.Tensor",
    K: "torch.Tensor",
    V: "torch.Tensor",
    causal: bool = True,
    sm_scale=None,
):
    import torch

    if Q.ndim != 4 or K.ndim != 4 or V.ndim != 4:
        raise ValueError("Q, K, and V must all be rank-4 tensors")
    if K.shape != V.shape:
        raise ValueError("K and V must have identical shapes")

    batch, q_heads, seq_len, head_dim = Q.shape
    k_batch, kv_heads, k_seq_len, k_head_dim = K.shape
    if (k_batch, k_seq_len, k_head_dim) != (batch, seq_len, head_dim):
        raise ValueError("Q, K, and V must have matching batch, sequence, and head dimensions")
    if not (Q.is_cuda and K.is_cuda and V.is_cuda):
        raise ValueError("Q, K, and V must be CUDA tensors")
    if not (Q.device == K.device == V.device):
        raise ValueError("Q, K, and V must be on the same CUDA device")
    if not (Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()):
        raise ValueError("Q, K, and V must be contiguous")

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    def to_cute(tensor):
        return (
            from_dlpack(tensor, assumed_align=16)
            .mark_layout_dynamic(leading_dim=3)
            .mark_compact_shape_dynamic(
                mode=3,
                stride_order=tensor.dim_order(),
                divisibility=128 // D_TYPE.width,
            )
        )

    capability = torch.cuda.get_device_capability(Q.device)
    use_wgmma = (
        capability[0] == 9
        and head_dim == 128
        and causal
        and seq_len % HopperWGMMAKernel.BLOCK_M == 0
    )
    if not use_wgmma:
        return _sdpa_fallback(Q, K, V, causal, sm_scale)
    backend = "sm90_wgmma"

    cache_key = (
        backend,
        Q.device.index,
        capability,
        tuple(Q.shape),
        tuple(Q.stride()),
        tuple(K.shape),
        tuple(K.stride()),
        tuple(V.stride()),
        Q.dtype,
        bool(causal),
        float(sm_scale),
    )

    with torch.cuda.device(Q.device):
        torch_stream = torch.cuda.current_stream()
        stream = cuda.CUstream(torch_stream.cuda_stream)
        Q_tensor = to_cute(Q)
        K_tensor = to_cute(K)
        V_tensor = to_cute(V)
        output = torch.empty_like(Q)
        output_tensor = to_cute(output)

        compiled_attn = _COMPILED_ATTN_CACHE.get(cache_key)
        if compiled_attn is None:
            attn = HopperWGMMAKernel(
                batch, q_heads, kv_heads, seq_len, head_dim, causal, sm_scale
            )
            compiled_attn = cute.compile(
                attn, Q_tensor, K_tensor, V_tensor, output_tensor, stream
            )
            _COMPILED_ATTN_CACHE[cache_key] = compiled_attn

        compiled_attn(Q_tensor, K_tensor, V_tensor, output_tensor, stream)
    return output


def _ref_attention(Q, K, V, causal, sm_scale):
    import torch

    q_heads = Q.shape[1]
    kv_heads = K.shape[1]
    rep = q_heads // kv_heads
    K_rep = K.repeat_interleave(rep, dim=1)
    V_rep = V.repeat_interleave(rep, dim=1)
    return torch.nn.functional.scaled_dot_product_attention(
        Q, K_rep, V_rep, scale=sm_scale, is_causal=causal
    )
