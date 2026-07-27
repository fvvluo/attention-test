# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
#
# ---------------------------------------------------------------------------
# DERIVED FILE — Liu Xiaochen personal SM90 attention kernel.
#
# Source: flash-attention-baseline/flash_attn/cute/flash_fwd_sm90.py
#   (the teaching baseline provided by the TA, itself derived from the
#    original Dao-AILab FlashAttention SM90 forward kernel).
# License: retained from the original file (see copyright line above).
#
# This is an independent copy living under ops/liuxiaochen_sm90_attention/.
# The class is renamed to LiuXiaochenFlashAttentionForwardSm90 so the personal
# run path drives THIS kernel, not the baseline's FlashAttentionForwardSm90.
# It reuses the common flash_attn.cute utilities (utils / mask / softmax /
# seqlen_info / tile_scheduler / pipeline / FlashAttentionForwardBase), which
# is explicitly permitted.
#
# First version keeps the baseline logic byte-for-byte identical: none of the
# six removed optimizations are restored yet.  The EXERCISE (n) markers below
# are preserved verbatim so they can be addressed one at a time later.
# ---------------------------------------------------------------------------
#
# SM90 (Hopper) forward pass for flash attention -- DE-OPTIMIZED TEACHING VERSION.
#
# This file is derived from the original SM90 kernel with six optimizations deliberately
# removed. It is functionally correct (under 1x64x8x131072x128) but substantially slower.
# Each removal is marked with an `EXERCISE (n)` comment at the site where the technique used to live
# (typically, more code is required to modify; they might be far away from the `EXERCISE (n)` comment).
#
#   [EASY] EXERCISE (1)  LPT tile scheduling            -> plain SingleTileScheduler
#   [EASY] EXERCISE (2)  Causal n-block skipping        -> every KV block is visited and masked
#   [MEDIUM] EXERCISE (3)  Warp specialization          -> one merged 256-thread mainloop
#   [EASY] EXERCISE (4)  Register redistribution        -> setmaxnreg calls deleted
#   [MEDIUM] EXERCISE (5)  Intra-warpgroup overlap      -> QK and PV serialized in one iteration
#   [HARD] EXERCISE (6)  Inter-warpgroup ping-pong      -> warp scheduler barriers deleted
#
# If you are stuck, refer to https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/flash_fwd_sm90.py.
#

from typing import Callable, Optional
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.utils import LayoutEnum
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.base_dsl.arch import Arch

from quack import copy_utils
from quack import layout_utils
from quack import sm90_utils
from quack.cute_dsl_utils import ParamsBase

from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from flash_attn.cute import utils
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute.softmax import Softmax
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.block_info import BlockInfo  # EXERCISE (2): causal n-block skipping
from flash_attn.cute.block_sparsity import BlockSparseTensors  # signature only
from flash_attn.cute.utils import AuxData  # signature only
from flash_attn.cute import pipeline as pipeline_custom
from flash_attn.cute.tile_scheduler import (
    TileSchedulerArguments,
    SingleTileScheduler,
)

from flash_attn.cute.flash_fwd import FlashAttentionForwardBase


class LiuXiaochenFlashAttentionForwardSm90Exercise2(FlashAttentionForwardBase):
    def __init__(
        self,
        *args,
        intra_wg_overlap: bool = True,
        mma_pv_is_rs: bool = True,
        paged_kv_non_tma: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert mma_pv_is_rs and not paged_kv_non_tma, (
            "this de-optimized SM90 kernel only supports mma_pv_is_rs=True, "
            "paged_kv_non_tma=False"
        )
        # EXERCISE (5): intra_wg_overlap is accepted for signature compatibility but ignored.
        # The mainloop below runs QK and PV back to back inside a single iteration. Is this efficient?
        self.intra_wg_overlap, self.mma_pv_is_rs = False, mma_pv_is_rs
        self.use_tma_Q = self.use_tma_O = self.use_tma_KV = True
        self.buffer_align_bytes = 1024
        self.cluster_shape_mn = (1, 1)
        assert self.arch.is_family_of(Arch.sm_90a), "Only SM 9.x is supported"
        assert not self.pack_gqa and not self.Q_in_regs
        assert not self.is_local, "sliding window / local attention was stripped out"
        assert self.score_mod is None and self.mask_mod is None
        assert self.tile_hdim == 128 and self.tile_hdimv == 128

    def _get_smem_layout_atom(self):
        # Q/K/V/O all have leading (contiguous) extent 128, so one atom covers them all.
        # sP is None: MMA-PV takes P straight from registers (mma_pv_is_rs).
        atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, self.dtype, self.tile_hdim),
            self.dtype,
        )
        return atom, atom, atom, atom, None

    def _get_tiled_mma(self):
        make_mma = partial(
            sm90_utils_basic.make_trivial_tiled_mma,
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            acc_dtype=Float32,
            atom_layout_mnk=(self.tile_m // 64, 1, 1),
        )
        tiled_mma_qk = make_mma(warpgroup.OperandMajorMode.K, tiler_mn=(64, self.tile_n))
        tiled_mma_pv = make_mma(
            warpgroup.OperandMajorMode.MN,
            tiler_mn=(64, self.tile_hdimv),
            a_source=warpgroup.OperandSource.RMEM,
        )
        return tiled_mma_qk, tiled_mma_pv

    def _get_shared_storage_cls(self):
        sQ_struct, sK_struct, sV_struct = [
            cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(layout)], self.buffer_align_bytes
            ]
            for layout in (self.sQ_layout, self.sK_layout, self.sV_layout)
        ]
        # 1 stage * 2 for Q (full + empty), num_stages * 2 each for K and V.
        mbar_Q_struct = cute.struct.MemRange[cutlass.Int64, 1 * 2]
        mbar_K_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        mbar_V_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]

        @cute.struct
        class SharedStorageQKV:
            mbar_ptr_Q: mbar_Q_struct
            mbar_ptr_K: mbar_K_struct
            mbar_ptr_V: mbar_V_struct
            sV: sV_struct
            sQ: sQ_struct
            sK: sK_struct

        return SharedStorageQKV

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (b, s_q, h, d)
        mK: cute.Tensor,  # (b, s_k, h_k, d)
        mV: cute.Tensor,  # (b, s_k, h_k, dv)
        mO: cute.Tensor,  # (b, s_q, h, dv)
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mPageTable: Optional[cute.Tensor] = None,  # (b_k, max_num_pages_per_seq)
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        learnable_sink: Optional[cute.Tensor] = None,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        aux_data: AuxData = AuxData(),
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        """Configures and launches the flash attention kernel.

        mQ/mK/mV/mO have the same dtype (fp16 or bf16) and the same layout:
        (batch_size, seqlen, num_head, head_dim):(_, _, _, 1)
        """
        assert all(
            t is None
            for t in (
                mCuSeqlensQ,
                mCuSeqlensK,
                mSeqUsedQ,
                mSeqUsedK,
                mPageTable,
                window_size_left,
                window_size_right,
                learnable_sink,
                blocksparse_tensors,
                aux_data.tensors,
            )
        ), "varlen, paged KV, local windows, sinks, block sparsity and score/mask mods were stripped"

        self._check_type(
            mQ.element_type,
            mK.element_type,
            mV.element_type,
            mO.element_type,
            mLSE.element_type if const_expr(mLSE is not None) else None,
            None,
            None,
            None,
            None,
        )

        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        # (b, s, h, d) -> (s, d, h, b)
        mQ, mK, mV, mO = [layout_utils.select(t, [1, 3, 2, 0]) for t in (mQ, mK, mV, mO)]
        if const_expr(mLSE is not None):
            mLSE = layout_utils.select(mLSE, [2, 1, 0])

        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        self.num_threads_per_warp_group = 128
        self.num_mma_threads = tiled_mma_qk.size
        self.num_wg_mma = self.num_mma_threads // self.num_threads_per_warp_group
        assert self.num_wg_mma == 2

        # EXERCISE (3): no warp specialization.
        # The same threads that run the WGMMAs also issue the TMA loads, sharing a single program counter.
        # How does adding warp specialization impact performance?
        self.num_threads = self.num_mma_threads
        self.num_producer_threads = 32  # one warp still elects to issue the TMAs
        self.num_Q_load_threads = self.num_threads_per_warp_group
        self.num_epilogue_threads = self.num_mma_threads

        # EXERCISE (4): the num_mma_regs / num_producer_regs table is gone. Without warp
        # specialization there is nobody to donate registers to, so every thread keeps the
        # compiler's default allocation out of the 65536-register file.
        self._setup_attributes()

        self.sQ_layout, self.sK_layout, self.sV_layout, self.sO_layout = [
            sm90_utils.make_smem_layout(self.dtype, LayoutEnum.ROW_MAJOR, shape, stage)
            for shape, stage in [
                ((self.tile_m, self.tile_hdim), None),
                ((self.tile_n, self.tile_hdim), self.num_stages),
                ((self.tile_n, self.tile_hdimv), self.num_stages),
                ((self.tile_m, self.tile_hdimv), None),
            ]
        ]
        SharedStorage = self._get_shared_storage_cls()

        # TMA atoms for Q/K/V (G2S) and O (S2G). No multicast (cluster is 1x1).
        self.tma_copy_bytes = {
            name: cute.size_in_bytes(self.dtype, cute.select(layout, mode=[0, 1]))
            for name, layout in [("Q", self.sQ_layout), ("K", self.sK_layout), ("V", self.sV_layout)]
        }
        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mQ, self.sQ_layout, (self.tile_m, self.tile_hdim)
        )
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mK,
            cute.select(self.sK_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdim),
            1,
        )
        tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mV,
            cute.select(self.sV_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdimv),
            1,
        )
        tma_atom_O, tma_tensor_O = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), mO, self.sO_layout, (self.tile_m, self.tile_hdimv)
        )

        # EXERCISE (1): What does the `lpt` field of `TileSchedulerArguments` mean? Does this impact efficiency?
        TileScheduler = SingleTileScheduler
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), self.tile_m),
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3]),
            1,  # num_splits
            cute.size(mK.shape[0]),
            mQ.shape[1],
            mV.shape[1],
            total_q=cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=(self.tile_m, self.tile_n),
            qhead_per_kvhead_packgqa=1,
            element_size=self.dtype.width // 8,
            is_persistent=False,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(softmax_scale, None)

        self.kernel(
            tma_tensor_Q,
            tma_tensor_K,
            tma_tensor_V,
            tma_tensor_O,
            mLSE,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            softmax_scale_log2,
            softmax_scale,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sO_layout,
            self.gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            tile_sched_params,
            TileScheduler,
            SharedStorage,
        ).launch(
            grid=TileScheduler.get_grid_shape(tile_sched_params),
            block=[self.num_threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_O: cute.CopyAtom,
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        SharedStorage: cutlass.Constexpr[Callable],
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            for tma_atom in (tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_O):
                cpasync.prefetch_descriptor(tma_atom)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # The pipelines survive the removal of warp specialization: the producer group is
        # still a single elected thread (warp 0), but it is now one of the 8 consumer warps
        # rather than a separate warpgroup. Producer and consumer states therefore advance
        # in lockstep and the pipeline never runs ahead.
        ThreadCooperativeGroup = partial(pipeline.CooperativeGroup, pipeline.Agent.Thread)
        tma_warp = ThreadCooperativeGroup(1)
        mma_warps = ThreadCooperativeGroup(self.num_mma_threads // cute.arch.WARP_SIZE)
        make_pipe = partial(
            pipeline_custom.PipelineTmaAsync.create,
            producer_group=tma_warp,
            consumer_group=mma_warps,
            defer_sync=True,
        )
        pipeline_q = make_pipe(
            barrier_storage=storage.mbar_ptr_Q.data_ptr(),
            num_stages=1,
            tx_count=self.tma_copy_bytes["Q"],
        )
        pipeline_k = make_pipe(
            barrier_storage=storage.mbar_ptr_K.data_ptr(),
            num_stages=self.num_stages,
            tx_count=self.tma_copy_bytes["K"],
        )
        pipeline_v = make_pipe(
            barrier_storage=storage.mbar_ptr_V.data_ptr(),
            num_stages=self.num_stages,
            tx_count=self.tma_copy_bytes["V"],
        )
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        # Transpose view of V, i.e. (head_dim_v, tile_n), for the PV tiled mma
        sVt = layout_utils.transpose_view(sV)
        sO = storage.sQ.get_tensor(sO_layout.outer, swizzle=sO_layout.inner, dtype=self.dtype)

        SeqlenInfoCls = partial(
            SeqlenInfoQK.create, seqlen_q_static=mQ.shape[0], seqlen_k_static=mK.shape[0]
        )
        AttentionMaskCls = partial(AttentionMask, self.tile_m, self.tile_n)
        TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # EXERCISE (3): a single entry point for all 256 threads. There is no
        # `if warp_idx < 4: load(...) else: mma(...)` split and no setmaxregister pair.
        tidx, _, _ = cute.arch.thread_idx()
        self.mainloop(
            tiled_mma_qk,
            tiled_mma_pv,
            mQ,
            mK,
            mV,
            mO,
            mLSE,
            sQ,
            sK,
            sV,
            sVt,
            sO,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            pipeline_q,
            pipeline_k,
            pipeline_v,
            gmem_tiled_copy_O,
            tidx,
            warp_idx,
            softmax_scale_log2,
            softmax_scale,
            SeqlenInfoCls,
            AttentionMaskCls,
            TileSchedulerCls,
        )

    @cute.jit
    def load_KV(
        self,
        tma_load_fn: Callable,
        pipeline_kv: pipeline.PipelineAsync,
        block: Int32,
        producer_state: pipeline.PipelineState,
    ):
        pipeline_kv.producer_acquire(producer_state)
        tma_load_fn(src_idx=block, producer_state=producer_state)
        pipeline_kv.producer_commit(producer_state)

    @cute.jit
    def mainloop(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sVt: cute.Tensor,
        sO: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_O: cute.CopyAtom,
        pipeline_q: pipeline.PipelineAsync,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        gmem_tiled_copy_O: cute.TiledCopy,
        tidx: Int32,
        warp_idx: Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
    ):
        """Loads and computes in one instruction stream, one N block at a time."""
        # tidx runs 0..255 now (no producer warpgroup to subtract off).
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
        wg_thread_layout = cute.make_layout(
            self.num_wg_mma, stride=self.num_threads_per_warp_group
        )
        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        wg_mma_qk = tiled_mma_qk.get_slice(wg_thread_layout(warp_group_idx))
        wg_mma_pv = tiled_mma_pv.get_slice(wg_thread_layout(warp_group_idx))
        _, tSrQ, tSrK = sm90_utils.partition_fragment_ABC(
            wg_mma_qk, (self.tile_m, self.tile_n, self.tile_hdim), sQ, sK
        )
        mma_qk_fn = partial(
            sm90_utils.gemm_zero_init, tiled_mma_qk, (self.tile_m, self.tile_n), tSrQ, tSrK
        )
        # A operand of the PV gemm is None -> P stays in registers (mma_pv_is_rs)
        acc_O, tOrP, tOrVt = sm90_utils.partition_fragment_ABC(
            wg_mma_pv, (self.tile_m, self.tile_hdimv, self.tile_n), None, sVt
        )
        mma_pv_fn = partial(sm90_utils.gemm_w_idx, tiled_mma_pv, acc_O, tOrP, tOrVt)

        # EXERCISE (6): mma_init() used to prime the warp-scheduler token ring here by
        # having warpgroup 1 arrive on NamedBarrierFwd.WarpSchedulerWG1. Both MMA
        # warpgroups now issue their WGMMAs whenever the hardware warp scheduler pleases.

        q_producer_phase = Int32(1)
        q_consumer_phase = Int32(0)
        kv_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_stages
        )
        kv_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_stages
        )
        softmax = Softmax.create(
            softmax_scale_log2,
            num_rows=acc_O.shape[0][0] * acc_O.shape[1],
            softmax_scale=softmax_scale,
        )

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            head_idx_kv = head_idx // self.qhead_per_kvhead  # GQA
            mask = AttentionMaskCls(seqlen)
            # Exercise (2). Every block gets the full mask treatment, including
            # seqlen masking on blocks that are entirely inside seqlen_k.
            mask_fn = partial(
                mask.apply_mask,
                batch_idx=batch_idx,
                head_idx=head_idx,
                m_block=m_block,
                thr_mma=thr_mma_qk,
                mask_seqlen=True,
                mask_causal=self.is_causal,
                mask_local=False,
                mask_mod=None,
            )

            gQ = cute.local_tile(
                mQ[None, None, head_idx, batch_idx], (self.tile_m, self.tile_hdim), (m_block, 0)
            )
            load_Q, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q, 0, cute.make_layout(1), gQ, sQ, single_stage=True
            )
            gK = cute.local_tile(
                mK[None, None, head_idx_kv, batch_idx], (self.tile_n, self.tile_hdim), (None, 0)
            )
            gV = cute.local_tile(
                mV[None, None, head_idx_kv, batch_idx], (self.tile_n, self.tile_hdimv), (None, 0)
            )
            tma_load_K, _, _ = copy_utils.tma_get_copy_fn(tma_atom_K, 0, cute.make_layout(1), gK, sK)
            tma_load_V, _, _ = copy_utils.tma_get_copy_fn(tma_atom_V, 0, cute.make_layout(1), gV, sV)
            load_K = partial(
                self.load_KV, copy_utils.tma_producer_copy_fn(tma_load_K, pipeline_k), pipeline_k
            )
            load_V = partial(
                self.load_KV, copy_utils.tma_producer_copy_fn(tma_load_V, pipeline_v), pipeline_v
            )

            # EXERCISE (2) RESTORED: causal n-block skipping.
            # Reuse the shared BlockInfo helper to narrow the KV block range so
            # that causal attention skips the whole blocks that sit entirely
            # above the diagonal (which the teaching version used to load,
            # multiply, and then mask to -inf in their entirety).  For
            # non-causal, get_n_block_min_max returns the full range, preserving
            # the original behavior.  pack_gqa/local are asserted off in __init__,
            # so qhead_per_kvhead_packgqa=1 and no split-KV.
            block_info = BlockInfo(
                self.tile_m,
                self.tile_n,
                self.is_causal,
                self.is_local,
                False,  # is_split_kv
                None,  # window_size_left
                None,  # window_size_right
                qhead_per_kvhead_packgqa=(
                    self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1
                ),
            )
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)

            if warp_idx == 0:
                pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                load_Q(tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0))
            q_producer_phase ^= 1
            pipeline_q.consumer_wait_w_index_phase(0, q_consumer_phase)

            # EXERCISE (5): a flat loop of self-contained iterations. Each one loads K[n]
            # and V[n], runs the QK gemm to completion, does the softmax, then runs the PV
            # gemm to completion. Try to overlap the GEMMs and inspect its performance impacts.
            kv_producer_state, kv_consumer_state = self.one_n_block(
                n_block_max - 1,
                warp_idx,
                load_K,
                load_V,
                pipeline_k,
                pipeline_v,
                kv_producer_state,
                kv_consumer_state,
                mma_qk_fn,
                mma_pv_fn,
                acc_O,
                tOrP,
                softmax,
                mask_fn,
                is_first=True,
            )
            # EXERCISE (2): descend only to n_block_min instead of 0, so causal
            # attention skips the fully-above-diagonal blocks entirely.  The
            # first block (n_block_max - 1) is handled above; this loop covers
            # [n_block_min, n_block_max - 2] in descending order.
            for n_tile in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                kv_producer_state, kv_consumer_state = self.one_n_block(
                    n_block_max - 2 - n_tile,
                    warp_idx,
                    load_K,
                    load_V,
                    pipeline_k,
                    pipeline_v,
                    kv_producer_state,
                    kv_consumer_state,
                    mma_qk_fn,
                    mma_pv_fn,
                    acc_O,
                    tOrP,
                    softmax,
                    mask_fn,
                    is_first=False,
                )

            # Release Q so the producer can load the next tile's Q
            pipeline_q.consumer_release_w_index(0)
            q_consumer_phase ^= 1

            # Normalize acc_O by row_sum and compute the lse
            softmax.rescale_O(acc_O, softmax.finalize())
            self.epilogue(
                acc_O,
                softmax.row_sum,
                mO,
                mLSE,
                sO,
                seqlen,
                gmem_tiled_copy_O,
                tma_atom_O,
                tiled_mma_pv,
                tidx,
                m_block,
                head_idx,
                batch_idx,
            )

            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

        # Redundant here (every TMA is consumer-waited inside the loop) but kept so the
        # producer-side accounting still balances if the loop structure is changed.
        if warp_idx == 0:
            pipeline_v.producer_tail(kv_producer_state)

    @cute.jit
    def one_n_block(
        self,
        n_block: Int32,
        warp_idx: Int32,
        load_K: Callable,
        load_V: Callable,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        kv_producer_state,
        kv_consumer_state,
        mma_qk_fn: Callable,
        mma_pv_fn: Callable,
        acc_O: cute.Tensor,
        tOrP: cute.Tensor,
        softmax: Softmax,
        mask_fn: Callable,
        is_first: cutlass.Constexpr = False,
    ):
        """Load K[n] and V[n], then S = Q@K.T, softmax, O += P@V -- all fully serialized."""
        # ---- load -------------------------------------------------------------------
        # EXERCISE (3)/(5): the loads live inside the compute loop and are issued for this
        # block only. The original skewed the streams (K of block n-1 issued alongside V of
        # block n) so that the consumer's deferred PV always had its V resident.
        if warp_idx == 0:
            load_K(n_block, kv_producer_state)
            load_V(n_block, kv_producer_state)
        kv_producer_state.advance()

        # ---- S = Q @ K.T ------------------------------------------------------------
        # EXERCISE (3)/(5): warp_scheduler_barrier_sync() used to guard this region.
        pipeline_k.consumer_wait(kv_consumer_state, pipeline_k.consumer_try_wait(kv_consumer_state))
        # wg_wait=0: block on the QK gemm immediately instead of leaving it in flight.
        acc_S = mma_qk_fn(B_idx=kv_consumer_state.index, wg_wait=0)
        pipeline_k.consumer_release(kv_consumer_state)

        # ---- softmax ----------------------------------------------------------------
        mask_fn(acc_S=acc_S, n_block=n_block)
        if const_expr(is_first):
            # row_scale unused: the PV gemm below writes acc_O instead of accumulating.
            softmax.online_softmax(acc_S, is_first=True)
            utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_S), tOrP)
        else:
            row_scale = softmax.online_softmax(acc_S, check_inf=True)
            utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_S), tOrP)
            # Must happen before the PV gemm accumulates into acc_O.
            softmax.rescale_O(acc_O, row_scale)

        # ---- O += P @ V -------------------------------------------------------------
        # EXERCISE (3)/(5): warp_scheduler_barrier_arrive() used to release the token here.
        pipeline_v.consumer_wait(kv_consumer_state, pipeline_v.consumer_try_wait(kv_consumer_state))
        mma_pv_fn(B_idx=kv_consumer_state.index, zero_init=is_first, wg_wait=0)
        pipeline_v.consumer_release(kv_consumer_state)
        kv_consumer_state.advance()

        return kv_producer_state, kv_consumer_state