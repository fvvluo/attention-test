import os, sys
_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "flash_attention_baseline")
print(_dir)
sys.path.insert(0, _dir)

# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
#
# SM90 (Hopper) forward pass for flash attention -- RE-OPTIMIZED VERSION.
#
# 基于助教的去优化教学版 flash_fwd_sm90.py, 把被移除的 6 个优化全部恢复:
#   EXERCISE (1)  LPT tile scheduling      -> SingleTileLPTScheduler + lpt=is_causal (causal 负载均衡)
#   EXERCISE (2)  Causal n-block skipping  -> BlockInfo.get_n_block_min_max (即 SOTA.py 的对角块裁剪优化)
#   EXERCISE (3)  Warp specialization      -> 独立 producer(TMA) warpgroup + 2 个 consumer(MMA) warpgroup
#   EXERCISE (4)  Register redistribution  -> setmaxregister_decrease/increase (producer 少/consumer 多)
#   EXERCISE (5)  Intra-warpgroup overlap  -> QK(n) 与 PV(n-1) 跨迭代软件流水 (first/last half + skew load)
#   EXERCISE (6)  Inter-warpgroup ping-pong-> WarpSchedulerWG1/WG2 named barrier + mma_init
#
# 仅保留本场景所需路径: TMA / mma_pv_is_rs / 非 varlen / 非 paged / 非 pack_gqa / 非 local / 无 score_mod。
# 对外 __init__ / __call__ 签名与 baseline 一致, 可直接替换。decode 不在本文件 (未改动)。

from typing import Callable, Optional
from types import SimpleNamespace
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
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.named_barrier import NamedBarrierFwd
from flash_attn.cute.block_sparsity import BlockSparseTensors  # signature only
from flash_attn.cute.utils import AuxData  # signature only
from flash_attn.cute import pipeline as pipeline_custom
from flash_attn.cute.tile_scheduler import (
    TileSchedulerArguments,
    SingleTileScheduler,
    SingleTileLPTScheduler,
)

from flash_attn.cute.flash_fwd import FlashAttentionForwardBase
from flash_attn.cute.cute_dsl_utils import to_cute_tensor

# ---- attention 接口 (类似 SOTA.py) 依赖 ----
import math
import torch
import triton
import triton.language as tl


class FlashAttentionForwardSm90(FlashAttentionForwardBase):
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
            "this SM90 kernel only supports mma_pv_is_rs=True, paged_kv_non_tma=False"
        )
        # EXERCISE (5) RESTORED: intra-warpgroup overlap enabled.
        self.intra_wg_overlap, self.mma_pv_is_rs = intra_wg_overlap, mma_pv_is_rs
        self.use_tma_Q = self.use_tma_O = self.use_tma_KV = True
        self.buffer_align_bytes = 1024
        self.cluster_shape_mn = (1, 1)
        assert self.arch.is_family_of(Arch.sm_90a), "Only SM 9.x is supported"
        assert not self.pack_gqa and not self.Q_in_regs
        assert not self.is_local, "sliding window / local attention was stripped out"
        assert self.score_mod is None and self.mask_mod is None
        assert self.tile_hdim == 128 and self.tile_hdimv == 128

    def _get_smem_layout_atom(self):
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
        mPageTable: Optional[cute.Tensor] = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        learnable_sink: Optional[cute.Tensor] = None,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        aux_data: AuxData = AuxData(),
        stream: cuda.CUstream = None,
    ):
        assert all(
            t is None
            for t in (
                mCuSeqlensQ, mCuSeqlensK, mSeqUsedQ, mSeqUsedK, mPageTable,
                window_size_left, window_size_right, learnable_sink,
                blocksparse_tensors, aux_data.tensors,
            )
        ), "varlen, paged KV, local windows, sinks, block sparsity and score/mask mods were stripped"

        self._check_type(
            mQ.element_type, mK.element_type, mV.element_type, mO.element_type,
            mLSE.element_type if const_expr(mLSE is not None) else None, None, None, None, None,
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

        # EXERCISE (3) RESTORED: dedicated producer warpgroup + num_wg_mma consumer warpgroups.
        self.num_threads = self.num_threads_per_warp_group * (self.num_wg_mma + 1)
        self.num_producer_threads = 32
        self.num_Q_load_threads = self.num_threads_per_warp_group
        self.num_epilogue_threads = self.num_mma_threads
        # EXERCISE (4) RESTORED: producer donates registers, consumer takes more.
        self.num_mma_regs, self.num_producer_regs = {1: (256, 56), 2: (240, 24), 3: (160, 32)}[
            self.num_wg_mma
        ]
        # EXERCISE (6) RESTORED: enable the warp-scheduler ping-pong barrier.
        self.use_scheduler_barrier = self.num_wg_mma >= 2 and self.tile_hdim <= 128
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

        self.tma_copy_bytes = {
            name: cute.size_in_bytes(self.dtype, cute.select(layout, mode=[0, 1]))
            for name, layout in [("Q", self.sQ_layout), ("K", self.sK_layout), ("V", self.sV_layout)]
        }
        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mQ, self.sQ_layout, (self.tile_m, self.tile_hdim)
        )
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mK,
            cute.select(self.sK_layout, mode=[0, 1]), (self.tile_n, self.tile_hdim), 1,
        )
        tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mV,
            cute.select(self.sV_layout, mode=[0, 1]), (self.tile_n, self.tile_hdimv), 1,
        )
        tma_atom_O, tma_tensor_O = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), mO, self.sO_layout, (self.tile_m, self.tile_hdimv)
        )

        # EXERCISE (1) RESTORED: LPT (longest-processing-time) scheduling balances the
        # triangular causal workload across CTAs; plain scheduler leaves SMs idle at the tail.
        TileScheduler = SingleTileLPTScheduler if const_expr(self.is_causal) else SingleTileScheduler
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
            lpt=self.is_causal,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(softmax_scale, None)

        self.kernel(
            tma_tensor_Q, tma_tensor_K, tma_tensor_V, tma_tensor_O, mLSE,
            tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_O,
            softmax_scale_log2, softmax_scale,
            self.sQ_layout, self.sK_layout, self.sV_layout, self.sO_layout,
            self.gmem_tiled_copy_O, tiled_mma_qk, tiled_mma_pv,
            tile_sched_params, TileScheduler, SharedStorage,
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
            barrier_storage=storage.mbar_ptr_Q.data_ptr(), num_stages=1,
            tx_count=self.tma_copy_bytes["Q"],
        )
        pipeline_k = make_pipe(
            barrier_storage=storage.mbar_ptr_K.data_ptr(), num_stages=self.num_stages,
            tx_count=self.tma_copy_bytes["K"],
        )
        pipeline_v = make_pipe(
            barrier_storage=storage.mbar_ptr_V.data_ptr(), num_stages=self.num_stages,
            tx_count=self.tma_copy_bytes["V"],
        )
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sVt = layout_utils.transpose_view(sV)
        sO = storage.sQ.get_tensor(sO_layout.outer, swizzle=sO_layout.inner, dtype=self.dtype)

        block_info = BlockInfo(
            self.tile_m, self.tile_n, self.is_causal, False, False, None, None,
            qhead_per_kvhead_packgqa=1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create, seqlen_q_static=mQ.shape[0], seqlen_k_static=mK.shape[0]
        )
        AttentionMaskCls = partial(AttentionMask, self.tile_m, self.tile_n)
        TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # EXERCISE (3) RESTORED: warp specialization -- warpgroup 0 loads, 1/2 compute.
        if warp_idx < 4:  # Producer warpgroup
            cute.arch.setmaxregister_decrease(self.num_producer_regs)  # EXERCISE (4)
            self.load(
                mQ, mK, mV, sQ, sK, sV, tma_atom_Q, tma_atom_K, tma_atom_V,
                pipeline_k, pipeline_v, pipeline_q, block_info, SeqlenInfoCls, TileSchedulerCls,
            )
        else:  # Consumer warpgroups
            cute.arch.setmaxregister_increase(self.num_mma_regs)  # EXERCISE (4)
            tidx, _, _ = cute.arch.thread_idx()
            tidx = tidx - 128
            self.mma(
                tiled_mma_qk, tiled_mma_pv, mO, mLSE, sQ, sK, sVt, sO,
                pipeline_k, pipeline_v, pipeline_q, gmem_tiled_copy_O, tma_atom_O, tidx,
                softmax_scale_log2, softmax_scale, block_info,
                SeqlenInfoCls, AttentionMaskCls, TileSchedulerCls,
            )

    @cute.jit
    def load_KV(
        self,
        tma_load_fn: Callable,
        pipeline_kv: pipeline.PipelineAsync,
        block: Int32,
        producer_state: pipeline.PipelineState,
    ):
        # producer_acquire is done by the caller (so K[n] and V[n-1] can use different states).
        tma_load_fn(src_idx=block, producer_state=producer_state)
        pipeline_kv.producer_commit(producer_state)

    @cute.jit
    def load(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        pipeline_q: pipeline.PipelineAsync,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        q_producer_phase = Int32(1)
        kv_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_stages
        )
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            head_idx_kv = head_idx // self.qhead_per_kvhead  # GQA

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

            # EXERCISE (2) RESTORED: causal skipping -- only visit KV blocks that touch the
            # diagonal or below; blocks fully above are never loaded/multiplied/masked.
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)

            if warp_idx_in_wg == 0:
                n_block = n_block_max - 1
                pipeline_k.producer_acquire(kv_producer_state)
                load_K(block=n_block, producer_state=kv_producer_state)
                pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                load_Q(tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0))
                q_producer_phase ^= 1
                # EXERCISE (5) RESTORED: skew the streams -- issue K[n-1] alongside V[n] so the
                # consumer's deferred PV(n) always finds its V resident (software pipeline).
                for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                    n_block_prev = n_block_max - i - 1
                    n_block = n_block_prev - 1
                    kv_producer_state_prev = kv_producer_state.clone()
                    kv_producer_state.advance()
                    pipeline_k.producer_acquire(kv_producer_state)
                    load_K(block=n_block, producer_state=kv_producer_state)
                    pipeline_v.producer_acquire(kv_producer_state_prev)
                    load_V(block=n_block_prev, producer_state=kv_producer_state_prev)
                n_block = n_block_min
                pipeline_v.producer_acquire(kv_producer_state)
                load_V(block=n_block, producer_state=kv_producer_state)
                kv_producer_state.advance()

            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

        if warp_idx_in_wg == 0:
            pipeline_v.producer_tail(kv_producer_state)

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sVt: cute.Tensor,
        sO: cute.Tensor,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        pipeline_q: pipeline.PipelineAsync,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: cute.CopyAtom,
        tidx: Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
    ):
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
        acc_O, tOrP, tOrVt = sm90_utils.partition_fragment_ABC(
            wg_mma_pv, (self.tile_m, self.tile_hdimv, self.tile_n), None, sVt
        )
        mma_pv_fn = partial(sm90_utils.gemm_w_idx, tiled_mma_pv, acc_O, tOrP, tOrVt)
        smem_copy_params = SimpleNamespace(smem_thr_copy_P=None, tPsP=None)

        self.mma_init()  # EXERCISE (6) RESTORED: prime the ping-pong token ring.

        q_consumer_phase = Int32(0)
        kv_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_stages
        )
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        softmax = Softmax.create(
            softmax_scale_log2,
            num_rows=acc_O.shape[0][0] * acc_O.shape[1],
            softmax_scale=softmax_scale,
        )

        # EXERCISE (5): middle blocks overlap QK(n) with PV(n-1); the first/last "half" blocks
        # bootstrap and drain the software pipeline.
        mma_one_n_block_all = partial(
            self.mma_one_n_block_intrawg_overlap,
            mma_qk_fn=mma_qk_fn, pipeline_k=pipeline_k, pipeline_v=pipeline_v,
            acc_O=acc_O, tOrP=tOrP, smem_copy_params=smem_copy_params, check_inf=True,
        )
        process_first_half_block = partial(
            self.first_half_block_overlap, mma_qk_fn=mma_qk_fn, pipeline_k=pipeline_k,
            tOrP=tOrP, smem_copy_params=smem_copy_params, softmax=softmax, acc_O=acc_O,
        )
        process_last_half_block = partial(
            self.last_half_block_overlap, pipeline_v=pipeline_v, mma_pv_fn=mma_pv_fn,
            softmax=softmax, acc_O=acc_O,
        )
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            mask = AttentionMaskCls(seqlen)
            mask_fn = partial(
                mask.apply_mask, batch_idx=batch_idx, head_idx=head_idx, m_block=m_block,
                thr_mma=thr_mma_qk, mask_causal=self.is_causal, mask_local=False, mask_mod=None,
            )
            mma_one_n_block = partial(mma_one_n_block_all, seqlen=seqlen, softmax=softmax)
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)  # EXERCISE (2)
            pipeline_q.consumer_wait_w_index_phase(0, q_consumer_phase)
            O_should_accumulate = False

            # First (diagonal) block: QK + softmax only; PV deferred to next iteration.
            kv_consumer_state = process_first_half_block(
                n_block=n_block_max - 1, seqlen=seqlen, kv_consumer_state=kv_consumer_state,
                mask_fn=mask_fn, is_first_block=True,
            )
            n_block_max -= 1
            # Causal-masked blocks near the diagonal.
            if const_expr(self.is_causal):
                n_block_causal = block_info.get_n_block_min_causal_local_mask(
                    seqlen, m_block, n_block_min
                )
                for n_tile in cutlass.range(n_block_max - n_block_causal, unroll=1):
                    kv_consumer_state = mma_one_n_block(
                        kv_consumer_state, n_block=n_block_max - 1 - n_tile,
                        mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                        mask_fn=partial(mask_fn, mask_seqlen=False),
                    )
                    O_should_accumulate = True
                n_block_max = cutlass.min(n_block_max, n_block_causal)
            # Fully-visible blocks (no causal masking needed).
            n_block_before = block_info.get_n_block_min_before_local_mask(seqlen, m_block, n_block_min)
            for n_tile in cutlass.range(n_block_max - n_block_before, unroll=1):
                kv_consumer_state = mma_one_n_block(
                    kv_consumer_state, n_block=n_block_max - 1 - n_tile,
                    mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                    mask_fn=partial(mask_fn, mask_seqlen=False),
                )
                O_should_accumulate = True
            pipeline_q.consumer_release_w_index(0)
            # Drain the pipeline: final deferred PV.
            kv_consumer_state = process_last_half_block(
                kv_consumer_state=kv_consumer_state, zero_init=not O_should_accumulate
            )
            q_consumer_phase ^= 1

            softmax.rescale_O(acc_O, softmax.finalize())
            self.epilogue(
                acc_O, softmax.row_sum, mO, mLSE, sO, seqlen, gmem_tiled_copy_O, tma_atom_O,
                tiled_mma_pv, tidx, m_block, head_idx, batch_idx,
            )
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def first_half_block_overlap(
        self,
        n_block: Int32,
        mma_qk_fn: Callable,
        kv_consumer_state,
        pipeline_k,
        tOrP: cute.Tensor,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        seqlen: SeqlenInfoQK,
        acc_O: cute.Tensor = None,
        mask_fn: Callable = None,
        is_first_block: bool = False,
    ):
        """QK + softmax of the first block; PV is deferred (intra-wg overlap)."""
        pipeline_k.consumer_wait(kv_consumer_state, pipeline_k.consumer_try_wait(kv_consumer_state))
        acc_S = mma_qk_fn(B_idx=kv_consumer_state.index, wg_wait=0)
        pipeline_k.consumer_release(kv_consumer_state)
        mask_fn(acc_S, n_block=n_block, mask_seqlen=True)
        softmax.online_softmax(acc_S, is_first=is_first_block)
        utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_S), tOrP)
        return kv_consumer_state

    @cute.jit
    def last_half_block_overlap(
        self,
        kv_consumer_state,
        pipeline_v,
        mma_pv_fn: Callable,
        zero_init: bool,
        softmax: Optional[Softmax] = None,
        acc_O: Optional[cute.Tensor] = None,
    ):
        """Final PV GEMM draining the intra-wg-overlap pipeline."""
        pipeline_v.consumer_wait(kv_consumer_state, pipeline_v.consumer_try_wait(kv_consumer_state))
        mma_pv_fn(B_idx=kv_consumer_state.index, zero_init=zero_init, wg_wait=0)
        pipeline_v.consumer_release(kv_consumer_state)
        kv_consumer_state.advance()
        return kv_consumer_state

    @cute.jit
    def mma_one_n_block_intrawg_overlap(
        self,
        smem_pipe_read,
        n_block: Int32,
        mma_qk_fn: Callable,
        mma_pv_fn: Callable,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        acc_O: cute.Tensor,
        tOrP: cute.Tensor,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        seqlen: SeqlenInfoQK,
        mask_fn: Optional[Callable] = None,
        check_inf: cutlass.Constexpr = True,
    ):
        """QK(n) issued while PV(n-1) is still in flight -> intra-warpgroup overlap (EX5),
        guarded by the WG1/WG2 scheduler barriers -> inter-warpgroup ping-pong (EX6)."""
        smem_pipe_read_v = smem_pipe_read.clone()
        smem_pipe_read.advance()
        pipeline_k.consumer_wait(smem_pipe_read, pipeline_k.consumer_try_wait(smem_pipe_read))
        self.warp_scheduler_barrier_sync()  # EXERCISE (6)
        acc_S = mma_qk_fn(B_idx=smem_pipe_read.index, wg_wait=-1)
        pipeline_v.consumer_wait(smem_pipe_read_v, pipeline_v.consumer_try_wait(smem_pipe_read_v))
        mma_pv_fn(B_idx=smem_pipe_read_v.index, wg_wait=-1)
        self.warp_scheduler_barrier_arrive()  # EXERCISE (6)
        warpgroup.wait_group(1)
        pipeline_k.consumer_release(smem_pipe_read)
        if const_expr(mask_fn is not None):
            mask_fn(acc_S=acc_S, n_block=n_block)
        row_scale = softmax.online_softmax(acc_S, check_inf=check_inf)
        warpgroup.wait_group(0)
        pipeline_v.consumer_release(smem_pipe_read_v)
        utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_S), tOrP)
        softmax.rescale_O(acc_O, row_scale)
        return smem_pipe_read

    @cute.jit
    def mma_init(self):
        # EXERCISE (6) RESTORED: warpgroup 1 arrives first so WG2 can start, forming the token ring.
        warp_group_idx = utils.canonical_warp_group_idx(sync=False)
        if const_expr(self.use_scheduler_barrier):
            if warp_group_idx == 1:
                cute.arch.barrier_arrive(
                    barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1),
                    number_of_threads=2 * self.num_threads_per_warp_group,
                )

    def warp_scheduler_barrier_sync(self):
        if const_expr(self.use_scheduler_barrier):
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1)
                - 1
                + utils.canonical_warp_group_idx(sync=False),
                number_of_threads=2 * self.num_threads_per_warp_group,
            )

    def warp_scheduler_barrier_arrive(self):
        if const_expr(self.use_scheduler_barrier):
            assert self.num_wg_mma in [2, 3]
            cur_wg = utils.canonical_warp_group_idx(sync=False) - 1
            next_wg = 1 - cur_wg if const_expr(self.num_wg_mma == 2) else (cur_wg + 1) % self.num_wg_mma
            cute.arch.barrier_arrive(
                barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1) + next_wg,
                number_of_threads=2 * self.num_threads_per_warp_group,
            )


# ============================================================
# attention 接口 (类似 SOTA.py): prefill -> SM90 CuTe DSL, decode -> 原 Triton (未改)
#   输入布局与 SOTA.py 一致: q/k/v = (batch, heads, seq, head_dim), 输出同 q。
# ============================================================
_Q_HEADS = 64
_KV_HEADS = 8
_N_CTX = 131072
_HEAD_DIM = 128
_GROUP_SIZE = _Q_HEADS // _KV_HEADS
_QK_SCALE = (1.0 / math.sqrt(_HEAD_DIM)) * math.log2(math.e)   # decode 用: 已含 log2(e)
_BLOCK_N_DECODE = 64
_N_SPLITS = 128
_SPLIT_LEN = _N_CTX // _N_SPLITS
_PAD_M = 16
_DT = {torch.float16: cutlass.Float16, torch.bfloat16: cutlass.BFloat16}
_COMPILED = {}


def _prefill(q, k, v, out):
    # SOTA 布局 (b, h, s, d) -> kernel 期望 (b, s, h, d)。permute 是零拷贝视图 (最内维仍连续)。
    # 若 to_cute_tensor 对 strided 视图报错, 可在 permute 后加 .contiguous() (out 需另 copy 回)。
    b, h, s, d = q.shape
    hkv = k.shape[1]
    mQ, mK, mV, mO = (to_cute_tensor(x.permute(0, 2, 1, 3)) for x in (q, k, v, out))
    scale = 1.0 / math.sqrt(d)
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    key = (q.device, q.dtype, h, hkv, s, d)
    args = (mQ, mK, mV, mO, None, scale,
            None, None, None, None, None, None, None, None, None, AuxData(), stream)
    fn = _COMPILED.get(key)
    if fn is None:
        op = FlashAttentionForwardSm90(
            _DT[q.dtype], d, d, h // hkv,
            is_causal=True, is_local=False, pack_gqa=False,
            tile_m=128, tile_n=128, num_stages=2, num_threads=384,
            Q_in_regs=False, intra_wg_overlap=True, mma_pv_is_rs=True,
            mask_mod=None, score_mod=None, has_aux_tensors=False,
            q_subtile_factor=1, paged_kv_non_tma=False,
        )
        fn = cute.compile(op, *args)
        _COMPILED[key] = fn
    fn(*args)


# ---- Decode: 原 SOTA.py 的 Triton Split-KV 两阶段 (未翻译) ----
@triton.jit
def _attn_decode_split_kernel(Q, K, V, M_buf, L_buf, Acc_buf, QK_SCALE: tl.constexpr,
    N_CTX: tl.constexpr, HEAD_DIM: tl.constexpr, GROUP_SIZE: tl.constexpr, BLOCK_N: tl.constexpr,
    SPLIT_LEN: tl.constexpr, N_SPLITS: tl.constexpr, PAD_M: tl.constexpr):
    kv_head = tl.program_id(0)
    split_id = tl.program_id(1)
    q_offset = kv_head * GROUP_SIZE * HEAD_DIM
    kv_offset = kv_head * N_CTX * HEAD_DIM
    offs_m = tl.arange(0, PAD_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    m_valid = offs_m < GROUP_SIZE
    q_rows = tl.where(m_valid, offs_m, 0)
    q = tl.load(Q + q_offset + q_rows[:, None] * HEAD_DIM + offs_k[None, :])
    kbase = kv_offset + split_id * SPLIT_LEN * HEAD_DIM
    k_ptrs = K + kbase + offs_k[:, None] + offs_n[None, :] * HEAD_DIM
    v_ptrs = V + kbase + offs_n[:, None] * HEAD_DIM + offs_k[None, :]
    m_i = tl.full([PAD_M], -float("inf"), tl.float32)
    l_i = tl.zeros([PAD_M], tl.float32)
    acc = tl.zeros([PAD_M, HEAD_DIM], tl.float32)
    for start_n in range(0, SPLIT_LEN, BLOCK_N):
        k = tl.load(k_ptrs + start_n * HEAD_DIM)
        qk = tl.dot(q, k) * QK_SCALE
        qk = tl.where(m_valid[:, None], qk, -float("inf"))
        m_next = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_next[:, None])
        alpha = tl.math.exp2(m_i - m_next)
        acc = acc * alpha[:, None]
        v = tl.load(v_ptrs + start_n * HEAD_DIM)
        acc = tl.dot(p.to(q.dtype), v, acc)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_next
    q_head = kv_head * GROUP_SIZE + offs_m
    ml_idx = q_head * N_SPLITS + split_id
    tl.store(M_buf + ml_idx, m_i, mask=m_valid)
    tl.store(L_buf + ml_idx, l_i, mask=m_valid)
    acc_idx = (q_head * N_SPLITS + split_id)[:, None] * HEAD_DIM + offs_k[None, :]
    tl.store(Acc_buf + acc_idx, acc, mask=m_valid[:, None])


@triton.jit
def _attn_decode_reduce_kernel(M_buf, L_buf, Acc_buf, Out, HEAD_DIM: tl.constexpr, N_SPLITS: tl.constexpr):
    q_head = tl.program_id(0)
    offs_s = tl.arange(0, N_SPLITS)
    offs_k = tl.arange(0, HEAD_DIM)
    m_s = tl.load(M_buf + q_head * N_SPLITS + offs_s)
    l_s = tl.load(L_buf + q_head * N_SPLITS + offs_s)
    m = tl.max(m_s, 0)
    scale = tl.math.exp2(m_s - m)
    denom = tl.sum(l_s * scale, 0)
    acc = tl.load(Acc_buf + q_head * N_SPLITS * HEAD_DIM + offs_s[:, None] * HEAD_DIM + offs_k[None, :])
    out = tl.sum(acc * scale[:, None], 0) / denom
    tl.store(Out + q_head * HEAD_DIM + offs_k, out.to(Out.dtype.element_ty))


def _decode(q, k, v, out):
    m_buf = torch.empty((_Q_HEADS, _N_SPLITS), dtype=torch.float32, device=q.device)
    l_buf = torch.empty((_Q_HEADS, _N_SPLITS), dtype=torch.float32, device=q.device)
    acc_buf = torch.empty((_Q_HEADS, _N_SPLITS, _HEAD_DIM), dtype=torch.float32, device=q.device)
    _attn_decode_split_kernel[(_KV_HEADS, _N_SPLITS)](q, k, v, m_buf, l_buf, acc_buf,
        QK_SCALE=_QK_SCALE, N_CTX=_N_CTX, HEAD_DIM=_HEAD_DIM, GROUP_SIZE=_GROUP_SIZE,
        BLOCK_N=_BLOCK_N_DECODE, SPLIT_LEN=_SPLIT_LEN, N_SPLITS=_N_SPLITS, PAD_M=_PAD_M)
    _attn_decode_reduce_kernel[(_Q_HEADS,)](m_buf, l_buf, acc_buf, out,
        HEAD_DIM=_HEAD_DIM, N_SPLITS=_N_SPLITS)


def attention(q, k, v, causal=True, sm_scale=None):
    out = torch.empty_like(q)
    if q.shape[2] != 1:
        _prefill(q, k, v, out)          # prefill: SM90 CuTe DSL (WGMMA + TMA + warp-spec)
    else:
        _decode(q, k, v, out)           # decode: 原 Triton
    return out
