# lhx 的 SM90 (Hopper) FlashAttention 前向算子 —— CuTe DSL 版
#
# 内核核心复制自 flash-attention-baseline/flash_attn/cute/flash_fwd_sm90.py
# （去优化教学版，带 6 个 EXERCISE 优化练习），其所需的全部 flash_attn.cute
# 依赖模块也已复制到 ops/lhx_cute/ 包中，与 baseline checkout 完全解耦，
# 便于在不改动 baseline 的前提下做优化实验。
#
# 接口封装参考 ops/_template.py / ops/_example_flash_attention.py：
#   attention(q, k, v, causal=True, sm_scale=None) -> output
#   q: (batch, q_heads, q_len, head_dim)，k/v: (batch, kv_heads, kv_len, head_dim)
#   输出 shape 与 q 相同。GQA 通过内核的 qhead_per_kvhead 路径原生支持
#   （不需要 repeat_interleave 展开 kv）。
#
# 注意：
#   - 仅支持 SM90（H 系列卡）、head_dim=128、fp16/bf16（内核自带 assert）。
#   - 首次调用某一 (dtype, head_dim, causal, qhead_per_kvhead) 组合时会触发
#     CuTe DSL JIT 编译（较慢），之后命中 _COMPILE_CACHE 直接发射内核；
#     batch / seqlen / heads 均为动态维度，同配置下换形状不会重新编译。
#
# ============================================================
# 以下为原 flash_fwd_sm90.py 的文件头注释（保留，含练习说明）
# ============================================================
# SM90 (Hopper) forward pass for flash attention -- DE-OPTIMIZED TEACHING VERSION.
#
# This file is derived from the original SM90 kernel with six optimizations deliberately
# removed. It is functionally correct (under 1x64x8x131072x128) but substantially slower.
# Each removal is marked with an `EXERCISE (n)` comment at the site where the technique used to live
# (typically, more code is required to modify; they might be far away from the `EXERCISE (n)` comment).
#
#   [DONE] EXERCISE (1)  LPT tile scheduling            -> SingleTileLPTScheduler
#   [DONE] EXERCISE (2)  Causal n-block skipping        -> skip invisible blocks; mask boundaries only
#   [DONE] EXERCISE (3)  Warp specialization            -> 128-thread TMA WG + 256-thread MMA WGs
#   [DONE] EXERCISE (4)  Register redistribution        -> producer 24, MMA consumers 240
#   [DONE] EXERCISE (5)  Intra-warpgroup overlap        -> QK[current] overlaps PV[previous]
#   [DONE] EXERCISE (6)  Inter-warpgroup ping-pong      -> named-barrier token ring
#
# If you are stuck, refer to https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/flash_fwd_sm90.py.

import math
from typing import Callable, Optional
from functools import partial

import torch

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

from .lhx_cute.cute_dsl_utils import assume_tensor_aligned, to_cute_tensor
from .lhx_cute import utils
from .lhx_cute.mask import AttentionMask
from .lhx_cute.named_barrier import NamedBarrierFwd
from .lhx_cute.softmax import Softmax
from .lhx_cute.seqlen_info import SeqlenInfoQK
from .lhx_cute.block_info import BlockInfo
from .lhx_cute.block_sparsity import BlockSparseTensors  # signature only
from .lhx_cute.flash_fwd_combine import FlashAttentionForwardCombine
from .lhx_cute.utils import AuxData
from .lhx_cute import pipeline as pipeline_custom
from .lhx_cute.tile_scheduler import (
    TileSchedulerArguments,
    SingleTileLPTScheduler,
)

from .lhx_cute.flash_fwd import FlashAttentionForwardBase

from .base import register


class FlashAttentionForwardSm90(FlashAttentionForwardBase):
    def __init__(
        self,
        *args,
        intra_wg_overlap: bool = True,
        mma_pv_is_rs: bool = True,
        paged_kv_non_tma: bool = False,
        is_split_kv: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.is_split_kv = is_split_kv
        assert mma_pv_is_rs and not paged_kv_non_tma, (
            "this de-optimized SM90 kernel only supports mma_pv_is_rs=True, "
            "paged_kv_non_tma=False"
        )
        assert intra_wg_overlap, "this optimized kernel requires intra_wg_overlap=True"
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
        # Q/K/V/O all have leading (contiguous) extent 128, so one atom covers them all.
        # sP is None: MMA-PV takes P straight from registers (mma_pv_is_rs).
        atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, self.dtype, self.tile_hdim),
            self.dtype,
        )
        # The split specialization replaces sO with an FP32 layout after the base
        # copy attributes are initialized; this placeholder keeps base setup unchanged.
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
        # Split-KV writes FP32 partial O through the buffer that aliases sQ. Give
        # that specialization a 64-KiB backing range; the non-split specialization
        # keeps the original 32-KiB range and therefore its current occupancy.
        q_storage_dtype = Float32 if self.is_split_kv else self.dtype
        q_storage_layout = self.sO_layout if self.is_split_kv else self.sQ_layout
        sQ_struct = cute.struct.Align[
            cute.struct.MemRange[q_storage_dtype, cute.cosize(q_storage_layout)],
            self.buffer_align_bytes,
        ]
        sK_struct, sV_struct = [
            cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(layout)], self.buffer_align_bytes
            ]
            for layout in (self.sK_layout, self.sV_layout)
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

        if const_expr(self.is_split_kv):
            if const_expr(not (mQ.element_type == mK.element_type == mV.element_type == self.dtype)):
                raise TypeError("Q, K and V must match the configured fp16/bf16 dtype")
            if const_expr(mO.element_type != Float32 or mLSE is None or mLSE.element_type != Float32):
                raise TypeError("Split-KV partial O and LSE must be Float32")
        else:
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
        # Q/K/V: (b, s, h, d) -> (s, d, h, b).
        mQ, mK, mV = [layout_utils.select(t, [1, 3, 2, 0]) for t in (mQ, mK, mV)]
        if const_expr(self.is_split_kv):
            # O partial: (split, b, s, h, d) -> (s, d, h, b, split)
            # LSE partial: (split, b, h, s) -> (s, h, b, split)
            mO = layout_utils.select(mO, [2, 4, 3, 1, 0])
            mLSE = layout_utils.select(mLSE, [3, 2, 1, 0])
            num_splits = mO.shape[4]
        else:
            mO = layout_utils.select(mO, [1, 3, 2, 0])
            if const_expr(mLSE is not None):
                mLSE = layout_utils.select(mLSE, [2, 1, 0])
            num_splits = Int32(1)

        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        self.num_threads_per_warp_group = 128
        self.num_mma_threads = tiled_mma_qk.size
        self.num_wg_mma = self.num_mma_threads // self.num_threads_per_warp_group
        assert self.num_wg_mma == 2

        # Warp specialization: one 128-thread producer warpgroup followed by the two
        # 128-thread WGMMA consumer warpgroups. Only warp 0 issues TMA instructions, but
        # keeping a complete producer warpgroup preserves WGMMA warpgroup alignment.
        self.num_producer_threads = self.num_threads_per_warp_group
        self.num_threads = self.num_producer_threads + self.num_mma_threads
        self.num_Q_load_threads = self.num_producer_threads
        self.num_epilogue_threads = self.num_mma_threads

        # Exercise 4: the 384-thread CTA starts with 168 registers/thread. Donate the
        # producer WG's registers to the two accumulator-heavy consumer WGs:
        # 24 + 2 * 240 = 504 = 3 * 168 warpgroup register units.
        self.num_producer_regs = 24
        self.num_mma_regs = 240
        assert self.num_producer_regs + self.num_wg_mma * self.num_mma_regs <= 504
        self._setup_attributes()

        self.sQ_layout, self.sK_layout, self.sV_layout = [
            sm90_utils.make_smem_layout(self.dtype, LayoutEnum.ROW_MAJOR, shape, stage)
            for shape, stage in [
                ((self.tile_m, self.tile_hdim), None),
                ((self.tile_n, self.tile_hdim), self.num_stages),
                ((self.tile_n, self.tile_hdimv), self.num_stages),
            ]
        ]
        output_dtype = Float32 if self.is_split_kv else self.dtype
        self.sO_layout = sm90_utils.make_smem_layout(
            output_dtype,
            LayoutEnum.ROW_MAJOR,
            (self.tile_m, self.tile_hdimv),
            None,
        )
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

        # EXERCISE (1): LPT schedules the longest causal Q tiles first to reduce tail effects.
        TileScheduler = SingleTileLPTScheduler
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), self.tile_m),
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3]),
            num_splits,
            cute.size(mK.shape[0]),
            mQ.shape[1],
            mV.shape[1],
            total_q=cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=(self.tile_m, self.tile_n),
            qhead_per_kvhead_packgqa=1,
            element_size=self.dtype.width // 8,
            is_persistent=False,
            is_split_kv=self.is_split_kv,
            lpt=self.is_causal,
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
            num_splits,
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
        num_splits: Int32,
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

        # Warp 0 in the producer warpgroup elects one TMA issuer; the two consumer
        # warpgroups independently wait on and release the shared-memory stages.
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

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner, dtype=self.dtype)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        # Transpose view of V, i.e. (head_dim_v, tile_n), for the PV tiled mma
        sVt = layout_utils.transpose_view(sV)
        output_dtype = Float32 if self.is_split_kv else self.dtype
        sO = storage.sQ.get_tensor(sO_layout.outer, swizzle=sO_layout.inner, dtype=output_dtype)

        SeqlenInfoCls = partial(
            SeqlenInfoQK.create, seqlen_q_static=mQ.shape[0], seqlen_k_static=mK.shape[0]
        )
        AttentionMaskCls = partial(AttentionMask, self.tile_m, self.tile_n)
        BlockInfoCls = partial(
            BlockInfo,
            self.tile_m,
            self.tile_n,
            self.is_causal,
            False,
            self.is_split_kv,
        )
        TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # Producer WG0 and consumer WG1/WG2 follow independent instruction streams.
        # The tiled MMA sees a consumer-local thread index in [0, 256), while role
        # selection and the TMA-O epilogue continue to use the global warp index.
        tidx, _, _ = cute.arch.thread_idx()
        if warp_idx < self.num_producer_threads // cute.arch.WARP_SIZE:
            # All four producer warps must execute the warpgroup-level operation.
            cute.arch.warpgroup_reg_dealloc(self.num_producer_regs)
            if warp_idx == 0:
                self.load_mainloop(
                    mQ,
                    mK,
                    mV,
                    sQ,
                    sK,
                    sV,
                    tma_atom_Q,
                    tma_atom_K,
                    tma_atom_V,
                    pipeline_q,
                    pipeline_k,
                    pipeline_v,
                    num_splits,
                    SeqlenInfoCls,
                    BlockInfoCls,
                    TileSchedulerCls,
                )
        else:
            # Both complete WGMMA warpgroups raise their per-thread register ceiling.
            cute.arch.warpgroup_reg_alloc(self.num_mma_regs)
            consumer_tidx = tidx - self.num_producer_threads
            self.compute_mainloop(
                tiled_mma_qk,
                tiled_mma_pv,
                mO,
                mLSE,
                sQ,
                sK,
                sVt,
                sO,
                tma_atom_O,
                pipeline_q,
                pipeline_k,
                pipeline_v,
                gmem_tiled_copy_O,
                consumer_tidx,
                softmax_scale_log2,
                softmax_scale,
                num_splits,
                SeqlenInfoCls,
                AttentionMaskCls,
                BlockInfoCls,
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
    def load_mainloop(
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
        pipeline_q: pipeline.PipelineAsync,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        BlockInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        """Producer-only TMA loop for Q, K, and V."""
        q_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, 1
        )
        kv_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_stages
        )

        tile_scheduler = TileSchedulerCls()
        block_info = BlockInfoCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            head_idx_kv = head_idx // self.qhead_per_kvhead

            gQ = cute.local_tile(
                mQ[None, None, head_idx, batch_idx],
                (self.tile_m, self.tile_hdim),
                (m_block, 0),
            )
            load_Q, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q, 0, cute.make_layout(1), gQ, sQ, single_stage=True
            )
            gK = cute.local_tile(
                mK[None, None, head_idx_kv, batch_idx],
                (self.tile_n, self.tile_hdim),
                (None, 0),
            )
            gV = cute.local_tile(
                mV[None, None, head_idx_kv, batch_idx],
                (self.tile_n, self.tile_hdimv),
                (None, 0),
            )
            tma_load_K, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_K, 0, cute.make_layout(1), gK, sK
            )
            tma_load_V, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_V, 0, cute.make_layout(1), gV, sV
            )
            load_K = partial(
                self.load_KV,
                copy_utils.tma_producer_copy_fn(tma_load_K, pipeline_k),
                pipeline_k,
            )
            load_V = partial(
                self.load_KV,
                copy_utils.tma_producer_copy_fn(tma_load_V, pipeline_v),
                pipeline_v,
            )

            pipeline_q.producer_acquire(q_producer_state)
            load_Q(
                tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(
                    q_producer_state.index
                )
            )
            q_producer_state.advance()

            # Producer and consumers must advance their K/V pipeline states for
            # exactly the same split-local half-open block range.
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx, num_splits
            )
            for n_tile in cutlass.range(n_block_max - n_block_min, unroll=1):
                n_block = n_block_max - 1 - n_tile
                load_K(n_block, kv_producer_state)
                load_V(n_block, kv_producer_state)
                kv_producer_state.advance()

            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

        pipeline_q.producer_tail(q_producer_state)
        pipeline_k.producer_tail(kv_producer_state.clone())
        pipeline_v.producer_tail(kv_producer_state)

    @cute.jit
    def warp_scheduler_barrier_init(self, warp_group_idx: Int32):
        """Seed the consumer warpgroup token ring with local WG0."""
        if warp_group_idx == 0:
            cute.arch.barrier_arrive(
                barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1),
                number_of_threads=self.num_mma_threads,
            )

    @cute.jit
    def warp_scheduler_barrier_sync(self, warp_group_idx: Int32):
        """Wait until this consumer warpgroup owns the WGMMA issue token."""
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1) + warp_group_idx,
            number_of_threads=self.num_mma_threads,
        )

    @cute.jit
    def warp_scheduler_barrier_arrive(self, warp_group_idx: Int32):
        """Pass the WGMMA issue token to the other consumer warpgroup."""
        next_wg = 1 - warp_group_idx
        cute.arch.barrier_arrive(
            barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1) + next_wg,
            number_of_threads=self.num_mma_threads,
        )

    @cute.jit
    def compute_mainloop(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sVt: cute.Tensor,
        sO: cute.Tensor,
        tma_atom_O: cute.CopyAtom,
        pipeline_q: pipeline.PipelineAsync,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        gmem_tiled_copy_O: cute.TiledCopy,
        tidx: Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        BlockInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        """Consumer-only QK, softmax, PV, and epilogue loop."""
        # tidx is local to the two consumer warpgroups and runs from 0 to 255.
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

        # Exercise 6: local consumer WG0 seeds the token ring. Each WGMMA issue
        # phase passes the token to the other consumer WG before scalar/SFU work.
        self.warp_scheduler_barrier_init(warp_group_idx)

        q_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, 1
        )
        k_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_stages
        )
        v_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_stages
        )
        softmax = Softmax.create(
            softmax_scale_log2,
            num_rows=acc_O.shape[0][0] * acc_O.shape[1],
            softmax_scale=softmax_scale,
        )

        tile_scheduler = TileSchedulerCls()
        block_info = BlockInfoCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            mask = AttentionMaskCls(seqlen)
            # EXERCISE (2): mask_fn is only called for boundary blocks below.
            # Blocks strictly before the causal / seqlen boundary are fully valid.
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

            # Crop the descending KV loop to this CTA's split-local range. The
            # producer uses the same helper, which keeps all pipeline states aligned.
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx, num_splits
            )
            n_block_count = n_block_max - n_block_min

            # Blocks [0, n_block_mask_start) are fully valid and skip apply_mask.
            # A causal block is fully valid iff its last K index is visible to the
            # first Q row in this tile. For non-causal attention, only a partial
            # final K block needs seqlen masking.
            if const_expr(self.is_causal):
                first_q_idx = m_block * self.tile_m
                first_q_k_end = first_q_idx + seqlen.seqlen_k - seqlen.seqlen_q + 1
                n_block_mask_start = cutlass.max(first_q_k_end // self.tile_n, 0)
            else:
                n_block_mask_start = seqlen.seqlen_k // self.tile_n

            pipeline_q.consumer_wait(
                q_consumer_state, pipeline_q.consumer_try_wait(q_consumer_state)
            )

            # Exercise 5 prologue: produce P for the first block but defer its PV.
            k_consumer_state = self.compute_first_qk(
                n_block_max - 1,
                warp_group_idx,
                pipeline_k,
                k_consumer_state,
                mma_qk_fn,
                tOrP,
                softmax,
                mask_fn,
                apply_mask=True,
            )

            # Steady state: QK[current] is the older WGMMA group and PV[previous]
            # is the newer group. The first PV initializes acc_O; later PVs accumulate.
            if n_block_count > 1:
                n_block = n_block_max - 2
                if n_block >= n_block_mask_start:
                    k_consumer_state, v_consumer_state = self.compute_overlapped_block(
                        n_block,
                        warp_group_idx,
                        pipeline_k,
                        pipeline_v,
                        k_consumer_state,
                        v_consumer_state,
                        mma_qk_fn,
                        mma_pv_fn,
                        acc_O,
                        tOrP,
                        softmax,
                        mask_fn,
                        apply_mask=True,
                        is_first_pv=True,
                    )
                else:
                    k_consumer_state, v_consumer_state = self.compute_overlapped_block(
                        n_block,
                        warp_group_idx,
                        pipeline_k,
                        pipeline_v,
                        k_consumer_state,
                        v_consumer_state,
                        mma_qk_fn,
                        mma_pv_fn,
                        acc_O,
                        tOrP,
                        softmax,
                        mask_fn,
                        apply_mask=False,
                        is_first_pv=True,
                    )

            for n_tile in cutlass.range(cutlass.max(n_block_count - 2, 0), unroll=1):
                n_block = n_block_max - 3 - n_tile
                if n_block >= n_block_mask_start:
                    k_consumer_state, v_consumer_state = self.compute_overlapped_block(
                        n_block,
                        warp_group_idx,
                        pipeline_k,
                        pipeline_v,
                        k_consumer_state,
                        v_consumer_state,
                        mma_qk_fn,
                        mma_pv_fn,
                        acc_O,
                        tOrP,
                        softmax,
                        mask_fn,
                        apply_mask=True,
                        is_first_pv=False,
                    )
                else:
                    k_consumer_state, v_consumer_state = self.compute_overlapped_block(
                        n_block,
                        warp_group_idx,
                        pipeline_k,
                        pipeline_v,
                        k_consumer_state,
                        v_consumer_state,
                        mma_qk_fn,
                        mma_pv_fn,
                        acc_O,
                        tOrP,
                        softmax,
                        mask_fn,
                        apply_mask=False,
                        is_first_pv=False,
                    )

            # Epilogue: the last P fragment has no following QK iteration to carry it.
            if n_block_count == 1:
                v_consumer_state = self.flush_last_pv(
                    warp_group_idx,
                    pipeline_v,
                    v_consumer_state,
                    mma_pv_fn,
                    zero_init=True,
                )
            else:
                v_consumer_state = self.flush_last_pv(
                    warp_group_idx,
                    pipeline_v,
                    v_consumer_state,
                    mma_pv_fn,
                    zero_init=False,
                )

            # Normalize acc_O by row_sum and compute the natural-log LSE.
            softmax.rescale_O(acc_O, softmax.finalize())
            if const_expr(self.is_split_kv):
                self.epilogue_split_kv(
                    acc_O,
                    softmax.row_sum,
                    mO,
                    mLSE,
                    sO,
                    seqlen,
                    tma_atom_O,
                    tiled_mma_pv,
                    tidx,
                    m_block,
                    head_idx,
                    batch_idx,
                    split_idx,
                )
            else:
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

            # sO aliases sQ, so Q cannot be released until the epilogue has completed
            # its shared-memory staging and TMA store.
            pipeline_q.consumer_release(q_consumer_state)
            q_consumer_state.advance()

            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def epilogue_split_kv(
        self,
        acc_O: cute.Tensor,
        lse: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        sO: cute.Tensor,
        seqlen: SeqlenInfoQK,
        tma_atom_O: cute.CopyAtom,
        tiled_mma: cute.TiledMma,
        tidx: Int32,
        m_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
        split_idx: Int32,
    ):
        """Store normalized FP32 partial O/LSE for one KV split."""
        rO = cute.make_fragment_like(acc_O, Float32)
        rO.store(acc_O.load())

        # All consumer threads must finish reading V before sV can be reused, and
        # all must participate in the register-to-shared epilogue copy.
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.Epilogue),
            number_of_threads=self.num_epilogue_threads,
        )
        smem_copy_atom_O = utils.get_smem_store_atom(
            self.arch.major * 10 + self.arch.minor, Float32
        )
        smem_thr_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma).get_slice(tidx)
        taccOrO = smem_thr_copy_O.retile(rO)
        taccOsO = smem_thr_copy_O.partition_D(sO)
        cute.copy(smem_copy_atom_O, taccOrO, taccOsO)

        cO = cute.make_identity_tensor((self.tile_m, self.tile_hdimv))

        # mLSE has internal layout (q, head, batch, split).
        mLSE_cur = seqlen.offset_batch_Q(mLSE, batch_idx, dim=2)[None, head_idx, split_idx]
        gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (m_block,))
        gLSE_expanded_layout = cute.append(
            gLSE.layout, cute.make_layout((self.tile_hdimv,), stride=(0,))
        )
        gLSE_expanded = cute.make_tensor(gLSE.iterator, gLSE_expanded_layout)
        thr_mma = tiled_mma.get_slice(tidx)
        taccOgLSE = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(gLSE_expanded))
        taccOcO = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cO))
        t0accOcO = layout_utils.reshape_acc_to_mn(thr_mma.get_slice(0).partition_C(cO))
        if taccOcO[0][1] == 0:
            for m in cutlass.range(cute.size(taccOgLSE.shape[1]), unroll_full=True):
                if (
                    t0accOcO[m, 0][0]
                    < seqlen.seqlen_q - m_block * self.tile_m - taccOcO[0][0]
                ):
                    taccOgLSE[m, 0] = lse[m]

        # mO has internal layout (q, d, head, batch, split).
        mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[
            None, None, head_idx, split_idx
        ]
        cute.arch.fence_view_async_shared()
        cute.arch.barrier_arrive(
            barrier_id=int(NamedBarrierFwd.Epilogue),
            number_of_threads=self.num_epilogue_threads + cute.arch.WARP_SIZE,
        )
        gO = cute.local_tile(mO_cur, (self.tile_m, self.tile_hdimv), (m_block, 0))
        store_O, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_O, 0, cute.make_layout(1), sO, gO, single_stage=True
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 4:
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.Epilogue),
                number_of_threads=self.num_epilogue_threads + cute.arch.WARP_SIZE,
            )
            store_O()
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=True)

    @cute.jit
    def compute_first_qk(
        self,
        n_block: Int32,
        warp_group_idx: Int32,
        pipeline_k: pipeline.PipelineAsync,
        k_consumer_state,
        mma_qk_fn: Callable,
        tOrP: cute.Tensor,
        softmax: Softmax,
        mask_fn: Callable,
        apply_mask: cutlass.Constexpr,
    ):
        """Compute the first score tile and retain P in registers for the next iteration."""
        pipeline_k.consumer_wait(
            k_consumer_state, pipeline_k.consumer_try_wait(k_consumer_state)
        )
        self.warp_scheduler_barrier_sync(warp_group_idx)
        acc_S = mma_qk_fn(B_idx=k_consumer_state.index, wg_wait=0)
        pipeline_k.consumer_release(k_consumer_state)
        k_consumer_state.advance()
        self.warp_scheduler_barrier_arrive(warp_group_idx)

        if const_expr(apply_mask):
            mask_fn(acc_S=acc_S, n_block=n_block)
        softmax.online_softmax(acc_S, is_first=True)
        utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_S), tOrP)
        return k_consumer_state

    @cute.jit
    def compute_overlapped_block(
        self,
        n_block: Int32,
        warp_group_idx: Int32,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        k_consumer_state,
        v_consumer_state,
        mma_qk_fn: Callable,
        mma_pv_fn: Callable,
        acc_O: cute.Tensor,
        tOrP: cute.Tensor,
        softmax: Softmax,
        mask_fn: Callable,
        apply_mask: cutlass.Constexpr,
        is_first_pv: cutlass.Constexpr,
    ):
        """Overlap QK[current], softmax[current], and PV[previous]."""
        # Wait for both operands before taking the WGMMA issue token, so a slow
        # TMA stage cannot block the other consumer warpgroup behind this one.
        pipeline_k.consumer_wait(
            k_consumer_state, pipeline_k.consumer_try_wait(k_consumer_state)
        )
        pipeline_v.consumer_wait(
            v_consumer_state, pipeline_v.consumer_try_wait(v_consumer_state)
        )

        # QK is committed first so wait_group(1) below makes its accumulator visible
        # while leaving the newer PV group eligible to execute asynchronously.
        self.warp_scheduler_barrier_sync(warp_group_idx)
        acc_S = mma_qk_fn(B_idx=k_consumer_state.index, wg_wait=-1)
        mma_pv_fn(
            B_idx=v_consumer_state.index,
            zero_init=is_first_pv,
            wg_wait=1,
        )
        pipeline_k.consumer_release(k_consumer_state)
        k_consumer_state.advance()
        self.warp_scheduler_barrier_arrive(warp_group_idx)

        # QK is complete, but PV can remain in flight while mask, exp, and row
        # reductions consume acc_S on the CUDA cores/SFU. The other consumer WG
        # now owns the issue token and can use the Tensor Cores during this phase.
        if const_expr(apply_mask):
            mask_fn(acc_S=acc_S, n_block=n_block)
        row_scale = softmax.online_softmax(acc_S, check_inf=True)

        # acc_O and tOrP are PV destinations/source operands, so neither may be
        # modified until the outstanding PV has consumed P and updated O.
        warpgroup.wait_group(0)
        pipeline_v.consumer_release(v_consumer_state)
        v_consumer_state.advance()
        softmax.rescale_O(acc_O, row_scale)
        utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_S), tOrP)
        return k_consumer_state, v_consumer_state

    @cute.jit
    def flush_last_pv(
        self,
        warp_group_idx: Int32,
        pipeline_v: pipeline.PipelineAsync,
        v_consumer_state,
        mma_pv_fn: Callable,
        zero_init: cutlass.Constexpr,
    ):
        """Consume the final deferred P/V tile after the QK loop is exhausted."""
        pipeline_v.consumer_wait(
            v_consumer_state, pipeline_v.consumer_try_wait(v_consumer_state)
        )
        self.warp_scheduler_barrier_sync(warp_group_idx)
        mma_pv_fn(
            B_idx=v_consumer_state.index,
            zero_init=zero_init,
            wg_wait=-1,
        )
        self.warp_scheduler_barrier_arrive(warp_group_idx)
        warpgroup.wait_group(0)
        pipeline_v.consumer_release(v_consumer_state)
        v_consumer_state.advance()
        return v_consumer_state


# ---------------------------------------------------------------------------
# ops 接口封装：把上面的 CuTe DSL 内核包成 benchmark 约定的函数签名
# ---------------------------------------------------------------------------

_TORCH_TO_CUTE_DTYPE = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
}

# Forward cache distinguishes split/non-split specializations, but concrete split
# counts remain dynamic. The combine cache is bucketed by its maximum split count.
_COMPILE_CACHE = {}
_COMBINE_CACHE = {}


def _select_decode_num_splits(q, k, causal, requested):
    """Choose a small split count that improves CTA-wave utilization."""
    batch, q_heads, q_len, _ = q.shape
    kv_len = k.shape[2]
    n_blocks = math.ceil(kv_len / 128)
    is_supported_decode = q_len == 1 and not causal and kv_len > 0

    if requested not in (None, 0):
        if requested < 1:
            raise ValueError(f"num_splits 必须是正整数、0 或 None，实际: {requested}")
        if requested > 1 and not is_supported_decode:
            raise ValueError("num_splits>1 第一版仅支持 q_len=1、causal=False 的 decode")
        if requested > min(256, n_blocks):
            raise ValueError(
                f"num_splits={requested} 超过可用 KV blocks={n_blocks} 或上限 256"
            )
        return int(requested)

    if not is_supported_decode or n_blocks <= 4:
        return 1

    num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    base_ctas = batch * q_heads
    max_splits = min(16, 256, n_blocks)

    def wave_efficiency(splits):
        ctas = base_ctas * splits
        waves = math.ceil(ctas / num_sms)
        return ctas / (waves * num_sms)

    efficiencies = [wave_efficiency(s) for s in range(1, max_splits + 1)]
    best = max(efficiencies)
    if best <= efficiencies[0] * 1.05:
        return 1
    target = best * 0.85
    return next(s for s, efficiency in enumerate(efficiencies, 1) if efficiency >= target)


def _run_splitkv_combine(out_partial, lse_partial, out):
    num_splits = out_partial.shape[0]
    log_max_splits = max(math.ceil(math.log2(num_splits)), 5)
    key = (out.dtype, out.shape[-1], log_max_splits)
    compiled = _COMBINE_CACHE.get(key)
    lse_for_combine = lse_partial.transpose(-1, -2)
    if compiled is None:
        combine = FlashAttentionForwardCombine(
            dtype=_TORCH_TO_CUTE_DTYPE[out.dtype],
            dtype_partial=Float32,
            head_dim=out.shape[-1],
            tile_m=8,
            k_block_size=128,
            log_max_splits=log_max_splits,
        )
        if not combine.can_implement(
            _TORCH_TO_CUTE_DTYPE[out.dtype],
            Float32,
            out.shape[-1],
            8,
            128,
            log_max_splits,
            256,
        ):
            raise RuntimeError("Split-KV combine configuration is unsupported")
        compiled = cute.compile(
            combine,
            to_cute_tensor(out_partial),
            to_cute_tensor(lse_for_combine),
            to_cute_tensor(out),
            None, None, None, None, None, None,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        _COMBINE_CACHE[key] = compiled
    compiled(out_partial, lse_for_combine, out, None, None, None, None, None, None)


def attention(q, k, v, causal=True, sm_scale=None, num_splits=None):
    """SM90 (Hopper) CuTe DSL FlashAttention 前向，支持 GQA。

    Args:
        q: shape (batch, q_heads, q_len, head_dim)
        k, v: shape (batch, kv_heads, kv_len, head_dim)，q_heads 必须是
            kv_heads 的整数倍（GQA 由内核的 qhead_per_kvhead 路径原生支持，
            无需把 k/v repeat 到 q_heads）
        causal: 是否使用因果掩码
        sm_scale: softmax 缩放系数，默认为 1/sqrt(head_dim)
        num_splits: decode Split-KV 数；None/0 自动选择，1 强制旧路径

    Returns:
        output: shape 与 q 相同，(batch, q_heads, q_len, head_dim)
    """
    batch, q_heads, q_len, head_dim = q.shape
    kv_heads = k.shape[1]
    if head_dim != 128 or k.shape[-1] != 128 or v.shape[-1] != 128:
        raise ValueError(f"本内核仅支持 head_dim=128，实际: q={head_dim} k={k.shape[-1]} v={v.shape[-1]}")
    if q.dtype not in _TORCH_TO_CUTE_DTYPE:
        raise ValueError(f"本内核仅支持 fp16/bf16，实际: {q.dtype}")
    if q_heads % kv_heads != 0:
        raise ValueError(f"q_heads={q_heads} 必须是 kv_heads={kv_heads} 的整数倍（GQA）")
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    qhead_per_kvhead = q_heads // kv_heads
    selected_splits = _select_decode_num_splits(q, k, bool(causal), num_splits)
    is_split_kv = selected_splits > 1

    # 内核使用 BSHD 布局；benchmark 内部统一是 BHSD，这里转成 BSHD。
    # 与 baseline 的 maybe_contiguous 行为一致：只要 head_dim 维 stride 为 1，
    # 就直接把转置后的 strided 视图交给内核（TMA 描述符原生支持任意 stride），
    # 避免每次调用额外做 3 次 contiguous 拷贝（decode 阶段这批拷贝约占 20% 耗时）。
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    q_t = q_t if q_t.stride(-1) == 1 else q_t.contiguous()
    k_t = k_t if k_t.stride(-1) == 1 else k_t.contiguous()
    v_t = v_t if v_t.stride(-1) == 1 else v_t.contiguous()
    # Final output is always BSHD. Split-KV additionally writes normalized FP32
    # partial O/LSE, then combines them in a second kernel.
    o_t = torch.empty((batch, q_len, q_heads, head_dim), dtype=q.dtype, device=q.device)
    if is_split_kv:
        kernel_o = torch.empty(
            (selected_splits, batch, q_len, q_heads, head_dim),
            dtype=torch.float32,
            device=q.device,
        )
        kernel_lse = torch.empty(
            (selected_splits, batch, q_heads, q_len),
            dtype=torch.float32,
            device=q.device,
        )
    else:
        kernel_o = o_t
        kernel_lse = None

    key = (q.dtype, head_dim, qhead_per_kvhead, bool(causal), is_split_kv)
    compiled = _COMPILE_CACHE.get(key)
    if compiled is None:
        fa_fwd = FlashAttentionForwardSm90(
            _TORCH_TO_CUTE_DTYPE[q.dtype],
            head_dim,
            head_dim,  # head_dim_v
            qhead_per_kvhead,
            is_causal=causal,
            is_local=False,
            pack_gqa=False,
            tile_m=128,
            tile_n=128,
            num_stages=2,
            num_threads=256,
            Q_in_regs=False,
            intra_wg_overlap=True,
            mma_pv_is_rs=True,
            is_split_kv=is_split_kv,
        )
        compiled = cute.compile(
            fa_fwd,
            to_cute_tensor(q_t),
            to_cute_tensor(k_t),
            to_cute_tensor(v_t),
            to_cute_tensor(kernel_o),
            to_cute_tensor(kernel_lse) if kernel_lse is not None else None,
            sm_scale,
            None, None, None, None,  # cu_seqlens_q/k, seqused_q/k
            None,  # page_table
            None, None,  # window_size_left/right
            None,  # learnable_sink
            None,  # blocksparse_tensors
            AuxData(),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        _COMPILE_CACHE[key] = compiled

    compiled(
        q_t, k_t, v_t, kernel_o,
        kernel_lse,
        sm_scale,
        None, None, None, None,
        None,
        None, None,
        None,
        None,
        AuxData(),
    )
    if is_split_kv:
        _run_splitkv_combine(kernel_o, kernel_lse, o_t)
    return o_t.transpose(1, 2)


register("lhx_flash_attention (cute-dsl sm90)", attention)
