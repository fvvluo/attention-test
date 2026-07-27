# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed-shape Qwen3 Decode attention for H20.

The production path is exactly two CuTe kernels: a split-KV M64 Hopper WGMMA
kernel that writes raw FP32 partials, followed by a warp-per-head raw split
reduction that writes BF16 output.  The only supported workload is
Q=[1,64,1,128], K/V=[1,8,131072,128], contiguous BF16 on the current CUDA
device (any available H20; not hard-coded to cuda:0).
"""

import argparse
import inspect as _inspect
import math
import os
import statistics
import sys
import threading
from contextlib import contextmanager

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.warpgroup as warpgroup
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import (
    Boolean,
    Int32,
    if_generate,
    not_,
    while_generate,
    yield_out,
    dsl_user_op,
)
from cutlass._mlir.dialects import nvvm
from cutlass._mlir._mlir_libs._cutlass_ir._mlir.ir import IntegerType


QWEN_BATCH = 1
QWEN_QUERY_HEADS = 64
QWEN_KV_HEADS = 8
QWEN_HEAD_DIM = 128
QWEN_CONTEXT = 128 * 1024
HEAD_RATIO = QWEN_QUERY_HEADS // QWEN_KV_HEADS
LOG2_E = 1.4426950408889634074
LN2 = 0.6931471805599453094
CONFIGS = {
    "m64-n128-kv5-s9": {
        "block_m": 64,
        "block_n": 128,
        "kv_stage": 5,
        "splits": 9,
    },
    "m64-n176-kv3-s8": {
        "block_m": 64,
        "block_n": 176,
        "kv_stage": 3,
        "splits": 8,
    },
    "m64-n176-kv3-s9": {
        "block_m": 64,
        "block_n": 176,
        "kv_stage": 3,
        "splits": 9,
    },
    "m64-n176-kv3-s10": {
        "block_m": 64,
        "block_n": 176,
        "kv_stage": 3,
        "splits": 10,
    },
    "m64-n176-kv3-s18": {
        "block_m": 64,
        "block_n": 176,
        "kv_stage": 3,
        "splits": 18,
    },
    "m64-n176-kv3-s19": {
        "block_m": 64,
        "block_n": 176,
        "kv_stage": 3,
        "splits": 19,
    },
    "m64-n176-kv3-s20": {
        "block_m": 64,
        "block_n": 176,
        "kv_stage": 3,
        "splits": 20,
    },
    "m64-n176-kv4-s9": {
        "block_m": 64,
        "block_n": 176,
        "kv_stage": 4,
        "splits": 9,
    },
    "m64-n176-kv4-s19": {
        "block_m": 64,
        "block_n": 176,
        "kv_stage": 4,
        "splits": 19,
    },
    "m64-n160-kv3-s19": {
        "block_m": 64,
        "block_n": 160,
        "kv_stage": 3,
        "splits": 19,
    },
    "m64-n192-kv3-s19": {
        "block_m": 64,
        "block_n": 192,
        "kv_stage": 3,
        "splits": 19,
    },
    "m64-n176-kv3-s19-r32-232": {
        "block_m": 64,
        "block_n": 176,
        "kv_stage": 3,
        "splits": 19,
        "num_regs_load": 32,
        "num_regs_mma": 232,
    },
    "m64-n176-kv3-s19-r24-248": {
        "block_m": 64,
        "block_n": 176,
        "kv_stage": 3,
        "splits": 19,
        "num_regs_load": 24,
        "num_regs_mma": 248,
    },
    "m64-n176-k2v2-s19": {
        "block_m": 64,
        "block_n": 176,
        "kv_stage": 2,
        "splits": 19,
        "kernel": "k2v2",
    },
    "m64-n176-k2v2-overlap-s19": {
        "block_m": 64,
        "block_n": 176,
        "kv_stage": 2,
        "splits": 19,
        "kernel": "k2v2-overlap",
    },
}
# Winner of repeated four-process interleaved tuning on GPU 4: most stable
# across thermal states while remaining faster than the FA3/P2 references.
AUTO_CONFIG = "m64-n176-kv3-s19"

_COMPILE_LOCK = threading.Lock()
_PARTIAL_KERNEL_CACHE = {}
_COMBINE_KERNEL_CACHE = {}
_WORKSPACE_CACHE = {}
_WORKSPACE_LOCK = threading.Lock()
_WORKSPACE_CACHE_LIMIT = 32
_WORKSPACE_LAYOUT_VERSION = 1
_PARTIAL_KERNEL_VERSION = 1
_COMBINE_KERNEL_VERSION = 1


_timelimit_has_res = "res" in _inspect.signature(
    nvvm.mbarrier_try_wait_parity_timelimit
).parameters
_MBARIER_PATCH_LOCK = threading.RLock()


def _try_wait_timelimit(llvm_ptr, phase_val, timeout, *, loc=None, ip=None):
    if _timelimit_has_res:
        i1 = IntegerType.get_signless(1)
        return nvvm.mbarrier_try_wait_parity_timelimit(
            i1, llvm_ptr, phase_val, timeout, loc=loc, ip=ip
        )
    return nvvm.mbarrier_try_wait_parity_timelimit(
        llvm_ptr, phase_val, timeout, loc=loc, ip=ip
    )


@dsl_user_op
def _optimized_mbarrier_wait(mbar_ptr, phase, *, loc=None, ip=None):
    llvm_ptr = mbar_ptr.llvm_ptr
    phase_val = Int32(phase).ir_value(loc=loc, ip=ip)
    true_value = lambda: Boolean(True).ir_value(loc=loc, ip=ip)
    timeout = Int32(10000000).ir_value(loc=loc, ip=ip)
    done = Boolean(_try_wait_timelimit(llvm_ptr, phase_val, timeout, loc=loc, ip=ip))
    done = if_generate(
        done,
        true_value,
        lambda: _try_wait_timelimit(llvm_ptr, phase_val, timeout, loc=loc, ip=ip),
        None,
        [Boolean],
        loc=loc,
        ip=ip,
    )
    done = if_generate(
        done,
        true_value,
        lambda: _try_wait_timelimit(llvm_ptr, phase_val, timeout, loc=loc, ip=ip),
        None,
        [Boolean],
        loc=loc,
        ip=ip,
    )

    def fallback():
        initial = Boolean(False).ir_value(loc=loc, ip=ip)
        loop = while_generate(
            [initial], lambda value: not_(value, loc=loc, ip=ip), loc=loc, ip=ip
        )
        with loop as (_,):
            result = Boolean(
                _try_wait_timelimit(llvm_ptr, phase_val, timeout, loc=loc, ip=ip)
            )
            yield_out([result], loc=loc, ip=ip)
        return Boolean(True).ir_value(loc=loc, ip=ip)

    if_generate(done, true_value, fallback, None, [Boolean], loc=loc, ip=ip)


@contextmanager
def _use_optimized_mbarrier_wait():
    import cutlass.cute.arch as arch_mod

    with _MBARIER_PATCH_LOCK:
        original = arch_mod.mbarrier_wait
        arch_mod.mbarrier_wait = _optimized_mbarrier_wait
        try:
            yield
        finally:
            arch_mod.mbarrier_wait = original


class HopperDecode128KRawForward:
    """M64 split-KV kernel with one load WG and one compute WG."""

    def __init__(
        self,
        qk_acc_dtype,
        pv_acc_dtype,
        mma_tiler,
        kv_stage,
        num_splits,
        num_regs_load=24,
        num_regs_mma=240,
    ):
        if not 1 <= num_splits <= 32:
            raise ValueError("num_splits must fit in one warp")
        self.num_splits = num_splits
        self.num_mma_warp_groups = 1
        self.qk_acc_dtype = qk_acc_dtype
        self.pv_acc_dtype = pv_acc_dtype
        self.cta_tiler = self.cta_tile_shape_mnk = mma_tiler
        self.qk_mma_tiler = mma_tiler
        self.pv_mma_tiler = (mma_tiler[0], mma_tiler[2], mma_tiler[1])
        self.cluster_shape_mn = (1, 1)
        self.atom_layout_mnk = (1, 1, 1)
        self.configured_kv_stage = kv_stage
        self.separate_kv = False

        self.threads_per_warp = 32
        self.num_threads_per_warp_group = 128
        self.num_warps_per_warp_group = 4
        self.load_warp_group_id = 0
        self.compute_warp_group_id = 1
        self.producer_warp_loadkv_id = 1
        self.num_regs_load = num_regs_load
        self.num_regs_mma = num_regs_mma
        self.threads_per_cta = 256
        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        self.q_stage = 1
        self.kv_stage = self.configured_kv_stage

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        o_raw: cute.Tensor,
        stats_raw: cute.Tensor,
        scale_softmax_log2: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        self.q_dtype = q.element_type
        self.k_dtype = k.element_type
        self.v_dtype = v.element_type
        self.o_dtype = o_raw.element_type

        if cutlass.const_expr(self.q_dtype != cutlass.BFloat16):
            raise TypeError("fixed Decode Q must be BFloat16")
        if cutlass.const_expr(self.k_dtype != self.q_dtype or self.v_dtype != self.q_dtype):
            raise TypeError("fixed Decode Q/K/V types must match")
        if cutlass.const_expr(self.o_dtype != cutlass.Float32):
            raise TypeError("raw O workspace must be Float32")
        if cutlass.const_expr(stats_raw.element_type != cutlass.Float32):
            raise TypeError("raw stats workspace must be Float32")
        if cutlass.const_expr(q.leading_dim != 1 or k.leading_dim != 1):
            raise RuntimeError("Q and K must be k-major")

        self._setup_attributes()
        self.q_layout = utils.LayoutEnum.from_tensor(q)
        self.k_layout = utils.LayoutEnum.from_tensor(k)
        self.v_layout = utils.LayoutEnum.from_tensor(v)
        self.q_major_mode = self.q_layout.sm90_mma_major_mode()
        self.k_major_mode = self.k_layout.sm90_mma_major_mode()
        self.v_major_mode = self.v_layout.sm90_mma_major_mode()

        qk_tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.k_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.qk_acc_dtype,
            self.atom_layout_mnk,
            self.qk_mma_tiler[:2],
        )
        pv_tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.v_dtype,
            self.v_dtype,
            cute.nvgpu.OperandMajorMode.K,
            self.v_major_mode,
            self.pv_acc_dtype,
            self.atom_layout_mnk,
            self.pv_mma_tiler[:2],
            warpgroup.OperandSource.RMEM,
        )

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        q_smem_layout_staged = sm90_utils.make_smem_layout_a(
            self.q_layout, self.qk_mma_tiler, self.q_dtype, self.q_stage
        )
        k_smem_layout_staged = sm90_utils.make_smem_layout_b(
            self.k_layout, self.qk_mma_tiler, self.k_dtype, self.kv_stage
        )
        v_smem_layout_staged = sm90_utils.make_smem_layout_b(
            self.v_layout, self.pv_mma_tiler, self.v_dtype, self.kv_stage
        )

        q_smem_layout = cute.slice_(q_smem_layout_staged, (None, None, 0))
        tma_atom_q, tma_tensor_q = self._make_tma_atoms_and_tensors(
            q,
            q_smem_layout_staged,
            (self.qk_mma_tiler[0], self.qk_mma_tiler[2]),
            1,
        )
        k_smem_layout = cute.slice_(k_smem_layout_staged, (None, None, 0))
        tma_atom_k, tma_tensor_k = self._make_tma_atoms_and_tensors(
            k,
            k_smem_layout_staged,
            (self.qk_mma_tiler[1], self.qk_mma_tiler[2]),
            1,
        )
        tma_atom_v, tma_tensor_v = self._make_tma_atoms_and_tensors(
            v,
            v_smem_layout_staged,
            (self.pv_mma_tiler[1], self.pv_mma_tiler[2]),
            1,
        )
        self.tma_copy_q_bytes = cute.size_in_bytes(self.q_dtype, q_smem_layout)
        self.tma_copy_kv_bytes = cute.size_in_bytes(self.k_dtype, k_smem_layout)

        # Insert a stride-zero fake-D mode so stats can reuse the PV C mapping.
        stats_raw = cute.make_tensor(
            stats_raw.iterator,
            cute.make_layout(
                (
                    stats_raw.shape[0],
                    self.pv_mma_tiler[1],
                    stats_raw.shape[1],
                    stats_raw.shape[2],
                    stats_raw.shape[3],
                ),
                stride=(
                    stats_raw.stride[0],
                    0,
                    stats_raw.stride[1],
                    stats_raw.stride[2],
                    stats_raw.stride[3],
                ),
            ),
        )

        if cutlass.const_expr(self.separate_kv):

            @cute.struct
            class SharedStorage:
                load_q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.q_stage * 2]
                load_k_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.kv_stage * 2]
                load_v_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.kv_stage * 2]
                sQ: cute.struct.Align[
                    cute.struct.MemRange[self.q_dtype, cute.cosize(q_smem_layout_staged)],
                    self.buffer_align_bytes,
                ]
                sK: cute.struct.Align[
                    cute.struct.MemRange[self.k_dtype, cute.cosize(k_smem_layout_staged)],
                    self.buffer_align_bytes,
                ]
                sV: cute.struct.Align[
                    cute.struct.MemRange[self.v_dtype, cute.cosize(v_smem_layout_staged)],
                    self.buffer_align_bytes,
                ]

        else:

            @cute.struct
            class SharedStorage:
                load_q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.q_stage * 2]
                load_kv_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.kv_stage * 2]
                sQ: cute.struct.Align[
                    cute.struct.MemRange[self.q_dtype, cute.cosize(q_smem_layout_staged)],
                    self.buffer_align_bytes,
                ]
                sK: cute.struct.Align[
                    cute.struct.MemRange[self.k_dtype, cute.cosize(k_smem_layout_staged)],
                    self.buffer_align_bytes,
                ]

        self.shared_storage = SharedStorage
        with _use_optimized_mbarrier_wait():
            self.kernel(
                qk_tiled_mma,
                pv_tiled_mma,
                tma_atom_q,
                tma_tensor_q,
                tma_atom_k,
                tma_tensor_k,
                tma_atom_v,
                tma_tensor_v,
                o_raw,
                stats_raw,
                scale_softmax_log2,
                q_smem_layout_staged,
                k_smem_layout_staged,
                v_smem_layout_staged,
            ).launch(
                grid=(self.num_splits, QWEN_KV_HEADS, 1),
                block=(self.threads_per_cta, 1, 1),
                cluster=self.cluster_shape_mnk,
                stream=stream,
                min_blocks_per_mp=1,
            )

    @cute.kernel
    def kernel(
        self,
        qk_tiled_mma: cute.TiledMma,
        pv_tiled_mma: cute.TiledMma,
        tma_atom_q: cute.CopyAtom,
        mQ_qdl: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        mK_kdl: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        mV_dkl: cute.Tensor,
        mO_qdl: cute.Tensor,
        mStats: cute.Tensor,
        scale_softmax_log2: cutlass.Float32,
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        split_idx, kv_head_idx, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group
        )

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        load_q_producer, load_q_consumer = self.make_and_init_load_q_pipeline(
            storage.load_q_mbar_ptr.data_ptr()
        )
        load_kv_producer, load_kv_consumer = self.make_and_init_load_kv_pipeline(
            storage.load_kv_mbar_ptr.data_ptr()
        )

        sQ = storage.sQ.get_tensor(
            q_smem_layout_staged.outer, swizzle=q_smem_layout_staged.inner
        )
        sK = storage.sK.get_tensor(
            k_smem_layout_staged.outer, swizzle=k_smem_layout_staged.inner
        )
        sV_ptr = cute.recast_ptr(sK.iterator, v_smem_layout_staged.inner)
        sV = cute.make_tensor(sV_ptr, v_smem_layout_staged.outer)

        seqlen_q = mQ_qdl.shape[0]
        seqlen_k = mK_kdl.shape[0]
        num_n_tiles = cute.ceil_div(seqlen_k, self.qk_mma_tiler[1])
        tile_begin = split_idx * num_n_tiles // self.num_splits
        tile_end = (split_idx + 1) * num_n_tiles // self.num_splits
        split_k_tiles = tile_end - tile_begin

        gQ_qdl = cute.flat_divide(
            mQ_qdl, cute.select(self.qk_mma_tiler, mode=[0, 2])
        )
        qk_thr_mma = qk_tiled_mma.get_slice(tidx)
        tSgQ_qdl = qk_thr_mma.partition_A(gQ_qdl)
        tQsQ, tQgQ_qdl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_q,
            0,
            cute.make_layout(1),
            cute.group_modes(sQ, 0, 2),
            cute.group_modes(tSgQ_qdl, 0, 3),
        )

        gK_kdl = cute.flat_divide(
            mK_kdl, cute.select(self.qk_mma_tiler, mode=[1, 2])
        )
        tSgK_kdl = qk_thr_mma.partition_B(gK_kdl)
        tKsK, tKgK_kdl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_k,
            0,
            cute.make_layout(1),
            cute.group_modes(sK, 0, 2),
            cute.group_modes(tSgK_kdl, 0, 3),
        )

        gV_dkl = cute.flat_divide(
            mV_dkl, cute.select(self.pv_mma_tiler, mode=[1, 2])
        )
        pv_thr_mma = pv_tiled_mma.get_slice(tidx)
        tSgV_dkl = pv_thr_mma.partition_B(gV_dkl)
        tVsV, tVgV_dkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_v,
            0,
            cute.make_layout(1),
            cute.group_modes(sV, 0, 2),
            cute.group_modes(tSgV_dkl, 0, 3),
        )

        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)

        if warp_group_idx == self.load_warp_group_id:
            cute.arch.setmaxregister_decrease(self.num_regs_load)
            producer_warp_role = warp_idx % self.num_warps_per_warp_group
            if producer_warp_role == self.producer_warp_loadkv_id:
                tQgQ = tQgQ_qdl[(None, None, 0, kv_head_idx)]
                q_handle = load_q_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_q,
                    tQgQ[(None, 0)],
                    tQsQ[(None, q_handle.index)],
                    tma_bar_ptr=q_handle.barrier,
                )

                tKgK = tKgK_kdl[(None, None, 0, kv_head_idx)]
                tVgV = tVgV_dkl[(None, 0, None, kv_head_idx)]
                k_index = tile_begin
                kv_load_count = 2 * split_k_tiles
                while kv_load_count > 0:
                    k_handle = load_kv_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_k,
                        tKgK[(None, k_index)],
                        tKsK[(None, k_handle.index)],
                        tma_bar_ptr=k_handle.barrier,
                    )
                    kv_load_count -= 1

                    v_handle = load_kv_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_v,
                        tVgV[(None, k_index)],
                        tVsV[(None, v_handle.index)],
                        tma_bar_ptr=v_handle.barrier,
                    )
                    k_index += 1
                    kv_load_count -= 1

        if warp_group_idx == self.compute_warp_group_id:
            cute.arch.setmaxregister_increase(self.num_regs_mma)
            wg_coord = (0, 0, kv_head_idx)

            tSsQ = qk_thr_mma.partition_A(sQ)
            tSsK = qk_thr_mma.partition_B(sK)
            tSrQ = qk_thr_mma.make_fragment_A(tSsQ)
            tSrK = qk_thr_mma.make_fragment_B(tSsK)
            tOsV = pv_thr_mma.partition_B(sV)
            tOrV = pv_thr_mma.make_fragment_B(tOsV)
            q_handle = load_q_consumer.wait()

            cP = cute.make_identity_tensor((mQ_qdl.shape[0], seqlen_k))
            gPcP = cute.local_tile(cP, self.qk_mma_tiler[:2], (None, None))
            ptPcP = qk_thr_mma.partition_C(gPcP)
            pv_acc_shape = pv_thr_mma.partition_shape_C(
                (self.pv_mma_tiler[0], self.pv_mma_tiler[1])
            )
            acc_pv = pv_thr_mma.make_fragment_C(pv_acc_shape)
            qk_acc_shape = qk_thr_mma.partition_shape_C(
                (self.qk_mma_tiler[0], self.qk_mma_tiler[1])
            )
            row_fragment_layout = cute.make_layout(
                cute.size(self.layout_acc_mn(pv_tiled_mma, acc_pv.layout), mode=[0])
            )
            s_max = cute.make_rmem_tensor_like(row_fragment_layout, cutlass.Float32)
            a_sum = cute.make_rmem_tensor_like(row_fragment_layout, cutlass.Float32)

            global_tile_idx = tile_begin
            tPcP = cute.slice_(ptPcP, (None, None, None, 0, global_tile_idx))
            kv_offset = global_tile_idx + 1
            acc_qk = qk_thr_mma.make_fragment_C(qk_acc_shape)
            k_handle = load_kv_consumer.wait_and_advance()

            cute.nvgpu.warpgroup.fence()
            self.gemm_zero_acc(
                qk_tiled_mma,
                tSrQ[(None, None, None, q_handle.index)],
                tSrK[(None, None, None, k_handle.index)],
                acc_qk,
            )
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)

            if cutlass.const_expr(QWEN_CONTEXT % self.qk_mma_tiler[1] != 0):
                self.mask_fixed_residue(acc_qk, tPcP, global_tile_idx)
            s_max, a_sum = self.softmax_step(
                acc_qk,
                qk_tiled_mma,
                s_max,
                a_sum,
                acc_qk,
                qk_tiled_mma,
                scale_softmax_log2,
                True,
            )
            acc_qk_fixed = self.make_acc_into_op(
                acc_qk, pv_tiled_mma.tv_layout_A, self.q_dtype
            )
            v_handle = load_kv_consumer.wait_and_advance()
            cute.nvgpu.warpgroup.fence()
            self.gemm_zero_acc(
                pv_tiled_mma,
                acc_qk_fixed,
                tOrV[(None, None, None, v_handle.index)],
                acc_pv,
            )
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)
            k_handle.release()
            v_handle.release()

            load_kv_consumer, kv_offset, s_max, a_sum = self.compute(
                split_k_tiles - 1,
                qk_thr_mma,
                acc_pv,
                qk_tiled_mma,
                pv_tiled_mma,
                load_kv_consumer,
                q_handle,
                tSrQ,
                tSrK,
                s_max,
                a_sum,
                tOrV,
                ptPcP,
                wg_coord,
                kv_offset,
                scale_softmax_log2,
                qk_acc_shape,
            )
            cute.nvgpu.warpgroup.wait_group(0)
            self.reduce_raw_sum(a_sum, acc_pv, pv_tiled_mma)
            q_handle.release()

            thr_mma = pv_tiled_mma.get_slice(tidx)
            cD = cute.make_identity_tensor(
                (self.pv_mma_tiler[0], self.pv_mma_tiler[1])
            )
            tOcD = thr_mma.partition_C(cD)
            tOcD_mn = cute.make_tensor(
                tOcD.iterator, self.layout_acc_mn(pv_tiled_mma, tOcD.layout)
            )

            # Raw stats: [max_log2, sum], no logarithm and no normalization.
            gStats_full = cute.local_tile(
                mStats, self.pv_mma_tiler[:2], (None, None, None, None, None)
            )
            for stat_idx in cutlass.range_constexpr(2):
                gStats = cute.slice_(
                    gStats_full,
                    (None, None, 0, 0, kv_head_idx, split_idx, stat_idx),
                )
                tOgStats = thr_mma.partition_C(gStats)
                tOgStats_mn = cute.make_tensor(
                    tOgStats.iterator,
                    self.layout_acc_mn(pv_tiled_mma, tOgStats.layout),
                )
                if tOcD[0][1] == 0:
                    for i in cutlass.range_constexpr(
                        cute.size(tOgStats_mn, mode=[0])
                    ):
                        if tOcD_mn[(i, 0)][0] < seqlen_q:
                            value = a_sum[i]
                            if stat_idx == 0:
                                value = s_max[i] * scale_softmax_log2
                            tOgStats_mn[(i, 0)] = value

            # Raw unnormalized R_j accumulator.
            gD = cute.local_tile(
                mO_qdl,
                self.pv_mma_tiler[:2],
                (0, 0, kv_head_idx, split_idx),
            )
            tOgD = thr_mma.partition_C(gD)
            acc_pv_mn = cute.make_tensor(
                acc_pv.iterator, self.layout_acc_mn(pv_tiled_mma, acc_pv.layout)
            )
            tOgD_mn = cute.make_tensor(
                tOgD.iterator, self.layout_acc_mn(pv_tiled_mma, tOgD.layout)
            )
            for i in cutlass.range_constexpr(cute.size(acc_pv_mn, mode=[0])):
                if tOcD_mn[(i, 0)][0] < seqlen_q:
                    cute.autovec_copy(acc_pv_mn[i, None], tOgD_mn[i, None])
        return

    @cute.jit
    def compute(
        self,
        k_tile_count: cutlass.Int32,
        qk_thr_mma: cute.ThrMma,
        acc_pv: cute.ThrMma,
        qk_tiled_mma: cute.TiledMma,
        pv_tiled_mma: cute.TiledMma,
        load_kv_consumer: pipeline.PipelineConsumer,
        q_handle: pipeline.PipelineConsumer.ImmutableResourceHandle,
        tSrQ: cute.Tensor,
        tSrK: cute.Tensor,
        s_max: cute.Tensor,
        a_sum: cute.Tensor,
        tOrV: cute.Tensor,
        ptPcP: cute.Tensor,
        wg_coord: tuple,
        kv_offset: cutlass.Int32,
        scale_softmax_log2: cutlass.Float32,
        qk_acc_shape: cute.Shape,
    ):
        while k_tile_count > 0:
            k_tile_count -= 1
            global_tile_idx = kv_offset
            tPcP = cute.slice_(
                ptPcP, (None, None, None, wg_coord[0], global_tile_idx)
            )
            kv_offset = global_tile_idx + 1
            acc_qk = qk_thr_mma.make_fragment_C(qk_acc_shape)
            k_handle = load_kv_consumer.wait_and_advance()
            cute.nvgpu.warpgroup.fence()
            self.gemm_zero_acc(
                qk_tiled_mma,
                tSrQ[(None, None, None, q_handle.index)],
                tSrK[(None, None, None, k_handle.index)],
                acc_qk,
            )
            cute.nvgpu.warpgroup.commit_group()
            token = load_kv_consumer.try_wait()
            cute.nvgpu.warpgroup.wait_group(0)
            if cutlass.const_expr(QWEN_CONTEXT % self.qk_mma_tiler[1] != 0):
                self.mask_fixed_residue(acc_qk, tPcP, global_tile_idx)
            s_max, a_sum = self.softmax_step(
                acc_qk,
                qk_tiled_mma,
                s_max,
                a_sum,
                acc_pv,
                pv_tiled_mma,
                scale_softmax_log2,
                False,
            )
            acc_qk_fixed = self.make_acc_into_op(
                acc_qk, pv_tiled_mma.tv_layout_A, self.q_dtype
            )
            v_handle = load_kv_consumer.wait_and_advance(token)
            cute.nvgpu.warpgroup.fence()
            pv_tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
            cute.gemm(
                pv_tiled_mma,
                acc_pv,
                acc_qk_fixed,
                tOrV[(None, None, None, v_handle.index)],
                acc_pv,
            )
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)
            k_handle.release()
            v_handle.release()
        return load_kv_consumer, kv_offset, s_max, a_sum

    @cute.jit
    def mask_fixed_residue(
        self,
        acc_qk: cute.Tensor,
        index_qk: cute.Tensor,
        global_tile_idx: cutlass.Int32,
    ):
        block_n = self.qk_mma_tiler[1]
        last_tile_index = QWEN_CONTEXT // block_n
        last_tile_valid_columns = QWEN_CONTEXT - last_tile_index * block_n
        if global_tile_idx == last_tile_index:
            for i in cutlass.range_constexpr(cute.size(acc_qk)):
                local_n = index_qk[i][1] - last_tile_index * block_n
                if local_n >= last_tile_valid_columns:
                    acc_qk[i] = -cutlass.Float32.inf

    @cute.jit
    def softmax_step(
        self,
        acc_qk: cute.ThrMma,
        tiled_mma_qk: cute.TiledMma,
        s_max: cute.Tensor,
        a_sum: cute.Tensor,
        acc_pv: cute.ThrMma,
        tiled_mma_pv: cute.TiledMma,
        scale_softmax_log2: cutlass.Float32,
        is_first_iter: bool,
    ):
        acc_qk_mn = cute.make_tensor(
            acc_qk.iterator, self.layout_acc_mn(tiled_mma_qk, acc_qk.layout)
        )
        reduction_target_qk = self.reduction_target_n(tiled_mma_qk)
        red_rank = cute.rank(reduction_target_qk)
        s_max_prev = None
        acc_pv_mn = None
        if cutlass.const_expr(is_first_iter):
            for i in cutlass.range_constexpr(cute.size(acc_qk_mn, mode=[0])):
                s_max[i] = acc_qk_mn[i, 0]
            for j in cutlass.range_constexpr(1, cute.size(acc_qk_mn, mode=[1])):
                for i in cutlass.range_constexpr(cute.size(acc_qk_mn, mode=[0])):
                    s_max[i] = cute.arch.fmax(s_max[i], acc_qk_mn[i, j])
        else:
            acc_pv_mn = cute.make_tensor(
                acc_pv.iterator, self.layout_acc_mn(tiled_mma_pv, acc_pv.layout)
            )
            s_max_prev = cute.make_rmem_tensor_like(s_max, s_max._dtype)

        for i in cutlass.range_constexpr(cute.size(acc_qk_mn, mode=[0])):
            if cutlass.const_expr(not is_first_iter):
                s_max_prev[i] = s_max[i]
                for j in cutlass.range_constexpr(cute.size(acc_qk_mn, mode=[1])):
                    s_max[i] = cutlass.max(s_max[i], acc_qk_mn[i, j])
            for r in cutlass.range_constexpr(red_rank):
                s_max[i] = cute.arch.warp_reduction_max(
                    s_max[i], threads_in_group=reduction_target_qk.shape[r]
                )
            local_max = s_max[i]
            if s_max[i] == -cutlass.Float32.inf:
                local_max = 0.0
            scale_max = scale_softmax_log2 * local_max
            for j in cutlass.range_constexpr(cute.size(acc_qk_mn, mode=[1])):
                acc_qk_mn[i, j] = cute.math.exp2(
                    scale_softmax_log2 * acc_qk_mn[i, j] - scale_max,
                    fastmath=True,
                )
            local_sum = 0.0
            if cutlass.const_expr(not is_first_iter):
                current_max = s_max[i]
                if current_max == -cutlass.Float32.inf:
                    current_max = 0.0
                correction = cute.math.exp2(
                    (s_max_prev[i] - current_max) * scale_softmax_log2,
                    fastmath=True,
                )
                a_sum[i] *= correction
                for j in cutlass.range_constexpr(cute.size(acc_pv_mn, mode=[1])):
                    acc_pv_mn[i, j] *= correction
                local_sum = a_sum[i]
            a_sum[i] = local_sum + acc_qk_mn[i, None].load().reduce(
                cute.ReductionOp.ADD, cutlass.Float32.zero, 0
            )
        return s_max, a_sum

    @cute.jit
    def reduce_raw_sum(self, a_sum, acc_pv, tiled_mma_pv):
        acc_pv_mn = cute.make_tensor(
            acc_pv.iterator, self.layout_acc_mn(tiled_mma_pv, acc_pv.layout)
        )
        reduction_target = self.reduction_target_n(tiled_mma_pv)
        for r in cutlass.range_constexpr(cute.rank(reduction_target)):
            for i in cutlass.range_constexpr(cute.size(acc_pv_mn, mode=[0])):
                a_sum[i] = cute.arch.warp_reduction_sum(
                    a_sum[i], threads_in_group=reduction_target.shape[r]
                )

    @cute.jit
    def reduction_target_n(self, tiled_mma):
        separated = self.layout_separate(
            tiled_mma.shape_mnk[0],
            cute.make_layout(tiled_mma.tv_layout_C.shape[0]),
            tiled_mma.tv_layout_C.stride[0],
        )
        return separated[1]

    @staticmethod
    def convert_c_layout_to_a_layout(c_layout, a_layout):
        return cute.make_layout(
            (
                a_layout,
                c_layout.shape[1],
                (c_layout.shape[2], cute.size(c_layout, mode=[0]) // cute.size(a_layout)),
            ),
            stride=(
                c_layout.stride[0],
                c_layout.stride[1],
                (
                    c_layout.stride[2],
                    cute.size(a_layout, mode=[2]) * c_layout.stride[0][2],
                ),
            ),
        )

    @cute.jit
    def make_acc_into_op(self, acc, operand_layout_tv, element_type):
        operand = cute.make_rmem_tensor_like(
            self.convert_c_layout_to_a_layout(acc.layout, operand_layout_tv.shape[1]),
            element_type,
        )
        operand_as_acc = cute.make_tensor(operand.iterator, acc.layout)
        operand_as_acc.store(acc.load().to(element_type))
        return operand

    @staticmethod
    def layout_separate(thr, src, ref):
        less = cute.make_layout(())
        greater_equal = cute.make_layout(())
        for index, value in enumerate(ref):
            if cutlass.const_expr(value < thr):
                less = cute.append(less, src[index])
            else:
                greater_equal = cute.append(greater_equal, src[index])
        if cutlass.const_expr(cute.rank(less) == 1):
            return cute.append(less, greater_equal)
        return cute.append(cute.append(cute.make_layout(()), less), greater_equal)

    @staticmethod
    @cute.jit
    def gemm_zero_acc(tiled_mma, a, b, c):
        rank_a = cute.rank(a)
        rank_b = cute.rank(b)
        rank_c = cute.rank(c)
        if cutlass.const_expr(rank_a == 2 and rank_b == 2 and rank_c == 1):
            for k_block_idx in range(cute.size(a, mode=[1]), unroll_full=True):
                tiled_mma.set(
                    cute.nvgpu.warpgroup.Field.ACCUMULATE, k_block_idx != 0
                )
                cute.gemm(
                    tiled_mma,
                    c,
                    a[None, k_block_idx],
                    b[None, k_block_idx],
                    c,
                )
        elif cutlass.const_expr(rank_a == 3 and rank_b == 3 and rank_c == 3):
            for k_block_idx in range(cute.size(a, mode=[2]), unroll_full=True):
                tiled_mma.set(
                    cute.nvgpu.warpgroup.Field.ACCUMULATE, k_block_idx != 0
                )
                cute.gemm(
                    tiled_mma,
                    c,
                    a[None, None, k_block_idx],
                    b[None, None, k_block_idx],
                    c,
                )
        else:
            assert 0

    @cute.jit
    def layout_acc_mn(self, tiled_mma, acc):
        separated = self.layout_separate(
            tiled_mma.shape_mnk[0], acc[0], tiled_mma.tv_layout_C.stride[1]
        )
        value_m = separated[0]
        value_n = separated[1]
        if cutlass.const_expr(cute.rank(value_m) == 1):
            value_m1 = cute.append(value_m, acc[1])
        else:
            value_m1 = cute.append(cute.append(cute.make_layout(()), value_m), acc[1])
        if cutlass.const_expr(cute.rank(value_n) == 1):
            value_n1 = cute.append(value_n, acc[2])
        else:
            value_n1 = cute.append(cute.append(cute.make_layout(()), value_n), acc[2])
        if cutlass.const_expr(cute.rank(value_m1) == 1):
            return cute.append(value_m1, value_n1)
        return cute.append(cute.append(cute.make_layout(()), value_m1), value_n1)

    def make_and_init_load_q_pipeline(self, barrier_ptr):
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_warps_per_warp_group
        )
        return pipeline.PipelineTmaAsync.create(
            barrier_storage=barrier_ptr,
            num_stages=self.q_stage,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=self.tma_copy_q_bytes,
            defer_sync=True,
        ).make_participants()

    def make_and_init_load_kv_pipeline(self, barrier_ptr):
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_warps_per_warp_group
        )
        return pipeline.PipelineTmaAsync.create(
            barrier_storage=barrier_ptr,
            num_stages=self.kv_stage,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=self.tma_copy_kv_bytes,
            defer_sync=True,
        ).make_participants()

    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor, smem_layout_staged, smem_tile, mcast_dim
    ):
        operation = (
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
        )
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        return cute.nvgpu.cpasync.make_tiled_tma_atom(
            operation,
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=mcast_dim,
        )


class HopperDecode128KRawForwardK2V2(HopperDecode128KRawForward):
    """M64 raw kernel with independent two-stage K and V TMA pipelines."""

    def __init__(self, qk_acc_dtype, pv_acc_dtype, mma_tiler, num_splits):
        super().__init__(qk_acc_dtype, pv_acc_dtype, mma_tiler, 2, num_splits)
        self.separate_kv = True
        self.overlap = False

    @cute.kernel
    def kernel(
        self,
        qk_tiled_mma: cute.TiledMma,
        pv_tiled_mma: cute.TiledMma,
        tma_atom_q: cute.CopyAtom,
        mQ_qdl: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        mK_kdl: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        mV_dkl: cute.Tensor,
        mO_qdl: cute.Tensor,
        mStats: cute.Tensor,
        scale_softmax_log2: cutlass.Float32,
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        split_idx, kv_head_idx, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group
        )

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        load_q_producer, load_q_consumer = self.make_and_init_load_q_pipeline(
            storage.load_q_mbar_ptr.data_ptr()
        )
        load_k_producer, load_k_consumer = self.make_and_init_load_kv_pipeline(
            storage.load_k_mbar_ptr.data_ptr()
        )
        load_v_producer, load_v_consumer = self.make_and_init_load_kv_pipeline(
            storage.load_v_mbar_ptr.data_ptr()
        )

        sQ = storage.sQ.get_tensor(
            q_smem_layout_staged.outer, swizzle=q_smem_layout_staged.inner
        )
        sK = storage.sK.get_tensor(
            k_smem_layout_staged.outer, swizzle=k_smem_layout_staged.inner
        )
        sV = storage.sV.get_tensor(
            v_smem_layout_staged.outer, swizzle=v_smem_layout_staged.inner
        )

        seqlen_q = mQ_qdl.shape[0]
        seqlen_k = mK_kdl.shape[0]
        num_n_tiles = cute.ceil_div(seqlen_k, self.qk_mma_tiler[1])
        tile_begin = split_idx * num_n_tiles // self.num_splits
        tile_end = (split_idx + 1) * num_n_tiles // self.num_splits
        split_k_tiles = tile_end - tile_begin

        gQ_qdl = cute.flat_divide(
            mQ_qdl, cute.select(self.qk_mma_tiler, mode=[0, 2])
        )
        qk_thr_mma = qk_tiled_mma.get_slice(tidx)
        tSgQ_qdl = qk_thr_mma.partition_A(gQ_qdl)
        tQsQ, tQgQ_qdl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_q,
            0,
            cute.make_layout(1),
            cute.group_modes(sQ, 0, 2),
            cute.group_modes(tSgQ_qdl, 0, 3),
        )

        gK_kdl = cute.flat_divide(
            mK_kdl, cute.select(self.qk_mma_tiler, mode=[1, 2])
        )
        tSgK_kdl = qk_thr_mma.partition_B(gK_kdl)
        tKsK, tKgK_kdl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_k,
            0,
            cute.make_layout(1),
            cute.group_modes(sK, 0, 2),
            cute.group_modes(tSgK_kdl, 0, 3),
        )

        gV_dkl = cute.flat_divide(
            mV_dkl, cute.select(self.pv_mma_tiler, mode=[1, 2])
        )
        pv_thr_mma = pv_tiled_mma.get_slice(tidx)
        tSgV_dkl = pv_thr_mma.partition_B(gV_dkl)
        tVsV, tVgV_dkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_v,
            0,
            cute.make_layout(1),
            cute.group_modes(sV, 0, 2),
            cute.group_modes(tSgV_dkl, 0, 3),
        )

        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)

        if warp_group_idx == self.load_warp_group_id:
            cute.arch.setmaxregister_decrease(self.num_regs_load)
            producer_warp_role = warp_idx % self.num_warps_per_warp_group
            if producer_warp_role == self.producer_warp_loadkv_id:
                tQgQ = tQgQ_qdl[(None, None, 0, kv_head_idx)]
                q_handle = load_q_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_q,
                    tQgQ[(None, 0)],
                    tQsQ[(None, q_handle.index)],
                    tma_bar_ptr=q_handle.barrier,
                )

                tKgK = tKgK_kdl[(None, None, 0, kv_head_idx)]
                tVgV = tVgV_dkl[(None, 0, None, kv_head_idx)]
                k_index = tile_begin
                tile_count = split_k_tiles
                while tile_count > 0:
                    k_handle = load_k_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_k,
                        tKgK[(None, k_index)],
                        tKsK[(None, k_handle.index)],
                        tma_bar_ptr=k_handle.barrier,
                    )
                    v_handle = load_v_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_v,
                        tVgV[(None, k_index)],
                        tVsV[(None, v_handle.index)],
                        tma_bar_ptr=v_handle.barrier,
                    )
                    k_index += 1
                    tile_count -= 1

        if warp_group_idx == self.compute_warp_group_id:
            cute.arch.setmaxregister_increase(self.num_regs_mma)
            wg_coord = (0, 0, kv_head_idx)

            tSsQ = qk_thr_mma.partition_A(sQ)
            tSsK = qk_thr_mma.partition_B(sK)
            tSrQ = qk_thr_mma.make_fragment_A(tSsQ)
            tSrK = qk_thr_mma.make_fragment_B(tSsK)
            tOsV = pv_thr_mma.partition_B(sV)
            tOrV = pv_thr_mma.make_fragment_B(tOsV)
            q_handle = load_q_consumer.wait()

            cP = cute.make_identity_tensor((mQ_qdl.shape[0], seqlen_k))
            gPcP = cute.local_tile(cP, self.qk_mma_tiler[:2], (None, None))
            ptPcP = qk_thr_mma.partition_C(gPcP)
            pv_acc_shape = pv_thr_mma.partition_shape_C(
                (self.pv_mma_tiler[0], self.pv_mma_tiler[1])
            )
            acc_pv = pv_thr_mma.make_fragment_C(pv_acc_shape)
            qk_acc_shape = qk_thr_mma.partition_shape_C(
                (self.qk_mma_tiler[0], self.qk_mma_tiler[1])
            )
            row_fragment_layout = cute.make_layout(
                cute.size(self.layout_acc_mn(pv_tiled_mma, acc_pv.layout), mode=[0])
            )
            s_max = cute.make_rmem_tensor_like(row_fragment_layout, cutlass.Float32)
            a_sum = cute.make_rmem_tensor_like(row_fragment_layout, cutlass.Float32)

            global_tile_idx = tile_begin
            tPcP = cute.slice_(ptPcP, (None, None, None, 0, global_tile_idx))
            kv_offset = global_tile_idx + 1
            acc_qk = qk_thr_mma.make_fragment_C(qk_acc_shape)
            k_handle = load_k_consumer.wait_and_advance()

            cute.nvgpu.warpgroup.fence()
            self.gemm_zero_acc(
                qk_tiled_mma,
                tSrQ[(None, None, None, q_handle.index)],
                tSrK[(None, None, None, k_handle.index)],
                acc_qk,
            )
            cute.nvgpu.warpgroup.commit_group()
            v_token = load_v_consumer.try_wait()
            cute.nvgpu.warpgroup.wait_group(0)
            k_handle.release()

            if cutlass.const_expr(QWEN_CONTEXT % self.qk_mma_tiler[1] != 0):
                self.mask_fixed_residue(acc_qk, tPcP, global_tile_idx)
            s_max, a_sum = self.softmax_step(
                acc_qk,
                qk_tiled_mma,
                s_max,
                a_sum,
                acc_qk,
                qk_tiled_mma,
                scale_softmax_log2,
                True,
            )
            acc_qk_fixed = self.make_acc_into_op(
                acc_qk, pv_tiled_mma.tv_layout_A, self.q_dtype
            )
            if cutlass.const_expr(self.overlap):
                acc_pv.fill(0.0)
                load_k_consumer, load_v_consumer, kv_offset, s_max, a_sum = (
                    self.compute_overlap_k2v2(
                        split_k_tiles - 1,
                        qk_thr_mma,
                        acc_pv,
                        acc_qk_fixed,
                        qk_tiled_mma,
                        pv_tiled_mma,
                        load_k_consumer,
                        load_v_consumer,
                        q_handle,
                        tSrQ,
                        tSrK,
                        s_max,
                        a_sum,
                        tOrV,
                        ptPcP,
                        wg_coord,
                        kv_offset,
                        scale_softmax_log2,
                        qk_acc_shape,
                    )
                )
            else:
                v_handle = load_v_consumer.wait_and_advance(v_token)
                cute.nvgpu.warpgroup.fence()
                self.gemm_zero_acc(
                    pv_tiled_mma,
                    acc_qk_fixed,
                    tOrV[(None, None, None, v_handle.index)],
                    acc_pv,
                )
                cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.wait_group(0)
                v_handle.release()

                load_k_consumer, load_v_consumer, kv_offset, s_max, a_sum = (
                    self.compute_serial_k2v2(
                        split_k_tiles - 1,
                        qk_thr_mma,
                        acc_pv,
                        qk_tiled_mma,
                        pv_tiled_mma,
                        load_k_consumer,
                        load_v_consumer,
                        q_handle,
                        tSrQ,
                        tSrK,
                        s_max,
                        a_sum,
                        tOrV,
                        ptPcP,
                        wg_coord,
                        kv_offset,
                        scale_softmax_log2,
                        qk_acc_shape,
                    )
                )
            cute.nvgpu.warpgroup.wait_group(0)
            self.reduce_raw_sum(a_sum, acc_pv, pv_tiled_mma)
            q_handle.release()

            thr_mma = pv_tiled_mma.get_slice(tidx)
            cD = cute.make_identity_tensor(
                (self.pv_mma_tiler[0], self.pv_mma_tiler[1])
            )
            tOcD = thr_mma.partition_C(cD)
            tOcD_mn = cute.make_tensor(
                tOcD.iterator, self.layout_acc_mn(pv_tiled_mma, tOcD.layout)
            )

            gStats_full = cute.local_tile(
                mStats, self.pv_mma_tiler[:2], (None, None, None, None, None)
            )
            for stat_idx in cutlass.range_constexpr(2):
                gStats = cute.slice_(
                    gStats_full,
                    (None, None, 0, 0, kv_head_idx, split_idx, stat_idx),
                )
                tOgStats = thr_mma.partition_C(gStats)
                tOgStats_mn = cute.make_tensor(
                    tOgStats.iterator,
                    self.layout_acc_mn(pv_tiled_mma, tOgStats.layout),
                )
                if tOcD[0][1] == 0:
                    for i in cutlass.range_constexpr(
                        cute.size(tOgStats_mn, mode=[0])
                    ):
                        if tOcD_mn[(i, 0)][0] < seqlen_q:
                            value = a_sum[i]
                            if stat_idx == 0:
                                value = s_max[i] * scale_softmax_log2
                            tOgStats_mn[(i, 0)] = value

            gD = cute.local_tile(
                mO_qdl,
                self.pv_mma_tiler[:2],
                (0, 0, kv_head_idx, split_idx),
            )
            tOgD = thr_mma.partition_C(gD)
            acc_pv_mn = cute.make_tensor(
                acc_pv.iterator, self.layout_acc_mn(pv_tiled_mma, acc_pv.layout)
            )
            tOgD_mn = cute.make_tensor(
                tOgD.iterator, self.layout_acc_mn(pv_tiled_mma, tOgD.layout)
            )
            for i in cutlass.range_constexpr(cute.size(acc_pv_mn, mode=[0])):
                if tOcD_mn[(i, 0)][0] < seqlen_q:
                    cute.autovec_copy(acc_pv_mn[i, None], tOgD_mn[i, None])
        return

    @cute.jit
    def compute_serial_k2v2(
        self,
        k_tile_count: cutlass.Int32,
        qk_thr_mma: cute.ThrMma,
        acc_pv: cute.ThrMma,
        qk_tiled_mma: cute.TiledMma,
        pv_tiled_mma: cute.TiledMma,
        load_k_consumer: pipeline.PipelineConsumer,
        load_v_consumer: pipeline.PipelineConsumer,
        q_handle: pipeline.PipelineConsumer.ImmutableResourceHandle,
        tSrQ: cute.Tensor,
        tSrK: cute.Tensor,
        s_max: cute.Tensor,
        a_sum: cute.Tensor,
        tOrV: cute.Tensor,
        ptPcP: cute.Tensor,
        wg_coord: tuple,
        kv_offset: cutlass.Int32,
        scale_softmax_log2: cutlass.Float32,
        qk_acc_shape: cute.Shape,
    ):
        while k_tile_count > 0:
            k_tile_count -= 1
            global_tile_idx = kv_offset
            tPcP = cute.slice_(
                ptPcP, (None, None, None, wg_coord[0], global_tile_idx)
            )
            kv_offset = global_tile_idx + 1
            acc_qk = qk_thr_mma.make_fragment_C(qk_acc_shape)
            k_handle = load_k_consumer.wait_and_advance()
            cute.nvgpu.warpgroup.fence()
            self.gemm_zero_acc(
                qk_tiled_mma,
                tSrQ[(None, None, None, q_handle.index)],
                tSrK[(None, None, None, k_handle.index)],
                acc_qk,
            )
            cute.nvgpu.warpgroup.commit_group()
            v_token = load_v_consumer.try_wait()
            cute.nvgpu.warpgroup.wait_group(0)
            k_handle.release()
            if cutlass.const_expr(QWEN_CONTEXT % self.qk_mma_tiler[1] != 0):
                self.mask_fixed_residue(acc_qk, tPcP, global_tile_idx)
            s_max, a_sum = self.softmax_step(
                acc_qk,
                qk_tiled_mma,
                s_max,
                a_sum,
                acc_pv,
                pv_tiled_mma,
                scale_softmax_log2,
                False,
            )
            acc_qk_fixed = self.make_acc_into_op(
                acc_qk, pv_tiled_mma.tv_layout_A, self.q_dtype
            )
            v_handle = load_v_consumer.wait_and_advance(v_token)
            cute.nvgpu.warpgroup.fence()
            pv_tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
            cute.gemm(
                pv_tiled_mma,
                acc_pv,
                acc_qk_fixed,
                tOrV[(None, None, None, v_handle.index)],
                acc_pv,
            )
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)
            v_handle.release()
        return load_k_consumer, load_v_consumer, kv_offset, s_max, a_sum


class HopperDecode128KRawForwardK2V2Overlap(HopperDecode128KRawForwardK2V2):
    """K2/V2 kernel overlapping QK(next) with PV(current) in one compute WG."""

    def __init__(self, qk_acc_dtype, pv_acc_dtype, mma_tiler, num_splits):
        super().__init__(qk_acc_dtype, pv_acc_dtype, mma_tiler, num_splits)
        self.overlap = True

    @cute.jit
    def compute_overlap_k2v2(
        self,
        k_tile_count: cutlass.Int32,
        qk_thr_mma: cute.ThrMma,
        acc_pv: cute.ThrMma,
        p_current: cute.Tensor,
        qk_tiled_mma: cute.TiledMma,
        pv_tiled_mma: cute.TiledMma,
        load_k_consumer: pipeline.PipelineConsumer,
        load_v_consumer: pipeline.PipelineConsumer,
        q_handle: pipeline.PipelineConsumer.ImmutableResourceHandle,
        tSrQ: cute.Tensor,
        tSrK: cute.Tensor,
        s_max: cute.Tensor,
        a_sum: cute.Tensor,
        tOrV: cute.Tensor,
        ptPcP: cute.Tensor,
        wg_coord: tuple,
        kv_offset: cutlass.Int32,
        scale_softmax_log2: cutlass.Float32,
        qk_acc_shape: cute.Shape,
    ):
        pv_tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
        k_tile_count -= 1
        (
            load_k_consumer,
            load_v_consumer,
            p_current,
            kv_offset,
            s_max,
            a_sum,
        ) = self.overlap_qk_pv_step(
            qk_thr_mma,
            acc_pv,
            p_current,
            qk_tiled_mma,
            pv_tiled_mma,
            load_k_consumer,
            load_v_consumer,
            q_handle,
            tSrQ,
            tSrK,
            s_max,
            a_sum,
            tOrV,
            ptPcP,
            wg_coord,
            kv_offset,
            scale_softmax_log2,
            qk_acc_shape,
        )

        while k_tile_count > 0:
            k_tile_count -= 1
            (
                load_k_consumer,
                load_v_consumer,
                p_current,
                kv_offset,
                s_max,
                a_sum,
            ) = self.overlap_qk_pv_step(
                qk_thr_mma,
                acc_pv,
                p_current,
                qk_tiled_mma,
                pv_tiled_mma,
                load_k_consumer,
                load_v_consumer,
                q_handle,
                tSrQ,
                tSrK,
                s_max,
                a_sum,
                tOrV,
                ptPcP,
                wg_coord,
                kv_offset,
                scale_softmax_log2,
                qk_acc_shape,
            )

        v_token = load_v_consumer.try_wait()
        v_handle = load_v_consumer.wait_and_advance(v_token)
        cute.nvgpu.warpgroup.fence()
        cute.gemm(
            pv_tiled_mma,
            acc_pv,
            p_current,
            tOrV[(None, None, None, v_handle.index)],
            acc_pv,
        )
        cute.nvgpu.warpgroup.commit_group()
        cute.nvgpu.warpgroup.wait_group(0)
        v_handle.release()
        return load_k_consumer, load_v_consumer, kv_offset, s_max, a_sum

    @cute.jit
    def overlap_qk_pv_step(
        self,
        qk_thr_mma: cute.ThrMma,
        acc_pv: cute.ThrMma,
        p_current: cute.Tensor,
        qk_tiled_mma: cute.TiledMma,
        pv_tiled_mma: cute.TiledMma,
        load_k_consumer: pipeline.PipelineConsumer,
        load_v_consumer: pipeline.PipelineConsumer,
        q_handle: pipeline.PipelineConsumer.ImmutableResourceHandle,
        tSrQ: cute.Tensor,
        tSrK: cute.Tensor,
        s_max: cute.Tensor,
        a_sum: cute.Tensor,
        tOrV: cute.Tensor,
        ptPcP: cute.Tensor,
        wg_coord: tuple,
        kv_offset: cutlass.Int32,
        scale_softmax_log2: cutlass.Float32,
        qk_acc_shape: cute.Shape,
    ):
        global_tile_idx = kv_offset
        tPcP = cute.slice_(
            ptPcP, (None, None, None, wg_coord[0], global_tile_idx)
        )
        kv_offset = global_tile_idx + 1
        acc_qk = qk_thr_mma.make_fragment_C(qk_acc_shape)
        k_handle = load_k_consumer.wait_and_advance()

        cute.nvgpu.warpgroup.fence()
        self.gemm_zero_acc(
            qk_tiled_mma,
            tSrQ[(None, None, None, q_handle.index)],
            tSrK[(None, None, None, k_handle.index)],
            acc_qk,
        )
        cute.nvgpu.warpgroup.commit_group()

        v_token = load_v_consumer.try_wait()
        v_handle = load_v_consumer.wait_and_advance(v_token)
        cute.nvgpu.warpgroup.fence()
        cute.gemm(
            pv_tiled_mma,
            acc_pv,
            p_current,
            tOrV[(None, None, None, v_handle.index)],
            acc_pv,
        )
        cute.nvgpu.warpgroup.commit_group()

        cute.nvgpu.warpgroup.wait_group(1)
        k_handle.release()
        if cutlass.const_expr(QWEN_CONTEXT % self.qk_mma_tiler[1] != 0):
            self.mask_fixed_residue(acc_qk, tPcP, global_tile_idx)
        s_max, a_sum, row_scale = self.softmax_overlap_step(
            acc_qk,
            qk_tiled_mma,
            s_max,
            a_sum,
            scale_softmax_log2,
        )

        cute.nvgpu.warpgroup.wait_group(0)
        v_handle.release()
        self.rescale_pv(acc_pv, pv_tiled_mma, row_scale)
        p_next = self.make_acc_into_op(
            acc_qk, pv_tiled_mma.tv_layout_A, self.q_dtype
        )
        return (
            load_k_consumer,
            load_v_consumer,
            p_next,
            kv_offset,
            s_max,
            a_sum,
        )

    @cute.jit
    def softmax_overlap_step(
        self,
        acc_qk: cute.ThrMma,
        tiled_mma_qk: cute.TiledMma,
        s_max: cute.Tensor,
        a_sum: cute.Tensor,
        scale_softmax_log2: cutlass.Float32,
    ):
        acc_qk_mn = cute.make_tensor(
            acc_qk.iterator, self.layout_acc_mn(tiled_mma_qk, acc_qk.layout)
        )
        reduction_target_qk = self.reduction_target_n(tiled_mma_qk)
        red_rank = cute.rank(reduction_target_qk)
        row_scale = cute.make_rmem_tensor_like(s_max, s_max._dtype)

        for i in cutlass.range_constexpr(cute.size(acc_qk_mn, mode=[0])):
            row_scale[i] = s_max[i]
            for j in cutlass.range_constexpr(cute.size(acc_qk_mn, mode=[1])):
                s_max[i] = cutlass.max(s_max[i], acc_qk_mn[i, j])
            for r in cutlass.range_constexpr(red_rank):
                s_max[i] = cute.arch.warp_reduction_max(
                    s_max[i], threads_in_group=reduction_target_qk.shape[r]
                )
            local_max = s_max[i]
            if local_max == -cutlass.Float32.inf:
                local_max = 0.0
            scale_max = scale_softmax_log2 * local_max
            for j in cutlass.range_constexpr(cute.size(acc_qk_mn, mode=[1])):
                acc_qk_mn[i, j] = cute.math.exp2(
                    scale_softmax_log2 * acc_qk_mn[i, j] - scale_max,
                    fastmath=True,
                )
            current_max = s_max[i]
            if current_max == -cutlass.Float32.inf:
                current_max = 0.0
            correction = cute.math.exp2(
                (row_scale[i] - current_max) * scale_softmax_log2,
                fastmath=True,
            )
            row_scale[i] = correction
            a_sum[i] *= correction
            a_sum[i] += acc_qk_mn[i, None].load().reduce(
                cute.ReductionOp.ADD, cutlass.Float32.zero, 0
            )
        return s_max, a_sum, row_scale

    @cute.jit
    def rescale_pv(
        self,
        acc_pv: cute.Tensor,
        tiled_mma_pv: cute.TiledMma,
        row_scale: cute.Tensor,
    ):
        acc_pv_mn = cute.make_tensor(
            acc_pv.iterator, self.layout_acc_mn(tiled_mma_pv, acc_pv.layout)
        )
        for i in cutlass.range_constexpr(cute.size(acc_pv_mn, mode=[0])):
            for j in cutlass.range_constexpr(cute.size(acc_pv_mn, mode=[1])):
                acc_pv_mn[i, j] *= row_scale[i]


class HopperDecode128KRawCombine:
    """One warp per query head combines raw split statistics and accumulators."""

    def __init__(self, num_splits):
        if not 1 <= num_splits <= 32:
            raise ValueError("num_splits must fit in one warp")
        self.num_splits = num_splits
        self.threads_per_cta = 32

    @cute.jit
    def __call__(
        self,
        o_raw: cute.Tensor,
        stats_raw: cute.Tensor,
        output: cute.Tensor,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(o_raw.element_type != cutlass.Float32):
            raise TypeError("raw O must be Float32")
        if cutlass.const_expr(stats_raw.element_type != cutlass.Float32):
            raise TypeError("raw stats must be Float32")
        if cutlass.const_expr(output.element_type != cutlass.BFloat16):
            raise TypeError("output must be BFloat16")

        load_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=32
        )
        store_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=16
        )
        thread_layout = cute.make_layout(self.threads_per_cta)
        value_layout = cute.make_layout(4)
        load_copy = cute.make_tiled_copy_tv(load_atom, thread_layout, value_layout)
        store_copy = cute.make_tiled_copy_tv(store_atom, thread_layout, value_layout)

        @cute.struct
        class SharedStorage:
            alpha: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.threads_per_cta], 128
            ]

        self.shared_storage = SharedStorage
        self.kernel(
            o_raw,
            stats_raw,
            output,
            load_atom,
            store_atom,
            load_copy,
            store_copy,
        ).launch(
            grid=(HEAD_RATIO, QWEN_KV_HEADS, 1),
            block=(self.threads_per_cta, 1, 1),
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        o_raw: cute.Tensor,
        stats_raw: cute.Tensor,
        output: cute.Tensor,
        load_atom: cute.CopyAtom,
        store_atom: cute.CopyAtom,
        load_copy: cute.TiledCopy,
        store_copy: cute.TiledCopy,
    ):
        lane, _, _ = cute.arch.thread_idx()
        ratio_idx, kv_head_idx, _ = cute.arch.block_idx()
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        alpha_smem = storage.alpha.get_tensor(cute.make_layout(self.threads_per_cta))

        split_max = -cutlass.Float32.inf
        split_sum = 0.0
        if lane < self.num_splits:
            split_max = stats_raw[kv_head_idx, ratio_idx, lane, 0]
            split_sum = stats_raw[kv_head_idx, ratio_idx, lane, 1]
        global_max = cute.arch.warp_reduction_max(split_max, threads_in_group=32)
        finite_max = global_max
        if global_max == -cutlass.Float32.inf:
            finite_max = 0.0
        alpha = 0.0
        if lane < self.num_splits:
            alpha = cute.math.exp2(split_max - finite_max, fastmath=True)
        denominator_term = alpha * split_sum
        denominator = cute.arch.warp_reduction_sum(
            denominator_term, threads_in_group=32
        )
        if denominator == 0.0 or denominator != denominator:
            alpha = 0.0
        alpha_smem[lane] = alpha
        cute.arch.sync_threads()

        load_thr = load_copy.get_slice(lane)
        first_partial = o_raw[0, kv_head_idx, ratio_idx, None]
        thread_partial = load_thr.partition_S(first_partial)
        thread_partial_tile = thread_partial[None, 0]
        fragment_partial = cute.make_fragment_like(thread_partial_tile)
        fragment_acc = cute.make_rmem_tensor_like(thread_partial_tile, cutlass.Float32)
        fragment_acc.fill(0.0)
        for split_idx in cutlass.range_constexpr(self.num_splits):
            partial = o_raw[split_idx, kv_head_idx, ratio_idx, None]
            thread_partial = load_thr.partition_S(partial)
            cute.copy(load_atom, thread_partial[None, 0], fragment_partial)
            fragment_acc.store(
                fragment_acc.load() + alpha_smem[split_idx] * fragment_partial.load()
            )
        if denominator != 0.0 and denominator == denominator:
            fragment_acc.store(fragment_acc.load() / denominator)

        head_idx = kv_head_idx * HEAD_RATIO + ratio_idx
        output_head = output[head_idx, None]
        store_thr = store_copy.get_slice(lane)
        thread_output = store_thr.partition_D(output_head)
        thread_output_tile = thread_output[None, 0]
        fragment_output = cute.make_fragment_like(thread_output_tile)
        fragment_output.store(fragment_acc.load().to(cutlass.BFloat16))
        cute.copy(store_atom, fragment_output, thread_output_tile)


def _resolve_config(config):
    if config == "auto":
        config = AUTO_CONFIG
    try:
        values = CONFIGS[config]
    except KeyError as exc:
        choices = ", ".join(("auto", *CONFIGS))
        raise ValueError(f"unknown config {config!r}; expected one of {choices}") from exc
    return config, values


def _normalize_sm_scale(sm_scale):
    import torch

    if sm_scale is None:
        return 1.0 / math.sqrt(QWEN_HEAD_DIM)
    if isinstance(sm_scale, (bool, torch.Tensor)):
        raise TypeError("sm_scale must be a positive finite real number or None")
    try:
        value = float(sm_scale)
    except (TypeError, ValueError) as exc:
        raise TypeError("sm_scale must be a positive finite real number or None") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("sm_scale must be a positive finite real number")
    if value > torch.finfo(torch.float32).max / LOG2_E:
        raise ValueError("sm_scale is not representable after log2(e) scaling")
    return value


def _validate_inputs(q, k, v, causal):
    import torch

    if not isinstance(causal, bool):
        raise TypeError("causal must be a bool")
    tensors = {"q": q, "k": k, "v": v}
    expected = {
        "q": (QWEN_BATCH, QWEN_QUERY_HEADS, 1, QWEN_HEAD_DIM),
        "k": (QWEN_BATCH, QWEN_KV_HEADS, QWEN_CONTEXT, QWEN_HEAD_DIM),
        "v": (QWEN_BATCH, QWEN_KV_HEADS, QWEN_CONTEXT, QWEN_HEAD_DIM),
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tuple(tensor.shape) != expected[name]:
            raise ValueError(f"{name} must have shape {expected[name]}, got {tuple(tensor.shape)}")
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"{name} must use torch.bfloat16, got {tensor.dtype}")
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if tensor.device != q.device:
            raise ValueError("q, k, and v must be on the same CUDA device")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous BHSD")
        if tensor.data_ptr() % 16:
            raise ValueError(f"{name} must be 16-byte aligned")
        if tensor.requires_grad:
            raise ValueError("qwen3_decode_attention is forward-only")

    properties = torch.cuda.get_device_properties(q.device)
    capability = torch.cuda.get_device_capability(q.device)
    if capability != (9, 0) or properties.multi_processor_count != 78:
        raise RuntimeError(
            "the fixed kernel requires H20 SM90 with 78 SMs; got "
            f"{properties.name}, sm={capability}, sms={properties.multi_processor_count}"
        )
    if "H20" not in properties.name.upper():
        raise RuntimeError(f"the fixed kernel requires NVIDIA H20, got {properties.name}")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "CUDA Graph capture is not supported by this Decode-only API; "
            "prepare fixed workspace/output separately"
        )


def _get_workspace(device, stream_handle, config_name, num_splits):
    import torch

    key = (
        device.index,
        int(stream_handle),
        config_name,
        num_splits,
        _WORKSPACE_LAYOUT_VERSION,
    )
    with _WORKSPACE_LOCK:
        workspace = _WORKSPACE_CACHE.get(key)
        if workspace is None:
            if len(_WORKSPACE_CACHE) >= _WORKSPACE_CACHE_LIMIT:
                del _WORKSPACE_CACHE[next(iter(_WORKSPACE_CACHE))]
            workspace = {
                "o": torch.empty(
                    (num_splits, QWEN_KV_HEADS, HEAD_RATIO, QWEN_HEAD_DIM),
                    dtype=torch.float32,
                    device=device,
                ),
                "stats": torch.empty(
                    (QWEN_KV_HEADS, HEAD_RATIO, num_splits, 2),
                    dtype=torch.float32,
                    device=device,
                ),
            }
            _WORKSPACE_CACHE[key] = workspace
    return workspace


def _as_cute_tensor(tensor, element_type, leading_dim):
    result = from_dlpack(tensor, assumed_align=16)
    result.element_type = element_type
    return result.mark_layout_dynamic(leading_dim=leading_dim)


def _launch_partial(q, k, v, workspace, sm_scale, config_name, values):
    import torch

    q_kernel = q.view(QWEN_KV_HEADS, HEAD_RATIO, QWEN_HEAD_DIM).permute(1, 2, 0)
    k_kernel = k.view(QWEN_KV_HEADS, QWEN_CONTEXT, QWEN_HEAD_DIM).permute(1, 2, 0)
    v_kernel = v.view(QWEN_KV_HEADS, QWEN_CONTEXT, QWEN_HEAD_DIM).permute(2, 1, 0)
    o_kernel = workspace["o"].permute(2, 3, 1, 0)
    stats_kernel = workspace["stats"].permute(1, 0, 2, 3)

    q_tensor = _as_cute_tensor(q_kernel, cutlass.BFloat16, 1)
    k_tensor = _as_cute_tensor(k_kernel, cutlass.BFloat16, 1)
    v_tensor = _as_cute_tensor(v_kernel, cutlass.BFloat16, 0)
    o_tensor = _as_cute_tensor(o_kernel, cutlass.Float32, 1)
    stats_tensor = _as_cute_tensor(stats_kernel, cutlass.Float32, 3)
    torch_stream = torch.cuda.current_stream(q.device)
    stream = cuda.CUstream(torch_stream.cuda_stream)
    capability = torch.cuda.get_device_capability(q.device)
    key = (
        _PARTIAL_KERNEL_VERSION,
        q.device.index,
        capability,
        config_name,
        values["splits"],
        "bf16-fp32-raw",
    )
    compiled = _PARTIAL_KERNEL_CACHE.get(key)
    scale_log2 = sm_scale * LOG2_E
    if compiled is None:
        with _COMPILE_LOCK:
            compiled = _PARTIAL_KERNEL_CACHE.get(key)
            if compiled is None:
                if values.get("kernel") == "k2v2-overlap":
                    operation = HopperDecode128KRawForwardK2V2Overlap(
                        cutlass.Float32,
                        cutlass.Float32,
                        (values["block_m"], values["block_n"], QWEN_HEAD_DIM),
                        values["splits"],
                    )
                elif values.get("kernel") == "k2v2":
                    operation = HopperDecode128KRawForwardK2V2(
                        cutlass.Float32,
                        cutlass.Float32,
                        (values["block_m"], values["block_n"], QWEN_HEAD_DIM),
                        values["splits"],
                    )
                else:
                    operation = HopperDecode128KRawForward(
                        cutlass.Float32,
                        cutlass.Float32,
                        (values["block_m"], values["block_n"], QWEN_HEAD_DIM),
                        values["kv_stage"],
                        values["splits"],
                        values.get("num_regs_load", 24),
                        values.get("num_regs_mma", 240),
                    )
                compiled = cute.compile(
                    operation,
                    q_tensor,
                    k_tensor,
                    v_tensor,
                    o_tensor,
                    stats_tensor,
                    scale_log2,
                    stream,
                )
                _PARTIAL_KERNEL_CACHE[key] = compiled
    compiled(
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        stats_tensor,
        scale_log2,
        stream,
    )
    for tensor in (q, k, v, workspace["o"], workspace["stats"]):
        tensor.record_stream(torch_stream)


def _launch_combine(workspace, output, config_name, values):
    import torch

    o_tensor = _as_cute_tensor(workspace["o"], cutlass.Float32, 3)
    stats_tensor = _as_cute_tensor(workspace["stats"], cutlass.Float32, 3)
    output_tensor = _as_cute_tensor(
        output.view(QWEN_QUERY_HEADS, QWEN_HEAD_DIM), cutlass.BFloat16, 1
    )
    torch_stream = torch.cuda.current_stream(output.device)
    stream = cuda.CUstream(torch_stream.cuda_stream)
    capability = torch.cuda.get_device_capability(output.device)
    key = (
        _COMBINE_KERNEL_VERSION,
        output.device.index,
        capability,
        values["splits"],
        "fp32-raw-bf16",
    )
    compiled = _COMBINE_KERNEL_CACHE.get(key)
    if compiled is None:
        with _COMPILE_LOCK:
            compiled = _COMBINE_KERNEL_CACHE.get(key)
            if compiled is None:
                operation = HopperDecode128KRawCombine(values["splits"])
                compiled = cute.compile(
                    operation, o_tensor, stats_tensor, output_tensor, stream
                )
                _COMBINE_KERNEL_CACHE[key] = compiled
    compiled(o_tensor, stats_tensor, output_tensor, stream)
    for tensor in (workspace["o"], workspace["stats"], output):
        tensor.record_stream(torch_stream)


def _run_decode(q, k, v, sm_scale, config_name, values, return_workspace=False):
    import torch

    torch_stream = torch.cuda.current_stream(q.device)
    workspace = _get_workspace(
        q.device, torch_stream.cuda_stream, config_name, values["splits"]
    )
    _launch_partial(q, k, v, workspace, sm_scale, config_name, values)
    output = torch.empty_like(q)
    _launch_combine(workspace, output, config_name, values)
    if return_workspace:
        return output, workspace
    return output


def qwen3_decode_attention(q, k, v, *, causal=True, sm_scale=None, config="auto"):
    """Run the fixed H20 128K forward-only Decode kernel.

    ``causal=True`` and ``causal=False`` are identical for the latest-token
    contract: both attend all 131072 cached keys.
    """

    _validate_inputs(q, k, v, causal)
    scale = _normalize_sm_scale(sm_scale)
    config_name, values = _resolve_config(config)
    return _run_decode(q, k, v, scale, config_name, values)


def qwen3_attention(q, k, v, *, causal=True, sm_scale=None, config="auto"):
    return qwen3_decode_attention(
        q, k, v, causal=causal, sm_scale=sm_scale, config=config
    )


attention = qwen3_attention


def _grouped_reference(q, k, v, sm_scale):
    import torch

    q_grouped = q.float().view(
        QWEN_BATCH, QWEN_KV_HEADS, HEAD_RATIO, 1, QWEN_HEAD_DIM
    )
    scores = torch.einsum("bhgqd,bhkd->bhgqk", q_grouped, k.float())
    probabilities = torch.softmax(scores * sm_scale, dim=-1)
    output = torch.einsum("bhgqk,bhkd->bhgqd", probabilities, v.float())
    return output.reshape(QWEN_BATCH, QWEN_QUERY_HEADS, 1, QWEN_HEAD_DIM)


def _make_inputs(seed):
    import torch

    torch.manual_seed(seed)
    device = torch.device("cuda", torch.cuda.current_device())
    q = torch.randn(
        (QWEN_BATCH, QWEN_QUERY_HEADS, 1, QWEN_HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    k = torch.randn(
        (QWEN_BATCH, QWEN_KV_HEADS, QWEN_CONTEXT, QWEN_HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    v = torch.randn_like(k)
    return q, k, v


def _error_metrics(actual, expected):
    error = (actual.float() - expected.float()).abs()
    return error.max().item(), error.mean().item()


def _time_cuda(invoke, warmup, iterations):
    import torch

    for _ in range(warmup):
        output = invoke()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = invoke()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    samples.sort()
    return statistics.median(samples), samples, output


def _effective_kv_gbps(time_us):
    logical_bytes = 2 * QWEN_KV_HEADS * QWEN_CONTEXT * QWEN_HEAD_DIM * 2
    return logical_bytes / (time_us * 1e-6) / 1e9


def _run_correctness(args):
    import torch

    q, k, v = _make_inputs(args.seed)
    for scale in args.scales:
        reference = _grouped_reference(q, k, v, scale)
        for causal in (False, True):
            output = qwen3_decode_attention(
                q, k, v, causal=causal, sm_scale=scale, config=args.config
            )
            torch.cuda.synchronize()
            max_abs, mean_abs = _error_metrics(output, reference)
            print(
                f"correctness causal={causal} scale={scale:.9g} "
                f"max_abs={max_abs:.6g} mean_abs={mean_abs:.6g}"
            )
            torch.testing.assert_close(
                output.float(), reference, atol=args.atol, rtol=args.rtol
            )
    print("full 128K grouped-reference correctness passed")


def _run_partial_compare(args):
    import torch
    import implement_attention_optimized3 as optimized3

    q, k, v = _make_inputs(args.seed)
    scale = args.scales[0]
    config_name, values = _resolve_config(args.config)
    output, workspace = _run_decode(
        q, k, v, scale, config_name, values, return_workspace=True
    )
    p2_output = optimized3.qwen3_decode_attention(
        q, k, v, causal=True, sm_scale=scale, num_splits=9
    )
    torch.cuda.synchronize()
    p2_workspaces = list(optimized3._QWEN3_DECODE_FIXED_WORKSPACE_CACHE.values())
    if not p2_workspaces:
        raise RuntimeError("optimized3 P2 workspace was not populated")
    p2_workspace = p2_workspaces[-1]

    raw_o = workspace["o"]
    raw_stats = workspace["stats"].permute(2, 0, 1, 3)
    raw_sum = raw_stats[..., 1]
    normalized = raw_o / raw_sum.unsqueeze(-1)
    raw_lse = (raw_stats[..., 0] + torch.log2(raw_sum)) * LN2
    partial_max, partial_mean = _error_metrics(normalized, p2_workspace["o"])
    lse_max, lse_mean = _error_metrics(raw_lse, p2_workspace["lse"])
    output_max, output_mean = _error_metrics(output, p2_output)
    print(
        f"P2 partial normalized max_abs={partial_max:.6g} mean_abs={partial_mean:.6g}"
    )
    print(f"P2 partial LSE max_abs={lse_max:.6g} mean_abs={lse_mean:.6g}")
    print(f"P2 final max_abs={output_max:.6g} mean_abs={output_mean:.6g}")
    torch.testing.assert_close(normalized, p2_workspace["o"], atol=args.atol, rtol=args.rtol)
    torch.testing.assert_close(raw_lse, p2_workspace["lse"], atol=args.atol, rtol=args.rtol)
    torch.testing.assert_close(output.float(), p2_output.float(), atol=args.atol, rtol=args.rtol)


def _load_fa3_func():
    hopper_path = "/dockerdata/linqihao/flash-attention/hopper"
    if hopper_path not in sys.path:
        sys.path.insert(0, hopper_path)
    from flash_attn_interface import flash_attn_func

    return flash_attn_func


def _fa3_invoke(flash_attn_func, q_bshd, k_bshd, v_bshd, sm_scale):
    result = flash_attn_func(
        q_bshd,
        k_bshd,
        v_bshd,
        softmax_scale=sm_scale,
        causal=False,
        num_splits=0,
        pack_gqa=None,
    )
    return result[0] if isinstance(result, tuple) else result


def _percentile(samples, fraction):
    ordered = sorted(samples)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _time_cuda_interleaved(invokes, warmup, iterations, rounds):
    import torch

    names = tuple(invokes)
    outputs = {}
    for warmup_idx in range(warmup):
        offset = warmup_idx % len(names)
        for name in names[offset:] + names[:offset]:
            outputs[name] = invokes[name]()
    torch.cuda.synchronize()

    all_rounds = []
    for round_idx in range(rounds):
        samples = {name: [] for name in names}
        for sample_idx in range(iterations):
            offset = (round_idx + sample_idx) % len(names)
            for name in names[offset:] + names[:offset]:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                outputs[name] = invokes[name]()
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end) * 1000.0)
        all_rounds.append(
            {
                name: {
                    "median": statistics.median(values),
                    "p95": _percentile(values, 0.95),
                }
                for name, values in samples.items()
            }
        )
    return all_rounds, outputs


def _run_benchmark(args):
    import torch
    import implement_attention_optimized3 as optimized3

    q, k, v = _make_inputs(args.seed)
    scale = args.scales[0]
    requested_configs = args.benchmark_configs or (args.config,)
    config_names = []
    for requested in requested_configs:
        config_name, _ = _resolve_config(requested)
        if config_name not in config_names:
            config_names.append(config_name)
    if AUTO_CONFIG not in config_names:
        config_names.append(AUTO_CONFIG)

    invokes = {}
    for config_name in config_names:
        invokes[config_name] = lambda name=config_name: qwen3_decode_attention(
            q, k, v, causal=True, sm_scale=scale, config=name
        )
    invokes["optimized3-P2"] = lambda: optimized3.qwen3_decode_attention(
        q, k, v, causal=True, sm_scale=scale, num_splits=9
    )

    flash_attn_func = _load_fa3_func()
    q_bshd = q.transpose(1, 2).contiguous()
    k_bshd = k.transpose(1, 2).contiguous()
    v_bshd = v.transpose(1, 2).contiguous()
    invokes["FA3-auto"] = lambda: _fa3_invoke(
        flash_attn_func, q_bshd, k_bshd, v_bshd, scale
    )

    round_results, outputs = _time_cuda_interleaved(
        invokes, args.warmup, args.iterations, args.rounds
    )
    for round_idx, result in enumerate(round_results, 1):
        print(f"benchmark round={round_idx} samples={args.iterations}")
        for name, metrics in result.items():
            print(
                f"  {name}: median={metrics['median']:.3f} us "
                f"p95={metrics['p95']:.3f} us"
            )

    median_of_medians = {
        name: statistics.median([result[name]["median"] for result in round_results])
        for name in invokes
    }
    auto_us = median_of_medians[AUTO_CONFIG]
    print("median-of-medians")
    for name, median_us in median_of_medians.items():
        suffix = f" speedup_vs_auto={auto_us / median_us:.4f}x"
        if name in CONFIGS:
            suffix += f" effective_kv={_effective_kv_gbps(median_us):.1f} GB/s"
        print(f"  {name}: {median_us:.3f} us{suffix}")

    auto_output = outputs[AUTO_CONFIG]
    max_errors = []
    for name, output in outputs.items():
        comparable = output.transpose(1, 2) if name == "FA3-auto" else output
        max_abs, mean_abs = _error_metrics(comparable, auto_output)
        max_errors.append(max_abs)
        print(f"  compare {name}: max_abs={max_abs:.6g} mean_abs={mean_abs:.6g}")
    if max(max_errors) > args.atol:
        raise AssertionError("benchmark comparison exceeded --atol")


def _run_profile(args):
    import torch
    from torch.autograd import DeviceType

    q, k, v = _make_inputs(args.seed)
    scale = args.scales[0]
    qwen3_decode_attention(q, k, v, sm_scale=scale, config=args.config)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        qwen3_decode_attention(q, k, v, sm_scale=scale, config=args.config)
    torch.cuda.synchronize()
    cuda_events = [event for event in prof.events() if event.device_type == DeviceType.CUDA]
    print(f"profile CUDA kernel events={len(cuda_events)}")
    for event in cuda_events:
        print(f"  {event.name}: {event.device_time_total:.3f} us")
    if len(cuda_events) != 2:
        raise AssertionError(f"expected exactly 2 CUDA kernels, observed {len(cuda_events)}")


def _run_smoke(args):
    import torch

    q, k, v = _make_inputs(args.seed)
    output = qwen3_decode_attention(
        q, k, v, causal=True, sm_scale=args.scales[0], config=args.config
    )
    torch.cuda.synchronize()
    if not torch.isfinite(output).all():
        raise AssertionError("smoke output contains non-finite values")
    print(f"compile smoke passed: output={tuple(output.shape)} dtype={output.dtype}")


def _parse_scales(value):
    try:
        values = tuple(float(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated floats") from exc
    if not values or any(not math.isfinite(item) or item <= 0 for item in values):
        raise argparse.ArgumentTypeError("scales must be positive finite values")
    return values


def _parse_benchmark_configs(value):
    names = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = [name for name in names if name != "auto" and name not in CONFIGS]
    if not names or unknown:
        choices = ", ".join(("auto", *CONFIGS))
        raise argparse.ArgumentTypeError(
            f"expected comma-separated configs from {choices}; unknown={unknown}"
        )
    return names


def main():
    parser = argparse.ArgumentParser(
        description="H20 fixed-128K Qwen3 Decode-only CuTe attention"
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "correctness", "partial-compare", "benchmark", "profile"),
        default="correctness",
    )
    parser.add_argument("--config", choices=("auto", *CONFIGS), default="auto")
    parser.add_argument(
        "--benchmark-configs",
        type=_parse_benchmark_configs,
        default=None,
        help="comma-separated configs timed together with current AUTO, P2, and FA3",
    )
    parser.add_argument(
        "--scales",
        type=_parse_scales,
        default=(1.0 / math.sqrt(QWEN_HEAD_DIM), 0.125),
    )
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=1)
    args = parser.parse_args()
    if args.atol <= 0 or args.rtol < 0:
        parser.error("--atol must be positive and --rtol non-negative")
    if args.warmup < 0 or args.iterations <= 0 or args.rounds <= 0:
        parser.error("--warmup must be non-negative; --iterations/--rounds must be positive")

    if args.mode == "smoke":
        _run_smoke(args)
    elif args.mode == "correctness":
        _run_correctness(args)
    elif args.mode == "partial-compare":
        _run_partial_compare(args)
    elif args.mode == "benchmark":
        _run_benchmark(args)
    else:
        _run_profile(args)


if __name__ == "__main__":
    main()
