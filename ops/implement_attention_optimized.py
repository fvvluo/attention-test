# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Optimized Hopper CuTe DSL attention for Qwen3-32B.

Targets fixed B=1, Hq=64, Hkv=8, D=128 workloads: tuned causal Prefill
and concurrent split-KV Pack-GQA Decode with FP32 LSE combination.
"""

import math
import os
import sys
import threading
from typing import Type, Tuple, Optional, Sequence

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute

import cutlass.cute.nvgpu.warpgroup as warpgroup
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass._mlir.dialects import math as _math

import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.runtime import from_dlpack

CUTLASS_CUTE_EXAMPLES = os.environ.get(
    "CUTLASS_CUTE_EXAMPLES", "/dockerdata/cutlass/examples/python/CuTeDSL"
)
if CUTLASS_CUTE_EXAMPLES not in sys.path:
    sys.path.insert(0, CUTLASS_CUTE_EXAMPLES)

from helpers import fmha_helpers as fmha_utils

from cutlass.cutlass_dsl import (
    Boolean, Int32, if_generate, while_generate, yield_out, not_, dsl_user_op,
)
from cutlass._mlir.dialects import nvvm
from cutlass._mlir._mlir_libs._cutlass_ir._mlir.ir import IntegerType
from contextlib import contextmanager


import inspect as _inspect


# Presets retained from H20 tuning; auto selects the validated N192/KV3 winner.
PREFILL_CONFIGS = {
    "n128-kv5": (128, 5),
    "n128-kv4": (128, 4),
    "n160-kv3": (160, 3),
    "n160-kv4": (160, 4),
    "n192-kv3": (192, 3),
}
# N128/KV5 remains the Decode baseline and an explicit Prefill option.
PREFILL_AUTO_CONFIG = "n192-kv3"
DECODE_KERNEL_CONFIG = "n128-kv5"
DEFAULT_DECODE_SPLIT_CANDIDATES = (8, 9, 10, 16, 18, 19, 32)


def _resolve_prefill_config(name: str) -> tuple[str, int, int]:
    if name == "auto":
        name = PREFILL_AUTO_CONFIG
    try:
        block_n, kv_stage = PREFILL_CONFIGS[name]
    except KeyError as exc:
        choices = ", ".join(("auto", *PREFILL_CONFIGS))
        raise ValueError(f"unknown prefill config {name!r}; expected one of {choices}") from exc
    return name, block_n, kv_stage


_timelimit_has_res = "res" in _inspect.signature(
    nvvm.mbarrier_try_wait_parity_timelimit
).parameters
_MBARRIER_PATCH_LOCK = threading.RLock()


def _try_wait_timelimit(llvm_ptr, phase_val, timeout, *, loc=None, ip=None):
    if _timelimit_has_res:
        i1 = IntegerType.get_signless(1)
        return nvvm.mbarrier_try_wait_parity_timelimit(
            i1, llvm_ptr, phase_val, timeout, loc=loc, ip=ip,
        )
    return nvvm.mbarrier_try_wait_parity_timelimit(
        llvm_ptr, phase_val, timeout, loc=loc, ip=ip,
    )


@dsl_user_op
def _optimized_mbarrier_wait(mbar_ptr, phase, *, loc=None, ip=None):
    llvm_ptr = mbar_ptr.llvm_ptr
    phase_val = Int32(phase).ir_value(loc=loc, ip=ip)
    _true = lambda: Boolean(True).ir_value(loc=loc, ip=ip)
    timeout = Int32(10000000).ir_value(loc=loc, ip=ip)
    d = Boolean(_try_wait_timelimit(llvm_ptr, phase_val, timeout, loc=loc, ip=ip))
    d = if_generate(d, _true,
        lambda: _try_wait_timelimit(llvm_ptr, phase_val, timeout, loc=loc, ip=ip),
        None, [Boolean], loc=loc, ip=ip)
    d = if_generate(d, _true,
        lambda: _try_wait_timelimit(llvm_ptr, phase_val, timeout, loc=loc, ip=ip),
        None, [Boolean], loc=loc, ip=ip)
    def _fallback():
        inner = Boolean(False).ir_value(loc=loc, ip=ip)
        ctx = while_generate([inner], lambda x: not_(x, loc=loc, ip=ip), loc=loc, ip=ip)
        with ctx as (_,):
            r = Boolean(_try_wait_timelimit(
                llvm_ptr, phase_val, timeout, loc=loc, ip=ip,
            ))
            yield_out([r], loc=loc, ip=ip)
        return Boolean(True).ir_value(loc=loc, ip=ip)
    if_generate(d, _true, _fallback, None, [Boolean], loc=loc, ip=ip)


@contextmanager
def _use_optimized_mbarrier_wait():
    import cutlass.cute.arch as arch_mod

    # CuTe tracing consults this process-global symbol. Serialize the temporary
    # patch so concurrent optimized-kernel compilations cannot restore it out of
    # order and leave a different trace observing the wrong implementation.
    with _MBARRIER_PATCH_LOCK:
        orig_wait = arch_mod.mbarrier_wait
        arch_mod.mbarrier_wait = _optimized_mbarrier_wait
        try:
            yield
        finally:
            arch_mod.mbarrier_wait = orig_wait


class HopperFusedMultiHeadAttentionForward:
    def __init__(
        self,
        qk_acc_dtype,
        pv_acc_dtype,
        mma_tiler,
        is_persistent,
        mask_type: fmha_utils.MaskEnum,
        kv_stage: int = 5,
        write_lse: bool = True,
    ):
        """Configure Hopper warp groups, MMA tiles, stages, and LSE output."""

        self.num_mma_warp_groups = 2
        self.qk_acc_dtype = qk_acc_dtype
        self.pv_acc_dtype = pv_acc_dtype
        self.cta_tiler = self.cta_tile_shape_mnk = (
            mma_tiler[0] * self.num_mma_warp_groups,
            mma_tiler[1],
            mma_tiler[2],
        )

        self.qk_mma_tiler = (
            mma_tiler[0],
            mma_tiler[1],
            mma_tiler[2],
        )

        self.pv_mma_tiler = (
            self.qk_mma_tiler[0],
            self.qk_mma_tiler[2],
            self.qk_mma_tiler[1],
        )

        self.cluster_shape_mn = (1, 1)
        self.atom_layout_mnk = (1, 1, 1)
        self.is_persistent = is_persistent
        self.mask_type = mask_type
        self.configured_kv_stage = kv_stage
        self.write_lse = write_lse
        # Deliberately disabled: neither compact-producer nor extra WGMMA
        # overlap is enabled without a deadlock-free H20 validation.
        self.enable_compact_producer = False
        self.enable_wgmma_overlap = False
        self.threads_per_warp = 32
        self.num_threads_per_warp_group = 128
        self.num_warps_per_warp_group = (
            self.num_threads_per_warp_group / self.threads_per_warp
        )

        # WarpGroupRole
        self.load_warp_group_id = 0
        self.compute_epilogue_0_warp_group_id = 1
        self.compute_epilogue_1_warp_group_id = 2
        # ProducerWarpRole
        self.producer_warp_loadkv_id = 1

        self.num_regs_load = 40 - 2 * 8
        self.threads_per_cta = (
            self.num_mma_warp_groups + 1
        ) * self.num_threads_per_warp_group
        self.num_regs_mma = 240
        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        self.q_stage = 2
        self.kv_stage = self.configured_kv_stage
        self.epi_stage = 2

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        o: cute.Tensor,
        lse: cute.Tensor,
        scale_softmax_log2: cutlass.Float32,
        scale_softmax: cutlass.Float32,
        scale_output: cutlass.Float32,
        window_size_left: Optional[cutlass.Int32],
        window_size_right: Optional[cutlass.Int32],
        stream: cuda.CUstream,
    ):
        # setup static attributes before smem/grid/tma computation
        self.q_dtype = q.element_type
        self.k_dtype = k.element_type
        self.v_dtype = v.element_type
        self.o_dtype = o.element_type

        # (s, d, ((h_r, h_k), b)), 0-stride for h_r to broadcast
        k = cute.make_tensor(
            k.iterator,
            cute.make_layout(
                (k.shape[0], k.shape[1], ((q.shape[2], k.shape[3]), k.shape[4])),
                stride=(
                    k.stride[0],
                    k.stride[1],
                    ((0, k.stride[3]), k.stride[4]),
                ),
            ),
        )

        # (d, s, ((h_r, h_k), b)), 0-stride for h_r to broadcast
        v = cute.make_tensor(
            v.iterator,
            cute.make_layout(
                (v.shape[1], v.shape[0], ((q.shape[2], v.shape[3]), v.shape[4])),
                stride=(
                    v.stride[1],
                    v.stride[0],
                    ((0, v.stride[3]), v.stride[4]),
                ),
            ),
        )

        # (s, d, ((h_r, h_k), b))
        q = cute.group_modes(cute.group_modes(q, begin=2, end=4), begin=2, end=4)
        o = cute.group_modes(cute.group_modes(o, begin=2, end=4), begin=2, end=4)

        # The callable keeps a tensor argument for a stable CuTe signature.  In
        # write_lse=False specializations it is a one-element dummy and the
        # compile-time branch below removes all layout use, log, and stores.
        if cutlass.const_expr(self.write_lse):
            # (s, ((h_r, h_k), b))
            lse = cute.make_tensor(
                lse.iterator,
                cute.make_layout(
                    (
                        lse.shape[0],
                        self.pv_mma_tiler[1],
                        ((lse.shape[2], lse.shape[3]), lse.shape[4]),
                    ),
                    stride=(
                        lse.stride[0],
                        0,
                        ((lse.stride[2], lse.stride[3]), lse.stride[4]),
                    ),
                ),
            )

        if cutlass.const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        if cutlass.const_expr(self.q_dtype != self.v_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype}")

        if cutlass.const_expr(q.leading_dim != 1):  # k-major
            raise RuntimeError("The layout of q is not supported")

        if cutlass.const_expr(k.leading_dim != 1):  # k-major
            raise RuntimeError("The layout of k is not supported")

        self._setup_attributes()

        tile_shape_mnk = self.cta_tiler
        self.epi_tile = sm90_utils.compute_tile_shape_or_override(
            tile_shape_mnk, self.o_dtype
        )

        self.q_layout = utils.LayoutEnum.from_tensor(q)
        self.k_layout = utils.LayoutEnum.from_tensor(k)
        self.v_layout = utils.LayoutEnum.from_tensor(v)
        self.o_layout = utils.LayoutEnum.from_tensor(o)

        self.q_major_mode = self.q_layout.sm90_mma_major_mode()
        self.k_major_mode = self.k_layout.sm90_mma_major_mode()
        self.v_major_mode = self.v_layout.sm90_mma_major_mode()

        p_major_mode = cute.nvgpu.OperandMajorMode.K
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
            p_major_mode,
            self.v_major_mode,
            self.pv_acc_dtype,
            self.atom_layout_mnk,
            self.pv_mma_tiler[:2],
            warpgroup.OperandSource.RMEM,
        )

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (qk_tiled_mma.thr_id.shape,),
        )

        q_smem_layout_staged = sm90_utils.make_smem_layout_a(
            self.q_layout,
            self.qk_mma_tiler,
            self.q_dtype,
            self.q_stage,
        )

        k_smem_layout_staged = sm90_utils.make_smem_layout_b(
            self.k_layout,
            self.qk_mma_tiler,
            self.k_dtype,
            self.kv_stage,
        )

        v_smem_layout_staged = sm90_utils.make_smem_layout_b(
            self.v_layout,
            self.pv_mma_tiler,
            self.v_dtype,
            self.kv_stage,
        )

        o_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            self.o_dtype,
            self.o_layout,
            self.epi_tile,
            self.epi_stage,
            cute.append(
                cute.append(self.epi_tile, self.epi_stage), self.num_mma_warp_groups
            ),
            smem_order=(1, 0, 2, 3) if self.o_layout.is_m_major_c() else (0, 1, 2, 3),
        )

        # TMA load for Q
        q_smem_layout = cute.slice_(q_smem_layout_staged, (None, None, 0))
        tma_atom_q, tma_tensor_q = self._make_tma_atoms_and_tensors(
            q,
            q_smem_layout_staged,
            (self.qk_mma_tiler[0], self.qk_mma_tiler[2]),
            self.cluster_shape_mnk[1],
        )

        # TMA load for K
        k_smem_layout = cute.slice_(k_smem_layout_staged, (None, None, 0))
        tma_atom_k, tma_tensor_k = self._make_tma_atoms_and_tensors(
            k,
            k_smem_layout_staged,
            (self.qk_mma_tiler[1], self.qk_mma_tiler[2]),
            self.cluster_shape_mnk[0],
        )

        # TMA load for V
        pv_tile_shape_mnk = (
            self.qk_mma_tiler[0],
            self.qk_mma_tiler[2],
            self.qk_mma_tiler[1],
        )
        tma_atom_v, tma_tensor_v = self._make_tma_atoms_and_tensors(
            v,
            v_smem_layout_staged,
            (pv_tile_shape_mnk[1], pv_tile_shape_mnk[2]),
            self.cluster_shape_mnk[0],
        )

        o_cta_v_layout = cute.composition(
            cute.make_identity_layout(o.shape), self.epi_tile
        )
        o_smem_layout = cute.slice_(o_smem_layout_staged, (None, None, 0, 0))

        tma_store_op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()
        tma_atom_o, tma_tensor_o = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_store_op,
            o,
            o_smem_layout,
            self.epi_tile,
        )

        q_copy_size = cute.size_in_bytes(self.q_dtype, q_smem_layout)
        k_copy_size = cute.size_in_bytes(self.k_dtype, k_smem_layout)
        self.tma_copy_q_bytes = q_copy_size
        self.tma_copy_kv_bytes = k_copy_size

        self.tile_sched_params, grid = fmha_utils.compute_grid(
            o.shape,
            self.cta_tiler,
            self.is_persistent,
        )

        @cute.struct
        class SharedStorage:
            # 2 for full/empty
            load_q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.q_stage * 2]
            load_kv_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.kv_stage * 2]
            MathWarpGroupOrderBarrier: cute.struct.MemRange[
                cutlass.Int64, self.num_mma_warp_groups
            ]

            sO: cute.struct.Align[
                cute.struct.MemRange[
                    self.o_dtype,
                    (
                        cute.cosize(o_smem_layout_staged)
                        if cutlass.const_expr(self.is_persistent)
                        else 0
                    ),
                ],
                self.buffer_align_bytes,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(q_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(k_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Enqueue the kernel on the requested CUDA stream
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
                tma_atom_o,
                tma_tensor_o,
                lse,
                scale_softmax_log2,
                scale_softmax,
                scale_output,
                window_size_left,
                window_size_right,
                q_smem_layout_staged,
                k_smem_layout_staged,
                v_smem_layout_staged,
                o_smem_layout_staged,
                self.tile_sched_params,
            ).launch(
                grid=grid,
                block=[self.threads_per_cta, 1, 1],
                cluster=self.cluster_shape_mnk,
                stream=stream,
                min_blocks_per_mp=1,
            )

    #  GPU device kernel
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
        tma_atom_o: cute.CopyAtom,
        mO_qdl: cute.Tensor,
        mLse_qdl: cute.Tensor,
        scale_softmax_log2: cutlass.Float32,
        scale_softmax: cutlass.Float32,
        scale_output: cutlass.Float32,
        window_size_left: Optional[cutlass.Int32],
        window_size_right: Optional[cutlass.Int32],
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        o_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: fmha_utils.FmhaStaticTileSchedulerParams,
    ):
        """Run the warp-specialized Hopper FMHA device kernel.

        A load warp group feeds staged TMA Q/K/V pipelines while two math warp
        groups execute WGMMA QK, online softmax, PV, optional LSE, and epilogue.
        """

        tidx, _, _ = cute.arch.thread_idx()

        # Alloc
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        load_q_producer, load_q_consumer = self.make_and_init_load_q_pipeline(
            storage.load_q_mbar_ptr.data_ptr()
        )
        load_kv_producer, load_kv_consumer = self.make_and_init_load_kv_pipeline(
            storage.load_kv_mbar_ptr.data_ptr()
        )
        tma_store_pipeline = self.make_and_init_tma_store_pipeline()

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group
        )

        math_wg_order_barrier = self.make_and_init_order_barrier(
            storage.MathWarpGroupOrderBarrier.data_ptr(),
            warp_group_idx - 1,
        )

        #  Generate smem tensor Q/K/V/O
        # (MMA, MMA_Q, MMA_D, PIPE)
        sQ = storage.sQ.get_tensor(
            q_smem_layout_staged.outer, swizzle=q_smem_layout_staged.inner
        )
        # (MMA, MMA_K, MMA_D, PIPE)
        sK = storage.sK.get_tensor(
            k_smem_layout_staged.outer, swizzle=k_smem_layout_staged.inner
        )
        # (MMA, MMA_K, MMA_D, PIPE)
        # Adjust swizzle info to reuse smem
        sV_ptr = cute.recast_ptr(sK.iterator, v_smem_layout_staged.inner)
        sV = cute.make_tensor(sV_ptr, v_smem_layout_staged.outer)

        if cutlass.const_expr(self.is_persistent):
            sO = storage.sO.get_tensor(
                o_smem_layout_staged.outer, swizzle=o_smem_layout_staged.inner
            )
        else:
            sO = cute.make_tensor(
                cute.recast_ptr(sQ.iterator, o_smem_layout_staged.inner, self.o_dtype),
                o_smem_layout_staged.outer,
            )

        seqlen_q = mQ_qdl.shape[0]
        gQ_qdl = cute.flat_divide(mQ_qdl, cute.select(self.qk_mma_tiler, mode=[0, 2]))
        qk_thr_mma = qk_tiled_mma.get_slice(tidx)
        tSgQ_qdl = qk_thr_mma.partition_A(gQ_qdl)

        tQsQ, tQgQ_qdl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_q,
            0,  # no multicast
            cute.make_layout(1),
            cute.group_modes(sQ, 0, 2),
            cute.group_modes(tSgQ_qdl, 0, 3),
        )

        seqlen_k = mK_kdl.shape[0]
        gK_kdl = cute.flat_divide(mK_kdl, cute.select(self.qk_mma_tiler, mode=[1, 2]))
        tSgK_kdl = qk_thr_mma.partition_B(gK_kdl)
        tKsK, tKgK_kdl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_k,
            0,  # no multicast
            cute.make_layout(1),
            cute.group_modes(sK, 0, 2),
            cute.group_modes(tSgK_kdl, 0, 3),
        )

        gV_dkl = cute.flat_divide(mV_dkl, cute.select(self.pv_mma_tiler, mode=[1, 2]))
        pv_thr_mma = pv_tiled_mma.get_slice(tidx)
        tSgV_dkl = pv_thr_mma.partition_B(gV_dkl)
        tVsV, tVgV_dkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_v,
            0,  # no multicast
            cute.make_layout(1),
            cute.group_modes(sV, 0, 2),
            cute.group_modes(tSgV_dkl, 0, 3),
        )

        producer_warp_role = warp_idx % 4  # self.num_warps_per_warp_group

        # Fence the mbarrier init to ensure all mbarrier initializations are visible
        # to all threads. This is critical for FP8 performance - without this fence,
        # the compiler may generate software polling loops instead of hardware waits.
        cute.arch.mbarrier_init_fence()

        # We need this to guarantee that the Pipeline init is visible
        # To all producers and consumer blocks in the Cluster
        # and to finish smem init
        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        else:
            cute.arch.sync_threads()

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_o)

        if warp_group_idx == self.load_warp_group_id:
            cute.arch.setmaxregister_decrease(self.num_regs_load)

            tile_sched = fmha_utils.create_fmha_static_tile_scheduler(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx

                q0_index = 0
                k_index = fmha_utils.FusedMask.get_trip_start(
                    self.mask_type,
                    curr_block_coord,
                    self.cta_tiler,
                    seqlen_q,
                    seqlen_k,
                    window_size_left,
                )
                fusion_tile_count = fmha_utils.FusedMask.get_trip_count(
                    self.mask_type,
                    curr_block_coord,
                    self.cta_tiler,
                    seqlen_q,
                    seqlen_k,
                    window_size_left,
                    window_size_right,
                )

                q_tile_count = self.num_mma_warp_groups
                k_tile_count = 2 * fusion_tile_count

                curr_block_coord_m = curr_block_coord[0]
                _tQgQ = tQgQ_qdl[(None, None, 0, curr_block_coord[2])]
                tQgQ = cute.domain_offset(
                    (0, curr_block_coord_m * self.num_mma_warp_groups), _tQgQ
                )

                if producer_warp_role == self.producer_warp_loadkv_id:
                    # LoadQ
                    if q_tile_count > 0:
                        q_handle = load_q_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_q,
                            tQgQ[(None, q0_index)],
                            tQsQ[(None, q_handle.index)],
                            tma_bar_ptr=q_handle.barrier,
                        )
                        q0_index += 1

                    q_tile_count -= 1

                    tKgK = tKgK_kdl[(None, None, 0, curr_block_coord[2])]
                    tVgV = tVgV_dkl[(None, 0, None, curr_block_coord[2])]

                    # Load K
                    if k_tile_count > 0:
                        k_handle = load_kv_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_k,
                            tKgK[(None, k_index)],
                            tKsK[(None, k_handle.index)],
                            tma_bar_ptr=k_handle.barrier,
                        )

                    k_tile_count -= 1

                    # Q1
                    if q_tile_count > 0:
                        q_handle = load_q_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_q,
                            tQgQ[(None, q0_index)],
                            tQsQ[(None, q_handle.index)],
                            tma_bar_ptr=q_handle.barrier,
                        )
                        q0_index += 1
                    q_tile_count -= 1

                    # LoadV
                    if k_tile_count > 0:
                        k_handle = load_kv_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_v,
                            tVgV[(None, k_index)],
                            tVsV[(None, k_handle.index)],
                            tma_bar_ptr=k_handle.barrier,
                        )

                        k_index += 1
                    k_tile_count -= 1

                    while k_tile_count > 0:
                        # Load KV
                        k_handle = load_kv_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_k,
                            tKgK[(None, k_index)],
                            tKsK[(None, k_handle.index)],
                            tma_bar_ptr=k_handle.barrier,
                        )

                        k_tile_count -= 1

                        v_handle = load_kv_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_v,
                            tVgV[(None, k_index)],
                            tVsV[(None, v_handle.index)],
                            tma_bar_ptr=v_handle.barrier,
                        )

                        k_index += 1
                        k_tile_count -= 1

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

        # Mainloop
        if (
            warp_group_idx == self.compute_epilogue_0_warp_group_id
            or warp_group_idx == self.compute_epilogue_1_warp_group_id
        ):
            cute.arch.setmaxregister_increase(self.num_regs_mma)

            tile_sched = fmha_utils.create_fmha_static_tile_scheduler(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            kOuterLoads = 1

            cP = cute.make_identity_tensor((mQ_qdl.shape[0], seqlen_k))
            gPcP = cute.local_tile(cP, self.qk_mma_tiler[:2], (None, None))

            while work_tile.is_valid_tile:
                for i in cutlass.range((warp_group_idx - 1) * kOuterLoads, unroll=1):
                    load_q_consumer.advance()

                curr_block_coord = work_tile.tile_idx

                # _wg_coord_1 is work_tile.tile_idx[1], which is always 0.
                _wg_coord_0 = self.num_mma_warp_groups * curr_block_coord[0] + (
                    warp_group_idx - 1
                )
                _wg_coord_1 = curr_block_coord[1]

                wg_coord = (_wg_coord_0, _wg_coord_1, *curr_block_coord[2:])

                # Mainloop setup QK
                tSsQ = qk_thr_mma.partition_A(sQ)  # (MMA,MMA_M,MMA_K,PIPE)
                tSsK = qk_thr_mma.partition_B(sK)  # (MMA,MMA_N,MMA_K,PIPE)
                tSrQ = qk_thr_mma.make_fragment_A(tSsQ)  # (MMA,MMA_M,MMA_K,PIPE)
                tSrK = qk_thr_mma.make_fragment_B(tSsK)  # (MMA,MMA_N,MMA_K,PIPE)

                # Prepare: MMA PV
                thr_mma_pv = pv_tiled_mma.get_slice(tidx)

                # Mainloop setup PV
                tOsV = thr_mma_pv.partition_B(sV)  # (MMA,MMA_N,MMA_K,PIPE)
                tOrV = thr_mma_pv.make_fragment_B(tOsV)  # (MMA,MMA_M,MMA_N,PIPE)

                q_handle = load_q_consumer.wait()

                # mapping into QK accumulator
                ptPcP = qk_thr_mma.partition_C(gPcP)

                # Allocate PV acc
                pv_acc_shape = pv_thr_mma.partition_shape_C(
                    (self.pv_mma_tiler[0], self.pv_mma_tiler[1])
                )
                acc_pv = pv_thr_mma.make_fragment_C(pv_acc_shape)

                qk_acc_shape = qk_thr_mma.partition_shape_C(
                    (self.qk_mma_tiler[0], self.qk_mma_tiler[1])
                )

                s_max_layout = cute.make_layout(
                    cute.size(self.layout_acc_mn(pv_tiled_mma, acc_pv.layout), mode=[0])
                )
                s_max = cute.make_rmem_tensor_like(s_max_layout, self.qk_acc_dtype)
                a_sum = cute.make_rmem_tensor_like(s_max, cutlass.Float32)

                kv_offset = fmha_utils.FusedMask.get_trip_start(
                    self.mask_type,
                    curr_block_coord,
                    self.cta_tiler,
                    seqlen_q,
                    seqlen_k,
                    window_size_left,
                )

                masked_leading_count = fmha_utils.FusedMask.get_masked_leading_count(
                    self.mask_type,
                    curr_block_coord,
                    self.cta_tiler,
                    seqlen_q,
                    seqlen_k,
                    window_size_left,
                    window_size_right,
                )
                unmasked_trip_count = fmha_utils.FusedMask.get_unmasked_trip_count(
                    self.mask_type,
                    curr_block_coord,
                    self.cta_tiler,
                    seqlen_q,
                    seqlen_k,
                    window_size_left,
                    window_size_right,
                )

                # mapping into QK accumulator
                tPcP = cute.slice_(ptPcP, (None, None, None, wg_coord[0], kv_offset))
                kv_offset += 1

                qk_acc_shape = qk_thr_mma.partition_shape_C(
                    (self.qk_mma_tiler[0], self.qk_mma_tiler[1])
                )

                # Allocate QK acc
                acc_qk = qk_thr_mma.make_fragment_C(qk_acc_shape)
                k_handle = load_kv_consumer.wait_and_advance()
                math_wg_order_barrier.wait()

                # MMA QK
                cute.nvgpu.warpgroup.fence()

                self.gemm_zero_acc(
                    qk_tiled_mma,
                    tSrQ[(None, None, None, q_handle.index)],
                    tSrK[(None, None, None, k_handle.index)],
                    acc_qk,
                )
                cute.nvgpu.warpgroup.commit_group()

                math_wg_order_barrier.arrive()

                # Wait for the pipeline MMAs to drain
                cute.nvgpu.warpgroup.wait_group(0)

                s_max, a_sum = self.softmax_step(
                    True,
                    self.mask_type,
                    acc_qk,
                    qk_tiled_mma,
                    tPcP,
                    s_max,
                    a_sum,
                    acc_qk,
                    qk_tiled_mma,
                    scale_softmax_log2,
                    seqlen_k,
                    seqlen_q,
                    window_size_left,
                    window_size_right,
                    True,
                )

                acc_qk_fixed = self.make_acc_into_op(
                    acc_qk, pv_tiled_mma.tv_layout_A, self.q_dtype
                )

                v_handle = load_kv_consumer.wait_and_advance()

                # MMA PV
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

                if masked_leading_count >= 1:
                    masked_leading_count -= 1
                    load_kv_consumer, k_tile_count, kv_offset, s_max, a_sum = (
                        self.compute(
                            True,
                            masked_leading_count,
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
                            seqlen_k,
                            seqlen_q,
                            qk_acc_shape,
                            window_size_left,
                            window_size_right,
                        )
                    )
                else:
                    unmasked_trip_count -= 1

                load_kv_consumer, k_tile_count, kv_offset, s_max, a_sum = self.compute(
                    False,
                    unmasked_trip_count,
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
                    seqlen_k,
                    seqlen_q,
                    qk_acc_shape,
                    window_size_left,
                    window_size_right,
                )

                k_tile_count = fmha_utils.FusedMask.get_masked_trailing_count(
                    self.mask_type,
                    curr_block_coord,
                    self.cta_tiler,
                    seqlen_q,
                    seqlen_k,
                    window_size_left,
                    window_size_right,
                    k_tile_count,
                )

                # Use fusion in softmax
                load_kv_consumer, k_tile_count, kv_offset, s_max, a_sum = self.compute(
                    True,
                    k_tile_count,
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
                    seqlen_k,
                    seqlen_q,
                    qk_acc_shape,
                    window_size_left,
                    window_size_right,
                )

                if cutlass.const_expr(self.is_persistent):
                    q_handle.release()

                # Wait for the pipeline MMAs to drain
                cute.nvgpu.warpgroup.wait_group(0)

                # Normalize the output.  Public prefill compiles the no-LSE
                # specialization so the logarithm and LSE register fragment do
                # not exist in that kernel.
                if cutlass.const_expr(self.write_lse):
                    lse = self.tail(
                        s_max, a_sum, acc_pv, pv_tiled_mma, scale_softmax, scale_output
                    )
                else:
                    self.tail_no_lse(a_sum, acc_pv, pv_tiled_mma, scale_output)

                if warp_group_idx == self.compute_epilogue_0_warp_group_id:
                    for i in cutlass.range_constexpr(
                        kOuterLoads * (self.num_mma_warp_groups - 0)
                    ):
                        load_q_consumer.advance()

                if cutlass.const_expr(self.num_mma_warp_groups >= 2):
                    if warp_group_idx == self.compute_epilogue_1_warp_group_id:
                        for i in cutlass.range_constexpr(
                            kOuterLoads * (self.num_mma_warp_groups - 1)
                        ):
                            load_q_consumer.advance()

                math_wg_order_barrier.wait()

                if cutlass.const_expr(self.write_lse):
                    # Store log-sum-exp (LSE) only in the decode-partial or
                    # explicit diagnostic specialization.
                    thr_mma = pv_tiled_mma.get_slice(tidx)

                    gLSE_full = cute.local_tile(
                        mLse_qdl, self.pv_mma_tiler[:2], (None, None, None)
                    )
                    gLSE = cute.slice_(
                        gLSE_full,
                        (None, None, wg_coord[0], wg_coord[1], wg_coord[2]),
                    )

                    tOgLSE = thr_mma.partition_C(gLSE)
                    cO = cute.make_identity_tensor(
                        (self.pv_mma_tiler[0], self.pv_mma_tiler[1])
                    )
                    tOcO = thr_mma.partition_C(cO)

                    if tOcO[0][1] == 0:
                        tOgLSE_mn = cute.make_tensor(
                            tOgLSE.iterator,
                            self.layout_acc_mn(pv_tiled_mma, tOgLSE.layout),
                        )
                        tOcO_mn = cute.make_tensor(
                            tOcO.iterator,
                            self.layout_acc_mn(pv_tiled_mma, tOcO.layout),
                        )
                        for i in cutlass.range_constexpr(
                            cute.size(tOgLSE_mn, mode=[0])
                        ):
                            if (
                                tOcO_mn[i][0]
                                + wg_coord[0] * self.pv_mma_tiler[0]
                                < seqlen_q
                            ):
                                tOgLSE_mn[(i, 0)] = lse[i]

                # Epilogue
                cO = cute.make_identity_tensor((self.cta_tiler[0], self.cta_tiler[2]))
                copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
                    self.o_layout,
                    elem_ty_d=self.o_dtype,
                    elem_ty_acc=self.pv_acc_dtype,
                )

                copy_atom_O = cute.make_copy_atom(
                    cute.nvgpu.warp.StMatrix8x8x16bOp(
                        self.o_layout.is_m_major_c(),
                        4,
                    ),
                    self.o_dtype,
                )

                tiled_copy_O_Atom = cute.make_tiled_copy_C_atom(
                    copy_atom_O, pv_tiled_mma
                )

                tiled_copy_r2s = cute.make_tiled_copy_S(
                    copy_atom_r2s,
                    tiled_copy_O_Atom,
                )

                thr_copy_r2s = tiled_copy_r2s.get_slice(
                    tidx % self.num_threads_per_warp_group
                )
                tRS_sD = thr_copy_r2s.partition_D(sO)
                tRS_rAcc = tiled_copy_r2s.retile(acc_pv)

                # Allocate D registers.
                rD_shape = cute.shape(thr_copy_r2s.partition_S(sO))
                tRS_rD_layout = cute.make_layout(rD_shape[:3])

                tRS_rD = cute.make_rmem_tensor_like(tRS_rD_layout, self.pv_acc_dtype)
                size_tRS_rD = cute.size(tRS_rD)

                gD = cute.local_tile(
                    mO_qdl,
                    self.pv_mma_tiler[:2],
                    (wg_coord[0], 0, wg_coord[2]),
                )

                sepi_for_tma_partition = cute.group_modes(sO, 0, 2)
                tcgc_for_tma_partition = cute.zipped_divide(gD, self.epi_tile)

                bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_o,
                    0,
                    cute.make_layout(1),
                    sepi_for_tma_partition,
                    tcgc_for_tma_partition,
                )

                epi_tile_num = cute.size(tcgc_for_tma_partition, mode=[1])

                for epi_idx in cutlass.range_constexpr(epi_tile_num):
                    # Copy from accumulators to D registers
                    for epi_v in cutlass.range_constexpr(size_tRS_rD):
                        tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]

                    # Type conversion
                    tRS_rD_out = cute.make_rmem_tensor_like(tRS_rD_layout, self.o_dtype)
                    acc_vec = tRS_rD.load()
                    tRS_rD_out.store(acc_vec.to(self.o_dtype))

                    # Copy from D registers to shared memory
                    epi_buffer = epi_idx % self.epi_stage
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rD_out,
                        tRS_sD[(None, None, None, epi_buffer, warp_group_idx - 1)],
                    )

                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )
                    pipeline.arrive_and_wait(
                        barrier_id=warp_group_idx,
                        num_threads=self.num_threads_per_warp_group,
                    )

                    # only one warp in each warpgroup copy shared memory to global memory
                    if warp_idx == 4 or warp_idx == 8:
                        cute.copy(
                            tma_atom_o,
                            bSG_sD[(None, epi_buffer, warp_group_idx - 1)],
                            bSG_gD[(None, epi_idx)],
                        )

                        tma_store_pipeline.producer_commit()
                        tma_store_pipeline.producer_acquire()

                    pipeline.arrive_and_wait(
                        barrier_id=warp_group_idx,
                        num_threads=self.num_threads_per_warp_group,
                    )

                math_wg_order_barrier.arrive()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
        return

    @cute.jit
    def compute(
        self,
        fusion: bool,
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
        seqlen_k: cutlass.Int32,
        seqlen_q: cutlass.Int32,
        qk_acc_shape: cute.Shape,
        window_size_left: Optional[cutlass.Int32],
        window_size_right: Optional[cutlass.Int32],
    ) -> Tuple[
        pipeline.PipelineConsumer,
        cutlass.Int32,
        cutlass.Int32,
        cute.Tensor,
        cute.Tensor,
    ]:
        while k_tile_count > 0:
            k_tile_count -= 1

            tPcP = cute.slice_(ptPcP, (None, None, None, wg_coord[0], kv_offset))
            kv_offset += 1

            # Allocate QK acc
            acc_qk = qk_thr_mma.make_fragment_C(qk_acc_shape)

            k_handle = load_kv_consumer.wait_and_advance()

            # MMA QK
            cute.nvgpu.warpgroup.fence()

            self.gemm_zero_acc(
                qk_tiled_mma,
                tSrQ[(None, None, None, q_handle.index)],
                tSrK[(None, None, None, k_handle.index)],
                acc_qk,
            )

            cute.nvgpu.warpgroup.commit_group()

            tok = load_kv_consumer.try_wait()

            # Wait for the pipeline MMAs to drain
            cute.nvgpu.warpgroup.wait_group(0)

            s_max, a_sum = self.softmax_step(
                fusion,
                self.mask_type,
                acc_qk,
                qk_tiled_mma,
                tPcP,
                s_max,
                a_sum,
                acc_pv,
                pv_tiled_mma,
                scale_softmax_log2,
                seqlen_k,
                seqlen_q,
                window_size_left,
                window_size_right,
            )

            acc_qk_fixed = self.make_acc_into_op(
                acc_qk, pv_tiled_mma.tv_layout_A, self.q_dtype
            )

            v_handle = load_kv_consumer.wait_and_advance(tok)

            # MMA PV
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

        return load_kv_consumer, k_tile_count, kv_offset, s_max, a_sum

    @cute.jit
    def softmax_step(
        self,
        fusion: bool,
        mask_type: fmha_utils.MaskEnum,
        acc_qk: cute.ThrMma,
        tiled_mma_qk: cute.TiledMma,
        count_qk: cute.Tensor,
        s_max: cute.Tensor,
        a_sum: cute.Tensor,
        acc_pv: cute.ThrMma,
        tiled_mma_pv: cute.TiledMma,
        scale_softmax_log2: cutlass.Float32,
        seqlen_k: cutlass.Int32,
        seqlen_q: cutlass.Int32,
        window_size_left: Optional[cutlass.Int32],
        window_size_right: Optional[cutlass.Int32],
        is_first_iter: bool = False,
    ) -> Tuple[cute.Tensor, cute.Tensor]:
        if cutlass.const_expr(fusion):
            fmha_utils.FusedMask.apply_mask(
                mask_type,
                acc_qk,
                count_qk,
                seqlen_q,
                seqlen_k,
                window_size_left,
                window_size_right,
            )

        acc_qk_mn = cute.make_tensor(
            acc_qk.iterator, self.layout_acc_mn(tiled_mma_qk, acc_qk.layout)
        )

        reduction_target_qk = self.reduction_target_n(tiled_mma_qk)
        red_rank = cute.rank(reduction_target_qk)

        s_max_prev = None
        acc_pv_mn = None
        if cutlass.const_expr(is_first_iter):
            # Linear reduction is faster for the first iteration
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

                # Linear reduction is faster here, as well
                for j in cutlass.range_constexpr(cute.size(acc_qk_mn, mode=[1])):
                    s_max[i] = cutlass.max(s_max[i], acc_qk_mn[i, j])

            # reduce max
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
                    scale_softmax_log2 * acc_qk_mn[i, j] - scale_max, fastmath=True
                )

            _a_sum = 0.0
            if cutlass.const_expr(not is_first_iter):
                s_max_cur = s_max[i]
                if s_max[i] == -cutlass.Float32.inf:
                    s_max_cur = 0.0
                scale_pv = cute.math.exp2(
                    (s_max_prev[i] - s_max_cur) * scale_softmax_log2, fastmath=True
                )
                a_sum[i] *= scale_pv

                for j in cutlass.range_constexpr(cute.size(acc_pv_mn, mode=[1])):
                    acc_pv_mn[i, j] *= scale_pv

                _a_sum = a_sum[i]

            a_sum[i] = _a_sum + acc_qk_mn[i, None].load().reduce(
                cute.ReductionOp.ADD, cutlass.Float32.zero, 0
            )

        return s_max, a_sum

    @cute.jit
    def reduction_target_n(self, tiled_mma):
        separated = self.layout_separate(
            tiled_mma.shape_mnk[0],
            cute.make_layout(tiled_mma.tv_layout_C.shape[0]),
            tiled_mma.tv_layout_C.stride[0],
        )
        return separated[1]

    @staticmethod
    def convert_c_layout_to_a_layout(c, a):
        return cute.make_layout(
            (a, c.shape[1], (c.shape[2], cute.size(c, mode=[0]) // cute.size(a))),
            stride=(
                c.stride[0],
                c.stride[1],
                (c.stride[2], cute.size(a, mode=[2]) * c.stride[0][2]),
            ),
        )

    @cute.jit
    def make_acc_into_op(self, acc, operand_layout_tv, Element):
        operand = cute.make_rmem_tensor_like(
            self.convert_c_layout_to_a_layout(acc.layout, operand_layout_tv.shape[1]),
            Element,
        )
        operand_as_acc = cute.make_tensor(operand.iterator, acc.layout)
        acc_vec = acc.load()
        operand_as_acc.store(acc_vec.to(Element))

        if cutlass.const_expr(Element.width == 8 and True):
            ## 00 11 22 33 00 11 22 33 acc layout
            ## 00 00 11 11 22 22 33 33 operand layout
            ## BB AA AA BB AA BB BB AA conflict-free exchange pattern
            ##                         16-bit exchange; so process two at a time potentially
            # int tid = threadIdx.x % 4;
            tidx, _, _ = cute.arch.thread_idx()
            tid = tidx % 4
            values_u32 = cute.recast_tensor(operand, cutlass.Uint32)
            for n in cutlass.range_constexpr(cute.size(values_u32, mode=[1])):
                for k in cutlass.range_constexpr(cute.size(values_u32, mode=[2])):
                    for ii in cutlass.range_constexpr(0, 8, 4):
                        values_tmp_0 = values_u32[ii // 2 + 0, n, k]
                        values_tmp_1 = values_u32[ii // 2 + 1, n, k]

                        ## step A:
                        ## t 1 v 0 -> t 0 v 1
                        ## t 2 v 0 -> t 1 v 0
                        ## t 0 v 1 -> t 2 v 0
                        ## t 3 v 1 -> t 3 v 1

                        v_to_send = 1
                        if tid == 1 or tid == 2:
                            v_to_send = 0

                        v_to_recv = v_to_send
                        t_to_recv_from = (0x3021 >> (tid * 4)) & 0xF

                        values_tmp_a = values_tmp_1
                        if v_to_send == 0:
                            values_tmp_a = values_tmp_0

                        values_tmp_a = cute.arch.shuffle_sync_op(
                            values_tmp_a, t_to_recv_from, 0xFFFFFFFF, 7199
                        )

                        # step B:
                        # t 0 v 0 -> t 0 v 0
                        # t 3 v 0 -> t 1 v 1
                        # t 1 v 1 -> t 2 v 1
                        # t 2 v 1 -> t 3 v 0

                        v_to_send = 1 - v_to_send
                        v_to_recv = 1 - v_to_recv
                        t_to_recv_from = (0x2130 >> (tid * 4)) & 0xF

                        values_tmp_b = values_tmp_1
                        if v_to_send == 0:
                            values_tmp_b = values_tmp_0

                        values_tmp_b = cute.arch.shuffle_sync_op(
                            values_tmp_b, t_to_recv_from, 0xFFFFFFFF, 7199
                        )

                        # __byte_perm
                        order = 0x5410
                        if v_to_send == 0:
                            order = 0x1054

                        values_u32[ii // 2 + 0, n, k] = cute.arch.prmt(
                            values_tmp_a,
                            values_tmp_b,
                            order,
                        )

                        order = 0x7632
                        if v_to_send == 0:
                            order = 0x3276
                        values_u32[ii // 2 + 1, n, k] = cute.arch.prmt(
                            values_tmp_a, values_tmp_b, order
                        )
        return operand

    @cute.jit
    def tail(self, s_max, a_sum, acc_pv, tiled_mma_pv, scale_softmax, scale_output):
        """Reduce softmax sums, compute LSE, and normalize the PV accumulator."""
        # Create tensor view of accumulated P*V values with M*N layout
        acc_pv_mn = cute.make_tensor(
            acc_pv.iterator, self.layout_acc_mn(tiled_mma_pv, acc_pv.layout)
        )
        reduction_target = self.reduction_target_n(tiled_mma_pv)
        red_rank = cute.rank(reduction_target)
        for r in cutlass.range_constexpr(red_rank):
            for i in cutlass.range_constexpr(cute.size(acc_pv_mn, mode=[0])):
                a_sum[i] = cute.arch.warp_reduction_sum(
                    a_sum[i], threads_in_group=reduction_target.shape[r]
                )

        acc_mn = cute.make_tensor(
            acc_pv.iterator, self.layout_acc_mn(tiled_mma_pv, acc_pv.layout)
        )

        lse = cute.make_rmem_tensor_like(a_sum, a_sum._dtype)
        for i in cutlass.range_constexpr(cute.size(acc_mn, mode=[0])):
            sum = a_sum[i]
            inv_sum = cute.arch.rcp_approx(sum)
            if sum == 0.0 or sum != sum:
                inv_sum = 1.0

            lse[i] = s_max[i] * scale_softmax + _math.log(sum)
            if sum == 0.0 or sum != sum:
                lse[i] = cutlass.Float32.inf

            rp_dropout = 1
            scale = rp_dropout * inv_sum
            for j in cutlass.range_constexpr(cute.size(acc_mn, mode=[1])):
                acc_mn[i, j] *= scale * scale_output

        return lse

    @cute.jit
    def tail_no_lse(self, a_sum, acc_pv, tiled_mma_pv, scale_output):
        """Normalize O without materializing LSE or evaluating a logarithm."""
        acc_pv_mn = cute.make_tensor(
            acc_pv.iterator, self.layout_acc_mn(tiled_mma_pv, acc_pv.layout)
        )
        reduction_target = self.reduction_target_n(tiled_mma_pv)
        red_rank = cute.rank(reduction_target)
        for r in cutlass.range_constexpr(red_rank):
            for i in cutlass.range_constexpr(cute.size(acc_pv_mn, mode=[0])):
                a_sum[i] = cute.arch.warp_reduction_sum(
                    a_sum[i], threads_in_group=reduction_target.shape[r]
                )

        for i in cutlass.range_constexpr(cute.size(acc_pv_mn, mode=[0])):
            sum = a_sum[i]
            inv_sum = cute.arch.rcp_approx(sum)
            if sum == 0.0 or sum != sum:
                inv_sum = 1.0
            for j in cutlass.range_constexpr(cute.size(acc_pv_mn, mode=[1])):
                acc_pv_mn[i, j] *= inv_sum * scale_output

    @staticmethod
    def layout_separate(thr, src, ref):
        lt = cute.make_layout(())
        ge = cute.make_layout(())

        for k, v in enumerate(ref):
            if cutlass.const_expr(v < thr):
                lt = cute.append(lt, src[k])
            else:
                ge = cute.append(ge, src[k])

        r = None
        if cutlass.const_expr(cute.rank(lt) == 1):
            r = cute.append(lt, ge)
        else:
            r = cute.append(cute.append(cute.make_layout(()), lt), ge)
        return r

    @staticmethod
    @cute.jit
    def gemm_zero_acc(tiled_mma, A, B, C):
        rA = cute.rank(A)
        rB = cute.rank(B)
        rC = cute.rank(C)
        if cutlass.const_expr(rA == 2 and rB == 2 and rC == 1):
            for k_block_idx in range(cute.size(A, mode=[1]), unroll_full=True):
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, k_block_idx != 0)
                cute.gemm(
                    tiled_mma,
                    C,
                    A[None, k_block_idx],
                    B[None, k_block_idx],
                    C,
                )
        elif cutlass.const_expr(rA == 3 and rB == 3 and rC == 3):
            for k_block_idx in range(cute.size(A, mode=[2]), unroll_full=True):
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, k_block_idx != 0)
                cute.gemm(
                    tiled_mma,
                    C,
                    A[None, None, k_block_idx],
                    B[None, None, k_block_idx],
                    C,
                )
        else:
            assert 0

    @cute.jit
    def layout_acc_mn(self, tiled_mma, acc):
        separated = self.layout_separate(
            tiled_mma.shape_mnk[0], acc[0], tiled_mma.tv_layout_C.stride[1]
        )

        V_M = separated[0]
        V_N = separated[1]
        V_M1 = None
        V_N1 = None
        if cutlass.const_expr(cute.rank(V_M) == 1):
            V_M1 = cute.append(V_M, acc[1])
        else:
            V_M1 = cute.append(cute.append(cute.make_layout(()), V_M), acc[1])

        if cutlass.const_expr(cute.rank(V_N) == 1):
            V_N1 = cute.append(V_N, acc[2])
        else:
            V_N1 = cute.append(cute.append(cute.make_layout(()), V_N), acc[2])
        r = None
        if cutlass.const_expr(cute.rank(V_M1) == 1):
            r = cute.append(V_M1, V_N1)
        else:
            r = cute.append(cute.append(cute.make_layout(()), V_M1), V_N1)
        return r

    def make_and_init_load_q_pipeline(self, load_q_mbar_ptr):
        load_q_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.load_warp_group_id]),
        )
        load_q_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_warps_per_warp_group,
        )
        return pipeline.PipelineTmaAsync.create(
            barrier_storage=load_q_mbar_ptr,
            num_stages=self.q_stage,
            producer_group=load_q_producer_group,
            consumer_group=load_q_consumer_group,
            tx_count=self.tma_copy_q_bytes,
            defer_sync=True,
        ).make_participants()

    def make_and_init_load_kv_pipeline(self, load_kv_mbar_ptr):
        load_kv_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.load_warp_group_id]),
        )
        load_kv_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_mma_warp_groups * self.num_warps_per_warp_group,
        )
        return pipeline.PipelineTmaAsync.create(
            barrier_storage=load_kv_mbar_ptr,
            num_stages=self.kv_stage,
            producer_group=load_kv_producer_group,
            consumer_group=load_kv_consumer_group,
            tx_count=self.tma_copy_kv_bytes,
            defer_sync=True,
        ).make_participants()

    def make_and_init_tma_store_pipeline(self):
        tma_store_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            1,
        )
        return pipeline.PipelineTmaStore.create(
            num_stages=self.epi_stage,
            producer_group=tma_store_producer_group,
        )

    def make_and_init_order_barrier(self, order_mbar_ptr, group_id):
        StagesPerMathWarpGroup = 1
        return pipeline.PipelineOrder.create(
            barrier_storage=order_mbar_ptr,
            depth=StagesPerMathWarpGroup,
            length=self.num_mma_warp_groups,
            group_id=group_id,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.num_threads_per_warp_group,
            ),
            defer_sync=True,
        )

    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor: cute.Tensor,
        smem_layout_staged: cute.ComposedLayout,
        smem_tile: tuple[int, int],
        mcast_dim: int,
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        """Build a staged global-to-shared TMA copy atom and tensor."""
        op = (
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
        )

        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op,
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=mcast_dim,
        )
        return tma_atom, tma_tensor

    @staticmethod
    def can_implement(
        q_shape: Tuple[int, int, int, int],
        k_shape: Tuple[int, int, int, int],
        in_dtype: Type[cutlass.Numeric],
        out_dtype: Type[cutlass.Numeric],
        qk_acc_dtype: Type[cutlass.Numeric],
        pv_acc_dtype: Type[cutlass.Numeric],
        mma_tiler_mn: Tuple[int, int],
        is_persistent: bool,
        scale_softmax: float,
        window_size: Tuple[int, int],
        iterations: int,
    ) -> Tuple[bool, str]:
        """Validate shapes, dtypes, tiling, masking, and resource constraints."""

        # Unpack parameters
        b, s_q, h, d = q_shape
        b_, s_k, h_k, d_ = k_shape
        window_size_left, window_size_right = window_size

        if b != b_:
            return False, "q & k must have the same batch size"

        if d != d_:
            return False, "q & k must have the same head dimension"

        if window_size_left >= s_k - 1:
            return False, "window_size_left must be less than s_k_max - 1"
        if window_size_right >= s_q - 1:
            return False, "window_size_right must be less than s_q_max - 1"

        if h % h_k != 0:
            return False, "h must be divisible by h_k"

        if in_dtype not in {cutlass.Float8E4M3FN, cutlass.Float16, cutlass.BFloat16}:
            return False, "in_dtype must be Float16, BFloat16, Float8E4M3FN"

        if out_dtype not in {cutlass.Float8E4M3FN, cutlass.Float16, cutlass.BFloat16}:
            return False, "out_dtype must be Float16, BFloat16, Float8E4M3FN"

        if qk_acc_dtype not in {cutlass.Float32}:
            return False, "qk_acc_dtype must be Float32"

        if pv_acc_dtype not in {cutlass.Float32}:
            return False, "pv_acc_dtype must be Float32"

        if iterations < 1:
            return False, "iterations must be at least 1"

        if (
            in_dtype.width == 16
            and out_dtype.width == 16
            and (
                (d_ == 256 and mma_tiler_mn[1] >= 128)
                or (d_ == 128 and mma_tiler_mn[1] >= 256)
            )
        ) or (
            in_dtype.width == 8
            and out_dtype.width == 8
            and d_ == 256
            and mma_tiler_mn[1] >= 256
        ):
            return False, "not enough smem"

        if is_persistent and (
            (
                in_dtype.width == 16
                and out_dtype.width == 16
                and (
                    (d_ == 128 and mma_tiler_mn[1] >= 256)
                    or (d_ == 256 and mma_tiler_mn[1] > 32)
                )
            )
            or (
                in_dtype.width == 8
                and out_dtype.width == 8
                and d_ == 256
                and mma_tiler_mn[1] == 256
            )
        ):
            return False, "not supported persistent"

        return True, None


QWEN_BATCH = 1
QWEN_QUERY_HEADS = 64
QWEN_KV_HEADS = 8
QWEN_HEAD_DIM = 128
QWEN_CONTEXT = 128 * 1024

_QWEN3_KERNEL_CACHE = {}
_QWEN3_COMPILE_LOCK = threading.Lock()
_DECODE_STREAM_CACHE = {}
_DECODE_STREAM_LOCK = threading.Lock()


def _get_decode_streams(device, count: int):
    """Return reusable auxiliary streams for concurrent split-KV launches."""
    import torch

    if count <= 1:
        return ()
    key = (device.index, count)
    streams = _DECODE_STREAM_CACHE.get(key)
    if streams is None:
        with _DECODE_STREAM_LOCK:
            streams = _DECODE_STREAM_CACHE.get(key)
            if streams is None:
                streams = tuple(torch.cuda.Stream(device=device) for _ in range(count))
                _DECODE_STREAM_CACHE[key] = streams
    return streams


def _validate_qwen3_inputs(q, k, v, *, decode: bool, causal: bool):
    import torch

    api_name = "qwen3_decode_attention" if decode else "qwen3_prefill_attention"
    tensors = {"q": q, "k": k, "v": v}
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.ndim != 4:
            raise ValueError(
                f"{name} must have BHSD rank 4, got shape {tuple(tensor.shape)}"
            )
    if not isinstance(causal, bool):
        raise TypeError(f"causal must be a bool, got {type(causal).__name__}")
    if any(tensor.requires_grad for tensor in tensors.values()):
        raise ValueError(f"{api_name} is forward-only and does not support autograd")

    expected_q_len = 1 if decode else None
    if q.shape[:2] != (QWEN_BATCH, QWEN_QUERY_HEADS) or q.shape[3] != QWEN_HEAD_DIM:
        raise ValueError(
            "q must have shape (1, 64, seqlen, 128), "
            f"got {tuple(q.shape)}"
        )
    if expected_q_len is not None and q.shape[2] != expected_q_len:
        raise ValueError(f"decode q must have shape (1, 64, 1, 128), got {tuple(q.shape)}")
    for name, tensor in (("k", k), ("v", v)):
        if tensor.shape[:2] != (QWEN_BATCH, QWEN_KV_HEADS) or tensor.shape[3] != QWEN_HEAD_DIM:
            raise ValueError(
                f"{name} must have shape (1, 8, seqlen, 128), got {tuple(tensor.shape)}"
            )
    if q.shape[2] <= 0 or k.shape[2] <= 0:
        raise ValueError("sequence lengths must be positive")
    if k.shape[2] != v.shape[2]:
        raise ValueError(
            f"k and v sequence lengths differ: {k.shape[2]} and {v.shape[2]}"
        )
    if not decode and q.shape[2] != k.shape[2]:
        raise ValueError(
            "prefill q, k, and v must have equal sequence lengths, got "
            f"{q.shape[2]}, {k.shape[2]}, and {v.shape[2]}"
        )

    for name, tensor in tensors.items():
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"{name} must use torch.bfloat16, got {tensor.dtype}")
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if tensor.device != q.device:
            raise ValueError(
                f"q, k, and v must be on the same CUDA device; {name} is on "
                f"{tensor.device} while q is on {q.device}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous in BHSD layout")
        if tensor.data_ptr() % 16 != 0:
            raise ValueError(f"{name} storage must be 16-byte aligned")

    with torch.cuda.device(q.device):
        capability = torch.cuda.get_device_capability(q.device)
        if capability != (9, 0):
            raise RuntimeError(
                f"{api_name} requires Hopper SM90, got compute capability "
                f"{capability[0]}.{capability[1]}"
            )


def _normalize_sm_scale(sm_scale):
    import torch

    if sm_scale is None:
        return 1.0 / math.sqrt(QWEN_HEAD_DIM)
    if isinstance(sm_scale, (bool, torch.Tensor)):
        raise TypeError("sm_scale must be a positive finite real number or None")
    try:
        sm_scale = float(sm_scale)
    except (TypeError, ValueError) as exc:
        raise TypeError("sm_scale must be a positive finite real number or None") from exc
    log2_e = 1.4426950408889634074
    float32_info = torch.finfo(torch.float32)
    if not math.isfinite(sm_scale) or not float32_info.tiny <= sm_scale <= float32_info.max / log2_e:
        raise ValueError(
            "sm_scale must be positive and representable in float32 after "
            f"log2(e) scaling, got {sm_scale}"
        )
    return sm_scale


def _run_hopper_fmha(
    q,
    k,
    v,
    *,
    causal: bool,
    bottom_right: bool,
    sm_scale: float,
    prefill_config: str,
    write_lse: bool,
):
    """Launch one Hopper problem; Decode K/V may be zero-copy strided slices."""
    import torch

    config_name, block_n, kv_stage = _resolve_prefill_config(prefill_config)
    batch, query_heads, q_len, head_dim = q.shape
    kv_heads, kv_len = k.shape[1], k.shape[2]
    if query_heads % kv_heads != 0:
        raise ValueError("query head count must be divisible by KV head count")
    h_r = query_heads // kv_heads
    if batch != QWEN_BATCH or head_dim != QWEN_HEAD_DIM:
        raise ValueError("the local Hopper kernel requires batch=1 and head_dim=128")
    with torch.cuda.device(q.device):
        output = torch.empty_like(q)
        if write_lse:
            lse_storage = torch.empty(
                (batch, q_len, kv_heads, h_r, 1),
                dtype=torch.float32,
                device=q.device,
            )
            lse_kernel = lse_storage.permute(1, 4, 3, 2, 0)
        else:
            # Preserve the non-optional CuTe signature; this specialization
            # never accesses the one-element dummy LSE tensor.
            lse_storage = torch.empty((1, 1, 1, 1, 1), dtype=torch.float32, device=q.device)
            lse_kernel = lse_storage

        q_kernel = q.view(
            batch, kv_heads, h_r, q_len, head_dim
        ).permute(3, 4, 2, 1, 0)
        k_kernel = k.unsqueeze(2).permute(3, 4, 2, 1, 0)
        v_kernel = v.unsqueeze(2).permute(3, 4, 2, 1, 0)
        o_kernel = output.view(
            batch, kv_heads, h_r, q_len, head_dim
        ).permute(3, 4, 2, 1, 0)

        def as_cute_tensor(tensor, element_type):
            cute_tensor = from_dlpack(tensor, assumed_align=16)
            cute_tensor.element_type = element_type
            return cute_tensor.mark_layout_dynamic(leading_dim=1)

        q_tensor = as_cute_tensor(q_kernel, cutlass.BFloat16)
        k_tensor = as_cute_tensor(k_kernel, cutlass.BFloat16)
        v_tensor = as_cute_tensor(v_kernel, cutlass.BFloat16)
        o_tensor = as_cute_tensor(o_kernel, cutlass.BFloat16)
        lse_tensor = as_cute_tensor(lse_kernel, cutlass.Float32)

        log2_e = 1.4426950408889634074
        scale_softmax_log2 = sm_scale * log2_e
        window_size_left = None
        window_size_right = cutlass.Int32(0) if causal else None
        mask_type = (
            fmha_utils.MaskEnum.WINDOW_MASK_INFERENCE
            if bottom_right and causal
            else fmha_utils.MaskEnum.WINDOW_MASK
        )
        if not causal and kv_len % block_n != 0:
            mask_type = fmha_utils.MaskEnum.RESIDUAL_MASK
        torch_stream = torch.cuda.current_stream(q.device)
        current_stream = cuda.CUstream(torch_stream.cuda_stream)

        kernel_args = (
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            lse_tensor,
            scale_softmax_log2,
            sm_scale,
            1.0,
            window_size_left,
            window_size_right,
            current_stream,
        )
        cache_key = (
            q.device.index,
            query_heads,
            kv_heads,
            q_len,
            kv_len,
            causal,
            bottom_right,
            config_name,
            write_lse,
            tuple(q.stride()),
            tuple(k.stride()),
            tuple(v.stride()),
            sm_scale,
        )
        compiled_fmha = _QWEN3_KERNEL_CACHE.get(cache_key)
        if compiled_fmha is None:
            with _QWEN3_COMPILE_LOCK:
                compiled_fmha = _QWEN3_KERNEL_CACHE.get(cache_key)
                if compiled_fmha is None:
                    fmha = HopperFusedMultiHeadAttentionForward(
                        cutlass.Float32,
                        cutlass.Float32,
                        (64, block_n, QWEN_HEAD_DIM),
                        False,
                        mask_type,
                        kv_stage=kv_stage,
                        write_lse=write_lse,
                    )
                    compiled_fmha = cute.compile(fmha, *kernel_args)
                    _QWEN3_KERNEL_CACHE[cache_key] = compiled_fmha

        compiled_fmha(*kernel_args)
        q.record_stream(torch_stream)
        k.record_stream(torch_stream)
        v.record_stream(torch_stream)
        if write_lse:
            return output, lse_storage[..., 0]
        return output, None


def qwen3_prefill_attention(
    q, k, v, *, causal=True, sm_scale=None, prefill_config="auto"
):
    """Run Prefill without materializing the unused LSE output."""
    _validate_qwen3_inputs(q, k, v, decode=False, causal=causal)
    sm_scale = _normalize_sm_scale(sm_scale)
    output, _ = _run_hopper_fmha(
        q,
        k,
        v,
        causal=causal,
        bottom_right=False,
        sm_scale=sm_scale,
        prefill_config=prefill_config,
        write_lse=False,
    )
    return output


def _normalize_split_candidates(candidates: Sequence[int]) -> tuple[int, ...]:
    if isinstance(candidates, (str, bytes)):
        raise TypeError("split_candidates must be a sequence of positive integers")
    normalized = []
    for candidate in candidates:
        if isinstance(candidate, bool) or not isinstance(candidate, int):
            raise TypeError("split_candidates must contain only positive integers")
        if candidate <= 0:
            raise ValueError("split_candidates must contain only positive integers")
        if candidate not in normalized:
            normalized.append(candidate)
    if not normalized:
        raise ValueError("split_candidates must not be empty")
    return tuple(sorted(normalized))


def _select_decode_splits(
    seqlen: int,
    num_splits: Optional[int],
    split_candidates: Sequence[int],
    *,
    block_n: int = 128,
) -> int:
    candidates = _normalize_split_candidates(split_candidates)
    max_splits = max(1, math.ceil(seqlen / block_n))
    if num_splits is None or num_splits == 0:
        # Each split contributes eight CTAs. On the fixed 78-SM H20, choose the
        # smallest candidate reaching 85% of the best wave efficiency.
        num_sms = 78
        valid = (1, *tuple(value for value in candidates if value <= max_splits))
        efficiencies = {
            value: (QWEN_KV_HEADS * value)
            / (math.ceil(QWEN_KV_HEADS * value / num_sms) * num_sms)
            for value in valid
        }
        best = max(efficiencies.values())
        requested = next(
            value for value in valid if efficiencies[value] >= 0.85 * best
        )
    else:
        if isinstance(num_splits, bool) or not isinstance(num_splits, int):
            raise TypeError("num_splits must be a non-negative integer or None")
        if num_splits < 0:
            raise ValueError("num_splits must be non-negative")
        requested = num_splits
    return max(1, min(requested, max_splits))


def _continuous_kv_ranges(
    seqlen: int, num_splits: int, *, block_n: int = 128
) -> tuple[tuple[int, int], ...]:
    num_tiles = math.ceil(seqlen / block_n)
    num_splits = max(1, min(num_splits, num_tiles))
    ranges = []
    for split_idx in range(num_splits):
        tile_begin = split_idx * num_tiles // num_splits
        tile_end = (split_idx + 1) * num_tiles // num_splits
        begin = tile_begin * block_n
        end = min(tile_end * block_n, seqlen)
        if begin < end:
            ranges.append((begin, end))
    return tuple(ranges)


def _combine_decode_partials(partial_outputs, partial_lses, *, output_dtype):
    """Combine Pack-GQA split outputs using stable FP32 LSE weights."""
    import torch

    o_partial = torch.stack([output.float() for output in partial_outputs], dim=0)
    lse_partial = torch.stack(
        [lse.squeeze(-1).permute(0, 2, 1) for lse in partial_lses], dim=0
    ).float()
    global_lse = torch.logsumexp(lse_partial, dim=0)
    weights = torch.exp(lse_partial - global_lse.unsqueeze(0))
    combined = (o_partial * weights.unsqueeze(-1)).sum(dim=0)
    return (
        combined.contiguous()
        .view(QWEN_BATCH, QWEN_QUERY_HEADS, 1, QWEN_HEAD_DIM)
        .to(dtype=output_dtype)
    )


def _qwen3_decode_attention_impl(
    q,
    k,
    v,
    *,
    causal: bool,
    sm_scale: float,
    num_splits: Optional[int],
    split_candidates: Sequence[int],
    return_stats: bool,
):
    """Run staged split-KV Decode behind the replaceable private interface."""
    import torch

    _, block_n, _ = _resolve_prefill_config(DECODE_KERNEL_CONFIG)
    actual_splits = _select_decode_splits(
        k.shape[2], num_splits, split_candidates, block_n=block_n
    )
    ranges = _continuous_kv_ranges(k.shape[2], actual_splits, block_n=block_n)

    # Pack each KV head's eight query heads into useful logical M rows while
    # keeping K/V as zero-copy cache views.
    h_r = QWEN_QUERY_HEADS // QWEN_KV_HEADS
    q_packed = q.view(QWEN_BATCH, QWEN_KV_HEADS, h_r, QWEN_HEAD_DIM)
    partial_outputs = []
    partial_lses = []

    def launch_partial(begin, end):
        return _run_hopper_fmha(
            q_packed,
            k[:, :, begin:end, :],
            v[:, :, begin:end, :],
            # Packed rows are heads, so each row attends the complete split;
            # the global combination restores full-cache attention.
            causal=False,
            bottom_right=False,
            sm_scale=sm_scale,
            prefill_config=DECODE_KERNEL_CONFIG,
            write_lse=True,
        )

    streams = _get_decode_streams(q.device, len(ranges))
    if streams:
        caller_stream = torch.cuda.current_stream(q.device)
        for stream, (begin, end) in zip(streams, ranges):
            stream.wait_stream(caller_stream)
            with torch.cuda.stream(stream):
                partial_output, partial_lse = launch_partial(begin, end)
            partial_outputs.append(partial_output)
            partial_lses.append(partial_lse)
        for stream in streams:
            caller_stream.wait_stream(stream)
        # wait_stream orders execution but not allocator lifetime; record the
        # partial storages before the asynchronous combine reads them.
        for partial_output, partial_lse in zip(partial_outputs, partial_lses):
            partial_output.record_stream(caller_stream)
            partial_lse.record_stream(caller_stream)
    else:
        partial_output, partial_lse = launch_partial(*ranges[0])
        partial_outputs.append(partial_output)
        partial_lses.append(partial_lse)

    output = _combine_decode_partials(
        partial_outputs, partial_lses, output_dtype=q.dtype
    )
    if not return_stats:
        return output
    return output, {
        "num_splits": len(ranges),
        "ranges": ranges,
        "partial_wgmma_launches": len(ranges),
        "packed_query_heads_per_kv_head": h_r,
        "combine": "FP32 log-sum-exp weights over BF16 partial outputs",
    }


def qwen3_decode_attention(
    q,
    k,
    v,
    *,
    causal=True,
    sm_scale=None,
    num_splits=0,
    split_candidates=DEFAULT_DECODE_SPLIT_CANDIDATES,
):
    """Run latest-token Decode over the complete KV cache."""
    _validate_qwen3_inputs(q, k, v, decode=True, causal=causal)
    sm_scale = _normalize_sm_scale(sm_scale)
    return _qwen3_decode_attention_impl(
        q,
        k,
        v,
        causal=causal,
        sm_scale=sm_scale,
        num_splits=num_splits,
        split_candidates=split_candidates,
        return_stats=False,
    )


def qwen3_attention(
    q,
    k,
    v,
    *,
    causal=True,
    sm_scale=None,
    num_splits=0,
    split_candidates=DEFAULT_DECODE_SPLIT_CANDIDATES,
    prefill_config="auto",
):
    """Dispatch fixed-shape Qwen3 attention to decode or prefill."""
    q_len = q.shape[2] if getattr(q, "ndim", None) == 4 else None
    if q_len == 1:
        return qwen3_decode_attention(
            q,
            k,
            v,
            causal=causal,
            sm_scale=sm_scale,
            num_splits=num_splits,
            split_candidates=split_candidates,
        )
    return qwen3_prefill_attention(
        q,
        k,
        v,
        causal=causal,
        sm_scale=sm_scale,
        prefill_config=prefill_config,
    )


def main() -> None:
    """CLI entry point; diagnostics live in the ``_attention_diagnostics`` module."""
    try:
        from . import _attention_diagnostics as _diag
    except ImportError:
        import _attention_diagnostics as _diag
    _diag.main_optimized(sys.modules[__name__])


if __name__ == "__main__":
    main()
