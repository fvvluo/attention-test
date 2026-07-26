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

"""Optimized Qwen3-32B Prefill and Decode attention for NVIDIA Hopper.

The CuTe DSL kernels fuse QK, online softmax, and PV with TMA transfers,
warp-specialized Hopper WGMMA, BF16 inputs/outputs, and FP32 accumulators. The
fixed target is B=1, S=131072, Hq=64, Hkv=8, D=128. Prefill uses a tuned
N192/KV3 pipeline and omits unused LSE work. Decode packs each group of eight
query heads into logical M rows, launches continuous split-KV work concurrently,
and combines split outputs with FP32 log-sum-exp weights.

This implementation is adapted from NVIDIA CUTLASS's Hopper FMHA CuTe DSL
example and retains its BSD-3-Clause license. Run ``--help`` for correctness,
full-length validation, and benchmark modes.
"""

import argparse
import math
import os
import sys
import threading
import time
from typing import Type, Tuple, Optional, Sequence

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute

import cutlass.cute.nvgpu.warpgroup as warpgroup
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
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


# Keep the original N128/KV5 baseline and the candidate configurations used in
# H20 tuning. ``auto`` below selects only a candidate that passed small-shape,
# sampled 128K, and interleaved 128K performance validation.
PREFILL_CONFIGS = {
    "n128-kv5": (128, 5),
    "n128-kv4": (128, 4),
    "n160-kv3": (160, 3),
    "n160-kv4": (160, 4),
    "n192-kv3": (192, 3),
}
# N192/KV3 passed small-shape and sampled 128K correctness and was the most
# stable H20 winner in an interleaved 128K A/B run (1977.91 ms median versus
# 2000.12 ms for N128/KV5). The original N128/KV5 preset remains available.
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
        """Initializes the configuration for a Hopper Fused Multi-Head Attention (FMHA) kernel.

        This configuration includes several key aspects:

        1.  Data Type Settings:
            - qk_acc_dtype: Data type for Q*K^T matrix multiplication accumulator
            - pv_acc_dtype: Data type for P*V matrix multiplication accumulator

        2.  MMA Instruction Settings:
            - mma_tiler: The (M, N, K) shape of the MMA instruction unit
            - qk_mma_tiler: MMA shape for Q*K^T computation
            - pv_mma_tiler: MMA shape for P*V computation

        3.  Kernel Execution Mode:
            - is_persistent: Boolean indicating whether to use persistent kernel mode
            - mask_type: Specifies the type of mask to use (no mask, residual mask, or causal mask)
            - window_size_left/right: Sliding window parameters for attention masking

        :param qk_acc_dtype: Data type for Q*K^T matrix multiplication accumulator
        :type qk_acc_dtype: Type[cutlass.Numeric]
        :param pv_acc_dtype: Data type for P*V matrix multiplication accumulator
        :type pv_acc_dtype: Type[cutlass.Numeric]
        :param mma_tiler: The (M, N, K) shape of the MMA instruction
        :type mma_tiler: Tuple[int, int, int]
        :param is_persistent: Whether to use persistent kernel mode
        :type is_persistent: bool
        :param mask_type: Type of mask to use
        :type mask_type: fmha_utils.MaskEnum
        """

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
        num_load_warp_groups = 1
        self.num_threads_per_warp_group = 128
        max_threads_per_block = (
            self.num_mma_warp_groups + num_load_warp_groups
        ) * self.num_threads_per_warp_group
        self.threads_per_cta = max_threads_per_block
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
        """The device kernel implementation of the Fused Multi-Head Attention for Hopper architecture.

        This kernel coordinates multiple specialized warps to perform different phases of the FMHA computation:
        1. Load warp group: Loads Q, K, V data from global memory to shared memory using TMA
        2. Comput warps groups: Performs matrix multiplications (Q*K^T and P*V) using Hopper TensorCores,
        then compute softmax normalization on attention scores with numerical stability.
        Handle final output transformation and storage.

        The kernel implements a complex pipeline with overlapping computation and memory operations,
        using tensor memory access (TMA) for efficient data loading, warp specialization for different
        computation phases, and optional attention masking for causal or residual attention patterns.

        Key optimizations include:
        - Warp group specialization for load, compute/epilogue phases
        - Pipeline stages between different warps for overlapping computation and memory access
        - Efficient shared memory layouts optimized for Hopper architecture
        - Support for different precision data types and accumulation types
        - Optional causal masking for autoregressive models
        - Sliding window attention masking for efficient long sequence processing

        :param qk_tiled_mma: Tiled MMA for Q*K^T matrix multiplication
        :type qk_tiled_mma: cute.TiledMma
        :param pv_tiled_mma: Tiled MMA for P*V matrix multiplication
        :type pv_tiled_mma: cute.TiledMma
        :param tma_atom_q: TMA copy atom for query tensor loading
        :type tma_atom_q: cute.CopyAtom
        :param mQ_qdl: Partitioned query tensor for TMA loading
        :type mQ_qdl: cute.Tensor
        :param tma_atom_k: TMA copy atom for key tensor loading
        :type tma_atom_k: cute.CopyAtom
        :param mK_kdl: Partitioned key tensor for TMA loading
        :type mK_kdl: cute.Tensor
        :param tma_atom_v: TMA copy atom for value tensor loading
        :type tma_atom_v: cute.CopyAtom
        :param mV_dkl: Partitioned value tensor for TMA loading
        :type mV_dkl: cute.Tensor
        :param tma_atom_o: TMA copy atom for output tensor storage
        :type tma_atom_o: cute.CopyAtom
        :param mO_qdl: Partitioned output tensor for TMA storage
        :type mO_qdl: cute.Tensor
        :param mLse_qdl: Tensor for lse
        :type mLse_qdl: cute.Tensor
        :param scale_softmax_log2: The log2 scale factor for softmax computation
        :type scale_softmax_log2: cutlass.Float32
        :param scale_softmax: The scale factor for softmax (currently unused)
        :type scale_softmax: cutlass.Float32
        :param scale_output: The scale factor for the final output
        :type scale_output: cutlass.Float32
        :param window_size_left: Left-side sliding window size for attention masking
        :type window_size_left: Optional[cutlass.Int32]
        :param window_size_right: Right-side sliding window size for attention masking
        :type window_size_right: Optional[cutlass.Int32]
        :param q_smem_layout_staged: Shared memory layout for query tensor with staging
        :type q_smem_layout_staged: cute.ComposedLayout
        :param k_smem_layout_staged: Shared memory layout for key tensor with staging
        :type k_smem_layout_staged: cute.ComposedLayout
        :param v_smem_layout_staged: Shared memory layout for value tensor with staging
        :type v_smem_layout_staged: cute.ComposedLayout
        :param o_smem_layout_staged: Shared memory layout for output tensor with staging
        :type o_smem_layout_staged: cute.ComposedLayout
        :param tile_sched_params: Scheduling parameters for work distribution across blocks
        :type tile_sched_params: fmha_utils.FmhaStaticTileSchedulerParams
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
        """
        Final processing step for FMHA that computes log-sum-exp (LSE) and scales the output.

        This function performs the following operations:
        1. Reduces the attention sums across warps using butterfly shuffle
        2. Computes the log-sum-exp (LSE) for numerical stability
        3. Applies softmax scaling and output scaling to the accumulated values
        4. Handles edge cases like zero sums and NaN values

        :param s_max: Maximum attention scores for each position (for numerical stability)
        :type s_max: cute.Tensor
        :param a_sum: Sum of attention scores after softmax
        :type a_sum: cute.Tensor
        :param acc_pv: Accumulated P*V values from the attention computation
        :type acc_pv: cute.ThrMma
        :param tiled_mma_pv: Tiled MMA for P*V computation
        :type tiled_mma_pv: cute.TiledMma
        :param scale_softmax: Scaling factor for softmax computation
        :type scale_softmax: cutlass.Float32
        :param scale_output: Scaling factor for final output
        :type scale_output: cutlass.Float32

        :return: Log-sum-exp values for each position
        :rtype: cute.Tensor
        """
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
        """Create TMA atoms and tensors for input tensors.

        :param tensor: Input tensor (A or B)
        :type tensor: cute.Tensor
        :param smem_layout_staged: Shared memory layout for the tensor
        :type smem_layout_staged: cute.ComposedLayout
        :param smem_tile: Shared memory tile shape
        :type smem_tile: Tuple[int, int]
        :param mcast_dim: Multicast dimension
        :type mcast_dim: int

        :return: TMA atom and tensor
        :rtype: Tuple[cute.CopyAtom, cute.Tensor]
        """
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
        """Check if the FMHA kernel can be implemented with the given parameters.

        This method validates that the input parameters are compatible with the Hopper
        Fused Multi-Head Attention implementation. It checks tensor shapes, data types,
        window sizes, and other constraints to ensure the kernel can be successfully
        compiled and executed.

        :param q_shape: Query tensor shape (B, S_q, H, D) where B=batch size, S_q=query sequence length,
                       H=number of heads, D=head dimension
        :type q_shape: Tuple[int, int, int, int]
        :param k_shape: Key tensor shape (B, S_k, H_k, D) where B=batch size, S_k=key sequence length,
                       H_k=number of key heads, D=head dimension
        :type k_shape: Tuple[int, int, int, int]
        :param in_dtype: Input data type for query, key and value tensors
        :type in_dtype: Type[cutlass.Numeric]
        :param out_dtype: Output data type for attention output
        :type out_dtype: Type[cutlass.Numeric]
        :param qk_acc_dtype: Accumulator data type for query-key matrix multiplication
        :type qk_acc_dtype: Type[cutlass.Numeric]
        :param pv_acc_dtype: Accumulator data type for probability-value matrix multiplication
        :type pv_acc_dtype: Type[cutlass.Numeric]
        :param mma_tiler_mn: Matrix multiply accumulate tile shape (M, N)
        :type mma_tiler_mn: Tuple[int, int]
        :param is_persistent: Whether to use persistent kernel optimization
        :type is_persistent: bool
        :param scale_softmax: Attention score scaling factor
        :type scale_softmax: float
        :param window_size: Sliding window size (left, right) for attention masking
        :type window_size: Tuple[int, int]
        :param iterations: Number of iterations to run for performance testing
        :type iterations: int

        :return: Tuple of (can_implement, error_message) where can_implement is True if the kernel
                 can be implemented, False otherwise, and error_message contains the reason for failure
        :rtype: Tuple[bool, str]
        """

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


class HopperDecode128KSplitKVForward(HopperFusedMultiHeadAttentionForward):
    """Single-launch split-KV Decode partial kernel for the fixed 128K contract.

    Fixed workload: B=1, Hq=64, Hkv=8, ratio=8, q_len=1, KV=131072, D=128,
    BF16 inputs, FP32 accumulators, latest-token full-cache attention over a
    contiguous BHSD KV cache.  One kernel launch covers grid=(split, kv_head):
    ``blockIdx.x`` selects the continuous split of N tiles, ``blockIdx.y`` the
    KV head, and each CTA packs the eight query heads of that KV head as
    logical M rows.  Every CTA writes a normalized BF16 partial O and an FP32
    partial LSE for its split; a host-side FP32 log-sum-exp combine merges the
    splits afterwards.

    Only ``__call__``/``kernel`` are overridden; the mainloop helpers
    (``compute``/``softmax_step``/``tail`` and the pipeline initializers) are
    inherited unchanged.  The general kernel's tile scheduler and fused-mask
    trip computation are replaced by a direct device-side continuous tile
    range, because the fixed contract has no causal, window, or residue work
    (131072 is an exact multiple of the 128-wide N tile).
    """

    def __init__(self, qk_acc_dtype, pv_acc_dtype, mma_tiler, kv_stage, num_splits: int):
        if num_splits <= 0:
            raise ValueError("num_splits must be positive")
        super().__init__(
            qk_acc_dtype,
            pv_acc_dtype,
            mma_tiler,
            False,
            fmha_utils.MaskEnum.WINDOW_MASK,
            kv_stage=kv_stage,
            write_lse=True,
        )
        self.num_splits = num_splits

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
        stream: cuda.CUstream,
    ):
        # q: (ratio, D, Hkv), k-major
        # k: (KV, D, Hkv), k-major
        # v: (D, KV, Hkv), d-major
        # o: (ratio, D, Hkv, split), k-major BF16 partial storage
        # lse: (ratio, Hkv, split) FP32; a stride-0 fake-D mode is inserted
        #      below so the inherited epilogue/LSE helpers keep their shape.
        self.q_dtype = q.element_type
        self.k_dtype = k.element_type
        self.v_dtype = v.element_type
        self.o_dtype = o.element_type

        # (ratio, fake_D, Hkv, split); fake_D stride 0, stored only at column 0.
        lse = cute.make_tensor(
            lse.iterator,
            cute.make_layout(
                (lse.shape[0], self.pv_mma_tiler[1], lse.shape[1], lse.shape[2]),
                stride=(lse.stride[0], 0, lse.stride[1], lse.stride[2]),
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

        # One launch covers all (split, kv_head) pairs.
        grid = (o.shape[3], o.shape[2], 1)

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
                q_smem_layout_staged,
                k_smem_layout_staged,
                v_smem_layout_staged,
                o_smem_layout_staged,
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
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        o_smem_layout_staged: cute.ComposedLayout,
    ):
        """Device kernel for the fixed 128K single-launch split-KV Decode.

        Same warp-specialized pipeline as the general kernel, but each CTA
        handles exactly one (split, kv_head) work item: ``blockIdx.x`` selects
        the split and ``blockIdx.y`` the KV head.  The continuous N-tile range
        ``[tile_begin, tile_end)`` is computed on device from the split index,
        so Python never slices the KV cache and no auxiliary streams are used.
        The fixed contract has no causal/window/residue mask work; only the
        TMA clamp handles the eight valid packed M rows.
        """

        tidx, _, _ = cute.arch.thread_idx()
        split_idx, kv_head_idx, _ = cute.arch.block_idx()

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

        sO = cute.make_tensor(
            cute.recast_ptr(sQ.iterator, o_smem_layout_staged.inner, self.o_dtype),
            o_smem_layout_staged.outer,
        )

        seqlen_q = mQ_qdl.shape[0]
        seqlen_k = mK_kdl.shape[0]

        # Continuous device-side split of the N tiles.  Coverage of
        # [0, num_n_tiles) across splits is exact, disjoint, and gap-free.
        num_n_tiles = cute.ceil_div(seqlen_k, self.qk_mma_tiler[1])
        tile_begin = split_idx * num_n_tiles // self.num_splits
        tile_end = (split_idx + 1) * num_n_tiles // self.num_splits
        split_k_tiles = tile_end - tile_begin

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
        # to all threads.
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

            q0_index = 0
            k_index = tile_begin
            q_tile_count = self.num_mma_warp_groups
            k_tile_count = 2 * split_k_tiles

            _tQgQ = tQgQ_qdl[(None, None, 0, kv_head_idx)]
            tQgQ = cute.domain_offset((0, 0), _tQgQ)

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

                tKgK = tKgK_kdl[(None, None, 0, kv_head_idx)]
                tVgV = tVgV_dkl[(None, 0, None, kv_head_idx)]

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

        # Mainloop
        if (
            warp_group_idx == self.compute_epilogue_0_warp_group_id
            or warp_group_idx == self.compute_epilogue_1_warp_group_id
        ):
            cute.arch.setmaxregister_increase(self.num_regs_mma)

            kOuterLoads = 1

            cP = cute.make_identity_tensor((mQ_qdl.shape[0], seqlen_k))
            gPcP = cute.local_tile(cP, self.qk_mma_tiler[:2], (None, None))

            for i in cutlass.range((warp_group_idx - 1) * kOuterLoads, unroll=1):
                load_q_consumer.advance()

            # One work item per CTA: the M dimension holds the eight packed
            # query-head rows, so each compute warpgroup owns one 64-row half
            # of the single logical M tile.
            _wg_coord_0 = warp_group_idx - 1
            _wg_coord_1 = 0

            wg_coord = (_wg_coord_0, _wg_coord_1, kv_head_idx)

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

            kv_offset = tile_begin

            # mapping into QK accumulator
            tPcP = cute.slice_(ptPcP, (None, None, None, wg_coord[0], kv_offset))
            kv_offset += 1

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

            # The fixed latest-token contract applies no causal, window, or
            # residue mask: every packed row visits its whole split.
            s_max, a_sum = self.softmax_step(
                False,
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
                None,
                None,
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

            load_kv_consumer, k_tile_count, kv_offset, s_max, a_sum = self.compute(
                False,
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
                seqlen_k,
                seqlen_q,
                qk_acc_shape,
                None,
                None,
            )

            # Wait for the pipeline MMAs to drain
            cute.nvgpu.warpgroup.wait_group(0)

            # Normalize the output and produce the per-split LSE.
            lse = self.tail(
                s_max, a_sum, acc_pv, pv_tiled_mma, scale_softmax, scale_output
            )

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

            # Store the per-split log-sum-exp (LSE); the split coordinate is
            # part of the destination layout.
            thr_mma = pv_tiled_mma.get_slice(tidx)

            gLSE_full = cute.local_tile(
                mLse_qdl, self.pv_mma_tiler[:2], (None, None, None, None)
            )
            gLSE = cute.slice_(
                gLSE_full,
                (None, None, wg_coord[0], wg_coord[1], kv_head_idx, split_idx),
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
                (wg_coord[0], 0, kv_head_idx, split_idx),
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
        return


def run(
    q_shape: Tuple[int, int, int, int],
    k_shape: Tuple[int, int, int, int],
    in_dtype: Type[cutlass.Numeric],
    out_dtype: Type[cutlass.Numeric],
    qk_acc_dtype: Type[cutlass.Numeric],
    pv_acc_dtype: Type[cutlass.Numeric],
    mma_tiler_mn: Tuple[int, int],
    is_persistent: bool,
    is_causal: bool,
    bottom_right_align: bool,
    scale_q: float,
    scale_k: float,
    scale_v: float,
    inv_scale_o: float,
    scale_softmax: float,
    window_size: Tuple[int, int],
    tolerance: float,
    warmup_iterations: int,
    iterations: int,
    skip_ref_check: bool,
    use_cold_l2: bool = False,
    execute_benchmark: bool = True,
    input_pattern: str = "random",
    kv_stage: int = 5,
    write_lse: bool = False,
    **kwargs,
):
    """Execute Fused Multi-Head Attention (FMHA) on Hopper architecture and validate results.

    This function creates random input tensors for query, key, and value, then performs the
    complete FMHA computation pipeline. It supports configurable data types, tiling parameters,
    and various attention masking options. Results can be validated against a PyTorch reference
    implementation or run multiple times for performance measurement.

    The implementation leverages specialized tensor memory operations and efficient math
    operations optimized for Hopper architecture, including pipelined computation stages
    for maximum throughput.

    :param q_shape: Query tensor shape (B, S_q, H, D) where B=batch size, S_q=query sequence length,
                    H=number of heads, D=head dimension.
                    If S_q is a tuple, it is the variable sequence length.
    :type q_shape: Tuple[int, int, int, int] | Tuple[int, Tuple[int, ...], int, int]
    :param k_shape: Key tensor shape (B, S_k, H_k, D) where B=batch size, S_k=key sequence length,
                    H_k=number of key heads (H must be divisible by H_k), D=head dimension.
                    If S_k is a tuple, it is the variable sequence length.
    :type k_shape: Tuple[int, int, int, int] | Tuple[int, Tuple[int, ...], int, int]
    :param in_dtype: Input data type for query, key and value tensors
    :type in_dtype: Type[cutlass.Numeric]
    :param out_dtype: Output data type for attention output
    :type out_dtype: Type[cutlass.Numeric]
    :param qk_acc_dtype: Accumulator data type for query-key matrix multiplication
    :type qk_acc_dtype: Type[cutlass.Numeric]
    :param pv_acc_dtype: Accumulator data type for probability-value matrix multiplication
    :type pv_acc_dtype: Type[cutlass.Numeric]
    :param mma_tiler_mn: Matrix multiply accumulate tile shape (M, N)
    :type mma_tiler_mn: Tuple[int, int]
    :param is_persistent: Whether to use persistent kernel optimization
    :type is_persistent: bool
    :param is_causal: Whether to apply causal masking
    :type is_causal: bool
    :param bottom_right_align: Whether to use bottom right align, under this settion, the end of q is aligned with the end of k.
    :type bottom_right_align: bool
    :param scale_q: Scaling factor for query tensor
    :type scale_q: float
    :param scale_k: Scaling factor for key tensor
    :type scale_k: float
    :param scale_v: Scaling factor for value tensor
    :type scale_v: float
    :param inv_scale_o: Inverse scaling factor for output tensor
    :type inv_scale_o: float
    :param scale_softmax: Attention score scaling factor (defaults to 1/sqrt(D) if set to 0)
    :type scale_softmax: float
    :param window_size: Sliding window size (left, right) for attention masking. Controls which positions each query can attend to. Negative values disable windowing.
    :type window_size: Tuple[int, int]
    :param tolerance: Maximum acceptable error for validation
    :type tolerance: float
    :param warmup_iterations: Number of warmup iterations
    :type warmup_iterations: int
    :param iterations: Number of iterations to run for performance testing
    :type iterations: int
    :param skip_ref_check: Skip validation against reference implementation
    :type skip_ref_check: bool
    :param use_cold_l2: Whether to use circular buffer strategy to ensure cold L2 cache
    :type use_cold_l2: bool

    :raises ValueError: If input shapes are incompatible or head dimension is unsupported
    :raises RuntimeError: If GPU is unavailable for computation
    :return: Execution time of the FMHA kernel in microseconds
    :rtype: float
    """
    import torch
    import cutlass.torch as cutlass_torch

    print("Running Hopper SM90 FMHA test with:")
    print(f"  q_shape: {q_shape}")
    print(f"  k_shape: {k_shape}")
    print(f"  in_dtype: {in_dtype}")
    print(f"  out_dtype: {out_dtype}")
    print(f"  qk_acc_dtype: {qk_acc_dtype}")
    print(f"  pv_acc_dtype: {pv_acc_dtype}")
    print(f"  mma_tiler_mn: {mma_tiler_mn}")
    print(f"  is_persistent: {is_persistent}")
    print(f"  is_causal: {is_causal}")
    print(f"  bottom_right_align: {bottom_right_align}")
    print(f"  scale_q: {scale_q}")
    print(f"  scale_k: {scale_k}")
    print(f"  scale_v: {scale_v}")
    print(f"  inv_scale_o: {inv_scale_o}")
    print(f"  scale_softmax: {scale_softmax}")
    print(f"  window_size: {window_size}")
    print(f"  tolerance: {tolerance}")
    print(f"  skip_ref_check: {skip_ref_check}")
    print(f"  use_cold_l2: {use_cold_l2}")
    print(f"  kv_stage: {kv_stage}")
    print(f"  write_lse: {write_lse}")

    # Prepare pytorch tensors: Q, K, V (random from 0 to 2) and O (all zero)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")
    torch.cuda.reset_peak_memory_stats()

    ret, msg = HopperFusedMultiHeadAttentionForward.can_implement(
        q_shape,
        k_shape,
        in_dtype,
        out_dtype,
        qk_acc_dtype,
        pv_acc_dtype,
        mma_tiler_mn,
        is_persistent,
        scale_softmax,
        window_size,
        iterations,
    )
    if not ret:
        raise TypeError(msg)

    # Unpack parameters
    b, s_q, h, d = q_shape
    b_, s_k, h_k, d_ = k_shape
    window_size_left, window_size_right = window_size
    if window_size_left == -1:
        window_size_left = None
    if window_size_right == -1:
        window_size_right = None

    h_r = h // h_k

    torch.manual_seed(1111)

    def create_and_permute_tensor(
        b, s, h_k, h_r, d, dtype, is_dynamic_layout=True, tensor_name=""
    ):
        # (b, s, h_k, h_r, d) -> (s, d, h_r, h_k, b)
        # torch SPDA order is (h_k, h_r), then kernel is (h_r, h_k)
        shape = (b, s, h_k, h_r, d)
        permute_order = (1, 4, 3, 2, 0)
        is_fp8 = dtype in {cutlass.Float8E4M3FN}
        leading_dim = 1
        if is_fp8 and tensor_name == "v":
            permute_order = (4, 1, 3, 2, 0)
            leading_dim = 0
            shape = (b, d, h_k, h_r, s)

        # torch does not support fp8 type
        torch_dtype = cutlass.torch.dtype(dtype) if not is_fp8 else torch.int8

        # Create dtype torch tensor (cpu)
        torch_tensor_cpu = cutlass_torch.create_and_permute_torch_tensor(
            shape,
            torch_dtype,
            permute_order=permute_order,
            init_type=cutlass.torch.TensorInitType.RANDOM,
            init_config=cutlass.torch.RandomInitConfig(
                min_val=-2,
                max_val=2,
            ),
        )
        # Create dtype torch tensor (gpu)
        torch_tensor_gpu = torch_tensor_cpu.cuda()

        f32_torch_tensor = None
        if not skip_ref_check or is_fp8:
            f32_torch_tensor = torch_tensor_cpu.to(dtype=torch.float32)

        # BF16 tensors already have the desired values and can be consumed directly.
        cute_tensor = from_dlpack(torch_tensor_gpu, assumed_align=16)
        cute_tensor.element_type = dtype
        if is_dynamic_layout:
            cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)
        if f32_torch_tensor is not None:
            cute_tensor = cutlass_torch.convert_cute_tensor(
                f32_torch_tensor,
                cute_tensor,
                dtype,
                is_dynamic_layout=is_dynamic_layout,
            )

        return f32_torch_tensor, cute_tensor, torch_tensor_gpu

    q_ref, q_tensor, q_torch = create_and_permute_tensor(
        b, s_q, h_k, h_r, d, in_dtype, is_dynamic_layout=True
    )
    k_ref, k_tensor, k_torch = create_and_permute_tensor(
        b, s_k, h_k, 1, d, in_dtype, is_dynamic_layout=True
    )
    v_ref, v_tensor, v_torch = create_and_permute_tensor(
        b, s_k, h_k, 1, d, in_dtype, is_dynamic_layout=True, tensor_name="v"
    )
    o_ref, o_tensor, o_torch = create_and_permute_tensor(
        b, s_q, h_k, h_r, d, out_dtype, is_dynamic_layout=True
    )
    if write_lse:
        lse_ref, lse_tensor, lse_torch = create_and_permute_tensor(
            b, s_q, h_k, h_r, 1, qk_acc_dtype, is_dynamic_layout=True
        )
    else:
        # CuTe currently requires the stable tensor argument in the callable
        # signature.  The no-LSE specialization never dereferences this single
        # FP32 element.
        lse_ref = None
        lse_torch = torch.empty((1, 1, 1, 1, 1), dtype=torch.float32, device="cuda")
        lse_tensor = from_dlpack(lse_torch, assumed_align=16)
        lse_tensor.element_type = cutlass.Float32
        lse_tensor = lse_tensor.mark_layout_dynamic(leading_dim=1)

    if input_pattern not in {"random", "prefix"}:
        raise ValueError(f"unsupported input pattern: {input_pattern}")
    if input_pattern == "prefix":
        if not skip_ref_check:
            raise ValueError("prefix input is only used by the full-length sampled check")
        q_torch.zero_()
        k_torch.zero_()
        positions = (torch.arange(s_k, device=v_torch.device) % 17 - 8).float() / 8
        kv_offsets = torch.arange(h_k, device=v_torch.device).float() / 4
        values = positions[:, None, None, None, None] + kv_offsets[None, None, None, :, None]
        v_torch.copy_(values.expand_as(v_torch))
        o_torch.zero_()
        if write_lse:
            lse_torch.zero_()

    mma_tiler = (*mma_tiler_mn, d)

    mask_type = fmha_utils.MaskEnum.WINDOW_MASK
    if bottom_right_align:
        mask_type = fmha_utils.MaskEnum.WINDOW_MASK_INFERENCE
    if is_causal:
        window_size_right = 0
    elif window_size_left is None and window_size_right is None:
        if s_k % mma_tiler_mn[1] != 0:
            mask_type = fmha_utils.MaskEnum.RESIDUAL_MASK

    # To avoid mask out the whole row which results in NaN in softmax
    def check_seqlen_valid(
        s_q, s_k, window_size_left, window_size_right, bottom_right_align
    ):
        for i in range(s_q):
            offset = 0 if not bottom_right_align else s_k - s_q

            s_q_start = 0 if window_size_left is None else i + offset - window_size_left
            s_q_end = (
                s_q if window_size_right is None else i + offset + window_size_right
            )
            s_q_min = max(s_q_start, 0)
            s_q_max = min(s_q_end, s_k)

            if s_q_max - s_q_min == 0 and (i != 0 and i != s_q - 1):
                return False
        return True

    need_check_seqlen_valid = (
        window_size_left is not None or window_size_right is not None
    )
    if need_check_seqlen_valid and not check_seqlen_valid(
        s_q,
        s_k,
        window_size_left,
        window_size_right,
        bottom_right_align,
    ):
        raise ValueError("sliding window doesn't support current setting")

    fmha = HopperFusedMultiHeadAttentionForward(
        qk_acc_dtype,
        pv_acc_dtype,
        mma_tiler,
        is_persistent,
        mask_type,
        kv_stage=kv_stage,
        write_lse=write_lse,
    )

    # Get current CUDA stream from PyTorch
    torch_stream = torch.cuda.current_stream()
    # Get the raw stream pointer as a CUstream
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    if scale_softmax == 0.0:  # default to 1/sqrt(head_dim)
        scale_softmax = 1.0 / math.sqrt(q_shape[3])

    scale_softmax = scale_q * scale_k * scale_softmax

    LOG2_E = 1.4426950408889634074
    scale_softmax_log2 = scale_softmax * LOG2_E
    scale_output = scale_v * inv_scale_o

    print("Compiling kernel with cute.compile ...")
    start_time = time.time()
    # compile fmha kernel
    compiled_fmha = cute.compile(
        fmha,
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        lse_tensor,
        scale_softmax_log2,
        scale_softmax,
        scale_output,
        (
            window_size_left
            if window_size_left is None
            else cutlass.Int32(window_size_left)
        ),
        (
            window_size_right
            if window_size_right is None
            else cutlass.Int32(window_size_right)
        ),
        current_stream,
    )
    compilation_time = time.time() - start_time
    print(f"Compilation time: {compilation_time:.4f} seconds")

    def invoke_compiled():
        compiled_fmha(
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            lse_tensor,
            scale_softmax_log2,
            scale_softmax,
            scale_output,
            (
                window_size_left
                if window_size_left is None
                else cutlass.Int32(window_size_left)
            ),
            (
                window_size_right
                if window_size_right is None
                else cutlass.Int32(window_size_right)
            ),
            current_stream,
        )

    def run_reference(q, k, v):
        from reference_attention import attention_reference

        if window_size_left is not None or window_size_right not in {None, 0}:
            raise ValueError("the project reference only supports dense causal attention")
        s_q_ref, d_ref, h_r_ref, h_k_ref, b_ref = q.shape
        s_k_ref = k.shape[0]
        q_bhsd = q.permute(4, 3, 2, 0, 1).contiguous().view(
            b_ref, h_k_ref * h_r_ref, s_q_ref, d_ref
        ).to(device=q_torch.device)
        k_bhsd = k.permute(4, 3, 2, 0, 1).contiguous().view(
            b_ref, h_k_ref, s_k_ref, d_ref
        ).to(device=q_torch.device)
        v_bhsd = v.permute(4, 3, 2, 0, 1).contiguous().view(
            b_ref, h_k_ref, s_k_ref, d_ref
        ).to(device=q_torch.device)
        output_bhsd = attention_reference(
            q_bhsd,
            k_bhsd.repeat_interleave(h_r_ref, dim=1),
            v_bhsd.repeat_interleave(h_r_ref, dim=1),
            causal=is_causal,
            softmax_scale=scale_softmax,
        )
        return (
            output_bhsd.view(b_ref, h_k_ref, h_r_ref, s_q_ref, d_ref)
            .permute(3, 4, 2, 1, 0)
            .contiguous()
            * scale_output
        )

    if not skip_ref_check:
        invoke_compiled()

        print("Verifying results with reference_attention.py...")
        o_ref = run_reference(q_ref, k_ref, v_ref)

        o_fp32_torch = o_torch.float()
        ref_o_f32_torch = o_ref.float()

        error = (o_fp32_torch - ref_o_f32_torch).abs()
        print(
            f"Reference error: max_abs={error.max().item():.6g}, "
            f"mean_abs={error.mean().item():.6g}"
        )
        torch.testing.assert_close(
            o_fp32_torch, ref_o_f32_torch, atol=tolerance, rtol=2e-02
        )
        print("Results verified successfully!")

    if not execute_benchmark:
        if skip_ref_check:
            invoke_compiled()
        torch.cuda.synchronize()
        return {
            "time_us": None,
            "compilation_time_s": compilation_time,
            "q": q_torch,
            "k": k_torch,
            "v": v_torch,
            "output": o_torch,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        }

    if use_cold_l2:
        raise ValueError("cold-L2 benchmarking is not supported by this fixed workload")
    for _ in range(warmup_iterations):
        invoke_compiled()
    torch.cuda.synchronize()
    elapsed_us = []
    for _ in range(iterations):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        invoke_compiled()
        end_event.record()
        end_event.synchronize()
        elapsed_us.append(start_event.elapsed_time(end_event) * 1000)
    elapsed_us.sort()
    midpoint = len(elapsed_us) // 2
    exec_time = (
        elapsed_us[midpoint]
        if len(elapsed_us) % 2
        else (elapsed_us[midpoint - 1] + elapsed_us[midpoint]) / 2
    )

    return {
        "time_us": exec_time,
        "compilation_time_s": compilation_time,
        "q": q_torch,
        "k": k_torch,
        "v": v_torch,
        "output": o_torch,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
    }


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


# Fixed 128K single-launch Decode configuration.  The conservative migration
# baseline keeps the validated N128/KV5 pipeline, two compute warpgroups, 128
# producer threads, and no intra-warpgroup overlap.
_DECODE_FIXED_BLOCK_N = 128
_DECODE_FIXED_KV_STAGE = 5
# Single split has no direct-output epilogue; it uses the existing staged path.
_DECODE_FIXED_SUPPORTED_SPLITS = (8, 9, 10, 16, 18, 19, 32)
_DECODE_FIXED_KV_LEN = QWEN_CONTEXT
# Bounded to keep long-running processes from accumulating one workspace per
# stream; the fixed workspace is small and cold entries are cheap to rebuild.
_DECODE_FIXED_WORKSPACE_CACHE_LIMIT = 64

_QWEN3_DECODE_FIXED_KERNEL_CACHE = {}
_QWEN3_DECODE_FIXED_WORKSPACE_CACHE = {}
_QWEN3_DECODE_FIXED_WORKSPACE_LOCK = threading.Lock()
_QWEN3_DEVICE_SM_COUNT_CACHE = {}
_QWEN3_DEVICE_SM_COUNT_LOCK = threading.Lock()


def _get_decode_fixed_workspace(device, stream_handle: int, num_splits: int):
    """Return the cached per-(device, stream, split) partial workspace.

    BF16 partial O is stored as [split, Hkv, ratio, D] and FP32 partial LSE as
    [split, Hkv, ratio].  Workspaces are never shared across streams, so
    sequential calls on one stream reuse the buffers without synchronization.
    """
    import torch

    key = (device.index, stream_handle, num_splits)
    with _QWEN3_DECODE_FIXED_WORKSPACE_LOCK:
        workspace = _QWEN3_DECODE_FIXED_WORKSPACE_CACHE.get(key)
        cache_hit = workspace is not None
        if workspace is None:
            if len(_QWEN3_DECODE_FIXED_WORKSPACE_CACHE) >= _DECODE_FIXED_WORKSPACE_CACHE_LIMIT:
                evicted_key = next(iter(_QWEN3_DECODE_FIXED_WORKSPACE_CACHE))
                del _QWEN3_DECODE_FIXED_WORKSPACE_CACHE[evicted_key]
            o_partial = torch.empty(
                (num_splits, QWEN_KV_HEADS, QWEN_QUERY_HEADS // QWEN_KV_HEADS, QWEN_HEAD_DIM),
                dtype=torch.bfloat16,
                device=device,
            )
            lse_partial = torch.empty(
                (num_splits, QWEN_KV_HEADS, QWEN_QUERY_HEADS // QWEN_KV_HEADS),
                dtype=torch.float32,
                device=device,
            )
            workspace = {
                "o": o_partial,
                "lse": lse_partial,
                "o_bytes": o_partial.numel() * o_partial.element_size(),
                "lse_bytes": lse_partial.numel() * lse_partial.element_size(),
            }
            _QWEN3_DECODE_FIXED_WORKSPACE_CACHE[key] = workspace
    return workspace, cache_hit


def _run_hopper_decode_fixed_128k(
    q,
    k,
    v,
    *,
    sm_scale: float,
    num_splits: int,
    o_partial,
    lse_partial,
):
    """Launch the single-launch fixed-128K split-KV Decode partial kernel.

    Q/K/V keep their original storage: Q is viewed as [ratio, D, Hkv], K as
    [KV, D, Hkv], and V as [D, KV, Hkv].  The kernel writes BF16 partial O as
    [ratio, D, Hkv, split] and FP32 partial LSE as [ratio, Hkv, split] into
    the caller-provided workspace; no Python K/V slices are created.
    """
    import torch

    h_r = QWEN_QUERY_HEADS // QWEN_KV_HEADS
    # Exact-rank views: q [ratio, D, Hkv], k [KV, D, Hkv], v [D, KV, Hkv],
    # o [ratio, D, Hkv, split], lse [ratio, Hkv, split].  The batch mode is
    # squeezed away because the fixed contract is B=1.
    q_kernel = q.view(QWEN_KV_HEADS, h_r, QWEN_HEAD_DIM).permute(1, 2, 0)
    k_kernel = k.view(QWEN_KV_HEADS, _DECODE_FIXED_KV_LEN, QWEN_HEAD_DIM).permute(1, 2, 0)
    v_kernel = v.view(QWEN_KV_HEADS, _DECODE_FIXED_KV_LEN, QWEN_HEAD_DIM).permute(2, 1, 0)
    o_kernel = o_partial.permute(2, 3, 1, 0)
    lse_kernel = lse_partial.permute(2, 1, 0)

    def as_cute_tensor(tensor, element_type, leading_dim):
        cute_tensor = from_dlpack(tensor, assumed_align=16)
        cute_tensor.element_type = element_type
        return cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)

    q_tensor = as_cute_tensor(q_kernel, cutlass.BFloat16, 1)
    k_tensor = as_cute_tensor(k_kernel, cutlass.BFloat16, 1)
    v_tensor = as_cute_tensor(v_kernel, cutlass.BFloat16, 0)
    o_tensor = as_cute_tensor(o_kernel, cutlass.BFloat16, 1)
    lse_tensor = as_cute_tensor(lse_kernel, cutlass.Float32, 0)

    log2_e = 1.4426950408889634074
    scale_softmax_log2 = sm_scale * log2_e
    torch_stream = torch.cuda.current_stream(q.device)
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    cache_key = (q.device.index, num_splits)
    compiled_fmha = _QWEN3_DECODE_FIXED_KERNEL_CACHE.get(cache_key)
    if compiled_fmha is None:
        with _QWEN3_COMPILE_LOCK:
            compiled_fmha = _QWEN3_DECODE_FIXED_KERNEL_CACHE.get(cache_key)
            if compiled_fmha is None:
                fmha = HopperDecode128KSplitKVForward(
                    cutlass.Float32,
                    cutlass.Float32,
                    (64, _DECODE_FIXED_BLOCK_N, QWEN_HEAD_DIM),
                    _DECODE_FIXED_KV_STAGE,
                    num_splits,
                )
                compiled_fmha = cute.compile(
                    fmha,
                    q_tensor,
                    k_tensor,
                    v_tensor,
                    o_tensor,
                    lse_tensor,
                    scale_softmax_log2,
                    sm_scale,
                    1.0,
                    current_stream,
                )
                _QWEN3_DECODE_FIXED_KERNEL_CACHE[cache_key] = compiled_fmha

    compiled_fmha(
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        lse_tensor,
        scale_softmax_log2,
        sm_scale,
        1.0,
        current_stream,
    )
    q.record_stream(torch_stream)
    k.record_stream(torch_stream)
    v.record_stream(torch_stream)


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
    """Launch one local Hopper WGMMA attention problem.

    K/V may be strided sequence slices. TMA consumes their actual strides, so
    decode does not copy or expand the KV cache.
    """
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
            # Optional tensor arguments are not reliable in this CuTe version.
            # This one-element allocation preserves the signature; the
            # write_lse=False specialization does not load or store it.
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
                    compiled_fmha = cute.compile(
                        fmha,
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
                    _QWEN3_KERNEL_CACHE[cache_key] = compiled_fmha

        compiled_fmha(
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
        q.record_stream(torch_stream)
        k.record_stream(torch_stream)
        v.record_stream(torch_stream)
        if write_lse:
            return output, lse_storage[..., 0]
        return output, None


def qwen3_prefill_attention(
    q, k, v, *, causal=True, sm_scale=None, prefill_config="auto"
):
    """Run Qwen3-32B prefill attention on contiguous BHSD tensors.

    The public forward API returns O only. Its compiled kernel omits LSE log and
    stores; a one-element dummy tensor remains solely because the current CuTe
    callable signature cannot reliably use an Optional tensor.
    """
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
    num_sms: Optional[int] = None,
) -> int:
    candidates = _normalize_split_candidates(split_candidates)
    max_splits = max(1, math.ceil(seqlen / block_n))
    if num_splits is None or num_splits == 0:
        # One Pack-GQA CTA handles each KV head, so a split contributes eight
        # independent CTAs. Match FA3's wave-efficiency idea: find the best SM
        # utilization, then choose the smallest split count reaching 85% of it.
        # The SM count is a runtime input from the caller (H20 has 78 SMs and
        # still selects 9 splits here); it is not an algorithmic constant.
        if num_sms is None:
            import torch

            num_sms = torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count
        if isinstance(num_sms, bool) or not isinstance(num_sms, int):
            raise TypeError("num_sms must be a positive integer or None")
        if num_sms <= 0:
            raise ValueError("num_sms must be positive")
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
    """Combine Pack-GQA split outputs with a stable FP32 LSE reduction.

    Each partial output is laid out as ``[B, Hkv, ratio, D]``: the eight query
    heads sharing one KV head are packed into the kernel's logical M dimension.
    The local Hopper kernel returns LSE as ``[B, ratio, Hkv, 1]``.
    """
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


def _combine_decode_fixed_partials(o_partial, lse_partial, *, output_dtype):
    """Combine the pre-stacked fixed-path partial workspace in FP32.

    ``o_partial`` is [split, Hkv, ratio, D] BF16 and ``lse_partial`` is
    [split, Hkv, ratio] FP32, both written by the single-launch partial
    kernel.  Applies the same stable log-sum-exp weighting as the staged
    combine without an intermediate ``torch.stack``.
    """
    import torch

    o_fp32 = o_partial.float()
    global_lse = torch.logsumexp(lse_partial, dim=0)
    weights = torch.exp(lse_partial - global_lse.unsqueeze(0))
    combined = (o_fp32 * weights.unsqueeze(-1)).sum(dim=0)
    return (
        combined.contiguous()
        .view(QWEN_BATCH, QWEN_QUERY_HEADS, 1, QWEN_HEAD_DIM)
        .to(dtype=output_dtype)
    )


def _can_use_qwen3_decode_fixed_128k(q, k, v, num_splits: int) -> bool:
    """Whether the fixed-128K single-launch path supports this exact call."""
    return (
        q.shape == (QWEN_BATCH, QWEN_QUERY_HEADS, 1, QWEN_HEAD_DIM)
        and k.shape == (QWEN_BATCH, QWEN_KV_HEADS, _DECODE_FIXED_KV_LEN, QWEN_HEAD_DIM)
        and v.shape == (QWEN_BATCH, QWEN_KV_HEADS, _DECODE_FIXED_KV_LEN, QWEN_HEAD_DIM)
        and num_splits in _DECODE_FIXED_SUPPORTED_SPLITS
    )


def _qwen3_decode_attention_fixed_128k_impl(
    q,
    k,
    v,
    *,
    causal: bool,
    sm_scale: float,
    num_splits: int,
    num_sms: int,
    return_stats: bool,
):
    """Fixed 128K fast path: one partial kernel launch + PyTorch FP32 combine.

    ``causal`` is intentionally unused: for the latest-token Decode contract,
    causal=True and causal=False both visit the complete KV cache.
    """
    import torch

    torch_stream = torch.cuda.current_stream(q.device)
    workspace, cache_hit = _get_decode_fixed_workspace(
        q.device, torch_stream.cuda_stream, num_splits
    )
    _run_hopper_decode_fixed_128k(
        q,
        k,
        v,
        sm_scale=sm_scale,
        num_splits=num_splits,
        o_partial=workspace["o"],
        lse_partial=workspace["lse"],
    )
    output = _combine_decode_fixed_partials(
        workspace["o"], workspace["lse"], output_dtype=q.dtype
    )
    if not return_stats:
        return output
    ranges = _continuous_kv_ranges(
        _DECODE_FIXED_KV_LEN, num_splits, block_n=_DECODE_FIXED_BLOCK_N
    )
    return output, {
        "path": "fixed-128k",
        "dispatch_reason": "exact 128K fixed-shape fast path",
        "num_sms": num_sms,
        "num_splits": num_splits,
        "ranges": ranges,
        "partial_kernel_launches": 1,
        "partial_wgmma_launches": 1,
        "logical_partial_ctas": num_splits * QWEN_KV_HEADS,
        "packed_query_heads_per_kv_head": QWEN_QUERY_HEADS // QWEN_KV_HEADS,
        "workspace_o_bytes": workspace["o_bytes"],
        "workspace_lse_bytes": workspace["lse_bytes"],
        "workspace_total_bytes": workspace["o_bytes"] + workspace["lse_bytes"],
        "workspace_cache_hit": cache_hit,
        "combine": "PyTorch FP32 log-sum-exp over pre-stacked BF16 partial workspace",
    }


def _qwen3_decode_attention_staged_impl(
    q,
    k,
    v,
    *,
    causal: bool,
    sm_scale: float,
    num_splits: int,
    num_sms: int,
    dispatch_reason: str,
    return_stats: bool,
):
    """Safe staged Hopper split-KV implementation.

    This is deliberately isolated behind a replacement-friendly interface and
    remains the fallback for any KV length or split configuration the fixed
    128K path does not support.
    """
    import torch

    _, block_n, _ = _resolve_prefill_config(DECODE_KERNEL_CONFIG)
    ranges = _continuous_kv_ranges(k.shape[2], num_splits, block_n=block_n)

    # Pack the eight query heads belonging to each KV head into logical M rows.
    # This changes the WGMMA workload from 64 mostly-empty query-head CTAs per
    # split to 8 CTAs whose first eight rows are all useful, while K/V remain
    # zero-copy strided views of the original cache.
    h_r = QWEN_QUERY_HEADS // QWEN_KV_HEADS
    q_packed = q.view(QWEN_BATCH, QWEN_KV_HEADS, h_r, QWEN_HEAD_DIM)
    partial_outputs = []
    partial_lses = []

    def launch_partial(begin, end):
        return _run_hopper_fmha(
            q_packed,
            k[:, :, begin:end, :],
            v[:, :, begin:end, :],
            # Packed M rows are query heads, not sequence positions. Every row
            # attends the complete continuous split for both public causal
            # choices; global split combination restores full-cache attention.
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
        # wait_stream establishes execution ordering but does not extend the
        # caching allocator lifetime onto the consumer stream. Record both
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
        "path": "staged-split-kv",
        "dispatch_reason": dispatch_reason,
        "num_sms": num_sms,
        "num_splits": len(ranges),
        "ranges": ranges,
        "partial_kernel_launches": len(ranges),
        "partial_wgmma_launches": len(ranges),
        "logical_partial_ctas": len(ranges) * QWEN_KV_HEADS,
        "packed_query_heads_per_kv_head": h_r,
        "workspace_o_bytes": 0,
        "workspace_lse_bytes": 0,
        "workspace_total_bytes": 0,
        "workspace_cache_hit": False,
        "combine": "FP32 log-sum-exp weights over BF16 partial outputs",
    }


def _get_device_sm_count(device) -> int:
    """Cache the per-device SM count so the hot Decode dispatch path does not
    repeatedly query CUDA device properties."""
    import torch

    key = device.index
    num_sms = _QWEN3_DEVICE_SM_COUNT_CACHE.get(key)
    if num_sms is None:
        with _QWEN3_DEVICE_SM_COUNT_LOCK:
            num_sms = _QWEN3_DEVICE_SM_COUNT_CACHE.get(key)
            if num_sms is None:
                num_sms = torch.cuda.get_device_properties(device).multi_processor_count
                _QWEN3_DEVICE_SM_COUNT_CACHE[key] = num_sms
    return num_sms


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
    """Dispatch Decode to the fixed 128K fast path or the staged fallback."""
    import torch

    _, block_n, _ = _resolve_prefill_config(DECODE_KERNEL_CONFIG)
    num_sms = _get_device_sm_count(q.device)
    actual_splits = _select_decode_splits(
        k.shape[2], num_splits, split_candidates, block_n=block_n, num_sms=num_sms
    )

    dispatch_reason = None
    if k.shape[2] != _DECODE_FIXED_KV_LEN:
        dispatch_reason = f"kv_len={k.shape[2]} != {_DECODE_FIXED_KV_LEN}"
    elif not _can_use_qwen3_decode_fixed_128k(q, k, v, actual_splits):
        dispatch_reason = f"num_splits={actual_splits} not in fixed supported set"
    elif torch.cuda.is_current_stream_capturing():
        dispatch_reason = "cuda graph capture is not validated for the fixed path"

    if dispatch_reason is None:
        return _qwen3_decode_attention_fixed_128k_impl(
            q,
            k,
            v,
            causal=causal,
            sm_scale=sm_scale,
            num_splits=actual_splits,
            num_sms=num_sms,
            return_stats=return_stats,
        )
    return _qwen3_decode_attention_staged_impl(
        q,
        k,
        v,
        causal=causal,
        sm_scale=sm_scale,
        num_splits=actual_splits,
        num_sms=num_sms,
        dispatch_reason=dispatch_reason,
        return_stats=return_stats,
    )


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
    """Decode attention for q=[1,64,1,128], k/v=[1,8,S,128] BF16 CUDA.

    For the fixed latest-token ``q_len=1`` contract, causal=True and
    causal=False are equivalent and both visit the complete KV cache. Packed
    query-head rows therefore run each continuous split without a sequence mask.
    """
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


def run_qwen3_prefill(
    seqlen: int,
    *,
    persistent: bool,
    tolerance: float,
    warmup: int,
    iterations: int,
    check_reference: bool,
    benchmark: bool,
    input_pattern: str = "random",
    prefill_config: str = "auto",
):
    if seqlen <= 0:
        raise ValueError("seqlen must be positive")
    _, block_n, kv_stage = _resolve_prefill_config(prefill_config)
    return run(
        q_shape=(QWEN_BATCH, seqlen, QWEN_QUERY_HEADS, QWEN_HEAD_DIM),
        k_shape=(QWEN_BATCH, seqlen, QWEN_KV_HEADS, QWEN_HEAD_DIM),
        in_dtype=cutlass.BFloat16,
        out_dtype=cutlass.BFloat16,
        qk_acc_dtype=cutlass.Float32,
        pv_acc_dtype=cutlass.Float32,
        mma_tiler_mn=(64, block_n),
        is_persistent=persistent,
        is_causal=True,
        bottom_right_align=False,
        scale_q=1.0,
        scale_k=1.0,
        scale_v=1.0,
        inv_scale_o=1.0,
        scale_softmax=1.0 / math.sqrt(QWEN_HEAD_DIM),
        window_size=(-1, -1),
        tolerance=tolerance,
        warmup_iterations=warmup,
        iterations=iterations,
        skip_ref_check=not check_reference,
        use_cold_l2=False,
        execute_benchmark=benchmark,
        input_pattern=input_pattern,
        kv_stage=kv_stage,
        write_lse=False,
    )


def _decode_grouped_reference(q, k, v, sm_scale: float):
    import torch

    h_r = QWEN_QUERY_HEADS // QWEN_KV_HEADS
    q_grouped = q.float().view(
        QWEN_BATCH, QWEN_KV_HEADS, h_r, 1, QWEN_HEAD_DIM
    )
    scores = torch.einsum("bhgqd,bhkd->bhgqk", q_grouped, k.float()) * sm_scale
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.einsum("bhgqk,bhkd->bhgqd", probabilities, v.float())
    return output.reshape(QWEN_BATCH, QWEN_QUERY_HEADS, 1, QWEN_HEAD_DIM)


def _median_of_sorted(sorted_samples):
    if not sorted_samples:
        return None
    midpoint = len(sorted_samples) // 2
    return (
        sorted_samples[midpoint]
        if len(sorted_samples) % 2
        else (sorted_samples[midpoint - 1] + sorted_samples[midpoint]) / 2
    )


def _nearest_rank_p95(sorted_samples):
    """Nearest-rank p95: smallest sample covering at least 95% of the data."""
    if not sorted_samples:
        return None
    rank = max(1, math.ceil(0.95 * len(sorted_samples)))
    return sorted_samples[min(rank, len(sorted_samples)) - 1]


def _decode_effective_kv_gbps(seqlen: int, time_us: float) -> float:
    """Effective KV bandwidth from the logical K+V read of one Decode token."""
    logical_bytes = 2 * QWEN_BATCH * QWEN_KV_HEADS * seqlen * QWEN_HEAD_DIM * 2
    return logical_bytes / (time_us * 1e-6) / 1e9


def _time_decode_invoke(invoke, warmup: int, iterations: int):
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
        samples.append(start.elapsed_time(end) * 1000)
    samples.sort()
    return samples, output


def run_qwen3_decode(
    seqlen: int,
    *,
    tolerance: float,
    warmup: int,
    iterations: int,
    check_reference: bool,
    benchmark: bool,
    causal: bool,
    num_splits: Optional[int],
    split_candidates: Sequence[int],
):
    import torch

    if seqlen <= 0:
        raise ValueError("seqlen must be positive")
    torch.manual_seed(1111)
    device = torch.device("cuda", torch.cuda.current_device())
    q = torch.randn(
        (QWEN_BATCH, QWEN_QUERY_HEADS, 1, QWEN_HEAD_DIM),
        device=device,
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        (QWEN_BATCH, QWEN_KV_HEADS, seqlen, QWEN_HEAD_DIM),
        device=device,
        dtype=torch.bfloat16,
    )
    v = torch.randn_like(k)
    sm_scale = 1.0 / math.sqrt(QWEN_HEAD_DIM)
    output, stats = _qwen3_decode_attention_impl(
        q,
        k,
        v,
        causal=causal,
        sm_scale=sm_scale,
        num_splits=num_splits,
        split_candidates=split_candidates,
        return_stats=True,
    )

    if check_reference:
        reference = _decode_grouped_reference(q, k, v, sm_scale)
        error = (output.float() - reference).abs()
        print(
            f"Decode S={seqlen} splits={stats['num_splits']} path={stats['path']} "
            f"max_abs={error.max().item():.6g}, mean_abs={error.mean().item():.6g}"
        )
        torch.testing.assert_close(
            output.float(), reference, atol=tolerance, rtol=2e-2
        )

    elapsed_us = None
    p95_us = None
    effective_gbps = None
    staged_timing = None
    if benchmark:
        def invoke():
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

        # The 128K benchmark explicitly measures the fixed path and the staged
        # fallback, interleaved round by round (A/B) so clock/thermal drift
        # affects both equally.  Other lengths only run the staged fallback.
        fixed_applies = stats["path"] == "fixed-128k"
        staged_invoke = None
        if fixed_applies:
            def staged_invoke():
                return _qwen3_decode_attention_staged_impl(
                    q,
                    k,
                    v,
                    causal=causal,
                    sm_scale=sm_scale,
                    num_splits=stats["num_splits"],
                    num_sms=stats["num_sms"],
                    dispatch_reason="explicit 128K A/B benchmark",
                    return_stats=False,
                )

            for _ in range(warmup):
                invoke()
                staged_invoke()
            torch.cuda.synchronize()
            fixed_samples = []
            staged_samples = []
            fixed_output = output
            for round_idx in range(iterations):
                # Alternate A/B order each round so neither path systematically
                # benefits from being measured first.
                order = (
                    ((invoke, fixed_samples), (staged_invoke, staged_samples))
                    if round_idx % 2 == 0
                    else ((staged_invoke, staged_samples), (invoke, fixed_samples))
                )
                for closure, samples in order:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    latest = closure()
                    end.record()
                    end.synchronize()
                    samples.append(start.elapsed_time(end) * 1000)
                    if closure is invoke:
                        fixed_output = latest
            output = fixed_output
            fixed_samples.sort()
            staged_samples.sort()
            elapsed_us = _median_of_sorted(fixed_samples)
            p95_us = _nearest_rank_p95(fixed_samples)
            staged_median = _median_of_sorted(staged_samples)
            staged_p95 = _nearest_rank_p95(staged_samples)
            staged_timing = {
                "time_us": staged_median,
                "p95_us": staged_p95,
                "effective_kv_gbps": _decode_effective_kv_gbps(seqlen, staged_median),
            }
        else:
            samples, output = _time_decode_invoke(invoke, warmup, iterations)
            elapsed_us = _median_of_sorted(samples)
            p95_us = _nearest_rank_p95(samples)
        effective_gbps = _decode_effective_kv_gbps(seqlen, elapsed_us)

    return {
        "time_us": elapsed_us,
        "p95_us": p95_us,
        "effective_kv_gbps": effective_gbps,
        "staged_timing": staged_timing,
        "q": q,
        "k": k,
        "v": v,
        "output": output,
        **stats,
    }


def check_full_length(result, seqlen: int, tolerance: float) -> None:
    import torch

    output = result["output"]
    sample_positions = sorted(
        {position for position in (0, 1, 127, 128, seqlen // 2, seqlen - 1) if position < seqlen}
    )
    total_error = 0.0
    sample_count = 0
    max_error = 0.0
    for position in sample_positions:
        count = position + 1
        remainder = count % 17
        position_sum = (remainder * (remainder - 1) / 2 - 8 * remainder) / 8
        for kv_head in range(QWEN_KV_HEADS):
            expected = position_sum / count + kv_head / 4
            actual = output[position, :, :, kv_head, 0].float()
            if not torch.isfinite(actual).all():
                raise AssertionError(
                    f"non-finite output at position={position}, kv_head={kv_head}"
                )
            error = (actual - expected).abs()
            max_error = max(max_error, error.max().item())
            total_error += error.sum().item()
            sample_count += error.numel()
    mean_error = total_error / sample_count
    print(
        f"128K sampled prefix-mean error: max_abs={max_error:.6g}, "
        f"mean_abs={mean_error:.6g}"
    )
    if max_error > tolerance:
        raise AssertionError(
            f"full-length sampled error {max_error:.6g} exceeds {tolerance:.6g}"
        )


def causal_tflops(seqlen: int, time_us: float) -> float:
    flops = (
        QWEN_BATCH
        * QWEN_QUERY_HEADS
        * seqlen
        * (seqlen + 1)
        * (QWEN_HEAD_DIM + QWEN_HEAD_DIM)
    )
    return flops / (time_us * 1e-6) / 1e12


def benchmark_fa3(result, warmup: int, iterations: int):
    import statistics
    import torch

    hopper_path = "/dockerdata/linqihao/flash-attention/hopper"
    if hopper_path not in sys.path:
        sys.path.insert(0, hopper_path)
    from flash_attn_interface import flash_attn_func

    q_internal, k_internal, v_internal = result["q"], result["k"], result["v"]
    seqlen = q_internal.shape[0]
    q = q_internal.permute(4, 0, 3, 2, 1).reshape(
        QWEN_BATCH, seqlen, QWEN_QUERY_HEADS, QWEN_HEAD_DIM
    ).contiguous()
    k = k_internal.permute(4, 0, 3, 2, 1).reshape(
        QWEN_BATCH, seqlen, QWEN_KV_HEADS, QWEN_HEAD_DIM
    ).contiguous()
    v = v_internal.permute(4, 0, 3, 2, 1).reshape(
        QWEN_BATCH, seqlen, QWEN_KV_HEADS, QWEN_HEAD_DIM
    ).contiguous()

    def invoke():
        return flash_attn_func(
            q,
            k,
            v,
            softmax_scale=1.0 / math.sqrt(QWEN_HEAD_DIM),
            causal=True,
            num_splits=1,
            pack_gqa=None,
        )

    with torch.inference_mode():
        baseline_output = invoke()
        for _ in range(warmup):
            baseline_output = invoke()
        torch.cuda.synchronize()
        elapsed_us = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            baseline_output = invoke()
            end.record()
            end.synchronize()
            elapsed_us.append(start.elapsed_time(end) * 1000)

    local_output = result["output"]
    max_sample_error = 0.0
    sample_positions = sorted(
        {position for position in (0, 127, seqlen // 2, seqlen - 1) if position < seqlen}
    )
    for position in sample_positions:
        for query_head in (0, 7, 8, 63):
            kv_head, ratio_head = divmod(query_head, QWEN_QUERY_HEADS // QWEN_KV_HEADS)
            local = local_output[position, :, ratio_head, kv_head, 0].float()
            baseline = baseline_output[0, position, query_head].float()
            if not torch.isfinite(local).all() or not torch.isfinite(baseline).all():
                raise AssertionError(
                    f"non-finite FA3 comparison at position={position}, head={query_head}"
                )
            max_sample_error = max(
                max_sample_error, (local - baseline).abs().max().item()
            )
    return statistics.median(elapsed_us), max_sample_error


def benchmark_fa3_decode(result, warmup: int, iterations: int):
    """Benchmark FA3 Decode with its automatic split and Pack-GQA heuristics."""
    import statistics
    import torch

    hopper_path = "/dockerdata/linqihao/flash-attention/hopper"
    if hopper_path not in sys.path:
        sys.path.insert(0, hopper_path)
    from flash_attn_interface import flash_attn_func

    q = result["q"].transpose(1, 2).contiguous()
    k = result["k"].transpose(1, 2).contiguous()
    v = result["v"].transpose(1, 2).contiguous()

    def invoke():
        output = flash_attn_func(
            q,
            k,
            v,
            softmax_scale=1.0 / math.sqrt(QWEN_HEAD_DIM),
            causal=False,
            num_splits=0,
            pack_gqa=None,
        )
        # Some FA3 builds return (output, lse, ...) tuples.
        return output[0] if isinstance(output, tuple) else output

    with torch.inference_mode():
        baseline_output = invoke()
        for _ in range(warmup):
            baseline_output = invoke()
        torch.cuda.synchronize()
        elapsed_us = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            baseline_output = invoke()
            end.record()
            end.synchronize()
            elapsed_us.append(start.elapsed_time(end) * 1000)

    local_output = result["output"].transpose(1, 2)
    max_error = (local_output.float() - baseline_output.float()).abs().max().item()
    elapsed_us.sort()
    return {
        "median_us": statistics.median(elapsed_us),
        "p95_us": _nearest_rank_p95(elapsed_us),
        "effective_kv_gbps": _decode_effective_kv_gbps(
            result["k"].shape[2], statistics.median(elapsed_us)
        ),
        "max_error": max_error,
    }


def benchmark_original_prefill(result, warmup: int, iterations: int):
    import statistics
    import torch
    import implement_attention as original

    q_internal, k_internal, v_internal = result["q"], result["k"], result["v"]
    seqlen = q_internal.shape[0]
    q = q_internal.permute(4, 3, 2, 0, 1).reshape(
        QWEN_BATCH, QWEN_QUERY_HEADS, seqlen, QWEN_HEAD_DIM
    ).contiguous()
    k = k_internal.permute(4, 3, 2, 0, 1).reshape(
        QWEN_BATCH, QWEN_KV_HEADS, seqlen, QWEN_HEAD_DIM
    ).contiguous()
    v = v_internal.permute(4, 3, 2, 0, 1).reshape(
        QWEN_BATCH, QWEN_KV_HEADS, seqlen, QWEN_HEAD_DIM
    ).contiguous()

    def invoke():
        return original.qwen3_prefill_attention(q, k, v, causal=True)

    with torch.inference_mode():
        baseline_output = invoke()
        for _ in range(warmup):
            baseline_output = invoke()
        torch.cuda.synchronize()
        elapsed_us = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            baseline_output = invoke()
            end.record()
            end.synchronize()
            elapsed_us.append(start.elapsed_time(end) * 1000)

    local_output = result["output"].permute(4, 3, 2, 0, 1).reshape_as(q)
    max_error = (local_output.float() - baseline_output.float()).abs().max().item()
    return statistics.median(elapsed_us), max_error


def parse_seqlens(value: str):
    try:
        values = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("sequence lengths must be positive")
    return values


def parse_split_candidates(value: str):
    try:
        return _normalize_split_candidates(tuple(int(item) for item in value.split(",")))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "expected comma-separated positive split counts"
        ) from exc


def decode_tflops(seqlen: int, time_us: float) -> float:
    flops = 4 * QWEN_BATCH * QWEN_QUERY_HEADS * seqlen * QWEN_HEAD_DIM
    return flops / (time_us * 1e-6) / 1e12


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qwen3-32B BF16 GQA prefill/decode attention on Hopper SM90"
    )
    parser.add_argument(
        "--mode", choices=("correctness", "full-check", "benchmark"), default="correctness"
    )
    parser.add_argument(
        "--phase", choices=("prefill", "decode", "both"), default="prefill"
    )
    parser.add_argument("--seqlen", type=int, default=QWEN_CONTEXT)
    parser.add_argument("--seqlens", type=parse_seqlens, default=[128, 257, 1024])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=5e-2)
    parser.add_argument("--persistent", action="store_true")
    parser.add_argument("--num-splits", type=int, default=0)
    parser.add_argument(
        "--split-candidates",
        type=parse_split_candidates,
        default=DEFAULT_DECODE_SPLIT_CANDIDATES,
    )
    parser.add_argument(
        "--prefill-config",
        choices=("auto", *PREFILL_CONFIGS),
        default="auto",
    )
    parser.add_argument("--compare-fa3", action="store_true")
    parser.add_argument("--compare-original", action="store_true")
    args = parser.parse_args()

    if args.warmup < 0 or args.iterations <= 0:
        parser.error("--warmup must be non-negative and --iterations must be positive")
    if args.tolerance <= 0:
        parser.error("--tolerance must be positive")
    if args.num_splits < 0:
        parser.error("--num-splits must be non-negative")

    run_prefill = args.phase in {"prefill", "both"}
    run_decode = args.phase in {"decode", "both"}

    if args.mode == "correctness":
        if run_prefill:
            for seqlen in args.seqlens:
                run_qwen3_prefill(
                    seqlen,
                    persistent=args.persistent,
                    tolerance=args.tolerance,
                    warmup=0,
                    iterations=1,
                    check_reference=True,
                    benchmark=False,
                    prefill_config=args.prefill_config,
                )
        if run_decode:
            import torch

            num_sms = torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count
            for seqlen in args.seqlens:
                split_requests = [1]
                requested = args.num_splits if args.num_splits else 0
                actual = _select_decode_splits(
                    seqlen, requested, args.split_candidates, num_sms=num_sms
                )
                if actual != 1:
                    split_requests.append(requested)
                for causal in (False, True):
                    for split_request in split_requests:
                        run_qwen3_decode(
                            seqlen,
                            tolerance=args.tolerance,
                            warmup=0,
                            iterations=1,
                            check_reference=True,
                            benchmark=False,
                            causal=causal,
                            num_splits=split_request,
                            split_candidates=args.split_candidates,
                        )
        print("Correctness checks passed")
        return

    if args.mode == "full-check":
        if run_prefill:
            result = run_qwen3_prefill(
                args.seqlen,
                persistent=args.persistent,
                tolerance=args.tolerance,
                warmup=0,
                iterations=1,
                check_reference=False,
                benchmark=False,
                input_pattern="prefix",
                prefill_config=args.prefill_config,
            )
            check_full_length(result, args.seqlen, args.tolerance)
            print("Prefill full-length sampled check passed")
        if run_decode:
            run_qwen3_decode(
                args.seqlen,
                tolerance=args.tolerance,
                warmup=0,
                iterations=1,
                check_reference=True,
                benchmark=False,
                causal=True,
                num_splits=args.num_splits,
                split_candidates=args.split_candidates,
            )
            print("Decode full-length reference check passed")
        return

    if run_prefill:
        result = run_qwen3_prefill(
            args.seqlen,
            persistent=args.persistent,
            tolerance=args.tolerance,
            warmup=args.warmup,
            iterations=args.iterations,
            check_reference=False,
            benchmark=True,
            prefill_config=args.prefill_config,
        )
        local_time_us = float(result["time_us"])
        resolved_config, _, _ = _resolve_prefill_config(args.prefill_config)
        print(
            f"Prefill local CuTe ({resolved_config}): {local_time_us:.3f} us, "
            f"{causal_tflops(args.seqlen, local_time_us):.3f} TFLOP/s, "
            f"peak_memory={result['peak_memory_bytes'] / 2**30:.3f} GiB"
        )
        if args.compare_fa3:
            fa3_time_us, sample_error = benchmark_fa3(
                result, args.warmup, args.iterations
            )
            print(
                f"FA3 pack_gqa=None: {fa3_time_us:.3f} us, "
                f"{causal_tflops(args.seqlen, fa3_time_us):.3f} TFLOP/s"
            )
            print(
                f"Speed ratio (FA3/local): {fa3_time_us / local_time_us:.4f}x, "
                f"sample max_abs={sample_error:.6g}"
            )
            if sample_error > args.tolerance:
                raise AssertionError(
                    f"FA3 sample error {sample_error:.6g} exceeds {args.tolerance:.6g}"
                )
        if args.compare_original:
            original_time_us, max_error = benchmark_original_prefill(
                result, args.warmup, args.iterations
            )
            print(
                f"Original prefill: {original_time_us:.3f} us, "
                f"original/local={original_time_us / local_time_us:.4f}x, "
                f"max_abs={max_error:.6g}"
            )

    if run_decode:
        result = run_qwen3_decode(
            args.seqlen,
            tolerance=args.tolerance,
            warmup=args.warmup,
            iterations=args.iterations,
            check_reference=False,
            benchmark=True,
            causal=True,
            num_splits=args.num_splits,
            split_candidates=args.split_candidates,
        )
        local_time_us = float(result["time_us"])
        print(
            f"Decode local CuTe ({result['path']}): {local_time_us:.3f} us, "
            f"p95={result['p95_us']:.3f} us, "
            f"{decode_tflops(args.seqlen, local_time_us):.3f} TFLOP/s, "
            f"effective_kv={result['effective_kv_gbps']:.1f} GB/s, "
            f"splits={result['num_splits']}, "
            f"partial_kernel_launches={result['partial_kernel_launches']}, "
            f"logical_partial_ctas={result['logical_partial_ctas']}, "
            f"num_sms={result['num_sms']}"
        )
        print(
            f"Decode dispatch: {result['dispatch_reason']}; "
            f"workspace_total={result['workspace_total_bytes']} bytes "
            f"(o={result['workspace_o_bytes']}, lse={result['workspace_lse_bytes']}), "
            f"workspace_cache_hit={result['workspace_cache_hit']}; "
            f"combine: {result['combine']}"
        )
        staged_timing = result["staged_timing"]
        if staged_timing is not None:
            print(
                f"Decode staged CuTe (interleaved A/B): "
                f"{staged_timing['time_us']:.3f} us, "
                f"p95={staged_timing['p95_us']:.3f} us, "
                f"effective_kv={staged_timing['effective_kv_gbps']:.1f} GB/s, "
                f"staged/fixed={staged_timing['time_us'] / local_time_us:.4f}x"
            )
        if args.compare_original:
            print("Original decode comparison unavailable: implement_attention.py has no decode API")
        if args.compare_fa3:
            fa3 = benchmark_fa3_decode(result, args.warmup, args.iterations)
            print(
                f"FA3 decode auto: {fa3['median_us']:.3f} us, "
                f"p95={fa3['p95_us']:.3f} us, "
                f"effective_kv={fa3['effective_kv_gbps']:.1f} GB/s, "
                f"FA3/local={fa3['median_us'] / local_time_us:.4f}x, "
                f"max_abs={fa3['max_error']:.6g}"
            )
            if fa3["max_error"] > args.tolerance:
                raise AssertionError(
                    f"FA3 decode error {fa3['max_error']:.6g} exceeds {args.tolerance:.6g}"
                )


if __name__ == "__main__":
    main()
