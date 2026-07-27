# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
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

"""Dual-compute-warpgroup Hopper causal FMHA M128 performance kernel (GQA).

Source skeleton:
  CUTLASS v4.6.0, commit e6233cba
  examples/python/CuTeDSL/cute/hopper/kernel/attention/fmha.py
  examples/python/CuTeDSL/helpers/fmha_helpers.py

M128N128D128 performance specialization of the verified M128 correctness
kernel, retargeted to the GQA contract: BF16 Q/K/V/O, FP32 QK/PV accumulators
and online-softmax state, B=1, GQA Hq=64 / Hkv=8 (blocked mapping, q head h
pairs with kv head h//8), causal prefill, and S=131072. One TMA producer
warpgroup plus two compute warpgroups (each computes 64 of the 128
CTA rows; 384 threads total), q_stage=2, K/V alternating ring with a
configurable kv_stage in {2,3,4,5}, epi_stage=2, TMA Q/K/V/O and QK/PV WGMMA.
Cluster shape is 1 and scheduling is non-persistent; there is no multicast.
Because BLOCK_M == BLOCK_N == 128, q_block is exactly the K diagonal block
and the only mask is local_k > local_q on that tile, sliced per warpgroup.
Register donation (load/mma) is configurable for search. It does not call a
production FlashAttention kernel.
"""

import inspect as _inspect
import math
import threading
from contextlib import contextmanager
from typing import Tuple

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
    dsl_user_op,
    if_generate,
    not_,
    while_generate,
    yield_out,
)
from cutlass._mlir.dialects import nvvm
from cutlass._mlir._mlir_libs._cutlass_ir._mlir.ir import IntegerType


_BLOCK_M = 128
_BLOCK_N = 128
_BATCH = 1
_Q_HEADS = 64
_KV_HEADS = 8
_GQA_GROUP_SIZE = _Q_HEADS // _KV_HEADS
_SEQUENCE = 131072
_HEAD_DIM = 128
_GRID = (_SEQUENCE // _BLOCK_M, _Q_HEADS, _BATCH)
_LOG2_E = 1.4426950408889634074
_SCALE_SOFTMAX_LOG2 = _LOG2_E / math.sqrt(_HEAD_DIM)
_QO_CUTE_SHAPE = (_SEQUENCE, _HEAD_DIM, _GQA_GROUP_SIZE, _KV_HEADS, _BATCH)
_QO_CUTE_STRIDE = (
    _HEAD_DIM,
    1,
    _SEQUENCE * _HEAD_DIM,
    _GQA_GROUP_SIZE * _SEQUENCE * _HEAD_DIM,
    _Q_HEADS * _SEQUENCE * _HEAD_DIM,
)
_KV_CUTE_SHAPE = (_SEQUENCE, _HEAD_DIM, 1, _KV_HEADS, _BATCH)
_KV_CUTE_STRIDE = (
    _HEAD_DIM,
    1,
    _KV_HEADS * _SEQUENCE * _HEAD_DIM,
    _SEQUENCE * _HEAD_DIM,
    _KV_HEADS * _SEQUENCE * _HEAD_DIM,
)


# Keep the bounded-polling mbarrier workaround used by the exact upstream
# example. It is active only while CuTe traces the launch during compilation.
_timelimit_has_res = "res" in _inspect.signature(
    nvvm.mbarrier_try_wait_parity_timelimit
).parameters


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
    timeout = Int32(10000000).ir_value(loc=loc, ip=ip)
    true_value = lambda: Boolean(True).ir_value(loc=loc, ip=ip)

    done = Boolean(
        _try_wait_timelimit(llvm_ptr, phase_val, timeout, loc=loc, ip=ip)
    )
    done = if_generate(
        done,
        true_value,
        lambda: _try_wait_timelimit(
            llvm_ptr, phase_val, timeout, loc=loc, ip=ip
        ),
        None,
        [Boolean],
        loc=loc,
        ip=ip,
    )
    done = if_generate(
        done,
        true_value,
        lambda: _try_wait_timelimit(
            llvm_ptr, phase_val, timeout, loc=loc, ip=ip
        ),
        None,
        [Boolean],
        loc=loc,
        ip=ip,
    )

    def fallback():
        inner = Boolean(False).ir_value(loc=loc, ip=ip)
        ctx = while_generate(
            [inner], lambda value: not_(value, loc=loc, ip=ip), loc=loc, ip=ip
        )
        with ctx as (_,):
            result = Boolean(
                _try_wait_timelimit(
                    llvm_ptr, phase_val, timeout, loc=loc, ip=ip
                )
            )
            yield_out([result], loc=loc, ip=ip)
        return Boolean(True).ir_value(loc=loc, ip=ip)

    if_generate(done, true_value, fallback, None, [Boolean], loc=loc, ip=ip)


@contextmanager
def _use_optimized_mbarrier_wait():
    import cutlass.cute.arch as arch_mod

    original_wait = arch_mod.mbarrier_wait
    arch_mod.mbarrier_wait = _optimized_mbarrier_wait
    try:
        yield
    finally:
        arch_mod.mbarrier_wait = original_wait


class Sm90CausalFmhaPerfM128Forward:
    """M128 dual-compute-warpgroup Hopper TMA/WGMMA FMHA forward kernel."""

    def __init__(self, kv_stage=4, load_regs=24, mma_regs=240):
        if kv_stage not in (2, 3, 4, 5):
            raise ValueError("kv_stage must be one of 2, 3, 4, 5")
        if not (0 < load_regs < 256 and 0 < mma_regs < 256):
            raise ValueError("load_regs/mma_regs must be in (0, 256)")

        # Two math warpgroups, each computing 64 rows; the CTA tile is M=128.
        self.num_mma_warp_groups = 2
        self.qk_acc_dtype = cutlass.Float32
        self.pv_acc_dtype = cutlass.Float32

        self.qk_mma_tiler = (64, _BLOCK_N, _HEAD_DIM)
        self.pv_mma_tiler = (64, _HEAD_DIM, _BLOCK_N)
        self.cta_tiler = (_BLOCK_M, _BLOCK_N, _HEAD_DIM)

        self.cluster_shape_mnk = (1, 1, 1)
        self.atom_layout_mnk = (1, 1, 1)

        self.load_warp_group_id = 0
        self.compute_epilogue_0_warp_group_id = 1
        self.compute_epilogue_1_warp_group_id = 2
        self.producer_warp_loadkv_id = 1

        self.num_threads_per_warp_group = 128
        self.num_warps_per_warp_group = 4
        # 1 TMA producer warpgroup + 2 compute warpgroups.
        self.threads_per_cta = 384
        # Producers mostly wait on TMA barriers, so they donate registers to
        # the compute warpgroups. Both values are searchable.
        self.num_regs_load = load_regs
        self.num_regs_mma = mma_regs
        self.buffer_align_bytes = 1024

        # Two 64-row Q stages form one 128-row CTA query block.
        self.q_stage = 2
        # K/V alternate in a single ring (kv_stage half-tiles in flight).
        self.kv_stage = kv_stage
        self.epi_stage = 2

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        o: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.q_dtype = q.element_type
        self.k_dtype = k.element_type
        self.v_dtype = v.element_type
        self.o_dtype = o.element_type

        if cutlass.const_expr(self.q_dtype != cutlass.BFloat16):
            raise TypeError("SM90 FMHA requires BF16 Q")
        if cutlass.const_expr(
            self.k_dtype != self.q_dtype
            or self.v_dtype != self.q_dtype
            or self.o_dtype != self.q_dtype
        ):
            raise TypeError("SM90 FMHA requires matching BF16 Q/K/V/O")
        if cutlass.const_expr(q.leading_dim != 1 or k.leading_dim != 1):
            raise RuntimeError("Q and K must be head-dimension major")
        if cutlass.const_expr(
            q.shape != _QO_CUTE_SHAPE
            or o.shape != _QO_CUTE_SHAPE
            or k.shape != _KV_CUTE_SHAPE
            or v.shape != _KV_CUTE_SHAPE
        ):
            raise ValueError("SM90 FMHA is fixed to B1 Hq64 Hkv8 S131072 D128")
        if cutlass.const_expr(
            q.stride != _QO_CUTE_STRIDE
            or o.stride != _QO_CUTE_STRIDE
            or k.stride != _KV_CUTE_STRIDE
            or v.stride != _KV_CUTE_STRIDE
        ):
            raise ValueError("SM90 FMHA requires the fixed contiguous BHSD strides")

        scale_softmax_log2 = cutlass.Float32(_SCALE_SOFTMAX_LOG2)

        # Incoming zero-copy views: Q/O are (S, D, Hq//Hkv, Hkv, B) and K/V
        # are (S, D, 1, Hkv, B). K/V are re-nested below so the q head group
        # mode (Hq//Hkv) broadcasts with stride 0 over each kv head, matching
        # the upstream FMHA GQA layout. The blocked pairing (q head h attends
        # kv head h // (Hq//Hkv)) is fixed by the torch view construction in
        # _as_cute_qo_view / _as_cute_kv_view.
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
        q = cute.group_modes(cute.group_modes(q, begin=2, end=4), begin=2, end=4)
        o = cute.group_modes(cute.group_modes(o, begin=2, end=4), begin=2, end=4)

        self.q_layout = utils.LayoutEnum.from_tensor(q)
        self.k_layout = utils.LayoutEnum.from_tensor(k)
        self.v_layout = utils.LayoutEnum.from_tensor(v)
        self.o_layout = utils.LayoutEnum.from_tensor(o)

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
            cute.nvgpu.warpgroup.OperandMajorMode.K,
            self.v_major_mode,
            self.pv_acc_dtype,
            self.atom_layout_mnk,
            self.pv_mma_tiler[:2],
            warpgroup.OperandSource.RMEM,
        )

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

        self.epi_tile = sm90_utils.compute_tile_shape_or_override(
            self.cta_tiler, self.o_dtype
        )
        o_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            self.o_dtype,
            self.o_layout,
            self.epi_tile,
            self.epi_stage,
            cute.append(
                cute.append(self.epi_tile, self.epi_stage),
                self.num_mma_warp_groups,
            ),
            smem_order=(
                (1, 0, 2, 3)
                if self.o_layout.is_m_major_c()
                else (0, 1, 2, 3)
            ),
        )

        q_smem_layout = cute.slice_(q_smem_layout_staged, (None, None, 0))
        tma_atom_q, tma_tensor_q = self._make_tma_atoms_and_tensors(
            q,
            q_smem_layout_staged,
            (self.qk_mma_tiler[0], self.qk_mma_tiler[2]),
        )

        k_smem_layout = cute.slice_(k_smem_layout_staged, (None, None, 0))
        tma_atom_k, tma_tensor_k = self._make_tma_atoms_and_tensors(
            k,
            k_smem_layout_staged,
            (self.qk_mma_tiler[1], self.qk_mma_tiler[2]),
        )
        tma_atom_v, tma_tensor_v = self._make_tma_atoms_and_tensors(
            v,
            v_smem_layout_staged,
            (self.pv_mma_tiler[1], self.pv_mma_tiler[2]),
        )

        o_smem_layout = cute.slice_(o_smem_layout_staged, (None, None, 0, 0))
        tma_atom_o, tma_tensor_o = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            o,
            o_smem_layout,
            self.epi_tile,
        )

        self.tma_copy_q_bytes = cute.size_in_bytes(self.q_dtype, q_smem_layout)
        self.tma_copy_kv_bytes = cute.size_in_bytes(self.k_dtype, k_smem_layout)

        @cute.struct
        class SharedStorage:
            load_q_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.q_stage * 2
            ]
            load_kv_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.kv_stage * 2
            ]
            MathWarpGroupOrderBarrier: cute.struct.MemRange[
                cutlass.Int64, self.num_mma_warp_groups
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[
                    self.q_dtype, cute.cosize(q_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            # K and V intentionally share this backing allocation.
            sK: cute.struct.Align[
                cute.struct.MemRange[
                    self.k_dtype, cute.cosize(k_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Fixed non-persistent grid: 1024 query tiles x 64 q heads x B1.
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
                scale_softmax_log2,
                q_smem_layout_staged,
                k_smem_layout_staged,
                v_smem_layout_staged,
                o_smem_layout_staged,
            ).launch(
                grid=_GRID,
                block=[self.threads_per_cta, 1, 1],
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
        tma_atom_o: cute.CopyAtom,
        mO_qdl: cute.Tensor,
        scale_softmax_log2: cutlass.Float32,
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        o_smem_layout_staged: cute.ComposedLayout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        q_block, head_idx, batch_idx = cute.arch.block_idx()
        head_batch = (head_idx, batch_idx)

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

        sQ = storage.sQ.get_tensor(
            q_smem_layout_staged.outer, swizzle=q_smem_layout_staged.inner
        )
        sK = storage.sK.get_tensor(
            k_smem_layout_staged.outer, swizzle=k_smem_layout_staged.inner
        )
        sV = cute.make_tensor(
            cute.recast_ptr(sK.iterator, v_smem_layout_staged.inner),
            v_smem_layout_staged.outer,
        )
        # Non-persistent mode reuses the Q allocation for the output epilogue.
        sO = cute.make_tensor(
            cute.recast_ptr(
                sQ.iterator, o_smem_layout_staged.inner, self.o_dtype
            ),
            o_smem_layout_staged.outer,
        )

        qk_thr_mma = qk_tiled_mma.get_slice(tidx)
        gQ_qdl = cute.flat_divide(
            mQ_qdl, cute.select(self.qk_mma_tiler, mode=[0, 2])
        )
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

        pv_thr_mma = pv_tiled_mma.get_slice(tidx)
        gV_dkl = cute.flat_divide(
            mV_dkl, cute.select(self.pv_mma_tiler, mode=[1, 2])
        )
        tSgV_dkl = pv_thr_mma.partition_B(gV_dkl)
        tVsV, tVgV_dkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_v,
            0,
            cute.make_layout(1),
            cute.group_modes(sV, 0, 2),
            cute.group_modes(tSgV_dkl, 0, 3),
        )

        producer_warp_role = warp_idx % self.num_warps_per_warp_group

        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_o)

        if warp_group_idx == self.load_warp_group_id:
            cute.arch.setmaxregister_decrease(self.num_regs_load)

            if producer_warp_role == self.producer_warp_loadkv_id:
                tQgQ_head = tQgQ_qdl[(None, None, 0, head_batch)]
                # Two 64-row Q stages form one 128-row CTA query block.
                tQgQ = cute.domain_offset((0, q_block * 2), tQgQ_head)
                tKgK = tKgK_kdl[(None, None, 0, head_batch)]
                tVgV = tVgV_dkl[(None, 0, None, head_batch)]

                q0_handle = load_q_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_q,
                    tQgQ[(None, 0)],
                    tQsQ[(None, q0_handle.index)],
                    tma_bar_ptr=q0_handle.barrier,
                )

                k_handle = load_kv_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_k,
                    tKgK[(None, 0)],
                    tKsK[(None, k_handle.index)],
                    tma_bar_ptr=k_handle.barrier,
                )

                q1_handle = load_q_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_q,
                    tQgQ[(None, 1)],
                    tQsQ[(None, q1_handle.index)],
                    tma_bar_ptr=q1_handle.barrier,
                )

                v_handle = load_kv_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_v,
                    tVgV[(None, 0)],
                    tVsV[(None, v_handle.index)],
                    tma_bar_ptr=v_handle.barrier,
                )

                # Causal specialization: the CTA consumes exactly K/V blocks
                # 0..q_block. Every one is a full in-bounds 128-column tile.
                kv_index = 1
                remaining_kv_pairs = q_block
                while remaining_kv_pairs > 0:
                    k_handle = load_kv_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_k,
                        tKgK[(None, kv_index)],
                        tKsK[(None, k_handle.index)],
                        tma_bar_ptr=k_handle.barrier,
                    )

                    v_handle = load_kv_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_v,
                        tVgV[(None, kv_index)],
                        tVsV[(None, v_handle.index)],
                        tma_bar_ptr=v_handle.barrier,
                    )
                    kv_index += 1
                    remaining_kv_pairs -= 1

        if (
            warp_group_idx == self.compute_epilogue_0_warp_group_id
            or warp_group_idx == self.compute_epilogue_1_warp_group_id
        ):
            cute.arch.setmaxregister_increase(self.num_regs_mma)

            wg_local = warp_group_idx - 1
            for _ in cutlass.range(wg_local, unroll=1):
                load_q_consumer.advance()

            tSsQ = qk_thr_mma.partition_A(sQ)
            tSsK = qk_thr_mma.partition_B(sK)
            tSrQ = qk_thr_mma.make_fragment_A(tSsQ)
            tSrK = qk_thr_mma.make_fragment_B(tSsK)

            thr_mma_pv = pv_tiled_mma.get_slice(tidx)
            tOsV = thr_mma_pv.partition_B(sV)
            tOrV = thr_mma_pv.make_fragment_B(tOsV)

            q_handle = load_q_consumer.wait()

            pv_acc_shape = pv_thr_mma.partition_shape_C(
                (self.pv_mma_tiler[0], self.pv_mma_tiler[1])
            )
            acc_pv = pv_thr_mma.make_fragment_C(pv_acc_shape)
            qk_acc_shape = qk_thr_mma.partition_shape_C(
                (self.qk_mma_tiler[0], self.qk_mma_tiler[1])
            )

            s_max_layout = cute.make_layout(
                cute.size(
                    self.layout_acc_mn(pv_tiled_mma, acc_pv.layout), mode=[0]
                )
            )
            s_max = cute.make_rmem_tensor_like(
                s_max_layout, self.qk_acc_dtype
            )
            a_sum = cute.make_rmem_tensor_like(s_max, cutlass.Float32)

            # Local coordinates for the one and only masked tile. WG0 covers
            # local rows 0..63 and WG1 covers 64..127.
            local_qk_coords = cute.make_identity_tensor(
                (_BLOCK_M, _BLOCK_N)
            )
            local_qk_tiles = cute.local_tile(
                local_qk_coords, self.qk_mma_tiler[:2], (None, None)
            )
            partitioned_local_qk = qk_thr_mma.partition_C(local_qk_tiles)
            diagonal_coords = cute.slice_(
                partitioned_local_qk,
                (None, None, None, wg_local, 0),
            )

            # The first tile is K/V block 0. For q_block==0 it is already the
            # diagonal; otherwise it is the first independent unmasked tile.
            acc_qk = qk_thr_mma.make_fragment_C(qk_acc_shape)
            k_handle = load_kv_consumer.wait_and_advance()
            math_wg_order_barrier.wait()

            cute.nvgpu.warpgroup.fence()
            self.gemm_zero_acc(
                qk_tiled_mma,
                tSrQ[(None, None, None, q_handle.index)],
                tSrK[(None, None, None, k_handle.index)],
                acc_qk,
            )
            cute.nvgpu.warpgroup.commit_group()
            math_wg_order_barrier.arrive()
            cute.nvgpu.warpgroup.wait_group(0)

            if q_block == 0:
                self.apply_diagonal_mask(acc_qk, diagonal_coords)

            s_max, a_sum = self.softmax_step(
                acc_qk,
                qk_tiled_mma,
                s_max,
                a_sum,
                acc_pv,
                pv_tiled_mma,
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

            # Strictly independent unmasked loop over K/V blocks
            # 1..q_block-1. Its body contains no causal, tail, or bounds tests.
            load_kv_consumer, _, s_max, a_sum = self.compute_unmasked(
                q_block - 1,
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
                scale_softmax_log2,
                qk_acc_shape,
            )

            # q_block>0 has one separate diagonal K/V block at q_block.
            if q_block > 0:
                diagonal_acc_qk = qk_thr_mma.make_fragment_C(qk_acc_shape)
                diagonal_k_handle = load_kv_consumer.wait_and_advance()

                cute.nvgpu.warpgroup.fence()
                self.gemm_zero_acc(
                    qk_tiled_mma,
                    tSrQ[(None, None, None, q_handle.index)],
                    tSrK[
                        (None, None, None, diagonal_k_handle.index)
                    ],
                    diagonal_acc_qk,
                )
                cute.nvgpu.warpgroup.commit_group()
                diagonal_token = load_kv_consumer.try_wait()
                cute.nvgpu.warpgroup.wait_group(0)

                # The only mask is local_k_col > local_q_row, and it is
                # applied before softmax_step performs the row maximum.
                self.apply_diagonal_mask(
                    diagonal_acc_qk, diagonal_coords
                )
                s_max, a_sum = self.softmax_step(
                    diagonal_acc_qk,
                    qk_tiled_mma,
                    s_max,
                    a_sum,
                    acc_pv,
                    pv_tiled_mma,
                    scale_softmax_log2,
                    False,
                )
                diagonal_qk_fixed = self.make_acc_into_op(
                    diagonal_acc_qk,
                    pv_tiled_mma.tv_layout_A,
                    self.q_dtype,
                )

                diagonal_v_handle = load_kv_consumer.wait_and_advance(
                    diagonal_token
                )
                cute.nvgpu.warpgroup.fence()
                pv_tiled_mma.set(
                    cute.nvgpu.warpgroup.Field.ACCUMULATE, True
                )
                cute.gemm(
                    pv_tiled_mma,
                    acc_pv,
                    diagonal_qk_fixed,
                    tOrV[
                        (None, None, None, diagonal_v_handle.index)
                    ],
                    acc_pv,
                )
                cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.wait_group(0)
                diagonal_k_handle.release()
                diagonal_v_handle.release()

            cute.nvgpu.warpgroup.wait_group(0)
            self.normalize_tail(a_sum, acc_pv, pv_tiled_mma)

            if warp_group_idx == self.compute_epilogue_0_warp_group_id:
                for _ in cutlass.range_constexpr(self.num_mma_warp_groups):
                    load_q_consumer.advance()
            if warp_group_idx == self.compute_epilogue_1_warp_group_id:
                for _ in cutlass.range_constexpr(
                    self.num_mma_warp_groups - 1
                ):
                    load_q_consumer.advance()

            math_wg_order_barrier.wait()

            q_mma_tile = q_block * self.num_mma_warp_groups + wg_local
            copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
                self.o_layout,
                elem_ty_d=self.o_dtype,
                elem_ty_acc=self.pv_acc_dtype,
            )
            copy_atom_o = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(
                    self.o_layout.is_m_major_c(), 4
                ),
                self.o_dtype,
            )
            tiled_copy_o_atom = cute.make_tiled_copy_C_atom(
                copy_atom_o, pv_tiled_mma
            )
            tiled_copy_r2s = cute.make_tiled_copy_S(
                copy_atom_r2s, tiled_copy_o_atom
            )
            thr_copy_r2s = tiled_copy_r2s.get_slice(
                tidx % self.num_threads_per_warp_group
            )
            tRS_sD = thr_copy_r2s.partition_D(sO)
            tRS_rAcc = tiled_copy_r2s.retile(acc_pv)

            rD_shape = cute.shape(thr_copy_r2s.partition_S(sO))
            tRS_rD_layout = cute.make_layout(rD_shape[:3])
            tRS_rD = cute.make_rmem_tensor_like(
                tRS_rD_layout, self.pv_acc_dtype
            )
            size_tRS_rD = cute.size(tRS_rD)

            gD = cute.local_tile(
                mO_qdl,
                self.pv_mma_tiler[:2],
                (q_mma_tile, 0, head_batch),
            )
            bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
                tma_atom_o,
                0,
                cute.make_layout(1),
                cute.group_modes(sO, 0, 2),
                cute.zipped_divide(gD, self.epi_tile),
            )
            epi_tile_num = cute.size(
                cute.zipped_divide(gD, self.epi_tile), mode=[1]
            )

            for epi_idx in cutlass.range_constexpr(epi_tile_num):
                for epi_v in cutlass.range_constexpr(size_tRS_rD):
                    tRS_rD[epi_v] = tRS_rAcc[
                        epi_idx * size_tRS_rD + epi_v
                    ]

                tRS_rD_out = cute.make_rmem_tensor_like(
                    tRS_rD_layout, self.o_dtype
                )
                tRS_rD_out.store(tRS_rD.load().to(self.o_dtype))

                epi_buffer = epi_idx % self.epi_stage
                cute.copy(
                    tiled_copy_r2s,
                    tRS_rD_out,
                    tRS_sD[
                        (
                            None,
                            None,
                            None,
                            epi_buffer,
                            warp_group_idx - 1,
                        )
                    ],
                )
                cute.arch.fence_proxy("async.shared", space="cta")
                pipeline.arrive_and_wait(
                    barrier_id=warp_group_idx,
                    num_threads=self.num_threads_per_warp_group,
                )

                if warp_idx == 4 or warp_idx == 8:
                    cute.copy(
                        tma_atom_o,
                        bSG_sD[
                            (None, epi_buffer, warp_group_idx - 1)
                        ],
                        bSG_gD[(None, epi_idx)],
                    )
                    tma_store_pipeline.producer_commit()
                    tma_store_pipeline.producer_acquire()

                pipeline.arrive_and_wait(
                    barrier_id=warp_group_idx,
                    num_threads=self.num_threads_per_warp_group,
                )

            math_wg_order_barrier.arrive()

    @cute.jit
    def compute_unmasked(
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
        scale_softmax_log2: cutlass.Float32,
        qk_acc_shape: cute.Shape,
    ) -> Tuple[
        pipeline.PipelineConsumer,
        cutlass.Int32,
        cute.Tensor,
        cute.Tensor,
    ]:
        """Process only full, in-bounds, strictly pre-diagonal K/V tiles."""
        while k_tile_count > 0:
            k_tile_count -= 1
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

            # Intentionally no causal, residual, tail, or bounds predicate here.
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
            pv_tiled_mma.set(
                cute.nvgpu.warpgroup.Field.ACCUMULATE, True
            )
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

        return load_kv_consumer, k_tile_count, s_max, a_sum

    @staticmethod
    @cute.jit
    def apply_diagonal_mask(acc_qk: cute.Tensor, local_qk: cute.Tensor):
        """Mask exactly local_k_col > local_q_row on a diagonal tile."""
        for i in cutlass.range(cute.size(acc_qk), unroll_full=True):
            local_q_row, local_k_col = local_qk[i]
            if local_k_col > local_q_row:
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
    ) -> Tuple[cute.Tensor, cute.Tensor]:
        """FP32 online softmax update from the upstream implementation."""
        acc_qk_mn = cute.make_tensor(
            acc_qk.iterator,
            self.layout_acc_mn(tiled_mma_qk, acc_qk.layout),
        )
        reduction_target_qk = self.reduction_target_n(tiled_mma_qk)
        red_rank = cute.rank(reduction_target_qk)

        s_max_prev = None
        acc_pv_mn = None
        if cutlass.const_expr(is_first_iter):
            for i in cutlass.range_constexpr(
                cute.size(acc_qk_mn, mode=[0])
            ):
                s_max[i] = acc_qk_mn[i, 0]
            for j in cutlass.range_constexpr(
                1, cute.size(acc_qk_mn, mode=[1])
            ):
                for i in cutlass.range_constexpr(
                    cute.size(acc_qk_mn, mode=[0])
                ):
                    s_max[i] = cute.arch.fmax(
                        s_max[i], acc_qk_mn[i, j]
                    )
        else:
            acc_pv_mn = cute.make_tensor(
                acc_pv.iterator,
                self.layout_acc_mn(tiled_mma_pv, acc_pv.layout),
            )
            s_max_prev = cute.make_rmem_tensor_like(s_max, s_max._dtype)

        for i in cutlass.range_constexpr(
            cute.size(acc_qk_mn, mode=[0])
        ):
            if cutlass.const_expr(not is_first_iter):
                s_max_prev[i] = s_max[i]
                for j in cutlass.range_constexpr(
                    cute.size(acc_qk_mn, mode=[1])
                ):
                    s_max[i] = cutlass.max(
                        s_max[i], acc_qk_mn[i, j]
                    )

            for r in cutlass.range_constexpr(red_rank):
                s_max[i] = cute.arch.warp_reduction_max(
                    s_max[i],
                    threads_in_group=reduction_target_qk.shape[r],
                )

            local_max = s_max[i]
            if s_max[i] == -cutlass.Float32.inf:
                local_max = 0.0
            scaled_max = scale_softmax_log2 * local_max

            for j in cutlass.range_constexpr(
                cute.size(acc_qk_mn, mode=[1])
            ):
                acc_qk_mn[i, j] = cute.math.exp2(
                    scale_softmax_log2 * acc_qk_mn[i, j] - scaled_max,
                    fastmath=True,
                )

            previous_sum = 0.0
            if cutlass.const_expr(not is_first_iter):
                current_max = s_max[i]
                if current_max == -cutlass.Float32.inf:
                    current_max = 0.0
                # Rescale old l and O only when the exact running maximum
                # changes. Otherwise alpha is exactly one, so skipping the
                # exp2 and full output-fragment multiply is equivalent.
                if current_max != s_max_prev[i]:
                    alpha = cute.math.exp2(
                        (s_max_prev[i] - current_max)
                        * scale_softmax_log2,
                        fastmath=True,
                    )
                    a_sum[i] *= alpha
                    for j in cutlass.range_constexpr(
                        cute.size(acc_pv_mn, mode=[1])
                    ):
                        acc_pv_mn[i, j] *= alpha
                previous_sum = a_sum[i]

            a_sum[i] = previous_sum + acc_qk_mn[i, None].load().reduce(
                cute.ReductionOp.ADD, cutlass.Float32.zero, 0
            )

        return s_max, a_sum

    @cute.jit
    def normalize_tail(self, a_sum, acc_pv, tiled_mma_pv):
        """Reduce FP32 l and normalize the FP32 output accumulator."""
        acc_pv_mn = cute.make_tensor(
            acc_pv.iterator,
            self.layout_acc_mn(tiled_mma_pv, acc_pv.layout),
        )
        reduction_target = self.reduction_target_n(tiled_mma_pv)
        red_rank = cute.rank(reduction_target)
        for r in cutlass.range_constexpr(red_rank):
            for i in cutlass.range_constexpr(
                cute.size(acc_pv_mn, mode=[0])
            ):
                a_sum[i] = cute.arch.warp_reduction_sum(
                    a_sum[i],
                    threads_in_group=reduction_target.shape[r],
                )

        for i in cutlass.range_constexpr(
            cute.size(acc_pv_mn, mode=[0])
        ):
            inv_sum = cute.arch.rcp_approx(a_sum[i])
            for j in cutlass.range_constexpr(
                cute.size(acc_pv_mn, mode=[1])
            ):
                acc_pv_mn[i, j] *= inv_sum

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
    def make_acc_into_op(self, acc, operand_layout_tv, element_type):
        operand = cute.make_rmem_tensor_like(
            self.convert_c_layout_to_a_layout(
                acc.layout, operand_layout_tv.shape[1]
            ),
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
        return cute.append(
            cute.append(cute.make_layout(()), less), greater_equal
        )

    @staticmethod
    @cute.jit
    def gemm_zero_acc(tiled_mma, a, b, c):
        rank_a = cute.rank(a)
        rank_b = cute.rank(b)
        rank_c = cute.rank(c)
        if cutlass.const_expr(rank_a == 2 and rank_b == 2 and rank_c == 1):
            for k_block_idx in range(
                cute.size(a, mode=[1]), unroll_full=True
            ):
                tiled_mma.set(
                    cute.nvgpu.warpgroup.Field.ACCUMULATE,
                    k_block_idx != 0,
                )
                cute.gemm(
                    tiled_mma,
                    c,
                    a[None, k_block_idx],
                    b[None, k_block_idx],
                    c,
                )
        elif cutlass.const_expr(
            rank_a == 3 and rank_b == 3 and rank_c == 3
        ):
            for k_block_idx in range(
                cute.size(a, mode=[2]), unroll_full=True
            ):
                tiled_mma.set(
                    cute.nvgpu.warpgroup.Field.ACCUMULATE,
                    k_block_idx != 0,
                )
                cute.gemm(
                    tiled_mma,
                    c,
                    a[None, None, k_block_idx],
                    b[None, None, k_block_idx],
                    c,
                )
        else:
            assert False

    @cute.jit
    def layout_acc_mn(self, tiled_mma, acc):
        separated = self.layout_separate(
            tiled_mma.shape_mnk[0],
            acc[0],
            tiled_mma.tv_layout_C.stride[1],
        )
        value_m = separated[0]
        value_n = separated[1]

        if cutlass.const_expr(cute.rank(value_m) == 1):
            value_m = cute.append(value_m, acc[1])
        else:
            value_m = cute.append(
                cute.append(cute.make_layout(()), value_m), acc[1]
            )

        if cutlass.const_expr(cute.rank(value_n) == 1):
            value_n = cute.append(value_n, acc[2])
        else:
            value_n = cute.append(
                cute.append(cute.make_layout(()), value_n), acc[2]
            )

        if cutlass.const_expr(cute.rank(value_m) == 1):
            return cute.append(value_m, value_n)
        return cute.append(
            cute.append(cute.make_layout(()), value_m), value_n
        )

    def make_and_init_load_q_pipeline(self, load_q_mbar_ptr):
        return pipeline.PipelineTmaAsync.create(
            barrier_storage=load_q_mbar_ptr,
            num_stages=self.q_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, 1
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_warps_per_warp_group
            ),
            tx_count=self.tma_copy_q_bytes,
            defer_sync=True,
        ).make_participants()

    def make_and_init_load_kv_pipeline(self, load_kv_mbar_ptr):
        return pipeline.PipelineTmaAsync.create(
            barrier_storage=load_kv_mbar_ptr,
            num_stages=self.kv_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, 1
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.num_mma_warp_groups
                * self.num_warps_per_warp_group,
            ),
            tx_count=self.tma_copy_kv_bytes,
            defer_sync=True,
        ).make_participants()

    def make_and_init_tma_store_pipeline(self):
        return pipeline.PipelineTmaStore.create(
            num_stages=self.epi_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, 1
            ),
        )

    def make_and_init_order_barrier(self, order_mbar_ptr, group_id):
        return pipeline.PipelineOrder.create(
            barrier_storage=order_mbar_ptr,
            depth=1,
            length=self.num_mma_warp_groups,
            group_id=group_id,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_threads_per_warp_group
            ),
            defer_sync=True,
        )

    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor: cute.Tensor,
        smem_layout_staged: cute.ComposedLayout,
        smem_tile: tuple[int, int],
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        return cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=1,
        )


_compile_lock = threading.Lock()
# Process-local compile cache keyed by (kv_stage, load_regs, mma_regs): each
# configuration traces a distinct kernel, so each gets its own callable.
_compiled_fmha_m128 = {}
_compile_count = 0


def _as_cute_qo_view(tensor):
    """Return the fixed zero-copy (S,D,8,8,1) CuTe view of BHSD Q/O.

    The contiguous BHSD head axis is split as h = h_kv * 8 + h_r, so the
    kernel's nested head coordinate (h_r, h_kv) pairs q head h with kv head
    h // 8: the blocked GQA convention used by FlashAttention-2. The returned
    torch view must stay alive for the duration of the launch.
    """
    torch_view = tensor.view(
        _BATCH, _KV_HEADS, _GQA_GROUP_SIZE, _SEQUENCE, _HEAD_DIM
    ).permute(3, 4, 2, 1, 0)
    cute_view = from_dlpack(
        torch_view,
        assumed_align=16,
        enable_tvm_ffi=False,
    )
    return cute_view, torch_view


def _as_cute_kv_view(tensor):
    """Return the fixed zero-copy (S,D,1,8,1) CuTe view of BHSD K/V.

    The singleton group mode is filled by the kernel with a stride-0
    broadcast over the q head group. The returned torch view must stay alive
    for the duration of the launch.
    """
    torch_view = tensor.permute(2, 3, 1, 0).unsqueeze(2)
    cute_view = from_dlpack(
        torch_view,
        assumed_align=16,
        enable_tvm_ffi=False,
    )
    return cute_view, torch_view


def _current_cu_stream(torch, device):
    torch_stream = torch.cuda.current_stream(device=device)
    return cuda.CUstream(torch_stream.cuda_stream)


def run_sm90_fmha_m128(
    q, k, v, out, kv_stage=4, load_regs=24, mma_regs=240
):
    """Launch the cached static-128K CuTe callable on PyTorch's current stream.

    Shape, strides, blocked GQA ratio, scale, and launch grid are compile-time
    constants. Compilations are cached only per register/stage configuration.
    """
    global _compile_count

    import torch

    q_shape = (_BATCH, _Q_HEADS, _SEQUENCE, _HEAD_DIM)
    kv_shape = (_BATCH, _KV_HEADS, _SEQUENCE, _HEAD_DIM)
    if tuple(q.shape) != q_shape or tuple(out.shape) != q_shape:
        raise ValueError("q/out must have fixed BHSD shape (1,64,131072,128)")
    if tuple(k.shape) != kv_shape or tuple(v.shape) != kv_shape:
        raise ValueError("k/v must have fixed BHSD shape (1,8,131072,128)")
    if not all(tensor.is_contiguous() for tensor in (q, k, v, out)):
        raise ValueError("q, k, v, and out must be contiguous BHSD tensors")

    q_cute, q_torch_view = _as_cute_qo_view(q)
    k_cute, k_torch_view = _as_cute_kv_view(k)
    v_cute, v_torch_view = _as_cute_kv_view(v)
    o_cute, o_torch_view = _as_cute_qo_view(out)
    current_stream = _current_cu_stream(torch, q.device)

    cache_key = (int(kv_stage), int(load_regs), int(mma_regs))
    compiled = _compiled_fmha_m128.get(cache_key)
    if compiled is None:
        with _compile_lock:
            compiled = _compiled_fmha_m128.get(cache_key)
            if compiled is None:
                fmha = Sm90CausalFmhaPerfM128Forward(
                    kv_stage=cache_key[0],
                    load_regs=cache_key[1],
                    mma_regs=cache_key[2],
                )
                compiled = cute.compile(
                    fmha,
                    q_cute,
                    k_cute,
                    v_cute,
                    o_cute,
                    current_stream,
                )
                _compiled_fmha_m128[cache_key] = compiled
                _compile_count += 1

    compiled(
        q_cute,
        k_cute,
        v_cute,
        o_cute,
        current_stream,
    )

    # Keep all DLPack-exporting torch views live through the direct call.
    _ = (q_torch_view, k_torch_view, v_torch_view, o_torch_view)
    return out


def get_compile_count():
    """Return successful process-local cute.compile invocations."""
    return _compile_count
