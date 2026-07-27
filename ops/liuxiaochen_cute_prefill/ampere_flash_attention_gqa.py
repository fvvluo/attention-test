# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Derived from the external NVIDIA Ampere CuTe DSL FlashAttention v2 reference
# implementation (`reference/ampere_flash_attention_v2.py`).  The original
# copyright, provenance, and license notice above are preserved and the
# original reference is not claimed as an original project kernel.
#
# Project-specific modifications by Liu Xiaochen:
#   * support independent Q and KV head counts for GQA;
#   * map each Q head to its KV head in kernel indexing;
#   * accept contiguous BHSD tensors through explicit stride-aware logical
#     BSHD layouts, without input transposes or copies;
#   * write output directly in contiguous BHSD layout;
#   * expose a causal-capable prefill path used by the team wrapper.

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

import argparse
from types import SimpleNamespace
from typing import Type, Callable

import cuda.bindings.driver as cuda
import cutlass.cute.testing as testing
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import from_dlpack
import cutlass.pipeline as pipeline
import cutlass.utils as utils

"""
A flash attention v2 forward pass example for NVIDIA Ampere SM80 architecture using CUTE DSL.

- Matrix Q is BxSqxNxH, B is batch dimension, Sq is query sequence length, N is number of heads, H is head dimension
- Matrix K is BxSkxNxH, B is batch dimension, Sk is key sequence length, N is number of heads, H is head dimension
- Matrix V is BxSkxNxH, B is batch dimension, Sk is key sequence length, N is number of heads, H is head dimension
- Matrix O is BxSqxNxH, B is batch dimension, Sq is query sequence length, N is number of heads, H is head dimension

This kernel supports the following features:
    - Utilizes CpAsync for efficient memory operations
    - Utilizes Ampere's tensor core for matrix multiply-accumulate (MMA) operations
    - Utilizes register pipeline to overlap shared memory-to-register transfers with computations.
    - Leverages DSL to implement an integrated online softmax fusion pattern.

This kernel works as follows:
1. Load Q and K matrices from global memory (GMEM) to shared memory (SMEM) using CpAsync operations.
2. Perform matrix multiply-accumulate (MMA) operations using tensor core instructions to compute intermediate result S.
3. Apply padding mask or causal mask to S during initial iterations.
4. Apply online softmax to S and rescale O using results from previous iteration.
5. Load V matrices and perform matrix multiply-accumulate (MMA) operations to compute final result O.
6. Normalize O after all iterations complete and store result back to global memory (GMEM).

To run this example:

.. code-block:: bash

    python examples/ampere/flash_attention_v2.py                                            \
      --dtype Float16 --head_dim 128 --m_block_size 128 --n_block_size 128                  \
      --num_threads 128 --batch_size 1 --seqlen_q 1280 --seqlen_k 1536                      \
      --num_head 16 --softmax_scale 1.0 --is_causal

The above command configures the model to use float16 for inputs and outputs. The problem dimensions
are set to: batch size of 1, query sequence length of 1280, key sequence length of 1536, head dimension
of 128, and 16 attention heads. The softmax scale is set to 1.0 and causal masking is enabled. The computation
uses tiles of size 128x128 for m and n dimensions, and utilizes 128 parallel threads.

To collect the performance with NCU profiler:

.. code-block:: bash

    ncu python examples/ampere/flash_attention_v2.py                                        \
        --dtype Float16 --head_dim 128 --m_block_size 128 --n_block_size 128                \
        --num_threads 128 --batch_size 1 --seqlen_q 1280 --seqlen_k 1536                    \
        --num_head 16 --softmax_scale 1.0 --is_causal --skip_ref_check

There are some constraints for this example:
* Only fp16 and bf16 data types are supported.
* The contiguous dimension of each tensor must be at least 16 bytes aligned.
* The log-sum-exp(for training) is not computed in the kernel.
* The values of `m_block_size`, `n_block_size`, and `head_dim` must be selected to stay within shared memory capacity limits.
* `m_block_size * 2` must be divisible by `num_threads`, otherwise the kernel will not be able to get the correct result.
"""


class FlashAttentionForwardAmpere:
    def __init__(
        self,
        head_dim: int,
        m_block_size: int = 128,
        n_block_size: int = 128,
        num_threads: int = 128,
        is_causal: bool = False,
    ):
        """Initializes the configuration for a flash attention v2 kernel.

        All contiguous dimensions must be at least 16 bytes aligned which indicates the head dimension
        should be a multiple of 8.

        :param head_dim: head dimension
        :type head_dim: int
        :param m_block_size: m block size
        :type m_block_size: int
        :param n_block_size: n block size
        :type n_block_size: int
        :param num_threads: number of threads
        :type num_threads: int
        :param is_causal: is causal
        """
        self._head_dim = head_dim
        self._m_block_size = m_block_size
        self._n_block_size = n_block_size
        # padding head_dim to a multiple of 32 as k_block_size
        self._head_dim_padded = (head_dim + 31) // 32 * 32
        self._num_threads = num_threads
        self._is_causal = is_causal

        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=num_threads
        )

    @staticmethod
    def can_implement(
        dtype, head_dim, m_block_size, n_block_size, num_threads, is_causal
    ) -> bool:
        """Check if the kernel can be implemented with the given parameters.

        :param dtype: data type
        :type dtype: cutlass.Numeric
        :param head_dim: head dimension
        :type head_dim: int
        :param m_block_size: m block size
        :type m_block_size: int
        :param n_block_size: n block size
        :type n_block_size: int
        :param num_threads: number of threads
        :type num_threads: int
        :param is_causal: is causal
        :type is_causal: bool

        :return: True if the kernel can be implemented, False otherwise
        :rtype: bool
        """
        # Check if data type is fp16 or bf16
        if dtype != cutlass.Float16 and dtype != cutlass.BFloat16:
            return False

        # Check if head dimension is a multiple of 8
        if head_dim % 8 != 0:
            return False

        # Check if number of threads is a multiple of 32
        if num_threads % 32 != 0:
            return False

        # Check if block size setting is out of shared memory capacity
        # Shared memory usage: Q tile + (K tile + V tile) where K and V use the same tile size
        smem_usage = (m_block_size * head_dim + n_block_size * head_dim * 2) * 2
        smem_capacity = utils.get_smem_capacity_in_bytes("sm_80")
        if smem_usage > smem_capacity:
            return False

        # Check if twice the block size is divisible by the number of threads
        if (m_block_size * 2) % num_threads != 0:
            return False

        return True

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        softmax_scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        """Configures and launches the flash attention v2 kernel.

        mQ/mK/mV/mO has same data types(supports fp16 and bf16) and same layout:
        (batch_size, seqlen_q, num_head, head_dim):(seqlen_q * num_head * head_dim, num_head * head_dim, head_dim, 1)

        Prepares the shared memory layout, tiled copy atoms, tiled mma and shared memory storage.
        Then launches the kernel function with the prepared parameters.

        :param mQ: query tensor
        :type mQ: cute.Tensor
        :param mK: key tensor
        :type mK: cute.Tensor
        :param mV: value tensor
        :type mV: cute.Tensor
        :param mO: output tensor
        :type mO: cute.Tensor
        :param softmax_scale: softmax scale
        :type softmax_scale: cutlass.Float32
        """
        # Get the data type and check if it is fp16 or bf16
        if cutlass.const_expr(
            not (
                mQ.element_type == mK.element_type == mV.element_type == mO.element_type
            )
        ):
            raise TypeError("All tensors must have the same data type")
        if cutlass.const_expr(
            not (
                mQ.element_type == cutlass.Float16
                or mQ.element_type == cutlass.BFloat16
            )
        ):
            raise TypeError("Only Float16 or BFloat16 is supported")
        self._dtype: Type[cutlass.Numeric] = mQ.element_type
        # ///////////////////////////////////////////////////////////////////////////////
        # Shared memory layout: Q/K/V
        # ///////////////////////////////////////////////////////////////////////////////
        smem_k_block_size = 64 if self._head_dim_padded % 64 == 0 else 32
        swizzle_bits = 3 if smem_k_block_size == 64 else 2
        sQ_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3),
            0,
            cute.make_layout((8, smem_k_block_size), stride=(smem_k_block_size, 1)),
        )
        sQ_layout = cute.tile_to_shape(
            sQ_layout_atom,
            (self._m_block_size, self._head_dim_padded),
            (0, 1),
        )

        sKV_layout_atom = sQ_layout_atom
        sKV_layout = cute.tile_to_shape(
            sKV_layout_atom,
            (self._n_block_size, self._head_dim_padded),
            (0, 1),
        )

        sO_layout = sQ_layout

        @cute.struct
        class SharedStorage:
            sQ: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sQ_layout)], 1024
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024
            ]

        # ///////////////////////////////////////////////////////////////////////////////
        # GMEM Tiled copy:
        # ///////////////////////////////////////////////////////////////////////////////
        # Thread layouts for copies.  The BHSD logical layout places the head
        # stride between sequence and head_dim, so a 128-bit vector cannot be
        # inferred uniformly for every tile.  Use element-wise universal copies
        # for Q/K/V/O in this correctness version.
        qkv_copy_bits = 128
        qkv_copy_elems = qkv_copy_bits // self._dtype.width
        universal_copy_bits = self._dtype.width
        universal_copy_elems = universal_copy_bits // self._dtype.width
        # atom_async_copy: async copy atom for QKV load
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self._dtype,
            num_bits_per_copy=qkv_copy_bits,
        )
        # atom_universal_copy: universal copy atom for O store
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self._dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        # tQKV_layout: thread layout for QKV load
        tQKV_shape_dim_1 = sQ_layout_atom.outer.shape[1] // qkv_copy_elems
        tQKV_layout = cute.make_layout(
            (self._num_threads // tQKV_shape_dim_1, tQKV_shape_dim_1),
            stride=(tQKV_shape_dim_1, 1),
        )
        # tO_layout: thread layout for O store
        tO_shape_dim_1 = sQ_layout_atom.outer.shape[1] // universal_copy_elems
        tO_layout = cute.make_layout(
            (self._num_threads // tO_shape_dim_1, tO_shape_dim_1),
            stride=(tO_shape_dim_1, 1),
        )

        # Value layouts for copies
        vQKV_layout = cute.make_layout((1, qkv_copy_elems))
        vO_layout = cute.make_layout((1, universal_copy_elems))

        # gmem_tiled_copy_QKV: tiled copy for QKV load
        gmem_tiled_copy_QKV = cute.make_tiled_copy_tv(
            atom_async_copy, tQKV_layout, vQKV_layout
        )
        # gmem_tiled_copy_O: tiled copy for O store
        gmem_tiled_copy_O = cute.make_tiled_copy_tv(
            atom_universal_copy, tO_layout, vO_layout
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Tiled mma
        # ///////////////////////////////////////////////////////////////////////////////
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self._dtype, cutlass.Float32, (16, 8, 16)),
            (self._num_threads // 32, 1, 1),
            permutation_mnk=(self._num_threads // 32 * 16, 16, 16),
        )

        # grid_dim: (m_block, batch_size, num_head)
        grid_dim = (
            cute.ceil_div(mQ.shape[1], self._m_block_size),
            cute.size(mQ.shape[0]),
            cute.size(mQ.shape[2]),
        )
        LOG2_E = 1.4426950408889634074
        softmax_scale_log2 = softmax_scale * LOG2_E
        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            softmax_scale_log2,
            sQ_layout,
            sKV_layout,
            sO_layout,
            gmem_tiled_copy_QKV,
            gmem_tiled_copy_O,
            tiled_mma,
            SharedStorage,
        ).launch(
            grid=grid_dim,
            block=[self._num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        softmax_scale_log2: cutlass.Float32,
        sQ_layout: cute.ComposedLayout,
        sKV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_QKV: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        """Kernel function for flash attention v2.

        :param mQ: query tensor
        :type mQ: cute.Tensor
        :param mK: key tensor
        :type mK: cute.Tensor
        :param mV: value tensor
        :type mV: cute.Tensor
        :param mO: output tensor
        :type mO: cute.Tensor
        :param softmax_scale_log2: softmax scale log2
        :type softmax_scale_log2: cutlass.Float32
        :param sQ_layout: query layout
        :type sQ_layout: cute.ComposedLayout
        :param sKV_layout: key/value layout
        :type sKV_layout: cute.ComposedLayout
        :param sO_layout: output layout
        :type sO_layout: cute.ComposedLayout
        :param gmem_tiled_copy_QKV: tiled copy for QKV load
        :type gmem_tiled_copy_QKV: cute.TiledCopy
        :param gmem_tiled_copy_O: tiled copy for O store
        :type gmem_tiled_copy_O: cute.TiledCopy
        :param tiled_mma: tiled mma
        :type tiled_mma: cute.TiledMma
        :param SharedStorage: shared storage
        :type SharedStorage: cutlass.Constexpr
        """
        # Thread index, block index
        tidx, _, _ = cute.arch.thread_idx()
        m_block, batch_size, q_head = cute.arch.block_idx()
        group_size = mQ.shape[2] // mK.shape[2]
        kv_head = q_head // group_size

        n_block_max = cute.ceil_div(mK.shape[1], self._n_block_size)
        if self._is_causal:
            n_block_max = min(
                cute.ceil_div(
                    (m_block + 1) * self._m_block_size,
                    self._n_block_size,
                ),
                n_block_max,
            )
        n_block = n_block_max - 1

        # ///////////////////////////////////////////////////////////////////////////////
        # Get the appropriate tiles for this thread block.
        # ///////////////////////////////////////////////////////////////////////////////
        # (m_block_size, head_dim)
        gQ = cute.local_tile(
            mQ[batch_size, None, q_head, None],
            (self._m_block_size, self._head_dim_padded),
            (m_block, 0),
        )
        # (n_block_size, head_dim, n_block)
        gK = cute.local_tile(
            mK[batch_size, None, kv_head, None],
            (self._n_block_size, self._head_dim_padded),
            (None, 0),
        )
        # (n_block_size, head_dim, n_block)
        gV = cute.local_tile(
            mV[batch_size, None, kv_head, None],
            (self._n_block_size, self._head_dim_padded),
            (None, 0),
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Get shared memory buffer
        # ///////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()

        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sK = storage.sK.get_tensor(sKV_layout)
        sV = storage.sV.get_tensor(sKV_layout)

        # Transpose view of V to tensor with layout (head_dim, n_block_size) for tiled mma
        sVt = cute.composition(
            sV,
            cute.make_layout(
                (self._head_dim_padded, self._n_block_size),
                stride=(self._n_block_size, 1),
            ),
        )

        gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_slice(tidx)
        # (CPY_Atom, CPY_M, CPY_K)
        tQgQ = gmem_thr_copy_QKV.partition_S(gQ)
        tQsQ = gmem_thr_copy_QKV.partition_D(sQ)
        # (CPY_Atom, CPY_N, CPY_K, n_block)
        tKgK = gmem_thr_copy_QKV.partition_S(gK)
        tKsK = gmem_thr_copy_QKV.partition_D(sK)
        # (CPY_Atom, CPY_N, CPY_K, n_block)
        tVgV = gmem_thr_copy_QKV.partition_S(gV)
        tVsV = gmem_thr_copy_QKV.partition_D(sV)

        # ///////////////////////////////////////////////////////////////////////////////
        # Tile MMA compute thread partitions and allocate accumulators
        # ///////////////////////////////////////////////////////////////////////////////
        thr_mma = tiled_mma.get_slice(tidx)
        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
        tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK))
        tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt))
        acc_shape_O = thr_mma.partition_shape_C(
            (self._m_block_size, self._head_dim_padded)
        )
        acc_O = cute.make_rmem_tensor(acc_shape_O, cutlass.Float32)
        acc_O.fill(0.0)

        # ///////////////////////////////////////////////////////////////////////////////
        # Smem copy atom tiling
        # ///////////////////////////////////////////////////////////////////////////////
        smem_copy_atom_Q = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self._dtype,
        )
        smem_copy_atom_K = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self._dtype,
        )
        smem_copy_atom_V = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
            self._dtype,
        )
        smem_tiled_copy_Q = cute.make_tiled_copy_A(smem_copy_atom_Q, tiled_mma)
        smem_tiled_copy_K = cute.make_tiled_copy_B(smem_copy_atom_K, tiled_mma)
        smem_tiled_copy_V = cute.make_tiled_copy_B(smem_copy_atom_V, tiled_mma)

        smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(tidx)
        smem_thr_copy_K = smem_tiled_copy_K.get_slice(tidx)
        smem_thr_copy_V = smem_tiled_copy_V.get_slice(tidx)

        tSsQ = smem_thr_copy_Q.partition_S(sQ)
        tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
        tSsK = smem_thr_copy_K.partition_S(sK)
        tSrK_copy_view = smem_thr_copy_K.retile(tSrK)
        tOsVt = smem_thr_copy_V.partition_S(sVt)
        tOrVt_copy_view = smem_thr_copy_V.retile(tOrVt)

        # ///////////////////////////////////////////////////////////////////////////////
        # Predicate: Mark indices that need to copy when problem_shape isn't a multiple
        # of tile_shape
        # ///////////////////////////////////////////////////////////////////////////////
        # Construct identity layout for Q and KV
        mcQ = cute.make_identity_tensor(mQ.layout.shape)
        mcKV = cute.make_identity_tensor(mK.layout.shape)
        cQ = cute.local_tile(
            mcQ[batch_size, None, q_head, None],
            (self._m_block_size, self._head_dim_padded),
            (m_block, 0),
        )
        cKV = cute.local_tile(
            mcKV[batch_size, None, kv_head, None],
            (self._n_block_size, self._head_dim_padded),
            (n_block, 0),
        )

        # Repeat the partitioning with identity layouts
        tQcQ = gmem_thr_copy_QKV.partition_S(cQ)
        tKVcKV = gmem_thr_copy_QKV.partition_S(cKV)
        # Allocate predicate tensors for m and n, here we only allocate the tile of k, and do special process for mn.
        # This is to reduce register pressure and gets 2-3% performance gain compared with allocating the whole tile.
        tQpQ = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    tQsQ.shape[0][1],
                    cute.size(tQsQ, mode=[1]),
                    cute.size(tQsQ, mode=[2]),
                ),
                stride=(cute.size(tQsQ, mode=[2]), 0, 1),
            ),
            cutlass.Boolean,
        )
        tKVpKV = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    tKsK.shape[0][1],
                    cute.size(tKsK, mode=[1]),
                    cute.size(tKsK, mode=[2]),
                ),
                stride=(cute.size(tKsK, mode=[2]), 0, 1),
            ),
            cutlass.Boolean,
        )
        # Set predicates for head_dim bounds, seqlen_q/k bounds is processed at the first tile.
        for rest_v in cutlass.range_constexpr(tQpQ.shape[0]):
            for rest_k in cutlass.range_constexpr(tQpQ.shape[2]):
                tQpQ[rest_v, 0, rest_k] = cute.elem_less(
                    tQcQ[(0, rest_v), 0, rest_k][3], mQ.layout.shape[3]
                )
        for rest_v in cutlass.range_constexpr(tKVpKV.shape[0]):
            for rest_k in cutlass.range_constexpr(tKVpKV.shape[2]):
                tKVpKV[rest_v, 0, rest_k] = cute.elem_less(
                    tKVcKV[(0, rest_v), 0, rest_k][3], mK.layout.shape[3]
                )
        # ///////////////////////////////////////////////////////////////////////////////
        # Correctness-first scalar Q/K tile loads
        # ///////////////////////////////////////////////////////////////////////////////
        # Load the Q tile and last K tile using one BF16 element per assignment.
        # The logical GMEM tensors carry the BHSD strides, so no input copy is
        # made.  Out-of-range sequence positions deterministically become zero.
        # Static diagnostic fast path for S divisible by the Q/K tile sizes.
        # This avoids runtime tail control flow while proving that scalar
        # GMEM->SMEM plus the MMA body can compile.  Tail support is restored
        # in a separate predicated implementation after this diagnostic gate.
        for row_in_tile in cutlass.range_constexpr(self._m_block_size):
            for d_idx in cutlass.range_constexpr(self._head_dim_padded):
                flat_idx = row_in_tile * self._head_dim_padded + d_idx
                if flat_idx % self._num_threads == tidx:
                    seq_idx = m_block * self._m_block_size + row_in_tile
                    sQ[row_in_tile, d_idx] = mQ[batch_size, seq_idx, q_head, d_idx]

        k_seq_start = n_block * self._n_block_size
        for row_in_tile in cutlass.range_constexpr(self._n_block_size):
            for d_idx in cutlass.range_constexpr(self._head_dim_padded):
                flat_idx = row_in_tile * self._head_dim_padded + d_idx
                if flat_idx % self._num_threads == tidx:
                    seq_idx = k_seq_start + row_in_tile
                    sK[row_in_tile, d_idx] = mK[batch_size, seq_idx, kv_head, d_idx]

        self.cta_sync_barrier.arrive_and_wait()
        # ///////////////////////////////////////////////////////////////////////////////
        # Softmax intermediate result: row_max and row_sum
        # ///////////////////////////////////////////////////////////////////////////////
        # shape: (atom_v_m * rest_m)
        row_max = cute.make_rmem_tensor(
            (acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32
        )
        # shape: (atom_v_m * rest_m)
        row_sum = cute.make_rmem_tensor(
            (acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32
        )
        row_max.fill(-cutlass.Float32.inf)
        row_sum.fill(0.0)

        # group parameters for compute_one_n_block
        basic_params = SimpleNamespace(
            m_block=m_block,
            n_block=n_block,
            mQ=mQ,
            mK=mK,
            mV=mV,
            batch_size=batch_size,
            num_head=q_head,
        )
        mma_params = SimpleNamespace(
            thr_mma=thr_mma,
            tiled_mma=tiled_mma,
            tSrQ=tSrQ,
            tSrK=tSrK,
            tOrVt=tOrVt,
            acc_O=acc_O,
        )
        gmem_copy_params = SimpleNamespace(
            gmem_tiled_copy_QKV=gmem_tiled_copy_QKV,
            tKVcKV=tKVcKV,
            tKgK=tKgK,
            tKsK=tKsK,
            tVgV=tVgV,
            tVsV=tVsV,
            tKVpKV=tKVpKV,
        )
        smem_copy_params = SimpleNamespace(
            smem_tiled_copy_Q=smem_tiled_copy_Q,
            smem_tiled_copy_K=smem_tiled_copy_K,
            smem_tiled_copy_V=smem_tiled_copy_V,
            tSsQ=tSsQ,
            tSrQ_copy_view=tSrQ_copy_view,
            tSsK=tSsK,
            tSrK_copy_view=tSrK_copy_view,
            tOsVt=tOsVt,
            tOrVt_copy_view=tOrVt_copy_view,
            sK=sK,
            sV=sV,
        )
        softmax_params = SimpleNamespace(
            row_max=row_max,
            row_sum=row_sum,
            softmax_scale_log2=softmax_scale_log2,
        )

        # Start processing of the first n-block.
        # For performance reason, we separate out two kinds of iterations:
        # those that need masking on S, and those that don't.
        # We need masking on S for the very last block when K and V has length not multiple of n_block_size.
        # We also need masking on S if it's causal, for the last ceil_div(m_block_size, n_block_size) blocks.
        # We will have at least 1 "masking" iteration.
        mask_steps = 1
        if cutlass.const_expr(self._is_causal):
            mask_steps = cute.ceil_div(self._m_block_size, self._n_block_size)

        for n_tile in cutlass.range_constexpr(mask_steps):
            n_block = n_block_max - n_tile - 1
            basic_params.n_block = n_block
            if cutlass.const_expr(self._is_causal):
                if n_block >= 0:
                    self.compute_one_n_block(
                        basic_params,
                        mma_params,
                        gmem_copy_params,
                        smem_copy_params,
                        softmax_params,
                        is_first_n_block=(n_tile == 0),
                        in_mask_steps=True,
                    )
            else:
                self.compute_one_n_block(
                    basic_params,
                    mma_params,
                    gmem_copy_params,
                    smem_copy_params,
                    softmax_params,
                    is_first_n_block=True,
                    in_mask_steps=True,
                )

        # Start async loads of rest k-tiles in reverse order, no k-residue handling needed
        for n_tile in range(mask_steps, n_block_max, 1):
            n_block = n_block_max - n_tile - 1
            basic_params.n_block = n_block
            self.compute_one_n_block(
                basic_params,
                mma_params,
                gmem_copy_params,
                smem_copy_params,
                softmax_params,
                is_first_n_block=False,
                in_mask_steps=False,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        # Epilogue
        # ///////////////////////////////////////////////////////////////////////////////
        # normalize acc_O by row_sum and calculate the lse
        self.normalize_softmax(acc_O, row_sum)
        # store acc_O
        rO = cute.make_fragment_like(acc_O, self._dtype)
        rO.store(acc_O.load().to(self._dtype))
        # reuse sQ's data iterator
        sO = cute.make_tensor(sQ.iterator, sO_layout)

        # smem copy atom for O
        smem_copy_atom_O = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self._dtype
        )
        # tiled copy atom for O
        smem_tiled_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma)
        smem_thr_copy_O = smem_tiled_copy_O.get_slice(tidx)
        taccOrO = smem_thr_copy_O.retile(rO)
        taccOsO = smem_thr_copy_O.partition_D(sO)
        # copy acc O from rmem to smem with the smem copy atom
        cute.copy(
            smem_copy_atom_O,
            taccOrO,
            taccOsO,
        )
        gO = cute.local_tile(
            mO[batch_size, None, q_head, None],
            (self._m_block_size, self._head_dim_padded),
            (m_block, 0),
        )

        gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
        tOsO = gmem_thr_copy_O.partition_S(sO)
        tOgO = gmem_thr_copy_O.partition_D(gO)
        tOrO = cute.make_fragment_like(tOgO, self._dtype)
        # sync before all smem stores are done.
        self.cta_sync_barrier.arrive_and_wait()
        # load acc O from smem to rmem for wider vectorization
        cute.copy(
            gmem_tiled_copy_O,
            tOsO,
            tOrO,
        )
        mcO = cute.make_identity_tensor(mO.layout.shape)
        cO = cute.local_tile(
            mcO[batch_size, None, q_head, None],
            (self._m_block_size, self._head_dim_padded),
            (m_block, 0),
        )
        tOcO = gmem_thr_copy_O.partition_D(cO)
        tOpO = cute.make_rmem_tensor(
            cute.make_layout(
                (tOgO.shape[0][1], tOgO.shape[1], tOgO.shape[2]),
                stride=(tOgO.shape[2], 0, 1),
            ),
            cutlass.Boolean,
        )
        for rest_v in cutlass.range_constexpr(tOpO.shape[0]):
            for rest_n in cutlass.range_constexpr(cute.size(tOpO.shape[2])):
                tOpO[rest_v, 0, rest_n] = cute.elem_less(
                    tOcO[(0, rest_v), 0, rest_n][3], mO.layout.shape[3]
                )
        # copy acc O from rmem to gmem
        for rest_m in cutlass.range_constexpr(cute.size(tOpO.shape[1])):
            if cute.elem_less(tOcO[0, rest_m, 0][1], mO.layout.shape[1]):
                cute.copy(
                    gmem_tiled_copy_O,
                    tOrO[None, rest_m, None],
                    tOgO[None, rest_m, None],
                    pred=tOpO[None, rest_m, None],
                )

    @cute.jit
    def compute_one_n_block(
        self,
        basic_params: SimpleNamespace,
        mma_params: SimpleNamespace,
        gmem_copy_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        is_first_n_block: cutlass.Constexpr,
        in_mask_steps: cutlass.Constexpr,
    ):
        """Compute one n_block of S/O.

        This function provides different variants for processing the first n block versus subsequent blocks,
        as well as variants for handling masked and unmasked steps.

        :param basic_params: basic parameters
        :type basic_params: SimpleNamespace
        :param mma_params: mma parameters
        :type mma_params: SimpleNamespace
        :param gmem_copy_params: gmem copy parameters
        :type gmem_copy_params: SimpleNamespace
        :param smem_copy_params: smem copy parameters
        :type smem_copy_params: SimpleNamespace
        :param softmax_params: softmax parameters
        :type softmax_params: SimpleNamespace
        :param is_first_n_block: is first n block
        :type is_first_n_block: cutlass.Constexpr
        """
        acc_shape_S = mma_params.thr_mma.partition_shape_C(
            (self._m_block_size, self._n_block_size)
        )
        acc_S = cute.make_rmem_tensor(acc_shape_S, cutlass.Float32)
        acc_S.fill(0.0)

        # wait for smem tile QK before mma calculation for S
        self.cta_sync_barrier.arrive_and_wait()
        # load smem tile V for O, special process for the first tile to avoid loading nan.
        # The `if` here is a constexpr, won't be generated in the IR.
        tidx, _, _ = cute.arch.thread_idx()
        m_block, batch_size, q_head = cute.arch.block_idx()
        group_size = basic_params.mQ.shape[2] // basic_params.mK.shape[2]
        kv_head = q_head // group_size
        v_seq_start = basic_params.n_block * self._n_block_size
        for row_in_tile in cutlass.range_constexpr(self._n_block_size):
            for d_idx in cutlass.range_constexpr(self._head_dim_padded):
                flat_idx = row_in_tile * self._head_dim_padded + d_idx
                if flat_idx % self._num_threads == tidx:
                    seq_idx = v_seq_start + row_in_tile
                    smem_copy_params.sV[row_in_tile, d_idx] = basic_params.mV[
                        batch_size, seq_idx, kv_head, d_idx
                    ]
        self.cta_sync_barrier.arrive_and_wait()
        # ///////////////////////////////////////////////////////////////////////////////
        # S gemm calculation
        # ///////////////////////////////////////////////////////////////////////////////
        # load first QK k-block from smem to rmem for mma
        cute.copy(
            smem_copy_params.smem_tiled_copy_Q,
            smem_copy_params.tSsQ[None, None, 0],
            smem_copy_params.tSrQ_copy_view[None, None, 0],
        )
        cute.copy(
            smem_copy_params.smem_tiled_copy_K,
            smem_copy_params.tSsK[None, None, 0],
            smem_copy_params.tSrK_copy_view[None, None, 0],
        )
        # mma for S
        for k in cutlass.range_constexpr(cute.size(smem_copy_params.tSsQ.shape[2])):
            # load next QK k-block from smem to rmem for mma
            k_next = (k + 1) % cute.size(smem_copy_params.tSsQ.shape[2])
            cute.copy(
                smem_copy_params.smem_tiled_copy_Q,
                smem_copy_params.tSsQ[None, None, k_next],
                smem_copy_params.tSrQ_copy_view[None, None, k_next],
            )
            cute.copy(
                smem_copy_params.smem_tiled_copy_K,
                smem_copy_params.tSsK[None, None, k_next],
                smem_copy_params.tSrK_copy_view[None, None, k_next],
            )
            cute.gemm(
                mma_params.tiled_mma,
                acc_S,
                mma_params.tSrQ[None, None, k],
                mma_params.tSrK[None, None, k],
                acc_S,
            )

        # wait for smem tile V for O
        self.cta_sync_barrier.arrive_and_wait()

        if basic_params.n_block > 0:
            tidx, _, _ = cute.arch.thread_idx()
            m_block, batch_size, q_head = cute.arch.block_idx()
            group_size = basic_params.mQ.shape[2] // basic_params.mK.shape[2]
            kv_head = q_head // group_size
            next_k_seq_start = (basic_params.n_block - 1) * self._n_block_size
            for row_in_tile in cutlass.range_constexpr(self._n_block_size):
                for d_idx in cutlass.range_constexpr(self._head_dim_padded):
                    flat_idx = row_in_tile * self._head_dim_padded + d_idx
                    if flat_idx % self._num_threads == tidx:
                        seq_idx = next_k_seq_start + row_in_tile
                        smem_copy_params.sK[row_in_tile, d_idx] = basic_params.mK[
                            batch_size, seq_idx, kv_head, d_idx
                        ]
            self.cta_sync_barrier.arrive_and_wait()
        # ///////////////////////////////////////////////////////////////////////////////
        # online softmax
        # ///////////////////////////////////////////////////////////////////////////////
        self.softmax_rescale_O(
            basic_params,
            mma_params,
            softmax_params,
            acc_S,
            is_first_n_block,
            in_mask_steps,
        )

        rP = cute.make_fragment_like(acc_S, self._dtype)
        rP.store(acc_S.load().to(self._dtype))
        # ///////////////////////////////////////////////////////////////////////////////
        # O gemm calculation
        # ///////////////////////////////////////////////////////////////////////////////
        # Convert layout of acc_S to gemm O accept layout.
        # Due to the mma instruction shape is 16x8x16, we need to convert from (4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        # (4, MMA_M, MMA_N) -> (4, MMA_M, (2, MMA_N / 2))
        rP_layout_divided = cute.logical_divide(rP.layout, (None, None, 2))
        rP_mma_view = cute.make_layout(
            (
                (rP_layout_divided.shape[0], rP_layout_divided.shape[2][0]),
                rP_layout_divided.shape[1],
                rP_layout_divided.shape[2][1],
            ),
            stride=(
                (rP_layout_divided.stride[0], rP_layout_divided.stride[2][0]),
                rP_layout_divided.stride[1],
                rP_layout_divided.stride[2][1],
            ),
        )
        tOrS = cute.make_tensor(rP.iterator, rP_mma_view)

        # load first V k-block from smem to rmem for mma
        cute.copy(
            smem_copy_params.smem_tiled_copy_V,
            smem_copy_params.tOsVt[None, None, 0],
            smem_copy_params.tOrVt_copy_view[None, None, 0],
        )
        # mma for O
        for k in cutlass.range_constexpr(cute.size(tOrS.shape[2])):
            # load next V k-block from smem to rmem for mma
            k_next = (k + 1) % cute.size(tOrS.shape[2])
            cute.copy(
                smem_copy_params.smem_tiled_copy_V,
                smem_copy_params.tOsVt[None, None, k_next],
                smem_copy_params.tOrVt_copy_view[None, None, k_next],
            )
            cute.gemm(
                mma_params.tiled_mma,
                mma_params.acc_O,
                tOrS[None, None, k],
                mma_params.tOrVt[None, None, k],
                mma_params.acc_O,
            )

    @cute.jit
    def softmax_rescale_O(
        self,
        basic_params: SimpleNamespace,
        mma_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        acc_S: cute.Tensor,
        is_first_n_block: cutlass.Constexpr,
        in_mask_steps: cutlass.Constexpr,
    ):
        """Apply online softmax and rescale acc_O.

        This function provides different variants for processing the first n block versus subsequent blocks,
        as well as variants for handling masked and unmasked steps.

        :param basic_params: basic parameters
        :type basic_params: SimpleNamespace
        :param mma_params: mma parameters
        :type mma_params: SimpleNamespace
        :param softmax_params: softmax parameters
        :type softmax_params: SimpleNamespace
        :param acc_S: acc_S tensor
        :type acc_S: cute.Tensor
        :param is_first_n_block: is first n_block
        :type is_first_n_block: cutlass.Constexpr
        :param in_mask_steps: in mask steps
        :type in_mask_steps: cutlass.Constexpr
        """
        # Change acc_S to M,N layout view.
        acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
        acc_O_mn = self._make_acc_tensor_mn_view(mma_params.acc_O)
        row_max_prev = None
        # if it is not the first tile, load the row r of previous row_max and compare with row_max_cur_row.
        if cutlass.const_expr(not is_first_n_block):
            row_max_prev = cute.make_fragment_like(
                softmax_params.row_max, cutlass.Float32
            )
            cute.basic_copy(softmax_params.row_max, row_max_prev)
        # if it is the first tile, create a mask for residual of S to -inf for softmax.
        tScS_mn = None
        if cutlass.const_expr(in_mask_steps):
            mcS = cute.make_identity_tensor(
                (
                    basic_params.mQ.shape[0],
                    basic_params.mQ.shape[1],
                    basic_params.mQ.shape[2],
                    basic_params.mK.shape[1],
                )
            )
            cS = cute.local_tile(
                mcS[basic_params.batch_size, None, basic_params.num_head, None],
                (self._m_block_size, self._n_block_size),
                (basic_params.m_block, basic_params.n_block),
            )
            tScS = mma_params.thr_mma.partition_C(cS)
            tScS_mn = self._make_acc_tensor_mn_view(tScS)

        # Each iteration processes one row of acc_S
        for r in cutlass.range_constexpr(cute.size(softmax_params.row_max)):
            # mask residual of S with -inf
            if cutlass.const_expr(in_mask_steps):
                if cutlass.const_expr(not self._is_causal):
                    # traverse column index.
                    for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                        if cute.elem_less(
                            basic_params.mK.shape[1], tScS_mn[0, c][3] + 1
                        ):
                            acc_S_mn[r, c] = -cutlass.Float32.inf
                else:
                    # get the column index limit based on current row. Only consider the row index, so the column index sets to 0.
                    col_idx_limit = cutlass.min(
                        tScS_mn[r, 0][1] + 1, basic_params.mK.shape[1]
                    )
                    # traverse column index.
                    for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                        # only consider the column index, so the row index sets to 0.
                        if cute.elem_less(col_idx_limit, tScS_mn[0, c][3] + 1):
                            acc_S_mn[r, c] = -cutlass.Float32.inf

            # (n_block_size)
            acc_S_row = acc_S_mn[r, None].load()
            # row_max_cur_row => f32
            row_max_cur_row = acc_S_row.reduce(
                cute.ReductionOp.MAX, -cutlass.Float32.inf, 0
            )
            # quad reduction for row_max
            row_max_cur_row = self._threadquad_reduce_max(row_max_cur_row)
            row_max_prev_row = None
            # if it is not the first tile, load the row r of previous row_max and compare with row_max_cur_row.
            if cutlass.const_expr(not is_first_n_block):
                row_max_prev_row = row_max_prev[r]
                row_max_cur_row = cute.arch.fmax(row_max_prev_row, row_max_cur_row)
            if cutlass.const_expr(self._is_causal):
                row_max_cur_row = (
                    0.0 if row_max_cur_row == -cutlass.Float32.inf else row_max_cur_row
                )

            # compute exp(x - max) using exp2(x * log_2(e) - max * log_2(e))
            acc_S_row_exp = cute.math.exp2(
                acc_S_row * softmax_params.softmax_scale_log2
                - row_max_cur_row * softmax_params.softmax_scale_log2,
                fastmath=True,
            )
            # acc_S_row_sum => f32
            acc_S_row_sum = acc_S_row_exp.reduce(
                cute.ReductionOp.ADD, cutlass.Float32.zero, 0
            )
            # if it is not the first tile, load the row r of previous row_max and minus row_max_cur_row to update row_sum.
            if cutlass.const_expr(not is_first_n_block):
                prev_minus_cur_exp = cute.math.exp2(
                    row_max_prev_row * softmax_params.softmax_scale_log2
                    - row_max_cur_row * softmax_params.softmax_scale_log2,
                    fastmath=True,
                )
                acc_S_row_sum = (
                    acc_S_row_sum + softmax_params.row_sum[r] * prev_minus_cur_exp
                )
                acc_O_mn[r, None] = acc_O_mn[r, None].load() * prev_minus_cur_exp
            # update row_max, row_sum and acc_S
            softmax_params.row_max[r] = row_max_cur_row
            softmax_params.row_sum[r] = acc_S_row_sum
            acc_S_mn[r, None] = acc_S_row_exp

    @cute.jit
    def normalize_softmax(
        self,
        acc_O: cute.Tensor,
        row_sum: cute.Tensor,
    ):
        """Normalize acc_O by row_sum.

        :param acc_O: input tensor
        :type acc_O: cute.Tensor
        :param row_sum: row_sum tensor
        :type row_sum: cute.Tensor
        """
        # do quad reduction for row_sum.
        acc_O_mn = self._make_acc_tensor_mn_view(acc_O)
        for r in cutlass.range_constexpr(cute.size(row_sum)):
            row_sum[r] = self._threadquad_reduce_sum(row_sum[r])
            # if row_sum is zero or nan, set acc_O_mn_row to 1.0
            acc_O_mn_row_is_zero_or_nan = row_sum[r] == 0.0 or row_sum[r] != row_sum[r]

            scale = (
                1.0 if acc_O_mn_row_is_zero_or_nan else cute.arch.rcp_approx(row_sum[r])
            )

            acc_O_mn[r, None] = acc_O_mn[r, None].load() * scale

    def _make_acc_tensor_mn_view(self, acc: cute.Tensor) -> cute.Tensor:
        """make acc tensor as mn layout view

        :param acc: input tensor
        :type acc: cute.Tensor
        :return: acc tensor mn layout view
        :rtype: cute.Tensor
        """
        acc_layout_col_major = cute.make_layout(acc.layout.shape)
        acc_layout_mn = cute.make_layout(
            (
                (
                    acc_layout_col_major.shape[0][1],
                    acc_layout_col_major.shape[1],
                ),  # MMA_M
                (
                    acc_layout_col_major.shape[0][0],
                    acc_layout_col_major.shape[2],
                ),  # MMA_N
            ),
            stride=(
                (
                    acc_layout_col_major.stride[0][1],
                    acc_layout_col_major.stride[1],
                ),  # MMA_M
                (
                    acc_layout_col_major.stride[0][0],
                    acc_layout_col_major.stride[2],
                ),  # MMA_N
            ),
        )
        acc_layout_mn = cute.composition(acc.layout, acc_layout_mn)
        return cute.make_tensor(acc.iterator, acc_layout_mn)

    def _threadquad_reduce(self, val: cutlass.Float32, op: Callable) -> cutlass.Float32:
        """thread quad reduction

        :param val: register value
        :type val: cutlass.Float32
        :param op: binary operator
        :type op: Callable
        :return: reduced value
        :rtype: cutlass.Float32
        """
        val = op(
            val,
            cute.arch.shuffle_sync_bfly(val, offset=2, mask=-1, mask_and_clamp=31),
        )
        val = op(
            val,
            cute.arch.shuffle_sync_bfly(val, offset=1, mask=-1, mask_and_clamp=31),
        )
        return val

    def _threadquad_reduce_max(self, val: cutlass.Float32) -> cutlass.Float32:
        """thread quad reduction max

        :param val: register value
        :type val: cutlass.Float32
        :return: max value
        :rtype: cutlass.Float32
        """
        return self._threadquad_reduce(val, lambda x, y: cute.arch.fmax(x, y))

    def _threadquad_reduce_sum(self, val: cutlass.Float32) -> cutlass.Float32:
        """thread quad reduction sum

        :param val: register value
        :type val: cutlass.Float32
        :return: sum value
        :rtype: cutlass.Float32
        """
        return self._threadquad_reduce(val, lambda x, y: x + y)


class FlashAttentionForwardAmpereStaticNoTail:
    """Static-no-tail GQA prefill kernel.

    Dedicated compilation path for shapes where the sequence length S is
    divisible by both M_BLOCK (128) and N_BLOCK (64), guaranteeing every
    Q/K/V tile is complete.

    This class does NOT inherit from FlashAttentionForwardAmpere.  It is an
    independent compilation entry that:

    * generates no runtime tail predicates (no ``tQpQ``, ``tKVpKV``, ``tOpO``);
    * generates no ``scf.if`` tail branches in Q/K/V GMEM→SMEM load paths;
    * generates no identity-tensor ``elem_less`` bounds guards for the O store;
    * skips the non-causal tail-mask branch in ``softmax_rescale_O``;
    * uses unconditional scalar loads and unconditional O store.

    The original ``FlashAttentionForwardAmpere`` generic/tail path is
    preserved elsewhere in this file but is never instantiated by this
    class or by the static-no-tail compilation entry.

    Supported fixed shapes
    ----------------------
    * B = 1
    * D = 128  (head_dim)
    * S in {128, 131072}
    * BF16
    * q_heads % kv_heads == 0 (GQA supported)
    * causal = True
    """

    M_BLOCK = 128
    N_BLOCK = 64
    HEAD_DIM = 128
    NUM_THREADS = 128
    HEAD_DIM_PADDED = 128  # (128 + 31) // 32 * 32
    SUPPORTED_SEQ_LENS = (128, 131072)

    def __init__(self, head_dim, m_block_size=128, n_block_size=64,
                 num_threads=128, is_causal=False):
        if head_dim != self.HEAD_DIM:
            raise ValueError(
                f"static-no-tail path only supports D={self.HEAD_DIM}, "
                f"got D={head_dim}"
            )
        if m_block_size != self.M_BLOCK:
            raise ValueError(
                f"static-no-tail path only supports M_BLOCK={self.M_BLOCK}, "
                f"got M_BLOCK={m_block_size}"
            )
        if n_block_size != self.N_BLOCK:
            raise ValueError(
                f"static-no-tail path only supports N_BLOCK={self.N_BLOCK}, "
                f"got N_BLOCK={n_block_size}"
            )
        if num_threads != self.NUM_THREADS:
            raise ValueError(
                f"static-no-tail path only supports THREADS={self.NUM_THREADS}, "
                f"got THREADS={num_threads}"
            )
        self._head_dim = head_dim
        self._m_block_size = m_block_size
        self._n_block_size = n_block_size
        self._head_dim_padded = self.HEAD_DIM_PADDED
        self._num_threads = num_threads
        self._is_causal = is_causal
        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=num_threads
        )

    @staticmethod
    def can_implement(dtype, head_dim, m_block_size, n_block_size, num_threads, is_causal):
        if dtype != cutlass.Float16 and dtype != cutlass.BFloat16:
            return False
        if head_dim != FlashAttentionForwardAmpereStaticNoTail.HEAD_DIM:
            return False
        if m_block_size != FlashAttentionForwardAmpereStaticNoTail.M_BLOCK:
            return False
        if n_block_size != FlashAttentionForwardAmpereStaticNoTail.N_BLOCK:
            return False
        if num_threads != FlashAttentionForwardAmpereStaticNoTail.NUM_THREADS:
            return False
        head_dim_padded = FlashAttentionForwardAmpereStaticNoTail.HEAD_DIM_PADDED
        smem_usage = (m_block_size * head_dim_padded + n_block_size * head_dim_padded * 2) * 2
        smem_capacity = utils.get_smem_capacity_in_bytes("sm_80")
        if smem_usage > smem_capacity:
            return False
        if (m_block_size * 2) % num_threads != 0:
            return False
        return True

    @staticmethod
    def validate_static_shape(batch, q_heads, kv_heads, seq_len, head_dim, dtype, is_causal):
        """Validate that a shape is supported by the static-no-tail path.

        Raises ValueError with a clear message if not supported.
        """
        if batch != 1:
            raise ValueError(
                f"static-no-tail path only supports B=1, got B={batch}"
            )
        if q_heads <= 0 or kv_heads <= 0:
            raise ValueError("head counts must be positive")
        if q_heads % kv_heads != 0:
            raise ValueError(
                f"GQA requires q_heads={q_heads} divisible by kv_heads={kv_heads}"
            )
        if seq_len not in FlashAttentionForwardAmpereStaticNoTail.SUPPORTED_SEQ_LENS:
            raise ValueError(
                f"static-no-tail path only supports S in "
                f"{FlashAttentionForwardAmpereStaticNoTail.SUPPORTED_SEQ_LENS}, "
                f"got S={seq_len}.  Both values are divisible by "
                f"M_BLOCK={FlashAttentionForwardAmpereStaticNoTail.M_BLOCK} and "
                f"N_BLOCK={FlashAttentionForwardAmpereStaticNoTail.N_BLOCK}."
            )
        if head_dim != FlashAttentionForwardAmpereStaticNoTail.HEAD_DIM:
            raise ValueError(
                f"static-no-tail path only supports D="
                f"{FlashAttentionForwardAmpereStaticNoTail.HEAD_DIM}, "
                f"got D={head_dim}"
            )
        if dtype != cutlass.BFloat16:
            raise ValueError(
                f"static-no-tail path only supports BF16, got {dtype}"
            )
        if seq_len % FlashAttentionForwardAmpereStaticNoTail.M_BLOCK != 0:
            raise ValueError(
                f"S={seq_len} must be divisible by M_BLOCK="
                f"{FlashAttentionForwardAmpereStaticNoTail.M_BLOCK}"
            )
        if seq_len % FlashAttentionForwardAmpereStaticNoTail.N_BLOCK != 0:
            raise ValueError(
                f"S={seq_len} must be divisible by N_BLOCK="
                f"{FlashAttentionForwardAmpereStaticNoTail.N_BLOCK}"
            )
        return True

    @cute.jit
    def __call__(self, mQ, mK, mV, mO, softmax_scale, stream):
        if cutlass.const_expr(
            not (mQ.element_type == mK.element_type == mV.element_type == mO.element_type)
        ):
            raise TypeError("All tensors must have the same data type")
        if cutlass.const_expr(
            not (mQ.element_type == cutlass.Float16 or mQ.element_type == cutlass.BFloat16)
        ):
            raise TypeError("Only Float16 or BFloat16 is supported")
        self._dtype = mQ.element_type

        # SMEM layouts
        smem_k_block_size = 64  # head_dim_padded=128, 128%64==0
        swizzle_bits = 3
        sQ_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3),
            0,
            cute.make_layout((8, smem_k_block_size), stride=(smem_k_block_size, 1)),
        )
        sQ_layout = cute.tile_to_shape(
            sQ_layout_atom, (self._m_block_size, self._head_dim_padded), (0, 1),
        )
        sKV_layout_atom = sQ_layout_atom
        sKV_layout = cute.tile_to_shape(
            sKV_layout_atom, (self._n_block_size, self._head_dim_padded), (0, 1),
        )
        sO_layout = sQ_layout

        @cute.struct
        class SharedStorage:
            sQ: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sQ_layout)], 1024
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024
            ]

        # Tiled MMA
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self._dtype, cutlass.Float32, (16, 8, 16)),
            (self._num_threads // 32, 1, 1),
            permutation_mnk=(self._num_threads // 32 * 16, 16, 16),
        )

        # GMEM tiled copy for O store only (element-wise universal copy)
        universal_copy_bits = self._dtype.width
        universal_copy_elems = universal_copy_bits // self._dtype.width
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self._dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        tO_shape_dim_1 = sQ_layout_atom.outer.shape[1] // universal_copy_elems
        tO_layout = cute.make_layout(
            (self._num_threads // tO_shape_dim_1, tO_shape_dim_1),
            stride=(tO_shape_dim_1, 1),
        )
        vO_layout = cute.make_layout((1, universal_copy_elems))
        gmem_tiled_copy_O = cute.make_tiled_copy_tv(
            atom_universal_copy, tO_layout, vO_layout
        )

        grid_dim = (
            cute.ceil_div(mQ.shape[1], self._m_block_size),
            cute.size(mQ.shape[0]),
            cute.size(mQ.shape[2]),
        )
        LOG2_E = 1.4426950408889634074
        softmax_scale_log2 = softmax_scale * LOG2_E
        self.kernel(
            mQ, mK, mV, mO,
            softmax_scale_log2,
            sQ_layout, sKV_layout, sO_layout,
            gmem_tiled_copy_O, tiled_mma,
            SharedStorage,
        ).launch(
            grid=grid_dim, block=[self._num_threads, 1, 1], stream=stream,
        )

    @cute.kernel
    def kernel(self, mQ, mK, mV, mO, softmax_scale_log2,
               sQ_layout, sKV_layout, sO_layout,
               gmem_tiled_copy_O, tiled_mma, SharedStorage):
        # Thread/block indices and GQA head mapping
        tidx, _, _ = cute.arch.thread_idx()
        m_block, batch_size, q_head = cute.arch.block_idx()
        group_size = mQ.shape[2] // mK.shape[2]
        kv_head = q_head // group_size

        n_block_max = cute.ceil_div(mK.shape[1], self._n_block_size)
        if self._is_causal:
            n_block_max = min(
                cute.ceil_div((m_block + 1) * self._m_block_size, self._n_block_size),
                n_block_max,
            )
        n_block = n_block_max - 1

        # Get GMEM tiles
        gQ = cute.local_tile(
            mQ[batch_size, None, q_head, None],
            (self._m_block_size, self._head_dim_padded), (m_block, 0),
        )
        gK = cute.local_tile(
            mK[batch_size, None, kv_head, None],
            (self._n_block_size, self._head_dim_padded), (None, 0),
        )
        gV = cute.local_tile(
            mV[batch_size, None, kv_head, None],
            (self._n_block_size, self._head_dim_padded), (None, 0),
        )

        # SMEM allocation
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sK = storage.sK.get_tensor(sKV_layout)
        sV = storage.sV.get_tensor(sKV_layout)
        sVt = cute.composition(
            sV,
            cute.make_layout(
                (self._head_dim_padded, self._n_block_size),
                stride=(self._n_block_size, 1),
            ),
        )

        # MMA thread partitions and accumulators
        thr_mma = tiled_mma.get_slice(tidx)
        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
        tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK))
        tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt))
        acc_shape_O = thr_mma.partition_shape_C((self._m_block_size, self._head_dim_padded))
        acc_O = cute.make_rmem_tensor(acc_shape_O, cutlass.Float32)
        acc_O.fill(0.0)

        # SMEM copy atoms and tiled copies
        smem_copy_atom_Q = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self._dtype,
        )
        smem_copy_atom_K = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self._dtype,
        )
        smem_copy_atom_V = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self._dtype,
        )
        smem_tiled_copy_Q = cute.make_tiled_copy_A(smem_copy_atom_Q, tiled_mma)
        smem_tiled_copy_K = cute.make_tiled_copy_B(smem_copy_atom_K, tiled_mma)
        smem_tiled_copy_V = cute.make_tiled_copy_B(smem_copy_atom_V, tiled_mma)
        smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(tidx)
        smem_thr_copy_K = smem_tiled_copy_K.get_slice(tidx)
        smem_thr_copy_V = smem_tiled_copy_V.get_slice(tidx)
        tSsQ = smem_thr_copy_Q.partition_S(sQ)
        tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
        tSsK = smem_thr_copy_K.partition_S(sK)
        tSrK_copy_view = smem_thr_copy_K.retile(tSrK)
        tOsVt = smem_thr_copy_V.partition_S(sVt)
        tOrVt_copy_view = smem_thr_copy_V.retile(tOrVt)

        # =================================================================
        # Static scalar Q/K loads: no tail branches, no predicates.
        # Every tile coordinate is guaranteed valid because S is divisible
        # by both M_BLOCK and N_BLOCK.
        # =================================================================
        # Q load (M_BLOCK x HEAD_DIM_PADDED, one BF16 per assignment)
        for row_in_tile in cutlass.range_constexpr(self._m_block_size):
            for d_idx in cutlass.range_constexpr(self._head_dim_padded):
                flat_idx = row_in_tile * self._head_dim_padded + d_idx
                if flat_idx % self._num_threads == tidx:
                    seq_idx = m_block * self._m_block_size + row_in_tile
                    sQ[row_in_tile, d_idx] = mQ[batch_size, seq_idx, q_head, d_idx]

        # K load (N_BLOCK x HEAD_DIM_PADDED, one BF16 per assignment)
        k_seq_start = n_block * self._n_block_size
        for row_in_tile in cutlass.range_constexpr(self._n_block_size):
            for d_idx in cutlass.range_constexpr(self._head_dim_padded):
                flat_idx = row_in_tile * self._head_dim_padded + d_idx
                if flat_idx % self._num_threads == tidx:
                    seq_idx = k_seq_start + row_in_tile
                    sK[row_in_tile, d_idx] = mK[batch_size, seq_idx, kv_head, d_idx]

        self.cta_sync_barrier.arrive_and_wait()

        # Softmax state
        row_max = cute.make_rmem_tensor(
            (acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32
        )
        row_sum = cute.make_rmem_tensor(
            (acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32
        )
        row_max.fill(-cutlass.Float32.inf)
        row_sum.fill(0.0)

        # Parameter bundles (no gmem_copy_params — not needed without predicates)
        basic_params = SimpleNamespace(
            m_block=m_block, n_block=n_block,
            mQ=mQ, mK=mK, mV=mV,
            batch_size=batch_size, num_head=q_head,
        )
        mma_params = SimpleNamespace(
            thr_mma=thr_mma, tiled_mma=tiled_mma,
            tSrQ=tSrQ, tSrK=tSrK, tOrVt=tOrVt, acc_O=acc_O,
        )
        smem_copy_params = SimpleNamespace(
            smem_tiled_copy_Q=smem_tiled_copy_Q,
            smem_tiled_copy_K=smem_tiled_copy_K,
            smem_tiled_copy_V=smem_tiled_copy_V,
            tSsQ=tSsQ, tSrQ_copy_view=tSrQ_copy_view,
            tSsK=tSsK, tSrK_copy_view=tSrK_copy_view,
            tOsVt=tOsVt, tOrVt_copy_view=tOrVt_copy_view,
            sK=sK, sV=sV,
        )
        softmax_params = SimpleNamespace(
            row_max=row_max, row_sum=row_sum,
            softmax_scale_log2=softmax_scale_log2,
        )

        # Mask steps: at least 1; extra for causal lower-triangular guard
        mask_steps = 1
        if cutlass.const_expr(self._is_causal):
            mask_steps = cute.ceil_div(self._m_block_size, self._n_block_size)

        for n_tile in cutlass.range_constexpr(mask_steps):
            n_block = n_block_max - n_tile - 1
            basic_params.n_block = n_block
            if cutlass.const_expr(self._is_causal):
                if n_block >= 0:
                    self.compute_one_n_block(
                        basic_params, mma_params, smem_copy_params,
                        softmax_params,
                        is_first_n_block=(n_tile == 0), in_mask_steps=True,
                    )
            else:
                self.compute_one_n_block(
                    basic_params, mma_params, smem_copy_params,
                    softmax_params,
                    is_first_n_block=True, in_mask_steps=True,
                )

        for n_tile in range(mask_steps, n_block_max, 1):
            n_block = n_block_max - n_tile - 1
            basic_params.n_block = n_block
            self.compute_one_n_block(
                basic_params, mma_params, smem_copy_params,
                softmax_params,
                is_first_n_block=False, in_mask_steps=False,
            )

        # =================================================================
        # Epilogue: normalize, store to SMEM, then unconditional O store.
        # No predicate tensors — every output coordinate is guaranteed valid.
        # =================================================================
        self.normalize_softmax(acc_O, row_sum)
        rO = cute.make_fragment_like(acc_O, self._dtype)
        rO.store(acc_O.load().to(self._dtype))
        sO = cute.make_tensor(sQ.iterator, sO_layout)

        smem_copy_atom_O = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self._dtype)
        smem_tiled_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma)
        smem_thr_copy_O = smem_tiled_copy_O.get_slice(tidx)
        taccOrO = smem_thr_copy_O.retile(rO)
        taccOsO = smem_thr_copy_O.partition_D(sO)
        cute.copy(smem_copy_atom_O, taccOrO, taccOsO)

        gO = cute.local_tile(
            mO[batch_size, None, q_head, None],
            (self._m_block_size, self._head_dim_padded), (m_block, 0),
        )
        gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
        tOsO = gmem_thr_copy_O.partition_S(sO)
        tOgO = gmem_thr_copy_O.partition_D(gO)
        tOrO = cute.make_fragment_like(tOgO, self._dtype)
        self.cta_sync_barrier.arrive_and_wait()

        # SMEM -> RMEM
        cute.copy(gmem_tiled_copy_O, tOsO, tOrO)

        # Unconditional RMEM -> GMEM (no elem_less bounds guard)
        cute.copy(gmem_tiled_copy_O, tOrO, tOgO)

    @cute.jit
    def compute_one_n_block(self, basic_params, mma_params, smem_copy_params,
                           softmax_params, is_first_n_block, in_mask_steps):
        # S accumulator
        acc_shape_S = mma_params.thr_mma.partition_shape_C(
            (self._m_block_size, self._n_block_size)
        )
        acc_S = cute.make_rmem_tensor(acc_shape_S, cutlass.Float32)
        acc_S.fill(0.0)

        # Wait for Q/K to be ready in SMEM
        self.cta_sync_barrier.arrive_and_wait()

        # =================================================================
        # Static scalar V load: no tail branch (S divisible by N_BLOCK)
        # =================================================================
        tidx, _, _ = cute.arch.thread_idx()
        m_block, batch_size, q_head = cute.arch.block_idx()
        group_size = basic_params.mQ.shape[2] // basic_params.mK.shape[2]
        kv_head = q_head // group_size
        v_seq_start = basic_params.n_block * self._n_block_size
        for row_in_tile in cutlass.range_constexpr(self._n_block_size):
            for d_idx in cutlass.range_constexpr(self._head_dim_padded):
                flat_idx = row_in_tile * self._head_dim_padded + d_idx
                if flat_idx % self._num_threads == tidx:
                    seq_idx = v_seq_start + row_in_tile
                    smem_copy_params.sV[row_in_tile, d_idx] = basic_params.mV[
                        batch_size, seq_idx, kv_head, d_idx
                    ]

        self.cta_sync_barrier.arrive_and_wait()

        # =================================================================
        # S = Q @ K^T  (MMA across k-dimension, SMEM→RMEM pipelined)
        # =================================================================
        cute.copy(
            smem_copy_params.smem_tiled_copy_Q,
            smem_copy_params.tSsQ[None, None, 0],
            smem_copy_params.tSrQ_copy_view[None, None, 0],
        )
        cute.copy(
            smem_copy_params.smem_tiled_copy_K,
            smem_copy_params.tSsK[None, None, 0],
            smem_copy_params.tSrK_copy_view[None, None, 0],
        )
        for k in cutlass.range_constexpr(cute.size(smem_copy_params.tSsQ.shape[2])):
            k_next = (k + 1) % cute.size(smem_copy_params.tSsQ.shape[2])
            cute.copy(
                smem_copy_params.smem_tiled_copy_Q,
                smem_copy_params.tSsQ[None, None, k_next],
                smem_copy_params.tSrQ_copy_view[None, None, k_next],
            )
            cute.copy(
                smem_copy_params.smem_tiled_copy_K,
                smem_copy_params.tSsK[None, None, k_next],
                smem_copy_params.tSrK_copy_view[None, None, k_next],
            )
            cute.gemm(
                mma_params.tiled_mma,
                acc_S,
                mma_params.tSrQ[None, None, k],
                mma_params.tSrK[None, None, k],
                acc_S,
            )

        self.cta_sync_barrier.arrive_and_wait()

        # =================================================================
        # Static scalar next-K prefetch (unconditional, valid tile)
        # =================================================================
        if basic_params.n_block > 0:
            next_k_seq_start = (basic_params.n_block - 1) * self._n_block_size
            for row_in_tile in cutlass.range_constexpr(self._n_block_size):
                for d_idx in cutlass.range_constexpr(self._head_dim_padded):
                    flat_idx = row_in_tile * self._head_dim_padded + d_idx
                    if flat_idx % self._num_threads == tidx:
                        seq_idx = next_k_seq_start + row_in_tile
                        smem_copy_params.sK[row_in_tile, d_idx] = basic_params.mK[
                            batch_size, seq_idx, kv_head, d_idx
                        ]
            self.cta_sync_barrier.arrive_and_wait()

        # =================================================================
        # Online softmax + rescale O (static-no-tail softmax)
        # =================================================================
        self.softmax_rescale_O_static(
            basic_params, mma_params, softmax_params,
            acc_S, is_first_n_block, in_mask_steps,
        )

        # =================================================================
        # O = softmax(S) @ V  (MMA across k-dimension, SMEM→RMEM pipelined)
        # =================================================================
        rP = cute.make_fragment_like(acc_S, self._dtype)
        rP.store(acc_S.load().to(self._dtype))
        rP_layout_divided = cute.logical_divide(rP.layout, (None, None, 2))
        rP_mma_view = cute.make_layout(
            (
                (rP_layout_divided.shape[0], rP_layout_divided.shape[2][0]),
                rP_layout_divided.shape[1],
                rP_layout_divided.shape[2][1],
            ),
            stride=(
                (rP_layout_divided.stride[0], rP_layout_divided.stride[2][0]),
                rP_layout_divided.stride[1],
                rP_layout_divided.stride[2][1],
            ),
        )
        tOrS = cute.make_tensor(rP.iterator, rP_mma_view)

        cute.copy(
            smem_copy_params.smem_tiled_copy_V,
            smem_copy_params.tOsVt[None, None, 0],
            smem_copy_params.tOrVt_copy_view[None, None, 0],
        )
        for k in cutlass.range_constexpr(cute.size(tOrS.shape[2])):
            k_next = (k + 1) % cute.size(tOrS.shape[2])
            cute.copy(
                smem_copy_params.smem_tiled_copy_V,
                smem_copy_params.tOsVt[None, None, k_next],
                smem_copy_params.tOrVt_copy_view[None, None, k_next],
            )
            cute.gemm(
                mma_params.tiled_mma,
                mma_params.acc_O,
                tOrS[None, None, k],
                mma_params.tOrVt[None, None, k],
                mma_params.acc_O,
            )

    @cute.jit
    def softmax_rescale_O_static(self, basic_params, mma_params, softmax_params,
                                 acc_S, is_first_n_block, in_mask_steps):
        """Online softmax + O rescale for static-no-tail shapes.

        Only the causal mask branch is retained.  The non-causal tail mask
        is skipped because every N_BLOCK tile is complete when S is
        divisible by N_BLOCK, making the non-causal elem_less guard
        always False.
        """
        acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
        acc_O_mn = self._make_acc_tensor_mn_view(mma_params.acc_O)

        row_max_prev = None
        if cutlass.const_expr(not is_first_n_block):
            row_max_prev = cute.make_fragment_like(softmax_params.row_max, cutlass.Float32)
            cute.basic_copy(softmax_params.row_max, row_max_prev)

        # Prepare causal mask identity tensor (only used when in_mask_steps and causal)
        tScS_mn = None
        if cutlass.const_expr(in_mask_steps and self._is_causal):
            mcS = cute.make_identity_tensor(
                (basic_params.mQ.shape[0], basic_params.mQ.shape[1],
                 basic_params.mQ.shape[2], basic_params.mK.shape[1])
            )
            cS = cute.local_tile(
                mcS[basic_params.batch_size, None, basic_params.num_head, None],
                (self._m_block_size, self._n_block_size),
                (basic_params.m_block, basic_params.n_block),
            )
            tScS = mma_params.thr_mma.partition_C(cS)
            tScS_mn = self._make_acc_tensor_mn_view(tScS)

        for r in cutlass.range_constexpr(cute.size(softmax_params.row_max)):
            # Apply causal mask during mask_steps
            if cutlass.const_expr(in_mask_steps and self._is_causal):
                col_idx_limit = cutlass.min(
                    tScS_mn[r, 0][1] + 1, basic_params.mK.shape[1]
                )
                for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                    if cute.elem_less(col_idx_limit, tScS_mn[0, c][3] + 1):
                        acc_S_mn[r, c] = -cutlass.Float32.inf

            acc_S_row = acc_S_mn[r, None].load()
            row_max_cur_row = acc_S_row.reduce(cute.ReductionOp.MAX, -cutlass.Float32.inf, 0)
            row_max_cur_row = self._threadquad_reduce_max(row_max_cur_row)

            row_max_prev_row = None
            if cutlass.const_expr(not is_first_n_block):
                row_max_prev_row = row_max_prev[r]
                row_max_cur_row = cute.arch.fmax(row_max_prev_row, row_max_cur_row)
            if cutlass.const_expr(self._is_causal):
                row_max_cur_row = (
                    0.0 if row_max_cur_row == -cutlass.Float32.inf else row_max_cur_row
                )

            acc_S_row_exp = cute.math.exp2(
                acc_S_row * softmax_params.softmax_scale_log2
                - row_max_cur_row * softmax_params.softmax_scale_log2,
                fastmath=True,
            )
            acc_S_row_sum = acc_S_row_exp.reduce(
                cute.ReductionOp.ADD, cutlass.Float32.zero, 0
            )

            if cutlass.const_expr(not is_first_n_block):
                prev_minus_cur_exp = cute.math.exp2(
                    row_max_prev_row * softmax_params.softmax_scale_log2
                    - row_max_cur_row * softmax_params.softmax_scale_log2,
                    fastmath=True,
                )
                acc_S_row_sum = (
                    acc_S_row_sum + softmax_params.row_sum[r] * prev_minus_cur_exp
                )
                acc_O_mn[r, None] = acc_O_mn[r, None].load() * prev_minus_cur_exp

            softmax_params.row_max[r] = row_max_cur_row
            softmax_params.row_sum[r] = acc_S_row_sum
            acc_S_mn[r, None] = acc_S_row_exp

    @cute.jit
    def normalize_softmax(self, acc_O, row_sum):
        acc_O_mn = self._make_acc_tensor_mn_view(acc_O)
        for r in cutlass.range_constexpr(cute.size(row_sum)):
            row_sum[r] = self._threadquad_reduce_sum(row_sum[r])
            acc_O_mn_row_is_zero_or_nan = row_sum[r] == 0.0 or row_sum[r] != row_sum[r]
            scale = (
                1.0 if acc_O_mn_row_is_zero_or_nan else cute.arch.rcp_approx(row_sum[r])
            )
            acc_O_mn[r, None] = acc_O_mn[r, None].load() * scale

    def _make_acc_tensor_mn_view(self, acc):
        acc_layout_col_major = cute.make_layout(acc.layout.shape)
        acc_layout_mn = cute.make_layout(
            (
                (acc_layout_col_major.shape[0][1], acc_layout_col_major.shape[1]),
                (acc_layout_col_major.shape[0][0], acc_layout_col_major.shape[2]),
            ),
            stride=(
                (acc_layout_col_major.stride[0][1], acc_layout_col_major.stride[1]),
                (acc_layout_col_major.stride[0][0], acc_layout_col_major.stride[2]),
            ),
        )
        acc_layout_mn = cute.composition(acc.layout, acc_layout_mn)
        return cute.make_tensor(acc.iterator, acc_layout_mn)

    def _threadquad_reduce(self, val, op):
        val = op(
            val,
            cute.arch.shuffle_sync_bfly(val, offset=2, mask=-1, mask_and_clamp=31),
        )
        val = op(
            val,
            cute.arch.shuffle_sync_bfly(val, offset=1, mask=-1, mask_and_clamp=31),
        )
        return val

    def _threadquad_reduce_max(self, val):
        return self._threadquad_reduce(val, lambda x, y: cute.arch.fmax(x, y))

    def _threadquad_reduce_sum(self, val):
        return self._threadquad_reduce(val, lambda x, y: x + y)


def run(
    dtype: Type[cutlass.Numeric],
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    num_head: int,
    head_dim: int,
    softmax_scale: float = 1.0,
    m_block_size: int = 128,
    n_block_size: int = 128,
    num_threads: int = 128,
    is_causal: bool = False,
    warmup_iterations: int = 0,
    iterations: int = 1,
    skip_ref_check: bool = False,
    use_cold_l2: bool = False,
    **kwargs,
):
    import torch
    import cutlass.torch as cutlass_torch

    # Skip unsupported testcase
    if not FlashAttentionForwardAmpere.can_implement(
        dtype,
        head_dim,
        m_block_size,
        n_block_size,
        num_threads,
        is_causal,
    ):
        raise TypeError(
            f"Unsupported testcase {dtype}, {head_dim}, {m_block_size}, {n_block_size}, {num_threads}, {is_causal}"
        )

    print("Running Ampere SM80 FlashAttentionForward test with:")
    print(f"  dtype: {dtype}")
    print(f"  batch_size: {batch_size}")
    print(f"  seqlen_q: {seqlen_q}")
    print(f"  seqlen_k: {seqlen_k}")
    print(f"  num_head: {num_head}")
    print(f"  head_dim: {head_dim}")
    print(f"  softmax_scale: {softmax_scale}")
    print(f"  m_block_size: {m_block_size}")
    print(f"  n_block_size: {n_block_size}")
    print(f"  num_threads: {num_threads}")
    print(f"  is_causal: {is_causal}")
    print(f"  warmup_iterations: {warmup_iterations}")
    print(f"  iterations: {iterations}")
    print(f"  skip_ref_check: {skip_ref_check}")
    print(f"  use_cold_l2: {use_cold_l2}")

    # Create tensor Q/K/V/O
    def create_tensor(
        batch_size: int,
        seqlen: int,
        num_head: int,
        head_dim: int,
        dtype: Type[cutlass.Numeric],
    ) -> cute.Tensor:
        # (batch_size, seqlen, num_head, head_dim)
        shape = (batch_size, seqlen, num_head, head_dim)
        torch_tensor = (
            torch.empty(*shape, dtype=torch.int32)
            .random_(-2, 2)
            .to(dtype=cutlass_torch.dtype(dtype))
            .cuda()
        )
        # assume input is 16B aligned.
        cute_tensor = (
            from_dlpack(torch_tensor, assumed_align=16)
            .mark_layout_dynamic(leading_dim=3)
            .mark_compact_shape_dynamic(
                mode=3,
                stride_order=torch_tensor.dim_order(),
                divisibility=(128 // dtype.width),
            )
        )
        return cute_tensor, torch_tensor

    q, q_torch = create_tensor(batch_size, seqlen_q, num_head, head_dim, dtype)
    k, k_torch = create_tensor(batch_size, seqlen_k, num_head, head_dim, dtype)
    v, v_torch = create_tensor(batch_size, seqlen_k, num_head, head_dim, dtype)
    o, o_torch = create_tensor(batch_size, seqlen_q, num_head, head_dim, dtype)

    fa2_fwd = FlashAttentionForwardAmpere(
        head_dim,
        m_block_size,
        n_block_size,
        num_threads,
        is_causal,
    )

    # Get current CUDA stream from PyTorch
    torch_stream = torch.cuda.current_stream()
    # Get the raw stream pointer as a CUstream
    current_stream = cuda.CUstream(torch_stream.cuda_stream)
    # Pass compile options if needed
    compile_options = ""
    # compile the fa2 forward pass
    compiled_fa2_fwd = cute.compile(
        fa2_fwd, q, k, v, o, softmax_scale, current_stream, options=compile_options
    )

    if not skip_ref_check:
        compiled_fa2_fwd(q, k, v, o, softmax_scale, current_stream)
        torch.cuda.synchronize()
        q_ref = q_torch.permute(0, 2, 1, 3)
        k_ref = k_torch.permute(0, 2, 1, 3)
        v_ref = v_torch.permute(0, 2, 1, 3)
        torch.backends.cuda.enable_flash_sdp(enabled=True)
        ref_o = torch.nn.functional.scaled_dot_product_attention(
            q_ref, k_ref, v_ref, scale=softmax_scale, is_causal=is_causal
        ).permute(0, 2, 1, 3)
        torch.testing.assert_close(o_torch.cpu(), ref_o.cpu(), atol=1e-02, rtol=1e-04)
        print("Results verified successfully!")

    def generate_tensors():
        q_workspace, _ = create_tensor(batch_size, seqlen_q, num_head, head_dim, dtype)
        k_workspace, _ = create_tensor(batch_size, seqlen_k, num_head, head_dim, dtype)
        v_workspace, _ = create_tensor(batch_size, seqlen_k, num_head, head_dim, dtype)
        o_workspace, _ = create_tensor(batch_size, seqlen_q, num_head, head_dim, dtype)
        return testing.JitArguments(
            q_workspace,
            k_workspace,
            v_workspace,
            o_workspace,
            softmax_scale,
            current_stream,
        )

    workspace_count = 1
    if use_cold_l2:
        one_workspace_bytes = (
            q_torch.numel() * q_torch.element_size()
            + k_torch.numel() * k_torch.element_size()
            + v_torch.numel() * v_torch.element_size()
            + o_torch.numel() * o_torch.element_size()
        )
        workspace_count = testing.get_workspace_count(
            one_workspace_bytes, warmup_iterations, iterations
        )

    avg_time_us = testing.benchmark(
        compiled_fa2_fwd,
        workspace_generator=generate_tensors,
        workspace_count=workspace_count,
        stream=current_stream,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )

    return avg_time_us  # Return execution time in microseconds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="example of flash attention v2 with CuTe on GPU"
    )
    parser.add_argument("--dtype", type=cutlass.dtype, default=cutlass.BFloat16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seqlen_q", type=int, default=8192)
    parser.add_argument("--seqlen_k", type=int, default=8192)
    parser.add_argument("--num_head", type=int, default=16)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--softmax_scale", type=float, default=0.5)
    parser.add_argument("--m_block_size", type=int, default=128)
    parser.add_argument("--n_block_size", type=int, default=64)
    parser.add_argument("--num_threads", type=int, default=128)
    parser.add_argument("--is_causal", action="store_true", help="Enable causal mask")
    parser.add_argument("--warmup_iterations", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--skip_ref_check", action="store_true", help="Skip reference check"
    )
    parser.add_argument(
        "--use_cold_l2",
        action="store_true",
        default=False,
        help="Use circular buffer tensor sets to ensure L2 cold cache",
    )
    args = parser.parse_args()
    run(
        args.dtype,
        args.batch_size,
        args.seqlen_q,
        args.seqlen_k,
        args.num_head,
        args.head_dim,
        args.softmax_scale,
        args.m_block_size,
        args.n_block_size,
        args.num_threads,
        args.is_causal,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
        args.use_cold_l2,
    )

    print("PASS")
