# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Bandwidth-oriented Qwen3 Decode attention for H20.

This is an independent rewrite targeting the measured ~4.5 TB/s HBM ceiling.
For the fixed workload, the essential K+V+Q read is 536.9 MB and the pure-HBM
roofline is ~119 us.

The first scalar-SIMT experiment reached high occupancy but only ~9% DRAM
throughput: a warp reduction per token and scalar BF16 loads made it compute
and instruction-latency bound.  The production design therefore uses one
SM80-compatible ``mma.sync.m16n8k16`` warp per (KV head, split):

* 128-bit ``cp.async`` K/V transfers and N=32/64 shared-memory tiles;
* M=16 warp-level BF16 Tensor-Core MMA, with the eight GQA rows packed into the
  valid half of the tile;
* 64..256 split batches for memory-level parallelism without WGMMA's 240
  registers/thread footprint;
* BF16 normalized partials plus FP32 max/sum statistics, followed by one
  two-level combine kernel.

Only workload supported: Q=[1,64,1,128], K/V=[1,8,131072,128], contiguous BF16
on the current CUDA device (any available H20; not hard-coded to cuda:0).
"""

import argparse
import math
import statistics
import sys
import threading
from types import SimpleNamespace
from typing import Callable, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.arch as arch
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import from_dlpack


QWEN_BATCH = 1
QWEN_QUERY_HEADS = 64
QWEN_KV_HEADS = 8
QWEN_HEAD_DIM = 128
QWEN_CONTEXT = 128 * 1024
HEAD_RATIO = QWEN_QUERY_HEADS // QWEN_KV_HEADS
LOG2_E = 1.4426950408889634074

# The split count divides 131072 exactly; each split length is also a multiple
# of every N tile below, so the hot mainloop has no K/V residue path.
CONFIGS = {
    "hmma-s64-n64": {"splits": 64, "block_n": 64},
    "hmma-s128-n64": {"splits": 128, "block_n": 64},
    "hmma-s256-n32": {"splits": 256, "block_n": 32},
    "hmma-s256-n64": {"splits": 256, "block_n": 64},
}
# Winner of three-round full-kernel interleaved tuning on H20 GPU 4.
AUTO_CONFIG = "hmma-s256-n32"

_COMPILE_LOCK = threading.Lock()
_FORWARD_KERNEL_CACHE = {}
_LAUNCH_PLAN_CACHE = {}
_LAUNCH_PLAN_LOCK = threading.Lock()
_WORKSPACE_CACHE = {}
_WORKSPACE_LOCK = threading.Lock()
_WORKSPACE_CACHE_LIMIT = 32
_LAUNCH_PLAN_CACHE_LIMIT = 32
_WORKSPACE_LAYOUT_VERSION = 5
_PARTIAL_KERNEL_VERSION = 3
_COMBINE_KERNEL_VERSION = 3

# Four warps cover all 128 output dimensions; each thread also reduces two
# stats entries in the 256-split winner.
COMBINE_THREADS = 128


class Decode128KWarpMmaPartial:
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
        mStats: cute.Tensor,
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
        # Thread layouts for copies
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self._dtype.width
        # atom_async_copy: async copy atom for QKV load
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self._dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        # atom_universal_copy: universal copy atom for O store
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self._dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        # tQKV_layout: thread layout for QKV load
        tQKV_shape_dim_1 = sQ_layout_atom.outer.shape[1] // async_copy_elems
        tQKV_layout = cute.make_layout(
            (self._num_threads // tQKV_shape_dim_1, tQKV_shape_dim_1),
            stride=(tQKV_shape_dim_1, 1),
        )
        # tO_layout: thread layout for O store
        tO_layout = tQKV_layout

        # Value layouts for copies
        vQKV_layout = cute.make_layout((1, async_copy_elems))
        vO_layout = vQKV_layout

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
            cute.ceil_div(mQ.shape[0], self._m_block_size),
            cute.size(mK.shape[0]),
            cute.size(mQ.shape[1]),
        )
        LOG2_E = 1.4426950408889634074
        softmax_scale_log2 = softmax_scale * LOG2_E
        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            mStats,
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
        mStats: cute.Tensor,
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
        m_block, batch_size, num_head = cute.arch.block_idx()

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
            mQ[None, num_head, None],
            (self._m_block_size, self._head_dim_padded),
            (m_block, 0),
        )
        # (n_block_size, head_dim, n_block)
        gK = cute.local_tile(
            mK[batch_size, None, num_head, None],
            (self._n_block_size, self._head_dim_padded),
            (None, 0),
        )
        # (n_block_size, head_dim, n_block)
        gV = cute.local_tile(
            mV[batch_size, None, num_head, None],
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
            mcQ[None, num_head, None],
            (self._m_block_size, self._head_dim_padded),
            (m_block, 0),
        )
        cKV = cute.local_tile(
            mcKV[batch_size, None, num_head, None],
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
                    tQcQ[(0, rest_v), 0, rest_k][2], mQ.layout.shape[2]
                )
        for rest_v in cutlass.range_constexpr(tKVpKV.shape[0]):
            for rest_k in cutlass.range_constexpr(tKVpKV.shape[2]):
                tKVpKV[rest_v, 0, rest_k] = cute.elem_less(
                    tKVcKV[(0, rest_v), 0, rest_k][3], mK.layout.shape[3]
                )
        # ///////////////////////////////////////////////////////////////////////////////
        # Prefetch Prologue
        # ///////////////////////////////////////////////////////////////////////////////
        # Start async loads of the last mn-tile, where we take care of the mn residue
        for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
            if cute.elem_less(tQcQ[0, m, 0][0], mQ.layout.shape[0]):
                cute.copy(
                    gmem_tiled_copy_QKV,
                    tQgQ[None, m, None],
                    tQsQ[None, m, None],
                    pred=tQpQ[None, m, None],
                )
            else:
                # Clear the smem tiles to account for predicated off loads
                tQsQ[None, m, None].fill(0)
        for n in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
            if cute.elem_less(tKVcKV[0, n, 0][1], mK.layout.shape[1]):
                cute.copy(
                    gmem_tiled_copy_QKV,
                    tKgK[None, n, None, n_block],
                    tKsK[None, n, None],
                    pred=tKVpKV[None, n, None],
                )
            else:
                # Clear the smem tiles to account for predicated off loads
                tKsK[None, n, None].fill(0)

        cute.arch.cp_async_commit_group()
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
            batch_size=batch_size,
            num_head=num_head,
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
        # Store stats with explicit logical coordinates. Partitioning an
        # eight-row global tensor with an M16 C layout can form out-of-range
        # addresses even when the eventual store is predicated; direct scalar
        # addressing keeps the generated write strictly inside the workspace.
        cLM = cute.make_identity_tensor((self._m_block_size, 1))
        cLM_thr = tiled_mma.get_slice(tidx).partition_C(cLM)
        for row_idx in cutlass.range_constexpr(cute.size(row_max)):
            row = cLM_thr[(0, row_idx), 0, 0][0]
            if cute.elem_less(row, HEAD_RATIO):
                mStats[row, batch_size, num_head, 0] = (
                    row_max[row_idx] * softmax_scale_log2
                )
                mStats[row, batch_size, num_head, 1] = row_sum[row_idx]


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
            mO[batch_size, None, num_head, None],
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
            mcO[batch_size, None, num_head, None],
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
            if cute.elem_less(tOcO[0, rest_m, 0][1], mQ.layout.shape[0]):
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
        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()
        # load smem tile V for O, special process for the first tile to avoid loading nan.
        # The `if` here is a constexpr, won't be generated in the IR.
        if is_first_n_block:
            for n in cutlass.range_constexpr(cute.size(gmem_copy_params.tVsV.shape[1])):
                if cute.elem_less(
                    gmem_copy_params.tKVcKV[0, n, 0][1],
                    basic_params.mK.layout.shape[1],
                ):
                    cute.copy(
                        gmem_copy_params.gmem_tiled_copy_QKV,
                        gmem_copy_params.tVgV[None, n, None, basic_params.n_block],
                        gmem_copy_params.tVsV[None, n, None],
                        pred=gmem_copy_params.tKVpKV[None, n, None],
                    )
                else:
                    gmem_copy_params.tVsV[None, n, None].fill(0.0)
        else:
            cute.copy(
                gmem_copy_params.gmem_tiled_copy_QKV,
                gmem_copy_params.tVgV[None, None, None, basic_params.n_block],
                gmem_copy_params.tVsV,
                pred=gmem_copy_params.tKVpKV,
            )

        cute.arch.cp_async_commit_group()
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
        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()

        if basic_params.n_block > 0:
            cute.copy(
                gmem_copy_params.gmem_tiled_copy_QKV,
                gmem_copy_params.tKgK[None, None, None, basic_params.n_block - 1],
                gmem_copy_params.tKsK,
                pred=gmem_copy_params.tKVpKV,
            )
            cute.arch.cp_async_commit_group()
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
                    1,
                    basic_params.mQ.shape[0],
                    basic_params.mQ.shape[1],
                    basic_params.mK.shape[1],
                )
            )
            cS = cute.local_tile(
                mcS[0, None, basic_params.num_head, None],
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



class Decode128KWarpMmaCombine:
    """Combine normalized BF16 split outputs using ratio-major FP32 stats.

    Stage 1 reduces [m_j, s_j] to M and Z=sum(exp2(m_j-M)*s_j).  Since the
    partial kernel stores O_j=R_j/s_j, stage 2 computes
    sum(exp2(m_j-M)*s_j*O_j)/Z.
    """

    def __init__(self, num_splits):
        if not 1 <= num_splits <= 288:
            raise ValueError("num_splits must be in [1, 288]")
        self.num_splits = num_splits
        self.threads_per_cta = COMBINE_THREADS
        self.num_warps = COMBINE_THREADS // 32

    @cute.jit
    def __call__(
        self,
        o_partial: cute.Tensor,  # [splits, HEAD_RATIO, KV_HEADS, D] bf16
        stats: cute.Tensor,      # [HEAD_RATIO, splits, KV_HEADS, 2] fp32
        output: cute.Tensor,     # [QUERY_HEADS, D]                  bf16
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(o_partial.element_type != cutlass.BFloat16):
            raise TypeError("normalized partial O must be BFloat16")
        if cutlass.const_expr(output.element_type != cutlass.BFloat16):
            raise TypeError("output must be BFloat16")

        @cute.struct
        class SharedStorage:
            red: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.num_warps], 128
            ]
            mm: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.num_warps], 128
            ]

        self.shared_storage = SharedStorage
        self.kernel(o_partial, stats, output).launch(
            grid=(HEAD_RATIO, QWEN_KV_HEADS, 1),
            block=(self.threads_per_cta, 1, 1),
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        o_partial: cute.Tensor,
        stats: cute.Tensor,
        output: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        ratio_idx, kv_head, _ = cute.arch.block_idx()
        warp_id = tid // 32
        lane = tid % 32

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        red_smem = storage.red.get_tensor(cute.make_layout(self.num_warps))
        mm_smem = storage.mm.get_tensor(cute.make_layout(self.num_warps))

        # ---- Stage 1a: per-thread local max over its strided split subset ----
        local_max = -cutlass.Float32.inf
        j = tid
        while j < self.num_splits:
            local_max = cute.arch.fmax(local_max, stats[ratio_idx, j, kv_head, 0])
            j += self.threads_per_cta
        # warp reduce, then cross-warp via shared
        warp_max = cute.arch.warp_reduction_max(local_max, threads_in_group=32)
        if lane == 0:
            mm_smem[warp_id] = warp_max
        cute.arch.sync_threads()
        global_max = -cutlass.Float32.inf
        for w in cutlass.range_constexpr(self.num_warps):
            global_max = cute.arch.fmax(global_max, mm_smem[w])
        finite_max = global_max
        if global_max == -cutlass.Float32.inf:
            finite_max = 0.0

        # ---- Stage 1b: denominator Z = sum_j alpha_j * s_j ----
        local_den = cutlass.Float32(0.0)
        j = tid
        while j < self.num_splits:
            m_j = stats[ratio_idx, j, kv_head, 0]
            s_j = stats[ratio_idx, j, kv_head, 1]
            alpha = cute.math.exp2(m_j - finite_max, fastmath=True)
            local_den += alpha * s_j
            j += self.threads_per_cta
        warp_den = cute.arch.warp_reduction_sum(local_den, threads_in_group=32)
        if lane == 0:
            red_smem[warp_id] = warp_den
        cute.arch.sync_threads()
        denominator = cutlass.Float32(0.0)
        for w in cutlass.range_constexpr(self.num_warps):
            denominator += red_smem[w]
        inv_den = cutlass.Float32(0.0)
        if denominator != 0.0 and denominator == denominator:
            inv_den = 1.0 / denominator

        # ---- Stage 2: weighted accumulation of raw partials over D ----
        # Threads 0..D-1 each own one head-dim element.
        if tid < QWEN_HEAD_DIM:
            d = tid
            acc = cutlass.Float32(0.0)
            for j in cutlass.range_constexpr(self.num_splits):
                m_j = stats[ratio_idx, j, kv_head, 0]
                s_j = stats[ratio_idx, j, kv_head, 1]
                alpha = cute.math.exp2(m_j - finite_max, fastmath=True)
                acc += (
                    alpha
                    * s_j
                    * o_partial[j, ratio_idx, kv_head, d].to(cutlass.Float32)
                )
            head_idx = kv_head * HEAD_RATIO + ratio_idx
            output[head_idx, d] = (acc * inv_den).to(cutlass.BFloat16)


class Decode128KWarpMmaForward:
    """One compiled host call that submits the two production CUDA kernels."""

    def __init__(self, num_splits, block_n):
        self.partial = Decode128KWarpMmaPartial(
            QWEN_HEAD_DIM,
            m_block_size=16,
            n_block_size=block_n,
            num_threads=32,
            is_causal=False,
        )
        self.combine = Decode128KWarpMmaCombine(num_splits)

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        partial: cute.Tensor,
        stats: cute.Tensor,
        output: cute.Tensor,
        softmax_scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        self.partial(q, k, v, partial, stats, softmax_scale, stream)
        self.combine(partial, stats, output, stream)


def _resolve_config(config):
    if config == "auto":
        config = AUTO_CONFIG
    try:
        values = CONFIGS[config]
    except KeyError as exc:
        choices = ", ".join(("auto", *CONFIGS))
        raise ValueError(
            f"unknown config {config!r}; expected one of {choices}"
        ) from exc
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
        raise TypeError(
            "sm_scale must be a positive finite real number or None"
        ) from exc
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
            raise ValueError(
                f"{name} must have shape {expected[name]}, got {tuple(tensor.shape)}"
            )
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
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "CUDA Graph capture is not supported by this Decode-only API; "
            "prepare fixed workspace/output separately"
        )
    properties = torch.cuda.get_device_properties(q.device)
    capability = torch.cuda.get_device_capability(q.device)
    if capability != (9, 0):
        raise RuntimeError(
            f"the fixed kernel requires SM90 (H20), got sm_{capability[0]}{capability[1]}"
        )
    if "H20" not in properties.name.upper():
        raise RuntimeError(
            f"the fixed kernel requires NVIDIA H20, got {properties.name}"
        )
    if properties.multi_processor_count != 78:
        raise RuntimeError(
            "the fixed kernel targets the 78-SM H20 SKU, got "
            f"{properties.multi_processor_count} SMs"
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
                    (num_splits, HEAD_RATIO, QWEN_KV_HEADS, QWEN_HEAD_DIM),
                    dtype=torch.bfloat16,
                    device=device,
                ),
                "stats": torch.empty(
                    (HEAD_RATIO, num_splits, QWEN_KV_HEADS, 2),
                    dtype=torch.float32,
                    device=device,
                ),
            }
            _WORKSPACE_CACHE[key] = workspace
    return workspace


def _as_cute_tensor(tensor, element_type, leading_dim, *, compact=False):
    result = from_dlpack(tensor, assumed_align=16)
    result.element_type = element_type
    result = result.mark_layout_dynamic(leading_dim=leading_dim)
    if compact:
        result = result.mark_compact_shape_dynamic(
            mode=leading_dim,
            divisibility=128 // element_type.width,
        )
    return result


def _get_launch_plan(q, k, v, workspace, config_name, values):
    num_splits = values["splits"]
    split_len = QWEN_CONTEXT // num_splits
    key = (
        q.device.index,
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        workspace["o"].data_ptr(),
        workspace["stats"].data_ptr(),
        config_name,
        _WORKSPACE_LAYOUT_VERSION,
    )
    with _LAUNCH_PLAN_LOCK:
        plan = _LAUNCH_PLAN_CACHE.get(key)
        if plan is not None:
            return plan

        q_view = q.view(QWEN_KV_HEADS, HEAD_RATIO, QWEN_HEAD_DIM).permute(1, 0, 2)
        k_view = (
            k.view(QWEN_KV_HEADS, QWEN_CONTEXT, QWEN_HEAD_DIM)
            .permute(1, 0, 2)
            .unflatten(0, (num_splits, split_len))
        )
        v_view = (
            v.view(QWEN_KV_HEADS, QWEN_CONTEXT, QWEN_HEAD_DIM)
            .permute(1, 0, 2)
            .unflatten(0, (num_splits, split_len))
        )
        plan = {
            "q": _as_cute_tensor(q_view, cutlass.BFloat16, 2, compact=True),
            "k": _as_cute_tensor(k_view, cutlass.BFloat16, 3, compact=True),
            "v": _as_cute_tensor(v_view, cutlass.BFloat16, 3, compact=True),
            "partial": _as_cute_tensor(
                workspace["o"], cutlass.BFloat16, 3, compact=True
            ),
            "stats": _as_cute_tensor(workspace["stats"], cutlass.Float32, 3),
            # Keep all DLPack owners alive while their wrappers are cached.
            "owners": (q, k, v, q_view, k_view, v_view, workspace),
        }
        if len(_LAUNCH_PLAN_CACHE) >= _LAUNCH_PLAN_CACHE_LIMIT:
            del _LAUNCH_PLAN_CACHE[next(iter(_LAUNCH_PLAN_CACHE))]
        _LAUNCH_PLAN_CACHE[key] = plan
        return plan


def _launch_forward(q, k, v, workspace, sm_scale, config_name, values):
    import torch

    plan = _get_launch_plan(q, k, v, workspace, config_name, values)
    output = torch.empty_like(q)
    output_tensor = _as_cute_tensor(
        output.view(QWEN_QUERY_HEADS, QWEN_HEAD_DIM),
        cutlass.BFloat16,
        1,
        compact=True,
    )
    torch_stream = torch.cuda.current_stream(q.device)
    stream = cuda.CUstream(torch_stream.cuda_stream)
    capability = torch.cuda.get_device_capability(q.device)
    key = (
        _PARTIAL_KERNEL_VERSION,
        _COMBINE_KERNEL_VERSION,
        q.device.index,
        capability,
        config_name,
        values["splits"],
        values["block_n"],
        "two-kernel-host-fused",
    )
    compiled = _FORWARD_KERNEL_CACHE.get(key)
    if compiled is None:
        with _COMPILE_LOCK:
            compiled = _FORWARD_KERNEL_CACHE.get(key)
            if compiled is None:
                operation = Decode128KWarpMmaForward(
                    values["splits"], values["block_n"]
                )
                compiled = cute.compile(
                    operation,
                    plan["q"],
                    plan["k"],
                    plan["v"],
                    plan["partial"],
                    plan["stats"],
                    output_tensor,
                    sm_scale,
                    stream,
                )
                _FORWARD_KERNEL_CACHE[key] = compiled
    compiled(
        plan["q"],
        plan["k"],
        plan["v"],
        plan["partial"],
        plan["stats"],
        output_tensor,
        sm_scale,
        stream,
    )
    for tensor in (q, k, v, workspace["o"], workspace["stats"], output):
        tensor.record_stream(torch_stream)
    return output


def _run_decode(q, k, v, sm_scale, config_name, values, return_workspace=False):
    import torch

    torch_stream = torch.cuda.current_stream(q.device)
    workspace = _get_workspace(
        q.device, torch_stream.cuda_stream, config_name, values["splits"]
    )
    output = _launch_forward(q, k, v, workspace, sm_scale, config_name, values)
    if return_workspace:
        return output, workspace
    return output


def qwen3_decode_attention(q, k, v, *, causal=True, sm_scale=None, config="auto"):
    """Run the fixed H20 128K warp-HMMA Decode kernel.

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


def _run_smoke(args):
    import torch

    q, k, v = _make_inputs(args.seed)
    output = qwen3_decode_attention(q, k, v, config=args.config)
    torch.cuda.synchronize()
    print(
        f"compile smoke passed: output={tuple(output.shape)} dtype={output.dtype}"
    )


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


def _build_benchmark_targets(spec):
    targets = {}
    for name in spec:
        name = name.strip()
        if not name:
            continue
        if name in CONFIGS or name == "auto":
            cfg = name
            targets[name] = lambda q, k, v, s, c=cfg: qwen3_decode_attention(
                q, k, v, causal=True, sm_scale=s, config=c
            )
        elif name == "decode-optimized":
            import implement_attention_decode_optimized as base

            targets[name] = lambda q, k, v, s: base.qwen3_decode_attention(
                q, k, v, causal=True, sm_scale=s, config="auto"
            )
        elif name == "fa3":
            targets[name] = _make_fa3_target()
        else:
            raise ValueError(f"unknown benchmark target {name!r}")
    return targets


def _make_fa3_target():
    hopper_path = "/dockerdata/linqihao/flash-attention/hopper"
    if hopper_path not in sys.path:
        sys.path.insert(0, hopper_path)
    from flash_attn_interface import flash_attn_func

    cached = {}

    def run(q, k, v, s):
        key = (q.data_ptr(), k.data_ptr(), v.data_ptr())
        packed = cached.get("packed")
        if packed is None or cached.get("key") != key:
            packed = (
                q.permute(0, 2, 1, 3).contiguous(),
                k.permute(0, 2, 1, 3).contiguous(),
                v.permute(0, 2, 1, 3).contiguous(),
            )
            cached["key"] = key
            cached["packed"] = packed
        qh, kh, vh = packed
        out = flash_attn_func(
            qh,
            kh,
            vh,
            causal=False,
            softmax_scale=s,
            num_splits=0,
            pack_gqa=None,
        )
        if isinstance(out, tuple):
            out = out[0]
        return out.permute(0, 2, 1, 3).contiguous()

    return run


def _run_benchmark(args):
    import torch

    q, k, v = _make_inputs(args.seed)
    scale = args.scales[0]
    spec = args.benchmark_configs.split(",") if args.benchmark_configs else [
        "auto"
    ]
    targets = _build_benchmark_targets(spec)
    reference = _grouped_reference(q, k, v, scale)
    auto_out = qwen3_decode_attention(
        q, k, v, causal=True, sm_scale=scale, config="auto"
    )
    torch.cuda.synchronize()

    round_medians = {name: [] for name in targets}
    for r in range(args.rounds):
        print(f"benchmark round={r + 1} samples={args.iterations}")
        for name, fn in targets.items():
            median, _, out = _time_cuda(
                lambda fn=fn: fn(q, k, v, scale), args.warmup, args.iterations
            )
            p95 = None
            round_medians[name].append((median, out))
            print(f"  {name}: median={median:.3f} us")
    print("median-of-medians")
    base_med = statistics.median(
        [m for m, _ in round_medians[next(iter(targets))]]
    )
    for name, samples in round_medians.items():
        med = statistics.median([m for m, _ in samples])
        speed = base_med / med if med else 0.0
        line = f"  {name}: {med:.3f} us speedup_vs_first={speed:.4f}x"
        if name in CONFIGS or name == "auto":
            line += f" effective_kv={_effective_kv_gbps(med):.1f} GB/s"
        print(line)
    for name, samples in round_medians.items():
        out = samples[-1][1]
        max_abs, mean_abs = _error_metrics(out, reference)
        print(f"  compare {name}: max_abs={max_abs:.6g} mean_abs={mean_abs:.6g}")


def _run_profile(args):
    import torch

    q, k, v = _make_inputs(args.seed)
    scale = args.scales[0]
    config_name, values = _resolve_config(args.config)
    for _ in range(args.warmup):
        qwen3_decode_attention(q, k, v, sm_scale=scale, config=config_name)
    torch.cuda.synchronize()
    activities = [torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities) as prof:
        qwen3_decode_attention(q, k, v, sm_scale=scale, config=config_name)
        torch.cuda.synchronize()
    cuda_events = [
        e
        for e in prof.events()
        if e.device_type == torch.autograd.DeviceType.CUDA and e.self_device_time_total > 0
    ]
    print(f"profile CUDA kernel events={len(cuda_events)}")
    for e in cuda_events:
        print(f"  {e.key[:70]}: {e.self_device_time_total:.2f} us")
    if len(cuda_events) != 2:
        raise AssertionError(
            f"expected exactly 2 CUDA kernels, observed {len(cuda_events)}"
        )


def _build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["smoke", "correctness", "benchmark", "profile"],
        default="smoke",
    )
    parser.add_argument("--config", default="auto")
    parser.add_argument("--benchmark-configs", default="auto,decode-optimized,fa3")
    parser.add_argument("--scales", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=1)
    return parser


def _parse_scales(text):
    if text is None:
        return [1.0 / math.sqrt(QWEN_HEAD_DIM), 0.125]
    return [float(x) for x in text.split(",")]


def main():
    args = _build_parser().parse_args()
    args.scales = _parse_scales(args.scales)
    if args.mode == "smoke":
        _run_smoke(args)
    elif args.mode == "correctness":
        _run_correctness(args)
    elif args.mode == "benchmark":
        _run_benchmark(args)
    elif args.mode == "profile":
        _run_profile(args)


if __name__ == "__main__":
    main()
