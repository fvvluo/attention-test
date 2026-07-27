"""Fixed-128K Qwen3 Decode attention for H20 in NVIDIA CuTe DSL.

The official benchmark counts 536,903,680 logical Q/K/V/O bytes, so the
4.5-TB/s assignment target is 119.31 us.  This independent Decode rewrite uses
full-utilization transposed WGMMA and persistent TMA workers; it is derived from
the user-provided ``fvvluo/attention-test`` ``wangzicheng`` reference and adds
strict fixed-shape validation, stream/device isolation, bounded caches, and
production verification entry points.

Design:
  * Split-KV (Flash-Decoding): grid = (num_splits, kv_heads, batch).
  * GQA group-packing: one CTA handles the GROUP=8 q_heads that share one
    kv_head, so every KV byte is read from HBM exactly once.
  * Transposed WGMMA dataflow so the tiny per-group M=8 never lands on the
    tensor-core M dimension:
      GEMM1: S^T[BLOCK_N, 8] = K_tile[BLOCK_N, D] @ Q^T[D, 8]   (M = kv rows)
      GEMM2: O^T[D, 8]        = V^T[D, BLOCK_N] @ P^T[BLOCK_N, 8] (M = head_dim)
    Both run at full WGMMA M=128 efficiency (compute ~29us << memory ~128us).
    Softmax reductions become per-column (per q_head) ops on the
    [kv, head] fragments; P^T round-trips through a tiny smem tile.
  * TMA producer/consumer warp-specialization with a multi-stage smem ring.
  * Partial kernel writes O_partial (bf16, normalized) + LSE (log2 domain);
    a small combine kernel reduces across splits.
"""

import argparse
import math
import statistics
import sys
import threading
from dataclasses import dataclass
from typing import Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Boolean, if_generate
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
from cutlass.cute.runtime import from_dlpack

import os as _os

D_TYPE = cutlass.BFloat16
ACC_TYPE = cutlass.Float32

BATCH = 1
Q_HEADS = 64
KV_HEADS = 8
Q_LEN = 1
KV_LEN = 128 * 1024
GROUP_M = Q_HEADS // KV_HEADS
HEAD_DIM = 128
NUM_THREADS = 256  # one producer warpgroup + one consumer warpgroup
LOG2_E = 1.4426950408889634074

CONFIGS = {
    "wgmma-balanced-n256-st1-w78": {
        "num_splits": 10,
        "block_n": 256,
        "num_stages": 1,
        "num_workers": 78,
        "balanced_heads": True,
    },
    "wgmma-s8-n256-st1-w64": {
        "num_splits": 8,
        "block_n": 256,
        "num_stages": 1,
        "num_workers": 64,
    },
    "wgmma-s9-n256-st1-w72": {
        "num_splits": 9,
        "block_n": 256,
        "num_stages": 1,
        "num_workers": 72,
    },
    "wgmma-s9-n256-k2v1-w72": {
        "num_splits": 9,
        "block_n": 256,
        "num_stages": 1,
        "num_k_stages": 2,
        "num_v_stages": 1,
        "num_workers": 72,
    },
    "wgmma-s9-n256-k1v2-w72": {
        "num_splits": 9,
        "block_n": 256,
        "num_stages": 1,
        "num_k_stages": 1,
        "num_v_stages": 2,
        "num_workers": 72,
    },
    "wgmma-s9-n256-k2v1-w72-t160": {
        "num_splits": 9,
        "block_n": 256,
        "num_stages": 1,
        "num_k_stages": 2,
        "num_v_stages": 1,
        "num_workers": 72,
        "num_threads": 160,
        "compact_roles": True,
    },
    "wgmma-s9-n256-st1-w72-t160": {
        "num_splits": 9,
        "block_n": 256,
        "num_stages": 1,
        "num_workers": 72,
        "num_threads": 160,
        "compact_roles": True,
    },
    "wgmma-s18-n256-st1-w72": {
        "num_splits": 18,
        "block_n": 256,
        "num_stages": 1,
        "num_workers": 72,
    },
    "wgmma-s19-n256-st1-w76": {
        "num_splits": 19,
        "block_n": 256,
        "num_stages": 1,
        "num_workers": 76,
    },
    "wgmma-s19-n256-st1-w78": {
        "num_splits": 19,
        "block_n": 256,
        "num_stages": 1,
        "num_workers": 78,
    },
    "wgmma-s38-n256-st1-w76": {
        "num_splits": 38,
        "block_n": 256,
        "num_stages": 1,
        "num_workers": 76,
    },
    "wgmma-s39-n256-st1-w78": {
        "num_splits": 39,
        "block_n": 256,
        "num_stages": 1,
        "num_workers": 78,
    },
    "wgmma-s8-n128-st2-w64": {
        "num_splits": 8,
        "block_n": 128,
        "num_stages": 2,
        "num_workers": 64,
    },
    "wgmma-s9-n128-st2-w72": {
        "num_splits": 9,
        "block_n": 128,
        "num_stages": 2,
        "num_workers": 72,
    },
    "wgmma-s10-n256-skew-w78-t160": {
        "num_splits": 10,
        "block_n": 256,
        "num_stages": 1,
        "num_workers": 78,
        "num_threads": 160,
        "compact_roles": True,
        "skew_tail": True,
    },
    "wgmma-s10-n128-st2-w78": {
        "num_splits": 10,
        "block_n": 128,
        "num_stages": 2,
        "num_workers": 78,
    },
    "wgmma-s13-n128-st2-w78": {
        "num_splits": 13,
        "block_n": 128,
        "num_stages": 2,
        "num_workers": 78,
    },
    "wgmma-s19-n128-st2-w78": {
        "num_splits": 19,
        "block_n": 128,
        "num_stages": 2,
        "num_workers": 78,
    },
    "wgmma-s26-n128-st2-w78": {
        "num_splits": 26,
        "block_n": 128,
        "num_stages": 2,
        "num_workers": 78,
    },
    "wgmma-s39-n128-st3-w78": {
        "num_splits": 39,
        "block_n": 128,
        "num_stages": 3,
        "num_workers": 78,
    },
    "wgmma-s39-n128-st2-w78": {
        "num_splits": 39,
        "block_n": 128,
        "num_stages": 2,
        "num_workers": 78,
    },
    "wgmma-s39-n128-k2v3-w78": {
        "num_splits": 39,
        "block_n": 128,
        "num_stages": 2,
        "num_k_stages": 2,
        "num_v_stages": 3,
        "num_workers": 78,
    },
    "wgmma-s39-n128-k3v2-w78": {
        "num_splits": 39,
        "block_n": 128,
        "num_stages": 2,
        "num_k_stages": 3,
        "num_v_stages": 2,
        "num_workers": 78,
    },
    "wgmma-s10-n128-k2v1-w80": {
        "num_splits": 10,
        "block_n": 128,
        "num_stages": 1,
        "num_k_stages": 2,
        "num_v_stages": 1,
        "num_workers": 80,
    },
    "wgmma-s16-n128-k2v1-w128": {
        "num_splits": 16,
        "block_n": 128,
        "num_stages": 1,
        "num_k_stages": 2,
        "num_v_stages": 1,
        "num_workers": 128,
    },
    "wgmma-s19-n128-k2v1-w152": {
        "num_splits": 19,
        "block_n": 128,
        "num_stages": 1,
        "num_k_stages": 2,
        "num_v_stages": 1,
        "num_workers": 152,
    },
    "wgmma-s19-n128-k1v2-w152": {
        "num_splits": 19,
        "block_n": 128,
        "num_stages": 1,
        "num_k_stages": 1,
        "num_v_stages": 2,
        "num_workers": 152,
    },
    "wgmma-s39-n128-k2v1-w156": {
        "num_splits": 39,
        "block_n": 128,
        "num_stages": 1,
        "num_k_stages": 2,
        "num_v_stages": 1,
        "num_workers": 156,
    },
    "wgmma-s39-n128-k1v2-w156": {
        "num_splits": 39,
        "block_n": 128,
        "num_stages": 1,
        "num_k_stages": 1,
        "num_v_stages": 2,
        "num_workers": 156,
    },
    "wgmma-s39-n128-st2-w156": {
        "num_splits": 39,
        "block_n": 128,
        "num_stages": 2,
        "num_workers": 156,
    },
    "wgmma-s39-n128-st1-w78": {
        "num_splits": 39,
        "block_n": 128,
        "num_stages": 1,
        "num_workers": 78,
    },
    "wgmma-s39-n128-st1-w156": {
        "num_splits": 39,
        "block_n": 128,
        "num_stages": 1,
        "num_workers": 156,
    },
    "wgmma-s39-n128-st1-w234": {
        "num_splits": 39,
        "block_n": 128,
        "num_stages": 1,
        "num_workers": 234,
    },
    "wgmma-s78-n128-st1-w234": {
        "num_splits": 78,
        "block_n": 128,
        "num_stages": 1,
        "num_workers": 234,
    },
    "wgmma-s117-n128-st1-w234": {
        "num_splits": 117,
        "block_n": 128,
        "num_stages": 1,
        "num_workers": 234,
    },
    "wgmma-s39-n64-st3-w156": {
        "num_splits": 39,
        "block_n": 64,
        "num_stages": 3,
        "num_workers": 156,
    },
    "wgmma-s39-n64-st2-w156": {
        "num_splits": 39,
        "block_n": 64,
        "num_stages": 2,
        "num_workers": 156,
    },
    "wgmma-s117-n64-st2-w234": {
        "num_splits": 117,
        "block_n": 64,
        "num_stages": 2,
        "num_workers": 234,
    },
    "wgmma-s117-n64-st1-w234": {
        "num_splits": 117,
        "block_n": 64,
        "num_stages": 1,
        "num_workers": 234,
    },
}
AUTO_CONFIG = "wgmma-s39-n128-st3-w78"
BLOCK_N = CONFIGS[AUTO_CONFIG]["block_n"]
NUM_STAGES = CONFIGS[AUTO_CONFIG]["num_stages"]
NUM_SPLITS = CONFIGS[AUTO_CONFIG]["num_splits"]

_COMPILE_LOCK = threading.Lock()
_COMPILED_CACHE = {}
_WORKSPACE_LOCK = threading.Lock()
_WORKSPACE_CACHE = {}
_WORKSPACE_CACHE_LIMIT = 32
_LAUNCH_PLAN_LOCK = threading.Lock()
_LAUNCH_PLAN_CACHE = {}
_LAUNCH_PLAN_CACHE_LIMIT = 1
_KERNEL_VERSION = 1
_WORKSPACE_VERSION = 1

# Debug bisect switches (compile-time constants read from env).
_DBG_SKIP_QCOPY = _os.environ.get("AD_SKIP_QCOPY") == "1"
_DBG_SKIP_KVLOOP = _os.environ.get("AD_SKIP_KVLOOP") == "1"
_DBG_SKIP_GEMM1 = _os.environ.get("AD_SKIP_GEMM1") == "1"
_DBG_SKIP_SOFTMAX = _os.environ.get("AD_SKIP_SOFTMAX") == "1"
_DBG_SKIP_GEMM2 = _os.environ.get("AD_SKIP_GEMM2") == "1"
_DBG_SKIP_EPI = _os.environ.get("AD_SKIP_EPI") == "1"
_DBG_SKIP_COMBINE = _os.environ.get("AD_SKIP_COMBINE") == "1"


# ---------------------------------------------------------------------------
# WGMMA accumulator helpers (layout plumbing borrowed from the prefill kernel)
# ---------------------------------------------------------------------------

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
def _wgmma_gemm(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    operand_a: cute.Tensor,
    operand_b: cute.Tensor,
    zero_init: cutlass.Constexpr[bool],
):
    """acc (MMA, MMA_M, MMA_N) += a (MMA, MMA_M, MMA_K) @ b (MMA, MMA_N, MMA_K)."""
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
    warpgroup.wait_group(0)


def _transpose_smem_view(tensor: cute.Tensor) -> cute.Tensor:
    shape = (tensor.shape[1], tensor.shape[0], *tensor.shape[2:])
    order = (1, 0, *range(2, cute.rank(tensor)))
    return cute.composition(tensor, cute.make_ordered_layout(shape, order=order))


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


# ---------------------------------------------------------------------------
# Flash-decoding kernel: partial + combine
# ---------------------------------------------------------------------------

class FlashDecodeKernel:
    GROUP_M = GROUP_M
    HEAD_DIM = HEAD_DIM

    def __init__(
        self,
        q_heads: int,
        kv_heads: int,
        kv_len: int,
        batch: int,
        num_splits: int = NUM_SPLITS,
        block_n: int = BLOCK_N,
        num_k_stages: int = NUM_STAGES,
        num_v_stages: int = NUM_STAGES,
        num_workers: int = 78,
        num_threads: int = NUM_THREADS,
        balanced_heads: bool = False,
        compact_roles: bool = False,
        sm_scale: float = 1.0 / math.sqrt(HEAD_DIM),
    ):
        if kv_heads * self.GROUP_M != q_heads:
            raise ValueError("kernel expects q_heads == kv_heads * 8")
        if kv_len % block_n != 0:
            raise ValueError("kv_len must be a multiple of block_n")
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.kv_len = kv_len
        self.batch = batch
        self.num_splits = num_splits
        self.num_workers = num_workers
        self.balanced_heads = balanced_heads
        self.compact_roles = compact_roles
        self.BLOCK_N = block_n
        self.K_STAGES = num_k_stages
        self.V_STAGES = num_v_stages
        self.NUM_THREADS = num_threads
        self.scale_log2 = sm_scale * math.log2(math.e)

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,      # (B, H, 1, D) bf16
        mK: cute.Tensor,      # (B, HK, S, D) bf16
        mV: cute.Tensor,      # (B, HK, S, D) bf16
        mOpart: cute.Tensor,  # (S_splits, HK, G, D) fp32
        mLSE: cute.Tensor,    # (S_splits, HK, G) fp32 (log2 domain)
        mO: cute.Tensor,      # (B, H, 1, D) bf16
        stream: cuda.CUstream,
    ):
        self._dtype = mQ.element_type

        # (H, D, B) view of Q/O; (S, D, (HK, B)) view of K/V.
        Q = cute.make_tensor(
            mQ.iterator,
            cute.make_layout(
                (mQ.shape[1], mQ.shape[3], mQ.shape[0]),
                stride=(mQ.stride[1], mQ.stride[3], mQ.stride[0]),
            ),
        )
        O = cute.make_tensor(
            mO.iterator,
            cute.make_layout(
                (mO.shape[1], mO.shape[3], mO.shape[0]),
                stride=(mO.stride[1], mO.stride[3], mO.stride[0]),
            ),
        )
        K, V = [
            cute.make_tensor(
                t.iterator,
                cute.make_layout(
                    (t.shape[2], t.shape[3], (t.shape[1], t.shape[0])),
                    stride=(t.stride[2], t.stride[3], (t.stride[1], t.stride[0])),
                ),
            )
            for t in (mK, mV)
        ]

        smem_atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                utils.LayoutEnum.ROW_MAJOR, self._dtype, self.HEAD_DIM
            ),
            self._dtype,
        )
        sK_layout = cute.tile_to_shape(
            smem_atom, (self.BLOCK_N, self.HEAD_DIM, self.K_STAGES), (0, 1, 2)
        )
        sV_layout = cute.tile_to_shape(
            smem_atom, (self.BLOCK_N, self.HEAD_DIM, self.V_STAGES), (0, 1, 2)
        )
        sQ_layout = cute.tile_to_shape(smem_atom, (self.GROUP_M, self.HEAD_DIM), (0, 1))
        sP_layout = cute.tile_to_shape(smem_atom, (self.GROUP_M, self.BLOCK_N), (0, 1))

        @cute.struct
        class SharedStorage:
            k_mbars: cute.struct.MemRange[cutlass.Int64, self.K_STAGES * 2]
            v_mbars: cute.struct.MemRange[cutlass.Int64, self.V_STAGES * 2]
            red: cute.struct.MemRange[cutlass.Float32, 2 * 4 * self.GROUP_M]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sQ_layout)], 1024
            ]
            sP: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sP_layout)], 1024
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sK_layout)], 1024
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sV_layout)], 1024
            ]

        tma_atom_k, tma_tensor_k = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            K,
            cute.select(sK_layout, mode=[0, 1]),
            (self.BLOCK_N, self.HEAD_DIM),
        )
        tma_atom_v, tma_tensor_v = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            V,
            cute.select(sV_layout, mode=[0, 1]),
            (self.BLOCK_N, self.HEAD_DIM),
        )

        # GEMM1: S^T[N=kv, 8] = K_tile @ Q^T   (A: K-major smem, B: K-major smem)
        tiled_mma_qk = sm90_utils.make_trivial_tiled_mma(
            self._dtype,
            self._dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            ACC_TYPE,
            atom_layout_mnk=(1, 1, 1),
            tiler_mn=(64, self.GROUP_M),
        )
        # GEMM2: O^T[D, 8] = V^T @ P^T         (A: MN-major smem, B: K-major smem)
        tiled_mma_pv = sm90_utils.make_trivial_tiled_mma(
            self._dtype,
            self._dtype,
            warpgroup.OperandMajorMode.MN,
            warpgroup.OperandMajorMode.K,
            ACC_TYPE,
            atom_layout_mnk=(1, 1, 1),
            tiler_mn=(64, self.GROUP_M),
        )

        self.partial_kernel(
            Q,
            tma_atom_k,
            tma_tensor_k,
            tma_atom_v,
            tma_tensor_v,
            mOpart,
            mLSE,
            cutlass.Float32(self.scale_log2),
            sQ_layout,
            sK_layout,
            sV_layout,
            sP_layout,
            tiled_mma_qk,
            tiled_mma_pv,
            SharedStorage,
        ).launch(
            grid=(self.num_workers, 1, 1),
            block=[self.NUM_THREADS, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

        self.combine_kernel(
            mOpart,
            mLSE,
            O,
        ).launch(
            grid=(self.q_heads, self.batch, 1),
            block=[128, 1, 1],
            smem=self.num_splits * 4,
            stream=stream,
        )



    # ------------------------------------------------------------------
    # Partial kernel
    # ------------------------------------------------------------------
    @cute.kernel
    def partial_kernel(
        self,
        mQ: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        mK: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        mV: cute.Tensor,
        mOpart: cute.Tensor,
        mLSE: cute.Tensor,
        scale_log2: cutlass.Float32,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        worker, _, _ = cute.arch.block_idx()

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_k)
            cpasync.prefetch_descriptor(tma_atom_v)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        pipeline_k = _PipelineTmaAsyncNoCluster.create(
            barrier_storage=storage.k_mbars.data_ptr(),
            num_stages=self.K_STAGES,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=cute.size_in_bytes(
                self._dtype, cute.select(sK_layout, mode=[0, 1])
            ),
            init_wait=False,
        )
        pipeline_v = _PipelineTmaAsyncNoCluster.create(
            barrier_storage=storage.v_mbars.data_ptr(),
            num_stages=self.V_STAGES,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=cute.size_in_bytes(
                self._dtype, cute.select(sV_layout, mode=[0, 1])
            ),
        )

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sVt = _transpose_smem_view(sV)

        tiles_total = self.kv_len // self.BLOCK_N

        num_items = self.num_splits * self.kv_heads * self.batch
        if cutlass.const_expr(self.balanced_heads):
            num_items = self.num_workers

        producer_warp = 0
        is_producer = warp_idx < 4
        if cutlass.const_expr(self.compact_roles):
            # WGMMA consumers must occupy an aligned four-warp group (warps 0..3).
            producer_warp = 4
            is_producer = warp_idx == producer_warp
        if is_producer:
            # ------------------------- producer warp -------------------------
            if warp_idx == producer_warp:
                producer_state_k = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.K_STAGES
                )
                producer_state_v = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.V_STAGES
                )
                for item in cutlass.range(worker, num_items, self.num_workers, unroll=1):
                    if cutlass.const_expr(self.balanced_heads):
                        kv_head = worker // 10
                        split = worker % 10
                        head_splits = 10
                        if worker >= 60:
                            kv_head = 6 + (worker - 60) // 9
                            split = (worker - 60) % 9
                            head_splits = 9
                        batch = 0
                        tile_beg = split * tiles_total // head_splits
                        tile_end = (split + 1) * tiles_total // head_splits
                    else:
                        kv_head = (item // self.batch) % self.kv_heads
                        split = item // (self.batch * self.kv_heads)
                        batch = item % self.batch
                        tile_beg = split * tiles_total // self.num_splits
                        tile_end = (split + 1) * tiles_total // self.num_splits
                    n_tiles = tile_end - tile_beg
                    gK = cute.local_tile(
                        mK[None, None, (kv_head, batch)],
                        (self.BLOCK_N, self.HEAD_DIM),
                        (None, 0),
                    )
                    gV = cute.local_tile(
                        mV[None, None, (kv_head, batch)],
                        (self.BLOCK_N, self.HEAD_DIM),
                        (None, 0),
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
                    for i in cutlass.range(0 if _DBG_SKIP_KVLOOP else n_tiles, unroll=1):
                        pipeline_k.producer_acquire(producer_state_k)
                        cute.copy(
                            tma_atom_k,
                            tKgK[None, tile_beg + i],
                            tKsK[None, producer_state_k.index],
                            tma_bar_ptr=pipeline_k.producer_get_barrier(producer_state_k),
                        )
                        pipeline_k.producer_commit(producer_state_k)
                        pipeline_v.producer_acquire(producer_state_v)
                        cute.copy(
                            tma_atom_v,
                            tVgV[None, tile_beg + i],
                            tVsV[None, producer_state_v.index],
                            tma_bar_ptr=pipeline_v.producer_get_barrier(producer_state_v),
                        )
                        pipeline_v.producer_commit(producer_state_v)
                        producer_state_k.advance()
                        producer_state_v.advance()
        else:
            # ------------------------- consumer -------------------------
            tidx2 = tidx - 128
            if cutlass.const_expr(self.compact_roles):
                tidx2 = tidx
            lane = tidx2 % 32
            warp_in_wg = tidx2 // 32

            wg_mma_qk = tiled_mma_qk.get_slice(0)
            wg_mma_pv = tiled_mma_pv.get_slice(0)
            thr_mma_qk = tiled_mma_qk.get_slice(tidx2)
            thr_mma_pv = tiled_mma_pv.get_slice(tidx2)

            tSrK = tiled_mma_qk.make_fragment_A(wg_mma_qk.partition_A(sK))
            tSrQ = tiled_mma_qk.make_fragment_B(wg_mma_qk.partition_B(sQ))
            tOsVt = tiled_mma_pv.make_fragment_A(wg_mma_pv.partition_A(sVt))
            tPsP = tiled_mma_pv.make_fragment_B(wg_mma_pv.partition_B(sP))

            acc_s_shape = tiled_mma_qk.partition_shape_C((self.BLOCK_N, self.GROUP_M))
            acc_o_shape = tiled_mma_pv.partition_shape_C((self.HEAD_DIM, self.GROUP_M))
            acc_O = cute.make_rmem_tensor(acc_o_shape, ACC_TYPE)
            acc_O_mn = _make_acc_tensor_mn_view(acc_O)

            cS = cute.make_identity_tensor((self.BLOCK_N, self.GROUP_M))
            tScS = thr_mma_qk.partition_C(cS)
            tScS_mn = _make_acc_tensor_mn_view(tScS)
            cO = cute.make_identity_tensor((self.HEAD_DIM, self.GROUP_M))
            tOcO = thr_mma_pv.partition_C(cO)
            tOcO_mn = _make_acc_tensor_mn_view(tOcO)

            NR = cute.size(tScS_mn.shape[0])
            NC = cute.size(tScS_mn.shape[1])
            row_max = cute.make_rmem_tensor((NC,), ACC_TYPE)
            row_sum = cute.make_rmem_tensor((NC,), ACC_TYPE)

            red = storage.red.data_ptr()

            q_copy_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self._dtype, num_bits_per_copy=128
            )
            q_tiled_copy = cute.make_tiled_copy_tv(
                q_copy_atom,
                cute.make_layout(
                    (self.GROUP_M, 128 // self.GROUP_M), stride=(128 // self.GROUP_M, 1)
                ),
                cute.make_layout((1, 128 // (128 // self.GROUP_M))),
            )
            q_thr_copy = q_tiled_copy.get_slice(tidx2)

            consumer_state_k = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.K_STAGES
            )
            consumer_state_v = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.V_STAGES
            )
            for item in cutlass.range(worker, num_items, self.num_workers, unroll=1):
                if cutlass.const_expr(self.balanced_heads):
                    kv_head = worker // 10
                    split = worker % 10
                    head_splits = 10
                    if worker >= 60:
                        kv_head = 6 + (worker - 60) // 9
                        split = (worker - 60) % 9
                        head_splits = 9
                    batch = 0
                    tile_beg = split * tiles_total // head_splits
                    tile_end = (split + 1) * tiles_total // head_splits
                else:
                    kv_head = (item // self.batch) % self.kv_heads
                    split = item // (self.batch * self.kv_heads)
                    batch = item % self.batch
                    tile_beg = split * tiles_total // self.num_splits
                    tile_end = (split + 1) * tiles_total // self.num_splits
                n_tiles = tile_end - tile_beg

                if not _DBG_SKIP_QCOPY:
                    gQ = cute.local_tile(
                        mQ[None, None, batch], (self.GROUP_M, self.HEAD_DIM), (kv_head, 0)
                    )
                    cute.copy(
                        q_tiled_copy,
                        q_thr_copy.partition_S(gQ),
                        q_thr_copy.partition_D(sQ),
                    )
                    cute.arch.fence_proxy("async.shared", space="cta")
                    cute.arch.barrier(barrier_id=1, number_of_threads=128)

                acc_O.fill(0.0)
                row_max.fill(-cutlass.Float32.inf)
                row_sum.fill(0.0)

                for i in cutlass.range(0 if _DBG_SKIP_KVLOOP else n_tiles, unroll=1):
                    acc_S = cute.make_rmem_tensor(acc_s_shape, ACC_TYPE)
                    pipeline_k.consumer_wait(consumer_state_k)
                    if not _DBG_SKIP_GEMM1:
                        _wgmma_gemm(
                            tiled_mma_qk,
                            acc_S,
                            tSrK[None, None, None, consumer_state_k.index],
                            tSrQ,
                            zero_init=True,
                        )
                    pipeline_k.consumer_release(consumer_state_k)
                    consumer_state_k.advance()

                    acc_S_mn = _make_acc_tensor_mn_view(acc_S)

                    # ---- online softmax over kv (columns of the fragment) ----
                    if _DBG_SKIP_SOFTMAX:
                        for c in cutlass.range_constexpr(NC):
                            for r in cutlass.range_constexpr(NR):
                                acc_S_mn[r, c] = acc_S_mn[r, c] * scale_log2
                    if not _DBG_SKIP_SOFTMAX:
                        # scale into log2 domain.
                        for c in cutlass.range_constexpr(NC):
                            for r in cutlass.range_constexpr(NR):
                                acc_S_mn[r, c] = acc_S_mn[r, c] * scale_log2
                        # per-warp column max: thread-local rows + butterfly (no smem).
                        warp_max = cute.make_rmem_tensor((NC,), ACC_TYPE)
                        for c in cutlass.range_constexpr(NC):
                            v = acc_S_mn[0, c]
                            for r in cutlass.range_constexpr(1, NR):
                                v = cute.arch.fmax(v, acc_S_mn[r, c])
                            v = cute.arch.fmax(v, cute.arch.shuffle_sync_bfly(v, offset=4))
                            v = cute.arch.fmax(v, cute.arch.shuffle_sync_bfly(v, offset=8))
                            v = cute.arch.fmax(v, cute.arch.shuffle_sync_bfly(v, offset=16))
                            warp_max[c] = v
                        # p with warp-local max (always <= 1, no overflow); per-warp sums.
                        warp_sum = cute.make_rmem_tensor((NC,), ACC_TYPE)
                        for c in cutlass.range_constexpr(NC):
                            p = cute.math.exp2(acc_S_mn[0, c] - warp_max[c], fastmath=True)
                            acc_S_mn[0, c] = p
                            tsum = p
                            for r in cutlass.range_constexpr(1, NR):
                                p = cute.math.exp2(acc_S_mn[r, c] - warp_max[c], fastmath=True)
                                acc_S_mn[r, c] = p
                                tsum += p
                            tsum += cute.arch.shuffle_sync_bfly(tsum, offset=4)
                            tsum += cute.arch.shuffle_sync_bfly(tsum, offset=8)
                            tsum += cute.arch.shuffle_sync_bfly(tsum, offset=16)
                            warp_sum[c] = tsum
                        # single cross-warp exchange of (max, sum) through smem
                        if lane < 4:
                            for c in cutlass.range_constexpr(NC):
                                red[warp_in_wg * self.GROUP_M + tScS_mn[0, c][1]] = warp_max[c]
                                red[4 * self.GROUP_M + warp_in_wg * self.GROUP_M + tScS_mn[0, c][1]] = warp_sum[c]
                        cute.arch.barrier(barrier_id=1, number_of_threads=128)
                        new_max = cute.make_rmem_tensor((NC,), ACC_TYPE)
                        alpha = cute.make_rmem_tensor((NC,), ACC_TYPE)
                        for c in cutlass.range_constexpr(NC):
                            col = tScS_mn[0, c][1]
                            mt = red[0 * self.GROUP_M + col]
                            for w in cutlass.range_constexpr(1, 4):
                                mt = cute.arch.fmax(mt, red[w * self.GROUP_M + col])
                            new_max[c] = cute.arch.fmax(row_max[c], mt)
                            alpha[c] = cute.math.exp2(row_max[c] - new_max[c], fastmath=True)
                            row_max[c] = new_max[c]
                            corr = cute.math.exp2(warp_max[c] - new_max[c], fastmath=True)
                            tsum = cutlass.Float32(0.0)
                            for w in cutlass.range_constexpr(4):
                                tsum += red[4 * self.GROUP_M + w * self.GROUP_M + col] * cute.math.exp2(
                                    red[w * self.GROUP_M + col] - new_max[c], fastmath=True
                                )
                            row_sum[c] = row_sum[c] * alpha[c] + tsum
                            # rescale O accumulator and P (warp-local -> global max correction)
                            for r in cutlass.range_constexpr(cute.size(acc_O_mn.shape[0])):
                                acc_O_mn[r, c] = acc_O_mn[r, c] * alpha[c]
                            for r in cutlass.range_constexpr(NR):
                                acc_S_mn[r, c] = acc_S_mn[r, c] * corr
                        # store P^T (bf16) to smem for the second WGMMA
                        for c in cutlass.range_constexpr(NC):
                            col = tScS_mn[0, c][1]
                            for r in cutlass.range_constexpr(NR):
                                sP[col, tScS_mn[r, c][0]] = cutlass.BFloat16(acc_S_mn[r, c])
                        cute.arch.fence_proxy("async.shared", space="cta")
                        cute.arch.barrier(barrier_id=1, number_of_threads=128)

                    pipeline_v.consumer_wait(consumer_state_v)
                    if not _DBG_SKIP_GEMM2:
                        _wgmma_gemm(
                            tiled_mma_pv,
                            acc_O,
                            tOsVt[None, None, None, consumer_state_v.index],
                            tPsP,
                            zero_init=False,
                        )
                    pipeline_v.consumer_release(consumer_state_v)
                    consumer_state_v.advance()

                # ------------------------- epilogue -------------------------
                if _DBG_SKIP_EPI:
                    pass
                if not _DBG_SKIP_EPI:
                    for c in cutlass.range_constexpr(NC):
                        total = row_sum[c]
                        inv = (
                            0.0
                            if total == 0.0 or total != total
                            else cute.arch.rcp_approx(total)
                        )
                        for r in cutlass.range_constexpr(cute.size(acc_O_mn.shape[0])):
                            acc_O_mn[r, c] = acc_O_mn[r, c] * inv
                        lse = (
                            -cutlass.Float32.inf
                            if total == 0.0 or total != total
                            else row_max[c] + cute.math.log2(total, fastmath=True)
                        )
                        if warp_in_wg == 0 and lane < 4:
                            mLSE[split, kv_head, tOcO_mn[0, c][1], batch] = lse
                    for c in cutlass.range_constexpr(NC):
                        col = tOcO_mn[0, c][1]
                        for r in cutlass.range_constexpr(cute.size(acc_O_mn.shape[0])):
                            mOpart[split, kv_head, col, tOcO_mn[r, c][0], batch] = cutlass.BFloat16(
                                acc_O_mn[r, c]
                            )


    # ------------------------------------------------------------------
    # Combine kernel: log-sum-exp reduction across splits
    # ------------------------------------------------------------------
    @cute.kernel
    def combine_kernel(
        self,
        mOpart: cute.Tensor,  # (S, HK, G, D, B) fp32
        mLSE: cute.Tensor,    # (S, HK, G, B) fp32 (log2 domain)
        mO: cute.Tensor,      # (H, D, B) bf16
    ):
        q_head, batch, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        kv_head = q_head // self.GROUP_M
        g = q_head % self.GROUP_M

        smem = cutlass.utils.SmemAllocator()
        lse_buf = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout(self.num_splits), 16
        )
        for idx in cutlass.range(tidx, self.num_splits, 128):
            if cutlass.const_expr(self.balanced_heads):
                lse_buf[idx] = -cutlass.Float32.inf
                if idx < 9 or kv_head < 6:
                    lse_buf[idx] = mLSE[idx, kv_head, g, batch]
            else:
                lse_buf[idx] = mLSE[idx, kv_head, g, batch]
        cute.arch.barrier()

        # num_splits is a compile-time constant: fully unroll for ILP.
        ws = cute.make_rmem_tensor((self.num_splits,), ACC_TYPE)
        lse_max = -cutlass.Float32.inf
        for s in cutlass.range_constexpr(self.num_splits):
            lse_max = cute.arch.fmax(lse_max, lse_buf[s])
        denom = cutlass.Float32(0.0)
        for s in cutlass.range_constexpr(self.num_splits):
            ws[s] = cute.math.exp2(lse_buf[s] - lse_max, fastmath=True)
            denom += ws[s]
        acc = cutlass.Float32(0.0)
        for s in cutlass.range_constexpr(self.num_splits):
            if cutlass.const_expr(self.balanced_heads):
                if s < 9 or kv_head < 6:
                    acc += ws[s] * cutlass.Float32(
                        mOpart[s, kv_head, g, tidx, batch]
                    )
            else:
                acc += ws[s] * cutlass.Float32(
                    mOpart[s, kv_head, g, tidx, batch]
                )
        inv = 0.0 if denom == 0.0 or denom != denom else cute.arch.rcp_approx(denom)
        mO[q_head, tidx, batch] = cutlass.BFloat16(acc * inv)


# ---------------------------------------------------------------------------
# Host-side wrapper with compile/buffer caching
# ---------------------------------------------------------------------------

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
        return 1.0 / math.sqrt(HEAD_DIM)
    if isinstance(sm_scale, (bool, torch.Tensor)):
        raise TypeError("sm_scale must be a positive finite real number or None")
    try:
        value = float(sm_scale)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(
            "sm_scale must be a positive finite real number or None"
        ) from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("sm_scale must be a positive finite real number")
    if value < 1.401298464324817e-45:
        raise ValueError("sm_scale is not representable as a positive float32")
    if value > torch.finfo(torch.float32).max / LOG2_E:
        raise ValueError("sm_scale is not representable after log2(e) scaling")
    return value


def _validate_inputs(q, k, v, causal):
    import torch

    if not isinstance(causal, bool):
        raise TypeError("causal must be a bool")
    tensors = {"q": q, "k": k, "v": v}
    expected = {
        "q": (BATCH, Q_HEADS, Q_LEN, HEAD_DIM),
        "k": (BATCH, KV_HEADS, KV_LEN, HEAD_DIM),
        "v": (BATCH, KV_HEADS, KV_LEN, HEAD_DIM),
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

    with torch.cuda.device(q.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("CUDA Graph capture is not supported")
        props = torch.cuda.get_device_properties(q.device)
        capability = torch.cuda.get_device_capability(q.device)
        if capability != (9, 0):
            raise RuntimeError(
                f"the fixed kernel requires SM90, got sm_{capability[0]}{capability[1]}"
            )
        if "H20" not in props.name.upper() or props.multi_processor_count != 78:
            raise RuntimeError(
                "the fixed kernel requires NVIDIA H20 with 78 SMs; got "
                f"{props.name}, sms={props.multi_processor_count}"
            )


def _to_cute_4d(tensor):
    return (
        from_dlpack(tensor, assumed_align=16)
        .mark_layout_dynamic(leading_dim=3)
        .mark_compact_shape_dynamic(
            mode=3,
            stride_order=tensor.dim_order(),
            divisibility=128 // D_TYPE.width,
        )
    )


def _get_workspace(device, stream_handle, config_name, values):
    import torch

    key = (
        device.index,
        int(stream_handle),
        config_name,
        _WORKSPACE_VERSION,
    )
    with _WORKSPACE_LOCK:
        workspace = _WORKSPACE_CACHE.get(key)
        if workspace is None:
            if len(_WORKSPACE_CACHE) >= _WORKSPACE_CACHE_LIMIT:
                del _WORKSPACE_CACHE[next(iter(_WORKSPACE_CACHE))]
            workspace = {
                "partial": torch.empty(
                    (
                        values["num_splits"],
                        KV_HEADS,
                        GROUP_M,
                        HEAD_DIM,
                        BATCH,
                    ),
                    dtype=torch.bfloat16,
                    device=device,
                ),
                "lse": torch.empty(
                    (values["num_splits"], KV_HEADS, GROUP_M, BATCH),
                    dtype=torch.float32,
                    device=device,
                ),
                "launch_lock": threading.Lock(),
            }
            _WORKSPACE_CACHE[key] = workspace
    return workspace


def _get_launch_plan(q, k, v, workspace, config_name):
    key = (
        q.device.index,
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        workspace["partial"].data_ptr(),
        workspace["lse"].data_ptr(),
        config_name,
        _WORKSPACE_VERSION,
    )
    with _LAUNCH_PLAN_LOCK:
        plan = _LAUNCH_PLAN_CACHE.get(key)
        if plan is not None:
            return plan
        plan = {
            "q": _to_cute_4d(q),
            "k": _to_cute_4d(k),
            "v": _to_cute_4d(v),
            "partial": from_dlpack(workspace["partial"], assumed_align=16),
            "lse": from_dlpack(workspace["lse"], assumed_align=16),
            "owners": (q, k, v, workspace),
        }
        if len(_LAUNCH_PLAN_CACHE) >= _LAUNCH_PLAN_CACHE_LIMIT:
            del _LAUNCH_PLAN_CACHE[next(iter(_LAUNCH_PLAN_CACHE))]
        _LAUNCH_PLAN_CACHE[key] = plan
        return plan


def _get_compiled(plan, output_tensor, stream, device, scale, config_name, values):
    capability = __import__("torch").cuda.get_device_capability(device)
    key = (
        _KERNEL_VERSION,
        device.index,
        capability,
        config_name,
        float(scale),
    )
    compiled = _COMPILED_CACHE.get(key)
    if compiled is None:
        with _COMPILE_LOCK:
            compiled = _COMPILED_CACHE.get(key)
            if compiled is None:
                kernel = FlashDecodeKernel(
                    Q_HEADS,
                    KV_HEADS,
                    KV_LEN,
                    BATCH,
                    num_splits=values["num_splits"],
                    block_n=values["block_n"],
                    num_k_stages=values.get("num_k_stages", values["num_stages"]),
                    num_v_stages=values.get("num_v_stages", values["num_stages"]),
                    num_workers=values["num_workers"],
                    num_threads=values.get("num_threads", NUM_THREADS),
                    balanced_heads=values.get("balanced_heads", False),
                    compact_roles=values.get("compact_roles", False),
                    sm_scale=scale,
                )
                compiled = cute.compile(
                    kernel,
                    plan["q"],
                    plan["k"],
                    plan["v"],
                    plan["partial"],
                    plan["lse"],
                    output_tensor,
                    stream,
                )
                _COMPILED_CACHE[key] = compiled
    return compiled


def _run_decode(q, k, v, scale, config_name, values, return_workspace=False):
    import torch

    torch_stream = torch.cuda.current_stream(q.device)
    stream = cuda.CUstream(torch_stream.cuda_stream)
    workspace = _get_workspace(q.device, torch_stream.cuda_stream, config_name, values)
    plan = _get_launch_plan(q, k, v, workspace, config_name)
    output = torch.empty_like(q)
    output_tensor = _to_cute_4d(output)
    compiled = _get_compiled(
        plan, output_tensor, stream, q.device, scale, config_name, values
    )
    with workspace["launch_lock"]:
        compiled(
            plan["q"],
            plan["k"],
            plan["v"],
            plan["partial"],
            plan["lse"],
            output_tensor,
            stream,
        )
    for tensor in (q, k, v, workspace["partial"], workspace["lse"], output):
        tensor.record_stream(torch_stream)
    if return_workspace:
        return output, workspace
    return output


def qwen3_decode_attention(q, k, v, *, causal=True, sm_scale=None, config="auto"):
    """Run fixed-shape H20 Qwen3 128K Decode attention."""
    import torch

    _validate_inputs(q, k, v, causal)
    with torch.cuda.device(q.device):
        scale = _normalize_sm_scale(sm_scale)
        config_name, values = _resolve_config(config)
        return _run_decode(q, k, v, scale, config_name, values)


def qwen3_attention(q, k, v, *, causal=True, sm_scale=None, config="auto"):
    return qwen3_decode_attention(
        q, k, v, causal=causal, sm_scale=sm_scale, config=config
    )


attention = qwen3_attention
attention_decode = qwen3_decode_attention
run = qwen3_decode_attention


def _grouped_reference(q, k, v, scale):
    import torch

    qg = q.float().view(BATCH, KV_HEADS, GROUP_M, Q_LEN, HEAD_DIM)
    scores = torch.einsum("bhgqd,bhkd->bhgqk", qg, k.float())
    probs = torch.softmax(scores * scale, dim=-1)
    out = torch.einsum("bhgqk,bhkd->bhgqd", probs, v.float())
    return out.reshape(BATCH, Q_HEADS, Q_LEN, HEAD_DIM)


def _make_inputs(seed):
    import torch

    torch.manual_seed(seed)
    device = torch.device("cuda", torch.cuda.current_device())
    q = torch.randn(
        (BATCH, Q_HEADS, Q_LEN, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    k = torch.randn(
        (BATCH, KV_HEADS, KV_LEN, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    return q, k, torch.randn_like(k)


def _error_metrics(actual, expected):
    error = (actual.float() - expected.float()).abs()
    return error.max().item(), error.mean().item()


def _official_time(invoke, warmup, iterations):
    import torch

    for _ in range(warmup):
        output = invoke()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        output = invoke()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations, output


def _run_smoke(args):
    import torch

    q, k, v = _make_inputs(args.seed)
    output = qwen3_decode_attention(q, k, v, config=args.config)
    torch.cuda.synchronize()
    print(f"smoke passed: shape={tuple(output.shape)} dtype={output.dtype}")


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
    print("full 128K correctness passed")


def _run_benchmark(args):
    q, k, v = _make_inputs(args.seed)
    scale = args.scales[0]
    configs = [name.strip() for name in args.benchmark_configs.split(",") if name]
    results = {name: [] for name in configs}
    for round_idx in range(args.rounds):
        print(f"benchmark round={round_idx + 1}")
        for name in configs:
            us, output = _official_time(
                lambda n=name: qwen3_decode_attention(
                    q, k, v, sm_scale=scale, config=n
                ),
                args.warmup,
                args.iterations,
            )
            results[name].append(us)
            print(
                f"  {name}: {us:.3f} us, "
                f"logical_bw={536903680 / us / 1e3:.1f} GB/s"
            )
    print("median-of-rounds")
    for name, samples in results.items():
        print(f"  {name}: {statistics.median(samples):.3f} us")


def _run_profile(args):
    import torch

    q, k, v = _make_inputs(args.seed)
    for _ in range(args.warmup):
        qwen3_decode_attention(q, k, v, config=args.config)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        qwen3_decode_attention(q, k, v, config=args.config)
        torch.cuda.synchronize()
    events = [
        event
        for event in prof.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
        and event.self_device_time_total > 0
    ]
    print(f"CUDA kernel events={len(events)}")
    for event in events:
        print(f"  {event.key[:80]}: {event.self_device_time_total:.2f} us")
    if len(events) != 2:
        raise AssertionError(f"expected exactly 2 CUDA kernels, got {len(events)}")


def _parse_scales(text):
    if text is None:
        return [1.0 / math.sqrt(HEAD_DIM), 0.125]
    return [float(item) for item in text.split(",")]


def _build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["smoke", "correctness", "benchmark", "profile"],
        default="smoke",
    )
    parser.add_argument("--config", default="auto")
    parser.add_argument("--benchmark-configs", default="auto")
    parser.add_argument("--scales", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=3)
    return parser


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
