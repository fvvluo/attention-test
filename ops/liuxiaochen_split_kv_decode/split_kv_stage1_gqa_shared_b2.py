#!/usr/bin/env python3
"""GQA Split-KV Stage-1 — Scheme B2 (shared K/V, parameterized tile size).

Scheme B (``split_kv_stage1_gqa_shared.py``) fixed ``TOKENS_PER_TILE=8``, which at
128K/split=128 means 128 tiles per CTA and ~256 ``sync_threads`` barriers per CTA.
That barrier count was suspected to dominate long-sequence latency.

Scheme B2 keeps the identical CTA structure (grid ``[KV_HEADS, split]``, 256
threads = 8 warps, warp ``w`` -> ``q_head = kv_head*8 + w``, K/V loaded once per
tile into shared memory, all 8 warps read from shared) but makes
``TOKENS_PER_TILE`` a compile-time parameter in {8, 16, 32, 64}, so a larger tile
processes more tokens per barrier and cuts barriers/CTA proportionally:

    tile=8  -> tokens_per_split/8  tiles -> 2x that many barriers
    tile=16 -> half the barriers of tile=8
    tile=32 -> a quarter
    tile=64 -> an eighth

Everything else (online-softmax math, per-warp state, raw FP32 workspace output)
is unchanged, so Stage-2 and Scheme A remain valid references.

Constraints: B=1, Hq=64, Hkv=8, q_len=1, D=128, BF16, causal=False,
split_count in {32,64,128,256}, kv_len % split_count == 0,
(kv_len//split_count) % TOKENS_PER_TILE == 0.
"""

import math
from numbers import Real

import torch

if not hasattr(torch, "float4_e2m1fn_x2"):
    torch.float4_e2m1fn_x2 = torch.uint8

import cuda.bindings.driver as cuda  # noqa: E402
import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass.cute.runtime import from_dlpack  # noqa: E402

from split_kv_stage1_gqa import (  # noqa: E402
    DTYPE_TORCH,
    GROUP_SIZE,
    HEAD_DIM,
    KV_HEADS,
    Q_HEADS,
    allocate_stage1_workspace,
    validate_stage1_workspace,
)

ALLOWED_SPLITS = (32, 64, 128, 256)
ALLOWED_TILES = (8, 16, 32, 64)

NUM_WARPS = GROUP_SIZE          # 8
WARP_SIZE = 32
NUM_THREADS = NUM_WARPS * WARP_SIZE  # 256
ELEMS_PER_LANE = HEAD_DIM // WARP_SIZE  # 4 (strided d = lane + e*32)


def _make_cute_tensor(tensor):
    return from_dlpack(tensor, assumed_align=16)


def _validate_inputs(Q, K, V, split_count, tokens_per_tile, softmax_scale):
    for name, tensor in {"Q": Q, "K": K, "V": V}.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} 必须是 torch.Tensor")
        if not tensor.is_cuda:
            raise ValueError(f"{name} 必须位于 CUDA device")
        if tensor.dtype != DTYPE_TORCH:
            raise TypeError(f"{name} dtype 必须是 {DTYPE_TORCH}，实际为 {tensor.dtype}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} 必须是 contiguous BHSD，禁止隐式复制")
        if tensor.data_ptr() % 16 != 0:
            raise ValueError(f"{name} 起始地址必须满足 16-byte 对齐")
    if not (Q.device == K.device == V.device):
        raise ValueError("Q/K/V 必须位于同一个 CUDA device")

    if tuple(Q.shape) != (1, Q_HEADS, 1, HEAD_DIM):
        raise ValueError(f"Q shape 必须是 (1, {Q_HEADS}, 1, {HEAD_DIM}), 实际 {tuple(Q.shape)}")
    if K.ndim != 4 or (K.shape[0], K.shape[1], K.shape[3]) != (1, KV_HEADS, HEAD_DIM):
        raise ValueError(f"K shape 必须是 (1, {KV_HEADS}, kv_len, {HEAD_DIM}), 实际 {tuple(K.shape)}")
    if tuple(V.shape) != tuple(K.shape):
        raise ValueError("V shape 必须与 K 完全相同")

    if split_count not in ALLOWED_SPLITS:
        raise ValueError(f"B2 仅支持 split_count in {ALLOWED_SPLITS}，实际 {split_count}")
    if tokens_per_tile not in ALLOWED_TILES:
        raise ValueError(f"B2 仅支持 tokens_per_tile in {ALLOWED_TILES}，实际 {tokens_per_tile}")

    kv_len = int(K.shape[2])
    if kv_len % split_count != 0:
        raise ValueError(f"要求 kv_len % split == 0；kv_len={kv_len}, split={split_count}")
    tokens_per_split = kv_len // split_count
    if tokens_per_split % tokens_per_tile != 0:
        raise ValueError(
            f"要求 (kv_len//split) % tile == 0；tokens_per_split={tokens_per_split}, tile={tokens_per_tile}"
        )

    if not isinstance(softmax_scale, Real) or isinstance(softmax_scale, bool):
        raise TypeError("softmax_scale 必须是实数")
    if not math.isfinite(float(softmax_scale)) or float(softmax_scale) <= 0:
        raise ValueError("softmax_scale 必须是有限正数")
    return kv_len, tokens_per_split


class SplitKVStage1GQASharedB2:
    """256-thread / 8-warp CTA; one CTA per (kv_head, split); parameterized tile."""

    def __init__(self, split_count, tokens_per_split, tokens_per_tile):
        self.split_count = split_count
        self.tokens_per_split = tokens_per_split
        self.tokens_per_tile = tokens_per_tile
        self.num_tiles = tokens_per_split // tokens_per_tile

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mMax: cute.Tensor,
        mSum: cute.Tensor,
        mP: cute.Tensor,
        softmax_scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        self.kernel(mQ, mK, mV, mMax, mSum, mP, softmax_scale).launch(
            grid=[KV_HEADS, self.split_count, 1],
            block=[NUM_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mMax: cute.Tensor,
        mSum: cute.Tensor,
        mP: cute.Tensor,
        softmax_scale: cutlass.Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        kv_head, split_idx, _ = cute.arch.block_idx()
        warp_id = tidx >> 5
        lane_id = tidx & 31

        q_head = kv_head * GROUP_SIZE + warp_id
        split_start = split_idx * self.tokens_per_split

        TPT = self.tokens_per_tile
        smem = cutlass.utils.SmemAllocator()
        sK = smem.allocate_tensor(
            element_type=cutlass.BFloat16,
            layout=cute.make_layout((TPT, HEAD_DIM), stride=(HEAD_DIM, 1)),
            byte_alignment=16,
        )
        sV = smem.allocate_tensor(
            element_type=cutlass.BFloat16,
            layout=cute.make_layout((TPT, HEAD_DIM), stride=(HEAD_DIM, 1)),
            byte_alignment=16,
        )

        q_frag = cute.make_rmem_tensor((ELEMS_PER_LANE,), cutlass.Float32)
        p_frag = cute.make_rmem_tensor((ELEMS_PER_LANE,), cutlass.Float32)
        for e in range(ELEMS_PER_LANE):
            d_idx = lane_id + e * WARP_SIZE
            q_frag[e] = mQ[0, q_head, 0, d_idx].to(cutlass.Float32)
            p_frag[e] = cutlass.Float32(0.0)

        warp_m = -cutlass.Float32.inf
        warp_l = cutlass.Float32(0.0)

        ELEMS_PER_TILE = TPT * HEAD_DIM
        LOADS_PER_THREAD = ELEMS_PER_TILE // NUM_THREADS

        for tile in range(self.num_tiles):
            tile_token0 = split_start + tile * TPT

            # Cooperative single GMEM load of the K/V tile into shared memory.
            for li in range(LOADS_PER_THREAD):
                flat = tidx + li * NUM_THREADS
                t = flat // HEAD_DIM
                d = flat % HEAD_DIM
                sK[t, d] = mK[0, kv_head, tile_token0 + t, d]
                sV[t, d] = mV[0, kv_head, tile_token0 + t, d]
            cute.arch.sync_threads()

            # Each warp streams its q_head over the whole tile from shared memory.
            for tk in range(TPT):
                dot = cutlass.Float32(0.0)
                for e in range(ELEMS_PER_LANE):
                    d_idx = lane_id + e * WARP_SIZE
                    k_val = sK[tk, d_idx].to(cutlass.Float32)
                    dot += q_frag[e] * k_val
                dot = cute.arch.warp_reduction_sum(dot, threads_in_group=WARP_SIZE)
                score = dot * softmax_scale

                new_m = cute.arch.fmax(warp_m, score)
                alpha = cute.math.exp(warp_m - new_m, fastmath=False)
                beta = cute.math.exp(score - new_m, fastmath=False)
                warp_l = alpha * warp_l + beta
                for e in range(ELEMS_PER_LANE):
                    d_idx = lane_id + e * WARP_SIZE
                    v_val = sV[tk, d_idx].to(cutlass.Float32)
                    p_frag[e] = alpha * p_frag[e] + beta * v_val
                warp_m = new_m

            cute.arch.sync_threads()

        if lane_id == 0:
            mMax[q_head, split_idx] = warp_m
            mSum[q_head, split_idx] = warp_l
        for e in range(ELEMS_PER_LANE):
            d_idx = lane_id + e * WARP_SIZE
            mP[q_head, split_idx, d_idx] = p_frag[e]


def compile_split_kv_stage1_gqa_shared_b2(
    Q, K, V, split_count, tokens_per_tile, softmax_scale, workspace=None
):
    """Compile B2 Stage-1; return ``run_stage1(cur_Q,cur_K,cur_V), workspace, kv_len``."""
    kv_len, tokens_per_split = _validate_inputs(
        Q, K, V, split_count, tokens_per_tile, softmax_scale
    )
    if workspace is None:
        workspace = allocate_stage1_workspace(split_count, device=Q.device)
    validate_stage1_workspace(workspace, split_count, device=Q.device, require_cuda=True)

    mQ, mK, mV = (_make_cute_tensor(t) for t in (Q, K, V))
    mMax, mSum, mP = (
        _make_cute_tensor(t)
        for t in (workspace.partial_max, workspace.partial_sum, workspace.partial_output)
    )

    device = Q.device
    torch_stream = torch.cuda.current_stream(device)
    stream = cuda.CUstream(torch_stream.cuda_stream)
    kernel = SplitKVStage1GQASharedB2(split_count, tokens_per_split, tokens_per_tile)
    compiled = cute.compile(
        kernel, mQ, mK, mV, mMax, mSum, mP, float(softmax_scale), stream
    )
    scale_f = float(softmax_scale)

    def run_stage1(cur_Q, cur_K, cur_V):
        current_stream = torch.cuda.current_stream(device)
        if current_stream.cuda_stream != torch_stream.cuda_stream:
            raise RuntimeError("Stage-1(B2) 必须在编译时使用的同一 CUDA stream 上运行")
        compiled(
            _make_cute_tensor(cur_Q),
            _make_cute_tensor(cur_K),
            _make_cute_tensor(cur_V),
            _make_cute_tensor(workspace.partial_max),
            _make_cute_tensor(workspace.partial_sum),
            _make_cute_tensor(workspace.partial_output),
            scale_f,
            stream,
        )
        return workspace

    return run_stage1, workspace, kv_len
