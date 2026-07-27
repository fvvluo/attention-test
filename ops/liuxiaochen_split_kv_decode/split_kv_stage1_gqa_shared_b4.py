#!/usr/bin/env python3
"""GQA Split-KV Stage-1 — Scheme B4.

Two experiments layered on B3 (``split_kv_stage1_gqa_shared_b3.py``):
  1. wider split_count {256,512,1024} (paired with the B4 two-level combine);
  2. optional chunked online-softmax (TOKENS_PER_GROUP in {1,2,4}) that merges
     the per-warp running (m,l,p) state once per GROUP instead of once per token,
     cutting the serial dependency chain and p-rescale count by GROUP.

GROUP=1 is exactly B3 math (per-token merge). GROUP>1 processes a group of G
tokens, computes a local (group_m, group_l, group_p) with G-way ILP, then merges
into the running state once. Result is mathematically identical to B3 (verified
bit-close). All else identical to B3: grid [KV_HEADS, split], 256 threads/8 warps,
warp w -> q_head kv_head*8+w, 64-bit vectorized K/V load into shared, per-warp
FP32 state, raw m/l/p workspace output.

Constraints: B=1,Hq=64,Hkv=8,q_len=1,D=128,BF16,causal=False;
split in {256,512,1024}; tile in {8,16,32,64}; group in {1,2,4};
kv_len % split == 0; (kv_len//split) % tile == 0; tile % group == 0.
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

ALLOWED_SPLITS = (256, 512, 1024)
ALLOWED_TILES = (8, 16, 32, 64)
ALLOWED_GROUPS = (1, 2, 4)

NUM_WARPS = GROUP_SIZE          # 8
WARP_SIZE = 32
NUM_THREADS = NUM_WARPS * WARP_SIZE  # 256
ELEMS_PER_LANE = HEAD_DIM // WARP_SIZE  # 4

BF16_PER_I64 = 4
D_I64 = HEAD_DIM // BF16_PER_I64  # 32


def _make_cute_tensor(t):
    return from_dlpack(t, assumed_align=16)


def _as_i64(t):
    return t.view(torch.int64)


def _validate_inputs(Q, K, V, split_count, tokens_per_tile, tokens_per_group, softmax_scale):
    for name, tensor in {"Q": Q, "K": K, "V": V}.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} 必须是 torch.Tensor")
        if not tensor.is_cuda:
            raise ValueError(f"{name} 必须位于 CUDA device")
        if tensor.dtype != DTYPE_TORCH:
            raise TypeError(f"{name} dtype 必须是 {DTYPE_TORCH}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} 必须是 contiguous BHSD")
        if tensor.data_ptr() % 16 != 0:
            raise ValueError(f"{name} 起始地址必须 16-byte 对齐")
    if not (Q.device == K.device == V.device):
        raise ValueError("Q/K/V 必须同 device")
    if tuple(Q.shape) != (1, Q_HEADS, 1, HEAD_DIM):
        raise ValueError(f"Q shape 必须 (1,{Q_HEADS},1,{HEAD_DIM}), got {tuple(Q.shape)}")
    if K.ndim != 4 or (K.shape[0], K.shape[1], K.shape[3]) != (1, KV_HEADS, HEAD_DIM):
        raise ValueError(f"K shape 必须 (1,{KV_HEADS},kv_len,{HEAD_DIM}), got {tuple(K.shape)}")
    if tuple(V.shape) != tuple(K.shape):
        raise ValueError("V shape 必须 == K")
    if split_count not in ALLOWED_SPLITS:
        raise ValueError(f"B4 split in {ALLOWED_SPLITS}, got {split_count}")
    if tokens_per_tile not in ALLOWED_TILES:
        raise ValueError(f"B4 tile in {ALLOWED_TILES}, got {tokens_per_tile}")
    if tokens_per_group not in ALLOWED_GROUPS:
        raise ValueError(f"B4 group in {ALLOWED_GROUPS}, got {tokens_per_group}")
    if tokens_per_tile % tokens_per_group != 0:
        raise ValueError(f"tile % group != 0: tile={tokens_per_tile}, group={tokens_per_group}")
    kv_len = int(K.shape[2])
    if kv_len % split_count != 0:
        raise ValueError(f"kv_len % split != 0: {kv_len} % {split_count}")
    tokens_per_split = kv_len // split_count
    if tokens_per_split % tokens_per_tile != 0:
        raise ValueError(f"(kv_len//split) % tile != 0: {tokens_per_split} % {tokens_per_tile}")
    if not isinstance(softmax_scale, Real) or isinstance(softmax_scale, bool):
        raise TypeError("softmax_scale 必须是实数")
    if not math.isfinite(float(softmax_scale)) or float(softmax_scale) <= 0:
        raise ValueError("softmax_scale 必须是有限正数")
    return kv_len, tokens_per_split


class SplitKVStage1GQASharedB4:
    def __init__(self, split_count, tokens_per_split, tokens_per_tile, tokens_per_group):
        self.split_count = split_count
        self.tokens_per_split = tokens_per_split
        self.tokens_per_tile = tokens_per_tile
        self.tokens_per_group = tokens_per_group
        self.num_tiles = tokens_per_split // tokens_per_tile
        self.groups_per_tile = tokens_per_tile // tokens_per_group

    @cute.jit
    def __call__(self, mQ, mK64, mV64, mMax, mSum, mP, softmax_scale, stream):
        self.kernel(mQ, mK64, mV64, mMax, mSum, mP, softmax_scale).launch(
            grid=[KV_HEADS, self.split_count, 1],
            block=[NUM_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mQ, mK64, mV64, mMax, mSum, mP, softmax_scale):
        tidx, _, _ = cute.arch.thread_idx()
        kv_head, split_idx, _ = cute.arch.block_idx()
        warp_id = tidx >> 5
        lane_id = tidx & 31

        q_head = kv_head * GROUP_SIZE + warp_id
        split_start = split_idx * self.tokens_per_split

        TPT = self.tokens_per_tile
        G = self.tokens_per_group
        smem = cutlass.utils.SmemAllocator()
        sK64 = smem.allocate_tensor(cutlass.Int64, cute.make_layout((TPT, D_I64), stride=(D_I64, 1)), 16)
        sV64 = smem.allocate_tensor(cutlass.Int64, cute.make_layout((TPT, D_I64), stride=(D_I64, 1)), 16)
        sK = cute.recast_tensor(sK64, cutlass.BFloat16)
        sV = cute.recast_tensor(sV64, cutlass.BFloat16)

        q_frag = cute.make_rmem_tensor((ELEMS_PER_LANE,), cutlass.Float32)
        p_frag = cute.make_rmem_tensor((ELEMS_PER_LANE,), cutlass.Float32)
        for e in range(ELEMS_PER_LANE):
            d_idx = lane_id + e * WARP_SIZE
            q_frag[e] = mQ[0, q_head, 0, d_idx].to(cutlass.Float32)
            p_frag[e] = cutlass.Float32(0.0)

        warp_m = -cutlass.Float32.inf
        warp_l = cutlass.Float32(0.0)

        I64_PER_TILE = TPT * D_I64
        LOADS_PER_THREAD = I64_PER_TILE // NUM_THREADS

        # group-local accumulators (per lane)
        g_scores = cute.make_rmem_tensor((G,), cutlass.Float32)
        gp_frag = cute.make_rmem_tensor((ELEMS_PER_LANE,), cutlass.Float32)

        for tile in range(self.num_tiles):
            tile_token0 = split_start + tile * TPT
            for li in range(LOADS_PER_THREAD):
                flat = tidx + li * NUM_THREADS
                t = flat // D_I64
                c = flat % D_I64
                sK64[t, c] = mK64[0, kv_head, tile_token0 + t, c]
                sV64[t, c] = mV64[0, kv_head, tile_token0 + t, c]
            cute.arch.sync_threads()

            for grp in range(self.groups_per_tile):
                base_tk = grp * G
                # 1) compute G scores (each warp-reduced), find group max
                group_m = -cutlass.Float32.inf
                for gi in range(G):
                    tk = base_tk + gi
                    dot = cutlass.Float32(0.0)
                    for e in range(ELEMS_PER_LANE):
                        d_idx = lane_id + e * WARP_SIZE
                        dot += q_frag[e] * sK[tk, d_idx].to(cutlass.Float32)
                    dot = cute.arch.warp_reduction_sum(dot, threads_in_group=WARP_SIZE)
                    sc = dot * softmax_scale
                    g_scores[gi] = sc
                    group_m = cute.arch.fmax(group_m, sc)
                # 2) group_l and group_p with G-way ILP
                group_l = cutlass.Float32(0.0)
                for e in range(ELEMS_PER_LANE):
                    gp_frag[e] = cutlass.Float32(0.0)
                for gi in range(G):
                    tk = base_tk + gi
                    w = cute.math.exp(g_scores[gi] - group_m, fastmath=False)
                    group_l += w
                    for e in range(ELEMS_PER_LANE):
                        d_idx = lane_id + e * WARP_SIZE
                        gp_frag[e] += w * sV[tk, d_idx].to(cutlass.Float32)
                # 3) merge group into running state once
                new_m = cute.arch.fmax(warp_m, group_m)
                alpha = cute.math.exp(warp_m - new_m, fastmath=False)
                beta = cute.math.exp(group_m - new_m, fastmath=False)
                warp_l = alpha * warp_l + beta * group_l
                for e in range(ELEMS_PER_LANE):
                    p_frag[e] = alpha * p_frag[e] + beta * gp_frag[e]
                warp_m = new_m

            cute.arch.sync_threads()

        if lane_id == 0:
            mMax[q_head, split_idx] = warp_m
            mSum[q_head, split_idx] = warp_l
        for e in range(ELEMS_PER_LANE):
            d_idx = lane_id + e * WARP_SIZE
            mP[q_head, split_idx, d_idx] = p_frag[e]


def compile_split_kv_stage1_gqa_shared_b4(
    Q, K, V, split_count, tokens_per_tile, tokens_per_group, softmax_scale, workspace=None
):
    kv_len, tokens_per_split = _validate_inputs(
        Q, K, V, split_count, tokens_per_tile, tokens_per_group, softmax_scale
    )
    if workspace is None:
        workspace = allocate_stage1_workspace(split_count, device=Q.device)
    validate_stage1_workspace(workspace, split_count, device=Q.device, require_cuda=True)

    K64, V64 = _as_i64(K), _as_i64(V)
    mQ = _make_cute_tensor(Q)
    mK64, mV64 = _make_cute_tensor(K64), _make_cute_tensor(V64)
    mMax, mSum, mP = (
        _make_cute_tensor(t)
        for t in (workspace.partial_max, workspace.partial_sum, workspace.partial_output)
    )

    device = Q.device
    torch_stream = torch.cuda.current_stream(device)
    stream = cuda.CUstream(torch_stream.cuda_stream)
    kernel = SplitKVStage1GQASharedB4(split_count, tokens_per_split, tokens_per_tile, tokens_per_group)
    compiled = cute.compile(kernel, mQ, mK64, mV64, mMax, mSum, mP, float(softmax_scale), stream)
    scale_f = float(softmax_scale)

    def run_stage1(cur_Q, cur_K, cur_V):
        if torch.cuda.current_stream(device).cuda_stream != torch_stream.cuda_stream:
            raise RuntimeError("Stage-1(B4) 必须在编译时同一 stream 上运行")
        compiled(
            _make_cute_tensor(cur_Q),
            _make_cute_tensor(_as_i64(cur_K)),
            _make_cute_tensor(_as_i64(cur_V)),
            _make_cute_tensor(workspace.partial_max),
            _make_cute_tensor(workspace.partial_sum),
            _make_cute_tensor(workspace.partial_output),
            scale_f,
            stream,
        )
        return workspace

    return run_stage1, workspace, kv_len
