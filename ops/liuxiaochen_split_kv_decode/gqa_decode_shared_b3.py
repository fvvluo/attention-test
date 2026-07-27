#!/usr/bin/env python3
"""End-to-end Scheme-B3 (shared K/V + 64-bit vectorized load) GQA decode runner.

Same interface/cache discipline as B2, but Stage-1 uses the B3 vectorized-load
kernel.  Cache key includes impl="shared_b3"; independent of A/B/B2/prefill/base.
Compiled kernels launch with the current q/k/v each call (no tensor-bound closure).

Env overrides: LIUXIAOCHEN_DECODE_SPLITS (default 256), LIUXIAOCHEN_DECODE_TILE (default 32).
"""

import os
import sys

import torch

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from split_kv_stage1_gqa import (  # noqa: E402
    GROUP_SIZE,
    HEAD_DIM,
    KV_HEADS,
    Q_HEADS,
    allocate_stage1_workspace,
)
from split_kv_stage1_gqa_shared_b3 import (  # noqa: E402
    ALLOWED_SPLITS,
    ALLOWED_TILES,
    compile_split_kv_stage1_gqa_shared_b3,
)
from split_kv_stage2_gqa_shared import compile_split_kv_stage2_gqa_shared  # noqa: E402

_WORKSPACE_CACHE = {}
_KERNEL_CACHE = {}


def _default_split_count():
    env = os.environ.get("LIUXIAOCHEN_DECODE_SPLITS")
    return int(env) if env else 256


def _default_tile():
    env = os.environ.get("LIUXIAOCHEN_DECODE_TILE")
    return int(env) if env else 32


def _cache_key(q, k, split_count, tokens_per_tile):
    return (
        "shared_b3", q.device.type, q.device.index,
        int(q.shape[1]), int(k.shape[1]), int(k.shape[2]), int(q.shape[3]),
        int(split_count), int(tokens_per_tile), str(q.dtype),
    )


def gqa_split_kv_decode_shared_b3(q, k, v, sm_scale=None, split_count=None, tokens_per_tile=None):
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q/k/v 必须是 4 维 BHSD tensor")
    if q.shape[2] != 1:
        raise ValueError(f"decode 路径要求 q_len==1，实际 q_len={q.shape[2]}")
    if q.shape[1] != Q_HEADS or k.shape[1] != KV_HEADS or v.shape[1] != KV_HEADS:
        raise ValueError(
            f"第一版仅支持 Hq={Q_HEADS}, Hkv={KV_HEADS}；实际 Hq={q.shape[1]}, Hk={k.shape[1]}, Hv={v.shape[1]}"
        )
    if q.shape[1] // k.shape[1] != GROUP_SIZE:
        raise ValueError(f"group_size 必须为 {GROUP_SIZE}")

    if split_count is None:
        split_count = _default_split_count()
    if tokens_per_tile is None:
        tokens_per_tile = _default_tile()
    if split_count not in ALLOWED_SPLITS:
        raise ValueError(f"B3 仅支持 split_count in {ALLOWED_SPLITS}，实际 {split_count}")
    if tokens_per_tile not in ALLOWED_TILES:
        raise ValueError(f"B3 仅支持 tokens_per_tile in {ALLOWED_TILES}，实际 {tokens_per_tile}")

    if sm_scale is None:
        sm_scale = 1.0 / (HEAD_DIM ** 0.5)
    sm_scale = float(sm_scale)

    key = _cache_key(q, k, split_count, tokens_per_tile)
    if key not in _KERNEL_CACHE:
        if key not in _WORKSPACE_CACHE:
            _WORKSPACE_CACHE[key] = allocate_stage1_workspace(split_count, device=q.device)
        workspace = _WORKSPACE_CACHE[key]
        run_stage1, workspace, _ = compile_split_kv_stage1_gqa_shared_b3(
            q, k, v, split_count, tokens_per_tile, sm_scale, workspace=workspace
        )
        run_stage2, output = compile_split_kv_stage2_gqa_shared(workspace, output=None)
        _KERNEL_CACHE[key] = (run_stage1, workspace, run_stage2, output)

    run_stage1, workspace, run_stage2, output = _KERNEL_CACHE[key]
    run_stage1(q, k, v)
    run_stage2()
    return output


def build_b3_runners(q, k, v, sm_scale, split_count, tokens_per_tile):
    if sm_scale is None:
        sm_scale = 1.0 / (HEAD_DIM ** 0.5)
    workspace = allocate_stage1_workspace(split_count, device=q.device)
    run_stage1, workspace, _ = compile_split_kv_stage1_gqa_shared_b3(
        q, k, v, split_count, tokens_per_tile, float(sm_scale), workspace=workspace
    )
    run_stage2, output = compile_split_kv_stage2_gqa_shared(workspace, output=None)
    return run_stage1, run_stage2, output
