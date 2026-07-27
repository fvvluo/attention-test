#!/usr/bin/env python3
"""End-to-end B4 GQA decode runner: wide split + two-level combine + chunked softmax.

gqa_split_kv_decode_shared_b4(q,k,v, sm_scale=None, split_count=512,
                              tokens_per_tile=32, tokens_per_group=1) -> BHSD BF16

Independent cache (impl="shared_b4" + split + tile + group). Launches with the
current q/k/v each call. Env: LIUXIAOCHEN_DECODE_SPLITS / _TILE / _GROUP.
Also exposes build_b4_runners for stage-decomposed timing.
"""

import os
import sys

import torch

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from split_kv_stage1_gqa import (  # noqa: E402
    GROUP_SIZE, HEAD_DIM, KV_HEADS, Q_HEADS, allocate_stage1_workspace,
)
from split_kv_stage1_gqa_shared_b4 import (  # noqa: E402
    ALLOWED_GROUPS, ALLOWED_SPLITS, ALLOWED_TILES,
    compile_split_kv_stage1_gqa_shared_b4,
)
from split_kv_stage2_gqa_shared_b4 import compile_combine_b4  # noqa: E402

_WORKSPACE_CACHE = {}
_KERNEL_CACHE = {}


def _default_split():
    e = os.environ.get("LIUXIAOCHEN_DECODE_SPLITS")
    return int(e) if e else 512


def _default_tile():
    e = os.environ.get("LIUXIAOCHEN_DECODE_TILE")
    return int(e) if e else 32


def _default_group():
    e = os.environ.get("LIUXIAOCHEN_DECODE_GROUP")
    return int(e) if e else 1


def _key(q, k, split, tile, group):
    return ("shared_b4", q.device.type, q.device.index, int(q.shape[1]), int(k.shape[1]),
            int(k.shape[2]), int(q.shape[3]), int(split), int(tile), int(group), str(q.dtype))


def gqa_split_kv_decode_shared_b4(q, k, v, sm_scale=None, split_count=None,
                                  tokens_per_tile=None, tokens_per_group=None):
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q/k/v 必须是 4 维 BHSD")
    if q.shape[2] != 1:
        raise ValueError(f"decode 要求 q_len==1, got {q.shape[2]}")
    if q.shape[1] != Q_HEADS or k.shape[1] != KV_HEADS or v.shape[1] != KV_HEADS:
        raise ValueError(f"仅支持 Hq={Q_HEADS},Hkv={KV_HEADS}")
    if q.shape[1] // k.shape[1] != GROUP_SIZE:
        raise ValueError(f"group_size 必须 {GROUP_SIZE}")

    split_count = split_count if split_count is not None else _default_split()
    tokens_per_tile = tokens_per_tile if tokens_per_tile is not None else _default_tile()
    tokens_per_group = tokens_per_group if tokens_per_group is not None else _default_group()
    if split_count not in ALLOWED_SPLITS:
        raise ValueError(f"B4 split in {ALLOWED_SPLITS}, got {split_count}")
    if tokens_per_tile not in ALLOWED_TILES:
        raise ValueError(f"B4 tile in {ALLOWED_TILES}, got {tokens_per_tile}")
    if tokens_per_group not in ALLOWED_GROUPS:
        raise ValueError(f"B4 group in {ALLOWED_GROUPS}, got {tokens_per_group}")

    if sm_scale is None:
        sm_scale = 1.0 / (HEAD_DIM ** 0.5)
    sm_scale = float(sm_scale)

    key = _key(q, k, split_count, tokens_per_tile, tokens_per_group)
    if key not in _KERNEL_CACHE:
        if key not in _WORKSPACE_CACHE:
            _WORKSPACE_CACHE[key] = allocate_stage1_workspace(split_count, device=q.device)
        ws = _WORKSPACE_CACHE[key]
        run_stage1, ws, _ = compile_split_kv_stage1_gqa_shared_b4(
            q, k, v, split_count, tokens_per_tile, tokens_per_group, sm_scale, workspace=ws
        )
        run_combine, output = compile_combine_b4(ws, output=None)
        _KERNEL_CACHE[key] = (run_stage1, ws, run_combine, output)

    run_stage1, ws, run_combine, output = _KERNEL_CACHE[key]
    run_stage1(q, k, v)
    run_combine()
    return output


def build_b4_runners(q, k, v, sm_scale, split_count, tokens_per_tile, tokens_per_group):
    if sm_scale is None:
        sm_scale = 1.0 / (HEAD_DIM ** 0.5)
    ws = allocate_stage1_workspace(split_count, device=q.device)
    run_stage1, ws, _ = compile_split_kv_stage1_gqa_shared_b4(
        q, k, v, split_count, tokens_per_tile, tokens_per_group, float(sm_scale), workspace=ws
    )
    run_combine, output = compile_combine_b4(ws, output=None)
    return run_stage1, run_combine, output
