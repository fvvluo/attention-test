#!/usr/bin/env python3
"""End-to-end Scheme-B (shared K/V) GQA Split-KV decode runner, fully in-repo.

Public entry:

    gqa_split_kv_decode_shared(q, k, v, sm_scale=None, split_count=64) -> BHSD BF16

Pipeline: validate BHSD inputs -> get/allocate device-keyed workspace ->
launch Scheme-B Stage-1 (grid [KV_HEADS, split]) -> launch Stage-2 GPU reduction
-> return BHSD BF16.

Cache design mirrors Scheme A but is fully independent (key includes
``impl="shared"``), so it never collides with Scheme A / prefill / baseline
caches.  The compiled kernels are launched with the *current* q/k/v every call
(``run_stage1(q,k,v)``) — no closure binds a specific input tensor.
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
from split_kv_stage1_gqa_shared import (  # noqa: E402
    ALLOWED_SPLITS,
    compile_split_kv_stage1_gqa_shared,
)
from split_kv_stage2_gqa_shared import compile_split_kv_stage2_gqa_shared  # noqa: E402

# Independent Scheme-B caches (never shared with Scheme A / prefill / baseline).
_WORKSPACE_CACHE = {}
_KERNEL_CACHE = {}


def _default_split_count(kv_len):
    """Scheme-B default policy; validation restricts to ALLOWED_SPLITS.

    128K default is split=64 (per D6-9). Smaller kv_len keeps a moderate split.
    """
    env = os.environ.get("LIUXIAOCHEN_DECODE_SPLITS")
    if env:
        return int(env)
    return 64


def _cache_key(q, k, split_count):
    return (
        "shared",
        q.device.type,
        q.device.index,
        int(q.shape[1]),
        int(k.shape[1]),
        int(k.shape[2]),
        int(q.shape[3]),
        int(split_count),
        str(q.dtype),
    )


def gqa_split_kv_decode_shared(q, k, v, sm_scale=None, split_count=None):
    """Scheme-B GQA Split-KV decode. BHSD in/out, BF16."""
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q/k/v 必须是 4 维 BHSD tensor")
    if q.shape[2] != 1:
        raise ValueError(f"decode 路径要求 q_len==1，实际 q_len={q.shape[2]}")
    if q.shape[1] != Q_HEADS or k.shape[1] != KV_HEADS or v.shape[1] != KV_HEADS:
        raise ValueError(
            f"第一版仅支持 Hq={Q_HEADS}, Hkv={KV_HEADS}；"
            f"实际 Hq={q.shape[1]}, Hk={k.shape[1]}, Hv={v.shape[1]}"
        )
    if q.shape[1] // k.shape[1] != GROUP_SIZE:
        raise ValueError(f"group_size 必须为 {GROUP_SIZE}")

    kv_len = int(k.shape[2])
    if split_count is None:
        split_count = _default_split_count(kv_len)
    if split_count not in ALLOWED_SPLITS:
        raise ValueError(f"Scheme B 仅支持 split_count in {ALLOWED_SPLITS}，实际为 {split_count}")

    if sm_scale is None:
        sm_scale = 1.0 / (HEAD_DIM ** 0.5)
    sm_scale = float(sm_scale)

    key = _cache_key(q, k, split_count)
    if key not in _KERNEL_CACHE:
        if key not in _WORKSPACE_CACHE:
            _WORKSPACE_CACHE[key] = allocate_stage1_workspace(split_count, device=q.device)
        workspace = _WORKSPACE_CACHE[key]

        run_stage1, workspace, _kv_len = compile_split_kv_stage1_gqa_shared(
            q, k, v, split_count, sm_scale, workspace=workspace
        )
        run_stage2, output = compile_split_kv_stage2_gqa_shared(workspace, output=None)
        _KERNEL_CACHE[key] = (run_stage1, workspace, run_stage2, output)

    run_stage1, workspace, run_stage2, output = _KERNEL_CACHE[key]
    run_stage1(q, k, v)
    run_stage2()
    return output
