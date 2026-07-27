#!/usr/bin/env python3
"""End-to-end GQA Split-KV decode runner (Scheme A), fully in-repo.

Public entry:

    gqa_split_kv_decode(q, k, v, sm_scale=None, split_count=None) -> BHSD BF16

Pipeline:
    1. validate BHSD inputs (B=1, Hq=64, Hkv=8, q_len=1, D=128, BF16);
    2. get/allocate a device-keyed workspace (cached across calls);
    3. launch Stage-1 GQA kernel (compiled kernel cached);
    4. launch Stage-2 GQA GPU reduction (compiled kernel cached);
    5. return BHSD BF16 output.

Design notes:
    * No dependency on the out-of-tree ``/dockerdata/lxc/cute_attn_128k`` module.
    * No K/V repeat_interleave, no contiguous copy, no BSHD copy, no PyTorch
      reduction, no baseline fallback.
    * All tensors and kernels use the input ``q``'s device — no hardcoded cuda:0.
    * Workspace and compiled kernels are cached in dicts keyed by the full
      problem signature; this cache is entirely private to this module and does
      NOT touch the baseline compile cache or the SM90 prefill compile cache.
"""

import os
import sys

import torch

# The sibling Stage-1/Stage-2 modules use bare (top-level) import names, matching
# the existing split_kv_decode.py / split_kv_stage1.py convention in this package.
# Put this package directory on sys.path so those bare imports resolve to the
# in-repo GQA files (and never to any out-of-tree module).
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from split_kv_stage1_gqa import (  # noqa: E402
    ALLOWED_SPLITS,
    DTYPE_TORCH,
    GROUP_SIZE,
    HEAD_DIM,
    KV_HEADS,
    Q_HEADS,
    allocate_stage1_workspace,
    compile_split_kv_stage1_gqa,
)
from split_kv_stage2_gqa import compile_split_kv_stage2_gqa  # noqa: E402


# Private caches, keyed by full problem signature.  Never shared with baseline
# or prefill caches.
_WORKSPACE_CACHE = {}   # key -> Stage1WorkspaceGQA
_OUTPUT_CACHE = {}      # key -> BHSD BF16 output tensor
_KERNEL_CACHE = {}      # key -> (run_stage1, workspace, run_stage2, output)


def _select_split_count(kv_len):
    """First-version fixed policy; validation restricts to ALLOWED_SPLITS."""
    return 8 if kv_len < 32768 else 16


def _cache_key(q, k, split_count):
    return (
        q.device.type,
        q.device.index,
        int(q.shape[1]),   # q_heads
        int(k.shape[1]),   # kv_heads
        int(k.shape[2]),   # kv_len
        int(q.shape[3]),   # head_dim
        int(split_count),
        str(q.dtype),
    )


def gqa_split_kv_decode(q, k, v, sm_scale=None, split_count=None):
    """Run the in-repo GQA Split-KV decode. Inputs/outputs are BHSD BF16.

    q: [1, Hq, 1, D]; k/v: [1, Hkv, kv_len, D]; returns [1, Hq, 1, D].
    """
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
        split_count = _select_split_count(kv_len)
    if split_count not in ALLOWED_SPLITS:
        raise ValueError(f"第一版仅支持 split_count in {ALLOWED_SPLITS}，实际为 {split_count}")

    if sm_scale is None:
        sm_scale = 1.0 / (HEAD_DIM ** 0.5)
    sm_scale = float(sm_scale)

    key = _cache_key(q, k, split_count)

    if key not in _KERNEL_CACHE:
        # Allocate/reuse workspace + output buffers (avoid per-call big alloc).
        if key not in _WORKSPACE_CACHE:
            _WORKSPACE_CACHE[key] = allocate_stage1_workspace(split_count, device=q.device)
        workspace = _WORKSPACE_CACHE[key]

        run_stage1, workspace, _kv_len = compile_split_kv_stage1_gqa(
            q, k, v, split_count, sm_scale, workspace=workspace
        )
        run_stage2, output = compile_split_kv_stage2_gqa(workspace, output=None)
        _OUTPUT_CACHE[key] = output
        _KERNEL_CACHE[key] = (run_stage1, workspace, run_stage2, output)

    run_stage1, workspace, run_stage2, output = _KERNEL_CACHE[key]
    # Pass the *current* q/k/v so a cached compiled kernel never reuses the
    # tensors captured at compile time.
    run_stage1(q, k, v)
    run_stage2()
    return output
