#!/usr/bin/env python3
"""B7 unified-interface dispatch adapter (Liu Xiaochen).

Exposes the team-standard entry:

    attention(q, k, v, causal=True, sm_scale=None) -> BHSD output

It validates the team-native BHSD decode inputs and forwards to the B7 warp-MMA
cp.async+PDL runner (runner_b7.mma_decode_b7). It performs NO data movement of
its own — no repeat_interleave, no expand/transpose+contiguous, no clone, no
dtype cast, no baseline/SDPA/CPU fallback. Unsupported inputs raise TypeError/
ValueError; there is never a silent fallback or wrong result.

B7 itself owns the compile/workspace cache (keyed by device/shape/split/dtype),
launched with the current q/k/v each call. This adapter only selects a validated
(num_splits, n_block) configuration from the verified table and calls B7.

Scope (current verified target):
    B=1, Hq=64, Hkv=8, q_len=1, D=128, BF16, SM90, full-visible-prefix decode
    (causal semantics on q_len==1 == non-causal; the team decode phase passes
    causal=False and K/V already contain exactly the visible prefix).
"""

import math
import os
import sys

import torch

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from runner_b7 import mma_decode_b7, _num_splits  # noqa: E402

_Q_HEADS = 64
_KV_HEADS = 8
_HEAD_DIM = 128
_N_BLOCK = 64

# Verified (kv_len -> num_splits) configurations. These are the shapes exercised
# by verify_mma_b7 / the D13-D16 correctness+perf runs. Anything else raises in
# the development phase (no silent heuristic).
_VERIFIED_SPLITS = {
    64: 1,
    512: 8,
    1024: 16,
    8192: 32,
    131072: 256,
}


def _select_num_splits(kv_len):
    if kv_len in _VERIFIED_SPLITS:
        return _VERIFIED_SPLITS[kv_len]
    raise ValueError(
        f"kv_len={kv_len} is not in the verified split table {sorted(_VERIFIED_SPLITS)}; "
        "refusing to guess a split in the development phase."
    )


def attention(q, k, v, causal=True, sm_scale=None):
    """Team-standard unified interface -> B7 warp-MMA decode. BHSD in/out, BF16."""
    # --- type / device / dtype / layout validation ---
    for name, t in (("q", q), ("k", k), ("v", v)):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{name} 必须是 torch.Tensor，实际为 {type(t)}")
        if not t.is_cuda:
            raise ValueError(f"{name} 必须位于 CUDA device")
    if not (q.device == k.device == v.device):
        raise ValueError("q/k/v 必须位于同一个 CUDA device")
    if not (q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16):
        raise TypeError("B7 dispatch 仅支持 bfloat16")
    major, minor = torch.cuda.get_device_capability(q.device)
    if (major, minor) != (9, 0):
        raise ValueError(f"B7 dispatch 需要 SM90，当前 SM{major}{minor}")

    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q/k/v 必须是 4 维 BHSD (batch, heads, seq_len, head_dim)")
    b, hq, q_len, d = q.shape
    _, hk, kv_len, dk = k.shape

    # --- shape / GQA validation ---
    if q_len != 1:
        raise ValueError(f"B7 dispatch 是 decode-only：要求 q_len==1，实际 q_len={q_len}")
    if d != _HEAD_DIM or dk != _HEAD_DIM:
        raise ValueError(f"B7 dispatch 要求 D=={_HEAD_DIM}")
    if tuple(k.shape) != tuple(v.shape):
        raise ValueError("k 与 v 必须同 shape")
    if hq % hk != 0:
        raise ValueError(f"Hq={hq} 必须是 Hkv={hk} 的整数倍")
    if b != 1 or hq != _Q_HEADS or hk != _KV_HEADS:
        raise ValueError(
            f"B7 dispatch 当前正式目标为 B=1, Hq={_Q_HEADS}, Hkv={_KV_HEADS}；"
            f"实际 B={b}, Hq={hq}, Hkv={hk}"
        )
    if kv_len % _N_BLOCK != 0:
        raise ValueError(f"B7 dispatch 要求 kv_len % {_N_BLOCK} == 0，实际 kv_len={kv_len}")
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        raise ValueError("B7 dispatch 要求 contiguous BHSD 输入（不做隐藏 copy）")

    # --- causal semantics ---
    # On q_len==1 the single query is the newest token and sees the entire KV
    # cache, so causal and non-causal are equivalent (there is no future to mask).
    # The team decode phase passes causal=False and builds K/V as exactly the
    # visible prefix, which is precisely B7's supported semantics. We therefore
    # accept both causal=True and causal=False identically for q_len==1, and do
    # NOT apply any positional mask. (If some caller passed q_len>1 with causal
    # it would already have been rejected above.)
    _ = causal  # intentionally semantics-equivalent for q_len==1

    # --- sm_scale (B7 runner applies it internally; pass through, no double scale) ---
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d)
    sm_scale = float(sm_scale)
    if not math.isfinite(sm_scale) or sm_scale <= 0.0:
        raise ValueError("sm_scale 必须是有限正数")

    num_splits = _select_num_splits(kv_len)
    return mma_decode_b7(q, k, v, sm_scale=sm_scale, num_splits=num_splits, n_block=_N_BLOCK)
