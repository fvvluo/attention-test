#!/usr/bin/env python3
"""Layered correctness verification for the Scheme-B (shared K/V) GQA decode.

Compares ``gqa_split_kv_decode_shared`` against:
  * flash-attention-baseline ``flash_attn.cute.flash_attn_func`` (routed like bench);
  * the Scheme-A ``gqa_split_kv_decode`` (independent implementation);
  * a PyTorch FP32 reference for small cases.

Usage (from repo root):
    python3 ops/liuxiaochen_split_kv_decode/verify_gqa_decode_shared.py --gpu 5

Rule (matches repo benchmark): PASS if max_abs <= 2e-2 OR max_rel <= 2e-2.
Verification only; never benchmarks.
"""

import argparse
import importlib
import os
import sys
from pathlib import Path

import torch

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from gqa_decode import gqa_split_kv_decode  # noqa: E402  (Scheme A)
from gqa_decode_shared import gqa_split_kv_decode_shared  # noqa: E402  (Scheme B)

Q_HEADS, KV_HEADS, GROUP, HEAD_DIM = 64, 8, 8, 128
DTYPE = torch.bfloat16
ABS_TOL, REL_TOL = 2e-2, 2e-2
# Scheme A only allows split in {8,16}; map the B split to a valid A split for the
# A/B cross-check (both are exact references of the same math).
A_SPLIT_FOR = {32: 16, 64: 16, 128: 16}


def route_baseline():
    repo_root = Path(__file__).resolve().parents[2]
    baseline_pkg_dir = (repo_root / "flash-attention-baseline" / "flash_attn").resolve()
    if not baseline_pkg_dir.is_dir():
        raise ImportError(f"找不到 flash-attention-baseline: {baseline_pkg_dir}")
    import flash_attn

    baseline_path = str(baseline_pkg_dir)
    flash_attn.__path__ = [baseline_path, *(p for p in flash_attn.__path__ if p != baseline_path)]
    importlib.invalidate_caches()
    from flash_attn.cute import flash_attn_func
    import flash_attn.cute as flash_attn_cute

    loaded_from = Path(flash_attn_cute.__file__).resolve()
    if baseline_pkg_dir not in loaded_from.parents:
        raise ImportError(f"flash_attn.cute 路由错误: {loaded_from}")

    def baseline(q, k, v, sm_scale=None):
        out = flash_attn_func(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            softmax_scale=sm_scale, causal=False,
        )
        if isinstance(out, tuple):
            out = out[0]
        return out.transpose(1, 2)

    return baseline


def torch_reference(q, k, v, sm_scale):
    qf = q.float()
    kf = k.float().repeat_interleave(GROUP, dim=1)
    vf = v.float().repeat_interleave(GROUP, dim=1)
    scores = torch.einsum("bhqd,bhkd->bhqk", qf, kf) * sm_scale
    weights = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bhkd->bhqd", weights, vf)


def compare(name, out, ref):
    of, rf = out.float(), ref.float()
    has_nan = bool(torch.isnan(of).any().item())
    has_inf = bool(torch.isinf(of).any().item())
    ad = (of - rf).abs()
    max_abs = ad.max().item()
    max_rel = (ad / (rf.abs() + 1e-6)).max().item()
    passed = (not has_nan) and (not has_inf) and ((max_abs <= ABS_TOL) or (max_rel <= REL_TOL))
    print(f"    [{name}] max_abs={max_abs:.3e} max_rel={max_rel:.3e} "
          f"nan={has_nan} inf={has_inf} -> {'PASS' if passed else 'FAIL'}")
    return passed


def run_case(baseline, device, kv_len, split_count, seed, do_sdpa):
    torch.manual_seed(seed)
    q = torch.randn(1, Q_HEADS, 1, HEAD_DIM, dtype=DTYPE, device=device)
    k = torch.randn(1, KV_HEADS, kv_len, HEAD_DIM, dtype=DTYPE, device=device)
    v = torch.randn(1, KV_HEADS, kv_len, HEAD_DIM, dtype=DTYPE, device=device)
    sm = 1.0 / (HEAD_DIM ** 0.5)

    out_b = gqa_split_kv_decode_shared(q, k, v, sm_scale=sm, split_count=split_count)
    print(f"  case kv_len={kv_len} split={split_count} seed={seed}: "
          f"output shape={tuple(out_b.shape)} dtype={out_b.dtype} device={out_b.device}")
    ok = True
    ok &= compare("B vs baseline", out_b, baseline(q, k, v, sm_scale=sm))
    out_a = gqa_split_kv_decode(q, k, v, sm_scale=sm, split_count=A_SPLIT_FOR[split_count])
    ok &= compare("B vs A", out_b, out_a)
    if do_sdpa:
        ok &= compare("B vs torch-ref", out_b, torch_reference(q, k, v, sm))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    print(f"device: {device} ({torch.cuda.get_device_name(device)})")
    baseline = route_baseline()

    # (kv_len, split, seeds, do_sdpa)
    cases = [
        (1024, 32, [0], True),
        (8192, 32, [0, 1, 2026], False),
        (8192, 64, [0, 1, 2026], False),
        (131072, 32, [0], False),
        (131072, 64, [0, 1, 2026], False),
        (131072, 128, [0], False),
    ]
    all_ok = True
    for kv_len, split, seeds, do_sdpa in cases:
        for seed in seeds:
            ok = run_case(baseline, device, kv_len, split, seed, do_sdpa)
            all_ok &= ok
            if not ok:
                print("FAIL — stopping.")
                sys.exit(2)
    print("ALL PASS" if all_ok else "SOME FAILED")
    sys.exit(0 if all_ok else 2)


if __name__ == "__main__":
    main()
