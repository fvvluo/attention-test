#!/usr/bin/env python3
"""Layered correctness verification for the in-repo GQA Split-KV decode.

Compares ``gqa_split_kv_decode`` against:
  * the flash-attention-baseline ``flash_attn.cute.flash_attn_func`` (routed
    exactly like bench_attention.get_baseline_fn);
  * a PyTorch FP32 reference (SDPA-equivalent manual softmax) for small cases.

Usage (from repo root):
    python3 ops/liuxiaochen_split_kv_decode/verify_gqa_decode.py --gpu 4

Correctness rule (matches the repo benchmark): PASS if
    max_abs <= 2e-2 OR max_rel <= 2e-2.

This is a verification script only; it never benchmarks or reports latency.
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

from gqa_decode import gqa_split_kv_decode  # noqa: E402

Q_HEADS = 64
KV_HEADS = 8
GROUP = Q_HEADS // KV_HEADS
HEAD_DIM = 128
DTYPE = torch.bfloat16
ABS_TOL = 2e-2
REL_TOL = 2e-2


def route_baseline():
    """Route flash_attn.cute to the in-repo baseline checkout, like bench_attention."""
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

    def baseline(q_bhsd, k_bhsd, v_bhsd, sm_scale=None):
        # BHSD -> BSHD view for the baseline wrapper.
        out = flash_attn_func(
            q_bhsd.transpose(1, 2),
            k_bhsd.transpose(1, 2),
            v_bhsd.transpose(1, 2),
            softmax_scale=sm_scale,
            causal=False,
        )
        if isinstance(out, tuple):
            out = out[0]
        return out.transpose(1, 2)

    return baseline


def torch_reference(q, k, v, sm_scale):
    """FP32 GQA decode reference: expand kv to q_heads, full softmax, no mask."""
    qf = q.float()  # [1, Hq, 1, D]
    kf = k.float()  # [1, Hkv, S, D]
    vf = v.float()
    kf = kf.repeat_interleave(GROUP, dim=1)  # -> [1, Hq, S, D]
    vf = vf.repeat_interleave(GROUP, dim=1)
    scores = torch.einsum("bhqd,bhkd->bhqk", qf, kf) * sm_scale  # [1,Hq,1,S]
    weights = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhqk,bhkd->bhqd", weights, vf)  # [1,Hq,1,D]
    return out


def compare(name, out, ref):
    of = out.float()
    rf = ref.float()
    has_nan = bool(torch.isnan(of).any().item())
    has_inf = bool(torch.isinf(of).any().item())
    abs_diff = (of - rf).abs()
    max_abs = abs_diff.max().item()
    max_rel = (abs_diff / (rf.abs() + 1e-6)).max().item()
    passed = (not has_nan) and (not has_inf) and ((max_abs <= ABS_TOL) or (max_rel <= REL_TOL))
    print(
        f"    [{name}] max_abs={max_abs:.3e} max_rel={max_rel:.3e} "
        f"nan={has_nan} inf={has_inf} -> {'PASS' if passed else 'FAIL'}"
    )
    return passed


def run_case(baseline, device, kv_len, split_count, seed, do_sdpa):
    torch.manual_seed(seed)
    q = torch.randn(1, Q_HEADS, 1, HEAD_DIM, dtype=DTYPE, device=device)
    k = torch.randn(1, KV_HEADS, kv_len, HEAD_DIM, dtype=DTYPE, device=device)
    v = torch.randn(1, KV_HEADS, kv_len, HEAD_DIM, dtype=DTYPE, device=device)
    sm_scale = 1.0 / (HEAD_DIM ** 0.5)

    out = gqa_split_kv_decode(q, k, v, sm_scale=sm_scale, split_count=split_count)
    print(
        f"  case kv_len={kv_len} split={split_count} seed={seed}: "
        f"output shape={tuple(out.shape)} dtype={out.dtype} device={out.device}"
    )

    ok = True
    base_out = baseline(q, k, v, sm_scale=sm_scale)
    ok &= compare("vs baseline", out, base_out)
    if do_sdpa:
        ref = torch_reference(q, k, v, sm_scale)
        ok &= compare("vs torch-ref", out, ref)
    return ok, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--split", type=int, default=None, help="override split_count")
    args = ap.parse_args()

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    print(f"device: {device} ({torch.cuda.get_device_name(device)})")

    baseline = route_baseline()

    # (kv_len, split_count, do_sdpa)
    cases = [
        (1024, 8, True),
        (8192, 8, False),
        (8192, 16, False),
        (131072, 16, False),
    ]
    seeds = [0, 1, 2026]

    all_ok = True
    for kv_len, split_count, do_sdpa in cases:
        sc = args.split if args.split is not None else split_count
        for seed in seeds:
            ok, _ = run_case(baseline, device, kv_len, sc, seed, do_sdpa)
            all_ok &= ok
            if not ok:
                print("FAIL — stopping.")
                sys.exit(2)

    print("ALL PASS" if all_ok else "SOME FAILED")
    sys.exit(0 if all_ok else 2)


if __name__ == "__main__":
    main()
