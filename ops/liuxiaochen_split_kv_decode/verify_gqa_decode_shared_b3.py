#!/usr/bin/env python3
"""Correctness verification for Scheme-B3 (shared + 64-bit vectorized load).

Compares B3 vs baseline / vs A / vs B2 / (small) vs SDPA.
Usage: python3 ops/liuxiaochen_split_kv_decode/verify_gqa_decode_shared_b3.py --gpu 6
Rule: PASS if max_abs<=2e-2 OR max_rel<=2e-2.
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

from gqa_decode import gqa_split_kv_decode as A  # noqa: E402
from gqa_decode_shared_b2 import gqa_split_kv_decode_shared_b2 as B2  # noqa: E402
from gqa_decode_shared_b3 import gqa_split_kv_decode_shared_b3 as B3  # noqa: E402

Q_HEADS, KV_HEADS, GROUP, HEAD_DIM = 64, 8, 8, 128
DTYPE = torch.bfloat16
ABS_TOL, REL_TOL = 2e-2, 2e-2


def route_baseline():
    baseline_pkg_dir = (Path(__file__).resolve().parents[2] / "flash-attention-baseline" / "flash_attn").resolve()
    import flash_attn
    bp = str(baseline_pkg_dir)
    flash_attn.__path__ = [bp, *(p for p in flash_attn.__path__ if p != bp)]
    importlib.invalidate_caches()
    from flash_attn.cute import flash_attn_func

    def baseline(q, k, v, sm_scale=None):
        out = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), softmax_scale=sm_scale, causal=False)
        return (out[0] if isinstance(out, tuple) else out).transpose(1, 2)
    return baseline


def torch_ref(q, k, v, sm):
    kf = k.float().repeat_interleave(GROUP, 1)
    vf = v.float().repeat_interleave(GROUP, 1)
    w = torch.softmax(torch.einsum("bhqd,bhkd->bhqk", q.float(), kf) * sm, -1)
    return torch.einsum("bhqk,bhkd->bhqd", w, vf)


def compare(name, out, ref):
    of, rf = out.float(), ref.float()
    nan = bool(torch.isnan(of).any().item()); inf = bool(torch.isinf(of).any().item())
    ad = (of - rf).abs(); ma = ad.max().item(); mr = (ad / (rf.abs() + 1e-6)).max().item()
    ok = (not nan) and (not inf) and ((ma <= ABS_TOL) or (mr <= REL_TOL))
    print(f"    [{name}] max_abs={ma:.3e} max_rel={mr:.3e} nan={nan} inf={inf} -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, required=True)
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu); dev = torch.device(f"cuda:{args.gpu}")
    print(f"device: {dev} ({torch.cuda.get_device_name(dev)})")
    baseline = route_baseline()
    sm = 1.0 / (HEAD_DIM ** 0.5)
    # (kv_len, tile, split, a_split, b2_split, seeds, sdpa)
    cases = [
        (1024, 32, 32, 16, 32, [0, 1, 2026], True),
        (8192, 32, 64, 16, 64, [0, 1, 2026], False),
        (131072, 32, 256, 16, 256, [0, 1, 2026], False),
    ]
    all_ok = True
    for kv_len, tile, split, a_split, b2_split, seeds, sdpa in cases:
        for seed in seeds:
            torch.manual_seed(seed)
            q = torch.randn(1, Q_HEADS, 1, HEAD_DIM, dtype=DTYPE, device=dev)
            k = torch.randn(1, KV_HEADS, kv_len, HEAD_DIM, dtype=DTYPE, device=dev)
            v = torch.randn(1, KV_HEADS, kv_len, HEAD_DIM, dtype=DTYPE, device=dev)
            out = B3(q, k, v, sm_scale=sm, split_count=split, tokens_per_tile=tile)
            print(f"  kv_len={kv_len} tile={tile} split={split} seed={seed}: out {tuple(out.shape)} {out.dtype} {out.device}")
            ok = True
            ok &= compare("B3 vs baseline", out, baseline(q, k, v, sm_scale=sm))
            ok &= compare("B3 vs A", out, A(q, k, v, sm_scale=sm, split_count=a_split))
            ok &= compare("B3 vs B2", out, B2(q, k, v, sm_scale=sm, split_count=b2_split, tokens_per_tile=tile))
            if sdpa:
                ok &= compare("B3 vs torch-ref", out, torch_ref(q, k, v, sm))
            all_ok &= ok
            if not ok:
                print("FAIL — stopping."); sys.exit(2)
    print("ALL PASS"); sys.exit(0)


if __name__ == "__main__":
    main()
