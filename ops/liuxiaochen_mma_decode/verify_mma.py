#!/usr/bin/env python3
"""B5 warp-MMA decode layered correctness (independent, by Liu Xiaochen).

Usage: python3 ops/liuxiaochen_mma_decode/verify_mma.py --gpu 6
Rule: PASS if max_abs<=2e-2 OR max_rel<=2e-2.
"""

import argparse
import importlib
import os
import sys
from pathlib import Path

import torch

_PKG = os.path.dirname(os.path.abspath(__file__))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from runner import mma_decode_b5, _num_splits  # noqa: E402

QH, KVH, G, D = 64, 8, 8, 128
DT = torch.bfloat16
ABS, REL = 2e-2, 2e-2


def route_baseline():
    bp = str((Path(__file__).resolve().parents[2] / "flash-attention-baseline" / "flash_attn").resolve())
    import flash_attn
    flash_attn.__path__ = [bp, *(p for p in flash_attn.__path__ if p != bp)]
    importlib.invalidate_caches()
    from flash_attn.cute import flash_attn_func

    def bl(q, k, v, sm):
        o = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), softmax_scale=sm, causal=False)
        return (o[0] if isinstance(o, tuple) else o).transpose(1, 2)
    return bl


def sdpa_ref(q, k, v, sm):
    kf = k.float().repeat_interleave(G, 1); vf = v.float().repeat_interleave(G, 1)
    w = torch.softmax(torch.einsum("bhqd,bhkd->bhqk", q.float(), kf) * sm, -1)
    return torch.einsum("bhqk,bhkd->bhqd", w, vf)


def cmp(name, out, ref):
    of, rf = out.float(), ref.float()
    nan = bool(torch.isnan(of).any()); inf = bool(torch.isinf(of).any())
    ad = (of - rf).abs(); ma = ad.max().item(); mr = (ad / (rf.abs() + 1e-6)).max().item()
    ok = (not nan) and (not inf) and ((ma <= ABS) or (mr <= REL))
    print(f"    [{name}] max_abs={ma:.3e} max_rel={mr:.3e} nan={nan} inf={inf} -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, required=True)
    a = ap.parse_args()
    torch.cuda.set_device(a.gpu); dev = torch.device(f"cuda:{a.gpu}")
    print(f"device: {dev} ({torch.cuda.get_device_name(dev)})")
    bl = route_baseline(); sm = 1.0 / (D ** 0.5)
    try:
        from gqa_decode_shared_b4 import gqa_split_kv_decode_shared_b4 as B4  # noqa
        have_b4 = True
    except Exception:
        have_b4 = False

    # (kv_len, num_splits, seeds, sdpa)
    cases = [
        (64, 1, [0, 1, 2026], True),
        (512, 1, [0, 1, 2026], True),
        (1024, 8, [0, 1, 2026], True),
        (8192, 16, [0, 1, 2026], False),
        (8192, 32, [0, 1, 2026], False),
        (131072, 128, [0], False),
        (131072, 256, [0, 1, 2026], False),
    ]
    all_ok = True
    for kv, ns, seeds, sdpa in cases:
        for seed in seeds:
            torch.manual_seed(seed)
            q = torch.randn(1, QH, 1, D, dtype=DT, device=dev)
            k = torch.randn(1, KVH, kv, D, dtype=DT, device=dev)
            v = torch.randn(1, KVH, kv, D, dtype=DT, device=dev)
            out = mma_decode_b5(q, k, v, sm_scale=sm, num_splits=ns)
            print(f"  kv={kv} splits={ns} seed={seed}: {tuple(out.shape)} {out.dtype}")
            ok = True
            ok &= cmp("B5 vs baseline", out, bl(q, k, v, sm))
            if sdpa:
                ok &= cmp("B5 vs SDPA", out, sdpa_ref(q, k, v, sm))
            all_ok &= ok
            if not ok:
                print("FAIL — stopping."); sys.exit(2)
    print("ALL PASS"); sys.exit(0)


if __name__ == "__main__":
    main()
