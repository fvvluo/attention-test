#!/usr/bin/env python3
"""Correctness + stage-decomposed timing for Scheme-B2 (shared, tuned tile).

Two modes:
  --mode correct : correctness gate (B2 vs baseline / vs A / vs B tile8 / vs SDPA)
  --mode stages  : Stage-1 / Stage-2 / end-to-end CUDA-event timing per tile

Usage (from repo root):
    python3 ops/liuxiaochen_split_kv_decode/verify_gqa_decode_shared_b2.py --gpu 5 --mode correct
    python3 ops/liuxiaochen_split_kv_decode/verify_gqa_decode_shared_b2.py --gpu 5 --mode stages

Never modifies bench_attention. Rule: PASS if max_abs<=2e-2 OR max_rel<=2e-2.
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

from gqa_decode import gqa_split_kv_decode  # noqa: E402  Scheme A
from gqa_decode_shared import gqa_split_kv_decode_shared  # noqa: E402  Scheme B tile8
from gqa_decode_shared_b2 import (  # noqa: E402
    gqa_split_kv_decode_shared_b2,
    build_b2_runners,
)

Q_HEADS, KV_HEADS, GROUP, HEAD_DIM = 64, 8, 8, 128
DTYPE = torch.bfloat16
ABS_TOL, REL_TOL = 2e-2, 2e-2
A_SPLIT_FOR = 16  # Scheme A supports {8,16}


def route_baseline():
    repo_root = Path(__file__).resolve().parents[2]
    baseline_pkg_dir = (repo_root / "flash-attention-baseline" / "flash_attn").resolve()
    import flash_attn

    bp = str(baseline_pkg_dir)
    flash_attn.__path__ = [bp, *(p for p in flash_attn.__path__ if p != bp)]
    importlib.invalidate_caches()
    from flash_attn.cute import flash_attn_func

    def baseline(q, k, v, sm_scale=None):
        out = flash_attn_func(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            softmax_scale=sm_scale, causal=False,
        )
        if isinstance(out, tuple):
            out = out[0]
        return out.transpose(1, 2)

    return baseline


def torch_reference(q, k, v, sm):
    qf = q.float()
    kf = k.float().repeat_interleave(GROUP, dim=1)
    vf = v.float().repeat_interleave(GROUP, dim=1)
    w = torch.softmax(torch.einsum("bhqd,bhkd->bhqk", qf, kf) * sm, dim=-1)
    return torch.einsum("bhqk,bhkd->bhqd", w, vf)


def compare(name, out, ref):
    of, rf = out.float(), ref.float()
    nan = bool(torch.isnan(of).any().item())
    inf = bool(torch.isinf(of).any().item())
    ad = (of - rf).abs()
    ma = ad.max().item()
    mr = (ad / (rf.abs() + 1e-6)).max().item()
    ok = (not nan) and (not inf) and ((ma <= ABS_TOL) or (mr <= REL_TOL))
    print(f"    [{name}] max_abs={ma:.3e} max_rel={mr:.3e} nan={nan} inf={inf} -> {'PASS' if ok else 'FAIL'}")
    return ok


def mk(seed, kv_len, dev):
    torch.manual_seed(seed)
    q = torch.randn(1, Q_HEADS, 1, HEAD_DIM, dtype=DTYPE, device=dev)
    k = torch.randn(1, KV_HEADS, kv_len, HEAD_DIM, dtype=DTYPE, device=dev)
    v = torch.randn(1, KV_HEADS, kv_len, HEAD_DIM, dtype=DTYPE, device=dev)
    return q, k, v


def mode_correct(baseline, dev):
    sm = 1.0 / (HEAD_DIM ** 0.5)
    cases = [
        (1024, 16, 32, [0], True),
        (8192, 16, 64, [0, 1, 2026], False),
        (8192, 32, 64, [0, 1, 2026], False),
        (8192, 64, 64, [0, 1, 2026], False),
        (131072, 16, 128, [0, 1, 2026], False),
        (131072, 32, 128, [0, 1, 2026], False),
        (131072, 64, 128, [0, 1, 2026], False),
    ]
    all_ok = True
    for kv_len, tile, split, seeds, sdpa in cases:
        for seed in seeds:
            q, k, v = mk(seed, kv_len, dev)
            outb2 = gqa_split_kv_decode_shared_b2(q, k, v, sm_scale=sm, split_count=split, tokens_per_tile=tile)
            print(f"  kv_len={kv_len} tile={tile} split={split} seed={seed}: "
                  f"out {tuple(outb2.shape)} {outb2.dtype} {outb2.device}")
            ok = True
            ok &= compare("B2 vs baseline", outb2, baseline(q, k, v, sm_scale=sm))
            ok &= compare("B2 vs A", outb2, gqa_split_kv_decode(q, k, v, sm_scale=sm, split_count=A_SPLIT_FOR))
            ok &= compare("B2 vs B(tile8)", outb2, gqa_split_kv_decode_shared(q, k, v, sm_scale=sm, split_count=min(split,128)))
            if sdpa:
                ok &= compare("B2 vs torch-ref", outb2, torch_reference(q, k, v, sm))
            all_ok &= ok
            if not ok:
                print("FAIL — stopping this tile.")
                sys.exit(2)
    print("ALL PASS")


def time_stage(fn, warmup=5, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def mode_stages(dev):
    sm = 1.0 / (HEAD_DIM ** 0.5)
    kv_len, split = 131072, 128
    q, k, v = mk(0, kv_len, dev)
    print(f"stage timing: kv_len={kv_len} split={split}")
    print("| tile | Stage1 ms | Stage2 ms | total ms | tiles/CTA | barriers/CTA |")
    print("|---:|---:|---:|---:|---:|---:|")
    for tile in [8, 16, 32, 64]:
        run1, run2, out = build_b2_runners(q, k, v, sm, split, tile)
        t1 = time_stage(lambda: run1(q, k, v))
        t2 = time_stage(lambda: run2())
        tt = time_stage(lambda: (run1(q, k, v), run2()))
        tiles = (kv_len // split) // tile
        barriers = tiles * 2
        print(f"| {tile} | {t1:.4f} | {t2:.4f} | {tt:.4f} | {tiles} | {barriers} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--mode", choices=["correct", "stages"], default="correct")
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu)
    dev = torch.device(f"cuda:{args.gpu}")
    print(f"device: {dev} ({torch.cuda.get_device_name(dev)})")
    if args.mode == "correct":
        mode_correct(route_baseline(), dev)
    else:
        mode_stages(dev)


if __name__ == "__main__":
    main()
