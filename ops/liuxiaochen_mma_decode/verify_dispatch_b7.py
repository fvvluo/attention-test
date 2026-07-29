#!/usr/bin/env python3
"""Direct correctness for the B7 unified-interface adapter (Liu Xiaochen).

Verifies dispatch_b7.attention == direct runner_b7.mma_decode_b7 (bit-identical)
and vs baseline / SDPA, plus checks: no input data_ptr change, no full-input
copy, output shape/dtype/device/stride, explicit sm_scale, NaN/Inf.

Usage: python3 ops/liuxiaochen_mma_decode/verify_dispatch_b7.py --gpu <idle>
"""

import argparse
import importlib
import math
import os
import sys
from pathlib import Path

import torch

_PKG = os.path.dirname(os.path.abspath(__file__))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from dispatch_b7 import attention as B7_DISPATCH  # noqa: E402
from runner_b7 import mma_decode_b7 as B7_DIRECT  # noqa: E402

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
    bl = route_baseline()
    default_sm = 1.0 / math.sqrt(D)
    # (kv_len, num_splits_for_direct, scales, seeds, sdpa)
    cases = [
        (64, 1, [None], [0, 1, 2026], True),
        (512, 8, [None], [0, 1, 2026], True),
        (8192, 32, [None], [0, 1, 2026], False),
        (131072, 256, [None], [0], False),
        # explicit scale coverage (default and a distinct legal scale)
        (512, 8, [default_sm, 0.05], [0], True),
        (131072, 256, [default_sm], [1, 2026], False),
    ]
    all_ok = True
    for kv, ns, scales, seeds, sdpa in cases:
        for scale in scales:
            for seed in seeds:
                torch.manual_seed(seed)
                q = torch.randn(1, QH, 1, D, dtype=DT, device=dev)
                k = torch.randn(1, KVH, kv, D, dtype=DT, device=dev)
                v = torch.randn(1, KVH, kv, D, dtype=DT, device=dev)
                qp, kp, vp = q.data_ptr(), k.data_ptr(), v.data_ptr()
                mem0 = torch.cuda.memory_allocated(dev)
                out = B7_DISPATCH(q, k, v, causal=True, sm_scale=scale)
                mem1 = torch.cuda.memory_allocated(dev)
                # inputs untouched (no clone/copy of q/k/v)
                same_ptr = (q.data_ptr() == qp and k.data_ptr() == kp and v.data_ptr() == vp)
                # a full K/V copy would add ~2*kv*KVH*D*2 bytes; flag if allocation grew hugely
                kv_bytes = 2 * kv * KVH * D * 2
                big_copy = (mem1 - mem0) > kv_bytes  # workspace is far smaller than a full K/V copy
                eff_sm = scale if scale is not None else default_sm
                direct = B7_DIRECT(q, k, v, sm_scale=eff_sm, num_splits=ns)
                print(f"  kv={kv} split={ns} scale={scale} seed={seed}: "
                      f"shape={tuple(out.shape)} dtype={out.dtype} dev={out.device} "
                      f"stride={out.stride()} same_input_ptr={same_ptr} "
                      f"alloc_delta={mem1-mem0}B big_copy={big_copy}")
                ok = True
                ok &= same_ptr and (not big_copy)
                if not same_ptr:
                    print("    [inputs] FAIL: q/k/v data_ptr changed")
                if big_copy:
                    print("    [copy] FAIL: allocation grew beyond a full K/V copy")
                # bit-identical vs direct runner
                d_ad = (out.float() - direct.float()).abs().max().item()
                bit_ok = (d_ad == 0.0)
                print(f"    [dispatch vs direct] max_abs={d_ad:.3e} -> {'PASS (bit-identical)' if bit_ok else 'FAIL'}")
                ok &= bit_ok
                ok &= cmp("dispatch vs baseline", out, bl(q, k, v, eff_sm))
                if sdpa:
                    ok &= cmp("dispatch vs SDPA", out, sdpa_ref(q, k, v, eff_sm))
                all_ok &= ok
                if not ok:
                    print("FAIL — stopping."); sys.exit(2)
    print("ALL PASS"); sys.exit(0)


if __name__ == "__main__":
    main()
