#!/usr/bin/env python3
"""Correctness suites for the stride-aware GQA CuTe prefill kernel."""

import argparse
import math
import sys

import torch
import torch.nn.functional as F

from .gqa_prefill import compile_gqa_prefill, run_gqa_prefill

DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16
HEAD_DIM = 128
TEAM_ATOL = 2e-2
TEAM_RTOL = 2e-2
STRICT_ATOL = 2e-3
STRICT_RTOL = 1e-2


def make_inputs(q_heads, kv_heads, seq_len, seed):
    torch.manual_seed(seed)
    q = torch.randn(1, q_heads, seq_len, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    k = torch.randn(1, kv_heads, seq_len, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    v = torch.randn_like(k)
    return q, k, v


def repeat_for_oracle(k, v, q_heads):
    group = q_heads // k.shape[1]
    if group == 1:
        return k, v
    return k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)


def sdpa_oracle(q, k, v, causal, sm_scale):
    k_ref, v_ref = repeat_for_oracle(k, v, q.shape[1])
    return F.scaled_dot_product_attention(
        q,
        k_ref,
        v_ref,
        is_causal=causal,
        scale=sm_scale,
    )


def metrics(actual, reference):
    actual_f32 = actual.float()
    reference_f32 = reference.float()
    diff = (actual_f32 - reference_f32).abs()
    rel = diff / reference_f32.abs().clamp_min(1e-12)
    team_pass = bool(
        (diff.max().item() <= TEAM_ATOL)
        or (rel.max().item() <= TEAM_RTOL)
    )
    try:
        torch.testing.assert_close(
            actual_f32,
            reference_f32,
            atol=STRICT_ATOL,
            rtol=STRICT_RTOL,
        )
        strict_result = f"PASS atol={STRICT_ATOL:g}, rtol={STRICT_RTOL:g}"
    except AssertionError as exc:
        strict_result = f"FAIL {str(exc).splitlines()[0]}"
    return {
        "max_abs": diff.max().item(),
        "max_rel": rel.max().item(),
        "has_nan": bool(torch.isnan(actual_f32).any().item()),
        "has_inf": bool(torch.isinf(actual_f32).any().item()),
        "team_pass": team_pass,
        "strict_result": strict_result,
    }


def print_tensor_state(label, q, k, v, output):
    print(f"  {label}")
    print(f"    q ptr={q.data_ptr()} stride={tuple(q.stride())}")
    print(f"    k ptr={k.data_ptr()} stride={tuple(k.stride())}")
    print(f"    v ptr={v.data_ptr()} stride={tuple(v.stride())}")
    print(f"    out shape={tuple(output.shape)} stride={tuple(output.stride())} dtype={output.dtype} device={output.device}")


def check_case(label, q_heads, kv_heads, seq_len, causal, sm_scale, seed):
    q, k, v = make_inputs(q_heads, kv_heads, seq_len, seed)
    before_ptrs = (q.data_ptr(), k.data_ptr(), v.data_ptr())
    before_storages = (
        q.untyped_storage().data_ptr(),
        k.untyped_storage().data_ptr(),
        v.untyped_storage().data_ptr(),
    )
    scale = HEAD_DIM**-0.5 if sm_scale is None else sm_scale
    output = run_gqa_prefill(q, k, v, causal=causal, sm_scale=sm_scale)
    torch.cuda.synchronize()
    reference = sdpa_oracle(q, k, v, causal, scale)
    result = metrics(output, reference)
    after_ptrs = (q.data_ptr(), k.data_ptr(), v.data_ptr())
    after_storages = (
        q.untyped_storage().data_ptr(),
        k.untyped_storage().data_ptr(),
        v.untyped_storage().data_ptr(),
    )

    print(f"\ncase={label}")
    print(
        f"  B=1 Hq={q_heads} Hkv={kv_heads} S={seq_len} D={HEAD_DIM} "
        f"causal={causal} sm_scale={sm_scale}"
    )
    print_tensor_state("input/output state", q, k, v, output)
    print(f"    input ptr unchanged={before_ptrs == after_ptrs}")
    print(f"    input storage unchanged={before_storages == after_storages}")
    print(
        f"    max_abs={result['max_abs']:.6e} max_rel={result['max_rel']:.6e} "
        f"NaN={result['has_nan']} Inf={result['has_inf']}"
    )
    print(f"    team PASS={result['team_pass']} (abs<=2e-2 OR rel<=2e-2)")
    print(f"    strict assert_close: {result['strict_result']}")
    ok = (
        before_ptrs == after_ptrs
        and before_storages == after_storages
        and output.shape == q.shape
        and output.dtype == q.dtype
        and output.device == q.device
        and output.is_contiguous()
        and result["team_pass"]
        and result["strict_result"].startswith("PASS")
        and not result["has_nan"]
        and not result["has_inf"]
    )
    print(f"  [{'PASS' if ok else 'FAIL'}]")
    return ok


def run_compile(seed):
    q, k, v = make_inputs(8, 8, 128, seed)
    _run = compile_gqa_prefill(q, k, v, causal=True, sm_scale=None)
    print("[PASS] compile-only")
    return True


def run_mha(seed):
    ok = True
    for causal in (False, True):
        ok = ok and check_case(
            f"mha128 causal={causal} scale=None",
            8,
            8,
            128,
            causal,
            None,
            seed,
        )
        ok = ok and check_case(
            f"mha128 causal={causal} scale=explicit",
            8,
            8,
            128,
            causal,
            HEAD_DIM**-0.5,
            seed + 1,
        )
        if not ok:
            return False
    print("[PASS] mha-small")
    return True


def run_gqa(seed):
    ok = True
    for causal in (False, True):
        ok = ok and check_case(
            f"gqa64_8_s128 causal={causal} scale=None",
            64,
            8,
            128,
            causal,
            None,
            seed,
        )
        ok = ok and check_case(
            f"gqa64_8_s128 causal={causal} scale=explicit",
            64,
            8,
            128,
            causal,
            HEAD_DIM**-0.5,
            seed + 1,
        )
        if not ok:
            return False
    print("[PASS] gqa-small")
    return True


def run_tail(seed):
    ok = check_case(
        "gqa64_8_s130 causal tail",
        64,
        8,
        130,
        True,
        None,
        seed,
    )
    if ok:
        print("[PASS] tail")
    return ok


def run_medium(seed):
    ok = True
    for seq_len in (1024, 8192):
        ok = ok and check_case(
            f"gqa64_8_s{seq_len} causal",
            64,
            8,
            seq_len,
            True,
            None,
            seed + seq_len,
        )
        if not ok:
            return False
    print("[PASS] medium")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=["compile", "mha-small", "gqa-small", "tail", "medium"],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()

    if args.suite == "compile":
        ok = run_compile(args.seed)
    elif args.suite == "mha-small":
        ok = run_mha(args.seed)
    elif args.suite == "gqa-small":
        ok = run_gqa(args.seed)
    elif args.suite == "tail":
        ok = run_tail(args.seed)
    else:
        ok = run_medium(args.seed)
    print(f"========== suite {args.suite}: {'PASS' if ok else 'FAIL'} ==========")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
