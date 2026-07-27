# Prefill (128k) + Decode demo for the simplified FlashAttention-3 (fwd-only, SM90/H20).
#
# Usage:
#   python prefill_decode_demo.py                      # correctness checks + 128k perf
#   python prefill_decode_demo.py --skip-checks        # perf only
#   python prefill_decode_demo.py --seqlen 65536 --heads 16 --hdim 128 --decode-steps 32

import argparse
import math

import torch

from flash_attn_interface import (
    flash_attn_func,
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
)


def attention_ref(q, k, v, causal=False):
    """Naive fp32 reference. q: (b, sq, h, d); k/v: (b, sk, h_k, d). GQA-aware."""
    h, h_k = q.shape[2], k.shape[2]
    g = h // h_k
    if g > 1:
        k = k.repeat_interleave(g, dim=2)
        v = v.repeat_interleave(g, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) / math.sqrt(q.shape[-1])
    if causal:
        sq, sk = q.shape[1], k.shape[1]
        # bottom-right aligned causal mask
        i = torch.arange(sk, device=q.device)[None, :]
        j = torch.arange(sq, device=q.device)[:, None] + (sk - sq)
        mask = i > j
        scores.masked_fill_(mask[None, None], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", probs, v.float())


def check(name, out, ref, dtype):
    tol = 2e-2 if dtype == torch.bfloat16 else 1e-2
    err = (out.float() - ref).abs().max().item()
    denom = ref.abs().max().item()
    ok = err <= tol
    print(f"  [{ 'PASS' if ok else 'FAIL'}] {name}: max abs err = {err:.3e} (ref max {denom:.2e})")
    return ok


def run_correctness(dtype):
    print("== Correctness (small sizes, vs fp32 reference) ==")
    torch.manual_seed(0)
    dev = "cuda"
    ok = True

    # 1) Prefill: varlen, causal
    seqlens = [1536, 2048]
    cu = torch.tensor([0] + list(torch.tensor(seqlens).cumsum(0)), dtype=torch.int32, device=dev)
    total, h, d = int(sum(seqlens)), 8, 128
    q = torch.randn(total, h, d, dtype=dtype, device=dev)
    k = torch.randn(total, h, d, dtype=dtype, device=dev)
    v = torch.randn(total, h, d, dtype=dtype, device=dev)
    out = flash_attn_varlen_func(q, k, v, cu, cu, max(seqlens), max(seqlens), causal=True)
    refs = []
    for i, s in enumerate(seqlens):
        sl = slice(int(cu[i]), int(cu[i + 1]))
        refs.append(attention_ref(q[sl][None], k[sl][None], v[sl][None], causal=True)[0])
    ok &= check("prefill varlen causal", out, torch.cat(refs), dtype)

    # 2) Decode: contiguous KV cache, q_len=1 and q_len=3
    b, kv_len, h, h_k, d = 2, 4096, 8, 2, 128
    k_cache = torch.randn(b, kv_len, h_k, d, dtype=dtype, device=dev)
    v_cache = torch.randn(b, kv_len, h_k, d, dtype=dtype, device=dev)
    for q_len in (1, 3):
        q = torch.randn(b, q_len, h, d, dtype=dtype, device=dev)
        out = flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens=kv_len, causal=True)
        ref = attention_ref(q, k_cache, v_cache, causal=True)
        ok &= check(f"decode contiguous cache q_len={q_len} (GQA {h}/{h_k})", out, ref, dtype)

    # 3) Decode: paged KV cache
    page_size = 64
    num_blocks_per_seq = kv_len // page_size
    k_paged = k_cache.reshape(b * num_blocks_per_seq, page_size, h_k, d).contiguous()
    v_paged = v_cache.reshape(b * num_blocks_per_seq, page_size, h_k, d).contiguous()
    page_table = torch.arange(b * num_blocks_per_seq, dtype=torch.int32, device=dev).reshape(b, -1)
    q = torch.randn(b, 1, h, d, dtype=dtype, device=dev)
    out = flash_attn_with_kvcache(
        q, k_paged, v_paged, cache_seqlens=kv_len, page_table=page_table, causal=True
    )
    ref = attention_ref(q, k_cache, v_cache, causal=True)
    ok &= check("decode paged cache q_len=1", out, ref, dtype)

    return ok


def bench(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def run_perf(args, dtype):
    s, b, h, d = args.seqlen, args.batch, args.heads, args.hdim
    h_k = args.kv_heads or h
    dev = "cuda"
    print(f"== Perf: batch={b} heads={h} kv_heads={h_k} hdim={d} dtype={dtype} ==")

    # ---- Prefill ----
    q = torch.randn(b, s, h, d, dtype=dtype, device=dev)
    k = torch.randn(b, s, h_k, d, dtype=dtype, device=dev)
    v = torch.randn(b, s, h_k, d, dtype=dtype, device=dev)
    ms = bench(lambda: flash_attn_func(q, k, v, causal=True), iters=args.iters)
    flops = 2 * 2 * s * s * h * d * b / 2  # QK^T + PV, causal halves it
    print(
        f"  Prefill  {s:>7} tokens: {ms:8.2f} ms  |  {flops / ms / 1e9:7.1f} TFLOP/s"
    )
    del q, k, v
    torch.cuda.empty_cache()

    # ---- Decode ----
    k_cache = torch.randn(b, s, h_k, d, dtype=dtype, device=dev)
    v_cache = torch.randn(b, s, h_k, d, dtype=dtype, device=dev)
    q = torch.randn(b, 1, h, d, dtype=dtype, device=dev)
    cache_seqlens = torch.full((b,), s, dtype=torch.int32, device=dev)
    kv_bytes = 2 * s * h_k * d * b * dtype.itemsize
    # num_splits matters a lot for decode: sweep and report the best
    # (0 = built-in heuristic).
    best = None
    for ns in (0, 2, 4, 8):
        ms = bench(
            lambda: flash_attn_with_kvcache(
                q, k_cache, v_cache, cache_seqlens=cache_seqlens, causal=True, num_splits=ns
            ),
            warmup=5,
            iters=args.decode_steps,
        )
        print(
            f"  Decode   {s:>7} ctx, 1 token, num_splits={ns}: {ms:8.3f} ms/token  |  "
            f"{kv_bytes / ms / 1e6:7.0f} GB/s eff. BW"
        )
        if best is None or ms < best[1]:
            best = (ns, ms)
    print(f"  Decode best: num_splits={best[0]}  {best[1]:.3f} ms/token")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seqlen", type=int, default=128 * 1024)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--kv-heads", type=int, default=None, help="default: same as --heads (MHA)")
    p.add_argument("--hdim", type=int, default=128, choices=[64, 128, 256])
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--decode-steps", type=int, default=32)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--skip-checks", action="store_true")
    args = p.parse_args()

    assert torch.cuda.get_device_capability()[0] == 9, "This build requires an SM90 GPU (Hopper)"
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    ok = True
    if not args.skip_checks:
        ok = run_correctness(dtype)
    run_perf(args, dtype)
    if not ok:
        raise SystemExit("correctness checks FAILED")
    print("Done.")


if __name__ == "__main__":
    main()
