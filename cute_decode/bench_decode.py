"""Standalone benchmark + correctness harness for the CuTe DSL decode kernel.

Fixed target shape: 1x64x8x131072x128 bf16. Reports effective KV bandwidth using
the same formula as the official bench_attention.py:
    bytes = elem_size * batch * head_dim * (2*q_heads*q_len + 2*kv_heads*kv_len)
Correctness checked against an fp32 grouped SDPA reference (max_abs <= 2e-2).
"""

import argparse
import statistics

import torch

import flash_decode_transposed_wgmma as da


def grouped_reference(q, k, v, sm_scale):
    b, hq, ql, d = q.shape
    hk = k.shape[1]
    g = hq // hk
    qg = q.float().view(b, hk, g, ql, d)
    scores = torch.einsum("bhgqd,bhkd->bhgqk", qg, k.float()) * sm_scale
    p = torch.softmax(scores, dim=-1)
    o = torch.einsum("bhgqk,bhkd->bhgqd", p, v.float())
    return o.reshape(b, hq, ql, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--q-heads", type=int, default=64)
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--kv-len", type=int, default=131072)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--num-splits", type=int, default=39)
    ap.add_argument("--block-n", type=int, default=128)
    ap.add_argument("--stages", type=int, default=3)
    ap.add_argument("--producers", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--regs-producer", type=int, default=0)
    ap.add_argument("--regs-consumer", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.cuda.set_device(args.gpu)
    dev = torch.device(f"cuda:{args.gpu}")
    torch.manual_seed(args.seed)
    sm_scale = 1.0 / (args.head_dim ** 0.5)

    q = torch.randn(args.batch, args.q_heads, 1, args.head_dim, dtype=torch.bfloat16, device=dev)
    k = torch.randn(args.batch, args.kv_heads, args.kv_len, args.head_dim, dtype=torch.bfloat16, device=dev)
    v = torch.randn(args.batch, args.kv_heads, args.kv_len, args.head_dim, dtype=torch.bfloat16, device=dev)

    def call():
        return da.attention_decode(
            q, k, v, sm_scale=sm_scale, causal=False,
            num_splits=args.num_splits, block_n=args.block_n, num_stages=args.stages,
            num_producer_warps=args.producers, num_workers=args.num_workers,
            regs_producer=args.regs_producer, regs_consumer=args.regs_consumer,
        )

    o = call()
    torch.cuda.synchronize()
    ref = grouped_reference(q, k, v, sm_scale)
    err = (o.float() - ref).abs()
    max_abs = err.max().item()
    rel = (err / (ref.abs() + 1e-6)).max().item()
    ok = (max_abs <= 2e-2) or (rel <= 2e-2)
    print(f"correctness: max_abs={max_abs:.3e} max_rel={rel:.3e} -> {'PASS' if ok else 'FAIL'}")

    for _ in range(args.warmup):
        call()
    torch.cuda.synchronize()
    # Measure like the official bench_attention.py: N iters between two events,
    # NO per-iter sync (per-iter sync serializes CPU launch overhead into the
    # timing and understates a fast, memory-bound kernel).
    reps = 5
    per_rep = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(args.iters):
            call()
        e.record()
        e.synchronize()
        per_rep.append(s.elapsed_time(e) / args.iters)
    per_rep.sort()
    ms = statistics.median(per_rep)
    samples = per_rep

    elem = 2
    nbytes = elem * args.batch * args.head_dim * (
        2 * args.q_heads * 1 + 2 * args.kv_heads * args.kv_len
    )
    gbps = nbytes / (ms / 1000) / 1e9
    print(
        f"cfg splits={args.num_splits} block_n={args.block_n} stages={args.stages} "
        f"prod={args.producers} regs={args.regs_producer}/{args.regs_consumer} "
        f"workers={args.num_workers}"
    )
    print(f"latency: median={ms:.4f} ms  min={samples[0]:.4f} ms  -> {gbps:.1f} GB/s")


if __name__ == "__main__":
    main()
