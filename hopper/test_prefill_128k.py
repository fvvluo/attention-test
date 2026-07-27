# 128k prefill correctness verification for the simplified FlashAttention-3 (fwd-only, SM90).
#
# A full fp32 reference at 128k needs ~68 GiB per head for the score matrix, so we
# verify an exact fp32 reference on a sample of query rows (first / last / random),
# computed with online softmax over KV chunks (no approximation in the reference).
#
# Usage: CUDA_VISIBLE_DEVICES=1 python test_prefill_128k.py [--seqlen 131072] [--heads 16] [--hdim 128]

import argparse
import math

import torch

from flash_attn_interface import flash_attn_func


@torch.no_grad()
def ref_attention_rows(q_rows, row_idx, k, v, scale, kv_chunk=8192):
    """Exact fp32 causal attention for selected query rows.

    q_rows: (n, h, d) query rows (bf16/fp16), row_idx: (n,) absolute positions,
    k/v: (s, h, d) full KV. Returns (n, h, d) fp32.
    """
    n, h, d = q_rows.shape
    s = k.shape[0]
    dev = q_rows.device
    out = torch.zeros(n, h, d, dtype=torch.float32, device=dev)
    m = torch.full((n, h), float("-inf"), dtype=torch.float32, device=dev)
    l = torch.zeros(n, h, dtype=torch.float32, device=dev)
    col = row_idx[:, None]  # (n, 1)
    for start in range(0, s, kv_chunk):
        kc = k[start : start + kv_chunk].float()  # (cs, h, d)
        vc = v[start : start + kv_chunk].float()
        scores = torch.einsum("nhd,chd->nhc", q_rows.float(), kc) * scale  # (n, h, cs)
        keys = start + torch.arange(kc.shape[0], device=dev)[None, :]  # (1, cs)
        scores.masked_fill_((keys > col)[:, None, :], float("-inf"))  # causal
        m_new = torch.maximum(m, scores.amax(-1))
        # rows fully masked so far keep m_new = -inf; guard exp
        m_safe = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)
        alpha = torch.exp(m - m_safe)
        alpha = torch.where(torch.isinf(m), torch.zeros_like(alpha), alpha)
        p = torch.exp(scores - m_safe[..., None])
        l = l * alpha + p.sum(-1)
        out = out * alpha[..., None] + torch.einsum("nhc,chd->nhd", p, vc)
        m = m_safe
    return out / l.clamp_min(1e-30)[..., None]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seqlen", type=int, default=128 * 1024)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--kv-heads", type=int, default=None)
    p.add_argument("--hdim", type=int, default=128, choices=[64, 128, 256])
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--sample-rows", type=int, default=512)
    args = p.parse_args()

    torch.manual_seed(42)
    dev = "cuda"
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    s, h, d = args.seqlen, args.heads, args.hdim
    h_k = args.kv_heads or h
    print(f"Prefill correctness: seqlen={s} heads={h} kv_heads={h_k} hdim={d} dtype={args.dtype}")

    q = torch.randn(1, s, h, d, dtype=dtype, device=dev)
    k = torch.randn(1, s, h_k, d, dtype=dtype, device=dev)
    v = torch.randn(1, s, h_k, d, dtype=dtype, device=dev)

    out = flash_attn_func(q, k, v, causal=True)[0]  # (s, h, d)
    torch.cuda.synchronize()

    # sample rows: first 64, last 64, and random interior rows
    n_edge = min(64, s)
    n_rand = max(args.sample_rows - 2 * n_edge, 0)
    idx = torch.cat(
        [
            torch.arange(n_edge, device=dev),
            torch.randint(n_edge, s - n_edge, (n_rand,), device=dev) if n_rand else torch.tensor([], dtype=torch.long, device=dev),
            torch.arange(s - n_edge, s, device=dev),
        ]
    ).long()

    # GQA: expand kv heads for the reference
    if h_k != h:
        g = h // h_k
        k_ref = k[0].repeat_interleave(g, dim=1)
        v_ref = v[0].repeat_interleave(g, dim=1)
    else:
        k_ref, v_ref = k[0], v[0]

    ref = ref_attention_rows(q[0, idx], idx, k_ref, v_ref, scale=1.0 / math.sqrt(d))
    got = out[idx].float()

    err = (got - ref).abs()
    rel = err / ref.abs().clamp_min(1e-3)
    tol = 2e-2 if dtype == torch.bfloat16 else 1e-2
    max_err = err.max().item()
    mean_err = err.mean().item()
    frac_within = (err <= tol).float().mean().item()
    ok = max_err <= tol
    print(f"  sampled rows: {idx.numel()} (first {n_edge} + random {n_rand} + last {n_edge})")
    print(f"  max abs err : {max_err:.4e}  (tol {tol:.0e})")
    print(f"  mean abs err: {mean_err:.4e}")
    print(f"  max rel err : {rel.max().item():.4e}")
    print(f"  rows within tol: {frac_within * 100:.3f}%")
    print(f"  lse-free check passed, result: [{'PASS' if ok else 'FAIL'}]")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
