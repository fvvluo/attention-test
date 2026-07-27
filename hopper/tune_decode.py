# num_splits sweep tuner for decode (flash_attn_with_kvcache) on long KV.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1 python tune_decode.py                          # MHA 16/16 @128k
#   CUDA_VISIBLE_DEVICES=1 python tune_decode.py --kv-heads 2             # GQA 16/2
#   CUDA_VISIBLE_DEVICES=1 python tune_decode.py --batch 4 --seqlen 65536

import argparse

import torch

from flash_attn_interface import flash_attn_with_kvcache


def bench(fn, warmup=5, iters=32):
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
    return start.elapsed_time(end) / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seqlen", type=int, default=128 * 1024)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--kv-heads", type=int, default=None)
    p.add_argument("--hdim", type=int, default=128, choices=[64, 128, 256])
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--q-len", type=int, default=1)
    p.add_argument("--iters", type=int, default=32)
    p.add_argument(
        "--splits",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128],
        help="num_splits values to try (0 = built-in heuristic)",
    )
    args = p.parse_args()

    torch.manual_seed(0)
    dev = "cuda"
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    s, b, h, d, q_len = args.seqlen, args.batch, args.heads, args.hdim, args.q_len
    h_k = args.kv_heads or h
    kv_bytes = 2 * s * h_k * d * b * dtype.itemsize

    print(
        f"decode tune: seqlen={s} batch={b} heads={h} kv_heads={h_k} "
        f"hdim={d} q_len={q_len} {args.dtype} | KV read/step = {kv_bytes / 1e9:.2f} GB"
    )

    k_cache = torch.randn(b, s, h_k, d, dtype=dtype, device=dev)
    v_cache = torch.randn(b, s, h_k, d, dtype=dtype, device=dev)
    q = torch.randn(b, q_len, h, d, dtype=dtype, device=dev)
    cache_seqlens = torch.full((b,), s, dtype=torch.int32, device=dev)

    results = []
    for ns in args.splits:
        ms = bench(
            lambda: flash_attn_with_kvcache(
                q, k_cache, v_cache, cache_seqlens=cache_seqlens, causal=True, num_splits=ns
            ),
            iters=args.iters,
        )
        bw = kv_bytes / ms / 1e6
        results.append((ms, ns, bw))
        print(f"  num_splits={ns:>4}: {ms:8.3f} ms/token | {bw:7.0f} GB/s")

    results.sort()
    print(f"\nBest: num_splits={results[0][1]}  {results[0][0]:.3f} ms/token "
          f"({results[0][2]:.0f} GB/s)")
    heuristic = next((r for r in results if r[1] == 0), None)
    if heuristic and heuristic[1] != results[0][1]:
        print(f"Heuristic (num_splits=0): {heuristic[0]:.3f} ms/token "
              f"-> tuned is {heuristic[0] / results[0][0]:.2f}x faster")


if __name__ == "__main__":
    main()
