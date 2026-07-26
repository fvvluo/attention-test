"""Speed test for FlashAttention (CuTe) forward: 128K prefill + decode.

Usage (from repo root, after: pip install -e "flash_attn/cute"):
  python bench_fa4_128k.py
"""

import torch

from flash_attn.cute import flash_attn_func


def _time_cuda(fn, warmup: int = 1, repeats: int = 10) -> float:
    """Return mean latency in ms over `repeats` runs after `warmup`."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats


def speed_test_prefill_128k(
    batch_size: int = 1,
    nheads: int = 32,
    nheads_kv: int = 8,
    headdim: int = 128,
    dtype=torch.bfloat16,
    causal: bool = True,
    warmup: int = 1,
    repeats: int = 10,
):
    """Context length 128K — prefill (seqlen_q == seqlen_k)."""
    device = "cuda"
    seqlen = 128 * 1024

    q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen, nheads_kv, headdim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen, nheads_kv, headdim, device=device, dtype=dtype)

    def run():
        return flash_attn_func(q, k, v, causal=causal)

    ms = _time_cuda(run, warmup=warmup, repeats=repeats)

    # causal: ~ half the GEMM FLOPs of full attention
    flops = 4 * batch_size * nheads * seqlen * seqlen * headdim
    if causal:
        flops //= 2
    tflops = flops / (ms / 1e3) / 1e12

    print(
        f"[prefill] seqlen={seqlen} batch={batch_size} "
        f"h={nheads}/{nheads_kv} d={headdim} causal={causal} "
        f"→ {ms:.3f} ms/iter ({repeats} runs, {warmup} warmup) | {tflops:.1f} TFLOPS"
    )
    return ms


def speed_test_decode_128k(
    batch_size: int = 1,
    nheads: int = 32,
    nheads_kv: int = 8,
    headdim: int = 128,
    dtype=torch.bfloat16,
    causal: bool = True,
    warmup: int = 1,
    repeats: int = 10,
):
    """Context length 128K — decode one token (seqlen_q == 1)."""
    device = "cuda"
    seqlen_k = 128 * 1024
    seqlen_q = 1

    q = torch.randn(batch_size, seqlen_q, nheads, headdim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen_k, nheads_kv, headdim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, nheads_kv, headdim, device=device, dtype=dtype)

    def run():
        return flash_attn_func(q, k, v, causal=causal)

    ms = _time_cuda(run, warmup=warmup, repeats=repeats)

    # Decode is usually bandwidth-bound: read K/V (+ Q) write O
    bytes_moved = (
        q.numel() + k.numel() + v.numel() + batch_size * seqlen_q * nheads * headdim
    ) * dtype.itemsize
    gbps = bytes_moved / (ms / 1e3) / 1e9

    print(
        f"[decode ] seqlen_q={seqlen_q} seqlen_k={seqlen_k} batch={batch_size} "
        f"h={nheads}/{nheads_kv} d={headdim} causal={causal} "
        f"→ {ms:.3f} ms/iter ({repeats} runs, {warmup} warmup) | {gbps:.1f} GB/s"
    )
    return ms


if __name__ == "__main__":
    torch.manual_seed(0)
    assert torch.cuda.is_available()

    # First call compiles the JIT kernel; warmup inside each fn covers that + 1 extra.
    speed_test_prefill_128k()
    speed_test_decode_128k()