"""Split-K flash-decoding kernel (Triton) for q_len == 1 decode on SM90.

Optimized derivative of fast_decode.py. Same two-kernel split-K structure, but
faster reduction + retuned tiles. Measured ~3.24-3.30 TB/s full path on H20 for
1x64x8x131072x128 bf16 vs. ~3.19 TB/s for fast_decode.py.

Why a dedicated decode path
---------------------------
The generic CuTe FMHA kernel treats decode like a small prefill: every q head
re-reads the whole KV cache (8x redundant traffic for a GQA group of 8) and the
grid only has batch*q_heads CTAs, which underfills the GPU.  This file
implements a flash-decoding style decode path:

  1. GQA packing - one program handles an entire KV-head's query group (the
     MMA M dimension covers the group's q heads), so K/V are read exactly once.
  2. Split-K - the KV sequence is chopped into chunks processed by different
     CTAs; partial (acc, m, l) triples are combined by a second lightweight
     kernel, so the grid fills all SMs (wave-aligned to avoid quantization).

What changed vs fast_decode.py
------------------------------
  * Reduce kernel rewrite: the original combined partials with a serial
    online-softmax rescale chain (each split depends on the previous). Here we
    load the whole (splits x rows) m/l planes at once, take the global max over
    splits, then do a single independent weighted sum of the partial
    accumulators. Independent loads pipeline; the reduce dropped 6.5us -> 2.6us.
  * BMR: for GQA the split kernel packs group<=BM rows into a padded WGMMA M
    tile; the reduce only touches pow2(group) rows and skips the padding rows
    of the partial accumulator (halves its traffic/work).
  * Retuned tiles: BM=16 (better mma.sync M utilization than BM=8), BN=64,
    num_stages=4. Split-only reaches ~3.33 TB/s (161us), essentially the H20
    DRAM read wall for this access pattern (a naive TMA read+reduce is slower).
  * Host-side: cached SM count, single pm/pl allocation.

Design notes (H20, measured on 1x64x8x131072x128 bf16)
------------------------------------------------------
  * Small M (mma.sync path) is essential: an M=64 WGMMA packing wastes 8x MMA on
    padding rows for GQA group=8; on H20's weak tensor cores (148 TFLOPS bf16)
    that alone caps decode at ~1.8-2.3 TB/s. Verified by wiring the CuTe DSL
    M=64 decode kernel: it only reached 1.78 TB/s here.
  * TMA descriptor loads beat plain vectorized loads; EVEN (kv_len % BN == 0)
    drops all masks.
  * splits = floor(2*SMs / (b*hk*mch)) aligns the grid to whole CTA waves
    (152 CTAs = 2x78 SMs on H20); a misaligned count costs up to 30%.
  * A fused single-kernel reduction (atomic last-CTA ticket, see fast_decode3.py)
    was retried and is still ~7% slower: the last CTA does the whole combine,
    lengthening the tail while other SMs idle, and every CTA pays an atomic.
    So the two-kernel structure is kept. The residual ~5us (reduce 2.6us +
    inter-kernel dispatch gap ~2.5us) is the only overhead over the read wall.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


_LOG2E = 1.4426950408889634
_LN2 = 0.6931471805599453

try:  # device-side TMA descriptors need a scratch allocator
    triton.set_allocator(
        lambda size, alignment, stream: torch.empty(
            size, dtype=torch.int8, device="cuda"
        )
    )
    _HAS_TMA = hasattr(tl, "make_tensor_descriptor")
except Exception:  # pragma: no cover - older triton
    _HAS_TMA = False


@triton.jit
def _decode_split_kernel(
    Q, K, V, Pacc, Pm, Pl,
    qk_scale,                    # softmax_scale * log2(e)
    kv_len, chunk_len,
    G, KVH, MCH, NSPLITS,
    stride_qb, stride_qh,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid_split = tl.program_id(0)
    pid_bk = tl.program_id(1)

    mc = pid_bk % MCH
    bkvh = pid_bk // MCH
    kvh = bkvh % KVH
    b = bkvh // KVH

    offs_m = mc * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    m_valid = offs_m < G

    qh = kvh * G + offs_m
    q = tl.load(
        Q + b * stride_qb + qh[:, None] * stride_qh + offs_d[None, :],
        mask=m_valid[:, None], other=0.0,
    )

    start = pid_split * chunk_len
    end = tl.minimum(start + chunk_len, kv_len)

    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)

    kbase = K + b * stride_kb + kvh * stride_kh
    vbase = V + b * stride_vb + kvh * stride_vh

    for n in range(start, end, BN):
        if EVEN:
            # kv_len % BN == 0: every tile is full, skip masks entirely
            k = tl.load(kbase + (n + tl.arange(0, BN))[:, None] * stride_kn + offs_d[None, :])
            s = tl.dot(q, tl.trans(k)) * qk_scale
        else:
            offs_n = n + tl.arange(0, BN)
            nmask = offs_n < end
            k = tl.load(
                kbase + offs_n[:, None] * stride_kn + offs_d[None, :],
                mask=nmask[:, None], other=0.0,
            )
            s = tl.dot(q, tl.trans(k)) * qk_scale
            s = tl.where(nmask[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        if EVEN:
            v = tl.load(vbase + (n + tl.arange(0, BN))[:, None] * stride_vn + offs_d[None, :])
        else:
            v = tl.load(
                vbase + offs_n[:, None] * stride_vn + offs_d[None, :],
                mask=nmask[:, None], other=0.0,
            )
        acc = acc * alpha[:, None] + tl.dot(p.to(V.dtype.element_ty), v)
        m_i = m_new

    pbase = pid_bk * NSPLITS + pid_split
    offs_pm = pbase * BM + tl.arange(0, BM)
    tl.store(Pm + offs_pm, m_i)
    tl.store(Pl + offs_pm, l_i)
    tl.store(Pacc + offs_pm[:, None] * D + offs_d[None, :], acc)


@triton.jit
def _decode_split_kernel_tma(
    Q, K, V, Pacc, Pm, Pl,
    qk_scale,
    kv_len, chunk_len,
    G, KVH, MCH, NSPLITS,
    stride_qb, stride_qh,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr, EVEN: tl.constexpr,
):
    pid_split = tl.program_id(0)
    pid_bk = tl.program_id(1)

    mc = pid_bk % MCH
    bkvh = pid_bk // MCH
    kvh = bkvh % KVH
    b = bkvh // KVH

    offs_m = mc * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    m_valid = offs_m < G

    qh = kvh * G + offs_m
    q = tl.load(
        Q + b * stride_qb + qh[:, None] * stride_qh + offs_d[None, :],
        mask=m_valid[:, None], other=0.0,
    )

    start = pid_split * chunk_len
    end = tl.minimum(start + chunk_len, kv_len)

    # TMA clamps box reads at the descriptor shape (zero-fill); the -inf where
    # on the logits (non-EVEN path) keeps the softmax correct on the tail.
    k_desc = tl.make_tensor_descriptor(
        K + b * stride_kb + kvh * stride_kh,
        shape=[kv_len, D], strides=[stride_kn, 1], block_shape=[BN, D],
    )
    v_desc = tl.make_tensor_descriptor(
        V + b * stride_vb + kvh * stride_vh,
        shape=[kv_len, D], strides=[stride_vn, 1], block_shape=[BN, D],
    )

    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)

    for n in range(start, end, BN):
        k = k_desc.load([n, 0])
        s = tl.dot(q, tl.trans(k)) * qk_scale
        if not EVEN:
            nmask = (n + tl.arange(0, BN)) < end
            s = tl.where(nmask[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = v_desc.load([n, 0])
        acc = acc * alpha[:, None] + tl.dot(p.to(V.dtype.element_ty), v)
        m_i = m_new

    pbase = pid_bk * NSPLITS + pid_split
    offs_pm = pbase * BM + tl.arange(0, BM)
    tl.store(Pm + offs_pm, m_i)
    tl.store(Pl + offs_pm, l_i)
    tl.store(Pacc + offs_pm[:, None] * D + offs_d[None, :], acc)


@triton.jit
def _decode_reduce_kernel(
    Pacc, Pm, Pl, Out, Lse,
    G, KVH, MCH, NSPLITS,
    stride_ob, stride_oh,
    stride_lb, stride_lh,
    PBM: tl.constexpr, BMR: tl.constexpr, D: tl.constexpr, BDR: tl.constexpr,
    HAS_LSE: tl.constexpr, NS: tl.constexpr,
):
    # PBM = physical row stride of the partial tensors (matches the split
    # kernel's BM). BMR = logical rows actually reduced; for GQA the split
    # kernel packs group<=BM rows into a padded M tile, so the reduce only
    # needs to touch BMR = pow2(group) rows and can skip reading the padding
    # rows of the partial accumulator entirely (halves reduce traffic/work).
    pid_bk = tl.program_id(0)
    pid_d = tl.program_id(1)

    mc = pid_bk % MCH
    bkvh = pid_bk // MCH
    kvh = bkvh % KVH
    b = bkvh // KVH

    offs_m = mc * PBM + tl.arange(0, BMR)
    offs_d = pid_d * BDR + tl.arange(0, BDR)
    m_valid = offs_m < (mc * PBM + G)

    # Load the whole (splits x BMR) m/l planes at once. They are tiny and this
    # removes the serial online-softmax dependency chain: take the global max
    # over splits first, then do an independent weighted sum of the partial
    # accumulators. Independent acc loads pipeline far better than the original
    # running-rescale loop, which dominated the reduce cost.
    offs_s = tl.arange(0, NS)
    valid_s = offs_s < NSPLITS
    base = (pid_bk * NSPLITS + offs_s)[:, None] * PBM + offs_m[None, :]
    m_all = tl.load(Pm + base, mask=valid_s[:, None], other=float("-inf"))  # [NS, BMR]
    l_all = tl.load(Pl + base, mask=valid_s[:, None], other=0.0)            # [NS, BMR]

    m_run = tl.max(m_all, 0)                                                # [BMR]
    w = tl.where(valid_s[:, None], tl.math.exp2(m_all - m_run[None, :]), 0.0)  # [NS, BMR]
    l_run = tl.sum(w * l_all, 0)                                            # [BMR]

    a_ptr = (Pacc
             + (pid_bk * NSPLITS + offs_s)[:, None, None] * (PBM * D)
             + offs_m[None, :, None] * D
             + offs_d[None, None, :])
    a_all = tl.load(a_ptr, mask=valid_s[:, None, None], other=0.0)          # [NS, BMR, BDR]
    acc = tl.sum(a_all * w[:, :, None], 0)                                  # [BMR, BDR]

    out = acc / l_run[:, None]
    oh = kvh * G + tl.arange(0, BMR)
    tl.store(
        Out + b * stride_ob + oh[:, None] * stride_oh + offs_d[None, :],
        out.to(Out.dtype.element_ty),
        mask=m_valid[:, None],
    )
    if HAS_LSE:
        # lse 只由第一个 d-block 写出,避免重复
        if pid_d == 0:
            lse = (m_run + tl.math.log2(l_run)) * 0.6931471805599453
            tl.store(Lse + b * stride_lb + oh * stride_lh, lse, mask=m_valid)


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


_SM_CACHE: dict = {}


def _sm_count(device: torch.device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    n = _SM_CACHE.get(idx)
    if n is None:
        n = torch.cuda.get_device_properties(idx).multi_processor_count
        _SM_CACHE[idx] = n
    return n


def flash_attention_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    layout: str = "bhsd",
    softmax_scale: float | None = None,
    return_lse: bool = False,
    BN: int = 64,
    num_warps: int = 4,
    num_stages: int = 4,
    ctas_per_sm: int = 2,
    splits: int | None = None,
    BM: int = 16,
    use_tma: bool | None = None,
):
    """Decode-only attention: q_len == 1, non-causal, q/k/v rank-4 contiguous.

    q: (b, hq, 1, d) for bhsd or (b, 1, hq, d) for bshd.
    k/v: (b, hk, sk, d) for bhsd or (b, sk, hk, d) for bshd.
    Returns out (same shape as q); with return_lse also (b, hq, 1) fp32 lse.
    """
    d = q.shape[-1]
    if layout == "bshd":
        b, _, hq, _ = q.shape
        sk, hk = k.shape[1], k.shape[2]
        stride_qb, stride_qh = q.stride(0), q.stride(2)
        stride_kb, stride_kh, stride_kn = k.stride(0), k.stride(2), k.stride(1)
        stride_vb, stride_vh, stride_vn = v.stride(0), v.stride(2), v.stride(1)
    else:
        b, hq = q.shape[0], q.shape[1]
        hk, sk = k.shape[1], k.shape[2]
        stride_qb, stride_qh = q.stride(0), q.stride(1)
        stride_kb, stride_kh, stride_kn = k.stride(0), k.stride(1), k.stride(2)
        stride_vb, stride_vh, stride_vn = v.stride(0), v.stride(1), v.stride(2)

    group = hq // hk
    mch = _cdiv(group, BM)

    # Pick the split count so the grid fills whole waves of CTAs (H20: 78 SMs,
    # ~1 CTA/SM resident for the default config, so ~2 waves of CTAs) while
    # every split still gets a multiple-of-BN chunk. The SM count is cached:
    # get_device_properties is a surprisingly expensive host call and this
    # runs on the decode critical path for every single token.
    num_sms = _sm_count(q.device)
    bk = b * hk * mch
    if splits is None:
        target = num_sms * ctas_per_sm
        splits = max(1, min(_cdiv(sk, BN), target // bk, 128))
    chunk = _cdiv(_cdiv(sk, splits), BN) * BN
    splits = _cdiv(sk, chunk)

    scale = float(softmax_scale) if softmax_scale is not None else 1.0 / math.sqrt(d)
    qk_scale = scale * _LOG2E

    device = q.device
    pacc = torch.empty((bk, splits, BM, d), device=device, dtype=torch.float32)
    # pm and pl share one allocation to halve host-side allocator overhead on
    # the decode critical path; the kernels take independent base pointers.
    pml = torch.empty((2, bk, splits, BM), device=device, dtype=torch.float32)
    pm = pml[0]
    pl = pml[1]

    out = torch.empty_like(q)
    if layout == "bshd":
        stride_ob, stride_oh = out.stride(0), out.stride(2)
    else:
        stride_ob, stride_oh = out.stride(0), out.stride(1)

    if return_lse:
        lse = torch.empty((b, hq), device=device, dtype=torch.float32)
        stride_lb, stride_lh = lse.stride(0), lse.stride(1)
    else:
        lse = pacc  # unused placeholder
        stride_lb = stride_lh = 0

    even = (sk % BN) == 0
    if use_tma is None:
        use_tma = _HAS_TMA

    split_kernel = _decode_split_kernel_tma if (use_tma and _HAS_TMA) else _decode_split_kernel
    split_kernel[(splits, bk)](
        q, k, v, pacc, pm, pl,
        qk_scale, sk, chunk,
        group, hk, mch, splits,
        stride_qb, stride_qh,
        stride_kb, stride_kh, stride_kn,
        stride_vb, stride_vh, stride_vn,
        BM=BM, BN=BN, D=d, EVEN=even,
        num_warps=num_warps, num_stages=num_stages,
    )

    # Smaller BDR spreads the reduction over more CTAs; the reduce kernel is
    # launch/occupancy bound (grid = bk * d/BDR), not bandwidth bound, so more
    # CTAs hide launch latency. NS is the padded (pow2) split count so the
    # whole m/l/acc planes load in one shot without a serial loop.
    # BMR: the split kernel packs group<=BM rows into a padded M tile; the
    # reduce only needs pow2(group) rows and skips the padding rows entirely.
    ns_pad = 1 << (splits - 1).bit_length()
    if mch == 1:
        bmr = 1 << (group - 1).bit_length()
        bmr = min(bmr, BM)
    else:
        bmr = BM  # multi-tile M: fall back to full physical rows
    BDR = 32 if d >= 32 else d
    if bk * (d // BDR) < num_sms and d >= 32:
        BDR = 16
    _decode_reduce_kernel[(bk, d // BDR)](
        pacc, pm, pl, out, lse,
        group, hk, mch, splits,
        stride_ob, stride_oh,
        stride_lb, stride_lh,
        PBM=BM, BMR=bmr, D=d, BDR=BDR, HAS_LSE=return_lse, NS=ns_pad,
        num_warps=4, num_stages=2,
    )

    if not return_lse:
        return out
    # Match the main path: lse is always returned as (b, hq, sq=1).
    return out, lse.view(b, hq, 1)
