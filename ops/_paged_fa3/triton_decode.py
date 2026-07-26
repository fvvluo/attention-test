"""triton_decode.py - Paged flash-decoding attention in Triton for Hopper (H20).

This is the DECODE-first camp deliverable (Task 1 decode + Task 2 paged KV):

  * single (or short) query length Q, attending to a very long KV cache (up to 128k)
  * KV stored in PAGED layout: a block table maps logical positions -> physical
    KV blocks of fixed size PAGE_SIZE (like vLLM / PagedAttention)
  * split-KV (flash-decoding): each (batch, head, kv-split) is one program; each
    computes a partial (O, m, l); a second reduction kernel LSE-combines the splits
  * GQA: multiple query heads share one KV head
  * numerically-stable online softmax (running max / running sum)

Why Triton for decode on H20: decode is memory-bound (single query vs 128k KV),
so performance is gated by HBM bandwidth reading K/V, not by tensor-core throughput.
Triton generates efficient coalesced loads and reaches ~FA-level here, while being
verifiable/runnable immediately. The CuTe DSL port targets the same algorithm.

Reference: FlashAttention-2 flash-decoding; vLLM PagedAttention.
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Phase 1: per-split partial attention over paged KV
# ---------------------------------------------------------------------------
@triton.jit
def _paged_decode_split_kernel(
    Q_ptr,              # [B, Hq, D]           query (decode: q_len folded into B or =1)
    Kcache_ptr,         # [num_blocks, PAGE, Hkv, D]  paged key cache
    Vcache_ptr,         # [num_blocks, PAGE, Hkv, D]  paged value cache
    BlockTable_ptr,     # [B, max_blocks]      logical->physical block id
    SeqLen_ptr,         # [B]                  actual kv length per sequence
    Out_partial_ptr,    # [B, Hq, NUM_SPLITS, D]  partial outputs
    Lse_partial_ptr,    # [B, Hq, NUM_SPLITS]     partial log-sum-exp (m + log l)
    sm_scale,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kp, stride_kh, stride_kd,
    stride_vb, stride_vp, stride_vh, stride_vd,
    stride_bt_b, stride_bt_x,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    Hq: tl.constexpr, Hkv: tl.constexpr,
    PAGE: tl.constexpr, D: tl.constexpr,
    NUM_SPLITS: tl.constexpr, BLOCK_N: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)          # query head
    s = tl.program_id(2)          # kv split index

    seqlen = tl.load(SeqLen_ptr + b)
    # split the [0, seqlen) key range across NUM_SPLITS programs
    split_len = tl.cdiv(seqlen, NUM_SPLITS)
    kv_start = s * split_len
    kv_end = tl.minimum(kv_start + split_len, seqlen)

    d_off = tl.arange(0, D)
    # load the single query vector for this (b, h): [D]
    q = tl.load(Q_ptr + b * stride_qb + h * stride_qh + d_off * stride_qd)
    q = (q * sm_scale).to(tl.float32)

    kv_head = h // (Hq // Hkv)    # GQA mapping

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([D], dtype=tl.float32)

    # iterate the key range for this split in tiles of BLOCK_N (<= PAGE ideally)
    for kv_pos in range(kv_start, kv_end, BLOCK_N):
        n_off = kv_pos + tl.arange(0, BLOCK_N)
        mask = n_off < kv_end

        # map each logical position to a physical page + offset via block table
        logical_block = n_off // PAGE
        in_block = n_off % PAGE
        # guard block-table index
        lb_safe = tl.where(logical_block < MAX_BLOCKS, logical_block, 0)
        phys_block = tl.load(
            BlockTable_ptr + b * stride_bt_b + lb_safe * stride_bt_x,
            mask=mask, other=0,
        )

        # gather K tile: [BLOCK_N, D]
        k_addr = (Kcache_ptr
                  + phys_block[:, None] * stride_kb
                  + in_block[:, None] * stride_kp
                  + kv_head * stride_kh
                  + d_off[None, :] * stride_kd)
        k = tl.load(k_addr, mask=mask[:, None], other=0.0).to(tl.float32)

        # scores = q . k^T  -> [BLOCK_N]
        scores = tl.sum(q[None, :] * k, axis=1)
        scores = tl.where(mask, scores, float("-inf"))

        # online softmax update
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)               # [BLOCK_N]

        # gather V tile: [BLOCK_N, D]
        v_addr = (Vcache_ptr
                  + phys_block[:, None] * stride_vb
                  + in_block[:, None] * stride_vp
                  + kv_head * stride_vh
                  + d_off[None, :] * stride_vd)
        v = tl.load(v_addr, mask=mask[:, None], other=0.0).to(tl.float32)

        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    # write partial O (normalized within split) and partial LSE
    out = acc / tl.where(l_i > 0.0, l_i, 1.0)
    o_ptr = (Out_partial_ptr + b * stride_ob + h * stride_oh
             + s * stride_os + d_off * stride_od)
    tl.store(o_ptr, out.to(Out_partial_ptr.dtype.element_ty))

    lse = m_i + tl.log(tl.where(l_i > 0.0, l_i, 1.0))
    tl.store(Lse_partial_ptr + b * stride_lb + h * stride_lh + s * stride_ls, lse)


# ---------------------------------------------------------------------------
# Phase 1b: GQA-PACKED per-split kernel.
# One program == (batch, KV-head, split). It loads each K/V tile ONCE and
# computes ALL G = Hq/Hkv query heads that share this KV head against it.
# This removes the G-fold redundant HBM reads of the naive per-query-head kernel
# -> ~G x more effective bandwidth, which is the whole game for memory-bound decode.
# ---------------------------------------------------------------------------
@triton.jit
def _paged_decode_split_gqa_kernel(
    Q_ptr,              # [B, Hq, D]
    Kcache_ptr,         # [num_blocks, PAGE, Hkv, D]
    Vcache_ptr,         # [num_blocks, PAGE, Hkv, D]
    BlockTable_ptr,     # [B, max_blocks]
    SeqLen_ptr,         # [B]
    Out_partial_ptr,    # [B, Hq, NUM_SPLITS, D]
    Lse_partial_ptr,    # [B, Hq, NUM_SPLITS]
    sm_scale,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kp, stride_kh, stride_kd,
    stride_vb, stride_vp, stride_vh, stride_vd,
    stride_bt_b, stride_bt_x,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    Hq: tl.constexpr, Hkv: tl.constexpr, G: tl.constexpr,
    PAGE: tl.constexpr, D: tl.constexpr,
    NUM_SPLITS: tl.constexpr, BLOCK_N: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
):
    b = tl.program_id(0)
    kvh = tl.program_id(1)        # KV head
    s = tl.program_id(2)          # split

    seqlen = tl.load(SeqLen_ptr + b)
    split_len = tl.cdiv(seqlen, NUM_SPLITS)
    kv_start = s * split_len
    kv_end = tl.minimum(kv_start + split_len, seqlen)

    d_off = tl.arange(0, D)                       # [D]
    g_off = tl.arange(0, G)                       # [G] query heads in this group
    qh = kvh * G + g_off                          # [G] actual query-head indices

    # load the G query vectors for this group: [G, D]
    q = tl.load(Q_ptr + b * stride_qb + qh[:, None] * stride_qh + d_off[None, :] * stride_qd)
    q = (q.to(tl.float32) * sm_scale)             # [G, D]

    m_i = tl.zeros([G], dtype=tl.float32) + float("-inf")
    l_i = tl.zeros([G], dtype=tl.float32)
    acc = tl.zeros([G, D], dtype=tl.float32)

    for kv_pos in range(kv_start, kv_end, BLOCK_N):
        n_off = kv_pos + tl.arange(0, BLOCK_N)    # [N]
        mask = n_off < kv_end

        logical_block = n_off // PAGE
        in_block = n_off % PAGE
        lb_safe = tl.where(logical_block < MAX_BLOCKS, logical_block, 0)
        phys_block = tl.load(
            BlockTable_ptr + b * stride_bt_b + lb_safe * stride_bt_x,
            mask=mask, other=0)

        # K tile loaded ONCE for the whole group: [N, D]
        k_addr = (Kcache_ptr + phys_block[:, None] * stride_kb
                  + in_block[:, None] * stride_kp + kvh * stride_kh
                  + d_off[None, :] * stride_kd)
        k = tl.load(k_addr, mask=mask[:, None], other=0.0).to(tl.float32)   # [N, D]

        # scores for all G heads: [G, D] @ [D, N] -> [G, N]
        scores = tl.dot(q, tl.trans(k))           # [G, N]
        scores = tl.where(mask[None, :], scores, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))     # [G]
        alpha = tl.exp(m_i - m_new)                          # [G]
        p = tl.exp(scores - m_new[:, None])                  # [G, N]

        v_addr = (Vcache_ptr + phys_block[:, None] * stride_vb
                  + in_block[:, None] * stride_vp + kvh * stride_vh
                  + d_off[None, :] * stride_vd)
        v = tl.load(v_addr, mask=mask[:, None], other=0.0).to(tl.float32)   # [N, D]

        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)  # [G,N]@[N,D]->[G,D]
        l_i = l_i * alpha + tl.sum(p, axis=1)                  # [G]
        m_i = m_new

    out = acc / tl.where(l_i > 0.0, l_i, 1.0)[:, None]         # [G, D]
    o_ptr = (Out_partial_ptr + b * stride_ob + qh[:, None] * stride_oh
             + s * stride_os + d_off[None, :] * stride_od)
    tl.store(o_ptr, out.to(Out_partial_ptr.dtype.element_ty))

    lse = m_i + tl.log(tl.where(l_i > 0.0, l_i, 1.0))          # [G]
    tl.store(Lse_partial_ptr + b * stride_lb + qh * stride_lh + s * stride_ls, lse)


# ---------------------------------------------------------------------------
# Phase 2: combine the NUM_SPLITS partials via LSE reduction
# ---------------------------------------------------------------------------
@triton.jit
def _combine_splits_kernel(
    Out_partial_ptr,    # [B, Hq, NUM_SPLITS, D]
    Lse_partial_ptr,    # [B, Hq, NUM_SPLITS]
    Out_ptr,            # [B, Hq, D]
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    stride_fb, stride_fh, stride_fd,
    D: tl.constexpr, NUM_SPLITS: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)

    s_off = tl.arange(0, NUM_SPLITS)          # [S]
    lse = tl.load(Lse_partial_ptr + b * stride_lb + h * stride_lh + s_off * stride_ls)

    m = tl.max(lse, axis=0)                    # scalar
    p = tl.exp(lse - m)                        # [S]
    denom = tl.sum(p, axis=0)                  # scalar
    w = p / denom                              # [S] combine weights

    # load ALL partial outputs at once: [S, D], weight-sum over S (vectorized)
    d_off = tl.arange(0, D)                    # [D]
    o_ptr = (Out_partial_ptr + b * stride_ob + h * stride_oh
             + s_off[:, None] * stride_os + d_off[None, :] * stride_od)
    o = tl.load(o_ptr).to(tl.float32)          # [S, D]
    acc = tl.sum(o * w[:, None], axis=0)       # [D]

    tl.store(Out_ptr + b * stride_fb + h * stride_fh + d_off * stride_fd,
             acc.to(Out_ptr.dtype.element_ty))


# ---------------------------------------------------------------------------
# Python entry point
# ---------------------------------------------------------------------------
def paged_decode_attention(
    q,                  # [B, Hq, D]        (decode step: one query per sequence)
    k_cache,            # [num_blocks, PAGE, Hkv, D]
    v_cache,            # [num_blocks, PAGE, Hkv, D]
    block_table,        # [B, max_blocks]   int32
    seq_lens,           # [B]               int32
    sm_scale=None,
    num_splits=None,
    block_n=64,          # best on H20 @128k GQA-packed (sweep: splits=256,block_n=64 -> 51% peak BW)
    gqa_packed=True,
):
    B, Hq, D = q.shape
    num_blocks, PAGE, Hkv, _ = k_cache.shape
    max_blocks = block_table.shape[1]
    G = Hq // Hkv
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)

    if num_splits is None:
        # Decode is embarrassingly parallel over the KV length and the grid is only
        # (B, Hq, num_splits). With B=1, Hq=32 we need MANY splits to fill ~78 SMs on
        # H20 and expose enough parallel HBM reads to approach peak bandwidth.
        max_len = int(seq_lens.max().item())
        # target ~256 splits at 128k (sweep-optimal on H20): each split reads ~512 keys,
        # enough parallel programs to fill the SMs and expose peak HBM read bandwidth.
        num_splits = max(8, min(256, triton.cdiv(max_len, 512)))
    # combine kernel uses tl.arange(0, NUM_SPLITS) -> must be a power of 2
    num_splits = 1 << (num_splits - 1).bit_length()

    dev = q.device
    out_partial = torch.empty((B, Hq, num_splits, D), device=dev, dtype=torch.float32)
    lse_partial = torch.full((B, Hq, num_splits), float("-inf"),
                             device=dev, dtype=torch.float32)

    # tl.dot needs M (=G) to be a power of 2 and >= 16 for good codegen; pad G up.
    G_pad = max(16, 1 << (G - 1).bit_length()) if gqa_packed else G

    if gqa_packed and G >= 1:
        grid1 = (B, Hkv, num_splits)
        _paged_decode_split_gqa_kernel[grid1](
            q, k_cache, v_cache, block_table, seq_lens,
            out_partial, lse_partial, sm_scale,
            q.stride(0), q.stride(1), q.stride(2),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
            block_table.stride(0), block_table.stride(1),
            out_partial.stride(0), out_partial.stride(1), out_partial.stride(2), out_partial.stride(3),
            lse_partial.stride(0), lse_partial.stride(1), lse_partial.stride(2),
            Hq=Hq, Hkv=Hkv, G=G, PAGE=PAGE, D=D,
            NUM_SPLITS=num_splits, BLOCK_N=block_n, MAX_BLOCKS=max_blocks,
            num_warps=4, num_stages=2,
        )
    else:
        grid1 = (B, Hq, num_splits)
        _paged_decode_split_kernel[grid1](
            q, k_cache, v_cache, block_table, seq_lens,
            out_partial, lse_partial, sm_scale,
            q.stride(0), q.stride(1), q.stride(2),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
            block_table.stride(0), block_table.stride(1),
            out_partial.stride(0), out_partial.stride(1), out_partial.stride(2), out_partial.stride(3),
            lse_partial.stride(0), lse_partial.stride(1), lse_partial.stride(2),
            Hq=Hq, Hkv=Hkv, PAGE=PAGE, D=D,
            NUM_SPLITS=num_splits, BLOCK_N=block_n, MAX_BLOCKS=max_blocks,
            num_warps=4, num_stages=2,
        )

    out = torch.empty((B, Hq, D), device=dev, dtype=q.dtype)
    grid2 = (B, Hq)
    _combine_splits_kernel[grid2](
        out_partial, lse_partial, out,
        out_partial.stride(0), out_partial.stride(1), out_partial.stride(2), out_partial.stride(3),
        lse_partial.stride(0), lse_partial.stride(1), lse_partial.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        D=D, NUM_SPLITS=num_splits,
        num_warps=4,
    )
    return out
