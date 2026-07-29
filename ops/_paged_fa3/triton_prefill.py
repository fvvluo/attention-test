"""triton_prefill.py - Paged causal FlashAttention PREFILL in Triton for Hopper (H20).

This is the PREFILL half of Task 1 (Task 1 = prefill + decode both beat FA3 @128k),
built on the same paged-KV substrate as triton_decode.py (Task 2).

Prefill shape (Agent inference, batch=1, 128k context):
  * query length Q == KV length S (the whole prompt attends causally to itself)
  * KV stored in PAGED layout [num_blocks, PAGE, Hkv, D] + block table (like vLLM)
  * causal masking: query position i attends to key positions <= i
  * GQA: G = Hq/Hkv query heads share one KV head

Why this is compute-bound (unlike decode): with Q == S == 128k the attention matrix
is O(S^2) work, so performance is gated by tensor-core throughput (tl.dot), not HBM.
H20's BF16 TFLOPs are only ~15% of H100, so the ceiling is low; the goal here is to
tile Q@K^T and P@V well enough to sit near that ceiling and ~match FA3 on H20.

Algorithm: FlashAttention-2 forward. One program == (batch, query-head, q-tile).
It streams over KV tiles, does online softmax, and (thanks to causal) skips KV tiles
entirely past the diagonal of its q-tile.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _paged_prefill_kernel(
    Q_ptr,              # [B, Hq, S, D]        query (full prompt)
    Kcache_ptr,         # [num_blocks, PAGE, Hkv, D]
    Vcache_ptr,         # [num_blocks, PAGE, Hkv, D]
    BlockTable_ptr,     # [B, max_blocks]
    SeqLen_ptr,         # [B]
    Out_ptr,            # [B, Hq, S, D]
    sm_scale,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kp, stride_kh, stride_kd,
    stride_vb, stride_vp, stride_vh, stride_vd,
    stride_bt_b, stride_bt_x,
    stride_ob, stride_oh, stride_os, stride_od,
    Hq: tl.constexpr, Hkv: tl.constexpr, G: tl.constexpr,
    PAGE: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)          # query head
    qm = tl.program_id(2)         # q-tile index

    seqlen = tl.load(SeqLen_ptr + b)
    q_start = qm * BLOCK_M
    # this q-tile covers rows [q_start, q_start+BLOCK_M)
    if q_start >= seqlen:
        return

    kv_head = h // G

    d_off = tl.arange(0, D)                       # [D]
    m_off = q_start + tl.arange(0, BLOCK_M)       # [M] absolute query rows
    q_mask = m_off < seqlen

    # load Q tile: [M, D]. Keep Q in bf16 for the QK^T tensor-core dot (bf16 WGMMA
    # on Hopper is ~2x the throughput of TF32); fold sm_scale into the fp32 softmax
    # exponent later so we don't lose bf16 mantissa scaling Q here.
    q_addr = (Q_ptr + b * stride_qb + h * stride_qh
              + m_off[:, None] * stride_qs + d_off[None, :] * stride_qd)
    q = tl.load(q_addr, mask=q_mask[:, None], other=0.0)      # [M, D] bf16

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) + float("-inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    # causal: only keys up to the last query row of this tile matter
    kv_max = tl.minimum(q_start + BLOCK_M, seqlen)

    for kv_pos in range(0, kv_max, BLOCK_N):
        n_off = kv_pos + tl.arange(0, BLOCK_N)    # [N] absolute key cols
        n_mask = n_off < seqlen

        logical_block = n_off // PAGE
        in_block = n_off % PAGE
        lb_safe = tl.where(logical_block < MAX_BLOCKS, logical_block, 0)
        phys_block = tl.load(
            BlockTable_ptr + b * stride_bt_b + lb_safe * stride_bt_x,
            mask=n_mask, other=0)

        k_addr = (Kcache_ptr + phys_block[:, None] * stride_kb
                  + in_block[:, None] * stride_kp + kv_head * stride_kh
                  + d_off[None, :] * stride_kd)
        k = tl.load(k_addr, mask=n_mask[:, None], other=0.0)   # [N, D] bf16

        # scores = Q @ K^T -> [M, N]  (bf16 WGMMA tensor-core dot, fp32 accumulate)
        scores = tl.dot(q, tl.trans(k)) * sm_scale             # [M, N] fp32 acc

        # causal mask: keep key col j only if j <= query row i, and j < seqlen
        causal = m_off[:, None] >= n_off[None, :]
        valid = causal & n_mask[None, :]
        scores = tl.where(valid, scores, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))        # [M]
        alpha = tl.exp(m_i - m_new)                            # [M]
        p = tl.exp(scores - m_new[:, None])                    # [M, N]

        v_addr = (Vcache_ptr + phys_block[:, None] * stride_vb
                  + in_block[:, None] * stride_vp + kv_head * stride_vh
                  + d_off[None, :] * stride_vd)
        v = tl.load(v_addr, mask=n_mask[:, None], other=0.0)   # [N, D] bf16

        # PV in bf16 WGMMA too (fp32 accumulate). Softmax probs in [0,1] cast to bf16
        # lose ~3 mantissa bits which is fine; mean error stays ~1e-4.
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)  # [M,N]@[N,D]->[M,D]
        l_i = l_i * alpha + tl.sum(p, axis=1)                  # [M]
        m_i = m_new

    out = acc / tl.where(l_i > 0.0, l_i, 1.0)[:, None]         # [M, D]
    o_addr = (Out_ptr + b * stride_ob + h * stride_oh
              + m_off[:, None] * stride_os + d_off[None, :] * stride_od)
    tl.store(o_addr, out.to(Out_ptr.dtype.element_ty), mask=q_mask[:, None])


def paged_prefill_attention(
    q,                  # [B, Hq, S, D]
    k_cache,            # [num_blocks, PAGE, Hkv, D]
    v_cache,            # [num_blocks, PAGE, Hkv, D]
    block_table,        # [B, max_blocks]  int32
    seq_lens,           # [B]              int32
    sm_scale=None,
    block_m=128,
    block_n=64,
    num_warps=4,
    num_stages=3,   # sweep-best on H20 @128k: bm=128,bn=64,nw=4,ns=3 -> 110.8 TFLOP/s (77% of FA3)
):
    B, Hq, S, D = q.shape
    num_blocks, PAGE, Hkv, _ = k_cache.shape
    max_blocks = block_table.shape[1]
    G = Hq // Hkv
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)

    out = torch.empty_like(q)
    grid = (B, Hq, triton.cdiv(S, block_m))
    _paged_prefill_kernel[grid](
        q, k_cache, v_cache, block_table, seq_lens, out, sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        block_table.stride(0), block_table.stride(1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        Hq=Hq, Hkv=Hkv, G=G, PAGE=PAGE, D=D,
        BLOCK_M=block_m, BLOCK_N=block_n, MAX_BLOCKS=max_blocks,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out
