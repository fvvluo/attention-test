# Copyright (c) 2023, Tri Dao.
# Simplified FlashAttention-3 interface: forward-only (inference), SM90 (Hopper),
# no backward, no dropout, no FP8, no softcap, no in-kernel KV append / rotary.

from typing import Optional, Tuple, Union

import torch

# isort: off
# We need to import the CUDA kernels after importing torch
import flash_attn_3._C  # Registers operators with PyTorch

# isort: on

flash_attn_3_gpu = torch.ops.flash_attn_3


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def round_up_headdim(head_size: int) -> int:
    # This build supports head dims 64, 128 and 256 (padded up if smaller).
    if head_size <= 64:
        return 64
    if head_size <= 128:
        return 128
    return 256


def _flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out_: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    page_table: Optional[torch.Tensor] = None,
    kv_batch_idx: Optional[torch.Tensor] = None,
    leftpad_k: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size_left: int = -1,
    window_size_right: int = -1,
    attention_chunk: int = 0,
    scheduler_metadata: Optional[torch.Tensor] = None,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    sm_margin: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k = [maybe_contiguous(x) for x in (q, k)]
    v = v.contiguous() if v.stride(-1) != 1 and v.stride(-3) != 1 else v
    cu_seqlens_q, cu_seqlens_k = [maybe_contiguous(x) for x in (cu_seqlens_q, cu_seqlens_k)]
    seqused_q, seqused_k = [maybe_contiguous(x) for x in (seqused_q, seqused_k)]
    page_table, kv_batch_idx, leftpad_k = [
        maybe_contiguous(x) for x in (page_table, kv_batch_idx, leftpad_k)
    ]
    out, softmax_lse, out_accum, softmax_lse_accum = flash_attn_3_gpu.fwd(
        q,
        k,
        v,
        None,  # k_new
        None,  # v_new
        None,  # q_v
        out_,
        cu_seqlens_q,
        cu_seqlens_k,
        None,  # cu_seqlens_k_new
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        page_table,
        kv_batch_idx,
        leftpad_k,
        None,  # rotary_cos
        None,  # rotary_sin
        None,  # seqlens_rotary
        None,  # q_descale
        None,  # k_descale
        None,  # v_descale
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        attention_chunk,
        0.0,  # softcap
        False,  # is_rotary_interleaved
        scheduler_metadata,
        num_splits,
        pack_gqa,
        sm_margin,
    )
    if out_accum is None:
        out_accum = torch.tensor([], device=out.device)
    if softmax_lse_accum is None:
        softmax_lse_accum = torch.tensor([], device=out.device)
    return out, softmax_lse, out_accum, softmax_lse_accum


@torch.no_grad()
def flash_attn_func(
    q,
    k,
    v,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    attention_chunk=0,
    num_splits=1,
    pack_gqa=None,
    sm_margin=0,
    return_attn_probs=False,
):
    """Forward-only flash attention (inference build, no backward).

    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. The number of heads in Q must be divisible by the number of heads in KV.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        num_splits: int. If > 1, split the key/value into this many chunks along the sequence.
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen).
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    out, softmax_lse, *_ = _flash_attn_forward(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        attention_chunk=attention_chunk,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        sm_margin=sm_margin,
    )
    return (out, softmax_lse) if return_attn_probs else out


@torch.no_grad()
def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    seqused_q=None,
    seqused_k=None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    attention_chunk=0,
    num_splits=1,
    pack_gqa=None,
    sm_margin=0,
    return_attn_probs=False,
):
    """Forward-only varlen flash attention (inference build, no backward).

    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim).
        v: (total_k, nheads_k, headdim).
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
            of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. Same for k/v.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
    Return:
        out: (total_q, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (nheads, total_q).
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    out, softmax_lse, *_ = _flash_attn_forward(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        attention_chunk=attention_chunk,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        sm_margin=sm_margin,
    )
    return (out, softmax_lse) if return_attn_probs else out


def flash_attn_combine(out_partial, lse_partial, out=None, out_dtype=None):
    return flash_attn_3_gpu.fwd_combine(out_partial, lse_partial, out, out_dtype)


@torch.no_grad()
def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    cache_seqlens: Optional[Union[int, torch.Tensor]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    attention_chunk=0,
    scheduler_metadata=None,
    num_splits=0,  # Can be tuned for speed
    pack_gqa=None,  # Can be tuned for speed
    sm_margin=0,  # Can be tuned if some SMs are used for communication
    return_softmax_lse=False,
):
    """Forward-only flash attention with KV cache (inference build, no backward).

    To update the cache with new K/V, write them into k_cache / v_cache from Python
    (e.g. k_cache[:, pos] = k_new) and bump cache_seqlens before calling this function.

    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. The number of heads in Q must be divisible by the number of heads in KV.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no page_table,
            or (num_blocks, page_block_size, nheads_k, headdim) if there's a page_table
            (i.e. paged KV cache). page_block_size can be arbitrary (e.g, 1, 2, 3, 64, etc.).
        v_cache: same as k_cache.
        cache_seqlens: int, or (batch_size,), dtype torch.int32. The sequence lengths of the
            KV cache.
        cache_batch_idx: (batch_size,), dtype torch.int32. The indices used to index into the
            KV cache. If None, we assume that the batch indices are [0, 1, ..., batch_size - 1].
        cache_leftpad: (batch_size,), dtype torch.int32. The index that the KV cache starts.
        page_table [optional]: (batch_size, max_num_blocks_per_seq), dtype torch.int32.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        num_splits: int. If > 1, split the key/value into this many chunks along the sequence.
            If num_splits == 1, we don't split the key/value. If num_splits == 0, we use a
            heuristic to automatically determine the number of splits (recommended for decode).
        return_softmax_lse: bool. Whether to return the logsumexp of the attention scores.

    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_softmax_lse=True]: (batch_size, nheads, seqlen).
    """
    assert k_cache.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v_cache.stride(-1) == 1, "v_cache must have contiguous last dimension"
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if cache_seqlens is not None and isinstance(cache_seqlens, int):
        cache_seqlens = torch.full(
            (q.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device
        )
        cache_seqlens = maybe_contiguous(cache_seqlens)
    out, softmax_lse, *rest = _flash_attn_forward(
        q,
        k_cache,
        v_cache,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=cache_seqlens,
        max_seqlen_q=max_seqlen_q,
        page_table=page_table,
        kv_batch_idx=cache_batch_idx,
        leftpad_k=cache_leftpad,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        attention_chunk=attention_chunk,
        scheduler_metadata=scheduler_metadata,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        sm_margin=sm_margin,
    )
    return (out, softmax_lse, *rest) if return_softmax_lse else out


def get_scheduler_metadata(
    batch_size, max_seqlen_q, max_seqlen_k, num_heads_q, num_heads_kv, headdim,
    cache_seqlens: torch.Tensor,
    qkv_dtype=torch.bfloat16,
    headdim_v=None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    page_size: Optional[int] = None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    attention_chunk=0,
    num_splits=0,  # Can be tuned for speed
    pack_gqa=None,  # Can be tuned for speed
    sm_margin=0,  # Can be tuned if some SMs are used for communication
):
    cache_seqlens = maybe_contiguous(cache_seqlens)
    if headdim_v is None:
        headdim_v = headdim
    scheduler_metadata = flash_attn_3_gpu.get_scheduler_metadata(
        batch_size, max_seqlen_q, max_seqlen_k, num_heads_q, num_heads_kv, headdim, headdim_v,
        qkv_dtype,
        cache_seqlens,
        cu_seqlens_q,
        None,  # cu_seqlens_k
        None,  # cu_seqlens_k_new
        None,  # seqused_q
        cache_leftpad,
        page_size,
        0,  # max_seqlen_k_new
        causal,
        window_size[0], window_size[1],
        attention_chunk,
        False,  # has_softcap
        num_splits,
        pack_gqa,
        sm_margin,
    )
    return scheduler_metadata
