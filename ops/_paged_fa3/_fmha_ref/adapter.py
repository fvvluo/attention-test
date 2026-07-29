"""Adapter that routes supported prefill shapes to NVIDIA's warp-specialized
TMA+WGMMA Hopper FMHA reference (vendored under paged_fa3/_fmha_ref).

Our public signature is cute_attention(q,k,v,...) with q,k,v = [B,S,H,D] bf16,
returning o=[B,Sq,Hq,D]. The reference kernel consumes 5-mode CuTe tensors laid
out as (s, d, h_r, h_k, b) with d as the contiguous (leading) mode and the head
axis split into (h_r, h_k) where h_k is the KV head. We build those views over
OUR torch memory with a head-mode layout that matches OUR contiguous GQA
convention (kv_head = q_head // (Hq//Hkv)):

  reference index (hri, hki) -> our q head hq = hki*h_r + hri

so reference kv_head == hki correctly indexes our KV head, and the h_r query
heads sharing it are our contiguous group hq = hki*h_r + hri, hri in [0,h_r).

Env CUTE_DSL_ARCH=sm_90a is mandatory for TMA (set at import).
"""
import os
os.environ.setdefault("CUTE_DSL_ARCH", "sm_90a")

import math
import torch
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass.cute.runtime import from_dlpack

from . import fmha as _fmha
from . import fmha_helpers as _fmha_utils

_LOG2_E = 1.4426950408889634074

# compile cache keyed on (B,Sq,Sk,Hq,Hkv,D,is_causal,mma_m,mma_n)
_REF_CACHE = {}


def can_use_fmha_ref(q, k, v, is_causal, mma_tiler_mn=(64, 128)):
    """Cheap eligibility gate. Reference supports bf16, D in {64,128}, contiguous
    row-major [B,S,H,D], Hq % Hkv == 0, and needs Sq/Sk workable with the tile."""
    if q.dtype != torch.bfloat16:
        return False
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        return False
    B, Sq, Hq, D = q.shape
    Bk, Sk, Hkv, Dk = k.shape
    if D != Dk or D not in (64, 128):
        return False
    if Hq % Hkv != 0:
        return False
    # need contiguous row-major (D innermost, stride 1) so we can build a k-major view
    for t in (q, k, v):
        if t.stride(-1) != 1:
            return False
    return True


def _build_qo_tensor(t, Hkv):
    """t: torch [B,S,Hq,D] row-major. Return CuTe tensor (s, d, h_r, h_k, b)
    with our contiguous GQA convention hq = hki*Hr + hri (Hkv outer, Hr inner in
    the head axis). We realize this as a torch *view* (no copy):
      [B,S,Hq,D] --view--> [B,S,Hkv,Hr,D] --permute(1,4,3,2,0)--> (S,D,Hr,Hkv,B)
    then from_dlpack + mark_layout_dynamic(leading_dim=1) (d is contiguous)."""
    B, S, Hq, D = t.shape
    Hr = Hq // Hkv
    view = t.view(B, S, Hkv, Hr, D).permute(1, 4, 3, 2, 0)  # (S,D,Hr,Hkv,B)
    ct = from_dlpack(view, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    ct.element_type = _dtype_of(t)
    return ct


def _build_kv_tensor(t):
    """t: torch [B,S,Hkv,D] row-major -> CuTe (s, d, h_r=1, h_k, b).
      [B,S,Hkv,D] --view--> [B,S,Hkv,1,D] --permute(1,4,3,2,0)--> (S,D,1,Hkv,B)"""
    B, S, Hkv, D = t.shape
    view = t.view(B, S, Hkv, 1, D).permute(1, 4, 3, 2, 0)  # (S,D,1,Hkv,B)
    ct = from_dlpack(view, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    ct.element_type = _dtype_of(t)
    return ct


def _build_lse_tensor(t, Hkv):
    """t: torch [B,S,Hq,1] fp32 row-major -> CuTe (s, 1, h_r, h_k, b).
      [B,S,Hq,1] --view--> [B,S,Hkv,Hr,1] --permute(1,4,3,2,0)--> (S,1,Hr,Hkv,B)"""
    B, S, Hq, _ = t.shape
    Hr = Hq // Hkv
    view = t.view(B, S, Hkv, Hr, 1).permute(1, 4, 3, 2, 0)  # (S,1,Hr,Hkv,B)
    ct = from_dlpack(view, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    ct.element_type = cutlass.Float32
    return ct


def _dtype_of(t):
    if t.dtype == torch.bfloat16:
        return cutlass.BFloat16
    if t.dtype == torch.float16:
        return cutlass.Float16
    if t.dtype == torch.float32:
        return cutlass.Float32
    raise TypeError(f"unsupported dtype {t.dtype}")


def fmha_ref_attention(q, k, v, sm_scale=None, is_causal=False,
                       mma_tiler_mn=(64, 128)):
    """q,k,v: [B,S,H,D] bf16 cuda. Returns o [B,Sq,Hq,D] bf16."""
    B, Sq, Hq, D = q.shape
    Hkv = k.shape[2]
    Sk = k.shape[1]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    o = torch.empty_like(q)
    lse = torch.empty((B, Sq, Hq, 1), dtype=torch.float32, device=q.device)

    q_ct = _build_qo_tensor(q, Hkv)
    k_ct = _build_kv_tensor(k)
    v_ct = _build_kv_tensor(v)
    o_ct = _build_qo_tensor(o, Hkv)
    lse_ct = _build_lse_tensor(lse, Hkv)

    scale_softmax = float(sm_scale)
    scale_softmax_log2 = scale_softmax * _LOG2_E
    scale_output = 1.0

    mask_type = _fmha_utils.MaskEnum.WINDOW_MASK
    wsl = None
    wsr = None
    if is_causal:
        wsr = 0
    elif Sk % mma_tiler_mn[1] != 0:
        mask_type = _fmha_utils.MaskEnum.RESIDUAL_MASK

    key = (B, Sq, Sk, Hq, Hkv, D, is_causal, mma_tiler_mn)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    if key not in _REF_CACHE:
        mma_tiler = (*mma_tiler_mn, D)
        fmha = _fmha.HopperFusedMultiHeadAttentionForward(
            cutlass.Float32,   # qk_acc
            cutlass.Float32,   # pv_acc
            mma_tiler,
            False,             # is_persistent
            mask_type,
        )
        _REF_CACHE[key] = cute.compile(
            fmha, q_ct, k_ct, v_ct, o_ct, lse_ct,
            cutlass.Float32(scale_softmax_log2),
            cutlass.Float32(scale_softmax),
            cutlass.Float32(scale_output),
            wsl if wsl is None else cutlass.Int32(wsl),
            wsr if wsr is None else cutlass.Int32(wsr),
            stream,
        )
    _REF_CACHE[key](
        q_ct, k_ct, v_ct, o_ct, lse_ct,
        cutlass.Float32(scale_softmax_log2),
        cutlass.Float32(scale_softmax),
        cutlass.Float32(scale_output),
        wsl if wsl is None else cutlass.Int32(wsl),
        wsr if wsr is None else cutlass.Int32(wsr),
        stream,
    )
    return o
