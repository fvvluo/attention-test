#!/usr/bin/env python3
"""B7 cp.async double-buffered warp-MMA GQA decode runner.

Independent implementation by Liu Xiaochen. Same public interface / cache
discipline as B5's runner; only Stage-1 differs (cp.async 2-stage pipeline).
Reuses the B5 combine kernel unchanged (identical partial_o/partial_lse format).
Identity: "b7-mma-cpasync-pdl". Independent compile/workspace caches.
"""

import math
import os
import sys

import torch

if not hasattr(torch, "float4_e2m1fn_x2"):
    torch.float4_e2m1fn_x2 = torch.uint8

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

_compile_cache = {}
_workspace_cache = {}


def _num_splits(seqlen_k, n_block=64, max_splits=256):
    num_blocks = seqlen_k // n_block
    ns = min(max_splits, num_blocks)
    while ns > 1 and num_blocks % ns != 0:
        ns -= 1
    return ns


def _workspace(q, num_splits):
    stream_id = int(torch.cuda.current_stream(q.device).cuda_stream)
    key = ("b7", q.device.index, stream_id, q.dtype, tuple(q.shape), num_splits)
    buf = _workspace_cache.get(key)
    if buf is None:
        b, hq, _, d = q.shape
        partial = torch.empty(b, hq, num_splits, d, device=q.device, dtype=torch.float32)
        lse = torch.empty(b, hq, num_splits, device=q.device, dtype=torch.float32)
        out = torch.empty(b, hq, d, device=q.device, dtype=q.dtype)
        buf = (partial, lse, out)
        _workspace_cache[key] = buf
    return buf


def mma_decode_b7(q, k, v, sm_scale=None, num_splits=None, n_block=64):
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q/k/v 必须是 BHSD 四维")
    if q.shape[2] != 1:
        raise ValueError("B7 decode 要求 q_len==1")
    if tuple(q.shape) != (1, 64, 1, 128):
        raise ValueError("B7 首版要求 Q shape (1,64,1,128)")
    if tuple(k.shape[:2]) != (1, 8) or k.shape != v.shape or k.shape[-1] != 128:
        raise ValueError("B7 首版要求 K/V shape (1,8,Sk,128)")
    if k.shape[2] % n_block != 0:
        raise ValueError(f"B7 要求 kv_len % {n_block} == 0")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("B7 仅支持 CUDA")
    if not (q.device == k.device == v.device):
        raise ValueError("Q/K/V 必须同 device")
    if not (q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16):
        raise ValueError("B7 仅支持 BF16")
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        raise ValueError("B7 要求 contiguous BHSD 输入")
    major, minor = torch.cuda.get_device_capability(q.device)
    if (major, minor) != (9, 0):
        raise ValueError(f"B7 需要 SM90，当前 SM{major}{minor}")

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    sm_scale = float(sm_scale)

    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from decode_mma_stage1_b7 import LiuXiaochenMmaDecodeStage1B7
    from decode_mma_combine_b7 import LiuXiaochenMmaDecodeCombineB7

    b, hq, _, d = q.shape
    kvh, sk = k.shape[1], k.shape[2]
    ns = num_splits if num_splits is not None else _num_splits(sk, n_block)
    partial, lse, out = _workspace(q, ns)

    q_c = from_dlpack(q.squeeze(2), assumed_align=16)
    k_c = from_dlpack(k.transpose(1, 2), assumed_align=16)
    v_c = from_dlpack(v.transpose(1, 2), assumed_align=16)
    p_c = from_dlpack(partial, assumed_align=16)
    l_c = from_dlpack(lse, assumed_align=16)
    o_c = from_dlpack(out, assumed_align=16)

    torch_stream = torch.cuda.current_stream(q.device)
    stream = cuda.CUstream(int(torch_stream.cuda_stream))
    scale_log2 = cutlass.Float32(sm_scale * 1.4426950408889634)

    key = ("b7-mma-cpasync-pdl", q.device.index, int(torch_stream.cuda_stream), q.dtype,
           tuple(q.shape), tuple(k.shape), n_block, ns, partial.dtype)
    compiled = _compile_cache.get(key)
    if compiled is None:
        s1 = LiuXiaochenMmaDecodeStage1B7(
            head_dim=d, q_heads=hq, kv_heads=kvh, seqlen_k=sk, num_splits=ns, n_block=n_block
        )
        cmb = LiuXiaochenMmaDecodeCombineB7(head_dim=d, num_splits=ns)
        stage1 = cute.compile(s1, q_c, k_c, v_c, p_c, l_c, scale_log2, stream)
        combine = cute.compile(cmb, p_c, l_c, o_c, stream)
        compiled = (stage1, combine)
        _compile_cache[key] = compiled

    stage1, combine = compiled
    stage1(q_c, k_c, v_c, p_c, l_c, scale_log2, stream)
    combine(p_c, l_c, o_c, stream)
    return out.unsqueeze(2)
