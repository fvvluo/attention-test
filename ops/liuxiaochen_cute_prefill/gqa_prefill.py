#!/usr/bin/env python3
"""Stride-aware BHSD GQA prefill wrapper for the team benchmark.

This module is a correctness implementation.  It accepts real team Q/K/V
without repeating KV heads or copying complete inputs, and writes output
directly in contiguous BHSD layout.  It does not register itself with
bench_attention and does not benchmark.
"""

import os
import sys

import torch

if not hasattr(torch, "float4_e2m1fn_x2"):
    torch.float4_e2m1fn_x2 = torch.uint8

import cuda.bindings.driver as cuda  # noqa: E402
import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass.cute.runtime import from_dlpack  # noqa: E402

from .ampere_flash_attention_gqa import FlashAttentionForwardAmpere  # noqa: E402

HEAD_DIM = 128
M_BLOCK_SIZE = 128
N_BLOCK_SIZE = 64
NUM_THREADS = 128
DTYPE_TORCH = torch.bfloat16
DTYPE_CUTLASS = cutlass.BFloat16
_COMPILE_CACHE = {}


def _make_cute_tensor(tensor):
    """Wrap a contiguous BHSD tensor as logical BSHD without copying."""
    if not tensor.is_contiguous():
        raise ValueError("tensor 必须是 contiguous BHSD，禁止通过隐式复制修复布局")
    logical = tensor.transpose(1, 2)
    return from_dlpack(logical, assumed_align=16).mark_layout_dynamic()


def _validate_inputs(q, k, v, causal, sm_scale):
    tensors = {"q": q, "k": k, "v": v}
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} 必须是 torch.Tensor")
        if not tensor.is_cuda:
            raise ValueError(f"{name} 必须位于 CUDA device")
        if tensor.dtype != DTYPE_TORCH:
            raise TypeError(f"{name} dtype 必须是 torch.bfloat16，实际为 {tensor.dtype}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} 必须是 contiguous BHSD")
        if tensor.data_ptr() % 16 != 0:
            raise ValueError(f"{name} 起始地址必须满足 16-byte 对齐")

    if not (q.device == k.device == v.device):
        raise ValueError("q/k/v 必须位于同一个 CUDA device")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q/k/v 必须是四维 BHSD tensor")
    batch, num_q_heads, seq_len, head_dim = q.shape
    if batch != 1:
        raise ValueError("第一版仅支持 B=1")
    if head_dim != HEAD_DIM:
        raise ValueError(f"第一版仅支持 D={HEAD_DIM}")
    if k.shape != v.shape:
        raise ValueError("k/v shape 必须相同")
    if k.shape[0] != batch or k.shape[2] != seq_len:
        raise ValueError("第一版仅支持 q_len == kv_len")
    num_kv_heads = k.shape[1]
    if num_kv_heads <= 0 or num_q_heads <= 0:
        raise ValueError("head 数必须为正")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"GQA 要求 q_heads={num_q_heads} 能被 kv_heads={num_kv_heads} 整除"
        )
    if not isinstance(causal, bool):
        raise TypeError("causal 必须是 bool")
    if sm_scale is None:
        scale = float(head_dim**-0.5)
    else:
        scale = float(sm_scale)
        if not torch.isfinite(torch.tensor(scale)) or scale <= 0:
            raise ValueError("sm_scale 必须是有限正数")
    return batch, num_q_heads, num_kv_heads, seq_len, head_dim, scale


def _get_compiled(q, k, v, out, causal, sm_scale):
    key = (
        q.device.index,
        q.dtype,
        q.shape,
        k.shape,
        out.shape,
        causal,
        float(sm_scale),
    )
    if key in _COMPILE_CACHE:
        return _COMPILE_CACHE[key]
    kernel = FlashAttentionForwardAmpere(
        HEAD_DIM, M_BLOCK_SIZE, N_BLOCK_SIZE, NUM_THREADS, causal
    )
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    compiled = cute.compile(
        kernel,
        _make_cute_tensor(q),
        _make_cute_tensor(k),
        _make_cute_tensor(v),
        _make_cute_tensor(out),
        float(sm_scale),
        stream,
    )
    _COMPILE_CACHE[key] = (compiled, stream)
    return _COMPILE_CACHE[key]


def run_gqa_prefill(q, k, v, causal=True, sm_scale=None):
    """Run the stride-aware GQA prefill correctness kernel."""
    _batch, _hq, _hkv, _seq, _dim, scale = _validate_inputs(
        q, k, v, causal, sm_scale
    )
    out = torch.empty_like(q)
    compiled, stream = _get_compiled(q, k, v, out, causal, scale)
    compiled(
        _make_cute_tensor(q),
        _make_cute_tensor(k),
        _make_cute_tensor(v),
        _make_cute_tensor(out),
        scale,
        stream,
    )
    return out


def compile_gqa_prefill(q, k, v, causal=True, sm_scale=None):
    """Compile for the given inputs without launching the kernel."""
    _batch, _hq, _hkv, _seq, _dim, scale = _validate_inputs(
        q, k, v, causal, sm_scale
    )
    out = torch.empty_like(q)
    compiled, stream = _get_compiled(q, k, v, out, causal, scale)

    def run():
        compiled(
            _make_cute_tensor(q),
            _make_cute_tensor(k),
            _make_cute_tensor(v),
            _make_cute_tensor(out),
            scale,
            stream,
        )
        return out

    return run
