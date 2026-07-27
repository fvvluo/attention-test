"""Final BHSD adapter for the fixed SM90 CuTe DSL M128 GQA prefill kernel.

The selected production configuration is locked to B1 Hq64 Hkv8 S131072 D128,
M128N128, KV stage 4, producer registers 24, and compute registers 240.
"""

import torch

from ._cute_dsl_sm90_fmha_perf_m128_kernel import (
    get_compile_count as _kernel_get_compile_count,
)
from ._cute_dsl_sm90_fmha_perf_m128_kernel import run_sm90_fmha_m128
from .base import register


_REGISTERED_NAME = "CuTe DSL SM90 FMHA final M128N128D128 S4 GQA64x8"
_EXPECTED_BATCH = 1
_EXPECTED_Q_HEADS = 64
_EXPECTED_KV_HEADS = 8
_EXPECTED_SEQUENCE = 131072
_EXPECTED_HEAD_DIM = 128
_KV_STAGE = 4
_LOAD_REGS = 24
_MMA_REGS = 240


def _validate_inputs(q, k, v, causal, sm_scale):
    if not isinstance(q, torch.Tensor) or not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
        raise TypeError("q, k, v must be torch.Tensor instances")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must be rank-4 BHSD tensors")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("q, k, v must be CUDA tensors")
    if k.device != q.device or v.device != q.device:
        raise ValueError("q, k, v must all be on the same CUDA device")
    if torch.cuda.current_device() != q.device.index:
        raise RuntimeError("the current CUDA device must match q, k, and v")

    properties = torch.cuda.get_device_properties(q.device)
    device_name = properties.name.upper()
    if (properties.major, properties.minor) != (9, 0) or "H20" not in device_name:
        raise RuntimeError(
            "CuTe DSL SM90 FMHA requires an NVIDIA H20 SM90a device"
        )

    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("q, k, v must all use torch.bfloat16")
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        raise ValueError("q, k, v must be contiguous BHSD tensors")
    if k.shape != v.shape:
        raise ValueError("k and v must have identical shapes")

    expected_q_shape = (
        _EXPECTED_BATCH,
        _EXPECTED_Q_HEADS,
        _EXPECTED_SEQUENCE,
        _EXPECTED_HEAD_DIM,
    )
    expected_kv_shape = (
        _EXPECTED_BATCH,
        _EXPECTED_KV_HEADS,
        _EXPECTED_SEQUENCE,
        _EXPECTED_HEAD_DIM,
    )
    if tuple(q.shape) != expected_q_shape:
        raise ValueError("expected q BHSD shape (1, 64, 131072, 128)")
    if tuple(k.shape) != expected_kv_shape or tuple(v.shape) != expected_kv_shape:
        raise ValueError("expected k/v BHSD shape (1, 8, 131072, 128)")
    if causal is not True:
        raise ValueError("this specialized kernel requires causal=True")
    if sm_scale is not None:
        raise ValueError("sm_scale must be None; scale is fixed to 1/sqrt(128)")

    if any(tensor.data_ptr() % 16 != 0 for tensor in (q, k, v)):
        raise ValueError("q, k, v must have at least 16-byte pointer alignment")


def attention(q, k, v, causal=True, sm_scale=None):
    """Run fixed BF16 causal GQA64x8 prefill attention on H20 using zero-copy BHSD views."""
    _validate_inputs(q, k, v, causal, sm_scale)
    out = torch.empty_like(q, memory_format=torch.contiguous_format)
    return run_sm90_fmha_m128(
        q,
        k,
        v,
        out,
        kv_stage=_KV_STAGE,
        load_regs=_LOAD_REGS,
        mma_regs=_MMA_REGS,
    )


def get_compile_count():
    """Forward the M128 kernel's process-local cute.compile invocation count."""
    return _kernel_get_compile_count()


register(_REGISTERED_NAME, attention)
