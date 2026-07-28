"""Unified BHSD adapter for the fixed SM90 prefill and WXL decode kernels."""

import torch

from ._cute_dsl_sm90_fmha_perf_m128_kernel import (
    get_compile_count as _prefill_get_compile_count,
)
from ._cute_dsl_sm90_fmha_perf_m128_kernel import run_sm90_fmha_m128
from ._wxl_cutedsl_decode_flash_attention import (
    get_decode_compile_count as _decode_get_compile_count,
)
from ._wxl_cutedsl_decode_flash_attention import run_wxl_sm90_gqa_decode
from .base import register


_REGISTERED_NAME = "CuTe DSL SM90 M128 prefill + WXL split-KV decode GQA64x8"
_EXPECTED_BATCH = 1
_EXPECTED_Q_HEADS = 64
_EXPECTED_KV_HEADS = 8
_EXPECTED_SEQUENCE = 131072
_EXPECTED_HEAD_DIM = 128
_KV_STAGE = 4
_LOAD_REGS = 24
_MMA_REGS = 240
_ALIGNMENT = 16


def _validate_common_inputs(q, k, v, causal):
    if not all(isinstance(tensor, torch.Tensor) for tensor in (q, k, v)):
        raise TypeError("q, k, and v must be torch.Tensor instances")
    if not all(tensor.ndim == 4 for tensor in (q, k, v)):
        raise ValueError("q, k, and v must be rank-4 BHSD tensors")
    if not isinstance(causal, bool):
        raise TypeError("causal must be a bool")
    if not all(tensor.is_cuda for tensor in (q, k, v)):
        raise ValueError("q, k, and v must be CUDA tensors")
    if k.device != q.device or v.device != q.device:
        raise ValueError("q, k, and v must all be on the same CUDA device")
    if torch.cuda.current_device() != q.device.index:
        raise RuntimeError("the current CUDA device must match q, k, and v")

    properties = torch.cuda.get_device_properties(q.device)
    if ((properties.major, properties.minor) != (9, 0)
            or "H20" not in properties.name.upper()):
        raise RuntimeError("the unified CuTe DSL operator requires an NVIDIA H20 SM90a device")

    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("q, k, and v must all use torch.bfloat16")
    if not all(tensor.is_contiguous() for tensor in (q, k, v)):
        raise ValueError("q, k, and v must be contiguous BHSD tensors")
    if k.shape != v.shape:
        raise ValueError("k and v must have identical shapes")

    expected_kv_shape = (
        _EXPECTED_BATCH,
        _EXPECTED_KV_HEADS,
        _EXPECTED_SEQUENCE,
        _EXPECTED_HEAD_DIM,
    )
    if tuple(k.shape) != expected_kv_shape:
        raise ValueError("expected k/v BHSD shape (1, 8, 131072, 128)")
    if any(tensor.data_ptr() % _ALIGNMENT for tensor in (q, k, v)):
        raise ValueError("q, k, and v must have at least 16-byte pointer alignment")


def _validate_prefill_inputs(q, causal, sm_scale):
    expected_q_shape = (
        _EXPECTED_BATCH,
        _EXPECTED_Q_HEADS,
        _EXPECTED_SEQUENCE,
        _EXPECTED_HEAD_DIM,
    )
    if tuple(q.shape) != expected_q_shape:
        raise ValueError("expected prefill q BHSD shape (1, 64, 131072, 128)")
    if causal is not True:
        raise ValueError("the prefill kernel requires causal=True")
    if sm_scale is not None:
        raise ValueError("prefill sm_scale must be None; scale is fixed to 1/sqrt(128)")


def _validate_decode_inputs(q, causal):
    expected_q_shape = (
        _EXPECTED_BATCH,
        _EXPECTED_Q_HEADS,
        1,
        _EXPECTED_HEAD_DIM,
    )
    if tuple(q.shape) != expected_q_shape:
        raise ValueError("expected decode q BHSD shape (1, 64, 1, 128)")
    if causal is not False:
        raise ValueError("the WXL decode kernel requires causal=False")


def attention(q, k, v, causal=True, sm_scale=None):
    """Route fixed BF16 GQA64x8 prefill and WXL decode attention on H20."""
    _validate_common_inputs(q, k, v, causal)

    q_len = q.shape[2]
    if q_len == 1:
        _validate_decode_inputs(q, causal)
        return run_wxl_sm90_gqa_decode(q, k, v, sm_scale=sm_scale)
    if q_len == _EXPECTED_SEQUENCE:
        _validate_prefill_inputs(q, causal, sm_scale)
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
    raise ValueError("supported q_len values are 131072 for prefill and 1 for decode")


def get_compile_count():
    """Return total process-local prefill and WXL decode compilations."""
    return _prefill_get_compile_count() + _decode_get_compile_count()


register(_REGISTERED_NAME, attention)
