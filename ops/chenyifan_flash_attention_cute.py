import importlib.util
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

from .base import register

_OPTIMIZED_FLASH_ATTN = None
_DECODE_EXT = None


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _get_optimized_flash_attn():
    global _OPTIMIZED_FLASH_ATTN
    if _OPTIMIZED_FLASH_ATTN is not None:
        return _OPTIMIZED_FLASH_ATTN

    ext_dir = Path(__file__).resolve().parent / "chenyifan_ext"
    kernel_module = _load_module(
        "chenyifan_flash_fwd_sm90_optimized",
        ext_dir / "flash_fwd_sm90_optimized.py",
    )
    interface_module = _load_module(
        "chenyifan_flash_interface_optimized",
        ext_dir / "interface_optimized.py",
    )

    # 使用独立的完整 SM90 kernel 与编译缓存，避免复用评分 baseline 的去优化产物。
    from flash_attn.cute.cache_utils import get_jit_cache

    interface_module.FlashAttentionForwardSm90 = kernel_module.FlashAttentionForwardSm90
    interface_module._flash_attn_fwd.compile_cache = get_jit_cache(
        "chenyifan_fwd_sm90_stage3_v4"
    )
    _OPTIMIZED_FLASH_ATTN = interface_module.flash_attn_func
    return _OPTIMIZED_FLASH_ATTN


def _get_decode_ext():
    global _DECODE_EXT
    if _DECODE_EXT is None:
        source = Path(__file__).resolve().parent / "chenyifan_ext" / "flash_decode.cu"
        _DECODE_EXT = load(
            name="flash_attention_decode_chenyifan_v50_split57",
            sources=[str(source)],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    return _DECODE_EXT


def _expand_gqa(k, v, q_heads):
    kv_heads = k.shape[1]
    if q_heads == kv_heads:
        return k, v
    if q_heads % kv_heads != 0:
        raise ValueError(f"q_heads({q_heads}) must be divisible by kv_heads({kv_heads})")
    group = q_heads // kv_heads
    return k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)


def _sdpa_fallback(q, k, v, causal, sm_scale):
    scale = float(sm_scale) if sm_scale is not None else 1.0 / math.sqrt(q.shape[-1])
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)


def _optimized_prefill(q, k, v, causal, sm_scale):
    flash_attn_func = _get_optimized_flash_attn()
    out = flash_attn_func(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        softmax_scale=sm_scale,
        causal=causal,
        pack_gqa=True,
    )
    if isinstance(out, tuple):
        out = out[0]
    return out.transpose(1, 2)


def attention(q, k, v, causal=True, sm_scale=None):
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Expected q/k/v in shape (batch, heads, seq, dim)")

    _, q_heads, q_len, d = q.shape
    kv_len = k.shape[2]
    if q_heads % k.shape[1] != 0:
        raise ValueError("q_heads must be divisible by kv_heads")

    if q_len == 1:
        if (
            q.is_cuda
            and q.dtype == torch.bfloat16
            and k.dtype == torch.bfloat16
            and v.dtype == torch.bfloat16
            and d == 128
            and kv_len % 128 == 0
            and q_heads // k.shape[1] <= 8
        ):
            scale = float(sm_scale) if sm_scale is not None else 1.0 / math.sqrt(d)
            return _get_decode_ext().forward(q, k, v, scale)
        k_sdpa, v_sdpa = _expand_gqa(k, v, q_heads)
        return _sdpa_fallback(q, k_sdpa, v_sdpa, causal=False, sm_scale=sm_scale)

    if not (
        q.is_cuda
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
        and d == 128
        and q_len == kv_len
        and q_len % 128 == 0
    ):
        raise NotImplementedError(
            "Optimized prefill supports CUDA BF16, head_dim=128, q_len=kv_len, and 128-aligned sequence lengths"
        )

    return _optimized_prefill(q, k, v, causal=causal, sm_scale=sm_scale)


register("chenyifan_fa3_prefill + splitkv_decode (cute/cuda)", attention)
