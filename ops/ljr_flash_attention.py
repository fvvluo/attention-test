# ljr 的 tensor-core 版 FlashAttention 接入
#
# 调用 mc_attention/flash_attention_tc.py（QK^T 与 P@V 都用 warp bf16 MMA、多 warp/CTA）。
# 与 ljr_flash_attention.py 接入方式一致，只是底层 kernel 换成 tensor core 版。
# kernel 原生吃 bf16、输出 fp32；本文件负责 4D->3D、GQA repeat、编译缓存。

import importlib.util
import math
import os

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from .base import register

_fa_path = os.path.join(os.path.dirname(__file__), "mc_attention", "flash_attention_tc.py")
_spec = importlib.util.spec_from_file_location("ljr_mc_flash_attention_tc", _fa_path)
_fa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fa)

flash_attention_tc = _fa.flash_attention_tc
MAX_HEAD_DIM_TC = _fa.MAX_HEAD_DIM_TC

_compiled_cache = {}


def _get_compiled(sq, sk, sv, so, scale, causal_offset, cache_key):
    compiled = _compiled_cache.get(cache_key)
    if compiled is None:
        compiled = cute.compile(
            flash_attention_tc,
            sq, sk, sv, so,
            cutlass.Float32(scale), cutlass.Int32(causal_offset),
        )
        _compiled_cache[cache_key] = compiled
    return compiled


def attention(q, k, v, causal=True, sm_scale=None):
    batch, q_heads, q_len, head_dim = q.shape
    kv_heads = k.shape[1]
    kv_len = k.shape[2]

    assert head_dim <= MAX_HEAD_DIM_TC, f"head_dim={head_dim} 超过 TC kernel 上限 {MAX_HEAD_DIM_TC}"
    assert head_dim % 16 == 0, "TC kernel 要求 head_dim 是 16 的整数倍"

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    if kv_heads != q_heads:
        group = q_heads // kv_heads
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)

    orig_dtype = q.dtype

    # kernel 原生吃 bf16（tensor core 输入），合并 batch*heads 到 BH，行连续。
    q3 = q.reshape(batch * q_heads, q_len, head_dim).contiguous().to(torch.bfloat16)
    k3 = k.reshape(batch * q_heads, kv_len, head_dim).contiguous().to(torch.bfloat16)
    v3 = v.reshape(batch * q_heads, kv_len, head_dim).contiguous().to(torch.bfloat16)
    out3 = torch.empty(batch * q_heads, q_len, head_dim, device=q.device, dtype=torch.float32)

    q_t = from_dlpack(q3, assumed_align=16)
    k_t = from_dlpack(k3, assumed_align=16)
    v_t = from_dlpack(v3, assumed_align=16)
    out_t = from_dlpack(out3, assumed_align=16)

    causal_offset = (kv_len - q_len) if causal else kv_len

    cache_key = (batch * q_heads, q_len, kv_len, head_dim)
    compiled = _get_compiled(q_t, k_t, v_t, out_t, sm_scale, causal_offset, cache_key)
    compiled(q_t, k_t, v_t, out_t, cutlass.Float32(sm_scale), cutlass.Int32(causal_offset))

    return out3.reshape(batch, q_heads, q_len, head_dim).to(orig_dtype)


register("ljr_flash_attention_tc (tensor core)", attention)
