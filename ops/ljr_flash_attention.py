# ljr 的 FlashAttention 算子接入
#
# 直接复用 mc_attention/flash_attention.py 里的 CuTe tiled kernel
# （warp-per-row + shared memory 复用 K/V 的 online-softmax FlashAttention）。
# 该 kernel 内部按 3D (BH, seq, head_dim)、fp32 组织，本文件负责把 benchmark
# 传入的 4D (batch, heads, seq, head_dim) / bf16·fp16 张量转换过去，处理 GQA，
# 编译并调用 kernel，再把结果 reshape / 转回原 dtype 返回。

import importlib.util
import math
import os

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from .base import register

# 直接按文件路径加载 mc_attention/flash_attention.py，绕开 mc_attention/__init__.py
# （其内部为绝对 import，作为顶层包导入会 ModuleNotFoundError）。这样只依赖目标
# kernel 文件本身，不牵连同目录下其它模块，也无需改动 mc_attention 包。
_fa_path = os.path.join(os.path.dirname(__file__), "mc_attention", "flash_attention.py")
_spec = importlib.util.spec_from_file_location("ljr_mc_flash_attention", _fa_path)
_fa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fa)

flash_attention_tiled = _fa.flash_attention_tiled
MAX_HEAD_DIM = _fa.MAX_HEAD_DIM


# 按 (BH, q_len, kv_len, head_dim) 缓存编译产物：
# benchmark 的 warmup + iters 会用同一形状反复调用，缓存避免每次重新编译，
# 否则长序列（128K）下重复 JIT 会把测速拖垮。
# 因果与否只影响运行期传入的 causal_offset（不改变编译产物），故不进 key。
_compiled_cache = {}


def _get_compiled(sample_q, sample_k, sample_v, sample_out, scale, causal_offset, cache_key):
    compiled = _compiled_cache.get(cache_key)
    if compiled is None:
        compiled = cute.compile(
            flash_attention_tiled,
            sample_q, sample_k, sample_v, sample_out,
            cutlass.Float32(scale), cutlass.Int32(causal_offset),
        )
        _compiled_cache[cache_key] = compiled
    return compiled


def attention(q, k, v, causal=True, sm_scale=None):
    """CuTe tiled FlashAttention（复用 mc_attention 的 kernel）。

    Args:
        q: (batch, q_heads, seq_len, head_dim)
        k, v: (batch, kv_heads, seq_len, head_dim)，GQA 时 q_heads 是 kv_heads 整数倍
        causal: 是否使用因果掩码
        sm_scale: softmax 缩放系数，None 时默认 1/sqrt(head_dim)

    Returns:
        output: 与 q 同形 (batch, q_heads, seq_len, head_dim)
    """
    batch, q_heads, q_len, head_dim = q.shape
    kv_heads = k.shape[1]
    kv_len = k.shape[2]

    assert head_dim <= MAX_HEAD_DIM, (
        f"head_dim={head_dim} 超过 kernel 上限 {MAX_HEAD_DIM}"
    )

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    # GQA：把 k/v 的 head 维按 group 展开到 q_heads，与常见 repeat_interleave 约定一致。
    if kv_heads != q_heads:
        group = q_heads // kv_heads
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)

    orig_dtype = q.dtype

    # kernel 走 fp32；合并 batch*heads 到最外层 BH，行连续，用 4B 对齐包 dlpack。
    q3 = q.reshape(batch * q_heads, q_len, head_dim).contiguous().float()
    k3 = k.reshape(batch * q_heads, kv_len, head_dim).contiguous().float()
    v3 = v.reshape(batch * q_heads, kv_len, head_dim).contiguous().float()
    out3 = torch.empty_like(q3)

    q_t = from_dlpack(q3, assumed_align=4)
    k_t = from_dlpack(k3, assumed_align=4)
    v_t = from_dlpack(v3, assumed_align=4)
    out_t = from_dlpack(out3, assumed_align=4)

    # 因果：query 行绝对位置 = query_row + (kv_len - q_len)，key_row 超过它就屏蔽。
    # 非因果：把 offset 设成 >= kv_len，令 kernel 里的 key_row <= query_row+offset
    # 恒成立，等价于全部 key 可见（无需单独的 causal 分支）。
    if causal:
        causal_offset = kv_len - q_len
    else:
        causal_offset = kv_len

    cache_key = (batch * q_heads, q_len, kv_len, head_dim)
    compiled = _get_compiled(
        q_t, k_t, v_t, out_t, sm_scale, causal_offset, cache_key
    )
    compiled(
        q_t, k_t, v_t, out_t,
        cutlass.Float32(sm_scale), cutlass.Int32(causal_offset),
    )

    output = out3.reshape(batch, q_heads, q_len, head_dim).to(orig_dtype)
    return output


register("ljr_flash_attention", attention)
