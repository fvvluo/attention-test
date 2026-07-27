"""Benchmark registration for Liu Xiaochen's personal SM90 attention kernel.

The actual kernel is the personal derived class
``LiuXiaochenFlashAttentionForwardSm90`` under
``ops/liuxiaochen_sm90_attention/``.  It is driven through the shared CuTe
compile/launch framework, but the compiled/executed kernel is the personal one,
NOT the baseline's ``flash_attn_func`` nor the baseline's SM90 class.

Interface: ``attention(q, k, v, causal=True, sm_scale=None)`` with BHSD in/out.
GQA (q_heads != kv_heads) is handled natively by the kernel — no
repeat_interleave, no contiguous layout copy.
"""

from .base import register


def attention(q, k, v, causal=True, sm_scale=None):
    # Imported lazily so that ops auto-scan does not trigger heavy CuTe imports
    # or baseline path routing until this op is actually invoked.
    from .liuxiaochen_sm90_attention.runner import sm90_attention_bhsd

    return sm90_attention_bhsd(q, k, v, causal=causal, sm_scale=sm_scale)


register("Liu Xiaochen SM90 baseline-derived", attention)
