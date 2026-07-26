from .base import register

from .implement_attention_optimized2 import qwen3_attention


def attention(q, k, v, causal=True, sm_scale=None):
    return qwen3_attention(
        q,
        k,
        v,
        causal=causal,
        sm_scale=sm_scale,
    )


register("yinjun Hopper CuTe DSL optimized II", attention)
