"""Decode-only adapter for the latest WXL SM90 split-KV kernel."""

from ._wxl_cutedsl_decode_flash_attention import run_wxl_sm90_gqa_decode
from .base import register


def attention(q, k, v, causal=True, sm_scale=None):
    return run_wxl_sm90_gqa_decode(q, k, v, sm_scale=sm_scale)


register("WXL CuTe DSL SM90 split-KV decode", attention, phases=("decode",))
