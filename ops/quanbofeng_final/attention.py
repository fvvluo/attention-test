"""最终 attention 入口。

输入/输出统一为 BHSD：
- q_len == 1：Flash-Decoding split-KV decode；
- q_len > 1（正式目标 q_len=131072）：Hopper TMA/WGMMA prefill。
"""

from .decode import decode_attention
from .prefill import attention as prefill_attention


def attention(q, k, v, causal=True, sm_scale=None):
    if q.shape[2] == 1:
        # 新 token 位于 KV 序列末尾，可访问完整 KV cache，因此 decode 非因果。
        return decode_attention(q, k, v, causal=False, sm_scale=sm_scale)
    return prefill_attention(q, k, v, causal=causal, sm_scale=sm_scale)
