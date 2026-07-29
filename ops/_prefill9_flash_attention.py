# ============================================================
# prefill9 接入算子：把 ops/_prefill9.py 的 prefill_attention 接入 benchmark。
#   _prefill9 是 prefill-only 的 FA3 风格 Hopper SM90 内核
#   （warp-specialized TMA+WGMMA，online-softmax，BF16/FP16/FP8），
#   入口布局 BSHD：q (B,S,H,D)、k/v (B,S,H_k,D)，返回 (B,S,H,D)。
#
# 本框架约定 BHSD：q (B,H,S,D)、k/v (B,H_k,S,D)，输出同 q。这里只做布局
# 适配 + 注册。文件名不以 `_` 开头，会被 ops 扫描器自动 import 并注册。
# ============================================================

import math

import torch

from .base import register
from . import _prefill9 as _p9


def attention(q, k, v, causal=True, sm_scale=None):
    # q: (B, H, S, D)  k/v: (B, H_k, S, D)  —— 框架 BHSD 约定
    b, h, s_q, d = q.shape
    h_k = k.shape[1]

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d)

    # decode（q_len==1）不是 prefill9 的目标场景，走 SDPA 兜底。
    if s_q == 1:
        rep = h // h_k
        kk = k.repeat_interleave(rep, dim=1)
        vv = v.repeat_interleave(rep, dim=1)
        return torch.nn.functional.scaled_dot_product_attention(
            q, kk, vv, scale=sm_scale, is_causal=False
        )

    # BHSD -> BSHD（prefill9 期望的布局），最后一维 D 连续。
    q_bshd = q.transpose(1, 2).contiguous()
    k_bshd = k.transpose(1, 2).contiguous()
    v_bshd = v.transpose(1, 2).contiguous()

    out_bshd = _p9.prefill_attention(
        q_bshd, k_bshd, v_bshd,
        causal=causal,
        sm_scale=float(sm_scale),
    )

    # BSHD -> BHSD
    return out_bshd.transpose(1, 2).contiguous()


register("prefill9 (cute FA3)", attention)
