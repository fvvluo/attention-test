# ============================================================
# prefill7 接入算子：把 ops/_prefill7.py 的 prefill_attention 接入 benchmark。
#   _prefill7.prefill_attention 是 prefill-only 的 FA3 风格 Hopper SM90 内核
#   （warp-specialized TMA+WGMMA，online-softmax，FP16/BF16/FP8），
#   入口布局为 BSHD：q (B,S,H,D)、k/v (B,S,H_k,D)，返回 (B,S,H,D)。
#
# 本框架约定 BHSD：q (B,H,S,D)、k/v (B,H_k,S,D)，输出同 q。这里只做布局
# 适配 + 注册，不改任何已有文件（_prefill7.py / _fmha_prefill_kernel.py 只读）。
#
# 说明：prefill7 是 prefill 专用内核，对 decode（q_len==1）没有优化；这里
# 若遇到 q_len==1 直接用 SDPA 兜底，评测重点是 prefill（q_len>1）路径。
# ============================================================

import math

import torch

from .base import register
from . import _prefill7 as _p7


def attention(q, k, v, causal=True, sm_scale=None):
    # q: (B, H, S, D)  k/v: (B, H_k, S, D)  —— 框架 BHSD 约定
    b, h, s_q, d = q.shape
    h_k = k.shape[1]

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d)

    # decode（q_len==1）不是 prefill7 的目标场景，走 SDPA 兜底。
    if s_q == 1:
        rep = h // h_k
        kk = k.repeat_interleave(rep, dim=1)
        vv = v.repeat_interleave(rep, dim=1)
        return torch.nn.functional.scaled_dot_product_attention(
            q, kk, vv, scale=sm_scale, is_causal=False
        )

    # BHSD -> BSHD（prefill7 期望的布局），最后一维 D 连续。
    q_bshd = q.transpose(1, 2).contiguous()
    k_bshd = k.transpose(1, 2).contiguous()
    v_bshd = v.transpose(1, 2).contiguous()

    out_bshd = _p7.prefill_attention(
        q_bshd, k_bshd, v_bshd,
        causal=causal,
        sm_scale=float(sm_scale),
    )

    # BSHD -> BHSD
    return out_bshd.transpose(1, 2).contiguous()


register("prefill7 (cute FA3)", attention)
