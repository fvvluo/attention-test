# ============================================================
# Prefill-optimized FlashAttention operator (CUTE DSL, Hopper SM90 / H20).
#
#   外部接口:  attention(q, k, v, causal=True, sm_scale=None) -> output
#   张量布局:  q/k/v/output 均为 (batch, heads, seq_len, head_dim)  [框架约定]
#
# 底层内核是 Dao-AILab CUTE-DSL 的 SM90 warp-specialized WGMMA 前向内核
# (TMA + WGMMA + 软件流水线 + online softmax)。全部依赖(48 个 .py)已放在
# 同目录的 ops/flash_attn/ 包内,本文件只做:
#   1) 让 ops/ 可被 import (从而 import flash_attn.cute)
#   2) 布局适配: 框架 (B,H,S,D) <-> 内核 (B,S,H,D)
#   3) GQA 展开
#   4) 以 H20 实测最优配置调用内核 (~144 TFLOPS)
#
# 运行期依赖(系统包): nvidia-cutlass-dsl==4.6.0, quack-kernels==0.6.1,
#                    torch(cu13), Hopper GPU(sm_90)。
# ============================================================

import math
import os
import sys

import torch

# ---- 让同目录下的 flash_attn 包可被 import ----
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from flash_attn.cute import interface as _fa  # noqa: E402

# H20 实测最优 SM90 prefill 配置(exhaustive SMEM-legal sweep + 波次对齐)。
_BEST_CFG = dict(
    tile_mn=(128, 128),
    mma_pv_is_rs=False,
    intra_wg_overlap=True,
    num_stages_override=2,
    num_threads=384,
    pack_gqa=False,
)


def attention(q, k, v, causal=True, sm_scale=None):
    """Prefill-optimized attention。

    Args:
        q:    (batch, q_heads,  seq_len, head_dim)  bf16/fp16 CUDA
        k, v: (batch, kv_heads, seq_len, head_dim)  q_heads 是 kv_heads 整数倍(GQA)
        causal:   是否因果掩码
        sm_scale: softmax 缩放,默认 1/sqrt(head_dim)

    Returns:
        output: (batch, q_heads, seq_len, head_dim),布局同 q
    """
    batch, q_heads, q_len, head_dim = q.shape
    kv_heads = k.shape[1]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    # GQA: 按框架约定(连续 group)把 k/v 展开到 q_heads,再喂内核。
    # (内核也原生支持 GQA,但这里显式展开以保证与 baseline 分组约定完全一致。)
    if kv_heads != q_heads:
        group = q_heads // kv_heads
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)

    # 布局适配: (B,H,S,D) -> (B,S,H,D),内核要求最后一维连续。
    q_bshd = q.transpose(1, 2).contiguous()
    k_bshd = k.transpose(1, 2).contiguous()
    v_bshd = v.transpose(1, 2).contiguous()

    out_bshd, *_ = _fa._flash_attn_fwd(
        q_bshd, k_bshd, v_bshd,
        softmax_scale=float(sm_scale),
        causal=causal,
        **_BEST_CFG,
    )

    # 转回框架布局 (B,S,H,D) -> (B,H,S,D)。
    return out_bshd.transpose(1, 2).contiguous()
