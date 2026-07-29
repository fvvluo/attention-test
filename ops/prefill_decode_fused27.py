# ============================================================
# 融合算子 v27：按 q_len 路由 prefill / decode，得到完整 FlashAttention
#   q_len == 1  -> ops/_decode19.py 的 CuTe DSL flash-decoding kernel
#                  （decode9 内核 + 调参：num_splits 39->62 + num_workers=496，
#                    当前实测最优 decode，与 fused21 同款）
#   q_len  > 1  -> ops/_prefill9.py 的 Hopper SM90 FMHA kernel（CuTe DSL）
# 只做"拼接/融合"，不修改任何已有文件
#   （_prefill9.py / _prefill9_kernel.py / _prefill9_helpers.py / _decode19.py 只读）。
# 全程 BHSD：
#   q:  (b, q_heads,  q_len,  d)
#   k/v:(b, kv_heads, kv_len, d)
#   out 与 q 同 shape。q_heads 是 kv_heads 的整数倍（GQA）；MHA 时相等。
#
# 组合说明：prefill 用 v9（_prefill9），decode 用 v19（fused21 的最优 decode）。
#   - _prefill9 自带配套 helper（_prefill9_helpers.py）+ 内核（_prefill9_kernel.py），
#     dual-mode import，不依赖 baseline 的外部 helpers 包 —— 从根上避开了 prefill8
#     因外部 helper 路由不匹配导致 kernel launch 后挂死的问题（实测 prefill9 跑通）。
#   - prefill9 入口 prefill_attention(q,k,v,causal,sm_scale) 为 BSHD 布局，
#     内部自理 GQA 展开 / 布局 / 编译缓存 / LSE 跳过（return_lse=False 默认走
#     纯推理路径，跳过 epilogue 的 LSE 写回）。融合层只做 BHSD<->BSHD 转发。
#   - decode13 沿用 decode9 的 attention_decode，默认 static_softmax=False 会走
#     online-softmax 分支（TYPE_UNSTABLE_JOIN 编译问题）；这里显式传生产配置绕开。
# ============================================================

import os
import sys

import torch

from .base import register

# decode13 依赖同目录若干模块（其内部按包内相对/顶层双模式 import），确保 ops/
# 在 sys.path 上以兼容其顶层 import 回退。
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from . import _prefill9 as _prefill_mod   # noqa: E402  (自带 helper 的 prefill v9)
from . import _decode19 as _decode_mod    # noqa: E402  (CuTe DSL decode kernel v19)


# ------------------------------------------------------------
# prefill 驱动（q_len > 1）：转发到 _prefill9.prefill_attention。
#   framework BHSD (b, h, s, d) -> prefill9 BSHD (b, s, h, d)，输出转回 BHSD。
#   prefill9 内部自理 GQA / 布局 / 编译缓存；return_lse 默认 False（跳过 LSE）。
# ------------------------------------------------------------
def _run_prefill(q, k, v, causal, sm_scale):
    q_bshd = q.transpose(1, 2).contiguous()
    k_bshd = k.transpose(1, 2).contiguous()
    v_bshd = v.transpose(1, 2).contiguous()

    out_bshd = _prefill_mod.prefill_attention(
        q_bshd, k_bshd, v_bshd,
        causal=causal,
        sm_scale=sm_scale,
    )
    return out_bshd.transpose(1, 2).contiguous()


# ------------------------------------------------------------
# decode 驱动（q_len == 1）：调用 _decode19.py，固定生产配置（不走 autotune）。
#   decode19 = decode13 内核 + 换配置：把 autotune 系一直想要的 BLOCK_N=256
#   "WINNER" 配置 (num_splits=9, num_workers=78, block_n=256, stages=1) 直接固化，
#   既取大 tile 的高带宽（tile 数减半->softmax 屏障减半），又避开 autotune 抖动。
# ------------------------------------------------------------
def _run_decode(q, k, v, sm_scale):
    def _c(x):
        return x if x.is_contiguous() else x.contiguous()

    return _decode_mod.attention_decode(
        _c(q), _c(k), _c(v),
        sm_scale=sm_scale,
        causal=False,
        num_splits=9, num_workers=78, block_n=256,
        num_stages=1, num_stages_v=1, num_producer_warps=1,
        fused=False, use_pdl=True, evict_first=True, static_softmax=True,
    )


# ------------------------------------------------------------
# 统一入口：按 q_len 路由（q_len==1 -> decode，否则 prefill）
# ------------------------------------------------------------
def attention(q, k, v, causal=True, sm_scale=None):
    if q.shape[2] == 1:
        return _run_decode(q, k, v, sm_scale)
    return _run_prefill(q, k, v, causal, sm_scale)


register("prefill_decode_fused27 (cute+cute)", attention)
