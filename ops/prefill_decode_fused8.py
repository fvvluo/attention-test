# ============================================================
# 融合算子 v8：按 q_len 路由 prefill / decode，得到完整 FlashAttention
#   q_len == 1  -> ops/_decode2.py 的 split-K flash-decoding kernel（Triton）
#   q_len  > 1  -> ops/_prefill3.py 的 Hopper SM90 FMHA kernel（CuTe DSL，
#                  新优化版 v3）
# 只做"拼接/融合"，不修改任何已有文件（_prefill3.py / _decode2.py 保持只读）。
# 全程 BHSD：
#   q:  (b, q_heads,  q_len,  d)
#   k/v:(b, kv_heads, kv_len, d)
#   out 与 q 同 shape。q_heads 是 kv_heads 的整数倍（GQA）；MHA 时相等。
# ============================================================

import math
import os
import sys

import torch

from .base import register


# ------------------------------------------------------------
# 让 ops/_prefill3.py 能 import：其 `from . import fmha_helpers` 会回退到
# 顶层 `import fmha_helpers`，该模块住在 baseline 的 CuTeDSL utils 目录。
# ------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_CUTEDSL_UTILS = os.path.join(
    _REPO_ROOT,
    "flash-attention-baseline",
    "csrc",
    "cutlass",
    "examples",
    "python",
    "CuTeDSL",
    "utils",
)
if os.path.isdir(_CUTEDSL_UTILS) and _CUTEDSL_UTILS not in sys.path:
    sys.path.insert(0, _CUTEDSL_UTILS)

# _prefill3.py 用 compute_grid(..., device_id=...) 调用，但 baseline 的
# fmha_helpers.compute_grid 签名不含 device_id。包一层吞掉它再转调原实现
# （单 GPU 下 device_id 恒为 0，忽略语义正确）。
import fmha_helpers as _fmha_helpers  # noqa: E402

if "device_id" not in _fmha_helpers.compute_grid.__code__.co_varnames:
    _orig_compute_grid = _fmha_helpers.compute_grid

    def _compute_grid_compat(o_shape, cta_tiler, is_persistent, *args, **kwargs):
        kwargs.pop("device_id", None)
        return _orig_compute_grid(o_shape, cta_tiler, is_persistent)

    _fmha_helpers.compute_grid = _compute_grid_compat

import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass.cute.runtime import from_dlpack  # noqa: E402
import cuda.bindings.driver as cuda  # noqa: E402

from . import _prefill3 as _prefill_mod  # noqa: E402  (新优化的 prefill kernel v3)
from . import _decode2 as _decode_mod  # noqa: E402  (decode kernel v2)


_TORCH_TO_CUTLASS = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}

_LOG2_E = 1.4426950408889634074

# 编译缓存：key -> dict（编译产物 + 输出/LSE cute 张量及其保活 buffer）
_PREFILL_CACHE = {}


# ------------------------------------------------------------
# 张量布局工具：BHSD torch -> prefill kernel 期望的 cute 张量。
# kernel 逻辑 shape (s, d, h_r, h_k, b)，物理等价 contiguous
# (b, s, h_k, h_r, d) 再 .permute(1,4,3,2,0)（k-major，leading_dim=1=d）。
# head 展平为 kv-major：qh = kv*h_r + r。
# ------------------------------------------------------------
def _make_cute_from_bhsd(x_bhsd, h_k, h_r, dtype_cutlass):
    b, H, s, d = x_bhsd.shape
    # (b, H, s, d) -> (b, s, H, d) contiguous -> (b, s, h_k, h_r, d)
    x = x_bhsd.permute(0, 2, 1, 3).contiguous()
    x = x.view(b, s, h_k, h_r, d)                 # 已连续，view 无拷贝
    x_view = x.permute(1, 4, 3, 2, 0)             # 逻辑 (s, d, h_r, h_k, b)

    cute_t = from_dlpack(x_view, assumed_align=16)
    cute_t.element_type = dtype_cutlass
    cute_t = cute_t.mark_layout_dynamic(leading_dim=1)
    return cute_t, x  # 保留 x 引用，避免其存储被回收


def _alloc_output_cute(b, h_r, h_k, s_q, d, dtype_cutlass, dtype_torch, device):
    """输出 cute 张量 + 其 torch 视图，物理 contiguous (b, s_q, h_k, h_r, d)。"""
    buf = torch.empty(b, s_q, h_k, h_r, d, dtype=dtype_torch, device=device)
    view = buf.permute(1, 4, 3, 2, 0)  # (s_q, d, h_r, h_k, b)
    cute_t = from_dlpack(view, assumed_align=16)
    cute_t.element_type = dtype_cutlass
    cute_t = cute_t.mark_layout_dynamic(leading_dim=1)
    return cute_t, buf


def _alloc_lse_cute(b, h_r, h_k, s_q, dtype_torch, device):
    """LSE 逻辑布局 (s_q, 1, h_r, h_k, b)，acc dtype = fp32。kernel 会写入，必须分配。"""
    buf = torch.empty(b, s_q, h_k, h_r, 1, dtype=dtype_torch, device=device)
    view = buf.permute(1, 4, 3, 2, 0)  # (s_q, 1, h_r, h_k, b)
    cute_t = from_dlpack(view, assumed_align=16)
    cute_t.element_type = cutlass.Float32
    cute_t = cute_t.mark_layout_dynamic(leading_dim=1)
    return cute_t, buf


# ------------------------------------------------------------
# prefill 驱动（q_len > 1）：调用 _prefill3.py 的 Hopper FMHA kernel
# ------------------------------------------------------------
def _run_prefill(q, k, v, causal, sm_scale):
    b, h, s_q, d = q.shape
    h_k = k.shape[1]
    s_k = k.shape[2]
    h_r = h // h_k

    if d not in (32, 64, 128, 256):
        raise ValueError(f"prefill 仅支持 head_dim ∈ {{32,64,128,256}}，收到 {d}")
    if h % h_k != 0:
        raise ValueError(f"q_heads={h} 必须是 kv_heads={h_k} 的整数倍")

    dtype_torch = q.dtype
    dtype_cutlass = _TORCH_TO_CUTLASS.get(dtype_torch)
    if dtype_cutlass not in (cutlass.Float16, cutlass.BFloat16):
        raise ValueError(f"prefill 仅支持 fp16/bf16，收到 {dtype_torch}")

    device = q.device
    mma_tile_shape_mn = (64, 128)
    is_persistent = False
    qk_acc = cutlass.Float32
    pv_acc = cutlass.Float32

    key = (b, h, h_k, s_q, s_k, d, dtype_torch, bool(causal))
    cached = _PREFILL_CACHE.get(key)

    # 每次用传入 q/k/v 重新构造输入 cute 张量（共享其存储，零拷贝）
    q_cute, _q_buf = _make_cute_from_bhsd(q, h_k, h_r, dtype_cutlass)
    k_cute, _k_buf = _make_cute_from_bhsd(k, h_k, 1, dtype_cutlass)
    v_cute, _v_buf = _make_cute_from_bhsd(v, h_k, 1, dtype_cutlass)

    if cached is None:
        # mask 类型：复刻 _prefill3.run 的逻辑
        window_size_right = None
        mask_type = _fmha_helpers.MaskEnum.WINDOW_MASK
        if causal:
            window_size_right = 0
        elif s_k % mma_tile_shape_mn[1] != 0:
            mask_type = _fmha_helpers.MaskEnum.RESIDUAL_MASK

        kv_stage = min(5, (232448 - 32768 - 2048) // (mma_tile_shape_mn[1] * d * 2))

        fmha = _prefill_mod.HopperFusedMultiHeadAttentionForward(
            qk_acc,
            pv_acc,
            (*mma_tile_shape_mn, d),
            is_persistent,
            mask_type,
            kv_stage=kv_stage,
        )

        scale_softmax = float(sm_scale) if sm_scale is not None else 1.0 / math.sqrt(d)
        scale_softmax_log2 = scale_softmax * _LOG2_E
        scale_output = 1.0

        o_cute, o_buf = _alloc_output_cute(
            b, h_r, h_k, s_q, d, dtype_cutlass, dtype_torch, device
        )
        lse_cute, lse_buf = _alloc_lse_cute(b, h_r, h_k, s_q, torch.float32, device)

        current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        wsr = None if window_size_right is None else cutlass.Int32(window_size_right)

        compiled = cute.compile(
            fmha,
            q_cute, k_cute, v_cute, o_cute, lse_cute,
            scale_softmax_log2, scale_softmax, scale_output,
            None, wsr, current_stream,
        )

        cached = dict(
            compiled=compiled,
            o_cute=o_cute, o_buf=o_buf,
            lse_cute=lse_cute, lse_buf=lse_buf,  # lse_buf 仅保活
            scale_softmax_log2=scale_softmax_log2,
            scale_softmax=scale_softmax,
            scale_output=scale_output,
            wsr=wsr,
            stream=current_stream,
        )
        _PREFILL_CACHE[key] = cached

    c = cached
    c["compiled"](
        q_cute, k_cute, v_cute, c["o_cute"], c["lse_cute"],
        c["scale_softmax_log2"], c["scale_softmax"], c["scale_output"],
        None, c["wsr"], c["stream"],
    )

    # o_buf 物理 (b, s_q, h_k, h_r, d) -> BHSD kv-major (b, h_k*h_r, s_q, d)。
    # reshape 合并 (h_k,h_r) 需物化，其结果即连续，无需再 .contiguous()。
    o = c["o_buf"].permute(0, 2, 3, 1, 4)          # (b, h_k, h_r, s_q, d)
    return o.reshape(b, h_k * h_r, s_q, d)


# ------------------------------------------------------------
# decode 驱动（q_len == 1）：调用 _decode2.py 的 split-K flash-decoding kernel
# ------------------------------------------------------------
def _run_decode(q, k, v, sm_scale):
    # 仅在非连续时才拷贝：decode kernel 按 stride 访问，只要求连续。
    # benchmark 传入的已是连续张量，无脑 .contiguous() 会多出 3 次 no-op
    # dispatch（decode 本体仅 ~0.17ms，任何固定开销占比都被放大）。
    def _c(x):
        return x if x.is_contiguous() else x.contiguous()

    return _decode_mod.flash_attention_decode(
        _c(q), _c(k), _c(v),
        layout="bhsd",
        softmax_scale=sm_scale,
        return_lse=False,
    )


# ------------------------------------------------------------
# 统一入口：按 q_len 路由（q_len==1 -> decode，否则 prefill）
# ------------------------------------------------------------
def attention(q, k, v, causal=True, sm_scale=None):
    if q.shape[2] == 1:
        return _run_decode(q, k, v, sm_scale)
    return _run_prefill(q, k, v, causal, sm_scale)


register("prefill_decode_fused8 (cute+triton)", attention)
