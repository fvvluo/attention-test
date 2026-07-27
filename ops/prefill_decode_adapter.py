# ============================================================
# 转接器算子：按 q_len 路由 prefill / decode
# ============================================================
#
#   q_len == 1  -> 调 ops/_decode.py 的 flash-decoding kernel（Triton）
#   q_len  > 1  -> 调 ops/_prefill.py 的 Hopper SM90 FMHA kernel（CuTe DSL）
#
# 本文件只做"拼接"：import 两个 kernel 模块并用统一的
# attention(q, k, v, causal, sm_scale) -> output 接口暴露给 benchmark，
# 不修改任何 kernel 源码（ops/_prefill.py / ops/_decode.py 保持只读）。
#
# 约定（见 bench_attention.py）：全程 BHSD
#   q:   (batch, q_heads,  q_len,  head_dim)
#   k/v: (batch, kv_heads, kv_len, head_dim)
#   out: 与 q 同 shape
#   q_heads 是 kv_heads 的整数倍（GQA）；标准 MHA 时相等。

import math
import os
import sys

import torch

from .base import register


# ------------------------------------------------------------
# 让 ops/_prefill.py 能被 import：
#   _prefill.py 里 `from . import fmha_helpers` 会失败（ops/ 下无此模块），
#   回退到顶层 `import fmha_helpers`。fmha_helpers 实际住在 baseline 的
#   CuTeDSL examples/utils 目录，这里把该目录放到 sys.path 首位。
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

# _prefill.py 的 "multi-GPU scheduler ID" 本地改动会用
# fmha_utils.compute_grid(..., device_id=...) 调用，但 baseline 自带的
# fmha_helpers.compute_grid 签名是 (o_shape, cta_tiler, is_persistent)，
# 不接受 device_id。这里给它包一层，吞掉 device_id 再转调原实现
# （单 GPU 运行下 device_id 恒为可见卡的 0，忽略语义正确）。
import fmha_helpers as _fmha_helpers  # noqa: E402

if "device_id" not in _fmha_helpers.compute_grid.__code__.co_varnames:
    _orig_compute_grid = _fmha_helpers.compute_grid

    def _compute_grid_compat(o_shape, cta_tiler, is_persistent, *args, **kwargs):
        kwargs.pop("device_id", None)
        return _orig_compute_grid(o_shape, cta_tiler, is_persistent)

    _fmha_helpers.compute_grid = _compute_grid_compat

# 现在可以安全 import 两个 kernel 模块
import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
import cutlass.torch as cutlass_torch  # noqa: E402
from cutlass.cute.runtime import from_dlpack  # noqa: E402
import cuda.bindings.driver as cuda  # noqa: E402

from . import _prefill as _prefill_mod  # noqa: E402
from . import _decode as _decode_mod  # noqa: E402


_TORCH_TO_CUTLASS = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}

_LOG2_E = 1.4426950408889634074

# 编译缓存：key -> (compiled_fmha, o_tensor, o_torch, lse_tensor,
#                   scale_softmax_log2, scale_softmax, scale_output,
#                   wsl, wsr, current_stream, b, h, s_q, d)
_PREFILL_CACHE = {}


# ------------------------------------------------------------
# 张量布局工具：把 BHSD 的 torch 张量转成 prefill kernel 期望的
# cute 张量。kernel 期望逻辑 shape (s, d, h_r, h_k, b)，物理内存等价于
# 一个 contiguous 的 (b, s, h_k, h_r, d) buffer 再 .permute(1,4,3,2,0)
# （即 _prefill.py 里 create_and_permute_tensor 的做法，leading_dim=1=d 为
#  k-major/连续维）。q 的 head 展平顺序是 (h_r, h_k)，即 qh = r*h_k + kv
# （见 _prefill.py run_torch_fmha 的 permute(4,2,3,0,1)）。
# ------------------------------------------------------------
def _make_cute_from_bhsd(x_bhsd, h_k, h_r, dtype_cutlass):
    """x_bhsd: (b, H, s, d)，H = h_r * h_k（q）或 H = h_k（k/v，h_r=1）。

    返回 (cute_tensor, torch_gpu_tensor)。torch_gpu_tensor 与 cute_tensor
    共享存储，可用于零拷贝读回输出。
    """
    b, H, s, d = x_bhsd.shape
    # harness/baseline 的 head 展平顺序是 kv-major: qh = kv*h_r + r
    # （与 baseline repeat_interleave、_decode.py 的 qh=kvh*G+r 一致）。
    # kernel create_and_permute_tensor 的逻辑 shape 是 (b, s, h_k, h_r, d)，
    # 其中 h_k 在前、h_r 在后，正好对应 kv-major，所以直接 reshape 即可。
    # (b, H, s, d) -> (b, s, H, d) -> (b, s, h_k, h_r, d)
    x = x_bhsd.permute(0, 2, 1, 3).contiguous()  # (b, s, H, d)
    x = x.view(b, s, h_k, h_r, d).contiguous()    # (b, s, h_k, h_r, d) contiguous
    # 取 create_and_permute_tensor 的逻辑视图: permute(1,4,3,2,0)
    #   (b, s, h_k, h_r, d) -> (s, d, h_r, h_k, b)
    x_view = x.permute(1, 4, 3, 2, 0)             # 逻辑 (s, d, h_r, h_k, b)

    cute_t = from_dlpack(x_view, assumed_align=16)
    cute_t.element_type = dtype_cutlass
    cute_t = cute_t.mark_layout_dynamic(leading_dim=1)
    return cute_t, x  # 返回连续 buffer x（保持引用，避免被回收）


def _alloc_output_cute(b, h_r, h_k, s_q, d, dtype_cutlass, dtype_torch, device):
    """按 (s_q, d, h_r, h_k, b) 逻辑布局分配输出 cute 张量 + 其 torch 视图。"""
    # 物理 contiguous (b, s_q, h_k, h_r, d)
    buf = torch.empty(b, s_q, h_k, h_r, d, dtype=dtype_torch, device=device)
    view = buf.permute(1, 4, 3, 2, 0)  # (s_q, d, h_r, h_k, b)
    cute_t = from_dlpack(view, assumed_align=16)
    cute_t.element_type = dtype_cutlass
    cute_t = cute_t.mark_layout_dynamic(leading_dim=1)
    return cute_t, buf


def _alloc_lse_cute(b, h_r, h_k, s_q, dtype_torch, device):
    """LSE 逻辑布局 (s_q, d=1, h_r, h_k, b)，acc dtype = fp32。"""
    buf = torch.empty(b, s_q, h_k, h_r, 1, dtype=dtype_torch, device=device)
    view = buf.permute(1, 4, 3, 2, 0)  # (s_q, 1, h_r, h_k, b)
    cute_t = from_dlpack(view, assumed_align=16)
    cute_t.element_type = cutlass.Float32
    cute_t = cute_t.mark_layout_dynamic(leading_dim=1)
    return cute_t, buf


# ------------------------------------------------------------
# prefill 驱动
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
    if dtype_cutlass is None or dtype_cutlass not in (cutlass.Float16, cutlass.BFloat16):
        raise ValueError(f"prefill 仅支持 fp16/bf16，收到 {dtype_torch}")

    device = q.device
    mma_tile_shape_mn = (64, 128)
    is_persistent = False
    qk_acc = cutlass.Float32
    pv_acc = cutlass.Float32

    key = (b, h, h_k, s_q, s_k, d, dtype_torch, bool(causal))
    cached = _PREFILL_CACHE.get(key)

    # 每次都用传入 q/k/v 重新构造输入 cute 张量（共享其存储，零拷贝）
    q_cute, _q_buf = _make_cute_from_bhsd(q, h_k, h_r, dtype_cutlass)
    k_cute, _k_buf = _make_cute_from_bhsd(k, h_k, 1, dtype_cutlass)
    v_cute, _v_buf = _make_cute_from_bhsd(v, h_k, 1, dtype_cutlass)

    if cached is None:
        # ---- 首次：构造 kernel + 编译 ----
        # mask 类型：复刻 _prefill.run 的逻辑
        window_size_left = None
        window_size_right = None
        mask_type = _fmha_helpers.MaskEnum.WINDOW_MASK
        if causal:
            window_size_right = 0
        elif window_size_left is None and window_size_right is None:
            if s_k % mma_tile_shape_mn[1] != 0:
                mask_type = _fmha_helpers.MaskEnum.RESIDUAL_MASK

        kv_stage = min(
            5,
            (232448 - 32768 - 2048) // (mma_tile_shape_mn[1] * d * 2),
        )

        fmha = _prefill_mod.HopperFusedMultiHeadAttentionForward(
            qk_acc,
            pv_acc,
            (*mma_tile_shape_mn, d),
            is_persistent,
            mask_type,
            kv_stage=kv_stage,
        )

        scale_softmax = sm_scale if sm_scale is not None else 1.0 / math.sqrt(d)
        scale_softmax = float(scale_softmax)
        scale_softmax_log2 = scale_softmax * _LOG2_E
        scale_output = 1.0  # scale_v * inv_scale_o，默认均为 1

        o_cute, o_buf = _alloc_output_cute(
            b, h_r, h_k, s_q, d, dtype_cutlass, dtype_torch, device
        )
        lse_cute, _lse_buf = _alloc_lse_cute(b, h_r, h_k, s_q, torch.float32, device)

        torch_stream = torch.cuda.current_stream()
        current_stream = cuda.CUstream(torch_stream.cuda_stream)

        wsl = None if window_size_left is None else cutlass.Int32(window_size_left)
        wsr = None if window_size_right is None else cutlass.Int32(window_size_right)

        compiled = cute.compile(
            fmha,
            q_cute, k_cute, v_cute, o_cute, lse_cute,
            scale_softmax_log2, scale_softmax, scale_output,
            wsl, wsr, current_stream,
        )

        cached = dict(
            compiled=compiled,
            o_cute=o_cute, o_buf=o_buf,
            lse_cute=lse_cute, _lse_buf=_lse_buf,
            scale_softmax_log2=scale_softmax_log2,
            scale_softmax=scale_softmax,
            scale_output=scale_output,
            wsl=wsl, wsr=wsr,
            stream=current_stream,
        )
        _PREFILL_CACHE[key] = cached

    c = cached
    c["compiled"](
        q_cute, k_cute, v_cute, c["o_cute"], c["lse_cute"],
        c["scale_softmax_log2"], c["scale_softmax"], c["scale_output"],
        c["wsl"], c["wsr"], c["stream"],
    )

    # o_buf 物理 (b, s_q, h_k, h_r, d)，逻辑输出等价 (s_q, d, h_r, h_k, b)。
    # 还原成 BHSD (b, H, s_q, d)，H 为 kv-major: qh = kv*h_r + r。
    o = c["o_buf"]  # (b, s_q, h_k, h_r, d)
    o = o.permute(0, 2, 3, 1, 4)          # (b, h_k, h_r, s_q, d)
    o = o.reshape(b, h_k * h_r, s_q, d)   # (b, H, s_q, d) kv-major
    return o.contiguous()


# ------------------------------------------------------------
# decode 驱动
# ------------------------------------------------------------
def _run_decode(q, k, v, sm_scale):
    # decode kernel 用 stride 访问，确保连续
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    return _decode_mod.flash_attention_decode(
        q, k, v,
        layout="bhsd",
        softmax_scale=sm_scale,
        return_lse=False,
    )


# ------------------------------------------------------------
# 统一入口：按 q_len 路由
# ------------------------------------------------------------
def attention(q, k, v, causal=True, sm_scale=None):
    q_len = q.shape[2]
    if q_len == 1:
        return _run_decode(q, k, v, sm_scale)
    return _run_prefill(q, k, v, causal, sm_scale)


register("prefill_decode_adapter (cute+triton)", attention)
#python3 bench_attention.py --shapes 1x64x8x131072x128 --dtype bf16 --causal --warmup 10 --iters 20 --gpu=2