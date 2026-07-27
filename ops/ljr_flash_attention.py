import importlib
import math

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from .base import register


def _ensure_baseline_routed():
    import bench_attention  # noqa: F401
    bench_attention.get_baseline_fn()


_FA_CLS = None
_CUDA_STREAM = None
_AuxData = None


def _get_fa_cls():
    global _FA_CLS, _CUDA_STREAM, _AuxData
    if _FA_CLS is None:
        _ensure_baseline_routed()
        import cuda.bindings.driver as cuda
        from flash_attn.cute.utils import AuxData
        # 载入本仓库自己的 kernel 文件（内部会 import 已路由的 flash_attn.cute.* 叶子构件）
        import importlib.util, os
        kpath = os.path.join(os.path.dirname(__file__), "mc_attention", "flash_attention_fa3_kernel.py")
        spec = importlib.util.spec_from_file_location("ljr_fa3_kernel", kpath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _FA_CLS = mod.LjrFlashFwdSm90
        _AuxData = AuxData
        _CUDA_STREAM = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    return _FA_CLS


_compiled_cache = {}


def attention(q, k, v, causal=True, sm_scale=None):
    batch, q_heads, q_len, head_dim = q.shape
    kv_heads = k.shape[1]
    kv_len = k.shape[2]
    assert head_dim == 128, "该 FA3 kernel 固定 head_dim=128"
    assert q_heads % kv_heads == 0, "GQA: q_heads 必须是 kv_heads 整数倍"
    qhead_per_kvhead = q_heads // kv_heads
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    FA = _get_fa_cls()
    orig_dtype = q.dtype

    # 零拷贝接口：kernel 逻辑上要 (b, s, h, d)，但内部用 TMA 按 stride 读，
    # 允许任意 stride。benchmark 传入 (b, h, s, d)，head_dim 已连续；只需 transpose(1,2)
    # 得到 (b, s, h, d) 的 *view*（不 .contiguous()），省掉每次 4 次大张量拷贝。
    # dtype 已是 bf16 时不再 .to()（避免拷贝）。
    def _prep(x):
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        return x.transpose(1, 2)  # (b,h,s,d) -> (b,s,h,d) view，d 仍连续

    q_bshd = _prep(q)
    k_bshd = _prep(k)
    v_bshd = _prep(v)
    # 输出按 (b,s,h,d) 连续分配，返回时 transpose 回 (b,h,s,d)
    o_bshd = torch.empty(
        (batch, q_len, q_heads, head_dim), device=q.device, dtype=torch.bfloat16
    )

    q_t = from_dlpack(q_bshd, assumed_align=16)
    k_t = from_dlpack(k_bshd, assumed_align=16)
    v_t = from_dlpack(v_bshd, assumed_align=16)
    o_t = from_dlpack(o_bshd, assumed_align=16)

    cache_key = (qhead_per_kvhead, bool(causal), q_len, kv_len)
    entry = _compiled_cache.get(cache_key)
    if entry is None:
        fa = FA(
            cutlass.BFloat16, head_dim, head_dim, qhead_per_kvhead,
            is_causal=bool(causal), is_local=False, pack_gqa=False,
            tile_m=128, tile_n=128, num_stages=2, num_threads=256, Q_in_regs=False,
        )
        aux = _AuxData()
        compiled = cute.compile(
            fa, q_t, k_t, v_t, o_t, None, cutlass.Float32(sm_scale),
            None, None, None, None, None, None, None, None, None, aux,
            _CUDA_STREAM,
        )
        entry = (compiled, aux)
        _compiled_cache[cache_key] = entry

    compiled, aux = entry
    compiled(
        q_t, k_t, v_t, o_t, None, cutlass.Float32(sm_scale),
        None, None, None, None, None, None, None, None, None, aux,
        _CUDA_STREAM,
    )
    return o_bshd.transpose(1, 2).to(orig_dtype)  # (b,s,h,d) -> (b,h,s,d)


register("ljr_flash_attention_fa3", attention)
