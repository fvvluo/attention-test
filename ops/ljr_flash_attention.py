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


_FA_CLS = None          # prefill kernel: LjrFlashFwdSm90
_FA_DECODE_CLS = None    # decode kernel: LjrFlashFwdDecodeSm90 (pack_gqa + split-KV)
_CUDA_STREAM = None
_AuxData = None


def _load_kernel_class(filename, clsname):
    import importlib.util, os
    kpath = os.path.join(os.path.dirname(__file__), "mc_attention", filename)
    spec = importlib.util.spec_from_file_location(f"ljr_{clsname}", kpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, clsname)


def _ensure_loaded():
    global _FA_CLS, _FA_DECODE_CLS, _CUDA_STREAM, _AuxData
    if _FA_CLS is None:
        _ensure_baseline_routed()
        import cuda.bindings.driver as cuda
        from flash_attn.cute.utils import AuxData
        _FA_CLS = _load_kernel_class("flash_attention_fa3_kernel.py", "LjrFlashFwdSm90")
        _FA_DECODE_CLS = _load_kernel_class(
            "flash_attention_fa3_decode_kernel.py", "LjrFlashFwdDecodeSm90"
        )
        _AuxData = AuxData
        _CUDA_STREAM = cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def _get_fa_cls():
    _ensure_loaded()
    return _FA_CLS


def _get_decode_cls():
    _ensure_loaded()
    return _FA_DECODE_CLS


_compiled_cache = {}


_TILE_N = 128  # kernel 固定 tile_n=128


def _decode_num_splits(q_len, kv_len, kv_heads, batch, device):
    """Split-KV（FlashDecoding）启发式：仅在 decode-like（q_len 很小）场景启用。

    decode kernel 用 pack_gqa：共享同一 kv-head 的 q-head 打包进 M 维，grid 从
    q_heads 掉到 kv_heads，未 split 时只有 kv_heads*batch 个 CTA，远小于 SM 数。
    切成 num_splits 段后 grid Y 轴变 kv_heads*num_splits*batch，填满 SM 并行访存。
    """
    if q_len > 8:
        return 1
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    num_ctas_base = kv_heads * batch
    n_block_total = (kv_len + _TILE_N - 1) // _TILE_N
    if n_block_total <= 4:
        return 1
    # 目标：把总 CTA 数拉高以充分并行 KV 访存；同时每个 split 至少覆盖若干 KV block
    # （太碎会让 tail/合并开销吃掉收益）。取「每 split 覆盖约 min_blocks_per_split 个块」为上限。
    min_blocks_per_split = 16
    num_splits = max(1, n_block_total // min_blocks_per_split)
    # 不超过 KV 总块数；下限保证至少填满 ~num_sms 的 CTA
    min_splits = max(1, (num_sms + num_ctas_base - 1) // num_ctas_base)
    return max(min_splits, min(n_block_total, num_splits))


def _prep_bshd(x):
    """(b,h,s,d) -> (b,s,h,d) 的零拷贝 view；dtype 非 bf16 时才转。"""
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    return x.transpose(1, 2)


def _attention_decode(q, k, v, sm_scale, qhead_per_kvhead, num_splits):
    """decode 路径：pack_gqa（消 qhead_per_kvhead× 冗余读）+ Split-KV（填满 SM）。

    每个 (kv-head, split) 一个 CTA，输出对本 KV 段的 partial-O(bf16)+partial-LSE(fp32)，
    再在 PyTorch 里按 logsumexp 合并。
    """
    batch, q_heads, q_len, head_dim = q.shape
    kv_len = k.shape[2]
    orig_dtype = q.dtype
    FA = _get_decode_cls()

    q_t = from_dlpack(_prep_bshd(q), assumed_align=16)
    k_t = from_dlpack(_prep_bshd(k), assumed_align=16)
    v_t = from_dlpack(_prep_bshd(v), assumed_align=16)

    # 分 split 的 partial-O(bf16) + partial-LSE(fp32)。
    o_partial = torch.empty(
        (num_splits, batch, q_len, q_heads, head_dim), device=q.device, dtype=torch.bfloat16
    )
    lse_partial = torch.empty(
        (num_splits, batch, q_len, q_heads), device=q.device, dtype=torch.float32
    )
    o_t = from_dlpack(o_partial, assumed_align=16)
    lse_t = from_dlpack(lse_partial, assumed_align=4)

    cache_key = ("decode", qhead_per_kvhead, q_len, kv_len, num_splits)
    entry = _compiled_cache.get(cache_key)
    if entry is None:
        fa = FA(
            cutlass.BFloat16, head_dim, head_dim, qhead_per_kvhead,
            is_causal=False, is_local=False, pack_gqa=True,
            tile_m=128, tile_n=128, num_stages=2, num_threads=256, Q_in_regs=False,
            num_splits=num_splits,
        )
        aux = _AuxData()
        compiled = cute.compile(
            fa, q_t, k_t, v_t, o_t, lse_t, cutlass.Float32(sm_scale),
            None, None, None, None, None, None, None, None, None, aux,
            _CUDA_STREAM,
        )
        entry = (compiled, aux)
        _compiled_cache[cache_key] = entry

    compiled, aux = entry
    compiled(
        q_t, k_t, v_t, o_t, lse_t, cutlass.Float32(sm_scale),
        None, None, None, None, None, None, None, None, None, aux,
        _CUDA_STREAM,
    )

    # 合并各 split：lse = logsumexp_s(lse_s)，O = Σ_s O_s * exp(lse_s - lse)。
    lse_f = lse_partial.float()                          # (S,b,s_q,h)
    lse = torch.logsumexp(lse_f, dim=0)                  # (b,s_q,h)
    w = (lse_f - lse.unsqueeze(0)).exp()                 # (S,b,s_q,h)
    w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    o = (o_partial.float() * w.unsqueeze(-1)).sum(0)     # (b,s_q,h,d)
    o = torch.nan_to_num(o)
    return o.transpose(1, 2).to(orig_dtype)              # (b,h,s_q,d)


def _attention_prefill(q, k, v, causal, sm_scale, qhead_per_kvhead):
    """prefill 路径：原 kernel（tile_m=128 大 M，num_splits=1），行为不变。"""
    batch, q_heads, q_len, head_dim = q.shape
    kv_len = k.shape[2]
    orig_dtype = q.dtype
    FA = _get_fa_cls()

    q_t = from_dlpack(_prep_bshd(q), assumed_align=16)
    k_t = from_dlpack(_prep_bshd(k), assumed_align=16)
    v_t = from_dlpack(_prep_bshd(v), assumed_align=16)
    o_bshd = torch.empty(
        (batch, q_len, q_heads, head_dim), device=q.device, dtype=torch.bfloat16
    )
    o_t = from_dlpack(o_bshd, assumed_align=16)

    cache_key = ("prefill", qhead_per_kvhead, bool(causal), q_len, kv_len)
    entry = _compiled_cache.get(cache_key)
    if entry is None:
        fa = FA(
            cutlass.BFloat16, head_dim, head_dim, qhead_per_kvhead,
            is_causal=bool(causal), is_local=False, pack_gqa=False,
            tile_m=128, tile_n=128, num_stages=2, num_threads=256, Q_in_regs=False,
            num_splits=1,
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
    return o_bshd.transpose(1, 2).to(orig_dtype)


def attention(q, k, v, causal=True, sm_scale=None):
    batch, q_heads, q_len, head_dim = q.shape
    kv_heads = k.shape[1]
    kv_len = k.shape[2]
    assert head_dim == 128, "该 FA3 kernel 固定 head_dim=128"
    assert q_heads % kv_heads == 0, "GQA: q_heads 必须是 kv_heads 整数倍"
    qhead_per_kvhead = q_heads // kv_heads
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    num_splits = _decode_num_splits(q_len, kv_len, kv_heads, batch, q.device)
    # decode 判定：q_len 小且能有效 split（num_splits>1）时走 pack_gqa+split 的 decode kernel。
    if q_len <= 8 and num_splits > 1:
        return _attention_decode(q, k, v, sm_scale, qhead_per_kvhead, num_splits)
    return _attention_prefill(q, k, v, causal, sm_scale, qhead_per_kvhead)


register("ljr_flash_attention_fa3 (WGMMA+TMA)", attention)
