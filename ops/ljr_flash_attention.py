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


# decode kernel 的最优 tile 配置（经扫描）：
#   tile_m=64  -> 单 MMA warpgroup(128 线程)，sQ/sO 减半，smem 从 192KB->96KB，
#                 让每 SM 能驻留 2 个 CTA，大幅提升 occupancy 与访存隐藏；
#   tile_n=64  -> sK/sV 减半，进一步压 smem 到 2 CTA/SM。
# 与 tile_m=128/tile_n=128 相比该组合在本 shape 快约 1.8x（0.67ms -> 0.375ms）。
_DECODE_TILE_M = 64
_DECODE_TILE_N = 32   # tile_n=32 -> smem 更小，每 SM 可驻留 3 个 CTA（tile_n=64 只有 2）
_DECODE_NUM_THREADS = 128
_DECODE_BLOCKS_PER_SPLIT = 16  # 每 split 覆盖的 KV block 数（tile_n=32 时 -> ~256 splits）
# warp specialization：True 时额外加一个 producer warpgroup（总 256 线程），
# producer 专发 K/V TMA、consumer 专做 MMA，解耦访存与计算以隐藏延迟。
_DECODE_WARP_SPECIALIZED = False


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
    n_block_total = (kv_len + _DECODE_TILE_N - 1) // _DECODE_TILE_N
    if n_block_total <= 4:
        return 1
    # 每个 split 覆盖约 _DECODE_BLOCKS_PER_SPLIT 个 KV block（太碎会让 tail/合并开销吃掉收益）。
    num_splits = max(1, n_block_total // _DECODE_BLOCKS_PER_SPLIT)
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
    # 注意 LSE 用 (ns, b, h, s_q)：head 在 seqlen 之前，与 baseline 的
    # lse_shape=(b,h,s_q) 约定一致，kernel 里 pack_gqa 的 head_idx=1 才对得上。
    o_partial = torch.empty(
        (num_splits, batch, q_len, q_heads, head_dim), device=q.device, dtype=torch.bfloat16
    )
    lse_partial = torch.empty(
        (num_splits, batch, q_heads, q_len), device=q.device, dtype=torch.float32
    )
    o_t = from_dlpack(o_partial, assumed_align=16)
    lse_t = from_dlpack(lse_partial, assumed_align=4)

    cache_key = ("decode", qhead_per_kvhead, q_len, kv_len, num_splits, _DECODE_WARP_SPECIALIZED)
    entry = _compiled_cache.get(cache_key)
    if entry is None:
        fa = FA(
            cutlass.BFloat16, head_dim, head_dim, qhead_per_kvhead,
            is_causal=False, is_local=False, pack_gqa=True,
            tile_m=_DECODE_TILE_M, tile_n=_DECODE_TILE_N, num_stages=2,
            num_threads=(_DECODE_NUM_THREADS + 128) if _DECODE_WARP_SPECIALIZED else _DECODE_NUM_THREADS,
            Q_in_regs=False,
            num_splits=num_splits,
            warp_specialized=_DECODE_WARP_SPECIALIZED,
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
    # lse_partial 是 (S,b,h,s_q)，转成与 o_partial (S,b,s_q,h) 对齐的 (S,b,s_q,h)。
    # 空 split 的 lse=-inf -> w=exp(-inf-lse)=0（合法）；对 w 兜底防极端全 -inf 行 nan。
    lse_f = lse_partial.float().transpose(2, 3)          # (S,b,s_q,h)
    lse = torch.logsumexp(lse_f, dim=0)                  # (b,s_q,h)
    w = (lse_f - lse.unsqueeze(0)).exp()                 # (S,b,s_q,h)
    w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    # einsum 单 kernel 做加权求和；端到端里其 launch 与主 kernel 重叠，略优于 (op*w).sum(0)。
    o = torch.einsum('sbqh,sbqhd->bqhd', w, o_partial.float())  # (b,s_q,h,d)
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


_DECODE_CUDA_EXT = None


def _get_decode_cuda_ext():
    """惰性编译手写 CUDA decode kernel（mma.m16n8k16，KV token 放 M 维）。"""
    global _DECODE_CUDA_EXT
    if _DECODE_CUDA_EXT is None:
        import os
        from torch.utils.cpp_extension import load
        src = os.path.join(os.path.dirname(__file__), "mc_attention", "flash_attention_decode.cu")
        _DECODE_CUDA_EXT = load(
            name="ljr_flash_decode_cuda",
            sources=[src],
            verbose=False,
            extra_cuda_cflags=["-O3", "--use_fast_math",
                               "-gencode=arch=compute_90a,code=sm_90a"],
        )
    return _DECODE_CUDA_EXT


def attention(q, k, v, causal=True, sm_scale=None):
    batch, q_heads, q_len, head_dim = q.shape
    kv_heads = k.shape[1]
    kv_len = k.shape[2]
    assert head_dim == 128, "该 FA3 kernel 固定 head_dim=128"
    assert q_heads % kv_heads == 0, "GQA: q_heads 必须是 kv_heads 整数倍"
    qhead_per_kvhead = q_heads // kv_heads
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    # decode（q_len==1）优先走手写 CUDA kernel（mma.m16n8k16：KV token 放 M 维、q-head 放 N 维，
    # split-KV + 1 warp/CTA），比 WGMMA/GEMV 都快。约束：D=128、kv_len%32==0、qpkv<=8。
    if q_len == 1 and kv_len % 32 == 0 and qhead_per_kvhead <= 8:
        return _get_decode_cuda_ext().forward(q, k, v, float(sm_scale))

    num_splits = _decode_num_splits(q_len, kv_len, kv_heads, batch, q.device)
    if q_len <= 8 and num_splits > 1:
        return _attention_decode(q, k, v, sm_scale, qhead_per_kvhead, num_splits)
    return _attention_prefill(q, k, v, causal, sm_scale, qhead_per_kvhead)


register("ljr_flash_attention_fa3 (WGMMA+TMA)", attention)
