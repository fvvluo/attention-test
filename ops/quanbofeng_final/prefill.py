# =============================================================================
# FlashAttention-3 / Hopper 技巧版注册层
#
# benchmark 传入 BHSD；Hopper kernel 使用 (S,D,Hr,Hkv,B) CuTe 逻辑布局。
# Hr = q_heads // kv_heads。K/V 的 Hr 维在 kernel 内以 0-stride 广播，避免
# repeat_interleave 产生实际显存拷贝。
# =============================================================================

import math

import torch

_modules = None
_compiled_cache = {}


def _get_modules():
    global _modules
    if _modules is None:
        import cuda.bindings.driver as cuda
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack
        from .prefill_kernel import (
            FlashAttention5GQACuteDSL,
            fmha_utils,
        )

        _modules = (
            cuda,
            cutlass,
            cute,
            from_dlpack,
            FlashAttention5GQACuteDSL,
            fmha_utils,
        )
    return _modules


def _to_cute(tensor, from_dlpack):
    """按实际 shape/stride 静态特化 TMA descriptor。

    编译缓存本来就按完整 shape 区分，因此无需动态 layout；静态 layout 能让
    编译器折叠更多步长、整除和边界地址计算，减少 NCU 看到的通用指令开销。
    """
    return from_dlpack(tensor, assumed_align=16)


def _make_kernel_views(q, k, v):
    """BHSD -> Hopper FMHA 所需的 5D 逻辑布局。

    Q/O: BHSD -> (B,Hkv,Hr,S,D) -> (S,D,Hr,Hkv,B)
    K/V: BHSD -> (S,D,1,Hkv,B)
    """
    batch, q_heads, q_len, head_dim = q.shape
    kv_heads, kv_len = k.shape[1], k.shape[2]
    heads_per_kv = q_heads // kv_heads

    q_5d = q.reshape(batch, kv_heads, heads_per_kv, q_len, head_dim)
    q_sdrrb = q_5d.permute(3, 4, 2, 1, 0)

    k_sdrrb = k.permute(2, 3, 1, 0).unsqueeze(2)
    v_sdrrb = v.permute(2, 3, 1, 0).unsqueeze(2)

    out_bhsd = torch.empty_like(q)
    out_5d = out_bhsd.reshape(batch, kv_heads, heads_per_kv, q_len, head_dim)
    out_sdrrb = out_5d.permute(3, 4, 2, 1, 0)

    # kernel5 编译期关闭 LSE 写回，仅需向接口提供正确的逻辑 shape。
    # 用一个标量的 0-stride view 代替 B*Hq*S 的完整 32MB LSE workspace。
    lse_scalar = torch.empty(1, device=q.device, dtype=torch.float32)
    lse_sdrrb = lse_scalar.as_strided(
        (q_len, 1, heads_per_kv, kv_heads, batch), (0, 0, 0, 0, 0)
    )

    return q_sdrrb, k_sdrrb, v_sdrrb, out_sdrrb, lse_sdrrb, out_bhsd


def attention(q, k, v, causal=True, sm_scale=None):
    """Hopper TMA + WGMMA + warp-specialized attention，支持 MHA/GQA。"""
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("kernel5 只支持 CUDA tensor")
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q/k/v 必须是 BHSD 四维 tensor")
    if k.shape != v.shape or q.shape[0] != k.shape[0]:
        raise ValueError("K/V shape 或 batch 不匹配")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("Q/K head_dim 不匹配")
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError("GQA 要求 q_heads 是 kv_heads 的整数倍")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("kernel5 仅支持 fp16/bf16；正式评测使用 bf16")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("Q/K/V dtype 必须相同")
    if q.shape[-1] not in (64, 128):
        raise ValueError("kernel5 当前只验证 head_dim=64/128")
    if causal and q.shape[2] != k.shape[2]:
        raise ValueError("causal prefill 要求 q_len == kv_len")

    major, minor = torch.cuda.get_device_capability(q.device)
    if (major, minor) != (9, 0):
        raise ValueError(f"kernel5 使用 Hopper WGMMA/TMA，需要 SM90；当前是 SM{major}{minor}")

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])

    cuda, cutlass, cute, from_dlpack, KernelClass, fmha_utils = _get_modules()

    # 保证 D 维连续；正常 benchmark 输入本来就是 contiguous。
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    q_view, k_view, v_view, o_view, lse_view, output = _make_kernel_views(q, k, v)

    q_cute = _to_cute(q_view, from_dlpack)
    k_cute = _to_cute(k_view, from_dlpack)
    v_cute = _to_cute(v_view, from_dlpack)
    o_cute = _to_cute(o_view, from_dlpack)
    # 0-stride LSE 占位视图没有 leading stride=1，且 no-LSE 编译路径不会访问它。
    lse_cute = from_dlpack(lse_view, assumed_align=16)

    q_len, kv_len, head_dim = q.shape[2], k.shape[2], q.shape[-1]
    # 固定为 A/B 实测后的 128K 配置，不再通过环境变量切换实验参数：
    # - 两个 WGMMA warp-group 各处理 64 行，CTA 合计处理 128 行 Q；
    # - N tile=128、Q/KV/epilogue pipeline=2/5/2；
    # - 本 CuTe DSL 的静态 persistent scheduler 实测略慢，因此关闭。
    tile_n = 128
    q_stage = 2
    kv_stage = 5
    epi_stage = 2
    is_persistent = False
    mma_tiler = (64, tile_n, head_dim)

    if causal:
        mask_type = fmha_utils.MaskEnum.WINDOW_MASK
        window_left = None
        window_right = cutlass.Int32(0)
    elif kv_len % mma_tiler[1] != 0:
        mask_type = fmha_utils.MaskEnum.RESIDUAL_MASK
        window_left = None
        window_right = None
    else:
        mask_type = fmha_utils.MaskEnum.WINDOW_MASK
        window_left = None
        window_right = None

    cache_key = (
        q.device.index,
        q.dtype,
        tuple(q.shape),
        tuple(k.shape),
        bool(causal),
        is_persistent,
        mma_tiler,
        q_stage,
        kv_stage,
        epi_stage,
        "overlap-v1",
    )

    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    scale = cutlass.Float32(float(sm_scale))
    scale_log2 = cutlass.Float32(float(sm_scale) * 1.4426950408889634)
    scale_output = cutlass.Float32(1.0)

    compiled = _compiled_cache.get(cache_key)
    if compiled is None:
        op = KernelClass(
            qk_acc_dtype=cutlass.Float32,
            pv_acc_dtype=cutlass.Float32,
            mma_tiler=mma_tiler,
            is_persistent=is_persistent,
            mask_type=mask_type,
            use_lpt_scheduler=False,  # A/B: LPT+L2 scheduler disabled
            store_lse=False,
            unit_output_scale=True,
            q_stage=q_stage,
            kv_stage=kv_stage,
            epi_stage=epi_stage,
        )
        compiled = cute.compile(
            op,
            q_cute,
            k_cute,
            v_cute,
            o_cute,
            lse_cute,
            scale_log2,
            scale,
            scale_output,
            window_left,
            window_right,
            stream,
        )
        _compiled_cache[cache_key] = compiled

    compiled(
        q_cute,
        k_cute,
        v_cute,
        o_cute,
        lse_cute,
        scale_log2,
        scale,
        scale_output,
        window_left,
        window_right,
        stream,
    )
    return output

