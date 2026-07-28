# =============================================================================
# Decode：正式 SM90/GQA8 形状使用 cp.async/M16 warp-MMA split-KV。
# =============================================================================

import math

import torch

_modules = None
_compile_cache = {}
_workspace_cache = {}


def _get_modules():
    global _modules
    if _modules is None:
        import cuda.bindings.driver as cuda
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack
        from .decode_mma_kernel import GQADecodeCombine, GQADecodeMma

        _modules = (
            cuda,
            cutlass,
            cute,
            from_dlpack,
            GQADecodeMma,
            GQADecodeCombine,
        )
    return _modules


def _num_splits(seqlen_k, n_block=64, max_splits=256):
    num_blocks = seqlen_k // n_block
    num_splits = min(max_splits, num_blocks)
    while num_splits > 1 and num_blocks % num_splits != 0:
        num_splits -= 1
    return num_splits


def _workspace(q, num_splits):
    stream_id = int(torch.cuda.current_stream(q.device).cuda_stream)
    key = (
        q.device.index,
        stream_id,
        q.dtype,
        tuple(q.shape),
        num_splits,
    )
    buffers = _workspace_cache.get(key)
    if buffers is None:
        batch, q_heads, _, head_dim = q.shape
        partial = torch.empty(
            batch,
            q_heads,
            num_splits,
            head_dim,
            device=q.device,
            dtype=torch.bfloat16,
        )
        lse = torch.empty(
            batch,
            q_heads,
            num_splits,
            device=q.device,
            dtype=torch.float32,
        )
        output = torch.empty(
            batch,
            q_heads,
            head_dim,
            device=q.device,
            dtype=q.dtype,
        )
        buffers = (partial, lse, output)
        _workspace_cache[key] = buffers
    return buffers


def flash_decode_mma(q, k, v, scale, n_block=64, max_splits=256):
    cuda, cutlass, cute, from_dlpack, DecodeClass, CombineClass = _get_modules()
    major, minor = torch.cuda.get_device_capability(q.device)
    if (major, minor) != (9, 0):
        raise ValueError(
            f"MMA decode target requires SM90; current device is SM{major}{minor}"
        )

    batch, q_heads, _, head_dim = q.shape
    kv_heads, seqlen_k = k.shape[1], k.shape[2]
    if seqlen_k % n_block != 0:
        raise ValueError(
            f"MMA decode requires kv_len divisible by {n_block}, got {seqlen_k}"
        )
    num_splits = _num_splits(seqlen_k, n_block, max_splits)
    partial, lse, output = _workspace(q, num_splits)

    # BHSD -> BSHD is a stride-only view; never materialize a 512 MiB K/V copy.
    q_view = q.squeeze(2)
    k_view = k.transpose(1, 2)
    v_view = v.transpose(1, 2)
    q_cute = from_dlpack(q_view, assumed_align=16)
    k_cute = from_dlpack(k_view, assumed_align=16)
    v_cute = from_dlpack(v_view, assumed_align=16)
    partial_cute = from_dlpack(partial, assumed_align=16)
    lse_cute = from_dlpack(lse, assumed_align=16)
    output_cute = from_dlpack(output, assumed_align=16)

    torch_stream = torch.cuda.current_stream(q.device)
    stream_id = int(torch_stream.cuda_stream)
    stream = cuda.CUstream(stream_id)
    scale_log2 = cutlass.Float32(float(scale) * 1.4426950408889634)
    key = (
        "sm90-gqa-m16n64-qcpasync-final",
        q.device.index,
        stream_id,
        q.dtype,
        tuple(q.shape),
        tuple(k.shape),
        tuple(v.shape),
        n_block,
        num_splits,
        partial.dtype,
    )
    compiled = _compile_cache.get(key)
    if compiled is None:
        decode_op = DecodeClass(
            head_dim=head_dim,
            q_heads=q_heads,
            kv_heads=kv_heads,
            seqlen_k=seqlen_k,
            num_splits=num_splits,
            n_block=n_block,
        )
        combine_op = CombineClass(head_dim=head_dim, num_splits=num_splits)
        stage1 = cute.compile(
            decode_op,
            q_cute,
            k_cute,
            v_cute,
            partial_cute,
            lse_cute,
            scale_log2,
            stream,
        )
        combine = cute.compile(
            combine_op,
            partial_cute,
            lse_cute,
            output_cute,
            stream,
        )
        compiled = (stage1, combine)
        _compile_cache[key] = compiled

    stage1, combine = compiled
    stage1(
        q_cute,
        k_cute,
        v_cute,
        partial_cute,
        lse_cute,
        scale_log2,
        stream,
    )
    combine(partial_cute, lse_cute, output_cute, stream)
    return output.unsqueeze(2)


def decode_attention(q, k, v, causal=False, sm_scale=None):
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q/k/v 必须是 BHSD 四维 tensor")
    if q.shape[2] != 1:
        raise ValueError("decode_attention 要求 q_len == 1")
    if causal:
        raise ValueError("bench decode 的 q_len=1 应传 causal=False")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("decode 仅支持 CUDA")
    if not (q.device == k.device == v.device):
        raise ValueError("Q/K/V 必须位于同一 CUDA device")
    if not (
        q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
    ):
        raise ValueError("最终 decode 仅支持 BF16")
    if tuple(q.shape) != (1, 64, 1, 128):
        raise ValueError("最终 decode 要求 Q shape 为 (1,64,1,128)")
    if k.shape != v.shape or tuple(k.shape[:2]) != (1, 8):
        raise ValueError("最终 decode 要求 K/V shape 为 (1,8,Sk,128)")
    if k.shape[-1] != 128 or k.shape[2] <= 0 or k.shape[2] % 64 != 0:
        raise ValueError("最终 decode 要求 D=128 且 kv_len 为正的 64 倍数")
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        raise ValueError("最终 decode 要求 contiguous BHSD 输入")
    if any(tensor.data_ptr() % 16 != 0 for tensor in (q, k, v)):
        raise ValueError("最终 decode 要求 16-byte aligned 输入")

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    sm_scale = float(sm_scale)
    if not math.isfinite(sm_scale) or sm_scale <= 0.0:
        raise ValueError("sm_scale 必须是有限正数")

    # 新版：转置-WGMMA + TMA 流水实现（kv_len 为 128 的倍数时启用，更快）；
    # 否则回退到旧的 M16 warp-MMA 实现。
    if k.shape[2] % 128 == 0:
        from .decode_transposed import decode as _transposed_decode
        return _transposed_decode(q, k, v, sm_scale=sm_scale)
    return flash_decode_mma(q, k, v, sm_scale)
