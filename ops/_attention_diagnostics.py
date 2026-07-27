# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Correctness, validation, and benchmark diagnostics for the Qwen3 Hopper attention kernels.

This module hosts everything that is not part of the production operator path:
the generic FMHA test runner, PyTorch/FA3 reference checks, CUDA-event
benchmarks, and the command-line interfaces.  ``implement_attention.py`` (the
frozen original baseline) and ``implement_attention_optimized.py`` import it
lazily from their ``main()`` entry points, so importing a production API never
pays for -- or traces -- any of this code.  Functions take the implementation
module as ``backend`` so a single copy serves both files without sharing or
subclassing their CuTe kernels.
"""

import argparse
import math
import sys
import time
from typing import Optional, Sequence, Tuple, Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


def _is_optimized(backend) -> bool:
    return hasattr(backend, "_resolve_prefill_config")


def _cuda_median_us(invoke, warmup: int, iterations: int):
    """Run ``invoke`` and return ``(median latency in us, last output)``."""
    import torch

    output = None
    for _ in range(warmup):
        output = invoke()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = invoke()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
    samples.sort()
    midpoint = len(samples) // 2
    median = (
        samples[midpoint]
        if len(samples) % 2
        else (samples[midpoint - 1] + samples[midpoint]) / 2
    )
    return median, output


def run(
    backend,
    q_shape: Tuple[int, int, int, int],
    k_shape: Tuple[int, int, int, int],
    in_dtype: Type[cutlass.Numeric],
    out_dtype: Type[cutlass.Numeric],
    qk_acc_dtype: Type[cutlass.Numeric],
    pv_acc_dtype: Type[cutlass.Numeric],
    mma_tiler_mn: Tuple[int, int],
    is_persistent: bool,
    is_causal: bool,
    bottom_right_align: bool,
    scale_q: float,
    scale_k: float,
    scale_v: float,
    inv_scale_o: float,
    scale_softmax: float,
    window_size: Tuple[int, int],
    tolerance: float,
    warmup_iterations: int,
    iterations: int,
    skip_ref_check: bool,
    use_cold_l2: bool = False,
    execute_benchmark: bool = True,
    input_pattern: str = "random",
    kv_stage: int = 5,
    write_lse: Optional[bool] = None,
    **kwargs,
):
    """Execute the FMHA kernel on Hopper and validate/benchmark it.

    ``backend`` is the implementation module providing the kernel class and
    helpers.  The original baseline kernel takes no ``kv_stage``/``write_lse``
    constructor arguments and always writes LSE; the optimized kernel honors
    both (defaulting to no LSE) so diagnostics match each production path.
    """
    import torch
    import cutlass.torch as cutlass_torch

    fmha_utils = backend.fmha_utils
    optimized = _is_optimized(backend)
    if write_lse is None:
        write_lse = not optimized

    print("Running Hopper SM90 FMHA test with:")
    print(f"  q_shape: {q_shape}")
    print(f"  k_shape: {k_shape}")
    print(f"  in_dtype: {in_dtype}")
    print(f"  out_dtype: {out_dtype}")
    print(f"  qk_acc_dtype: {qk_acc_dtype}")
    print(f"  pv_acc_dtype: {pv_acc_dtype}")
    print(f"  mma_tiler_mn: {mma_tiler_mn}")
    print(f"  is_persistent: {is_persistent}")
    print(f"  is_causal: {is_causal}")
    print(f"  bottom_right_align: {bottom_right_align}")
    print(f"  scale_q: {scale_q}")
    print(f"  scale_k: {scale_k}")
    print(f"  scale_v: {scale_v}")
    print(f"  inv_scale_o: {inv_scale_o}")
    print(f"  scale_softmax: {scale_softmax}")
    print(f"  window_size: {window_size}")
    print(f"  tolerance: {tolerance}")
    print(f"  skip_ref_check: {skip_ref_check}")
    print(f"  use_cold_l2: {use_cold_l2}")
    print(f"  kv_stage: {kv_stage}")
    print(f"  write_lse: {write_lse}")

    # Prepare pytorch tensors: Q, K, V (random from 0 to 2) and O (all zero)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")
    torch.cuda.reset_peak_memory_stats()

    ret, msg = backend.HopperFusedMultiHeadAttentionForward.can_implement(
        q_shape,
        k_shape,
        in_dtype,
        out_dtype,
        qk_acc_dtype,
        pv_acc_dtype,
        mma_tiler_mn,
        is_persistent,
        scale_softmax,
        window_size,
        iterations,
    )
    if not ret:
        raise TypeError(msg)

    # Unpack parameters
    b, s_q, h, d = q_shape
    b_, s_k, h_k, d_ = k_shape
    window_size_left, window_size_right = window_size
    if window_size_left == -1:
        window_size_left = None
    if window_size_right == -1:
        window_size_right = None

    h_r = h // h_k

    torch.manual_seed(1111)

    def create_and_permute_tensor(
        b, s, h_k, h_r, d, dtype, is_dynamic_layout=True, tensor_name=""
    ):
        # (b, s, h_k, h_r, d) -> (s, d, h_r, h_k, b)
        # torch SPDA order is (h_k, h_r), then kernel is (h_r, h_k)
        shape = (b, s, h_k, h_r, d)
        permute_order = (1, 4, 3, 2, 0)
        is_fp8 = dtype in {cutlass.Float8E4M3FN}
        leading_dim = 1
        if is_fp8 and tensor_name == "v":
            permute_order = (4, 1, 3, 2, 0)
            leading_dim = 0
            shape = (b, d, h_k, h_r, s)

        # torch does not support fp8 type
        torch_dtype = cutlass.torch.dtype(dtype) if not is_fp8 else torch.int8

        # Create dtype torch tensor (cpu)
        torch_tensor_cpu = cutlass_torch.create_and_permute_torch_tensor(
            shape,
            torch_dtype,
            permute_order=permute_order,
            init_type=cutlass.torch.TensorInitType.RANDOM,
            init_config=cutlass.torch.RandomInitConfig(
                min_val=-2,
                max_val=2,
            ),
        )
        # Create dtype torch tensor (gpu)
        torch_tensor_gpu = torch_tensor_cpu.cuda()

        f32_torch_tensor = None
        if not skip_ref_check or is_fp8:
            f32_torch_tensor = torch_tensor_cpu.to(dtype=torch.float32)

        # BF16 tensors already have the desired values and can be consumed directly.
        cute_tensor = from_dlpack(torch_tensor_gpu, assumed_align=16)
        cute_tensor.element_type = dtype
        if is_dynamic_layout:
            cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)
        if f32_torch_tensor is not None:
            cute_tensor = cutlass_torch.convert_cute_tensor(
                f32_torch_tensor,
                cute_tensor,
                dtype,
                is_dynamic_layout=is_dynamic_layout,
            )

        return f32_torch_tensor, cute_tensor, torch_tensor_gpu

    q_ref, q_tensor, q_torch = create_and_permute_tensor(
        b, s_q, h_k, h_r, d, in_dtype, is_dynamic_layout=True
    )
    k_ref, k_tensor, k_torch = create_and_permute_tensor(
        b, s_k, h_k, 1, d, in_dtype, is_dynamic_layout=True
    )
    v_ref, v_tensor, v_torch = create_and_permute_tensor(
        b, s_k, h_k, 1, d, in_dtype, is_dynamic_layout=True, tensor_name="v"
    )
    o_ref, o_tensor, o_torch = create_and_permute_tensor(
        b, s_q, h_k, h_r, d, out_dtype, is_dynamic_layout=True
    )
    if write_lse:
        lse_ref, lse_tensor, lse_torch = create_and_permute_tensor(
            b, s_q, h_k, h_r, 1, qk_acc_dtype, is_dynamic_layout=True
        )
    else:
        # CuTe currently requires the stable tensor argument in the callable
        # signature.  The no-LSE specialization never dereferences this single
        # FP32 element.
        lse_ref = None
        lse_torch = torch.empty((1, 1, 1, 1, 1), dtype=torch.float32, device="cuda")
        lse_tensor = from_dlpack(lse_torch, assumed_align=16)
        lse_tensor.element_type = cutlass.Float32
        lse_tensor = lse_tensor.mark_layout_dynamic(leading_dim=1)

    if input_pattern not in {"random", "prefix"}:
        raise ValueError(f"unsupported input pattern: {input_pattern}")
    if input_pattern == "prefix":
        if not skip_ref_check:
            raise ValueError("prefix input is only used by the full-length sampled check")
        q_torch.zero_()
        k_torch.zero_()
        positions = (torch.arange(s_k, device=v_torch.device) % 17 - 8).float() / 8
        kv_offsets = torch.arange(h_k, device=v_torch.device).float() / 4
        values = positions[:, None, None, None, None] + kv_offsets[None, None, None, :, None]
        v_torch.copy_(values.expand_as(v_torch))
        o_torch.zero_()
        if write_lse:
            lse_torch.zero_()

    mma_tiler = (*mma_tiler_mn, d)

    mask_type = fmha_utils.MaskEnum.WINDOW_MASK
    if bottom_right_align:
        mask_type = fmha_utils.MaskEnum.WINDOW_MASK_INFERENCE
    if is_causal:
        window_size_right = 0
    elif window_size_left is None and window_size_right is None:
        if s_k % mma_tiler_mn[1] != 0:
            mask_type = fmha_utils.MaskEnum.RESIDUAL_MASK

    # To avoid mask out the whole row which results in NaN in softmax
    def check_seqlen_valid(
        s_q, s_k, window_size_left, window_size_right, bottom_right_align
    ):
        for i in range(s_q):
            offset = 0 if not bottom_right_align else s_k - s_q

            s_q_start = 0 if window_size_left is None else i + offset - window_size_left
            s_q_end = (
                s_q if window_size_right is None else i + offset + window_size_right
            )
            s_q_min = max(s_q_start, 0)
            s_q_max = min(s_q_end, s_k)

            if s_q_max - s_q_min == 0 and (i != 0 and i != s_q - 1):
                return False
        return True

    need_check_seqlen_valid = (
        window_size_left is not None or window_size_right is not None
    )
    if need_check_seqlen_valid and not check_seqlen_valid(
        s_q,
        s_k,
        window_size_left,
        window_size_right,
        bottom_right_align,
    ):
        raise ValueError("sliding window doesn't support current setting")

    kernel_kwargs = (
        {"kv_stage": kv_stage, "write_lse": write_lse} if optimized else {}
    )
    fmha = backend.HopperFusedMultiHeadAttentionForward(
        qk_acc_dtype,
        pv_acc_dtype,
        mma_tiler,
        is_persistent,
        mask_type,
        **kernel_kwargs,
    )

    # Get current CUDA stream from PyTorch
    torch_stream = torch.cuda.current_stream()
    # Get the raw stream pointer as a CUstream
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    if scale_softmax == 0.0:  # default to 1/sqrt(head_dim)
        scale_softmax = 1.0 / math.sqrt(q_shape[3])

    scale_softmax = scale_q * scale_k * scale_softmax

    LOG2_E = 1.4426950408889634074
    scale_softmax_log2 = scale_softmax * LOG2_E
    scale_output = scale_v * inv_scale_o

    print("Compiling kernel with cute.compile ...")
    start_time = time.time()
    # compile fmha kernel
    compiled_fmha = cute.compile(
        fmha,
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        lse_tensor,
        scale_softmax_log2,
        scale_softmax,
        scale_output,
        (
            window_size_left
            if window_size_left is None
            else cutlass.Int32(window_size_left)
        ),
        (
            window_size_right
            if window_size_right is None
            else cutlass.Int32(window_size_right)
        ),
        current_stream,
    )
    compilation_time = time.time() - start_time
    print(f"Compilation time: {compilation_time:.4f} seconds")

    def invoke_compiled():
        compiled_fmha(
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            lse_tensor,
            scale_softmax_log2,
            scale_softmax,
            scale_output,
            (
                window_size_left
                if window_size_left is None
                else cutlass.Int32(window_size_left)
            ),
            (
                window_size_right
                if window_size_right is None
                else cutlass.Int32(window_size_right)
            ),
            current_stream,
        )

    def run_reference(q, k, v):
        from reference_attention import attention_reference

        if window_size_left is not None or window_size_right not in {None, 0}:
            raise ValueError("the project reference only supports dense causal attention")
        s_q_ref, d_ref, h_r_ref, h_k_ref, b_ref = q.shape
        s_k_ref = k.shape[0]
        q_bhsd = q.permute(4, 3, 2, 0, 1).contiguous().view(
            b_ref, h_k_ref * h_r_ref, s_q_ref, d_ref
        ).to(device=q_torch.device)
        k_bhsd = k.permute(4, 3, 2, 0, 1).contiguous().view(
            b_ref, h_k_ref, s_k_ref, d_ref
        ).to(device=q_torch.device)
        v_bhsd = v.permute(4, 3, 2, 0, 1).contiguous().view(
            b_ref, h_k_ref, s_k_ref, d_ref
        ).to(device=q_torch.device)
        output_bhsd = attention_reference(
            q_bhsd,
            k_bhsd.repeat_interleave(h_r_ref, dim=1),
            v_bhsd.repeat_interleave(h_r_ref, dim=1),
            causal=is_causal,
            softmax_scale=scale_softmax,
        )
        return (
            output_bhsd.view(b_ref, h_k_ref, h_r_ref, s_q_ref, d_ref)
            .permute(3, 4, 2, 1, 0)
            .contiguous()
            * scale_output
        )

    if not skip_ref_check:
        invoke_compiled()

        print("Verifying results with reference_attention.py...")
        o_ref = run_reference(q_ref, k_ref, v_ref)

        o_fp32_torch = o_torch.float()
        ref_o_f32_torch = o_ref.float()

        error = (o_fp32_torch - ref_o_f32_torch).abs()
        print(
            f"Reference error: max_abs={error.max().item():.6g}, "
            f"mean_abs={error.mean().item():.6g}"
        )
        torch.testing.assert_close(
            o_fp32_torch, ref_o_f32_torch, atol=tolerance, rtol=2e-02
        )
        print("Results verified successfully!")

    if not execute_benchmark:
        if skip_ref_check:
            invoke_compiled()
        torch.cuda.synchronize()
        return {
            "time_us": None,
            "compilation_time_s": compilation_time,
            "q": q_torch,
            "k": k_torch,
            "v": v_torch,
            "output": o_torch,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        }

    if use_cold_l2:
        raise ValueError("cold-L2 benchmarking is not supported by this fixed workload")
    exec_time, _ = _cuda_median_us(invoke_compiled, warmup_iterations, iterations)

    return {
        "time_us": exec_time,
        "compilation_time_s": compilation_time,
        "q": q_torch,
        "k": k_torch,
        "v": v_torch,
        "output": o_torch,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
    }


def run_qwen3_prefill(
    backend,
    seqlen: int,
    *,
    persistent: bool,
    tolerance: float,
    warmup: int,
    iterations: int,
    check_reference: bool,
    benchmark: bool,
    input_pattern: str = "random",
    prefill_config: str = "auto",
):
    if seqlen <= 0:
        raise ValueError("seqlen must be positive")
    optimized = _is_optimized(backend)
    if optimized:
        _, block_n, kv_stage = backend._resolve_prefill_config(prefill_config)
    else:
        block_n, kv_stage = 128, 5
    return run(
        backend,
        q_shape=(backend.QWEN_BATCH, seqlen, backend.QWEN_QUERY_HEADS, backend.QWEN_HEAD_DIM),
        k_shape=(backend.QWEN_BATCH, seqlen, backend.QWEN_KV_HEADS, backend.QWEN_HEAD_DIM),
        in_dtype=cutlass.BFloat16,
        out_dtype=cutlass.BFloat16,
        qk_acc_dtype=cutlass.Float32,
        pv_acc_dtype=cutlass.Float32,
        mma_tiler_mn=(64, block_n),
        is_persistent=persistent,
        is_causal=True,
        bottom_right_align=False,
        scale_q=1.0,
        scale_k=1.0,
        scale_v=1.0,
        inv_scale_o=1.0,
        scale_softmax=1.0 / math.sqrt(backend.QWEN_HEAD_DIM),
        window_size=(-1, -1),
        tolerance=tolerance,
        warmup_iterations=warmup,
        iterations=iterations,
        skip_ref_check=not check_reference,
        use_cold_l2=False,
        execute_benchmark=benchmark,
        input_pattern=input_pattern,
        kv_stage=kv_stage,
        write_lse=not optimized,
    )


def _decode_grouped_reference(backend, q, k, v, sm_scale: float):
    import torch

    h_r = backend.QWEN_QUERY_HEADS // backend.QWEN_KV_HEADS
    q_grouped = q.float().view(
        backend.QWEN_BATCH, backend.QWEN_KV_HEADS, h_r, 1, backend.QWEN_HEAD_DIM
    )
    scores = torch.einsum("bhgqd,bhkd->bhgqk", q_grouped, k.float()) * sm_scale
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.einsum("bhgqk,bhkd->bhgqd", probabilities, v.float())
    return output.reshape(
        backend.QWEN_BATCH, backend.QWEN_QUERY_HEADS, 1, backend.QWEN_HEAD_DIM
    )


def run_qwen3_decode(
    backend,
    seqlen: int,
    *,
    tolerance: float,
    warmup: int,
    iterations: int,
    check_reference: bool,
    benchmark: bool,
    causal: bool,
    num_splits: Optional[int],
    split_candidates: Sequence[int],
):
    import torch

    if seqlen <= 0:
        raise ValueError("seqlen must be positive")
    torch.manual_seed(1111)
    device = torch.device("cuda", torch.cuda.current_device())
    q = torch.randn(
        (backend.QWEN_BATCH, backend.QWEN_QUERY_HEADS, 1, backend.QWEN_HEAD_DIM),
        device=device,
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        (backend.QWEN_BATCH, backend.QWEN_KV_HEADS, seqlen, backend.QWEN_HEAD_DIM),
        device=device,
        dtype=torch.bfloat16,
    )
    v = torch.randn_like(k)
    sm_scale = 1.0 / math.sqrt(backend.QWEN_HEAD_DIM)
    output, stats = backend._qwen3_decode_attention_impl(
        q,
        k,
        v,
        causal=causal,
        sm_scale=sm_scale,
        num_splits=num_splits,
        split_candidates=split_candidates,
        return_stats=True,
    )

    if check_reference:
        reference = _decode_grouped_reference(backend, q, k, v, sm_scale)
        error = (output.float() - reference).abs()
        print(
            f"Decode S={seqlen} splits={stats['num_splits']} "
            f"max_abs={error.max().item():.6g}, mean_abs={error.mean().item():.6g}"
        )
        torch.testing.assert_close(
            output.float(), reference, atol=tolerance, rtol=2e-2
        )

    elapsed_us = None
    if benchmark:
        def invoke():
            return backend._qwen3_decode_attention_impl(
                q,
                k,
                v,
                causal=causal,
                sm_scale=sm_scale,
                num_splits=num_splits,
                split_candidates=split_candidates,
                return_stats=False,
            )

        elapsed_us, output = _cuda_median_us(invoke, warmup, iterations)

    return {
        "time_us": elapsed_us,
        "q": q,
        "k": k,
        "v": v,
        "output": output,
        **stats,
    }


def check_full_length(backend, result, seqlen: int, tolerance: float) -> None:
    import torch

    output = result["output"]
    sample_positions = sorted(
        {position for position in (0, 1, 127, 128, seqlen // 2, seqlen - 1) if position < seqlen}
    )
    total_error = 0.0
    sample_count = 0
    max_error = 0.0
    for position in sample_positions:
        count = position + 1
        remainder = count % 17
        position_sum = (remainder * (remainder - 1) / 2 - 8 * remainder) / 8
        for kv_head in range(backend.QWEN_KV_HEADS):
            expected = position_sum / count + kv_head / 4
            actual = output[position, :, :, kv_head, 0].float()
            if not torch.isfinite(actual).all():
                raise AssertionError(
                    f"non-finite output at position={position}, kv_head={kv_head}"
                )
            error = (actual - expected).abs()
            max_error = max(max_error, error.max().item())
            total_error += error.sum().item()
            sample_count += error.numel()
    mean_error = total_error / sample_count
    print(
        f"128K sampled prefix-mean error: max_abs={max_error:.6g}, "
        f"mean_abs={mean_error:.6g}"
    )
    if max_error > tolerance:
        raise AssertionError(
            f"full-length sampled error {max_error:.6g} exceeds {tolerance:.6g}"
        )


def causal_tflops(backend, seqlen: int, time_us: float) -> float:
    flops = (
        backend.QWEN_BATCH
        * backend.QWEN_QUERY_HEADS
        * seqlen
        * (seqlen + 1)
        * (backend.QWEN_HEAD_DIM + backend.QWEN_HEAD_DIM)
    )
    return flops / (time_us * 1e-6) / 1e12


def decode_tflops(backend, seqlen: int, time_us: float) -> float:
    flops = (
        4 * backend.QWEN_BATCH * backend.QWEN_QUERY_HEADS * seqlen * backend.QWEN_HEAD_DIM
    )
    return flops / (time_us * 1e-6) / 1e12


def _import_fa3():
    hopper_path = "/dockerdata/linqihao/flash-attention/hopper"
    if hopper_path not in sys.path:
        sys.path.insert(0, hopper_path)
    from flash_attn_interface import flash_attn_func

    return flash_attn_func


def benchmark_fa3(backend, result, warmup: int, iterations: int, *, pack_gqa):
    import torch

    flash_attn_func = _import_fa3()

    q_internal, k_internal, v_internal = result["q"], result["k"], result["v"]
    seqlen = q_internal.shape[0]
    q = q_internal.permute(4, 0, 3, 2, 1).reshape(
        backend.QWEN_BATCH, seqlen, backend.QWEN_QUERY_HEADS, backend.QWEN_HEAD_DIM
    ).contiguous()
    k = k_internal.permute(4, 0, 3, 2, 1).reshape(
        backend.QWEN_BATCH, seqlen, backend.QWEN_KV_HEADS, backend.QWEN_HEAD_DIM
    ).contiguous()
    v = v_internal.permute(4, 0, 3, 2, 1).reshape(
        backend.QWEN_BATCH, seqlen, backend.QWEN_KV_HEADS, backend.QWEN_HEAD_DIM
    ).contiguous()

    def invoke():
        return flash_attn_func(
            q,
            k,
            v,
            softmax_scale=1.0 / math.sqrt(backend.QWEN_HEAD_DIM),
            causal=True,
            num_splits=1,
            pack_gqa=pack_gqa,
        )

    with torch.inference_mode():
        median_us, baseline_output = _cuda_median_us(invoke, warmup + 1, iterations)

    local_output = result["output"]
    max_sample_error = 0.0
    sample_positions = sorted(
        {position for position in (0, 127, seqlen // 2, seqlen - 1) if position < seqlen}
    )
    for position in sample_positions:
        for query_head in (0, 7, 8, 63):
            kv_head, ratio_head = divmod(
                query_head, backend.QWEN_QUERY_HEADS // backend.QWEN_KV_HEADS
            )
            local = local_output[position, :, ratio_head, kv_head, 0].float()
            baseline = baseline_output[0, position, query_head].float()
            if not torch.isfinite(local).all() or not torch.isfinite(baseline).all():
                raise AssertionError(
                    f"non-finite FA3 comparison at position={position}, head={query_head}"
                )
            max_sample_error = max(
                max_sample_error, (local - baseline).abs().max().item()
            )
    return median_us, max_sample_error


def benchmark_fa3_decode(backend, result, warmup: int, iterations: int):
    """Benchmark FA3 Decode with its automatic split and Pack-GQA heuristics."""
    import torch

    flash_attn_func = _import_fa3()

    q = result["q"].transpose(1, 2).contiguous()
    k = result["k"].transpose(1, 2).contiguous()
    v = result["v"].transpose(1, 2).contiguous()

    def invoke():
        return flash_attn_func(
            q,
            k,
            v,
            softmax_scale=1.0 / math.sqrt(backend.QWEN_HEAD_DIM),
            causal=False,
            num_splits=0,
            pack_gqa=None,
        )

    with torch.inference_mode():
        median_us, baseline_output = _cuda_median_us(invoke, warmup + 1, iterations)

    local_output = result["output"].transpose(1, 2)
    max_error = (local_output.float() - baseline_output.float()).abs().max().item()
    return median_us, max_error


def benchmark_original_prefill(backend, result, warmup: int, iterations: int):
    import torch
    import implement_attention as original

    q_internal, k_internal, v_internal = result["q"], result["k"], result["v"]
    seqlen = q_internal.shape[0]
    q = q_internal.permute(4, 3, 2, 0, 1).reshape(
        backend.QWEN_BATCH, backend.QWEN_QUERY_HEADS, seqlen, backend.QWEN_HEAD_DIM
    ).contiguous()
    k = k_internal.permute(4, 3, 2, 0, 1).reshape(
        backend.QWEN_BATCH, backend.QWEN_KV_HEADS, seqlen, backend.QWEN_HEAD_DIM
    ).contiguous()
    v = v_internal.permute(4, 3, 2, 0, 1).reshape(
        backend.QWEN_BATCH, backend.QWEN_KV_HEADS, seqlen, backend.QWEN_HEAD_DIM
    ).contiguous()

    def invoke():
        return original.qwen3_prefill_attention(q, k, v, causal=True)

    with torch.inference_mode():
        median_us, baseline_output = _cuda_median_us(invoke, warmup + 1, iterations)

    local_output = result["output"].permute(4, 3, 2, 0, 1).reshape_as(q)
    max_error = (local_output.float() - baseline_output.float()).abs().max().item()
    return median_us, max_error


def parse_seqlens(value: str):
    try:
        values = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("sequence lengths must be positive")
    return values


def parse_split_candidates(backend, value: str):
    try:
        return backend._normalize_split_candidates(
            tuple(int(item) for item in value.split(","))
        )
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "expected comma-separated positive split counts"
        ) from exc


def _add_common_arguments(parser) -> None:
    parser.add_argument(
        "--mode", choices=("correctness", "full-check", "benchmark"), default="correctness"
    )
    parser.add_argument("--seqlen", type=int, default=None)
    parser.add_argument("--seqlens", type=parse_seqlens, default=[128, 257, 1024])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=5e-2)
    parser.add_argument("--persistent", action="store_true")
    parser.add_argument("--compare-fa3", action="store_true")


def _check_common_arguments(parser, args) -> None:
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("--warmup must be non-negative and --iterations must be positive")
    if args.tolerance <= 0:
        parser.error("--tolerance must be positive")


def main_original(backend) -> None:
    parser = argparse.ArgumentParser(
        description="Qwen3-32B BF16 causal GQA prefill attention on Hopper SM90"
    )
    _add_common_arguments(parser)
    parser.set_defaults(seqlen=backend.QWEN_CONTEXT)
    args = parser.parse_args()
    _check_common_arguments(parser, args)

    if args.mode == "correctness":
        for seqlen in args.seqlens:
            run_qwen3_prefill(
                backend,
                seqlen,
                persistent=args.persistent,
                tolerance=args.tolerance,
                warmup=0,
                iterations=1,
                check_reference=True,
                benchmark=False,
            )
        print("Correctness checks passed")
        return

    if args.mode == "full-check":
        result = run_qwen3_prefill(
            backend,
            args.seqlen,
            persistent=args.persistent,
            tolerance=args.tolerance,
            warmup=0,
            iterations=1,
            check_reference=False,
            benchmark=False,
            input_pattern="prefix",
        )
        check_full_length(backend, result, args.seqlen, args.tolerance)
        print("Full-length sampled check passed")
        return

    result = run_qwen3_prefill(
        backend,
        args.seqlen,
        persistent=args.persistent,
        tolerance=args.tolerance,
        warmup=args.warmup,
        iterations=args.iterations,
        check_reference=False,
        benchmark=True,
    )
    local_time_us = float(result["time_us"])
    print(
        f"Local CuTe: {local_time_us:.3f} us, "
        f"{causal_tflops(backend, args.seqlen, local_time_us):.3f} TFLOP/s, "
        f"peak_memory={result['peak_memory_bytes'] / 2**30:.3f} GiB"
    )
    if args.compare_fa3:
        fa3_time_us, sample_error = benchmark_fa3(
            backend, result, args.warmup, args.iterations, pack_gqa=True
        )
        print(
            f"FA3: {fa3_time_us:.3f} us, "
            f"{causal_tflops(backend, args.seqlen, fa3_time_us):.3f} TFLOP/s"
        )
        print(
            f"Speed ratio (FA3/local): {fa3_time_us / local_time_us:.4f}x, "
            f"sample max_abs={sample_error:.6g}"
        )
        if sample_error > args.tolerance:
            raise AssertionError(
                f"FA3 sample error {sample_error:.6g} exceeds {args.tolerance:.6g}"
            )


def main_optimized(backend) -> None:
    parser = argparse.ArgumentParser(
        description="Qwen3-32B BF16 GQA prefill/decode attention on Hopper SM90"
    )
    _add_common_arguments(parser)
    parser.set_defaults(seqlen=backend.QWEN_CONTEXT)
    parser.add_argument(
        "--phase", choices=("prefill", "decode", "both"), default="prefill"
    )
    parser.add_argument("--num-splits", type=int, default=0)
    parser.add_argument(
        "--split-candidates",
        type=lambda value: parse_split_candidates(backend, value),
        default=backend.DEFAULT_DECODE_SPLIT_CANDIDATES,
    )
    parser.add_argument(
        "--prefill-config",
        choices=("auto", *backend.PREFILL_CONFIGS),
        default="auto",
    )
    parser.add_argument("--compare-original", action="store_true")
    args = parser.parse_args()
    _check_common_arguments(parser, args)
    if args.num_splits < 0:
        parser.error("--num-splits must be non-negative")

    run_prefill = args.phase in {"prefill", "both"}
    run_decode = args.phase in {"decode", "both"}

    if args.mode == "correctness":
        if run_prefill:
            for seqlen in args.seqlens:
                run_qwen3_prefill(
                    backend,
                    seqlen,
                    persistent=args.persistent,
                    tolerance=args.tolerance,
                    warmup=0,
                    iterations=1,
                    check_reference=True,
                    benchmark=False,
                    prefill_config=args.prefill_config,
                )
        if run_decode:
            for seqlen in args.seqlens:
                split_requests = [1]
                requested = args.num_splits if args.num_splits else 0
                actual = backend._select_decode_splits(
                    seqlen, requested, args.split_candidates
                )
                if actual != 1:
                    split_requests.append(requested)
                for causal in (False, True):
                    for split_request in split_requests:
                        run_qwen3_decode(
                            backend,
                            seqlen,
                            tolerance=args.tolerance,
                            warmup=0,
                            iterations=1,
                            check_reference=True,
                            benchmark=False,
                            causal=causal,
                            num_splits=split_request,
                            split_candidates=args.split_candidates,
                        )
        print("Correctness checks passed")
        return

    if args.mode == "full-check":
        if run_prefill:
            result = run_qwen3_prefill(
                backend,
                args.seqlen,
                persistent=args.persistent,
                tolerance=args.tolerance,
                warmup=0,
                iterations=1,
                check_reference=False,
                benchmark=False,
                input_pattern="prefix",
                prefill_config=args.prefill_config,
            )
            check_full_length(backend, result, args.seqlen, args.tolerance)
            print("Prefill full-length sampled check passed")
        if run_decode:
            run_qwen3_decode(
                backend,
                args.seqlen,
                tolerance=args.tolerance,
                warmup=0,
                iterations=1,
                check_reference=True,
                benchmark=False,
                causal=True,
                num_splits=args.num_splits,
                split_candidates=args.split_candidates,
            )
            print("Decode full-length reference check passed")
        return

    if run_prefill:
        result = run_qwen3_prefill(
            backend,
            args.seqlen,
            persistent=args.persistent,
            tolerance=args.tolerance,
            warmup=args.warmup,
            iterations=args.iterations,
            check_reference=False,
            benchmark=True,
            prefill_config=args.prefill_config,
        )
        local_time_us = float(result["time_us"])
        resolved_config, _, _ = backend._resolve_prefill_config(args.prefill_config)
        print(
            f"Prefill local CuTe ({resolved_config}): {local_time_us:.3f} us, "
            f"{causal_tflops(backend, args.seqlen, local_time_us):.3f} TFLOP/s, "
            f"peak_memory={result['peak_memory_bytes'] / 2**30:.3f} GiB"
        )
        if args.compare_fa3:
            fa3_time_us, sample_error = benchmark_fa3(
                backend, result, args.warmup, args.iterations, pack_gqa=None
            )
            print(
                f"FA3 pack_gqa=None: {fa3_time_us:.3f} us, "
                f"{causal_tflops(backend, args.seqlen, fa3_time_us):.3f} TFLOP/s"
            )
            print(
                f"Speed ratio (FA3/local): {fa3_time_us / local_time_us:.4f}x, "
                f"sample max_abs={sample_error:.6g}"
            )
            if sample_error > args.tolerance:
                raise AssertionError(
                    f"FA3 sample error {sample_error:.6g} exceeds {args.tolerance:.6g}"
                )
        if args.compare_original:
            original_time_us, max_error = benchmark_original_prefill(
                backend, result, args.warmup, args.iterations
            )
            print(
                f"Original prefill: {original_time_us:.3f} us, "
                f"original/local={original_time_us / local_time_us:.4f}x, "
                f"max_abs={max_error:.6g}"
            )

    if run_decode:
        result = run_qwen3_decode(
            backend,
            args.seqlen,
            tolerance=args.tolerance,
            warmup=args.warmup,
            iterations=args.iterations,
            check_reference=False,
            benchmark=True,
            causal=True,
            num_splits=args.num_splits,
            split_candidates=args.split_candidates,
        )
        local_time_us = float(result["time_us"])
        print(
            f"Decode staged CuTe: {local_time_us:.3f} us, "
            f"{decode_tflops(backend, args.seqlen, local_time_us):.3f} TFLOP/s, "
            f"splits={result['num_splits']}, "
            f"partial_wgmma_launches={result['partial_wgmma_launches']}"
        )
        print(
            "Decode path: 8 query heads are Pack-GQA M rows; independent KV "
            "splits launch on concurrent CUDA streams, followed by FP32 LSE combine"
        )
        if args.compare_original:
            print("Original decode comparison unavailable: implement_attention.py has no decode API")
        if args.compare_fa3:
            fa3_time_us, max_error = benchmark_fa3_decode(
                backend, result, args.warmup, args.iterations
            )
            print(
                f"FA3 decode auto: {fa3_time_us:.3f} us, "
                f"FA3/local={fa3_time_us / local_time_us:.4f}x, "
                f"max_abs={max_error:.6g}"
            )
            if max_error > args.tolerance:
                raise AssertionError(
                    f"FA3 decode error {max_error:.6g} exceeds {args.tolerance:.6g}"
                )
