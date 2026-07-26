#!/usr/bin/env python3
"""Project-owned GPU Stage-2 reduction for Split-KV decode.

Combines the raw FP32 Stage-1 workspace completely on GPU:

    m = max_s(partial_max[h, s])
    w_s = exp(partial_max[h, s] - m)
    denominator = sum_s(w_s * partial_sum[h, s])
    output[h, d] = sum_s(w_s * partial_output[h, s, d]) / denominator

The reduction uses only FP32 intermediates and keeps an FP32 BSHD output.  A
BF16 output may additionally be produced for comparison with the existing
BF16 references.  This module does not benchmark and does not report latency
or speedup.
"""

import torch

# torch 2.5.1 lacks this attribute, while cutlass.torch.dtype() accesses it.
if not hasattr(torch, "float4_e2m1fn_x2"):
    torch.float4_e2m1fn_x2 = torch.uint8

import cuda.bindings.driver as cuda  # noqa: E402
import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass.cute.runtime import from_dlpack  # noqa: E402

from cute_flash_attn import DTYPE_TORCH, HEAD_DIM, NUM_HEADS  # noqa: E402
from split_kv_stage1 import Stage1Workspace, validate_stage1_workspace  # noqa: E402

NUM_THREADS = HEAD_DIM
MAX_SPLITS = 16


class SplitKVStage2:
    """One CTA per head; thread d reduces all splits independently."""

    def __init__(self, split_count):
        self.split_count = split_count

    @cute.jit
    def __call__(
        self,
        mMax: cute.Tensor,
        mSum: cute.Tensor,
        mP: cute.Tensor,
        mOut: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(mMax, mSum, mP, mOut).launch(
            grid=[NUM_HEADS, 1, 1],
            block=[NUM_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mMax: cute.Tensor,
        mSum: cute.Tensor,
        mP: cute.Tensor,
        mOut: cute.Tensor,
    ):
        head_idx, _, _ = cute.arch.block_idx()
        d_idx, _, _ = cute.arch.thread_idx()

        global_max = -cutlass.Float32.inf
        for split_idx in range(self.split_count):
            global_max = cute.arch.fmax(
                global_max, mMax[head_idx, split_idx]
            )

        denominator = cutlass.Float32(0.0)
        numerator = cutlass.Float32(0.0)
        for split_idx in range(self.split_count):
            weight = cute.math.exp(
                mMax[head_idx, split_idx] - global_max,
                fastmath=False,
            )
            denominator += weight * mSum[head_idx, split_idx]
            numerator += weight * mP[head_idx, split_idx, d_idx]

        mOut[0, 0, head_idx, d_idx] = numerator / denominator


def validate_stage2_inputs(
    workspace,
    output_fp32,
    *,
    device=None,
    output_dtype=torch.float32,
):
    split_count = workspace.partial_max.shape[1]
    if split_count < 1 or split_count > MAX_SPLITS:
        raise ValueError(f"split_count 必须在 1..{MAX_SPLITS}，实际为 {split_count}")
    validate_stage1_workspace(
        workspace,
        split_count,
        device=device or workspace.partial_max.device,
        require_cuda=True,
    )
    if not isinstance(output_fp32, torch.Tensor):
        raise TypeError("output_fp32 必须是 torch.Tensor")
    if tuple(output_fp32.shape) != (1, 1, NUM_HEADS, HEAD_DIM):
        raise ValueError(
            f"output_fp32 shape 必须是 (1,1,{NUM_HEADS},{HEAD_DIM})，实际为 {tuple(output_fp32.shape)}"
        )
    if output_fp32.dtype != torch.float32:
        raise TypeError("output_fp32 必须是 float32")
    if not output_fp32.is_cuda:
        raise ValueError("output_fp32 必须位于 CUDA device")
    if not output_fp32.is_contiguous():
        raise ValueError("output_fp32 必须 contiguous")
    if output_fp32.data_ptr() % 16 != 0:
        raise ValueError("output_fp32 起始地址必须满足 16-byte 对齐")
    if device is not None and output_fp32.device != torch.device(device):
        raise ValueError(
            f"output_fp32 必须位于 {device}，实际为 {output_fp32.device}"
        )
    if output_dtype not in (torch.float32, DTYPE_TORCH):
        raise TypeError("output_dtype 必须是 torch.float32 或项目 BF16 dtype")


def allocate_stage2_output(workspace):
    return torch.empty(
        1,
        1,
        NUM_HEADS,
        HEAD_DIM,
        dtype=torch.float32,
        device=workspace.partial_max.device,
    )


def _make_cute_tensor(tensor):
    return from_dlpack(tensor, assumed_align=16)


def compile_split_kv_stage2(
    workspace,
    output_fp32=None,
    *,
    output_dtype=torch.float32,
):
    """Compile Stage-2 and return ``run_stage2, output_fp32, final_output``."""
    split_count = workspace.partial_max.shape[1]
    if output_fp32 is None:
        output_fp32 = allocate_stage2_output(workspace)
    validate_stage2_inputs(
        workspace,
        output_fp32,
        device=workspace.partial_max.device,
        output_dtype=output_dtype,
    )

    torch_stream = torch.cuda.current_stream(workspace.partial_max.device)
    stream = cuda.CUstream(torch_stream.cuda_stream)
    kernel = SplitKVStage2(split_count)
    mMax, mSum, mP, mOut = (
        _make_cute_tensor(tensor)
        for tensor in (
            workspace.partial_max,
            workspace.partial_sum,
            workspace.partial_output,
            output_fp32,
        )
    )
    compiled = cute.compile(kernel, mMax, mSum, mP, mOut, stream)

    def run_stage2():
        current_stream = torch.cuda.current_stream(workspace.partial_max.device)
        if current_stream.cuda_stream != torch_stream.cuda_stream:
            raise RuntimeError(
                "Stage-2 必须在编译时使用的同一 CUDA stream 上运行"
            )
        compiled(
            _make_cute_tensor(workspace.partial_max),
            _make_cute_tensor(workspace.partial_sum),
            _make_cute_tensor(workspace.partial_output),
            _make_cute_tensor(output_fp32),
            stream,
        )
        return output_fp32.to(output_dtype)

    return run_stage2, output_fp32, output_dtype


def run_split_kv_gpu_reduction(workspace, *, output_dtype=torch.float32):
    """Run Stage-2 with a fresh FP32 output buffer."""
    run_stage2, output_fp32, _ = compile_split_kv_stage2(
        workspace, output_dtype=output_dtype
    )
    final_output = run_stage2()
    return final_output, output_fp32
