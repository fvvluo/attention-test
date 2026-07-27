#!/usr/bin/env python3
"""Project-owned GQA GPU Stage-2 reduction for Split-KV decode (Scheme A).

Derived from ``split_kv_stage2.py`` (this repo's MHA Stage-2 kernel).  Only the
head count changes (grid is now ``Q_HEADS`` instead of the MHA ``NUM_HEADS``);
the reduction math and pure-GPU implementation are unchanged.

Combines the raw FP32 Stage-1 workspace completely on GPU:

    m = max_s(partial_max[h, s])
    w_s = exp(partial_max[h, s] - m)
    denominator = sum_s(w_s * partial_sum[h, s])
    output[h, d] = sum_s(w_s * partial_output[h, s, d]) / denominator

Only FP32 intermediates are used; the final output is BF16 BHSD ``[1, Hq, 1, D]``.
No torch.max / torch.sum / PyTorch reduction / CPU reduction / baseline fallback.
"""

import torch

# torch 2.5.1 lacks this attribute, while cutlass.torch.dtype() accesses it.
if not hasattr(torch, "float4_e2m1fn_x2"):
    torch.float4_e2m1fn_x2 = torch.uint8

import cuda.bindings.driver as cuda  # noqa: E402
import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass.cute.runtime import from_dlpack  # noqa: E402

from split_kv_stage1_gqa import (  # noqa: E402
    HEAD_DIM,
    Q_HEADS,
    DTYPE_TORCH,
    Stage1WorkspaceGQA,
    validate_stage1_workspace,
)

NUM_THREADS = HEAD_DIM  # 128, one thread per output channel d
MAX_SPLITS = 16


class SplitKVStage2GQA:
    """One CTA per q_head; thread d reduces all splits independently."""

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
            grid=[Q_HEADS, 1, 1],
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
        q_head, _, _ = cute.arch.block_idx()
        d_idx, _, _ = cute.arch.thread_idx()

        global_max = -cutlass.Float32.inf
        for split_idx in range(self.split_count):
            global_max = cute.arch.fmax(global_max, mMax[q_head, split_idx])

        denominator = cutlass.Float32(0.0)
        numerator = cutlass.Float32(0.0)
        for split_idx in range(self.split_count):
            weight = cute.math.exp(mMax[q_head, split_idx] - global_max, fastmath=False)
            denominator += weight * mSum[q_head, split_idx]
            numerator += weight * mP[q_head, split_idx, d_idx]

        # mOut is BHSD [1, Hq, 1, D]; write BF16 result.
        mOut[0, q_head, 0, d_idx] = (numerator / denominator).to(mOut.element_type)


def validate_stage2_inputs(workspace, output, *, device=None):
    split_count = workspace.partial_max.shape[1]
    if split_count < 1 or split_count > MAX_SPLITS:
        raise ValueError(f"split_count 必须在 1..{MAX_SPLITS}，实际为 {split_count}")
    validate_stage1_workspace(
        workspace, split_count, device=device or workspace.partial_max.device, require_cuda=True
    )
    if not isinstance(output, torch.Tensor):
        raise TypeError("output 必须是 torch.Tensor")
    if tuple(output.shape) != (1, Q_HEADS, 1, HEAD_DIM):
        raise ValueError(
            f"output shape 必须是 (1, {Q_HEADS}, 1, {HEAD_DIM})（BHSD），实际为 {tuple(output.shape)}"
        )
    if output.dtype != DTYPE_TORCH:
        raise TypeError(f"output 必须是 {DTYPE_TORCH}")
    if not output.is_cuda:
        raise ValueError("output 必须位于 CUDA device")
    if not output.is_contiguous():
        raise ValueError("output 必须 contiguous")
    if output.data_ptr() % 16 != 0:
        raise ValueError("output 起始地址必须满足 16-byte 对齐")
    if device is not None and output.device != torch.device(device):
        raise ValueError(f"output 必须位于 {device}，实际为 {output.device}")


def allocate_stage2_output(workspace):
    return torch.empty(
        1, Q_HEADS, 1, HEAD_DIM, dtype=DTYPE_TORCH, device=workspace.partial_max.device
    )


def _make_cute_tensor(tensor):
    return from_dlpack(tensor, assumed_align=16)


def compile_split_kv_stage2_gqa(workspace, output=None):
    """Compile Stage-2 and return ``run_stage2, output``."""
    split_count = workspace.partial_max.shape[1]
    if output is None:
        output = allocate_stage2_output(workspace)
    validate_stage2_inputs(workspace, output, device=workspace.partial_max.device)

    torch_stream = torch.cuda.current_stream(workspace.partial_max.device)
    stream = cuda.CUstream(torch_stream.cuda_stream)
    kernel = SplitKVStage2GQA(split_count)
    mMax, mSum, mP, mOut = (
        _make_cute_tensor(t)
        for t in (workspace.partial_max, workspace.partial_sum, workspace.partial_output, output)
    )
    compiled = cute.compile(kernel, mMax, mSum, mP, mOut, stream)

    def run_stage2():
        current_stream = torch.cuda.current_stream(workspace.partial_max.device)
        if current_stream.cuda_stream != torch_stream.cuda_stream:
            raise RuntimeError("Stage-2 必须在编译时使用的同一 CUDA stream 上运行")
        compiled(
            _make_cute_tensor(workspace.partial_max),
            _make_cute_tensor(workspace.partial_sum),
            _make_cute_tensor(workspace.partial_output),
            _make_cute_tensor(output),
            stream,
        )
        return output

    return run_stage2, output
