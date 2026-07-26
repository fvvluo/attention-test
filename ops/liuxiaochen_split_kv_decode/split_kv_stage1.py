#!/usr/bin/env python3
"""Project-owned q_len=1 Split-KV Stage-1 CuTe kernel.

Each CTA processes one ``(head, split)`` pair and writes the raw FP32
max-shifted online-softmax state:

    m_s = max(score)
    l_s = sum(exp(score - m_s))
    p_s = sum(exp(score - m_s) * V)

``p_s`` is never divided by ``l_s`` and is never rounded to BF16.  The first
correctness version uses a PyTorch FP32 Stage-2 reduction; it is not a complete
performance implementation and must not be used for latency or speedup claims.
"""

import math
from dataclasses import dataclass
from numbers import Real

import torch

# torch 2.5.1 lacks this attribute, while cutlass.torch.dtype() accesses it.
if not hasattr(torch, "float4_e2m1fn_x2"):
    torch.float4_e2m1fn_x2 = torch.uint8

import cuda.bindings.driver as cuda  # noqa: E402
import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass.cute.runtime import from_dlpack  # noqa: E402

from cute_flash_attn import DTYPE_TORCH, HEAD_DIM, NUM_HEADS  # noqa: E402
from split_kv_decode import build_split_plan, resolve_visible_kv_len  # noqa: E402

NUM_THREADS = 128
NUM_WARPS = 4
WARP_SIZE = 32
ELEMS_PER_LANE = HEAD_DIM // WARP_SIZE


@dataclass(frozen=True)
class Stage1Workspace:
    partial_max: torch.Tensor
    partial_sum: torch.Tensor
    partial_output: torch.Tensor

    @property
    def nbytes(self):
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.partial_max, self.partial_sum, self.partial_output)
        )


def stage1_workspace_nbytes(split_count):
    if not isinstance(split_count, int) or isinstance(split_count, bool):
        raise TypeError("split_count 必须是整数")
    if split_count <= 0:
        raise ValueError("split_count 必须大于 0")
    return (NUM_HEADS * split_count * 2 + NUM_HEADS * split_count * HEAD_DIM) * 4


def allocate_stage1_workspace(split_count, *, device):
    stage1_workspace_nbytes(split_count)
    return Stage1Workspace(
        partial_max=torch.empty(
            NUM_HEADS, split_count, dtype=torch.float32, device=device
        ),
        partial_sum=torch.empty(
            NUM_HEADS, split_count, dtype=torch.float32, device=device
        ),
        partial_output=torch.empty(
            NUM_HEADS, split_count, HEAD_DIM, dtype=torch.float32, device=device
        ),
    )


def validate_stage1_workspace(
    workspace, split_count, *, device=None, require_cuda=False
):
    if not isinstance(workspace, Stage1Workspace):
        raise TypeError("workspace 必须是 Stage1Workspace")

    expected_shapes = {
        "partial_max": (NUM_HEADS, split_count),
        "partial_sum": (NUM_HEADS, split_count),
        "partial_output": (NUM_HEADS, split_count, HEAD_DIM),
    }
    storages = []
    for name, expected_shape in expected_shapes.items():
        tensor = getattr(workspace, name)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"workspace.{name} 必须是 torch.Tensor")
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"workspace.{name} shape 必须是 {expected_shape}，实际为 {tuple(tensor.shape)}"
            )
        if tensor.dtype != torch.float32:
            raise TypeError(f"workspace.{name} 必须是 float32")
        if not tensor.is_contiguous():
            raise ValueError(f"workspace.{name} 必须 contiguous")
        if require_cuda and not tensor.is_cuda:
            raise ValueError(f"workspace.{name} 必须位于 CUDA device")
        if device is not None and tensor.device != torch.device(device):
            raise ValueError(
                f"workspace.{name} 必须位于 {device}，实际为 {tensor.device}"
            )
        if tensor.data_ptr() % 16 != 0:
            raise ValueError(f"workspace.{name} 起始地址必须满足 16-byte 对齐")
        storages.append(tensor.untyped_storage().data_ptr())

    if len(set(storages)) != len(storages):
        raise ValueError("Stage-1 workspace tensor storage 不得重叠")

    expected_nbytes = stage1_workspace_nbytes(split_count)
    if workspace.nbytes != expected_nbytes:
        raise ValueError(
            f"workspace 字节数必须是 {expected_nbytes}，实际为 {workspace.nbytes}"
        )
    return expected_nbytes


def _make_cute_tensor(tensor):
    return from_dlpack(tensor, assumed_align=16)


def _validate_inputs(Q, K, V, split_count, softmax_scale, query_position):
    tensors = {"Q": Q, "K": K, "V": V}
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} 必须是 torch.Tensor")
        if not tensor.is_cuda:
            raise ValueError(f"{name} 必须位于 CUDA device")
        if tensor.dtype != DTYPE_TORCH:
            raise TypeError(f"{name} dtype 必须是 {DTYPE_TORCH}，实际为 {tensor.dtype}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} 必须是 contiguous BSHD")
        if tensor.data_ptr() % 16 != 0:
            raise ValueError(f"{name} 起始地址必须满足 16-byte 对齐")

    if not (Q.device == K.device == V.device):
        raise ValueError("Q/K/V 必须位于同一个 CUDA device")
    if tuple(Q.shape) != (1, 1, NUM_HEADS, HEAD_DIM):
        raise ValueError(
            f"Q shape 必须是 (1, 1, {NUM_HEADS}, {HEAD_DIM})，实际为 {tuple(Q.shape)}"
        )
    if K.ndim != 4 or tuple(K.shape[:1] + K.shape[2:]) != (
        1,
        NUM_HEADS,
        HEAD_DIM,
    ):
        raise ValueError(
            f"K shape 必须是 (1, kv_len, {NUM_HEADS}, {HEAD_DIM})，实际为 {tuple(K.shape)}"
        )
    if tuple(V.shape) != tuple(K.shape):
        raise ValueError("V shape 必须与 K 完全相同")

    if not isinstance(softmax_scale, Real) or isinstance(softmax_scale, bool):
        raise TypeError("softmax_scale 必须是实数")
    if not math.isfinite(float(softmax_scale)) or float(softmax_scale) <= 0:
        raise ValueError("softmax_scale 必须是有限正数")

    visible_kv_len = resolve_visible_kv_len(K.shape[1], query_position)
    return build_split_plan(visible_kv_len, split_count)


class SplitKVStage1:
    """128-thread, four-warp SIMT Stage-1 kernel specialized for D=128."""

    def __init__(self, split_count):
        self.split_count = split_count

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mMax: cute.Tensor,
        mSum: cute.Tensor,
        mP: cute.Tensor,
        visible_kv_len: cutlass.Int32,
        softmax_scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            mQ,
            mK,
            mV,
            mMax,
            mSum,
            mP,
            visible_kv_len,
            softmax_scale,
        ).launch(
            grid=[NUM_HEADS, self.split_count, 1],
            block=[NUM_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mMax: cute.Tensor,
        mSum: cute.Tensor,
        mP: cute.Tensor,
        visible_kv_len: cutlass.Int32,
        softmax_scale: cutlass.Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        head_idx, split_idx, _ = cute.arch.block_idx()
        warp_id = tidx >> 5
        lane_id = tidx & 31

        base = visible_kv_len // self.split_count
        long_count = visible_kv_len % self.split_count
        split_start = cutlass.Int32(0)
        split_length = cutlass.Int32(0)
        if split_idx < long_count:
            split_start = split_idx * (base + 1)
            split_length = base + 1
        else:
            split_start = long_count * (base + 1) + (split_idx - long_count) * base
            split_length = base
        split_end = split_start + split_length

        q_frag = cute.make_rmem_tensor((ELEMS_PER_LANE,), cutlass.Float32)
        p_frag = cute.make_rmem_tensor((ELEMS_PER_LANE,), cutlass.Float32)
        for elem in range(ELEMS_PER_LANE):
            d_idx = lane_id * ELEMS_PER_LANE + elem
            q_frag[elem] = mQ[0, 0, head_idx, d_idx].to(cutlass.Float32)
            p_frag[elem] = cutlass.Float32(0.0)

        warp_m = -cutlass.Float32.inf
        warp_l = cutlass.Float32(0.0)
        for token_idx in cutlass.range(
            split_start + warp_id, split_end, NUM_WARPS, unroll=1
        ):
            dot = cutlass.Float32(0.0)
            for elem in range(ELEMS_PER_LANE):
                d_idx = lane_id * ELEMS_PER_LANE + elem
                k_value = mK[0, token_idx, head_idx, d_idx].to(cutlass.Float32)
                dot += q_frag[elem] * k_value
            dot = cute.arch.warp_reduction_sum(dot, threads_in_group=WARP_SIZE)
            score = dot * softmax_scale

            new_m = cute.arch.fmax(warp_m, score)
            alpha = cute.math.exp(warp_m - new_m, fastmath=False)
            beta = cute.math.exp(score - new_m, fastmath=False)
            warp_l = alpha * warp_l + beta
            for elem in range(ELEMS_PER_LANE):
                d_idx = lane_id * ELEMS_PER_LANE + elem
                v_value = mV[0, token_idx, head_idx, d_idx].to(cutlass.Float32)
                p_frag[elem] = alpha * p_frag[elem] + beta * v_value
            warp_m = new_m

        smem = cutlass.utils.SmemAllocator()
        s_m = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((NUM_WARPS,)),
            byte_alignment=16,
        )
        s_l = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((NUM_WARPS,)),
            byte_alignment=16,
        )
        s_p = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout(
                (NUM_WARPS, HEAD_DIM), stride=(HEAD_DIM, 1)
            ),
            byte_alignment=16,
        )

        if lane_id == 0:
            s_m[warp_id] = warp_m
            s_l[warp_id] = warp_l
        for elem in range(ELEMS_PER_LANE):
            d_idx = lane_id * ELEMS_PER_LANE + elem
            s_p[warp_id, d_idx] = p_frag[elem]
        cute.arch.sync_threads()

        if warp_id == 0:
            merged_m = s_m[0]
            for warp in range(1, NUM_WARPS):
                merged_m = cute.arch.fmax(merged_m, s_m[warp])

            merged_l = cutlass.Float32(0.0)
            merged_p = cute.make_rmem_tensor(
                (ELEMS_PER_LANE,), cutlass.Float32
            )
            merged_p.fill(0.0)
            for warp in range(NUM_WARPS):
                warp_scale = cute.math.exp(s_m[warp] - merged_m, fastmath=False)
                merged_l += warp_scale * s_l[warp]
                for elem in range(ELEMS_PER_LANE):
                    d_idx = lane_id * ELEMS_PER_LANE + elem
                    merged_p[elem] += warp_scale * s_p[warp, d_idx]

            if lane_id == 0:
                mMax[head_idx, split_idx] = merged_m
                mSum[head_idx, split_idx] = merged_l
            for elem in range(ELEMS_PER_LANE):
                d_idx = lane_id * ELEMS_PER_LANE + elem
                mP[head_idx, split_idx, d_idx] = merged_p[elem]


def compile_split_kv_stage1(
    Q,
    K,
    V,
    split_count,
    softmax_scale,
    query_position=None,
    workspace=None,
):
    """Compile Stage-1 and return ``run_stage1, workspace, split_plan``."""
    plan = _validate_inputs(
        Q, K, V, split_count, softmax_scale, query_position
    )
    if workspace is None:
        workspace = allocate_stage1_workspace(split_count, device=Q.device)
    validate_stage1_workspace(
        workspace, split_count, device=Q.device, require_cuda=True
    )

    mQ, mK, mV = (_make_cute_tensor(tensor) for tensor in (Q, K, V))
    mMax, mSum, mP = (
        _make_cute_tensor(tensor)
        for tensor in (
            workspace.partial_max,
            workspace.partial_sum,
            workspace.partial_output,
        )
    )

    torch_stream = torch.cuda.current_stream(Q.device)
    stream = cuda.CUstream(torch_stream.cuda_stream)
    kernel = SplitKVStage1(split_count)
    compiled = cute.compile(
        kernel,
        mQ,
        mK,
        mV,
        mMax,
        mSum,
        mP,
        plan.visible_kv_len,
        float(softmax_scale),
        stream,
    )

    def run_stage1():
        current_stream = torch.cuda.current_stream(Q.device)
        if current_stream.cuda_stream != torch_stream.cuda_stream:
            raise RuntimeError(
                "Stage-1 必须在编译时使用的同一 CUDA stream 上运行"
            )
        compiled(
            _make_cute_tensor(Q),
            _make_cute_tensor(K),
            _make_cute_tensor(V),
            _make_cute_tensor(workspace.partial_max),
            _make_cute_tensor(workspace.partial_sum),
            _make_cute_tensor(workspace.partial_output),
            plan.visible_kv_len,
            float(softmax_scale),
            stream,
        )
        return workspace

    return run_stage1, workspace, plan


def reduce_stage1_fp32(workspace, *, output_dtype=torch.float32):
    """Correctness-only PyTorch FP32 Stage-2 reduction."""
    split_count = workspace.partial_max.shape[1]
    validate_stage1_workspace(
        workspace,
        split_count,
        device=workspace.partial_max.device,
        require_cuda=workspace.partial_max.is_cuda,
    )

    for name, tensor in (
        ("partial_max", workspace.partial_max),
        ("partial_sum", workspace.partial_sum),
        ("partial_output", workspace.partial_output),
    ):
        if not bool(torch.isfinite(tensor).all().item()):
            raise RuntimeError(f"Stage-1 {name} 包含 NaN/Inf")
    if not bool((workspace.partial_sum > 0).all().item()):
        raise RuntimeError("Stage-1 partial_sum 必须全部大于 0")

    global_max = workspace.partial_max.amax(dim=1)
    split_scale = torch.exp(workspace.partial_max - global_max.unsqueeze(1))
    denominator = (split_scale * workspace.partial_sum).sum(dim=1)
    numerator = (
        split_scale.unsqueeze(-1) * workspace.partial_output
    ).sum(dim=1)
    if not bool(torch.isfinite(denominator).all().item()) or not bool(
        (denominator > 0).all().item()
    ):
        raise RuntimeError("Stage-2 denominator 非法")

    output_fp32 = numerator / denominator.unsqueeze(-1)
    if not bool(torch.isfinite(output_fp32).all().item()):
        raise RuntimeError("Stage-2 output 包含 NaN/Inf")
    return output_fp32.view(1, 1, NUM_HEADS, HEAD_DIM).to(output_dtype)
