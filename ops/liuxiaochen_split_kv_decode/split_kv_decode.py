#!/usr/bin/env python3
"""Decode split mapping 与 online-softmax merge 的 correctness scaffold。

本模块不是正式 split-KV kernel、优化实现或性能实现。它只复用现有的
``FlashAttentionForwardAmpere``，让不同 KV chunk 作为 batch 并行产生局部归一化
输出，再用 PyTorch correctness oracle 重算每个 chunk 的 local max/sum，验证 split
mapping 与 merge 公式。

重要限制：

* reference kernel 不输出 local LSE；本模块不验证未来 kernel 的 LSE 输出。
* oracle 会重新读取 K 并重算 QK，因此不得用于 latency、speedup 或总 workspace peak。
* 只支持 batch=1、q_len=1、32 heads、head_dim=128、bf16 decode。
* 底层 kernel 始终是 non-causal。``query_position`` 仅通过裁剪可见 KV 前缀实现
  单 query decode 的全局 causal 语义，不支持通用 multi-query causal attention。
* K/V split 全部使用共享原 storage 的 contiguous view；不 padding，也不复制完整 K/V。
"""

import math
from dataclasses import dataclass
from numbers import Real

import torch

# torch 2.5.1 缺少该属性，而 cutlass.torch.dtype() 会无条件访问。
if not hasattr(torch, "float4_e2m1fn_x2"):
    torch.float4_e2m1fn_x2 = torch.uint8

import cuda.bindings.driver as cuda  # noqa: E402
import cutlass.cute  # noqa: E402

from cute_flash_attn import (  # noqa: E402
    DTYPE_TORCH,
    HEAD_DIM,
    NUM_HEADS,
    build_kernel,
    device,
    make_cute_tensor,
)

ORACLE_TILE_SIZE = 4096


@dataclass(frozen=True)
class SplitChunk:
    """一个逻辑 KV split 在 visible prefix 中的半开区间。"""

    split_index: int
    start: int
    end: int

    @property
    def length(self):
        return self.end - self.start


@dataclass(frozen=True)
class SplitGroup:
    """长度相同、可作为紧凑 batch view 一次 launch 的连续 split 组。"""

    chunk_length: int
    first_split: int
    split_count: int
    token_start: int
    token_end: int


@dataclass(frozen=True)
class SplitPlan:
    """纯整数 split mapping；构建和检查不触发 GPU kernel。"""

    visible_kv_len: int
    split_count: int
    chunks: tuple
    groups: tuple


def resolve_visible_kv_len(kv_len, query_position=None):
    """返回单 query decode 可见的 KV prefix 长度。"""
    if not isinstance(kv_len, int) or isinstance(kv_len, bool) or kv_len <= 0:
        raise ValueError(f"kv_len 必须是正整数，实际为 {kv_len!r}")
    if query_position is None:
        return kv_len
    if not isinstance(query_position, int) or isinstance(query_position, bool):
        raise TypeError("query_position 必须是整数或 None")
    if not 0 <= query_position < kv_len:
        raise ValueError(
            f"query_position 必须满足 0 <= query_position < kv_len；"
            f"实际为 query_position={query_position}, kv_len={kv_len}"
        )
    return query_position + 1


def build_split_plan(visible_kv_len, split_count):
    """均衡切分 visible prefix，长 chunk 在前，且永不产生零长度 split。"""
    if not isinstance(visible_kv_len, int) or isinstance(visible_kv_len, bool):
        raise TypeError("visible_kv_len 必须是整数")
    if visible_kv_len <= 0:
        raise ValueError("visible_kv_len 必须大于 0")
    if not isinstance(split_count, int) or isinstance(split_count, bool):
        raise TypeError("split_count 必须是整数")
    if split_count <= 0:
        raise ValueError("split_count 必须大于 0")
    if split_count > visible_kv_len:
        raise ValueError(
            f"split_count={split_count} 不能大于 visible_kv_len={visible_kv_len}，"
            "禁止产生零长度 split"
        )

    short_len, long_count = divmod(visible_kv_len, split_count)
    long_len = short_len + 1
    lengths = [long_len] * long_count + [short_len] * (split_count - long_count)

    chunks = []
    cursor = 0
    for split_index, length in enumerate(lengths):
        if length <= 0:
            raise RuntimeError("内部错误：split plan 产生了零长度 chunk")
        chunks.append(SplitChunk(split_index, cursor, cursor + length))
        cursor += length
    if cursor != visible_kv_len:
        raise RuntimeError("内部错误：split boundaries 未覆盖完整 visible KV prefix")

    groups = []
    if long_count:
        long_token_end = long_count * long_len
        groups.append(SplitGroup(long_len, 0, long_count, 0, long_token_end))
    short_count = split_count - long_count
    if short_count:
        token_start = long_count * long_len
        groups.append(
            SplitGroup(short_len, long_count, short_count, token_start, visible_kv_len)
        )

    return SplitPlan(visible_kv_len, split_count, tuple(chunks), tuple(groups))


def _validate_shared_contiguous_view(original, view, expected_numel, label):
    if view.untyped_storage().data_ptr() != original.untyped_storage().data_ptr():
        raise RuntimeError(f"{label} 未与原 tensor 共享 storage，疑似发生了复制")
    if not view.is_contiguous():
        raise RuntimeError(f"{label} 必须是 contiguous view")
    if view.numel() != expected_numel:
        raise RuntimeError(
            f"{label} 元素数不匹配：expected={expected_numel}, actual={view.numel()}"
        )
    if view.data_ptr() % 16 != 0:
        raise RuntimeError(f"{label} 起始地址不满足 16-byte 对齐")


def make_kv_group_views(tensor, plan, label="KV"):
    """按 plan 返回最多两个无 padding、无复制的长/短 chunk batch view。"""
    if tensor.ndim < 2 or tensor.shape[0] != 1:
        raise ValueError(f"{label} 必须至少为二维且 batch=1")
    if tensor.shape[1] < plan.visible_kv_len:
        raise ValueError(
            f"{label} seq_len={tensor.shape[1]} 小于 visible_kv_len={plan.visible_kv_len}"
        )
    if not tensor.is_contiguous():
        raise ValueError(f"{label} 必须是 contiguous，禁止通过隐式复制修复布局")

    visible = tensor[:, : plan.visible_kv_len]
    _validate_shared_contiguous_view(
        tensor, visible, plan.visible_kv_len * math.prod(tensor.shape[2:]), f"{label}_visible"
    )

    trailing_shape = tuple(tensor.shape[2:])
    trailing_numel = math.prod(trailing_shape)
    views = []
    for group_index, group in enumerate(plan.groups):
        segment = visible[:, group.token_start : group.token_end]
        group_view = segment.view(
            group.split_count, group.chunk_length, *trailing_shape
        )
        expected_numel = group.split_count * group.chunk_length * trailing_numel
        _validate_shared_contiguous_view(
            tensor, group_view, expected_numel, f"{label}_group_{group_index}"
        )
        if group.token_end - group.token_start != group.split_count * group.chunk_length:
            raise RuntimeError(f"{label}_group_{group_index} 包含 padding")
        views.append(group_view)
    return tuple(views)


def make_output_group_views(output, plan):
    """把统一 O_split workspace 按逻辑 split 顺序切成 group view。"""
    if output.shape[0] != plan.split_count or not output.is_contiguous():
        raise ValueError("O_split 必须是 contiguous，且第一维等于 split_count")
    views = []
    for group_index, group in enumerate(plan.groups):
        view = output[group.first_split : group.first_split + group.split_count]
        _validate_shared_contiguous_view(
            output, view, view.numel(), f"O_split_group_{group_index}"
        )
        views.append(view)
    return tuple(views)


def _validate_decode_inputs(Q, K, V, split_count, softmax_scale, query_position):
    tensors = {"Q": Q, "K": K, "V": V}
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} 必须是 torch.Tensor")
        if not tensor.is_cuda:
            raise ValueError(f"{name} 必须位于 CUDA device")
        if tensor.dtype != DTYPE_TORCH:
            raise TypeError(f"{name} dtype 必须是 {DTYPE_TORCH}，实际为 {tensor.dtype}")
        if tensor.device != device:
            raise ValueError(f"{name} 必须位于 {device}，实际为 {tensor.device}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} 必须是 contiguous BSHD，禁止隐式复制")
        if tensor.data_ptr() % 16 != 0:
            raise ValueError(f"{name} 起始地址必须满足 16-byte 对齐")

    expected_q_shape = (1, 1, NUM_HEADS, HEAD_DIM)
    if tuple(Q.shape) != expected_q_shape:
        raise ValueError(f"Q shape 必须是 {expected_q_shape}，实际为 {tuple(Q.shape)}")
    if K.ndim != 4 or tuple(K.shape[:1] + K.shape[2:]) != (1, NUM_HEADS, HEAD_DIM):
        raise ValueError(
            f"K shape 必须是 (1, kv_len, {NUM_HEADS}, {HEAD_DIM})，实际为 {tuple(K.shape)}"
        )
    if tuple(V.shape) != tuple(K.shape):
        raise ValueError(f"V shape 必须与 K 相同，K={tuple(K.shape)}, V={tuple(V.shape)}")

    if not isinstance(softmax_scale, Real) or isinstance(softmax_scale, bool):
        raise TypeError("softmax_scale 必须是实数")
    if not math.isfinite(float(softmax_scale)) or float(softmax_scale) <= 0:
        raise ValueError("softmax_scale 必须是有限正数")

    visible_kv_len = resolve_visible_kv_len(K.shape[1], query_position)
    plan = build_split_plan(visible_kv_len, split_count)
    if split_count == 1:
        raise ValueError(
            "split_count=1 必须直接调用 cute_flash_attn.compile_cute_attn 原始 decode 路径，"
            "不得经过 statistics oracle 或 merge"
        )
    return plan


def _torch_local_softmax_stats_oracle(
    Q, K_visible, plan, softmax_scale, oracle_tile_size
):
    """用 FP32 tile 重算各 split 的 local max/sum，仅供 correctness merge。"""
    q_fp32 = Q[0, 0].to(torch.float32)
    local_max = []
    local_sum = []

    for chunk in plan.chunks:
        chunk_max = None
        chunk_sum = None
        K_chunk = K_visible[0, chunk.start : chunk.end]
        for tile_start in range(0, chunk.length, oracle_tile_size):
            tile_end = min(tile_start + oracle_tile_size, chunk.length)
            K_tile_fp32 = K_chunk[tile_start:tile_end].to(torch.float32)
            scores = torch.einsum("hd,thd->ht", q_fp32, K_tile_fp32)
            scores.mul_(float(softmax_scale))

            tile_max = scores.amax(dim=-1)
            tile_sum = torch.exp(scores - tile_max.unsqueeze(-1)).sum(dim=-1)
            if chunk_max is None:
                chunk_max = tile_max
                chunk_sum = tile_sum
            else:
                merged_max = torch.maximum(chunk_max, tile_max)
                chunk_sum = (
                    torch.exp(chunk_max - merged_max) * chunk_sum
                    + torch.exp(tile_max - merged_max) * tile_sum
                )
                chunk_max = merged_max

        local_max.append(chunk_max)
        local_sum.append(chunk_sum)

    return torch.stack(local_max), torch.stack(local_sum)


def compile_split_kv_scaffold(
    Q,
    K,
    V,
    split_count,
    softmax_scale,
    query_position=None,
    oracle_tile_size=ORACLE_TILE_SIZE,
):
    """编译 correctness scaffold，返回 ``run_once/run_local/combine/plan``。

    ``run_local`` 仅运行现有 CuTe kernel，产生各 split 的局部归一化输出。
    ``combine`` 使用 tiled PyTorch statistics oracle 合并结果，且必须在
    ``run_local`` 之后调用。``run_once`` 串联二者。

    这些 callable 只用于正确性验证，不得用于性能计时或 workspace 峰值报告。
    ``split_count=1`` 会被拒绝；调用方必须直接使用原始 CuTe decode 路径。
    """
    if not isinstance(oracle_tile_size, int) or isinstance(oracle_tile_size, bool):
        raise TypeError("oracle_tile_size 必须是整数")
    if oracle_tile_size <= 0:
        raise ValueError("oracle_tile_size 必须大于 0")

    plan = _validate_decode_inputs(
        Q, K, V, split_count, softmax_scale, query_position
    )
    K_visible = K[:, : plan.visible_kv_len]
    V_visible = V[:, : plan.visible_kv_len]
    K_groups = make_kv_group_views(K, plan, "K")
    V_groups = make_kv_group_views(V, plan, "V")

    Q_batched = Q.expand(split_count, 1, NUM_HEADS, HEAD_DIM).contiguous()
    O_split = torch.empty(
        split_count, 1, NUM_HEADS, HEAD_DIM, dtype=DTYPE_TORCH, device=device
    )
    Q_groups = make_output_group_views(Q_batched, plan)
    O_groups = make_output_group_views(O_split, plan)

    torch_stream = torch.cuda.current_stream(Q.device)
    stream = cuda.CUstream(torch_stream.cuda_stream)
    launches = []
    for Q_group, K_group, V_group, O_group in zip(
        Q_groups, K_groups, V_groups, O_groups
    ):
        mQ, mK, mV, mO = (
            make_cute_tensor(tensor)
            for tensor in (Q_group, K_group, V_group, O_group)
        )
        compiled = cutlass.cute.compile(
            build_kernel(), mQ, mK, mV, mO, float(softmax_scale), stream
        )
        launches.append((compiled, Q_group, K_group, V_group, O_group))

    local_outputs_ready = False

    def _check_stream():
        current = torch.cuda.current_stream(Q.device)
        if current.cuda_stream != torch_stream.cuda_stream:
            raise RuntimeError(
                "correctness scaffold 必须在编译时使用的同一 CUDA stream 上运行"
            )

    def run_local():
        nonlocal local_outputs_ready
        _check_stream()
        local_outputs_ready = False
        for compiled, Q_group, K_group, V_group, O_group in launches:
            compiled(
                make_cute_tensor(Q_group),
                make_cute_tensor(K_group),
                make_cute_tensor(V_group),
                make_cute_tensor(O_group),
                float(softmax_scale),
                stream,
            )
        local_outputs_ready = True
        return O_split

    def combine():
        if not local_outputs_ready:
            raise RuntimeError("combine() 前必须先调用 run_local()")
        _check_stream()
        m_s, l_s = _torch_local_softmax_stats_oracle(
            Q,
            K_visible,
            plan,
            float(softmax_scale),
            oracle_tile_size,
        )
        global_max = m_s.amax(dim=0)
        weights = torch.exp(m_s - global_max.unsqueeze(0)) * l_s
        denominator = weights.sum(dim=0)
        numerator = (
            weights.unsqueeze(-1) * O_split[:, 0].to(torch.float32)
        ).sum(dim=0)
        output = (numerator / denominator.unsqueeze(-1)).to(DTYPE_TORCH)
        return output.view(1, 1, NUM_HEADS, HEAD_DIM)

    def run_once():
        run_local()
        return combine()

    return run_once, run_local, combine, plan
