#!/usr/bin/env python3
"""Layered correctness tests for decode Split-KV work.

Suites separate mapping, the existing reference kernel's local behavior, the
known BF16-normalized-output limitation, and the project-owned FP32-partial
Stage-1 kernel.  No suite reports latency or speedup, and this script contains
no 128k test case.
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F

# torch 2.5.1 lacks this attribute, while cutlass.torch.dtype() accesses it.
if not hasattr(torch, "float4_e2m1fn_x2"):
    torch.float4_e2m1fn_x2 = torch.uint8

_REFERENCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference")
sys.path.insert(0, _REFERENCE_DIR)

from cute_flash_attn import (  # noqa: E402
    DTYPE_TORCH,
    HEAD_DIM,
    NUM_HEADS,
    compile_cute_attn,
    device,
)
from split_kv_decode import (  # noqa: E402
    build_split_plan,
    compile_split_kv_scaffold,
    make_kv_group_views,
    make_output_group_views,
    resolve_visible_kv_len,
)
from split_kv_stage1 import (  # noqa: E402
    Stage1Workspace,
    allocate_stage1_workspace,
    compile_split_kv_stage1,
    reduce_stage1_fp32,
    stage1_workspace_nbytes,
    validate_stage1_workspace,
)
from split_kv_stage2 import (  # noqa: E402
    compile_split_kv_stage2,
    run_split_kv_gpu_reduction,
)

DEFAULT_ATOL = 2e-3
DEFAULT_RTOL = 1e-2
ORACLE_TILE_SIZE = 1024


@dataclass(frozen=True)
class ValidationCase:
    label: str
    kv_len: int
    split_counts: tuple
    query_position: int | None = None


STAGE1_CASES = (
    ValidationCase("visible8_of9", 9, (2,), query_position=7),
    ValidationCase("kv128", 128, (2, 4, 8)),
    ValidationCase("tail130", 130, (2, 4, 8)),
    ValidationCase("kv1024", 1024, (2, 4, 8)),
    ValidationCase("kv8192", 8192, (2, 4, 8)),
)
STAGE2_SMALL_CASES = (
    ValidationCase("visible8_of9", 9, (2,), query_position=7),
    ValidationCase("kv128", 128, (1, 2, 4, 8)),
    ValidationCase("tail130", 130, (2, 4, 8)),
    ValidationCase("kv1024", 1024, (2, 4, 8)),
    ValidationCase("kv8192", 8192, (2, 4, 8)),
)
EXPECTED_WORKSPACE_BYTES = {
    1: 16640,
    2: 33280,
    4: 66560,
    8: 133120,
    16: 266240,
}


def _expect_exception(expected_type, fn, label):
    try:
        fn()
    except expected_type:
        return
    except Exception as exc:
        raise AssertionError(
            f"{label}: 预期 {expected_type.__name__}，实际为 {type(exc).__name__}: {exc}"
        ) from exc
    raise AssertionError(f"{label}: 预期抛出 {expected_type.__name__}")


def _comparison(actual, reference, atol, rtol):
    actual_fp32 = actual.to(torch.float32)
    reference_fp32 = reference.to(torch.float32)
    diff = (actual_fp32 - reference_fp32).abs()
    tolerance = float(atol) + float(rtol) * reference_fp32.abs()
    finite = bool(
        (
            torch.isfinite(actual_fp32).all()
            & torch.isfinite(reference_fp32).all()
        ).item()
    )
    relative = diff / reference_fp32.abs().clamp_min(1e-12)
    return {
        "max_abs": diff.max().item(),
        "max_rel": relative.max().item(),
        "close": finite and bool((diff <= tolerance).all().item()),
        "has_nan": bool(torch.isnan(actual_fp32).any().item()),
        "has_inf": bool(torch.isinf(actual_fp32).any().item()),
    }


def _format_comparison(name, metrics):
    return (
        f"    {name:<30} max_abs={metrics['max_abs']:.6e}, "
        f"max_rel={metrics['max_rel']:.6e}, NaN={metrics['has_nan']}, "
        f"Inf={metrics['has_inf']}, close={metrics['close']}"
    )


def _make_inputs(kv_len, seed):
    torch.manual_seed(seed)
    Q = torch.randn(
        1, 1, NUM_HEADS, HEAD_DIM, dtype=DTYPE_TORCH, device=device
    )
    K = torch.randn(
        1, kv_len, NUM_HEADS, HEAD_DIM, dtype=DTYPE_TORCH, device=device
    )
    V = torch.randn_like(K)
    return Q, K, V


def sdpa_ref(Q, K_visible, V_visible, softmax_scale):
    Q_t, K_t, V_t = (
        tensor.transpose(1, 2) for tensor in (Q, K_visible, V_visible)
    )
    return F.scaled_dot_product_attention(
        Q_t,
        K_t,
        V_t,
        is_causal=False,
        scale=float(softmax_scale),
    ).transpose(1, 2)


def _run_original_decode(Q, K_visible, V_visible, softmax_scale):
    run_original, (_placeholder_q, _placeholder_k, _placeholder_v, O_original) = (
        compile_cute_attn(1, K_visible.shape[1], float(softmax_scale))
    )
    return run_original(q=Q, k=K_visible, v=V_visible, o=O_original).clone()


def _merge_raw_states(maximum, denominator, numerator):
    global_max = maximum.amax(dim=1)
    split_scale = torch.exp(maximum - global_max.unsqueeze(1))
    merged_sum = (split_scale * denominator).sum(dim=1)
    merged_output = (
        split_scale.unsqueeze(-1) * numerator
    ).sum(dim=1) / merged_sum.unsqueeze(-1)
    return merged_output.view(1, 1, NUM_HEADS, HEAD_DIM)


def _tiled_fp32_raw_oracle(Q, K, V, plan, softmax_scale, tile_size=ORACLE_TILE_SIZE):
    """Return raw FP32 m/l/p per logical split without full K/V FP32 copies."""
    q_fp32 = Q[0, 0].to(torch.float32)
    split_max = []
    split_sum = []
    split_output = []

    for chunk in plan.chunks:
        chunk_max = None
        chunk_sum = None
        chunk_output = None
        for tile_start in range(chunk.start, chunk.end, tile_size):
            tile_end = min(tile_start + tile_size, chunk.end)
            K_tile = K[0, tile_start:tile_end].to(torch.float32)
            V_tile = V[0, tile_start:tile_end].to(torch.float32)
            scores = torch.einsum("hd,thd->ht", q_fp32, K_tile)
            scores.mul_(float(softmax_scale))

            tile_max = scores.amax(dim=-1)
            tile_exp = torch.exp(scores - tile_max.unsqueeze(-1))
            tile_sum = tile_exp.sum(dim=-1)
            tile_output = torch.einsum("ht,thd->hd", tile_exp, V_tile)

            if chunk_max is None:
                chunk_max = tile_max
                chunk_sum = tile_sum
                chunk_output = tile_output
            else:
                merged_max = torch.maximum(chunk_max, tile_max)
                old_scale = torch.exp(chunk_max - merged_max)
                tile_scale = torch.exp(tile_max - merged_max)
                chunk_sum = old_scale * chunk_sum + tile_scale * tile_sum
                chunk_output = (
                    old_scale.unsqueeze(-1) * chunk_output
                    + tile_scale.unsqueeze(-1) * tile_output
                )
                chunk_max = merged_max

        split_max.append(chunk_max)
        split_sum.append(chunk_sum)
        split_output.append(chunk_output)

    return Stage1Workspace(
        partial_max=torch.stack(split_max, dim=1),
        partial_sum=torch.stack(split_sum, dim=1),
        partial_output=torch.stack(split_output, dim=1),
    )


def run_mapping_tests():
    """CPU-only split mapping, view, rejection and workspace checks."""
    cases = (
        (10, None, 3, ((0, 4), (4, 7), (7, 10))),
        (13, 9, 3, ((0, 4), (4, 7), (7, 10))),
        (7, None, 4, ((0, 2), (2, 4), (4, 6), (6, 7))),
        (8, None, 4, ((0, 2), (2, 4), (4, 6), (6, 8))),
        (5, None, 5, ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5))),
    )

    for physical_kv_len, query_position, split_count, expected_bounds in cases:
        visible_kv_len = resolve_visible_kv_len(physical_kv_len, query_position)
        plan = build_split_plan(visible_kv_len, split_count)
        actual_bounds = tuple((chunk.start, chunk.end) for chunk in plan.chunks)
        if actual_bounds != expected_bounds:
            raise AssertionError(
                f"boundaries 不匹配：expected={expected_bounds}, actual={actual_bounds}"
            )
        if any(chunk.length <= 0 for chunk in plan.chunks):
            raise AssertionError("发现零长度 split")

        source = torch.arange(physical_kv_len * 6, dtype=torch.int64).view(
            1, physical_kv_len, 2, 3
        )
        group_views = make_kv_group_views(source, plan, "mapping_source")
        reconstructed_chunks = []
        for group, group_view in zip(plan.groups, group_views):
            if group_view.untyped_storage().data_ptr() != source.untyped_storage().data_ptr():
                raise AssertionError("group view 没有与 visible source 共享 storage")
            if not group_view.is_contiguous():
                raise AssertionError("group view 不是 contiguous")
            reconstructed_chunks.extend(
                group_view[index] for index in range(group.split_count)
            )
        reconstructed = torch.cat(reconstructed_chunks, dim=0)
        if not torch.equal(reconstructed, source[0, :visible_kv_len]):
            raise AssertionError("logical chunks 无法还原 visible prefix")

        O_split = torch.full((split_count, 1, 2, 2), -1, dtype=torch.int64)
        O_groups = make_output_group_views(O_split, plan)
        for group, O_group in zip(plan.groups, O_groups):
            for local_index in range(group.split_count):
                O_group[local_index].fill_(group.first_split + local_index)
        if not torch.equal(
            O_split[:, 0, 0, 0], torch.arange(split_count, dtype=torch.int64)
        ):
            raise AssertionError("group 写回 O_split 的逻辑顺序不正确")

    _expect_exception(ValueError, lambda: build_split_plan(3, 4), "split > visible")
    _expect_exception(ValueError, lambda: build_split_plan(3, 0), "split == 0")
    _expect_exception(ValueError, lambda: resolve_visible_kv_len(3, -1), "negative query")
    _expect_exception(ValueError, lambda: resolve_visible_kv_len(3, 3), "query == kv_len")

    for split_count, expected_bytes in EXPECTED_WORKSPACE_BYTES.items():
        workspace = allocate_stage1_workspace(split_count, device="cpu")
        actual_bytes = validate_stage1_workspace(
            workspace, split_count, device="cpu", require_cuda=False
        )
        if actual_bytes != expected_bytes:
            raise AssertionError(
                f"S={split_count}: workspace={actual_bytes}, expected={expected_bytes}"
            )
        if stage1_workspace_nbytes(split_count) != expected_bytes:
            raise AssertionError("workspace formula 与实际分配不一致")
        print(f"  workspace S={split_count:>2}: {actual_bytes} bytes")

    print("[PASS] mapping tests")
    return True


def run_existing_local_checks(atol, rtol, seed):
    """Check the old reference kernel's per-chunk local normalized outputs."""
    Q, K, V = _make_inputs(9, seed)
    scale = HEAD_DIM**-0.5
    run_once, run_local, _combine, plan = compile_split_kv_scaffold(
        Q, K, V, 2, scale, query_position=7
    )
    del run_once
    local_cute = run_local().clone()
    torch.cuda.synchronize()

    all_ok = True
    for chunk in plan.chunks:
        local_sdpa = sdpa_ref(
            Q,
            K[:, chunk.start:chunk.end],
            V[:, chunk.start:chunk.end],
            scale,
        )
        metrics = _comparison(
            local_cute[chunk.split_index : chunk.split_index + 1],
            local_sdpa,
            atol,
            rtol,
        )
        print(_format_comparison(f"existing local split {chunk.split_index}", metrics))
        all_ok = all_ok and metrics["close"]
    print(f"[{'PASS' if all_ok else 'FAIL'}] existing-local")
    return all_ok


def run_bf16_limitation_diagnostic(atol, rtol, seed):
    """Reproduce the known BF16 local-normalized merge precision limitation."""
    Q, K, V = _make_inputs(9, seed)
    scale = HEAD_DIM**-0.5
    run_once, run_local, combine, plan = compile_split_kv_scaffold(
        Q, K, V, 2, scale, query_position=7
    )
    del run_once
    local_cute = run_local().clone()
    legacy_output = combine().clone()
    K_visible = K[:, : plan.visible_kv_len]
    V_visible = V[:, : plan.visible_kv_len]
    full_sdpa = sdpa_ref(Q, K_visible, V_visible, scale)
    raw_oracle = _tiled_fp32_raw_oracle(Q, K, V, plan, scale)
    exact_merged = _merge_raw_states(
        raw_oracle.partial_max,
        raw_oracle.partial_sum,
        raw_oracle.partial_output,
    )
    full_plan = build_split_plan(plan.visible_kv_len, 1)
    full_oracle = _tiled_fp32_raw_oracle(Q, K, V, full_plan, scale)
    exact_full = _merge_raw_states(
        full_oracle.partial_max,
        full_oracle.partial_sum,
        full_oracle.partial_output,
    )

    local_ok = True
    for chunk in plan.chunks:
        local_sdpa = sdpa_ref(
            Q,
            K[:, chunk.start:chunk.end],
            V[:, chunk.start:chunk.end],
            scale,
        )
        local_metrics = _comparison(
            local_cute[chunk.split_index : chunk.split_index + 1],
            local_sdpa,
            atol,
            rtol,
        )
        local_ok = local_ok and local_metrics["close"]
        print(_format_comparison(f"local split {chunk.split_index}", local_metrics))

    exact_metrics = _comparison(exact_merged, exact_full, atol, rtol)
    legacy_metrics = _comparison(legacy_output, full_sdpa, atol, rtol)
    print(_format_comparison("exact FP32 merge vs full", exact_metrics))
    print(_format_comparison("BF16 normalized merge vs SDPA", legacy_metrics))

    if not local_ok or not exact_metrics["close"]:
        print("[FAIL] bf16-diagnostic prerequisites")
        return False
    if legacy_metrics["has_nan"] or legacy_metrics["has_inf"]:
        print("[FAIL] bf16-diagnostic produced NaN/Inf")
        return False
    if legacy_metrics["close"]:
        print("[XPASS] known BF16-normalized-output limitation not reproduced")
    else:
        print("[EXPECTED_LIMITATION] BF16 local normalized O loses merge precision")
    return True


def _stage1_smoke_inputs(seed):
    Q, K, V = _make_inputs(9, seed)
    return Q, K, V, 2, 7, HEAD_DIM**-0.5


def run_stage1_compile(seed):
    Q, K, V, split_count, query_position, scale = _stage1_smoke_inputs(seed)
    _run_stage1, workspace, plan = compile_split_kv_stage1(
        Q, K, V, split_count, scale, query_position=query_position
    )
    print(
        f"[PASS] stage1-compile: visible={plan.visible_kv_len}, split={split_count}, "
        f"workspace={workspace.nbytes} bytes"
    )
    return True


def run_stage1_smoke(seed):
    Q, K, V, split_count, query_position, scale = _stage1_smoke_inputs(seed)
    run_stage1, workspace, plan = compile_split_kv_stage1(
        Q, K, V, split_count, scale, query_position=query_position
    )
    run_stage1()
    torch.cuda.synchronize()

    valid = (
        bool(torch.isfinite(workspace.partial_max).all().item())
        and bool(torch.isfinite(workspace.partial_sum).all().item())
        and bool(torch.isfinite(workspace.partial_output).all().item())
        and bool((workspace.partial_sum > 0).all().item())
    )
    print(
        f"[{'PASS' if valid else 'FAIL'}] stage1-smoke: visible={plan.visible_kv_len}, "
        f"split={split_count}, m={tuple(workspace.partial_max.shape)}, "
        f"l={tuple(workspace.partial_sum.shape)}, p={tuple(workspace.partial_output.shape)}"
    )
    return valid


def run_stage1_case(case, atol, rtol, seed):
    Q, K, V = _make_inputs(case.kv_len, seed)
    visible_kv_len = resolve_visible_kv_len(case.kv_len, case.query_position)
    K_visible = K[:, :visible_kv_len]
    V_visible = V[:, :visible_kv_len]
    scale = HEAD_DIM**-0.5

    O_sdpa = sdpa_ref(Q, K_visible, V_visible, scale)
    O_original = _run_original_decode(Q, K_visible, V_visible, scale)
    original_metrics = _comparison(O_original, O_sdpa, atol, rtol)
    print(
        f"\ncase={case.label}, physical_kv={case.kv_len}, "
        f"query_position={case.query_position}, visible_kv={visible_kv_len}, "
        f"atol={atol:g}, rtol={rtol:g}"
    )
    print(_format_comparison("original CuTe vs SDPA", original_metrics))
    if not original_metrics["close"]:
        print("[FAIL] original CuTe baseline")
        return False

    full_plan = build_split_plan(visible_kv_len, 1)
    full_oracle = _tiled_fp32_raw_oracle(Q, K, V, full_plan, scale)
    O_fp32_oracle = _merge_raw_states(
        full_oracle.partial_max,
        full_oracle.partial_sum,
        full_oracle.partial_output,
    )

    for split_count in case.split_counts:
        run_stage1, workspace, plan = compile_split_kv_stage1(
            Q,
            K,
            V,
            split_count,
            scale,
            query_position=case.query_position,
        )
        run_stage1()
        torch.cuda.synchronize()
        local_oracle = _tiled_fp32_raw_oracle(Q, K, V, plan, scale)

        local_m = _comparison(
            workspace.partial_max, local_oracle.partial_max, atol, rtol
        )
        local_l = _comparison(
            workspace.partial_sum, local_oracle.partial_sum, atol, rtol
        )
        local_p = _comparison(
            workspace.partial_output, local_oracle.partial_output, atol, rtol
        )
        O_stage1_fp32 = reduce_stage1_fp32(
            workspace, output_dtype=torch.float32
        )
        O_stage1_bf16 = O_stage1_fp32.to(DTYPE_TORCH)
        final_vs_fp32 = _comparison(O_stage1_fp32, O_fp32_oracle, atol, rtol)
        final_vs_sdpa = _comparison(O_stage1_bf16, O_sdpa, atol, rtol)
        final_vs_original = _comparison(O_stage1_bf16, O_original, atol, rtol)

        print(f"  split={split_count}")
        for name, metrics in (
            ("local m vs FP32 oracle", local_m),
            ("local l vs FP32 oracle", local_l),
            ("local p vs FP32 oracle", local_p),
            ("Stage1 FP32 vs FP32 oracle", final_vs_fp32),
            ("Stage1 BF16 vs SDPA", final_vs_sdpa),
            ("Stage1 BF16 vs original CuTe", final_vs_original),
        ):
            print(_format_comparison(name, metrics))

        split_ok = all(
            metrics["close"]
            for metrics in (
                local_m,
                local_l,
                local_p,
                final_vs_fp32,
                final_vs_sdpa,
                final_vs_original,
            )
        )
        print(f"  [{'PASS' if split_ok else 'FAIL'}] {case.label}/split={split_count}")
        if not split_ok:
            return False
    return True


def run_stage1_small(atol, rtol, seed):
    for case_index, case in enumerate(STAGE1_CASES):
        if not run_stage1_case(case, atol, rtol, seed + case_index):
            print("首个 Stage-1 失败已定位；停止后续 case。")
            return False
    print("[PASS] stage1-small")
    return True


def _stage2_smoke_inputs(seed):
    Q, K, V = _make_inputs(9, seed)
    return Q, K, V, 2, 7, HEAD_DIM**-0.5


def run_stage2_compile(seed):
    Q, K, V, split_count, query_position, scale = _stage2_smoke_inputs(seed)
    run_stage1, workspace, plan = compile_split_kv_stage1(
        Q, K, V, split_count, scale, query_position=query_position
    )
    run_stage1()
    torch.cuda.synchronize()
    _run_stage2, output_fp32, output_dtype = compile_split_kv_stage2(
        workspace, output_dtype=torch.float32
    )
    print(
        f"[PASS] stage2-compile: visible={plan.visible_kv_len}, "
        f"split={split_count}, output={tuple(output_fp32.shape)} "
        f"dtype={output_fp32.dtype}->{output_dtype}"
    )
    return True


def run_stage2_smoke(atol, rtol, seed):
    Q, K, V, split_count, query_position, scale = _stage2_smoke_inputs(seed)
    run_stage1, workspace, plan = compile_split_kv_stage1(
        Q, K, V, split_count, scale, query_position=query_position
    )
    run_stage1()
    torch.cuda.synchronize()
    output_gpu_fp32, output_fp32 = run_split_kv_gpu_reduction(
        workspace, output_dtype=torch.float32
    )
    torch.cuda.synchronize()

    output_torch_fp32 = reduce_stage1_fp32(
        workspace, output_dtype=torch.float32
    )
    metrics = _comparison(output_gpu_fp32, output_torch_fp32, atol, rtol)
    finite = (
        bool(torch.isfinite(output_gpu_fp32).all().item())
        and bool(torch.isfinite(output_fp32).all().item())
    )
    valid = finite and metrics["close"]
    print(
        f"[{'PASS' if valid else 'FAIL'}] stage2-smoke: visible={plan.visible_kv_len}, "
        f"split={split_count}, output={tuple(output_gpu_fp32.shape)} "
        f"dtype={output_gpu_fp32.dtype}"
    )
    print(_format_comparison("GPU Stage2 vs PyTorch FP32", metrics))
    return valid


def run_stage2_case(case, atol, rtol, seed):
    Q, K, V = _make_inputs(case.kv_len, seed)
    visible_kv_len = resolve_visible_kv_len(case.kv_len, case.query_position)
    K_visible = K[:, :visible_kv_len]
    V_visible = V[:, :visible_kv_len]
    scale = HEAD_DIM**-0.5

    O_sdpa = sdpa_ref(Q, K_visible, V_visible, scale)
    O_original = _run_original_decode(Q, K_visible, V_visible, scale)
    original_metrics = _comparison(O_original, O_sdpa, atol, rtol)
    print(
        f"\ncase={case.label}, physical_kv={case.kv_len}, "
        f"query_position={case.query_position}, visible_kv={visible_kv_len}, "
        f"atol={atol:g}, rtol={rtol:g}"
    )
    print(_format_comparison("original CuTe vs SDPA", original_metrics))
    if not original_metrics["close"]:
        print("[FAIL] original CuTe baseline")
        return False

    full_plan = build_split_plan(visible_kv_len, 1)
    full_oracle = _tiled_fp32_raw_oracle(Q, K, V, full_plan, scale)
    O_fp32_oracle = _merge_raw_states(
        full_oracle.partial_max,
        full_oracle.partial_sum,
        full_oracle.partial_output,
    )

    for split_count in case.split_counts:
        run_stage1, workspace, plan = compile_split_kv_stage1(
            Q,
            K,
            V,
            split_count,
            scale,
            query_position=case.query_position,
        )
        run_stage1()
        torch.cuda.synchronize()
        local_oracle = _tiled_fp32_raw_oracle(Q, K, V, plan, scale)

        local_m = _comparison(
            workspace.partial_max, local_oracle.partial_max, atol, rtol
        )
        local_l = _comparison(
            workspace.partial_sum, local_oracle.partial_sum, atol, rtol
        )
        local_p = _comparison(
            workspace.partial_output, local_oracle.partial_output, atol, rtol
        )
        if not (local_m["close"] and local_l["close"] and local_p["close"]):
            print(f"  [FAIL] {case.label}/split={split_count}: Stage-1 input invalid")
            for name, metrics in (
                ("local m vs FP32 oracle", local_m),
                ("local l vs FP32 oracle", local_l),
                ("local p vs FP32 oracle", local_p),
            ):
                print(_format_comparison(name, metrics))
            return False

        output_gpu_fp32, output_fp32 = run_split_kv_gpu_reduction(
            workspace, output_dtype=torch.float32
        )
        output_torch_fp32 = reduce_stage1_fp32(
            workspace, output_dtype=torch.float32
        )
        output_gpu_bf16 = output_gpu_fp32.to(DTYPE_TORCH)

        stage2_vs_torch = _comparison(
            output_gpu_fp32, output_torch_fp32, atol, rtol
        )
        pipeline_vs_fp32 = _comparison(
            output_gpu_fp32, O_fp32_oracle, atol, rtol
        )
        pipeline_vs_sdpa = _comparison(
            output_gpu_bf16, O_sdpa, atol, rtol
        )
        pipeline_vs_original = _comparison(
            output_gpu_bf16, O_original, atol, rtol
        )

        print(f"  split={split_count}, workspace={workspace.nbytes} bytes")
        for name, metrics in (
            ("GPU Stage2 vs PyTorch FP32", stage2_vs_torch),
            ("GPU pipeline vs FP32 oracle", pipeline_vs_fp32),
            ("GPU pipeline BF16 vs SDPA", pipeline_vs_sdpa),
            ("GPU pipeline BF16 vs original", pipeline_vs_original),
        ):
            print(_format_comparison(name, metrics))

        split_ok = all(
            metrics["close"]
            for metrics in (
                stage2_vs_torch,
                pipeline_vs_fp32,
                pipeline_vs_sdpa,
                pipeline_vs_original,
            )
        )
        print(f"  [{'PASS' if split_ok else 'FAIL'}] {case.label}/split={split_count}")
        if not split_ok:
            return False
    return True


def run_stage2_cases(cases, label, atol, rtol, seed):
    for case_index, case in enumerate(cases):
        if not run_stage2_case(case, atol, rtol, seed + case_index):
            print(f"首个 Stage-2 失败已定位；停止 {label} 后续 case。")
            return False
    print(f"[PASS] {label}")
    return True


def _print_scope_notice(suite, atol, rtol):
    print("========== Decode Split-KV layered correctness ==========")
    print(f"suite={suite}, atol={atol:g}, rtol={rtol:g}")
    print("Stage-1 输出 raw FP32 m/l/p；Stage-2 可为 GPU FP32 reduction。")
    print("不运行 benchmark，不报告 latency/speedup；性能仍未验证。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=[
            "mapping",
            "existing-local",
            "bf16-diagnostic",
            "stage1-compile",
            "stage1-smoke",
            "stage1-small",
            "stage2-compile",
            "stage2-smoke",
            "stage2-small",
            "all-small",
        ],
        default="mapping",
    )
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()

    if (
        not math.isfinite(args.atol)
        or not math.isfinite(args.rtol)
        or args.atol < 0
        or args.rtol < 0
    ):
        parser.error("--atol/--rtol 必须为有限非负数")

    _print_scope_notice(args.suite, args.atol, args.rtol)
    if args.suite == "mapping":
        ok = run_mapping_tests()
    elif args.suite == "existing-local":
        ok = run_existing_local_checks(args.atol, args.rtol, args.seed)
    elif args.suite == "bf16-diagnostic":
        ok = run_bf16_limitation_diagnostic(args.atol, args.rtol, args.seed)
    elif args.suite == "stage1-compile":
        ok = run_stage1_compile(args.seed)
    elif args.suite == "stage1-smoke":
        ok = run_stage1_smoke(args.seed)
    elif args.suite == "stage1-small":
        ok = run_stage1_small(args.atol, args.rtol, args.seed)
    elif args.suite == "stage2-compile":
        ok = run_stage2_compile(args.seed)
    elif args.suite == "stage2-smoke":
        ok = run_stage2_smoke(args.atol, args.rtol, args.seed)
    elif args.suite == "stage2-small":
        ok = run_stage2_cases(
            STAGE2_SMALL_CASES,
            "stage2-small",
            args.atol,
            args.rtol,
            args.seed,
        )
    else:
        ok = run_mapping_tests()
        if ok:
            ok = run_existing_local_checks(args.atol, args.rtol, args.seed)
        if ok:
            ok = run_bf16_limitation_diagnostic(args.atol, args.rtol, args.seed)
        if ok:
            ok = run_stage1_smoke(args.seed)
        if ok:
            ok = run_stage1_small(args.atol, args.rtol, args.seed)

    print(f"========== suite {args.suite}: {'PASS' if ok else 'FAIL'} ==========")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
