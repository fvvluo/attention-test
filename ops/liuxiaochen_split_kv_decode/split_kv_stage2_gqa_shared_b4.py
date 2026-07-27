#!/usr/bin/env python3
"""Two-level pure-GPU combine for B4 (supports split up to 1024).

Level-1 (only used when split_count > SINGLE_LEVEL_MAX): partition the
``split_count`` per-split states into ``num_groups`` contiguous groups of size
``group_span`` (<=256). Each CTA (one per (q_head, group)) merges its group's
raw (m,l,p) into a group-level raw (m,l,p) written to an intermediate FP32
workspace of shape [Q_HEADS, num_groups] / [Q_HEADS, num_groups, D].

Level-2: merge the ``num_groups`` group states into the final BF16 output. When
split_count <= SINGLE_LEVEL_MAX we skip level-1 and directly merge all splits
(this is the same math as split_kv_stage2_gqa_shared).

All FP32 intermediates; no torch.max/sum; no CPU; pure CuTe kernels.
"""

import torch

if not hasattr(torch, "float4_e2m1fn_x2"):
    torch.float4_e2m1fn_x2 = torch.uint8

import cuda.bindings.driver as cuda  # noqa: E402
import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass.cute.runtime import from_dlpack  # noqa: E402

from split_kv_stage1_gqa import DTYPE_TORCH, HEAD_DIM, Q_HEADS  # noqa: E402

NUM_THREADS = HEAD_DIM  # 128
SINGLE_LEVEL_MAX = 256  # split_count <= this -> single-level merge
GROUP_SPAN = 128        # level-1 group size for split>256


def _mk(t):
    return from_dlpack(t, assumed_align=16)


class _MergeToRaw:
    """Merge a contiguous span of splits into group-level raw (m,l,p). Grid [Q_HEADS,num_groups]."""

    def __init__(self, group_span, num_groups, in_splits):
        self.group_span = group_span
        self.num_groups = num_groups
        self.in_splits = in_splits

    @cute.jit
    def __call__(self, mMax, mSum, mP, oMax, oSum, oP, stream):
        self.kernel(mMax, mSum, mP, oMax, oSum, oP).launch(
            grid=[Q_HEADS, self.num_groups, 1], block=[NUM_THREADS, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, mMax, mSum, mP, oMax, oSum, oP):
        q_head, grp, _ = cute.arch.block_idx()
        d_idx, _, _ = cute.arch.thread_idx()
        start = grp * self.group_span
        # group max
        gmax = -cutlass.Float32.inf
        for j in range(self.group_span):
            s = start + j
            if s < self.in_splits:
                gmax = cute.arch.fmax(gmax, mMax[q_head, s])
        denom = cutlass.Float32(0.0)
        numer = cutlass.Float32(0.0)
        for j in range(self.group_span):
            s = start + j
            if s < self.in_splits:
                w = cute.math.exp(mMax[q_head, s] - gmax, fastmath=False)
                denom += w * mSum[q_head, s]
                numer += w * mP[q_head, s, d_idx]
        # write group-level raw state: m=gmax, l=denom, p=numer (un-normalized, shifted by gmax)
        if d_idx == 0:
            oMax[q_head, grp] = gmax
            oSum[q_head, grp] = denom
        oP[q_head, grp, d_idx] = numer


class _MergeToOut:
    """Merge all (group or split) raw states into final BF16 out. Grid [Q_HEADS,1]."""

    def __init__(self, n_in):
        self.n_in = n_in

    @cute.jit
    def __call__(self, mMax, mSum, mP, mOut, stream):
        self.kernel(mMax, mSum, mP, mOut).launch(
            grid=[Q_HEADS, 1, 1], block=[NUM_THREADS, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, mMax, mSum, mP, mOut):
        q_head, _, _ = cute.arch.block_idx()
        d_idx, _, _ = cute.arch.thread_idx()
        gmax = -cutlass.Float32.inf
        for s in range(self.n_in):
            gmax = cute.arch.fmax(gmax, mMax[q_head, s])
        denom = cutlass.Float32(0.0)
        numer = cutlass.Float32(0.0)
        for s in range(self.n_in):
            w = cute.math.exp(mMax[q_head, s] - gmax, fastmath=False)
            denom += w * mSum[q_head, s]
            numer += w * mP[q_head, s, d_idx]
        mOut[0, q_head, 0, d_idx] = (numer / denom).to(mOut.element_type)


def _alloc_group_ws(num_groups, device):
    return (
        torch.empty(Q_HEADS, num_groups, dtype=torch.float32, device=device),
        torch.empty(Q_HEADS, num_groups, dtype=torch.float32, device=device),
        torch.empty(Q_HEADS, num_groups, HEAD_DIM, dtype=torch.float32, device=device),
    )


def compile_combine_b4(workspace, output=None):
    """Return (run_combine, output). Two-level when split>SINGLE_LEVEL_MAX."""
    split_count = int(workspace.partial_max.shape[1])
    device = workspace.partial_max.device
    if output is None:
        output = torch.empty(1, Q_HEADS, 1, HEAD_DIM, dtype=DTYPE_TORCH, device=device)

    torch_stream = torch.cuda.current_stream(device)
    stream = cuda.CUstream(torch_stream.cuda_stream)

    if split_count <= SINGLE_LEVEL_MAX:
        k2 = _MergeToOut(split_count)
        c2 = cute.compile(
            k2, _mk(workspace.partial_max), _mk(workspace.partial_sum),
            _mk(workspace.partial_output), _mk(output), stream,
        )

        def run_combine():
            c2(_mk(workspace.partial_max), _mk(workspace.partial_sum),
               _mk(workspace.partial_output), _mk(output), stream)
            return output
        return run_combine, output

    # two-level
    num_groups = (split_count + GROUP_SPAN - 1) // GROUP_SPAN
    gMax, gSum, gP = _alloc_group_ws(num_groups, device)
    k1 = _MergeToRaw(GROUP_SPAN, num_groups, split_count)
    c1 = cute.compile(
        k1, _mk(workspace.partial_max), _mk(workspace.partial_sum), _mk(workspace.partial_output),
        _mk(gMax), _mk(gSum), _mk(gP), stream,
    )
    k2 = _MergeToOut(num_groups)
    c2 = cute.compile(k2, _mk(gMax), _mk(gSum), _mk(gP), _mk(output), stream)

    def run_combine():
        c1(_mk(workspace.partial_max), _mk(workspace.partial_sum), _mk(workspace.partial_output),
           _mk(gMax), _mk(gSum), _mk(gP), stream)
        c2(_mk(gMax), _mk(gSum), _mk(gP), _mk(output), stream)
        return output

    return run_combine, output
