# Copyright (c) 2026. Prefill-optimized FlashAttention for NVIDIA Hopper (SM90).
# SPDX-License-Identifier: BSD-3-Clause
#
# This module wraps a warp-specialized, TMA + WGMMA FlashAttention forward kernel
# written in the CUTLASS CUTE Python DSL, specialized for the *prefill* phase of
# LLM inference / training.
#
# Prefill characteristics (vs. decode):
#   * Q sequence length is large (== KV length), so the workload is strongly
#     COMPUTE-bound rather than memory-bound.
#   * The dominant cost is the two batched GEMMs  S = Q @ K^T  and  O = softmax(S) @ V.
#   * To saturate the Hopper TensorCores we therefore use:
#       - Large CTA tiles (2 MMA warp-groups => 256 rows of Q per CTA).
#       - Warp specialization: 1 producer warp-group issuing TMA loads, 2 consumer
#         warp-groups running back-to-back WGMMA (QK^T and PV) fully overlapped with
#         the online-softmax rescale.
#       - Deep multi-stage smem pipeline (K/V double/quintuple buffered) so the
#         TensorCores never stall on global memory.
#       - Persistent scheduling: CTAs stay resident and stream tiles, amortizing
#         launch / prologue overhead across the many tiles of a long sequence.
#   * FP32 accumulation for numerical fidelity of the softmax.
#
# The underlying device kernel lives in ``fmha_prefill_kernel.py`` (class
# ``HopperFusedMultiHeadAttentionForward``). This file provides a clean, cached,
# PyTorch-friendly entry point:  ``prefill_attention(q, k, v, causal=...)``.
#
# References:
#   * Dao et al., "FlashAttention-2" (2023)
#   * Shah et al., "FlashAttention-3: Fast and Accurate Attention with
#     Asynchrony and Low-precision" (2024)  -- warp-specialized producer/consumer,
#     WGMMA/TMA overlap, which this kernel structurally follows.
#   * NVIDIA CUTLASS CUTE-DSL Hopper FMHA example.

import math
from functools import lru_cache
from typing import Optional, Tuple

import torch

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass.cute.runtime import from_dlpack

try:  # dual-mode import: works as a package (from .) or as flat modules on sys.path
    from . import _prefill9_helpers as fmha_utils
    from ._prefill9_kernel import HopperFusedMultiHeadAttentionForward
except ImportError:
    import _prefill9_helpers as fmha_utils
    from _prefill9_kernel import HopperFusedMultiHeadAttentionForward

LOG2_E = 1.4426950408889634074

# Map torch dtype -> cutlass numeric type.
_TORCH_TO_CUTLASS = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
}

_FP8_DTYPES = {torch.float8_e4m3fn}


def _prefill_config(head_dim: int) -> Tuple[Tuple[int, int], bool]:
    """Return (mma_tiler_mn, is_persistent) tuned for prefill at a given head dim.

    Each MMA warp-group processes ``mma_tiler_mn[0]`` rows of Q; with 2 MMA
    warp-groups the effective CTA row tile is ``2 * mma_tiler_mn[0]``.  The N
    dimension (KV block) is ``mma_tiler_mn[1]``.  The K dimension of the MMA is
    the head dim.

    For prefill we prefer the largest tiles that fit in Hopper's 228 KB smem so
    the TensorCores stay saturated:
      * D<=128: (64, 128) -> 128x128 CTA tile, persistent.  This is the sweet
        spot that maxes WGMMA occupancy while leaving smem for a deep KV pipeline.
      * D==256: (64, 128) non-persistent (smem pressure is higher for D=256).
    """
    if head_dim <= 128:
        return (64, 128), True
    else:  # 256
        return (64, 128), False


class _PrefillPlan:
    """Compiled, reusable prefill-attention plan for one static problem config.

    Compilation (JIT of the CUTE kernel) is expensive, so we cache one plan per
    unique (shape, dtype, causal) signature and reuse the compiled kernel across
    calls with matching signatures.
    """

    def __init__(
        self,
        h: int,
        h_k: int,
        d: int,
        in_dtype,
        out_dtype,
        causal: bool,
        sm_scale: float,
        kv_aligned: bool = True,
        tuning: Optional[dict] = None,
        want_lse: bool = True,
    ):
        self.h, self.h_k, self.d = h, h_k, d
        self.in_dtype, self.out_dtype = in_dtype, out_dtype
        self.causal = causal
        self.sm_scale = sm_scale
        self.want_lse = want_lse
        self.h_r = h // h_k
        self.is_fp8 = in_dtype == cutlass.Float8E4M3FN
        self.kv_aligned = kv_aligned
        # Map cutlass out dtype back to a torch dtype for output allocation.
        _CUTLASS_TO_TORCH = {
            cutlass.Float16: torch.float16,
            cutlass.BFloat16: torch.bfloat16,
            cutlass.Float8E4M3FN: torch.float8_e4m3fn,
        }
        self.out_torch_dtype = _CUTLASS_TO_TORCH[out_dtype]

        tuning = dict(tuning) if tuning else {}
        # Tuned prefill defaults (empirically best on H20).
        #   FP16/BF16, D<=128: kv_stage=4 is the deepest KV pipeline that fits
        #     smem at the 128x128 tile; the FP16 Tensor cores are the bottleneck.
        #   FP8, D<=128: K/V are 1 byte, so a wider N=192 KV block fits and feeds
        #     the 2x-faster FP8 Tensor cores better (263 vs 260 TFLOPS at N=128).
        mma_tiler_mn, is_persistent = _prefill_config(d)
        if d <= 128:
            if self.is_fp8:
                mma_tiler_mn = (64, 192)
                tuning.setdefault("kv_stage", 5)
            else:
                tuning.setdefault("kv_stage", 4)
                # Non-persistent launches many CTAs (better multi-wave latency
                # hiding of the softmax gaps) and measured marginally faster than
                # persistent for BF16 prefill on H20 (~143.3 vs ~142.9).
                tuning.setdefault("is_persistent", False)
        mma_tiler_mn = tuning.get("mma_tiler_mn", mma_tiler_mn)
        is_persistent = tuning.get("is_persistent", is_persistent)
        self.mma_tiler = (*mma_tiler_mn, d)
        self.is_persistent = is_persistent
        self._tuning = tuning

        # Mask selection for prefill.  Depends only on `causal` and whether the
        # KV length is tile-aligned (kv_aligned) -- NOT the exact seqlen -- so the
        # plan (and its expensive compilation) is shared across all seqlens of the
        # same alignment class.
        if causal:
            self.mask_type = fmha_utils.MaskEnum.WINDOW_MASK
            self.window_size_left = None
            self.window_size_right = 0
        else:
            self.window_size_left = None
            self.window_size_right = None
            # Residual mask only needed when KV length is not a multiple of the
            # N tile; otherwise the cheaper no-mask fast path is valid.
            if not kv_aligned:
                self.mask_type = fmha_utils.MaskEnum.RESIDUAL_MASK
            else:
                self.mask_type = fmha_utils.MaskEnum.WINDOW_MASK

        self.scale_softmax = sm_scale
        self.scale_softmax_log2 = sm_scale * LOG2_E
        self.scale_output = 1.0

        # Specialization: skip LSE compute+writeback when the caller doesn't
        # need it (pure inference).  Read by the kernel's __init__.
        HopperFusedMultiHeadAttentionForward._override_write_lse = want_lse
        self._fmha = HopperFusedMultiHeadAttentionForward(
            cutlass.Float32,  # qk_acc_dtype
            cutlass.Float32,  # pv_acc_dtype
            self.mma_tiler,
            self.is_persistent,
            self.mask_type,
        )

        # Apply pipeline / register tuning overrides onto the kernel instance.
        if "kv_stage" in tuning or "q_stage" in tuning or "epi_stage" in tuning:
            qs = tuning.get("q_stage", 2)
            kvs = tuning.get("kv_stage", 5)
            eps = tuning.get("epi_stage", 2)

            def _setup(self_k, _qs=qs, _kvs=kvs, _eps=eps):
                self_k.q_stage = _qs
                self_k.kv_stage = _kvs
                self_k.epi_stage = _eps

            import types
            self._fmha._setup_attributes = types.MethodType(_setup, self._fmha)
        if "num_regs_mma" in tuning:
            self._fmha.num_regs_mma = tuning["num_regs_mma"]
        if "num_regs_load" in tuning:
            self._fmha.num_regs_load = tuning["num_regs_load"]

        self._compiled = None  # lazily compiled on first call with real tensors

    # ---- tensor layout helpers -------------------------------------------------
    # The device kernel expects tensors laid out as (S, D, ((H_r, H_k), B)) with a
    # K-major (contiguous D) leading dimension for Q/K and specific handling for V.
    # We take standard torch tensors in (B, S, H, D) and permute a *view* so the
    # kernel's expected 5D grouping (B, S, H_k, H_r, D) -> permuted is satisfied.

    def _make_cute_tensor(self, t5d: torch.Tensor, dtype, leading_dim: int):
        ct = from_dlpack(t5d, assumed_align=16)
        ct.element_type = dtype
        ct = ct.mark_layout_dynamic(leading_dim=leading_dim)
        return ct

    def _prep(self, q, k, v, o, lse):
        """Reshape/permute torch tensors (B,S,H,D) into the kernel's 5D layout and
        wrap as cute tensors.  Returns the 5 cute tensors in kernel order.

        The kernel expects logical layout (S, D, ((H_r, H_k), B)) with the D axis
        contiguous.  We start from a physically-contiguous (B, S, H_k, H_r, D)
        tensor (H splits as (H_k outer, H_r inner) to match torch SDPA / GQA
        broadcast), then take the strided view permute (1,4,3,2,0) ->
        (S, D, H_r, H_k, B).  Axis 1 (D) is the contiguous leading dim.
        """
        h_k, h_r, d = self.h_k, self.h_r, self.d

        # Runtime shapes (NOT the baked self.s_q/s_k/b): the compiled CUTE kernel
        # was built with mark_layout_dynamic, so ONE compilation handles any
        # seqlen/batch of the same (h, h_k, d, dtype, mask) signature.  Reading
        # runtime shapes here is what lets the plan cache be seqlen-agnostic and
        # avoids a ~2s recompile per new sequence length.
        b = q.shape[0]
        s_q = q.shape[1]
        s_k = k.shape[1]
        # Q / O / LSE carry the full H = H_k * H_r query heads.
        qk = q.reshape(b, s_q, h_k, h_r, d).permute(1, 4, 3, 2, 0)
        ok = o.reshape(b, s_q, h_k, h_r, d).permute(1, 4, 3, 2, 0)
        lk = lse.reshape(b, s_q, h_k, h_r, 1).permute(1, 4, 3, 2, 0)
        # K carries only H_k KV heads (H_r broadcast handled inside the kernel).
        kk = k.reshape(b, s_k, h_k, 1, d).permute(1, 4, 3, 2, 0)

        if self.is_fp8:
            # FP8 PV GEMM needs V K-major (== V^T, S contiguous).  Provide V as a
            # physically (b, d, h_k, 1, s)-contiguous tensor, then permute
            # (4,1,3,2,0) -> (s, d, 1, h_k, b) with leading_dim=0 (S contiguous).
            v_t = v.reshape(b, s_k, h_k, 1, d).permute(0, 4, 2, 3, 1).contiguous()
            vv = v_t.permute(4, 1, 3, 2, 0)
            v_leading = 0
        else:
            vv = v.reshape(b, s_k, h_k, 1, d).permute(1, 4, 3, 2, 0)
            v_leading = 1

        return (
            self._make_cute_tensor(qk, self.in_dtype, 1),
            self._make_cute_tensor(kk, self.in_dtype, 1),
            self._make_cute_tensor(vv, self.in_dtype, v_leading),
            self._make_cute_tensor(ok, self.out_dtype, 1),
            self._make_cute_tensor(lk, cutlass.Float32, 1),
        )

    def __call__(self, q, k, v):
        # Runtime shapes, so one compiled plan serves any seqlen/batch.
        b, s_q, h, d = q.shape[0], q.shape[1], self.h, self.d
        o = torch.empty(b, s_q, h, d, dtype=self.out_torch_dtype, device=q.device)
        lse = torch.empty(b, s_q, h, 1, dtype=torch.float32, device=q.device)

        cq, ck, cv, co, clse = self._prep(q, k, v, o, lse)

        wl = None if self.window_size_left is None else cutlass.Int32(self.window_size_left)
        wr = None if self.window_size_right is None else cutlass.Int32(self.window_size_right)

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

        if self._compiled is None:
            self._compiled = cute.compile(
                self._fmha,
                cq, ck, cv, co, clse,
                cutlass.Float32(self.scale_softmax_log2),
                cutlass.Float32(self.scale_softmax),
                cutlass.Float32(self.scale_output),
                wl, wr, stream,
            )

        self._compiled(
            cq, ck, cv, co, clse,
            cutlass.Float32(self.scale_softmax_log2),
            cutlass.Float32(self.scale_softmax),
            cutlass.Float32(self.scale_output),
            wl, wr, stream,
        )
        return o, lse.squeeze(-1)


@lru_cache(maxsize=128)
def _get_plan(h, h_k, d, in_dtype_key, out_dtype_key, causal, sm_scale,
              kv_aligned=True, tuning_key=None, want_lse=True):
    in_dtype = _TORCH_TO_CUTLASS[in_dtype_key]
    out_dtype = _TORCH_TO_CUTLASS[out_dtype_key]
    tuning = dict(tuning_key) if tuning_key else None
    return _PrefillPlan(
        h, h_k, d, in_dtype, out_dtype, causal, sm_scale,
        kv_aligned=kv_aligned, tuning=tuning, want_lse=want_lse,
    )


def prefill_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
    tuning: Optional[dict] = None,
    out_dtype: Optional[torch.dtype] = None,
):
    """Prefill-optimized FlashAttention forward for Hopper.

    Args:
        q: (B, S_q, H, D)   query,  fp16/bf16/fp8_e4m3, contiguous.
        k: (B, S_k, H_k, D) key,    same dtype as q.  H must be divisible by H_k (GQA/MHA).
        v: (B, S_k, H_k, D) value,  same dtype as q.
        causal: apply causal (lower-triangular, bottom-right unaligned) masking.
        sm_scale: softmax scale; defaults to 1/sqrt(D).
        return_lse: if True also return log-sum-exp (B, S_q, H) in fp32.
        out_dtype: output dtype; defaults to the input dtype for fp16/bf16 and to
            bf16 for fp8 inputs (fp8 accumulation still happens in fp32 internally).

    Returns:
        o: (B, S_q, H, D) attention output.
        (optionally) lse: (B, S_q, H) fp32 log-sum-exp.

    FP8 (torch.float8_e4m3fn) inputs run the QK^T and PV GEMMs at the H20's ~2x
    FP8 Tensor-core rate -- the recommended path to exceed the FP16 compute
    ceiling on this GPU.  Inputs should already be scaled into fp8 range.
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda, "inputs must be CUDA tensors"
    assert q.dtype in _TORCH_TO_CUTLASS, f"unsupported dtype {q.dtype}"
    assert q.dtype == k.dtype == v.dtype, "q/k/v dtypes must match"

    b, s_q, h, d = q.shape
    b_k, s_k, h_k, d_k = k.shape
    assert b == b_k and d == d_k, "batch/head-dim mismatch"
    assert h % h_k == 0, "H must be divisible by H_k (GQA)"
    assert d in (32, 64, 128, 256), f"unsupported head dim {d}"

    is_fp8 = q.dtype in _FP8_DTYPES
    if out_dtype is None:
        out_dtype = torch.bfloat16 if is_fp8 else q.dtype

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    tuning_key = tuple(sorted(tuning.items())) if tuning else None
    # KV-block N tile (fp8 uses 192, else 128); alignment picks mask fast-path.
    _n_tile = 192 if (is_fp8 and d <= 128) else 128
    if tuning and "mma_tiler_mn" in tuning:
        _n_tile = tuning["mma_tiler_mn"][1]
    kv_aligned = (s_k % _n_tile == 0)
    plan = _get_plan(
        h, h_k, d, q.dtype, out_dtype, bool(causal), float(sm_scale),
        kv_aligned, tuning_key, bool(return_lse),
    )
    o, lse = plan(q, k, v)
    if return_lse:
        return o, lse
    return o
