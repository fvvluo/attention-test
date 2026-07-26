# liuxiaochen Split-KV Decode Kernels

This directory contains a project-owned CuTe DSL implementation of a
q_len=1 decode Split-KV attention pipeline.  It is a source snapshot for
review and correctness work, not a registered team benchmark operator.

## Files

- `split_kv_stage1.py` — project-owned Stage-1 kernel.
  - One CTA handles one `(head, split)` pair.
  - It scans that split's KV range and writes raw FP32 state.
  - Outputs are local max, local sum, and unnormalized partial numerator:
    - `partial_max: [H, S]`, FP32
    - `partial_sum: [H, S]`, FP32
    - `partial_output: [H, S, D]`, FP32
  - `partial_output` is not divided by `partial_sum` and is not converted to BF16.
- `split_kv_stage2.py` — project-owned GPU Stage-2 reduction kernel.
  - One CTA handles one head.
  - It combines the raw FP32 Stage-1 workspace on GPU using FP32 max,
    weights, denominator, and numerator.
- `split_kv_decode.py` — split mapping helpers, legacy BF16 diagnostic
  scaffold, and tiled PyTorch correctness oracle.
- `verify_split_kv.py` — layered correctness test entry for mapping,
  legacy limitation diagnostics, Stage-1 and Stage-2.

## Correctness Status

The Split-KV correctness checks have passed:

- small cases;
- 32K cases;
- 128K cases;
- GPU Stage-2 reduction correctness;
- complete Stage-1 + GPU Stage-2 GPU pipeline correctness.

These are correctness results only.  No completed latency, speedup,
benchmark, or Nsight conclusion is reported here.

## Fixed Scope

Current implementation is fixed for decode:

- `q_len = 1`
- `B = 1`
- `H = 32`
- `D = 128`
- supported split counts: 1, 2, 4, 8, 16

It does not implement the team final prefill GQA benchmark shape:

```text
attention(q, k, v, causal=True, sm_scale=None)
q:     [B, q_heads, seq_len, D]
k/v:   [B, kv_heads, seq_len, D]
final: [1, 64, 8, 131072, 128]
dtype: bfloat16
```

In particular, this decode kernel is not a generic prefill kernel, does not
support `q_heads=64` with `kv_heads=8` GQA in its current public interface,
does not take the team's BHSD layout, and does not implement a generic
causal multi-query attention path.

## Auto-Scan Boundary

This directory must not be treated as a benchmark operator implementation
for `bench_attention.py`, and should not be auto-scanned or registered as a
team attention backend.  It intentionally contains no callable matching the
team final benchmark interface.

## External Reference Boundary

The kernels in this directory are project-owned implementations.  The
external NVIDIA CUTLASS CuTe DSL examples in the original project's
`cute_attn_128k/reference/` directory are used only for learning, comparison,
and baseline calls.  They are not the code body of
`split_kv_stage1.py` or `split_kv_stage2.py`, and no external source's
copyright, provenance, or license notice should be removed or altered.

The current source snapshot still depends on the original wrapper and
external reference for its baseline comparison path.  See the dependency
audit in the accompanying work report before attempting to run this copy as
a standalone package.

## Current Standalone Status

The four copied Python files are byte-identical to the original project
sources.  They are not yet an independently importable package in this team
worktree because they import `cute_flash_attn`, which is not copied here and
which itself depends on the external Ampere reference file for baseline
comparison.  No fake operator, compatibility shim, or automatic registration
is provided.
