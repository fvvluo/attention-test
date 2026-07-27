# Liu Xiaochen CuTe GQA Prefill

This package contains a stride-aware, zero-input-copy GQA prefill
correctness implementation for the team FlashAttention benchmark.

## Source Boundary

`ampere_flash_attention_gqa.py` is derived from the external NVIDIA Ampere
CuTe DSL FlashAttention v2 reference implementation.  Its original copyright,
license, and provenance are preserved.  Project-specific changes are limited
to:

- independent Q and KV head counts for GQA;
- `q_head -> kv_head` mapping in kernel indexing;
- stride-aware logical BSHD views of contiguous BHSD inputs;
- direct contiguous BHSD output;
- a causal-capable prefill path used by this wrapper.

The derived file is not claimed as the original external reference, and the
external reference is not claimed as a project-owned kernel.

## Current Scope

This phase targets correctness and team interface adaptation, not final
performance.

Current limits:

- `B = 1`
- `q_len == kv_len`
- `D = 128`
- `dtype = torch.bfloat16`
- `q_heads % kv_heads == 0`
- `causal=True/False`
- `sm_scale=None` means `D ** -0.5`

## Layout and GQA

Team tensors use contiguous BHSD:

```text
Q:     [B, Hq, S, D]
K/V:   [B, Hkv, S, D]
Output: [B, Hq, S, D]
```

The wrapper creates logical BSHD CuTe tensors using only stride-aware views:

```text
Q/O logical BSHD stride: (Hq*S*D, D, S*D, 1)
K/V logical BSHD stride: (Hkv*S*D, D, S*D, 1)
```

No `transpose(...).contiguous()`, `permute(...).contiguous()`, full input
copy, or `repeat_interleave` is used.

Each CTA handles one Q head and one query tile.  The kernel maps:

```text
group_size = q_heads / kv_heads
kv_head = q_head / group_size
```

This removes full K/V replication, but shared K/V may still be read
independently by multiple Q-head CTAs.  Cross-Q-head K/V tile reuse is a
future performance optimization and is not claimed here.

## Verification

The correctness suites are intentionally separate from the benchmark
registration path.  The underscore development entry
`ops/_liuxiaochen_cute_prefill_dev.py` is not auto-registered by
`ops/__init__.py`.

No latency, speedup, benchmark, or Nsight result is reported in this phase.
