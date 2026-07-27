"""Benchmark registration for Liu Xiaochen's personal SM90 attention kernel.

The actual kernel is the personal derived class
``LiuXiaochenFlashAttentionForwardSm90`` under
``ops/liuxiaochen_sm90_attention/``.  It is driven through the shared CuTe
compile/launch framework, but the compiled/executed kernel is the personal one,
NOT the baseline's ``flash_attn_func`` nor the baseline's SM90 class.

Interface: ``attention(q, k, v, causal=True, sm_scale=None)`` with BHSD in/out.
GQA (q_heads != kv_heads) is handled natively by the kernel — no
repeat_interleave, no contiguous layout copy.
"""

from .base import register


def attention(q, k, v, causal=True, sm_scale=None):
    # Imported lazily so that ops auto-scan does not trigger heavy CuTe imports
    # or baseline path routing until this op is actually invoked.
    #
    # Dispatch on the team-native BHSD layout: q/k/v are
    # (batch, heads, seq_len, head_dim), so seq_len is dim 2.
    q_len = q.shape[2]
    kv_len = k.shape[2]

    if q_len == 1:
        # Decode: a single query attends the full KV cache. The benchmark always
        # passes causal=False in the decode phase (the new token sees all cached
        # KV). Route to the in-repo GQA Split-KV decode kernels.
        #
        # Both are shared-K/V (grid [KV_HEADS, split]): each kv_head's K/V is read
        # from HBM once and shared by its 8 q_heads (no ~8x re-read, no repeat).
        #   B3: vectorized 64-bit GMEM->SMEM load, per-token online softmax.
        #   B4: B3 + wider split (two-level GPU combine) + chunked online softmax
        #       (group=4), which reduces the serial dependency chain and dynamic
        #       instruction count — a clear win at long kv_len.
        # Measured crossover (H20, clean): B4 >= B3 at kv_len >= ~16384, and wins
        # by ~15-18% at 128K; at very short kv_len the two are ~tied. So the
        # default "auto" policy sends long sequences to B4 and short to B3.
        #
        # No silent fallback: the selected impl's exceptions propagate. No baseline
        # fallback, no repeat_interleave, no input contiguous, no CPU/torch combine.
        import os
        import sys

        _decode_dir = os.path.join(
            os.path.dirname(__file__), "liuxiaochen_split_kv_decode"
        )
        if _decode_dir not in sys.path:
            sys.path.insert(0, _decode_dir)

        # Length-adaptive threshold: B4 for kv_len >= this, else B3.
        B4_MIN_KV_LEN = 16384

        def _valid_tile(kv, split, prefer=(32, 16, 8)):
            tps = kv // split
            for t in prefer:
                if tps % t == 0:
                    return t
            return None

        def _run_b3():
            from gqa_decode_shared_b3 import gqa_split_kv_decode_shared_b3
            # B3 uses split in {32,64,128,256}. Pick the largest split that both
            # divides kv_len and yields a tile-divisible tokens/split.
            split = int(os.environ.get("LIUXIAOCHEN_DECODE_SPLITS", "0")) or None
            tile = int(os.environ.get("LIUXIAOCHEN_DECODE_TILE", "0")) or None
            if split is None:
                for s in (256, 128, 64, 32):
                    if kv_len % s == 0 and _valid_tile(kv_len, s) is not None:
                        split = s
                        break
            if tile is None:
                tile = _valid_tile(kv_len, split)
            return gqa_split_kv_decode_shared_b3(
                q, k, v, sm_scale=sm_scale, split_count=split, tokens_per_tile=tile
            )

        def _run_b4():
            from gqa_decode_shared_b4 import gqa_split_kv_decode_shared_b4
            split = int(os.environ.get("LIUXIAOCHEN_DECODE_SPLITS", "0")) or None
            tile = int(os.environ.get("LIUXIAOCHEN_DECODE_TILE", "0")) or None
            group = int(os.environ.get("LIUXIAOCHEN_DECODE_GROUP", "0")) or 4
            if split is None:
                for s in (512, 256, 1024):
                    if kv_len % s == 0 and _valid_tile(kv_len, s) is not None:
                        split = s
                        break
            if tile is None:
                # tile must be divisible by group; prefer 32.
                for t in (32, 16, 8):
                    if (kv_len // split) % t == 0 and t % group == 0:
                        tile = t
                        break
            return gqa_split_kv_decode_shared_b4(
                q, k, v, sm_scale=sm_scale,
                split_count=split, tokens_per_tile=tile, tokens_per_group=group,
            )

        impl = os.environ.get("LIUXIAOCHEN_DECODE_IMPL", "auto").strip().lower()
        if impl == "auto":
            return _run_b4() if kv_len >= B4_MIN_KV_LEN else _run_b3()
        elif impl in ("b4", "shared_b4"):
            return _run_b4()
        elif impl in ("b3", "shared_b3"):
            return _run_b3()
        elif impl in ("independent", "a"):
            from gqa_decode import gqa_split_kv_decode
            return gqa_split_kv_decode(q, k, v, sm_scale=sm_scale)
        elif impl in ("shared", "b"):
            from gqa_decode_shared import gqa_split_kv_decode_shared
            return gqa_split_kv_decode_shared(q, k, v, sm_scale=sm_scale)
        elif impl in ("shared_b2", "b2"):
            from gqa_decode_shared_b2 import gqa_split_kv_decode_shared_b2
            return gqa_split_kv_decode_shared_b2(q, k, v, sm_scale=sm_scale)
        else:
            raise ValueError(
                f"未知 LIUXIAOCHEN_DECODE_IMPL={impl!r}；"
                "可选 auto|b3|b4|independent|a|shared|b|shared_b2|b2"
            )
    elif q_len == kv_len:
        # Prefill: reuse the personal SM90 forward kernel (Exercise 1 + 2).
        from .liuxiaochen_sm90_attention.runner import sm90_attention_bhsd

        return sm90_attention_bhsd(q, k, v, causal=causal, sm_scale=sm_scale)
    else:
        raise ValueError(
            f"unsupported shape: q_len={q_len}, kv_len={kv_len}; only "
            "q_len==1 (decode) or q_len==kv_len (prefill) are supported"
        )


register("Liu Xiaochen SM90 baseline-derived", attention)
