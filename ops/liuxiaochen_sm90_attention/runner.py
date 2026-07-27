"""Personal SM90 attention run path.

This module drives the *personal* derived kernel
``LiuXiaochenFlashAttentionForwardSm90`` (defined in
``flash_fwd_sm90_liuxiaochen.py``), NOT the baseline's
``flash_attn.cute.flash_attn_func`` and NOT the baseline's
``FlashAttentionForwardSm90`` class.

We reuse the baseline's private forward driver
``flash_attn.cute.interface._flash_attn_fwd`` purely as a
compile/launch/scheduler framework (this is common infrastructure, explicitly
allowed to be reused).  Before calling it we monkey-patch the
``FlashAttentionForwardSm90`` name in that module's namespace to point at our
derived class, so the kernel that actually gets compiled and executed is the
personal one.

Layout: benchmark hands us BHSD ``(batch, heads, seq_len, head_dim)``.  We pass
BSHD views to the driver via ``.transpose(1, 2)`` (a view, no contiguous copy),
exactly like the baseline wrapper.  GQA (q_heads != kv_heads) is handled
natively by the kernel; we do NOT repeat_interleave K/V.
"""

from pathlib import Path
import importlib


_driver = None  # cached (fn, module) after first setup


def _route_baseline_flash_attn():
    """Route ``flash_attn.cute`` to the in-repo baseline checkout.

    Mirrors ``bench_attention.get_baseline_fn`` so that the common utilities
    imported by our derived kernel resolve to the same baseline checkout the
    benchmark validates against.
    """
    repo_root = Path(__file__).resolve().parents[2]
    baseline_pkg_dir = (repo_root / "flash-attention-baseline" / "flash_attn").resolve()
    if not baseline_pkg_dir.is_dir():
        raise ImportError(f"找不到 flash-attention-baseline: {baseline_pkg_dir}")

    import flash_attn

    baseline_path = str(baseline_pkg_dir)
    flash_attn.__path__ = [
        baseline_path,
        *(p for p in flash_attn.__path__ if p != baseline_path),
    ]
    importlib.invalidate_caches()

    import flash_attn.cute as flash_attn_cute

    loaded_from = Path(flash_attn_cute.__file__).resolve()
    if baseline_pkg_dir not in loaded_from.parents:
        raise ImportError(
            "flash_attn.cute 路由错误: "
            f"期望从 {baseline_pkg_dir} 加载，实际为 {loaded_from}"
        )
    return baseline_pkg_dir


def _setup():
    global _driver
    if _driver is not None:
        return _driver

    _route_baseline_flash_attn()

    # Import the shared compile/launch driver (framework reuse).
    import flash_attn.cute.interface as _bi

    # Import the personal derived kernel class.
    from .flash_fwd_sm90_liuxiaochen import (
        LiuXiaochenFlashAttentionForwardSm90Exercise2,
    )

    # Inject the personal class in place of the baseline's SM90 class so that
    # _flash_attn_fwd instantiates and compiles OUR kernel on the SM90 path.
    # The unique Exercise2 class name gives the JIT/generated symbol a distinct
    # identity, so we can confirm from logs that OUR kernel is compiled/run.
    _bi.FlashAttentionForwardSm90 = LiuXiaochenFlashAttentionForwardSm90Exercise2

    _driver = (_bi._flash_attn_fwd, _bi)
    return _driver


# Private compile cache dedicated to the personal kernel.  CRITICAL: the
# baseline's _flash_attn_fwd.compile_cache is keyed only by dtype/shape/causal/
# tile/... and does NOT include the kernel class identity.  When the benchmark
# runs baseline first (populating that cache) and then our op with the same
# shape, our compile_key collides and we'd silently re-execute the BASELINE's
# compiled kernel instead of ours.  We therefore swap in our own cache dict for
# the duration of each personal call so our injected Exercise2 class is actually
# compiled and launched, then restore the baseline cache.
_PERSONAL_COMPILE_CACHE = {}


def sm90_attention_bhsd(q, k, v, causal=True, sm_scale=None):
    """Run the personal SM90 kernel on BHSD inputs, return BHSD output."""
    if q.shape[-1] != 128 or k.shape[-1] != 128 or v.shape[-1] != 128:
        raise ValueError("个人 SM90 kernel 仅支持 head_dim=128")

    flash_attn_fwd, _bi = _setup()

    # BHSD -> BSHD view (no contiguous, no repeat_interleave).
    q_bshd = q.transpose(1, 2)
    k_bshd = k.transpose(1, 2)
    v_bshd = v.transpose(1, 2)

    # Isolate our compilation from the baseline's shared cache (see note above).
    _saved_cache = getattr(flash_attn_fwd, "compile_cache", None)
    flash_attn_fwd.compile_cache = _PERSONAL_COMPILE_CACHE
    try:
        out = flash_attn_fwd(
            q_bshd,
            k_bshd,
            v_bshd,
            softmax_scale=sm_scale,
            causal=causal,
        )
    finally:
        if _saved_cache is not None:
            flash_attn_fwd.compile_cache = _saved_cache
    if isinstance(out, tuple):
        out = out[0]
    # BSHD -> BHSD view back.
    return out.transpose(1, 2)
