"""Development entry for Liu Xiaochen's GQA prefill correctness kernel.

The leading underscore prevents ops/__init__.py from auto-registering this
module with the team benchmark.  This module is intentionally not a benchmark
operator yet.
"""

from .liuxiaochen_cute_prefill.gqa_prefill import (
    compile_gqa_prefill,
    run_gqa_prefill,
)

__all__ = ["compile_gqa_prefill", "run_gqa_prefill"]
