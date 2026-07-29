#!/usr/bin/env python3
"""Team-benchmark entry that runs ONLY the B7 decode adapter (Liu Xiaochen).

Reuses the team bench_attention.py timing/correctness logic unmodified:
  1. imports bench_attention (which builds the shared ops.OPS registry);
  2. clears OPS and registers ONLY dispatch_b7.attention under a unique name, so
     B7 is the sole custom op evaluated (does not join other members' default
     full runs, does not modify ops/__init__.py or bench_attention.py);
  3. forwards the same CLI args and calls bench_attention.main().

Run e.g.:
    python3 -m ops.liuxiaochen_mma_decode.bench_dispatch_b7 \
        --gpu 6 --shapes 1x64x8x131072x128 --dtype bf16 --causal \
        --check-only --phases decode
"""

import os
import sys

# Ensure repo root on sys.path so `import bench_attention` / `import ops` work
# regardless of the module invocation cwd.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

_REG_NAME = "Liu Xiaochen B7 Decode (MMA+cp.async+PDL)"


def main():
    # Import the team benchmark unchanged. Importing it also triggers
    # `from ops import OPS`, which auto-scans root ops modules.
    import bench_attention
    from ops.base import OPS, register

    # Isolate B7: drop every auto-registered op so only B7 is evaluated. This
    # does not touch any file on disk; it only edits the in-process registry.
    OPS.clear()

    from dispatch_b7 import attention as b7_attention
    register(_REG_NAME, b7_attention)

    # Reuse the team benchmark's own main(): it reads sys.argv via parse_args and
    # iterates ops.OPS — no timing/correctness/baseline logic is copied here.
    bench_attention.main()


if __name__ == "__main__":
    main()
