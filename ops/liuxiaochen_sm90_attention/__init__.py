"""Liu Xiaochen personal SM90 attention package.

Derived from the TA teaching baseline flash_fwd_sm90.py.  Kept intentionally
light: importing this package must NOT trigger heavy CuTe/kernel imports or any
op registration.  The actual benchmark registration lives in the top-level
module ops/liuxiaochen_flash_attention.py, which imports from here lazily.
"""
