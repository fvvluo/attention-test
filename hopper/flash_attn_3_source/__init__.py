from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("flash_attn_3")
except PackageNotFoundError:  # not installed (e.g. running from a source tree)
    __version__ = "3.0.0"

__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
    "flash_attn_with_kvcache",
    "flash_attn_combine",
    "get_scheduler_metadata",
]


def __getattr__(name):
    # Lazy re-export (PEP 562): avoids a circular import between this package
    # and the top-level `flash_attn_interface` module, which itself imports
    # `flash_attn_3._C` to register the CUDA ops.
    if name in __all__:
        import flash_attn_interface

        return getattr(flash_attn_interface, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
