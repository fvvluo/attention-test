# 自动扫描并导入本目录下所有算子模块，触发各模块的 register() 注册副作用。
#
# 团队成员新增算子时，只需在本目录下新建一个 .py 文件（除 base.py 外），
# 无需手动修改这个文件 —— 新文件会被自动发现并导入。

import importlib
import pkgutil
import sys as _sys
import types as _types

# --- self-healing flash_attn shim ---------------------------------------
# bench_attention.py 会先 `import flash_attn`，随后把 flash_attn.__path__
# 重定向到 vendored flash-attention-baseline，再导入纯 CuTe-DSL 的
# `flash_attn.cute`。本机（经常被重装镜像）没有安装顶层 flash_attn 包，
# 那句裸 import 会抛 ModuleNotFoundError，导致 bench 在加载 baseline 之前
# 就崩溃。这里保证一个最小的顶层 flash_attn 包存在（空 __path__、无编译扩展），
# 使 bench 后续的 __path__ 重定向 + `flash_attn.cute` 导入能够成功。
# 只改动我们自己的 ops 包；baseline 本身完全不动。
if "flash_attn" not in _sys.modules:
    try:
        import flash_attn  # noqa: F401
    except Exception:
        _fa = _types.ModuleType("flash_attn")
        _fa.__path__ = []  # 命名空间式；bench 会覆写它
        _fa.__doc__ = (
            "minimal shim injected by ops/__init__.py so bench can redirect "
            "flash_attn.__path__ to the vendored baseline and load flash_attn.cute"
        )
        _sys.modules["flash_attn"] = _fa
# ------------------------------------------------------------------------

from .base import OPS, register  # noqa: F401  (register 供子模块使用)

_package_name = __name__
_package_path = __path__

for _, _module_name, _is_pkg in pkgutil.iter_modules(_package_path):
    if _module_name == "base":
        continue
    if _module_name.startswith("_"):
        # 以下划线开头的模块（如 _template.py）视为模板/草稿，不自动导入，
        # 避免团队成员还没写完就被扫描到导致报错。
        continue
    importlib.import_module(f"{_package_name}.{_module_name}")

__all__ = ["OPS", "register"]
