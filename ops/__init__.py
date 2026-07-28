# 自动扫描并导入本目录下所有算子模块，触发各模块的 register() 注册副作用。
#
# 团队成员新增算子时，只需在本目录下新建一个 .py 文件（除 base.py 外），
# 无需手动修改这个文件 —— 新文件会被自动发现并导入。

import importlib
import pkgutil

from .base import OPS, register  # noqa: F401  (register 供子模块使用)

_package_name = __name__
_package_path = __path__

for _, _module_name, _is_pkg in pkgutil.iter_modules(_package_path):
    if _module_name == "base":
        continue
    if _is_pkg:
        # 子包（如 quanbofeng_final/）是算子的具体实现，不是注册脚本，
        # 由对应的注册模块自行相对导入，这里不重复扫描导入。
        continue
    if _module_name.startswith("_"):
        # 以下划线开头的模块（如 _template.py）视为模板/草稿，不自动导入，
        # 避免团队成员还没写完就被扫描到导致报错。
        continue
    importlib.import_module(f"{_package_name}.{_module_name}")

__all__ = ["OPS", "register"]
