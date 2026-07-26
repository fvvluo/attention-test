# lhx_flash_attention 算子的本地 CuTe DSL 支持包。
#
# 本目录下的模块原样复制自 flash-attention-baseline/flash_attn/cute/ 中
# flash_fwd_sm90.py（SM90 前向内核）所需的全部 flash_attn.cute 依赖，
# 并把包内交叉导入改写为相对导入，从而与 baseline checkout 完全解耦：
# 修改本目录下的文件不会影响 bench_attention.py 用作对照的 baseline。
#
# 外部依赖（环境中已安装的包，不在本目录复制）：cutlass (nvidia-cutlass-dsl)、
# quack、cuda-bindings、torch。
#
# 注意：__init__.py 保持为空（ops/__init__.py 会自动扫描导入本包），
# 具体子模块由 ops/lhx_flash_attention.py 按需导入。
