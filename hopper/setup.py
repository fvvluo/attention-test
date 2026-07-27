# Simplified FlashAttention-3 build: forward-only, SM90 (Hopper, e.g. H20),
# head dims 64/128/256, bf16 + fp16, with split-KV and paged-KV support.
# No backward / dropout / FP8 / softcap / sm80 / sm100.

import subprocess
from pathlib import Path

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = Path(__file__).parent.resolve()
repo_dir = this_dir.parent
cutlass_dir = repo_dir / "csrc" / "cutlass"

if not (cutlass_dir / "include" / "cutlass" / "cutlass.h").exists():
    print("cutlass not found, running `git submodule update --init csrc/cutlass`")
    subprocess.run(
        ["git", "submodule", "update", "--init", "csrc/cutlass"], cwd=repo_dir, check=True
    )


def nvidia_cu13_include():
    # Slim CUDA toolkit images miss library headers (cusparse.h, cudss.h, ...)
    # that torch headers require; the pip `nvidia.cu13` package ships them.
    # Appended with low priority (-isystem, after the toolkit's own headers).
    try:
        import importlib.util

        spec = importlib.util.find_spec("nvidia.cu13")
        if spec and spec.submodule_search_locations:
            p = Path(spec.submodule_search_locations[0]) / "include"
            if p.exists():
                return str(p)
    except Exception:
        pass
    return None


fallback_include_args = []
_cu13_inc = nvidia_cu13_include()
if _cu13_inc:
    fallback_include_args = ["-isystem", _cu13_inc]

FEATURE_DEFINES = [
    "-DFLASHATTENTION_DISABLE_BACKWARD",
    "-DFLASHATTENTION_DISABLE_DROPOUT",
    "-DFLASHATTENTION_DISABLE_SM8x",
    "-DFLASHATTENTION_DISABLE_FP8",
    "-DFLASHATTENTION_DISABLE_SOFTCAP",
    "-DFLASHATTENTION_DISABLE_PACKGQA",
    "-DFLASHATTENTION_DISABLE_APPENDKV",
    "-DFLASHATTENTION_DISABLE_HDIM96",
    "-DFLASHATTENTION_DISABLE_HDIM192",
    "-DFLASHATTENTION_DISABLE_HDIMDIFF64",
    "-DFLASHATTENTION_DISABLE_HDIMDIFF192",
]

HEAD_DIMS = [64, 128, 256]
DTYPES = ["bf16", "fp16"]
SUFFIXES = ["", "_split", "_paged", "_paged_split"]  # plain / split-KV / paged / paged+split

sources = ["flash_api.cpp", "flash_fwd_combine.cu", "flash_prepare_scheduler.cu"]
sources += [
    f"instantiations/flash_fwd_hdim{hdim}_{dtype}{suffix}_sm90.cu"
    for hdim in HEAD_DIMS
    for dtype in DTYPES
    for suffix in SUFFIXES
]
for s in sources:
    assert (this_dir / s).exists(), f"missing source file: {s}"

nvcc_flags = [
    "-O3",
    "-std=c++17",
    "--ftemplate-backtrace-limit=0",  # To debug template code
    "--use_fast_math",
    "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",  # Necessary for the WGMMA shapes that we use
    "-DCUTLASS_ENABLE_GDC_FOR_SM90",  # For PDL
    "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
    "-DNDEBUG",  # Important, otherwise performance is severely impacted
    "-gencode",
    "arch=compute_90a,code=sm_90a",
] + FEATURE_DEFINES

setup(
    name="flash_attn_3",
    version="3.0.0",
    description="FlashAttention-3 (simplified: fwd-only, SM90)",
    packages=find_packages(exclude=("build", "dist")),
    py_modules=["flash_attn_interface"],
    ext_modules=[
        CUDAExtension(
            name="flash_attn_3._C",
            sources=[str(this_dir / s) for s in sources],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"] + FEATURE_DEFINES + fallback_include_args,
                "nvcc": nvcc_flags + fallback_include_args,
            },
            include_dirs=[this_dir, cutlass_dir / "include"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
    install_requires=["torch", "ninja"],
)
