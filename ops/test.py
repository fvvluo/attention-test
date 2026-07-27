import argparse

import torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from mc_matrix import matrix_mul

from mc_benchmark import benchmark_kernel

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type = int, default = 4096)
    parser.add_argument("--m", type = int, default = 4096)
    parser.add_argument("--k", type = int, default = 4096)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Requires Linux and an NVDIA CUDA GPU.")
    if args.n <= 0:
        raise ValueError("--n must be positive")
    if args.m <= 0:
        raise ValueError("--m must be positive")
    if args.k <= 0:
        raise ValueError("--k must be positive")
    dtype = torch.float32
    device = torch.device("cuda")

    a = torch.rand((args.n, args.k), dtype = dtype, device = device)
    b = torch.rand((args.k, args.m), dtype = dtype, device = device)
    result = torch.zeros((args.n, args.m), dtype = dtype, device = device)

    assert a.is_contiguous()
    assert b.is_contiguous()
    assert result.is_contiguous()

    assert a.data_ptr() % 16 == 0
    assert b.data_ptr() % 16 == 0
    assert result.data_ptr() % 16 == 0

    a_cute = from_dlpack(a, assumed_align = 16)
    b_cute = from_dlpack(b, assumed_align = 16)
    result_cute = from_dlpack(result, assumed_align = 16)

    compiled_matrix_mul = cute.compile(
        matrix_mul,
        a_cute, b_cute, result_cute,
        options = "--generate-line-info"
    )
    compiled_matrix_mul(a_cute, b_cute, result_cute)

    expected = a @ b

    max_error = (result - expected).abs().max().item()
    print(f"max absolute error: {max_error:.3e}")

    if max_error > 1e-5:
        raise ValueError("The answer may be wrong.")

    average_time = benchmark_kernel(compiled_matrix_mul, a_cute, b_cute, result_cute)
    print(f"average running time: {average_time:.3f}ms")

    average_time = benchmark_kernel(lambda a, b: a @ b, a, b)
    print(f"standard average running time: {average_time:.3f}ms")

if __name__ == "__main__":
    main()