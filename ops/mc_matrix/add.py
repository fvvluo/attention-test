import cutlass
import cutlass.cute as cute
from cutlass.utils import SmemAllocator
from cutlass.cute.nvgpu import cpasync

THREADS_PER_BLOCK = 256
WARP_SIZE = 32

@cute.kernel
def matrix_add_kernel(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    block_x, block_y, _ = cute.arch.block_idx()
    block_dim_x, block_dim_y, _ =cute.arch.block_dim()
    thread_x, thread_y, _ = cute.arch.thread_idx()

    column = block_x * block_dim_x + thread_x
    row = block_y * block_dim_y + thread_y

    if row < c.shape[0] and column < c.shape[1]:
        c[row, column] = a[row, column] + b[row, column]

@cute.jit
def matrix_add(a, b, c):
    block_dim_x = WARP_SIZE
    block_dim_y = THREADS_PER_BLOCK // WARP_SIZE
    block_num_x = (c.shape[1] + block_dim_x - 1) // block_dim_x
    block_num_y = (c.shape[0] + block_dim_y - 1) // block_dim_y

    matrix_add_kernel(a, b, c).launch(
        block = (block_dim_x, block_dim_y, 1),
        grid = (block_num_x, block_num_y, 1)
    )