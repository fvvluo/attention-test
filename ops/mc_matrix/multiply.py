import cutlass
import cutlass.cute as cute
from cutlass.utils import SmemAllocator
from cutlass.cute.nvgpu import cpasync


THREADS_PER_BLOCK = 256
WARP_SIZE = 32

BLOCK_N = 64
BLOCK_M = 64
BLOCK_K = 16

THREAD_N = 16
THREAD_M = 16

THREAD_TILE_N = 4
THREAD_TILE_M = 4

STAGES = 2

@cute.jit
def make_fp32_tiled_copy(major_size: cutlass.Constexpr, use_async: cutlass.Constexpr = False):
    copy_op = cpasync.CopyG2SOp() if use_async else cute.nvgpu.CopyUniversalOp()
    copy_atom = cute.make_copy_atom(
        copy_op,
        cutlass.Float32,
        num_bits_per_copy = 128
    )

    copy_elements = 4
    threads_per_row = major_size // copy_elements

    thread_layout = cute.make_ordered_layout(
        (THREADS_PER_BLOCK // threads_per_row, threads_per_row),
        order = (1, 0)
    )
    value_layout = cute.make_layout((1, copy_elements))

    tiled_copy = cute.make_tiled_copy_tv(
        copy_atom,
        thread_layout,
        value_layout
    )
    return copy_atom, tiled_copy

@cute.jit
def load_tile(
    a, b, shared_a, shared_b,
    copy_atom_a, tiled_copy_a,
    copy_atom_b, tiled_copy_b,
    block_x, block_y, thread_idx,
    tile_k, stage
):
    global_a_tile = cute.local_tile(a, (BLOCK_N, BLOCK_K), (block_y, tile_k))
    global_b_tile = cute.local_tile(b, (BLOCK_K, BLOCK_M), (tile_k, block_x))

    smem_a = shared_a[stage, None, :BLOCK_K]
    smem_b = shared_b[stage, None, :BLOCK_M]

    thread_copy_a = tiled_copy_a.get_slice(thread_idx)
    cute.copy(
        copy_atom_a,
        thread_copy_a.partition_S(global_a_tile),
        thread_copy_a.partition_D(smem_a)
    )

    thread_copy_b = tiled_copy_b.get_slice(thread_idx)
    cute.copy(
        copy_atom_b,
        thread_copy_b.partition_S(global_b_tile),
        thread_copy_b.partition_D(smem_b)
    )

    cute.arch.cp_async_commit_group()

@cute.kernel
def matrix_mul_kernel(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    block_x, block_y, _ = cute.arch.block_idx()
    thread_x, thread_y, _ = cute.arch.thread_idx()

    column = block_x * BLOCK_M + thread_x * THREAD_TILE_M
    row = block_y * BLOCK_N + thread_y * THREAD_TILE_N

    layout_a = cute.make_layout(
        (STAGES, BLOCK_N, BLOCK_K),
        stride = (BLOCK_N * BLOCK_K, BLOCK_K, 1)
    )
    layout_b = cute.make_layout(
        (STAGES, BLOCK_K, BLOCK_M),
        stride = (BLOCK_K * BLOCK_M, BLOCK_M, 1)
    )

    smem = SmemAllocator()

    shared_a = smem.allocate_tensor(
        cutlass.Float32,
        layout_a,
        byte_alignment = 16
    )
    shared_b = smem.allocate_tensor(
        cutlass.Float32,
        layout_b,
        byte_alignment = 16
    )

    accumulator_layout = cute.make_layout(
        (THREAD_TILE_N, THREAD_TILE_M),
        stride = (THREAD_TILE_M, 1)
    )
    accumulator = cute.make_rmem_tensor(
        accumulator_layout,
        cutlass.Float32
    )

    for i in cutlass.range_constexpr(THREAD_TILE_N):
        for j in cutlass.range_constexpr(THREAD_TILE_M):
            accumulator[i, j] = cutlass.Float32(0.0)

    num_tiles_k = (a.shape[1] + BLOCK_K - 1) // BLOCK_K

    copy_atom_a, tiled_copy_a = make_fp32_tiled_copy(BLOCK_K, use_async = True)
    copy_atom_b, tiled_copy_b = make_fp32_tiled_copy(BLOCK_M, use_async = True)

    register_a = cute.make_rmem_tensor(
        (THREAD_TILE_N,),
        cutlass.Float32
    )
    register_b = cute.make_rmem_tensor(
        (THREAD_TILE_M,),
        cutlass.Float32
    )

    thread_idx = thread_x + thread_y * THREAD_M

    for tile in cutlass.range_constexpr(STAGES - 1):
        if tile < num_tiles_k:
            load_tile(
                a, b, shared_a, shared_b,
                copy_atom_a, tiled_copy_a,
                copy_atom_b, tiled_copy_b,
                block_x, block_y, thread_idx,
                tile, tile
            )

    for tile_k in range(num_tiles_k):
        read_stage = tile_k % STAGES
        next_tile = tile_k + STAGES - 1

        if next_tile < num_tiles_k:
            cute.arch.cp_async_wait_group(STAGES - 2)
        else:
            cute.arch.cp_async_wait_group(0)

        cute.arch.sync_threads()

        if next_tile < num_tiles_k:
            write_stage = next_tile % STAGES
            load_tile(
                a, b, shared_a, shared_b,
                copy_atom_a, tiled_copy_a,
                copy_atom_b, tiled_copy_b,
                block_x, block_y, thread_idx,
                next_tile, write_stage
            )

        smem_a = shared_a[read_stage, None, None]
        smem_b = shared_b[read_stage, None, None]

        for k in cutlass.range_constexpr(BLOCK_K):
            for i in cutlass.range_constexpr(THREAD_TILE_N):
                register_a[i] = smem_a[thread_y * THREAD_TILE_N + i, k]
            for j in cutlass.range_constexpr(THREAD_TILE_M):
                register_b[j] = smem_b[k, thread_x * THREAD_TILE_M + j]
            for i in cutlass.range_constexpr(THREAD_TILE_N):
                for j in cutlass.range_constexpr(THREAD_TILE_M):
                    accumulator[i, j] += register_a[i] * register_b[j]

        cute.arch.sync_threads()

    copy_atom_c = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        cutlass.Float32,
        num_bits_per_copy = 128
    )
    global_c_tile = cute.local_tile(
        c,
        (BLOCK_N, BLOCK_M),
        (block_y, block_x)
    )
    for i in cutlass.range_constexpr(THREAD_TILE_N):
        thread_c_tile = cute.local_tile(
            global_c_tile,
            (1, THREAD_TILE_M),
            (thread_y * THREAD_TILE_N + i, thread_x)
        )

        register_row = accumulator[i, None]
        global_row = thread_c_tile[0, None]

        cute.copy(
            copy_atom_c,
            register_row,
            global_row
        )

@cute.jit
def matrix_mul(a, b, c):
    block_num_x = (c.shape[1] + BLOCK_M - 1) // BLOCK_M
    block_num_y = (c.shape[0] + BLOCK_N - 1) // BLOCK_N

    matrix_mul_kernel(a, b, c).launch(
        block = (THREAD_M, THREAD_N, 1),
        grid = (block_num_x, block_num_y, 1)
    )