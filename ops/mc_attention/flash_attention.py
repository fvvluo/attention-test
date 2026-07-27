import math

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import SmemAllocator


THREADS_PER_BLOCK = 256
WARP_SIZE = 32
WARPS_PER_BLOCK = THREADS_PER_BLOCK // WARP_SIZE

# 每个 lane 在寄存器里最多持有多少个 head_dim 元素（编译期上限）。
# 支持的最大 head_dim = MAX_ELEMENTS_PER_LANE * WARP_SIZE。
# head_dim 运行时可变，可以不是 WARP_SIZE 的整数倍。
MAX_ELEMENTS_PER_LANE = 8
MAX_HEAD_DIM = MAX_ELEMENTS_PER_LANE * WARP_SIZE

# ---- tiled flash attention 的分块参数 ----
# 一个 block 处理 WARPS_PER_BLOCK 行 query（每个 warp 一行，保留 warp 级并行点积），
# 外层按 TILE_N 分块遍历 key/value，把 K/V 块搬进 shared memory，block 内所有 warp 复用。
# 这样 K/V 只从 global memory 读 ceil(seq_k / TILE_N) 次，而不是 seq_k 次。
TILE_N = 32

@cute.jit
def warp_sum(value):
    for offset in [16, 8, 4, 2, 1]:
        value += cute.arch.shuffle_sync_bfly(value, offset)
    return value


@cute.kernel
def flash_attention_kernel(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    output: cute.Tensor,
    scale: cutlass.Float32
):
    block_x, _, _ = cute.arch.block_idx()
    thread_x, _, _ = cute.arch.thread_idx()

    lane_idx = thread_x % WARP_SIZE
    warp_idx = thread_x // WARP_SIZE
    query_row = block_x * WARPS_PER_BLOCK + warp_idx

    if query_row < output.shape[0]:
        num_keys = k.shape[0]
        head_dim = q.shape[1]

        # 寄存器数组按编译期上限分配，实际有效元素数由运行时 head_dim 决定。
        register_q = cute.make_rmem_tensor(
            (MAX_ELEMENTS_PER_LANE,),
            cutlass.Float32
        )
        accumulator = cute.make_rmem_tensor(
            (MAX_ELEMENTS_PER_LANE,),
            cutlass.Float32
        )

        # 每个 lane 负责列 (lane_idx, lane_idx + 32, lane_idx + 64, ...)；越界列填 0。
        for e in cutlass.range_constexpr(MAX_ELEMENTS_PER_LANE):
            column = lane_idx + e * WARP_SIZE
            if column < head_dim:
                register_q[e] = cutlass.Float32(q[query_row, column])
            else:
                register_q[e] = cutlass.Float32(0.0)
            accumulator[e] = cutlass.Float32(0.0)

        running_max = cutlass.Float32(float("-inf"))
        running_sum = cutlass.Float32(0.0)

        for key_row in range(num_keys):
            partial = cutlass.Float32(0.0)
            for e in cutlass.range_constexpr(MAX_ELEMENTS_PER_LANE):
                column = lane_idx + e * WARP_SIZE
                if column < head_dim:
                    partial += register_q[e] * cutlass.Float32(
                        k[key_row, column]
                    )

            # warp 内规约得到完整点积，即当前 (query_row, key_row) 的注意力分数。
            score = warp_sum(partial) * scale

            # ---- online softmax：running max / running sum 重缩放 ----
            new_max = cute.arch.fmax(running_max, score)
            correction = cute.exp(running_max - new_max, fastmath = False)
            probability = cute.exp(score - new_max, fastmath = False)

            running_sum = running_sum * correction + probability

            for e in cutlass.range_constexpr(MAX_ELEMENTS_PER_LANE):
                column = lane_idx + e * WARP_SIZE
                if column < head_dim:
                    value = cutlass.Float32(v[key_row, column])
                    accumulator[e] = accumulator[e] * correction + probability * value

            running_max = new_max

        inverse_sum = cutlass.Float32(1.0) / running_sum
        for e in cutlass.range_constexpr(MAX_ELEMENTS_PER_LANE):
            column = lane_idx + e * WARP_SIZE
            if column < head_dim:
                output[query_row, column] = accumulator[e] * inverse_sum

@cute.jit
def flash_attention(q, k, v, output, scale):
    block_num = (output.shape[0] + WARPS_PER_BLOCK - 1) // WARPS_PER_BLOCK

    flash_attention_kernel(q, k, v, output, scale).launch(
        block = (THREADS_PER_BLOCK, 1, 1),
        grid = (block_num, 1, 1)
    )


@cute.kernel
def flash_attention_tiled_kernel(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    output: cute.Tensor,
    scale: cutlass.Float32,
    causal_offset: cutlass.Int32
):
    # 张量为 3D (BH, seq, head_dim)，BH = batch * q_heads 已合并到最外层维度。
    # grid 第二维 bh_idx 选中当前 (batch, head) 切片，切片内的注意力数学与 2D 版一致。
    block_x, bh_idx, _ = cute.arch.block_idx()
    thread_x, _, _ = cute.arch.thread_idx()

    lane_idx = thread_x % WARP_SIZE
    warp_idx = thread_x // WARP_SIZE

    num_queries = output.shape[1]
    num_keys = k.shape[1]
    head_dim = q.shape[2]

    # 沿用 basic 的高并行映射：一个 warp 负责一行 query，
    # warp 内 32 lane 沿 head_dim 切分、用 warp_sum 协作算点积。
    # 在此之上叠加 tiling：一个 block 的所有 warp 复用同一份 K/V smem 分块。
    query_row = block_x * WARPS_PER_BLOCK + warp_idx

    # K/V 分块搬进 shared memory，block 内所有 warp 共享。
    # smem 按运行时 head_dim 分配，避免用 MAX_HEAD_DIM 浪费共享内存。
    layout_k = cute.make_layout(
        (TILE_N, head_dim),
        stride = (head_dim, 1)
    )
    layout_v = cute.make_layout(
        (TILE_N, head_dim),
        stride = (head_dim, 1)
    )

    smem = SmemAllocator()
    shared_k = smem.allocate_tensor(cutlass.Float32, layout_k, byte_alignment = 16)
    shared_v = smem.allocate_tensor(cutlass.Float32, layout_v, byte_alignment = 16)

    # 本 warp 的 query 行按 lane 切分留在寄存器：lane 持列 (lane, lane+32, ...)。
    register_q = cute.make_rmem_tensor((MAX_ELEMENTS_PER_LANE,), cutlass.Float32)
    accumulator = cute.make_rmem_tensor((MAX_ELEMENTS_PER_LANE,), cutlass.Float32)

    valid_query = query_row < num_queries

    for e in cutlass.range_constexpr(MAX_ELEMENTS_PER_LANE):
        column = lane_idx + e * WARP_SIZE
        if column < head_dim and valid_query:
            register_q[e] = cutlass.Float32(q[bh_idx, query_row, column])
        else:
            register_q[e] = cutlass.Float32(0.0)
        accumulator[e] = cutlass.Float32(0.0)

    running_max = cutlass.Float32(float("-inf"))
    running_sum = cutlass.Float32(0.0)

    # causal 早退：block 内 8 个 warp 各负责一行 query，最大 query 行是
    # block_x*WARPS_PER_BLOCK + (WARPS_PER_BLOCK-1)。它能看到的最靠后 key 的绝对
    # 位置是 max_query_row + causal_offset，凡是 key_base 超过它的 tile 整块都被
    # 因果屏蔽，无需加载/计算。非因果时 causal_offset >= num_keys，边界自然覆盖全部
    # key，等价于不早退。取 min(边界+1, num_keys) 作为实际要遍历的 key 数上界。
    max_query_row = block_x * WARPS_PER_BLOCK + (WARPS_PER_BLOCK - 1)
    causal_key_limit = max_query_row + causal_offset + 1
    effective_keys = causal_key_limit
    if effective_keys > num_keys:
        effective_keys = num_keys

    num_key_tiles = (effective_keys + TILE_N - 1) // TILE_N

    for tile in range(num_key_tiles):
        key_base = tile * TILE_N

        # 协作加载：block 的全部 THREADS_PER_BLOCK 个线程把当前 K/V 块搬进 smem。
        for index in range(thread_x, TILE_N * head_dim, THREADS_PER_BLOCK):
            local_row = index // head_dim
            local_col = index % head_dim
            key_row = key_base + local_row
            if key_row < num_keys:
                shared_k[local_row, local_col] = cutlass.Float32(k[bh_idx, key_row, local_col])
                shared_v[local_row, local_col] = cutlass.Float32(v[bh_idx, key_row, local_col])
            else:
                shared_k[local_row, local_col] = cutlass.Float32(0.0)
                shared_v[local_row, local_col] = cutlass.Float32(0.0)

        cute.arch.sync_threads()

        if valid_query:
            # 块内遍历 TILE_N 个 key；每个点积仍由 warp 的 32 lane 协作完成。
            for local_row in cutlass.range_constexpr(TILE_N):
                key_row = key_base + local_row
                # 因果掩码：query 行的绝对位置为 query_row + causal_offset（decode 时
                # causal_offset = kv_len - q_len，把短 query 对齐到序列末尾）。
                # 绝对位置更靠后的 key 被屏蔽，直接跳过其 online-softmax 更新。
                # 非因果场景由调用方把 causal_offset 设成 >= num_keys，使全部 key 可见。
                if key_row < num_keys and key_row <= query_row + causal_offset:
                    partial = cutlass.Float32(0.0)
                    for e in cutlass.range_constexpr(MAX_ELEMENTS_PER_LANE):
                        column = lane_idx + e * WARP_SIZE
                        if column < head_dim:
                            partial += register_q[e] * shared_k[local_row, column]

                    score = warp_sum(partial) * scale

                    new_max = cute.arch.fmax(running_max, score)
                    correction = cute.exp(running_max - new_max, fastmath = False)
                    probability = cute.exp(score - new_max, fastmath = False)

                    running_sum = running_sum * correction + probability

                    for e in cutlass.range_constexpr(MAX_ELEMENTS_PER_LANE):
                        column = lane_idx + e * WARP_SIZE
                        if column < head_dim:
                            accumulator[e] = (
                                accumulator[e] * correction
                                + probability * shared_v[local_row, column]
                            )

                    running_max = new_max

        # 下一轮要覆盖写 shared memory，等所有线程读完当前块再继续。
        cute.arch.sync_threads()

    if valid_query:
        inverse_sum = cutlass.Float32(1.0) / running_sum
        for e in cutlass.range_constexpr(MAX_ELEMENTS_PER_LANE):
            column = lane_idx + e * WARP_SIZE
            if column < head_dim:
                output[bh_idx, query_row, column] = accumulator[e] * inverse_sum

@cute.jit
def flash_attention_tiled(q, k, v, output, scale, causal_offset):
    # 张量为 3D (BH, seq, head_dim)：block_num 按 query 序列长度分块，
    # grid 第二维为 BH，一次 launch 覆盖所有 (batch, head) 切片。
    # causal_offset >= num_keys 时等价于非因果（全部 key 可见）。
    block_num = (output.shape[1] + WARPS_PER_BLOCK - 1) // WARPS_PER_BLOCK
    batch_heads = output.shape[0]

    flash_attention_tiled_kernel(q, k, v, output, scale, causal_offset).launch(
        block = (THREADS_PER_BLOCK, 1, 1),
        grid = (block_num, batch_heads, 1)
    )


def torch_attention_reference(q, k, v):
    scale = 1.0 / math.sqrt(q.shape[1])
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    probabilities = torch.softmax(scores, dim = -1)
    output = torch.matmul(probabilities, v.float())
    return output


def benchmark_kernel(kernel, *args, warmup: int = 10, iterations: int = 100) -> float:
    for _ in range(warmup):
        kernel(*args)

    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing = True)
    end_event = torch.cuda.Event(enable_timing = True)

    start_event.record()

    for _ in range(iterations):
        kernel(*args)

    end_event.record()

    end_event.synchronize()

    total_time = start_event.elapsed_time(end_event)
    average_time = total_time / iterations

    return average_time


if __name__ == "__main__":
    seq_len_q = 4096
    seq_len_k = 4096
    head_dim = 128

    assert head_dim <= MAX_HEAD_DIM, (
        f"head_dim={head_dim} 超过上限 {MAX_HEAD_DIM}，请调大 MAX_ELEMENTS_PER_LANE"
    )

    q = torch.randn(seq_len_q, head_dim, device = "cuda", dtype = torch.float32)
    k = torch.randn(seq_len_k, head_dim, device = "cuda", dtype = torch.float32)
    v = torch.randn(seq_len_k, head_dim, device = "cuda", dtype = torch.float32)
    output = torch.empty(seq_len_q, head_dim, device = "cuda", dtype = torch.float32)

    scale = 1.0 / math.sqrt(head_dim)

    # head_dim 可变时行首不一定 16B 对齐，用 4B（fp32 元素）对齐更稳妥。
    q_tensor = from_dlpack(q, assumed_align = 4)
    k_tensor = from_dlpack(k, assumed_align = 4)
    v_tensor = from_dlpack(v, assumed_align = 4)
    output_tensor = from_dlpack(output, assumed_align = 4)

    # tiled 版现在按 3D (BH, seq, head_dim) 组织，单头测试即 BH=1。
    q3 = q.unsqueeze(0)
    k3 = k.unsqueeze(0)
    v3 = v.unsqueeze(0)
    output3 = output.unsqueeze(0)
    q_tensor3 = from_dlpack(q3, assumed_align = 4)
    k_tensor3 = from_dlpack(k3, assumed_align = 4)
    v_tensor3 = from_dlpack(v3, assumed_align = 4)
    output_tensor3 = from_dlpack(output3, assumed_align = 4)

    reference = torch_attention_reference(q, k, v)

    # ---- 基础版（warp-per-row）----
    compiled = cute.compile(
        flash_attention,
        q_tensor, k_tensor, v_tensor, output_tensor,
        cutlass.Float32(scale)
    )
    compiled(q_tensor, k_tensor, v_tensor, output_tensor, cutlass.Float32(scale))
    torch.cuda.synchronize()
    basic_error = (output - reference).abs().max().item()

    # ---- tiled 版（shared memory 复用 K/V）----
    output.zero_()
    # __main__ 里做非因果测试：causal_offset 取 >= seq_len_k，令全部 key 可见。
    tiled_causal_offset = cutlass.Int32(seq_len_k)
    compiled_tiled = cute.compile(
        flash_attention_tiled,
        q_tensor3, k_tensor3, v_tensor3, output_tensor3,
        cutlass.Float32(scale), tiled_causal_offset
    )
    compiled_tiled(
        q_tensor3, k_tensor3, v_tensor3, output_tensor3,
        cutlass.Float32(scale), tiled_causal_offset
    )
    torch.cuda.synchronize()
    tiled_error = (output - reference).abs().max().item()

    print(f"basic max abs error: {basic_error:.6e}")
    print(f"tiled max abs error: {tiled_error:.6e}")

    basic_time = benchmark_kernel(
        compiled,
        q_tensor, k_tensor, v_tensor, output_tensor, cutlass.Float32(scale)
    )
    tiled_time = benchmark_kernel(
        compiled_tiled,
        q_tensor3, k_tensor3, v_tensor3, output_tensor3,
        cutlass.Float32(scale), tiled_causal_offset
    )
    torch_time = benchmark_kernel(
        torch_attention_reference,
        q, k, v
    )

    print(f"shape: q={seq_len_q}x{head_dim}, k=v={seq_len_k}x{head_dim}")
    print(f"basic (warp-per-row): {basic_time:.4f} ms")
    print(f"tiled (smem reuse):   {tiled_time:.4f} ms")
    print(f"torch reference:      {torch_time:.4f} ms")
    print(f"tiled speedup vs basic: {basic_time / tiled_time:.2f}x")
    print(f"tiled speedup vs torch: {torch_time / tiled_time:.2f}x")