import math

import torch

import cutlass
import cutlass.cute as cute


THREADS_PER_BLOCK = 256
WARP_SIZE = 32
WARPS_PER_BLOCK = THREADS_PER_BLOCK // WARP_SIZE

ATTENTION_BLOCK_N = 16
ATTENTION_BLOCK_M = 16

@cute.jit
def warp_sum(value):
    for offset in [16, 8, 4, 2, 1]:
        value += cute.arch.shuffle_sync_bfly(value, offset)
    return value

@cute.kernel
def stable_softmax_kernel(x: cute.Tensor, y: cute.Tensor):
    block_x, _, _ = cute.arch.block_idx()
    thread_x, _, _ = cute.arch.thread_idx()

    lane_idx = thread_x % WARP_SIZE
    warp_idx = thread_x // WARP_SIZE
    row = block_x * WARPS_PER_BLOCK + warp_idx

    if row < x.shape[0]:
        local_max = cutlass.Float32(float("-inf"))

        for column in range(lane_idx, x.shape[1], WARP_SIZE):
            value = cutlass.Float32(x[row, column])
            local_max = cute.arch.fmax(local_max, value)

        row_max = cute.arch.warp_redux_sync(local_max, "fmax")

        local_sum = cutlass.Float32(0.0)

        for column in range(lane_idx, x.shape[1], WARP_SIZE):
            value = cutlass.Float32(x[row, column])
            local_sum += cute.exp(
                value - row_max,
                fastmath = False
            )

        row_sum = warp_sum(local_sum)
        inverse_sum = cutlass.Float32(1.0) / row_sum

        for column in range(lane_idx, x.shape[1], WARP_SIZE):
            value = cutlass.Float32(x[row, column])
            probability = cute.exp(value - row_max, fastmath = False) * inverse_sum
            y[row, column] = probability

@cute.jit
def stable_softmax(x, y):
    block_num = (x.shape[0] + WARPS_PER_BLOCK - 1) // WARPS_PER_BLOCK

    stable_softmax_kernel(x, y).launch(
        block = (THREADS_PER_BLOCK, 1, 1),
        grid = (block_num, 1, 1)
    )


@cute.kernel
def attention_scores_kernel(
    q: cute.Tensor,
    k: cute.Tensor,
    scores: cute.Tensor,
    scale: cutlass.Float32
):
    block_x, block_y, _ = cute.arch.block_idx()
    thread_x, thread_y, _ = cute.arch.thread_idx()

    key_row = block_x * ATTENTION_BLOCK_M + thread_x
    query_row = block_y * ATTENTION_BLOCK_N + thread_y

    if query_row < scores.shape[0] and key_row < scores.shape[1]:
        accumulator = cutlass.Float32(0.0)

        for d in range(q.shape[1]):
            accumulator += q[query_row, d] * k[key_row, d]

        scores[query_row, key_row] = accumulator * scale

@cute.jit
def attention_scores(q, k, scores, scale):
    block_num_x = (scores.shape[1] + ATTENTION_BLOCK_M - 1) // ATTENTION_BLOCK_M
    block_num_y = (scores.shape[0] + ATTENTION_BLOCK_N - 1) // ATTENTION_BLOCK_N

    attention_scores_kernel(q, k, scores, scale).launch(
        block = (ATTENTION_BLOCK_M, ATTENTION_BLOCK_N, 1),
        grid = (block_num_x, block_num_y, 1)
    )


@cute.kernel
def attention_value_kernel(
    probabilities: cute.Tensor,
    v: cute.Tensor,
    output: cute.Tensor
):
    block_x, block_y, _ = cute.arch.block_idx()
    thread_x, thread_y, _ = cute.arch.thread_idx()

    value_column = block_x * ATTENTION_BLOCK_M + thread_x
    query_row = block_y * ATTENTION_BLOCK_N + thread_y

    if query_row < output.shape[0] and value_column < output.shape[1]:
        accumulator = cutlass.Float32(0.0)

        for key_row in range(probabilities.shape[1]):
            accumulator += (
                cutlass.Float32(probabilities[query_row, key_row])
                * cutlass.Float32(v[key_row, value_column])
            )

        output[query_row, value_column] = accumulator

@cute.jit
def attention_value(probabilities, v, output):
    block_num_x = (output.shape[1] + ATTENTION_BLOCK_M - 1) // ATTENTION_BLOCK_M
    block_num_y = (output.shape[0] + ATTENTION_BLOCK_N - 1) // ATTENTION_BLOCK_N

    attention_value_kernel(probabilities, v, output).launch(
        block = (ATTENTION_BLOCK_M, ATTENTION_BLOCK_N, 1),
        grid = (block_num_x, block_num_y, 1)
    )


@cute.jit
def attention(q, k, v, scores, probabilities, output, scale):
    attention_scores(q, k, scores, scale)
    stable_softmax(scores, probabilities)
    attention_value(probabilities, v, output)


def torch_attention_reference(q, k, v):
    scale = 1.0 / math.sqrt(q.shape[1])
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    probabilities = torch.softmax(scores, dim = -1)
    output = torch.matmul(probabilities, v.float())
    return scores, probabilities, output
