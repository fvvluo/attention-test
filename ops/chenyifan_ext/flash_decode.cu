#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {

constexpr int kHeadDim = 128;
constexpr int kWarps = 1;
constexpr int kThreads = kWarps * 32;
constexpr int kTokensPerTile = 32;
constexpr int kKStride = kHeadDim + 16;
constexpr int kVStride = kHeadDim + 16;
constexpr int kQStride = kHeadDim + 8;
constexpr int kPStride = kTokensPerTile + 8;
constexpr int kMaxSplits = 64;
constexpr int kMaxGroup = 8;   // max q-heads per kv-head (decode M dim), decoupled from kWarps

__device__ __forceinline__ float warp_sum(float x) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_down_sync(0xffffffff, x, offset);
    }
    return __shfl_sync(0xffffffff, x, 0);
}

__device__ __forceinline__ float warp_max(float x) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_down_sync(0xffffffff, x, offset));
    }
    return __shfl_sync(0xffffffff, x, 0);
}

__device__ __forceinline__ void cp_async_16(void* dst, const void* src) {
    const unsigned dst_shared = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst_shared), "l"(src));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group 0;\n" ::);
}

// D[16x8] += A[16x16] * B[16x8]^T ; A row-major, B col-major, bf16 in, f32 out.
// Thread->fragment mapping per PTX ISA mma.m16n8k16.
__device__ __forceinline__ void mma_m16n8k16(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
}

// ldmatrix.x4: load a 16x16 bf16 matrix tile from shared into 4 regs/thread
// (row-major source), producing the A-operand fragment layout for m16n8k16.
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* smem_ptr) {
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], const void* smem_ptr) {
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(addr));
}

// ldmatrix.x2: load a 16x8 bf16 tile (B-operand fragment for m16n8k16).
__device__ __forceinline__ void ldmatrix_x2(uint32_t (&r)[2], const void* smem_ptr) {
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1])
        : "r"(addr));
}

// Transposed load for row-major V[K,N] into the col-major mma B fragment.
__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t (&r)[2], const void* smem_ptr) {
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1])
        : "r"(addr));
}

__global__ void decode_split_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    float* __restrict__ partial_o,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    int batch,
    int q_heads,
    int kv_heads,
    int kv_len,
    int splits,
    float scale) {
    __shared__ __align__(16) __nv_bfloat16 s_q[16][kQStride];  // 16 mma-M rows + bank padding
    __shared__ __align__(16) __nv_bfloat16 s_k[2][kTokensPerTile][kKStride];
    __shared__ __align__(16) __nv_bfloat16 s_v[2][kTokensPerTile][kVStride];
    __shared__ __align__(16) __nv_bfloat16 s_pbf[16][kPStride];  // P (bf16), padded for PV ldmatrix
    __shared__ float s_row_max[kMaxGroup];
    __shared__ float s_row_sum[kMaxGroup];
    __shared__ float s_alpha[kMaxGroup];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int split = blockIdx.x;
    const int kv_head = blockIdx.y;
    const int b = blockIdx.z;
    const int group = q_heads / kv_heads;
    const bool active = warp < group;
    const int q_head = kv_head * group + warp;

    const int tiles = (kv_len + kTokensPerTile - 1) / kTokensPerTile;
    const int tile_begin = static_cast<int>((static_cast<long long>(tiles) * split) / splits);
    const int tile_end = static_cast<int>((static_cast<long long>(tiles) * (split + 1)) / splits);

    // warp0 owns online softmax stats (per head) and the O accumulator.
    if (warp == 0 && lane < kMaxGroup) {
        s_row_max[lane] = -INFINITY;
        s_row_sum[lane] = 0.0f;
    }
    // O^T accumulator: 8 blocks of [16 d x 8 head], persistent across KV tiles.
    float acc_o[kHeadDim / 16][4];
    #pragma unroll
    for (int db = 0; db < kHeadDim / 16; ++db) {
        #pragma unroll
        for (int r = 0; r < 4; ++r) acc_o[db][r] = 0.0f;
    }

    const long long q_group_offset =
        (static_cast<long long>(b) * q_heads + kv_head * group) * kHeadDim;
    constexpr int kQVecsPerRow = kHeadDim / 8;
    constexpr int kQVectors = kMaxGroup * kQVecsPerRow;  // rows 0..7: real q heads
    constexpr int kQVectorsPad = 16 * kQVecsPerRow;      // rows 8..15: mma zero pad
    for (int vec = tid; vec < kQVectorsPad; vec += kThreads) {
        const int row = vec / kQVecsPerRow;
        const int vec_in_row = vec % kQVecsPerRow;
        reinterpret_cast<uint4*>(&s_q[row][vec_in_row * 8])[0] =
            vec < kQVectors
                ? reinterpret_cast<const uint4*>(q + q_group_offset)[vec]
                : make_uint4(0, 0, 0, 0);
    }

    const long long kv_head_offset =
        (static_cast<long long>(b) * kv_heads + kv_head) * kv_len * kHeadDim;

    constexpr int kVectorsPerTensor = kTokensPerTile * kHeadDim / 8;
    if (tile_begin < tile_end) {
        const long long first_offset =
            kv_head_offset + static_cast<long long>(tile_begin * kTokensPerTile) * kHeadDim;
        for (int vec = tid; vec < kVectorsPerTensor; vec += kThreads) {
            const int token = vec / (kHeadDim / 8);
            const int vec_in_row = vec % (kHeadDim / 8);
            cp_async_16(
                &s_k[0][token][vec_in_row * 8],
                reinterpret_cast<const uint4*>(k + first_offset) + vec);
            cp_async_16(
                &s_v[0][token][vec_in_row * 8],
                reinterpret_cast<const uint4*>(v + first_offset) + vec);
        }
        cp_async_commit();
        cp_async_wait();
    }
    __syncthreads();

    for (int tile = tile_begin; tile < tile_end; ++tile) {
        const int stage = (tile - tile_begin) & 1;
        const int next_tile = tile + 1;
        const int token_begin = tile * kTokensPerTile;
        const int valid_tokens = min(kTokensPerTile, kv_len - token_begin);

        if (next_tile < tile_end) {
            const int next_stage = stage ^ 1;
            const long long next_offset =
                kv_head_offset + static_cast<long long>(next_tile * kTokensPerTile) * kHeadDim;
            for (int vec = tid; vec < kVectorsPerTensor; vec += kThreads) {
                const int token = vec / (kHeadDim / 8);
                const int vec_in_row = vec % (kHeadDim / 8);
                cp_async_16(
                    &s_k[next_stage][token][vec_in_row * 8],
                    reinterpret_cast<const uint4*>(k + next_offset) + vec);
                cp_async_16(
                    &s_v[next_stage][token][vec_in_row * 8],
                    reinterpret_cast<const uint4*>(v + next_offset) + vec);
            }
            cp_async_commit();
        }

        // ---- QK with tokens in mma-M and heads in mma-N: no padded M rows ----
        // K[16 token,128] * Q^T[128,8 head] uses every lane of m16n8k16.
        // Two M blocks cover 32 tokens, halving QK mma count from 32 to 16.
        if (warp == 0) {
            float acc[2][4];
            #pragma unroll
            for (int tb = 0; tb < 2; ++tb) {
                #pragma unroll
                for (int r = 0; r < 4; ++r) acc[tb][r] = 0.0f;
            }
            #pragma unroll
            for (int kblk = 0; kblk < kHeadDim / 16; ++kblk) {
                const int d0 = kblk * 16;
                uint32_t q_frag[2];
                // Q[head,dim] is the col-major storage of logical Q^T[dim,head].
                ldmatrix_x2(q_frag, &s_q[lane % 8][d0 + (lane / 8) * 8]);
                #pragma unroll
                for (int tb = 0; tb < 2; ++tb) {
                    uint32_t k_frag[4];
                    // A is a row-major 16-token x 16-dim K tile.
                    ldmatrix_x4(
                        k_frag,
                        &s_k[stage][tb * 16 + (lane % 16)][d0 + (lane / 16) * 8]);
                    mma_m16n8k16(acc[tb], k_frag, q_frag);
                }
            }

            // D mapping: lane%4 selects a pair of head columns; lane/4 selects
            // token rows {r,r+8}. Reduce each head across lanes spaced by 4.
            const int token_row = lane >> 2;
            const int head0 = 2 * (lane & 3);
            const int head1 = head0 + 1;
            float max0 = -INFINITY;
            float max1 = -INFINITY;
            #pragma unroll
            for (int tb = 0; tb < 2; ++tb) {
                acc[tb][0] = (tb * 16 + token_row) < valid_tokens
                    ? acc[tb][0] * scale : -INFINITY;
                acc[tb][1] = (tb * 16 + token_row) < valid_tokens
                    ? acc[tb][1] * scale : -INFINITY;
                acc[tb][2] = (tb * 16 + token_row + 8) < valid_tokens
                    ? acc[tb][2] * scale : -INFINITY;
                acc[tb][3] = (tb * 16 + token_row + 8) < valid_tokens
                    ? acc[tb][3] * scale : -INFINITY;
                max0 = fmaxf(max0, fmaxf(acc[tb][0], acc[tb][2]));
                max1 = fmaxf(max1, fmaxf(acc[tb][1], acc[tb][3]));
            }
            #pragma unroll
            for (int offset = 4; offset <= 16; offset <<= 1) {
                max0 = fmaxf(max0, __shfl_xor_sync(0xffffffff, max0, offset));
                max1 = fmaxf(max1, __shfl_xor_sync(0xffffffff, max1, offset));
            }
            const float prev_max0 = s_row_max[head0];
            const float prev_max1 = s_row_max[head1];
            const float prev_sum0 = s_row_sum[head0];
            const float prev_sum1 = s_row_sum[head1];
            const float new_max0 = fmaxf(prev_max0, max0);
            const float new_max1 = fmaxf(prev_max1, max1);
            const float alpha0 = prev_max0 == -INFINITY ? 0.0f : __expf(prev_max0 - new_max0);
            const float alpha1 = prev_max1 == -INFINITY ? 0.0f : __expf(prev_max1 - new_max1);

            float sum0 = 0.0f;
            float sum1 = 0.0f;
            #pragma unroll
            for (int tb = 0; tb < 2; ++tb) {
                const int t0 = tb * 16 + token_row;
                const int t8 = t0 + 8;
                const float p00 = __expf(acc[tb][0] - new_max0);
                const float p01 = __expf(acc[tb][1] - new_max1);
                const float p08 = __expf(acc[tb][2] - new_max0);
                const float p09 = __expf(acc[tb][3] - new_max1);
                sum0 += p00 + p08;
                sum1 += p01 + p09;
                s_pbf[head0][t0] = __float2bfloat16(p00);
                s_pbf[head1][t0] = __float2bfloat16(p01);
                s_pbf[head0][t8] = __float2bfloat16(p08);
                s_pbf[head1][t8] = __float2bfloat16(p09);
                s_pbf[head0 + 8][t0] = __float2bfloat16(0.0f);
                s_pbf[head1 + 8][t0] = __float2bfloat16(0.0f);
                s_pbf[head0 + 8][t8] = __float2bfloat16(0.0f);
                s_pbf[head1 + 8][t8] = __float2bfloat16(0.0f);
            }
            #pragma unroll
            for (int offset = 4; offset <= 16; offset <<= 1) {
                sum0 += __shfl_xor_sync(0xffffffff, sum0, offset);
                sum1 += __shfl_xor_sync(0xffffffff, sum1, offset);
            }
            if (token_row == 0) {
                s_row_max[head0] = new_max0;
                s_row_max[head1] = new_max1;
                s_row_sum[head0] = prev_sum0 * alpha0 + sum0;
                s_row_sum[head1] = prev_sum1 * alpha1 + sum1;
                s_alpha[head0] = alpha0;
                s_alpha[head1] = alpha1;
            }
            __syncwarp();

            // O^T mma D layout: each lane owns two head columns for d rows r/r+8.
            const float out_alpha0 = s_alpha[2 * (lane & 3)];
            const float out_alpha1 = s_alpha[2 * (lane & 3) + 1];
            #pragma unroll
            for (int db = 0; db < kHeadDim / 16; ++db) {
                acc_o[db][0] *= out_alpha0;
                acc_o[db][1] *= out_alpha1;
                acc_o[db][2] *= out_alpha0;
                acc_o[db][3] *= out_alpha1;
            }

            // PV as O^T = V^T[128 d,32 token] * P^T[32 token,8 head].
            // This fills all m16n8 lanes and halves PV mma count from 32 to 16.
            #pragma unroll
            for (int kblk = 0; kblk < kTokensPerTile / 16; ++kblk) {
                const int t0 = kblk * 16;
                uint32_t p_frag[2];
                // P[head,token] is col-major storage of P^T[token,head].
                ldmatrix_x2(
                    p_frag,
                    &s_pbf[lane % 8][t0 + (((lane / 8) & 1) * 8)]);
                #pragma unroll
                for (int db = 0; db < kHeadDim / 16; ++db) {
                    const int d0 = db * 16;
                    uint32_t v_frag[4];
                    // Transpose the row-major V[token,d] 16x16 tile into A[d,token].
                    const int matrix = lane / 8;
                    const int src_token = t0 + (lane % 8) + (matrix / 2) * 8;
                    const int src_d = d0 + (matrix & 1) * 8;
                    ldmatrix_x4_trans(v_frag, &s_v[stage][src_token][src_d]);
                    mma_m16n8k16(acc_o[db], v_frag, p_frag);
                }
            }
        }

        if (next_tile < tile_end) {
            cp_async_wait();
            __syncthreads();
        } else {
            __syncthreads();
        }
    }

    // ---- transpose O^T mma fragments back to partial_o[head,d] ----
    if (warp == 0) {
        const int drow = lane >> 2;
        const int head0 = 2 * (lane & 3);
        const int head1 = head0 + 1;
        const int q_head0 = kv_head * group + head0;
        const int q_head1 = kv_head * group + head1;
        const long long stat0 =
            (static_cast<long long>(b) * q_heads + q_head0) * splits + split;
        const long long stat1 =
            (static_cast<long long>(b) * q_heads + q_head1) * splits + split;
        #pragma unroll
        for (int db = 0; db < kHeadDim / 16; ++db) {
            const int d0 = db * 16 + drow;
            partial_o[stat0 * kHeadDim + d0] = acc_o[db][0];
            partial_o[stat1 * kHeadDim + d0] = acc_o[db][1];
            partial_o[stat0 * kHeadDim + d0 + 8] = acc_o[db][2];
            partial_o[stat1 * kHeadDim + d0 + 8] = acc_o[db][3];
        }
        if (drow == 0) {
            partial_m[stat0] = s_row_max[head0];
            partial_l[stat0] = s_row_sum[head0];
            partial_m[stat1] = s_row_max[head1];
            partial_l[stat1] = s_row_sum[head1];
        }
    }
}

__global__ void decode_combine_kernel(
    const float* __restrict__ partial_o,
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    __nv_bfloat16* __restrict__ output,
    int q_heads,
    int splits) {
    const int q_head = blockIdx.x;
    const int b = blockIdx.y;
    const int d = threadIdx.x;
    const long long stat_base =
        (static_cast<long long>(b) * q_heads + q_head) * splits;
    __shared__ float weights[kMaxSplits];
    __shared__ float denominator;

    if (d == 0) {
        float global_max = -INFINITY;
        for (int split = 0; split < splits; ++split) {
            global_max = fmaxf(global_max, partial_m[stat_base + split]);
        }
        float denom = 0.0f;
        for (int split = 0; split < splits; ++split) {
            const float weight = __expf(partial_m[stat_base + split] - global_max);
            weights[split] = weight;
            denom += partial_l[stat_base + split] * weight;
        }
        denominator = denom;
    }
    __syncthreads();

    float numerator = 0.0f;
    for (int split = 0; split < splits; ++split) {
        numerator += partial_o[(stat_base + split) * kHeadDim + d] * weights[split];
    }

    const long long out_idx =
        (static_cast<long long>(b) * q_heads + q_head) * kHeadDim + d;
    output[out_idx] = __float2bfloat16(numerator / denominator);
}

int choose_splits(int batch, int kv_heads, int kv_len, int num_sms) {
    const int max_useful_splits = min(kMaxSplits, kv_len / kTokensPerTile);
    const int target_blocks = 4 * num_sms;
    const int splits_for_occupancy =
        (target_blocks + batch * kv_heads - 1) / (batch * kv_heads);
    return max(1, min(max_useful_splits, splits_for_occupancy));
}

}  // namespace

torch::Tensor decode_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    double sm_scale = 0.08838834764831845) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q/k/v must be CUDA tensors");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(), "q/k/v must be on the same device");
    c10::cuda::CUDAGuard device_guard(q.device());
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(k.scalar_type() == torch::kBFloat16, "k must be bfloat16");
    TORCH_CHECK(v.scalar_type() == torch::kBFloat16, "v must be bfloat16");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "expected [B,H,S,D]");
    TORCH_CHECK(q.size(2) == 1, "decode expects q_len=1");
    TORCH_CHECK(q.size(3) == kHeadDim && k.size(3) == kHeadDim && v.size(3) == kHeadDim,
                "decode supports head_dim=128");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0), "batch mismatch");
    TORCH_CHECK(k.size(1) == v.size(1) && k.size(2) == v.size(2), "K/V shape mismatch");
    TORCH_CHECK(q.size(0) > 0 && q.size(1) > 0 && k.size(1) > 0 && k.size(2) > 0,
                "batch, head counts, and kv_len must be positive");
    TORCH_CHECK(q.size(1) % k.size(1) == 0, "q_heads must be divisible by kv_heads");
    TORCH_CHECK(q.size(1) / k.size(1) <= kMaxGroup, "decode supports at most 8 q heads per kv head");
    TORCH_CHECK(k.size(2) % 128 == 0, "decode requires kv_len divisible by 128");

    auto q_contig = q.contiguous();
    auto k_contig = k.contiguous();
    auto v_contig = v.contiguous();
    const int batch = q_contig.size(0);
    const int q_heads = q_contig.size(1);
    const int kv_heads = k_contig.size(1);
    const int kv_len = k_contig.size(2);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(k_contig.data_ptr()) % 16 == 0,
                "K base pointer must be 16-byte aligned");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(v_contig.data_ptr()) % 16 == 0,
                "V base pointer must be 16-byte aligned");
    cudaDeviceProp prop;
    TORCH_CHECK(cudaGetDeviceProperties(&prop, q.get_device()) == cudaSuccess,
                "failed to query CUDA device properties");
    const int splits = choose_splits(batch, kv_heads, kv_len, prop.multiProcessorCount);
    const float scale = static_cast<float>(sm_scale);

    auto partial_o = torch::empty(
        {batch, q_heads, splits, kHeadDim}, q_contig.options().dtype(torch::kFloat32));
    auto partial_m = torch::empty(
        {batch, q_heads, splits}, q_contig.options().dtype(torch::kFloat32));
    auto partial_l = torch::empty(
        {batch, q_heads, splits}, q_contig.options().dtype(torch::kFloat32));
    auto output = torch::empty({batch, q_heads, 1, kHeadDim}, q_contig.options());

    auto stream = c10::cuda::getCurrentCUDAStream(q.get_device());
    const dim3 split_grid(splits, kv_heads, batch);
    decode_split_kernel<<<split_grid, kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_contig.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k_contig.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v_contig.data_ptr()),
        partial_o.data_ptr<float>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        batch,
        q_heads,
        kv_heads,
        kv_len,
        splits,
        scale);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "decode split kernel launch failed");

    const dim3 combine_grid(q_heads, batch);
    decode_combine_kernel<<<combine_grid, kHeadDim, 0, stream>>>(
        partial_o.data_ptr<float>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        q_heads,
        splits);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "decode combine kernel launch failed");
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    m.def(
        "forward",
        &decode_forward,
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("sm_scale") = 0.08838834764831845,
        "GQA decode forward (BF16, D128)");
}
