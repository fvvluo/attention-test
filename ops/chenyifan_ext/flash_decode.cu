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
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kTokensPerTile = 32;
constexpr int kKStride = kHeadDim + 8;
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

// ldmatrix.x2: load a 16x8 bf16 tile (B-operand fragment for m16n8k16).
__device__ __forceinline__ void ldmatrix_x2(uint32_t (&r)[2], const void* smem_ptr) {
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
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
    __shared__ __align__(16) __nv_bfloat16 s_q[16][kHeadDim];  // pad to 16 rows for mma M
    __shared__ __align__(16) __nv_bfloat16 s_k[2][kTokensPerTile][kKStride];
    __shared__ __align__(16) __nv_bfloat16 s_v[2][kTokensPerTile][kHeadDim];
    __shared__ float s_probs[kMaxGroup][kTokensPerTile];
    __shared__ __align__(16) __nv_bfloat16 s_pbf[16][kTokensPerTile];  // P (bf16) for PV mma
    __shared__ float s_alpha[kMaxGroup];                               // per-head rescale
    __shared__ float s_row_max[kMaxGroup];
    __shared__ float s_row_sum[kMaxGroup];

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
    // acc_o[16 n8-blocks over d=128][4 f32], persistent across tiles (warp0 only).
    float acc_o[kHeadDim / 8][4];
    #pragma unroll
    for (int nb = 0; nb < kHeadDim / 8; ++nb) {
        #pragma unroll
        for (int r = 0; r < 4; ++r) acc_o[nb][r] = 0.0f;
    }

    const long long q_group_offset =
        (static_cast<long long>(b) * q_heads + kv_head * group) * kHeadDim;
    constexpr int kQVectors = kWarps * kHeadDim / 8;      // 128 (rows 0..7)
    constexpr int kQVectorsPad = 16 * kHeadDim / 8;       // 256 (rows 0..15)
    if (tid < kQVectors) {
        reinterpret_cast<uint4*>(s_q)[tid] =
            reinterpret_cast<const uint4*>(q + q_group_offset)[tid];
    } else if (tid < kQVectorsPad) {
        reinterpret_cast<uint4*>(s_q)[tid] = make_uint4(0, 0, 0, 0);
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
                reinterpret_cast<uint4*>(&s_v[0][0][0]) + vec,
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
                    reinterpret_cast<uint4*>(&s_v[next_stage][0][0]) + vec,
                    reinterpret_cast<const uint4*>(v + next_offset) + vec);
            }
            cp_async_commit();
        }

        // ---- QK via tensor-core mma (warp 0 computes full 8x32 score) ----
        // A = Q[16(head, 8 valid) x 16(dim)], B = K[16(token) x 16(dim)] col-major,
        // D[m(head)][n(token)] += sum_d Q[m][d]*K[n][d].  Loop d over 8 k16 blocks,
        // token over 4 n8 blocks (32 tokens). One warp owns the whole tile.
        if (warp == 0) {
            // acc[nblk][4]: 4 f32 per (m,n) fragment, per PTX m16n8k16 D-layout.
            float acc[4][4];
            #pragma unroll
            for (int nb = 0; nb < 4; ++nb) {
                #pragma unroll
                for (int r = 0; r < 4; ++r) acc[nb][r] = 0.0f;
            }
            #pragma unroll
            for (int kblk = 0; kblk < kHeadDim / 16; ++kblk) {
                const int d0 = kblk * 16;
                uint32_t a_frag[4];
                // A source: row = lane%16 (head, only 0..7 valid, 8..15 zero-padded
                // via s_q padding), col group = (lane/16)*8 within the 16-dim block.
                const __nv_bfloat16* a_ptr = &s_q[lane % 16][d0 + (lane / 16) * 8];
                ldmatrix_x4(a_frag, a_ptr);
                #pragma unroll
                for (int nb = 0; nb < 4; ++nb) {
                    const int tok0 = nb * 8;
                    uint32_t b_frag[2];
                    // B source (col-major for mma): row = token, col = dim.
                    // ldmatrix.x2 loads a 16x8 tile; provide per-lane row address.
                    const __nv_bfloat16* b_ptr =
                        &s_k[stage][tok0 + (lane % 8)][d0 + (lane / 8) * 8];
                    ldmatrix_x2(b_frag, b_ptr);
                    mma_m16n8k16(acc[nb], a_frag, b_frag);
                }
            }
            // Write scores to s_probs. D-fragment layout (m16n8k16):
            //   c0,c1 -> row = group,    col = 2*(lane%4) + {0,1}
            //   c2,c3 -> row = group+8,  col = 2*(lane%4) + {0,1}
            // where group = lane/4. Only rows 0..7 (heads) are valid.
            const int frag_group = lane / 4;
            const int col_base = 2 * (lane % 4);
            #pragma unroll
            for (int nb = 0; nb < 4; ++nb) {
                const int tok0 = nb * 8;
                // c0,c1: row = frag_group (head 0..7); c2,c3: row = frag_group+8 (unused)
                #pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int head = frag_group;         // rows 0..7 -> heads
                    const int token = tok0 + col_base + c;
                    if (head < group && token < kTokensPerTile) {
                        s_probs[head][token] =
                            token < valid_tokens ? acc[nb][c] * scale : -INFINITY;
                    }
                }
            }
        }
        __syncthreads();

        // ---- Online softmax (warp0, lanes 0..7 own one head row each) + PV mma ----
        if (warp == 0) {
            // 1) per-head online softmax over the 32-token tile; produce bf16 P.
            if (lane < group) {
                const int head = lane;
                const float prev_max = s_row_max[head];
                const float prev_sum = s_row_sum[head];
                float tile_max = -INFINITY;
                #pragma unroll
                for (int t = 0; t < kTokensPerTile; ++t) {
                    tile_max = fmaxf(tile_max, s_probs[head][t]);
                }
                const float new_max = fmaxf(prev_max, tile_max);
                const float alpha =
                    prev_max == -INFINITY ? 0.0f : __expf(prev_max - new_max);
                float tile_sum = 0.0f;
                #pragma unroll
                for (int t = 0; t < kTokensPerTile; ++t) {
                    const float p = __expf(s_probs[head][t] - new_max);
                    tile_sum += p;
                    s_pbf[head][t] = __float2bfloat16(p);
                }
                s_alpha[head] = alpha;
                s_row_max[head] = new_max;
                s_row_sum[head] = prev_sum * alpha + tile_sum;
            }
            // zero-pad P rows 8..15 (mma M padding).
            if (lane >= group && lane < 16) {
                #pragma unroll
                for (int t = 0; t < kTokensPerTile; ++t) {
                    s_pbf[lane][t] = __float2bfloat16(0.0f);
                }
            }
            __syncwarp();

            // 2) rescale persistent O accumulator by per-head alpha.
            //    D-layout: acc_o[nb][0,1] -> head = lane/4 ; [2,3] -> head=lane/4+8 (unused)
            const float alpha_h = s_alpha[lane / 4];
            #pragma unroll
            for (int nb = 0; nb < kHeadDim / 8; ++nb) {
                acc_o[nb][0] *= alpha_h;
                acc_o[nb][1] *= alpha_h;
            }

            // 3) PV mma: O[16(head) x 128(d)] += P[16 x 32(token)] @ V[32(token) x 128(d)]
            //    A = P (row-major), B = V (col-major). token=32 -> 2 k16; d=128 -> 16 n8.
            #pragma unroll
            for (int kblk = 0; kblk < kTokensPerTile / 16; ++kblk) {
                const int t0 = kblk * 16;
                uint32_t a_frag[4];
                // A source: row = lane%16 (head), col group = (lane/16)*8 within 16 tokens.
                ldmatrix_x4(a_frag, &s_pbf[lane % 16][t0 + (lane / 16) * 8]);
                #pragma unroll
                for (int nb = 0; nb < kHeadDim / 8; ++nb) {
                    const int d0 = nb * 8;
                    uint32_t b_frag[2];
                    // B col-major m16n8k16 (k=token, n=d). Per PTX B-fragment map:
                    //   b0 -> rows {2*(lane%4), 2*(lane%4)+1}, col = lane/4
                    //   b1 -> rows {2*(lane%4)+8, +9},         col = lane/4
                    const int nrow = 2 * (lane % 4);
                    const int ncol = lane / 4;   // 0..7 -> d within n8
                    __nv_bfloat162 b0 = make_bfloat162(
                        s_v[stage][t0 + nrow][d0 + ncol],
                        s_v[stage][t0 + nrow + 1][d0 + ncol]);
                    __nv_bfloat162 b1 = make_bfloat162(
                        s_v[stage][t0 + nrow + 8][d0 + ncol],
                        s_v[stage][t0 + nrow + 9][d0 + ncol]);
                    b_frag[0] = *reinterpret_cast<const uint32_t*>(&b0);
                    b_frag[1] = *reinterpret_cast<const uint32_t*>(&b1);
                    mma_m16n8k16(acc_o[nb], a_frag, b_frag);
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

    // ---- write partial results (warp0 holds acc_o) ----
    if (warp == 0) {
        const int frag_group = lane / 4;   // head for c0,c1
        const int col_base = 2 * (lane % 4);
        const int head = frag_group;
        if (head < group) {
            const int q_head_out = kv_head * group + head;
            const long long stat_idx =
                (static_cast<long long>(b) * q_heads + q_head_out) * splits + split;
            const long long out_base = stat_idx * kHeadDim;
            // acc_o[nb][0,1] -> d = nb*8 + col_base + {0,1}
            #pragma unroll
            for (int nb = 0; nb < kHeadDim / 8; ++nb) {
                const int d0 = nb * 8 + col_base;
                partial_o[out_base + d0 + 0] = acc_o[nb][0];
                partial_o[out_base + d0 + 1] = acc_o[nb][1];
            }
            if (lane % 4 == 0) {
                partial_m[stat_idx] = s_row_max[head];
                partial_l[stat_idx] = s_row_sum[head];
            }
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
    TORCH_CHECK(q.size(1) / k.size(1) <= kWarps, "decode supports at most 8 q heads per kv head");
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
