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
    __shared__ __align__(16) __nv_bfloat16 s_q[kWarps][kHeadDim];
    __shared__ __align__(16) __nv_bfloat16 s_k[2][kTokensPerTile][kKStride];
    __shared__ __align__(16) __nv_bfloat16 s_v[2][kTokensPerTile][kHeadDim];
    __shared__ float s_probs[kWarps][kTokensPerTile];

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

    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float out[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    const long long q_group_offset =
        (static_cast<long long>(b) * q_heads + kv_head * group) * kHeadDim;
    constexpr int kQVectors = kWarps * kHeadDim / 8;
    if (tid < kQVectors) {
        reinterpret_cast<uint4*>(s_q)[tid] =
            reinterpret_cast<const uint4*>(q + q_group_offset)[tid];
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

        if (active) {
            const int subgroup = lane >> 3;
            const int subgroup_lane = lane & 7;
            #pragma unroll
            for (int token_base = 0; token_base < kTokensPerTile; token_base += 4) {
                const int token = token_base + subgroup;
                const int d = subgroup_lane * 16;
                float dot = 0.0f;
                #pragma unroll
                for (int i = 0; i < 16; i += 2) {
                    const float2 q_pair = __bfloat1622float2(
                        *reinterpret_cast<const __nv_bfloat162*>(&s_q[warp][d + i]));
                    const float2 k_pair = __bfloat1622float2(
                        *reinterpret_cast<const __nv_bfloat162*>(&s_k[stage][token][d + i]));
                    dot = fmaf(q_pair.x, k_pair.x, dot);
                    dot = fmaf(q_pair.y, k_pair.y, dot);
                }
                #pragma unroll
                for (int offset = 4; offset > 0; offset >>= 1) {
                    dot += __shfl_down_sync(0xffffffff, dot, offset, 8);
                }
                if (subgroup_lane == 0) {
                    s_probs[warp][token] = token < valid_tokens ? dot * scale : -INFINITY;
                }
            }

            __syncwarp();
            const float previous_max = __shfl_sync(0xffffffff, row_max, 0);
            const float previous_sum = __shfl_sync(0xffffffff, row_sum, 0);
            const float score = s_probs[warp][lane];
            const float tile_max = warp_max(score);
            const float new_max = fmaxf(previous_max, tile_max);
            float alpha = 0.0f;
            if (lane == 0) {
                alpha = previous_max == -INFINITY ? 0.0f : __expf(previous_max - new_max);
            }
            alpha = __shfl_sync(0xffffffff, alpha, 0);
            const float probability = __expf(score - new_max);
            s_probs[warp][lane] = probability;
            const float tile_sum = warp_sum(probability);
            if (lane == 0) {
                row_sum = previous_sum * alpha + tile_sum;
                row_max = new_max;
            }
            __syncwarp();

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                out[i] *= alpha;
            }
            #pragma unroll
            for (int token = 0; token < kTokensPerTile; ++token) {
                const float probability = s_probs[warp][token];
                const int d = lane * 4;
                const float2 v01 = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(&s_v[stage][token][d]));
                const float2 v23 = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(&s_v[stage][token][d + 2]));
                out[0] = fmaf(probability, v01.x, out[0]);
                out[1] = fmaf(probability, v01.y, out[1]);
                out[2] = fmaf(probability, v23.x, out[2]);
                out[3] = fmaf(probability, v23.y, out[3]);
            }
        }

        if (next_tile < tile_end) {
            cp_async_wait();
            __syncthreads();
        }
    }

    if (active) {
        const long long stat_idx =
            (static_cast<long long>(b) * q_heads + q_head) * splits + split;
        if (lane == 0) {
            partial_m[stat_idx] = row_max;
            partial_l[stat_idx] = row_sum;
        }
        const long long out_idx = stat_idx * kHeadDim;
        reinterpret_cast<float4*>(partial_o + out_idx)[lane] =
            make_float4(out[0], out[1], out[2], out[3]);
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
