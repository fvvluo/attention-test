#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kHeadDim = 128;
constexpr int kThreads = 32;          // 1 warp / CTA
constexpr int kTokensPerTile = 32;    // 每个 KV tile 的 token 数（=2 个 m16 块）
constexpr int kMaxSplits = 64;
constexpr int kMaxGroup = 8;
constexpr int kMBlk = kTokensPerTile / 16;  // m16 token 块数          // 每 kv-head 最多 q-head 数（MMA-N）
constexpr int kKStride = kHeadDim;        // sK/sV 每行元素数（swizzle 避 bank conflict）
constexpr int kQStride = kHeadDim + 8;    // sQ 加 padding 避 ldmatrix 列主访问 bank conflict
constexpr int kPStride = kTokensPerTile + 8;  // sP 同理

// ---- warp 归约 ----
__device__ __forceinline__ float warp_sum(float x) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) x += __shfl_down_sync(0xffffffff, x, off);
    return __shfl_sync(0xffffffff, x, 0);
}
__device__ __forceinline__ float warp_max(float x) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) x = fmaxf(x, __shfl_down_sync(0xffffffff, x, off));
    return __shfl_sync(0xffffffff, x, 0);
}

// ---- cp.async 16B ----
__device__ __forceinline__ void cp_async_16(void* dst, const void* src) {
    unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(src));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_async_wait0() { asm volatile("cp.async.wait_group 0;\n" ::); }

// ---- swizzle：按 token 行低位 XOR 16B 列块，避 smem bank conflict。
// 逻辑列 col（bf16 元素索引，16 对齐）-> 物理列。cp.async 写入与 ldmatrix 读取用同一映射。
__device__ __forceinline__ int swz(int row, int col) {
    // 16B = 8 bf16 一块；用 row 的低 3 位异或块号
    int blk = col >> 3;
    int xblk = blk ^ (row & 7);
    return (xblk << 3) | (col & 7);
}

// ---- mma.m16n8k16 bf16 -> f32。D[4] += A[4] * B[2]。A: 16x16 row.col; B: 16x8. ----
__device__ __forceinline__ void mma_m16n8k16(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* p) {
    unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], const void* p) {
    unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldmatrix_x2(uint32_t (&r)[2], const void* p) {
    unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}

// ============================ split kernel ============================
// grid = (splits, kv_heads, batch)，block = 32（1 warp）。
__global__ __launch_bounds__(kThreads, 16) void decode_split_kernel(
    const __nv_bfloat16* __restrict__ q,   // (b, q_heads, 1, D)
    const __nv_bfloat16* __restrict__ k,   // (b, kv_heads, kv_len, D)
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ partial_o, // (b, q_heads, splits, D)
    float* __restrict__ partial_m,         // (b, q_heads, splits)
    float* __restrict__ partial_l,
    int q_heads, int kv_heads, int kv_len, int splits, float scale) {

    const int lane = threadIdx.x;
    const int split = blockIdx.x;
    const int kv_head = blockIdx.y;
    const int b = blockIdx.z;
    const int group = q_heads / kv_heads;   // qpkv, <=8

    // 双缓冲 smem
    __shared__ __align__(16) __nv_bfloat16 s_k[2][kTokensPerTile][kKStride];
    __shared__ __align__(16) __nv_bfloat16 s_v[2][kTokensPerTile][kKStride];
    __shared__ __align__(16) __nv_bfloat16 s_q[kMaxGroup][kQStride];
    __shared__ __align__(16) __nv_bfloat16 s_p[kMaxGroup][kPStride];  // P^T: [head][token]
    __shared__ float s_max[kMaxGroup];
    __shared__ float s_sum[kMaxGroup];
    __shared__ float s_alpha[kMaxGroup];

    // ---- 载入 Q 到 smem（8 head × 128 dim）----
    // q 布局 (b, q_heads, 1, D) 连续；本 kv_head 的 group 个 q-head 全局索引 = kv_head*group + h。
    for (int idx = lane; idx < group * kHeadDim; idx += kThreads) {
        int h = idx / kHeadDim;
        int d = idx % kHeadDim;
        long long qoff = ((long long)(b * q_heads + kv_head * group + h)) * kHeadDim + d;
        s_q[h][d] = q[qoff];
    }
    for (int h = lane; h < group; h += kThreads) { s_max[h] = -INFINITY; s_sum[h] = 0.0f; }

    // ---- KV tile 范围（本 split）----
    const int tiles = kv_len / kTokensPerTile;
    const int tile_begin = (int)(((long long)tiles * split) / splits);
    const int tile_end = (int)(((long long)tiles * (split + 1)) / splits);

    const long long kv_head_off = ((long long)(b * kv_heads + kv_head)) * kv_len * kHeadDim;
    constexpr int kVecsPerTile = kTokensPerTile * kHeadDim / 8;  // 16B=8bf16 单位

    // O^T 累加器：每 lane 持 D/16=8 个 db 块，每块 4 个 float（对应 O^T 的 (dim,head) 片）
    float acc_o[kHeadDim / 16][4];
    #pragma unroll
    for (int db = 0; db < kHeadDim / 16; ++db)
        #pragma unroll
        for (int r = 0; r < 4; ++r) acc_o[db][r] = 0.0f;

    // prologue: 预取第一个 tile 到 stage 0
    if (tile_begin < tile_end) {
        long long off = kv_head_off + (long long)(tile_begin * kTokensPerTile) * kHeadDim;
        for (int vec = lane; vec < kVecsPerTile; vec += kThreads) {
            int tok = vec / (kHeadDim / 8);
            int vic = vec % (kHeadDim / 8);
            cp_async_16(&s_k[0][tok][swz(tok, vic * 8)], reinterpret_cast<const uint4*>(k + off) + vec);
            cp_async_16(&s_v[0][tok][swz(tok, vic * 8)], reinterpret_cast<const uint4*>(v + off) + vec);
        }
        cp_async_commit();
        cp_async_wait0();
    }
    __syncwarp();

    for (int tile = tile_begin; tile < tile_end; ++tile) {
        const int stage = (tile - tile_begin) & 1;
        const int token0 = tile * kTokensPerTile;
        const int valid = min(kTokensPerTile, kv_len - token0);

        // 预取下一个 tile
        if (tile + 1 < tile_end) {
            const int ns = stage ^ 1;
            long long off = kv_head_off + (long long)((tile + 1) * kTokensPerTile) * kHeadDim;
            for (int vec = lane; vec < kVecsPerTile; vec += kThreads) {
                int tok = vec / (kHeadDim / 8);
                int vic = vec % (kHeadDim / 8);
                cp_async_16(&s_k[ns][tok][swz(tok, vic * 8)], reinterpret_cast<const uint4*>(k + off) + vec);
                cp_async_16(&s_v[ns][tok][swz(tok, vic * 8)], reinterpret_cast<const uint4*>(v + off) + vec);
            }
            cp_async_commit();
        }

        // ==== QK: S[16 token, 8 head] = K[16 token,16 dim] @ Q^T[16 dim,8 head]，两个 m16 块覆盖 32 token ====
        float acc_s[kMBlk][4];  // [tb][4]: m16n8k16 D 输出
        #pragma unroll
        for (int tb = 0; tb < kMBlk; ++tb)
            #pragma unroll
            for (int r = 0; r < 4; ++r) acc_s[tb][r] = 0.0f;

        #pragma unroll
        for (int kb = 0; kb < kHeadDim / 16; ++kb) {
            const int d0 = kb * 16;
            // B 操作数 = Q^T[16 dim, 8 head]，来自 s_q[head][dim] 的列主存储。ldmatrix.x2。
            uint32_t q_frag[2];
            // s_q[head][dim]: 取 head=lane%8, dim=d0+(lane/8)*8 的 8x8 块
            ldmatrix_x2(q_frag, &s_q[lane % 8][d0 + (lane / 8) * 8]);
            #pragma unroll
            for (int tb = 0; tb < kMBlk; ++tb) {
                // A 操作数 = K[16 token,16 dim] row-major，ldmatrix.x4
                const int krow = tb * 16 + (lane % 16);
                const int kcol = d0 + (lane / 16) * 8;
                uint32_t k_frag[4];
                ldmatrix_x4(k_frag, &s_k[stage][krow][swz(krow, kcol)]);
                mma_m16n8k16(acc_s[tb], k_frag, q_frag);
            }
        }

        // ==== softmax ====
        // m16n8k16 的 D 输出布局：lane 拥有 (token_row = lane/4, head_pair = lane%4)。
        // acc_s[tb][0/1] 对应 token_row（head0/head1），[2/3] 对应 token_row+8。
        const int token_row = lane >> 2;
        const int head0 = 2 * (lane & 3);
        const int head1 = head0 + 1;
        float m0 = -INFINITY, m1 = -INFINITY;
        #pragma unroll
        for (int tb = 0; tb < kMBlk; ++tb) {
            acc_s[tb][0] = (tb * 16 + token_row     < valid && head0 < group) ? acc_s[tb][0] * scale : -INFINITY;
            acc_s[tb][1] = (tb * 16 + token_row     < valid && head1 < group) ? acc_s[tb][1] * scale : -INFINITY;
            acc_s[tb][2] = (tb * 16 + token_row + 8 < valid && head0 < group) ? acc_s[tb][2] * scale : -INFINITY;
            acc_s[tb][3] = (tb * 16 + token_row + 8 < valid && head1 < group) ? acc_s[tb][3] * scale : -INFINITY;
            m0 = fmaxf(m0, fmaxf(acc_s[tb][0], acc_s[tb][2]));
            m1 = fmaxf(m1, fmaxf(acc_s[tb][1], acc_s[tb][3]));
        }
        // 跨 lane 归约（同 head 的所有 token_row 分布在 lane 间，步长 4）
        #pragma unroll
        for (int off = 4; off <= 16; off <<= 1) {
            m0 = fmaxf(m0, __shfl_xor_sync(0xffffffff, m0, off));
            m1 = fmaxf(m1, __shfl_xor_sync(0xffffffff, m1, off));
        }
        const float pm0 = s_max[head0], pm1 = s_max[head1];
        const float nm0 = fmaxf(pm0, m0), nm1 = fmaxf(pm1, m1);
        const float a0 = pm0 == -INFINITY ? 0.0f : __expf(pm0 - nm0);
        const float a1 = pm1 == -INFINITY ? 0.0f : __expf(pm1 - nm1);
        float sum0 = 0.0f, sum1 = 0.0f;
        #pragma unroll
        for (int tb = 0; tb < kMBlk; ++tb) {
            const int t0 = tb * 16 + token_row;
            const int t8 = t0 + 8;
            const float p0 = __expf(acc_s[tb][0] - nm0);
            const float p1 = __expf(acc_s[tb][1] - nm1);
            const float p8 = __expf(acc_s[tb][2] - nm0);
            const float p9 = __expf(acc_s[tb][3] - nm1);
            sum0 += p0 + p8; sum1 += p1 + p9;
            // 写 P^T 到 smem: s_p[head][token]
            if (head0 < group) { s_p[head0][t0] = __float2bfloat16(p0); s_p[head0][t8] = __float2bfloat16(p8); }
            if (head1 < group) { s_p[head1][t0] = __float2bfloat16(p1); s_p[head1][t8] = __float2bfloat16(p9); }
        }
        #pragma unroll
        for (int off = 4; off <= 16; off <<= 1) {
            sum0 += __shfl_xor_sync(0xffffffff, sum0, off);
            sum1 += __shfl_xor_sync(0xffffffff, sum1, off);
        }
        if (token_row == 0) {
            if (head0 < group) { s_max[head0] = nm0; s_sum[head0] = s_sum[head0] * a0 + sum0; s_alpha[head0] = a0; }
            if (head1 < group) { s_max[head1] = nm1; s_sum[head1] = s_sum[head1] * a1 + sum1; s_alpha[head1] = a1; }
        }
        __syncwarp();

        // ==== rescale 已累加的 acc_o（每 tile 一次）====
        const float oa0 = s_alpha[2 * (lane & 3)];
        const float oa1 = s_alpha[2 * (lane & 3) + 1];
        #pragma unroll
        for (int db = 0; db < kHeadDim / 16; ++db) {
            acc_o[db][0] *= oa0; acc_o[db][1] *= oa1;
            acc_o[db][2] *= oa0; acc_o[db][3] *= oa1;
        }

        // ==== PV: O^T[128 dim, 8 head] = V^T[128 dim,16 token] @ P^T[16 token,8 head]，两个 token 块 ====
        #pragma unroll
        for (int kb = 0; kb < kTokensPerTile / 16; ++kb) {
            const int t0 = kb * 16;
            // B = P^T[16 token, 8 head]，来自 s_p[head][token] 列主存储。ldmatrix.x2
            uint32_t p_frag[2];
            ldmatrix_x2(p_frag, &s_p[lane % 8][t0 + ((lane / 8) & 1) * 8]);
            #pragma unroll
            for (int db = 0; db < kHeadDim / 16; ++db) {
                const int d0 = db * 16;
                // A = V^T[16 dim, 16 token]，由 row-major V[token,dim] 转置 load。ldmatrix.x4.trans
                const int mat = lane / 8;
                const int src_tok = t0 + (lane % 8) + (mat / 2) * 8;
                const int src_d = d0 + (mat & 1) * 8;
                uint32_t v_frag[4];
                ldmatrix_x4_trans(v_frag, &s_v[stage][src_tok][swz(src_tok, src_d)]);
                mma_m16n8k16(acc_o[db], v_frag, p_frag);
            }
        }

        if (tile + 1 < tile_end) { cp_async_wait0(); __syncwarp(); }
    }

    // ==== 写 partial：O^T fragment -> partial_o[head][d] ====
    const int drow = lane >> 2;
    const int oh0 = 2 * (lane & 3), oh1 = oh0 + 1;
    const int qh0 = kv_head * group + oh0, qh1 = kv_head * group + oh1;
    const long long st0 = ((long long)(b * q_heads + qh0)) * splits + split;
    const long long st1 = ((long long)(b * q_heads + qh1)) * splits + split;
    #pragma unroll
    for (int db = 0; db < kHeadDim / 16; ++db) {
        const int d0 = db * 16 + drow;
        if (oh0 < group) {
            partial_o[st0 * kHeadDim + d0]     = __float2bfloat16(acc_o[db][0]);
            partial_o[st0 * kHeadDim + d0 + 8] = __float2bfloat16(acc_o[db][2]);
        }
        if (oh1 < group) {
            partial_o[st1 * kHeadDim + d0]     = __float2bfloat16(acc_o[db][1]);
            partial_o[st1 * kHeadDim + d0 + 8] = __float2bfloat16(acc_o[db][3]);
        }
    }
    if (drow == 0) {
        if (oh0 < group) { partial_m[st0] = s_max[oh0]; partial_l[st0] = s_sum[oh0]; }
        if (oh1 < group) { partial_m[st1] = s_max[oh1]; partial_l[st1] = s_sum[oh1]; }
    }
}

// ============================ combine kernel ============================
// grid = (q_heads, batch)，block = 128（每 thread 一个 dim）。
__global__ void decode_combine_kernel(
    const __nv_bfloat16* __restrict__ partial_o,
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    __nv_bfloat16* __restrict__ output,
    int q_heads, int splits) {
    const int q_head = blockIdx.x;
    const int b = blockIdx.y;
    const int d = threadIdx.x;
    const long long base = ((long long)(b * q_heads + q_head)) * splits;
    __shared__ float weights[kMaxSplits];
    __shared__ float denom_s;

    if (d < 32) {
        float lmax = -INFINITY;
        for (int s = d; s < splits; s += 32) lmax = fmaxf(lmax, partial_m[base + s]);
        const float gmax = warp_max(lmax);
        float ldenom = 0.0f;
        for (int s = d; s < splits; s += 32) {
            float w = __expf(partial_m[base + s] - gmax);
            weights[s] = w;
            ldenom += partial_l[base + s] * w;
        }
        float dn = warp_sum(ldenom);
        if (d == 0) denom_s = dn;
    }
    __syncthreads();

    float num = 0.0f;
    for (int s = 0; s < splits; ++s)
        num += __bfloat162float(partial_o[(base + s) * kHeadDim + d]) * weights[s];
    const long long oidx = ((long long)(b * q_heads + q_head)) * kHeadDim + d;
    output[oidx] = __float2bfloat16(num / (denom_s + 1e-30f));
}

int choose_splits(int batch, int kv_heads, int kv_len, int num_sms) {
    const int max_useful = min(kMaxSplits, kv_len / kTokensPerTile);
    const int target = 6 * num_sms;
    const int occ = max(1, target / (batch * kv_heads));
    return max(1, min(max_useful, occ));
}

}  // namespace

torch::Tensor decode_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, double sm_scale) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q/k/v must be CUDA");
    c10::cuda::CUDAGuard guard(q.device());
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "bf16 only");
    TORCH_CHECK(q.dim() == 4 && q.size(2) == 1 && q.size(3) == kHeadDim, "expect [B,Hq,1,128]");
    TORCH_CHECK(k.size(3) == kHeadDim && k.size(2) % kTokensPerTile == 0, "kv_len % 32 == 0");
    TORCH_CHECK(q.size(1) % k.size(1) == 0 && q.size(1) / k.size(1) <= kMaxGroup, "GQA<=8");

    auto qc = q.contiguous(), kc = k.contiguous(), vc = v.contiguous();
    const int batch = qc.size(0), q_heads = qc.size(1), kv_heads = kc.size(1), kv_len = kc.size(2);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, q.get_device());
    const int splits = choose_splits(batch, kv_heads, kv_len, prop.multiProcessorCount);
    const float scale = (float)sm_scale;

    auto po = torch::empty({batch, q_heads, splits, kHeadDim}, qc.options());
    auto pm = torch::empty({batch, q_heads, splits}, qc.options().dtype(torch::kFloat32));
    auto pl = torch::empty({batch, q_heads, splits}, qc.options().dtype(torch::kFloat32));
    auto out = torch::empty({batch, q_heads, 1, kHeadDim}, qc.options());

    auto stream = c10::cuda::getCurrentCUDAStream(q.get_device());
    dim3 sg(splits, kv_heads, batch);
    decode_split_kernel<<<sg, kThreads, 0, stream>>>(
        (const __nv_bfloat16*)qc.data_ptr(), (const __nv_bfloat16*)kc.data_ptr(),
        (const __nv_bfloat16*)vc.data_ptr(), (__nv_bfloat16*)po.data_ptr(),
        pm.data_ptr<float>(), pl.data_ptr<float>(),
        q_heads, kv_heads, kv_len, splits, scale);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "split kernel launch failed");

    dim3 cg(q_heads, batch);
    decode_combine_kernel<<<cg, kHeadDim, 0, stream>>>(
        (__nv_bfloat16*)po.data_ptr(), pm.data_ptr<float>(), pl.data_ptr<float>(),
        (__nv_bfloat16*)out.data_ptr(), q_heads, splits);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "combine kernel launch failed");
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &decode_forward, "ljr GQA decode (bf16, D128)",
          pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
          pybind11::arg("sm_scale") = 0.08838834764831845);
}
