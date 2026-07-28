#!/usr/bin/env python3
"""B5 single-level pure-GPU LSE-weighted split combine.

Independent implementation by Liu Xiaochen. Design reference (idea only):
LSE-weighted split reduction as used in Flash-Decoding / quanbofeng's decode;
code written from scratch with the public CuTe DSL.

One CTA (128 threads) per query head. Reads FP32 normalized-O partials + FP32
LSE per split, produces final BF16 output. Pure GPU, no torch.max/sum, no CPU.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute


class LiuXiaochenMmaDecodeCombineB5:
    def __init__(self, head_dim, num_splits):
        self.head_dim = head_dim
        self.num_splits = num_splits
        self.num_threads = 128

    @cute.jit
    def __call__(self, mPartial, mLSE, mO, stream: cuda.CUstream):
        @cute.struct
        class Smem:
            weights: cute.struct.MemRange[cutlass.Float32, self.num_splits]
            reduction: cute.struct.MemRange[cutlass.Float32, self.num_threads]

        self.kernel(mPartial, mLSE, mO, Smem).launch(
            grid=[mO.shape[1], mO.shape[0], 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mPartial, mLSE, mO, Smem: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        q_head, batch, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(Smem)
        weights = storage.weights.get_tensor(cute.make_layout(self.num_splits))
        reduction = storage.reduction.get_tensor(cute.make_layout(self.num_threads))

        # 1) global max of LSE.
        local_max = -cutlass.Float32.inf
        s = tidx
        while s < self.num_splits:
            local_max = cute.arch.fmax(local_max, mLSE[batch, q_head, s])
            s += self.num_threads
        reduction[tidx] = local_max
        cute.arch.sync_threads()
        gmax = -cutlass.Float32.inf
        for i in cutlass.range_constexpr(self.num_threads):
            gmax = cute.arch.fmax(gmax, reduction[i])
        cute.arch.sync_threads()

        # 2) weights = exp(lse - gmax), denom.
        local_sum = cutlass.Float32(0.0)
        s = tidx
        while s < self.num_splits:
            w = cute.math.exp(mLSE[batch, q_head, s] - gmax, fastmath=True)
            weights[s] = w
            local_sum += w
            s += self.num_threads
        reduction[tidx] = local_sum
        cute.arch.sync_threads()
        denom = cutlass.Float32(0.0)
        for i in cutlass.range_constexpr(self.num_threads):
            denom += reduction[i]
        inv = cute.arch.rcp_approx(denom)
        s = tidx
        while s < self.num_splits:
            weights[s] *= inv
            s += self.num_threads
        cute.arch.sync_threads()

        # 3) weighted sum over splits per dim.
        dim = tidx
        while dim < self.head_dim:
            acc = cutlass.Float32(0.0)
            for si in cutlass.range_constexpr(self.num_splits):
                acc += mPartial[batch, q_head, si, dim].to(cutlass.Float32) * weights[si]
            mO[batch, q_head, dim] = acc.to(mO.element_type)
            dim += self.num_threads
