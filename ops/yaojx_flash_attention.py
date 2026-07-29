

import math
from typing import Type

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.nvgpu import cpasync, warp
import cutlass.utils as utils

from .base import register

try:
    import cuda.bindings.driver as cuda
    _HAS_CUDA_PY = True
except Exception:  # pragma: no cover
    _HAS_CUDA_PY = False


LOG2_E = 1.4426950408889634074
_DECODE_QLEN_THRESHOLD = 16
_COMPILE_CACHE = {}


class FlashAttentionPrefill:


    def __init__(
        self,
        head_dim: int,
        m_block_size: int = 128,
        n_block_size: int = 128,
        num_threads: int = 128,
        is_causal: bool = True,
    ):
        self._head_dim = head_dim

        self._m_block_size = m_block_size
        self._n_block_size = n_block_size
        self._num_threads = num_threads
        self._is_causal = is_causal

    # ------------------------------------------------------------------ #
    # host 端：布局/copy atom/MMA 设置 + launch
    # ------------------------------------------------------------------ #
    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,      # [B, H,    S, D]
        mK: cute.Tensor,      # [B, H_kv, S, D]
        mV: cute.Tensor,      # [B, H_kv, S, D]
        mO: cute.Tensor,      # [B, H,    S, D]
        softmax_scale: cutlass.Float32,
        stream,
    ):
        self._dtype: Type[cutlass.Numeric] = mQ.element_type

        # ---- 共享内存 swizzled 布局 (Q/K/V/O) ----
        smem_k_block = 64 if self._head_dim_padded % 64 == 0 else 32
        swizzle_bits = 3 if smem_k_block == 64 else 2
        sQ_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3),
            0,
            cute.make_layout((8, smem_k_block), stride=(smem_k_block, 1)),
        )
        sQ_layout = cute.tile_to_shape(
            sQ_layout_atom, (self._m_block_size, self._head_dim_padded), (0, 1))
        sKV_layout = cute.tile_to_shape(
            sQ_layout_atom, (self._n_block_size, self._head_dim_padded), (0, 1))
        sO_layout = sQ_layout

        # ---- gmem copy atoms ----
        copy_bits = 128
        async_elems = copy_bits // self._dtype.width
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self._dtype,
            num_bits_per_copy=copy_bits,
        )
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self._dtype, num_bits_per_copy=copy_bits)

        t_dim1 = sQ_layout_atom.outer.shape[1] // async_elems
        tQKV_layout = cute.make_layout(
            (self._num_threads // t_dim1, t_dim1), stride=(t_dim1, 1))
        vQKV_layout = cute.make_layout((1, async_elems))
        gmem_tiled_copy_QKV = cute.make_tiled_copy_tv(
            atom_async_copy, tQKV_layout, vQKV_layout)
        gmem_tiled_copy_O = cute.make_tiled_copy_tv(
            atom_universal_copy, tQKV_layout, vQKV_layout)

        # ---- tensor-core MMA ----
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self._dtype, cutlass.Float32, (16, 8, 16)),
            (self._num_threads // 32, 1, 1),
            permutation_mnk=(self._num_threads // 32 * 16, 16, 16),
        )

        # ---- smem->reg copy atoms (ldmatrix) ----
        smem_copy_atom_QK = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self._dtype)
        smem_copy_atom_V = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self._dtype)
        smem_tiled_copy_Q = cute.make_tiled_copy_A(smem_copy_atom_QK, tiled_mma)
        smem_tiled_copy_K = cute.make_tiled_copy_B(smem_copy_atom_QK, tiled_mma)
        smem_tiled_copy_V = cute.make_tiled_copy_B(smem_copy_atom_V, tiled_mma)

        @cute.struct
        class SharedStorage:
            sQ: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sQ_layout)], 1024]
            sK: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024]
            sV: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024]

        H = cute.size(mQ.shape[1])
        H_kv = cute.size(mK.shape[1])
        self._gqa_group = H // H_kv

        softmax_scale_log2 = softmax_scale * LOG2_E

        grid_dim = (
            cute.ceil_div(cute.size(mQ.shape[2]), self._m_block_size),  # m blocks
            cute.size(mQ.shape[0]),                                     # batch
            H,                                                          # query heads
        )
        self.kernel(
            mQ, mK, mV, mO, softmax_scale_log2,
            sQ_layout, sKV_layout, sO_layout,
            gmem_tiled_copy_QKV, gmem_tiled_copy_O,
            tiled_mma,
            smem_tiled_copy_Q, smem_tiled_copy_K, smem_tiled_copy_V,
            SharedStorage,
        ).launch(
            grid=grid_dim,
            block=[self._num_threads, 1, 1],
            stream=stream,
        )

   
    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mO: cute.Tensor,
        softmax_scale_log2: cutlass.Float32,
        # [修复] swizzled 的 _ComposedLayout 不能作为 cute.Layout 类型的运行时
        # kernel 参数传入，改为编译期常量（Constexpr）捕获。
        sQ_layout: cutlass.Constexpr,
        sKV_layout: cutlass.Constexpr,
        sO_layout: cutlass.Constexpr,
        gmem_tiled_copy_QKV: cute.TiledCopy, gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        smem_tiled_copy_Q: cute.TiledCopy,
        smem_tiled_copy_K: cute.TiledCopy,
        smem_tiled_copy_V: cute.TiledCopy,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        m_block, batch, head = cute.arch.block_idx()
        kv_head = head // self._gqa_group  # GQA 映射

        seqlen_q = cute.size(mQ.shape[2])
        seqlen_k = cute.size(mK.shape[2])

        # 需要处理的 KV block 数（causal 提前退出）
        n_block_max = cute.ceil_div(seqlen_k, self._n_block_size)
        if cutlass.const_expr(self._is_causal):
            n_block_max = min(
                cute.ceil_div((m_block + 1) * self._m_block_size, self._n_block_size),
                n_block_max,
            )

        # 取出当前 (batch, head) 的 [S, D] 全局 tile
        mQ_bh = mQ[batch, head, None, None]
        mK_bh = mK[batch, kv_head, None, None]
        mV_bh = mV[batch, kv_head, None, None]
        mO_bh = mO[batch, head, None, None]

        gQ = cute.local_tile(
            mQ_bh, (self._m_block_size, self._head_dim_padded), (m_block, 0))
        gK = cute.local_tile(
            mK_bh, (self._n_block_size, self._head_dim_padded), (None, 0))
        gV = cute.local_tile(
            mV_bh, (self._n_block_size, self._head_dim_padded), (None, 0))

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sK = storage.sK.get_tensor(sKV_layout)
        sV = storage.sV.get_tensor(sKV_layout)
        # V 的转置视图 (head_dim, n_block)，用于 P@V 的 MMA
        sVt = cute.composition(
            sV,
            cute.make_layout(
                (self._head_dim_padded, self._n_block_size),
                stride=(self._n_block_size, 1)),
        )

        # ---- gmem -> smem 划分 ----
        gmem_thr_copy = gmem_tiled_copy_QKV.get_slice(tidx)
        tQgQ = gmem_thr_copy.partition_S(gQ)
        tQsQ = gmem_thr_copy.partition_D(sQ)
        tKgK = gmem_thr_copy.partition_S(gK)
        tKsK = gmem_thr_copy.partition_D(sK)
        tVgV = gmem_thr_copy.partition_S(gV)
        tVsV = gmem_thr_copy.partition_D(sV)

        # ---- MMA fragments ----
        thr_mma = tiled_mma.get_slice(tidx)
        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
        tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK))
        tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt))

        acc_shape_O = thr_mma.partition_shape_C(
            (self._m_block_size, self._head_dim_padded))
        acc_O = cute.make_rmem_tensor(acc_shape_O, cutlass.Float32)
        acc_O.fill(0.0)

        # smem->reg copy 划分
        smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(tidx)
        smem_thr_copy_K = smem_tiled_copy_K.get_slice(tidx)
        smem_thr_copy_V = smem_tiled_copy_V.get_slice(tidx)
        tSsQ = smem_thr_copy_Q.partition_S(sQ)
        tSrQ_copy = smem_thr_copy_Q.retile(tSrQ)
        tSsK = smem_thr_copy_K.partition_S(sK)
        tSrK_copy = smem_thr_copy_K.retile(tSrK)
        tOsVt = smem_thr_copy_V.partition_S(sVt)
        tOrVt_copy = smem_thr_copy_V.retile(tOrVt)

        # ---- 序幕：发射 Q + 最后一块 K 的加载 ----
        cute.copy(gmem_tiled_copy_QKV, tQgQ, tQsQ)
        cute.arch.cp_async_commit_group()

        n_block = n_block_max - 1
        cute.copy(gmem_tiled_copy_QKV, tKgK[None, None, None, n_block], tKsK)
        cute.arch.cp_async_commit_group()

        # 等待 Q 与首块 K 到位
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        
        cute.copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy)
        cute.arch.barrier()

        
       
        tSrQ.store(
            (tSrQ.load().to(cutlass.Float32) * softmax_scale_log2).to(self._dtype))

        # ---- 在线 softmax 的运行统计 ----
        acc_O_mn = self._acc_mn_view(acc_O)
        n_rows = cute.size(acc_O_mn.shape[0])
        row_max = cute.make_rmem_tensor((n_rows,), cutlass.Float32)
        row_sum = cute.make_rmem_tensor((n_rows,), cutlass.Float32)
        row_max.fill(-cutlass.Float32.inf)
        row_sum.fill(0.0)

        # 需要掩码的前导（对角线）block 数
        mask_steps = 1
        if cutlass.const_expr(self._is_causal):
            mask_steps = cute.ceil_div(self._m_block_size, self._n_block_size)

        # ---- 主循环：先处理需要掩码的对角块，再处理无需掩码的主体 ----
        for i in cutlass.range_constexpr(mask_steps):
            nb = n_block_max - 1 - i
            if nb >= 0:
                self._one_n_block(
                    nb, i == 0, True,
                    m_block, seqlen_q, seqlen_k,
                    tKgK, tKsK, tVgV, tVsV,
                    tSrQ, tSrK, tOrVt,
                    tSsK, tSrK_copy, tOsVt, tOrVt_copy,
                    smem_tiled_copy_K, smem_tiled_copy_V,
                    gmem_tiled_copy_QKV, thr_mma,
                    acc_O, row_max, row_sum, softmax_scale_log2,
                )

        for nb in cutlass.range(n_block_max - 1 - mask_steps, -1, -1, unroll=1):
            self._one_n_block(
                nb, False, False,
                m_block, seqlen_q, seqlen_k,
                tKgK, tKsK, tVgV, tVsV,
                tSrQ, tSrK, tOrVt,
                tSsK, tSrK_copy, tOsVt, tOrVt_copy,
                smem_tiled_copy_K, smem_tiled_copy_V,
                gmem_tiled_copy_QKV, thr_mma,
                acc_O, row_max, row_sum, softmax_scale_log2,
            )

        # ---- 收尾：按 row_sum 归一化并写出 O ----
        self._normalize(acc_O, row_sum)
        self._write_output(
            acc_O, mO_bh, m_block, sO_layout, gmem_tiled_copy_O,
            thr_mma, tidx, seqlen_q, storage, sQ_layout)

   
    @cute.jit
    def _one_n_block(
        self, n_block, is_first, do_mask,
        m_block, seqlen_q, seqlen_k,
        tKgK, tKsK, tVgV, tVsV,
        tSrQ, tSrK, tOrVt,
        tSsK, tSrK_copy, tOsVt, tOrVt_copy,
        smem_tiled_copy_K, smem_tiled_copy_V,
        gmem_tiled_copy_QKV, thr_mma,
        acc_O, row_max, row_sum, softmax_scale_log2,
    ):
        tiled_mma = thr_mma  # thr_mma 已绑定 tidx，用于 gemm 时传 tiled_mma
        has_next = n_block - 1 >= 0

        
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

       
        cute.copy(smem_tiled_copy_K, tSsK, tSrK_copy)
        cute.arch.barrier()  # WAR：确保所有 warp 读完 sK 再覆盖

        
        cute.copy(gmem_tiled_copy_QKV, tVgV[None, None, None, n_block], tVsV)
        cute.arch.cp_async_commit_group()
        if has_next:
            cute.copy(
                gmem_tiled_copy_QKV, tKgK[None, None, None, n_block - 1], tKsK)
            cute.arch.cp_async_commit_group()

        # S = Q @ K^T  (fp32 累加)
        acc_shape_S = thr_mma.partition_shape_C(
            (self._m_block_size, self._n_block_size))
        acc_S = cute.make_rmem_tensor(acc_shape_S, cutlass.Float32)
        acc_S.fill(0.0)
        cute.gemm(tiled_mma, acc_S, tSrQ, tSrK, acc_S)

        # 对角块施加 causal mask
        if cutlass.const_expr(self._is_causal):
            if do_mask:
                self._apply_causal_mask(
                    acc_S, m_block, n_block, seqlen_q, seqlen_k)

        # 在线 softmax：更新 row_max/row_sum，重缩放 acc_O，得到 P
        self._softmax_rescale(
            acc_S, acc_O, row_max, row_sum, softmax_scale_log2, is_first)

        # 将 P 转成输入 dtype 供第二个 MMA
        rP = cute.make_rmem_tensor(acc_S.shape, self._dtype)
        rP_frag = thr_mma.make_fragment_A(rP)
        rP_frag.store(acc_S.load().to(self._dtype))

        # 等待 V 到位（group A）；若已预取下一块 K（group B），用 wait_group(1)
        # 让下一块 K 继续在后台加载，与本块 P@V 重叠
        if has_next:
            cute.arch.cp_async_wait_group(1)
        else:
            cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()
        cute.copy(smem_tiled_copy_V, tOsVt, tOrVt_copy)
        cute.gemm(tiled_mma, acc_O, rP_frag, tOrVt, acc_O)

    
    @staticmethod
    def _acc_mn_view(acc):
        l = acc.layout
        mn = cute.make_layout(
            ((l.shape[0][0], l.shape[1]), (l.shape[0][1], l.shape[2])),
            stride=((l.stride[0][0], l.stride[1]), (l.stride[0][1], l.stride[2])),
        )
        return cute.make_tensor(acc.iterator, mn)

    @cute.jit
    def _apply_causal_mask(self, acc_S, m_block, n_block, seqlen_q, seqlen_k):
        acc_S_mn = self._acc_mn_view(acc_S)
        tidx, _, _ = cute.arch.thread_idx()
        row_off = m_block * self._m_block_size
        col_off = n_block * self._n_block_size
        for r in cutlass.range_constexpr(cute.size(acc_S_mn.shape[0])):
            for c in cutlass.range_constexpr(cute.size(acc_S_mn.shape[1])):
                q_idx = row_off + self._row_coord(r, tidx)
                k_idx = col_off + self._col_coord(c, tidx)
                # 运行时（staged）布尔不能用 Python 的 or/and/not，改用位运算 |
                masked = (k_idx > q_idx) | (k_idx >= seqlen_k)
                if masked:
                    acc_S_mn[r, c] = -cutlass.Float32.inf

    @staticmethod
    def _row_coord(r, tidx):
        # m16n8k16: row = warp*16 + lane//4 + 8*(r%2)
        lane = tidx % 32
        warp = tidx // 32
        return warp * 16 + (lane // 4) + 8 * (r % 2)

    @staticmethod
    def _col_coord(c, tidx):
        lane = tidx % 32
        return (lane % 4) * 2 + (c % 2) + 8 * (c // 2)

    @cute.jit
    def _softmax_rescale(
        self, acc_S, acc_O, row_max, row_sum, scale_log2, is_first):
        # 注：scale_log2 已在扫描前折叠进 Q（见 kernel 中 [优化3]），
        # 因此 acc_S 已是 log2 域分数，这里不再逐元素乘 scale。
        acc_S_mn = self._acc_mn_view(acc_S)
        acc_O_mn = self._acc_mn_view(acc_O)
        for r in cutlass.range_constexpr(cute.size(acc_S_mn.shape[0])):
            # 本块的行最大值
            cur = -cutlass.Float32.inf
            for c in cutlass.range_constexpr(cute.size(acc_S_mn.shape[1])):
                cur = max(cur, acc_S_mn[r, c])
            cur = self._quad_reduce_max(cur)
            new_max = max(row_max[r], cur)

            correction = cute.math.exp2(row_max[r] - new_max, fastmath=True)
            s = 0.0
            for c in cutlass.range_constexpr(cute.size(acc_S_mn.shape[1])):
                p = cute.math.exp2(acc_S_mn[r, c] - new_max, fastmath=True)
                acc_S_mn[r, c] = p
                s += p
            if not is_first:
                row_sum[r] = row_sum[r] * correction + s
                for c in cutlass.range_constexpr(cute.size(acc_O_mn.shape[1])):
                    acc_O_mn[r, c] = acc_O_mn[r, c] * correction
            else:
                row_sum[r] = s
            row_max[r] = new_max

    @staticmethod
    def _quad_reduce_max(val):
        val = max(val, cute.arch.shuffle_sync_bfly(val, offset=2, mask=-1, mask_and_clamp=31))
        val = max(val, cute.arch.shuffle_sync_bfly(val, offset=1, mask=-1, mask_and_clamp=31))
        return val

    @staticmethod
    def _quad_reduce_sum(val):
        val = val + cute.arch.shuffle_sync_bfly(val, offset=2, mask=-1, mask_and_clamp=31)
        val = val + cute.arch.shuffle_sync_bfly(val, offset=1, mask=-1, mask_and_clamp=31)
        return val

    @cute.jit
    def _normalize(self, acc_O, row_sum):
        acc_O_mn = self._acc_mn_view(acc_O)
        for r in cutlass.range_constexpr(cute.size(row_sum)):
            rs = self._quad_reduce_sum(row_sum[r])
            # 运行时值不能用三元表达式(会触发 __bool__)，改用动态 if/else
            if (rs == 0.0) | (rs != rs):
                inv = cutlass.Float32(1.0)
            else:
                inv = cute.arch.rcp_approx(rs)
            for c in cutlass.range_constexpr(cute.size(acc_O_mn.shape[1])):
                acc_O_mn[r, c] = acc_O_mn[r, c] * inv

    @cute.jit
    def _write_output(
        self, acc_O, mO_bh, m_block, sO_layout, gmem_tiled_copy_O,
        thr_mma, tidx, seqlen_q, storage, sQ_layout):
        # 先经 smem 暂存 O，再合并写回 gmem
        sO = storage.sQ.get_tensor(sO_layout)  # 复用 Q 的 smem 区域
        cute.arch.barrier()
        rO = cute.make_rmem_tensor(acc_O.shape, self._dtype)
        rO.store(acc_O.load().to(self._dtype))
        tOsO = thr_mma.partition_C(sO)
        tOsO.store(rO.load())
        cute.arch.barrier()

        gO = cute.local_tile(
            mO_bh, (self._m_block_size, self._head_dim_padded), (m_block, 0))
        gmem_thr_copy = gmem_tiled_copy_O.get_slice(tidx)
        tOsO2 = gmem_thr_copy.partition_S(sO)
        tOgO = gmem_thr_copy.partition_D(gO)
        cute.copy(gmem_tiled_copy_O, tOsO2, tOgO)



class FlashAttentionDecode:
    

    def __init__(
        self,
        head_dim: int,
        q_len: int,                 # 每个 q head 的 query 长度（Python int）
        gqa_group: int,             # H // H_kv（Python int，用于 range_constexpr）
        num_splits: int,
        split_size: int,
        n_block_size: int = 64,
        num_threads: int = 128,
        is_causal: bool = True,
    ):
        self._head_dim = head_dim
        self._head_dim_padded = int((head_dim + 7) // 8 * 8)
        self._q_len = int(q_len)          # 每个 q head 的 query 行数
        self._gqa_group = int(gqa_group)  # 组内 q head 数（Python int）
        # 组融合后，一个 block 处理的“行”数 = 组内 q head 数 * 每 head query 行数
        self._rows = self._gqa_group * self._q_len
        self._num_splits = num_splits
        self._split_size = split_size
        self._n_block_size = n_block_size
        self._num_threads = num_threads
        self._is_causal = is_causal

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,        # [B, H, Sq, D]
        mK: cute.Tensor,        # [B, H_kv, Sk, D]
        mV: cute.Tensor,        # [B, H_kv, Sk, D]
        mO: cute.Tensor,        # [B, H, Sq, D]
        mOpart: cute.Tensor,    # [B, H, num_splits, Sq, D]  (fp32)
        mLpart: cute.Tensor,    # [B, H, num_splits, Sq]      (fp32, base-2 lse)
        softmax_scale: cutlass.Float32,
        stream,
    ):
        self._dtype = mQ.element_type
        H = cute.size(mQ.shape[1])
        H_kv = cute.size(mK.shape[1])
        # 注：gqa_group / q_len 已在构造函数中以 Python int 传入（供 range_constexpr）。
        # seqlen_k 是动态值(cute.size)，不能在 host 算好再传进 kernel(会违反
        # region isolation)，改在 split_kernel 内部从 mK 参数重新计算。
        self._num_warps = self._num_threads // 32
        # 每个 lane 负责的 head-dim 元素个数（连续），保证向量化 + 合并访存
        self._vec = (self._head_dim + 31) // 32
        softmax_scale_log2 = softmax_scale * LOG2_E

        # split kernel grid: (split, batch, kv_head) —— 按 KV head 划分，组内共享 K/V
        grid_split = (self._num_splits, cute.size(mQ.shape[0]), H_kv)
        self.split_kernel(
            mQ, mK, mV, mOpart, mLpart, softmax_scale_log2,
        ).launch(
            grid=grid_split, block=[self._num_threads, 1, 1], stream=stream)

        # reduction grid: (batch, head) —— 仍按每个 q head 归并各 split
        grid_red = (cute.size(mQ.shape[0]), H, 1)
        self.reduce_kernel(
            mOpart, mLpart, mO,
        ).launch(
            grid=grid_red, block=[self._num_threads, 1, 1], stream=stream)

   
    @cute.kernel
    def split_kernel(
        self, mQ, mK, mV, mOpart, mLpart, scale_log2: cutlass.Float32):
        tidx, _, _ = cute.arch.thread_idx()
        split, batch, head = cute.arch.block_idx()
        kv_head = head // self._gqa_group

        D = self._head_dim
        VEC = self._vec
        NW = self._num_warps
        warp_id = tidx // 32
        lane = tidx % 32

        # seqlen_k 必须在 kernel 区域内从参数 mK 计算，不能引用 host 端的动态值
        seqlen_k = cute.size(mK.shape[2])

        kv_start = split * self._split_size
        kv_end = min(kv_start + self._split_size, seqlen_k)

        mQ_bh = mQ[batch, head, None, None]      # [Sq, D]
        mK_bh = mK[batch, kv_head, None, None]   # [Sk, D]
        mV_bh = mV[batch, kv_head, None, None]

        # ---- 本 lane 的 Q 元素载入寄存器（跨 key 复用）----
        # 把 scale_log2 一次性折叠进 Q，dot(q,k) 直接得到 log2 域分数，
        # 省去每个 key 的一次缩放乘法。
        qreg = cute.make_rmem_tensor((self._q_len, VEC), cutlass.Float32)
        m_i = cute.make_rmem_tensor((self._q_len,), cutlass.Float32)
        l_i = cute.make_rmem_tensor((self._q_len,), cutlass.Float32)
        acc = cute.make_rmem_tensor((self._q_len, VEC), cutlass.Float32)
        for i in cutlass.range_constexpr(self._q_len):
            m_i[i] = -cutlass.Float32.inf
            l_i[i] = 0.0
            for e in cutlass.range_constexpr(VEC):
                gd = lane * VEC + e
                # gd 为运行时值，不能用三元表达式做边界判断，改用动态 if/else
                if gd < D:
                    qreg[i, e] = cutlass.Float32(mQ_bh[i, gd]) * scale_log2
                else:
                    qreg[i, e] = cutlass.Float32(0.0)
                acc[i, e] = 0.0

        q_pos_base = seqlen_k - self._q_len  # 第一个 q 行的全局位置

        # ---- 本 warp 处理 keys: kv_start+warp_id, +NW, +2NW, ... ----
        k = kv_start + warp_id
        while k < kv_end:
            for i in cutlass.range_constexpr(self._q_len):
                # dot(q_i, k_k)：每 lane 先算自己 VEC 个元素，再 warp 归约
                partial = cutlass.Float32(0.0)
                for e in cutlass.range_constexpr(VEC):
                    gd = lane * VEC + e
                    if gd < D:
                        kv = cutlass.Float32(mK_bh[k, gd])
                    else:
                        kv = cutlass.Float32(0.0)
                    partial += qreg[i, e] * kv
                dot = self._warp_reduce_sum(partial)
                s = dot  # scale_log2 已折叠进 qreg

                # 用“正向条件” in_range 取代对运行时布尔做 Python 的 not
                in_range = True
                if cutlass.const_expr(self._is_causal):
                    in_range = k <= (q_pos_base + i)
                if in_range:
                    new_max = max(m_i[i], s)
                    corr = cute.math.exp2(m_i[i] - new_max, fastmath=True)
                    p = cute.math.exp2(s - new_max, fastmath=True)
                    l_i[i] = l_i[i] * corr + p
                    for e in cutlass.range_constexpr(VEC):
                        gd = lane * VEC + e
                        if gd < D:
                            vv = cutlass.Float32(mV_bh[k, gd])
                        else:
                            vv = cutlass.Float32(0.0)
                        acc[i, e] = acc[i, e] * corr + p * vv
                    m_i[i] = new_max
            k += NW

        # ---- 跨 warp 合并 (m, l, acc) via shared memory ----
        smem = utils.SmemAllocator()
        sm_m = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((NW, self._q_len),
                                              stride=(self._q_len, 1)), 16)
        sm_l = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((NW, self._q_len),
                                              stride=(self._q_len, 1)), 16)
        sacc = smem.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((NW, self._q_len, 32, VEC),
                             stride=(self._q_len * 32 * VEC, 32 * VEC, VEC, 1)), 16)

        cute.arch.barrier()
        for i in cutlass.range_constexpr(self._q_len):
            if lane == 0:
                sm_m[warp_id, i] = m_i[i]
                sm_l[warp_id, i] = l_i[i]
            for e in cutlass.range_constexpr(VEC):
                sacc[warp_id, i, lane, e] = acc[i, e]
        cute.arch.barrier()

        # 仅 warp 0 负责合并并写出（其余 warp 数据相同，避免重复写）
        mOp_bh = mOpart[batch, head, split, None, None]  # [Sq, D]
        mLp_bh = mLpart[batch, head, split, None]        # [Sq]
        if warp_id == 0:
            for i in cutlass.range_constexpr(self._q_len):
                mg = -cutlass.Float32.inf
                for w in cutlass.range_constexpr(NW):
                    mg = max(mg, sm_m[w, i])

                lg = cutlass.Float32(0.0)
                og = cute.make_rmem_tensor((VEC,), cutlass.Float32)
                for e in cutlass.range_constexpr(VEC):
                    og[e] = 0.0

                valid = mg > -cutlass.Float32.inf
                if valid:
                    for w in cutlass.range_constexpr(NW):
                        wgt = cute.math.exp2(sm_m[w, i] - mg, fastmath=True)
                        lg += sm_l[w, i] * wgt
                        for e in cutlass.range_constexpr(VEC):
                            og[e] += sacc[w, i, lane, e] * wgt

                if (lg == 0.0) | (lg != lg):
                    inv = cutlass.Float32(1.0)
                else:
                    inv = cute.arch.rcp_approx(lg)
                for e in cutlass.range_constexpr(VEC):
                    gd = lane * VEC + e
                    if gd < D:
                        mOp_bh[i, gd] = og[e] * inv  # 已归一化的部分结果
                if lane == 0:
                    if valid & (lg > 0.0):
                        mLp_bh[i] = mg + cute.math.log2(lg, fastmath=True)
                    else:
                        mLp_bh[i] = -cutlass.Float32.inf

    
    @cute.kernel
    def reduce_kernel(self, mOpart, mLpart, mO):
        tidx, _, _ = cute.arch.thread_idx()
        batch, head, _ = cute.arch.block_idx()
        D = self._head_dim

        mLp_bh = mLpart[batch, head, None, None]        # [num_splits, Sq]
        mOp_bh = mOpart[batch, head, None, None, None]  # [num_splits, Sq, D]
        mO_bh = mO[batch, head, None, None]             # [Sq, D]

        for i in cutlass.range_constexpr(self._q_len):
            # 各 split 的全局最大值（base-2 lse）
            gmax = -cutlass.Float32.inf
            for s in cutlass.range_constexpr(self._num_splits):
                gmax = max(gmax, mLp_bh[s, i])
            denom = cutlass.Float32(0.0)
            valid = gmax > -cutlass.Float32.inf
            if valid:
                for s in cutlass.range_constexpr(self._num_splits):
                    denom += cute.math.exp2(mLp_bh[s, i] - gmax, fastmath=True)
            if (denom == 0.0) | (denom != denom):
                inv = cutlass.Float32(1.0)
            else:
                inv = cute.arch.rcp_approx(denom)

            d = tidx
            if d < D:
                out = cutlass.Float32(0.0)
                if valid:
                    for s in cutlass.range_constexpr(self._num_splits):
                        w = cute.math.exp2(mLp_bh[s, i] - gmax, fastmath=True)
                        out += mOp_bh[s, i, d] * w
                mO_bh[i, d] = (out * inv).to(self._dtype)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _warp_reduce_sum(val):
        # 32 lane butterfly 全归约，得到完整 D 维点积（正确覆盖 4 个 sub-lane 组）
        val = val + cute.arch.shuffle_sync_bfly(val, offset=16, mask=-1, mask_and_clamp=31)
        val = val + cute.arch.shuffle_sync_bfly(val, offset=8, mask=-1, mask_and_clamp=31)
        val = val + cute.arch.shuffle_sync_bfly(val, offset=4, mask=-1, mask_and_clamp=31)
        val = val + cute.arch.shuffle_sync_bfly(val, offset=2, mask=-1, mask_and_clamp=31)
        val = val + cute.arch.shuffle_sync_bfly(val, offset=1, mask=-1, mask_and_clamp=31)
        return val



def _torch_to_cute(t):
    """把 contiguous 的 [B, H, S, D] torch 张量包装成 CuTe 张量（D 维连续）。"""
    return (
        from_dlpack(t, assumed_align=16)
        .mark_layout_dynamic(leading_dim=3)  # D 连续（最后一维，stride 1）
    )


def _current_stream():
    ts = torch.cuda.current_stream()
    if _HAS_CUDA_PY:
        return cuda.CUstream(ts.cuda_stream)
    return ts.cuda_stream


def _next_pow2(x):
    return 1 << (max(1, x) - 1).bit_length()


def _run_prefill(q, k, v, causal, sm_scale):
    B, H, S, D = q.shape
    o = torch.empty_like(q)

    key = ("prefill", D, causal, q.dtype)
    if key not in _COMPILE_CACHE:
        op = FlashAttentionPrefill(
            head_dim=D, m_block_size=128, n_block_size=128,
            num_threads=128, is_causal=causal)
        mQ, mK, mV, mO = (_torch_to_cute(t) for t in (q, k, v, o))
        compiled = cute.compile(
            op, mQ, mK, mV, mO, cutlass.Float32(sm_scale), _current_stream())
        _COMPILE_CACHE[key] = compiled

    mQ, mK, mV, mO = (_torch_to_cute(t) for t in (q, k, v, o))
    _COMPILE_CACHE[key](mQ, mK, mV, mO, cutlass.Float32(sm_scale), _current_stream())
    return o


def _run_decode(q, k, v, causal, sm_scale):
    B, H, Sq, D = q.shape
    Sk = k.shape[2]
    o = torch.empty_like(q)

    # 选择 split 数以填满 GPU：block 总数 = B*H*num_splits 需远大于 SM 数
    num_sm = torch.cuda.get_device_properties(q.device).multi_processor_count
    n_block = 64
    target = num_sm * 12
    num_splits = max(1, min((Sk + n_block - 1) // n_block,
                            (target + B * H - 1) // (B * H)))
    split_size = ((Sk + num_splits - 1) // num_splits + n_block - 1) // n_block * n_block
    num_splits = (Sk + split_size - 1) // split_size

    o_part = torch.empty((B, H, num_splits, Sq, D), device=q.device, dtype=torch.float32)
    l_part = torch.empty((B, H, num_splits, Sq), device=q.device, dtype=torch.float32)

    key = ("decode", D, Sq, num_splits, split_size, causal, q.dtype)
    op = FlashAttentionDecode(
        head_dim=D, q_len=Sq, num_splits=num_splits, split_size=split_size,
        n_block_size=n_block, num_threads=128,
        is_causal=causal)

    def wrap(t):
        return from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=t.dim() - 1)

    mQ, mK, mV, mO = (_torch_to_cute(t) for t in (q, k, v, o))
    mOp = wrap(o_part)
    mLp = wrap(l_part)

    if key not in _COMPILE_CACHE:
        _COMPILE_CACHE[key] = cute.compile(
            op, mQ, mK, mV, mO, mOp, mLp,
            cutlass.Float32(sm_scale), _current_stream())
    _COMPILE_CACHE[key](
        mQ, mK, mV, mO, mOp, mLp, cutlass.Float32(sm_scale), _current_stream())
    return o


def attention(q, k, v, causal=True, sm_scale=None):
    """FlashAttention (CuTe DSL)。支持 GQA、causal，自动分派 prefill / decode。

    Args:
        q: [B, H,    q_len,  D]
        k: [B, H_kv, kv_len, D]
        v: [B, H_kv, kv_len, D]
        causal: 是否施加因果掩码。
        sm_scale: softmax 缩放；默认 1/sqrt(D)。
    Returns:
        o: [B, H, q_len, D]，dtype 与 q 相同。
    """
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4
    B, H, q_len, D = q.shape
    Bk, H_kv, kv_len, Dk = k.shape
    assert D == Dk == v.shape[-1], "head_dim mismatch"
    assert H % H_kv == 0, "H must be divisible by H_kv (GQA)"
    assert q.is_cuda, "CuTe DSL FlashAttention requires CUDA tensors"
    assert q.dtype in (torch.float16, torch.bfloat16), "use fp16/bf16"

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    q, k, v = (t.contiguous() for t in (q, k, v))

    if q_len <= _DECODE_QLEN_THRESHOLD and q_len < kv_len:
        return _run_decode(q, k, v, causal, float(sm_scale))
    return _run_prefill(q, k, v, causal, float(sm_scale))


register("yaojx_flash_attention(CuTe DSL)", attention)
