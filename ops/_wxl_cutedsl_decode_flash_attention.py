# =============================================================================
# 高性能 Decode Attention — CuTeDSL Split-KV 实现 (SM90 / H20)
# =============================================================================
#
# 目标: H20 (4.0 TB/s HBM3), 达到 3.5 TB/s 有效带宽 (~87.5% utilization)
# 场景: B=1, Hq=64, Hkv=8, D=128, q_len=1, kv_len=131072, bf16
# 数据量: K+V = 2 × 8 × 131072 × 128 × 2B = 512 MB
# 目标延迟: 512MB / 3.5TB/s ≈ 146 μs
#
# 设计原理:
#   Decode 是纯访存密集型——瓶颈是 KV cache 从 HBM 的读取。
#   计算强度极低 (FLOP/Byte ≈ 0.125)，MMA 计算完全被 TMA 搬运掩盖。
#
# 核心策略:
#   1. "双流交错 WGMMA" — GEMM1(S=K·Q^T) 和 GEMM2(O=V·P^T) 的 K-tiles 交错到
#      同一环形缓冲中，利用 GEMM1 计算时间窗口完成 V 的 TMA 预取，消除 V 加载气泡
#   2. "自适应动态分裂" — 基于 SM 数量和 kv_len 动态选择 split 数，确保每个 CTA
#      处理的 tile 数量一致（±1），彻底消除 tail effect
#   3. "GQA 寄存器复用" — 8 个 Q heads 打包为 WGMMA M=16 (pad 8→16)，Q 驻留 smem
#      全程不换出，KV 各 tile 只读一次
#   4. "Warp-compact 角色分工" — 160 线程 (1 producer warp + 1 consumer WG)，
#      减少同步开销和 smem 压力
#   5. "log2 域在线 softmax" — 利用 exp2/log2 硬件指令，比 exp/ln 快 ~15%
#
# =============================================================================

import math
import threading
from dataclasses import dataclass
from typing import Optional

import torch

# CuTeDSL imports
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import cutlass.pipeline as pipeline
from cutlass.cutlass_dsl import Boolean, if_generate
from cutlass.pipeline import pipeline_init_wait
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute.runtime import from_dlpack


# =============================================================================
# 常量与配置
# =============================================================================
BF16 = cutlass.BFloat16
F32 = cutlass.Float32
LOG2E = math.log2(math.e)  # ≈ 1.4427

# 默认 kernel 参数 (H20 优化)
GROUP_SIZE = 8       # GQA ratio: 64/8
HEAD_DIM = 128
TILE_N = 192         # 每次主循环处理的 KV 行数 (偏大减少 softmax 同步次数)
NUM_STAGES_K = 2     # K 的 TMA 环形缓冲级数
NUM_STAGES_V = 1     # V 的环形级数 (K 计算期间 V 延迟预取)
NUM_THREADS = 160    # 128 consumer + 32 producer
DEFAULT_SPLITS = 48  # 初始 split 数 (会动态调整)


# =============================================================================
# 流水线同步原语: 轻量级 TMA 环形缓冲
# =============================================================================
# 针对 "1 个 producer warp + 1 个 consumer warpgroup" 的非对称配置，
# 使用最小化的 barrier arrive 计数避免死锁。
@dataclass(frozen=True)
class _AsyncTmaPipeline(pipeline.PipelineAsync):
    """单 producer warp / 单 consumer warpgroup 的 TMA 环形流水线。"""

    @staticmethod
    def create(bar_ptr, stages, prod_group, cons_group, tx_bytes, init_wait=True):
        full = pipeline.PipelineAsync._make_sync_object(
            bar_ptr.align(min_align=8), stages,
            (pipeline.PipelineOp.TmaLoad, prod_group), tx_bytes)
        empty = pipeline.PipelineAsync._make_sync_object(
            bar_ptr.align(min_align=8) + stages, stages,
            (pipeline.PipelineOp.AsyncThread, cons_group))
        if cutlass.const_expr(init_wait):
            pipeline_init_wait()
        return _AsyncTmaPipeline(full, empty, stages, None, None)

    def producer_acquire(self, state, token: Optional[Boolean] = None):
        """Producer 等待 consumer 释放 stage。"""
        if_generate(token is None or token == 0,
                    lambda: self.sync_object_empty.wait(state.index, state.phase))
        self.sync_object_full.arrive(state.index, self.producer_mask)

    def producer_commit(self, state):
        """TMA commit — 对于 TMA load, hardware 自动 arrive, 此处为空。"""
        pass

    def consumer_release(self, state):
        """Consumer 释放一个 stage，只需 1 个线程 arrive。"""
        if_generate(cute.arch.thread_idx()[0] % 128 == 0,
                    lambda: self.sync_object_empty.arrive(state.index, self.consumer_mask))


# =============================================================================
# 辅助函数
# =============================================================================
@cute.jit
def _fire_wgmma(mma, acc, frag_a, frag_b, accumulate):
    """发射 WGMMA 指令序列，沿 K 维累加。"""
    warpgroup.fence()
    atom = cute.make_mma_atom(mma.op)
    atom.set(warpgroup.Field.ACCUMULATE, accumulate)
    for kk in cutlass.range_constexpr(cute.size(frag_a.shape[2])):
        cute.gemm(atom, acc, frag_a[None, None, kk], frag_b[None, None, kk], acc)
        atom.set(warpgroup.Field.ACCUMULATE, True)
    warpgroup.commit_group()
    warpgroup.wait_group(0)


def _acc_to_mn_view(acc):
    """将 WGMMA 累加器的物理布局重排为逻辑 (M, N) 二维视图。"""
    base = cute.make_layout(acc.shape)
    mn = cute.make_layout(
        (
            (base.shape[0][1], base.shape[1]),
            (base.shape[0][0], *base.shape[0][2:], base.shape[2]),
            *base.shape[3:],
        ),
        stride=(
            (base.stride[0][1], base.stride[1]),
            (base.stride[0][0], *base.stride[0][2:], base.stride[2]),
            *base.stride[3:],
        ),
    )
    return cute.make_tensor(acc.iterator, cute.composition(acc.layout, mn))


def _transpose_01(t):
    """交换 tensor 前两维的视图（不搬数据）。"""
    shp = (t.shape[1], t.shape[0], *t.shape[2:])
    order = (1, 0, *range(2, cute.rank(t)))
    return cute.composition(t, cute.make_ordered_layout(shp, order=order))


def _compute_splits(kv_len, tile_n, target_splits=DEFAULT_SPLITS):
    """自适应计算 split 数量，确保每个 split 的 tile 数量均衡 (±1)。
    
    策略: 先确定每个 split 处理的 tile 数 (tiles_per_split)，
    然后反推需要多少 splits 刚好覆盖所有 tiles。
    这避免了 "最后一个 split 特别小" 的 tail effect。
    """
    total_tiles = (kv_len + tile_n - 1) // tile_n
    tiles_per = max(1, (total_tiles + target_splits - 1) // target_splits)
    actual_splits = (total_tiles + tiles_per - 1) // tiles_per
    return actual_splits


# =============================================================================
# 主 Kernel 类: DecodeAttentionSplitKV
# =============================================================================
class DecodeAttentionSplitKV:
    """
    双流交错 WGMMA + TMA 流水的 Split-KV Flash-Decoding。
    
    数据流:
        HBM ─TMA─→ SMEM (K ring + V ring)
                      ↓ WGMMA descriptor
              Tensor Core: GEMM1 S^T[N,G] = K[N,D] · Q^T[D,G]
                      ↓ registers
              Online Softmax (butterfly + smem cross-warp reduce)
                      ↓ smem (P^T tile)
              Tensor Core: GEMM2 O^T[D,G] = V^T[D,N] · P^T[N,G]
                      ↓ registers → global memory
              O_partial + LSE (per-split)
                      ↓ combine kernel
              Final Output O (bf16)
    """

    def __init__(
        self,
        q_heads: int,
        kv_heads: int,
        kv_len: int,
        batch: int,
        num_splits: int = DEFAULT_SPLITS,
        tile_n: int = TILE_N,
        k_stages: int = NUM_STAGES_K,
        v_stages: int = NUM_STAGES_V,
        num_workers: int = None,
        sm_scale: float = None,
    ):
        assert q_heads == kv_heads * GROUP_SIZE
        assert kv_len % tile_n == 0, f"kv_len ({kv_len}) must be divisible by tile_n ({tile_n})"

        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.kv_len = kv_len
        self.batch = batch
        self.num_splits = num_splits
        self.tile_n = tile_n
        self.k_stages = k_stages
        self.v_stages = v_stages
        self.num_workers = num_workers
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(HEAD_DIM)
        self.scale_log2 = sm_scale * LOG2E

    # -------------------------------------------------------------------------
    # Host-side launch
    # -------------------------------------------------------------------------
    @cute.jit
    def __call__(self, mQ, mK, mV, mOpart, mLSE, mO, stream: cuda.CUstream):
        self.dt = mQ.element_type

        # 逻辑视图构建:
        # Q: (H_q, D, B)  — 从 (B, H_q, D) 重排
        # K: (S, D, (H_kv, B))  — 从 (B, S, H_kv, D) 重排
        # V: (S, D, (H_kv, B))
        # O: (H_q, D, B)
        Q = cute.make_tensor(mQ.iterator, cute.make_layout(
            (mQ.shape[1], mQ.shape[2], mQ.shape[0]),
            stride=(mQ.stride[1], mQ.stride[2], mQ.stride[0])))
        O = cute.make_tensor(mO.iterator, cute.make_layout(
            (mO.shape[1], mO.shape[2], mO.shape[0]),
            stride=(mO.stride[1], mO.stride[2], mO.stride[0])))
        # mK shape = (B, S, Hkv, D) → K = (S, D, (Hkv, B))
        K = cute.make_tensor(mK.iterator, cute.make_layout(
            (mK.shape[1], mK.shape[3], (mK.shape[2], mK.shape[0])),
            stride=(mK.stride[1], mK.stride[3], (mK.stride[2], mK.stride[0]))))
        V = cute.make_tensor(mV.iterator, cute.make_layout(
            (mV.shape[1], mV.shape[3], (mV.shape[2], mV.shape[0])),
            stride=(mV.stride[1], mV.stride[3], (mV.stride[2], mV.stride[0]))))

        # SMEM layout atoms (swizzled for bank-conflict-free access)
        atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(utils.LayoutEnum.ROW_MAJOR, self.dt, HEAD_DIM),
            self.dt)
        kL = cute.tile_to_shape(atom, (self.tile_n, HEAD_DIM, self.k_stages), (0, 1, 2))
        vL = cute.tile_to_shape(atom, (self.tile_n, HEAD_DIM, self.v_stages), (0, 1, 2))
        qL = cute.tile_to_shape(atom, (GROUP_SIZE, HEAD_DIM), (0, 1))
        pL = cute.tile_to_shape(atom, (GROUP_SIZE, self.tile_n), (0, 1))

        # SMEM struct definition
        sS_size = self.tile_n * GROUP_SIZE  # f32 buffer for softmax scores
        @cute.struct
        class SmemLayout:
            k_barriers: cute.struct.MemRange[cutlass.Int64, self.k_stages * 2]
            v_barriers: cute.struct.MemRange[cutlass.Int64, self.v_stages * 2]
            reduce_buf: cute.struct.MemRange[F32, 2 * 4 * GROUP_SIZE]  # cross-warp max/sum
            sS_buf: cute.struct.Align[cute.struct.MemRange[F32, sS_size], 1024]  # softmax score buf
            sQ: cute.struct.Align[cute.struct.MemRange[self.dt, cute.cosize(qL)], 1024]
            sP: cute.struct.Align[cute.struct.MemRange[self.dt, cute.cosize(pL)], 1024]
            sK: cute.struct.Align[cute.struct.MemRange[self.dt, cute.cosize(kL)], 1024]
            sV: cute.struct.Align[cute.struct.MemRange[self.dt, cute.cosize(vL)], 1024]

        # TMA descriptors
        ka, kt = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), K,
            cute.select(kL, mode=[0, 1]), (self.tile_n, HEAD_DIM))
        va, vt = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), V,
            cute.select(vL, mode=[0, 1]), (self.tile_n, HEAD_DIM))

        # WGMMA configs
        # GEMM1: S^T[N,G] = K[N,D] · Q^T[D,G] → A: K-major, B: K-major
        mma_gemm1 = sm90_utils.make_trivial_tiled_mma(
            self.dt, self.dt,
            warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K,
            F32, (1, 1, 1), (64, GROUP_SIZE))
        # GEMM2: O^T[D,G] = V^T[D,N] · P^T[N,G] → A: MN-major, B: K-major
        mma_gemm2 = sm90_utils.make_trivial_tiled_mma(
            self.dt, self.dt,
            warpgroup.OperandMajorMode.MN, warpgroup.OperandMajorMode.K,
            F32, (1, 1, 1), (64, GROUP_SIZE))

        # Launch split kernel (persistent grid)
        self.split_kernel(
            Q, ka, kt, va, vt, mOpart, mLSE, F32(self.scale_log2),
            qL, kL, vL, pL, mma_gemm1, mma_gemm2, SmemLayout,
        ).launch(
            grid=(self.num_workers, 1, 1),
            block=[NUM_THREADS, 1, 1],
            smem=SmemLayout.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=True,
        )

        # Launch combine kernel (PDL: overlaps with split tail)
        self.combine_kernel(mOpart, mLSE, O).launch(
            grid=(self.q_heads, self.batch, 1),
            block=[128, 1, 1],
            smem=self.num_splits * 4,
            stream=stream,
            use_pdl=True,
        )

    # -------------------------------------------------------------------------
    # Split Kernel: 每个 CTA 循环处理 (split, kv_head, batch) 任务
    # -------------------------------------------------------------------------
    @cute.kernel
    def split_kernel(self, mQ, ka, mK, va, mV, mOpart, mLSE, scale_log2,
                     qL, kL, vL, pL, mma_gemm1, mma_gemm2,
                     SmemLayout: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        warp_id = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        worker_id, _, _ = cute.arch.block_idx()

        # TMA 描述符预取
        if warp_id == 0:
            cpasync.prefetch_descriptor(ka)
            cpasync.prefetch_descriptor(va)

        # SMEM 分配
        sm = cutlass.utils.SmemAllocator()
        storage = sm.allocate(SmemLayout)

        # Pipeline 初始化
        prod_grp = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        cons_grp = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        tx_k = cute.size_in_bytes(self.dt, cute.select(kL, mode=[0, 1]))
        tx_v = cute.size_in_bytes(self.dt, cute.select(vL, mode=[0, 1]))
        pipe_k = _AsyncTmaPipeline.create(
            storage.k_barriers.data_ptr(), self.k_stages, prod_grp, cons_grp,
            tx_k, init_wait=False)
        pipe_v = _AsyncTmaPipeline.create(
            storage.v_barriers.data_ptr(), self.v_stages, prod_grp, cons_grp,
            tx_v)

        # SMEM tensor views
        sQ = storage.sQ.get_tensor(qL.outer, swizzle=qL.inner)
        sP = storage.sP.get_tensor(pL.outer, swizzle=pL.inner)
        sK = storage.sK.get_tensor(kL.outer, swizzle=kL.inner)
        sV = storage.sV.get_tensor(vL.outer, swizzle=vL.inner)
        sVt = _transpose_01(sV)  # V 转置视图
        # sS_buf: flat f32 buffer for softmax scores, shape (tile_n, GROUP_SIZE)
        sS_buf = cute.make_tensor(
            storage.sS_buf.data_ptr(),
            cute.make_layout((self.tile_n, GROUP_SIZE), stride=(GROUP_SIZE, 1)))

        # Work item 计算
        tiles_total = self.kv_len // self.tile_n
        num_items = self.num_splits * self.kv_heads * self.batch

        if warp_id == 4:
            # ============== Producer Warp (TMA 发射) ==============
            ps_k = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.k_stages)
            ps_v = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.v_stages)

            for item in cutlass.range(worker_id, num_items, self.num_workers, unroll=1):
                kvh = (item // self.batch) % self.kv_heads
                sp = item // (self.batch * self.kv_heads)
                bt = item % self.batch

                # 均衡分割: [t0, t1) 为本 split 的 tile 范围
                t0 = sp * tiles_total // self.num_splits
                t1 = (sp + 1) * tiles_total // self.num_splits

                gK = cute.local_tile(mK[None, None, (kvh, bt)],
                                     (self.tile_n, HEAD_DIM), (None, 0))
                gV = cute.local_tile(mV[None, None, (kvh, bt)],
                                     (self.tile_n, HEAD_DIM), (None, 0))
                tKs, tKg = cpasync.tma_partition(
                    ka, 0, cute.make_layout(1),
                    cute.group_modes(sK, 0, 2), cute.group_modes(gK, 0, 2))
                tVs, tVg = cpasync.tma_partition(
                    va, 0, cute.make_layout(1),
                    cute.group_modes(sV, 0, 2), cute.group_modes(gV, 0, 2))

                for i in cutlass.range(t1 - t0, unroll=1):
                    # K TMA
                    pipe_k.producer_acquire(ps_k)
                    cute.copy(ka, tKg[None, t0 + i], tKs[None, ps_k.index],
                              tma_bar_ptr=pipe_k.producer_get_barrier(ps_k))
                    pipe_k.producer_commit(ps_k)
                    ps_k.advance()
                    # V TMA (交错发射)
                    pipe_v.producer_acquire(ps_v)
                    cute.copy(va, tVg[None, t0 + i], tVs[None, ps_v.index],
                              tma_bar_ptr=pipe_v.producer_get_barrier(ps_v))
                    pipe_v.producer_commit(ps_v)
                    ps_v.advance()
        else:
            # ============== Consumer Warpgroup (WGMMA 计算) ==============
            t2 = tidx  # consumer thread id (0~127)
            lane = t2 % 32
            warp_in_wg = t2 // 32

            # MMA slice partitions
            sl_g1 = mma_gemm1.get_slice(0)
            sl_g2 = mma_gemm2.get_slice(0)
            thr_g1 = mma_gemm1.get_slice(t2)
            thr_g2 = mma_gemm2.get_slice(t2)

            # Fragment references for WGMMA descriptors
            rK = mma_gemm1.make_fragment_A(sl_g1.partition_A(sK))
            rQ = mma_gemm1.make_fragment_B(sl_g1.partition_B(sQ))
            rVt = mma_gemm2.make_fragment_A(sl_g2.partition_A(sVt))
            rP = mma_gemm2.make_fragment_B(sl_g2.partition_B(sP))

            # Accumulator shapes
            accS_shape = mma_gemm1.partition_shape_C((self.tile_n, GROUP_SIZE))
            accO_shape = mma_gemm2.partition_shape_C((HEAD_DIM, GROUP_SIZE))
            accO = cute.make_rmem_tensor(accO_shape, F32)
            accO_mn = _acc_to_mn_view(accO)

            # Identity tensors for coordinate mapping
            idS = cute.make_identity_tensor((self.tile_n, GROUP_SIZE))
            cS = _acc_to_mn_view(thr_g1.partition_C(idS))
            idO = cute.make_identity_tensor((HEAD_DIM, GROUP_SIZE))
            cO = _acc_to_mn_view(thr_g2.partition_C(idO))

            NR = cute.size(cS.shape[0])  # rows per thread in S
            NC = cute.size(cS.shape[1])  # cols per thread in S

            # Softmax state (per-column = per q_head)
            rmax = cute.make_rmem_tensor((NC,), F32)
            rsum = cute.make_rmem_tensor((NC,), F32)
            xchg = storage.reduce_buf.data_ptr()  # cross-warp exchange buffer

            # Q copy setup (register → smem)
            qcopy = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.dt,
                                        num_bits_per_copy=128)
            qtc = cute.make_tiled_copy_tv(
                qcopy,
                cute.make_layout((GROUP_SIZE, 128 // GROUP_SIZE),
                                 stride=(128 // GROUP_SIZE, 1)),
                cute.make_layout((1, 128 // (128 // GROUP_SIZE))))
            qthr = qtc.get_slice(t2)

            # Pipeline consumer states
            cs_k = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.k_stages)
            cs_v = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.v_stages)

            for item in cutlass.range(worker_id, num_items, self.num_workers, unroll=1):
                kvh = (item // self.batch) % self.kv_heads
                sp = item // (self.batch * self.kv_heads)
                bt = item % self.batch

                t0 = sp * tiles_total // self.num_splits
                t1 = (sp + 1) * tiles_total // self.num_splits

                # Load Q for this kv_head group into smem
                gQ = cute.local_tile(mQ[None, None, bt], (GROUP_SIZE, HEAD_DIM), (kvh, 0))
                cute.copy(qtc, qthr.partition_S(gQ), qthr.partition_D(sQ))
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.barrier(barrier_id=1, number_of_threads=128)

                # Reset accumulators
                accO.fill(0.0)
                rmax.fill(-F32.inf)
                rsum.fill(0.0)

                # Main loop over tiles in this split
                for i in cutlass.range(t1 - t0, unroll=1):
                    # ---- GEMM1: S^T = K · Q^T ----
                    accS = cute.make_rmem_tensor(accS_shape, F32)
                    pipe_k.consumer_wait(cs_k)
                    _fire_wgmma(mma_gemm1, accS,
                                rK[None, None, None, cs_k.index], rQ, False)
                    pipe_k.consumer_release(cs_k)
                    cs_k.advance()
                    accS_mn = _acc_to_mn_view(accS)

                    # ---- Online Softmax ----
                    self._online_softmax(
                        accS_mn, accO_mn, rmax, rsum, cS, cO,
                        sP, sS_buf, scale_log2, NR, NC, self.tile_n // 128,
                        warp_in_wg, lane, xchg, t2)

                    # ---- GEMM2: O^T += V^T · P^T ----
                    pipe_v.consumer_wait(cs_v)
                    _fire_wgmma(mma_gemm2, accO,
                                rVt[None, None, None, cs_v.index], rP, True)
                    pipe_v.consumer_release(cs_v)
                    cs_v.advance()

                # ---- Epilogue: normalize and store ----
                self._write_partial(
                    accO_mn, rmax, rsum, cO, mOpart, mLSE,
                    sp, kvh, bt, NC, t2)

        # PDL: 通知依赖的 combine kernel 可以提前启动
        cute.arch.griddepcontrol_launch_dependents()

    # -------------------------------------------------------------------------
    # Online Softmax (log2 域) — SMEM-based safe implementation
    # -------------------------------------------------------------------------
    @cute.jit
    def _online_softmax(self, accS_mn, accO_mn, rmax, rsum, cS, cO,
                        sP, sS_buf, scale_log2, NR: cutlass.Constexpr,
                        NC: cutlass.Constexpr, ROWS_PER_THREAD: cutlass.Constexpr,
                        warp_in_wg, lane, xchg, t2):
        # 1) Scale scores and write to smem S buffer for column-wise reduce
        #    sS_buf shape: (tile_n, GROUP_SIZE) as f32
        for c in cutlass.range_constexpr(NC):
            for r in cutlass.range_constexpr(NR):
                score = accS_mn[r, c] * scale_log2
                accS_mn[r, c] = score
                # Write to smem at correct (row, col) position
                sS_buf[cS[r, c][0], cS[r, c][1]] = score
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.barrier(barrier_id=1, number_of_threads=128)

        # 2) Each thread reduces one or more GROUP columns from smem
        #    128 threads, 8 columns → each thread handles columns strided
        #    Thread t2 handles column (t2 % GROUP_SIZE) if t2 < GROUP_SIZE * (tile_n / stride)
        #    Simple approach: thread t2 reduces col = t2 % GROUP_SIZE over rows
        #    But we need ALL threads to know max for their own NC columns.
        #    Simpler: each of first 8 threads computes max/sum for one column,
        #    then broadcast via smem.
        col_max = cute.make_rmem_tensor((GROUP_SIZE,), F32)
        col_sum = cute.make_rmem_tensor((GROUP_SIZE,), F32)

        # Each thread participates in reducing all columns
        # Strategy: divide tile_n rows among 128 threads, each computes partial max/sum
        # Then reduce across threads via shuffle + smem
        local_max = cute.make_rmem_tensor((GROUP_SIZE,), F32)
        local_sum = cute.make_rmem_tensor((GROUP_SIZE,), F32)
        for g in cutlass.range_constexpr(GROUP_SIZE):
            local_max[g] = -F32.inf
        for g in cutlass.range_constexpr(GROUP_SIZE):
            local_sum[g] = F32(0.0)

        # Phase 1: each thread scans its assigned rows
        row_start = t2 * ROWS_PER_THREAD
        for ri in cutlass.range_constexpr(ROWS_PER_THREAD):
            row = row_start + ri
            for g in cutlass.range_constexpr(GROUP_SIZE):
                val = sS_buf[row, g]
                local_max[g] = cute.arch.fmax(local_max[g], val)

        # Phase 2: warp-level reduce max via shuffle
        for g in cutlass.range_constexpr(GROUP_SIZE):
            m = local_max[g]
            m = cute.arch.fmax(m, cute.arch.shuffle_sync_bfly(m, offset=1))
            m = cute.arch.fmax(m, cute.arch.shuffle_sync_bfly(m, offset=2))
            m = cute.arch.fmax(m, cute.arch.shuffle_sync_bfly(m, offset=4))
            m = cute.arch.fmax(m, cute.arch.shuffle_sync_bfly(m, offset=8))
            m = cute.arch.fmax(m, cute.arch.shuffle_sync_bfly(m, offset=16))
            local_max[g] = m

        # Phase 3: cross-warp reduce max via smem (4 warps)
        if lane == 0:
            for g in cutlass.range_constexpr(GROUP_SIZE):
                xchg[warp_in_wg * GROUP_SIZE + g] = local_max[g]
        cute.arch.barrier(barrier_id=1, number_of_threads=128)

        for g in cutlass.range_constexpr(GROUP_SIZE):
            gm = xchg[g]
            for w in cutlass.range_constexpr(1, 4):
                gm = cute.arch.fmax(gm, xchg[w * GROUP_SIZE + g])
            col_max[g] = gm

        # Phase 4: compute sum using global max
        for ri in cutlass.range_constexpr(ROWS_PER_THREAD):
            row = row_start + ri
            for g in cutlass.range_constexpr(GROUP_SIZE):
                val = sS_buf[row, g]
                local_sum[g] += cute.math.exp2(val - col_max[g], fastmath=True)

        # Phase 5: warp-level reduce sum
        for g in cutlass.range_constexpr(GROUP_SIZE):
            s = local_sum[g]
            s += cute.arch.shuffle_sync_bfly(s, offset=1)
            s += cute.arch.shuffle_sync_bfly(s, offset=2)
            s += cute.arch.shuffle_sync_bfly(s, offset=4)
            s += cute.arch.shuffle_sync_bfly(s, offset=8)
            s += cute.arch.shuffle_sync_bfly(s, offset=16)
            local_sum[g] = s

        # Phase 6: cross-warp reduce sum
        if lane == 0:
            for g in cutlass.range_constexpr(GROUP_SIZE):
                xchg[4 * GROUP_SIZE + warp_in_wg * GROUP_SIZE + g] = local_sum[g]
        cute.arch.barrier(barrier_id=1, number_of_threads=128)

        for g in cutlass.range_constexpr(GROUP_SIZE):
            gs = F32(0.0)
            for w in cutlass.range_constexpr(4):
                gs += xchg[4 * GROUP_SIZE + w * GROUP_SIZE + g]
            col_sum[g] = gs

        # Phase 7: update online softmax state & rescale O accumulator
        for c in cutlass.range_constexpr(NC):
            col = cS[0, c][1]
            new_max = cute.arch.fmax(rmax[c], col_max[col])
            alpha = cute.math.exp2(rmax[c] - new_max, fastmath=True)
            # correction factor for current block's sum
            block_sum_corrected = col_sum[col] * cute.math.exp2(col_max[col] - new_max, fastmath=True)
            rsum[c] = rsum[c] * alpha + block_sum_corrected
            rmax[c] = new_max
            # Rescale O accumulator
            for r in cutlass.range_constexpr(cute.size(accO_mn.shape[0])):
                accO_mn[r, c] = accO_mn[r, c] * alpha

        # Phase 8: write P to smem using updated global max (rmax)
        #   P[i] = exp2(score[i] - rmax) — probability relative to global max
        for c in cutlass.range_constexpr(NC):
            col = cS[0, c][1]
            for r in cutlass.range_constexpr(NR):
                p = cute.math.exp2(accS_mn[r, c] - rmax[c], fastmath=True)
                sP[col, cS[r, c][0]] = BF16(p)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.barrier(barrier_id=1, number_of_threads=128)

    # -------------------------------------------------------------------------
    # Epilogue: Normalize O, store partial results + LSE
    # -------------------------------------------------------------------------
    @cute.jit
    def _write_partial(self, accO_mn, rmax, rsum, cO, mOpart, mLSE,
                       sp, kvh, bt, NC: cutlass.Constexpr, t2):
        # Normalize O and store partial output + LSE
        for c in cutlass.range_constexpr(NC):
            col = cO[0, c][1]
            total = rsum[c]
            inv = 0.0 if total == 0.0 or total != total else cute.arch.rcp_approx(total)
            for r in cutlass.range_constexpr(cute.size(accO_mn.shape[0])):
                accO_mn[r, c] = accO_mn[r, c] * inv
                mOpart[sp, kvh, col, cO[r, c][0], bt] = BF16(accO_mn[r, c])
            # LSE = max + log2(sum) in log2 domain (combine uses exp2)
            lse = (-F32.inf if total == 0.0 or total != total
                   else rmax[c] + cute.math.log2(total, fastmath=True))
            # Multiple threads may write the same col — values are identical, safe redundant write
            if t2 < GROUP_SIZE:
                mLSE[sp, kvh, col, bt] = lse

    # -------------------------------------------------------------------------
    # Combine Kernel: 跨 splits 的 log-sum-exp 加权归约
    # -------------------------------------------------------------------------
    @cute.kernel
    def combine_kernel(self, mOpart, mLSE, mO):
        qh, bt, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        kvh = qh // GROUP_SIZE
        g = qh % GROUP_SIZE

        # 加载所有 splits 的 LSE 到 smem
        sm = cutlass.utils.SmemAllocator()
        lse_buf = sm.allocate_tensor(F32, cute.make_layout(self.num_splits), 16)

        cute.arch.griddepcontrol_wait()  # PDL: 等待 split kernel 完成

        for i in cutlass.range(tidx, self.num_splits, 128):
            lse_buf[i] = mLSE[i, kvh, g, bt]
        cute.arch.barrier()

        # 求全局 max
        m = -F32.inf
        for s in cutlass.range_constexpr(self.num_splits):
            m = cute.arch.fmax(m, lse_buf[s])

        # 计算权重
        den = F32(0.0)
        w = cute.make_rmem_tensor((self.num_splits,), F32)
        for s in cutlass.range_constexpr(self.num_splits):
            w[s] = cute.math.exp2(lse_buf[s] - m, fastmath=True)
            den += w[s]

        # 加权求和 O (每个线程处理 1 个 head_dim 元素)
        acc = F32(0.0)
        for s in cutlass.range_constexpr(self.num_splits):
            acc += w[s] * F32(mOpart[s, kvh, g, tidx, bt])

        # Normalize and store
        inv = 0.0 if den == 0.0 or den != den else cute.arch.rcp_approx(den)
        mO[qh, tidx, bt] = BF16(acc * inv)


# =============================================================================
# Python wrapper for the unified prefill/decode adapter
# =============================================================================
_COMPILE_LOCK = threading.Lock()
_BUFFER_LOCK = threading.Lock()
_VIEW_LOCK = threading.Lock()
_COMPILED_CACHE = {}
_BUFFER_CACHE = {}
_VIEW_CACHE = {}
_COMPILE_COUNT = 0

_EXPECTED_Q_SHAPE = (1, 64, 1, 128)
_EXPECTED_KV_SHAPE = (1, 8, 131072, 128)
_ALIGNMENT = 16


def _make_cute_tensor(t, assumed_align=_ALIGNMENT):
    """Convert a PyTorch tensor to the stride-dynamic view expected by WXL."""
    return (from_dlpack(t, assumed_align=assumed_align)
            .mark_layout_dynamic(leading_dim=t.dim() - 1)
            .mark_compact_shape_dynamic(
                mode=t.dim() - 1,
                stride_order=t.dim_order(),
                divisibility=128 // BF16.width))


def _normalize_sm_scale(sm_scale):
    scale = 1.0 / math.sqrt(HEAD_DIM) if sm_scale is None else float(sm_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("sm_scale must be finite and positive")
    return scale


def _validate_decode_inputs(q, k, v):
    if not all(isinstance(tensor, torch.Tensor) for tensor in (q, k, v)):
        raise TypeError("q, k, and v must be torch.Tensor instances")
    if not all(tensor.ndim == 4 for tensor in (q, k, v)):
        raise ValueError("q, k, and v must be rank-4 BHSD tensors")
    if tuple(q.shape) != _EXPECTED_Q_SHAPE:
        raise ValueError("expected q BHSD shape (1, 64, 1, 128)")
    if tuple(k.shape) != _EXPECTED_KV_SHAPE or tuple(v.shape) != _EXPECTED_KV_SHAPE:
        raise ValueError("expected k/v BHSD shape (1, 8, 131072, 128)")
    if not all(tensor.is_cuda for tensor in (q, k, v)):
        raise ValueError("q, k, and v must be CUDA tensors")
    if k.device != q.device or v.device != q.device:
        raise ValueError("q, k, and v must be on the same CUDA device")
    if torch.cuda.current_device() != q.device.index:
        raise RuntimeError("the current CUDA device must match q, k, and v")

    properties = torch.cuda.get_device_properties(q.device)
    if ((properties.major, properties.minor) != (9, 0)
            or "H20" not in properties.name.upper()):
        raise RuntimeError("WXL decode requires an NVIDIA H20 SM90a device")

    if not all(tensor.dtype == torch.bfloat16 for tensor in (q, k, v)):
        raise TypeError("q, k, and v must all use torch.bfloat16")
    if not all(tensor.is_contiguous() for tensor in (q, k, v)):
        raise ValueError("q, k, and v must be contiguous BHSD tensors")
    if any(tensor.data_ptr() % _ALIGNMENT for tensor in (q, k, v)):
        raise ValueError("q, k, and v must have at least 16-byte pointer alignment")


def _select_tile_n(kv_len):
    if kv_len % TILE_N == 0 and TILE_N % 128 == 0:
        return TILE_N
    for candidate in (256, 128):
        if kv_len % candidate == 0:
            return candidate
    raise ValueError("WXL decode requires kv_len divisible by 256 or 128")


def _get_buffers(device, stream_handle, num_splits):
    key = (device.index, stream_handle, num_splits, _EXPECTED_Q_SHAPE, _EXPECTED_KV_SHAPE)
    buffers = _BUFFER_CACHE.get(key)
    if buffers is None:
        with _BUFFER_LOCK:
            buffers = _BUFFER_CACHE.get(key)
            if buffers is None:
                buffers = (
                    torch.empty(
                        (num_splits, _EXPECTED_KV_SHAPE[1], GROUP_SIZE, HEAD_DIM, 1),
                        dtype=torch.bfloat16,
                        device=device,
                    ),
                    torch.empty(
                        (num_splits, _EXPECTED_KV_SHAPE[1], GROUP_SIZE, 1),
                        dtype=torch.float32,
                        device=device,
                    ),
                    torch.empty(
                        (_EXPECTED_Q_SHAPE[0], _EXPECTED_Q_SHAPE[1], HEAD_DIM),
                        dtype=torch.bfloat16,
                        device=device,
                    ),
                )
                _BUFFER_CACHE[key] = buffers
    return buffers


def _get_views(q, k, v, buffers, stream_handle):
    opart, lse, out = buffers
    q_dec = q[:, :, 0, :]
    k_bshd = k.transpose(1, 2)
    v_bshd = v.transpose(1, 2)
    key = (
        q.device.index,
        stream_handle,
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        opart.data_ptr(),
        lse.data_ptr(),
        out.data_ptr(),
    )
    cached = _VIEW_CACHE.get(key)
    if cached is None:
        with _VIEW_LOCK:
            cached = _VIEW_CACHE.get(key)
            if cached is None:
                views = (
                    _make_cute_tensor(q_dec),
                    _make_cute_tensor(k_bshd),
                    _make_cute_tensor(v_bshd),
                    from_dlpack(opart, assumed_align=_ALIGNMENT),
                    from_dlpack(lse, assumed_align=_ALIGNMENT),
                    _make_cute_tensor(out),
                )
                stream = cuda.CUstream(stream_handle)
                exporters = (q, k, v, q_dec, k_bshd, v_bshd, opart, lse, out)
                cached = (views, stream, exporters)
                _VIEW_CACHE[key] = cached
    return cached


def run_wxl_sm90_gqa_decode(q, k, v, sm_scale=None):
    """Run the cached WXL TMA/WGMMA split-KV decode on the current stream."""
    global _COMPILE_COUNT

    _validate_decode_inputs(q, k, v)
    scale = _normalize_sm_scale(sm_scale)
    tile_n = _select_tile_n(_EXPECTED_KV_SHAPE[2])
    num_splits = _compute_splits(
        _EXPECTED_KV_SHAPE[2], tile_n, target_splits=DEFAULT_SPLITS
    )
    properties = torch.cuda.get_device_properties(q.device)
    num_workers = properties.multi_processor_count
    torch_stream = torch.cuda.current_stream(device=q.device)
    stream_handle = int(torch_stream.cuda_stream)

    buffers = _get_buffers(q.device, stream_handle, num_splits)
    opart, lse, out = buffers
    views, stream, exporters = _get_views(q, k, v, buffers, stream_handle)
    q_cute, k_cute, v_cute, opart_cute, lse_cute, out_cute = views

    compile_key = (
        q.device.index,
        _EXPECTED_Q_SHAPE,
        _EXPECTED_KV_SHAPE,
        num_splits,
        tile_n,
        NUM_STAGES_K,
        NUM_STAGES_V,
        num_workers,
        scale.hex(),
    )
    compiled = _COMPILED_CACHE.get(compile_key)
    if compiled is None:
        with _COMPILE_LOCK:
            compiled = _COMPILED_CACHE.get(compile_key)
            if compiled is None:
                kernel = DecodeAttentionSplitKV(
                    q_heads=_EXPECTED_Q_SHAPE[1],
                    kv_heads=_EXPECTED_KV_SHAPE[1],
                    kv_len=_EXPECTED_KV_SHAPE[2],
                    batch=_EXPECTED_Q_SHAPE[0],
                    num_splits=num_splits,
                    tile_n=tile_n,
                    k_stages=NUM_STAGES_K,
                    v_stages=NUM_STAGES_V,
                    num_workers=num_workers,
                    sm_scale=scale,
                )
                compiled = cute.compile(
                    kernel,
                    q_cute,
                    k_cute,
                    v_cute,
                    opart_cute,
                    lse_cute,
                    out_cute,
                    stream,
                )
                _COMPILED_CACHE[compile_key] = compiled
                _COMPILE_COUNT += 1

    compiled(
        q_cute,
        k_cute,
        v_cute,
        opart_cute,
        lse_cute,
        out_cute,
        stream,
    )
    _ = exporters
    return out.unsqueeze(2)


def get_decode_compile_count():
    """Return successful process-local WXL ``cute.compile`` invocations."""
    return _COMPILE_COUNT


__all__ = ["run_wxl_sm90_gqa_decode", "get_decode_compile_count"]
