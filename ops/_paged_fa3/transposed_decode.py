# =============================================================================
# 转置-WGMMA + TMA 流水的 Flash-Decoding（自研实现）
# =============================================================================
#
# 目标形状：B=1, Hq=64, Hkv=8, D=128, q_len=1, kv_len=131072, bf16。
# 访存密集：KV = 512MB, H20 HBM ~4TB/s -> 理论下界 ~0.128ms。
#
# 核心思想（decode 的 GQA GEMV 用 WGMMA 满效率跑）：
#   一组 8 个 q_head 共享一个 kv_head。若把这 8 个 head 放进 WGMMA 的 M 维，
#   M=8 太小、TensorCore 严重浪费。因此【转置】数据流，让长维（kv 行 / head_dim）
#   占据 M 维（=满 M）：
#     S1: St[N_kv, 8] = Kt[N_kv, D] @ Q[D, 8]          # M = kv 行
#     S2: Ot[D, 8]    = Vt[D, N_kv] @ Pt[N_kv, 8]      # M = head_dim
#   两个 GEMM 都跑在满 WGMMA 效率上，计算被访存完全掩盖。
#
# 流水：persistent grid（每 SM 一个常驻 CTA），TMA 生产者 warp + WGMMA 消费者
#       warpgroup，多级 smem 环形缓冲。softmax 在 [kv,head] 片段上按列（每 head）
#       归约。partial kernel 写归一化 O + LSE(log2)，combine kernel 跨 split 归约。
#
# 说明：本实现用官方 cutlass.pipeline.PipelineTmaAsync 搭 TMA 流水，softmax/combine
#       为自研写法。
# =============================================================================

import math
from dataclasses import dataclass
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90
import cutlass.pipeline as pipeline
from cutlass.cutlass_dsl import Boolean, if_generate
from cutlass.pipeline import pipeline_init_wait
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute.runtime import from_dlpack


# 单-warp-TMA-生产者 / 整-warpgroup-消费者的两阶段流水。
# 官方 PipelineTmaAsync 默认的 producer/consumer arrive 计数与"仅 1 个 warp
# 发 TMA、128 线程消费"这种非对称配置不匹配，会死锁；这里按该配置自建同步：
#   - full barrier 由 TMA 硬件在数据到达时 arrive（生产者侧 commit 为空操作）；
#   - empty barrier 由每个 128 线程消费组里的单一线程 arrive 一次。
@dataclass(frozen=True)
class _TmaRing(pipeline.PipelineAsync):
    @staticmethod
    def build(bar_ptr, stages, prod_grp, cons_grp, tx_bytes, do_wait=True):
        full = pipeline.PipelineAsync._make_sync_object(
            bar_ptr.align(min_align=8), stages,
            (pipeline.PipelineOp.TmaLoad, prod_grp), tx_bytes)
        empty = pipeline.PipelineAsync._make_sync_object(
            bar_ptr.align(min_align=8) + stages, stages,
            (pipeline.PipelineOp.AsyncThread, cons_grp))
        if cutlass.const_expr(do_wait):
            pipeline_init_wait()
        return _TmaRing(full, empty, stages, None, None)

    def producer_acquire(self, state, tok: Optional[Boolean] = None):
        if_generate(tok is None or tok == 0,
                    lambda: self.sync_object_empty.wait(state.index, state.phase))
        self.sync_object_full.arrive(state.index, self.producer_mask)

    def producer_commit(self, state):
        pass

    def consumer_release(self, state):
        if_generate(cute.arch.thread_idx()[0] % 128 == 0,
                    lambda: self.sync_object_empty.arrive(state.index, self.consumer_mask))

BF16 = cutlass.BFloat16
F32 = cutlass.Float32

GROUP = 8          # 每个 kv_head 下的 q_head 数
DIM = 128          # head_dim
TILE_KV = 256      # 每次主循环处理的 kv 行数（大 tile 减少 softmax 次数->减少 barrier 气泡）
STAGES = 1         # K 的 TMA 环形缓冲级数（大 tile 下单级即可，省 SMEM）
DEF_VSTAGES = 2    # V 双级环形缓冲：在 H20 上更好遮蔽 TMA 延迟
THREADS = 256      # 128 生产者 + 128 消费者（各一个 warpgroup）
DEF_SPLITS = 9     # 默认 split 数（items=72≈78SM，每CTA独占近整个head，
                   # partial往返极小+combine极轻+DRAM读局部性好 -> 3586 vs sp39 的 3452）


def _acc_as_mn(acc):
    """把 SM90 WGMMA 累加器的物理布局重排成逻辑 (M, N) 视图。"""
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


def _swap01(t):
    """交换前两维的 smem 视图（用于 V 的转置读取）。"""
    shp = (t.shape[1], t.shape[0], *t.shape[2:])
    order = (1, 0, *range(2, cute.rank(t)))
    return cute.composition(t, cute.make_ordered_layout(shp, order=order))


@cute.jit
def _wgmma(mma, acc, a, b, accumulate):
    """沿 K 维发射一串 WGMMA，结果累加到 acc。"""
    warpgroup.fence()
    atom = cute.make_mma_atom(mma.op)
    atom.set(warpgroup.Field.ACCUMULATE, accumulate)
    for kk in cutlass.range_constexpr(cute.size(a.shape[2])):
        cute.gemm(atom, acc, a[None, None, kk], b[None, None, kk], acc)
        atom.set(warpgroup.Field.ACCUMULATE, True)
    warpgroup.commit_group()
    warpgroup.wait_group(0)


class TransposedDecode:
    """转置-WGMMA + TMA 流水的 flash-decoding split-KV kernel。"""

    def __init__(self, q_heads, kv_heads, kv_len, batch,
                 splits=DEF_SPLITS, tile_kv=TILE_KV, stages=STAGES, workers=78,
                 sm_scale=None, v_stages=None):
        assert q_heads == kv_heads * GROUP, "要求 q_heads == kv_heads*8"
        assert kv_len % tile_kv == 0
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.kv_len = kv_len
        self.batch = batch
        self.splits = splits
        self.tile_kv = tile_kv
        self.stages = stages
        # V 用独立（更深）的流水级数：softmax 期间 K 已释放，多缓存的 V 可提前预取，
        # 减少 TMA 断流气泡。默认与 K 相同。
        self.v_stages = v_stages if v_stages is not None else stages
        self.workers = workers
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(DIM)
        self.scale_log2 = sm_scale * math.log2(math.e)

    @cute.jit
    def __call__(self, mQ, mK, mV, mOpart, mLSE, mO, stream: cuda.CUstream):
        self.dt = mQ.element_type

        # Q/O 逻辑视图 (H, D, B)；K/V 逻辑视图 (S, D, (HK, B))。
        Q = cute.make_tensor(mQ.iterator, cute.make_layout(
            (mQ.shape[1], mQ.shape[3], mQ.shape[0]),
            stride=(mQ.stride[1], mQ.stride[3], mQ.stride[0])))
        O = cute.make_tensor(mO.iterator, cute.make_layout(
            (mO.shape[1], mO.shape[3], mO.shape[0]),
            stride=(mO.stride[1], mO.stride[3], mO.stride[0])))
        K = cute.make_tensor(mK.iterator, cute.make_layout(
            (mK.shape[2], mK.shape[3], (mK.shape[1], mK.shape[0])),
            stride=(mK.stride[2], mK.stride[3], (mK.stride[1], mK.stride[0]))))
        V = cute.make_tensor(mV.iterator, cute.make_layout(
            (mV.shape[2], mV.shape[3], (mV.shape[1], mV.shape[0])),
            stride=(mV.stride[2], mV.stride[3], (mV.stride[1], mV.stride[0]))))

        atom = warpgroup.make_smem_layout_atom(
            sm90.get_smem_layout_atom(utils.LayoutEnum.ROW_MAJOR, self.dt, DIM),
            self.dt)
        kL = cute.tile_to_shape(atom, (self.tile_kv, DIM, self.stages), (0, 1, 2))
        vL = cute.tile_to_shape(atom, (self.tile_kv, DIM, self.v_stages), (0, 1, 2))
        qL = cute.tile_to_shape(atom, (GROUP, DIM), (0, 1))
        pL = cute.tile_to_shape(atom, (GROUP, self.tile_kv), (0, 1))

        @cute.struct
        class Smem:
            kbar: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
            vbar: cute.struct.MemRange[cutlass.Int64, self.v_stages * 2]
            xchg: cute.struct.MemRange[F32, 2 * 4 * GROUP]
            sQ: cute.struct.Align[cute.struct.MemRange[self.dt, cute.cosize(qL)], 1024]
            sP: cute.struct.Align[cute.struct.MemRange[self.dt, cute.cosize(pL)], 1024]
            sK: cute.struct.Align[cute.struct.MemRange[self.dt, cute.cosize(kL)], 1024]
            sV: cute.struct.Align[cute.struct.MemRange[self.dt, cute.cosize(vL)], 1024]

        ka, kt = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), K,
            cute.select(kL, mode=[0, 1]), (self.tile_kv, DIM))
        va, vt = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), V,
            cute.select(vL, mode=[0, 1]), (self.tile_kv, DIM))

        # GEMM1: St[N,8] = K[N,D] @ Q[D,8]（A/B 都 K-major）
        mma_qk = sm90.make_trivial_tiled_mma(
            self.dt, self.dt, warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K,
            F32, (1, 1, 1), (64, GROUP))
        # GEMM2: Ot[D,8] = Vt[D,N] @ Pt[N,8]（A MN-major, B K-major）
        mma_pv = sm90.make_trivial_tiled_mma(
            self.dt, self.dt, warpgroup.OperandMajorMode.MN, warpgroup.OperandMajorMode.K,
            F32, (1, 1, 1), (64, GROUP))

        self.split_kernel(
            Q, ka, kt, va, vt, mOpart, mLSE, F32(self.scale_log2),
            qL, kL, vL, pL, mma_qk, mma_pv, Smem,
        ).launch(grid=(self.workers, 1, 1), block=[THREADS, 1, 1],
                 smem=Smem.size_in_bytes(), stream=stream, min_blocks_per_mp=1,
                 use_pdl=True)

        self.combine_kernel(mOpart, mLSE, O).launch(
            grid=(self.q_heads, self.batch, 1), block=[128, 1, 1],
            smem=self.splits * 4, stream=stream, use_pdl=True)

    @cute.kernel
    def split_kernel(self, mQ, ka, mK, va, mV, mOpart, mLSE, scale_log2,
                     qL, kL, vL, pL, mma_qk, mma_pv, Smem: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        wi = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        worker, _, _ = cute.arch.block_idx()

        if wi == 0:
            cpasync.prefetch_descriptor(ka)
            cpasync.prefetch_descriptor(va)

        sm = cutlass.utils.SmemAllocator()
        st = sm.allocate(Smem)

        pg = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        cg = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        txk = cute.size_in_bytes(self.dt, cute.select(kL, mode=[0, 1]))
        txv = cute.size_in_bytes(self.dt, cute.select(vL, mode=[0, 1]))
        pipe_k = _TmaRing.build(st.kbar.data_ptr(), self.stages, pg, cg, txk, do_wait=False)
        pipe_v = _TmaRing.build(st.vbar.data_ptr(), self.v_stages, pg, cg, txv)

        sQ = st.sQ.get_tensor(qL.outer, swizzle=qL.inner)
        sP = st.sP.get_tensor(pL.outer, swizzle=pL.inner)
        sK = st.sK.get_tensor(kL.outer, swizzle=kL.inner)
        sV = st.sV.get_tensor(vL.outer, swizzle=vL.inner)
        sVt = _swap01(sV)

        tiles = self.kv_len // self.tile_kv
        items = self.splits * self.kv_heads * self.batch

        if wi < 4:
            # ---------------- 生产者（仅 warp0 发 TMA）----------------
            if wi == 0:
                ps_k = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.stages)
                ps_v = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.v_stages)
                for it in cutlass.range(worker, items, self.workers, unroll=1):
                    kvh = (it // self.batch) % self.kv_heads
                    sp = it // (self.batch * self.kv_heads)
                    bt = it % self.batch
                    t0 = sp * tiles // self.splits
                    t1 = (sp + 1) * tiles // self.splits
                    gK = cute.local_tile(mK[None, None, (kvh, bt)],
                                         (self.tile_kv, DIM), (None, 0))
                    gV = cute.local_tile(mV[None, None, (kvh, bt)],
                                         (self.tile_kv, DIM), (None, 0))
                    tKs, tKg = cpasync.tma_partition(
                        ka, 0, cute.make_layout(1),
                        cute.group_modes(sK, 0, 2), cute.group_modes(gK, 0, 2))
                    tVs, tVg = cpasync.tma_partition(
                        va, 0, cute.make_layout(1),
                        cute.group_modes(sV, 0, 2), cute.group_modes(gV, 0, 2))
                    for i in cutlass.range(t1 - t0, unroll=1):
                        pipe_k.producer_acquire(ps_k)
                        cute.copy(ka, tKg[None, t0 + i], tKs[None, ps_k.index],
                                  tma_bar_ptr=pipe_k.producer_get_barrier(ps_k))
                        pipe_k.producer_commit(ps_k)
                        ps_k.advance()
                        pipe_v.producer_acquire(ps_v)
                        cute.copy(va, tVg[None, t0 + i], tVs[None, ps_v.index],
                                  tma_bar_ptr=pipe_v.producer_get_barrier(ps_v))
                        pipe_v.producer_commit(ps_v)
                        ps_v.advance()
        else:
            # ---------------- 消费者 warpgroup（128 线程）----------------
            t2 = tidx - 128
            lane = t2 % 32
            wwg = t2 // 32

            sl_qk = mma_qk.get_slice(0)
            sl_pv = mma_pv.get_slice(0)
            thr_qk = mma_qk.get_slice(t2)
            thr_pv = mma_pv.get_slice(t2)

            rK = mma_qk.make_fragment_A(sl_qk.partition_A(sK))
            rQ = mma_qk.make_fragment_B(sl_qk.partition_B(sQ))
            rVt = mma_pv.make_fragment_A(sl_pv.partition_A(sVt))
            rP = mma_pv.make_fragment_B(sl_pv.partition_B(sP))

            accS_shape = mma_qk.partition_shape_C((self.tile_kv, GROUP))
            accO_shape = mma_pv.partition_shape_C((DIM, GROUP))
            accO = cute.make_rmem_tensor(accO_shape, F32)
            accO_mn = _acc_as_mn(accO)

            idS = cute.make_identity_tensor((self.tile_kv, GROUP))
            cS = _acc_as_mn(thr_qk.partition_C(idS))
            idO = cute.make_identity_tensor((DIM, GROUP))
            cO = _acc_as_mn(thr_pv.partition_C(idO))

            NR = cute.size(cS.shape[0])
            NC = cute.size(cS.shape[1])
            rmax = cute.make_rmem_tensor((NC,), F32)
            rsum = cute.make_rmem_tensor((NC,), F32)
            xchg = st.xchg.data_ptr()

            qcopy = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.dt,
                                        num_bits_per_copy=128)
            qtc = cute.make_tiled_copy_tv(
                qcopy,
                cute.make_layout((GROUP, 128 // GROUP), stride=(128 // GROUP, 1)),
                cute.make_layout((1, 128 // (128 // GROUP))))
            qthr = qtc.get_slice(t2)

            cs_k = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.stages)
            cs_v = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.v_stages)
            for it in cutlass.range(worker, items, self.workers, unroll=1):
                kvh = (it // self.batch) % self.kv_heads
                sp = it // (self.batch * self.kv_heads)
                bt = it % self.batch
                t0 = sp * tiles // self.splits
                t1 = (sp + 1) * tiles // self.splits

                gQ = cute.local_tile(mQ[None, None, bt], (GROUP, DIM), (kvh, 0))
                cute.copy(qtc, qthr.partition_S(gQ), qthr.partition_D(sQ))
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.barrier(barrier_id=1, number_of_threads=128)

                accO.fill(0.0)
                rmax.fill(-F32.inf)
                rsum.fill(0.0)

                for i in cutlass.range(t1 - t0, unroll=1):
                    accS = cute.make_rmem_tensor(accS_shape, F32)
                    pipe_k.consumer_wait(cs_k)
                    _wgmma(mma_qk, accS, rK[None, None, None, cs_k.index], rQ, False)
                    pipe_k.consumer_release(cs_k)
                    cs_k.advance()
                    accS_mn = _acc_as_mn(accS)

                    self._softmax(accS_mn, accO_mn, rmax, rsum, cS, cO, sP,
                                  scale_log2, NR, NC, wwg, lane, xchg)

                    pipe_v.consumer_wait(cs_v)
                    _wgmma(mma_pv, accO, rVt[None, None, None, cs_v.index], rP, True)
                    pipe_v.consumer_release(cs_v)
                    cs_v.advance()

                self._epilogue(accO_mn, rmax, rsum, cO, mOpart, mLSE,
                               sp, kvh, bt, NC, wwg, lane)

        # PDL：本 CTA 全部 item 完成、partial/LSE 已写。触发依赖的 combine 提前启动，
        # 把 combine 的 launch 开销藏进 split 尾部（consumer 已写完 mOpart 后）。
        cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def _softmax(self, accS_mn, accO_mn, rmax, rsum, cS, cO, sP,
                 scale_log2, NR: cutlass.Constexpr, NC: cutlass.Constexpr,
                 wwg, lane, xchg):
        # 缩放到 log2 域
        for c in cutlass.range_constexpr(NC):
            for r in cutlass.range_constexpr(NR):
                accS_mn[r, c] = accS_mn[r, c] * scale_log2
        # warp 内每列(每head) max：线程本地跨行 + butterfly
        wmax = cute.make_rmem_tensor((NC,), F32)
        for c in cutlass.range_constexpr(NC):
            m = accS_mn[0, c]
            for r in cutlass.range_constexpr(1, NR):
                m = cute.arch.fmax(m, accS_mn[r, c])
            m = cute.arch.fmax(m, cute.arch.shuffle_sync_bfly(m, offset=4))
            m = cute.arch.fmax(m, cute.arch.shuffle_sync_bfly(m, offset=8))
            m = cute.arch.fmax(m, cute.arch.shuffle_sync_bfly(m, offset=16))
            wmax[c] = m
        # 概率(用 warp-local max, 保证 <=1) + warp 内 sum
        wsum = cute.make_rmem_tensor((NC,), F32)
        for c in cutlass.range_constexpr(NC):
            s = F32(0.0)
            for r in cutlass.range_constexpr(NR):
                p = cute.math.exp2(accS_mn[r, c] - wmax[c], fastmath=True)
                accS_mn[r, c] = p
                s += p
            s += cute.arch.shuffle_sync_bfly(s, offset=4)
            s += cute.arch.shuffle_sync_bfly(s, offset=8)
            s += cute.arch.shuffle_sync_bfly(s, offset=16)
            wsum[c] = s
        # 跨 4 个 warp 交换 (max,sum) 一次
        if lane < 4:
            for c in cutlass.range_constexpr(NC):
                col = cS[0, c][1]
                xchg[wwg * GROUP + col] = wmax[c]
                xchg[4 * GROUP + wwg * GROUP + col] = wsum[c]
        cute.arch.barrier(barrier_id=1, number_of_threads=128)
        # 合并全局 max/sum，rescale O 和 P
        for c in cutlass.range_constexpr(NC):
            col = cS[0, c][1]
            gm = xchg[col]
            for w in cutlass.range_constexpr(1, 4):
                gm = cute.arch.fmax(gm, xchg[w * GROUP + col])
            nm = cute.arch.fmax(rmax[c], gm)
            alpha = cute.math.exp2(rmax[c] - nm, fastmath=True)
            rmax[c] = nm
            corr = cute.math.exp2(wmax[c] - nm, fastmath=True)
            add = F32(0.0)
            for w in cutlass.range_constexpr(4):
                add += xchg[4 * GROUP + w * GROUP + col] * cute.math.exp2(
                    xchg[w * GROUP + col] - nm, fastmath=True)
            rsum[c] = rsum[c] * alpha + add
            for r in cutlass.range_constexpr(cute.size(accO_mn.shape[0])):
                accO_mn[r, c] = accO_mn[r, c] * alpha
            for r in cutlass.range_constexpr(NR):
                accS_mn[r, c] = accS_mn[r, c] * corr
        # 写 P^T 到 smem 供 GEMM2
        for c in cutlass.range_constexpr(NC):
            col = cS[0, c][1]
            for r in cutlass.range_constexpr(NR):
                sP[col, cS[r, c][0]] = BF16(accS_mn[r, c])
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.barrier(barrier_id=1, number_of_threads=128)

    @cute.jit
    def _epilogue(self, accO_mn, rmax, rsum, cO, mOpart, mLSE,
                  sp, kvh, bt, NC: cutlass.Constexpr, wwg, lane):
        for c in cutlass.range_constexpr(NC):
            tot = rsum[c]
            inv = 0.0 if tot == 0.0 or tot != tot else cute.arch.rcp_approx(tot)
            for r in cutlass.range_constexpr(cute.size(accO_mn.shape[0])):
                accO_mn[r, c] = accO_mn[r, c] * inv
            lse = (-F32.inf if tot == 0.0 or tot != tot
                   else rmax[c] + cute.math.log2(tot, fastmath=True))
            if wwg == 0 and lane < 4:
                mLSE[sp, kvh, cO[0, c][1], bt] = lse
        for c in cutlass.range_constexpr(NC):
            col = cO[0, c][1]
            for r in cutlass.range_constexpr(cute.size(accO_mn.shape[0])):
                mOpart[sp, kvh, col, cO[r, c][0], bt] = BF16(accO_mn[r, c])

    @cute.kernel
    def combine_kernel(self, mOpart, mLSE, mO):
        qh, bt, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        kvh = qh // GROUP
        g = qh % GROUP
        sm = cutlass.utils.SmemAllocator()
        buf = sm.allocate_tensor(F32, cute.make_layout(self.splits), 16)
        cute.arch.griddepcontrol_wait()   # PDL：等 split 写完 partial/LSE
        for i in cutlass.range(tidx, self.splits, 128):
            buf[i] = mLSE[i, kvh, g, bt]
        cute.arch.barrier()
        m = -F32.inf
        for s in cutlass.range_constexpr(self.splits):
            m = cute.arch.fmax(m, buf[s])
        den = F32(0.0)
        w = cute.make_rmem_tensor((self.splits,), F32)
        for s in cutlass.range_constexpr(self.splits):
            w[s] = cute.math.exp2(buf[s] - m, fastmath=True)
            den += w[s]
        acc = F32(0.0)
        for s in cutlass.range_constexpr(self.splits):
            acc += w[s] * F32(mOpart[s, kvh, g, tidx, bt])
        inv = 0.0 if den == 0.0 or den != den else cute.arch.rcp_approx(den)
        mO[qh, tidx, bt] = BF16(acc * inv)


# ---------------------------------------------------------------------------
# Host 包装
# ---------------------------------------------------------------------------
_CACHE = {}
_BUF = {}
_VIEW_CACHE = {}


def _cute4d(t):
    key = (id(t), t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.device.index)
    view = _VIEW_CACHE.get(key)
    if view is not None:
        return view
    view = (from_dlpack(t, assumed_align=16)
            .mark_layout_dynamic(leading_dim=3)
            .mark_compact_shape_dynamic(mode=3, stride_order=t.dim_order(),
                                        divisibility=128 // BF16.width))
    if len(_VIEW_CACHE) >= 64:
        _VIEW_CACHE.clear()
    _VIEW_CACHE[key] = view
    return view


def decode(q, k, v, sm_scale=None, splits=DEF_SPLITS, tile_kv=TILE_KV,
           stages=STAGES, workers=None, v_stages=DEF_VSTAGES):
    import torch
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    B, H, _, D = q.shape
    HK, S = k.shape[1], k.shape[2]
    key = (q.device.index, B, H, HK, S, D, splits, tile_kv, stages, v_stages, workers, float(sm_scale))
    with torch.cuda.device(q.device):
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        ent = _CACHE.get(key)
        if ent is None:
            if workers is None:
                # 给 combine/调度保留一个 SM；H20 78 SM 时使用 77 workers。
                workers = max(1, torch.cuda.get_device_properties(q.device).multi_processor_count - 1)
            ker = TransposedDecode(H, HK, S, B, splits=splits, tile_kv=tile_kv, v_stages=v_stages,
                                   stages=stages, workers=workers, sm_scale=sm_scale)
            opart = torch.empty((splits, HK, GROUP, D, B), dtype=torch.bfloat16, device=q.device)
            lse = torch.empty((splits, HK, GROUP, B), dtype=torch.float32, device=q.device)
            o = torch.empty_like(q)
            opart_view = from_dlpack(opart, assumed_align=16)
            lse_view = from_dlpack(lse, assumed_align=16)
            output_view = _cute4d(o)
            args = (_cute4d(q), _cute4d(k), _cute4d(v),
                    opart_view, lse_view, output_view, stream)
            comp = cute.compile(ker, *args)
            ent = (comp, opart, lse, o, opart_view, lse_view, output_view)
            _CACHE[key] = ent
        comp, opart, lse, o, opart_view, lse_view, output_view = ent
        comp(_cute4d(q), _cute4d(k), _cute4d(v),
             opart_view, lse_view, output_view, stream)
    return o
