
#   2. 把下面 TODO 标记的地方替换成你自己的实现。
#   3. 直接运行 `python bench_attention.py` ，你的算子会被自动发现、
#      自动与 baseline 做正确性校验 + 性能对比，无需改动任何其他文件。
#
# 接入 TODO 清单：
#   [ ] 1. 把函数体替换成你自己的 attention 实现（可以是 Triton / CUDA / CuTe DSL 等）
#   [ ] 2. 确认函数签名保持 (q, k, v, causal=True, sm_scale=None) -> output
#   [ ] 3. 确认输出 shape 与 q 相同: (batch, heads, seq_len, head_dim)
#   [ ] 4. 把最后一行 register() 的 name 改成能区分你实现方式的唯一名字
#   [ ] 5. 先跑 `python bench_attention.py --check-only` 验证正确性 PASS
#   [ ] 6. 再跑 `python bench_attention.py` 看性能对比（耗时 / TFLOPS）
#
# 可以参考同目录下 _example_flash_attention.py（纯 PyTorch online-softmax 实现，
# 仅供参考，文件名以 "_" 开头不会被自动扫描注册），里面有完整的分块 + online softmax 算法示例。

import torch

from .base import register

from cutlass import cute
import math

# ============================================================================
# 目标形状最优配置:
#   q :  (1, 64, 131072, 128)   -> batch=1, q_heads=64, seq_len=131072, head_dim=128
#   k/v: (1,  8, 131072, 128)   -> batch=1, kv_heads=8, seq_len=131072, head_dim=128
#
# 路由: seq_len_q(131072) >= BLOCK_M -> prefill_kernel (计算受限, causal)
#       seq_len_q==1(自回归生成)         -> decode_kernel  (KV 带宽受限, seq_len_kv=131072)
#       group_size = q_heads/kv_heads = 64/8 = 8 (GQA, 每8个q head共享1个kv head)
#       head_dim=128 == HEAD_DIM, 无需改动
#
# 性能要点:
#   prefill: 1024 个 KV 块的内循环, 需足够深的流水线隐藏 TMA 延迟 -> STAGES=3
#            (通过 O 复用 Q 的共享内存区释放 32KB, 使 STAGES=3 也能落进 228KB)
#   decode : 单 query 对 131072 长度 KV 做 reduction, 属带宽/延迟受限,
#            必须把 KV 切成足够多的分片并行 (原固定 4 分片严重欠并行) -> 自适应分片
# ============================================================================

# 一个 CTA 内 8 个 warp (2 个 warpgroup):
#   MMA-M 方向 8*16 = 128 = BLOCK_M, 一次 MMA 即覆盖整块 M, 消除多趟 M 迭代,
#   拉满 Tensor Core 利用率 (长序列 causal 场景为计算受限, 这是主要收益点)。
WARPS_PER_CTA=8

# 大 tile 保持 128x128: 长序列下算术强度高, 大 tile 最大化 MMA 效率;
# 128 也与 HEAD_DIM / WARPS_PER_CTA*16 对齐, 无边角浪费。
BLOCK_M=128
BLOCK_N=128
HEAD_DIM=128
INV_SQRT_D=1.0/math.sqrt(float(HEAD_DIM))

# 共享内存预算 (SM100, fp16, 单 CTA 上限约 228KB):
#   单个 K/V tile = BLOCK_N*HEAD_DIM*2B = 32KB
#   O 只在收尾写出, Q 在循环前就已读入寄存器 rQ, 二者生命周期不重叠,
#   因此让 O 复用 Q 的共享内存区 (见 prefill_kernel), 省下 32KB:
#     Q/O(32) + K(STAGES*32) + V(STAGES*32)
#     STAGES=3 -> 32 + 96 + 96 = 224KB <= 228KB (放得下, 且流水线更深)。
#   3 级流水 (2 个在途预取) 能更好地隐藏 1024 次 KV 加载的 TMA 延迟。
PIPELINE_STAGES=3

# decode 每个 KV 分片(CTA)的块大小; 128 块 = 一次 TMA 载入的 KV 行数。
DECODE_BLOCK_KV=128
# decode 的 KV 分割上限: 单条 query 面对 131072(=1024块) KV 时, 固定 4 分片会让
# 每个 CTA 串行处理 256 块, 临界路径极长。提高到 64 分片 -> 每 CTA 仅 16 块,
# 大幅缩短 decode 延迟; reduce 阶段只需合并 64 个部分结果, 开销可忽略。
DECODE_MAX_SPLITS = 64
# 兼容旧引用: 短 KV 时的最小分片数下限。
DECODE_CTAS_PER_KV_DEFAULT = 4
NEG_INF=-float('inf')
ZERO=0.0


def _scale_fragment(frag, scalar):
    cute.for_each(frag, lambda v: v * scalar)
# fr=fr*sc缩放

def _apply_exp(frag):
    cute.for_each(frag, lambda v: math.exp(v))
# fr=exp(fr)

def _fill_value(frag, value):
    cute.fill(frag, value)


def elem_max(a, b):
    result=cute.make_fragment_like(a, dtype=cute.float32)
    if cute.size(b) != cute.size(a):
        cute.for_each(result,
                      lambda v_new, va, vb: max(va, vb),
                      a, cute.broadcast(b, a.shape()))
    else:
        cute.for_each(result,
                      lambda v_new, va, vb: max(va, vb),
                      a, b)
    return result
# 逐个max，返回新fr

def elem_sub_inplace(a, b):
    if cute.size(b) != cute.size(a):
        cute.for_each(a,
                      lambda va, vb: va - vb,
                      cute.broadcast(b, a.shape()))
    else:
        cute.for_each(a,
                      lambda va, vb: va - vb,
                      b)
# 逐个a-

def elem_mul(a, b):
    result = cute.make_fragment_like(a, dtype=cute.float32)
    cute.for_each(result, lambda v_new, va, vb: va * vb, a, b)
    return result

def elem_mul_add(a, b, c):
    result=cute.make_fragment_like(a, dtype=cute.float32)
    cute.for_each(result,
    lambda v_new, va, vb, vc: va * vb + vc,
        a, b, c
    )
    return result
# 逐个a*b+c,返回新fr

def elem_div_inplace(a, b):
    cute.for_each(a, lambda va, vb: va / vb, b)


def sub_broadcast_inplace(a, b):
    cute.for_each(a,
                  lambda va, vb: va - vb,
                   cute.broadcast(b, a.shape()))
# 每个a减标量b，返回新fr

def div_broadcast(a, b):
    cute.for_each(a,
                  lambda va, vb: va / vb,
                  cute.broadcast(b, a.shape()))
# 每个/标量b

def mul_broadcast(a, b):
    cute.for_each(a,
                  lambda va, vb: va * vb,
                  cute.broadcast(b, a.shape()))
# 每个*b

def _copy_fragment(dst, src):
    cute.copy_n(src, dst)
# copy src->dst

def _to_f16(frag_f32):
    # 将 fp32 fragment 转成 fp16（第二个 GEMM 的 A 操作数需要 fp16）
    frag_f16 = cute.make_fragment_like(frag_f32, dtype=cute.float16)
    cute.copy_n(frag_f32, frag_f16)
    return frag_f16


def _make_row_fragment(tiled_mma, init):
    # m/l 是“每行一个标量”，其形状应与 C 累加器沿列(E<1>)归约后的行向量一致，
    # 从而与 MMA 对行的线程划分匹配（而不是整块 [M, N]）。
    c_frag = tiled_mma.make_fragment_C()
    cute.fill(c_frag, 0.0)
    row_like = cute.reduce(cute.maximum{}, c_frag, cute.E<1>{})
    return cute.make_fragment_like(row_like, dtype=cute.float32, init=init)


def _apply_causal_mask(rS, q_block_idx, kv_block_idx,
                       block_m=BLOCK_M, block_n=BLOCK_N):
    # 对角块的逐元素 causal 掩码：全局 key 位置 > 全局 query 位置的元素置 -inf。
    q_base = q_block_idx * block_m
    k_base = kv_block_idx * block_n
    # 块级快速判断：整块都落在下三角内（含对角）则无需逐元素掩码。
    if k_base + block_n - 1 <= q_base:
        return
    crd = cute.make_identity_tensor(cute.Shape[block_m, block_n])
    cute.for_each(
        rS,
        lambda s, coord: NEG_INF
        if (k_base + cute.get(coord, 1)) > (q_base + cute.get(coord, 0))
        else s,
        crd
    )


def _flash_update_step(rS, reg_m, reg_l, reg_o):
# m_new = max(reg_m, rowmax(rS))
# P = exp(rS - m_new)
# l_new = exp(reg_m - m_new) * reg_l + rowsum(P)
# reg_o = exp(reg_m - m_new) * reg_o
# 返回 (m_new, l_new,P)
    rS_max = cute.reduce(cute.maximum{}, rS, cute.E<1>{})
    rM_new = elem_max(reg_m, rS_max)
    elem_sub_inplace(reg_m,rM_new)
    _apply_exp(reg_m)
    sub_broadcast_inplace(rS, rM_new)
    _apply_exp(rS)
    rL_local = cute.reduce(cute.plus{}, rS, cute.E<1>{})
    reg_l_new = elem_mul_add(reg_m, reg_l, rL_local)
    mul_broadcast(reg_o, reg_m)
    _copy_fragment(reg_m, rM_new)
    _copy_fragment(reg_l, reg_l_new)
    return rS



def _consume_kv_tile(k_pipe, v_pipe, smem_K, smem_V,
                    tiled_mma, rQ, copy_s2r_K, copy_s2r_V,
                    reg_m, reg_l, reg_o,
                    rK_work,rV_work,rS_work,
                    causal=False, q_block_idx=0, kv_block_idx=0,
                    sm_scale=INV_SQRT_D):
# 一个KVtile
    k_pipe.consumer_wait()
    k_stage = k_pipe.consumer_acquire()
    v_pipe.consumer_wait()
    v_stage = v_pipe.consumer_acquire()

    cute.copy(copy_s2r_K, smem_K(k_stage), rK_work)
    cute.copy(copy_s2r_V, smem_V(v_stage), rV_work)

    cute.fill(rS_work, 0.0)
    cute.gemm(tiled_mma, rQ, rK_work, rS_work)
    _scale_fragment(rS_work, sm_scale)

    if causal:
        _apply_causal_mask(rS_work, q_block_idx, kv_block_idx)

    rP = _flash_update_step(rS_work, reg_m, reg_l, reg_o)

    # P 是 fp32 累加器，MMA 需要 fp16 的 A 操作数
    rP_f16 = _to_f16(rP)
    cute.gemm(tiled_mma, rP_f16, rV_work, reg_o)

    k_pipe.consumer_release(k_stage)
    v_pipe.consumer_release(v_stage)

def _build_prefill_config():
    mma_atom=cute.MMA_Atom[
        cute.SM100,
        cute.F16F16F16F32_TN,
        16,16,32
    ]
    tiled_mma=cute.make_tiled_mma(
        mma_atom,
        cute.make_layout(cute.Shape[cute.Int<WARPS_PER_CTA>, 1, 1]),
        cute.make_layout(cute.Shape[1, 1, 1])
    )
    gQ_shape=cute.Shape[BLOCK_M,HEAD_DIM]
    sQ_layout=cute.make_layout(gQ_shape,swizzle=cute.swizzle.B_128)
    gK_shape=cute.Shape[BLOCK_N,HEAD_DIM]
    sK_layout=cute.make_layout(gK_shape,swizzle=cute.swizzle.B_128)
    gV_shape=cute.Shape[BLOCK_N,HEAD_DIM]
    sV_layout=cute.make_layout(gV_shape,swizzle=cute.swizzle.B_128)
    gO_shape=cute.Shape[BLOCK_M,HEAD_DIM]
    sO_layout=cute.make_layout(gO_shape,swizzle=cute.swizzle.B_128)

    copy_g2s_Q=cute.make_tiled_copy(
        cute.Copy_Atom[cute.TMA,cute.F16],
        cute.make_layout(gQ_shape),
        sQ_layout
    )
    copy_g2s_K=cute.make_tiled_copy(
        cute.Copy_Atom[cute.TMA,cute.F16],
        cute.make_layout(gK_shape),
        sK_layout
    )
    copy_g2s_V=cute.make_tiled_copy(
        cute.Copy_Atom[cute.TMA,cute.F16],
        cute.make_layout(gV_shape),
        sV_layout
    )
    copy_s2r_Q=cute.make_tiled_copy(
        cute.Copy_Atom[cute.SM90,cute.F16,cute.F16],
        sQ_layout,
        tiled_mma.get_layout_A()
    )
    copy_s2r_K=cute.make_tiled_copy(
        cute.Copy_Atom[cute.SM90,cute.F16,cute.F16],
        sK_layout,
        tiled_mma.get_layout_B()
    )
    copy_s2r_V=cute.make_tiled_copy(
        cute.Copy_Atom[cute.SM90,cute.F16,cute.F16],
        sV_layout,
        tiled_mma.get_layout_B()
    )
    copy_r2s_O=cute.make_tiled_copy(
        cute.Copy_Atom[cute.SM90,cute.F16,cute.F32],
        sO_layout,
        tiled_mma.get_layout_C()
    )
    copy_s2g_O=cute.make_tiled_copy(
        cute.Copy_Atom[cute.TMA,cute.F16],
        cute.make_layout(gO_shape),
        sO_layout
    )
    pred_Q=cute.make_predicate(gQ_shape)
    pred_K=cute.make_predicate(gK_shape)
    pred_V=cute.make_predicate(gV_shape)
    pred_O=cute.make_predicate(gO_shape)

    return (tiled_mma,
            gQ_shape, sQ_layout, gK_shape, sK_layout,
            gV_shape, sV_layout, gO_shape, sO_layout,
            copy_g2s_Q, copy_g2s_K, copy_g2s_V,
            copy_s2r_Q, copy_s2r_K, copy_s2r_V,
            copy_r2s_O, copy_s2g_O,
            pred_Q, pred_K, pred_V, pred_O)


def prefill_kernel(Q_gmem,K_gmem,V_gmem,O_gmem,L_gmem,M_gmem,causal:bool=False,
                   sm_scale=INV_SQRT_D):
    (tiled_mma,
     gQ_shape, sQ_layout, gK_shape, sK_layout,
     gV_shape, sV_layout, gO_shape, sO_layout,
     copy_g2s_Q, copy_g2s_K, copy_g2s_V,
     copy_s2r_Q, copy_s2r_K, copy_s2r_V,
     copy_r2s_O, copy_s2g_O,
     pred_Q, pred_K, pred_V, pred_O) = _build_prefill_config()

    cta_coord=cute.blockIdx(cute.Int<0>{})
    grid_size=cute.gridDim(cute.Int<0>{})

    smem_base=cute.smem_ptr()
    smem_Q = cute.make_tensor(smem_base, sQ_layout)

    smem_K_offset=cute.round_up(cute.cosize(sQ_layout),128)
    smem_K_2stage = cute.make_layout(
        cute.Shape[cute.Int<PIPELINE_STAGES>{}, cute.Shape[BLOCK_N, HEAD_DIM]],
        swizzle=cute.swizzle.B_128
    )
    smem_K = cute.make_tensor(smem_base+smem_K_offset,smem_K_2stage)

    smem_V_offset=cute.round_up(smem_K_offset+cute.cosize(smem_K_2stage),128)
    smem_V_2stage = cute.make_layout(
        cute.Shape[cute.Int<PIPELINE_STAGES>{}, cute.Shape[BLOCK_N, HEAD_DIM]],
        swizzle=cute.swizzle.B_128
    )
    smem_V = cute.make_tensor(smem_base+smem_V_offset,smem_V_2stage)

    # O 复用 Q 的共享内存区: Q 在 KV 循环前已读入寄存器 rQ, 之后其 smem 不再被使用;
    # O 只在收尾阶段写出。二者生命周期不重叠, 复用可省 32KB, 从而支持 STAGES=3。
    smem_O = cute.make_tensor(smem_base, sO_layout)

    num_q_blocks = int(cute.size(Q_gmem, 0)) // BLOCK_M
    num_kv_blocks = int(cute.size(K_gmem, 0)) // BLOCK_N

    work_rK = cute.make_fragment_like(tiled_mma.make_fragment_B(),
                                      dtype=cute.float16)
    work_rV = cute.make_fragment_like(tiled_mma.make_fragment_B(),
                                      dtype=cute.float16)
    work_rS = cute.make_fragment_like(tiled_mma.make_fragment_C(),
                                      dtype=cute.float32)


    for q_block_idx in range(int(cta_coord),
        num_q_blocks, grid_size):

        tgQ=cute.local_tile(
            Q_gmem,
            cute.Shape[BLOCK_M,HEAD_DIM],
            cute.make_coord(q_block_idx)
        )
        # m/l 是每行一个标量（行向量），o 才是整块 [BLOCK_M, HEAD_DIM]
        reg_m=_make_row_fragment(tiled_mma, NEG_INF)
        reg_l=_make_row_fragment(tiled_mma, ZERO)
        reg_o=cute.make_fragment_like(tiled_mma.make_fragment_C(),
        dtype=cute.float32,init=ZERO)

        # O 与 Q 复用同一 smem 区: 覆盖 Q 之前必须等待上一轮 O 的 TMA store 读完该区域,
        # 否则新 Q 会覆盖尚未存完的 O (WAR 冒险)。首轮无在途 store, 该等待为空操作。
        cute.tma_store_wait()
        cute.synchronize()

        cute.copy(copy_g2s_Q, tgQ, smem_Q)
        cute.tma_store_fence()

        rQ = cute.make_fragment_like(tiled_mma.make_fragment_A(),
        dtype=cute.float16)
        cute.copy(copy_s2r_Q, smem_Q, rQ)

        k_pipe = cute.make_pipeline(copy_g2s_K, PIPELINE_STAGES)
        v_pipe = cute.make_pipeline(copy_g2s_V, PIPELINE_STAGES)
        copy_g2s_K_pred = copy_g2s_K.with(pred_K)
        copy_g2s_V_pred = copy_g2s_V.with(pred_V)

        # causal 时只处理到当前 q 块对应的 KV 块
        if causal:
            num_kv_blocks_effective = q_block_idx + 1
        else:
            num_kv_blocks_effective = num_kv_blocks

        preload_count = min(PIPELINE_STAGES - 1, num_kv_blocks_effective)

        for kv_idx in range(preload_count):
            tgK = cute.local_tile(
                K_gmem,
                cute.Shape[BLOCK_N, HEAD_DIM],
                cute.make_coord(kv_idx)
            )
            tgV = cute.local_tile(
                V_gmem,
                cute.Shape[BLOCK_N, HEAD_DIM],
                cute.make_coord(kv_idx)
            )
            k_stage = k_pipe.producer_acquire()
            v_stage = v_pipe.producer_acquire()

            cute.copy(copy_g2s_K_pred, tgK, smem_K(k_stage))
            cute.copy(copy_g2s_V_pred, tgV, smem_V(v_stage))

            k_pipe.producer_commit(k_stage)
            v_pipe.producer_commit(v_stage)

        # 消费块索引需独立计数：流水线里“正在消费”的块比“正在加载”的块滞后 preload_count
        consume_idx = 0
        for kv_idx in range(preload_count, num_kv_blocks_effective):
            tgK = cute.local_tile(
                K_gmem,
                cute.Shape[BLOCK_N, HEAD_DIM],
                cute.make_coord(kv_idx)
            )
            tgV = cute.local_tile(
                V_gmem,
                cute.Shape[BLOCK_N, HEAD_DIM],
                cute.make_coord(kv_idx)
            )
            k_stage = k_pipe.producer_acquire()
            v_stage = v_pipe.producer_acquire()

            cute.copy(copy_g2s_K_pred, tgK, smem_K(k_stage))
            cute.copy(copy_g2s_V_pred, tgV, smem_V(v_stage))

            k_pipe.producer_commit(k_stage)
            v_pipe.producer_commit(v_stage)

            _consume_kv_tile(k_pipe, v_pipe, smem_K, smem_V,
            tiled_mma, rQ, copy_s2r_K, copy_s2r_V,
            reg_m, reg_l, reg_o,work_rK,work_rV,work_rS,
            causal=causal, q_block_idx=q_block_idx, kv_block_idx=consume_idx,
            sm_scale=sm_scale)
            consume_idx += 1

        for _ in range(preload_count):
            _consume_kv_tile(k_pipe, v_pipe, smem_K, smem_V,
            tiled_mma, rQ, copy_s2r_K, copy_s2r_V,
            reg_m, reg_l, reg_o,work_rK,work_rV,work_rS,
            causal=causal, q_block_idx=q_block_idx, kv_block_idx=consume_idx,
            sm_scale=sm_scale)
            consume_idx += 1

        elem_div_inplace(reg_o, reg_l)

        tgL = cute.local_tile(L_gmem, cute.Shape[BLOCK_M],
        cute.make_coord(q_block_idx))
        cute.copy_n(reg_l, tgL)
        tgM = cute.local_tile(M_gmem, cute.Shape[BLOCK_M],
        cute.make_coord(q_block_idx))
        cute.copy_n(reg_m, tgM)
        cute.copy(copy_r2s_O, reg_o, smem_O)
        cute.synchronize()
        tgO = cute.local_tile(O_gmem, cute.Shape[BLOCK_M, HEAD_DIM],
        cute.make_coord(q_block_idx))
        copy_s2g_O_pred=copy_s2g_O.with(pred_O)
        cute.copy(copy_s2g_O_pred, smem_O, tgO)


def _build_decode_config():
    gK_shape = cute.Shape[DECODE_BLOCK_KV, HEAD_DIM]
    sK_layout = cute.make_layout(gK_shape, swizzle=cute.swizzle.B_128)

    gV_shape = cute.Shape[DECODE_BLOCK_KV, HEAD_DIM]
    sV_layout = cute.make_layout(gV_shape, swizzle=cute.swizzle.B_128)

    gQ_shape = cute.Shape[1, HEAD_DIM]
    sQ_layout = cute.make_layout(gQ_shape,swizzle=cute.swizzle.B_128)

    mma_atom = cute.MMA_Atom[
        cute.SM100,
        cute.F16F16F16F32_TN,
        16, 16, 32
    ]
    tiled_mma = cute.make_tiled_mma(
        mma_atom,
        cute.make_layout(cute.Shape[1, 1, 1]),
        cute.make_layout(cute.Shape[1, 1, 1])
    )

    copy_g2s_K = cute.make_tiled_copy(
        cute.Copy_Atom[cute.TMA, cute.F16],
        cute.make_layout(gK_shape),
        sK_layout
    )
    copy_g2s_V = cute.make_tiled_copy(
        cute.Copy_Atom[cute.TMA, cute.F16],
        cute.make_layout(gV_shape),
        sV_layout
    )
    copy_g2s_Q = cute.make_tiled_copy(
        cute.Copy_Atom[cute.TMA, cute.F16],
        cute.make_layout(gQ_shape),
        sQ_layout
    )
    copy_s2r_Q = cute.make_tiled_copy(
        cute.Copy_Atom[cute.SM90, cute.F16, cute.F16],
        sQ_layout,
        tiled_mma.get_layout_A()
    )
    copy_s2r_K = cute.make_tiled_copy(
        cute.Copy_Atom[cute.SM90, cute.F16, cute.F16],
        sK_layout,
        tiled_mma.get_layout_B()
    )
    copy_s2r_V = cute.make_tiled_copy(
        cute.Copy_Atom[cute.SM90, cute.F16, cute.F16],
        sV_layout,
        tiled_mma.get_layout_B()
    )

    pred_K = cute.make_predicate(gK_shape)
    pred_V = cute.make_predicate(gV_shape)

    return (tiled_mma,
            gQ_shape, sQ_layout, gK_shape, sK_layout,
            gV_shape, sV_layout,
            copy_g2s_Q, copy_g2s_K, copy_g2s_V,
            copy_s2r_Q, copy_s2r_K, copy_s2r_V,
            pred_K, pred_V)

def decode_kernel(Q_gmem, K_gmem, V_gmem,
                  partial_O_gmem, partial_L_gmem, partial_M_gmem,
                  sm_scale=INV_SQRT_D):
    (tiled_mma,
     gQ_shape, sQ_layout, gK_shape, sK_layout,
     gV_shape, sV_layout,
     copy_g2s_Q, copy_g2s_K, copy_g2s_V,
     copy_s2r_Q, copy_s2r_K, copy_s2r_V,
     pred_K, pred_V) = _build_decode_config()

    cta_id = int(cute.blockIdx())
    num_ctas = int(cute.gridDim())
    seq_len_kv = int(cute.size(K_gmem, 0))

    smem_base = cute.smem_ptr()

    smem_Q = cute.make_tensor(smem_base, sQ_layout)

    smem_K_offset = cute.round_up(cute.cosize(sQ_layout), 128)
    smem_K_2stage = cute.make_layout(
        cute.Shape[cute.Int<PIPELINE_STAGES>{}, cute.Shape[DECODE_BLOCK_KV, HEAD_DIM]],
        swizzle=cute.swizzle.B_128
    )
    smem_K = cute.make_tensor(smem_base + smem_K_offset, smem_K_2stage)

    smem_V_offset = cute.round_up(
        smem_K_offset + cute.cosize(smem_K_2stage), 128
    )
    smem_V_2stage = cute.make_layout(
        cute.Shape[cute.Int<PIPELINE_STAGES>{}, cute.Shape[DECODE_BLOCK_KV, HEAD_DIM]],
        swizzle=cute.swizzle.B_128
    )
    smem_V = cute.make_tensor(smem_base + smem_V_offset, smem_V_2stage)

    tgQ = cute.local_tile(Q_gmem, cute.Shape[1, HEAD_DIM], cute.make_coord(0))
    cute.copy(copy_g2s_Q, tgQ, smem_Q)
    cute.synchronize()

    rQ = cute.make_fragment_like(tiled_mma.make_fragment_A(),
                                 dtype=cute.float16)
    cute.copy(copy_s2r_Q, smem_Q, rQ)

    total_kv_blocks = (seq_len_kv + DECODE_BLOCK_KV - 1) // DECODE_BLOCK_KV
    kv_blocks_per_cta = (total_kv_blocks + num_ctas - 1) // num_ctas
    kv_start_block = cta_id * kv_blocks_per_cta
    kv_end_block = min(kv_start_block + kv_blocks_per_cta, total_kv_blocks)

    if kv_start_block >= kv_end_block:
        return

    # m/l 为行向量（decode 时行数=1），o 为整块 [1, HEAD_DIM]
    reg_m = _make_row_fragment(tiled_mma, NEG_INF)
    reg_l = _make_row_fragment(tiled_mma, ZERO)
    reg_o = cute.make_fragment_like(tiled_mma.make_fragment_C(),
                                    dtype=cute.float32,
                                    init=ZERO)

    work_rK = cute.make_fragment_like(tiled_mma.make_fragment_B(),
                                      dtype=cute.float16)
    work_rV = cute.make_fragment_like(tiled_mma.make_fragment_B(),
                                      dtype=cute.float16)
    work_rS = cute.make_fragment_like(tiled_mma.make_fragment_C(),
                                      dtype=cute.float32)

    k_pipe = cute.make_pipeline(copy_g2s_K, PIPELINE_STAGES)
    v_pipe = cute.make_pipeline(copy_g2s_V, PIPELINE_STAGES)
    copy_g2s_K_pred = copy_g2s_K.with(pred_K)
    copy_g2s_V_pred = copy_g2s_V.with(pred_V)

    kv_blocks_range = list(range(kv_start_block, kv_end_block))

    if len(kv_blocks_range) == 0:
        return

    preload_count = min(PIPELINE_STAGES - 1, len(kv_blocks_range))

    for i in range(preload_count):
        block_idx = kv_blocks_range[i]
        tgK = cute.local_tile(
            K_gmem,
            cute.Shape[DECODE_BLOCK_KV, HEAD_DIM],
            cute.make_coord(block_idx)
        )
        tgV = cute.local_tile(
            V_gmem,
            cute.Shape[DECODE_BLOCK_KV, HEAD_DIM],
            cute.make_coord(block_idx)
        )
        k_stage = k_pipe.producer_acquire()
        v_stage = v_pipe.producer_acquire()
        cute.copy(copy_g2s_K_pred, tgK, smem_K(k_stage))
        cute.copy(copy_g2s_V_pred, tgV, smem_V(v_stage))
        k_pipe.producer_commit(k_stage)
        v_pipe.producer_commit(v_stage)

    for i in range(preload_count, len(kv_blocks_range)):
        next_block = kv_blocks_range[i]

        tgK = cute.local_tile(
            K_gmem,
            cute.Shape[DECODE_BLOCK_KV, HEAD_DIM],
            cute.make_coord(next_block)
        )
        tgV = cute.local_tile(
            V_gmem,
            cute.Shape[DECODE_BLOCK_KV, HEAD_DIM],
            cute.make_coord(next_block)
        )
        k_stage = k_pipe.producer_acquire()
        v_stage = v_pipe.producer_acquire()
        cute.copy(copy_g2s_K_pred, tgK, smem_K(k_stage))
        cute.copy(copy_g2s_V_pred, tgV, smem_V(v_stage))
        k_pipe.producer_commit(k_stage)
        v_pipe.producer_commit(v_stage)


        _consume_decode_tile(k_pipe, v_pipe, smem_K, smem_V,
                             tiled_mma, rQ, copy_s2r_K, copy_s2r_V,
                             reg_m, reg_l, reg_o,
                             work_rK, work_rV, work_rS,
                             sm_scale=sm_scale)

    for _ in range(preload_count):
        _consume_decode_tile(
            k_pipe, v_pipe, smem_K, smem_V,
            tiled_mma, rQ, copy_s2r_K, copy_s2r_V,
            reg_m, reg_l, reg_o,
            work_rK, work_rV, work_rS,
            sm_scale=sm_scale
        )

    tg_partial_O = cute.local_tile(
        partial_O_gmem,
        cute.Shape[1, HEAD_DIM],
        cute.make_coord(cta_id)
    )
    cute.copy_n(reg_o, tg_partial_O)
    tgL = cute.local_tile(partial_L_gmem, cute.Shape[1], cute.make_coord(cta_id))
    cute.copy_n(reg_l, tgL)
    tgM = cute.local_tile(partial_M_gmem, cute.Shape[1], cute.make_coord(cta_id))
    cute.copy_n(reg_m, tgM)

def _consume_decode_tile(k_pipe, v_pipe, smem_K, smem_V,
                    tiled_mma, rQ, copy_s2r_K, copy_s2r_V,
                    reg_m, reg_l, reg_o,
                    rK_work, rV_work, rS_work,
                    sm_scale=INV_SQRT_D):
    k_pipe.consumer_wait()
    k_stage = k_pipe.consumer_acquire()
    v_pipe.consumer_wait()
    v_stage = v_pipe.consumer_acquire()

    cute.copy(copy_s2r_K, smem_K(k_stage), rK_work)
    cute.copy(copy_s2r_V, smem_V(v_stage), rV_work)

    cute.fill(rS_work, 0.0)
    cute.gemm(tiled_mma, rQ, rK_work, rS_work)
    _scale_fragment(rS_work, sm_scale)

    rP = _flash_update_step(rS_work, reg_m, reg_l, reg_o)

    # P 是 fp32 累加器，MMA 需要 fp16 的 A 操作数
    rP_f16 = _to_f16(rP)
    cute.gemm(tiled_mma, rP_f16, rV_work, reg_o)

    k_pipe.consumer_release(k_stage)
    v_pipe.consumer_release(v_stage)

def decode_reduce_kernel(partial_O_gmem, partial_L_gmem, partial_M_gmem,
                         O_gmem, L_gmem, M_gmem,
                         num_ctas: int):
    m_global = NEG_INF

    for cta_idx in range(num_ctas):
        tgM_i = cute.local_tile(
            partial_M_gmem, cute.Shape[1], cute.make_coord(cta_idx)
        )
        m_i_frag = cute.make_fragment(
            cute.Shape[1], dtype=cute.float32
        )
        cute.copy_n(tgM_i, m_i_frag)
        m_val = cute.get(m_i_frag, 0)
        if m_val > m_global:
            m_global = m_val

    m_global_frag = cute.make_fragment(
        cute.Shape[1], dtype=cute.float32, init=m_global
    )

    # 优化: 第二趟 - 用全局 m_max 做加权合并
    l_new = ZERO
    o_new = cute.make_fragment(
        cute.Shape[HEAD_DIM], dtype=cute.float32, init=ZERO
    )

    # 逐个cta合并
    for cta_idx in range(num_ctas):
        tgM_i = cute.local_tile(
            partial_M_gmem, cute.Shape[1], cute.make_coord(cta_idx)
        )
        tgL_i = cute.local_tile(
            partial_L_gmem, cute.Shape[1], cute.make_coord(cta_idx)
        )
        m_i = cute.make_fragment(
            cute.Shape[1], dtype=cute.float32
        )
        l_i = cute.make_fragment(
            cute.Shape[1], dtype=cute.float32
        )
        cute.copy_n(tgM_i, m_i)
        cute.copy_n(tgL_i, l_i)

        elem_sub_inplace(m_i, m_global_frag)
        _apply_exp(m_i)

        w=cute.get(m_i,0)
        l_i_val=cute.get(l_i,0)
        l_new += w*l_i_val


        tg_O_i = cute.local_tile(
            partial_O_gmem,
            cute.Shape[1, HEAD_DIM],
            cute.make_coord(cta_idx)
        )
        o_i = cute.make_fragment(
            cute.Shape[HEAD_DIM], dtype=cute.float32, init=0.0
        )
        cute.copy_n(tg_O_i, o_i)

        cute.for_each(o_new,
                      lambda vn, w, vi: vn + w * vi,
                      cute.broadcast(m_i, o_new.shape()),
                      o_i)

    l_new_frag = cute.make_fragment(
        cute.Shape[1], dtype=cute.float32, init=l_new
    )
    cute.for_each(o_new,
                  lambda vo, vl: vo / vl,
                  cute.broadcast(l_new_frag, o_new.shape()))

    tgO = cute.local_tile(
        O_gmem, cute.Shape[1, HEAD_DIM], cute.make_coord(0)
    )
    cute.copy_n(o_new, tgO)

    tgL = cute.local_tile(L_gmem, cute.Shape[1], cute.make_coord(0))
    cute.copy_n(l_new_frag, tgL)

    tgM = cute.local_tile(M_gmem, cute.Shape[1], cute.make_coord(0))
    cute.copy_n(m_global_frag, tgM)

def decode_reduce_kernel_vectorized(
    partial_O_gmem, partial_L_gmem,
            partial_M_gmem,
            O_gmem, L_gmem, M_gmem,
            num_ctas: int):
    m_global = NEG_INF

    for cta_idx in range(num_ctas):
        tgM_i = cute.local_tile(
            partial_M_gmem, cute.Shape[1], cute.make_coord(cta_idx)
        )
        m_i = cute.make_fragment(cute.Shape[1], dtype=cute.float32)
        cute.copy_n(tgM_i, m_i)
        m_val = cute.get(m_i, 0)
        if m_val > m_global:
            m_global = m_val

    # 批量合并
    m_global_frag = cute.make_fragment(
        cute.Shape[1], dtype=cute.float32, init=m_global
    )

    l_accum = ZERO
    o_accum = cute.make_fragment(
        cute.Shape[HEAD_DIM], dtype=cute.float32, init=ZERO
    )

    for cta_idx in range(num_ctas):
        tgM_i = cute.local_tile(
            partial_M_gmem, cute.Shape[1], cute.make_coord(cta_idx)
        )
        tgL_i = cute.local_tile(
            partial_L_gmem, cute.Shape[1], cute.make_coord(cta_idx)
        )
        m_i = cute.make_fragment(cute.Shape[1], dtype=cute.float32)
        l_i = cute.make_fragment(cute.Shape[1], dtype=cute.float32)
        cute.copy_n(tgM_i, m_i)
        cute.copy_n(tgL_i, l_i)

        # w_i = exp(m_i - m_global)
        elem_sub_inplace(m_i, m_global_frag)
        _apply_exp(m_i)
        w_i=cute.get(m_i,0)
        l_accum += w_i * cute.get(l_i, 0)

        tg_O_i = cute.local_tile(
            partial_O_gmem,
            cute.Shape[1, HEAD_DIM],
            cute.make_coord(cta_idx)
        )
        o_i = cute.make_fragment(
            cute.Shape[HEAD_DIM], dtype=cute.float32, init=ZERO
        )
        cute.copy_n(tg_O_i, o_i)

        # 使用 elem_mul_add 模式：o_accum += w_i * o_i
        cute.for_each(o_accum,
                      lambda va, w, vi: va + w * vi,
                      cute.broadcast(m_i, o_accum.shape()),
                      o_i)

    # 归一化
    l_frag = cute.make_fragment(cute.Shape[1], dtype=cute.float32,
                                init=l_accum)
    cute.for_each(o_accum,
                  lambda vo, vl: vo / vl,
                  cute.broadcast(l_frag, o_accum.shape()))

    # 写回
    cute.copy_n(o_accum, cute.local_tile(
        O_gmem, cute.Shape[1, HEAD_DIM], cute.make_coord(0)))
    cute.copy_n(l_frag, cute.local_tile(
        L_gmem, cute.Shape[1], cute.make_coord(0)))
    cute.copy_n(m_global_frag, cute.local_tile(
        M_gmem, cute.Shape[1], cute.make_coord(0)))


def _alloc_gmem(shape, dtype=cute.float32):
    # 分配一段全局显存并按给定 shape 视图化（用于中间/输出缓冲）
    return cute.make_tensor(
        cute.make_gmem_ptr(dtype, cute.cosize(cute.make_layout(shape))),
        cute.make_layout(shape)
    )


def _alloc_gmem_like(t):
    # 分配一个与 t 同 shape、同 dtype 的全局显存张量
    return cute.make_tensor(
        cute.make_gmem_ptr(t.dtype, cute.cosize(t.layout)),
        t.layout
    )


def _run_single_head(Q_bh, K_bh, V_bh, O_bh,
                     seq_len_q, seq_len_kv, head_dim,
                     causal, sm_scale):
    # 单个 (batch, head) 上的 FlashAttention：Q/K/V/O 均为 (seq_len, head_dim)
    L_bh = _alloc_gmem(cute.Shape[seq_len_q])
    M_bh = _alloc_gmem(cute.Shape[seq_len_q])

    # 按 query 长度自动分派 prefill / decode
    mode = 'prefill' if seq_len_q >= BLOCK_M else 'decode'

    if mode == 'prefill':
        prefill_kernel(Q_bh, K_bh, V_bh,
                       O_bh, L_bh, M_bh,
                       causal=causal, sm_scale=sm_scale)

    else:  # decode: 单 query 对超长 KV 做 reduction, 带宽/延迟受限
        kv_blocks = (seq_len_kv + DECODE_BLOCK_KV - 1) // DECODE_BLOCK_KV
        # 自适应 KV 分割: 尽量多分片以缩短每个 CTA 的临界路径, 上限 DECODE_MAX_SPLITS。
        # 例: seq_len_kv=131072 -> kv_blocks=1024 -> 64 分片, 每 CTA 仅 16 块
        #     (原来固定 4 分片时每 CTA 需串行 256 块, 延迟高 16 倍)。
        num_decode_ctas = min(kv_blocks, DECODE_MAX_SPLITS)

        partial_O_gmem = _alloc_gmem(cute.Shape[num_decode_ctas, head_dim])
        partial_L_gmem = _alloc_gmem(cute.Shape[num_decode_ctas])
        partial_M_gmem = _alloc_gmem(cute.Shape[num_decode_ctas])

        decode_kernel(Q_bh, K_bh, V_bh,
                      partial_O_gmem, partial_L_gmem, partial_M_gmem,
                      sm_scale=sm_scale)

        decode_reduce_kernel(partial_O_gmem, partial_L_gmem, partial_M_gmem,
                             O_bh, L_bh, M_bh, num_decode_ctas)


def flash_attention(q, k, v, causal=True, sm_scale=None):
    # q: (batch, q_heads, seq_len, head_dim)
    # k, v: (batch, kv_heads, seq_len, head_dim)，q_heads 必须是 kv_heads 的整数倍
    # 输出 shape 与 q 相同: (batch, q_heads, seq_len, head_dim)
    batch = int(cute.size(q, 0))
    q_heads = int(cute.size(q, 1))
    seq_len_q = int(cute.size(q, 2))
    head_dim = int(cute.size(q, 3))

    kv_heads = int(cute.size(k, 1))
    seq_len_kv = int(cute.size(k, 2))

    assert q_heads % kv_heads == 0, (
        f"q_heads({q_heads}) 必须是 kv_heads({kv_heads}) 的整数倍"
    )

    # sm_scale 默认值处理：1/sqrt(head_dim)
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(float(head_dim))

    # GQA/MQA：每 group_size 个 q head 共享一个 kv head
    group_size = q_heads // kv_heads

    O_gmem = _alloc_gmem_like(q)

    for b in range(batch):
        for h in range(q_heads):
            kv_h = h // group_size  # 该 q head 对应的 kv head

            # 取出单个 (batch, head) 的 2D 视图: (seq_len, head_dim)
            Q_bh = q[b, h]
            K_bh = k[b, kv_h]
            V_bh = v[b, kv_h]
            O_bh = O_gmem[b, h]

            _run_single_head(Q_bh, K_bh, V_bh, O_bh,
                             seq_len_q, seq_len_kv, head_dim,
                             causal, sm_scale)

    return O_gmem


def run_target_shape(causal=True):
    # 针对目标形状构造 q/k/v 并执行 FlashAttention:
    #   q :  (1, 64, 131072, 128)
    #   k/v: (1,  8, 131072, 128)   (kv_heads=8, GQA, group_size=8)
    BATCH = 1
    Q_HEADS = 64
    KV_HEADS = 8
    SEQ_LEN = 131072
    HD = 128

    # 半精度输入 (与内核 MMA 的 F16 A/B 操作数一致)
    q = _alloc_gmem(cute.Shape[BATCH, Q_HEADS, SEQ_LEN, HD], dtype=cute.float16)
    k = _alloc_gmem(cute.Shape[BATCH, KV_HEADS, SEQ_LEN, HD], dtype=cute.float16)
    v = _alloc_gmem(cute.Shape[BATCH, KV_HEADS, SEQ_LEN, HD], dtype=cute.float16)

    # sm_scale 默认 1/sqrt(head_dim); causal 默认 True (长序列因果注意力)
    o = flash_attention(q, k, v, causal=causal)
    return o


if __name__ == '__main__':
    run_target_shape(causal=True)




register("yaojixiu_flashattention(CuTe DSL)", attention)
