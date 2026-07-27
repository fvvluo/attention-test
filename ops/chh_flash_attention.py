# ============================================================
# 算子接入模板 —— 复制这个文件，改名后即可接入 benchmark
# ============================================================
#
# 使用方法（3 步接入）：
#   1. 复制本文件到同目录下，改名为 <你的名字或算子名>_flash_attention.py
#      （例如 zhangsan_flash_attention.py），注意文件名不要以 "_" 开头，
#      否则会被自动扫描忽略。
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


import math

import torch
import triton
import triton.language as tl

from .base import register


def _triton_alloc(size: int, alignment: int, stream):
    # device-side TMA descriptor (tl.make_tensor_descriptor) 需要的全局显存暂存区
    return torch.empty(size, dtype=torch.uint8, device='cuda')


triton.set_allocator(_triton_alloc)


def _prefill_configs():
    """prefill kernel 的 autotune 候选（共享显存超限的配置会被自动跳过）。"""
    cfgs = []
    for BR, BC, w, s in [(128, 64, 4, 2), (128, 64, 4, 3), (128, 64, 8, 2),
                         (128, 128, 8, 2), (128, 128, 8, 3),
                         (64, 64, 4, 2), (64, 64, 4, 4),
                         (64, 128, 4, 2), (64, 128, 4, 3),
                         (128, 32, 4, 3)]:
        cfgs.append(triton.Config({'BR': BR, 'BC': BC}, num_warps=w, num_stages=s))
    return cfgs


def _decode_configs():
    """decode partial kernel 的 autotune 候选（BR 固定 16，只调 BC/warps/stages）。"""
    cfgs = []
    for BC, w, s in [(64, 4, 2), (64, 4, 4), (128, 4, 2), (128, 4, 3),
                     (128, 4, 4), (128, 8, 3), (256, 4, 2), (256, 8, 3)]:
        cfgs.append(triton.Config({'BC': BC}, num_warps=w, num_stages=s))
    return cfgs


@triton.jit
def _fa_inner(o, l, m, q, k_desc, v_desc,
              n_kv, rows, offset, qk_scale, j_lo, j_hi,
              D: tl.constexpr, BC: tl.constexpr,
              MASKED: tl.constexpr, CAUSAL: tl.constexpr):
    """遍历 kv block [j_lo, j_hi)，做 online-softmax 累积。

    MASKED=False 表示这些 block 对所有 query 行完全可见且完全在界内，
    无需任何掩码；MASKED=True 时对越界列（以及 CAUSAL 时的未来列）置 -inf。
    """
    for j in range(j_lo, j_hi):
        k = k_desc.load([j * BC, 0])  # TMA：越界自动补零
        s = tl.dot(q, tl.trans(k)) * qk_scale

        if MASKED:
            cols = j * BC + tl.arange(0, BC)
            mask = (cols < n_kv)[None, :]
            if CAUSAL:
                mask = mask & (cols[None, :] <= (rows + offset)[:, None])
            s = tl.where(mask, s, float("-inf"))

        # qk_scale 已乘入 log2(e)，m/s 处于 log2 域，exp2(x) == exp(x_original)
        m_new = tl.maximum(m, tl.max(s, axis=1))
        alpha = tl.math.exp2(m - m_new)  # 旧统计量的缩放系数（首个 block 时 m=-inf -> alpha=0）
        p = tl.math.exp2(s - m_new[:, None])
        l = l * alpha + tl.sum(p, axis=1)

        v = v_desc.load([j * BC, 0])
        o = o * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m = m_new
    return o, l, m


@triton.autotune(configs=_prefill_configs(), key=['n_q', 'n_kv', 'D'])
@triton.jit
def flash_attn_kernel(q_ptr, k_ptr, v_ptr, o_ptr,
                      sqb, sqh, sqm, sqd,
                      skb, skh, skn, skd,
                      svb, svh, svn, svd,
                      sob, soh, som, sod,
                      n_q, n_kv, h_q, g, qk_scale,
                      D: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr,
                      CAUSAL: tl.constexpr):
    # grid = (cdiv(n_q, BR), batch * q_heads)
    i = tl.program_id(axis=0)
    bh = tl.program_id(axis=1)
    b = bh // h_q
    h = bh % h_q
    hk = h // g  # GQA: 连续的 g 个 q head 共享同一个 kv head

    q_ptr += b * sqb + h * sqh
    k_ptr += b * skb + hk * skh
    v_ptr += b * svb + hk * svh
    o_ptr += b * sob + h * soh

    # TMA descriptors（device-side 创建，基址已定位到本 (b, head)）
    q_desc = tl.make_tensor_descriptor(q_ptr, shape=[n_q, D], strides=[sqm, sqd],
                                       block_shape=[BR, D])
    k_desc = tl.make_tensor_descriptor(k_ptr, shape=[n_kv, D], strides=[skn, skd],
                                       block_shape=[BC, D])
    v_desc = tl.make_tensor_descriptor(v_ptr, shape=[n_kv, D], strides=[svn, svd],
                                       block_shape=[BC, D])
    o_desc = tl.make_tensor_descriptor(o_ptr, shape=[n_q, D], strides=[som, sod],
                                       block_shape=[BR, D])
    q = q_desc.load([i * BR, 0])

    o = tl.zeros([BR, D], dtype=tl.float32)
    l = tl.zeros([BR], dtype=tl.float32)
    m = tl.full([BR], float("-inf"), dtype=tl.float32)

    # decode (n_q < n_kv) 时，query 位置 i 的绝对位置为 i + offset
    offset = n_kv - n_q
    rows = i * BR + tl.arange(0, BR)

    # 两段式循环：
    # phase 1 [0, full)：对本 q block 的所有行完全可见、完全在界内，不加掩码；
    # phase 2 [full, hi)：对角线/尾部 block，加边界 + 因果掩码。
    if CAUSAL:
        full = tl.minimum(n_kv, i * BR + offset + 1) // BC
        limit = tl.minimum(n_kv, i * BR + BR + offset)
    else:
        full = n_kv // BC
        limit = n_kv
    hi = tl.cdiv(limit, BC)

    o, l, m = _fa_inner(o, l, m, q, k_desc, v_desc,
                        n_kv, rows, offset, qk_scale, 0, full,
                        D, BC, False, CAUSAL)
    o, l, m = _fa_inner(o, l, m, q, k_desc, v_desc,
                        n_kv, rows, offset, qk_scale, full, hi,
                        D, BC, True, CAUSAL)

    o = o / l[:, None]
    o_desc.store([i * BR, 0], o.to(o_ptr.dtype.element_ty))  # TMA：越界自动裁剪


@triton.autotune(configs=_decode_configs(), key=['n_kv', 'D'])
@triton.jit
def _decode_partial_kernel(q_ptr, k_ptr, v_ptr, op_ptr, mp_ptr, lp_ptr,
                           sqm, sqd,
                           skb, skh, skn, skd,
                           svb, svh, svn, svd,
                           bhn, n_q, n_kv, h_kv, g, qk_scale, chunk, splits,
                           D: tl.constexpr, M_PAD: tl.constexpr,
                           BC: tl.constexpr, CAUSAL: tl.constexpr):
    """FlashDecoding + GQA 共享，第一段：grid = (splits, b*h_kv)。

    每个 program 负责一个 kv head 的一段 kv [lo, hi)，并一次性载入该组全部
    g 个 q head（共 GNQ = g*n_q 行）与同一份 K/V 做 attention —— K/V 只从
    HBM 读一次，而不是像朴素 GQA 那样每个 q head 各读一遍（g 倍冗余流量）。
    输出该段的 partial (o, m, l)（未归一化），供 merge kernel 合并。
    """
    s_id = tl.program_id(axis=0)
    bhk = tl.program_id(axis=1)
    b = bhk // h_kv
    hk = bhk % h_kv

    k_ptr += b * skb + hk * skh
    v_ptr += b * svb + hk * svh

    GNQ = g * n_q
    # q 为连续 (b, h, n_q, d) 布局，本组 g 个 head 的 GNQ 行地址连续，
    # 可整体视为 [bhn, D] 矩阵从 row0 开始的一块
    row0 = (b * (h_kv * g) + hk * g) * n_q
    q_desc = tl.make_tensor_descriptor(q_ptr, shape=[bhn, D], strides=[sqm, sqd],
                                       block_shape=[M_PAD, D])
    k_desc = tl.make_tensor_descriptor(k_ptr, shape=[n_kv, D], strides=[skn, skd],
                                       block_shape=[BC, D])
    v_desc = tl.make_tensor_descriptor(v_ptr, shape=[n_kv, D], strides=[svn, svd],
                                       block_shape=[BC, D])
    q = q_desc.load([row0, 0])  # TMA：超过 GNQ 的 padding 行自动补零

    lo = s_id * chunk
    hi = tl.minimum(n_kv, lo + chunk)

    o = tl.zeros([M_PAD, D], dtype=tl.float32)
    l = tl.zeros([M_PAD], dtype=tl.float32)
    m = tl.full([M_PAD], float("-inf"), dtype=tl.float32)

    offset = n_kv - n_q
    rows = tl.arange(0, M_PAD)
    r_local = rows % n_q  # tile 行 -> head 内的 query 位置（padding 行不算出）

    for j0 in range(lo, hi, BC):
        k = k_desc.load([j0, 0])
        s = tl.dot(q, tl.trans(k)) * qk_scale

        cols = j0 + tl.arange(0, BC)
        mask = (cols < hi)[None, :]
        if CAUSAL:
            mask = mask & (cols[None, :] <= (r_local + offset)[:, None])
        s = tl.where(mask, s, float("-inf"))

        m_new = tl.maximum(m, tl.max(s, axis=1))
        # 该段可能对某些行全被掩码（m_new=-inf），用 0 兜底避免 exp(nan)
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.math.exp2(m - m_safe)
        p = tl.math.exp2(s - m_safe[:, None])
        l = l * alpha + tl.sum(p, axis=1)

        v = v_desc.load([j0, 0])
        o = o * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m = m_new

    pid = bhk * splits + s_id
    dcols = tl.arange(0, D)
    valid = rows < GNQ
    tl.store(op_ptr + (pid * GNQ + rows)[:, None] * D + dcols[None, :], o,
             mask=valid[:, None])
    tl.store(mp_ptr + pid * GNQ + rows, m, mask=valid)
    tl.store(lp_ptr + pid * GNQ + rows, l, mask=valid)


@triton.jit
def _decode_merge_kernel(op_ptr, mp_ptr, lp_ptr, o_ptr,
                         sob, soh, som, sod,
                         n_q, h_q, h_kv, g, splits,
                         D: tl.constexpr, BR: tl.constexpr, S_PAD: tl.constexpr):
    """FlashDecoding 第二段：grid = (b*h_q,)。

    定位本 q head 在其 kv 组 partial 缓冲中的行段（组内第 gi 个 head）。
    先用一次 2D load 算出所有 split 的全局 max 与权重，再在不相关的循环里
    累加 o（迭代间无依赖，可流水），最后归一化写出。
    """
    bh = tl.program_id(axis=0)
    b = bh // h_q
    h = bh % h_q
    hk = h // g
    gi = h % g
    o_ptr += b * sob + h * soh

    GNQ = g * n_q
    base_row = gi * n_q

    rows = tl.arange(0, BR)
    rmask = rows < n_q
    dcols = tl.arange(0, D)
    sids = tl.arange(0, S_PAD)
    smask = (sids < splits)[:, None] & rmask[None, :]
    pids = (b * h_kv + hk) * splits + sids

    # 一次性载入全部 split 的 (m, l)，算全局 max 与各 split 权重
    m_all = tl.load(mp_ptr + pids[:, None] * GNQ + base_row + rows[None, :],
                    mask=smask, other=float("-inf"))
    l_all = tl.load(lp_ptr + pids[:, None] * GNQ + base_row + rows[None, :],
                    mask=smask, other=0.0)
    m_star = tl.max(m_all, axis=0)
    m_safe = tl.where(m_star == float("-inf"), 0.0, m_star)
    w_all = tl.math.exp2(m_all - m_safe[None, :])  # 无效 split 的 m=-inf -> w=0
    l = tl.sum(l_all * w_all, axis=0)

    o = tl.zeros([BR, D], dtype=tl.float32)
    for sid in range(splits):
        pid = (b * h_kv + hk) * splits + sid
        m_s = tl.load(mp_ptr + pid * GNQ + base_row + rows, mask=rmask,
                      other=float("-inf"))
        w = tl.math.exp2(m_s - m_safe)
        o_s = tl.load(op_ptr + (pid * GNQ + base_row + rows)[:, None] * D
                      + dcols[None, :], mask=rmask[:, None], other=0.0)
        o += w[:, None] * o_s

    o = o / l[:, None]  # l==0 只会出现在 rmask=False 的 padding 行，不会写出
    tl.store(o_ptr + rows[:, None] * som + dcols[None, :] * sod,
             o.to(o_ptr.dtype.element_ty), mask=rmask[:, None])


def _prep(q, k, v, sm_scale):
    """公共检查 + 布局/缩放预处理，返回 (q, k, v, qk_scale)。"""
    b, h, n_q, d = q.shape
    _, h_kv, n_kv, _ = k.shape
    assert k.shape == v.shape
    assert q.dtype in (torch.float16, torch.bfloat16), "TMA kernel 仅支持 fp16/bf16 q/k/v"
    assert q.device.type == "cuda" and k.device == q.device and v.device == q.device
    assert h % h_kv == 0, "q_heads 必须是 kv_heads 的整数倍"
    assert 16 <= d <= 256 and (d & (d - 1)) == 0, "head_dim 需为 [16,256] 内 2 的幂"
    # TMA 要求最后一维连续且 16B 对齐；非连续 view 先物化
    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()
    assert q.data_ptr() % 16 == 0 and k.data_ptr() % 16 == 0 and v.data_ptr() % 16 == 0
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d)
    qk_scale = sm_scale * 1.4426950408889634  # 乘入 log2(e)，kernel 内用 exp2
    return q, k, v, qk_scale


def prefill(q, k, v, causal=True, sm_scale=None):
    """prefill 专用内核：任意 n_q（也支持 n_q != n_kv 的 append/chunked 场景，
    但 n_q <= 16 且 kv 很长时 decode() 更合适）。

    q: (batch, q_heads, seq_len, head_dim); k/v: (batch, kv_heads, seq_len, head_dim)
    返回与 q 同 shape 的输出。
    """
    b, h, n_q, d = q.shape
    n_kv = k.shape[2]
    if causal:
        assert n_kv >= n_q, "causal 时要求 kv 不短于 q"
    q, k, v, qk_scale = _prep(q, k, v, sm_scale)
    o = torch.empty_like(q)
    g = h // k.shape[1]

    grid = lambda meta: (triton.cdiv(n_q, meta['BR']), b * h)
    flash_attn_kernel[grid](q, k, v, o,
                            *q.stride(), *k.stride(), *v.stride(), *o.stride(),
                            n_q, n_kv, h, g, qk_scale,
                            D=d, CAUSAL=causal)
    return o


def decode(q, k, v, causal=True, sm_scale=None):
    """decode 专用内核（FlashDecoding split-K + GQA 组共享）。

    要求 n_q <= 16 且 g*n_q <= 128。同一 kv 组内的所有 q head 在一个 program
    里共享同一份 K/V（K/V 只从 HBM 读一次），kv 再按 splits 段并行，适合
    q_len 极短、kv 很长的自回归生成场景。
    """
    b, h, n_q, d = q.shape
    _, h_kv, n_kv, _ = k.shape
    g = h // h_kv
    assert n_q <= 16, "decode 内核要求 n_q <= 16，否则请用 prefill()"
    assert g * n_q <= 128, "GQA 组行数 g*n_q 超过 128，请用 prefill()"
    if causal:
        assert n_kv >= n_q, "causal 时要求 kv 不短于 q"
    q, k, v, qk_scale = _prep(q, k, v, sm_scale)
    o = torch.empty_like(q)

    GNQ = g * n_q
    M_PAD = max(16, triton.next_power_of_2(GNQ))
    BHK = b * h_kv
    splits = min(max(triton.cdiv(4 * 78, BHK), 1), min(triton.cdiv(n_kv, 256), 128))
    chunk = triton.cdiv(triton.cdiv(n_kv, splits), 128) * 128
    splits = triton.cdiv(n_kv, chunk)

    o_part = torch.empty(BHK * splits * GNQ * d, device=q.device, dtype=torch.float32)
    m_part = torch.empty(BHK * splits * GNQ, device=q.device, dtype=torch.float32)
    l_part = torch.empty(BHK * splits * GNQ, device=q.device, dtype=torch.float32)
    _decode_partial_kernel[(splits, BHK)](q, k, v, o_part, m_part, l_part,
                                          d, 1,
                                          *k.stride(), *v.stride(),
                                          b * h * n_q, n_q, n_kv, h_kv, g,
                                          qk_scale, chunk, splits,
                                          D=d, M_PAD=M_PAD, CAUSAL=causal)
    _decode_merge_kernel[(b * h,)](o_part, m_part, l_part, o,
                                   *o.stride(),
                                   n_q, h, h_kv, g, splits,
                                   D=d, BR=16, S_PAD=128, num_warps=4)
    return o


def attention(q, k, v, causal=True, sm_scale=None):
    """统一入口：按形状分发到 prefill / decode 两个专用内核。

    q: (batch, q_heads, seq_len, head_dim); k/v: (batch, kv_heads, seq_len, head_dim)，
    q_heads 必须是 kv_heads 的整数倍（GQA）。causal 时 query 对齐到 kv 末尾
    （支持 n_q != n_kv）。返回与 q 同 shape 的输出。
    """
    n_q = q.shape[2]
    n_kv = k.shape[2]
    g = q.shape[1] // k.shape[1]
    if n_q <= 16 and n_kv >= 1024 and g * n_q <= 128:
        return decode(q, k, v, causal=causal, sm_scale=sm_scale)
    return prefill(q, k, v, causal=causal, sm_scale=sm_scale)


# TODO: 把下面这个 name 改成能区分你实现方式的唯一名字，
# 例如 "zhangsan_flash_attention (triton)" / "lisi_flash_attention (cuda)"
register("chenhonghua_flash_attention (triton)", attention)
