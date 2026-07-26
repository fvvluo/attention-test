# ============================================================
# 算子接入模板 —— 复制这个文件，改名后即可接入 benchmark
# ============================================================
#
# 使用方法（3 步接入）：
#   1. 复制本文件到同目录下，改名为 <你的名字或算子名>_flash_attention.py
#      （例如 zhangsan_flash_attention.py），注意文件名不要以 "_" 开头，
#      否则会被自动扫描忽略（本文件就是因为以 "_" 开头才不会被扫描到）。
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


import math

import torch
import triton
import triton.language as tl



_Q_HEADS = 64
_KV_HEADS = 8
_N_CTX = 131072
_HEAD_DIM = 128
_GROUP_SIZE = _Q_HEADS // _KV_HEADS
_BLOCK_M = 128
_BLOCK_N = 64
_NUM_M_BLOCKS = _N_CTX // _BLOCK_M
_QK_SCALE = (1.0 / math.sqrt(_HEAD_DIM)) * math.log2(math.e)
_BLOCK_N_DECODE = 128


@triton.jit
def _attn_prefill_kernel(
    Q,
    K,
    V,
    Out,
    QK_SCALE: tl.constexpr,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    group_head = tl.program_id(0)
    start_m = tl.program_id(1)
    kv_head = tl.program_id(2)
    q_head = kv_head * GROUP_SIZE + group_head

    q_offset = q_head * N_CTX * HEAD_DIM
    kv_offset = kv_head * N_CTX * HEAD_DIM

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)

    q_ptrs = Q + q_offset + offs_m[:, None] * HEAD_DIM + offs_k[None, :]
    k_ptrs = K + kv_offset + offs_k[:, None] + offs_n[None, :] * HEAD_DIM
    v_ptrs = V + kv_offset + offs_n[:, None] * HEAD_DIM + offs_k[None, :]
    o_ptrs = Out + q_offset + offs_m[:, None] * HEAD_DIM + offs_k[None, :]

    q = tl.load(q_ptrs)
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    diagonal_start = start_m * BLOCK_M

    # Every key before this query tile is visible to every query in the tile.
    for start_n in range(0, diagonal_start, BLOCK_N):
        k = tl.load(k_ptrs + start_n * HEAD_DIM)
        qk = tl.dot(q, k) * QK_SCALE

        m_next = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_next[:, None])
        alpha = tl.math.exp2(m_i - m_next)
        acc = acc * alpha[:, None]
        v = tl.load(v_ptrs + start_n * HEAD_DIM)
        acc = tl.dot(p.to(q.dtype), v, acc)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_next

    # BLOCK_M / BLOCK_N == 2, so both diagonal tiles must be masked.
    for start_n in range(diagonal_start, diagonal_start + BLOCK_M, BLOCK_N):
        k = tl.load(k_ptrs + start_n * HEAD_DIM)
        qk = tl.dot(q, k) * QK_SCALE
        causal_mask = offs_m[:, None] >= start_n + offs_n[None, :]
        qk = tl.where(causal_mask, qk, -float("inf"))

        m_next = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_next[:, None])
        alpha = tl.math.exp2(m_i - m_next)
        acc = acc * alpha[:, None]
        v = tl.load(v_ptrs + start_n * HEAD_DIM)
        acc = tl.dot(p.to(q.dtype), v, acc)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_next

    tl.store(o_ptrs, (acc / l_i[:, None]).to(q.dtype))

@triton.jit
def _attn_decode_kernel(
    Q,
    K,
    V,
    Out,
    QK_SCALE: tl.constexpr,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    kv_head = tl.program_id(0)

    # q_len == 1: Q/Out 的每个 head 只有一行。K/V 仍是全长 N_CTX。
    q_offset = kv_head * GROUP_SIZE * HEAD_DIM
    kv_offset = kv_head * N_CTX * HEAD_DIM

    offs_g = tl.arange(0, GROUP_SIZE)   # 同一 kv_head 下的 GROUP_SIZE 个 q_head 当作 M 维
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)

    # 一次把这一组 GROUP_SIZE 个 query 向量读进来: [GROUP_SIZE, HEAD_DIM]
    q_ptrs = Q + q_offset + offs_g[:, None] * HEAD_DIM + offs_k[None, :]
    k_ptrs = K + kv_offset + offs_k[:, None] + offs_n[None, :] * HEAD_DIM
    v_ptrs = V + kv_offset + offs_n[:, None] * HEAD_DIM + offs_k[None, :]
    o_ptrs = Out + q_offset + offs_g[:, None] * HEAD_DIM + offs_k[None, :]

    q = tl.load(q_ptrs)
    m_i = tl.full([GROUP_SIZE], -float("inf"), tl.float32)
    l_i = tl.zeros([GROUP_SIZE], tl.float32)
    acc = tl.zeros([GROUP_SIZE, HEAD_DIM], tl.float32)

    # decode 无因果 mask: 新 token 位置为 N_CTX-1, 可见全部 KV。
    # N_CTX % BLOCK_N == 0, 无需尾块掩码。
    for start_n in range(0, N_CTX, BLOCK_N):
        k = tl.load(k_ptrs + start_n * HEAD_DIM)
        qk = tl.dot(q, k) * QK_SCALE

        m_next = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_next[:, None])
        alpha = tl.math.exp2(m_i - m_next)
        acc = acc * alpha[:, None]
        v = tl.load(v_ptrs + start_n * HEAD_DIM)
        acc = tl.dot(p.to(q.dtype), v, acc)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_next

    tl.store(o_ptrs, (acc / l_i[:, None]).to(q.dtype))

def attention(q, k, v, causal=True, sm_scale=None):
    out = torch.empty_like(q)
    if q.shape[2]!=1:
        grid = (_GROUP_SIZE, _NUM_M_BLOCKS, _KV_HEADS)
        _attn_prefill_kernel[grid](
            q,
            k,
            v,
            out,
            QK_SCALE=_QK_SCALE,
            N_CTX=_N_CTX,
            HEAD_DIM=_HEAD_DIM,
            GROUP_SIZE=_GROUP_SIZE,
            BLOCK_M=_BLOCK_M,
            BLOCK_N=_BLOCK_N,
            # num_warps=8,
            # num_stages=4,
        )
    else:
        grid = (8,)
        _attn_decode_kernel[grid](
            q, k, v, out,
            QK_SCALE=_QK_SCALE,
            N_CTX=_N_CTX,                      # cache 容量 / 行 stride
            HEAD_DIM=_HEAD_DIM,
            GROUP_SIZE=_GROUP_SIZE,
            BLOCK_N=_BLOCK_N_DECODE,
        )
    return out
import math

import torch
import triton
import triton.language as tl


_Q_HEADS = 64
_KV_HEADS = 8
_N_CTX = 131072
_HEAD_DIM = 128
_GROUP_SIZE = _Q_HEADS // _KV_HEADS
_BLOCK_M = 128
_BLOCK_N = 64
_NUM_M_BLOCKS = _N_CTX // _BLOCK_M
_QK_SCALE = (1.0 / math.sqrt(_HEAD_DIM)) * math.log2(math.e)
_BLOCK_N_DECODE = 64

# ---- Flash-Decoding (Split-KV) 相关常量 ----
# decode: q_len == 1。把长 KV 沿序列切成 _N_SPLITS 段并行, 再由 reduce kernel 合并。
_N_SPLITS = 128                          # KV 切段数; 131072/128 = 1024 每段
_SPLIT_LEN = _N_CTX // _N_SPLITS         # = 1024, 每段长度 (恰好 % BLOCK_N == 0)
_PAD_M = 16                              # GROUP_SIZE=8 < 16, tl.dot 的 M 维需 pad 到 16


@triton.jit
def _attn_prefill_kernel(
    Q,
    K,
    V,
    Out,
    QK_SCALE: tl.constexpr,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    group_head = tl.program_id(0)
    start_m = tl.program_id(1)
    kv_head = tl.program_id(2)
    q_head = kv_head * GROUP_SIZE + group_head

    q_offset = q_head * N_CTX * HEAD_DIM
    kv_offset = kv_head * N_CTX * HEAD_DIM

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)

    q_ptrs = Q + q_offset + offs_m[:, None] * HEAD_DIM + offs_k[None, :]
    k_ptrs = K + kv_offset + offs_k[:, None] + offs_n[None, :] * HEAD_DIM
    v_ptrs = V + kv_offset + offs_n[:, None] * HEAD_DIM + offs_k[None, :]
    o_ptrs = Out + q_offset + offs_m[:, None] * HEAD_DIM + offs_k[None, :]

    q = tl.load(q_ptrs)
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    diagonal_start = start_m * BLOCK_M

    # Every key before this query tile is visible to every query in the tile.
    for start_n in range(0, diagonal_start, BLOCK_N):
        k = tl.load(k_ptrs + start_n * HEAD_DIM)
        qk = tl.dot(q, k) * QK_SCALE

        m_next = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_next[:, None])
        alpha = tl.math.exp2(m_i - m_next)
        acc = acc * alpha[:, None]
        v = tl.load(v_ptrs + start_n * HEAD_DIM)
        acc = tl.dot(p.to(q.dtype), v, acc)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_next

    # BLOCK_M / BLOCK_N == 2, so both diagonal tiles must be masked.
    for start_n in range(diagonal_start, diagonal_start + BLOCK_M, BLOCK_N):
        k = tl.load(k_ptrs + start_n * HEAD_DIM)
        qk = tl.dot(q, k) * QK_SCALE
        causal_mask = offs_m[:, None] >= start_n + offs_n[None, :]
        qk = tl.where(causal_mask, qk, -float("inf"))

        m_next = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_next[:, None])
        alpha = tl.math.exp2(m_i - m_next)
        acc = acc * alpha[:, None]
        v = tl.load(v_ptrs + start_n * HEAD_DIM)
        acc = tl.dot(p.to(q.dtype), v, acc)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_next

    tl.store(o_ptrs, (acc / l_i[:, None]).to(q.dtype))

# ============================================================
# Flash-Decoding: Split-KV 两阶段 decode (q_len == 1)
#   Stage 1: _attn_decode_split_kernel  grid=(KV_HEADS, N_SPLITS)
#            每个 program 只扫一段 KV, 输出局部 acc / m / l 到中间 buffer。
#   Stage 2: _attn_decode_reduce_kernel grid=(Q_HEADS,)
#            把每个 q_head 的 N_SPLITS 段局部结果用在线 softmax 合并成最终输出。
# 关键: M 维 = GROUP_SIZE=8 < 16, 不满足 tl.dot 的 MMA 最小尺寸,
#       故把 M pad 到 PAD_M=16, 后 (PAD_M-GROUP_SIZE) 行用 -inf 屏蔽, 只写回前 GROUP_SIZE 行。
# 本 benchmark 场景: kv_len == N_CTX == 131072, N_CTX % SPLIT_LEN == 0, SPLIT_LEN % BLOCK_N == 0,
#       故无需运行时 kv_len 尾块掩码 (与原 decode 假设一致)。
# ============================================================
@triton.jit
def _attn_decode_split_kernel(
    Q,
    K,
    V,
    M_buf,            # [Q_HEADS, N_SPLITS]            每段局部 max (log2 域)
    L_buf,            # [Q_HEADS, N_SPLITS]            每段局部 exp2 和
    Acc_buf,          # [Q_HEADS, N_SPLITS, HEAD_DIM]  每段局部未归一化加权和
    QK_SCALE: tl.constexpr,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_LEN: tl.constexpr,
    N_SPLITS: tl.constexpr,
    PAD_M: tl.constexpr,
):
    kv_head = tl.program_id(0)
    split_id = tl.program_id(1)

    q_offset = kv_head * GROUP_SIZE * HEAD_DIM
    kv_offset = kv_head * N_CTX * HEAD_DIM

    offs_m = tl.arange(0, PAD_M)         # M pad 到 16; 前 GROUP_SIZE 行有效
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    m_valid = offs_m < GROUP_SIZE        # [PAD_M] 有效行掩码

    # 越界的 pad 行读第 0 行数据(反正后面会 -inf 屏蔽, 只求指针合法)
    q_rows = tl.where(m_valid, offs_m, 0)
    q_ptrs = Q + q_offset + q_rows[:, None] * HEAD_DIM + offs_k[None, :]
    q = tl.load(q_ptrs)                  # [PAD_M, HEAD_DIM]

    kbase = kv_offset + split_id * SPLIT_LEN * HEAD_DIM
    k_ptrs = K + kbase + offs_k[:, None] + offs_n[None, :] * HEAD_DIM
    v_ptrs = V + kbase + offs_n[:, None] * HEAD_DIM + offs_k[None, :]

    m_i = tl.full([PAD_M], -float("inf"), tl.float32)
    l_i = tl.zeros([PAD_M], tl.float32)
    acc = tl.zeros([PAD_M, HEAD_DIM], tl.float32)

    # 只扫本段 [0, SPLIT_LEN); decode 无因果 mask。
    for start_n in range(0, SPLIT_LEN, BLOCK_N):
        k = tl.load(k_ptrs + start_n * HEAD_DIM)          # [HEAD_DIM, BLOCK_N]
        qk = tl.dot(q, k) * QK_SCALE                      # [PAD_M, BLOCK_N]
        qk = tl.where(m_valid[:, None], qk, -float("inf"))  # pad 行屏蔽

        m_next = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_next[:, None])
        alpha = tl.math.exp2(m_i - m_next)
        acc = acc * alpha[:, None]
        v = tl.load(v_ptrs + start_n * HEAD_DIM)          # [BLOCK_N, HEAD_DIM]
        acc = tl.dot(p.to(q.dtype), v, acc)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_next

    # 写回局部结果 (只写前 GROUP_SIZE 行)。acc 此处未除 l, 归一化留到 reduce。
    q_head = kv_head * GROUP_SIZE + offs_m
    ml_idx = q_head * N_SPLITS + split_id
    tl.store(M_buf + ml_idx, m_i, mask=m_valid)
    tl.store(L_buf + ml_idx, l_i, mask=m_valid)
    acc_idx = (q_head * N_SPLITS + split_id)[:, None] * HEAD_DIM + offs_k[None, :]
    tl.store(Acc_buf + acc_idx, acc, mask=m_valid[:, None])


@triton.jit
def _attn_decode_reduce_kernel(
    M_buf,            # [Q_HEADS, N_SPLITS]
    L_buf,            # [Q_HEADS, N_SPLITS]
    Acc_buf,          # [Q_HEADS, N_SPLITS, HEAD_DIM]
    Out,              # [Q_HEADS, HEAD_DIM]
    HEAD_DIM: tl.constexpr,
    N_SPLITS: tl.constexpr,
):
    q_head = tl.program_id(0)
    offs_s = tl.arange(0, N_SPLITS)
    offs_k = tl.arange(0, HEAD_DIM)

    m_s = tl.load(M_buf + q_head * N_SPLITS + offs_s)     # [N_SPLITS]
    l_s = tl.load(L_buf + q_head * N_SPLITS + offs_s)     # [N_SPLITS]

    m = tl.max(m_s, 0)                                    # 全局 max
    scale = tl.math.exp2(m_s - m)                         # 每段 rescale 因子 [N_SPLITS]
    denom = tl.sum(l_s * scale, 0)                        # 全局归一化分母

    acc_ptrs = Acc_buf + q_head * N_SPLITS * HEAD_DIM + offs_s[:, None] * HEAD_DIM + offs_k[None, :]
    acc = tl.load(acc_ptrs)                               # [N_SPLITS, HEAD_DIM]
    out = tl.sum(acc * scale[:, None], 0) / denom         # 加权合并 + 归一化

    tl.store(Out + q_head * HEAD_DIM + offs_k, out.to(Out.dtype.element_ty))

def attention(q, k, v, causal=True, sm_scale=None):
    out = torch.empty_like(q)
    if q.shape[2]!=1:
        grid = (_GROUP_SIZE, _NUM_M_BLOCKS, _KV_HEADS)
        _attn_prefill_kernel[grid](
            q,
            k,
            v,
            out,
            QK_SCALE=_QK_SCALE,
            N_CTX=_N_CTX,
            HEAD_DIM=_HEAD_DIM,
            GROUP_SIZE=_GROUP_SIZE,
            BLOCK_M=_BLOCK_M,
            BLOCK_N=_BLOCK_N,
            # num_warps=8,
            # num_stages=4,
        )
    else:
        # ---- Flash-Decoding: Split-KV 两阶段 ----
        # 中间 buffer (fp32): 各段局部 m / l / acc
        m_buf = torch.empty((_Q_HEADS, _N_SPLITS), dtype=torch.float32, device=q.device)
        l_buf = torch.empty((_Q_HEADS, _N_SPLITS), dtype=torch.float32, device=q.device)
        acc_buf = torch.empty((_Q_HEADS, _N_SPLITS, _HEAD_DIM), dtype=torch.float32, device=q.device)

        # Stage 1: 沿 KV 切段并行, grid = (KV_HEADS, N_SPLITS)
        _attn_decode_split_kernel[(_KV_HEADS, _N_SPLITS)](
            q, k, v,
            m_buf, l_buf, acc_buf,
            QK_SCALE=_QK_SCALE,
            N_CTX=_N_CTX,
            HEAD_DIM=_HEAD_DIM,
            GROUP_SIZE=_GROUP_SIZE,
            BLOCK_N=_BLOCK_N_DECODE,
            SPLIT_LEN=_SPLIT_LEN,
            N_SPLITS=_N_SPLITS,
            PAD_M=_PAD_M,
        )
        # Stage 2: 合并各段, grid = (Q_HEADS,)
        _attn_decode_reduce_kernel[(_Q_HEADS,)](
            m_buf, l_buf, acc_buf, out,
            HEAD_DIM=_HEAD_DIM,
            N_SPLITS=_N_SPLITS,
        )
    return out



# TODO: 把下面这个 name 改成能区分你实现方式的唯一名字，
# 例如 "zhangsan_flash_attention (triton)" / "lisi_flash_attention (cuda)"
register("sunyichen_FA_triton", attention)
