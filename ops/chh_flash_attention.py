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



import math

import torch
import triton
import triton.language as tl

from .base import register


@triton.jit
def flash_attn_kernel(q_ptr, k_ptr, v_ptr, o_ptr,
                      sqb, sqh, sqm, sqd,
                      skb, skh, skn, skd,
                      svb, svh, svn, svd,
                      sob, soh, som, sod,
                      n_q, n_kv, h_q, g, sm_scale,
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

    q_offsets = tl.make_block_ptr(q_ptr,
                                  shape=(n_q, D),
                                  strides=(sqm, sqd),
                                  offsets=(i * BR, 0),
                                  block_shape=(BR, D),
                                  order=(1, 0))
    q = tl.load(q_offsets, boundary_check=(0, 1), padding_option='zero')

    o = tl.zeros([BR, D], dtype=tl.float32)
    l = tl.zeros([BR], dtype=tl.float32)
    m = tl.full([BR], float("-inf"), dtype=tl.float32)

    # decode (n_q < n_kv) 时，query 位置 i 的绝对位置为 i + offset
    offset = n_kv - n_q
    rows = i * BR + tl.arange(0, BR)

    for j in range(tl.cdiv(n_kv, BC)):
        k_offsets = tl.make_block_ptr(k_ptr,
                                      shape=(n_kv, D),
                                      strides=(skn, skd),
                                      offsets=(j * BC, 0),
                                      block_shape=(BC, D),
                                      order=(1, 0))
        k = tl.load(k_offsets, boundary_check=(0, 1), padding_option='zero')
        s = tl.dot(q, tl.trans(k)) * sm_scale

        # 两类掩码都置 -inf：
        # 1) 越界列（n_kv 不是 BC 整数倍时，尾部零填充的 k 不能参与 softmax）
        # 2) 因果掩码：query 绝对位置 rows+offset 只能看到 <= 它的 key
        cols = j * BC + tl.arange(0, BC)
        mask = (cols < n_kv)[None, :]
        if CAUSAL:
            mask = mask & (cols[None, :] <= (rows + offset)[:, None])
        s = tl.where(mask, s, float("-inf"))

        m_new = tl.maximum(m, tl.max(s, axis=1))
        alpha = tl.exp(m - m_new)  # 旧统计量的缩放系数（首个 block 时 m=-inf -> alpha=0）
        p = tl.exp(s - m_new[:, None])
        l = l * alpha + tl.sum(p, axis=1)

        v_offsets = tl.make_block_ptr(v_ptr,
                                      shape=(n_kv, D),
                                      strides=(svn, svd),
                                      offsets=(j * BC, 0),
                                      block_shape=(BC, D),
                                      order=(1, 0))
        v = tl.load(v_offsets, boundary_check=(0, 1), padding_option='zero')
        o = o * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m = m_new

    o = o / l[:, None]
    o_offsets = tl.make_block_ptr(o_ptr,
                                  shape=(n_q, D),
                                  strides=(som, sod),
                                  offsets=(i * BR, 0),
                                  block_shape=(BR, D),
                                  order=(1, 0))
    tl.store(o_offsets, o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))


def attention(q, k, v, causal=True, sm_scale=None):
    """
    Args:
        q: shape (batch, q_heads, seq_len, head_dim)
        k, v: shape (batch, kv_heads, seq_len, head_dim), 其中 q_heads 必须是
            kv_heads 的整数倍(GQA); 标准 MHA 时 q_heads == kv_heads。
            kernel 内部通过 h // g 找到对应的 kv head，无需显式 broadcast。
        causal: 是否使用因果掩码 (只看当前位置及之前的 token)。
            支持 n_q != n_kv 的 decode 场景（此时要求 n_kv >= n_q，
            query 对齐到 kv 末尾）。
        sm_scale: softmax 缩放系数，默认为 1/sqrt(head_dim)

    Returns:
        output: shape 与 q 相同，(batch, q_heads, seq_len, head_dim)
    """
    b, h, n_q, d = q.shape
    _, h_kv, n_kv, _ = k.shape
    assert k.shape == v.shape
    assert h % h_kv == 0, "q_heads 必须是 kv_heads 的整数倍"
    assert 16 <= d <= 256 and (d & (d - 1)) == 0, "head_dim 需为 [16,256] 内 2 的幂"
    if causal:
        assert n_kv >= n_q, "causal 时要求 kv 不短于 q"
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d)

    o = torch.empty_like(q)
    g = h // h_kv

    BR, BC = (64, 64) if d >= 128 else (128, 64)
    grid = (triton.cdiv(n_q, BR), b * h)
    flash_attn_kernel[grid](q, k, v, o,
                            *q.stride(), *k.stride(), *v.stride(), *o.stride(),
                            n_q, n_kv, h, g, sm_scale,
                            D=d, BR=BR, BC=BC, CAUSAL=causal,
                            num_warps=4, num_stages=2)
    return o


# TODO: 把下面这个 name 改成能区分你实现方式的唯一名字，
# 例如 "zhangsan_flash_attention (triton)" / "lisi_flash_attention (cuda)"
register("chenhonghua_flash_attention (triton)", attention)
