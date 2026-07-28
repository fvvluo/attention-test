# FA3-Hopper 算子接入：prefill 走打包在本仓库 attention-test/hopper/ 的 SM90
# kernel，decode 走自研 Triton flash-decoding kernel（针对 H20 调优）。
#
# 接入说明：
#   - attention-test/hopper/ 是简化版 FlashAttention-3（仅前向、推理用，
#     SM90/Hopper，支持 head_dim 64/128/256、MQA/GQA、连续或 paged KV-Cache），
#     已随本仓库一起打包（含编译好的 flash_attn_3/_C*.so），无外部路径依赖。
#   - benchmark 统一用 BHSD 布局 (batch, heads, seq_len, head_dim)，hopper 接口用
#     BSHD 布局 (batch, seq_len, heads, head_dim)，这里做 transpose 适配（纯 view，
#     无数据拷贝）。
#   - prefill 阶段（q_len == kv_len）走 flash_attn_func，FA3 的 prefill kernel
#     在 H20 上表现已经很好，保持不变。
#   - decode 阶段（q_len < kv_len，本框架固定 q_len=1）是自回归生成场景，
#     瓶颈是 KV-Cache 的显存带宽。FA3 的 decode 实际复用 prefill 形态的 kernel：
#     kBlockM=128 的 M-tile 里只有 group*q_len（评测形状下为 8）行是真实数据，
#     16 倍的 M 维 padding 让 kernel 卡在 WGMMA 流水上（H20 实测显存利用率仅
#     ~10%）。因此 decode 改用下面的 Triton flash-decoding kernel：
#       * BLOCK_M 收紧到 16（消除 padding 浪费），split-KV 多段并行 + 在线
#         softmax 跨 split 规约合并，num_splits 按形状首次调用时实测选优；
#       * K/V 用 host 侧 4D TMA descriptor 加载（kernel 内零描述符开销，
#         越界自动补 0 免 mask），比 cp.async 高约 3%；
#       * PDL 链式重叠（SM90 programmatic dependent launch）：combine 在 split
#         收尾时即被预启动，相邻两次调用也互相重叠，隐藏 launch 延迟；
#       * combine 按 32-split 分块向量化，workspace 跨调用复用。
#     空闲 H20 上评��形状端到端实测 ~3.15-3.18 TB/s（该机实测带宽上限
#     ~3.3-3.6 TB/s），比 FA3 decode 快 ~3x。注意：本机 GPU 常被其他容器
#     租户共享，GPU 有占用时该数值会大幅下跌，测性能请先挑空闲卡
#     （nvidia-smi 看 utilization）。
#   - Triton 路径不满足条件时（head_dim 非 2 的幂、pack_rows > 128、q 布局
#     不可折叠等）自动回退 FA3 flash_attn_with_kvcache；也可用环境变量
#     FA3_DECODE_BACKEND=fa3 强制回退。
#   - decode 的 num_splits 对性能影响很大，默认用内置启发式（目标 ~5 倍 SM 数
#     的 CTA），可用环境变量 FA3_DECODE_NUM_SPLITS 覆盖做调优。
#   - decode 阶段框架传 causal=False（单个新 token 位于序列末尾，等价于能看到全部
#     缓存）；Triton kernel 内部实现了 bottom-right 对齐的因果掩码，q_len>1 的
#     增量 decode 传 causal=True 时语义也与 FA3 一致。

import os
import sys
from pathlib import Path

import torch

from .base import register

# hopper/ 运行时代码已打包进本仓库 attention-test/hopper/（含编译好的
# flash_attn_3/_C*.so），不再依赖仓库外的 ../../hopper。手动加到 sys.path：
# ops/fa3_hopper_flash_attention.py -> parents[1] == attention-test/。
# 注意必须先 import torch 再 import flash_attn_3._C（依赖 libc10.so），上面的
# `import torch` 已经保证了顺序。
_HOPPER_DIR = Path(__file__).resolve().parents[1] / "hopper"
if str(_HOPPER_DIR) not in sys.path:
    sys.path.insert(0, str(_HOPPER_DIR))

from flash_attn_interface import flash_attn_func, flash_attn_with_kvcache  # noqa: E402

_DECODE_NUM_SPLITS = int(os.environ.get("FA3_DECODE_NUM_SPLITS", "0"))
_DECODE_BACKEND = os.environ.get("FA3_DECODE_BACKEND", "triton")

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda.gdc import gdc_launch_dependents, gdc_wait
    from triton.tools.tensor_descriptor import TensorDescriptor

    _HAS_TRITON = True
except Exception:  # noqa: BLE001
    _HAS_TRITON = False


# PDL（programmatic dependent launch，SM90+）：combine kernel 在 split kernel
# 收尾阶段就被预启动（gdc_wait 等待上游 grid 完成、内存可见），下一次调用的
# split kernel 也能与上一次 combine 重叠，隐藏两次 kernel launch 延迟。
# 懒检测：第一次调用时按 device capability 决定是否启用。
_USE_PDL = None


def _use_pdl(device) -> bool:
    global _USE_PDL
    if _USE_PDL is None:
        try:
            major, _minor = torch.cuda.get_device_capability(device)
            _USE_PDL = major >= 9
        except Exception:  # noqa: BLE001
            _USE_PDL = False
    return _USE_PDL


if _HAS_TRITON:

    @triton.jit
    def _decode_split_kernel(
        Q, K_DESC, V_DESC, OutP, MlkP,
        sm_scale,
        kv_len, q_len, kv_heads, num_splits,
        pack_rows,  # group * q_len（M 维真实行数）
        stride_qb, stride_qh, stride_qm,  # Q 视为 (b, kv_heads, pack_rows, d) 的步长
        stride_ob, stride_oh, stride_os, stride_om,
        stride_eb, stride_eh, stride_es, stride_em,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HDIM: tl.constexpr,
        CAUSAL: tl.constexpr,
        USE_GDC: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)  # b * kv_heads + h_kv
        pid_s = tl.program_id(1)   # split 编号
        b = pid_bh // kv_heads
        h = pid_bh % kv_heads

        offs_m = tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HDIM)
        m_mask = offs_m < pack_rows

        # 打包行 r = g*q_len + t，对应 q-head (h*group + g)、q 位置 t
        q_ptrs = Q + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :]
        q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

        # 本 split 负责的 KV 区间 [n_start, n_end)
        n_per = tl.cdiv(kv_len, num_splits)
        n_start = pid_s * n_per
        n_end = tl.minimum(n_start + n_per, kv_len)

        m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, HDIM), tl.float32)

        for n0 in range(n_start, n_end, BLOCK_N):
            # K/V 用 2D TMA descriptor 加载：TMA 直接把 (BLOCK_N, HDIM) tile 以
            # tensor-core 友好的 swizzle layout 搬进 shared，供 tl.dot 使用；越界自动
            # 补 0，无需 mask。（实测：改用 1D 连续 tl.load 虽然纯读更快，但完整 kernel
            # 里 tl.dot 需要 TMA 的 2D swizzle 布局，1D load 反而更慢，故保留 TMA。）
            k = K_DESC.load([b, h, n0, 0]).reshape(BLOCK_N, HDIM)
            s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale
            n_mask = (n0 + tl.arange(0, BLOCK_N)) < n_end
            s = tl.where(n_mask[None, :], s, float("-inf"))
            if CAUSAL:
                # bottom-right 对齐：行 r 的 q 位置 t = r % q_len，
                # 可见 kv 位置 n <= kv_len - q_len + t（与 FA3 的 causal 语义一致）
                t = offs_m % q_len
                causal_ok = (n0 + tl.arange(0, BLOCK_N))[None, :] <= (kv_len - q_len) + t[:, None]
                s = tl.where(causal_ok, s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            v = V_DESC.load([b, h, n0, 0]).reshape(BLOCK_N, HDIM)
            acc += tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
            m_i = m_new

        if USE_GDC:
            # PDL 链式重叠：下一次调用的本 kernel 可能与上一次调用的 combine
            # 并发执行，二者读写同一份 workspace —— 写部分结果前等上游 grid
            # 全部完成（内存可见），避免竞争
            gdc_wait()
        # 写部分结果：未归一化的 acc，以及 (m_i, l_i) 供 combine 做跨 split 规约
        o_ptrs = OutP + b * stride_ob + h * stride_oh + pid_s * stride_os + offs_m[:, None] * stride_om + offs_d[None, :]
        tl.store(o_ptrs, acc, mask=m_mask[:, None])
        e_ptrs = MlkP + b * stride_eb + h * stride_eh + pid_s * stride_es + offs_m * stride_em
        tl.store(e_ptrs, m_i, mask=m_mask)
        tl.store(e_ptrs + stride_em * BLOCK_M, l_i, mask=m_mask)
        if USE_GDC:
            # 让 dependent（combine kernel）尽早预启动
            gdc_launch_dependents()

    @triton.jit
    def _decode_combine_kernel(
        OutP, MlkP, Out,
        num_splits,
        kv_heads,
        stride_ob, stride_oh, stride_os, stride_om,
        stride_eb, stride_eh, stride_es, stride_em,
        stride_fb, stride_fh, stride_fm,
        BLOCK_M: tl.constexpr,
        HDIM: tl.constexpr,
        MAX_SPLITS: tl.constexpr,
        SPLIT_CHUNK: tl.constexpr,
        USE_GDC: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)  # b * kv_heads + h_kv
        pid_m = tl.program_id(1)   # pack 行号（grid 第二维 == pack_rows）
        b = pid_bh // kv_heads
        h = pid_bh % kv_heads

        if USE_GDC:
            # 先放行下一次调用的 split kernel 预启动（它写 workspace 前会
            # gdc_wait 等本 grid 完成），再等上游 split grid 完成
            gdc_launch_dependents()
            gdc_wait()

        offs_d = tl.arange(0, HDIM)
        sp_offs = tl.arange(0, MAX_SPLITS)
        sp_mask = sp_offs < num_splits

        # 全局规约因子：m_max 与 l_sum（各 split 在线 softmax 的 (m, l) 合并）
        e_ptrs = MlkP + b * stride_eb + h * stride_eh + sp_offs * stride_es + pid_m * stride_em
        m_s = tl.load(e_ptrs, mask=sp_mask, other=float("-inf"))
        l_s = tl.load(e_ptrs + stride_em * BLOCK_M, mask=sp_mask, other=0.0)
        m_max = tl.max(m_s, 0)
        l_sum = tl.sum(tl.where(sp_mask, tl.exp(m_s - m_max), 0.0) * l_s, 0)
        inv_l = 1.0 / tl.maximum(l_sum, 1e-30)

        # 按 SPLIT_CHUNK 分块做 2D 向量化加载与加权求和，
        # 避免逐 split 标量循环（ns 较大时寄存器也不会爆）。
        # 注意分子权重只有 exp(m_s - m_max)：部分和 acc_s 是未归一化的
        # Σ exp(x - m_s)·v，本身已含 l_s；l_s 只用于上面的分母 l_sum。
        acc = tl.zeros((HDIM,), tl.float32)
        for s0 in range(0, num_splits, SPLIT_CHUNK):
            offs_s = s0 + tl.arange(0, SPLIT_CHUNK)
            s_mask = offs_s < num_splits
            ec_ptrs = MlkP + b * stride_eb + h * stride_eh + offs_s * stride_es + pid_m * stride_em
            m_c = tl.load(ec_ptrs, mask=s_mask, other=float("-inf"))
            w = tl.where(s_mask, tl.exp(m_c - m_max) * inv_l, 0.0)
            o_ptrs = (OutP + b * stride_ob + h * stride_oh
                      + offs_s[:, None] * stride_os + pid_m * stride_om + offs_d[None, :])
            o = tl.load(o_ptrs, mask=s_mask[:, None], other=0.0)
            acc += tl.sum(o * w[:, None], 0)
        f_ptrs = Out + b * stride_fb + h * stride_fh + pid_m * stride_fm + offs_d
        tl.store(f_ptrs, acc.to(Out.dtype.element_ty))


# Triton decode 的最佳配置（H20 上扫出来的），可用环境变量覆盖
_DECODE_BLOCK_N = int(os.environ.get("FA3_DECODE_BLOCK_N", "64"))
_DECODE_NUM_WARPS = int(os.environ.get("FA3_DECODE_NUM_WARPS", "4"))
_DECODE_NUM_STAGES = int(os.environ.get("FA3_DECODE_NUM_STAGES", "2"))

# split 部分结果 workspace 缓存：内部临时缓冲区，按形状复用，
# 避免每次调用两次 torch.empty（caching allocator 也有 ~5us 开销）。
# 输出 tensor 每次新分配（返回给调用方，不能复用）。
_WS_CACHE: dict = {}


def _get_workspace(batch, kv_heads, num_splits, block_m, d, device):
    key = (batch, kv_heads, num_splits, block_m, d, str(device))
    ws = _WS_CACHE.get(key)
    if ws is None:
        if len(_WS_CACHE) > 64:
            _WS_CACHE.clear()
        # out_p 存各 split 的未归一化部分和 acc（分子），用 bf16 而非 fp32：
        # split kernel 写、combine kernel 读这份 workspace 是除 KV 之外的主要额外
        # 流量，bf16 把它的读写字节减半，降低写放大。split kernel 的 tl.store 会把
        # fp32 acc 自动转 bf16 存；combine 的 tl.load 得到 bf16 后与 fp32 权重相乘
        # 自动升 fp32 归约，数值无损到影响精度（实测 max abs err 3e-5，远内于容差）。
        # (m, l) 统计量对数值敏感，仍保持 fp32。
        out_p = torch.empty((batch, kv_heads, num_splits, block_m, d),
                            dtype=torch.bfloat16, device=device)
        mlk_p = torch.empty((batch, kv_heads, num_splits, 2 * block_m),
                            dtype=torch.float32, device=device)
        ws = (out_p, mlk_p)
        _WS_CACHE[key] = ws
    return ws


def _launch_decode(q, k, v, out, out_p, mlk_p, causal, sm_scale,
                   q_len, kv_len, kv_heads, num_splits, pack_rows,
                   sqb, sqh, sqm, block_m, block_n):
    b = q.shape[0]
    d = q.shape[3]
    pdl = _use_pdl(q.device)
    # K/V 的 (b, kv_heads, kv_len, d) 4D TMA descriptor，box (1,1,BLOCK_N,HDIM)，
    # host 侧创建一次，kernel 内零描述符开销（比 device 侧创建更快）
    k_desc = TensorDescriptor.from_tensor(k, block_shape=[1, 1, block_n, d])
    v_desc = TensorDescriptor.from_tensor(v, block_shape=[1, 1, block_n, d])
    # Triton launch 依赖当前 CUDA context，调用方未必把当前设备设为输入所在卡
    with torch.cuda.device(q.device):
        _decode_split_kernel[(b * kv_heads, num_splits)](
            q, k_desc, v_desc, out_p, mlk_p,
            sm_scale, kv_len, q_len, kv_heads, num_splits, pack_rows,
            sqb, sqh, sqm,
            out_p.stride(0), out_p.stride(1), out_p.stride(2), out_p.stride(3),
            mlk_p.stride(0), mlk_p.stride(1), mlk_p.stride(2), mlk_p.stride(3),
            BLOCK_M=block_m, BLOCK_N=block_n, HDIM=d, CAUSAL=causal, USE_GDC=pdl,
            num_warps=_DECODE_NUM_WARPS, num_stages=_DECODE_NUM_STAGES,
            launch_pdl=pdl,
        )
        _decode_combine_kernel[(b * kv_heads, pack_rows)](
            out_p, mlk_p, out,
            num_splits, kv_heads,
            out_p.stride(0), out_p.stride(1), out_p.stride(2), out_p.stride(3),
            mlk_p.stride(0), mlk_p.stride(1), mlk_p.stride(2), mlk_p.stride(3),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_M=block_m, HDIM=d, MAX_SPLITS=triton.next_power_of_2(num_splits),
            SPLIT_CHUNK=32, USE_GDC=pdl,
            num_warps=4,
            launch_pdl=pdl,
        )


# num_splits 调优结果缓存：形状 -> 最优 ns。
# 固定启发式在 H20 上不够稳（评测形状下 ns=48 与 49 差 ~15%，wave 量化敏感），
# 因此每个新形状首次调用时在真实输入上对一小组候选做快速实测，
# 之后命中缓存零开销（benchmark 的 warmup 阶段会吸收调优成本）。
_NS_CACHE: dict = {}


def _autotune_num_splits(q, k, v, causal, sm_scale, q_len, kv_len, kv_heads,
                         pack_rows, sqb, sqh, sqm, block_m, block_n, candidates):
    b, d = q.shape[0], q.shape[3]
    scratch = torch.empty((b, kv_heads, pack_rows, d), dtype=q.dtype, device=q.device)
    best_ns, best_t = candidates[0], float("inf")
    with torch.cuda.device(q.device):
        for ns in candidates:
            out_p, mlk_p = _get_workspace(b, kv_heads, ns, block_m, d, q.device)

            def fn():
                _launch_decode(q, k, v, scratch, out_p, mlk_p, causal, sm_scale,
                               q_len, kv_len, kv_heads, ns, pack_rows,
                               sqb, sqh, sqm, block_m, block_n)

            fn()
            torch.cuda.synchronize(q.device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            fn()
            fn()
            end.record()
            torch.cuda.synchronize(q.device)
            t = start.elapsed_time(end)
            if t < best_t:
                best_t, best_ns = t, ns
    return best_ns


def _triton_decode(q, k, v, causal, sm_scale):
    """Triton flash-decoding decode。输入输出均为 BHSD。"""
    b, q_heads, q_len, d = q.shape
    kv_heads, kv_len = k.shape[1], k.shape[2]
    group = q_heads // kv_heads
    pack_rows = group * q_len
    block_m = max(16, triton.next_power_of_2(pack_rows))
    if sm_scale is None:
        sm_scale = d ** -0.5

    # q (b, q_heads, q_len, d) 折叠为 (b, kv_heads, group*q_len, d) 的视图：
    # 要求 stride_h == q_len * stride_m（连续 BHSD 天然满足），此时
    # 行 r = g*q_len + t 的地址 = h*(group*stride_h) + r*stride_m。
    sqb, sqh, sqm = q.stride(0), group * q.stride(1), q.stride(2)

    block_n = _DECODE_BLOCK_N

    tune_key = (b, q_heads, q_len, kv_heads, kv_len, d, q.dtype, causal, str(q.device))
    num_splits = _NS_CACHE.get(tune_key)
    if num_splits is None:
        if _DECODE_NUM_SPLITS > 0:
            num_splits = _DECODE_NUM_SPLITS
        else:
            kv_blocks = (kv_len + block_n - 1) // block_n
            candidates = sorted({min(c, kv_blocks) for c in
                                 (1, 8, 16, 24, 32, 40, 48, 64, 96)})
            num_splits = _autotune_num_splits(
                q, k, v, causal, sm_scale, q_len, kv_len, kv_heads,
                pack_rows, sqb, sqh, sqm, block_m, block_n, candidates)
        _NS_CACHE[tune_key] = num_splits

    dev = q.device
    out_p, mlk_p = _get_workspace(b, kv_heads, num_splits, block_m, d, dev)
    out = torch.empty((b, kv_heads, pack_rows, d), dtype=q.dtype, device=dev)

    _launch_decode(q, k, v, out, out_p, mlk_p, causal, sm_scale,
                   q_len, kv_len, kv_heads, num_splits, pack_rows,
                   sqb, sqh, sqm, block_m, block_n)
    # out (b, kv_heads, group*q_len, d) -> (b, q_heads, q_len, d)
    return out.view(b, kv_heads, group, q_len, d).view(b, q_heads, q_len, d)


def _triton_decode_supported(q, k, v) -> bool:
    """检查 Triton decode 路径是否适用于当前输入，不适用则回退 FA3。"""
    if not _HAS_TRITON or _DECODE_BACKEND == "fa3":
        return False
    if q.dtype not in (torch.float16, torch.bfloat16):
        return False
    b, q_heads, q_len, d = q.shape
    kv_heads = k.shape[1]
    if q_heads % kv_heads != 0:
        return False
    # HDIM 作为 tl.arange 的 constexpr 必须是 2 的幂
    if d < 16 or d > 256 or (d & (d - 1)) != 0:
        return False
    pack_rows = (q_heads // kv_heads) * q_len
    if pack_rows > 128:
        return False
    # q 的 (q_heads, q_len) 两维需可折叠为一维 pack_rows
    if q_len > 1 and q.stride(1) != q_len * q.stride(2):
        return False
    if k.stride(-1) != 1 or v.stride(-1) != 1:
        return False
    # K/V 走 TMA：基地址 16B 对齐，且除最后一维外各 stride 换算成字节需是
    # 16B 的倍数（fp16/bf16 元素 2B，即 stride 需是 8 的倍数）
    for t in (k, v):
        if t.data_ptr() % 16 != 0:
            return False
        if any(t.stride(i) % 8 != 0 for i in range(3)):
            return False
    return True


def attention(q, k, v, causal=True, sm_scale=None):
    """FA3-Hopper prefill + Triton flash-decoding decode，自动分流。

    Args:
        q: shape (batch, q_heads, q_len, head_dim)
        k, v: shape (batch, kv_heads, kv_len, head_dim)，q_heads 是 kv_heads 的
            整数倍（GQA 原生支持，无需手动 repeat）。
        causal: 因果掩码（prefill 由 --causal 控制；decode 框架固定传 False）
        sm_scale: softmax 缩放系数，None 时用默认 1/sqrt(head_dim)

    Returns:
        output: shape 与 q 相同
    """
    q_len, kv_len = q.shape[2], k.shape[2]

    if q_len != kv_len and _triton_decode_supported(q, k, v):
        # decode：q_len=1（或很短的增量）attend 整段 KV-Cache，访存密集，
        # 走 Triton flash-decoding kernel（H20 上比 FA3 decode 快约 3.3x）
        return _triton_decode(q, k, v, causal, sm_scale)

    # BHSD -> BSHD（transpose 后最后一维 stride 仍为 1，满足 kernel 要求）
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)

    if q_len == kv_len:
        # prefill：一次性处理完整 prompt
        out = flash_attn_func(q_t, k_t, v_t, softmax_scale=sm_scale, causal=causal)
    else:
        # 回退路径：Triton decode 不适用时用 FA3 kvcache
        out = flash_attn_with_kvcache(
            q_t,
            k_t,
            v_t,
            cache_seqlens=kv_len,
            softmax_scale=sm_scale,
            causal=causal,
            num_splits=_DECODE_NUM_SPLITS,
        )

    return out.transpose(1, 2)


register("fa3_hopper (sm90 cuda)", attention)
