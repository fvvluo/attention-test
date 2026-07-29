# CuteDSL 转置-WGMMA decode 算子（bf16, H20 SM90a）。
#
# decode（q_len==1）走转置-WGMMA split-KV kernel，bf16、无量化，官方 bench 实测
# ~3.55 TB/s 有效 KV 带宽（正确性 max_abs≈1e-4 PASS），已达 H20 HBM 硬件上限区。
#
# 核心设计：
#   - 转置 WGMMA：把 KV-rows(BLOCK_N) 和 head_dim 放到 tensor-core 的 M 维，
#     GQA GROUP=8 只落在小 N 维 —— 两个 GEMM 都满 M=128 效率（避免 GROUP=8 占 M 的浪费）。
#       GEMM1: S^T[BLOCK_N,8] = K[BLOCK_N,D] @ Q^T[D,8]     (kv 行在 M)
#       GEMM2: O^T[D,8]       = V^T[D,BLOCK_N] @ P^T[BLOCK_N,8] (head_dim 在 M)
#   - GQA group-packing：一个 CTA 处理共享同一 kv_head 的 8 个 q_head，每 KV 字节只读一次。
#   - 持久化 split-KV grid + grid-stride 遍历，规避 wave 量化；微型 combine kernel 跨 split 归约。
#   - warp-specialized 生产者/消费者：一个 warpgroup 发 K/V TMA，另一个跑双 WGMMA + 在线 softmax。
#   - PDL 重叠 combine、L2 evict_first(KV 只读一次不污染 L2)、V pipeline 更浅(num_stages_v=2)。

import sys
from pathlib import Path

from .base import register

_CUTE = Path(__file__).resolve().parents[1] / "cute_decode"
if str(_CUTE) not in sys.path:
    sys.path.insert(0, str(_CUTE))

_mod = None
def _get():
    global _mod
    if _mod is None:
        import flash_decode_transposed_wgmma as m
        _mod = m
    return _mod


def attention(q, k, v, causal=True, sm_scale=None):
    """decode(q_len==1) 走转置-WGMMA CuteDSL kernel；其他形状回退 SDPA（kernel 内部处理）。

    q: (batch, q_heads, q_len, head_dim), k/v: (batch, kv_heads, kv_len, head_dim)。
    """
    return _get().attention(q, k, v, causal=causal, sm_scale=sm_scale)


register("cute_transposed_wgmma (sm90 bf16)", attention)
