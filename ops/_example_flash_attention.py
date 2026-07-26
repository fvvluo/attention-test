# 示例算子：纯 PyTorch 实现的 online-softmax（分块流式 softmax）FlashAttention
#
# 这是一个"教学版"实现，帮助团队成员理解 FlashAttention 的核心算法思想：
# 不必先算出完整的 (S, S) attention 矩阵，而是按 K/V 的 block 依次遍历，
# 维护一个 running max(m)、running sum(l) 和输出累积(o)，
# 每来一个新 block 就用在线 softmax 公式把之前的结果"重新缩放"后累加进去。
#
# 注意：这里用的是纯 PyTorch 算子拼出来的版本，主要目的是验证算法正确性，
# 并不会比 torch 自带的 scaled_dot_product_attention 更快（甚至会更慢，
# 因为没有用到 CUDA kernel 级别的 fusion）。团队成员可以参考这个实现的
# 分块逻辑，替换成自己的 Triton / CUDA 版本。

import math

import torch

from .base import register


def attention(q, k, v, causal=True, sm_scale=None, block_size=128):
    """纯 PyTorch online-softmax 版 FlashAttention，支持 GQA。

    Args:
        q: shape (batch, q_heads, seq_len, head_dim)
        k, v: shape (batch, kv_heads, seq_len, head_dim)，其中 q_heads 必须是
            kv_heads 的整数倍（标准 MHA 时 q_heads == kv_heads）
        causal: 是否使用因果掩码（只看当前位置及之前的 token）
        sm_scale: softmax 缩放系数，默认为 1/sqrt(head_dim)
        block_size: K/V 分块大小

    Returns:
        output: shape 与 q 相同，(batch, q_heads, seq_len, head_dim)
    """
    batch, q_heads, q_len, head_dim = q.shape
    kv_heads = k.shape[1]
    kv_len = k.shape[2]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    if kv_heads != q_heads:
        # GQA：把 k/v 的 head 维度按 group 展开成 q_heads，group 内的多个
        # q head 共享同一份 kv（repeat_interleave 保证与常见 GQA 分组约定
        # 一致：q_heads 里连续 group 个 head 对应同一个 kv head）。
        group = q_heads // kv_heads
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)

    device = q.device
    dtype = q.dtype

    # running 统计量：
    # m: 每个 query 位置当前见过的最大 score（用于数值稳定）
    # l: 每个 query 位置当前的 softmax 分母（未归一化的 exp 和）
    # o: 每个 query 位置当前的输出累积（未归一化）
    m = torch.full((batch, q_heads, q_len, 1), float("-inf"), device=device, dtype=dtype)
    l = torch.zeros((batch, q_heads, q_len, 1), device=device, dtype=dtype)
    o = torch.zeros_like(q)

    # 分块数量取决于 kv 的长度（而不是 q 的长度）：
    # decode 阶段 q_len=1、kv_len=完整 KV-Cache 长度，仍需要把 kv 分成多个
    # block 依次遍历，否则会漏掉大部分 key/value（这是本文件曾经存在的 bug）。
    num_blocks = math.ceil(kv_len / block_size)

    # decode（q_len < kv_len）场景下的因果对齐偏移：query 位置 i 在"全局序列"里
    # 实际对应的绝对位置是 (kv_len - q_len + i)，即最后 q_len 个 token；
    # 这样 prefill（q_len == kv_len，offset=0）和 decode 都能复用同一套掩码逻辑。
    causal_offset = kv_len - q_len

    for block_idx in range(num_blocks):
        kv_start = block_idx * block_size
        kv_end = min(kv_start + block_size, kv_len)

        k_block = k[:, :, kv_start:kv_end, :]  # (batch, heads, block, head_dim)
        v_block = v[:, :, kv_start:kv_end, :]

        # 当前 block 的 attention score: (batch, heads, q_len, block)
        scores = torch.matmul(q, k_block.transpose(-2, -1)) * sm_scale

        if causal:
            # 因果掩码：query 位置 i（绝对位置 i + causal_offset）只能看到
            # 绝对位置 j <= i + causal_offset 的 key。
            q_idx = torch.arange(q_len, device=device).unsqueeze(-1) + causal_offset  # (q_len, 1)
            kv_idx = torch.arange(kv_start, kv_end, device=device).unsqueeze(0)  # (1, block)
            mask = kv_idx > q_idx  # True 表示需要屏蔽
            if mask.any():
                scores = scores.masked_fill(mask, float("-inf"))
                # 如果这个 block 对某个 query 位置全部被屏蔽，直接跳过后续更新（避免 NaN）
                if torch.all(mask):
                    continue

        # 当前 block 内的最大值，用于在线更新 running max
        block_max = scores.max(dim=-1, keepdim=True).values  # (batch, heads, q_len, 1)
        new_m = torch.maximum(m, block_max)

        # 把旧的 running sum / output 按新的 max 重新缩放
        alpha = torch.exp(m - new_m)  # 旧统计量的缩放系数
        alpha = torch.where(torch.isinf(m), torch.zeros_like(alpha), alpha)

        # 当前 block 的 exp(score - new_m)
        p = torch.exp(scores - new_m)  # (batch, heads, q_len, block)

        new_l = l * alpha + p.sum(dim=-1, keepdim=True)
        new_o = o * alpha + torch.matmul(p, v_block)

        m, l, o = new_m, new_l, new_o

    # 归一化，避免除 0（理论上 l 不应为 0，除非某行被全部因果屏蔽，这里做个保护）
    l_safe = torch.where(l == 0, torch.ones_like(l), l)
    output = o / l_safe

    return output


# 注：本文件名以 "_" 开头，不会被 ops/__init__.py 自动扫描导入，
# 所以这行 register() 实际不会执行——仅保留作为参考实现的完整示例。
register("example", attention)
