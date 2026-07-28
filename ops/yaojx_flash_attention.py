
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


def attention(q, k, v, causal=True, sm_scale=None):
    """TODO: 替换成你自己的 FlashAttention 实现。

    Args:
        q: shape (batch, q_heads, seq_len, head_dim)
        k, v: shape (batch, kv_heads, seq_len, head_dim)，其中 q_heads 必须是
            kv_heads 的整数倍（GQA）；标准 MHA 时 q_heads == kv_heads。
            如果不需要支持 GQA，可以在函数开头假设 q_heads == kv_heads；
            如果要支持 GQA，需要自己把 k/v 的 head 维度 broadcast/repeat 到
            q_heads（可参考 _example_flash_attention.py 里的 repeat_interleave 用法）
        causal: 是否使用因果掩码（只看当前位置及之前的 token）
        sm_scale: softmax 缩放系数，默认为 1/sqrt(head_dim)，如果传 None 需要自己处理默认值

    Returns:
        output: shape 与 q 相同，(batch, q_heads, seq_len, head_dim)
    """
    raise NotImplementedError(
        "请把这个模板复制成新文件并实现你自己的 attention 算子，"
        "不要直接运行模板本身。"
    )



register("yaojixiu_flashattention(CuTe DSL)", attention)
