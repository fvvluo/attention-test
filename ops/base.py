# 算子注册中心
#
# 所有自定义 attention 算子都通过 register(name, fn) 注册到全局字典 OPS 中。
# 主脚本 bench_attention.py 只依赖 OPS 字典，新增算子时无需改动主脚本。

from typing import Callable, Dict

# 全局算子注册表：name -> 算子函数
OPS: Dict[str, Callable] = {}


def register(name: str, fn: Callable) -> None:
    """注册一个 attention 算子实现。

    Args:
        name: 算子名称，用于在结果表格中展示，必须全局唯一。
        fn: 算子函数，约定签名为
            fn(q, k, v, causal: bool = True, sm_scale: float = None) -> output
            其中 q 的 shape 为 (batch, q_heads, seq_len, head_dim)，k/v 的
            shape 为 (batch, kv_heads, seq_len, head_dim)，output 的 shape
            与 q 相同。标准 MHA 时 q_heads == kv_heads；GQA 时 q_heads 必须是
            kv_heads 的整数倍，算子实现内部需要自行把 k/v 的 head 维度
            broadcast/repeat 到 q_heads（可参考 _example_flash_attention.py）。

    Raises:
        ValueError: 如果 name 已经被注册过（防止团队成员重名冲突）。
    """
    if name in OPS:
        raise ValueError(
            f"算子名称 \"{name}\" 已经被注册过，请换一个唯一的名字。"
            f"当前已注册算子: {list(OPS.keys())}"
        )
    OPS[name] = fn
