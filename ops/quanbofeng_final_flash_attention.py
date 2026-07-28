"""新版 attention-test 的自动注册适配层。

实际实现全部位于同目录的子包 ``ops/quanbofeng_final``：
- prefill：Hopper TMA/WGMMA Tensor Core kernel；
- decode：GQA-packed cp.async/M16 warp-MMA split-KV + combine kernel。
"""

from .quanbofeng_final import attention

from .base import register


register("quanbofeng_final (WGMMA prefill + Flash-Decoding)", attention)
