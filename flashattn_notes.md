# flashattn.py 架构演进笔记

记录 `flashattn.py`（Triton 版 FlashAttention forward，教学 + 实用）从初版骨架到
当前架构的全部优化过程、实测数据与经验教训，供后续迭代参考。

- 硬件：NVIDIA H20（78 SMs，HBM ~4 TB/s），测试统一用 GPU 5
- 环境：base `python3`（triton 3.7.1）或 conda env `chh`（triton 3.6.0），结果一致
- 自测：`python3 flashattn.py`（12 组正确性用例 vs FP32 参考 + prefill/decode 基准）

## 当前架构

```
attention(q, k, v)                      # 统一入口，按形状分发
├── prefill(q, k, v)                    # n_q > 16 或 kv 较短
│   └── flash_attn_kernel               # grid=(cdiv(n_q,BR), b*h)
│       └── _fa_inner x2                # 两段式：无掩码整块 + 对角线掩码块
└── decode(q, k, v)                     # n_q <= 16 且 g*n_q <= 128 且 kv >= 1024
    └── _decode_fused_kernel            # grid=(splits, b*h_kv)，GQA 组共享，
                                        # partial + 信号量触发的 merge 融合为单次启动
```

要点：TMA 描述符（`tl.make_tensor_descriptor`，device 端创建）、FP32 累加器、
`exp2`（`qk_scale = sm_scale * log2e`）、decode partial 的 o 暂存为 fp16（m/l 为
fp32）、prefill kernel 与 decode fused kernel 均带 `@triton.autotune`，
decode 的 splits 由 `_tune_decode_splits` 实测缓存。

## 演进时间线（实测数据）

| 阶段 | prefill n=4096 | decode kv=8192 | decode kv=131072 |
|---|---|---|---|
| 初版骨架（修复后） | 4.80 ms | 96-118 GB/s（走 prefill kernel） | — |
| ① 两段式因果循环 | **2.20 ms**（2.2x） | 同左 | — |
| ② decode split-K（partial+merge） | 2.21 | 363 GB/s | — |
| ③ autotune | 2.21 | 368 | — |
| ④ exp2 + 折入 log2e | 2.13 | 371 | — |
| ⑤ TMA 描述符替换 make_block_ptr | 2.23 | 405 | — |
| ⑥ prefill/decode 硬拆分 + GQA 组共享 | 2.23 | **954 GB/s** | — |
| ⑦ CPU 优化（见下） | 2.23 | 1393 GB/s（24 µs/次） | 2627 GB/s |
| ⑧ autotune 回归 + chunk 入 key + splits 调优 | 2.23 | — | **2929 GB/s** |
| ⑨ 回退两 kernel | 2.22 | 1085 GB/s（41 µs/次） | 2913 GB/s |
| ⑩ fp16 partials | 2.25 | 839 | 2889 GB/s |
| ⑪ 重新融合单 kernel（信号量，当前） | 2.23 | **1054 GB/s（32 µs/次）** | 2892 GB/s |

参考：torch SDPA prefill 2.12 ms；decode kv=8192 ~4080 GB/s（有效）。

各阶段内容：

1. **两段式因果循环**：kv 循环拆成"完全可见块（无掩码）+ 对角线/尾部块（掩码）"，
   causal 下省掉约一半无效计算。收益最大（2.2x）。
2. **decode split-K（FlashDecoding）**：decode 时 q_len=1，把 kv 切成 splits 段并行
   算 partial (o,m,l)，第二个 kernel 用 log-sum-exp 合并。
3. **autotune**：prefill 对 BR/BC/warps/stages 扫配置空间，key=['n_q','n_kv','D']。
4. **exp2**：硬件 exp2 指令，`qk_scale` 预乘 log2(e)，全 kernel 替换。约 4%。
5. **TMA 描述符**：`tl.make_tensor_descriptor` 替代废弃的 make_block_ptr，
   越界自动补零/裁剪；host 侧 `triton.set_allocator` 提供暂存（已缓存复用）。
6. **GQA 组共享（decode 关键）**：grid 从 (splits, b*h) 改为 (splits, b*h_kv)，
   同一 kv 组的 g 个 q head 拼成 [M_PAD, D] tile 共享一次 K/V 读取。
   之前 K/V 被读 g=8 遍（"有效带宽 405"但实际 HBM 已 2.9 TB/s 打满），
   共享后有效=实际，405→954 GB/s。
7. **CPU 开销治理**：profiling 发现短 kv 时 CPU 发射耗时 57 µs > GPU 25 µs
   （CPU-bound）。措施：暂存 buffer 缓存、kernel 内由形状推导 stride（去掉 12 个
   stride 实参）、TMA 暂存复用。曾用"信号量 + 最后一 CTA merge"把两次启动并成
   一次（CPU 57→24 µs），后在 ⑨ 回退。
8. **autotune 回归与 splits 调优**（回退 bug 的修复，见教训 3/4）。
9. **回退两 kernel**：当时实测融合与两 kernel 性能一致（2562 vs 2543 GB/s
   @128K），曾回退为两 kernel 以求结构简单。
10. **fp16 partials**：o_part fp32→fp16（m/l 保持 fp32），partial 写与 merge 读
    流量减半；误差 ~3e-5 → ~5e-5（容差 2e-2 无虞），128K 约 +1%。
11. **重新融合单 kernel（信号量，当前）**：短 kv 下省一次 launch（41→32 µs/次），
    长 kv 无差异（2892 GB/s）。组内最后一个 program 用 acq_rel 原子计数判断后
    做 merge，并自复位计数器（免 host 侧 zero_ 启动）。结论：信号量融合在
    CPU-bound 场景是稳定正收益，故作为最终架构保留。

## 关键教训

1. **先诊断再优化**。decode 曾经的最大瓶颈不是 kernel 而是 CPU launch
   （57 µs vs GPU 25 µs）；不 profiling 就会优化错地方。区分
   `do_bench`（含 L2 清理、暴露单次耗时）与 back-to-back amortized
   （CPU/GPU 流水重叠）两种测量。
2. **GQA decode 的带宽大头是 K/V 重复读**。"有效带宽低"不等于"没打满 HBM"：
   按 q head 各读一遍时 HBM 早已打满，组共享才是正解。
3. **autotune 的 key 必须覆盖所有影响最优配置的参数**。曾经 key 缺 `chunk`：
   autotuner 在第一个 splits 候选上选出 BC=64/st=4，沿用到所有 splits，导致
   128K 从 2900 掉到 2600 GB/s；key 加入 `chunk` 后恢复并超过（2929）。
4. **splits 是 host 侧 grid 参数，triton.autotune 调不到**。固定公式（2 CTA/SM）
   落在 20，而峰值在 ~26-27（3023 GB/s partial-only）。现由
   `_tune_decode_splits` 对 {1.5,2,2.7,4}×SM/BHK 候选实测一次并缓存。
5. **信号量融合是稳定的短-kv 正收益**：省一次 launch（~8-10 µs/次），长 kv 无
   差异；代价是内存序复杂度（acq_rel + debug_barrier + 计数器自复位）。
   ⑨⑪ 两次反复说明：架构取舍要按目标负载（短 vs 长 kv）决定，并留下实测记录。
6. **自测也会骗人**：`_decode_run` 曾误用未传入的 `causal`，靠 `__main__` 循环
   变量泄漏成模块全局才"通过"；独立脚本复现才暴露。

## 已知差距与后续方向

- decode 短 kv（8192）距 SDPA 仍 ~4x：SDPA 单次 C++ 启动 ~11 µs CPU + 深度调优
  的 CUDA kernel。Triton 内继续压只能靠 CUDA Graph（用户侧捕获）或 C++ launcher。
- decode 长 kv ~2900 GB/s ≈ HBM 峰值的 73%（实际可持续带宽的 ~85%）。
  **persistent decode kernel 已实现并实测：回退**。grid-stride 复用描述符 +
  固定 CTA 数的版本峰值 2988 GB/s（partial-only, splits=39），而"每工作项一个
  program"的简单结构为 3039 GB/s（splits=27）——独立 program 的前言开销已被
  硬件调度器互相掩盖，grid-stride 反而在工作项边界暴露流水线排水。完整
  A/B 数据见 /tmp/ab_persist.py 的实验记录（2026-07-26）。
- **fp16 partials 已实现**（2026-07-26）：o_part 由 fp32 改 fp16（m/l 保持 fp32），
  partial 写与 merge 读流量减半；decode 各用例误差由 ~3e-5 升到 ~5e-5（容差
  2e-2 无虞），128K 约 2860 → 2889 GB/s（~1%，merge 本来只占 ~4%）。
- 剩余方向：更长 chunk + 更深 num_stages（少 CTA 也能保持字节在途）。
- prefill 2.22 ms 已与 SDPA 持平；再进一步需要 GQA 感知的 L2 调度与
  warp specialization（Triton 表达不了，需 CUDA/CuTe DSL，即 flash-attention-baseline
  里 cute 实现的路线）。
