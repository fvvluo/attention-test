# FlashAttention 算子对比 Benchmark

这是一个方便团队成员接入自定义 FlashAttention 算子、并与官方 baseline 对比正确性与
性能的测试框架。baseline 按 FlashAttention-3 → FlashAttention-2 →
PyTorch 官方 `torch.nn.functional.scaled_dot_product_attention` 优先级自动降级
选取（具体降级条件见下方"输出说明"）。

## 目录结构

```
test/
├── README.md                       # 本文档
├── ops/
│   ├── __init__.py                 # 自动扫描并导入本目录下所有算子模块
│   ├── base.py                     # 算子注册接口 register(name, fn)
│   ├── _template.py                # 接入模板，复制改名即可开始写自己的算子
│   └── _example_flash_attention.py # 示例算子：纯 PyTorch online-softmax FlashAttention（仅供参考，不会被自动扫描注册）
└── bench_attention.py               # 主 benchmark 脚本
```

## 🚀 快速接入 TODO 清单（新成员从这里开始）

> 目标：3 步之内把你自己写的 FlashAttention 算子接入 benchmark，和 PyTorch 官方
> baseline 自动对比正确性与性能。全程**不需要改动任何已有文件**。

- [ ] **1. 复制模板**：把 `ops/_template.py` 复制一份到 `ops/` 目录下，改名为
      `<你的名字或算子名>_flash_attention.py`（**注意文件名不要以 `_` 开头**，
      否则会被自动扫描忽略——这也是为什么模板本身叫 `_template.py`，不会被误跑）。
- [ ] **2. 实现算子**：把模板里的 `attention(q, k, v, causal=True, sm_scale=None)`
      函数体换成你自己的实现（Triton / CUDA / CuTe DSL 都可以），保持签名和输出
      shape `(batch, q_heads, seq_len, head_dim)` 不变。q 的 shape 为
      `(batch, q_heads, seq_len, head_dim)`，k/v 的 shape 为
      `(batch, kv_heads, seq_len, head_dim)`；标准 MHA 时 `q_heads == kv_heads`，
      GQA 时 `q_heads` 必须是 `kv_heads` 的整数倍，需要自己在实现里把 k/v 的
      head 维度 broadcast/repeat 到 `q_heads`（可参考
      `ops/_example_flash_attention.py` 里的 `repeat_interleave` 用法）。
- [ ] **3. 改注册名**：把文件末尾 `register("TODO_改成你的算子名字", attention)`
      的名字改成能一眼区分实现方式的唯一名字，例如 `"zhangsan_fa (triton)"`。
- [ ] **4. 先查正确性**：运行 `python bench_attention.py --check-only`，
      确认你的算子那一行显示 `PASS`（先保证对，再谈快）。
- [ ] **5. 再看性能**：去掉 `--check-only` 正式跑一遍，对比你的算子和 baseline
      的耗时（ms）/ TFLOPS，以及 `vs baseline` 列（相对 baseline 的耗时占比），
      迭代优化。

接入完成后，`ops/` 目录大概是这样（每人一个文件，互不干扰）：

```
ops/
├── base.py                          # 不要改：注册表
├── __init__.py                      # 不要改：自动扫描发现新算子
├── _template.py                     # 参考：接入模板（以 _ 开头，不会被扫描运行）
├── _example_flash_attention.py      # 参考：纯 PyTorch online-softmax 示例实现（以 _ 开头，不会被扫描运行）
├── zhangsan_flash_attention.py      # 你新增的算子（示例命名）
└── lisi_flash_attention.py          # 队友新增的算子（示例命名）
```

## 如何新增自己的算子（详细说明）

1. 在 `test/ops/` 目录下新建一个 `.py` 文件（文件名任意，但**不要以 `_` 开头**，
   建议直接从 `ops/_template.py` 复制，例如复制为 `my_flash_attention.py`）。
2. 实现统一签名的函数：

   ```python
   def attention(q, k, v, causal=True, sm_scale=None):
       # q shape: (batch, q_heads, seq_len, head_dim)
       # k, v shape: (batch, kv_heads, seq_len, head_dim)
       # 标准 MHA: q_heads == kv_heads；GQA: q_heads 是 kv_heads 的整数倍，
       # 需要自己把 k/v 的 head 维度 broadcast/repeat 到 q_heads
       # 返回值 shape 与 q 相同
       # ...
       return output
   ```

3. 在文件末尾调用 `register()` 完成注册（`name` 建议写清楚实现方式，方便在结果表格中区分）：

   ```python
   from .base import register

   register("my_flash_attention (triton)", attention)
   ```

4. 完成！不需要修改任何其他文件。运行 `bench_attention.py` 时会自动发现并对比你的算子。

> 可以参考 `ops/_example_flash_attention.py`（仅供参考，文件名以 `_` 开头不会被
> 自动扫描注册，避免和你自己接入的算子一起被跑到），里面是一个纯 PyTorch 实现的
> online-softmax（分块流式 softmax）版 FlashAttention，展示了算法的核心思路；
> 也可以直接从 `ops/_template.py` 复制起手，里面已经写好了接入步骤的注释。

## 如何运行

```bash
cd test

# 使用默认参数运行（两组形状，fp16，causal）
python bench_attention.py

# 自定义形状（batch x heads x seq_len x head_dim，逗号分隔多组）
python bench_attention.py --shapes 1x8x1024x64,1x8x4096x64,2x16x2048x128

# GQA：batch x q_heads x kv_heads x seq_len x head_dim（q_heads 必须是 kv_heads 的整数倍）
python bench_attention.py --shapes 1x32x8x4096x128

# 关闭因果掩码 + fp32
python bench_attention.py --dtype fp32 --no-causal

# 只做正确性校验，不测速（适合先验证算子写对了没有）
python bench_attention.py --check-only

# 调整 warmup / 迭代次数
python bench_attention.py --warmup 10 --iters 50

# 只跑 decode 阶段（模拟 KV-Cache 访存场景，关注 GB/s）
python bench_attention.py --phases decode
```

## 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--shapes` | 形状列表，逗号分隔，支持两种格式：标准 MHA 用 4 段 `batchxheadsxseq_lenxhead_dim`（q/kv head 数相同）；GQA 用 5 段 `batchxq_headsxkv_headsxseq_lenxhead_dim`（`q_heads` 必须是 `kv_heads` 的整数倍） | `1x8x1024x64,1x8x4096x64` |
| `--dtype` | 计算数据类型，可选 `fp16` / `fp32` / `bf16` | `fp16` |
| `--causal` / `--no-causal` | 是否使用因果掩码（只看当前位置及之前的 token） | 开启 |
| `--warmup` | 正式计时前的 warmup 迭代次数 | `5` |
| `--iters` | 正式计时的迭代次数，取平均耗时 | `20` |
| `--check-only` | 只做正确性校验，跳过性能测速 | 关闭 |
| `--phases` | 要跑的阶段，逗号分隔，可选 `prefill` / `decode` | `prefill,decode` |

## 输出说明

对每个形状会打印一个对比表格，包含：

- **baseline**：按 FlashAttention-3 → FlashAttention-2 → PyTorch 官方
  `scaled_dot_product_attention` 优先级自动降级选出的参照实现，展示其耗时 / TFLOPS。
- 每个已注册算子的：
  - 耗时（ms，真实测量）、估算 TFLOPS/GB·s（基于理论 FLOPs/字节数公式，仅供参考）
  - **相对 baseline 的耗时占比**（`vs baseline` 列，`baseline耗时 / 算子耗时 * 100%`，
    基于真实耗时计算，是精确值；数值越大代表比 baseline 越快）
  - 与 baseline 输出的**最大绝对误差**
  - 正确性校验结果（`PASS` / `FAIL`）：
    - fp16 / bf16 容差：绝对误差 ≤ 2e-2 或相对误差 ≤ 2e-2
    - fp32 容差：绝对误差 ≤ 1e-4 或相对误差 ≤ 1e-4
  - 如果算子运行时抛异常（比如显存不足、shape 不支持），会标注为“运行失败”，
    不会影响其他算子继续测试。

在非 `--check-only` 模式下，每个 shape 的 prefill/decode 两个阶段都跑完后，
还会额外打印一行 `[小结]`，对比 baseline 在 prefill（耗时/TFLOPS）和
decode（耗时/GB·s）两个阶段的表现，直观展示"prefill 拼算力、decode 拼带宽"
的差异。

## 环境要求

- 只依赖 `torch`（建议 `torch>=2.1.0`），无需额外安装依赖。
- 有 GPU 时自动使用 CUDA 并用 `torch.cuda.Event` 精确计时；
  没有 GPU 时会自动降级到 CPU（用 `time.perf_counter()` 计时），
  但此时性能数据仅供参考，不代表真实 GPU 性能。
