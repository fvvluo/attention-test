# FlashAttention 算子对比 Benchmark

这是一个方便团队成员接入自定义 FlashAttention 算子、并与官方 baseline 对比正确性与
性能的测试框架。baseline 固定使用 `flash-attention-baseline` 提供的
`flash_attn.cute.flash_attn_func`；脚本会优先路由到该目录并校验加载来源，避免与
`flash-attention` 或 site-packages 中的同名实现冲突。

## ✅ 最终测试命令

以下命令使用当前最新版 benchmark，在进程可见的 `cuda:0` 上同时测试 Prefill 和
Decode。**其余文档中出现的命令均为用法示例，非最终测试命令**：

```bash
python3 bench_attention.py --gpu 0 --shapes 1x64x8x131072x128 --dtype bf16 --causal \
    --prefill-warmup 10 --prefill-iters 10 --decode-warmup 100 --decode-iters 100
```

- `--gpu 0`：选择当前进程可见的第 0 张 GPU；
- `--shapes 1x64x8x131072x128`：GQA，`batch=1, q_heads=64, kv_heads=8,
  seq_len=131072（128K）, head_dim=128`；
- `--dtype bf16`；
- `--causal`：开启 Prefill 因果掩码（Decode 会按 benchmark 规则关闭 causal）；
- Prefill 使用 10 次预热和 10 次正式计时，Decode 使用 100 次预热和 100 次正式计时。
- 序列长度 128K 属于长序列场景，Prefill 阶段单次前向本身就要几秒，运行期间
  终端可能暂时没有中间输出，可另开窗口用 `nvidia-smi` 确认 GPU 是否仍在计算。

## 目录结构

```
test/
├── README.md                                      # 本文档
├── flash-attention-baseline/                      # 固定对比 baseline
├── ops/
│   ├── __init__.py                                # 自动扫描并导入本目录下所有算子模块
│   ├── base.py                                    # 算子注册接口 register(name, fn)
│   ├── cute_dsl_flash_attention_m128.py           # 最终 M128N128 stage-4 Prefill adapter
│   ├── _cute_dsl_sm90_fmha_perf_m128_kernel.py    # 最终静态 128K CuTe DSL kernel
│   ├── _template.py                               # 接入模板，复制改名即可开始写自己的算子
│   └── _example_flash_attention.py                # 示例算子（不会被自动扫描注册）
└── bench_attention.py                             # 主 benchmark 脚本
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
- [ ] **4. 先查正确性**：运行 `python bench_attention.py --gpu 0 --check-only`，
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

### 如果用 C++ / CUDA 扩展实现算子

用纯 PyTorch 或 Triton 写算子可以跳过这一节。写惯 C++/CUDA 的同学可以按平时的
习惯，把 kernel 写成正常的 `.cu`/`.cpp` 文件，用 `torch.utils.cpp_extension.load()`
在运行时编译成 Python 可以调用的模块，再在 `attention()` 里调用它、`register()`
注册即可——接入方式和纯 Python 算子完全一样，只是 `attention()` 函数体里改成
调 C++/CUDA 编译出来的函数。

**第 1 步：正常写 kernel 文件（`.cu` 写 CUDA kernel，`.cpp` 写 pybind11 绑定）**

假设你的算子文件夹结构是（放在 `ops/` 目录下，跟自己的 `.py` 文件放一起）：

```
ops/
└── zhangsan_ext/
    ├── kernel.cu      # CUDA kernel 实现
    └── binding.cpp    # 声明函数 + pybind11 绑定，暴露给 Python 调用
```

`kernel.cu`（写你自己的 attention 计算逻辑，这里演示一个占位的 copy kernel）：

```cpp
#include <torch/extension.h>

__global__ void copy_kernel(const float* q, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = q[idx];
    }
}

torch::Tensor copy_forward(torch::Tensor q) {
    auto out = torch::empty_like(q);
    int n = q.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    copy_kernel<<<blocks, threads>>>(q.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}
```

`binding.cpp`（声明函数签名 + 用 pybind11 暴露给 Python，几乎每个算子都是这个套路，
改函数名即可）：

```cpp
#include <torch/extension.h>

torch::Tensor copy_forward(torch::Tensor q);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("copy_forward", &copy_forward, "copy forward (CUDA)");
}
```

**第 2 步：在 `ops/zhangsan_flash_attention.py` 里用 `load()` 加载编译，然后照常注册**

```python
import torch
from torch.utils.cpp_extension import load
from .base import register

# 用 load() 编译 csrc 目录下的 .cu/.cpp 文件（第一次运行会自动编译，
# 之后源码不变就直接复用缓存，不需要手动 make/cmake）。
# name 务必改成带自己署名的唯一字符串（原因见下方"注意事项"）。
_mod = load(
    name="flash_attn_zhangsan",   # <- 改成自己的名字
    sources=["ops/zhangsan_ext/binding.cpp", "ops/zhangsan_ext/kernel.cu"],
    verbose=False,  # 编译报错排查时可临时改成 True 看完整编译日志
)


def attention(q, k, v, causal=True, sm_scale=None):
    # 这里换成真正的 attention 计算逻辑；示例 kernel 只支持 fp32，
    # 所以需要转 dtype，真正实现里如果 kernel 原生支持 fp16/bf16 就不需要转
    q_f32 = q.float().contiguous()
    out = _mod.copy_forward(q_f32)
    return out.to(q.dtype)


register("zhangsan_fa (cuda)", attention)
```

就这样，跟纯 Python 算子的接入流程完全一样，`bench_attention.py` 不需要改。

**关于 cutlass**：接入方式一样，在 `.cu` 文件里 `#include <cutlass/xxx.h>` 直接用
即可，头文件在 `flash-attention/csrc/cutlass/include/` 下面（submodule 自带，
不需要额外安装）；`load()` 的 `extra_include_paths` 参数可以指定这个路径：

```python
_mod = load(
    name="flash_attn_zhangsan_cutlass",
    sources=[...],
    extra_include_paths=["flash-attention/csrc/cutlass/include"],
)
```

**注意事项 1（这台机器的 CUDA 版本坑）**：这台机器装的是 CUDA 13.0（`nvcc
--version` 可查），但 PyTorch 是用 CUDA 12.4 编译的（`python -c "import torch;
print(torch.version.cuda)"` 可查），**如果用传统的 `setup.py build_ext` /
`pip install -e .` 方式编译，会直接报错**（`RuntimeError: detected CUDA
version mismatches...`），因为这种方式会做严格的版本校验。**必须用
`torch.utils.cpp_extension.load()`/`load_inline()` 在运行时编译**（本节上面
的方法），它不会触发这个版本校验，已验证可以正常编译运行。

**注意事项 2（多人共享账户的 name 冲突）**：

如果用 `torch.utils.cpp_extension.load_inline`（或 `load`）在线编译 C++/CUDA
kernel，**编译缓存目录是按你传的 `name` 参数命名的**（路径形如
`~/.cache/torch_extensions/py<ver>_cu<ver>/<name>/`），而这台机器大家共用同一个
账户、同一个缓存目录。如果两人都用了同一个 `name`（很容易发生，因为都是照抄同一份
模板起步），且刚好在差不多的时间点第一次触发编译（并发编译），会导致其中一人的
编译产物被另一人覆盖，**运行时不会报错，只会悄悄跑着别人的代码**，非常隐蔽难查。

解决方法很简单：把 `name` 改成带自己署名的唯一字符串，例如：

```python
mod = load_inline(name="flash_attn_zhangsan", cpp_sources=..., cuda_sources=...)
```

## ⚠️ 多人共享 GPU 机器时必读

这台机器有多张 GPU，供团队成员并行使用。脚本默认使用 `cuda:0`（即第一张卡），
**如果大家都直接运行 `python bench_attention.py`，会全部挤到同一张卡上**，
互相抢显存、抢算力，导致：
- 显存不足报 `CUDA out of memory`（尤其是跑大 `--shapes` 时）；
- 测出来的耗时 / TFLOPS 数据被别人的负载干扰，完全不可信。

**解决方法：每人用 `--gpu` 参数指定自己独占的卡号再运行**：

```bash
# 用 nvidia-smi 先看看哪张卡空闲（显存占用低、无进程）
nvidia-smi

# 假设你分到了第 3 张卡（编号从 0 开始）
python bench_attention.py --gpu 3 --shapes 1x8x1024x128
```

也可以用等价的环境变量 `CUDA_VISIBLE_DEVICES` 指定：

```bash
CUDA_VISIBLE_DEVICES=3 python bench_attention.py --gpu 0 --shapes 1x8x1024x128
```

团队内约定好各自固定使用哪个卡号（例如按人头分配 0~7），避免临时撞卡。

## 如何运行

> 以下均为用法示例，展示各参数的效果，**不是最终测试命令**（最终测试命令见文档
> 开头"✅ 最终测试命令"一节）。

```bash
cd test

# 使用默认参数运行（两组形状，fp16，causal）
python bench_attention.py --gpu 0

# 自定义形状（batch x heads x seq_len x head_dim，逗号分隔多组）
python bench_attention.py --gpu 0 --shapes 1x8x1024x128,1x8x4096x128,2x16x2048x128

# GQA：batch x q_heads x kv_heads x seq_len x head_dim（q_heads 必须是 kv_heads 的整数倍）
python bench_attention.py --gpu 0 --shapes 1x32x8x4096x128

# bf16 + 关闭因果掩码
python bench_attention.py --gpu 0 --dtype bf16 --no-causal

# 只做正确性校验，不测速（适合先验证算子写对了没有）
python bench_attention.py --gpu 0 --check-only

# 调整 warmup / 迭代次数
python bench_attention.py --gpu 0 --warmup 10 --iters 50

# 只跑 decode 阶段（模拟 KV-Cache 访存场景，关注 GB/s）
python bench_attention.py --gpu 0 --phases decode
```

## 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--shapes` | 形状列表，逗号分隔，支持两种格式：标准 MHA 用 4 段 `batchxheadsxseq_lenxhead_dim`（q/kv head 数相同）；GQA 用 5 段 `batchxq_headsxkv_headsxseq_lenxhead_dim`（`q_heads` 必须是 `kv_heads` 的整数倍）；当前 baseline 要求 `head_dim=128` | `1x8x1024x128,1x8x4096x128` |
| `--gpu` | 指定使用的 GPU 卡号（必填；多人共享一台机器时用来分卡，避免抢占同一张卡） | 无，必须显式指定 |
| `--dtype` | 计算数据类型，可选 `fp16` / `bf16` | `fp16` |
| `--causal` / `--no-causal` | 是否使用因果掩码（只看当前位置及之前的 token） | 开启 |
| `--warmup` | 正式计时前的 warmup 迭代次数 | `5` |
| `--iters` | 正式计时的迭代次数，取平均耗时 | `20` |
| `--check-only` | 只做正确性校验，跳过性能测速 | 关闭 |
| `--phases` | 要跑的阶段，逗号分隔，可选 `prefill` / `decode` | `prefill,decode` |

## 输出说明

对每个形状会打印一个对比表格，包含：

- **baseline**：固定使用 `flash-attention-baseline` 提供的
  `flash_attn.cute.flash_attn_func`，展示其耗时 / TFLOPS。
- 每个已注册算子的：
  - 耗时（ms，真实测量）、估算 TFLOPS/GB·s（基于理论 FLOPs/字节数公式，仅供参考）
  - **相对 baseline 的耗时占比**（`vs baseline` 列，`baseline耗时 / 算子耗时 * 100%`，
    基于真实耗时计算，是精确值；数值越大代表比 baseline 越快）
  - 与 baseline 输出的**最大绝对误差**
  - 正确性校验结果（`PASS` / `FAIL`）：
    - fp16 / bf16 容差：绝对误差 ≤ 2e-2 或相对误差 ≤ 2e-2
  - 如果算子运行时抛异常（比如显存不足、shape 不支持），会标注为“运行失败”，
    不会影响其他算子继续测试。

在非 `--check-only` 模式下，每个 shape 的 prefill/decode 两个阶段都跑完后，
还会额外打印一行 `[小结]`，对比 baseline 在 prefill（耗时/TFLOPS）和
decode（耗时/GB·s）两个阶段的表现，直观展示"prefill 拼算力、decode 拼带宽"
的差异。

## 环境要求

baseline 路径按**仓库相对路径**固定为
`./flash-attention-baseline/flash_attn/cute`，不是机器相关的绝对路径。脚本会把它
放到 `flash_attn` 子模块搜索路径首位，并校验实际加载文件；如果解析到
`./flash-attention` 或 site-packages 中的另一份 `flash_attn.cute`，会直接报错，
不会静默使用错误实现。

在仓库根目录安装 baseline 及其依赖：

```bash
# CUDA 12.x
python3 -m pip install -e "./flash-attention-baseline/flash_attn/cute"

# CUDA 13.x
python3 -m pip install -e "./flash-attention-baseline/flash_attn/cute[cu13]"
```

上述命令会根据 `flash-attention-baseline/flash_attn/cute/pyproject.toml` 自动安装：

- `torch`
- `einops`
- `typing_extensions`
- `nvidia-cutlass-dsl==4.6.0.dev0`（CUDA 13 使用对应的 `cu13` extra）
- `apache-tvm-ffi>=0.1.12,<0.2`
- `torch-c-dlpack-ext`
- `quack-kernels>=0.5.3`

安装后可检查实际路由：

```bash
python3 -c "import bench_attention; _, name = bench_attention.get_baseline_fn(); print(name)"
```

输出路径应位于
`flash-attention-baseline/flash_attn/cute`。当前环境如果出现
`No module named 'quack'`，说明尚未安装上述 baseline 依赖。

- 需要 CUDA GPU，并使用 `torch.cuda.Event` 精确计时。
- **多人共享同一台机器时**，务必通过 `--gpu` 参数或 `CUDA_VISIBLE_DEVICES`
  指定各自的 GPU 卡号再运行（见上方"多人共享 GPU 机器时必读"），避免抢卡
  导致结果不可信或 OOM。
