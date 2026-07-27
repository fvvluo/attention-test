"""FlashAttention 自定义算子 vs flash-attention-baseline 对比 benchmark 脚本。

baseline 固定使用 `flash-attention-baseline` 提供的
`flash_attn.cute.flash_attn_func`。脚本会将该 checkout 的 `flash_attn` 路径置于
其他 FlashAttention 实现之前，并校验实际加载来源，避免同名包冲突。

Prefill / Decode 说明：
    对每一组 --shapes 给出的形状 (batch, q_heads, kv_heads, seq_len, head_dim)
    （标准 MHA 时 q_heads == kv_heads，GQA 时 q_heads 是 kv_heads 的整数倍），
    脚本会分别构造并测两个阶段：
      - prefill: q_len = kv_len = seq_len，模拟一次性处理完整 prompt
                 （计算密集，FLOPs 随 seq_len^2 增长）
      - decode : q_len = 1, kv_len = seq_len，模拟自回归生成时用一条新 token
                 去 attend 已缓存的 KV-Cache（访存密集，性能瓶颈是带宽而非算力）
    两个阶段会分别打印独立的对比表格（耗时 / TFLOPS / GB/s）。

最终测试命令（本次评测实际使用的命令，非用法示例）：
    python3 bench_attention.py --gpu 0 --shapes 1x64x8x131072x128 --dtype bf16 --causal \
        --prefill-warmup 10 --prefill-iters 10 --decode-warmup 100 --decode-iters 100
    （prefill 用 10+10；decode 单次耗时极小，用 100+100 降低测量噪声，
     总耗时几乎不变，因为总时间由 prefill 主导。这也是各参数的默认值。）

用法示例（以下均为参数用法演示，非最终测试命令）：
    python bench_attention.py --gpu 0
    python bench_attention.py --gpu 0 --shapes 1x8x1024x128,1x8x4096x128 --dtype fp16 --no-causal
    python bench_attention.py --gpu 0 --check-only          # 只做正确性校验，不测速
    # GQA：q_heads=32, kv_heads=8（q_heads 必须是 kv_heads 的整数倍）
    python bench_attention.py --gpu 0 --shapes 1x32x8x4096x128
    # 多人共享一台机器时，用 --gpu 指定自己的卡号，避免大家都抢 cuda:0
    python bench_attention.py --gpu 3

新增自定义算子的方法请见同目录下 README.md。
"""

import argparse
import importlib
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch

from ops import OPS


def _display_width(s: str) -> int:
    """计算字符串在终端里的实际显示宽度（中日韩全角字符占 2 列，其余占 1 列）。

    Python 的 f"{s:<45}" 是按 len(s)（字符数）对齐，但中文字符实际显示宽度是
    英文字符的 2 倍，混合中英文时会导致列错位，因此这里手动按显示宽度计算。
    """
    width = 0
    for ch in s:
        width += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return width


def _pad(s: str, width: int, align: str = "<") -> str:
    """按显示宽度（而非字符数）左对齐或右对齐填充字符串到指定宽度。"""
    pad_len = max(0, width - _display_width(s))
    if align == "<":
        return s + " " * pad_len
    return " " * pad_len + s


def _truncate(s: str, width: int) -> str:
    """按显示宽度截断字符串到指定宽度以内，超长部分用 "..." 代替。

    保证每一行都严格是单行输出，不会因为算子名过长（尤其是带中文说明的
    baseline 名字）而把整行撑开或被迫换行，从而破坏表格对齐。
    """
    if _display_width(s) <= width:
        return s
    ellipsis = "..."
    budget = width - _display_width(ellipsis)
    out = ""
    cur = 0
    for ch in s:
        w = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        if cur + w > budget:
            break
        out += ch
        cur += w
    return out + ellipsis


# ------------------------- 参数解析 -------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="FlashAttention 算子对比 benchmark")
    parser.add_argument(
        "--shapes",
        type=str,
        default="1x8x1024x128,1x8x4096x128",
        help="形状列表，逗号分隔，每组形状支持两种格式："
        "标准 MHA 用 4 段 batchxheadsxseq_lenxhead_dim（q/kv head 数相同）；"
        "GQA 用 5 段 batchxq_headsxkv_headsxseq_lenxhead_dim（q_heads 必须是 "
        "kv_heads 的整数倍），例如 1x8x1024x128,1x32x8x4096x128。"
        "其中 seq_len 在 prefill 阶段用作 q_len=kv_len，在 decode 阶段用作 "
        "kv_len（q_len 固定为 1）",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16"],
        help="计算使用的数据类型",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        required=True,
        help="指定使用的 GPU 卡号（必填，多人共享一台机器时用来分卡，避免抢占同一张卡）",
    )
    parser.add_argument(
        "--causal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否使用因果掩码（仅作用于 prefill 阶段，默认开启，用 --no-causal 关闭）",
    )
    parser.add_argument("--prefill-warmup", type=int, default=10, help="prefill 阶段正式计时前的 warmup 次数")
    parser.add_argument("--prefill-iters", type=int, default=10, help="prefill 阶段正式计时的迭代次数")
    parser.add_argument(
        "--decode-warmup",
        type=int,
        default=100,
        help="decode 阶段正式计时前的 warmup 次数（decode 单次耗时远小于 "
        "prefill，用更多次数降低测量噪声）",
    )
    parser.add_argument("--decode-iters", type=int, default=100, help="decode 阶段正式计时的迭代次数")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只做正确性校验，不进行性能测速",
    )
    parser.add_argument(
        "--phases",
        type=str,
        default="prefill,decode",
        help="要跑的阶段，逗号分隔，可选 prefill / decode，默认两者都跑",
    )
    return parser.parse_args()


def parse_shapes(shapes_str: str) -> List[Tuple[int, int, int, int, int]]:
    """解析形状字符串，支持两种格式，统一返回 5 元组：

        (batch, q_heads, kv_heads, seq_len, head_dim)

    - 4 段 "batchxheadsxseq_lenxhead_dim"（标准 MHA）：
      展开为 q_heads = kv_heads = heads。
      例如 "1x8x1024x128" -> (1, 8, 8, 1024, 128)
    - 5 段 "batchxq_headsxkv_headsxseq_lenxhead_dim"（GQA，q_heads 必须是
      kv_heads 的整数倍）：
      例如 "1x32x8x4096x128" -> (1, 32, 8, 4096, 128)
    """
    shapes = []
    for item in shapes_str.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split("x")
        if len(parts) == 4:
            batch, heads, seq_len, head_dim = (int(p) for p in parts)
            q_heads, kv_heads = heads, heads
        elif len(parts) == 5:
            batch, q_heads, kv_heads, seq_len, head_dim = (int(p) for p in parts)
            if q_heads % kv_heads != 0:
                raise ValueError(
                    f"形状 \"{item}\" 中 q_heads={q_heads} 必须是 "
                    f"kv_heads={kv_heads} 的整数倍（GQA 要求）"
                )
        else:
            raise ValueError(
                f"形状格式错误: \"{item}\"，应为 4 段 batchxheadsxseq_lenxhead_dim"
                f"（标准 MHA）或 5 段 batchxq_headsxkv_headsxseq_lenxhead_dim"
                f"（GQA），例如 1x8x1024x128 或 1x32x8x4096x128"
            )
        shapes.append((batch, q_heads, kv_heads, seq_len, head_dim))
    if not shapes:
        raise ValueError("未解析出任何有效形状，请检查 --shapes 参数")
    return shapes


DTYPE_MAP = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
}

# 不同 dtype 下的正确性容差
TOLERANCE = {
    torch.float16: dict(abs_tol=2e-2, rel_tol=2e-2),
    torch.bfloat16: dict(abs_tol=2e-2, rel_tol=2e-2),
    torch.float32: dict(abs_tol=1e-4, rel_tol=1e-4),
}


# ------------------------- FLOPs / 显存估算 -------------------------

def attn_flops(batch, q_heads, kv_heads, q_len, kv_len, head_dim, causal: bool) -> float:
    """估算前向 attention 的浮点运算量。

    主要由两次矩阵乘法构成：
      QK^T: 2 * batch*q_heads*q_len*kv_len*head_dim
      P V : 2 * batch*q_heads*q_len*kv_len*head_dim
    GQA 场景下每个 q head 都要做一次完整的 QK^T/PV（kv 会被广播/repeat 到
    对应的 q head 上），因此 FLOPs 按 q_heads 计算，kv_heads 不直接影响。
    causal 且 q_len == kv_len（典型 prefill 场景）时，约一半的计算被掩码掉，
    粗略打 0.5 折扣；q_len != kv_len（典型 decode 场景，q_len=1）时，
    单个 query 会 attend 到全部 kv，没有掩码浪费，因此不打折扣。
    """
    flops = 4.0 * batch * q_heads * q_len * kv_len * head_dim
    if causal and q_len == kv_len:
        flops *= 0.5
    return flops


def attn_bytes(batch, q_heads, kv_heads, q_len, kv_len, head_dim, dtype: torch.dtype) -> float:
    """粗略估算 q/k/v/output 的显存搬运量（仅供参考，非严格值）。

    读 q（q_heads, q_len）+ 读 k,v（各 kv_heads, kv_len）+ 写 output（q_heads,
    q_len）。GQA 场景下 kv_heads < q_heads，k/v 的实际搬运量更小，这也是
    GQA 相比 MHA 能降低 KV-Cache 带宽压力的原因。
    decode 阶段 q_len=1，这里的字节数基本由 kv_len（KV-Cache 大小）主导，
    对应"访存密集"的直觉。
    """
    elem_size = torch.tensor([], dtype=dtype).element_size()
    return elem_size * batch * head_dim * (
        2.0 * q_heads * q_len + 2.0 * kv_heads * kv_len
    )


# ------------------------- 输入生成 & baseline -------------------------

def make_inputs(batch, q_heads, kv_heads, q_len, kv_len, head_dim, dtype, device):
    q = torch.randn(batch, q_heads, q_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, kv_heads, kv_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch, kv_heads, kv_len, head_dim, dtype=dtype, device=device)
    return q, k, v


def get_baseline_fn() -> Tuple[Callable, str]:
    """加载 flash-attention-baseline，并固定其为正确性与性能基线。"""
    baseline_package_dir = (
        Path(__file__).resolve().parent / "flash-attention-baseline" / "flash_attn"
    ).resolve()
    if not baseline_package_dir.is_dir():
        raise ImportError(f"找不到 flash-attention-baseline: {baseline_package_dir}")

    # FA2 和两个外部 checkout 都使用 flash_attn 包名。先加载父包，再将指定
    # baseline checkout 放到子模块搜索路径首位，确保 flash_attn.cute 不会命中
    # flash-attention/ 或 site-packages 中的另一份实现。
    import flash_attn

    baseline_package_path = str(baseline_package_dir)
    flash_attn.__path__ = [
        baseline_package_path,
        *(path for path in flash_attn.__path__ if path != baseline_package_path),
    ]
    importlib.invalidate_caches()

    from flash_attn.cute import flash_attn_func
    import flash_attn.cute as flash_attn_cute

    loaded_from = Path(flash_attn_cute.__file__).resolve()
    if baseline_package_dir not in loaded_from.parents:
        raise ImportError(
            "flash_attn.cute 路由错误: "
            f"期望从 {baseline_package_dir} 加载，实际为 {loaded_from}"
        )

    def baseline(q, k, v, causal=True, sm_scale=None):
        if any(tensor.shape[-1] != 128 for tensor in (q, k, v)):
            raise ValueError("flash-attention-baseline 仅支持 head_dim=128")

        # flash_attn_func 使用 BSHD；benchmark 内部统一使用 BHSD。
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        out = flash_attn_func(
            q_t,
            k_t,
            v_t,
            softmax_scale=sm_scale,
            causal=causal,
        )
        if isinstance(out, tuple):
            out = out[0]
        return out.transpose(1, 2)

    name = f"flash_attn.cute.flash_attn_func (flash-attention-baseline: {loaded_from})"
    return baseline, name


# ------------------------- 正确性校验 -------------------------

@dataclass
class CorrectnessResult:
    passed: bool
    max_abs_diff: float
    max_rel_diff: float
    error: Optional[str] = None


def check_correctness(output: torch.Tensor, baseline_output: torch.Tensor, dtype: torch.dtype) -> CorrectnessResult:
    tol = TOLERANCE.get(dtype, dict(abs_tol=1e-4, rel_tol=1e-4))

    out_f32 = output.float()
    base_f32 = baseline_output.float()

    abs_diff = (out_f32 - base_f32).abs()
    max_abs_diff = abs_diff.max().item()

    rel_diff = abs_diff / (base_f32.abs() + 1e-6)
    max_rel_diff = rel_diff.max().item()

    passed = (max_abs_diff <= tol["abs_tol"]) or (max_rel_diff <= tol["rel_tol"])

    return CorrectnessResult(passed=passed, max_abs_diff=max_abs_diff, max_rel_diff=max_rel_diff)


# ------------------------- 性能测速 -------------------------

@dataclass
class BenchResult:
    avg_ms: float
    tflops: float
    gbps: float


def benchmark_fn(fn, q, k, v, causal, sm_scale, warmup, iters, device) -> BenchResult:
    """warmup 若干次后，正式计时 iters 次，取平均耗时。"""
    batch, q_heads, q_len, head_dim = q.shape
    kv_heads = k.shape[1]
    kv_len = k.shape[2]

    for _ in range(warmup):
        fn(q, k, v, causal=causal, sm_scale=sm_scale)
    if device.type == "cuda":
        torch.cuda.synchronize()

    if device.type == "cuda":
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        for _ in range(iters):
            fn(q, k, v, causal=causal, sm_scale=sm_scale)
        end_evt.record()
        torch.cuda.synchronize()
        elapsed_ms = start_evt.elapsed_time(end_evt) / iters
    else:
        start = time.perf_counter()
        for _ in range(iters):
            fn(q, k, v, causal=causal, sm_scale=sm_scale)
        end = time.perf_counter()
        elapsed_ms = (end - start) * 1000 / iters

    flops = attn_flops(batch, q_heads, kv_heads, q_len, kv_len, head_dim, causal)
    tflops = flops / (elapsed_ms / 1000) / 1e12

    num_bytes = attn_bytes(batch, q_heads, kv_heads, q_len, kv_len, head_dim, q.dtype)
    gbps = num_bytes / (elapsed_ms / 1000) / 1e9

    return BenchResult(avg_ms=elapsed_ms, tflops=tflops, gbps=gbps)


# ------------------------- 结果汇总打印 -------------------------

@dataclass
class OpReport:
    name: str
    correctness: Optional[CorrectnessResult] = None
    bench: Optional[BenchResult] = None
    run_error: Optional[str] = None


PHASE_INFO = {
    "prefill": "q_len = kv_len = seq_len，计算密集（FLOPs ~ seq_len^2），关注 TFLOPS",
    "decode": "q_len = 1, kv_len = seq_len（模拟 KV-Cache），访存密集，关注 GB/s",
}


def print_report(
    phase: str,
    shape,
    q_len: int,
    kv_len: int,
    dtype_name,
    causal,
    baseline_name: str,
    baseline_bench: Optional[BenchResult],
    reports: List[OpReport],
):
    batch, q_heads, kv_heads, seq_len, head_dim = shape

    # 列宽按显示宽度计算（而非字符数），避免中文字符把列挤歪；
    # 算子名（含 baseline 说明文字）超宽时会被截断加 "..."，保证每行都是
    # 单行输出、严格按列对齐，不会因为换行导致数值列错位。
    NAME_W, MS_W, TFLOPS_W, GBPS_W, VS_W, DIFF_W, PASS_W = 58, 11, 9, 9, 12, 12, 8
    TOTAL_W = NAME_W + MS_W + TFLOPS_W + GBPS_W + VS_W + DIFF_W + PASS_W

    print("=" * TOTAL_W)
    print(
        f"[{phase.upper()}] {PHASE_INFO.get(phase, '')}"
    )
    print(
        f"形状: batch={batch} q_heads={q_heads} kv_heads={kv_heads} q_len={q_len} "
        f"kv_len={kv_len} head_dim={head_dim} "
        f"| dtype={dtype_name} | causal={causal if phase == 'prefill' else '不适用(decode)'}"
    )
    print("-" * TOTAL_W)

    def print_row(name: str, ms: str, tflops: str, gbps: str, vs: str, diff: str, pas: str, suffix: str = ""):
        """打印一行数据；name 超过 NAME_W 时截断加省略号，保证严格单行对齐。"""
        values = (
            _pad(ms, MS_W, ">") + _pad(tflops, TFLOPS_W, ">") + _pad(gbps, GBPS_W, ">")
            + _pad(vs, VS_W, ">") + _pad(diff, DIFF_W, ">") + _pad(pas, PASS_W, ">") + suffix
        )
        print(_pad(_truncate(name, NAME_W), NAME_W) + values)

    header = (
        _pad("算子", NAME_W) + _pad("耗时(ms)", MS_W, ">") + _pad("TFLOPS", TFLOPS_W, ">")
        + _pad("GB/s", GBPS_W, ">") + _pad("vs baseline", VS_W, ">")
        + _pad("最大绝差", DIFF_W, ">") + _pad("正确性", PASS_W, ">")
    )
    print(header)
    print("-" * TOTAL_W)

    if baseline_bench is not None:
        print_row(
            "baseline",
            f"{baseline_bench.avg_ms:.3f}", f"{baseline_bench.tflops:.2f}", f"{baseline_bench.gbps:.1f}",
            "-", "-", "-",
        )

    # 记录每个算子的 vs baseline 数值，供下方"突出显示"小节使用
    # (name, speedup_pct, passed)：passed=None 表示运行失败/无正确性结果，
    # 汇总表格里需要一起标注出来，避免只看这个表格误以为"快"就是"对"。
    speedups: List[Tuple[str, Optional[float], Optional[bool]]] = []

    for r in reports:
        if r.run_error is not None:
            print_row(r.name, "运行失败", "-", "-", "-", "-", "FAIL", suffix=f"  ({r.run_error})")
            speedups.append((r.name, None, False))
            continue

        ms_str = f"{r.bench.avg_ms:.3f}" if r.bench else "-"
        tflops_str = f"{r.bench.tflops:.2f}" if r.bench else "-"
        gbps_str = f"{r.bench.gbps:.1f}" if r.bench else "-"
        # vs baseline：baseline_ms / 算子_ms，用真实耗时做比值，
        # 两者理论 FLOPs 公式相同会相互抵消，因此该比值是精确值而非估算值。
        # >100% 表示比 baseline 快，<100% 表示比 baseline 慢。
        speedup_pct: Optional[float] = None
        if r.bench and baseline_bench is not None:
            speedup_pct = baseline_bench.avg_ms / r.bench.avg_ms * 100
            speedup_str = f"{speedup_pct:.1f}%"
        else:
            speedup_str = "-"
        passed = r.correctness.passed if r.correctness else None
        speedups.append((r.name, speedup_pct, passed))
        abs_diff_str = f"{r.correctness.max_abs_diff:.2e}" if r.correctness else "-"
        pass_str = "PASS" if passed else "FAIL"

        print_row(r.name, ms_str, tflops_str, gbps_str, speedup_str, abs_diff_str, pass_str)

    print("=" * TOTAL_W)

    # 单独突出显示 vs baseline，同样用表格对齐（不是纯文本），方便一眼对应到算子名。
    # 正确性 FAIL 的算子即使速度很快也会被标注出来（前面加 ⚠），避免有人只看这个
    # 汇总表格就误以为"快"等于"对"——一个啥都没算的空算子也可能显示"快 10x"。
    valid_speedups = [(name, pct, passed) for name, pct, passed in speedups if pct is not None]
    if valid_speedups:
        VS_NAME_W, VS_PCT_W, VS_RATIO_W = NAME_W, 12, 20
        print(f">>> vs baseline（{_truncate(baseline_name, 60)}）：数值越大代表比 baseline 越快")
        print(
            _pad("算子", VS_NAME_W) + _pad("占比", VS_PCT_W, ">") + _pad("倍数", VS_RATIO_W, ">")
        )
        print("-" * (VS_NAME_W + VS_PCT_W + VS_RATIO_W))
        for name, pct, passed in valid_speedups:
            ratio = pct / 100
            ratio_str = f"快 {ratio:.2f}x" if ratio >= 1 else f"慢 {1 / ratio:.2f}x" if ratio > 0 else "慢"
            display_name = name if passed else f"[FAIL] {name}"
            suffix = "" if passed else "  <- 正确性 FAIL，此速度数据无意义"
            print(
                _pad(_truncate(display_name, VS_NAME_W), VS_NAME_W)
                + _pad(f"{pct:.1f}%", VS_PCT_W, ">") + _pad(ratio_str, VS_RATIO_W, ">")
                + suffix
            )

    print()


def print_phase_summary(shape, phase_benches: dict):
    """在每个 shape 的 prefill/decode 都跑完后，打印一个简短的对比小结，
    直观展示"prefill 拼算力、decode 拼带宽"的差异。
    """
    batch, q_heads, kv_heads, seq_len, head_dim = shape
    prefill = phase_benches.get("prefill")
    decode = phase_benches.get("decode")
    if prefill is None or decode is None:
        return
    print(
        f"[小结] shape=({batch},{q_heads},{kv_heads},{seq_len},{head_dim}) baseline: "
        f"prefill {prefill.avg_ms:.3f}ms / {prefill.tflops:.2f} TFLOPS   "
        f"decode {decode.avg_ms:.3f}ms / {decode.gbps:.1f} GB/s"
    )
    print()


# ------------------------- 主流程 -------------------------

def main():
    args = parse_args()

    if args.gpu is not None:
        if not torch.cuda.is_available():
            raise RuntimeError(f"指定了 --gpu {args.gpu}，但当前环境检测不到可用的 CUDA 设备")
        if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
            raise ValueError(
                f"--gpu {args.gpu} 超出范围，当前可见 GPU 数量为 "
                f"{torch.cuda.device_count()}（可用卡号: 0~{torch.cuda.device_count() - 1}）"
            )
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"使用设备: {device}")
    if device.type == "cuda":
        print(f"GPU 型号: {torch.cuda.get_device_name(device)}")
    else:
        print("警告: 未检测到 GPU，将使用 CPU 运行。CPU 模式下的性能数据仅供参考，"
              "不代表实际 GPU 性能；FlashAttention-2/3 均无法在 CPU 上运行，"
              "baseline 会自动降级为 SDPA。")

    dtype = DTYPE_MAP[args.dtype]
    shapes = parse_shapes(args.shapes)
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    for p in phases:
        if p not in PHASE_INFO:
            raise ValueError(f"未知阶段 \"{p}\"，可选值: {list(PHASE_INFO.keys())}")

    if not OPS:
        print("警告: 当前没有任何已注册的自定义算子（ops/ 目录下除 base.py 外没有找到算子文件）。")

    baseline_fn, baseline_name = get_baseline_fn()
    print(f"baseline: {baseline_name}")
    print()

    for shape in shapes:
        batch, q_heads, kv_heads, seq_len, head_dim = shape
        phase_baseline_benches = {}

        for phase in phases:
            if phase == "prefill":
                q_len, kv_len = seq_len, seq_len
                phase_causal = args.causal
                phase_warmup = args.prefill_warmup
                phase_iters = args.prefill_iters
            else:  # decode
                q_len, kv_len = 1, seq_len
                # decode 阶段新 token 天然位于序列末尾，能 attend 到全部已缓存的
                # kv，等价于非因果；这里显式关闭 causal，避免部分朴素实现在
                # q_len != kv_len 时对绝对位置的因果掩码处理出错。
                phase_causal = False
                phase_warmup = args.decode_warmup
                phase_iters = args.decode_iters

            q, k, v = make_inputs(batch, q_heads, kv_heads, q_len, kv_len, head_dim, dtype, device)
            sm_scale = None  # 使用默认缩放 1/sqrt(head_dim)

            # baseline 输出，作为正确性校验的参照
            try:
                baseline_output = baseline_fn(q, k, v, causal=phase_causal, sm_scale=sm_scale)
            except Exception as e:  # noqa: BLE001
                print(f"[{phase}] baseline 计算失败，跳过该形状: {e}")
                continue

            baseline_bench = None
            if not args.check_only:
                baseline_bench = benchmark_fn(
                    baseline_fn, q, k, v, phase_causal, sm_scale,
                    phase_warmup, phase_iters, device,
                )
                phase_baseline_benches[phase] = baseline_bench

            reports: List[OpReport] = []
            for name, fn in OPS.items():
                report = OpReport(name=name)
                try:
                    output = fn(q, k, v, causal=phase_causal, sm_scale=sm_scale)
                    report.correctness = check_correctness(output, baseline_output, dtype)

                    if not args.check_only:
                        report.bench = benchmark_fn(
                            fn, q, k, v, phase_causal, sm_scale,
                            phase_warmup, phase_iters, device,
                        )
                except Exception as e:  # noqa: BLE001
                    report.run_error = str(e)

                reports.append(report)

            print_report(
                phase, shape, q_len, kv_len, args.dtype, phase_causal,
                baseline_name, baseline_bench, reports,
            )

        if not args.check_only:
            print_phase_summary(shape, phase_baseline_benches)


if __name__ == "__main__":
    main()