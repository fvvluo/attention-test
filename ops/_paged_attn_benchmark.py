"""Benchmark & correctness harness for the H20 paged-attention decode kernel.

设计对应 `h20_paged_attn_design.md`（已按评审意见修正：见文件末尾 NOTES）。

分层验证策略
------------
L0  manager 自检   : append/block_table/gather 还原出的 KV 与"真实历史"逐字节一致
                     （与任何 CUDA 内核无关，先隔离 manager 逻辑）。
L1  数值黄金参考   : paged decode 结果 vs torch SDPA(fp32, 真实历史) —— 任意 seq_len 可用。
L2  连续内核交叉   : 当 seq_len % 128 == 0 时，把 gather 出的连续 KV 喂给现有
                     `attention_decode`，验证 paged == 连续 == SDPA。
L3  边界覆盖       : seq_len%128!=0（末页不满）、跨页边界(127->128->129)、
                     短序列导致的空 split。
L4  性能/带宽      : 目标 shape 131072 的 latency 与 KV 带宽。

paged CUDA 内核尚未实现时：
  - 自动回退到纯 torch 的 `paged_decode_ref`（gather 有效 token + SDPA），
    用于验证 manager 与 harness 本身；
  - 一旦实现，把 `PagedKVDecoder` 暴露为 `_wzc_paged_attn_decode.PagedKVDecoder`
    （接口见设计 §6），本 harness 自动接入并做 L1/L2 对拍。

用法示例
--------
  python paged_attn_benchmark.py                 # 跑正确性套件（含边界）
  python paged_attn_benchmark.py --bench         # 额外跑 131072 性能
  python paged_attn_benchmark.py --poison        # 未用页填 NaN，压测末页 mask
"""

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Optional

import torch

# 允许既作为脚本运行（cwd=ops/）也被 ops 包自动扫描导入时解析同目录模块
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 现有连续内核（同目录）
from _wzc_attn_decode import (
    attention_decode,
    GROUP_M,
    HEAD_DIM,
    BLOCK_N,
    NUM_SPLITS,
)

# 可选：真实 paged CUDA 内核（实现后放到该模块，接口见设计 §6）
try:
    from _wzc_paged_attn_decode import PagedKVDecoder  # type: ignore
except Exception:  # noqa: BLE001
    PagedKVDecoder = None

PAGE_SIZE = BLOCK_N  # 核心不变式：page_size == BLOCK_N == 128


# ---------------------------------------------------------------------------
# Paged KV manager（纯 PyTorch，等价于设计 §2/§4 的 host 侧 manager）
# ---------------------------------------------------------------------------
class PagedKVManager:
    """页池 + block_table + seq_len + 空闲页栈；封装 append / gather。

    采用评审建议的 **kv_head 优先** 页内布局，使每个 (page, kv_head) 的
    128x128 tile 在显存里完全连续（利于 TMA / swizzle）：
        kv_cache[num_pages, kv_heads, page_size, head_dim]
    """

    def __init__(
        self,
        max_seqs: int,
        max_seq_len: int,
        kv_heads: int,
        head_dim: int,
        page_size: int = PAGE_SIZE,
        num_pages: Optional[int] = None,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        poison: bool = False,
    ):
        self.max_seqs = max_seqs
        self.max_seq_len = max_seq_len
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        self.max_pages_per_seq = (max_seq_len + page_size - 1) // page_size
        if num_pages is None:
            num_pages = max_seqs * self.max_pages_per_seq
        self.num_pages = num_pages

        init = torch.nan if poison else 0.0
        shape = (num_pages, kv_heads, page_size, head_dim)
        self.kv_cache_k = torch.full(shape, init, dtype=dtype, device=device)
        self.kv_cache_v = torch.full(shape, init, dtype=dtype, device=device)
        self.block_table = torch.full(
            (max_seqs, self.max_pages_per_seq), -1, dtype=torch.int32, device=device
        )
        self.seq_len = torch.zeros(max_seqs, dtype=torch.int32, device=device)
        # 空闲页栈（host 侧），与设计 §2 free_pages 一致
        self._free = list(range(num_pages - 1, -1, -1))

    # -- 页分配 -------------------------------------------------------------
    def _alloc_page(self) -> int:
        if not self._free:
            raise RuntimeError("page pool exhausted")
        return self._free.pop()

    def reset(self, seq_id: int):
        """归还该序列占用的物理页，seq_len=0（设计 §6 reset）。"""
        L = int(self.seq_len[seq_id].item())
        n = (L + self.page_size - 1) // self.page_size
        for t in range(n):
            p = int(self.block_table[seq_id, t].item())
            if p >= 0:
                self._free.append(p)
                self.block_table[seq_id, t] = -1
        self.seq_len[seq_id] = 0

    # -- append（设计 §4）---------------------------------------------------
    def append_kv(self, seq_id: int, k_new: torch.Tensor, v_new: torch.Tensor):
        """k_new/v_new: (kv_heads, head_dim)。写一个新 token 的 K/V。"""
        L = int(self.seq_len[seq_id].item())
        page = L // self.page_size
        off = L % self.page_size
        if off == 0:  # 跨入新页
            p = self._alloc_page()
            self.block_table[seq_id, page] = p
            # 设计 §2：新分配页首用前清零。这样即使 --poison 把空闲页池填成 NaN，
            # 一旦某页被某序列启用即被清零，末页 padding 行为 0（而非 NaN），
            # 内核 GEMM2 的 `P(=0 for masked) * V(=0 padding)` 不会产生 0*NaN=NaN。
            # mask 仍负责把越界行的 S 置 -inf（正确性保证）；清零负责数值鲁棒性。
            self.kv_cache_k[p].zero_()
            self.kv_cache_v[p].zero_()
        p = int(self.block_table[seq_id, page].item())
        self.kv_cache_k[p, :, off, :] = k_new.to(self.dtype)
        self.kv_cache_v[p, :, off, :] = v_new.to(self.dtype)
        self.seq_len[seq_id] = L + 1

    def bulk_append(self, seq_id: int, k_hist: torch.Tensor, v_hist: torch.Tensor):
        """批量写入 (kv_heads, L, head_dim)，供性能测试快速构造长序列。"""
        assert int(self.seq_len[seq_id].item()) == 0, "bulk_append 需从空序列开始"
        L = k_hist.shape[1]
        n = (L + self.page_size - 1) // self.page_size
        pad = n * self.page_size - L
        kp = torch.nn.functional.pad(k_hist, (0, 0, 0, pad)).to(self.dtype)
        vp = torch.nn.functional.pad(v_hist, (0, 0, 0, pad)).to(self.dtype)
        # (kv_heads, n, page_size, head_dim) -> (n, kv_heads, page_size, head_dim)
        kp = kp.view(self.kv_heads, n, self.page_size, self.head_dim).permute(1, 0, 2, 3)
        vp = vp.view(self.kv_heads, n, self.page_size, self.head_dim).permute(1, 0, 2, 3)
        pages = [self._alloc_page() for _ in range(n)]
        for t, p in enumerate(pages):
            self.block_table[seq_id, t] = p
            self.kv_cache_k[p] = kp[t]
            self.kv_cache_v[p] = vp[t]
        self.seq_len[seq_id] = L

    # -- gather：paged -> 连续（供 L0/L1/L2）---------------------------------
    def gather(self, seq_id: int):
        """还原出 (kv_heads, L, head_dim) 的连续 K/V（仅取有效 token）。"""
        L = int(self.seq_len[seq_id].item())
        n = (L + self.page_size - 1) // self.page_size
        pages = self.block_table[seq_id, :n].tolist()
        ks = torch.cat([self.kv_cache_k[p] for p in pages], dim=1)  # (hk, n*ps, hd)
        vs = torch.cat([self.kv_cache_v[p] for p in pages], dim=1)
        return ks[:, :L, :].contiguous(), vs[:, :L, :].contiguous()


# ---------------------------------------------------------------------------
# 参考实现
# ---------------------------------------------------------------------------
def sdpa_ref(q_new: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float):
    """黄金参考：fp32 SDPA。q_new (q_heads, head_dim); k/v (kv_heads, L, head_dim)。"""
    q_heads = q_new.shape[0]
    kv_heads = k.shape[0]
    rep = q_heads // kv_heads
    q = q_new.view(1, q_heads, 1, HEAD_DIM).float()
    kk = k.unsqueeze(0).repeat_interleave(rep, dim=1).float()
    vv = v.unsqueeze(0).repeat_interleave(rep, dim=1).float()
    o = torch.nn.functional.scaled_dot_product_attention(q, kk, vv, scale=sm_scale)
    return o.view(q_heads, HEAD_DIM)  # fp32


def paged_decode_ref(mgr: PagedKVManager, q_new: torch.Tensor, seq_id: int, sm_scale: float):
    """paged 参考路径：gather 有效 token -> SDPA。用于自检 manager / 无 CUDA 内核时兜底。"""
    k, v = mgr.gather(seq_id)
    return sdpa_ref(q_new, k, v, sm_scale)


def continuous_decode(q_new: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float):
    """现有连续内核。要求 L % 128 == 0。q_new (q_heads,hd); k/v (kv_heads,L,hd) bf16。"""
    q = q_new.view(1, q_new.shape[0], 1, HEAD_DIM).contiguous()
    kk = k.unsqueeze(0).contiguous()
    vv = v.unsqueeze(0).contiguous()
    o = attention_decode(q, kk, vv, sm_scale=sm_scale)
    return o.view(q_new.shape[0], HEAD_DIM).float()


def paged_decode_cuda(mgr: PagedKVManager, q_new: torch.Tensor, seq_id: int, sm_scale: float):
    """真实 paged CUDA 内核（实现后接入）。接口按设计 §6 的 step()。"""
    if PagedKVDecoder is None:
        return None
    o = PagedKVDecoder.decode(  # 约定的无状态入口；也可用实例 step()
        q_new=q_new,
        kv_cache_k=mgr.kv_cache_k,
        kv_cache_v=mgr.kv_cache_v,
        block_table=mgr.block_table,
        seq_len=mgr.seq_len,
        seq_id=seq_id,
        sm_scale=sm_scale,
        num_splits=NUM_SPLITS,
        page_size=mgr.page_size,
    )
    return o.view(q_new.shape[0], HEAD_DIM).float()


# ---------------------------------------------------------------------------
# 误差工具
# ---------------------------------------------------------------------------
def err_stats(a: torch.Tensor, b: torch.Tensor):
    a = a.float()
    b = b.float()
    abs_err = (a - b).abs()
    rel_err = abs_err / (b.abs() + 1e-6)
    return abs_err.max().item(), rel_err.max().item()


def _rand_kv(kv_heads, L, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    k = torch.randn(kv_heads, L, HEAD_DIM, generator=g, dtype=torch.bfloat16, device=device)
    v = torch.randn(kv_heads, L, HEAD_DIM, generator=g, dtype=torch.bfloat16, device=device)
    return k, v


def _rand_q(q_heads, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(q_heads, HEAD_DIM, generator=g, dtype=torch.bfloat16, device=device)


# ---------------------------------------------------------------------------
# L0: manager 自检
# ---------------------------------------------------------------------------
def test_manager(kv_heads, q_heads, device, poison):
    print("\n[L0] manager append/gather 自检 (逐字节还原真实历史)")
    seq_lens = [1, 127, 128, 129, 256, 300]
    ok = True
    for sid, L in enumerate(seq_lens):
        mgr = PagedKVManager(
            max_seqs=1, max_seq_len=L, kv_heads=kv_heads, head_dim=HEAD_DIM,
            device=device, poison=poison,
        )
        k_hist, v_hist = _rand_kv(kv_heads, L, device, seed=100 + L)
        # 逐 token append，模拟真实 decode
        for t in range(L):
            mgr.append_kv(0, k_hist[:, t, :], v_hist[:, t, :])
        gk, gv = mgr.gather(0)
        same = torch.equal(gk, k_hist) and torch.equal(gv, v_hist)
        assert int(mgr.seq_len[0].item()) == L
        print(f"    seq_len={L:>4}  gather==truth: {same}")
        ok = ok and same
    print(f"  => L0 {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# L1/L2/L3: 数值正确性（含边界）
# ---------------------------------------------------------------------------
def test_correctness(kv_heads, q_heads, device, poison, tol_abs, tol_rel):
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    # 覆盖：跨页边界、末页不满、短序列(空 split)、较长
    seq_lens = [1, 63, 127, 128, 129, 255, 256, 257, 511, 512, 1000, 4096, 4097]
    print("\n[L1/L3] paged decode vs SDPA(fp32) —— 覆盖末页不满/跨页/空split")
    has_cuda_paged = PagedKVDecoder is not None
    print(f"    (真实 paged CUDA 内核: {'已接入' if has_cuda_paged else '未实现 -> 用 torch paged 参考自检 harness'})")

    all_ok = True
    for sid, L in enumerate(seq_lens):
        mgr = PagedKVManager(
            max_seqs=1, max_seq_len=L, kv_heads=kv_heads, head_dim=HEAD_DIM,
            device=device, poison=poison,
        )
        k_hist, v_hist = _rand_kv(kv_heads, L, device, seed=200 + L)
        mgr.bulk_append(0, k_hist, v_hist)
        q_new = _rand_q(q_heads, device, seed=900 + L)

        ref = sdpa_ref(q_new, k_hist, v_hist, sm_scale)  # 黄金参考（真实历史）

        # L0 附加: gather 与真实历史一致
        gk, gv = mgr.gather(0)
        assert torch.equal(gk, k_hist) and torch.equal(gv, v_hist), f"gather mismatch L={L}"

        # paged 路径（优先 CUDA，否则 torch 参考）
        out = paged_decode_cuda(mgr, q_new, 0, sm_scale)
        tag = "cuda-paged"
        if out is None:
            out = paged_decode_ref(mgr, q_new, 0, sm_scale)
            tag = "torch-paged"
        a, r = err_stats(out, ref)
        ok = a <= tol_abs
        n_tiles = (L + 127) // 128
        empty_splits = max(0, NUM_SPLITS - n_tiles)
        last_partial = (L % 128) != 0
        print(
            f"    L={L:>5} [{tag}]  abs={a:.3e} rel={r:.3e}  "
            f"n_tiles={n_tiles:<3} empty_splits={empty_splits:<3} "
            f"last_partial={int(last_partial)}  {'ok' if ok else 'FAIL'}"
        )
        all_ok = all_ok and ok

    # L2: 连续内核交叉验证（仅 L%128==0）
    print("\n[L2] paged vs 连续内核 attention_decode (仅 seq_len%128==0)")
    for L in [128, 256, 512, 4096]:
        mgr = PagedKVManager(
            max_seqs=1, max_seq_len=L, kv_heads=kv_heads, head_dim=HEAD_DIM,
            device=device, poison=poison,
        )
        k_hist, v_hist = _rand_kv(kv_heads, L, device, seed=300 + L)
        mgr.bulk_append(0, k_hist, v_hist)
        q_new = _rand_q(q_heads, device, seed=950 + L)

        ref = sdpa_ref(q_new, k_hist, v_hist, sm_scale)
        cont = continuous_decode(q_new, k_hist, v_hist, sm_scale)
        gk, gv = mgr.gather(0)
        cont_paged = continuous_decode(q_new, gk, gv, sm_scale)  # gather 后喂连续内核

        a1, _ = err_stats(cont, ref)
        a2, _ = err_stats(cont_paged, cont)
        ok = (a1 <= tol_abs) and (a2 <= 1e-3)
        print(f"    L={L:>5}  cont-vs-sdpa abs={a1:.3e}  gather-vs-direct abs={a2:.3e}  {'ok' if ok else 'FAIL'}")
        all_ok = all_ok and ok

    print(f"\n  => 数值正确性 {'PASS' if all_ok else 'FAIL'}")
    return all_ok


# ---------------------------------------------------------------------------
# L3b: 增量 decode（逐步 append，模拟真实推理）
# ---------------------------------------------------------------------------
def test_incremental(kv_heads, q_heads, device, poison, tol_abs, steps, device_seed=0):
    print(f"\n[L3b] 增量 decode：逐步 append 到 seq_len={steps}，每步对拍 SDPA")
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    mgr = PagedKVManager(
        max_seqs=1, max_seq_len=steps, kv_heads=kv_heads, head_dim=HEAD_DIM,
        device=device, poison=poison,
    )
    k_hist, v_hist = _rand_kv(kv_heads, steps, device, seed=42)
    worst = 0.0
    check_at = set([1, 2, 127, 128, 129, 130, 255, 256, 257, steps])
    for t in range(steps):
        mgr.append_kv(0, k_hist[:, t, :], v_hist[:, t, :])
        L = t + 1
        if L not in check_at:
            continue
        q_new = _rand_q(q_heads, device, seed=1000 + L)
        ref = sdpa_ref(q_new, k_hist[:, :L, :], v_hist[:, :L, :], sm_scale)
        out = paged_decode_cuda(mgr, q_new, 0, sm_scale)
        if out is None:
            out = paged_decode_ref(mgr, q_new, 0, sm_scale)
        a, _ = err_stats(out, ref)
        worst = max(worst, a)
        print(f"    step L={L:>4}  abs={a:.3e}")
    ok = worst <= tol_abs
    print(f"  => L3b worst_abs={worst:.3e}  {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# L4: 性能 / 带宽
# ---------------------------------------------------------------------------
def bench(kv_heads, q_heads, kv_len, device, warmup, iters):
    print(f"\n[L4] 性能 @ seq_len={kv_len}")
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    mgr = PagedKVManager(
        max_seqs=1, max_seq_len=kv_len, kv_heads=kv_heads, head_dim=HEAD_DIM,
        device=device,
    )
    k_hist, v_hist = _rand_kv(kv_heads, kv_len, device, seed=7)
    mgr.bulk_append(0, k_hist, v_hist)
    q_new = _rand_q(q_heads, device, seed=8)

    def run_paged():
        out = paged_decode_cuda(mgr, q_new, 0, sm_scale)
        if out is None:
            raise RuntimeError("paged CUDA 内核未实现，无法测性能")
        return out

    # 连续内核基线（同一份 KV，L%128==0 时可比）
    kv_bytes = 2 * kv_heads * kv_len * HEAD_DIM * 2  # bf16, K+V 各读一遍

    def _time(fn):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters

    ms_cont = _time(lambda: continuous_decode(q_new, k_hist, v_hist, sm_scale))
    print(f"    连续内核  latency={ms_cont:.4f} ms  kv_bw={kv_bytes/ms_cont/1e9:.1f} GB/s")

    if PagedKVDecoder is not None:
        ms_paged = _time(run_paged)
        deg = (ms_paged - ms_cont) / ms_cont * 100
        print(f"    paged内核 latency={ms_paged:.4f} ms  kv_bw={kv_bytes/ms_paged/1e9:.1f} GB/s  退化={deg:+.1f}%")
        print("    (设计目标: 退化 <2%，即与连续版本基本持平)")
    else:
        print("    paged CUDA 内核未实现，跳过 paged 性能（实现后自动对比）")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Paged attention decode benchmark")
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--q-heads", type=int, default=64)
    ap.add_argument("--kv-len", type=int, default=131072, help="性能测试的 seq_len")
    ap.add_argument("--tol-abs", type=float, default=2e-2, help="bf16 decode 绝对误差上限")
    ap.add_argument("--tol-rel", type=float, default=1e-1)
    ap.add_argument("--incremental-steps", type=int, default=300)
    ap.add_argument("--poison", action="store_true", help="未用页填 NaN，压测末页 mask")
    ap.add_argument("--bench", action="store_true", help="额外跑 131072 性能")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA 设备")
    device = "cuda"
    assert args.q_heads == args.kv_heads * GROUP_M, "内核约定 q_heads == kv_heads*8"

    print("=" * 72)
    print(f"Paged Attention Decode Benchmark  (page_size={PAGE_SIZE}, num_splits={NUM_SPLITS})")
    print(f"q_heads={args.q_heads} kv_heads={args.kv_heads} head_dim={HEAD_DIM} poison={args.poison}")
    print("=" * 72)

    results = []
    results.append(("L0 manager", test_manager(args.kv_heads, args.q_heads, device, args.poison)))
    results.append(("L1/L2/L3 numeric", test_correctness(
        args.kv_heads, args.q_heads, device, args.poison, args.tol_abs, args.tol_rel)))
    results.append(("L3b incremental", test_incremental(
        args.kv_heads, args.q_heads, device, args.poison, args.tol_abs, args.incremental_steps)))

    if args.bench:
        bench(args.kv_heads, args.q_heads, args.kv_len, device, args.warmup, args.iters)

    print("\n" + "=" * 72)
    for name, ok in results:
        print(f"  {name:<20} {'PASS' if ok else 'FAIL'}")
    print("=" * 72)
    if not all(ok for _, ok in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# NOTES —— 与设计评审对应的修正点（供内核实现阶段核对）
# ---------------------------------------------------------------------------
# 1) split 再平衡用现有 floor 累积公式（均衡且无负 tile 数），只把 tiles_total
#    换成 n_tiles = ceil(seq_len/128)：
#        tile_beg = split * n_tiles // num_splits
#        tile_end = (split+1) * n_tiles // num_splits
#    不要用 ceil(n_tiles/num_splits) 前压式分法。
# 2) 末页 mask 的判据是 **全局** kv 下标：
#        global_row = (tile_beg + i) * 128 + tScS_mn[r,c][0]
#        if global_row >= seq_len:  acc_S_mn[r,c] = -inf   # 在列 max 之前
#    且仅全局最后一个 tile (tile_beg+i == n_tiles-1 且 seq_len%128!=0) 需要。
# 3) 空 split 现有 epilogue 已写出 O_part=0 / LSE=-inf（acc_O.fill(0)+row_sum=0），
#    改内核时务必保持 O_part 清零——combine 里 0*NaN=NaN 会污染。
# 4) block_table 不做整行寄存器预载（长度是运行期值）；循环内逐 tile 读，或对
#    i+1 软件预取即可，L2 常驻、延迟被 TMA 掩盖。
# 5) 页内布局用 [num_pages, kv_heads, page_size, head_dim]，使每个 (page,kv_head)
#    的 128x128 tile 连续，便于 TMA/swizzle/对齐。
# 6) 新分配页首用前清零（本 manager 默认 0；--poison 故意填 NaN 以验证 mask）。
