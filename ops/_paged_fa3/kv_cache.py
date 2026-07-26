"""kv_cache.py - vLLM-style paged KV-cache memory pool + block allocator.

Task 2 memory management. A single big KV tensor is carved into fixed-size
PAGE-token blocks. Sequences grab blocks from a free-list on demand and record
their logical->physical mapping in a per-sequence block table. This is what lets
an engine serve many long sequences without pre-allocating a contiguous
[max_seqs, max_len, ...] KV tensor (which would waste memory to the longest
possible sequence and fragment badly).

Layout matches the paged decode kernels (cute_mma_decode.paged_mma_decode_cute,
cute_decode.paged_decode_cute, triton_decode.paged_decode_attention):

    k_cache / v_cache : [num_blocks, PAGE, Hkv, D]   (one pool tensor each)
    block_table[b]    : list of physical page ids for logical blocks of seq b
    seq_lens[b]       : current token count of seq b

The pool never moves a block once allocated; growth just appends a new physical
page to the sequence's block table. Freeing returns pages to the free-list for
reuse -> no fragmentation of the KV bytes themselves.
"""
from __future__ import annotations

import torch


class PagedKVCache:
    """Fixed-size paged KV block pool with a free-list allocator.

    Args:
        num_blocks: total physical pages in the pool.
        page:       tokens per page (block size). Must match the kernel n_block.
        num_kv_heads, head_dim: KV geometry.
        dtype, device: pool tensor dtype/device (bf16 on cuda for the kernels).
    """

    def __init__(self, num_blocks: int, page: int, num_kv_heads: int,
                 head_dim: int, dtype=torch.bfloat16, device="cuda"):
        self.num_blocks = int(num_blocks)
        self.page = int(page)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.dtype = dtype
        self.device = device

        # the two pool tensors — this is ALL the KV memory that ever exists.
        self.k_cache = torch.zeros(num_blocks, page, num_kv_heads, head_dim,
                                   dtype=dtype, device=device)
        self.v_cache = torch.zeros(num_blocks, page, num_kv_heads, head_dim,
                                   dtype=dtype, device=device)

        # free-list of physical page ids (stack; pop/push are O(1)).
        self._free = list(range(num_blocks))
        # per-sequence state: seq_id -> {"blocks": [phys,...], "len": int}
        self._seqs: dict[int, dict] = {}

    # -------------------------------------------------------------- pool stats
    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - len(self._free)

    def pool_bytes(self) -> int:
        return self.k_cache.numel() * self.k_cache.element_size() + \
            self.v_cache.numel() * self.v_cache.element_size()

    # ------------------------------------------------------------- allocation
    def _alloc_block(self) -> int:
        if not self._free:
            raise MemoryError(
                f"KV pool exhausted: all {self.num_blocks} blocks in use "
                f"(need to free a sequence or grow the pool).")
        return self._free.pop()

    def add_sequence(self, seq_id: int, prompt_len: int = 0) -> None:
        """Register a new sequence, pre-allocating enough blocks for prompt_len."""
        if seq_id in self._seqs:
            raise ValueError(f"sequence {seq_id} already exists")
        need = (prompt_len + self.page - 1) // self.page
        blocks = [self._alloc_block() for _ in range(need)]
        self._seqs[seq_id] = {"blocks": blocks, "len": int(prompt_len)}

    def free_sequence(self, seq_id: int) -> None:
        """Return all of a sequence's blocks to the free-list."""
        s = self._seqs.pop(seq_id, None)
        if s is None:
            return
        # push back (reused LIFO); order irrelevant for correctness.
        self._free.extend(s["blocks"])

    def append_token(self, seq_id: int) -> int:
        """Grow a sequence by one token, allocating a new page if needed.
        Returns the new sequence length."""
        s = self._seqs[seq_id]
        new_len = s["len"] + 1
        need = (new_len + self.page - 1) // self.page
        while len(s["blocks"]) < need:
            s["blocks"].append(self._alloc_block())
        s["len"] = new_len
        return new_len

    # ------------------------------------------------------------------ writes
    def write_prompt(self, seq_id: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Scatter a prompt's contiguous K/V [T, Hkv, D] into the sequence's pages."""
        s = self._seqs[seq_id]
        T = k.shape[0]
        assert v.shape[0] == T
        assert s["len"] >= T, "call add_sequence(prompt_len=T) first"
        for i, phys in enumerate(s["blocks"]):
            lo = i * self.page
            if lo >= T:
                break
            hi = min(lo + self.page, T)
            n = hi - lo
            self.k_cache[phys, :n] = k[lo:hi]
            self.v_cache[phys, :n] = v[lo:hi]

    def write_token(self, seq_id: int, k_tok: torch.Tensor, v_tok: torch.Tensor,
                    pos: int) -> None:
        """Write one token's K/V [Hkv, D] at absolute position `pos`."""
        s = self._seqs[seq_id]
        phys = s["blocks"][pos // self.page]
        off = pos % self.page
        self.k_cache[phys, off] = k_tok
        self.v_cache[phys, off] = v_tok

    # -------------------------------------------------------- kernel interface
    def build_block_table(self, seq_ids: list[int]):
        """Return (block_table [B, max_blocks] int32, seq_lens [B] int32) for the
        given sequences, right-padded so the kernel gets a rectangular table."""
        max_blocks = max(len(self._seqs[s]["blocks"]) for s in seq_ids)
        B = len(seq_ids)
        bt = torch.zeros(B, max_blocks, dtype=torch.int32, device=self.device)
        sl = torch.zeros(B, dtype=torch.int32, device=self.device)
        for b, sid in enumerate(seq_ids):
            s = self._seqs[sid]
            blk = s["blocks"]
            bt[b, :len(blk)] = torch.tensor(blk, dtype=torch.int32,
                                            device=self.device)
            sl[b] = s["len"]
        return bt, sl

    def seq_len(self, seq_id: int) -> int:
        return self._seqs[seq_id]["len"]
