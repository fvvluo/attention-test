from dataclasses import dataclass
from typing import Optional, Tuple


DECODE_TILE_N = 64
DECODE_QHEADS_PER_CTA = 8
DECODE_TARGET_CTAS_PER_SM = 2


@dataclass(frozen=True)
class DecodeSplitConfig:
    num_sms: int
    target_ctas_per_sm: int
    target_grid_blocks: int
    num_blocks: int
    q_groups: int
    split_fanout: int
    auto_num_splits: int
    requested_num_splits: Optional[int]
    num_splits: int
    actual_grid_blocks: int
    estimated_waves: float
    base_blocks_per_split: int
    long_splits: int
    blocks_per_split: int

    @property
    def min_split_blocks(self) -> int:
        return self.base_blocks_per_split

    @property
    def max_split_blocks(self) -> int:
        return self.blocks_per_split


def _require_positive(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def compute_decode_split_config(
    batch_size: int,
    q_heads: int,
    kv_heads: int,
    kv_len: int,
    num_sms: int,
    num_splits: Optional[int] = None,
    tile_n: int = DECODE_TILE_N,
    qheads_per_cta: int = DECODE_QHEADS_PER_CTA,
    target_ctas_per_sm: int = DECODE_TARGET_CTAS_PER_SM,
) -> DecodeSplitConfig:
    for name, value in (
        ("batch_size", batch_size),
        ("q_heads", q_heads),
        ("kv_heads", kv_heads),
        ("kv_len", kv_len),
        ("num_sms", num_sms),
        ("tile_n", tile_n),
        ("qheads_per_cta", qheads_per_cta),
        ("target_ctas_per_sm", target_ctas_per_sm),
    ):
        _require_positive(name, value)
    if q_heads % kv_heads != 0:
        raise ValueError(f"q_heads={q_heads} must be divisible by kv_heads={kv_heads}")

    num_blocks = (kv_len + tile_n - 1) // tile_n
    q_heads_per_kv_head = q_heads // kv_heads
    q_groups = (q_heads_per_kv_head + qheads_per_cta - 1) // qheads_per_cta
    split_fanout = batch_size * kv_heads * q_groups
    target_grid_blocks = num_sms * target_ctas_per_sm
    auto_num_splits = max(
        1,
        min(num_blocks, target_grid_blocks // split_fanout),
    )

    if num_splits is not None:
        if isinstance(num_splits, bool) or not isinstance(num_splits, int):
            raise ValueError(f"num_splits must be an integer, got {num_splits!r}")
        if not 1 <= num_splits <= num_blocks:
            raise ValueError(
                f"num_splits must be in [1, {num_blocks}], got {num_splits}"
            )
        resolved_num_splits = num_splits
    else:
        resolved_num_splits = auto_num_splits

    base_blocks_per_split = num_blocks // resolved_num_splits
    long_splits = num_blocks % resolved_num_splits
    blocks_per_split = base_blocks_per_split + (1 if long_splits else 0)
    actual_grid_blocks = resolved_num_splits * split_fanout

    return DecodeSplitConfig(
        num_sms=num_sms,
        target_ctas_per_sm=target_ctas_per_sm,
        target_grid_blocks=target_grid_blocks,
        num_blocks=num_blocks,
        q_groups=q_groups,
        split_fanout=split_fanout,
        auto_num_splits=auto_num_splits,
        requested_num_splits=num_splits,
        num_splits=resolved_num_splits,
        actual_grid_blocks=actual_grid_blocks,
        estimated_waves=actual_grid_blocks / target_grid_blocks,
        base_blocks_per_split=base_blocks_per_split,
        long_splits=long_splits,
        blocks_per_split=blocks_per_split,
    )


def split_block_range(config: DecodeSplitConfig, split_idx: int) -> Tuple[int, int]:
    if isinstance(split_idx, bool) or not isinstance(split_idx, int):
        raise ValueError(f"split_idx must be an integer, got {split_idx!r}")
    if not 0 <= split_idx < config.num_splits:
        raise ValueError(
            f"split_idx must be in [0, {config.num_splits}), got {split_idx}"
        )

    split_count = config.base_blocks_per_split
    if split_idx < config.long_splits:
        split_count += 1
    first_n_block = (
        split_idx * config.base_blocks_per_split
        + min(split_idx, config.long_splits)
    )
    return first_n_block, split_count


def format_decode_split_config(config: DecodeSplitConfig) -> str:
    source = "override" if config.requested_num_splits is not None else "auto"
    return (
        "[decode-config] "
        f"source={source} "
        f"num_sms={config.num_sms} "
        f"target_ctas_per_sm={config.target_ctas_per_sm} "
        f"target_grid_blocks={config.target_grid_blocks} "
        f"split_fanout={config.split_fanout} "
        f"num_blocks={config.num_blocks} "
        f"q_groups={config.q_groups} "
        f"auto_num_splits={config.auto_num_splits} "
        f"num_splits={config.num_splits} "
        f"actual_grid_blocks={config.actual_grid_blocks} "
        f"estimated_waves={config.estimated_waves:.6f} "
        f"base_blocks_per_split={config.base_blocks_per_split} "
        f"long_splits={config.long_splits} "
        f"min_split_blocks={config.min_split_blocks} "
        f"max_split_blocks={config.max_split_blocks}"
    )
