from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = (REPO_ROOT / "ops" / "lhx_flash_attention.py").read_text()
GQA_SOURCE = SOURCE.split("class GqaDecodeSm90:", 1)[1].split(
    "# ops 接口封装", 1
)[0]


class DecodeSplitWiringTest(unittest.TestCase):
    def test_main_launch_targets_three_ctas_per_sm(self):
        self.assertIn(
            "grid=(self.num_splits, self.kv_heads * self.q_groups, self.batch_size)",
            GQA_SOURCE,
        )
        self.assertIn("min_blocks_per_mp=3", GQA_SOURCE)
        self.assertNotIn("min_blocks_per_mp=2", GQA_SOURCE)

    def test_partial_buffers_use_resolved_split_count(self):
        self.assertIn(
            "(batch, resolved_num_splits, q_heads, head_dim)",
            SOURCE,
        )
        self.assertIn("(batch, resolved_num_splits, q_heads)", SOURCE)

    def test_cache_and_kernel_constructor_use_resolved_split_count(self):
        decode_host = SOURCE.split("def _run_decode_sm90(", 1)[1].split(
            "def prefill_attention(", 1
        )[0]
        cache_key = decode_host.split("key = (", 1)[1].split(")\n    compiled", 1)[0]
        constructor = decode_host.split("decode = GqaDecodeSm90(", 1)[1].split(
            ")\n", 1
        )[0]

        self.assertIn("resolved_num_splits", cache_key)
        self.assertIn("blocks_per_split", cache_key)
        self.assertIn("resolved_num_splits", constructor)
        self.assertIn("blocks_per_split", constructor)

    def test_reduce_consumes_all_resolved_splits(self):
        self.assertEqual(
            GQA_SOURCE.count(
                "for split_idx in cutlass.range_constexpr(self.num_splits):"
            ),
            2,
        )

    def test_override_and_diagnostics_are_opt_in(self):
        self.assertIn("num_splits: Optional[int] = None", SOURCE)
        self.assertIn("debug_config: bool = False", SOURCE)
        self.assertIn("if debug_config:\n        print(format_decode_split_config(split_config))", SOURCE)


if __name__ == "__main__":
    unittest.main()
