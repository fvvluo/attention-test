import unittest

from decode_split_config import (
    compute_decode_split_config,
    format_decode_split_config,
    split_block_range,
)


class DecodeSplitConfigTest(unittest.TestCase):
    def h20_config(self, num_splits=None):
        return compute_decode_split_config(
            batch_size=1,
            q_heads=64,
            kv_heads=8,
            kv_len=131072,
            num_sms=78,
            num_splits=num_splits,
        )

    def assert_complete_coverage(self, config):
        covered_blocks = []
        split_counts = []
        for split_idx in range(config.num_splits):
            first_n_block, split_count = split_block_range(config, split_idx)
            split_counts.append(split_count)
            covered_blocks.extend(range(first_n_block, first_n_block + split_count))

        self.assertEqual(covered_blocks, list(range(config.num_blocks)))
        self.assertEqual(len(covered_blocks), len(set(covered_blocks)))
        self.assertLessEqual(max(split_counts) - min(split_counts), 1)
        self.assertEqual(covered_blocks[-1], config.num_blocks - 1)
        return split_counts

    def test_h20_primary_shape_auto_config(self):
        config = self.h20_config()

        self.assertEqual(config.num_sms, 78)
        self.assertEqual(config.target_ctas_per_sm, 3)
        self.assertEqual(config.target_grid_blocks, 234)
        self.assertEqual(config.num_blocks, 2048)
        self.assertEqual(config.q_groups, 1)
        self.assertEqual(config.split_fanout, 8)
        self.assertEqual(config.auto_num_splits, 29)
        self.assertIsNone(config.requested_num_splits)
        self.assertEqual(config.num_splits, 29)
        self.assertEqual(config.actual_grid_blocks, 232)
        self.assertAlmostEqual(config.estimated_waves, 232 / 234)
        self.assertEqual(config.base_blocks_per_split, 70)
        self.assertEqual(config.long_splits, 18)
        self.assertEqual(config.min_split_blocks, 70)
        self.assertEqual(config.max_split_blocks, 71)
        self.assertEqual((config.num_splits, 8, 1), (29, 8, 1))

    def test_h20_primary_shape_distribution_and_coverage(self):
        split_counts = self.assert_complete_coverage(self.h20_config())

        self.assertEqual(split_counts[:18], [71] * 18)
        self.assertEqual(split_counts[18:], [70] * 11)

    def test_twenty_split_override(self):
        config = self.h20_config(num_splits=20)

        self.assertEqual(config.auto_num_splits, 29)
        self.assertEqual(config.requested_num_splits, 20)
        self.assertEqual(config.num_splits, 20)
        self.assertEqual(config.actual_grid_blocks, 160)
        self.assertAlmostEqual(config.estimated_waves, 160 / 234)
        self.assertEqual(config.base_blocks_per_split, 102)
        self.assertEqual(config.long_splits, 8)
        self.assertEqual(config.blocks_per_split, 103)
        self.assert_complete_coverage(config)

    def test_auto_splits_are_limited_by_num_blocks(self):
        config = compute_decode_split_config(
            batch_size=1,
            q_heads=8,
            kv_heads=1,
            kv_len=128,
            num_sms=78,
        )

        self.assertEqual(config.num_blocks, 2)
        self.assertEqual(config.num_splits, 2)
        self.assertEqual(self.assert_complete_coverage(config), [1, 1])

    def test_large_fanout_still_uses_one_split(self):
        config = compute_decode_split_config(
            batch_size=32,
            q_heads=64,
            kv_heads=8,
            kv_len=1024,
            num_sms=1,
        )

        self.assertGreater(config.split_fanout, config.target_grid_blocks)
        self.assertEqual(config.auto_num_splits, 1)
        self.assertEqual(config.num_splits, 1)
        self.assert_complete_coverage(config)

    def test_q_groups_increase_split_fanout(self):
        config = compute_decode_split_config(
            batch_size=1,
            q_heads=72,
            kv_heads=8,
            kv_len=131072,
            num_sms=78,
        )

        self.assertEqual(config.q_groups, 2)
        self.assertEqual(config.split_fanout, 16)
        self.assertEqual(config.num_splits, 14)
        self.assert_complete_coverage(config)

    def test_invalid_overrides_are_rejected(self):
        for num_splits in (0, -1, 2049, True, 1.5):
            with self.subTest(num_splits=num_splits):
                with self.assertRaises(ValueError):
                    self.h20_config(num_splits=num_splits)

    def test_invalid_shape_inputs_are_rejected(self):
        valid = {
            "batch_size": 1,
            "q_heads": 64,
            "kv_heads": 8,
            "kv_len": 131072,
            "num_sms": 78,
        }
        for name in valid:
            invalid = dict(valid)
            invalid[name] = 0
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    compute_decode_split_config(**invalid)

        invalid_heads = dict(valid)
        invalid_heads["q_heads"] = 63
        with self.assertRaises(ValueError):
            compute_decode_split_config(**invalid_heads)

    def test_split_index_validation(self):
        config = self.h20_config()
        for split_idx in (-1, config.num_splits, True, 1.5):
            with self.subTest(split_idx=split_idx):
                with self.assertRaises(ValueError):
                    split_block_range(config, split_idx)

    def test_diagnostic_format(self):
        diagnostic = format_decode_split_config(self.h20_config())

        for expected in (
            "source=auto",
            "num_sms=78",
            "target_ctas_per_sm=3",
            "target_grid_blocks=234",
            "split_fanout=8",
            "num_splits=29",
            "actual_grid_blocks=232",
            "estimated_waves=0.991453",
            "base_blocks_per_split=70",
            "long_splits=18",
            "min_split_blocks=70",
            "max_split_blocks=71",
        ):
            self.assertIn(expected, diagnostic)


if __name__ == "__main__":
    unittest.main()
