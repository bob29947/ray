"""Focused CPU-only tests for the cloud map-fusion benchmark contract."""

from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import benchmark_core as core
import worker


def _summary(index, rows, hash_sums, numeric=None, categorical=None):
    return {
        "block_index": index,
        "rows": rows,
        "bytes": rows * 4,
        "hash_sums": hash_sums,
        "null_count": 0,
        "nan_count": 0,
        "numeric_zero_counts": numeric or [0] * core.NUMERIC_COLUMN_COUNT,
        "categorical_zero_counts": categorical or [0] * core.CATEGORICAL_COLUMN_COUNT,
        "categorical_range_violations": 0,
    }


class WorkloadContractTest(unittest.TestCase):
    def test_locked_shape_and_cardinality_multiset(self):
        self.assertEqual(len(core.numeric_columns()), 681)
        self.assertEqual(len(core.categorical_columns()), 74)
        self.assertEqual(core.TOTAL_COLUMN_COUNT, 755)

        cardinalities = core.categorical_cardinalities()
        self.assertEqual(len(cardinalities), 74)
        self.assertEqual(sum(cardinalities), 150_738)
        self.assertEqual(min(cardinalities), 2)
        self.assertEqual(max(cardinalities), 15_345)
        self.assertEqual(
            Counter(cardinalities),
            Counter(
                {
                    15_345: 1,
                    12_000: 2,
                    8_000: 4,
                    4_000: 8,
                    2_500: 12,
                    1_000: 15,
                    200: 10,
                    19: 20,
                    11: 1,
                    2: 1,
                }
            ),
        )

    def test_expected_missing_counts_match_brute_force(self):
        rows = 237
        numeric = core.expected_numeric_zero_counts(rows)
        categorical = core.expected_categorical_zero_counts(rows)
        for index in range(core.NUMERIC_COLUMN_COUNT):
            expected = sum((row + 7 * index) % 100 in (0, 1) for row in range(rows))
            self.assertEqual(numeric[index], expected)
        for index in range(core.CATEGORICAL_COLUMN_COUNT):
            expected = sum((row + 11 * index) % 100 in (0, 1) for row in range(rows))
            self.assertEqual(categorical[index], expected)

    def test_schema_is_exact_and_ordered(self):
        schema = core.expected_schema_pairs()
        self.assertEqual(len(schema), 755)
        self.assertEqual(schema[:2], [("n0000", "float"), ("n0001", "float")])
        self.assertEqual(schema[680], ("n0680", "float"))
        self.assertEqual(schema[681], ("c000", "int32"))
        self.assertEqual(schema[-1], ("c073", "int32"))
        core.validate_schema(schema)
        with self.assertRaises(core.TrialRejected):
            core.validate_schema(schema[:-1])


class PlanAndDigestTest(unittest.TestCase):
    def test_plan_contract(self):
        isolated = [f"MapBatches({name})" for name in core.STAGE_NAMES]
        self.assertEqual(
            core.validate_actor_plan("isolated", isolated)["actor_map_count"], 5
        )
        self.assertEqual(
            core.validate_actor_plan("fused", [core.FUSED_PLAN_NAME])[
                "actor_map_count"
            ],
            1,
        )
        with self.assertRaises(core.TrialRejected):
            core.validate_actor_plan("fused", isolated)

    def test_digest_is_independent_of_block_boundaries_and_order(self):
        split = core.combine_block_summaries(
            [_summary(1, 3, [20, 40]), _summary(0, 2, [10, 30])]
        )
        joined = core.combine_block_summaries([_summary(0, 5, [30, 70])])
        self.assertEqual(split["digest"], joined["digest"])
        self.assertEqual(split["hash_sums"], [30, 70])
        self.assertEqual(split["rows"], 5)

    def test_correctness_accepts_locked_counts(self):
        rows = 10_003
        summary = _summary(
            0,
            rows,
            [1, 2],
            core.expected_numeric_zero_counts(rows),
            core.expected_categorical_zero_counts(rows),
        )
        combined = core.combine_block_summaries([summary])
        self.assertEqual(core.correctness_reasons(combined, rows), [])
        combined["nan_count"] = 1
        self.assertIn(
            "final output contains 1 NaNs", core.correctness_reasons(combined, rows)
        )


class MetricsAndReceiptTest(unittest.TestCase):
    def test_mount_and_diskstats_parsing(self):
        mountinfo = "\n".join(
            (
                "24 1 8:1 / / rw - ext4 /dev/root rw",
                "25 24 259:7 / /mnt/nvme rw - ext4 /dev/nvme2n1 rw",
            )
        )
        self.assertEqual(worker._mount_device_number(mountinfo), (259, 7))
        diskstats = "259 7 nvme2n1 1 2 3 4 5 6 7 8 9 10 11"
        self.assertEqual(
            worker._diskstats_bytes(diskstats, (259, 7)),
            (3 * 512, 7 * 512, "nvme2n1"),
        )

    def test_batch_probe_requires_equal_totals_and_seven_fused_gpus(self):
        snapshot = {
            "stages": {
                name: {"rows": 1_000, "batches": 4} for name in core.STAGE_NAMES
            },
            "gpu_slots": [f"node-{index}:0" for index in range(7)],
        }
        self.assertEqual(core.batch_probe_reasons(snapshot, 1_000, "fused"), [])
        snapshot["gpu_slots"].pop()
        self.assertTrue(core.batch_probe_reasons(snapshot, 1_000, "fused"))

    def test_ray_data_metrics_are_collected_recursively(self):
        parent = SimpleNamespace(
            parents=[],
            base_name="first",
            extra_metrics={"num_tasks_finished": 2, "bytes_inputs_received": 10},
        )
        root = SimpleNamespace(
            parents=[parent],
            base_name="second",
            extra_metrics={"num_tasks_finished": 3, "bytes_inputs_received": 20},
            global_bytes_spilled=0,
            global_bytes_restored=0,
            dataset_bytes_spilled=0,
        )
        metrics = core.collect_ray_data_metrics(root)
        self.assertEqual(metrics["totals"]["num_tasks_finished"], 5)
        self.assertEqual(metrics["totals"]["bytes_inputs_received"], 30)
        self.assertEqual(
            [item["name"] for item in metrics["operators"]], ["first", "second"]
        )

    def test_atomic_json_receipt_replaces_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "results", "result.json")
            core.atomic_write_json(path, {"valid": False})
            core.atomic_write_json(path, {"valid": True, "mode": "fused"})
            self.assertEqual(
                json.loads(path.read_text()), {"valid": True, "mode": "fused"}
            )
            self.assertEqual(list(path.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
