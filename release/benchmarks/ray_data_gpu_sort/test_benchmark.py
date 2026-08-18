"""Focused CPU-only checks for the portable benchmark contract."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from .data import cohort_slices, plan_dict, scaled_slices, smoke_slices
from .spec import EXPECTED_BLOCKS, EXPECTED_ROWS, cell_by_name, cells
from .worker import _tuple_leq


def _manifest() -> dict:
    names = [f"column_{index}" for index in range(109)]
    names[:5] = ["Origin", "Dest", "FlightDate", "CRSDepTime", "OriginAirportID"]
    names[55] = "DistanceGroup"
    quotient, remainder = divmod(EXPECTED_ROWS, EXPECTED_BLOCKS)
    row_groups = []
    for index in range(EXPECTED_BLOCKS):
        month_index = index % 153
        year = 2013 + (3 + month_index) // 12
        month = (3 + month_index) % 12 + 1
        row_groups.append(
            {
                "path": f"parquet/year={year}/month={month:02d}/part-{index}.parquet",
                "row_group": 0,
                "rows": quotient + (index < remainder),
                "year": year,
                "month": month,
            }
        )
    return {"schema_names": names, "row_groups": row_groups}


class WorkloadTest(unittest.TestCase):
    def test_six_cells_keep_their_keys(self):
        manifest = _manifest()
        self.assertEqual(len(cells(manifest)), 6)
        self.assertEqual(len(cell_by_name(manifest, "narrow").columns), 5)
        self.assertEqual(len(cell_by_name(manifest, "core").columns), 57)
        self.assertEqual(len(cell_by_name(manifest, "full").columns), 110)

    def test_exact_cohort_and_scales(self):
        manifest = _manifest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = cohort_slices(root, manifest)
            scaled = scaled_slices(root, manifest, 245, 100)
            smoke = smoke_slices(root, manifest)
        self.assertEqual(len(base), EXPECTED_BLOCKS)
        self.assertEqual(sum(item.rows for item in base), EXPECTED_ROWS)
        self.assertEqual(sum(item.rows for item in scaled), EXPECTED_ROWS * 245 // 100)
        self.assertGreater(len(scaled), 2 * EXPECTED_BLOCKS)
        self.assertLess(len(scaled), 3 * EXPECTED_BLOCKS)
        self.assertEqual(sum(item.rows for item in smoke), 160_000)
        self.assertEqual(
            [item.row_id_start for item in smoke], list(range(0, 160_000, 10_000))
        )

    def test_plan_digest_is_stable(self):
        manifest = _manifest()
        with tempfile.TemporaryDirectory() as directory:
            slices = cohort_slices(Path(directory), manifest)
        first = plan_dict(slices, kind="trend")
        second = plan_dict(slices, kind="trend")
        self.assertEqual(first["digest"], second["digest"])
        self.assertEqual(first["rows"], EXPECTED_ROWS)


class OrderingTest(unittest.TestCase):
    def test_ascending_null_last_tuple_order(self):
        self.assertTrue(_tuple_leq(("ATL", 1), ("ATL", 2)))
        self.assertTrue(_tuple_leq(("ATL", 2), ("ATL", 2)))
        self.assertTrue(_tuple_leq(("ATL", 2), (None, 1)))
        self.assertFalse(_tuple_leq((None, 1), ("ATL", 2)))


if __name__ == "__main__":
    unittest.main()
