"""CPU-only helpers for the wide cuDF actor-map fusion benchmark.

This module intentionally has no import-time Ray or cuDF dependency.  The cloud
worker imports it both on the driver and in ordinary CPU tasks used to create and
validate Arrow blocks.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


NUMERIC_COLUMN_COUNT = 681
CATEGORICAL_COLUMN_COUNT = 74
TOTAL_COLUMN_COUNT = NUMERIC_COLUMN_COUNT + CATEGORICAL_COLUMN_COUNT

NUMERIC_PERIOD = 100
CATEGORICAL_PERIOD = 100

STAGE_NAMES = (
    "NumericLog1p",
    "NumericStandardize",
    "NumericFillNulls",
    "CategoricalToString",
    "CategoricalVocabularyLookup",
)
FUSED_PLAN_NAME = f"MapBatches({'->'.join(STAGE_NAMES)})"


class TrialRejected(RuntimeError):
    """Raised when an observation violates the benchmark acceptance contract."""


def numeric_columns() -> tuple[str, ...]:
    return tuple(f"n{index:04d}" for index in range(NUMERIC_COLUMN_COUNT))


def categorical_columns() -> tuple[str, ...]:
    return tuple(f"c{index:03d}" for index in range(CATEGORICAL_COLUMN_COUNT))


def categorical_cardinalities() -> tuple[int, ...]:
    """Return the locked 74-column fitted-vocabulary cardinality permutation.

    The multiset ranges from 2 to 15,345 and contains 150,738 entries in total:
    ``1x15345, 2x12000, 4x8000, 8x4000, 12x2500, 15x1000,
    10x200, 20x19, 1x11, 1x2``.  An affine permutation interleaves the groups
    deterministically without depending on ``random`` implementation details.
    """

    grouped = (
        [15_345]
        + [12_000] * 2
        + [8_000] * 4
        + [4_000] * 8
        + [2_500] * 12
        + [1_000] * 15
        + [200] * 10
        + [19] * 20
        + [11]
        + [2]
    )
    assert len(grouped) == CATEGORICAL_COLUMN_COUNT
    return tuple(grouped[(17 * index + 23) % len(grouped)] for index in range(74))


def numeric_means() -> tuple[float, ...]:
    # Every non-missing generated value is non-negative.  Negative fitted means
    # make a post-standardization zero an unambiguous missing-value fill.
    return tuple(-0.5 - (index % 17) / 32.0 for index in range(NUMERIC_COLUMN_COUNT))


def numeric_scales() -> tuple[float, ...]:
    return tuple(0.75 + (index % 13) / 16.0 for index in range(NUMERIC_COLUMN_COUNT))


def expected_schema_pairs() -> list[tuple[str, str]]:
    return [
        *(zip(numeric_columns(), ("float",) * NUMERIC_COLUMN_COUNT)),
        *(zip(categorical_columns(), ("int32",) * CATEGORICAL_COLUMN_COUNT)),
    ]


def count_residue(rows: int, modulus: int, residue: int) -> int:
    """Count ``0 <= value < rows`` with ``value % modulus == residue``."""

    if rows < 0:
        raise ValueError("rows must be non-negative")
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    residue %= modulus
    if rows <= residue:
        return 0
    return 1 + (rows - 1 - residue) // modulus


def expected_numeric_zero_counts(rows: int) -> list[int]:
    """Expected fill-zero counts after the numeric missing-value stage."""

    result = []
    for index in range(NUMERIC_COLUMN_COUNT):
        offset = 7 * index
        null_residue = (-offset) % NUMERIC_PERIOD
        nan_residue = (1 - offset) % NUMERIC_PERIOD
        result.append(
            count_residue(rows, NUMERIC_PERIOD, null_residue)
            + count_residue(rows, NUMERIC_PERIOD, nan_residue)
        )
    return result


def expected_categorical_zero_counts(rows: int) -> list[int]:
    """Expected null/OOV-to-zero counts after vocabulary lookup."""

    result = []
    for index in range(CATEGORICAL_COLUMN_COUNT):
        offset = 11 * index
        null_residue = (-offset) % CATEGORICAL_PERIOD
        oov_residue = (1 - offset) % CATEGORICAL_PERIOD
        result.append(
            count_residue(rows, CATEGORICAL_PERIOD, null_residue)
            + count_residue(rows, CATEGORICAL_PERIOD, oov_residue)
        )
    return result


def make_arrow_block(start: int, stop: int):
    """Create one deterministic Arrow block for global row range ``[start, stop)``."""

    if start < 0 or stop < start:
        raise ValueError(f"invalid row range: [{start}, {stop})")

    import numpy as np
    import pyarrow as pa

    row_ids = np.arange(start, stop, dtype=np.int64)
    numeric_residues = row_ids % NUMERIC_PERIOD
    categorical_residues = row_ids % CATEGORICAL_PERIOD
    arrays = []

    for index in range(NUMERIC_COLUMN_COUNT):
        values = (
            (row_ids % 4093).astype(np.float32) + np.float32((index % 97) / 8.0)
        ) / np.float32(256.0)
        shifted = (numeric_residues + 7 * index) % NUMERIC_PERIOD
        null_mask = shifted == 0
        values[shifted == 1] = np.nan
        arrays.append(pa.array(values, mask=null_mask, type=pa.float32()))

    for index, cardinality in enumerate(categorical_cardinalities()):
        # Unit stride visits every fitted token even for non-power-of-two sizes.
        values = ((row_ids + index) % cardinality).astype(np.int32)
        shifted = (categorical_residues + 11 * index) % CATEGORICAL_PERIOD
        null_mask = shifted == 0
        values[shifted == 1] = cardinality + 7
        arrays.append(pa.array(values, mask=null_mask, type=pa.int32()))

    return pa.Table.from_arrays(
        arrays,
        names=(*numeric_columns(), *categorical_columns()),
    )


def normalize_schema(schema: Any) -> list[tuple[str, str]]:
    """Convert a PyArrow or Ray schema to stable ``(name, type)`` pairs."""

    base_schema = getattr(schema, "base_schema", schema)
    return [(field.name, str(field.type)) for field in base_schema]


def validate_schema(schema_pairs: Sequence[Sequence[str]]) -> None:
    actual = [tuple(pair) for pair in schema_pairs]
    expected = expected_schema_pairs()
    if actual != expected:
        first_difference = next(
            (
                index
                for index, pair in enumerate(zip(actual, expected))
                if pair[0] != pair[1]
            ),
            min(len(actual), len(expected)),
        )
        raise TrialRejected(
            "output schema mismatch at column "
            f"{first_difference}: got {actual[first_difference : first_difference + 1]}, "
            f"expected {expected[first_difference : first_difference + 1]}"
        )


def expected_actor_plan_names(mode: str) -> list[str]:
    if mode == "isolated":
        return [f"MapBatches({name})" for name in STAGE_NAMES]
    if mode == "fused":
        return [FUSED_PLAN_NAME]
    raise ValueError(f"unknown mode: {mode!r}")


def validate_actor_plan(mode: str, actor_map_names: Sequence[str]) -> dict[str, Any]:
    actual = list(actor_map_names)
    expected = expected_actor_plan_names(mode)
    if actual != expected:
        raise TrialRejected(
            f"{mode} physical plan had actor maps {actual!r}; expected {expected!r}"
        )
    return {
        "actor_map_count": len(actual),
        "actor_map_names": actual,
        "expected_actor_map_names": expected,
    }


def combine_block_summaries(summaries: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Combine block summaries into a boundary- and order-independent digest.

    Each validation task hashes every row twice with cuDF xxhash64 and returns the
    two uint64 sums.  Addition modulo 2**64 makes the final value independent of
    Ray's output block boundaries and scheduling order.
    """

    ordered = sorted(summaries, key=lambda item: int(item["block_index"]))
    indices = [int(item["block_index"]) for item in ordered]
    if indices != list(range(len(ordered))):
        raise TrialRejected(f"digest blocks were not contiguous: {indices!r}")

    hash_sums = [0, 0]
    numeric_zero_counts = [0] * NUMERIC_COLUMN_COUNT
    categorical_zero_counts = [0] * CATEGORICAL_COLUMN_COUNT
    result = {
        "rows": 0,
        "bytes": 0,
        "blocks": len(ordered),
        "null_count": 0,
        "nan_count": 0,
        "categorical_range_violations": 0,
    }

    for item in ordered:
        for seed_index, value in enumerate(item["hash_sums"]):
            hash_sums[seed_index] = (hash_sums[seed_index] + int(value)) % (1 << 64)
        for key in (
            "rows",
            "bytes",
            "null_count",
            "nan_count",
            "categorical_range_violations",
        ):
            result[key] += int(item[key])
        for index, count in enumerate(item["numeric_zero_counts"]):
            numeric_zero_counts[index] += int(count)
        for index, count in enumerate(item["categorical_zero_counts"]):
            categorical_zero_counts[index] += int(count)

    result.update(
        {
            "digest": (f"xxh64-row-sum-v1:{hash_sums[0]:016x}:{hash_sums[1]:016x}"),
            "hash_sums": hash_sums,
            "numeric_zero_counts": numeric_zero_counts,
            "categorical_zero_counts": categorical_zero_counts,
        }
    )
    return result


def correctness_reasons(summary: Mapping[str, Any], rows: int) -> list[str]:
    reasons = []
    if int(summary["rows"]) != rows:
        reasons.append(f"output rows {summary['rows']} != requested rows {rows}")
    if int(summary["null_count"]) != 0:
        reasons.append(f"final output contains {summary['null_count']} nulls")
    if int(summary["nan_count"]) != 0:
        reasons.append(f"final output contains {summary['nan_count']} NaNs")
    if int(summary["categorical_range_violations"]) != 0:
        reasons.append(
            "final categorical output contains "
            f"{summary['categorical_range_violations']} out-of-range values"
        )
    if list(summary["numeric_zero_counts"]) != expected_numeric_zero_counts(rows):
        reasons.append(
            "numeric null/NaN-to-zero counts do not match the source contract"
        )
    if list(summary["categorical_zero_counts"]) != expected_categorical_zero_counts(
        rows
    ):
        reasons.append(
            "categorical null/OOV-to-zero counts do not match source contract"
        )
    return reasons


_RUNTIME_METRIC_KEYS = (
    "num_tasks_submitted",
    "num_tasks_finished",
    "num_tasks_failed",
    "bytes_inputs_received",
    "bytes_inputs_of_submitted_tasks",
    "bytes_outputs_of_finished_tasks",
    "obj_store_mem_spilled",
    "obj_store_mem_freed",
    "obj_store_mem_used",
)


def collect_ray_data_metrics(stats: Any) -> dict[str, Any]:
    """Extract stable task/object-store counters from a DatasetStats tree."""

    operators = []

    def visit(node: Any) -> None:
        for parent in getattr(node, "parents", ()) or ():
            visit(parent)
        extra = getattr(node, "extra_metrics", {}) or {}
        if extra:
            operators.append(
                {
                    "name": getattr(node, "base_name", ""),
                    **{key: extra.get(key, 0) for key in _RUNTIME_METRIC_KEYS},
                }
            )

    visit(stats)
    totals = {
        key: sum(int(operator.get(key, 0) or 0) for operator in operators)
        for key in _RUNTIME_METRIC_KEYS
    }
    totals.update(
        {
            "global_bytes_spilled": int(getattr(stats, "global_bytes_spilled", 0) or 0),
            "global_bytes_restored": int(
                getattr(stats, "global_bytes_restored", 0) or 0
            ),
            "dataset_bytes_spilled": int(
                getattr(stats, "dataset_bytes_spilled", 0) or 0
            ),
        }
    )
    return {"operators": operators, "totals": totals}


def batch_probe_reasons(snapshot: Mapping[str, Any], rows: int, mode: str) -> list[str]:
    reasons = []
    stages = snapshot.get("stages", {})
    missing = [stage for stage in STAGE_NAMES if stage not in stages]
    if missing:
        return [f"missing batch-probe events for stages {missing!r}"]

    batch_counts = []
    for stage in STAGE_NAMES:
        metrics = stages[stage]
        if int(metrics["rows"]) != rows:
            reasons.append(f"{stage} observed {metrics['rows']} rows instead of {rows}")
        batch_counts.append(int(metrics["batches"]))
    if len(set(batch_counts)) != 1:
        reasons.append(
            f"stage batch totals differ: {dict(zip(STAGE_NAMES, batch_counts))}"
        )

    if mode == "fused" and len(snapshot.get("gpu_slots", ())) < 7:
        reasons.append(
            "fused run used fewer than seven GPUs: "
            f"{len(snapshot.get('gpu_slots', ()))}"
        )
    return reasons


def atomic_write_json(path: os.PathLike[str] | str, payload: Mapping[str, Any]) -> None:
    """Durably replace ``path`` with one compact, sorted JSON document."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
