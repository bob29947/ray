import pickle
from dataclasses import replace

import pytest

from ray.data._internal.datasource.parquet_range import (
    ParquetColumnSize,
    ParquetRangePlanningError,
    ParquetRangeSuitabilityPolicy,
    ParquetRowGroupMetadata,
    evaluate_parquet_range_suitability,
    plan_parquet_row_groups,
    try_plan_parquet_row_groups,
)


def _row_group(
    row_group_id,
    key_min,
    key_max,
    *,
    path="data.parquet",
    source_identity="size=1000;mtime_ns=7",
    num_rows=10,
    null_count=0,
    encoded_bytes=100,
    uncompressed_bytes=200,
    column_sizes=(),
):
    return ParquetRowGroupMetadata(
        path=path,
        source_identity=source_identity,
        row_group_id=row_group_id,
        num_rows=num_rows,
        key_min=key_min,
        key_max=key_max,
        null_count=null_count,
        encoded_bytes=encoded_bytes,
        uncompressed_bytes=uncompressed_bytes,
        column_sizes=column_sizes,
    )


def _bounds(plan):
    return [
        (partition.lower_bound, partition.upper_bound)
        for partition in plan.partitions
    ]


def _fragment_map(partition):
    return {
        fragment.path: (fragment.source_identity, fragment.row_group_ids)
        for fragment in partition.fragments
    }


def test_integral_ranges_are_inclusive_contiguous_and_domain_capped():
    plan = plan_parquet_row_groups(
        [_row_group(1, 10, 19), _row_group(0, 0, 9)],
        num_partitions=2,
    )

    assert _bounds(plan) == [(0, 9), (10, 19)]
    assert _fragment_map(plan.partitions[0]) == {
        "data.parquet": ("size=1000;mtime_ns=7", (0,))
    }
    assert _fragment_map(plan.partitions[1]) == {
        "data.parquet": ("size=1000;mtime_ns=7", (1,))
    }
    assert plan.row_groups[0].row_group_id == 0

    capped = plan_parquet_row_groups(
        [_row_group(0, -1, 1)], num_partitions=10
    )
    assert _bounds(capped) == [(-1, -1), (0, 0), (1, 1)]


def test_overlapping_row_groups_are_duplicated_and_require_exact_filtering():
    rows_by_row_group = {
        0: tuple(range(0, 20, 2)),
        1: tuple(range(1, 20, 2)),
    }
    sizes = (
        ParquetColumnSize("User", 20, 40),
        ParquetColumnSize("payload", 80, 160),
    )
    plan = plan_parquet_row_groups(
        [
            _row_group(0, 0, 18, column_sizes=sizes),
            _row_group(1, 1, 19, column_sizes=sizes),
        ],
        num_partitions=2,
        projection=["User"],
    )

    assert _bounds(plan) == [(0, 9), (10, 19)]
    assert [
        fragment.row_group_ids
        for partition in plan.partitions
        for fragment in partition.fragments
    ] == [(0, 1), (0, 1)]

    emitted = []
    for partition in plan.partitions:
        for row_group_id in partition.fragments[0].row_group_ids:
            emitted.extend(
                value
                for value in rows_by_row_group[row_group_id]
                if partition.lower_bound <= value <= partition.upper_bound
            )
    assert sorted(emitted) == list(range(20))
    assert len(emitted) == len(set(emitted))

    metrics = plan.metrics
    assert metrics.source_row_groups == 2
    assert metrics.scanned_row_groups == 4
    assert metrics.duplicate_row_groups == 2
    assert metrics.source_encoded_bytes == 40
    assert metrics.scanned_encoded_bytes == 80
    assert metrics.duplicate_encoded_bytes == 40
    assert metrics.scan_amplification == pytest.approx(2.0)
    assert metrics.row_scan_amplification == pytest.approx(2.0)
    assert metrics.uncompressed_scan_amplification == pytest.approx(2.0)
    assert metrics.projection_size_complete


def test_projection_sizes_and_conservative_full_row_group_fallback():
    column_sizes = (
        ParquetColumnSize("User", 10, 20),
        ParquetColumnSize("payload", 90, 180),
    )
    projected = plan_parquet_row_groups(
        [_row_group(0, 0, 9, column_sizes=column_sizes)],
        num_partitions=1,
        projection=["User"],
    )
    assert projected.projection == ("User",)
    assert projected.metrics.source_encoded_bytes == 10
    assert projected.metrics.source_uncompressed_bytes == 20
    assert projected.metrics.projection_size_complete

    fallback = plan_parquet_row_groups(
        [_row_group(0, 0, 9)],
        num_partitions=1,
        projection=["User"],
    )
    assert fallback.metrics.source_encoded_bytes == 100
    assert fallback.metrics.source_uncompressed_bytes == 200
    assert not fallback.metrics.projection_size_complete

    with pytest.raises(ParquetRangePlanningError) as exc_info:
        plan_parquet_row_groups(
            [_row_group(0, 0, 9, column_sizes=column_sizes)],
            num_partitions=1,
            projection=["missing"],
        )
    assert exc_info.value.reason_code == "missing_projection_column"


def test_overlap_metrics_and_sparse_load_skew_include_empty_ranges():
    plan = plan_parquet_row_groups(
        [
            _row_group(0, 0, 0),
            _row_group(1, 99, 99),
        ],
        num_partitions=4,
    )

    assert _bounds(plan) == [(0, 24), (25, 49), (50, 74), (75, 99)]
    assert plan.partitions[1].fragments == ()
    assert plan.partitions[2].fragments == ()
    assert plan.metrics.scan_amplification == pytest.approx(1.0)
    assert plan.metrics.encoded_load_skew == pytest.approx(2.0)
    assert plan.metrics.uncompressed_load_skew == pytest.approx(2.0)
    assert plan.metrics.load_skew == pytest.approx(2.0)
    assert plan.suitability.reason_codes == ("load_skew_exceeded",)


@pytest.mark.parametrize(
    ("row_group", "reason_code"),
    [
        (_row_group(0, None, 1), "missing_statistics"),
        (_row_group(0, 0, None), "missing_statistics"),
        (_row_group(0, 0.0, 1.0), "unsupported_key_type"),
        (_row_group(0, False, 1), "unsupported_key_type"),
        (_row_group(0, 2, 1), "invalid_statistics"),
        (_row_group(0, 0, 1, null_count=None), "unknown_null_count"),
        (_row_group(0, 0, 1, null_count=1), "null_keys"),
        (_row_group(0, 0, 1, source_identity=""), "missing_source_identity"),
        (_row_group(0, 0, 1, encoded_bytes=0), "missing_size_metadata"),
    ],
)
def test_unsafe_metadata_has_deterministic_reason_code(row_group, reason_code):
    with pytest.raises(ParquetRangePlanningError) as exc_info:
        plan_parquet_row_groups([row_group], num_partitions=1)
    assert exc_info.value.reason_code == reason_code


def test_duplicate_row_groups_and_inconsistent_source_identity_fail_closed():
    row_group = _row_group(0, 0, 1)
    with pytest.raises(ParquetRangePlanningError) as exc_info:
        plan_parquet_row_groups([row_group, row_group], num_partitions=1)
    assert exc_info.value.reason_code == "duplicate_row_group"

    with pytest.raises(ParquetRangePlanningError) as exc_info:
        plan_parquet_row_groups(
            [
                row_group,
                _row_group(1, 2, 3, source_identity="changed"),
            ],
            num_partitions=1,
        )
    assert exc_info.value.reason_code == "source_identity_mismatch"


@pytest.mark.parametrize("num_partitions", [0, -1, True, 1.0, "1"])
def test_invalid_partition_count_fails_closed(num_partitions):
    result = try_plan_parquet_row_groups(
        [_row_group(0, 0, 1)], num_partitions=num_partitions
    )
    assert not result.planned
    assert result.rejection_reason == "invalid_partition_count"
    assert result.fallback_reason_codes == ("invalid_partition_count",)


def test_suitability_reason_codes_are_stable_and_thresholds_are_inclusive():
    metrics = plan_parquet_row_groups(
        [_row_group(0, 0, 9)], num_partitions=1
    ).metrics
    policy = ParquetRangeSuitabilityPolicy(
        max_scan_amplification=1.25,
        max_load_skew=1.5,
        max_partition_encoded_bytes=100,
        max_partition_uncompressed_bytes=200,
    )
    boundary = replace(
        metrics,
        scan_amplification=1.25,
        row_scan_amplification=1.25,
        uncompressed_scan_amplification=1.25,
        load_skew=1.5,
        largest_partition_encoded_bytes=100,
        largest_partition_uncompressed_bytes=200,
    )
    assert evaluate_parquet_range_suitability(boundary, policy).suitable

    rejected = replace(
        boundary,
        scan_amplification=1.250001,
        row_scan_amplification=1.250001,
        uncompressed_scan_amplification=1.250001,
        load_skew=1.500001,
        largest_partition_encoded_bytes=101,
        largest_partition_uncompressed_bytes=201,
    )
    assert evaluate_parquet_range_suitability(
        rejected, policy
    ).reason_codes == (
        "encoded_scan_amplification_exceeded",
        "row_scan_amplification_exceeded",
        "uncompressed_scan_amplification_exceeded",
        "load_skew_exceeded",
        "partition_encoded_bytes_exceeded",
        "partition_uncompressed_bytes_exceeded",
    )


def test_plan_and_failure_result_are_pickle_serializable():
    plan = plan_parquet_row_groups(
        [
            _row_group(0, 0, 9, path="b.parquet", source_identity="b-v1"),
            _row_group(0, 10, 19, path="a.parquet", source_identity="a-v1"),
        ],
        num_partitions=2,
    )
    assert pickle.loads(pickle.dumps(plan)) == plan

    result = try_plan_parquet_row_groups(
        [_row_group(0, None, 9)], num_partitions=2
    )
    restored = pickle.loads(pickle.dumps(result))
    assert restored == result
    assert restored.rejection_reason == "missing_statistics"
    assert restored.exception_type == "ParquetRangePlanningError"


def test_empty_dataset_and_invalid_projection_are_rejected():
    with pytest.raises(ParquetRangePlanningError) as exc_info:
        plan_parquet_row_groups([], num_partitions=1)
    assert exc_info.value.reason_code == "empty_dataset"

    for projection in ([], "User", ["User", "User"], [""]):
        with pytest.raises(ParquetRangePlanningError) as exc_info:
            plan_parquet_row_groups(
                [_row_group(0, 0, 1)],
                num_partitions=1,
                projection=projection,
            )
        assert exc_info.value.reason_code == "invalid_projection"
