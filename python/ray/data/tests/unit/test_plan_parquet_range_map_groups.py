from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pyarrow as pa
import pytest

from ray.data._internal.datasource.parquet_range import (
    ParquetColumnSize,
    ParquetRangePlanningResult,
    ParquetRangeSuitability,
    ParquetRowGroupMetadata,
    plan_parquet_row_groups,
)
from ray.data._internal.datasource.parquet_range_v1 import ParquetV1MetadataResult
from ray.data._internal.logical.operators import InputData, MapGroups
from ray.data._internal.planner.parquet_range_map_groups_selector import (
    ParquetRangeMapGroupsCandidate,
    ParquetRangeMapGroupsSelectionResult,
)
from ray.data._internal.planner.plan_parquet_range_map_groups import (
    ParquetRangeMapGroupsPlanningResult,
    try_plan_parquet_range_map_groups,
)
from ray.data.context import DataContext, ShuffleStrategy


def _row_group(row_group_id, key_min, key_max):
    column_sizes = (
        ParquetColumnSize("User", 10, 20),
        ParquetColumnSize("Card", 10, 20),
        ParquetColumnSize("payload", 10, 20),
    )
    return ParquetRowGroupMetadata(
        path="/tmp/input.parquet",
        source_identity="posix-v1:1:2:3:4",
        row_group_id=row_group_id,
        num_rows=10,
        key_min=key_min,
        key_max=key_max,
        null_count=0,
        encoded_bytes=30,
        uncompressed_bytes=60,
        column_sizes=column_sizes,
    )


def _schema():
    return pa.schema(
        [
            pa.field("User", pa.int64()),
            pa.field("Card", pa.int32()),
            pa.field("payload", pa.int64()),
        ]
    )


def _happy_state(monkeypatch):
    import ray.data._internal.planner.plan_parquet_range_map_groups as planner

    datasource = SimpleNamespace(_file_schema=_schema())
    read_op = SimpleNamespace(
        ray_remote_args={"label_selector": {"ray.io/node-id": "node-1"}}
    )
    tokenizer_op = SimpleNamespace(name="tokenizer")
    map_groups_op = SimpleNamespace(name="map_groups", num_partitions=2)
    candidate = ParquetRangeMapGroupsCandidate(
        read_op=read_op,
        datasource=datasource,
        tokenizer_op=tokenizer_op,
        map_groups_op=map_groups_op,
        partition_key="User",
        group_keys=("User", "Card"),
        num_partitions=2,
        worker_concurrency=2,
        projection=("User", "Card", "payload"),
        compute=SimpleNamespace(),
        ray_remote_args={
            "num_cpus": 1,
            "num_gpus": 1,
            "label_selector": {"accelerator": "v100"},
        },
    )
    selection = ParquetRangeMapGroupsSelectionResult(candidate, None)
    metadata = ParquetV1MetadataResult(
        row_groups=(_row_group(0, 0, 9), _row_group(1, 10, 19)),
        projection=candidate.projection,
        elapsed_s=0.25,
        rejection_reason=None,
        rejection_message=None,
        exception_type=None,
    )
    layout = plan_parquet_row_groups(
        metadata.row_groups,
        num_partitions=2,
        projection=candidate.projection,
    )
    range_result = ParquetRangePlanningResult(
        plan=layout,
        rejection_reason=None,
        rejection_message=None,
        exception_type=None,
    )
    physical_operator = object()

    selector = Mock(return_value=selection)
    metadata_adapter = Mock(return_value=metadata)
    range_planner = Mock(return_value=range_result)
    schema_validator = Mock(return_value=None)
    gpu_capacity = Mock(return_value=100_000)
    builder = Mock(return_value=physical_operator)
    monkeypatch.setattr(planner, "select_parquet_range_map_groups_candidate", selector)
    monkeypatch.setattr(
        planner, "try_extract_v1_parquet_row_group_metadata", metadata_adapter
    )
    monkeypatch.setattr(planner, "try_plan_parquet_row_groups", range_planner)
    monkeypatch.setattr(planner, "validate_cudf_projection_schema", schema_validator)
    monkeypatch.setattr(planner, "_detect_local_gpu_memory_bytes", gpu_capacity)
    monkeypatch.setattr(planner, "build_parquet_range_map_groups_operator", builder)

    context = DataContext()
    return SimpleNamespace(
        planner=planner,
        candidate=candidate,
        selection=selection,
        metadata=metadata,
        layout=layout,
        range_result=range_result,
        physical_operator=physical_operator,
        selector=selector,
        metadata_adapter=metadata_adapter,
        range_planner=range_planner,
        schema_validator=schema_validator,
        gpu_capacity=gpu_capacity,
        builder=builder,
        context=context,
    )


def test_selected_plan_builds_once_and_merges_local_label_selector(monkeypatch):
    state = _happy_state(monkeypatch)

    result = try_plan_parquet_range_map_groups(object(), state.context)

    assert result.selected
    assert result.physical_operator is state.physical_operator
    assert result.fallback_reason is None
    assert result.footer_planning_time_s >= state.metadata.elapsed_s
    assert result.estimated_peak_gpu_memory_bytes > 0
    assert result.gpu_memory_budget_bytes == 70_000
    state.metadata_adapter.assert_called_once_with(
        state.candidate.datasource,
        "User",
        ("User", "Card", "payload"),
    )
    state.range_planner.assert_called_once()
    state.schema_validator.assert_called_once_with(
        state.candidate.datasource._file_schema,
        state.candidate.projection,
    )
    state.builder.assert_called_once()
    builder_args = state.builder.call_args.kwargs
    assert builder_args["layout"] is state.layout
    assert builder_args["worker_concurrency"] == 2
    assert builder_args["ray_remote_args"] == {
        "num_cpus": 1,
        "num_gpus": 1,
        "label_selector": {
            "accelerator": "v100",
            "ray.io/node-id": "node-1",
        },
    }


def test_read_scheduling_strategy_is_preserved_by_fused_actor(monkeypatch):
    state = _happy_state(monkeypatch)
    candidate = replace(
        state.candidate,
        read_op=SimpleNamespace(ray_remote_args={"scheduling_strategy": "SPREAD"}),
        ray_remote_args={"num_cpus": 1, "num_gpus": 1},
    )
    state.selector.return_value = ParquetRangeMapGroupsSelectionResult(candidate, None)

    result = try_plan_parquet_range_map_groups(object(), state.context)

    assert result.selected
    assert state.builder.call_args.kwargs["ray_remote_args"] == {
        "num_cpus": 1,
        "num_gpus": 1,
        "scheduling_strategy": "SPREAD",
    }


@pytest.mark.parametrize(
    ("read_args", "actor_args", "reason"),
    [
        (
            {"scheduling_strategy": "SPREAD"},
            {
                "num_cpus": 1,
                "num_gpus": 1,
                "scheduling_strategy": "DEFAULT",
            },
            "actor_scheduling_strategy_conflicts_with_read",
        ),
        (
            {"label_selector": {"ray.io/node-id": "node-1"}},
            {
                "num_cpus": 1,
                "num_gpus": 1,
                "scheduling_strategy": "SPREAD",
            },
            "actor_label_selector_conflicts_with_scheduling_strategy",
        ),
    ],
)
def test_incompatible_actor_scheduling_falls_back_before_build(
    monkeypatch, read_args, actor_args, reason
):
    state = _happy_state(monkeypatch)
    candidate = replace(
        state.candidate,
        read_op=SimpleNamespace(ray_remote_args=read_args),
        ray_remote_args=actor_args,
    )
    state.selector.return_value = ParquetRangeMapGroupsSelectionResult(candidate, None)

    result = try_plan_parquet_range_map_groups(object(), state.context)

    assert result.fallback_reason == reason
    state.builder.assert_not_called()


def test_structural_and_metadata_rejections_never_call_builder(monkeypatch):
    state = _happy_state(monkeypatch)
    state.selector.return_value = ParquetRangeMapGroupsSelectionResult(
        candidate=None,
        fallback_reason="tokenizer_batch_format_not_cudf",
    )

    result = try_plan_parquet_range_map_groups(object(), state.context)

    assert result.fallback_reason == "tokenizer_batch_format_not_cudf"
    state.metadata_adapter.assert_not_called()
    state.builder.assert_not_called()

    state = _happy_state(monkeypatch)
    state.metadata_adapter.return_value = replace(
        state.metadata,
        row_groups=(),
        rejection_reason="missing_statistics",
        rejection_message="missing",
    )

    result = try_plan_parquet_range_map_groups(object(), state.context)

    assert result.fallback_reason == "missing_statistics"
    assert result.footer_planning_time_s == 0.25
    state.range_planner.assert_not_called()
    state.builder.assert_not_called()


def test_row_group_planning_and_suitability_rejections_skip_builder(monkeypatch):
    state = _happy_state(monkeypatch)
    state.range_planner.return_value = ParquetRangePlanningResult(
        plan=None,
        rejection_reason="missing_statistics",
        rejection_message="missing",
        exception_type="ParquetRangePlanningError",
    )

    result = try_plan_parquet_range_map_groups(object(), state.context)

    assert result.fallback_reason == "missing_statistics"
    state.builder.assert_not_called()

    state = _happy_state(monkeypatch)
    unsuitable = replace(
        state.layout,
        suitability=ParquetRangeSuitability(
            suitable=False,
            reason_codes=("scan_amplification_exceeded",),
        ),
    )
    state.range_planner.return_value = replace(state.range_result, plan=unsuitable)

    result = try_plan_parquet_range_map_groups(object(), state.context)

    assert result.fallback_reason == "scan_amplification_exceeded"
    state.builder.assert_not_called()


def test_partition_count_and_projection_metadata_are_exact(monkeypatch):
    state = _happy_state(monkeypatch)
    wrong_count = replace(state.layout, partitions=state.layout.partitions[:1])
    state.range_planner.return_value = replace(state.range_result, plan=wrong_count)

    result = try_plan_parquet_range_map_groups(object(), state.context)

    assert result.fallback_reason == "range_partition_count_mismatch"
    state.schema_validator.assert_not_called()
    state.builder.assert_not_called()

    state = _happy_state(monkeypatch)
    incomplete = replace(
        state.layout,
        metrics=replace(state.layout.metrics, projection_size_complete=False),
    )
    state.range_planner.return_value = replace(state.range_result, plan=incomplete)

    result = try_plan_parquet_range_map_groups(object(), state.context)

    assert result.fallback_reason == "projected_size_metadata_incomplete"
    state.schema_validator.assert_not_called()
    state.builder.assert_not_called()


def test_schema_and_memory_failures_skip_builder(monkeypatch):
    state = _happy_state(monkeypatch)
    state.schema_validator.return_value = "unsupported_projection_type"

    result = try_plan_parquet_range_map_groups(object(), state.context)

    assert result.fallback_reason == "unsupported_projection_type"
    state.builder.assert_not_called()

    state = _happy_state(monkeypatch)
    state.gpu_capacity.return_value = None

    result = try_plan_parquet_range_map_groups(object(), state.context)

    assert result.fallback_reason == "gpu_memory_capacity_unavailable"
    state.range_planner.assert_not_called()
    state.builder.assert_not_called()

    state = _happy_state(monkeypatch)
    state.context.set_config(
        "parquet_range_map_groups_estimated_output_bytes_per_row", 100_000
    )

    result = try_plan_parquet_range_map_groups(object(), state.context)

    assert result.fallback_reason == "estimated_gpu_memory_exceeded"
    assert result.estimated_peak_gpu_memory_bytes > result.gpu_memory_budget_bytes
    state.builder.assert_not_called()


@pytest.mark.parametrize(
    ("key", "value", "reason"),
    [
        (
            "parquet_range_map_groups_max_scan_amplification",
            0,
            "invalid_parquet_range_map_groups_max_scan_amplification",
        ),
        (
            "parquet_range_map_groups_max_load_skew",
            True,
            "invalid_parquet_range_map_groups_max_load_skew",
        ),
        (
            "parquet_range_map_groups_max_partition_encoded_bytes",
            -1,
            "invalid_parquet_range_map_groups_max_partition_encoded_bytes",
        ),
        (
            "parquet_range_map_groups_gpu_memory_bytes",
            0,
            "invalid_parquet_range_map_groups_gpu_memory_bytes",
        ),
        (
            "parquet_range_map_groups_gpu_memory_fraction",
            1.01,
            "invalid_parquet_range_map_groups_gpu_memory_fraction",
        ),
        (
            "parquet_range_map_groups_estimated_output_bytes_per_row",
            0,
            "invalid_parquet_range_map_groups_estimated_output_bytes_per_row",
        ),
        (
            "parquet_range_map_groups_peak_memory_multiplier",
            float("inf"),
            "invalid_parquet_range_map_groups_peak_memory_multiplier",
        ),
    ],
)
def test_invalid_planning_configs_fail_before_builder(monkeypatch, key, value, reason):
    state = _happy_state(monkeypatch)
    state.context.set_config(key, value)

    result = try_plan_parquet_range_map_groups(object(), state.context)

    assert result.fallback_reason == reason
    state.builder.assert_not_called()


def test_builder_failure_is_pre_execution_fallback(monkeypatch):
    state = _happy_state(monkeypatch)
    state.builder.side_effect = ValueError("invalid physical configuration")

    result = try_plan_parquet_range_map_groups(object(), state.context)

    assert result.fallback_reason == "physical_plan_construction_failed"
    state.builder.assert_called_once()


class _NamedPhysical:
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name


def _map_groups_op():
    def group_fn(batch):
        return batch

    return MapGroups(
        key=["User", "Card"],
        fn=group_fn,
        num_partitions=2,
        shuffle_strategy=ShuffleStrategy.HASH_SHUFFLE,
        input_dependencies=[InputData([])],
    )


def _planning_result(*, physical_operator=None, fallback_reason=None):
    return ParquetRangeMapGroupsPlanningResult(
        physical_operator=physical_operator,
        fallback_reason=fallback_reason,
        structural_selection=ParquetRangeMapGroupsSelectionResult(
            None, fallback_reason
        ),
    )


def test_map_groups_seam_returns_special_plan_before_fallback(monkeypatch):
    import ray.data._internal.planner.plan_map_groups_op as seam

    special = _NamedPhysical("ParquetRangeMapGroups")
    special_planner = Mock(return_value=_planning_result(physical_operator=special))
    fallback_exchange = Mock(side_effect=AssertionError("fallback exchange called"))
    fallback_map = Mock(side_effect=AssertionError("fallback map called"))
    monkeypatch.setattr(seam, "try_plan_parquet_range_map_groups", special_planner)
    monkeypatch.setattr(seam, "plan_all_to_all_op", fallback_exchange)
    monkeypatch.setattr(seam, "plan_udf_map_op", fallback_map)

    result = seam.plan_map_groups_op(_map_groups_op(), [object()], DataContext())

    assert result is special
    special_planner.assert_called_once()
    fallback_exchange.assert_not_called()
    fallback_map.assert_not_called()


def test_map_groups_flag_off_retains_unannotated_fallback_name(monkeypatch):
    import ray.data._internal.planner.plan_map_groups_op as seam

    exchange = object()
    fallback = _NamedPhysical("MapBatches(group_fn)")
    monkeypatch.setattr(seam, "plan_all_to_all_op", Mock(return_value=exchange))
    monkeypatch.setattr(seam, "plan_udf_map_op", Mock(return_value=fallback))

    result = seam.plan_map_groups_op(_map_groups_op(), [object()], DataContext())

    assert result is fallback
    assert result.name == "MapBatches(group_fn)"


def test_map_groups_fallback_reason_is_annotated(monkeypatch):
    import ray.data._internal.planner.plan_map_groups_op as seam

    exchange = object()
    fallback = _NamedPhysical("MapBatches(group_fn)")
    monkeypatch.setattr(
        seam,
        "try_plan_parquet_range_map_groups",
        Mock(return_value=_planning_result(fallback_reason="missing_statistics")),
    )
    monkeypatch.setattr(seam, "plan_all_to_all_op", Mock(return_value=exchange))
    monkeypatch.setattr(seam, "plan_udf_map_op", Mock(return_value=fallback))

    result = seam.plan_map_groups_op(_map_groups_op(), [object()], DataContext())

    assert result is fallback
    assert result.name == (
        "MapBatches(group_fn) " "[ParquetRangeMapGroupsFallback=missing_statistics]"
    )
