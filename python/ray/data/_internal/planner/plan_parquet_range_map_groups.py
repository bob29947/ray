"""Fail-closed planning for the experimental Parquet range ``map_groups`` path."""

from __future__ import annotations

import math
import numbers
import time
from dataclasses import dataclass
from typing import Optional

from ray.data._internal.datasource.parquet_range import (
    ParquetRangeSuitabilityPolicy,
    try_plan_parquet_row_groups,
)
from ray.data._internal.datasource.parquet_range_v1 import (
    try_extract_v1_parquet_row_group_metadata,
)
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.logical.interfaces import LogicalOperator
from ray.data._internal.planner.parquet_range_map_groups import (
    build_parquet_range_map_groups_operator,
    validate_cudf_projection_schema,
)
from ray.data._internal.planner.parquet_range_map_groups_selector import (
    ParquetRangeMapGroupsSelectionResult,
    select_parquet_range_map_groups_candidate,
)
from ray.data.context import DataContext


@dataclass(frozen=True)
class ParquetRangeMapGroupsPlanningResult:
    """Physical selection result used by the normal ``MapGroups`` lowering."""

    physical_operator: Optional[PhysicalOperator]
    fallback_reason: Optional[str]
    structural_selection: ParquetRangeMapGroupsSelectionResult
    footer_planning_time_s: float = 0.0
    estimated_peak_gpu_memory_bytes: Optional[int] = None
    gpu_memory_budget_bytes: Optional[int] = None

    @property
    def selected(self) -> bool:
        return self.physical_operator is not None


def _fallback(
    reason: str,
    selection: ParquetRangeMapGroupsSelectionResult,
    *,
    footer_planning_time_s: float = 0.0,
    estimated_peak_gpu_memory_bytes: Optional[int] = None,
    gpu_memory_budget_bytes: Optional[int] = None,
) -> ParquetRangeMapGroupsPlanningResult:
    return ParquetRangeMapGroupsPlanningResult(
        physical_operator=None,
        fallback_reason=reason,
        structural_selection=selection,
        footer_planning_time_s=footer_planning_time_s,
        estimated_peak_gpu_memory_bytes=estimated_peak_gpu_memory_bytes,
        gpu_memory_budget_bytes=gpu_memory_budget_bytes,
    )


def _positive_real_config(
    context: DataContext, key: str, default: float
) -> tuple[Optional[float], Optional[str]]:
    value = context.get_config(key, default)
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        return None, f"invalid_{key}"
    return float(value), None


def _optional_positive_int_config(
    context: DataContext, key: str, default: Optional[int]
) -> tuple[Optional[int], Optional[str]]:
    value = context.get_config(key, default)
    if value is None:
        return None, None
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Integral)
        or int(value) <= 0
    ):
        return None, f"invalid_{key}"
    return int(value), None


def _detect_local_gpu_memory_bytes() -> Optional[int]:
    """Return the smallest local GPU capacity without scheduling GPU work."""

    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            count = pynvml.nvmlDeviceGetCount()
            capacities = [
                int(
                    pynvml.nvmlDeviceGetMemoryInfo(
                        pynvml.nvmlDeviceGetHandleByIndex(index)
                    ).total
                )
                for index in range(count)
            ]
        finally:
            pynvml.nvmlShutdown()
        if capacities:
            return min(capacities)
    except Exception:
        pass

    try:
        import cupy

        count = int(cupy.cuda.runtime.getDeviceCount())
        capacities = [
            int(cupy.cuda.runtime.getDeviceProperties(index)["totalGlobalMem"])
            for index in range(count)
        ]
        if capacities:
            return min(capacities)
    except Exception:
        pass
    return None


def _merged_actor_args(candidate) -> tuple[Optional[dict], Optional[str]]:
    actor_args = dict(candidate.ray_remote_args)
    read_args = candidate.read_op.ray_remote_args
    read_selector = dict(read_args.get("label_selector", {}))
    actor_selector = dict(actor_args.get("label_selector", {}))
    for key, value in read_selector.items():
        if key in actor_selector and actor_selector[key] != value:
            return None, "actor_label_selector_conflicts_with_read"
        actor_selector[key] = value
    read_strategy = read_args.get("scheduling_strategy")
    actor_strategy = actor_args.get("scheduling_strategy")
    if (
        read_strategy is not None
        and actor_strategy is not None
        and read_strategy != actor_strategy
    ):
        return None, "actor_scheduling_strategy_conflicts_with_read"
    if read_strategy is not None:
        actor_args["scheduling_strategy"] = read_strategy
    if actor_selector:
        if actor_args.get("scheduling_strategy") is not None:
            return None, "actor_label_selector_conflicts_with_scheduling_strategy"
        actor_args["label_selector"] = actor_selector
    return actor_args, None


def try_plan_parquet_range_map_groups(
    op: LogicalOperator,
    data_context: DataContext,
) -> ParquetRangeMapGroupsPlanningResult:
    """Plan the fused backend, or return one deterministic pre-execution reason."""

    selection = select_parquet_range_map_groups_candidate(op, data_context)
    if not selection.selected:
        return _fallback(
            selection.fallback_reason or "structural_selection_failed", selection
        )
    candidate = selection.candidate
    assert candidate is not None

    metadata = try_extract_v1_parquet_row_group_metadata(
        candidate.datasource,
        candidate.partition_key,
        candidate.projection,
    )
    if not metadata.extracted:
        return _fallback(
            metadata.rejection_reason or "parquet_metadata_extraction_failed",
            selection,
            footer_planning_time_s=metadata.elapsed_s,
        )

    max_amplification, reason = _positive_real_config(
        data_context,
        "parquet_range_map_groups_max_scan_amplification",
        1.25,
    )
    if reason:
        return _fallback(reason, selection, footer_planning_time_s=metadata.elapsed_s)
    max_skew, reason = _positive_real_config(
        data_context,
        "parquet_range_map_groups_max_load_skew",
        1.5,
    )
    if reason:
        return _fallback(reason, selection, footer_planning_time_s=metadata.elapsed_s)
    max_encoded_bytes, reason = _optional_positive_int_config(
        data_context,
        "parquet_range_map_groups_max_partition_encoded_bytes",
        None,
    )
    if reason:
        return _fallback(reason, selection, footer_planning_time_s=metadata.elapsed_s)

    configured_gpu_bytes, reason = _optional_positive_int_config(
        data_context,
        "parquet_range_map_groups_gpu_memory_bytes",
        None,
    )
    if reason:
        return _fallback(reason, selection, footer_planning_time_s=metadata.elapsed_s)
    gpu_memory_bytes = configured_gpu_bytes or _detect_local_gpu_memory_bytes()
    if gpu_memory_bytes is None:
        return _fallback(
            "gpu_memory_capacity_unavailable",
            selection,
            footer_planning_time_s=metadata.elapsed_s,
        )
    memory_fraction, reason = _positive_real_config(
        data_context,
        "parquet_range_map_groups_gpu_memory_fraction",
        0.70,
    )
    if reason or memory_fraction is None or memory_fraction > 1.0:
        return _fallback(
            reason or "invalid_parquet_range_map_groups_gpu_memory_fraction",
            selection,
            footer_planning_time_s=metadata.elapsed_s,
        )
    gpu_memory_budget = int(gpu_memory_bytes * memory_fraction)

    plan_started = time.perf_counter()
    range_result = try_plan_parquet_row_groups(
        metadata.row_groups,
        num_partitions=candidate.num_partitions,
        projection=candidate.projection,
        suitability_policy=ParquetRangeSuitabilityPolicy(
            max_scan_amplification=max_amplification,
            max_load_skew=max_skew,
            max_partition_encoded_bytes=max_encoded_bytes,
            max_partition_uncompressed_bytes=gpu_memory_budget,
        ),
    )
    footer_planning_time_s = metadata.elapsed_s + (time.perf_counter() - plan_started)
    if not range_result.planned:
        return _fallback(
            range_result.rejection_reason or "parquet_range_planning_failed",
            selection,
            footer_planning_time_s=footer_planning_time_s,
            gpu_memory_budget_bytes=gpu_memory_budget,
        )
    layout = range_result.plan
    assert layout is not None
    if not layout.suitability.suitable:
        return _fallback(
            layout.suitability.reason_codes[0],
            selection,
            footer_planning_time_s=footer_planning_time_s,
            gpu_memory_budget_bytes=gpu_memory_budget,
        )
    if len(layout.partitions) != candidate.num_partitions:
        return _fallback(
            "range_partition_count_mismatch",
            selection,
            footer_planning_time_s=footer_planning_time_s,
            gpu_memory_budget_bytes=gpu_memory_budget,
        )
    if not layout.metrics.projection_size_complete:
        return _fallback(
            "projected_size_metadata_incomplete",
            selection,
            footer_planning_time_s=footer_planning_time_s,
            gpu_memory_budget_bytes=gpu_memory_budget,
        )

    schema_reason = validate_cudf_projection_schema(
        candidate.datasource._file_schema, candidate.projection
    )
    if schema_reason:
        return _fallback(
            schema_reason,
            selection,
            footer_planning_time_s=footer_planning_time_s,
            gpu_memory_budget_bytes=gpu_memory_budget,
        )

    output_bytes_per_row, reason = _positive_real_config(
        data_context,
        "parquet_range_map_groups_estimated_output_bytes_per_row",
        64.0,
    )
    if reason:
        return _fallback(
            reason, selection, footer_planning_time_s=footer_planning_time_s
        )
    peak_multiplier, reason = _positive_real_config(
        data_context,
        "parquet_range_map_groups_peak_memory_multiplier",
        1.5,
    )
    if reason:
        return _fallback(
            reason, selection, footer_planning_time_s=footer_planning_time_s
        )
    assert output_bytes_per_row is not None and peak_multiplier is not None
    estimated_peak_gpu_memory = max(
        int(
            math.ceil(
                (
                    partition.estimated_uncompressed_bytes
                    + partition.estimated_scanned_rows * output_bytes_per_row
                )
                * peak_multiplier
            )
        )
        for partition in layout.partitions
    )
    if estimated_peak_gpu_memory > gpu_memory_budget:
        return _fallback(
            "estimated_gpu_memory_exceeded",
            selection,
            footer_planning_time_s=footer_planning_time_s,
            estimated_peak_gpu_memory_bytes=estimated_peak_gpu_memory,
            gpu_memory_budget_bytes=gpu_memory_budget,
        )

    actor_args, reason = _merged_actor_args(candidate)
    if reason:
        return _fallback(
            reason,
            selection,
            footer_planning_time_s=footer_planning_time_s,
            estimated_peak_gpu_memory_bytes=estimated_peak_gpu_memory,
            gpu_memory_budget_bytes=gpu_memory_budget,
        )
    assert actor_args is not None
    try:
        physical_operator = build_parquet_range_map_groups_operator(
            layout=layout,
            footer_planning_time_s=footer_planning_time_s,
            projection=candidate.projection,
            partition_key=candidate.partition_key,
            group_keys=candidate.group_keys,
            source_schema=candidate.datasource._file_schema,
            tokenizer_op=candidate.tokenizer_op,
            map_groups_op=candidate.map_groups_op,
            partition_contract=candidate.partition_contract,
            partition_contract_fallback_reason=(
                candidate.partition_contract_fallback_reason
            ),
            data_context=data_context,
            ray_remote_args=actor_args,
            estimated_peak_gpu_memory_bytes=estimated_peak_gpu_memory,
            gpu_memory_budget_bytes=gpu_memory_budget,
        )
    except Exception:
        # This occurs before worker construction or either UDF invocation, so
        # retaining ordinary execution is side-effect safe.
        return _fallback(
            "physical_plan_construction_failed",
            selection,
            footer_planning_time_s=footer_planning_time_s,
            estimated_peak_gpu_memory_bytes=estimated_peak_gpu_memory,
            gpu_memory_budget_bytes=gpu_memory_budget,
        )
    return ParquetRangeMapGroupsPlanningResult(
        physical_operator=physical_operator,
        fallback_reason=None,
        structural_selection=selection,
        footer_planning_time_s=footer_planning_time_s,
        estimated_peak_gpu_memory_bytes=estimated_peak_gpu_memory,
        gpu_memory_budget_bytes=gpu_memory_budget,
    )


__all__ = [
    "ParquetRangeMapGroupsPlanningResult",
    "try_plan_parquet_range_map_groups",
]
