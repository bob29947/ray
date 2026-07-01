"""Experimental protocol for partition-equivalent ``map_groups`` callables.

The ordinary per-group callable remains the compatibility implementation.  A
side-effect-free callable may additionally declare a synchronous implementation
that consumes one already-grouped partition.  Older Ray versions ignore these
attributes and continue invoking the ordinary callable for every group.  Ray
validates the protocol shape and normal batch outputs, while the callable's
``equivalent_to_per_group`` declaration explicitly owns value and group-coverage
equivalence; proving arbitrary Python programs equivalent is not possible.
The normal planner memory gate remains in force, but Ray cannot infer arbitrary
temporary allocations inside the alternate implementation; runtime OOMs and
other failures propagate without retrying the ordinary callable.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional, Tuple

from ray.data.context import DataContext


MAP_GROUPS_PARTITION_EXECUTION_ENABLED_CONFIG = "map_groups_partition_execution_enabled"
MAP_GROUPS_PARTITION_PROTOCOL_ATTRIBUTE = "__ray_data_map_groups_partition_protocol__"
MAP_GROUPS_PARTITION_UDF_ATTRIBUTE = "__ray_data_map_groups_partition__"
MAP_GROUPS_PARTITION_PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class MapGroupsPartitionContext:
    """Immutable group layout passed to a partition-equivalent callable."""

    group_keys: Tuple[str, ...]
    input_group_boundaries: Tuple[int, ...]

    @property
    def num_groups(self) -> int:
        return max(0, len(self.input_group_boundaries) - 1)


@dataclass(frozen=True)
class MapGroupsPartitionContract:
    """Validated, process-serializable v1 protocol description."""

    udf: Callable[..., Any]
    batch_format: str


def resolve_map_groups_partition_contract(
    fn: Any,
    data_context: DataContext,
    *,
    batch_format: Optional[str],
    zero_copy_batch: bool = True,
    fn_args: Optional[Iterable[Any]] = None,
    fn_kwargs: Optional[Mapping[str, Any]] = None,
) -> tuple[Optional[MapGroupsPartitionContract], Optional[str]]:
    """Resolve a strict v1 contract without invoking application code."""

    if (
        data_context.get_config(MAP_GROUPS_PARTITION_EXECUTION_ENABLED_CONFIG, False)
        is not True
    ):
        return None, None

    descriptor = getattr(fn, MAP_GROUPS_PARTITION_PROTOCOL_ATTRIBUTE, None)
    partition_udf = getattr(fn, MAP_GROUPS_PARTITION_UDF_ATTRIBUTE, None)
    if descriptor is None and partition_udf is None:
        return None, None
    if not inspect.isfunction(fn):
        return None, "group_partition_callable_unsupported"
    if type(descriptor) is not dict or set(descriptor) != {
        "version",
        "batch_format",
        "side_effect_free",
        "equivalent_to_per_group",
    }:
        return None, "group_partition_protocol_invalid"
    if descriptor["version"] != MAP_GROUPS_PARTITION_PROTOCOL_VERSION:
        return None, "group_partition_protocol_version_unsupported"
    if descriptor["side_effect_free"] is not True:
        return None, "group_partition_side_effect_contract_missing"
    if descriptor["equivalent_to_per_group"] is not True:
        return None, "group_partition_equivalence_contract_missing"
    if (
        not isinstance(descriptor["batch_format"], str)
        or not descriptor["batch_format"]
        or descriptor["batch_format"] != batch_format
    ):
        return None, "group_partition_batch_format_mismatch"
    if zero_copy_batch is not True:
        return None, "group_partition_zero_copy_required"
    if not inspect.isfunction(partition_udf) or (
        inspect.iscoroutinefunction(partition_udf)
        or inspect.isasyncgenfunction(partition_udf)
    ):
        return None, "group_partition_udf_invalid"
    try:
        inspect.signature(partition_udf).bind(
            object(),
            object(),
            *(tuple(fn_args or ())),
            **dict(fn_kwargs or {}),
        )
    except (TypeError, ValueError):
        return None, "group_partition_udf_signature_incompatible"
    return (
        MapGroupsPartitionContract(
            udf=partition_udf,
            batch_format=descriptor["batch_format"],
        ),
        None,
    )


__all__ = [
    "MAP_GROUPS_PARTITION_EXECUTION_ENABLED_CONFIG",
    "MAP_GROUPS_PARTITION_PROTOCOL_ATTRIBUTE",
    "MAP_GROUPS_PARTITION_PROTOCOL_VERSION",
    "MAP_GROUPS_PARTITION_UDF_ATTRIBUTE",
    "MapGroupsPartitionContext",
    "MapGroupsPartitionContract",
    "resolve_map_groups_partition_contract",
]
