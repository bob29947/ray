"""Fail-closed structural selection for Parquet range ``map_groups``.

This module only inspects an optimized logical plan.  It performs no footer or
filesystem I/O and never constructs the fused physical backend.  Every rejected
shape returns a stable reason code so the ordinary shuffle implementation remains
the default for anything outside the deliberately narrow first implementation.
"""

from __future__ import annotations

import inspect
import numbers
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ray.data._internal.compute import ActorPoolStrategy, TaskPoolStrategy
from ray.data._internal.datasource.parquet_datasource import ParquetDatasource
from ray.data._internal.logical.interfaces import LogicalOperator
from ray.data._internal.logical.operators import MapBatches, MapGroups, Read
from ray.data._internal.planner.map_groups_partition_protocol import (
    resolve_map_groups_partition_contract,
)
from ray.data.block import CallableClass
from ray.data.context import DataContext, ShuffleStrategy

PARQUET_RANGE_MAP_GROUPS_ENABLED_CONFIG = "parquet_range_map_groups_enabled"
PARTITION_PRESERVATION_ATTRIBUTE = "__ray_data_preserves_partitioning__"
_REPLAY_REMOTE_ARGS = {
    "max_restarts",
    "max_task_retries",
    "max_retries",
    "retry_exceptions",
}


@dataclass(frozen=True)
class ParquetRangeMapGroupsCandidate:
    """Validated inputs needed by the metadata planner and physical builder."""

    read_op: Read
    datasource: ParquetDatasource
    tokenizer_op: MapBatches
    map_groups_op: MapGroups
    partition_key: str
    group_keys: Tuple[str, ...]
    num_partitions: int
    worker_concurrency: int
    projection: Tuple[str, ...]
    compute: ActorPoolStrategy
    ray_remote_args: Dict[str, Any]
    partition_contract: Optional[Any] = None
    partition_contract_fallback_reason: Optional[str] = None


@dataclass(frozen=True)
class ParquetRangeMapGroupsSelectionResult:
    """Structural selection result; a rejection always retains normal execution."""

    candidate: Optional[ParquetRangeMapGroupsCandidate]
    fallback_reason: Optional[str]

    @property
    def selected(self) -> bool:
        return self.candidate is not None


def _fallback(reason: str) -> ParquetRangeMapGroupsSelectionResult:
    return ParquetRangeMapGroupsSelectionResult(
        candidate=None,
        fallback_reason=reason,
    )


def _partition_preservation_contract(
    fn: Any,
) -> Tuple[Optional[Tuple[str, ...]], Optional[str]]:
    if not hasattr(fn, PARTITION_PRESERVATION_ATTRIBUTE):
        return None, "missing_partitioning_contract"
    value = getattr(fn, PARTITION_PRESERVATION_ATTRIBUTE)
    if (
        not isinstance(value, tuple)
        or not value
        or any(not isinstance(column, str) or not column for column in value)
        or len(set(value)) != len(value)
    ):
        return None, "invalid_partitioning_contract"
    return value, None


def _is_sync_callable_class(fn: Any) -> bool:
    return isinstance(fn, CallableClass) and not (
        inspect.iscoroutinefunction(fn.__call__)
        or inspect.isasyncgenfunction(fn.__call__)
    )


def _is_sync_plain_function(fn: Any) -> bool:
    return inspect.isfunction(fn) and not (
        inspect.iscoroutinefunction(fn) or inspect.isasyncgenfunction(fn)
    )


def _fixed_actor_pool_size(compute: Any) -> Optional[int]:
    if not isinstance(compute, ActorPoolStrategy):
        return None
    if not (
        compute.min_size == compute.max_size == compute.initial_size
        and isinstance(compute.min_size, numbers.Integral)
        and not isinstance(compute.min_size, bool)
        and int(compute.min_size) > 0
    ):
        return None
    return int(compute.min_size)


def _fixed_task_pool_size(compute: Any) -> Optional[int]:
    if not isinstance(compute, TaskPoolStrategy):
        return None
    if (
        not isinstance(compute.size, numbers.Integral)
        or isinstance(compute.size, bool)
        or int(compute.size) <= 0
    ):
        return None
    return int(compute.size)


def _uses_one_gpu(ray_remote_args: Dict[str, Any]) -> bool:
    num_gpus = ray_remote_args.get("num_gpus")
    return (
        isinstance(num_gpus, numbers.Real)
        and not isinstance(num_gpus, bool)
        and float(num_gpus) == 1.0
    )


def _label_selector(
    ray_remote_args: Dict[str, Any],
) -> Tuple[Optional[Dict[str, str]], bool]:
    selector = ray_remote_args.get("label_selector")
    if selector is None:
        return {}, True
    if not isinstance(selector, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in selector.items()
    ):
        return None, False
    return selector, True


def _selectors_conflict(
    read_selector: Dict[str, str], actor_selector: Dict[str, str]
) -> bool:
    return any(
        key in read_selector and read_selector[key] != value
        for key, value in actor_selector.items()
    )


def _validate_v1_parquet_datasource(
    read_op: Read,
    datasource: ParquetDatasource,
) -> Tuple[Optional[Tuple[str, ...]], Optional[str]]:
    if read_op.datasource_or_legacy_reader is not datasource:
        return None, "parquet_reader_mismatch"

    try:
        import pyarrow.fs as pafs

        filesystem = datasource._filesystem
        if type(filesystem) is not pafs.LocalFileSystem:
            return None, "parquet_filesystem_not_local"
    except (AttributeError, ImportError):
        return None, "parquet_filesystem_not_local"

    paths = getattr(datasource, "_pq_paths", None)
    if (
        not isinstance(paths, list)
        or not paths
        or any(
            not isinstance(path, str) or not path or not os.path.isabs(path)
            for path in paths
        )
    ):
        return None, "parquet_source_paths_invalid"

    projection_map = getattr(datasource, "_projection_map", None)
    if projection_map is None or projection_map == {}:
        return None, "parquet_projection_missing"
    if not isinstance(projection_map, dict) or any(
        not isinstance(column, str) or not column or projected != column
        for column, projected in projection_map.items()
    ):
        return None, "parquet_projection_invalid"
    projection = tuple(projection_map)

    if getattr(datasource, "_partition_columns", None):
        return None, "parquet_partition_columns_unsupported"
    if getattr(datasource, "_predicate_expr", None) is not None:
        return None, "parquet_predicate_unsupported"
    if getattr(datasource, "_block_udf", None) is not None:
        return None, "parquet_block_udf_unsupported"
    if getattr(datasource, "_shuffle", None) is not None:
        return None, "parquet_file_shuffle_unsupported"
    if getattr(datasource, "_read_schema", None) is not None:
        return None, "parquet_schema_override_unsupported"
    if getattr(datasource, "_include_paths", False) or getattr(
        datasource, "_include_row_hash", False
    ):
        return None, "parquet_synthetic_columns_unsupported"

    scanner_kwargs = getattr(datasource, "_scanner_kwargs", None)
    default_scanner_keys = {
        "batch_size",
        "batch_readahead",
        "fragment_readahead",
        "fragment_scan_options",
    }
    if not isinstance(scanner_kwargs, dict) or not set(scanner_kwargs).issubset(
        default_scanner_keys
    ):
        return None, "parquet_scan_options_unsupported"

    read_remote_args = read_op.ray_remote_args
    if not isinstance(read_remote_args, dict) or set(read_remote_args) - {
        "label_selector",
        "scheduling_strategy",
    }:
        return None, "read_remote_args_unsupported"
    _, valid_selector = _label_selector(read_remote_args)
    if not valid_selector:
        return None, "read_label_selector_invalid"

    return projection, None


def select_parquet_range_map_groups_candidate(
    op: LogicalOperator,
    data_context: DataContext,
) -> ParquetRangeMapGroupsSelectionResult:
    """Inspect ``op`` and return a safe structural candidate or fallback reason."""

    if (
        data_context.get_config(PARQUET_RANGE_MAP_GROUPS_ENABLED_CONFIG, False)
        is not True
    ):
        return _fallback("parquet_range_map_groups_disabled")
    if data_context.checkpoint_config is not None:
        return _fallback("checkpointing_unsupported")
    if data_context.retried_map_errors:
        return _fallback("map_error_retries_unsupported")
    if data_context.actor_task_retry_on_errors:
        return _fallback("actor_task_retries_unsupported")
    if data_context.actor_init_retry_on_errors:
        return _fallback("actor_init_retries_unsupported")
    if data_context.max_errored_blocks != 0:
        return _fallback("errored_blocks_unsupported")
    if type(op) is not MapGroups:
        return _fallback("root_not_map_groups")
    map_groups_op = op
    if (
        len(map_groups_op.input_dependencies) != 1
        or type(map_groups_op.input_dependencies[0]) is not MapBatches
    ):
        return _fallback("map_groups_input_not_tokenizer")
    tokenizer_op = map_groups_op.input_dependencies[0]
    if (
        len(tokenizer_op.input_dependencies) != 1
        or type(tokenizer_op.input_dependencies[0]) is not Read
    ):
        return _fallback("tokenizer_input_not_read")
    read_op = tokenizer_op.input_dependencies[0]
    if type(read_op.datasource) is not ParquetDatasource:
        return _fallback("read_not_v1_parquet")
    datasource = read_op.datasource

    projection, reason = _validate_v1_parquet_datasource(read_op, datasource)
    if reason is not None:
        return _fallback(reason)
    assert projection is not None

    if (
        not isinstance(map_groups_op.key, (list, tuple))
        or len(map_groups_op.key) < 2
        or any(
            not isinstance(column, str) or not column for column in map_groups_op.key
        )
        or len(set(map_groups_op.key)) != len(map_groups_op.key)
    ):
        return _fallback("group_keys_not_composite")
    group_keys = tuple(map_groups_op.key)

    if not map_groups_op.num_partitions_explicit:
        return _fallback("num_partitions_not_explicit")
    if (
        not isinstance(map_groups_op.num_partitions, numbers.Integral)
        or isinstance(map_groups_op.num_partitions, bool)
        or int(map_groups_op.num_partitions) <= 0
    ):
        return _fallback("num_partitions_invalid")
    num_partitions = int(map_groups_op.num_partitions)
    if map_groups_op.shuffle_strategy not in (
        ShuffleStrategy.HASH_SHUFFLE,
        ShuffleStrategy.GPU_SHUFFLE,
    ):
        return _fallback("shuffle_strategy_unsupported")

    if not isinstance(tokenizer_op.fn, CallableClass):
        return _fallback("tokenizer_not_callable_class")
    if not _is_sync_callable_class(tokenizer_op.fn):
        return _fallback("tokenizer_async_unsupported")
    contract, reason = _partition_preservation_contract(tokenizer_op.fn)
    if reason is not None:
        return _fallback(reason)
    assert contract is not None
    if len(contract) != 1:
        return _fallback("partitioning_contract_not_single_column")
    partition_key = contract[0]
    if partition_key != group_keys[0]:
        return _fallback("partition_key_not_leading_group_key")
    if partition_key not in projection:
        return _fallback("partition_key_not_projected")
    if any(group_key not in projection for group_key in group_keys):
        return _fallback("group_key_not_projected")

    if tokenizer_op.batch_format != "cudf":
        return _fallback("tokenizer_batch_format_not_cudf")
    if (
        not isinstance(tokenizer_op.batch_size, numbers.Integral)
        or isinstance(tokenizer_op.batch_size, bool)
        or int(tokenizer_op.batch_size) <= 0
    ):
        return _fallback("tokenizer_batch_size_not_fixed")
    if tokenizer_op.can_modify_num_rows:
        return _fallback("tokenizer_may_modify_row_count")
    if tokenizer_op.per_block_limit is not None:
        return _fallback("tokenizer_per_block_limit_unsupported")
    if tokenizer_op.ray_remote_args_fn is not None:
        return _fallback("tokenizer_dynamic_remote_args_unsupported")
    if set(tokenizer_op.ray_remote_args) & _REPLAY_REMOTE_ARGS:
        return _fallback("tokenizer_retry_options_unsupported")
    tokenizer_pool_size = _fixed_actor_pool_size(tokenizer_op.compute)
    if tokenizer_pool_size is None:
        return _fallback("tokenizer_actor_pool_not_fixed")
    if tokenizer_op.compute.enable_true_multi_threading or (
        tokenizer_op.compute.max_tasks_in_flight_per_actor not in (None, 1)
    ):
        return _fallback("tokenizer_actor_concurrency_unsupported")
    if not _uses_one_gpu(tokenizer_op.ray_remote_args):
        return _fallback("tokenizer_gpu_resource_not_one")

    if isinstance(map_groups_op.fn, CallableClass):
        return _fallback("group_udf_callable_class_unsupported")
    if not inspect.isfunction(map_groups_op.fn):
        return _fallback("group_udf_not_plain_function")
    if not _is_sync_plain_function(map_groups_op.fn):
        return _fallback("group_udf_async_unsupported")
    partition_contract, reason = resolve_map_groups_partition_contract(
        map_groups_op.fn,
        data_context,
        batch_format=map_groups_op.batch_format,
        zero_copy_batch=map_groups_op.zero_copy_batch,
        fn_args=map_groups_op.fn_args,
        fn_kwargs=map_groups_op.fn_kwargs,
    )
    if map_groups_op.batch_format != "cudf":
        return _fallback("group_batch_format_not_cudf")
    if map_groups_op.ray_remote_args_fn is not None:
        return _fallback("group_dynamic_remote_args_unsupported")
    if set(map_groups_op.ray_remote_args) & _REPLAY_REMOTE_ARGS:
        return _fallback("group_retry_options_unsupported")
    group_pool_size = _fixed_task_pool_size(map_groups_op.compute)
    if group_pool_size is None:
        return _fallback("group_task_pool_not_fixed")
    if group_pool_size != tokenizer_pool_size:
        return _fallback("tokenizer_group_pool_size_mismatch")
    worker_concurrency = tokenizer_pool_size
    if worker_concurrency > num_partitions:
        return _fallback("worker_concurrency_exceeds_num_partitions")
    if not _uses_one_gpu(map_groups_op.ray_remote_args):
        return _fallback("group_gpu_resource_not_one")

    tokenizer_args = tokenizer_op.ray_remote_args
    group_args = map_groups_op.ray_remote_args
    read_selector, read_selector_valid = _label_selector(read_op.ray_remote_args)
    tokenizer_selector, tokenizer_selector_valid = _label_selector(tokenizer_args)
    group_selector, group_selector_valid = _label_selector(group_args)
    if not (read_selector_valid and tokenizer_selector_valid and group_selector_valid):
        return _fallback("actor_label_selector_invalid")
    assert read_selector is not None
    assert tokenizer_selector is not None
    assert group_selector is not None
    if _selectors_conflict(read_selector, tokenizer_selector) or _selectors_conflict(
        read_selector, group_selector
    ):
        return _fallback("actor_label_selector_conflicts_with_read")
    if _selectors_conflict(tokenizer_selector, group_selector) or _selectors_conflict(
        group_selector, tokenizer_selector
    ):
        return _fallback("tokenizer_group_label_selectors_conflict")
    tokenizer_resources = {
        key: value for key, value in tokenizer_args.items() if key != "label_selector"
    }
    group_resources = {
        key: value for key, value in group_args.items() if key != "label_selector"
    }
    if tokenizer_resources != group_resources:
        return _fallback("tokenizer_group_resources_incompatible")
    actor_args = dict(tokenizer_resources)
    merged_actor_selector = {**tokenizer_selector, **group_selector}
    if merged_actor_selector:
        actor_args["label_selector"] = merged_actor_selector

    candidate = ParquetRangeMapGroupsCandidate(
        read_op=read_op,
        datasource=datasource,
        tokenizer_op=tokenizer_op,
        map_groups_op=map_groups_op,
        partition_key=partition_key,
        group_keys=group_keys,
        num_partitions=num_partitions,
        worker_concurrency=worker_concurrency,
        projection=projection,
        compute=tokenizer_op.compute,
        ray_remote_args=actor_args,
        partition_contract=partition_contract,
        partition_contract_fallback_reason=reason,
    )
    return ParquetRangeMapGroupsSelectionResult(
        candidate=candidate,
        fallback_reason=None,
    )


__all__ = [
    "PARQUET_RANGE_MAP_GROUPS_ENABLED_CONFIG",
    "PARTITION_PRESERVATION_ATTRIBUTE",
    "ParquetRangeMapGroupsCandidate",
    "ParquetRangeMapGroupsSelectionResult",
    "select_parquet_range_map_groups_candidate",
]
