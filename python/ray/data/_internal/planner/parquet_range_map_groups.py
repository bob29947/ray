"""Fused local-cuDF execution for metadata-planned ``map_groups``.

This module is intentionally a physical backend, not a user-facing API.  The
ordinary application remains ``read_parquet().map_batches().groupby().map_groups()``.
An independent fail-closed selector and the V1 metadata adapter decide whether
this backend is safe before this module constructs any workers.
"""

from __future__ import annotations

import collections.abc
import math
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Dict, Generator, Iterable, Iterator, List, Optional, Tuple

import ray
from ray.data._internal.compute import ActorPoolStrategy
from ray.data._internal.datasource.parquet_range import (
    ParquetRangeLayout,
    ParquetRowGroupFragment,
)
from ray.data._internal.datasource.parquet_range_v1 import (
    verify_posix_source_identity,
)
from ray.data._internal.execution.interfaces import (
    BlockEntry,
    ExecutionOptions,
    RefBundle,
)
from ray.data._internal.execution.interfaces.task_context import TaskContext
from ray.data._internal.execution.operators.actor_pool_map_operator import (
    ActorPoolMapOperator,
)
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.execution.operators.map_transformer import (
    BlockMapTransformFn,
    CustomOpStatsReportFn,
    MapTransformer,
)
from ray.data._internal.logical.operators import MapBatches, MapGroups
from ray.data._internal.output_buffer import BlockOutputBuffer, OutputBlockSizeOption
from ray.data._internal.planner.plan_udf_map_op import (
    _TransformingBatchIterator,
    _get_udf,
    _validate_batch_output,
)
from ray.data.block import (
    Block,
    BlockAccessor,
    BlockMetadata,
    CustomOpStats,
    _is_cudf_dataframe,
)
from ray.data.context import DataContext


@dataclass(frozen=True)
class ParquetRangeMapGroupsWork:
    """One immutable key range assigned to one fixed GPU actor."""

    partition_id: int
    lower_bound: int
    upper_bound: int
    fragments: Tuple[ParquetRowGroupFragment, ...]
    projection: Tuple[str, ...]
    partition_key: str
    group_keys: Tuple[str, ...]
    source_schema: Any
    estimated_scanned_rows: int
    estimated_encoded_bytes: int
    estimated_uncompressed_bytes: int


@dataclass(frozen=True)
class ParquetRangeMapGroupsStats(CustomOpStats):
    """Worker-reported timings and transfer measurements for one range."""

    partition_id: int
    lower_bound: int
    upper_bound: int
    selected_row_groups: int
    estimated_scanned_rows: int
    input_rows: int
    tokenized_rows: int
    groups_invoked: int
    output_rows: int
    output_blocks: int
    read_time_s: float
    map_time_s: float
    sort_time_s: float
    group_time_s: float
    output_time_s: float
    gpu_peak_memory_bytes: int
    host_transfer_bytes: int
    rmm_pool_initial_bytes: int
    rmm_pool_maximum_bytes: int


@dataclass(frozen=True)
class _RmmPoolConfig:
    """Realized bounds for the dedicated actor's device-memory pool."""

    initial_bytes: int
    maximum_bytes: int


@dataclass
class _RmmPoolState:
    """Keep the installed pool and its upstream resource alive in the actor."""

    config: _RmmPoolConfig
    upstream: Any
    pool: Any


def _resolve_rmm_pool_config(
    *,
    free_bytes: int,
    gpu_memory_budget_bytes: int,
    initial_bytes: int,
    reserve_bytes: int,
) -> _RmmPoolConfig:
    """Clamp configured RMM bounds to the planner budget and live free memory."""

    alignment = 256

    def align_down(value: int) -> int:
        return int(value) // alignment * alignment

    if free_bytes <= 0:
        raise RuntimeError("RMM reported no free GPU memory for the fused actor.")
    if gpu_memory_budget_bytes <= 0:
        raise ValueError("gpu_memory_budget_bytes must be positive")
    if initial_bytes <= 0:
        raise ValueError("RMM initial pool size must be positive")
    if reserve_bytes < 0:
        raise ValueError("RMM reserved GPU memory must be nonnegative")

    live_limit = align_down(max(0, free_bytes - reserve_bytes))
    maximum = min(align_down(gpu_memory_budget_bytes), live_limit)
    if maximum < alignment:
        raise RuntimeError(
            "Insufficient free GPU memory for the fused actor's RMM pool: "
            f"free_bytes={free_bytes}, reserve_bytes={reserve_bytes}, "
            f"gpu_memory_budget_bytes={gpu_memory_budget_bytes}"
        )
    return _RmmPoolConfig(
        initial_bytes=min(align_down(initial_bytes), maximum),
        maximum_bytes=maximum,
    )


def _configure_rmm_pool(
    *,
    gpu_memory_budget_bytes: int,
    initial_bytes: int,
    reserve_bytes: int,
) -> _RmmPoolState:
    """Install an actor-local RMM pool before constructing either GPU UDF.

    A selected range actor is a dedicated one-GPU process, so changing its
    process-global allocator cannot affect unrelated Ray tasks.  Initializing
    here avoids serialized cudaMalloc/cudaFree calls across the tokenizer and
    thousands of synchronous ``map_groups`` invocations.
    """

    import rmm

    free_bytes, _ = rmm.mr.available_device_memory()
    config = _resolve_rmm_pool_config(
        free_bytes=int(free_bytes),
        gpu_memory_budget_bytes=gpu_memory_budget_bytes,
        initial_bytes=initial_bytes,
        reserve_bytes=reserve_bytes,
    )
    # Cloudpickle may import a callable's module while constructing this map
    # worker, before the transformer's init hook runs. Wrapping the current
    # resource preserves any such live allocations; ``rmm.reinitialize``
    # would invalidate them.
    upstream = rmm.mr.get_current_device_resource()
    pool = rmm.mr.PoolMemoryResource(
        upstream,
        initial_pool_size=config.initial_bytes,
        maximum_pool_size=config.maximum_bytes,
    )
    rmm.mr.set_current_device_resource(pool)

    # Import CuPy only after RMM owns the device resource.  The tokenizer and
    # map_groups callable then share the same bounded allocator as cuDF.
    import cupy
    from rmm.allocators.cupy import rmm_cupy_allocator

    cupy.cuda.set_allocator(rmm_cupy_allocator)
    return _RmmPoolState(config=config, upstream=upstream, pool=pool)


def validate_cudf_projection_schema(
    schema: Any, projection: Tuple[str, ...]
) -> Optional[str]:
    """Return a fallback reason for Arrow types not normalized by this backend."""

    import pyarrow as pa

    if schema is None:
        return "missing_source_schema"
    if not projection:
        return "empty_projection"
    for column in projection:
        indices = schema.get_all_field_indices(column)
        if len(indices) != 1:
            return "invalid_projection_schema"
        arrow_type = schema.field(indices[0]).type
        supported = (
            pa.types.is_integer(arrow_type)
            or pa.types.is_floating(arrow_type)
            or pa.types.is_boolean(arrow_type)
            or pa.types.is_string(arrow_type)
            or pa.types.is_large_string(arrow_type)
            or (pa.types.is_timestamp(arrow_type) and arrow_type.tz is None)
        )
        if not supported:
            return "unsupported_projection_type"
    return None


def _normalize_cudf_frame(frame: Any, work: ParquetRangeMapGroupsWork) -> Any:
    """Match the primitive dtypes produced by Arrow-to-cuDF batch conversion."""

    import pyarrow as pa
    import numpy as np
    from cudf.api.types import is_string_dtype

    actual_columns = tuple(frame.columns)
    if set(actual_columns) != set(work.projection):
        raise ValueError(
            "Direct cuDF Parquet read returned columns different from the "
            f"planned projection: expected={work.projection!r}, "
            f"actual={actual_columns!r}"
        )
    frame = frame.loc[:, list(work.projection)]
    for column in work.projection:
        field = work.source_schema.field(column)
        arrow_type = field.type
        series = frame[column]
        if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
            if not is_string_dtype(series.dtype):
                frame[column] = series.astype("str")
        elif pa.types.is_timestamp(arrow_type):
            expected = f"datetime64[{arrow_type.unit}]"
            if str(series.dtype) != expected:
                frame[column] = series.astype(expected)
        else:
            expected = str(np.dtype(arrow_type.to_pandas_dtype()))
            if str(series.dtype) != expected:
                frame[column] = series.astype(expected)
    return frame.reset_index(drop=True)


def _iter_exact_cudf_batches(
    frames: Iterable[Any], batch_size: int, cudf: Any
) -> Iterator[Any]:
    """Rebatch a row-group stream without repeatedly copying its remainder."""

    # Each entry holds a frame and the first unconsumed row.  In particular,
    # do not materialize ``combined.iloc[batch_size:]`` after every batch: one
    # large Parquet read would otherwise copy progressively smaller tails and
    # turn exact batching into quadratic device-memory traffic.
    pending = deque()
    pending_rows = 0

    def take(rows: int) -> Any:
        nonlocal pending_rows
        pieces = []
        remaining = rows
        while remaining:
            frame, offset = pending[0]
            available = len(frame) - offset
            consumed = min(available, remaining)
            pieces.append(frame.iloc[offset : offset + consumed])
            if consumed == available:
                pending.popleft()
            else:
                pending[0] = (frame, offset + consumed)
            pending_rows -= consumed
            remaining -= consumed
        combined = (
            pieces[0] if len(pieces) == 1 else cudf.concat(pieces, ignore_index=True)
        )
        return combined.reset_index(drop=True)

    for frame in frames:
        frame_rows = len(frame)
        if frame_rows == 0:
            continue
        pending.append((frame, 0))
        pending_rows += frame_rows
        while pending_rows >= batch_size:
            yield take(batch_size)
    if pending_rows:
        yield take(pending_rows)


def _invoke_row_preserving_tokenizer(
    batch: Any,
    tokenizer_fn: Any,
    *,
    partition_key: str,
    zero_copy_batch: bool,
    cudf: Any,
) -> Any:
    udf_batch = batch if zero_copy_batch else batch.copy(deep=True)
    outputs = list(_TransformingBatchIterator((udf_batch,), tokenizer_fn))
    if not outputs:
        raise ValueError("The partition-preserving tokenizer returned no rows.")
    for output in outputs:
        if not _is_cudf_dataframe(output):
            raise TypeError(
                "The metadata range backend requires the tokenizer to return "
                "cudf.DataFrame batches so rows remain GPU-resident."
            )
    output = (
        outputs[0] if len(outputs) == 1 else cudf.concat(outputs, ignore_index=True)
    ).reset_index(drop=True)
    if len(output) != len(batch):
        raise ValueError(
            "The tokenizer violated udf_modifying_row_count=False: "
            f"input_rows={len(batch)}, output_rows={len(output)}"
        )
    if partition_key not in output.columns:
        raise ValueError(
            f"The tokenizer removed preserved partition column {partition_key!r}."
        )
    input_key = batch[partition_key].reset_index(drop=True)
    output_key = output[partition_key].reset_index(drop=True)
    input_missing = input_key.isnull()
    output_missing = output_key.isnull()
    values_equal = (input_key == output_key).fillna(False)
    # Keep the success path to one device synchronization per tokenizer batch.
    # Detailed diagnostics may synchronize again only after the contract fails.
    preserved = (~input_missing & ~output_missing & values_equal).all()
    if not bool(preserved):
        if bool((input_missing | output_missing).any()):
            raise ValueError(
                f"Preserved partition column {partition_key!r} contains nulls."
            )
        raise ValueError(
            f"The tokenizer changed preserved partition column {partition_key!r}."
        )
    return output


def _group_boundaries(frame: Any, group_keys: Tuple[str, ...], cupy: Any) -> List[int]:
    """Return host boundary indices while retaining all row columns on GPU."""

    row_count = len(frame)
    if row_count == 0:
        return [0]
    changed = cupy.zeros(row_count, dtype=cupy.bool_)
    changed[0] = True
    for column in group_keys:
        values = frame[column]
        previous = values.shift(1)
        value_missing = values.isnull()
        previous_missing = previous.isnull()
        # Ray's ordinary group-by path places consecutive missing/NaN keys in
        # one group. Parquet-to-cuDF exposes both through ``isnull()``; mask
        # equal missing pairs before comparing their values because NaN != NaN.
        both_missing = value_missing & previous_missing
        value_changed = ((values != previous) & ~both_missing).fillna(False).to_cupy()
        missing_changed = (value_missing != previous_missing).to_cupy()
        changed |= value_changed | missing_changed
    starts = cupy.flatnonzero(changed).get().tolist()
    return [int(value) for value in starts] + [row_count]


def _is_sorted_by_group_keys(
    frame: Any, group_keys: Tuple[str, ...], array_module: Any
) -> bool:
    """Return whether adjacent rows are lexicographically group-key ordered."""

    row_count = len(frame)
    if row_count < 2:
        return True

    def as_array(series: Any) -> Any:
        if hasattr(series, "to_cupy"):
            return series.to_cupy()
        return array_module.asarray(series.to_numpy())

    equal_prefix = array_module.ones(row_count - 1, dtype=array_module.bool_)
    out_of_order = array_module.zeros(row_count - 1, dtype=array_module.bool_)
    for column in group_keys:
        values = frame[column]
        previous = values.iloc[:-1].reset_index(drop=True)
        current = values.iloc[1:].reset_index(drop=True)
        previous_missing = previous.isnull()
        current_missing = current.isnull()
        both_present = ~previous_missing & ~current_missing
        less = ((current < previous) & both_present).fillna(False)
        # ``sort_values`` puts null/NaN keys last. A present value following a
        # missing value is therefore descending at the first differing key.
        missing_descends = previous_missing & ~current_missing
        out_of_order |= equal_prefix & as_array(less | missing_descends)

        both_missing = previous_missing & current_missing
        equal = both_missing | ((current == previous) & both_present).fillna(False)
        equal_prefix &= as_array(equal)

    return not bool(array_module.any(out_of_order))


def _iter_group_outputs(group_fn: Any, group: Any) -> Iterator[Any]:
    """Mirror ``_apply_udf_to_groups`` iterator semantics and validation."""

    result = group_fn(group)
    if isinstance(result, collections.abc.Iterator):
        outputs = result
    else:
        outputs = iter((result,))
    for output in outputs:
        _validate_batch_output(output)
        yield output


def _validate_group_keys(frame: Any, group_keys: Tuple[str, ...]) -> None:
    missing_group_keys = tuple(key for key in group_keys if key not in frame.columns)
    if missing_group_keys:
        raise ValueError(
            f"The tokenizer output is missing group keys {missing_group_keys!r}."
        )


def _sample_gpu_memory(cupy: Any) -> int:
    try:
        free_bytes, total_bytes = cupy.cuda.runtime.memGetInfo()
        return int(total_bytes - free_bytes)
    except Exception:
        return 0


def _execute_range(
    work: ParquetRangeMapGroupsWork,
    *,
    tokenizer_fn: Any,
    group_fn: Any,
    tokenizer_batch_size: int,
    tokenizer_zero_copy_batch: bool,
    group_zero_copy_batch: bool,
    target_max_block_size: int,
    rmm_pool_config: _RmmPoolConfig,
) -> Generator[Block, None, ParquetRangeMapGroupsStats]:
    """Read, tokenize, sort, and invoke every group exactly once on one GPU."""

    import cudf
    import cupy

    read_time_s = 0.0
    map_time_s = 0.0
    sort_time_s = 0.0
    group_time_s = 0.0
    output_time_s = 0.0
    input_rows = 0
    tokenized_rows = 0
    group_count = 0
    output_rows = 0
    host_transfer_bytes = 0
    gpu_peak_memory_bytes = _sample_gpu_memory(cupy)

    def raw_frames() -> Iterator[Any]:
        nonlocal read_time_s, input_rows, gpu_peak_memory_bytes
        for fragment in work.fragments:
            verify_posix_source_identity(fragment.path, fragment.source_identity)
            started = time.perf_counter()
            # One multi-row-group read per file is substantially faster than
            # repeatedly re-entering cuDF's Parquet reader. Exact range
            # filtering below removes overlap amplification before the UDF.
            frame = cudf.read_parquet(
                fragment.path,
                columns=list(work.projection),
                row_groups=list(fragment.row_group_ids),
            )
            frame = _normalize_cudf_frame(frame, work)
            frame = frame[
                (frame[work.partition_key] >= work.lower_bound)
                & (frame[work.partition_key] <= work.upper_bound)
            ].reset_index(drop=True)
            read_time_s += time.perf_counter() - started
            input_rows += len(frame)
            gpu_peak_memory_bytes = max(gpu_peak_memory_bytes, _sample_gpu_memory(cupy))
            if len(frame):
                yield frame
            verify_posix_source_identity(fragment.path, fragment.source_identity)

    tokenized_batches: List[Any] = []
    for raw_batch in _iter_exact_cudf_batches(raw_frames(), tokenizer_batch_size, cudf):
        started = time.perf_counter()
        tokenized = _invoke_row_preserving_tokenizer(
            raw_batch,
            tokenizer_fn,
            partition_key=work.partition_key,
            zero_copy_batch=tokenizer_zero_copy_batch,
            cudf=cudf,
        )
        map_time_s += time.perf_counter() - started
        tokenized_rows += len(tokenized)
        tokenized_batches.append(tokenized)
        gpu_peak_memory_bytes = max(gpu_peak_memory_bytes, _sample_gpu_memory(cupy))

    output_blocks = 0
    output_buffer = BlockOutputBuffer(
        OutputBlockSizeOption.of(target_max_block_size=target_max_block_size)
    )
    if tokenized_batches:
        started = time.perf_counter()
        tokenized_partition = (
            tokenized_batches[0]
            if len(tokenized_batches) == 1
            else cudf.concat(tokenized_batches, ignore_index=True)
        )
        tokenized_batches.clear()
        _validate_group_keys(tokenized_partition, work.group_keys)
        if not _is_sorted_by_group_keys(tokenized_partition, work.group_keys, cupy):
            tokenized_partition = tokenized_partition.sort_values(
                list(work.group_keys), ignore_index=True, na_position="last"
            )
        boundaries = _group_boundaries(tokenized_partition, work.group_keys, cupy)
        host_transfer_bytes += max(0, len(boundaries) - 1) * 8
        sort_time_s += time.perf_counter() - started
        gpu_peak_memory_bytes = max(gpu_peak_memory_bytes, _sample_gpu_memory(cupy))

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            group_count += 1
            group = tokenized_partition.iloc[start:end]
            if not group_zero_copy_batch:
                group = group.copy(deep=True)
            started = time.perf_counter()
            outputs = _iter_group_outputs(group_fn, group)
            for output in outputs:
                group_time_s += time.perf_counter() - started
                started = time.perf_counter()
                output_buffer.add_batch(output)
                for output_block in output_buffer.iter_ready_blocks():
                    accessor = BlockAccessor.for_block(output_block)
                    output_rows += accessor.num_rows()
                    output_blocks += 1
                    host_transfer_bytes += accessor.size_bytes()
                    output_time_s += time.perf_counter() - started
                    yield output_block
                    started = time.perf_counter()
                output_time_s += time.perf_counter() - started
                started = time.perf_counter()
            group_time_s += time.perf_counter() - started
            if group_count % 128 == 0:
                gpu_peak_memory_bytes = max(
                    gpu_peak_memory_bytes, _sample_gpu_memory(cupy)
                )

    started = time.perf_counter()
    output_buffer.finalize()
    for output_block in output_buffer.iter_ready_blocks():
        accessor = BlockAccessor.for_block(output_block)
        output_rows += accessor.num_rows()
        output_blocks += 1
        host_transfer_bytes += accessor.size_bytes()
        output_time_s += time.perf_counter() - started
        yield output_block
        started = time.perf_counter()
    output_time_s += time.perf_counter() - started

    stats = ParquetRangeMapGroupsStats(
        partition_id=work.partition_id,
        lower_bound=work.lower_bound,
        upper_bound=work.upper_bound,
        selected_row_groups=sum(
            len(fragment.row_group_ids) for fragment in work.fragments
        ),
        estimated_scanned_rows=work.estimated_scanned_rows,
        input_rows=input_rows,
        tokenized_rows=tokenized_rows,
        groups_invoked=group_count,
        output_rows=output_rows,
        output_blocks=output_blocks,
        read_time_s=read_time_s,
        map_time_s=map_time_s,
        sort_time_s=sort_time_s,
        group_time_s=group_time_s,
        output_time_s=output_time_s,
        gpu_peak_memory_bytes=gpu_peak_memory_bytes,
        host_transfer_bytes=host_transfer_bytes,
        rmm_pool_initial_bytes=rmm_pool_config.initial_bytes,
        rmm_pool_maximum_bytes=rmm_pool_config.maximum_bytes,
    )
    return stats


class ParquetRangeMapGroupsActorPoolMapOperator(ActorPoolMapOperator):
    """Fixed actor pool that exposes planner and worker metrics in stats output."""

    def __init__(
        self,
        *args: Any,
        plan_metrics: Dict[str, Any],
        actor_start_timeout_s: float,
        **kwargs: Any,
    ) -> None:
        self._parquet_range_plan_metrics = plan_metrics
        self._actor_start_timeout_s = actor_start_timeout_s
        super().__init__(*args, **kwargs)

    def start(self, options: ExecutionOptions) -> None:
        super().start(options)
        # All actors must be live before the first range is dispatched. With one
        # in-flight task per actor, this guarantees a distinct actor/GPU per range.
        pending = self._actor_pool.get_pending_actor_refs()
        if pending:
            ray.get(pending, timeout=self._actor_start_timeout_s)

    def _worker_stats(self) -> List[Dict[str, Any]]:
        by_partition: Dict[int, ParquetRangeMapGroupsStats] = {}
        for block_stats in self._output_blocks_stats:
            task_stats = block_stats.task_exec_stats
            if task_stats is None:
                continue
            for custom_stats in task_stats.custom_op_stats:
                if isinstance(custom_stats, ParquetRangeMapGroupsStats):
                    by_partition[custom_stats.partition_id] = custom_stats
        return [
            asdict(by_partition[partition_id]) for partition_id in sorted(by_partition)
        ]

    def _extra_metrics(self) -> Dict[str, Any]:
        metrics = super()._extra_metrics()
        metrics["parquet_range_map_groups_plan"] = self._parquet_range_plan_metrics
        worker_stats = self._worker_stats()
        if worker_stats:
            metrics["parquet_range_map_groups_workers"] = worker_stats
        return metrics


def _make_work(
    layout: ParquetRangeLayout,
    *,
    projection: Tuple[str, ...],
    partition_key: str,
    group_keys: Tuple[str, ...],
    source_schema: Any,
) -> Tuple[ParquetRangeMapGroupsWork, ...]:
    return tuple(
        ParquetRangeMapGroupsWork(
            partition_id=partition.partition_id,
            lower_bound=partition.lower_bound,
            upper_bound=partition.upper_bound,
            fragments=partition.fragments,
            projection=projection,
            partition_key=partition_key,
            group_keys=group_keys,
            source_schema=source_schema,
            estimated_scanned_rows=partition.estimated_scanned_rows,
            estimated_encoded_bytes=partition.estimated_encoded_bytes,
            estimated_uncompressed_bytes=partition.estimated_uncompressed_bytes,
        )
        for partition in layout.partitions
    )


def build_parquet_range_map_groups_operator(
    *,
    layout: ParquetRangeLayout,
    footer_planning_time_s: float,
    projection: Tuple[str, ...],
    partition_key: str,
    group_keys: Tuple[str, ...],
    source_schema: Any,
    tokenizer_op: MapBatches,
    map_groups_op: MapGroups,
    data_context: DataContext,
    ray_remote_args: Dict[str, Any],
    estimated_peak_gpu_memory_bytes: int = 0,
    gpu_memory_budget_bytes: int = 0,
) -> ParquetRangeMapGroupsActorPoolMapOperator:
    """Build the shuffle-free fixed-GPU physical operator for a validated plan."""

    works = _make_work(
        layout,
        projection=projection,
        partition_key=partition_key,
        group_keys=group_keys,
        source_schema=source_schema,
    )
    if len(works) != map_groups_op.num_partitions:
        raise ValueError(
            "The range layout must contain exactly one range per requested actor."
        )

    def input_data_factory(_: int) -> List[RefBundle]:
        bundles = []
        for work in works:
            work_ref = ray.put(work)
            bundles.append(
                RefBundle(
                    blocks=(
                        BlockEntry(
                            ref=work_ref,
                            metadata=BlockMetadata(
                                num_rows=work.estimated_scanned_rows,
                                size_bytes=max(1, work.estimated_encoded_bytes),
                                input_files=tuple(
                                    fragment.path for fragment in work.fragments
                                ),
                                exec_stats=None,
                            ),
                        ),
                    ),
                    owns_blocks=False,
                    schema=None,
                )
            )
        return bundles

    input_op = InputDataBuffer(data_context, input_data_factory=input_data_factory)
    tokenizer_fn, tokenizer_init_fn = _get_udf(
        tokenizer_op.fn,
        tokenizer_op.fn_args,
        tokenizer_op.fn_kwargs,
        tokenizer_op.fn_constructor_args,
        tokenizer_op.fn_constructor_kwargs,
        compute=tokenizer_op.compute,
    )
    group_fn, group_init_fn = _get_udf(
        map_groups_op.fn,
        map_groups_op.fn_args,
        map_groups_op.fn_kwargs,
        map_groups_op.fn_constructor_args,
        map_groups_op.fn_constructor_kwargs,
        compute=map_groups_op.compute,
    )

    rmm_pool_initial_bytes = data_context.get_config(
        "parquet_range_map_groups_rmm_pool_initial_bytes", 8 * 1024**3
    )
    rmm_pool_reserve_bytes = data_context.get_config(
        "parquet_range_map_groups_rmm_pool_reserve_bytes", 2 * 1024**3
    )
    if (
        isinstance(rmm_pool_initial_bytes, bool)
        or not isinstance(rmm_pool_initial_bytes, int)
        or rmm_pool_initial_bytes <= 0
    ):
        raise ValueError(
            "parquet_range_map_groups_rmm_pool_initial_bytes must be a positive "
            "integer"
        )
    if (
        isinstance(rmm_pool_reserve_bytes, bool)
        or not isinstance(rmm_pool_reserve_bytes, int)
        or rmm_pool_reserve_bytes < 0
    ):
        raise ValueError(
            "parquet_range_map_groups_rmm_pool_reserve_bytes must be a nonnegative "
            "integer"
        )
    if gpu_memory_budget_bytes <= 0:
        raise ValueError("The fused GPU actor requires a positive memory budget.")

    # This dictionary is process-local after Ray deserializes the actor's map
    # transformer. The init hook populates it before execute can run.
    allocator_state: Dict[str, _RmmPoolState] = {}

    def init_fn() -> None:
        allocator_state["rmm"] = _configure_rmm_pool(
            gpu_memory_budget_bytes=gpu_memory_budget_bytes,
            initial_bytes=rmm_pool_initial_bytes,
            reserve_bytes=rmm_pool_reserve_bytes,
        )
        tokenizer_init_fn()
        group_init_fn()

    tokenizer_batch_size = tokenizer_op.batch_size
    tokenizer_zero_copy_batch = tokenizer_op.zero_copy_batch
    group_zero_copy_batch = map_groups_op.zero_copy_batch
    target_max_block_size = data_context.target_max_block_size or 128 * 1024**2

    def execute(
        blocks: Iterable[ParquetRangeMapGroupsWork],
        _: TaskContext,
        report_custom_op_stats: CustomOpStatsReportFn,
    ) -> Iterator[Block]:
        rmm_pool_state = allocator_state.get("rmm")
        if rmm_pool_state is None:
            raise RuntimeError(
                "ParquetRangeMapGroups actor executed before RMM initialization."
            )
        for work in blocks:
            range_outputs = _execute_range(
                work,
                tokenizer_fn=tokenizer_fn,
                group_fn=group_fn,
                tokenizer_batch_size=tokenizer_batch_size,
                tokenizer_zero_copy_batch=tokenizer_zero_copy_batch,
                group_zero_copy_batch=group_zero_copy_batch,
                target_max_block_size=target_max_block_size,
                rmm_pool_config=rmm_pool_state.config,
            )
            # Keep one block behind so completed worker metrics are attached to
            # its metadata. Earlier bounded blocks can flow to the writer while
            # this actor continues processing the range.
            pending_block = None
            while True:
                try:
                    output_block = next(range_outputs)
                except StopIteration as completed:
                    stats = completed.value
                    break
                if pending_block is not None:
                    yield pending_block
                pending_block = output_block
            report_custom_op_stats(stats)
            if pending_block is not None:
                yield pending_block

    map_transformer = MapTransformer(
        [
            BlockMapTransformFn(
                execute,
                disable_block_shaping=True,
                reports_custom_op_stats=True,
            )
        ],
        init_fn=init_fn,
    )
    compute = ActorPoolStrategy(
        size=len(works),
        max_tasks_in_flight_per_actor=1,
        enable_true_multi_threading=False,
    )
    actor_args = dict(ray_remote_args)
    # Never replay a tokenizer or group UDF after it may have produced effects.
    actor_args["max_restarts"] = 0
    actor_args["max_task_retries"] = 0
    actor_start_timeout_s = data_context.get_config(
        "parquet_range_map_groups_actor_start_timeout_s", 300.0
    )
    if (
        isinstance(actor_start_timeout_s, bool)
        or not isinstance(actor_start_timeout_s, (int, float))
        or not math.isfinite(float(actor_start_timeout_s))
        or float(actor_start_timeout_s) <= 0
    ):
        raise ValueError(
            "parquet_range_map_groups_actor_start_timeout_s must be finite and positive"
        )

    metrics = layout.metrics
    plan_metrics = {
        "selected": True,
        "partition_key": partition_key,
        "group_keys": group_keys,
        "num_partitions": len(works),
        "footer_planning_time_s": footer_planning_time_s,
        "scan_amplification": metrics.scan_amplification,
        "row_scan_amplification": metrics.row_scan_amplification,
        "uncompressed_scan_amplification": (metrics.uncompressed_scan_amplification),
        "load_skew": metrics.load_skew,
        "estimated_peak_gpu_memory_bytes": estimated_peak_gpu_memory_bytes,
        "gpu_memory_budget_bytes": gpu_memory_budget_bytes,
        "rmm_pool_initial_bytes": rmm_pool_initial_bytes,
        "rmm_pool_reserve_bytes": rmm_pool_reserve_bytes,
        "ranges": tuple(
            {
                "partition_id": work.partition_id,
                "lower_bound": work.lower_bound,
                "upper_bound": work.upper_bound,
                "estimated_scanned_rows": work.estimated_scanned_rows,
                "estimated_encoded_bytes": work.estimated_encoded_bytes,
                "estimated_uncompressed_bytes": work.estimated_uncompressed_bytes,
            }
            for work in works
        ),
    }
    name = (
        f"ParquetRangeMapGroups({partition_key}, partitions={len(works)}, "
        f"amplification={metrics.scan_amplification:.3f}, "
        f"skew={metrics.load_skew:.3f})"
    )
    return ParquetRangeMapGroupsActorPoolMapOperator(
        map_transformer,
        input_op,
        data_context,
        compute,
        name=name,
        supports_fusion=False,
        ray_remote_args=actor_args,
        plan_metrics=plan_metrics,
        actor_start_timeout_s=float(actor_start_timeout_s),
    )
