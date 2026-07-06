"""Opt-in footer-planned Parquet/cuDF shuffle elision.

This module intentionally recognizes one narrow pipeline. If every contract is
not explicit and verifiable, planning returns ``None`` and ``map_groups`` keeps
its ordinary hash-shuffle implementation.
"""

from __future__ import annotations

import collections
import collections.abc
import functools
import inspect
import logging
import numbers
import os
import pickle
import time
import urllib.parse
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Optional

if TYPE_CHECKING:
    from ray.data._internal.execution.interfaces import PhysicalOperator
    from ray.data._internal.logical.operators import MapGroups
    from ray.data.context import DataContext

logger = logging.getLogger(__name__)

_GIB = 1024**3
_MAX_AMPLIFICATION = 1.25
_MAX_SKEW = 1.5
_MIN_NET_SAVINGS_PER_GPU = 256 * 1024**2
_ROW_GROUPS_PER_READ = 32
_KVIKIO_THREADS = 32
_KVIKIO_TASK_SIZE = 16 * 1024**2


class _PlanningRejected(Exception):
    """A normal reason to retain the stock Ray Data plan."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise _PlanningRejected(reason)


@dataclass(frozen=True)
class _SourceIdentity:
    size: int
    marker: str


@dataclass(frozen=True)
class _RowGroup:
    path: str
    identity: _SourceIdentity
    row_group: int
    num_rows: int
    lower: int
    upper: int
    compressed_bytes: int
    uncompressed_bytes: int


@dataclass(frozen=True)
class _FileWork:
    path: str
    identity: _SourceIdentity
    row_groups: tuple[int, ...]


@dataclass(frozen=True)
class _RangeWork:
    lower: int
    upper: int
    files: tuple[_FileWork, ...]


@dataclass(frozen=True)
class _FooterSummary:
    row_groups: tuple[_RowGroup, ...]
    footer_bytes: int
    elapsed_s: float


@dataclass(frozen=True)
class _RangePlan:
    work: tuple[_RangeWork, ...]
    gpu_budget: int
    amplification: tuple[float, float, float]
    skew: float
    peak_bytes: int
    net_savings: float


@dataclass(frozen=True)
class _Candidate:
    read_spec: Any
    tokenizer_op: Any
    config: Any
    group_keys: tuple[str, ...]
    num_partitions: int
    num_workers: int
    resources: dict[str, Any]
    target_max_block_size: int
    retried_io_errors: tuple[str, ...]


def _is_synchronous(callable_: Any) -> bool:
    if not callable(callable_):
        return False
    if inspect.isroutine(callable_) or isinstance(callable_, functools.partial):
        function = callable_
    else:
        function = callable_.__call__
    return not inspect.iscoroutinefunction(function) and not inspect.isasyncgenfunction(
        function
    )


def _is_plain_argument(value: Any) -> bool:
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return True
    if isinstance(value, (tuple, list)):
        return all(_is_plain_argument(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_plain_argument(item)
            for key, item in value.items()
        )
    return False


def _fixed_actor_pool_size(compute: Any) -> Optional[int]:
    from ray.data._internal.compute import ActorPoolStrategy

    if not isinstance(compute, ActorPoolStrategy):
        return None
    size = compute.min_size
    fixed = (
        isinstance(size, numbers.Integral)
        and not isinstance(size, bool)
        and size > 0
        and size == compute.max_size
        and size == compute.initial_size
        and compute.max_tasks_in_flight_per_actor == 1
        and not compute.enable_true_multi_threading
    )
    return int(size) if fixed else None


def _fixed_task_pool_size(compute: Any) -> Optional[int]:
    from ray.data._internal.compute import TaskPoolStrategy

    size = compute.size if isinstance(compute, TaskPoolStrategy) else None
    if isinstance(size, numbers.Integral) and not isinstance(size, bool) and size > 0:
        return int(size)
    return None


def _select_candidate(op: "MapGroups", data_context: "DataContext") -> _Candidate:
    import pyarrow as pa

    from ray.data._internal.datasource.parquet_datasource import ParquetDatasource
    from ray.data._internal.logical.operators import MapBatches, Read
    from ray.data.context import ShuffleStrategy

    config = op.parquet_cudf_shuffle_elision
    _require(config is not None, "not_configured")
    _require(op.shuffle_strategy == ShuffleStrategy.HASH_SHUFFLE, "not_hash_shuffle")
    for option in (
        "checkpoint_config",
        "retried_map_errors",
        "actor_task_retry_on_errors",
        "actor_init_retry_on_errors",
    ):
        _require(not getattr(data_context, option, None), f"context_{option}")
    _require(data_context.max_errored_blocks == 0, "context_max_errored_blocks")

    group_keys = (op.key,) if isinstance(op.key, str) else tuple(op.key or ())
    _require(
        bool(group_keys)
        and len(group_keys) == len(set(group_keys))
        and all(isinstance(key, str) and key for key in group_keys),
        "group_keys",
    )
    _require(
        op.num_partitions_explicit
        and isinstance(op.num_partitions, numbers.Integral)
        and not isinstance(op.num_partitions, bool)
        and op.num_partitions > 0,
        "partitions",
    )

    _require(len(op.input_dependencies) == 1, "tokenizer_shape")
    tokenizer_op = op.input_dependencies[0]
    _require(type(tokenizer_op) is MapBatches, "tokenizer_shape")
    _require(len(tokenizer_op.input_dependencies) == 1, "read_shape")
    read_op = tokenizer_op.input_dependencies[0]
    _require(type(read_op) is Read, "read_shape")
    datasource = read_op.datasource
    _require(
        type(datasource) is ParquetDatasource
        and read_op.datasource_or_legacy_reader is datasource,
        "datasource",
    )

    read_result = datasource._get_parquet_cudf_direct_read_spec()
    _require(read_result.spec is not None, f"read_{read_result.reason}")
    read_spec = read_result.spec
    _require(all(key in read_spec.projection for key in group_keys), "group_projection")
    _require(group_keys[0] in read_spec.file_schema.names, "range_key")
    _require(
        pa.types.is_integer(read_spec.file_schema.field(group_keys[0]).type),
        "range_key_type",
    )
    for field in read_spec.file_schema:
        if field.name not in read_spec.projection:
            continue
        unsupported = (
            pa.types.is_nested(field.type)
            or pa.types.is_decimal(field.type)
            or pa.types.is_dictionary(field.type)
            or (pa.types.is_timestamp(field.type) and field.type.tz is not None)
            or isinstance(field.type, pa.ExtensionType)
        )
        _require(not unsupported, "projection_type")

    _require(read_op.per_block_limit is None, "read_limit")
    allowed_read_args = {"scheduling_strategy", "label_selector"}
    _require(set(read_op.ray_remote_args) <= allowed_read_args, "read_resources")

    tokenizer_options_ok = (
        inspect.isclass(tokenizer_op.fn)
        and _is_synchronous(tokenizer_op.fn)
        and tokenizer_op.batch_format == "cudf"
        and tokenizer_op.zero_copy_batch
        and not tokenizer_op.can_modify_num_rows
        and isinstance(tokenizer_op.batch_size, numbers.Integral)
        and not isinstance(tokenizer_op.batch_size, bool)
        and tokenizer_op.batch_size > 0
        and tokenizer_op.per_block_limit is None
        and tokenizer_op.ray_remote_args_fn is None
    )
    _require(tokenizer_options_ok, "tokenizer_options")
    _require(
        all(
            _is_plain_argument(value)
            for value in (
                tokenizer_op.fn_args,
                tokenizer_op.fn_kwargs,
                tokenizer_op.fn_constructor_args,
                tokenizer_op.fn_constructor_kwargs,
                op.fn_args,
                op.fn_kwargs,
            )
        ),
        "udf_arguments",
    )
    workers = _fixed_actor_pool_size(tokenizer_op.compute)
    _require(workers is not None and workers <= op.num_partitions, "tokenizer_pool")

    group_options_ok = (
        _is_synchronous(op.fn)
        and op.batch_format == "cudf"
        and op.zero_copy_batch
        and op.fn_constructor_args is None
        and op.fn_constructor_kwargs is None
        and op.ray_remote_args_fn is None
        and not inspect.isclass(config.partition_fn)
        and _is_synchronous(config.partition_fn)
    )
    _require(group_options_ok, "group_options")
    _require(_fixed_task_pool_size(op.compute) == workers, "group_pool")

    tokenizer_resources = dict(tokenizer_op.ray_remote_args)
    group_resources = dict(op.ray_remote_args)
    _require(tokenizer_resources == group_resources, "worker_resources")
    retry_options = {
        "max_restarts",
        "max_task_retries",
        "max_retries",
        "retry_exceptions",
    }
    _require(not retry_options.intersection(group_resources), "worker_retries")
    _require(float(group_resources.get("num_gpus", 0)) == 1.0, "one_gpu")
    _require(group_resources.get("max_concurrency", 1) in (None, 1), "concurrency")

    resources = dict(group_resources)
    for option in ("scheduling_strategy", "label_selector"):
        if option not in read_op.ray_remote_args:
            continue
        _require(
            option not in resources
            or resources[option] == read_op.ray_remote_args[option],
            f"read_{option}_conflict",
        )
        resources[option] = read_op.ray_remote_args[option]

    target_bytes = data_context.target_max_block_size
    _require(
        isinstance(target_bytes, numbers.Integral)
        and not isinstance(target_bytes, bool)
        and target_bytes > 0,
        "target_block_size",
    )
    retried_io_errors = tuple(data_context.retried_io_errors)
    _require(
        all(isinstance(pattern, str) for pattern in retried_io_errors),
        "context_retried_io_errors",
    )
    if read_spec.source_kind == "s3":
        _require(
            not os.environ.get("AWS_ENDPOINT_URL")
            and not os.environ.get("AWS_ENDPOINT_URL_S3"),
            "s3_endpoint_override",
        )

    return _Candidate(
        read_spec=read_spec,
        tokenizer_op=tokenizer_op,
        config=config,
        group_keys=group_keys,
        num_partitions=int(op.num_partitions),
        num_workers=workers,
        resources=resources,
        target_max_block_size=int(target_bytes),
        retried_io_errors=retried_io_errors,
    )


def _split_s3_path(path: str) -> tuple[str, str]:
    bucket, separator, key = path.partition("/")
    if not separator or not bucket or not key:
        raise _PlanningRejected("s3_path")
    return bucket, key


def _new_s3_client(region: str) -> Any:
    if os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL_S3"):
        raise RuntimeError("Native S3 execution does not support endpoint overrides")
    import botocore.session

    session = botocore.session.get_session()
    if session.get_credentials() is None:
        raise RuntimeError("No ambient AWS credentials are available")
    return session.create_client("s3", region_name=region)


def _source_identity(
    read_spec: Any, path: str, s3_client: Any = None
) -> _SourceIdentity:
    if read_spec.source_kind == "local":
        import pyarrow.fs as pafs

        info = read_spec.filesystem.get_file_info(path)
        _require(info.type == pafs.FileType.File, "source_unavailable")
        _require(info.size is not None and info.size >= 0, "source_identity")
        _require(info.mtime_ns is not None, "source_identity")
        return _SourceIdentity(int(info.size), f"mtime_ns:{int(info.mtime_ns)}")

    if s3_client is None:
        s3_client = _new_s3_client(read_spec.region)
    bucket, key = _split_s3_path(path)
    response = s3_client.head_object(Bucket=bucket, Key=key)
    size = response.get("ContentLength")
    _require(isinstance(size, numbers.Integral) and size >= 0, "source_identity")
    version = response.get("VersionId")
    etag = response.get("ETag")
    modified = response.get("LastModified")
    if version and version != "null":
        marker = f"version:{version}"
    else:
        _require(bool(etag) and modified is not None, "source_identity")
        timestamp = (
            modified.isoformat() if hasattr(modified, "isoformat") else str(modified)
        )
        marker = f"etag:{etag};last_modified:{timestamp}"
    return _SourceIdentity(int(size), marker)


def _read_footer_summary(read_spec: Any, range_key: str) -> _FooterSummary:
    import pyarrow.parquet as pq

    started = time.perf_counter()
    rows: list[_RowGroup] = []
    footer_bytes = 0
    s3_client = (
        _new_s3_client(read_spec.region) if read_spec.source_kind == "s3" else None
    )

    for path, listed_size in zip(read_spec.paths, read_spec.listed_file_sizes):
        identity = _source_identity(read_spec, path, s3_client)
        _require(identity.size == listed_size, "listed_size_changed")
        with read_spec.filesystem.open_input_file(path) as source:
            metadata = pq.ParquetFile(source).metadata
        _require(
            _source_identity(read_spec, path, s3_client) == identity,
            "source_changed",
        )
        footer_bytes += int(getattr(metadata, "serialized_size", 0) or 0)

        column_indices: dict[str, list[int]] = {}
        for index in range(metadata.num_columns):
            name = metadata.schema.column(index).path.split(".", 1)[0]
            column_indices.setdefault(name, []).append(index)
        _require(
            all(column in column_indices for column in read_spec.projection),
            "missing_column",
        )
        _require(len(column_indices.get(range_key, ())) == 1, "nested_range_key")
        range_key_index = column_indices[range_key][0]

        for row_group_id in range(metadata.num_row_groups):
            row_group = metadata.row_group(row_group_id)
            if row_group.num_rows == 0:
                continue
            statistics = row_group.column(range_key_index).statistics
            statistics_ok = (
                statistics is not None
                and statistics.has_min_max
                and statistics.min is not None
                and statistics.max is not None
                and statistics.null_count == 0
            )
            _require(statistics_ok, "key_statistics")
            lower, upper = statistics.min, statistics.max
            _require(
                isinstance(lower, numbers.Integral)
                and not isinstance(lower, bool)
                and isinstance(upper, numbers.Integral)
                and not isinstance(upper, bool)
                and lower <= upper,
                "key_statistics",
            )

            compressed = 0
            uncompressed = 0
            for column in read_spec.projection:
                for index in column_indices[column]:
                    chunk = row_group.column(index)
                    compressed_size = chunk.total_compressed_size
                    uncompressed_size = chunk.total_uncompressed_size
                    _require(
                        compressed_size is not None
                        and compressed_size >= 0
                        and uncompressed_size is not None
                        and uncompressed_size >= 0,
                        "size_statistics",
                    )
                    compressed += int(compressed_size)
                    uncompressed += int(uncompressed_size)
            _require(compressed > 0 and uncompressed > 0, "size_statistics")
            rows.append(
                _RowGroup(
                    path=path,
                    identity=identity,
                    row_group=row_group_id,
                    num_rows=int(row_group.num_rows),
                    lower=int(lower),
                    upper=int(upper),
                    compressed_bytes=compressed,
                    uncompressed_bytes=uncompressed,
                )
            )

    _require(bool(rows), "empty_source")
    rows.sort(key=lambda row: (row.lower, row.upper, row.path, row.row_group))
    return _FooterSummary(tuple(rows), footer_bytes, time.perf_counter() - started)


def _plan_ranges(
    rows: tuple[_RowGroup, ...],
    *,
    num_partitions: int,
    num_workers: int,
    shuffle_bytes_per_row: float,
    peak_gpu_bytes_per_row: float,
    gpu_memory_bytes: int,
) -> _RangePlan:
    total_rows = sum(row.num_rows for row in rows)
    total_compressed = sum(row.compressed_bytes for row in rows)
    total_uncompressed = sum(row.uncompressed_bytes for row in rows)
    budget = min(int(gpu_memory_bytes * 0.70), gpu_memory_bytes - 2 * _GIB)
    _require(budget > 0, "gpu_budget")

    domain_lower = min(row.lower for row in rows)
    domain_upper = max(row.upper for row in rows)
    domain_size = domain_upper - domain_lower + 1
    for count in (num_partitions, num_partitions * 2, num_partitions * 4):
        if count > domain_size:
            continue
        ranges: list[tuple[int, int, tuple[_RowGroup, ...]]] = []
        loads: list[tuple[int, int, int]] = []
        for index in range(count):
            lower = domain_lower + domain_size * index // count
            upper = domain_lower + domain_size * (index + 1) // count - 1
            selected = tuple(
                row for row in rows if row.upper >= lower and row.lower <= upper
            )
            load = (
                sum(row.num_rows for row in selected),
                sum(row.compressed_bytes for row in selected),
                sum(row.uncompressed_bytes for row in selected),
            )
            ranges.append((lower, upper, selected))
            loads.append(load)
        if any(load[0] == 0 for load in loads):
            continue

        duplicated_totals = tuple(sum(load[i] for load in loads) for i in range(3))
        source_totals = (total_rows, total_compressed, total_uncompressed)
        amplification = tuple(duplicated_totals[i] / source_totals[i] for i in range(3))
        skew = max(
            max(load[i] for load in loads) / (duplicated_totals[i] / count)
            for i in range(3)
        )
        peak_bytes = max(
            max(int(load[0] * peak_gpu_bytes_per_row), load[2]) for load in loads
        )
        duplicated_read_bytes = duplicated_totals[1] - total_compressed
        net_savings = total_rows * shuffle_bytes_per_row - duplicated_read_bytes
        thresholds_ok = (
            all(value <= _MAX_AMPLIFICATION for value in amplification)
            and skew <= _MAX_SKEW
            and peak_bytes <= budget
            and net_savings >= num_workers * _MIN_NET_SAVINGS_PER_GPU
        )
        if not thresholds_ok:
            continue

        work: list[_RangeWork] = []
        for lower, upper, selected in ranges:
            files: dict[tuple[str, _SourceIdentity], list[int]] = {}
            for row in selected:
                files.setdefault((row.path, row.identity), []).append(row.row_group)
            file_work = tuple(
                _FileWork(path, identity, tuple(sorted(row_groups)))
                for (path, identity), row_groups in sorted(
                    files.items(), key=lambda item: item[0][0]
                )
            )
            work.append(_RangeWork(lower, upper, file_work))
        return _RangePlan(
            work=tuple(work),
            gpu_budget=budget,
            amplification=amplification,
            skew=skew,
            peak_bytes=peak_bytes,
            net_savings=net_savings,
        )

    raise _PlanningRejected("range_thresholds")


def _refresh_aws_credentials(
    region: str, session: Any = None, credentials: Any = None
) -> tuple[Any, Any]:
    if os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL_S3"):
        raise RuntimeError("Native S3 execution does not support endpoint overrides")
    if session is None:
        import botocore.session

        session = botocore.session.get_session()
    if credentials is None:
        credentials = session.get_credentials()
    if credentials is None:
        raise RuntimeError("No ambient AWS credentials are available")
    frozen = credentials.get_frozen_credentials()
    if not frozen.access_key or not frozen.secret_key:
        raise RuntimeError("The ambient AWS credential provider is incomplete")

    os.environ["AWS_ACCESS_KEY_ID"] = frozen.access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = frozen.secret_key
    if frozen.token:
        os.environ["AWS_SESSION_TOKEN"] = frozen.token
    else:
        os.environ.pop("AWS_SESSION_TOKEN", None)
    os.environ["AWS_DEFAULT_REGION"] = region
    os.environ["AWS_REGION"] = region
    return session, credentials


def _configure_kvikio(region: str, session: Any, credentials: Any) -> None:
    _refresh_aws_credentials(region, session, credentials)
    os.environ["KVIKIO_NTHREADS"] = str(_KVIKIO_THREADS)
    os.environ["KVIKIO_TASK_SIZE"] = str(_KVIKIO_TASK_SIZE)

    import kvikio.defaults as defaults
    from kvikio.remote_file import is_remote_file_available

    if not is_remote_file_available():
        raise RuntimeError("KvikIO remote I/O is unavailable")
    defaults.set({"num_threads": _KVIKIO_THREADS, "task_size": _KVIKIO_TASK_SIZE})
    applied = (int(defaults.get("num_threads")), int(defaults.get("task_size")))
    if applied != (_KVIKIO_THREADS, _KVIKIO_TASK_SIZE):
        raise RuntimeError("KvikIO rejected its actor-local configuration")


def _s3_uri(path: str) -> str:
    return "s3://" + urllib.parse.quote(path, safe="/")


def _rebatched_frames(
    frames: Iterable[Any], batch_size: int, cudf: Any
) -> Iterator[Any]:
    """Yield exact-size batches without repeatedly concatenating a growing tail."""

    pending: collections.deque[list[Any]] = collections.deque()
    available = 0

    def take(rows: int) -> Any:
        nonlocal available
        pieces = []
        remaining = rows
        while remaining:
            frame, offset = pending[0]
            count = min(remaining, len(frame) - offset)
            pieces.append(frame.iloc[offset : offset + count])
            offset += count
            remaining -= count
            available -= count
            if offset == len(frame):
                pending.popleft()
            else:
                pending[0][1] = offset
        if len(pieces) == 1:
            return pieces[0].reset_index(drop=True)
        return cudf.concat(pieces, ignore_index=True)

    for frame in frames:
        if len(frame) == 0:
            continue
        pending.append([frame, 0])
        available += len(frame)
        while available >= batch_size:
            yield take(batch_size)
    if available:
        yield take(available)


def _is_sorted(frame: Any, keys: tuple[str, ...]) -> bool:
    if len(frame) < 2:
        return True
    try:
        import pylibcudf as plc

        table, _ = frame.loc[:, list(keys)].to_pylibcudf(copy=False)
        return bool(
            plc.sorting.is_sorted(
                table,
                [plc.types.Order.ASCENDING] * len(keys),
                [plc.types.NullOrder.AFTER] * len(keys),
            )
        )
    except (AttributeError, ImportError, TypeError):
        return False


def _group_boundaries(frame: Any, keys: tuple[str, ...], cupy: Any) -> tuple[int, ...]:
    if len(frame) == 0:
        return (0,)
    changed = cupy.zeros(len(frame), dtype=cupy.bool_)
    changed[0] = True
    for key in keys:
        values = frame[key]
        previous = values.shift(1)
        missing = values.isnull()
        previous_missing = previous.isnull()
        changed |= ((values != previous) & ~(missing & previous_missing)).fillna(
            False
        ).to_cupy() | (missing != previous_missing).to_cupy()
    return tuple(int(index) for index in cupy.flatnonzero(changed).get()) + (
        len(frame),
    )


def _split_output(batch: Any, target_bytes: int) -> Iterator[Any]:
    if isinstance(batch, dict):
        values = tuple(batch.values())
        rows = len(values[0]) if values else 0
        size = sum(int(getattr(value, "nbytes", 0)) for value in values)

        def take(start: int, end: int) -> Any:
            return {key: value[start:end] for key, value in batch.items()}

    else:
        rows = len(batch)
        memory_usage = getattr(batch, "memory_usage", None)
        size = (
            int(memory_usage(deep=True).sum())
            if memory_usage is not None
            else int(getattr(batch, "nbytes", 0))
        )

        def take(start: int, end: int) -> Any:
            if hasattr(batch, "iloc"):
                return batch.iloc[start:end]
            return batch.slice(start, end - start)

    if rows == 0:
        yield batch
        return
    rows_per_output = max(1, int(rows * target_bytes / size)) if size else rows
    for start in range(0, rows, rows_per_output):
        yield take(start, min(rows, start + rows_per_output))


def _one_or_generator(value: Any) -> Iterator[Any]:
    return value if isinstance(value, collections.abc.Iterator) else iter((value,))


class _ParquetCudfShuffleElisionWorker:
    """Reusable actor that reads, tokenizes, and applies one range partition."""

    def __init__(self, *, spec: dict[str, Any]):
        self.__dict__.update(spec)
        self._aws_session = None
        self._aws_credentials = None
        self._s3_client = None
        if self.source_kind == "s3":
            self._aws_session, self._aws_credentials = _refresh_aws_credentials(
                self.region
            )
            self._s3_client = self._aws_session.create_client(
                "s3", region_name=self.region
            )

        import rmm

        free_memory, _ = rmm.mr.available_device_memory()
        maximum = min(self.gpu_budget, max(0, int(free_memory) - 2 * _GIB))
        maximum = maximum // 256 * 256
        if maximum <= 0:
            raise RuntimeError("Insufficient live GPU memory for shuffle elision")
        self._rmm_upstream = rmm.mr.get_current_device_resource()
        self._rmm_pool = rmm.mr.PoolMemoryResource(
            self._rmm_upstream,
            initial_pool_size=min(_GIB, maximum),
            maximum_pool_size=maximum,
        )
        rmm.mr.set_current_device_resource(self._rmm_pool)

        import cupy
        from rmm.allocators.cupy import rmm_cupy_allocator

        cupy.cuda.set_allocator(rmm_cupy_allocator)
        if self.source_kind == "s3":
            _configure_kvikio(self.region, self._aws_session, self._aws_credentials)
        import cudf

        if self.source_kind == "s3":
            cudf.set_option("kvikio_remote_io", True)
            if not cudf.get_option("kvikio_remote_io"):
                raise RuntimeError("cuDF rejected KvikIO remote I/O")
        self.cudf = cudf
        self.cupy = cupy
        self.tokenizer = self.tokenizer_cls(
            *(self.tokenizer_constructor_args or ()),
            **(self.tokenizer_constructor_kwargs or {}),
        )
        self.tokenizer_args = tuple(self.tokenizer_args or ())
        self.tokenizer_kwargs = dict(self.tokenizer_kwargs or {})
        self.group_args = tuple(self.group_args or ())
        self.group_kwargs = dict(self.group_kwargs or {})

    def _identity(self, path: str) -> _SourceIdentity:
        if self.source_kind == "local":
            import pyarrow.fs as pafs

            info = pafs.LocalFileSystem().get_file_info(path)
            if (
                info.type != pafs.FileType.File
                or info.size is None
                or info.mtime_ns is None
            ):
                raise RuntimeError(f"Parquet source is unavailable: {path!r}")
            return _SourceIdentity(int(info.size), f"mtime_ns:{int(info.mtime_ns)}")
        bucket, key = _split_s3_path(path)
        response = self._s3_client.head_object(Bucket=bucket, Key=key)
        size = response["ContentLength"]
        version = response.get("VersionId")
        if version and version != "null":
            marker = f"version:{version}"
        else:
            modified = response.get("LastModified")
            timestamp = (
                modified.isoformat()
                if hasattr(modified, "isoformat")
                else str(modified)
            )
            marker = f"etag:{response.get('ETag')};last_modified:{timestamp}"
        return _SourceIdentity(int(size), marker)

    def _verify_identity(self, file: _FileWork) -> None:
        if self._identity(file.path) != file.identity:
            raise RuntimeError(f"Parquet source changed after planning: {file.path!r}")

    def _read_frames(self, work: _RangeWork) -> Iterator[Any]:
        from ray.data._internal.util import iterate_with_retry

        for file in work.files:
            path = _s3_uri(file.path) if self.source_kind == "s3" else file.path
            for start in range(0, len(file.row_groups), _ROW_GROUPS_PER_READ):
                kwargs = {
                    "columns": list(self.projection),
                    "row_groups": list(
                        file.row_groups[start : start + _ROW_GROUPS_PER_READ]
                    ),
                }
                if self.source_kind == "s3":
                    if not self.cudf.get_option("kvikio_remote_io"):
                        raise RuntimeError("cuDF disabled KvikIO remote I/O")
                    kwargs.update(
                        engine="cudf",
                        dataset_kwargs={"partitioning": None},
                        use_pandas_metadata=False,
                        categorical_partitions=False,
                    )

                def read_chunk():
                    self._verify_identity(file)
                    frame = self.cudf.read_parquet(path, **kwargs)
                    self._verify_identity(file)
                    return (frame,)

                if self.source_kind == "s3":
                    frame = tuple(
                        iterate_with_retry(
                            read_chunk,
                            "read a Parquet row-group chunk with KvikIO",
                            match=list(self.retried_io_errors),
                        )
                    )[0]
                else:
                    frame = next(iter(read_chunk()))
                frame = frame[
                    (frame[self.range_key] >= work.lower)
                    & (frame[self.range_key] <= work.upper)
                ].reset_index(drop=True)
                if len(frame):
                    yield frame

    def __call__(self, batch: dict[str, Any]) -> Iterator[Any]:
        if self.source_kind == "s3":
            _refresh_aws_credentials(
                self.region, self._aws_session, self._aws_credentials
            )
            self._s3_client.close()
            self._s3_client = self._aws_session.create_client(
                "s3", region_name=self.region
            )
        work = pickle.loads(bytes(batch["work"][0]))
        tokenized_frames = []
        for raw in _rebatched_frames(
            self._read_frames(work), self.tokenizer_batch_size, self.cudf
        ):
            input_key = raw[self.range_key].reset_index(drop=True).copy(deep=True)
            result = self.tokenizer(raw, *self.tokenizer_args, **self.tokenizer_kwargs)
            outputs = list(_one_or_generator(result))
            if not outputs or any(
                not isinstance(output, self.cudf.DataFrame) for output in outputs
            ):
                raise TypeError("The tokenizer must return cuDF DataFrames")
            output = (
                outputs[0]
                if len(outputs) == 1
                else self.cudf.concat(outputs, ignore_index=True)
            ).reset_index(drop=True)
            key_preserved = (
                len(output) == len(raw)
                and self.range_key in output
                and bool(
                    (input_key == output[self.range_key].reset_index(drop=True))
                    .fillna(False)
                    .all()
                )
            )
            if not key_preserved:
                raise ValueError(
                    "The tokenizer violated its row/key preservation contract"
                )
            tokenized_frames.append(output)
            del raw, input_key, result, outputs, output

        if not tokenized_frames:
            return
        frame = (
            tokenized_frames[0]
            if len(tokenized_frames) == 1
            else self.cudf.concat(tokenized_frames, ignore_index=True)
        )
        tokenized_frames.clear()
        if any(key not in frame for key in self.group_keys):
            raise ValueError("The tokenizer removed a group key")
        if not _is_sorted(frame, self.group_keys):
            frame = frame.sort_values(
                list(self.group_keys), ignore_index=True, na_position="last"
            )
        boundaries = _group_boundaries(frame, self.group_keys, self.cupy)

        from ray.data.grouped_data import MapGroupsPartitionContext

        context = MapGroupsPartitionContext(
            group_keys=self.group_keys,
            input_group_boundaries=boundaries,
        )
        result = self.partition_fn(
            frame, context, *self.group_args, **self.group_kwargs
        )
        for output in _one_or_generator(result):
            yield from _split_output(output, self.target_bytes)


def _descriptor_bundles(work: tuple[_RangeWork, ...]) -> list[Any]:
    import pyarrow as pa
    import ray
    from ray.data._internal.execution.interfaces import BlockEntry, RefBundle
    from ray.data.block import BlockAccessor

    bundles = []
    for descriptor in work:
        block = pa.table({"work": [pickle.dumps(descriptor, protocol=5)]})
        metadata = BlockAccessor.for_block(block).get_metadata()
        bundles.append(
            RefBundle(
                blocks=(BlockEntry(ray.put(block), metadata),),
                schema=block.schema,
                owns_blocks=True,
            )
        )
    return bundles


def _build_physical_operator(
    op: "MapGroups",
    data_context: "DataContext",
    candidate: _Candidate,
    footer: _FooterSummary,
    plan: _RangePlan,
) -> "PhysicalOperator":
    from ray.data._internal.compute import ActorPoolStrategy
    from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
    from ray.data._internal.execution.operators.map_operator import MapOperator
    from ray.data._internal.execution.operators.map_transformer import (
        BatchMapTransformFn,
        MapTransformer,
    )
    from ray.data._internal.output_buffer import OutputBlockSizeOption
    from ray.data._internal.planner.plan_udf_map_op import (
        _generate_transform_fn_for_map_batches,
        _get_udf,
    )

    tokenizer_op = candidate.tokenizer_op
    worker_spec = {
        "source_kind": candidate.read_spec.source_kind,
        "region": candidate.read_spec.region,
        "projection": candidate.read_spec.projection,
        "range_key": candidate.group_keys[0],
        "group_keys": candidate.group_keys,
        "tokenizer_cls": tokenizer_op.fn,
        "tokenizer_args": tokenizer_op.fn_args,
        "tokenizer_kwargs": tokenizer_op.fn_kwargs,
        "tokenizer_constructor_args": tokenizer_op.fn_constructor_args,
        "tokenizer_constructor_kwargs": tokenizer_op.fn_constructor_kwargs,
        "tokenizer_batch_size": int(tokenizer_op.batch_size),
        "partition_fn": candidate.config.partition_fn,
        "group_args": op.fn_args,
        "group_kwargs": op.fn_kwargs,
        "target_bytes": candidate.target_max_block_size,
        "gpu_budget": plan.gpu_budget,
        "retried_io_errors": candidate.retried_io_errors,
    }
    compute = ActorPoolStrategy(
        size=candidate.num_workers, max_tasks_in_flight_per_actor=1
    )
    worker_fn, init_fn = _get_udf(
        _ParquetCudfShuffleElisionWorker,
        (),
        {},
        (),
        {"spec": worker_spec},
        compute,
    )
    transform = BatchMapTransformFn(
        _generate_transform_fn_for_map_batches(worker_fn),
        batch_size=1,
        batch_format="numpy",
        zero_copy_batch=True,
        is_udf=True,
        output_block_size_option=OutputBlockSizeOption.of(
            target_max_block_size=candidate.target_max_block_size
        ),
    )
    transformer = MapTransformer([transform], init_fn=init_fn)
    input_op = InputDataBuffer(
        data_context,
        input_data_factory=lambda _target_size: _descriptor_bundles(plan.work),
    )
    resources = dict(candidate.resources)
    resources.update(max_restarts=0, max_task_retries=0)
    name = (
        "ParquetCudfShuffleElision["
        f"backend={candidate.read_spec.source_kind},ranges={len(plan.work)},"
        f"amplification={max(plan.amplification):.3f},skew={plan.skew:.3f}]"
    )
    logger.info(
        "Selected %s: footer_bytes=%d footer_time_s=%.3f peak_bytes=%d "
        "gpu_budget=%d net_savings_bytes=%d actors=%d probe_bytes=0",
        name,
        footer.footer_bytes,
        footer.elapsed_s,
        plan.peak_bytes,
        plan.gpu_budget,
        int(plan.net_savings),
        candidate.num_workers,
    )
    return MapOperator.create(
        transformer,
        input_op,
        data_context,
        name=name,
        compute_strategy=compute,
        supports_fusion=False,
        ray_remote_args=resources,
    )


def try_plan_parquet_cudf_shuffle_elision(
    op: "MapGroups", data_context: "DataContext"
) -> Optional["PhysicalOperator"]:
    """Return the selected physical operator, or ``None`` for stock lowering."""

    try:
        candidate = _select_candidate(op, data_context)
        footer = _read_footer_summary(candidate.read_spec, candidate.group_keys[0])
        plan = _plan_ranges(
            footer.row_groups,
            num_partitions=candidate.num_partitions,
            num_workers=candidate.num_workers,
            shuffle_bytes_per_row=float(candidate.config.shuffle_bytes_per_input_row),
            peak_gpu_bytes_per_row=float(candidate.config.peak_gpu_bytes_per_input_row),
            gpu_memory_bytes=int(candidate.config.gpu_memory_bytes),
        )
        return _build_physical_operator(op, data_context, candidate, footer, plan)
    except _PlanningRejected as error:
        logger.info(
            "ParquetCudfShuffleElision fallback: reason=%s probe_bytes=0", error
        )
        return None
    except Exception:
        logger.warning(
            "ParquetCudfShuffleElision planning failed; using the stock plan",
            exc_info=True,
        )
        return None
