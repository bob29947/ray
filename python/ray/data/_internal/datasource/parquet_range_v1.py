"""V1 local-Parquet adapter for the source-neutral range planner.

This module only adapts state already owned by ``ParquetDatasource``.  It never
resolves, expands, or relists input paths.  PyArrow and filesystem-specific
imports stay lazy so importing the pure planner does not acquire those
dependencies.
"""

from __future__ import annotations

import numbers
import os
import stat
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from ray.data._internal.datasource.parquet_range import (
    ParquetColumnSize,
    ParquetRangePlanningError,
    ParquetRowGroupMetadata,
)


_POSIX_IDENTITY_VERSION = "posix-v1"


class ParquetSourceIdentityError(RuntimeError):
    """A source could not be verified against its planning-time identity."""

    def __init__(self, message: str, *, reason_code: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True)
class PosixSourceIdentity:
    """POSIX fields that identify the exact file observed during planning."""

    device: int
    inode: int
    size: int
    mtime_ns: int

    def encode(self) -> str:
        return ":".join(
            (
                _POSIX_IDENTITY_VERSION,
                str(self.device),
                str(self.inode),
                str(self.size),
                str(self.mtime_ns),
            )
        )

    @classmethod
    def decode(cls, value: str) -> "PosixSourceIdentity":
        if not isinstance(value, str):
            raise ParquetSourceIdentityError(
                "A POSIX source identity must be a string.",
                reason_code="invalid_source_identity",
            )
        parts = value.split(":")
        if len(parts) != 5 or parts[0] != _POSIX_IDENTITY_VERSION:
            raise ParquetSourceIdentityError(
                "The POSIX source identity has an unsupported format.",
                reason_code="invalid_source_identity",
            )
        try:
            fields = tuple(int(part) for part in parts[1:])
        except ValueError as exc:
            raise ParquetSourceIdentityError(
                "The POSIX source identity contains a non-integral field.",
                reason_code="invalid_source_identity",
            ) from exc
        # POSIX permits timestamps before the Unix epoch.  Device, inode, and
        # size are the fields that must be non-negative.
        if any(field < 0 for field in fields[:3]):
            raise ParquetSourceIdentityError(
                "The POSIX source identity contains a negative field.",
                reason_code="invalid_source_identity",
            )
        return cls(*fields)


@dataclass(frozen=True)
class ParquetV1MetadataResult:
    """Serializable fail-closed result of adapting V1 fragment metadata."""

    row_groups: Tuple[ParquetRowGroupMetadata, ...]
    projection: Optional[Tuple[str, ...]]
    elapsed_s: float
    rejection_reason: Optional[str]
    rejection_message: Optional[str]
    exception_type: Optional[str]

    @property
    def extracted(self) -> bool:
        return self.rejection_reason is None


def _planning_error(message: str, reason_code: str) -> ParquetRangePlanningError:
    return ParquetRangePlanningError(message, reason_code=reason_code)


def _as_posix_path(path: Any) -> str:
    try:
        value = os.fspath(path)
    except TypeError as exc:
        raise ParquetSourceIdentityError(
            "A local Parquet source path must be path-like.",
            reason_code="invalid_source_path",
        ) from exc
    if not isinstance(value, str) or not value or not os.path.isabs(value):
        raise ParquetSourceIdentityError(
            "A local Parquet source path must be an absolute string path.",
            reason_code="invalid_source_path",
        )
    return value


def _stat_posix_source(path: Any) -> PosixSourceIdentity:
    source_path = _as_posix_path(path)
    try:
        source_stat = os.stat(source_path)
    except OSError as exc:
        raise ParquetSourceIdentityError(
            f"Unable to stat local Parquet source {source_path!r}: {exc}",
            reason_code="source_unavailable",
        ) from exc
    if not stat.S_ISREG(source_stat.st_mode):
        raise ParquetSourceIdentityError(
            f"Local Parquet source {source_path!r} is not a regular file.",
            reason_code="unsupported_source_type",
        )
    return PosixSourceIdentity(
        device=int(source_stat.st_dev),
        inode=int(source_stat.st_ino),
        size=int(source_stat.st_size),
        mtime_ns=int(source_stat.st_mtime_ns),
    )


def capture_posix_source_identity(path: Any) -> str:
    """Capture a compact worker-serializable identity for a local file."""

    return _stat_posix_source(path).encode()


def verify_posix_source_identity(path: Any, expected: str) -> bool:
    """Verify a source immediately before reading it on a worker.

    Returns ``True`` on a match.  Missing, replaced, resized, or modified files
    raise ``ParquetSourceIdentityError`` so execution fails once instead of
    rerunning a side-effecting UDF through the normal shuffle path.
    """

    expected_identity = PosixSourceIdentity.decode(expected)
    current_identity = _stat_posix_source(path)
    if current_identity != expected_identity:
        raise ParquetSourceIdentityError(
            f"Local Parquet source {os.fspath(path)!r} changed after planning.",
            reason_code="source_changed",
        )
    return True


def _capture_for_planning(path: Any) -> PosixSourceIdentity:
    try:
        return _stat_posix_source(path)
    except ParquetSourceIdentityError as exc:
        raise _planning_error(str(exc), exc.reason_code) from exc


def _validate_local_v1_state(datasource: Any) -> Tuple[Any, ...]:
    try:
        fragments = tuple(datasource._pq_fragments)
        filesystem = datasource._filesystem
    except (AttributeError, TypeError) as exc:
        raise _planning_error(
            "Expected initialized ParquetDatasource V1 state.",
            "invalid_datasource_state",
        ) from exc

    import pyarrow.fs as pa_fs

    if not isinstance(filesystem, pa_fs.LocalFileSystem):
        raise _planning_error(
            "Metadata range planning currently requires a local POSIX filesystem.",
            "unsupported_filesystem",
        )
    if not fragments:
        raise _planning_error(
            "The ParquetDatasource contains no resolved fragments.",
            "empty_dataset",
        )

    paths = getattr(datasource, "_pq_paths", None)
    if paths is not None:
        try:
            expected_paths = tuple(paths)
        except TypeError as exc:
            raise _planning_error(
                "ParquetDatasource paths are not a sequence.",
                "invalid_datasource_state",
            ) from exc
        if len(expected_paths) != len(fragments):
            raise _planning_error(
                "ParquetDatasource paths and fragments are out of sync.",
                "fragment_state_mismatch",
            )
    return fragments


def _normalize_key(key: str) -> str:
    if not isinstance(key, str) or not key or "." in key:
        raise _planning_error(
            "The range key must be a non-empty top-level column name.",
            "invalid_key_column",
        )
    return key


def _normalize_projection(
    datasource: Any,
    projection: Optional[Sequence[str]],
    *,
    key: str,
) -> Optional[Tuple[str, ...]]:
    if projection is None:
        return None
    if isinstance(projection, (str, bytes)):
        raise _planning_error(
            "projection must be a sequence of column names.",
            "invalid_projection",
        )
    try:
        columns = tuple(projection)
    except TypeError as exc:
        raise _planning_error(
            "projection must be a sequence of column names.",
            "invalid_projection",
        ) from exc
    if any(not isinstance(column, str) or not column for column in columns):
        raise _planning_error(
            "projection must contain non-empty column names.",
            "invalid_projection",
        )

    non_file_columns = set(getattr(datasource, "_partition_columns", ()))
    if getattr(datasource, "_include_paths", False):
        non_file_columns.add("path")
    if getattr(datasource, "_include_row_hash", False):
        non_file_columns.add("row_hash")
    physical_columns = {column for column in columns if column not in non_file_columns}
    physical_columns.add(key)
    return tuple(sorted(physical_columns))


def _top_level_column(path: str) -> str:
    return path.split(".", 1)[0]


def _column_indices_by_name(metadata: Any) -> Dict[str, Tuple[int, ...]]:
    indices: Dict[str, list] = {}
    for column_index in range(metadata.num_columns):
        column_path = metadata.schema.column(column_index).path
        top_level = _top_level_column(column_path)
        indices.setdefault(top_level, []).append(column_index)
    return {name: tuple(value) for name, value in indices.items()}


def _selected_row_group_ids(fragment: Any, metadata: Any) -> Tuple[int, ...]:
    row_groups = getattr(fragment, "row_groups", None)
    if row_groups is None:
        return tuple(range(metadata.num_row_groups))
    try:
        row_group_ids = tuple(int(row_group.id) for row_group in row_groups)
    except (AttributeError, TypeError, ValueError) as exc:
        raise _planning_error(
            f"Fragment {fragment.path!r} has invalid row-group state.",
            "invalid_fragment_state",
        ) from exc
    if (
        len(set(row_group_ids)) != len(row_group_ids)
        or any(
            row_group_id < 0 or row_group_id >= metadata.num_row_groups
            for row_group_id in row_group_ids
        )
    ):
        raise _planning_error(
            f"Fragment {fragment.path!r} has invalid row-group ids.",
            "invalid_fragment_state",
        )
    return tuple(sorted(row_group_ids))


def _positive_chunk_size(value: Any, *, path: str, row_group_id: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Integral)
        or int(value) <= 0
    ):
        raise _planning_error(
            f"Missing byte-size metadata in {path!r}, row group {row_group_id}.",
            "missing_size_metadata",
        )
    return int(value)


def _extract_row_group(
    *,
    path: str,
    source_identity: str,
    metadata: Any,
    row_group_id: int,
    key: str,
    projected_columns: Tuple[str, ...],
    indices_by_name: Dict[str, Tuple[int, ...]],
) -> Optional[ParquetRowGroupMetadata]:
    row_group = metadata.row_group(row_group_id)
    if row_group.num_rows == 0:
        return None

    key_indices = indices_by_name.get(key, ())
    if not key_indices:
        raise _planning_error(
            f"Key column {key!r} was not found in {path!r}.",
            "missing_key_column",
        )
    if len(key_indices) != 1:
        raise _planning_error(
            f"Key column {key!r} is not a primitive top-level column in {path!r}.",
            "unsupported_key_column",
        )

    statistics = row_group.column(key_indices[0]).statistics
    if (
        statistics is None
        or not statistics.has_min_max
        or statistics.min is None
        or statistics.max is None
    ):
        raise _planning_error(
            f"Missing key min/max in {path!r}, row group {row_group_id}.",
            "missing_statistics",
        )
    if statistics.null_count is None:
        raise _planning_error(
            f"Missing key null count in {path!r}, row group {row_group_id}.",
            "unknown_null_count",
        )
    null_count = int(statistics.null_count)
    if null_count != 0:
        raise _planning_error(
            f"Key column {key!r} contains nulls in {path!r}, "
            f"row group {row_group_id}.",
            "null_keys",
        )
    if (
        isinstance(statistics.min, bool)
        or isinstance(statistics.max, bool)
        or not isinstance(statistics.min, numbers.Integral)
        or not isinstance(statistics.max, numbers.Integral)
    ):
        raise _planning_error(
            f"Key column {key!r} has non-integral statistics in {path!r}.",
            "unsupported_key_type",
        )
    key_min = int(statistics.min)
    key_max = int(statistics.max)
    if key_min > key_max:
        raise _planning_error(
            f"Key minimum exceeds maximum in {path!r}, row group {row_group_id}.",
            "invalid_statistics",
        )

    chunk_sizes = []
    for column_index in range(row_group.num_columns):
        chunk = row_group.column(column_index)
        chunk_sizes.append(
            (
                _positive_chunk_size(
                    chunk.total_compressed_size,
                    path=path,
                    row_group_id=row_group_id,
                ),
                _positive_chunk_size(
                    chunk.total_uncompressed_size,
                    path=path,
                    row_group_id=row_group_id,
                ),
            )
        )

    column_sizes = []
    for column in projected_columns:
        column_indices = indices_by_name.get(column, ())
        if not column_indices:
            raise _planning_error(
                f"Projected column {column!r} was not found in {path!r}.",
                "missing_projection_column",
            )
        column_sizes.append(
            ParquetColumnSize(
                column=column,
                encoded_bytes=sum(chunk_sizes[index][0] for index in column_indices),
                uncompressed_bytes=sum(
                    chunk_sizes[index][1] for index in column_indices
                ),
            )
        )

    return ParquetRowGroupMetadata(
        path=path,
        source_identity=source_identity,
        row_group_id=row_group_id,
        num_rows=int(row_group.num_rows),
        key_min=key_min,
        key_max=key_max,
        null_count=null_count,
        encoded_bytes=sum(size[0] for size in chunk_sizes),
        uncompressed_bytes=sum(size[1] for size in chunk_sizes),
        column_sizes=tuple(column_sizes),
    )


def _extract_v1_metadata(
    datasource: Any,
    key: str,
    projection: Optional[Sequence[str]],
) -> Tuple[Tuple[ParquetRowGroupMetadata, ...], Optional[Tuple[str, ...]]]:
    fragments = _validate_local_v1_state(datasource)
    key = _normalize_key(key)
    normalized_projection = _normalize_projection(
        datasource, projection, key=key
    )

    entries = []
    seen_paths = set()
    expected_paths = getattr(datasource, "_pq_paths", None)
    for fragment_index, wrapped_fragment in enumerate(fragments):
        try:
            fragment = wrapped_fragment.original
            path = fragment.path
            listed_size = wrapped_fragment.file_size
        except AttributeError as exc:
            raise _planning_error(
                "ParquetDatasource contains an invalid fragment wrapper.",
                "invalid_fragment_state",
            ) from exc
        try:
            path = _as_posix_path(path)
        except ParquetSourceIdentityError as exc:
            raise _planning_error(str(exc), exc.reason_code) from exc
        if expected_paths is not None and path != expected_paths[fragment_index]:
            raise _planning_error(
                "ParquetDatasource paths and fragments are out of sync.",
                "fragment_state_mismatch",
            )
        if path in seen_paths:
            raise _planning_error(
                f"Duplicate resolved fragment for {path!r}.",
                "duplicate_fragment",
            )
        seen_paths.add(path)

        identity_before = _capture_for_planning(path)
        if (
            isinstance(listed_size, numbers.Integral)
            and int(listed_size) != identity_before.size
        ):
            raise _planning_error(
                f"Local Parquet source {path!r} changed after V1 discovery.",
                "source_changed",
            )
        try:
            metadata = fragment.metadata
        except Exception as exc:
            raise _planning_error(
                f"Failed to read Parquet footer for {path!r}: {exc}",
                "footer_read_failed",
            ) from exc

        identity_after = _capture_for_planning(path)
        if identity_after != identity_before:
            raise _planning_error(
                f"Local Parquet source {path!r} changed while reading metadata.",
                "source_changed_during_planning",
            )

        indices_by_name = _column_indices_by_name(metadata)
        projected_columns = (
            tuple(sorted(indices_by_name))
            if normalized_projection is None
            else normalized_projection
        )
        for row_group_id in _selected_row_group_ids(fragment, metadata):
            entry = _extract_row_group(
                path=path,
                source_identity=identity_before.encode(),
                metadata=metadata,
                row_group_id=row_group_id,
                key=key,
                projected_columns=projected_columns,
                indices_by_name=indices_by_name,
            )
            if entry is not None:
                entries.append(entry)

    if not entries:
        raise _planning_error(
            "The resolved Parquet fragments contain no non-empty row groups.",
            "empty_dataset",
        )
    return tuple(entries), normalized_projection


def extract_v1_parquet_row_group_metadata(
    datasource: Any,
    key: str,
    projection: Optional[Sequence[str]],
) -> Tuple[ParquetRowGroupMetadata, ...]:
    """Extract metadata from the datasource's exact resolved V1 fragments."""

    row_groups, _ = _extract_v1_metadata(datasource, key, projection)
    return row_groups


def try_extract_v1_parquet_row_group_metadata(
    datasource: Any,
    key: str,
    projection: Optional[Sequence[str]],
) -> ParquetV1MetadataResult:
    """Adapt V1 metadata without allowing the optimization to fail a read."""

    started = time.perf_counter()
    try:
        row_groups, normalized_projection = _extract_v1_metadata(
            datasource, key, projection
        )
    except ParquetRangePlanningError as exc:
        return ParquetV1MetadataResult(
            row_groups=(),
            projection=None,
            elapsed_s=time.perf_counter() - started,
            rejection_reason=exc.reason_code,
            rejection_message=str(exc),
            exception_type=type(exc).__name__,
        )
    except Exception as exc:
        return ParquetV1MetadataResult(
            row_groups=(),
            projection=None,
            elapsed_s=time.perf_counter() - started,
            rejection_reason="adapter_internal_error",
            rejection_message=str(exc),
            exception_type=type(exc).__name__,
        )
    return ParquetV1MetadataResult(
        row_groups=row_groups,
        projection=normalized_projection,
        elapsed_s=time.perf_counter() - started,
        rejection_reason=None,
        rejection_message=None,
        exception_type=None,
    )
