"""Pure metadata planning for Parquet key-range reads.

The planner in this module has no filesystem or PyArrow dependency.  Datasource
adapters are responsible for extracting complete row-group metadata and an
opaque source identity from their native fragment representation.  Keeping that
boundary explicit lets both datasource implementations share the same safety
checks and cost model.

A row group is assigned to every inclusive key range that its min/max
statistics overlap.  Executors must therefore filter every scanned row group by
the partition bounds before producing rows.
"""

from __future__ import annotations

import bisect
import math
import numbers
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple


class ParquetRangePlanningError(ValueError):
    """An expected condition that makes metadata range planning unsafe."""

    def __init__(self, message: str, *, reason_code: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True)
class ParquetColumnSize:
    """Encoded and uncompressed bytes for one projected logical column.

    Datasource adapters should aggregate nested physical column chunks under
    the logical projection name that selects them.
    """

    column: str
    encoded_bytes: int
    uncompressed_bytes: int


@dataclass(frozen=True)
class ParquetRowGroupMetadata:
    """Source-neutral metadata required to plan one non-empty row group.

    ``source_identity`` is an opaque, compact identity captured with the
    footer, such as a file size/mtime pair or object-store version identifier.
    Every row group for a path must carry the same identity.  The executor can
    compare it with a freshly observed identity before reading.

    The row-group byte totals are always required.  ``column_sizes`` is
    optional; when present, it lets the planner estimate a projected scan.  If
    absent, the planner conservatively uses the full row-group totals.
    """

    path: str
    source_identity: str
    row_group_id: int
    num_rows: int
    key_min: Any
    key_max: Any
    null_count: Optional[int]
    encoded_bytes: int
    uncompressed_bytes: int
    column_sizes: Tuple[ParquetColumnSize, ...] = ()


@dataclass(frozen=True)
class ParquetRowGroupFragment:
    """Row groups from one immutable source selected for a range."""

    path: str
    source_identity: str
    row_group_ids: Tuple[int, ...]


@dataclass(frozen=True)
class ParquetRangePartition:
    """Fragments overlapping one inclusive integral key range."""

    partition_id: int
    lower_bound: int
    upper_bound: int
    fragments: Tuple[ParquetRowGroupFragment, ...]
    estimated_scanned_row_groups: int
    estimated_scanned_rows: int
    estimated_encoded_bytes: int
    estimated_uncompressed_bytes: int


@dataclass(frozen=True)
class ParquetRangeMetrics:
    """Projected scan, overlap duplication, and partition-balance estimates."""

    num_partitions: int
    source_row_groups: int
    source_rows: int
    source_encoded_bytes: int
    source_uncompressed_bytes: int
    scanned_row_groups: int
    scanned_rows: int
    scanned_encoded_bytes: int
    scanned_uncompressed_bytes: int
    duplicate_row_groups: int
    duplicate_rows: int
    duplicate_encoded_bytes: int
    duplicate_uncompressed_bytes: int
    scan_amplification: float
    row_scan_amplification: float
    uncompressed_scan_amplification: float
    encoded_load_skew: float
    uncompressed_load_skew: float
    load_skew: float
    largest_partition_rows: int
    largest_partition_encoded_bytes: int
    largest_partition_uncompressed_bytes: int
    projection_size_complete: bool


@dataclass(frozen=True)
class ParquetRangeSuitabilityPolicy:
    """Fail-closed thresholds for selecting a metadata range plan."""

    max_scan_amplification: float = 1.25
    max_load_skew: float = 1.5
    max_partition_encoded_bytes: Optional[int] = None
    max_partition_uncompressed_bytes: Optional[int] = 1024**3


DEFAULT_PARQUET_RANGE_SUITABILITY_POLICY = ParquetRangeSuitabilityPolicy()


@dataclass(frozen=True)
class ParquetRangeSuitability:
    """Suitability decision with stable machine-readable reason codes."""

    suitable: bool
    reason_codes: Tuple[str, ...]


@dataclass(frozen=True)
class ParquetRangeLayout:
    """Serializable output of the source-neutral metadata planner."""

    key_min: int
    key_max: int
    projection: Optional[Tuple[str, ...]]
    row_groups: Tuple[ParquetRowGroupMetadata, ...]
    partitions: Tuple[ParquetRangePartition, ...]
    metrics: ParquetRangeMetrics
    suitability: ParquetRangeSuitability


@dataclass(frozen=True)
class ParquetRangePlanningResult:
    """Fail-closed result returned by :func:`try_plan_parquet_row_groups`."""

    plan: Optional[ParquetRangeLayout]
    rejection_reason: Optional[str]
    rejection_message: Optional[str]
    exception_type: Optional[str]

    @property
    def planned(self) -> bool:
        return self.plan is not None

    @property
    def suitable(self) -> bool:
        return self.plan is not None and self.plan.suitability.suitable

    @property
    def fallback_reason_codes(self) -> Tuple[str, ...]:
        """Stable reasons why an optimizer should retain its normal path."""

        if self.plan is None:
            return (self.rejection_reason,) if self.rejection_reason else ()
        return self.plan.suitability.reason_codes


def _planning_error(message: str, reason_code: str) -> ParquetRangePlanningError:
    return ParquetRangePlanningError(message, reason_code=reason_code)


def _normalize_projection(
    projection: Optional[Sequence[str]],
) -> Optional[Tuple[str, ...]]:
    if projection is None:
        return None
    if isinstance(projection, (str, bytes)):
        raise _planning_error(
            "projection must be a non-empty sequence of column names.",
            "invalid_projection",
        )
    try:
        columns = tuple(projection)
    except TypeError as exc:
        raise _planning_error(
            "projection must be a non-empty sequence of column names.",
            "invalid_projection",
        ) from exc
    if not columns or any(
        not isinstance(column, str) or not column for column in columns
    ):
        raise _planning_error(
            "projection must contain non-empty column names.",
            "invalid_projection",
        )
    if len(set(columns)) != len(columns):
        raise _planning_error(
            "projection must not contain duplicate columns.",
            "invalid_projection",
        )
    return tuple(sorted(columns))


def _validate_num_partitions(num_partitions: int) -> int:
    if (
        isinstance(num_partitions, bool)
        or not isinstance(num_partitions, numbers.Integral)
        or int(num_partitions) < 1
    ):
        raise _planning_error(
            "num_partitions must be a positive integer.",
            "invalid_partition_count",
        )
    return int(num_partitions)


def _validate_policy(policy: ParquetRangeSuitabilityPolicy) -> None:
    if not isinstance(policy, ParquetRangeSuitabilityPolicy):
        raise _planning_error(
            "suitability_policy must be a ParquetRangeSuitabilityPolicy.",
            "invalid_policy",
        )
    for name, value in (
        ("max_scan_amplification", policy.max_scan_amplification),
        ("max_load_skew", policy.max_load_skew),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, numbers.Real)
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise _planning_error(
                f"{name} must be a finite positive number.", "invalid_policy"
            )
    for name, value in (
        ("max_partition_encoded_bytes", policy.max_partition_encoded_bytes),
        (
            "max_partition_uncompressed_bytes",
            policy.max_partition_uncompressed_bytes,
        ),
    ):
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, numbers.Integral)
            or int(value) < 1
        ):
            raise _planning_error(
                f"{name} must be a positive integer or None.", "invalid_policy"
            )


def _positive_int(
    value: Any,
    *,
    name: str,
    identity: Tuple[str, int],
    reason_code: str = "invalid_row_group_metadata",
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Integral)
        or int(value) <= 0
    ):
        raise _planning_error(
            f"{name} must be a positive integer for row group {identity!r}.",
            reason_code,
        )
    return int(value)


def _normalize_column_sizes(
    column_sizes: Sequence[ParquetColumnSize], *, identity: Tuple[str, int]
) -> Tuple[ParquetColumnSize, ...]:
    try:
        entries = tuple(column_sizes)
    except TypeError as exc:
        raise _planning_error(
            f"column_sizes must be a sequence for row group {identity!r}.",
            "invalid_column_size_metadata",
        ) from exc

    normalized = []
    names = set()
    for entry in entries:
        if not isinstance(entry, ParquetColumnSize):
            raise _planning_error(
                f"Invalid column-size metadata for row group {identity!r}.",
                "invalid_column_size_metadata",
            )
        if not isinstance(entry.column, str) or not entry.column:
            raise _planning_error(
                f"Column-size names must be non-empty for row group {identity!r}.",
                "invalid_column_size_metadata",
            )
        if entry.column in names:
            raise _planning_error(
                f"Duplicate size metadata for column {entry.column!r} in "
                f"row group {identity!r}.",
                "duplicate_column_size_metadata",
            )
        names.add(entry.column)
        normalized.append(
            ParquetColumnSize(
                column=entry.column,
                encoded_bytes=_positive_int(
                    entry.encoded_bytes,
                    name="encoded_bytes",
                    identity=identity,
                    reason_code="invalid_column_size_metadata",
                ),
                uncompressed_bytes=_positive_int(
                    entry.uncompressed_bytes,
                    name="uncompressed_bytes",
                    identity=identity,
                    reason_code="invalid_column_size_metadata",
                ),
            )
        )
    return tuple(sorted(normalized, key=lambda entry: entry.column))


def _normalize_row_groups(
    row_groups: Sequence[ParquetRowGroupMetadata],
) -> Tuple[ParquetRowGroupMetadata, ...]:
    try:
        entries = tuple(row_groups)
    except TypeError as exc:
        raise _planning_error(
            "row_groups must be a sequence of ParquetRowGroupMetadata values.",
            "invalid_row_group_metadata",
        ) from exc
    if not entries:
        raise _planning_error(
            "No non-empty Parquet row groups were provided.", "empty_dataset"
        )

    normalized = []
    identities = set()
    source_identities: Dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, ParquetRowGroupMetadata):
            raise _planning_error(
                "row_groups must contain ParquetRowGroupMetadata values.",
                "invalid_row_group_metadata",
            )
        if not isinstance(entry.path, str) or not entry.path:
            raise _planning_error(
                "Every row group must have a non-empty path.",
                "invalid_row_group_metadata",
            )
        if not isinstance(entry.source_identity, str) or not entry.source_identity:
            raise _planning_error(
                f"Source {entry.path!r} has no immutable identity.",
                "missing_source_identity",
            )
        if isinstance(entry.row_group_id, bool) or not isinstance(
            entry.row_group_id, numbers.Integral
        ):
            raise _planning_error(
                f"row_group_id must be a non-negative integer for {entry.path!r}.",
                "invalid_row_group_metadata",
            )
        row_group_id = int(entry.row_group_id)
        identity = (entry.path, row_group_id)
        if row_group_id < 0:
            raise _planning_error(
                f"row_group_id must be non-negative for {entry.path!r}.",
                "invalid_row_group_metadata",
            )
        if identity in identities:
            raise _planning_error(
                f"Duplicate row-group metadata for {identity!r}.",
                "duplicate_row_group",
            )
        identities.add(identity)

        previous_source_identity = source_identities.setdefault(
            entry.path, entry.source_identity
        )
        if previous_source_identity != entry.source_identity:
            raise _planning_error(
                f"Source identity changed within metadata for {entry.path!r}.",
                "source_identity_mismatch",
            )

        num_rows = _positive_int(entry.num_rows, name="num_rows", identity=identity)
        if entry.key_min is None or entry.key_max is None:
            raise _planning_error(
                f"Missing min/max statistics for row group {identity!r}.",
                "missing_statistics",
            )
        if (
            isinstance(entry.key_min, bool)
            or isinstance(entry.key_max, bool)
            or not isinstance(entry.key_min, numbers.Integral)
            or not isinstance(entry.key_max, numbers.Integral)
        ):
            raise _planning_error(
                f"Key statistics must be integral for row group {identity!r}.",
                "unsupported_key_type",
            )
        key_min = int(entry.key_min)
        key_max = int(entry.key_max)
        if key_min > key_max:
            raise _planning_error(
                f"Minimum key exceeds maximum key for row group {identity!r}.",
                "invalid_statistics",
            )

        if entry.null_count is None:
            raise _planning_error(
                f"Missing null count for row group {identity!r}.",
                "unknown_null_count",
            )
        if isinstance(entry.null_count, bool) or not isinstance(
            entry.null_count, numbers.Integral
        ):
            raise _planning_error(
                f"null_count must be an integer for row group {identity!r}.",
                "invalid_row_group_metadata",
            )
        null_count = int(entry.null_count)
        if null_count < 0 or null_count > num_rows:
            raise _planning_error(
                f"Invalid null count for row group {identity!r}.",
                "invalid_row_group_metadata",
            )
        if null_count != 0:
            raise _planning_error(
                f"Range keys contain nulls in row group {identity!r}.",
                "null_keys",
            )

        encoded_bytes = _positive_int(
            entry.encoded_bytes,
            name="encoded_bytes",
            identity=identity,
            reason_code="missing_size_metadata",
        )
        uncompressed_bytes = _positive_int(
            entry.uncompressed_bytes,
            name="uncompressed_bytes",
            identity=identity,
            reason_code="missing_size_metadata",
        )
        column_sizes = _normalize_column_sizes(
            entry.column_sizes, identity=identity
        )
        normalized.append(
            ParquetRowGroupMetadata(
                path=entry.path,
                source_identity=entry.source_identity,
                row_group_id=row_group_id,
                num_rows=num_rows,
                key_min=key_min,
                key_max=key_max,
                null_count=null_count,
                encoded_bytes=encoded_bytes,
                uncompressed_bytes=uncompressed_bytes,
                column_sizes=column_sizes,
            )
        )

    return tuple(sorted(normalized, key=lambda entry: (entry.path, entry.row_group_id)))


def _projected_sizes(
    entry: ParquetRowGroupMetadata,
    projection: Optional[Tuple[str, ...]],
) -> Tuple[int, int, bool]:
    if projection is None:
        return entry.encoded_bytes, entry.uncompressed_bytes, True
    if not entry.column_sizes:
        return entry.encoded_bytes, entry.uncompressed_bytes, False

    by_column = {column.column: column for column in entry.column_sizes}
    missing = tuple(column for column in projection if column not in by_column)
    if missing:
        raise _planning_error(
            f"Projection columns {missing!r} are absent from size metadata for "
            f"row group {(entry.path, entry.row_group_id)!r}.",
            "missing_projection_column",
        )
    return (
        sum(by_column[column].encoded_bytes for column in projection),
        sum(by_column[column].uncompressed_bytes for column in projection),
        True,
    )


def _key_ranges(
    key_min: int, key_max: int, requested_partitions: int
) -> Tuple[Tuple[int, int], ...]:
    domain_width = key_max - key_min + 1
    partition_count = min(requested_partitions, domain_width)
    return tuple(
        (
            key_min + (partition_id * domain_width) // partition_count,
            key_min + ((partition_id + 1) * domain_width) // partition_count - 1,
        )
        for partition_id in range(partition_count)
    )


def _build_partitions(
    entries: Tuple[ParquetRowGroupMetadata, ...],
    sizes: Dict[Tuple[str, int], Tuple[int, int]],
    ranges: Tuple[Tuple[int, int], ...],
) -> Tuple[ParquetRangePartition, ...]:
    range_starts = [lower for lower, _ in ranges]
    range_ends = [upper for _, upper in ranges]
    fragments = [dict() for _ in ranges]
    row_group_counts = [0 for _ in ranges]
    rows = [0 for _ in ranges]
    encoded_bytes = [0 for _ in ranges]
    uncompressed_bytes = [0 for _ in ranges]

    for entry in entries:
        first_partition = bisect.bisect_left(range_ends, entry.key_min)
        last_partition = bisect.bisect_right(range_starts, entry.key_max) - 1
        entry_encoded_bytes, entry_uncompressed_bytes = sizes[
            (entry.path, entry.row_group_id)
        ]
        for partition_id in range(first_partition, last_partition + 1):
            source_key = (entry.path, entry.source_identity)
            fragments[partition_id].setdefault(source_key, []).append(
                entry.row_group_id
            )
            row_group_counts[partition_id] += 1
            rows[partition_id] += entry.num_rows
            encoded_bytes[partition_id] += entry_encoded_bytes
            uncompressed_bytes[partition_id] += entry_uncompressed_bytes

    return tuple(
        ParquetRangePartition(
            partition_id=partition_id,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            fragments=tuple(
                ParquetRowGroupFragment(
                    path=path,
                    source_identity=source_identity,
                    row_group_ids=tuple(sorted(row_group_ids)),
                )
                for (path, source_identity), row_group_ids in sorted(
                    fragments[partition_id].items()
                )
            ),
            estimated_scanned_row_groups=row_group_counts[partition_id],
            estimated_scanned_rows=rows[partition_id],
            estimated_encoded_bytes=encoded_bytes[partition_id],
            estimated_uncompressed_bytes=uncompressed_bytes[partition_id],
        )
        for partition_id, (lower_bound, upper_bound) in enumerate(ranges)
    )


def _build_metrics(
    entries: Tuple[ParquetRowGroupMetadata, ...],
    sizes: Dict[Tuple[str, int], Tuple[int, int]],
    partitions: Tuple[ParquetRangePartition, ...],
    *,
    projection_size_complete: bool,
) -> ParquetRangeMetrics:
    source_rows = sum(entry.num_rows for entry in entries)
    source_encoded_bytes = sum(
        sizes[(entry.path, entry.row_group_id)][0] for entry in entries
    )
    source_uncompressed_bytes = sum(
        sizes[(entry.path, entry.row_group_id)][1] for entry in entries
    )
    scanned_row_groups = sum(
        partition.estimated_scanned_row_groups for partition in partitions
    )
    scanned_rows = sum(partition.estimated_scanned_rows for partition in partitions)
    scanned_encoded_bytes = sum(
        partition.estimated_encoded_bytes for partition in partitions
    )
    scanned_uncompressed_bytes = sum(
        partition.estimated_uncompressed_bytes for partition in partitions
    )
    largest_encoded_bytes = max(
        partition.estimated_encoded_bytes for partition in partitions
    )
    largest_uncompressed_bytes = max(
        partition.estimated_uncompressed_bytes for partition in partitions
    )
    encoded_mean = scanned_encoded_bytes / len(partitions)
    uncompressed_mean = scanned_uncompressed_bytes / len(partitions)
    encoded_load_skew = largest_encoded_bytes / encoded_mean
    uncompressed_load_skew = largest_uncompressed_bytes / uncompressed_mean

    return ParquetRangeMetrics(
        num_partitions=len(partitions),
        source_row_groups=len(entries),
        source_rows=source_rows,
        source_encoded_bytes=source_encoded_bytes,
        source_uncompressed_bytes=source_uncompressed_bytes,
        scanned_row_groups=scanned_row_groups,
        scanned_rows=scanned_rows,
        scanned_encoded_bytes=scanned_encoded_bytes,
        scanned_uncompressed_bytes=scanned_uncompressed_bytes,
        duplicate_row_groups=scanned_row_groups - len(entries),
        duplicate_rows=scanned_rows - source_rows,
        duplicate_encoded_bytes=scanned_encoded_bytes - source_encoded_bytes,
        duplicate_uncompressed_bytes=(
            scanned_uncompressed_bytes - source_uncompressed_bytes
        ),
        scan_amplification=scanned_encoded_bytes / source_encoded_bytes,
        row_scan_amplification=scanned_rows / source_rows,
        uncompressed_scan_amplification=(
            scanned_uncompressed_bytes / source_uncompressed_bytes
        ),
        encoded_load_skew=encoded_load_skew,
        uncompressed_load_skew=uncompressed_load_skew,
        load_skew=max(encoded_load_skew, uncompressed_load_skew),
        largest_partition_rows=max(
            partition.estimated_scanned_rows for partition in partitions
        ),
        largest_partition_encoded_bytes=largest_encoded_bytes,
        largest_partition_uncompressed_bytes=largest_uncompressed_bytes,
        projection_size_complete=projection_size_complete,
    )


def evaluate_parquet_range_suitability(
    metrics: ParquetRangeMetrics,
    policy: ParquetRangeSuitabilityPolicy = (
        DEFAULT_PARQUET_RANGE_SUITABILITY_POLICY
    ),
) -> ParquetRangeSuitability:
    """Apply policy thresholds and return stable fallback reason codes."""

    if not isinstance(metrics, ParquetRangeMetrics):
        raise _planning_error(
            "metrics must be a ParquetRangeMetrics value.", "invalid_metrics"
        )
    _validate_policy(policy)
    reasons = []
    if metrics.scan_amplification > policy.max_scan_amplification:
        reasons.append("encoded_scan_amplification_exceeded")
    if metrics.row_scan_amplification > policy.max_scan_amplification:
        reasons.append("row_scan_amplification_exceeded")
    if metrics.uncompressed_scan_amplification > policy.max_scan_amplification:
        reasons.append("uncompressed_scan_amplification_exceeded")
    if metrics.load_skew > policy.max_load_skew:
        reasons.append("load_skew_exceeded")
    if (
        policy.max_partition_encoded_bytes is not None
        and metrics.largest_partition_encoded_bytes
        > policy.max_partition_encoded_bytes
    ):
        reasons.append("partition_encoded_bytes_exceeded")
    if (
        policy.max_partition_uncompressed_bytes is not None
        and metrics.largest_partition_uncompressed_bytes
        > policy.max_partition_uncompressed_bytes
    ):
        reasons.append("partition_uncompressed_bytes_exceeded")
    return ParquetRangeSuitability(
        suitable=not reasons, reason_codes=tuple(reasons)
    )


def plan_parquet_row_groups(
    row_groups: Sequence[ParquetRowGroupMetadata],
    *,
    num_partitions: int,
    projection: Optional[Sequence[str]] = None,
    suitability_policy: ParquetRangeSuitabilityPolicy = (
        DEFAULT_PARQUET_RANGE_SUITABILITY_POLICY
    ),
) -> ParquetRangeLayout:
    """Plan validated row groups into equal-width inclusive key ranges.

    The number of partitions is capped by the width of the observed integral
    domain.  Sparse empty ranges are retained because they are relevant to load
    skew.  When projected column sizes are unavailable, byte estimates
    conservatively fall back to full row-group sizes.
    """

    partition_count = _validate_num_partitions(num_partitions)
    normalized_projection = _normalize_projection(projection)
    _validate_policy(suitability_policy)
    entries = _normalize_row_groups(row_groups)

    sizes: Dict[Tuple[str, int], Tuple[int, int]] = {}
    projection_size_complete = True
    for entry in entries:
        encoded_bytes, uncompressed_bytes, exact = _projected_sizes(
            entry, normalized_projection
        )
        sizes[(entry.path, entry.row_group_id)] = (
            encoded_bytes,
            uncompressed_bytes,
        )
        projection_size_complete = projection_size_complete and exact

    key_min = min(entry.key_min for entry in entries)
    key_max = max(entry.key_max for entry in entries)
    ranges = _key_ranges(key_min, key_max, partition_count)
    partitions = _build_partitions(entries, sizes, ranges)
    metrics = _build_metrics(
        entries,
        sizes,
        partitions,
        projection_size_complete=projection_size_complete,
    )
    return ParquetRangeLayout(
        key_min=key_min,
        key_max=key_max,
        projection=normalized_projection,
        row_groups=entries,
        partitions=partitions,
        metrics=metrics,
        suitability=evaluate_parquet_range_suitability(
            metrics, policy=suitability_policy
        ),
    )


def try_plan_parquet_row_groups(
    row_groups: Sequence[ParquetRowGroupMetadata],
    *,
    num_partitions: int,
    projection: Optional[Sequence[str]] = None,
    suitability_policy: ParquetRangeSuitabilityPolicy = (
        DEFAULT_PARQUET_RANGE_SUITABILITY_POLICY
    ),
) -> ParquetRangePlanningResult:
    """Attempt planning without allowing metadata optimization to fail a job."""

    try:
        plan = plan_parquet_row_groups(
            row_groups,
            num_partitions=num_partitions,
            projection=projection,
            suitability_policy=suitability_policy,
        )
    except ParquetRangePlanningError as exc:
        return ParquetRangePlanningResult(
            plan=None,
            rejection_reason=exc.reason_code,
            rejection_message=str(exc),
            exception_type=type(exc).__name__,
        )
    except Exception as exc:
        return ParquetRangePlanningResult(
            plan=None,
            rejection_reason="internal_error",
            rejection_message=str(exc),
            exception_type=type(exc).__name__,
        )
    return ParquetRangePlanningResult(
        plan=plan,
        rejection_reason=None,
        rejection_message=None,
        exception_type=None,
    )
