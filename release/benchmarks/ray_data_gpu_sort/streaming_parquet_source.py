"""Lazy, single-pass Parquet source for the frozen 1 TB BTS benchmark.

The source deliberately separates planning from reading.  Constructing the
datasource (and calling :meth:`get_read_tasks`) only validates immutable plan
metadata and creates ordinary Python callables.  A callable opens its Parquet
row group only when Ray executes the read task.

Reader attempts are claimed in a trial-owned directory.  This makes accidental
Dataset replay fail loudly instead of silently rereading the 1 TB input, and
provides bounded, cross-process timestamp telemetry without a controller RPC on
the read critical path.
"""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import socket
import time
from typing import Any, Iterator, Mapping, Optional, Sequence

import pyarrow as pa

from ray.data.block import BlockMetadata
from ray.data.context import DataContext
from ray.data.datasource import Datasource, ReadTask

from .streaming_parquet_common import (
    TARGET_BLOCKS,
    TARGET_DECODED_BYTES,
    TARGET_PLAN_DIGEST,
    TARGET_ROWS,
)
from .spec import digest


_ROW_ID = "row_id"
_TELEMETRY_VERSION = 1
_MAX_EVENTS_PER_TASK = 2
_READ_TASK_SUBMISSION_WINDOW = 256

# ``manifest.json`` records the exact native Arrow bytes for every complete
# normalized BTS row group.  ``read_projection`` performs an IPC round trip,
# which removes unused validity buffers, and appends an int64 row_id.  The
# normalized schema ordinarily retains 25 validity bitmaps.  These frozen row
# groups have one fewer or additional all-null bitmaps; keeping the compact
# exception set here is substantially easier to audit than a 9,151-entry byte
# table.  The opt-in exact-metadata integration test recomputes every distinct
# block from Parquet and verifies this layout contract.
_ROW_GROUPS_WITH_24_VALIDITY_BITMAPS = frozenset(
    {
        (2013, 11, 1),
        (2014, 2, 2),
        (2014, 6, 3),
        (2014, 10, 3),
        (2015, 2, 1),
        (2015, 4, 0),
        (2015, 4, 3),
        (2015, 5, 0),
        (2015, 5, 2),
        (2016, 3, 2),
        (2016, 7, 1),
        (2016, 8, 1),
        (2016, 10, 0),
        (2016, 12, 1),
        (2017, 1, 0),
        (2017, 1, 2),
        (2017, 2, 1),
        (2017, 3, 0),
        (2017, 4, 2),
        (2017, 8, 0),
        (2018, 2, 1),
        (2018, 2, 3),
        (2018, 5, 3),
        (2018, 5, 4),
        (2018, 7, 1),
        (2019, 1, 3),
        (2019, 7, 4),
        (2020, 1, 1),
        (2020, 1, 2),
        (2020, 1, 3),
        (2020, 3, 3),
        (2020, 4, 1),
        (2020, 8, 0),
        (2020, 8, 1),
        (2020, 9, 0),
        (2020, 11, 0),
        (2021, 3, 1),
        (2021, 6, 2),
        (2021, 9, 2),
        (2022, 12, 3),
        (2023, 1, 3),
        (2023, 2, 1),
        (2023, 3, 0),
        (2023, 5, 2),
        (2023, 9, 1),
        (2023, 11, 1),
        (2024, 2, 1),
        (2024, 7, 1),
        (2024, 8, 1),
        (2025, 6, 3),
        (2025, 9, 2),
    }
)
_ROW_GROUP_VALIDITY_BITMAP_EXCEPTIONS = {
    (2017, 11, 3): 35,
    (2018, 12, 4): 41,
    (2019, 11, 4): 35,
    (2023, 10, 4): 35,
}
_PARTIAL_BLOCK_BYTES = {
    # Frozen terminal 1 TB slice.
    (2020, 12, 0, 40_855): 34_783_844,
    # The 16-block correctness smoke reads a 10,000-row prefix.
    (2025, 9, 1, 10_000): 8_495_890,
    (2025, 9, 2, 10_000): 8_475_085,
    (2025, 9, 3, 10_000): 8_522_193,
    (2025, 10, 0, 10_000): 8_511_840,
    (2025, 10, 1, 10_000): 8_542_836,
    (2025, 10, 2, 10_000): 8_482_010,
    (2025, 10, 3, 10_000): 8_492_741,
    (2025, 10, 4, 10_000): 8_502_238,
    (2025, 11, 0, 10_000): 8_495_244,
    (2025, 11, 1, 10_000): 8_490_369,
    (2025, 11, 2, 10_000): 8_516_705,
    (2025, 11, 3, 10_000): 8_474_651,
    (2025, 12, 0, 10_000): 8_506_470,
    (2025, 12, 1, 10_000): 8_471_425,
    (2025, 12, 2, 10_000): 8_523_157,
    (2025, 12, 3, 10_000): 8_483_163,
}


class FrozenPlanError(ValueError):
    """The supplied workload differs from the frozen benchmark contract."""


class NonReplayableReadError(RuntimeError):
    """A frozen input task was invoked more than once in one trial."""


@dataclass(frozen=True)
class FrozenReadSlice:
    """One deterministic read-task descriptor."""

    ordinal: int
    path: str
    row_group: int
    rows: int
    row_id_start: int
    year: int
    month: int
    copy: int
    planned_size_bytes: int


def native_schema_from_manifest(manifest: Mapping[str, Any]) -> pa.Schema:
    """Reconstruct the exact 109-column schema without opening Parquet.

    The normalized BTS manifest stores Arrow's canonical schema string and its
    ordered field names.  Its types are scalar aliases understood by PyArrow.
    Refusing nested/multiline fields is intentional: accepting a lossy parser
    here would weaken the benchmark's schema proof.
    """

    encoded = manifest.get("schema")
    names = manifest.get("schema_names")
    if not isinstance(encoded, str) or not isinstance(names, list):
        raise FrozenPlanError("Manifest has no canonical schema/schema_names")

    fields: list[pa.Field] = []
    for line in encoded.splitlines():
        try:
            name, type_name = line.split(": ", 1)
        except ValueError as error:
            raise FrozenPlanError(
                f"Unsupported manifest schema line: {line!r}"
            ) from error
        if not name or not type_name or type_name.startswith(("struct<", "list<")):
            raise FrozenPlanError(f"Unsupported manifest schema line: {line!r}")
        try:
            arrow_type = pa.type_for_alias(type_name)
        except (KeyError, ValueError) as error:
            raise FrozenPlanError(
                f"Unsupported manifest Arrow type {type_name!r}"
            ) from error
        fields.append(pa.field(name, arrow_type, nullable=True))

    schema = pa.schema(fields)
    expected_names = tuple(str(name) for name in names)
    if tuple(schema.names) != expected_names or str(schema) != encoded:
        raise FrozenPlanError("Manifest schema text and schema_names disagree")
    return schema


def schema_from_manifest(manifest: Mapping[str, Any]) -> pa.Schema:
    """Return the exact 110-column benchmark schema, including ``row_id``."""

    return native_schema_from_manifest(manifest).append(pa.field(_ROW_ID, pa.int64()))


def exact_block_size_bytes(
    plan: Mapping[str, Any], manifest: Mapping[str, Any]
) -> tuple[int, ...]:
    """Return exact post-``read_projection`` bytes for a frozen BTS plan.

    Complete blocks are calculated from immutable normalized-manifest metadata;
    known partial prefixes use frozen, independently verified Arrow sizes.  No
    Parquet file is opened, so callers can supply exact ``ReadTask`` metadata
    while preserving lazy graph construction.
    """

    raw_slices = plan.get("slices")
    row_groups = manifest.get("row_groups")
    if not isinstance(raw_slices, list) or not isinstance(row_groups, list):
        raise FrozenPlanError("Plan or manifest is missing row-group metadata")
    native: dict[tuple[int, int, int, str], tuple[int, int]] = {}
    for value in row_groups:
        if not isinstance(value, Mapping):
            raise FrozenPlanError("Manifest row-group metadata is not an object")
        try:
            key = (
                int(value["year"]),
                int(value["month"]),
                int(value["row_group"]),
                str(value["path"]),
            )
            geometry = (int(value["rows"]), int(value["decoded_bytes"]))
        except (KeyError, TypeError, ValueError) as error:
            raise FrozenPlanError("Invalid manifest row-group metadata") from error
        if key in native:
            raise FrozenPlanError(f"Duplicate manifest row group: {key}")
        native[key] = geometry

    result: list[int] = []
    for ordinal, value in enumerate(raw_slices):
        if not isinstance(value, Mapping):
            raise FrozenPlanError(f"Plan slice {ordinal} is not an object")
        try:
            year = int(value["year"])
            month = int(value["month"])
            row_group = int(value["row_group"])
            rows = int(value["rows"])
            path = str(value["path"])
        except (KeyError, TypeError, ValueError) as error:
            raise FrozenPlanError(f"Invalid plan slice {ordinal}") from error
        manifest_value = native.get((year, month, row_group, path))
        if manifest_value is None:
            raise FrozenPlanError(f"Plan slice {ordinal} is absent from the manifest")
        native_rows, native_bytes = manifest_value
        partial = _PARTIAL_BLOCK_BYTES.get((year, month, row_group, rows))
        if rows != native_rows:
            if partial is None:
                raise FrozenPlanError(
                    f"No exact byte metadata for partial plan slice {ordinal}"
                )
            result.append(partial)
            continue
        bitmap_key = (year, month, row_group)
        bitmap_count = _ROW_GROUP_VALIDITY_BITMAP_EXCEPTIONS.get(bitmap_key, 25)
        if bitmap_key in _ROW_GROUPS_WITH_24_VALIDITY_BITMAPS:
            bitmap_count = 24
        validity_bytes = bitmap_count * ((rows + 7) // 8)
        result.append(native_bytes + 8 * rows - validity_bytes)
    return tuple(result)


def _validate_digest(plan: Mapping[str, Any], expected_digest: str) -> None:
    supplied = plan.get("digest")
    unsigned = {key: value for key, value in plan.items() if key != "digest"}
    calculated = digest(unsigned)
    if supplied != calculated:
        raise FrozenPlanError(
            f"Plan digest is invalid: supplied={supplied!r}, calculated={calculated!r}"
        )
    if calculated != expected_digest:
        raise FrozenPlanError(
            f"Plan digest changed: {calculated!r}, expected {expected_digest!r}"
        )


def _allocate_planned_bytes(
    rows: Sequence[int], total_size_bytes: int
) -> tuple[int, ...]:
    """Allocate a deterministic aggregate estimate for non-frozen test plans.

    Production benchmark callers supply :func:`exact_block_size_bytes`.  This
    fallback keeps the Datasource reusable in small synthetic unit tests while
    making no claim of per-block exactness.
    """

    if total_size_bytes < 0:
        raise FrozenPlanError("Expected decoded bytes must be nonnegative")
    total_rows = sum(rows)
    if total_rows <= 0:
        if total_size_bytes:
            raise FrozenPlanError("A zero-row plan cannot have decoded bytes")
        return tuple(0 for _ in rows)
    if total_size_bytes == 0:
        raise FrozenPlanError("A nonempty plan needs a positive byte estimate")

    result: list[int] = []
    cumulative_rows = 0
    assigned = 0
    for count in rows:
        cumulative_rows += count
        cumulative_bytes = total_size_bytes * cumulative_rows // total_rows
        result.append(cumulative_bytes - assigned)
        assigned = cumulative_bytes
    if assigned != total_size_bytes:
        raise AssertionError("Planned byte allocation lost bytes")
    return tuple(result)


def _validated_slices(
    plan: Mapping[str, Any],
    *,
    expected_digest: str,
    expected_rows: int,
    expected_blocks: int,
    expected_decoded_bytes: int,
    block_size_bytes: Optional[Sequence[int]],
) -> tuple[FrozenReadSlice, ...]:
    _validate_digest(plan, expected_digest)
    raw_slices = plan.get("slices")
    if not isinstance(raw_slices, list):
        raise FrozenPlanError("Plan slices must be a list")
    if plan.get("blocks") != len(raw_slices) or len(raw_slices) != expected_blocks:
        raise FrozenPlanError(
            f"Plan has {len(raw_slices)} blocks, expected {expected_blocks}"
        )

    normalized: list[dict[str, Any]] = []
    next_row_id = 0
    for ordinal, value in enumerate(raw_slices):
        if not isinstance(value, Mapping):
            raise FrozenPlanError(f"Plan slice {ordinal} is not an object")
        try:
            item = {
                "path": str(value["path"]),
                "row_group": int(value["row_group"]),
                "rows": int(value["rows"]),
                "row_id_start": int(value["row_id_start"]),
                "year": int(value["year"]),
                "month": int(value["month"]),
                "copy": int(value.get("copy", 0)),
            }
        except (KeyError, TypeError, ValueError) as error:
            raise FrozenPlanError(f"Invalid plan slice {ordinal}: {value!r}") from error
        if not Path(item["path"]).is_absolute():
            raise FrozenPlanError(f"Plan slice {ordinal} path is not absolute")
        if item["row_group"] < 0 or item["rows"] <= 0 or item["copy"] < 0:
            raise FrozenPlanError(f"Plan slice {ordinal} has invalid geometry")
        if item["row_id_start"] != next_row_id:
            raise FrozenPlanError(
                f"Plan slice {ordinal} row_id_start={item['row_id_start']}, "
                f"expected {next_row_id}"
            )
        next_row_id += item["rows"]
        normalized.append(item)

    if plan.get("rows") != next_row_id or next_row_id != expected_rows:
        raise FrozenPlanError(f"Plan has {next_row_id} rows, expected {expected_rows}")

    row_counts = tuple(item["rows"] for item in normalized)
    if block_size_bytes is None:
        sizes = _allocate_planned_bytes(row_counts, expected_decoded_bytes)
    else:
        try:
            sizes = tuple(int(value) for value in block_size_bytes)
        except (TypeError, ValueError) as error:
            raise FrozenPlanError(
                "Block byte metadata must contain integers"
            ) from error
        if len(sizes) != expected_blocks or any(value < 0 for value in sizes):
            raise FrozenPlanError("Block byte metadata has invalid geometry")
        if sum(sizes) != expected_decoded_bytes:
            raise FrozenPlanError(
                f"Block byte metadata sums to {sum(sizes)}, "
                f"expected {expected_decoded_bytes}"
            )

    return tuple(
        FrozenReadSlice(ordinal=ordinal, planned_size_bytes=sizes[ordinal], **item)
        for ordinal, item in enumerate(normalized)
    )


def _append_event(directory: str, payload: Mapping[str, Any]) -> None:
    """Append one bounded JSON event atomically across local worker processes."""

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    event_path = root / "source-events.jsonl"
    encoded = (
        json.dumps(dict(payload), sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    descriptor = os.open(event_path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise OSError(f"Short telemetry append: {written}/{len(encoded)} bytes")
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _event(
    *,
    event: str,
    item: FrozenReadSlice,
    plan_digest: str,
    **details: Any,
) -> dict[str, Any]:
    return {
        "version": _TELEMETRY_VERSION,
        "event": event,
        "ordinal": item.ordinal,
        "plan_digest": plan_digest,
        "monotonic_ns": time.monotonic_ns(),
        "wall_time_ns": time.time_ns(),
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        **details,
    }


def _claim(directory: str, item: FrozenReadSlice, plan_digest: str) -> None:
    root = Path(directory)
    claims = root / "claims"
    claims.mkdir(parents=True, exist_ok=True)
    claim = claims / f"{item.ordinal:05d}"
    try:
        claim.mkdir()
    except FileExistsError as error:
        raise NonReplayableReadError(
            f"Frozen input task {item.ordinal} was already consumed"
        ) from error
    _append_event(
        directory,
        _event(event="started", item=item, plan_digest=plan_digest),
    )


def _read_slice(
    item: FrozenReadSlice,
    *,
    native_schema: pa.Schema,
    output_schema: pa.Schema,
    columns: tuple[str, ...],
    telemetry_directory: str,
    plan_digest: str,
) -> list[pa.Table]:
    """Execute one claimed, non-replayable row-group read."""

    _claim(telemetry_directory, item, plan_digest)
    started_ns = time.monotonic_ns()
    try:
        # Reuse the accepted benchmark projection.  Its bounded IPC round-trip
        # removes spare variable-width buffer capacity, which is part of the
        # frozen decoded-byte accounting.
        from .data import read_projection

        table = read_projection(
            {
                "path": item.path,
                "row_group": item.row_group,
                "rows": item.rows,
                "row_id_start": item.row_id_start,
                "year": item.year,
                "month": item.month,
                "copy": item.copy,
            },
            columns,
        )
        if not table.schema.equals(output_schema, check_metadata=True):
            raise FrozenPlanError(
                f"Reader {item.ordinal} changed the frozen schema: {table.schema}"
            )
        if not table.schema.remove_metadata().equals(
            native_schema.append(pa.field(_ROW_ID, pa.int64())).remove_metadata()
        ):
            raise FrozenPlanError(f"Reader {item.ordinal} changed native Arrow types")
        if table.num_rows != item.rows:
            raise FrozenPlanError(
                f"Reader {item.ordinal} produced {table.num_rows}/{item.rows} rows"
            )
    except BaseException as error:
        _append_event(
            telemetry_directory,
            _event(
                event="failed",
                item=item,
                plan_digest=plan_digest,
                elapsed_ns=time.monotonic_ns() - started_ns,
                error_type=type(error).__name__,
                error=str(error)[:512],
            ),
        )
        raise

    _append_event(
        telemetry_directory,
        _event(
            event="produced",
            item=item,
            plan_digest=plan_digest,
            elapsed_ns=time.monotonic_ns() - started_ns,
            rows=table.num_rows,
            decoded_bytes=table.nbytes,
            planned_size_bytes=item.planned_size_bytes,
        ),
    )
    return [table]


def _make_read_fn(
    item: FrozenReadSlice,
    *,
    native_schema: pa.Schema,
    output_schema: pa.Schema,
    columns: tuple[str, ...],
    telemetry_directory: str,
    plan_digest: str,
):
    """Return a small callable that captures one descriptor, never the source."""

    def read_frozen_bts_slice() -> list[pa.Table]:
        return _read_slice(
            item,
            native_schema=native_schema,
            output_schema=output_schema,
            columns=columns,
            telemetry_directory=telemetry_directory,
            plan_digest=plan_digest,
        )

    return read_frozen_bts_slice


def read_source_telemetry(
    directory: Path,
    *,
    plan_digest: str,
    expected_blocks: int,
) -> dict[str, Any]:
    """Read and validate bounded source telemetry after (or during) a trial."""

    path = directory / "source-events.jsonl"
    if not path.is_file():
        return {
            "events": 0,
            "started_tasks": 0,
            "produced_tasks": 0,
            "failed_tasks": 0,
            "duplicate_started_events": 0,
            "duplicate_produced_events": 0,
            "completed_ordinals": [],
            "missing_ordinals": list(range(expected_blocks)),
            "exact_once_complete": expected_blocks == 0,
            "by_ordinal": {},
            "produced_rows": 0,
            "produced_decoded_bytes": 0,
            "planned_decoded_bytes": 0,
            "produced_bytes_match_planned": expected_blocks == 0,
            "byte_metadata_mismatch_ordinals": [],
            "byte_metadata_semantics": (
                "ReadTask size_bytes is exact when exact_block_size_bytes was "
                "supplied; produced_decoded_bytes independently records table.nbytes"
            ),
            "first_read_started_monotonic_ns": None,
            "last_input_produced_monotonic_ns": None,
            "first_read_started_wall_time_ns": None,
            "last_input_produced_wall_time_ns": None,
        }

    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.endswith("\n"):
                # A live reader may be between append and close. Ignore only a
                # final incomplete record when inspecting in-flight telemetry.
                break
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Invalid source telemetry at line {line_number}"
                ) from error
            if value.get("plan_digest") != plan_digest:
                raise RuntimeError("Source telemetry contains another frozen plan")
            ordinal = value.get("ordinal")
            if not isinstance(ordinal, int) or not 0 <= ordinal < expected_blocks:
                raise RuntimeError(f"Invalid source telemetry ordinal: {ordinal!r}")
            records.append(value)

    if len(records) > expected_blocks * _MAX_EVENTS_PER_TASK:
        raise RuntimeError("Source telemetry exceeded its bounded event contract")
    events_per_task: dict[int, int] = {}
    for value in records:
        ordinal = int(value["ordinal"])
        events_per_task[ordinal] = events_per_task.get(ordinal, 0) + 1
    if any(count > _MAX_EVENTS_PER_TASK for count in events_per_task.values()):
        raise RuntimeError("One source task exceeded its bounded event contract")

    allowed_events = {"started", "produced", "failed"}
    unknown = [
        value.get("event")
        for value in records
        if value.get("event") not in allowed_events
    ]
    if unknown:
        raise RuntimeError(f"Source telemetry has unknown events: {unknown[:4]}")
    started = [value for value in records if value.get("event") == "started"]
    produced = [value for value in records if value.get("event") == "produced"]
    failed = [value for value in records if value.get("event") == "failed"]
    started_by_ordinal = {int(value["ordinal"]): value for value in started}
    produced_by_ordinal = {int(value["ordinal"]): value for value in produced}
    failed_by_ordinal = {int(value["ordinal"]): value for value in failed}
    duplicate_started = len(started) - len(started_by_ordinal)
    duplicate_produced = len(produced) - len(produced_by_ordinal)
    completed_ordinals = sorted(produced_by_ordinal)
    missing_ordinals = sorted(set(range(expected_blocks)) - produced_by_ordinal.keys())
    observed_ordinals = sorted(
        started_by_ordinal.keys()
        | produced_by_ordinal.keys()
        | failed_by_ordinal.keys()
    )
    by_ordinal = {}
    for ordinal in observed_ordinals:
        started_value = started_by_ordinal.get(ordinal, {})
        produced_value = produced_by_ordinal.get(ordinal, {})
        failed_value = failed_by_ordinal.get(ordinal, {})
        by_ordinal[str(ordinal)] = {
            "started_monotonic_ns": started_value.get("monotonic_ns"),
            "started_wall_time_ns": started_value.get("wall_time_ns"),
            "produced_monotonic_ns": produced_value.get("monotonic_ns"),
            "produced_wall_time_ns": produced_value.get("wall_time_ns"),
            "elapsed_ns": produced_value.get(
                "elapsed_ns", failed_value.get("elapsed_ns")
            ),
            "rows": produced_value.get("rows"),
            "decoded_bytes": produced_value.get("decoded_bytes"),
            "planned_size_bytes": produced_value.get("planned_size_bytes"),
            "failed": ordinal in failed_by_ordinal,
            "error_type": failed_value.get("error_type"),
            "error": failed_value.get("error"),
        }
    byte_mismatches = sorted(
        int(value["ordinal"])
        for value in produced
        if int(value.get("decoded_bytes", -1))
        != int(value.get("planned_size_bytes", -2))
    )
    return {
        "events": len(records),
        "started_tasks": len(started_by_ordinal),
        "produced_tasks": len(produced_by_ordinal),
        "failed_tasks": len(failed_by_ordinal),
        "duplicate_started_events": duplicate_started,
        "duplicate_produced_events": duplicate_produced,
        "completed_ordinals": completed_ordinals,
        "missing_ordinals": missing_ordinals,
        "exact_once_complete": (
            len(started_by_ordinal) == expected_blocks
            and len(produced_by_ordinal) == expected_blocks
            and not failed_by_ordinal
            and duplicate_started == 0
            and duplicate_produced == 0
        ),
        "by_ordinal": by_ordinal,
        "produced_rows": sum(int(value.get("rows", 0)) for value in produced),
        "produced_decoded_bytes": sum(
            int(value.get("decoded_bytes", 0)) for value in produced
        ),
        "planned_decoded_bytes": sum(
            int(value.get("planned_size_bytes", 0)) for value in produced
        ),
        "produced_bytes_match_planned": not byte_mismatches,
        "byte_metadata_mismatch_ordinals": byte_mismatches,
        "byte_metadata_semantics": (
            "ReadTask size_bytes is exact when exact_block_size_bytes was supplied; "
            "produced_decoded_bytes independently records table.nbytes"
        ),
        "first_read_started_monotonic_ns": min(
            (int(value["monotonic_ns"]) for value in started), default=None
        ),
        "last_input_produced_monotonic_ns": max(
            (int(value["monotonic_ns"]) for value in produced), default=None
        ),
        "first_read_started_wall_time_ns": min(
            (int(value["wall_time_ns"]) for value in started), default=None
        ),
        "last_input_produced_wall_time_ns": max(
            (int(value["wall_time_ns"]) for value in produced), default=None
        ),
    }


class FrozenBTSPlanDatasource(Datasource):
    """Exact, lazy, non-replayable source for a frozen BTS slice plan.

    ``expected_decoded_bytes`` is the exact aggregate for benchmark plans.
    Benchmark callers pass :func:`exact_block_size_bytes`, so deterministic
    per-block metadata is known before reading.  Ray also derives each yielded
    RefBundle's byte metadata from the actual Arrow table, and source telemetry
    independently records ``table.nbytes`` for the buffering proof.
    """

    _supports_lazy_read_task_submission = True

    def __init__(
        self,
        plan: Mapping[str, Any],
        *,
        schema: pa.Schema,
        telemetry_directory: Path,
        columns: Optional[Sequence[str]] = None,
        expected_plan_digest: str = TARGET_PLAN_DIGEST,
        expected_rows: int = TARGET_ROWS,
        expected_blocks: int = TARGET_BLOCKS,
        expected_decoded_bytes: int = TARGET_DECODED_BYTES,
        block_size_bytes: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        if not isinstance(schema, pa.Schema):
            raise TypeError("schema must be a pyarrow.Schema")
        if not schema.names or schema.names[-1] != _ROW_ID:
            raise FrozenPlanError("Frozen output schema must end with row_id")
        row_id_field = schema[-1]
        if row_id_field.type != pa.int64() or not row_id_field.nullable:
            raise FrozenPlanError("row_id must be a nullable int64 Arrow field")
        if schema.names.count(_ROW_ID) != 1:
            raise FrozenPlanError("Frozen output schema must contain one row_id")
        native_schema = schema.remove(len(schema) - 1)
        if not telemetry_directory.is_absolute():
            raise FrozenPlanError("Telemetry directory must be absolute")

        selected = tuple(schema.names if columns is None else columns)
        required = tuple(schema.names)
        if selected != required:
            raise FrozenPlanError(
                "Frozen BTS source must read all native columns followed by row_id"
            )
        output_schema = schema
        slices = _validated_slices(
            plan,
            expected_digest=expected_plan_digest,
            expected_rows=expected_rows,
            expected_blocks=expected_blocks,
            expected_decoded_bytes=expected_decoded_bytes,
            block_size_bytes=block_size_bytes,
        )

        self._plan_digest = expected_plan_digest
        self._native_schema = native_schema
        self._output_schema = output_schema
        self._columns = selected
        self._telemetry_directory = str(telemetry_directory)
        self._slices = slices
        self._expected_decoded_bytes = int(expected_decoded_bytes)

    @property
    def plan_digest(self) -> str:
        return self._plan_digest

    @property
    def block_descriptors(self) -> tuple[FrozenReadSlice, ...]:
        return self._slices

    @property
    def output_schema(self) -> pa.Schema:
        return self._output_schema

    @property
    def _read_task_count(self) -> int:
        """Exact task count for Ray's internal lazy-read planner contract."""

        return len(self._slices)

    @property
    def _read_task_submission_window(self) -> int:
        """Bound descriptor ObjectRefs waiting for downstream read workers."""

        return _READ_TASK_SUBMISSION_WINDOW

    def estimate_inmemory_data_size(self) -> int:
        return self._expected_decoded_bytes

    def _iter_read_tasks(
        self,
        parallelism: int,
        per_task_row_limit: Optional[int] = None,
        data_context: Optional[DataContext] = None,
    ) -> Iterator[ReadTask]:
        """Yield one lightweight descriptor per frozen row group, exactly once."""

        del parallelism, data_context
        if per_task_row_limit is not None:
            raise FrozenPlanError(
                "A per-task row limit would change the frozen block geometry"
            )
        for item in self._slices:
            yield ReadTask(
                _make_read_fn(
                    item,
                    native_schema=self._native_schema,
                    output_schema=self._output_schema,
                    columns=self._columns,
                    telemetry_directory=self._telemetry_directory,
                    plan_digest=self._plan_digest,
                ),
                BlockMetadata(
                    num_rows=item.rows,
                    size_bytes=item.planned_size_bytes,
                    input_files=(item.path,),
                    exec_stats=None,
                ),
                schema=self._output_schema,
            )

    def get_read_tasks(
        self,
        parallelism: int,
        per_task_row_limit: Optional[int] = None,
        data_context: Optional[DataContext] = None,
    ) -> list[ReadTask]:
        # Keep the public Datasource contract list-shaped for metadata inference
        # and direct callers. The callables themselves remain non-replayable.
        # Ray's physical planner detects the private iterator contract above and
        # never materializes this list for execution.
        return list(
            self._iter_read_tasks(
                parallelism,
                per_task_row_limit=per_task_row_limit,
                data_context=data_context,
            )
        )
