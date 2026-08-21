"""Run one isolated streaming Parquet -> sort -> Parquet observation."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import queue
import shutil
import statistics
import subprocess
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from .backend_stats import get_last_run_stats, value
from .data import plan_dict, smoke_slices
from .streaming_parquet_common import (
    EXPECTED_CPUS,
    EXPECTED_GPUS,
    GPU_COMMUNICATION_ENVIRONMENT,
    LOCAL_RUN_MIN_FREE_BYTES,
    SORT_KEY_STATS,
    SUPPORTED_SORT_KEYS,
    TARGET_BLOCKS,
    TARGET_DECODED_BYTES,
    TARGET_ROWS,
    exact_plan,
    write_json,
)
from .streaming_parquet_runtime import (
    ResourceMonitor,
    _gpu_reasons,
    _remaining_files,
    _start_ray,
    _sync_filesystem,
)
from .spec import load_manifest
from .streaming_parquet_e2e import (
    IMPLEMENTATION_BRANCH,
    INPUT_BUFFER_BUDGET_BYTES,
    MAX_WRITERS,
    MAX_WRITER_INPUT_BYTES,
    SAMPLE_ROWS_PER_INPUT_BLOCK,
    SMOKE_DECODED_BYTES,
    STREAMING_SORT_BASE_COMMIT,
    TARGET_FILE_BYTES,
    TARGET_ROW_GROUP_BYTES,
    _dataset_identity,
    _oom_snapshot,
)
from .worker import _SPILL_FIELDS, _spill_delta, _spill_snapshot

VALID_BACKENDS = ("gpu", "pyarrow")
VALID_WORKLOADS = ("smoke", "full")
TELEMETRY_RPC_TIMEOUT_S = 15.0
VALIDATION_BATCH_ROWS = 65_536
PROGRESS_SAMPLE_S = 30.0


def _write_phase(path: Path, phase: str, **details: Any) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "phase": phase,
        "wall_time_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns(),
        **details,
    }
    write_json(path, payload)
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
    return payload


def _compact_source_progress(value: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "events",
        "started_tasks",
        "produced_tasks",
        "failed_tasks",
        "duplicate_started_events",
        "duplicate_produced_events",
        "produced_rows",
        "produced_decoded_bytes",
        "planned_decoded_bytes",
        "produced_bytes_match_planned",
        "first_read_started_wall_time_ns",
        "last_input_produced_wall_time_ns",
    )
    return {name: value.get(name) for name in keys}


class ProgressCheckpoint:
    """Periodically persist compact capacity evidence during the timed call."""

    def __init__(
        self,
        path: Path,
        *,
        backend: str,
        source_snapshot: Callable[[], Mapping[str, Any]],
        resources: ResourceMonitor,
        sample_s: float = PROGRESS_SAMPLE_S,
    ) -> None:
        self.path = path
        self.backend = backend
        self.source_snapshot = source_snapshot
        self.resources = resources
        self.sample_s = sample_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[str] = None
        self._samples = 0
        self._logical_bytes_written = 0
        self._last: dict[str, Any] = {}

    def _sample(self) -> None:
        source = _compact_source_progress(self.source_snapshot())
        payload = {
            "schema_version": 1,
            "wall_time_ns": time.time_ns(),
            "backend": self.backend,
            "source": source,
            "cpu_stats": _load_cpu_stats() if self.backend == "pyarrow" else {},
            "resources": self.resources.to_dict(),
        }
        write_json(self.path, payload)
        self._samples += 1
        self._logical_bytes_written += self.path.stat().st_size
        self._last = payload

    def __enter__(self) -> "ProgressCheckpoint":
        def loop() -> None:
            try:
                while not self._stop.wait(self.sample_s):
                    self._sample()
            except BaseException as error:
                self._error = f"{type(error).__name__}: {error}"

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        try:
            self._sample()
        except BaseException as error:
            self._error = self._error or f"final: {type(error).__name__}: {error}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sample_interval_s": self.sample_s,
            "samples": self._samples,
            "logical_bytes_written": self._logical_bytes_written,
            "error": self._error,
            "last": self._last,
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git_identity() -> dict[str, Any]:
    root = _repo_root()

    def output(*args: str) -> str:
        return subprocess.check_output(args, cwd=root, text=True).strip()

    head = output("git", "rev-parse", "HEAD")
    branch = output("git", "branch", "--show-current")
    status = output("git", "status", "--porcelain", "--untracked-files=all")
    descended = (
        subprocess.run(
            (
                "git",
                "merge-base",
                "--is-ancestor",
                STREAMING_SORT_BASE_COMMIT,
                head,
            ),
            cwd=root,
            check=False,
        ).returncode
        == 0
    )
    return {
        "head": head,
        "branch": branch,
        "clean": not status,
        "status": status.splitlines(),
        "streaming_sort_base_is_ancestor": descended,
    }


def _safe_runtime(path: Path, *, must_not_exist: bool) -> Path:
    if not path.is_absolute():
        raise ValueError("runtime must be an absolute child of /raid")
    resolved = path.resolve(strict=False)
    raid = Path("/raid").resolve()
    if resolved == raid or not resolved.is_relative_to(raid):
        raise ValueError(f"runtime must be below /raid: {resolved}")
    if len(resolved.relative_to(raid).parts) < 3:
        raise ValueError("runtime needs at least three components below /raid")
    if must_not_exist and resolved.exists():
        raise FileExistsError(f"refusing to reuse runtime: {resolved}")
    return resolved


def _bounded_call(call: Callable[[], Any], timeout_s: float) -> dict[str, Any]:
    """Run optional telemetry with a hard driver-side wait bound."""

    result_queue: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def invoke() -> None:
        try:
            result_queue.put((True, call()))
        except BaseException as error:
            result_queue.put((False, f"{type(error).__name__}: {error}"), block=False)

    thread = threading.Thread(target=invoke, daemon=True)
    thread.start()
    try:
        ok, payload = result_queue.get(timeout=timeout_s)
    except queue.Empty:
        return {
            "ok": False,
            "timeout": True,
            "timeout_s": timeout_s,
            "error": "telemetry RPC exceeded its bounded wait",
        }
    if not ok:
        return {"ok": False, "timeout": False, "error": payload}
    return {"ok": True, "timeout": False, "value": payload}


def _safe_spill_snapshot(ray: Any) -> dict[str, Any]:
    return _bounded_call(lambda: _spill_snapshot(ray), TELEMETRY_RPC_TIMEOUT_S)


def _spill_evidence(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    if before.get("ok") and after.get("ok"):
        return {
            "available": True,
            "delta": _spill_delta(before["value"], after["value"]),
            "source": "single-node raylet cumulative GetNodeStats counters",
            "before": before["value"],
            "after": after["value"],
        }
    return {
        "available": False,
        "delta": None,
        "source": "single-node raylet cumulative GetNodeStats counters",
        "before_error": before.get("error"),
        "after_error": after.get("error"),
        "nonfatal": True,
    }


def _configure_context(backend: str, run_directory: Path) -> dict[str, Any]:
    from ray.data import DataContext
    from ray.data.context import ShuffleStrategy

    context = DataContext.get_current()
    if context.execution_options.preserve_order:
        raise RuntimeError("fresh DataContext unexpectedly preserves order")
    if context.target_max_block_size != 128 << 20:
        raise RuntimeError(
            "fresh DataContext target_max_block_size is not the 128 MiB default"
        )
    if context.use_polars or context.use_polars_sort:
        raise RuntimeError("fresh DataContext unexpectedly enables Polars")
    if backend == "pyarrow":
        if context.shuffle_strategy != ShuffleStrategy.HASH_SHUFFLE:
            raise RuntimeError(
                "fresh DataContext shuffle_strategy is not the HASH_SHUFFLE default"
            )
        context.shuffle_strategy = ShuffleStrategy.SORT_SHUFFLE_PUSH_BASED
        return {
            "backend": "pyarrow",
            "shuffle_strategy": context.shuffle_strategy.value,
            "preserve_order": context.execution_options.preserve_order,
            "target_max_block_size": context.target_max_block_size,
            "use_polars": context.use_polars,
            "use_polars_sort": context.use_polars_sort,
            "pre_override_shuffle_strategy": ShuffleStrategy.HASH_SHUFFLE.value,
            "cpu_sort_overrides": ["shuffle_strategy"],
        }

    # Preserve source order only in the GPU runtime.  It is required by the
    # streaming GPU operator and must not leak into the fresh CPU process.
    context.execution_options.preserve_order = True
    context.gpu_shuffle_num_actors = EXPECTED_GPUS
    frozen = {
        "gpu_sort_sample_seed": 0,
        "gpu_sort_streaming_sample_rows_per_block": SAMPLE_ROWS_PER_INPUT_BLOCK,
        "gpu_sort_input_buffer_budget_bytes": INPUT_BUFFER_BUDGET_BYTES,
        "gpu_sort_auto_wave_fraction": 0.50,
        "gpu_sort_exchange_batch_bytes": 512 << 20,
        "gpu_sort_run_chunk_bytes": 512 << 20,
        "gpu_sort_merge_fan_in": 4,
        "gpu_sort_external_run_store": "local_disk",
        "gpu_sort_external_run_directory": str(run_directory.resolve()),
        "gpu_sort_external_run_min_free_bytes": LOCAL_RUN_MIN_FREE_BYTES,
        "gpu_sort_external_run_max_live_bytes": None,
        "gpu_sort_ucx_tls": GPU_COMMUNICATION_ENVIRONMENT["UCX_TLS"],
        "gpu_sort_ucx_sockaddr_tls_priority": GPU_COMMUNICATION_ENVIRONMENT[
            "UCX_SOCKADDR_TLS_PRIORITY"
        ],
        "gpu_sort_ucxx_progress_mode": GPU_COMMUNICATION_ENVIRONMENT[
            "RAPIDSMPF_UCXX_PROGRESS_MODE"
        ],
    }
    for name, amount in frozen.items():
        context.set_config(name, amount)
    return {
        "backend": "gpu",
        "gpu_shuffle_num_actors": context.gpu_shuffle_num_actors,
        "preserve_order": context.execution_options.preserve_order,
        "target_max_block_size": context.target_max_block_size,
        **frozen,
    }


def _load_cpu_stats() -> dict[str, Any]:
    """Read both generic blocking and exact push-task telemetry if available."""

    from ray.data._internal.execution.operators import base_physical_operator
    from ray.data._internal.planner.exchange import (
        push_based_shuffle_task_scheduler,
    )

    generic_getter = getattr(
        base_physical_operator, "get_last_all_to_all_stats", lambda: {}
    )
    push_getter = getattr(
        push_based_shuffle_task_scheduler,
        "get_last_push_based_shuffle_stats",
        lambda: {},
    )
    generic = dict(generic_getter() or {})
    push = dict(push_getter() or {})
    if generic:
        generic["blocking_all_to_all"] = True
        generic["inputs_complete_at_ns"] = generic.get("eos_received_at_ns")
        generic["input_blocks_received"] = generic.get("input_blocks")
    if push:
        push["first_map_task_submitted_at_ns"] = push.get(
            "first_push_map_task_submitted_at_ns"
        )
    return {"blocking_all_to_all": generic, "push_based_shuffle": push}


def _schema_fingerprint(schema: Any) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _frequency_digest(counts: Mapping[str, int]) -> str:
    encoded = json.dumps(
        sorted((str(key), int(count)) for key, count in counts.items()),
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_durable_parquet(
    output_directory: Path,
    sink_result: Mapping[str, Any],
    expected_schema: Any,
    *,
    sort_key: str,
    expected_rows: int,
    expected_row_id_sum: int,
    expected_sort_key_stats: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Reopen the durable output in bounded batches after the E2E timer."""

    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    manifest_path = output_directory / "manifest.json"
    success_path = output_directory / "_SUCCESS"
    manifest_payload = manifest_path.read_bytes()
    manifest = json.loads(manifest_payload)
    success = json.loads(success_path.read_bytes())
    embedded_manifest = sink_result.get("manifest")
    embedded_success = sink_result.get("success")
    files = manifest.get("files", [])
    expected_names = [f"part-{index:05d}.parquet" for index in range(len(files))]
    names = [str(item.get("path")) for item in files]

    encoded_schema = manifest.get("schema_base64")
    manifest_schema = None
    if encoded_schema:
        import pyarrow as pa

        manifest_schema = pa.ipc.read_schema(
            pa.BufferReader(base64.b64decode(encoded_schema))
        )

    counts: dict[str, int] = {}
    rows = row_id_sum = null_sort_key_rows = 0
    global_ordered = True
    nulls_last = True
    seen_null = False
    previous_non_null_last: Optional[Any] = None
    exact_schemas = True
    arrow_schema_metadata_stored = True
    versions: set[str] = set()
    compressions: set[str] = set()
    encodings: set[str] = set()
    missing_statistics = 0
    row_groups = 0
    physical_bytes = 0
    metadata_physical_bytes = 0
    first_sort_key = last_sort_key = None
    observed_min = observed_max = None
    per_file_geometry_mismatches: list[str] = []

    for index, item in enumerate(files):
        path = output_directory / str(item["path"])
        if path.name != f"part-{index:05d}.parquet" or path.parent != output_directory:
            raise RuntimeError(f"unsafe or nondeterministic manifest path: {path}")
        parquet = pq.ParquetFile(path)
        exact_schemas = exact_schemas and parquet.schema_arrow.equals(
            expected_schema, check_metadata=True
        )
        metadata = parquet.metadata
        if metadata.num_rows != int(item["num_rows"]):
            per_file_geometry_mismatches.append(
                f"{path.name}: rows {metadata.num_rows} != {item['num_rows']}"
            )
        if metadata.num_row_groups != int(item["row_group_count"]):
            per_file_geometry_mismatches.append(
                f"{path.name}: row groups {metadata.num_row_groups} != "
                f"{item['row_group_count']}"
            )
        if int(item["decoded_bytes"]) > TARGET_FILE_BYTES:
            per_file_geometry_mismatches.append(
                f"{path.name}: decoded file bytes exceed {TARGET_FILE_BYTES}"
            )
        if int(item["max_row_group_decoded_bytes"]) > TARGET_ROW_GROUP_BYTES:
            per_file_geometry_mismatches.append(
                f"{path.name}: decoded row group exceeds {TARGET_ROW_GROUP_BYTES}"
            )
        versions.add(str(metadata.format_version))
        file_metadata = metadata.metadata or {}
        arrow_schema_metadata_stored = (
            arrow_schema_metadata_stored and b"ARROW:schema" in file_metadata
        )
        physical = path.stat().st_size
        physical_bytes += physical
        metadata_physical_bytes += int(item["file_size_bytes"])
        if physical != int(item["file_size_bytes"]):
            raise RuntimeError(f"Parquet file size changed after commit: {path}")
        row_groups += metadata.num_row_groups
        for group_index in range(metadata.num_row_groups):
            group = metadata.row_group(group_index)
            for column_index in range(group.num_columns):
                column = group.column(column_index)
                compressions.add(str(column.compression).upper())
                encodings.update(str(value) for value in column.encodings)
                if column.statistics is None:
                    missing_statistics += 1

        for batch in parquet.iter_batches(
            batch_size=VALIDATION_BATCH_ROWS,
            columns=[sort_key, "row_id"],
            use_threads=True,
        ):
            key_values = batch.column(0)
            row_ids = batch.column(1)
            batch_nulls = key_values.null_count
            null_sort_key_rows += batch_nulls
            non_null_count = len(key_values) - batch_nulls
            if seen_null and non_null_count:
                nulls_last = False
            non_null_values = key_values.slice(0, non_null_count)
            if batch_nulls:
                null_suffix = key_values.slice(non_null_count)
                misplaced_null = bool(
                    non_null_values.null_count or null_suffix.null_count != batch_nulls
                )
                if misplaced_null:
                    nulls_last = False
                    non_null_values = pc.drop_null(key_values)
                    non_null_count = len(non_null_values)
                seen_null = True
            if non_null_count:
                batch_first = non_null_values[0].as_py()
                batch_last = non_null_values[-1].as_py()
                if previous_non_null_last is not None:
                    global_ordered = (
                        global_ordered and previous_non_null_last <= batch_first
                    )
                if non_null_count > 1:
                    global_ordered = global_ordered and bool(
                        pc.all(
                            pc.less_equal(
                                non_null_values.slice(0, non_null_count - 1),
                                non_null_values.slice(1),
                            )
                        ).as_py()
                    )
                if first_sort_key is None:
                    first_sort_key = batch_first
                    observed_min = batch_first
                previous_non_null_last = batch_last
                last_sort_key = batch_last
                observed_max = batch_last
            if batch_nulls:
                last_sort_key = None
            for frequency in pc.value_counts(key_values):
                key = frequency["values"].as_py()
                count = int(frequency["counts"].as_py())
                if key is not None:
                    text_key = str(key)
                    counts[text_key] = counts.get(text_key, 0) + count
            row_id_sum += int(pc.sum(row_ids).as_py() or 0)
            rows += batch.num_rows

    partials = sorted(
        str(path.relative_to(output_directory))
        for path in output_directory.rglob("*.partial")
    )
    staged = sorted(
        str(path.relative_to(output_directory))
        for path in output_directory.rglob("*.staged")
    )
    actual_parquet_names = sorted(
        path.name for path in output_directory.glob("*.parquet")
    )
    file_sizes = [int(item["file_size_bytes"]) for item in files]
    writer_settings = manifest.get("writer", {})
    expected_writer_settings = {
        "parquet_version": "2.6",
        "compression": "zstd",
        "compression_level": 3,
        "dictionary_encoding": True,
        "write_statistics": True,
        "data_page_bytes": 1 << 20,
        "row_group_decoded_bytes": TARGET_ROW_GROUP_BYTES,
        "max_decoded_file_bytes": TARGET_FILE_BYTES,
        "target_bytes_per_write": TARGET_FILE_BYTES,
        "recommended_max_writers": MAX_WRITERS,
    }
    manifest_totals = manifest.get("totals", {})
    manifest_totals_consistent = (
        int(manifest_totals.get("num_rows", -1))
        == sum(int(item["num_rows"]) for item in files)
        and int(manifest_totals.get("decoded_bytes", -1))
        == sum(int(item["decoded_bytes"]) for item in files)
        and int(manifest_totals.get("file_size_bytes", -1)) == sum(file_sizes)
        and int(manifest_totals.get("file_count", -1)) == len(files)
    )
    result = {
        "source": (
            "bounded ParquetFile.iter_batches reopen after sink syncfs and outside "
            "the primary E2E timer"
        ),
        "sort_key": sort_key,
        "sort_key_arrow_type": str(expected_schema.field(sort_key).type),
        "rows": rows,
        "row_id_sum": row_id_sum,
        "null_sort_key_rows": null_sort_key_rows,
        "sort_key_cardinality": len(counts),
        "sort_key_frequency_digest": _frequency_digest(counts),
        "sort_key_min": observed_min,
        "sort_key_max": observed_max,
        "ordered": global_ordered,
        "nulls_last": nulls_last,
        "first_sort_key": first_sort_key,
        "last_sort_key": last_sort_key,
        "exact_schema": exact_schemas,
        "expected_schema_fingerprint": _schema_fingerprint(expected_schema),
        "manifest_schema_exact": (
            manifest_schema is not None
            and manifest_schema.equals(expected_schema, check_metadata=True)
        ),
        "arrow_schema_metadata_stored": arrow_schema_metadata_stored,
        "parquet_format_versions": sorted(versions),
        "compressions": sorted(compressions),
        "encodings": sorted(encodings),
        "missing_column_statistics": missing_statistics,
        "file_count": len(files),
        "row_group_count": row_groups,
        "physical_bytes": physical_bytes,
        "manifest_physical_bytes": metadata_physical_bytes,
        "manifest_writer_settings": writer_settings,
        "expected_writer_settings": expected_writer_settings,
        "manifest_writer_settings_exact": writer_settings == expected_writer_settings,
        "manifest_totals_consistent": manifest_totals_consistent,
        "per_file_geometry_mismatches": per_file_geometry_mismatches,
        "file_size_bytes": {
            "min": min(file_sizes, default=0),
            "median": statistics.median(file_sizes) if file_sizes else 0,
            "max": max(file_sizes, default=0),
        },
        "deterministic_names": names == expected_names,
        "actual_files_match_manifest": actual_parquet_names == expected_names,
        "partials": partials,
        "staged_files": staged,
        "manifest_matches_driver_receipt": manifest == embedded_manifest,
        "success_matches_driver_receipt": success == embedded_success,
        "success_manifest": success.get("manifest"),
        "success_manifest_sha256": success.get("manifest_sha256"),
        "actual_manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
    }
    expected = {
        "sort_key": sort_key,
        "sort_key_arrow_type": (
            expected_sort_key_stats.get("arrow_type")
            if expected_sort_key_stats is not None
            and expected_sort_key_stats.get("arrow_type") is not None
            else str(expected_schema.field(sort_key).type)
        ),
        "rows": expected_rows,
        "row_id_sum": expected_row_id_sum,
        "ordered": True,
        "nulls_last": True,
        "exact_schema": True,
        "manifest_schema_exact": True,
        "arrow_schema_metadata_stored": True,
        "parquet_format_versions": ["2.6"],
        "compressions": ["ZSTD"],
        "missing_column_statistics": 0,
        "manifest_writer_settings_exact": True,
        "manifest_totals_consistent": True,
        "per_file_geometry_mismatches": [],
        "deterministic_names": True,
        "actual_files_match_manifest": True,
        "partials": [],
        "staged_files": [],
        "manifest_matches_driver_receipt": True,
        "success_matches_driver_receipt": True,
        "success_manifest": "manifest.json",
        "success_manifest_sha256": result["actual_manifest_sha256"],
    }
    if expected_sort_key_stats is not None:
        for source_name, result_name in (
            ("null_rows", "null_sort_key_rows"),
            ("cardinality", "sort_key_cardinality"),
            ("frequency_digest", "sort_key_frequency_digest"),
            ("min", "sort_key_min"),
            ("max", "sort_key_max"),
        ):
            if source_name in expected_sort_key_stats:
                expected[result_name] = expected_sort_key_stats[source_name]
    result["rejection_reasons"] = [
        f"{name}={result.get(name)!r}, expected {wanted!r}"
        for name, wanted in expected.items()
        if result.get(name) != wanted
    ]
    if not ({"RLE_DICTIONARY", "PLAIN_DICTIONARY"} & encodings):
        result["rejection_reasons"].append(
            "Parquet metadata contains no dictionary encoding"
        )
    result["valid"] = not result["rejection_reasons"]
    return result


def _timeline(
    *,
    e2e_started_wall_ns: int,
    e2e_finished_wall_ns: int,
    source: Mapping[str, Any],
    sink: Mapping[str, Any],
    gpu: Mapping[str, Any],
    cpu: Mapping[str, Any],
) -> dict[str, Any]:
    sink_telemetry = sink.get("telemetry", {})
    blocking = cpu.get("blocking_all_to_all", {})
    push = cpu.get("push_based_shuffle", {})
    gpu_processing_times = [
        int(item["first_ingest_started_at_ns"])
        for item in gpu.get("ranks", [])
        if item.get("first_ingest_started_at_ns") is not None
    ]
    events = {
        "e2e_started_at_ns": e2e_started_wall_ns,
        "first_input_read_at_ns": source.get("first_read_started_wall_time_ns"),
        "last_input_produced_at_ns": source.get("last_input_produced_wall_time_ns"),
        "gpu_ranks_started_at_ns": gpu.get("ranks_started_at_ns"),
        "first_gpu_processing_at_ns": min(gpu_processing_times, default=None),
        "first_gpu_run_committed_at_ns": gpu.get("first_gpu_run_committed_at_ns"),
        "gpu_eos_at_ns": gpu.get("inputs_complete_at_ns"),
        "gpu_finalization_started_at_ns": gpu.get("finalization_started_at_ns"),
        "first_gpu_output_ready_at_ns": gpu.get("first_output_bundle_ready_at_ns"),
        "last_gpu_output_ready_at_ns": gpu.get("last_output_bundle_ready_at_ns"),
        "gpu_finalization_complete_at_ns": gpu.get("finalization_complete_at_ns"),
        "cpu_eos_at_ns": blocking.get("inputs_complete_at_ns"),
        "first_cpu_shuffle_task_submitted_at_ns": push.get(
            "first_map_task_submitted_at_ns"
        ),
        "first_parquet_writer_at_ns": sink_telemetry.get(
            "first_writer_work_started_at_ns"
        ),
        "first_parquet_output_at_ns": sink_telemetry.get(
            "first_parquet_file_completed_at_ns"
        ),
        "last_parquet_output_at_ns": sink_telemetry.get(
            "last_parquet_file_completed_at_ns"
        ),
        "manifest_closed_at_ns": sink_telemetry.get("manifest_closed_at_ns"),
        "success_closed_at_ns": sink_telemetry.get("success_closed_at_ns"),
        "drain_complete_at_ns": sink_telemetry.get("drain_completed_at_ns"),
        "e2e_finished_at_ns": e2e_finished_wall_ns,
    }
    last_input = events["last_input_produced_at_ns"]
    first_run = events["first_gpu_run_committed_at_ns"]
    gpu_done = events["gpu_finalization_complete_at_ns"]
    first_writer = events["first_parquet_writer_at_ns"]
    return {
        "clock": "time.time_ns wall clock shared by local DGX processes",
        "events": events,
        "overlap_proofs": {
            "read_and_gpu_run_creation": bool(
                first_run is not None
                and last_input is not None
                and int(first_run) < int(last_input)
            ),
            "gpu_merge_or_extraction_and_parquet_write": bool(
                first_writer is not None
                and gpu_done is not None
                and int(first_writer) < int(gpu_done)
            ),
            "cpu_shuffle_started_before_eos": bool(
                push.get("first_map_task_submitted_at_ns") is not None
                and blocking.get("inputs_complete_at_ns") is not None
                and int(push["first_map_task_submitted_at_ns"])
                < int(blocking["inputs_complete_at_ns"])
            ),
        },
    }


def _capacity_failure(error: BaseException) -> bool:
    text = f"{type(error).__name__}: {error}".lower()
    tokens = (
        "outofmemory",
        "out of memory",
        "memoryerror",
        "object store is full",
        "objectstorefull",
        "no space left on device",
        "disk quota exceeded",
        "killed due to memory pressure",
    )
    return any(token in text for token in tokens)


def _timed_pipeline_capacity_failure(
    args: argparse.Namespace, state: Mapping[str, Any]
) -> bool:
    """Return whether the terminal write itself proved a CPU capacity failure.

    A later validation, telemetry, or cleanup failure must never be relabeled as
    a sort capacity result merely because the primary timer had run earlier.
    """

    return bool(
        args.backend == "pyarrow"
        and args.workload == "full"
        and state.get("timed_pipeline_started") is True
        and state.get("timed_pipeline_finished") is True
        and state.get("pipeline_succeeded") is False
        and state.get("pipeline_failure_is_capacity") is True
    )


def _run_trial(args: argparse.Namespace, state: dict[str, Any]) -> dict[str, Any]:
    identity = _git_identity()
    if (
        not identity["clean"]
        or identity["branch"] != IMPLEMENTATION_BRANCH
        or identity["head"] != args.expected_git_head
        or not identity["streaming_sort_base_is_ancestor"]
    ):
        raise RuntimeError(f"worker checkout is not frozen: {identity}")

    dataset_identity = _dataset_identity(args.dataset_root, verify_parquet_files=True)
    state["dataset_identity"] = dataset_identity
    if dataset_identity["identity_digest"] != args.expected_dataset_identity_digest:
        raise RuntimeError("worker Parquet source identity changed before execution")

    runtime = _safe_runtime(args.runtime, must_not_exist=True)
    runtime.mkdir(parents=True, exist_ok=False)
    spill_directory = runtime / "ray-spill"
    run_directory = runtime / "gpu-runs"
    output_directory = runtime / "output"
    telemetry_directory = runtime / "telemetry"
    source_telemetry_directory = telemetry_directory / "source"
    for path in (spill_directory, run_directory, output_directory, telemetry_directory):
        path.mkdir()
    phase_path = telemetry_directory / "phase.json"
    progress_path = telemetry_directory / "progress.json"
    state.update(
        {
            "phase_path": str(phase_path),
            "progress_path": str(progress_path),
            "runtime": str(runtime),
        }
    )
    _write_phase(phase_path, "preparing", timed_pipeline_started=False)

    if args.backend == "gpu":
        os.environ.update(GPU_COMMUNICATION_ENVIRONMENT)
    ray = None
    owned = None
    try:
        ray, owned = _start_ray(runtime, args.backend)
        context = _configure_context(args.backend, run_directory)

        from .streaming_parquet_sink import StreamingParquetDatasink
        from .streaming_parquet_source import (
            FrozenBTSPlanDatasource,
            exact_block_size_bytes,
            read_source_telemetry,
            schema_from_manifest,
        )

        manifest = load_manifest(args.dataset_root)
        expected_schema = schema_from_manifest(manifest)
        expected_columns = (*tuple(manifest["schema_names"]), "row_id")
        if tuple(expected_schema.names) != expected_columns:
            raise RuntimeError("source schema and frozen 110-column payload differ")
        if (
            str(expected_schema.field(args.sort_key).type)
            != SORT_KEY_STATS[args.sort_key]["arrow_type"]
        ):
            raise RuntimeError(f"sort-key type changed: {args.sort_key}")
        if args.workload == "full":
            plan, _ = exact_plan(args.dataset_root)
            expected_rows = TARGET_ROWS
            expected_blocks = TARGET_BLOCKS
            expected_decoded_bytes = TARGET_DECODED_BYTES
        else:
            slices = smoke_slices(manifest)
            plan = plan_dict(slices, kind="streaming-smoke")
            expected_rows = int(plan["rows"])
            expected_blocks = int(plan["blocks"])
            expected_decoded_bytes = SMOKE_DECODED_BYTES
        if plan["digest"] != args.expected_plan_digest:
            raise RuntimeError(
                f"plan digest changed: {plan['digest']} != {args.expected_plan_digest}"
            )

        block_size_bytes = exact_block_size_bytes(plan, manifest)
        if sum(block_size_bytes) != expected_decoded_bytes:
            raise RuntimeError("exact source byte metadata changed")
        source = FrozenBTSPlanDatasource(
            plan,
            schema=expected_schema,
            columns=expected_columns,
            telemetry_directory=source_telemetry_directory,
            expected_plan_digest=args.expected_plan_digest,
            expected_rows=expected_rows,
            expected_blocks=expected_blocks,
            expected_decoded_bytes=expected_decoded_bytes,
            block_size_bytes=block_size_bytes,
        )
        dataset = ray.data.read_datasource(
            source,
            override_num_blocks=expected_blocks,
            concurrency=EXPECTED_CPUS,
        )
        sort_kwargs: dict[str, Any] = {
            "key": [args.sort_key],
            "descending": [False],
        }
        if args.backend == "gpu":
            sort_kwargs["backend"] = "gpu"
        sorted_dataset = dataset.sort(**sort_kwargs)
        sink = StreamingParquetDatasink(
            output_directory,
            max_decoded_file_bytes=TARGET_FILE_BYTES,
            row_group_decoded_bytes=TARGET_ROW_GROUP_BYTES,
            target_bytes_per_write=TARGET_FILE_BYTES,
            recommended_max_writers=MAX_WRITERS,
        )

        construction_files = sorted(
            str(path.relative_to(runtime))
            for path in runtime.rglob("*")
            if path.is_file()
        )
        source_before = read_source_telemetry(
            source_telemetry_directory,
            plan_digest=args.expected_plan_digest,
            expected_blocks=expected_blocks,
        )
        if source_before["events"] != 0:
            raise RuntimeError("lazy graph construction executed a source reader")
        if any(output_directory.iterdir()):
            raise RuntimeError("lazy graph construction wrote sink output")

        _write_phase(
            phase_path,
            "pipeline-ready",
            timed_pipeline_started=False,
            construction_files=construction_files,
        )
        pre_timer_drain = _sync_filesystem(runtime)
        # This is a fresh isolated runtime and no Ray data work occurs before the
        # primary timer.  A zero baseline is therefore exact and avoids allowing
        # a timed-out diagnostic RPC thread to leak into the measured interval.
        spill_before = {
            "ok": True,
            "timeout": False,
            "value": {name: 0 for name in _SPILL_FIELDS},
            "source": "fresh isolated Ray runtime before its first data action",
        }

        monitor = ResourceMonitor()
        pipeline_error: Optional[BaseException] = None
        progress = ProgressCheckpoint(
            progress_path,
            backend=args.backend,
            source_snapshot=lambda: read_source_telemetry(
                source_telemetry_directory,
                plan_digest=args.expected_plan_digest,
                expected_blocks=expected_blocks,
            ),
            resources=monitor,
        )
        progress_started = False
        try:
            progress.__enter__()
            progress_started = True
            _write_phase(
                phase_path,
                "pipeline-arming",
                timed_pipeline_started=False,
            )
            with monitor:
                # Resource/NVML initialization can fail or hang independently of
                # the Dataset action. Keep that work durably outside the timed
                # capacity phase, then arm the watchdog immediately before the
                # primary clock and terminal call.
                timed_oom_baseline = _oom_snapshot()
                running_phase = _write_phase(
                    phase_path,
                    "pipeline-running",
                    timed_pipeline_started=True,
                    kernel_oom_baseline=timed_oom_baseline,
                )
                state.update(
                    {
                        "timed_pipeline_started": True,
                        "pipeline_running_wall_ns": running_phase["wall_time_ns"],
                        "pipeline_running_monotonic_ns": running_phase["monotonic_ns"],
                    }
                )
                e2e_started_wall_ns = time.time_ns()
                # Keep the primary timer immediately adjacent to the only
                # terminal Dataset action.  All checkpoint/phase setup is done.
                e2e_started_ns = time.perf_counter_ns()
                try:
                    sorted_dataset.write_datasink(sink, concurrency=MAX_WRITERS)
                except BaseException as error:
                    pipeline_error = error
                finally:
                    e2e_finished_ns = time.perf_counter_ns()
                    e2e_finished_wall_ns = time.time_ns()
                    timed_oom_finished = _oom_snapshot()
                    state.update(
                        {
                            "e2e_started_wall_ns": e2e_started_wall_ns,
                            "e2e_started_ns": e2e_started_ns,
                            "e2e_finished_ns": e2e_finished_ns,
                            "e2e_finished_wall_ns": e2e_finished_wall_ns,
                            "timed_pipeline_finished": True,
                            "pipeline_succeeded": pipeline_error is None,
                            "pipeline_failure_is_capacity": bool(
                                pipeline_error is not None
                                and _capacity_failure(pipeline_error)
                            ),
                        }
                    )
                    # End the controller's primary-phase watchdog before the
                    # resource monitor and progress checkpoint perform their
                    # excluded teardown work.
                    _write_phase(
                        phase_path,
                        "pipeline-finished",
                        timed_pipeline_started=True,
                        timed_pipeline_finished=True,
                        pipeline_succeeded=pipeline_error is None,
                        pipeline_failure_is_capacity=bool(
                            pipeline_error is not None
                            and _capacity_failure(pipeline_error)
                        ),
                        pipeline_failure=(
                            f"{type(pipeline_error).__name__}: {pipeline_error}"
                            if pipeline_error is not None
                            else None
                        ),
                        e2e_started_wall_ns=e2e_started_wall_ns,
                        e2e_finished_wall_ns=e2e_finished_wall_ns,
                        e2e_elapsed_s=(e2e_finished_ns - e2e_started_ns) / 1e9,
                        kernel_oom_baseline=timed_oom_baseline,
                        kernel_oom_finished=timed_oom_finished,
                    )
        finally:
            if progress_started:
                progress.__exit__(None, None, None)

        spill_after = _safe_spill_snapshot(ray)
        source_stats = read_source_telemetry(
            source_telemetry_directory,
            plan_digest=args.expected_plan_digest,
            expected_blocks=expected_blocks,
        )
        gpu_stats = get_last_run_stats() if args.backend == "gpu" else {}
        cpu_stats = _load_cpu_stats() if args.backend == "pyarrow" else {}
        run_files = _remaining_files(run_directory)
        if pipeline_error is not None:
            state["failure_evidence"] = {
                "source": source_stats,
                "sink_telemetry": sink.telemetry,
                "gpu_stats": gpu_stats,
                "cpu_stats": cpu_stats,
                "ray_object_store_io": _spill_evidence(spill_before, spill_after),
                "resources": monitor.to_dict(),
                "progress_checkpoint": progress.to_dict(),
                "run_directory_remaining_files": run_files,
                "elapsed_s": (e2e_finished_ns - e2e_started_ns) / 1e9,
            }
            raise pipeline_error

        sink_result = sink.result
        if sink_result is None:
            raise RuntimeError("terminal Datasink returned without a durable receipt")

        state["failure_evidence"] = {
            "source": source_stats,
            "sink": sink_result,
            "gpu_stats": gpu_stats,
            "cpu_stats": cpu_stats,
            "ray_object_store_io": _spill_evidence(spill_before, spill_after),
            "resources": monitor.to_dict(),
            "progress_checkpoint": progress.to_dict(),
            "run_directory_remaining_files": run_files,
            "elapsed_s": (e2e_finished_ns - e2e_started_ns) / 1e9,
        }
        _write_phase(
            phase_path,
            "validation-running",
            timed_pipeline_started=True,
            timed_pipeline_finished=True,
            pipeline_succeeded=True,
            pipeline_failure_is_capacity=False,
            e2e_started_wall_ns=e2e_started_wall_ns,
            e2e_finished_wall_ns=e2e_finished_wall_ns,
            e2e_elapsed_s=(e2e_finished_ns - e2e_started_ns) / 1e9,
        )
        durable = _validate_durable_parquet(
            output_directory,
            sink_result,
            expected_schema,
            sort_key=args.sort_key,
            expected_rows=expected_rows,
            expected_row_id_sum=expected_rows * (expected_rows - 1) // 2,
            expected_sort_key_stats=(
                SORT_KEY_STATS[args.sort_key]
                if args.workload == "full"
                else SORT_KEY_STATS[args.sort_key]["smoke"]
            ),
        )
        _write_phase(
            phase_path,
            "validation-finished",
            timed_pipeline_started=True,
            timed_pipeline_finished=True,
            pipeline_succeeded=True,
            pipeline_failure_is_capacity=False,
            e2e_started_wall_ns=e2e_started_wall_ns,
            e2e_finished_wall_ns=e2e_finished_wall_ns,
            e2e_elapsed_s=(e2e_finished_ns - e2e_started_ns) / 1e9,
            durable_validation_valid=bool(durable.get("valid")),
        )

        reasons: list[str] = []
        resource_stats = monitor.to_dict()
        progress_stats = progress.to_dict()
        if resource_stats.get("error"):
            reasons.append(f"resource monitoring failed: {resource_stats['error']}")
        if int(resource_stats.get("samples", 0)) < 2:
            reasons.append("resource monitoring produced fewer than two samples")
        if progress_stats.get("error"):
            reasons.append(f"capacity checkpointing failed: {progress_stats['error']}")
        if not source_stats["exact_once_complete"]:
            reasons.append("lazy source was not consumed exactly once")
        if source_stats["produced_rows"] != expected_rows:
            reasons.append("lazy source row count differs from the frozen plan")
        if not source_stats.get("produced_bytes_match_planned"):
            reasons.append("lazy source per-block byte metadata was not exact")
        if args.workload == "full" and (
            source_stats["produced_decoded_bytes"] != TARGET_DECODED_BYTES
        ):
            reasons.append(
                "lazy source decoded bytes differ from the frozen 1 TB contract"
            )
        sink_telemetry = sink_result["telemetry"]
        if not sink_telemetry.get("transaction_committed"):
            reasons.append("Parquet sink transaction was not durably committed")
        if int(sink_telemetry.get("num_rows", -1)) != expected_rows:
            reasons.append("Parquet sink row count differs from the source")
        if (
            int(sink_telemetry.get("writer_input_decoded_bytes_max", 0))
            > TARGET_FILE_BYTES
        ):
            reasons.append("one writer task exceeded its 4 GiB decoded input target")
        if int(sink_telemetry.get("configured_peak_writer_input_bytes", -1)) != (
            MAX_WRITER_INPUT_BYTES
        ):
            reasons.append("sink did not preserve the frozen 64 GiB writer bound")
        if int(sink_telemetry.get("writer_input_target_overshoot_tasks", -1)) != 0:
            reasons.append("a writer task exceeded the hard 4 GiB input bound")
        if not durable.get("valid"):
            reasons.extend(
                f"durable Parquet: {item}"
                for item in durable.get("rejection_reasons", [])
            )
        if run_files:
            reasons.append(f"GPU run directory leaked files: {run_files[:8]}")

        if args.backend == "gpu":
            reasons.extend(_gpu_reasons(gpu_stats, "local_disk"))
            if int(value(gpu_stats, "input_rows", default=-1)) != expected_rows:
                reasons.append("GPU telemetry input rows differ from the source")
            input_refs = int(value(gpu_stats, "input_object_refs_received", default=-1))
            released_refs = int(
                value(gpu_stats, "released_input_object_refs", default=-1)
            )
            if input_refs < expected_blocks:
                reasons.append("GPU received fewer refs than planned source blocks")
            if released_refs != input_refs:
                reasons.append("GPU did not release every source ObjectRef")
            if not bool(
                value(gpu_stats, "all_input_object_refs_released", default=False)
            ):
                reasons.append("GPU ObjectRef release proof failed")
            if not bool(value(gpu_stats, "input_buffer_within_bound", default=False)):
                reasons.append("GPU input buffer exceeded budget plus one block")
            first_run = value(gpu_stats, "first_gpu_run_committed_at_ns", default=None)
            last_source = source_stats.get("last_input_produced_wall_time_ns")
            if args.workload == "full":
                if (
                    first_run is None
                    or last_source is None
                    or int(first_run) >= int(last_source)
                ):
                    reasons.append(
                        "GPU did not commit a local run before source completion"
                    )
                if not bool(
                    value(
                        gpu_stats,
                        "gpu_processing_began_before_eos",
                        default=False,
                    )
                ):
                    reasons.append("GPU processing did not begin before EOS")
        else:
            blocking = cpu_stats.get("blocking_all_to_all", {})
            push = cpu_stats.get("push_based_shuffle", {})
            first_map = push.get("first_map_task_submitted_at_ns")
            eos = blocking.get("inputs_complete_at_ns")
            if not blocking.get("blocking_all_to_all"):
                reasons.append("CPU sort did not report stock blocking AllToAll")
            if int(blocking.get("input_blocks_received", -1)) < expected_blocks:
                reasons.append("CPU AllToAll received fewer than the source blocks")
            if first_map is None or eos is None or int(first_map) < int(eos):
                reasons.append(
                    "CPU push task timing does not prove post-EOS submission"
                )

        timeline = _timeline(
            e2e_started_wall_ns=e2e_started_wall_ns,
            e2e_finished_wall_ns=e2e_finished_wall_ns,
            source=source_stats,
            sink=sink_result,
            gpu=gpu_stats,
            cpu=cpu_stats,
        )
        if (
            args.backend == "gpu"
            and args.workload == "full"
            and not timeline["overlap_proofs"]["read_and_gpu_run_creation"]
        ):
            reasons.append("timeline lacks read/GPU-run overlap")
        if (
            args.backend == "pyarrow"
            and timeline["overlap_proofs"]["cpu_shuffle_started_before_eos"]
        ):
            reasons.append("CPU push unexpectedly started before EOS")

        e2e_s = (e2e_finished_ns - e2e_started_ns) / 1e9
        artifact = {
            "valid": not reasons,
            "status": "accepted" if not reasons else "rejected",
            "trial_name": args.trial_name,
            "backend": args.backend,
            "workload": args.workload,
            "sort_key": args.sort_key,
            "timed_pipeline_started": True,
            "timed_pipeline_finished": True,
            "pipeline_succeeded": True,
            "pipeline_failure_is_capacity": False,
            "git": identity,
            "dataset_identity": dataset_identity,
            "plan": {
                "kind": plan["kind"],
                "rows": expected_rows,
                "blocks": expected_blocks,
                "decoded_bytes": expected_decoded_bytes,
                "digest": plan["digest"],
            },
            "context": context,
            "lazy_graph": {
                "construction_submitted_reader_tasks": False,
                "construction_reader_descriptor_object_refs": 0,
                "construction_source_events": source_before["events"],
                "construction_files": construction_files,
                "terminal_action": "sorted_dataset.write_datasink",
                "forbidden_intermediate_actions": [],
                "source_non_replayable": True,
                "planned_read_tasks": expected_blocks,
                "read_task_submission_mode": "bounded-one-shot",
                "read_task_descriptor_window": source._read_task_submission_window,
            },
            "timings_s": {
                "e2e": e2e_s,
                "filesystem_drain": sink_telemetry.get("syncfs_s"),
                "sink_commit": sink_telemetry.get("commit_elapsed_s"),
                "parquet_writer_wall_rank_sum": sink_telemetry.get(
                    "writer_elapsed_s_sum"
                ),
                "parquet_encoding_wall_rank_sum": (
                    float(sink_telemetry.get("parquet_write_s_sum", 0.0))
                    + float(sink_telemetry.get("parquet_close_s_sum", 0.0))
                ),
                "parquet_encoding_process_cpu_rank_sum": sink_telemetry.get(
                    "parquet_encoding_cpu_s_sum"
                ),
            },
            "timeline": timeline,
            "source": source_stats,
            "sink": sink_result,
            "durable_validation": durable,
            "gpu_stats": gpu_stats,
            "cpu_stats": cpu_stats,
            "ray_object_store_io": _spill_evidence(spill_before, spill_after),
            "resources": resource_stats,
            "progress_checkpoint": progress_stats,
            "pre_timer_filesystem_drain": {
                **pre_timer_drain,
                "outside_primary_timer": True,
            },
            "run_directory_remaining_files": run_files,
            "rejection_reasons": reasons,
        }
        return artifact
    finally:
        if ray is not None:
            try:
                ray.shutdown()
            except BaseException:
                pass
        if owned is not None:
            resolved = owned.resolve()
            if resolved.parent == Path("/dev/shm/rgs") and len(resolved.name) == 12:
                shutil.rmtree(resolved, ignore_errors=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--trial-name", required=True)
    parser.add_argument("--backend", choices=VALID_BACKENDS, required=True)
    parser.add_argument("--workload", choices=VALID_WORKLOADS, required=True)
    parser.add_argument("--sort-key", choices=SUPPORTED_SORT_KEYS, required=True)
    parser.add_argument("--expected-plan-digest", required=True)
    parser.add_argument("--expected-git-head", required=True)
    parser.add_argument("--expected-dataset-identity-digest", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    state: dict[str, Any] = {"timed_pipeline_started": False}
    result: dict[str, Any]
    try:
        result = _run_trial(args, state)
    except BaseException as error:
        capacity = _timed_pipeline_capacity_failure(args, state)
        elapsed = None
        if (
            state.get("e2e_started_ns") is not None
            and state.get("e2e_finished_ns") is not None
        ):
            elapsed = (
                int(state["e2e_finished_ns"]) - int(state["e2e_started_ns"])
            ) / 1e9
        result = {
            "valid": False,
            "status": "capacity-failure" if capacity else "rejected",
            "trial_name": args.trial_name,
            "backend": args.backend,
            "workload": args.workload,
            "sort_key": args.sort_key,
            "dataset_identity": state.get("dataset_identity"),
            "timed_pipeline_started": bool(state.get("timed_pipeline_started")),
            "timed_pipeline_finished": bool(state.get("timed_pipeline_finished")),
            "pipeline_succeeded": state.get("pipeline_succeeded"),
            "pipeline_failure_is_capacity": bool(
                state.get("pipeline_failure_is_capacity")
            ),
            "timings_s": {"e2e_until_failure": elapsed},
            "rejection_reasons": [f"{type(error).__name__}: {error}"],
            "exception": traceback.format_exc(),
        }
        if state.get("failure_evidence") is not None:
            result["failure_evidence"] = state["failure_evidence"]
        phase_value = state.get("phase_path")
        if phase_value is not None and Path(phase_value).is_file():
            result["last_durable_phase"] = json.loads(
                Path(phase_value).read_text(encoding="utf-8")
            )
    write_json(args.output, result)
    phase_value = state.get("phase_path")
    if phase_value is not None:
        try:
            elapsed = None
            if (
                state.get("e2e_started_ns") is not None
                and state.get("e2e_finished_ns") is not None
            ):
                elapsed = (
                    int(state["e2e_finished_ns"]) - int(state["e2e_started_ns"])
                ) / 1e9
            _write_phase(
                Path(phase_value),
                "worker-complete",
                timed_pipeline_started=bool(state.get("timed_pipeline_started")),
                timed_pipeline_finished=bool(state.get("timed_pipeline_finished")),
                pipeline_succeeded=state.get("pipeline_succeeded"),
                pipeline_failure_is_capacity=bool(
                    state.get("pipeline_failure_is_capacity")
                ),
                e2e_started_wall_ns=state.get("e2e_started_wall_ns"),
                e2e_finished_wall_ns=state.get("e2e_finished_wall_ns"),
                e2e_elapsed_s=elapsed,
                status=result.get("status"),
                valid=bool(result.get("valid")),
            )
        except BaseException:
            # The result artifact is authoritative after normal worker return.
            pass
    return (
        0
        if result.get("valid")
        else (2 if result["status"] == "capacity-failure" else 1)
    )


if __name__ == "__main__":
    raise SystemExit(main())
