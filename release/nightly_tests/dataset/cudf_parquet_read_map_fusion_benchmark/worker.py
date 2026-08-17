#!/usr/bin/env python3
"""Run one cold S3 Parquet-to-GPU-tokenizer observation on an existing Ray cluster.

This is benchmark-only code.  The stopwatch starts immediately before the
ordinary ``read_parquet`` call and stops when the mapped Dataset is materialized.
Plan inspection, checksums, and correctness checks intentionally run afterward.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse

APPROVED_BATCH_SIZES = (8_388_608, 16_777_216, 33_554_432)
APPROVED_FUSED_ACTOR_MAX_CONCURRENCY = (1, 2, 3)
APPROVED_MAP_BATCHES_MAX_CONCURRENCY = (1, 2, 3, 4)
APPROVED_NUM_GPUS_PER_ACTOR = (0.5, 1.0)
APPROVED_PARQUET_ROW_GROUP_CHUNK_MULTIPLIERS = (1, 2, 4)
APPROVED_NODE_COUNTS = (8,)
CPUS_PER_NODE = 16
GPUS_PER_NODE = 1
MERCHANT_HASH_SIZE = 2_000
CHECKSUM_ALGORITHM = "tfm-tokenizer-uint64-commutative-v1"
SPILL_FIELDS = (
    "spilled_bytes_total",
    "spilled_objects_total",
    "restored_bytes_total",
    "restored_objects_total",
    "spill_time_total_s",
    "restore_time_total_s",
)
TOKENIZER_COLUMNS = (
    "User",
    "Card",
    "Year",
    "Month",
    "Day",
    "Time",
    "Amount",
    "Use Chip",
    "Merchant Name",
    "Merchant State",
    "Zip",
    "MCC",
    "Is Fraud?",
)
OUTPUT_COLUMNS = ("uc_key", "ts", "token_ids", "label")
UINT64_MASK = (1 << 64) - 1
CRITICAL_PROFILE_MARKER = "FUSION_CRITICAL_PROFILE="
CRITICAL_PROFILE_SUPPORT = "fusion_critical_profile_support.py"
CRITICAL_PROFILE_STAGE_GROUPS = {
    "cudf_s3_read_decode": ("cudf_read",),
    "gpu_tokenizer": ("udf",),
    "batch_to_block": ("batch_to_block",),
    "output_buffer": ("output_buffer_add_batch", "output_buffer_next"),
    "yield_backpressure": (
        "block_yield_suspended",
        "metadata_yield_suspended",
    ),
}


def _mount_device_number(
    mountinfo: str, mountpoint: str = "/mnt/nvme"
) -> tuple[int, int] | None:
    """Return the kernel device number for the visible NVMe bind mount."""

    matches = []
    for line in mountinfo.splitlines():
        fields = line.split()
        if len(fields) < 6 or fields[4] != mountpoint:
            continue
        try:
            major, minor = (int(value) for value in fields[2].split(":", 1))
        except (TypeError, ValueError):
            continue
        matches.append((major, minor))
    return matches[-1] if matches else None


def _diskstats_bytes(
    diskstats: str, device: tuple[int, int]
) -> tuple[int, int, str] | None:
    """Read byte counters by device number without relying on /dev symlinks."""

    for line in diskstats.splitlines():
        fields = line.split()
        if len(fields) < 10:
            continue
        try:
            observed = (int(fields[0]), int(fields[1]))
            sectors_read = int(fields[5])
            sectors_written = int(fields[9])
        except ValueError:
            continue
        if observed == device:
            # Linux exposes these two /proc/diskstats fields in 512-byte sectors.
            return sectors_read * 512, sectors_written * 512, fields[2]
    return None


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _self_hash(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("artifact_sha256", None)
    result["artifact_sha256"] = _digest(result)
    return result


def _artifact_hash_is_valid(value: Mapping[str, Any]) -> bool:
    claimed = value.get("artifact_sha256")
    body = {key: item for key, item in value.items() if key != "artifact_sha256"}
    return isinstance(claimed, str) and claimed == _digest(body)


@dataclass(frozen=True)
class Cohort:
    name: str
    paths: tuple[str, ...]
    rows: int
    columns: tuple[str, ...]
    compressed_bytes: int | None
    selected_compressed_bytes: int | None
    expected_output_schema_sha256: str | None
    expected_checksum: Mapping[str, Any] | None
    object_count: int
    content_digest: str
    manifest_path: str
    manifest_sha256: str
    manifest_digest: str
    cohort_digest: str


def _plain_s3_uri(value: object, *, label: str) -> str:
    uri = str(value)
    parsed = urlparse(uri)
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or not parsed.path.lstrip("/")
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"{label} is not a plain S3 object URI: {uri!r}")
    return uri


def _normalize_expected_checksum(value: object) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("expected_tokenizer_checksum must be an object")
    required = ("algorithm", "rows", "sum1", "sum2")
    if any(name not in value for name in required):
        raise ValueError(
            "expected_tokenizer_checksum must contain algorithm, rows, sum1, and sum2"
        )
    result = {name: value[name] for name in required}
    if result["algorithm"] != CHECKSUM_ALGORITHM:
        raise ValueError("expected_tokenizer_checksum uses an unknown algorithm")
    if type(result["rows"]) is not int or result["rows"] <= 0:
        raise ValueError("expected_tokenizer_checksum rows must be a positive integer")
    for name in ("sum1", "sum2"):
        digest = result[name]
        if (
            not isinstance(digest, str)
            or len(digest) != 16
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                f"expected_tokenizer_checksum {name} must be 16 hex digits"
            )
    return result


def _cohort_record(document: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    raw = document.get("cohorts")
    if isinstance(raw, Mapping):
        record = raw.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"cohort {name!r} is missing")
        return record
    if isinstance(raw, list):
        matches = [
            item
            for item in raw
            if isinstance(item, Mapping)
            and str(item.get("name", item.get("id", ""))) == name
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected one cohort named {name!r}, found {len(matches)}"
            )
        return matches[0]
    raise ValueError("manifest cohorts must be an object or a list")


def _object_uri(document: Mapping[str, Any], item: object, *, label: str) -> str:
    if not isinstance(item, Mapping):
        raise ValueError(f"{label} must be an object")
    if item.get("uri") is not None:
        return _plain_s3_uri(item["uri"], label=label)
    bucket = document.get("bucket")
    key = item.get("key")
    if not isinstance(bucket, str) or not bucket or "/" in bucket:
        raise ValueError("a manifest with object keys must declare one S3 bucket")
    if not isinstance(key, str) or not key or key.startswith("/"):
        raise ValueError(f"{label} has no valid S3 object key")
    return _plain_s3_uri(f"s3://{bucket}/{key}", label=label)


def _content_digest(
    objects: object, paths: Sequence[str], record: Mapping[str, Any]
) -> str:
    declared = record.get("content_sha256")
    if declared is not None:
        if (
            not isinstance(declared, str)
            or len(declared) != 64
            or any(character not in "0123456789abcdef" for character in declared)
        ):
            raise ValueError("content_sha256 must be a lowercase SHA-256")
        return declared
    if not isinstance(objects, list):
        return _digest(list(paths))
    identities = []
    for index, item in enumerate(objects):
        if not isinstance(item, Mapping):
            raise ValueError("cohort object inventory entries must be objects")
        identities.append(
            {
                "position": index,
                "copy_index": item.get("copy_index"),
                "split": item.get("split"),
                "source_key": item.get("source_key"),
                "source_etag": item.get("source_etag", item.get("etag")),
                "content_sha256": item.get("sha256", item.get("checksum_sha256")),
                "rows": item.get("rows"),
                "bytes": item.get("size", item.get("bytes")),
                "selected_compressed_bytes": item.get("selected_compressed_bytes"),
            }
        )
    return _digest(identities)


def load_cohort(path: Path, name: str) -> Cohort:
    path = path.resolve()
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError("cohort manifest must contain one JSON object")
    record = _cohort_record(document, name)

    raw_paths = record.get("paths", record.get("uris"))
    objects = record.get("objects", record.get("files"))
    if raw_paths is None and isinstance(objects, list):
        raw_paths = [
            _object_uri(document, item, label=f"cohort {name!r} object")
            for item in objects
        ]
    if not isinstance(raw_paths, list) or not raw_paths:
        raise ValueError(f"cohort {name!r} has no S3 paths")
    paths = tuple(
        _plain_s3_uri(item, label=f"cohort {name!r} path") for item in raw_paths
    )
    if len(paths) != len(set(paths)):
        raise ValueError(f"cohort {name!r} repeats an S3 object URI")

    rows = record.get("rows", record.get("num_rows"))
    if type(rows) is not int or rows <= 0:
        raise ValueError(f"cohort {name!r} must declare a positive integer row count")
    columns = record.get("columns", document.get("columns", TOKENIZER_COLUMNS))
    if (
        not isinstance(columns, (list, tuple))
        or not columns
        or not all(isinstance(item, str) and item for item in columns)
        or len(columns) != len(set(columns))
    ):
        raise ValueError("projected columns must be distinct nonempty strings")

    compressed = record.get(
        "compressed_bytes", record.get("bytes", record.get("total_bytes"))
    )
    if compressed is not None and (type(compressed) is not int or compressed <= 0):
        raise ValueError("compressed_bytes must be a positive integer when present")
    selected_compressed = record.get("selected_compressed_bytes")
    if selected_compressed is not None and (
        type(selected_compressed) is not int or selected_compressed <= 0
    ):
        raise ValueError(
            "selected_compressed_bytes must be a positive integer when present"
        )
    schema_digest = record.get("expected_output_schema_sha256")
    if schema_digest is not None and (
        not isinstance(schema_digest, str)
        or len(schema_digest) != 64
        or any(character not in "0123456789abcdef" for character in schema_digest)
    ):
        raise ValueError("expected_output_schema_sha256 must be a lowercase SHA-256")
    expected_checksum = _normalize_expected_checksum(
        record.get("expected_tokenizer_checksum")
    )
    if expected_checksum is not None and expected_checksum["rows"] != rows:
        raise ValueError("expected_tokenizer_checksum rows differ from cohort rows")

    declared_manifest_digest = document.get("manifest_sha256")
    manifest_body = {
        key: value for key, value in document.items() if key != "manifest_sha256"
    }
    calculated_manifest_digest = _digest(manifest_body)
    if declared_manifest_digest is not None and (
        declared_manifest_digest != calculated_manifest_digest
    ):
        raise ValueError("manifest_sha256 does not match the manifest contents")

    return Cohort(
        name=name,
        paths=paths,
        rows=rows,
        columns=tuple(columns),
        compressed_bytes=compressed,
        selected_compressed_bytes=selected_compressed,
        expected_output_schema_sha256=schema_digest,
        expected_checksum=expected_checksum,
        object_count=len(objects) if isinstance(objects, list) else len(paths),
        content_digest=_content_digest(objects, paths, record),
        manifest_path=str(path),
        manifest_sha256=_file_sha256(path),
        manifest_digest=calculated_manifest_digest,
        cohort_digest=_digest(record),
    )


def _node_affinity(node_id: str) -> Any:
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    return NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)


def _critical_profile_runtime_env(
    ray_module: Any,
    run_id: str,
    *,
    support_path: Path | None = None,
) -> dict[str, Any]:
    """Ship the benchmark hook without adding options to ``map_batches``."""

    path = support_path or Path(__file__).resolve().with_name(CRITICAL_PROFILE_SUPPORT)
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"critical profile support is missing: {path}")
    module_name = "fusion_critical_profile_support"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load critical profile support: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    ray_module.cloudpickle.register_pickle_by_value(module)
    return {
        "env_vars": {
            module.PROFILE_ENV: "1",
            module.LAZY_PROFILE_ENV: "1",
            module.RUN_ID_ENV: run_id,
        },
        "worker_process_setup_hook": module.install_profile_hooks,
    }


def _critical_profile_event_sort_key(event: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(event.get("node_id", "")),
        str(event.get("actor_id", "")),
        str(event.get("event", "")),
        str(event.get("op_name", "")),
        int(event.get("task_idx", -1)),
        float(event.get("wall_s", 0.0)),
        int(event.get("pid", -1)),
    )


def _decode_critical_profile_lines(
    lines: Iterable[str], run_id: str
) -> tuple[list[dict[str, Any]], list[str]]:
    events = []
    errors = []
    for line_number, line in enumerate(lines, start=1):
        marker = line.find(CRITICAL_PROFILE_MARKER)
        if marker < 0:
            continue
        payload = line[marker + len(CRITICAL_PROFILE_MARKER) :].strip()
        try:
            event = json.loads(payload)
        except json.JSONDecodeError as error:
            errors.append(f"line {line_number}: {error.msg}")
            continue
        if not isinstance(event, dict):
            errors.append(f"line {line_number}: profile event is not an object")
            continue
        if event.get("profile_run_id") == run_id:
            events.append(event)
    return sorted(events, key=_critical_profile_event_sort_key), errors


def _profile_distribution(values: Iterable[float]) -> dict[str, int | float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0}

    def percentile(fraction: float) -> float:
        return ordered[round((len(ordered) - 1) * fraction)]

    return {
        "count": len(ordered),
        "sum": sum(ordered),
        "mean": sum(ordered) / len(ordered),
        "min": ordered[0],
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "max": ordered[-1],
    }


def _critical_profile_summary(
    events: Sequence[Mapping[str, Any]],
    *,
    expected_actor_count: int,
    expected_node_ids: set[str],
) -> dict[str, Any]:
    actor_ready = [event for event in events if event.get("event") == "actor_ready"]
    tokenizers = [
        event for event in events if event.get("event") == "tokenizer_initialized"
    ]
    tasks = [event for event in events if event.get("event") == "task_complete"]
    stages: dict[str, dict[str, int | float]] = {}
    for task in tasks:
        task_stages = task.get("stages", {})
        if not isinstance(task_stages, Mapping):
            continue
        for name, values in task_stages.items():
            if not isinstance(name, str) or not isinstance(values, Mapping):
                continue
            aggregate = stages.setdefault(
                name,
                {"calls": 0, "elapsed_s": 0.0, "rows": 0, "bytes": 0},
            )
            for field in ("calls", "rows", "bytes"):
                aggregate[field] += int(values.get(field, 0))
            aggregate["elapsed_s"] += float(values.get("elapsed_s", 0.0))

    phases = {}
    for phase, stage_names in CRITICAL_PROFILE_STAGE_GROUPS.items():
        phases[phase] = {
            "stages": list(stage_names),
            "calls": sum(
                int(stages.get(name, {}).get("calls", 0)) for name in stage_names
            ),
            "actor_time_s": sum(
                float(stages.get(name, {}).get("elapsed_s", 0.0))
                for name in stage_names
            ),
        }
        phases[phase]["ideal_wall_s_at_full_actor_use"] = (
            phases[phase]["actor_time_s"] / expected_actor_count
        )

    reasons = []
    actor_nodes = {str(event.get("node_id")) for event in actor_ready}
    if len(actor_ready) != expected_actor_count:
        reasons.append(
            f"expected {expected_actor_count} actor_ready events, got {len(actor_ready)}"
        )
    if actor_nodes != expected_node_ids:
        reasons.append("actor_ready events do not cover the exact active nodes")
    if len(tokenizers) != expected_actor_count:
        reasons.append("tokenizer initialization events do not match the actor count")
    if not tasks:
        reasons.append("no fused task completion events were collected")
    if any(event.get("success") is not True for event in (*actor_ready, *tasks)):
        reasons.append("one or more actor or task profile events failed")
    for stage in (
        "cudf_read",
        "udf",
        "batch_to_block",
        "output_buffer_add_batch",
        "output_buffer_next",
        "block_yield_suspended",
    ):
        if int(stages.get(stage, {}).get("calls", 0)) == 0:
            reasons.append(f"required stage {stage!r} has no samples")

    return {
        "valid": not reasons,
        "rejection_reasons": reasons,
        "actor_count": len(actor_ready),
        "tokenizer_init_count": len(tokenizers),
        "task_count": len(tasks),
        "actor_init_s": _profile_distribution(
            float(event.get("init_elapsed_s", 0.0)) for event in actor_ready
        ),
        "tokenizer_init_s": _profile_distribution(
            float(event.get("init_elapsed_s", 0.0)) for event in tokenizers
        ),
        "task_elapsed_s": _profile_distribution(
            float(event.get("task_elapsed_s", 0.0)) for event in tasks
        ),
        "stages": dict(sorted(stages.items())),
        "phases": phases,
        "timing_note": (
            "Phase actor times sum across actors. output_buffer includes the nested "
            "batch_to_block stage; yield_backpressure is time suspended after a "
            "worker yields output or metadata."
        ),
    }


def _collect_critical_profile(
    ray_module: Any,
    alive: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    expected_actor_count: int,
) -> dict[str, Any]:
    """Read every node's local worker logs after the stopwatch has stopped."""

    @ray_module.remote(num_cpus=0)
    class ProfileLogReader:
        def read(self, selected_run_id: str) -> dict[str, Any]:
            import ray
            from ray._private.worker import global_worker

            node_id = str(ray.get_runtime_context().get_node_id())
            session_dir = Path(global_worker.node.address_info["session_dir"])
            events = []
            errors = []
            files = sorted((session_dir / "logs").glob("worker-*.out"))
            for path in files:
                decoded, failures = _decode_critical_profile_lines(
                    path.read_text(encoding="utf-8", errors="replace").splitlines(),
                    selected_run_id,
                )
                events.extend(decoded)
                errors.extend(f"{path.name}: {failure}" for failure in failures)
            return {
                "node_id": node_id,
                "log_file_count": len(files),
                "parse_errors": errors,
                "events": sorted(events, key=_critical_profile_event_sort_key),
            }

    readers = [
        ProfileLogReader.options(
            scheduling_strategy=_node_affinity(str(node["NodeID"]))
        ).remote()
        for node in alive
    ]
    collected = sorted(
        ray_module.get([reader.read.remote(run_id) for reader in readers]),
        key=lambda item: item["node_id"],
    )
    events = sorted(
        (event for node in collected for event in node["events"]),
        key=_critical_profile_event_sort_key,
    )
    expected_node_ids = {str(node["NodeID"]) for node in alive}
    summary = _critical_profile_summary(
        events,
        expected_actor_count=expected_actor_count,
        expected_node_ids=expected_node_ids,
    )
    parse_errors = [
        f"{node['node_id']}: {error}"
        for node in collected
        for error in node["parse_errors"]
    ]
    reasons = [*parse_errors, *summary["rejection_reasons"]]
    body = {
        "schema_version": 1,
        "kind": "ray_data_fused_map_critical_profile",
        "run_id": run_id,
        "valid": not reasons,
        "rejection_reasons": reasons,
        "collection": {
            "mechanism": "one post-timing node-affine Ray log reader per active node",
            "nodes": [
                {
                    "node_id": node["node_id"],
                    "log_file_count": node["log_file_count"],
                    "event_count": len(node["events"]),
                    "parse_errors": node["parse_errors"],
                }
                for node in collected
            ],
        },
        "summary": summary,
        "worker_events": events,
    }
    return _self_hash(body)


def _alive_nodes(ray_module: Any) -> list[dict[str, Any]]:
    return sorted(
        (dict(node) for node in ray_module.nodes() if node.get("Alive")),
        key=lambda item: str(item.get("NodeID", "")),
    )


def validate_cluster_shape(ray_module: Any, nodes: int) -> dict[str, Any]:
    alive = _alive_nodes(ray_module)
    resources = ray_module.cluster_resources()
    expected = {
        "nodes": nodes,
        "cpus": float(nodes * CPUS_PER_NODE),
        "gpus": float(nodes * GPUS_PER_NODE),
    }
    observed = {
        "nodes": len(alive),
        "cpus": float(resources.get("CPU", 0)),
        "gpus": float(resources.get("GPU", 0)),
    }
    if observed != expected:
        raise RuntimeError(
            f"Ray cluster shape differs: expected={expected}, observed={observed}"
        )
    node_rows = []
    for node in alive:
        available = node.get("Resources") or {}
        node_shape = {
            "cpu": float(available.get("CPU", 0)),
            "gpu": float(available.get("GPU", 0)),
        }
        if node_shape != {"cpu": float(CPUS_PER_NODE), "gpu": 1.0}:
            raise RuntimeError(
                f"Ray node {node.get('NodeID')} is not CPU=16/GPU=1: {node_shape}"
            )
        node_rows.append(
            {
                "node_id": str(node["NodeID"]),
                "node_ip": str(node.get("NodeManagerAddress", "")),
                **node_shape,
            }
        )
    return {"expected": expected, "observed": observed, "nodes": node_rows}


def configure_data_context(context: Any, *, fused: bool) -> None:
    context.enable_progress_bars = False
    context.use_datasource_v2 = True
    context.enable_cudf_parquet_read_fusion = fused
    context.parquet_chunker_target_chunk_size = None


@contextlib.contextmanager
def _temporary_fused_parquet_chunk_multiplier(
    multiplier: int, *, fusion_module: Any | None = None
) -> Iterable[None]:
    """Change only the fused row-group work size for one benchmark execution."""

    if multiplier not in APPROVED_PARQUET_ROW_GROUP_CHUNK_MULTIPLIERS:
        raise ValueError(
            f"unsupported Parquet row-group chunk multiplier: {multiplier}"
        )
    if multiplier == 1:
        yield
        return

    if fusion_module is None:
        from ray.data._internal.logical.rules import (
            cudf_parquet_read_fusion as fusion_module,
        )

    original = fusion_module._logical_config_if_eligible

    @functools.wraps(original)
    def use_larger_row_group_chunks(*args: Any, **kwargs: Any) -> Any:
        config = original(*args, **kwargs)
        if config is None:
            return None
        # The map batch stays unchanged. Only the row-group range sent to each
        # fused actor task grows for this benchmark process.
        return replace(config, batch_size=config.batch_size * multiplier)

    fusion_module._logical_config_if_eligible = use_larger_row_group_chunks
    try:
        yield
    finally:
        if fusion_module._logical_config_if_eligible is not use_larger_row_group_chunks:
            raise RuntimeError("fused Parquet logical config changed during the probe")
        fusion_module._logical_config_if_eligible = original


@contextlib.contextmanager
def _temporary_fused_actor_concurrency(
    max_concurrency: int | None, *, fusion_rule: Any | None = None
) -> Iterable[dict[str, int]]:
    """Change only the synthetic fused actor for one benchmark execution."""

    if max_concurrency is None:
        yield {}
        return
    if max_concurrency not in APPROVED_FUSED_ACTOR_MAX_CONCURRENCY:
        raise ValueError(f"unsupported fused actor concurrency: {max_concurrency}")
    if fusion_rule is None:
        from ray.data._internal.logical.rules.cudf_parquet_read_fusion import (
            FuseCudfParquetReadIntoMapBatches as fusion_rule,
        )

    original_descriptor = fusion_rule.__dict__["_create_fused_operator"]
    original = fusion_rule._create_fused_operator
    receipt: dict[str, int] = {}

    @functools.wraps(original)
    def create_with_concurrency(
        read_physical: Any,
        read_logical: Any,
        downstream_physical: Any,
        downstream_logical: Any,
        input_physical_op: Any,
        plan: Any,
        config: Any,
    ) -> Any:
        downstream_logical = replace(
            downstream_logical,
            ray_remote_args={
                **downstream_logical.ray_remote_args,
                "max_concurrency": max_concurrency,
            },
        )
        fused_physical, fused_logical = original(
            read_physical,
            read_logical,
            downstream_physical,
            downstream_logical,
            input_physical_op,
            plan,
            config,
        )
        receipt.update(
            max_concurrency=fused_physical._actor_pool.max_actor_concurrency(),
            max_tasks_in_flight_per_actor=(
                fused_physical._actor_pool.max_tasks_in_flight_per_actor()
            ),
        )
        return fused_physical, fused_logical

    fusion_rule._create_fused_operator = staticmethod(create_with_concurrency)
    try:
        yield receipt
    finally:
        if fusion_rule._create_fused_operator is not create_with_concurrency:
            raise RuntimeError("fused actor creation changed during the probe")
        fusion_rule._create_fused_operator = original_descriptor


def _actor_geometry(nodes: int, num_gpus_per_actor: float) -> dict[str, int | float]:
    """Resolve a fixed actor pool that consumes the cluster's exact GPUs."""

    if (
        type(num_gpus_per_actor) not in (int, float)
        or num_gpus_per_actor not in APPROVED_NUM_GPUS_PER_ACTOR
    ):
        raise ValueError(f"unsupported GPUs per actor: {num_gpus_per_actor}")
    total_gpus = nodes * GPUS_PER_NODE
    actors = int(total_gpus / num_gpus_per_actor)
    if actors * num_gpus_per_actor != total_gpus:
        raise ValueError("actor geometry does not consume the exact cluster GPUs")
    return {
        "min_actors": actors,
        "max_actors": actors,
        "num_gpus_per_actor": num_gpus_per_actor,
        "total_reserved_gpus": total_gpus,
    }


@contextlib.contextmanager
def _temporary_fractional_gpu_fusion(
    num_gpus_per_actor: float, *, fusion_module: Any | None = None
) -> Iterable[None]:
    """Let a benchmark-only half-GPU actor pass the production one-GPU guard."""

    if num_gpus_per_actor == 1:
        yield
        return
    if num_gpus_per_actor != 0.5:
        raise ValueError(f"unsupported GPUs per actor: {num_gpus_per_actor}")
    if fusion_module is None:
        from ray.data._internal.logical.rules import (
            cudf_parquet_read_fusion as fusion_module,
        )

    original = fusion_module._logical_config_if_eligible

    @functools.wraps(original)
    def allow_half_gpu_actor(read: Any, downstream: Any, context: Any) -> Any:
        remote_args = downstream.ray_remote_args
        if remote_args.get("num_gpus") != num_gpus_per_actor:
            return original(read, downstream, context)
        downstream = replace(
            downstream,
            ray_remote_args={**remote_args, "num_gpus": 1},
        )
        return original(read, downstream, context)

    fusion_module._logical_config_if_eligible = allow_half_gpu_actor
    try:
        yield
    finally:
        if fusion_module._logical_config_if_eligible is not allow_half_gpu_actor:
            raise RuntimeError("fused Parquet eligibility changed during the probe")
        fusion_module._logical_config_if_eligible = original


def timed_pipeline(
    ray_module: Any,
    cohort: Cohort,
    *,
    nodes: int,
    batch_size: int,
    num_gpus_per_actor: float = 1,
    map_batches_max_concurrency: int | None = None,
    fused_actor_max_concurrency: int | None = None,
    parquet_row_group_chunk_multiplier: int = 1,
    clock: Any = time.perf_counter_ns,
) -> tuple[Any, Any, dict[str, int], dict[str, int]]:
    """Run the only timed region.  Keep post-run inspection out of this helper."""

    from src.ray_tokenize import GPUTokenizer

    geometry = _actor_geometry(nodes, num_gpus_per_actor)
    with (
        _temporary_fractional_gpu_fusion(num_gpus_per_actor),
        _temporary_fused_parquet_chunk_multiplier(parquet_row_group_chunk_multiplier),
        _temporary_fused_actor_concurrency(
            fused_actor_max_concurrency
        ) as resolved_fused_actor_pool,
    ):
        started_ns = clock()
        map_batches_kwargs = {
            "fn_constructor_kwargs": {
                "merchant_hash_size": MERCHANT_HASH_SIZE,
            },
            "batch_format": "cudf",
            "batch_size": batch_size,
            "zero_copy_batch": True,
            "compute": ray_module.data.ActorPoolStrategy(size=geometry["max_actors"]),
            "num_gpus": num_gpus_per_actor,
        }
        if map_batches_max_concurrency is not None:
            map_batches_kwargs["max_concurrency"] = map_batches_max_concurrency
        dataset = ray_module.data.read_parquet(
            list(cohort.paths), columns=list(cohort.columns)
        ).map_batches(GPUTokenizer, **map_batches_kwargs)
        materialized = dataset.materialize()
        finished_ns = clock()
    # materialize() replaces the returned Dataset's lineage with InputData.  Keep
    # the lazy Dataset so post-run checks can inspect the plan that actually ran.
    return (
        dataset,
        materialized,
        {
            "started_ns": started_ns,
            "finished_ns": finished_ns,
            "elapsed_ns": finished_ns - started_ns,
        },
        dict(resolved_fused_actor_pool),
    )


def _monitor_actor_class(ray_module: Any) -> Any:
    @ray_module.remote(num_cpus=0)
    class Monitor:
        def __init__(self, spill_directory: str, sample_s: float = 0.2) -> None:
            import threading

            self.spill_directory = Path(spill_directory)
            self.sample_s = sample_s
            self.stop_event = threading.Event()
            self.thread = None
            self.samples: list[dict[str, Any]] = []

        @staticmethod
        def _non_loopback_network(psutil: Any) -> tuple[int, int]:
            counters = psutil.net_io_counters(pernic=True)
            return (
                sum(
                    int(value.bytes_recv)
                    for name, value in counters.items()
                    if name != "lo"
                ),
                sum(
                    int(value.bytes_sent)
                    for name, value in counters.items()
                    if name != "lo"
                ),
            )

        @staticmethod
        def _nvme_counters() -> tuple[int | None, int | None, list[str]]:
            try:
                mountinfo = Path("/proc/self/mountinfo").read_text(encoding="utf-8")
                diskstats = Path("/proc/diskstats").read_text(encoding="utf-8")
            except OSError:
                return None, None, []
            device = _mount_device_number(mountinfo)
            if device is None:
                return None, None, []
            counters = _diskstats_bytes(diskstats, device)
            label = f"{device[0]}:{device[1]}"
            if counters is None:
                return None, None, [label]
            read_bytes, write_bytes, name = counters
            return read_bytes, write_bytes, [f"{label}:{name}"]

        def _sample(self) -> dict[str, Any]:
            import psutil
            import ray

            received, sent = self._non_loopback_network(psutil)
            nvme_read, nvme_write, nvme_devices = self._nvme_counters()
            memory = psutil.virtual_memory()
            gpu = None
            initialized = False
            try:
                import pynvml

                pynvml.nvmlInit()
                initialized = True
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu = {
                    "utilization_pct": int(utilization.gpu),
                    "memory_used_bytes": int(gpu_memory.used),
                    "memory_total_bytes": int(gpu_memory.total),
                }
            except BaseException:
                pass
            finally:
                if initialized:
                    pynvml.nvmlShutdown()
            return {
                "time_ns": time.time_ns(),
                "node_id": str(ray.get_runtime_context().get_node_id()),
                "network_received_bytes": received,
                "network_sent_bytes": sent,
                "nvme_read_bytes": nvme_read,
                "nvme_write_bytes": nvme_write,
                "nvme_devices": nvme_devices,
                "host_memory_used_bytes": int(memory.used),
                "host_memory_available_bytes": int(memory.available),
                "gpu": gpu,
            }

        def _run(self) -> None:
            while not self.stop_event.wait(self.sample_s):
                self.samples.append(self._sample())

        def start(self) -> str:
            import threading

            self.samples = [self._sample()]
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            return self.samples[0]["node_id"]

        def stop(self) -> dict[str, Any]:
            self.stop_event.set()
            if self.thread is not None:
                self.thread.join(timeout=5)
            self.samples.append(self._sample())
            first, last = self.samples[0], self.samples[-1]
            gpu = [sample["gpu"] for sample in self.samples if sample["gpu"]]

            def delta(name: str) -> int | None:
                if first[name] is None or last[name] is None:
                    return None
                return int(last[name]) - int(first[name])

            return {
                "node_id": first["node_id"],
                "samples": len(self.samples),
                "network_received_bytes": delta("network_received_bytes"),
                "network_sent_bytes": delta("network_sent_bytes"),
                "nvme_read_bytes": delta("nvme_read_bytes"),
                "nvme_write_bytes": delta("nvme_write_bytes"),
                "nvme_devices": first["nvme_devices"],
                "peak_host_memory_used_bytes": max(
                    sample["host_memory_used_bytes"] for sample in self.samples
                ),
                "minimum_host_memory_available_bytes": min(
                    sample["host_memory_available_bytes"] for sample in self.samples
                ),
                "gpu_samples": len(gpu),
                "gpu_mean_utilization_pct": (
                    statistics.mean(item["utilization_pct"] for item in gpu)
                    if gpu
                    else None
                ),
                "gpu_peak_utilization_pct": max(
                    (item["utilization_pct"] for item in gpu), default=None
                ),
                "gpu_peak_memory_used_bytes": max(
                    (item["memory_used_bytes"] for item in gpu), default=None
                ),
                "gpu_total_memory_bytes": max(
                    (item["memory_total_bytes"] for item in gpu), default=None
                ),
            }

    return Monitor


def _start_monitors(
    ray_module: Any, alive: Sequence[Mapping[str, Any]], spill_directory: Path
) -> list[Any]:
    actor_class = _monitor_actor_class(ray_module)
    actors = [
        actor_class.options(
            scheduling_strategy=_node_affinity(str(node["NodeID"]))
        ).remote(str(spill_directory))
        for node in alive
    ]
    ids = ray_module.get([actor.start.remote() for actor in actors])
    if set(ids) != {str(node["NodeID"]) for node in alive}:
        raise RuntimeError("resource monitors did not cover every active node")
    return actors


def _stop_monitors(ray_module: Any, actors: Sequence[Any]) -> dict[str, Any]:
    nodes = ray_module.get([actor.stop.remote() for actor in actors])
    totals = {}
    for name in (
        "network_received_bytes",
        "network_sent_bytes",
        "nvme_read_bytes",
        "nvme_write_bytes",
    ):
        values = [item[name] for item in nodes]
        totals[name] = None if any(value is None for value in values) else sum(values)
    return {"nodes": nodes, "totals": totals}


def _transport_realization(
    ray_module: Any, alive: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Verify each Ray process's transport environment without issuing I/O."""

    environment_names = (
        "CUDF_KVIKIO_REMOTE_IO",
        "KVIKIO_NTHREADS",
        "KVIKIO_TASK_SIZE",
    )
    expected_environment = {name: os.environ.get(name) for name in environment_names}
    expected_remote_io = expected_environment["CUDF_KVIKIO_REMOTE_IO"] == "1"

    @ray_module.remote(num_cpus=0)
    class Probe:
        def read(self) -> dict[str, Any]:
            import cudf
            import ray

            process_environment = {
                name: os.environ.get(name) for name in environment_names
            }
            cudf_remote_io = bool(cudf.get_option("kvikio_remote_io"))
            return {
                "node_id": str(ray.get_runtime_context().get_node_id()),
                "process_environment": process_environment,
                "cudf_kvikio_remote_io": cudf_remote_io,
                "matches": process_environment == expected_environment
                and cudf_remote_io == expected_remote_io,
            }

    actors = [
        Probe.options(scheduling_strategy=_node_affinity(str(node["NodeID"]))).remote()
        for node in alive
    ]
    nodes = ray_module.get([actor.read.remote() for actor in actors])
    expected_ids = {str(node["NodeID"]) for node in alive}
    observed_ids = {str(node["node_id"]) for node in nodes}
    return {
        "valid": observed_ids == expected_ids
        and all(node["matches"] for node in nodes),
        "expected_environment": expected_environment,
        "nodes": sorted(nodes, key=lambda node: node["node_id"]),
    }


def _spill_snapshot(ray_module: Any) -> dict[str, Any]:
    try:
        from ray._private.internal_api import get_state_from_address, node_stats

        state = get_state_from_address(ray_module.get_runtime_context().gcs_address)
        alive = sorted(
            (node for node in state.node_table() if node["Alive"]),
            key=lambda item: str(item.get("NodeID", "")),
        )
        totals: dict[str, int | float] = dict.fromkeys(SPILL_FIELDS, 0)
        nodes = []
        for node in alive:
            stats = node_stats(
                node_manager_address=node["NodeManagerAddress"],
                node_manager_port=node["NodeManagerPort"],
                include_memory_info=False,
            ).store_stats
            fields = {name: getattr(stats, name) for name in SPILL_FIELDS}
            for name, amount in fields.items():
                totals[name] += amount
            nodes.append(
                {
                    "node_id": str(node.get("NodeID", "")),
                    "node_ip": str(node.get("NodeManagerAddress", "")),
                    **fields,
                }
            )
        return {"available": True, "nodes": nodes, "totals": totals}
    except BaseException as error:
        return {
            "available": False,
            "error": f"{type(error).__name__}: {error}",
            "nodes": [],
            "totals": {},
        }


def _stable_spill_snapshot(ray_module: Any) -> dict[str, Any]:
    previous = _spill_snapshot(ray_module)
    if not previous.get("available"):
        return previous
    stable = 0
    deadline = time.monotonic() + 8
    while stable < 2 and time.monotonic() < deadline:
        time.sleep(0.25)
        current = _spill_snapshot(ray_module)
        if not current.get("available"):
            return current
        stable = stable + 1 if current["totals"] == previous["totals"] else 0
        previous = current
    return previous


def _spill_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    if not before.get("available") or not after.get("available"):
        return {
            "available": False,
            "before_error": before.get("error"),
            "after_error": after.get("error"),
        }
    prior = {item["node_id"]: item for item in before["nodes"]}
    nodes = []
    for current in after["nodes"]:
        old = prior.get(current["node_id"], {})
        nodes.append(
            {
                "node_id": current["node_id"],
                "node_ip": current["node_ip"],
                **{name: current[name] - old.get(name, 0) for name in SPILL_FIELDS},
            }
        )
    return {
        "available": True,
        "nodes": nodes,
        "totals": {
            name: after["totals"][name] - before["totals"][name]
            for name in SPILL_FIELDS
        },
    }


def _dataset_metadata(dataset: Any) -> dict[str, int]:
    blocks = rows = size_bytes = 0
    for bundle in dataset.iter_internal_ref_bundles():
        blocks += len(bundle.block_refs)
        for metadata in bundle.metadata:
            if metadata.num_rows is None or metadata.size_bytes is None:
                raise RuntimeError("materialized output has incomplete block metadata")
            rows += int(metadata.num_rows)
            size_bytes += int(metadata.size_bytes)
    return {"blocks": blocks, "rows": rows, "size_bytes": size_bytes}


def _arrow_schema(dataset: Any) -> Any:
    schema = dataset.schema(fetch_if_missing=True)
    return getattr(schema, "base_schema", schema)


def _schema_evidence(dataset: Any) -> dict[str, Any]:
    schema = _arrow_schema(dataset)
    serialized = schema.serialize().to_pybytes()
    fields = []
    for field in schema:
        field_type = field.type
        shape = getattr(field_type, "shape", None)
        value_type = getattr(field_type, "value_type", None)
        fields.append(
            {
                "name": field.name,
                "type": str(field_type),
                "shape": list(shape) if shape is not None else None,
                "value_type": str(value_type) if value_type is not None else None,
            }
        )
    return {
        "names": list(schema.names),
        "text": str(schema),
        "sha256": hashlib.sha256(serialized).hexdigest(),
        "fields": fields,
    }


def _validate_schema(evidence: Mapping[str, Any]) -> list[str]:
    reasons = []
    if tuple(evidence["names"]) != OUTPUT_COLUMNS:
        reasons.append(
            f"output columns differ: {evidence['names']} != {list(OUTPUT_COLUMNS)}"
        )
        return reasons
    fields = {field["name"]: field for field in evidence["fields"]}
    for name in ("uc_key", "ts", "label"):
        if fields[name]["type"] != "int64":
            reasons.append(f"{name} is not int64: {fields[name]['type']}")
    tensor = fields["token_ids"]
    if tensor["shape"] not in ([12], None):
        reasons.append(f"token_ids shape is not [12]: {tensor['shape']}")
    if "int32" not in tensor["type"] and tensor["value_type"] != "int32":
        reasons.append(f"token_ids does not contain int32 values: {tensor}")
    return reasons


def verify_physical_plan(
    dataset: Any,
    *,
    fused: bool,
    expected_actor_pool: Mapping[str, int],
    expected_actor_geometry: Mapping[str, int | float],
    executed_actor_pool: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    from ray.data._internal.execution.operators.actor_pool_map_operator import (
        ActorPoolMapOperator,
    )
    from ray.data._internal.logical.optimizers import get_execution_plan
    from ray.data._internal.logical.operators import MapBatches, ReadFiles
    from ray.data._internal.logical.rules.cudf_parquet_read_fusion import (
        _CudfBatchMapTransformFn,
        _CudfParquetReader,
    )

    num_gpus_per_actor = float(expected_actor_geometry["num_gpus_per_actor"])
    with _temporary_fractional_gpu_fusion(num_gpus_per_actor):
        physical, _ = get_execution_plan(dataset._logical_plan)
    final = physical.dag
    transforms = (
        final.get_map_transformer().get_transform_fns()
        if isinstance(final, ActorPoolMapOperator)
        else []
    )
    actual_fused = (
        isinstance(final, ActorPoolMapOperator)
        and len(transforms) == 2
        and isinstance(getattr(transforms[0], "_fn", None), _CudfParquetReader)
        and isinstance(transforms[1], _CudfBatchMapTransformFn)
    )
    logical_types = [type(item).__name__ for item in physical.op_map.values()]
    if actual_fused != fused:
        raise RuntimeError(
            f"physical plan mismatch: expected fused={fused}, got {actual_fused}"
        )
    if not fused:
        if not isinstance(final, ActorPoolMapOperator):
            raise RuntimeError("isolated plan does not end in an actor map")
        if not any(isinstance(item, ReadFiles) for item in physical.op_map.values()):
            raise RuntimeError("isolated plan has no separate ReadFiles operator")
        if not any(isinstance(item, MapBatches) for item in physical.op_map.values()):
            raise RuntimeError("isolated plan has no MapBatches operator")
    resolved_actor_pool = dict(executed_actor_pool or {}) or {
        "max_concurrency": final._actor_pool.max_actor_concurrency(),
        "max_tasks_in_flight_per_actor": (
            final._actor_pool.max_tasks_in_flight_per_actor()
        ),
    }
    if resolved_actor_pool != expected_actor_pool:
        raise RuntimeError(
            f"actor pool differs: expected {dict(expected_actor_pool)}, "
            f"got {resolved_actor_pool}"
        )
    resolved_actor_geometry = {
        "min_actors": final._actor_pool.min_size(),
        "max_actors": final._actor_pool.max_size(),
        "num_gpus_per_actor": final._ray_remote_args.get("num_gpus"),
        "total_reserved_gpus": (
            final._actor_pool.max_size() * final._ray_remote_args.get("num_gpus", 0)
        ),
    }
    if resolved_actor_geometry != expected_actor_geometry:
        raise RuntimeError(
            f"actor geometry differs: expected {dict(expected_actor_geometry)}, "
            f"got {resolved_actor_geometry}"
        )
    return {
        "verified": True,
        "fused": actual_fused,
        "final_operator": type(final).__name__,
        "final_name": final.name,
        "transforms": [type(item).__name__ for item in transforms],
        "logical_operator_types": logical_types,
        "resolved_actor_pool": resolved_actor_pool,
        "resolved_actor_geometry": resolved_actor_geometry,
        "physical_operators": [
            {"type": type(item).__name__, "name": item.name}
            for item in final.post_order_iter()
        ],
    }


def _uint64_sum(values: Any, weight: int, salt: int) -> tuple[int, int]:
    import numpy as np

    array = np.asarray(values)
    unsigned = array.astype(np.uint64, copy=False)
    first = int((unsigned * np.uint64(weight)).sum(dtype=np.uint64))
    second = int(
        ((unsigned ^ np.uint64(salt)) * np.uint64(weight ^ salt)).sum(dtype=np.uint64)
    )
    return first, second


def checksum_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    if tuple(batch) != OUTPUT_COLUMNS:
        raise ValueError(f"checksum batch columns differ: {tuple(batch)!r}")
    rows = len(batch["uc_key"])
    if any(len(batch[name]) != rows for name in OUTPUT_COLUMNS):
        raise ValueError("checksum batch columns have different row counts")
    token_ids = np.asarray(batch["token_ids"])
    if token_ids.ndim != 2 or token_ids.shape != (rows, 12):
        raise ValueError(
            f"token_ids must have shape ({rows}, 12), got {token_ids.shape}"
        )

    first = second = 0
    constants = {
        "uc_key": (0x9E3779B185EBCA87, 0x243F6A8885A308D3),
        "ts": (0xC2B2AE3D27D4EB4F, 0x13198A2E03707344),
        "label": (0x165667B19E3779F9, 0xA4093822299F31D0),
    }
    for name, (weight, salt) in constants.items():
        left, right = _uint64_sum(batch[name], weight, salt)
        first = (first + left) & UINT64_MASK
        second = (second + right) & UINT64_MASK
    for column in range(12):
        weight = (0xD6E8FEB86659FD93 + 2 * column) & UINT64_MASK
        salt = (0x94D049BB133111EB ^ (column * 0x9E3779B1)) & UINT64_MASK
        left, right = _uint64_sum(token_ids[:, column], weight, salt)
        first = (first + left) & UINT64_MASK
        second = (second + right) & UINT64_MASK
    return {
        "rows": np.asarray([rows], dtype=np.int64),
        "sum1": np.asarray([first], dtype=np.uint64),
        "sum2": np.asarray([second], dtype=np.uint64),
    }


def _combine_checksum_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = first = second = batches = 0
    for record in records:
        batches += 1
        rows += int(record["rows"])
        first = (first + int(record["sum1"])) & UINT64_MASK
        second = (second + int(record["sum2"])) & UINT64_MASK
    return {
        "algorithm": CHECKSUM_ALGORITHM,
        "rows": rows,
        "batches": batches,
        "sum1": f"{first:016x}",
        "sum2": f"{second:016x}",
    }


def compute_checksum(dataset: Any) -> dict[str, Any]:
    records = dataset.map_batches(
        checksum_batch,
        batch_format="numpy",
        batch_size=1_048_576,
        zero_copy_batch=True,
    ).take_all()
    return _combine_checksum_records(records)


def _fairness_configuration(
    cohort: Cohort,
    nodes: int,
    batch_size: int,
    parquet_row_group_chunk_multiplier: int = 1,
    map_batches_max_concurrency: int | None = None,
    num_gpus_per_actor: float = 1,
) -> dict[str, Any]:
    geometry = _actor_geometry(nodes, num_gpus_per_actor)
    return {
        "api": "read_parquet(...).map_batches(...).materialize()",
        "content_digest": cohort.content_digest,
        "columns": list(cohort.columns),
        "rows": cohort.rows,
        "nodes": nodes,
        "actors": geometry["max_actors"],
        "actors_per_node": geometry["max_actors"] / nodes,
        "cpus_per_node": CPUS_PER_NODE,
        "gpus_per_node": GPUS_PER_NODE,
        "num_gpus_per_actor": num_gpus_per_actor,
        "total_reserved_gpus": geometry["total_reserved_gpus"],
        "batch_size": batch_size,
        "parquet_row_group_chunk_multiplier": parquet_row_group_chunk_multiplier,
        "max_tasks_in_flight_per_actor": "Ray default (omitted)",
        "max_concurrency": (
            map_batches_max_concurrency
            if map_batches_max_concurrency is not None
            else "Ray default (omitted)"
        ),
        "runtime_env": "omitted",
        "object_store_memory": "Ray default",
        "transport_environment": {
            "cudf_kvikio_remote_io": os.environ.get("CUDF_KVIKIO_REMOTE_IO"),
            "kvikio_nthreads": os.environ.get("KVIKIO_NTHREADS"),
            "kvikio_task_size": os.environ.get("KVIKIO_TASK_SIZE"),
            "aws_region": os.environ.get("AWS_DEFAULT_REGION"),
        },
    }


def run_observation(args: argparse.Namespace) -> dict[str, Any]:
    cohort = load_cohort(args.manifest, args.cohort)
    fused = args.mode == "fused"
    fairness = _fairness_configuration(
        cohort,
        args.nodes,
        args.batch_size,
        args.parquet_row_group_chunk_multiplier,
        args.map_batches_max_concurrency,
        args.num_gpus_per_actor,
    )
    started_wall = time.time()

    import ray
    import ray.data
    from ray.data.context import DataContext

    # The controller runs this artifact as a script. Serialize its benchmark
    # helpers by value so Ray workers never depend on importing a driver module.
    ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])
    profile_run_id = None
    runtime_env = None
    if args.critical_profile:
        profile_run_id = _digest(
            {"fairness_configuration": fairness, "critical_profile_schema": 1}
        )
        runtime_env = _critical_profile_runtime_env(ray, profile_run_id)
    ray.init(
        address="auto",
        logging_level="ERROR",
        **({"runtime_env": runtime_env} if runtime_env is not None else {}),
    )
    monitors: list[Any] = []
    materialized = None
    timing = None
    monitor_result = None
    spill_before = None
    spill_after = None
    critical_profile = None
    executed_actor_pool = None
    try:
        shape = validate_cluster_shape(ray, args.nodes)
        alive = _alive_nodes(ray)
        configure_data_context(DataContext.get_current(), fused=fused)

        spill_before = _stable_spill_snapshot(ray)
        monitors = _start_monitors(ray, alive, args.spill_directory)
        planned, materialized, timing, executed_actor_pool = timed_pipeline(
            ray,
            cohort,
            nodes=args.nodes,
            batch_size=args.batch_size,
            num_gpus_per_actor=args.num_gpus_per_actor,
            map_batches_max_concurrency=args.map_batches_max_concurrency,
            fused_actor_max_concurrency=args.fused_actor_max_concurrency,
            parquet_row_group_chunk_multiplier=(
                args.parquet_row_group_chunk_multiplier
            ),
        )
        monitor_result = _stop_monitors(ray, monitors)
        monitors = []
        spill_after = _stable_spill_snapshot(ray)
        # Everything below is deliberately outside the performance boundary.
        if profile_run_id is not None:
            critical_profile = _collect_critical_profile(
                ray,
                alive,
                run_id=profile_run_id,
                expected_actor_count=int(fairness["actors"]),
            )
        transport = _transport_realization(ray, alive)
        plan = verify_physical_plan(
            planned,
            fused=fused,
            expected_actor_pool={
                "max_concurrency": (
                    args.fused_actor_max_concurrency
                    or args.map_batches_max_concurrency
                    or 1
                ),
                "max_tasks_in_flight_per_actor": 2
                * (
                    args.fused_actor_max_concurrency
                    or args.map_batches_max_concurrency
                    or 1
                ),
            },
            expected_actor_geometry=_actor_geometry(
                args.nodes, args.num_gpus_per_actor
            ),
            executed_actor_pool=executed_actor_pool or None,
        )
        metadata = _dataset_metadata(materialized)
        schema = _schema_evidence(materialized)
        checksum = compute_checksum(materialized)
        stats = materialized.stats()

        reasons = _validate_schema(schema)
        if metadata["rows"] != cohort.rows:
            reasons.append(
                f"materialized rows differ: {metadata['rows']} != {cohort.rows}"
            )
        if checksum["rows"] != cohort.rows:
            reasons.append(f"checksum rows differ: {checksum['rows']} != {cohort.rows}")
        if cohort.expected_output_schema_sha256 is not None and (
            schema["sha256"] != cohort.expected_output_schema_sha256
        ):
            reasons.append("output schema fingerprint differs from the cohort manifest")
        if cohort.expected_checksum is not None:
            expected = dict(cohort.expected_checksum)
            observed = {
                key: checksum[key] for key in ("algorithm", "rows", "sum1", "sum2")
            }
            if observed != expected:
                reasons.append("tokenizer checksum differs from the cohort manifest")
        spill = _spill_delta(spill_before, spill_after)
        if not spill.get("available"):
            reasons.append("Ray spill telemetry is unavailable")
        if any(
            node.get("gpu_mean_utilization_pct") is None
            for node in monitor_result["nodes"]
        ):
            reasons.append("GPU telemetry is unavailable on one or more active nodes")
        if any(
            node.get("nvme_read_bytes") is None or node.get("nvme_write_bytes") is None
            for node in monitor_result["nodes"]
        ):
            reasons.append("NVMe telemetry is unavailable on one or more active nodes")
        if not transport["valid"]:
            reasons.append(
                "cuDF remote I/O or KvikIO process environment differs from the "
                "selected env"
            )
        if critical_profile is not None and not critical_profile["valid"]:
            reasons.append("critical-path profile is incomplete or invalid")

        elapsed_s = timing["elapsed_ns"] / 1e9
        result = {
            "schema_version": 1,
            "kind": "s3_parquet_cudf_map_fusion_observation",
            "status": "accepted" if not reasons else "rejected",
            "valid": not reasons,
            "rejection_reasons": reasons,
            "mode": args.mode,
            "fusion_flag": fused,
            "map_batches_max_concurrency": args.map_batches_max_concurrency,
            "num_gpus_per_actor": args.num_gpus_per_actor,
            "benchmark_fused_actor_max_concurrency": (args.fused_actor_max_concurrency),
            "parquet_row_group_chunk_multiplier": (
                args.parquet_row_group_chunk_multiplier
            ),
            "timing_boundary": (
                "immediately before read_parquet through map_batches output "
                "materialized in Ray's object store"
            ),
            "elapsed_s": elapsed_s,
            "rows_per_s": cohort.rows / elapsed_s,
            "timing": timing,
            "acceptance_policy": {
                "required_paired_speedup": 3.0,
                "valid_pairs_required": 1,
            },
            "cohort": {
                "name": cohort.name,
                "paths": list(cohort.paths),
                "rows": cohort.rows,
                "columns": list(cohort.columns),
                "compressed_bytes": cohort.compressed_bytes,
                "selected_compressed_bytes": cohort.selected_compressed_bytes,
                "object_count": cohort.object_count,
                "content_digest": cohort.content_digest,
                "manifest_path": cohort.manifest_path,
                "manifest_sha256": cohort.manifest_sha256,
                "manifest_digest": cohort.manifest_digest,
                "cohort_digest": cohort.cohort_digest,
            },
            "fairness_configuration": fairness,
            "fairness_configuration_sha256": _digest(fairness),
            "shape": shape,
            "plan": plan,
            "output": {
                **metadata,
                "schema": schema,
            },
            "checksum": checksum,
            "resources": monitor_result,
            "transport_realization": transport,
            "ray_object_store_io": spill,
            **(
                {"critical_profile": critical_profile}
                if critical_profile is not None
                else {}
            ),
            "stats": stats,
            "environment": {
                "cudf_kvikio_remote_io": os.environ.get("CUDF_KVIKIO_REMOTE_IO"),
                "kvikio_nthreads": os.environ.get("KVIKIO_NTHREADS"),
                "kvikio_task_size": os.environ.get("KVIKIO_TASK_SIZE"),
                "aws_region": os.environ.get("AWS_DEFAULT_REGION"),
            },
            "ray_version": ray.__version__,
            "ray_commit": getattr(ray, "__commit__", "unknown"),
            "worker_source_sha256": _file_sha256(Path(__file__).resolve()),
            "started_wall_s": started_wall,
            "finished_wall_s": time.time(),
        }
        return _self_hash(result)
    finally:
        if monitors:
            try:
                _stop_monitors(ray, monitors)
            except BaseException:
                pass
        ray.shutdown()


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--cohort", required=True)
    value.add_argument("--mode", choices=("isolated", "fused"), required=True)
    value.add_argument("--nodes", type=int, required=True)
    value.add_argument("--batch-size", type=int, required=True)
    value.add_argument(
        "--num-gpus-per-actor",
        type=float,
        choices=APPROVED_NUM_GPUS_PER_ACTOR,
        default=1.0,
        help="benchmark actor geometry; queue remains omitted",
    )
    value.add_argument(
        "--map-batches-max-concurrency",
        type=int,
        choices=APPROVED_MAP_BATCHES_MAX_CONCURRENCY,
        default=None,
        help="explicit max_concurrency passed to map_batches; omitted by default",
    )
    value.add_argument(
        "--fused-actor-max-concurrency",
        type=int,
        choices=APPROVED_FUSED_ACTOR_MAX_CONCURRENCY,
        default=None,
        help="benchmark-only physical fused actor concurrency probe",
    )
    value.add_argument("--parquet-row-group-chunk-multiplier", type=int, default=1)
    value.add_argument(
        "--critical-profile",
        action="store_true",
        help="collect benchmark-only fused actor phase timings after materialization",
    )
    value.add_argument("--spill-directory", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.nodes not in APPROVED_NODE_COUNTS:
        values = ", ".join(str(value) for value in APPROVED_NODE_COUNTS)
        raise SystemExit(f"--nodes must be one of {values}")
    if args.critical_profile and args.mode != "fused":
        raise SystemExit("--critical-profile requires --mode fused")
    if args.batch_size not in APPROVED_BATCH_SIZES:
        values = ", ".join(str(value) for value in APPROVED_BATCH_SIZES)
        raise SystemExit(f"--batch-size must be one of {values}")
    if args.mode != "fused" and args.fused_actor_max_concurrency is not None:
        raise SystemExit("--fused-actor-max-concurrency requires --mode fused")
    if (
        args.map_batches_max_concurrency is not None
        and args.fused_actor_max_concurrency is not None
    ):
        raise SystemExit(
            "public map_batches concurrency and the physical probe are mutually exclusive"
        )
    if args.num_gpus_per_actor != 1 and args.map_batches_max_concurrency is None:
        raise SystemExit(
            "fractional actors require explicit --map-batches-max-concurrency"
        )
    if (
        args.parquet_row_group_chunk_multiplier
        not in APPROVED_PARQUET_ROW_GROUP_CHUNK_MULTIPLIERS
    ):
        values = ", ".join(
            str(value) for value in APPROVED_PARQUET_ROW_GROUP_CHUNK_MULTIPLIERS
        )
        raise SystemExit(
            f"--parquet-row-group-chunk-multiplier must be one of {values}"
        )
    for name in ("manifest", "spill_directory", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if not args.manifest.is_file():
        raise SystemExit(f"manifest does not exist: {args.manifest}")
    if not args.spill_directory.is_dir():
        raise SystemExit(f"spill directory does not exist: {args.spill_directory}")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite result: {args.output}")

    try:
        result = run_observation(args)
        return_code = 0 if result["valid"] else 1
    except BaseException as error:
        result = _self_hash(
            {
                "schema_version": 1,
                "kind": "s3_parquet_cudf_map_fusion_observation",
                "status": "failed",
                "valid": False,
                "mode": args.mode,
                "cohort": args.cohort,
                "batch_size": args.batch_size,
                "num_gpus_per_actor": args.num_gpus_per_actor,
                "map_batches_max_concurrency": args.map_batches_max_concurrency,
                "benchmark_fused_actor_max_concurrency": (
                    args.fused_actor_max_concurrency
                ),
                "parquet_row_group_chunk_multiplier": (
                    args.parquet_row_group_chunk_multiplier
                ),
                "critical_profile": args.critical_profile,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "worker_source_sha256": _file_sha256(Path(__file__).resolve()),
            }
        )
        return_code = 1
    if not _artifact_hash_is_valid(result):
        raise RuntimeError("worker created an invalid self-hash")
    _atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "artifact_sha256": result["artifact_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
