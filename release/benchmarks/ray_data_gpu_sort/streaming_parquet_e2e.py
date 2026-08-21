"""Controller for the frozen streaming Parquet -> sort -> Parquet DGX study.

The controller never imports Ray.  It performs read-only preflight checks,
warms the common BTS source outside the measured interval, starts one isolated
worker process per observation, and removes only that worker's validated,
trial-owned runtime after compact evidence has been preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .data import plan_dict, smoke_slices
from .streaming_parquet_common import (
    EXPECTED_CPUS,
    EXPECTED_GPUS,
    GPU_COMMUNICATION_ENVIRONMENT,
    LOCAL_RUN_MIN_FREE_BYTES,
    MIN_CAMPAIGN_FREE_BYTES,
    MIN_HOST_AVAILABLE_BYTES,
    OBJECT_STORE_BYTES,
    TARGET_BLOCKS,
    TARGET_DECODED_BYTES,
    TARGET_ORIGIN_CARDINALITY,
    TARGET_ORIGIN_FREQUENCY_DIGEST,
    TARGET_PLAN_DIGEST,
    TARGET_ROWS,
    TARGET_ROW_ID_SUM,
    _meminfo,
    _nvidia_inventory,
    _process_group_alive,
    _ray_process_inventory,
    _terminate_process_group,
    _wait_for_gpu_idle,
    _warm_source_cache,
    exact_plan,
    read_json,
    write_json,
)
from .spec import (
    DATASET_ROOT,
    RAY_COMMIT,
    RAY_VERSION,
    RAY_WHEEL_SHA256,
    load_manifest,
)

PACKAGE = "release.benchmarks.ray_data_gpu_sort.streaming_parquet_e2e_worker"
IMPLEMENTATION_BRANCH = "streaming-parquet-gpu-sort-e2e"
STREAMING_SORT_BASE_COMMIT = "1442fad4af1b7e58e5ceba27a3a8a8dfdc6f95fc"
RAID_ROOT = Path("/raid")
TRIAL_TIMEOUT_S = 24 * 60 * 60
SMOKE_TIMEOUT_S = 60 * 60
PRE_PIPELINE_TIMEOUT_S = 60 * 60
POST_PIPELINE_TIMEOUT_S = 12 * 60 * 60
WORKER_POLL_S = 0.5

INPUT_BUFFER_BUDGET_BYTES = 100_000_000_000
TARGET_ROW_GROUP_BYTES = 256 << 20
TARGET_FILE_BYTES = 4 << 30
MAX_WRITERS = 16
MAX_WRITER_INPUT_BYTES = 64 << 30
SAMPLE_ROWS_PER_INPUT_BLOCK = 64
SMOKE_DECODED_BYTES = 135_986_017
GATE_SCHEMA_VERSION = 1
GATE_TEST_ROOT = Path("/raid/spark-team/rgs")


@dataclass(frozen=True)
class GateSpec:
    name: str
    nodeid: str
    environment: tuple[tuple[str, str], ...]
    uses_ray: bool


GATE_SEQUENCE = (
    GateSpec(
        "exact-frozen-bts-bytes",
        (
            "release/benchmarks/ray_data_gpu_sort/tests/"
            "test_streaming_parquet_source.py::"
            "test_exact_block_metadata_matches_every_distinct_arrow_block"
        ),
        (("RAY_DATA_VERIFY_FROZEN_BTS_BYTES", "1"),),
        False,
    ),
    GateSpec(
        "gpu-streaming-forced-external",
        (
            "python/ray/data/tests/test_gpu_sort_streaming_integration.py::"
            "test_streaming_origin_sort_externalizes_before_eos_and_releases_inputs"
        ),
        (
            ("RAY_DATA_RUN_GPU_SORT_INTEGRATION", "1"),
            (
                "RAY_DATA_GPU_SORT_TEST_RUN_DIR",
                str(GATE_TEST_ROOT / "gpu-runs"),
            ),
        ),
        True,
    ),
    GateSpec(
        "cpu-push-forced-spill",
        (
            "release/benchmarks/ray_data_gpu_sort/tests/"
            "test_streaming_parquet_forced_spill_integration.py::"
            "test_cpu_push_sort_forces_ray_spill_and_commits_sorted_parquet"
        ),
        (
            ("RAY_DATA_RUN_CPU_PUSH_FORCED_SPILL_INTEGRATION", "1"),
            ("RAY_DATA_FORCED_SPILL_TEST_ROOT", str(GATE_TEST_ROOT)),
        ),
        True,
    ),
)


@dataclass(frozen=True)
class TrialSpec:
    name: str
    backend: str
    workload: str

    @property
    def is_gpu(self) -> bool:
        return self.backend == "gpu"


SMOKE_SEQUENCE = (
    TrialSpec("smoke-gpu-local", "gpu", "smoke"),
    TrialSpec("smoke-cpu-push", "pyarrow", "smoke"),
)
RUN_SEQUENCE = (
    TrialSpec("gpu-streaming-local-1tb", "gpu", "full"),
    TrialSpec("cpu-streaming-push-1tb", "pyarrow", "full"),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git_identity(root: Path) -> dict[str, Any]:
    def output(*args: str) -> str:
        return subprocess.check_output(args, cwd=root, text=True).strip()

    try:
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
            "status": status.splitlines(),
            "clean": not status,
            "streaming_sort_base_commit": STREAMING_SORT_BASE_COMMIT,
            "streaming_sort_base_is_ancestor": descended,
        }
    except (OSError, subprocess.CalledProcessError) as error:
        return {"clean": False, "error": repr(error)}


def _safe_campaign_root(path: Path, *, must_not_exist: bool) -> Path:
    if not path.is_absolute():
        raise ValueError("Campaign root must be an absolute path under /raid")
    resolved = path.resolve(strict=False)
    if resolved == RAID_ROOT or not resolved.is_relative_to(RAID_ROOT):
        raise ValueError(f"Campaign root must be below /raid: {resolved}")
    if len(resolved.relative_to(RAID_ROOT).parts) < 2:
        raise ValueError("Campaign root needs at least two components below /raid")
    if must_not_exist and resolved.exists():
        raise FileExistsError(f"Refusing to overwrite campaign root: {resolved}")
    return resolved


def _dataset_identity(
    dataset_root: Path, *, verify_parquet_files: bool
) -> dict[str, Any]:
    """Bind the frozen logical plan to its manifest and physical Parquet bytes."""

    root = dataset_root.resolve()
    manifest_path = root / "manifest.json"
    manifest_payload = manifest_path.read_bytes()
    manifest = load_manifest(root)
    declared_manifest_sha256 = manifest.get("manifest_sha256")
    if (
        not isinstance(declared_manifest_sha256, str)
        or len(declared_manifest_sha256) != 64
        or any(char not in "0123456789abcdef" for char in declared_manifest_sha256)
    ):
        raise RuntimeError("dataset manifest has no valid declared SHA-256")
    _, slices = exact_plan(root)
    selected = sorted({str(item.path) for item in slices})

    months = manifest.get("months")
    if not isinstance(months, list):
        raise RuntimeError("dataset manifest has no month-file identities")
    declared: dict[str, str] = {}
    for item in months:
        if not isinstance(item, Mapping):
            raise RuntimeError("dataset manifest month entry is not an object")
        raw_path = item.get("parquet_path")
        sha256 = item.get("parquet_sha256")
        if not isinstance(raw_path, str) or not isinstance(sha256, str):
            raise RuntimeError("dataset manifest month lacks Parquet SHA-256")
        source_path = Path(raw_path)
        path = source_path.resolve()
        if (
            source_path.is_symlink()
            or not path.is_relative_to(root)
            or str(path) in declared
        ):
            raise RuntimeError(f"unsafe or duplicate manifest Parquet path: {path}")
        if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
            raise RuntimeError(f"invalid manifest Parquet SHA-256: {path}")
        declared[str(path)] = sha256

    records: list[dict[str, Any]] = []
    for raw_path in selected:
        source_path = Path(raw_path)
        path = source_path.resolve()
        expected = declared.get(str(path))
        if expected is None:
            raise RuntimeError(f"selected Parquet file is absent from manifest: {path}")
        if (
            source_path.is_symlink()
            or not path.is_file()
            or not path.is_relative_to(root)
        ):
            raise RuntimeError(f"selected Parquet file is unsafe or missing: {path}")
        if verify_parquet_files:
            actual = _sha256_file(path)
            if actual != expected:
                raise RuntimeError(
                    f"selected Parquet SHA-256 changed: {path}: {actual} != {expected}"
                )
        records.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": expected,
            }
        )

    records_payload = json.dumps(
        records, sort_keys=True, separators=(",", ":")
    ).encode()
    body = {
        "dataset_root": str(root),
        "manifest_file_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "manifest_declared_sha256": declared_manifest_sha256,
        "selected_parquet_files": len(records),
        "selected_parquet_physical_bytes": sum(item["bytes"] for item in records),
        "selected_parquet_sha256_digest": hashlib.sha256(records_payload).hexdigest(),
    }
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return {**body, "identity_digest": hashlib.sha256(encoded).hexdigest()}


def frozen_campaign(dataset_root: Path) -> dict[str, Any]:
    plan, _ = exact_plan(dataset_root)
    smoke = plan_dict(smoke_slices(load_manifest(dataset_root)), kind="streaming-smoke")
    body = {
        "schema_version": 1,
        "question": (
            "Streaming GPU local-RunStore sort versus streaming CPU push-based "
            "PyArrow for one identical Parquet-to-Parquet pipeline"
        ),
        "workload": {
            "rows": TARGET_ROWS,
            "planned_input_blocks": TARGET_BLOCKS,
            "decoded_input_bytes": TARGET_DECODED_BYTES,
            "columns": 110,
            "key": "Origin",
            "descending": False,
            "nulls": "last",
            "plan_digest": plan["digest"],
            "row_id_sum": TARGET_ROW_ID_SUM,
            "origin_cardinality": TARGET_ORIGIN_CARDINALITY,
            "origin_frequency_digest": TARGET_ORIGIN_FREQUENCY_DIGEST,
        },
        "dataset_identity": _dataset_identity(dataset_root, verify_parquet_files=False),
        "resources": {
            "logical_cpus": EXPECTED_CPUS,
            "gpu_arm_v100s": EXPECTED_GPUS,
            "object_store_bytes": OBJECT_STORE_BYTES,
            "ray_target_max_block_size_bytes": 512 << 20,
            "gpu_input_buffer_budget_bytes": INPUT_BUFFER_BUDGET_BYTES,
            "gpu_exchange_chunk_bytes": 512 << 20,
            "gpu_run_chunk_bytes": 512 << 20,
            "gpu_merge_fan_in": 4,
            "gpu_wave_fraction": 0.5,
            "gpu_sample_seed": 0,
            "gpu_streaming_sample_rows_per_input_block": (SAMPLE_ROWS_PER_INPUT_BLOCK),
            "gpu_local_run_min_free_bytes": LOCAL_RUN_MIN_FREE_BYTES,
            "gpu_communication_environment": GPU_COMMUNICATION_ENVIRONMENT,
        },
        "parquet_sink": {
            "compression": "zstd",
            "compression_level": 3,
            "dictionary": True,
            "version": "2.6",
            "store_schema": True,
            "write_statistics": True,
            "data_page_size": 1 << 20,
            "target_decoded_row_group_bytes": TARGET_ROW_GROUP_BYTES,
            "target_decoded_file_bytes": TARGET_FILE_BYTES,
            "max_concurrent_writers": MAX_WRITERS,
            "max_decoded_writer_input_in_flight": MAX_WRITER_INPUT_BYTES,
        },
        "timing_boundary": (
            "immediately before write_datasink triggers the first lazy reader "
            "through data files, manifest/_SUCCESS, and syncfs completion"
        ),
        "smoke": {
            "blocks": smoke["blocks"],
            "rows": smoke["rows"],
            "decoded_bytes": SMOKE_DECODED_BYTES,
            "digest": smoke["digest"],
            "sequence": [asdict(item) for item in SMOKE_SEQUENCE],
        },
        "prerequisite_gates": [
            {"name": item.name, "nodeid": item.nodeid} for item in GATE_SEQUENCE
        ],
        "performance_sequence": [asdict(item) for item in RUN_SEQUENCE],
        "ray": {
            "version": RAY_VERSION,
            "commit": RAY_COMMIT,
            "wheel_sha256": RAY_WHEEL_SHA256,
        },
    }
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return {**body, "campaign_digest": hashlib.sha256(encoded).hexdigest()}


def preflight(
    dataset_root: Path,
    *,
    campaign_root: Path | None = None,
    require_clean_git: bool = False,
) -> dict[str, Any]:
    reasons: list[str] = []
    if dataset_root.resolve() != DATASET_ROOT.resolve():
        reasons.append(
            f"dataset root is {dataset_root.resolve()}, expected frozen {DATASET_ROOT.resolve()}"
        )
    try:
        plan, slices = exact_plan(dataset_root)
        dataset_identity = _dataset_identity(dataset_root, verify_parquet_files=True)
        missing = sorted(
            {item.path for item in slices if not Path(item.path).is_file()}
        )
        if missing:
            reasons.append(f"missing {len(missing)} frozen source files")
    except BaseException as error:
        plan, slices, missing, dataset_identity = {}, (), [], {}
        reasons.append(f"frozen input: {type(error).__name__}: {error}")

    cpus = os.cpu_count() or 0
    if cpus != EXPECTED_CPUS:
        reasons.append(f"logical CPU count is {cpus}, expected {EXPECTED_CPUS}")
    memory = _meminfo()
    if int(memory.get("MemAvailable", 0)) < MIN_HOST_AVAILABLE_BYTES:
        reasons.append("host available memory is below the frozen preflight minimum")
    usage = shutil.disk_usage(RAID_ROOT)
    if usage.free < MIN_CAMPAIGN_FREE_BYTES:
        reasons.append(
            f"/raid has {usage.free} free bytes, need {MIN_CAMPAIGN_FREE_BYTES}"
        )
    gpu = _nvidia_inventory()
    if not gpu.get("valid"):
        reasons.extend(
            f"GPU preflight: {item}" for item in gpu.get("rejection_reasons", [])
        )
    ray_processes = _ray_process_inventory()
    if not ray_processes.get("valid"):
        reasons.append("an existing Ray runtime is active; refusing to interfere")
    git = _git_identity(_repo_root())
    if require_clean_git:
        if not git.get("clean"):
            reasons.append("benchmark execution requires a clean committed worktree")
        if git.get("branch") != IMPLEMENTATION_BRANCH:
            reasons.append(
                f"benchmark execution requires branch {IMPLEMENTATION_BRANCH!r}"
            )
        if not git.get("streaming_sort_base_is_ancestor"):
            reasons.append(
                "benchmark commit does not descend from the tested sort branch"
            )
        if git.get("head") == STREAMING_SORT_BASE_COMMIT:
            reasons.append("benchmark implementation has not been committed")

    campaign_value = None
    if campaign_root is not None:
        try:
            campaign_value = str(
                _safe_campaign_root(campaign_root, must_not_exist=True)
            )
        except (ValueError, OSError) as error:
            reasons.append(f"campaign root: {error}")
    return {
        "valid": not reasons,
        "rejection_reasons": reasons,
        "plan": {key: plan.get(key) for key in ("kind", "rows", "blocks", "digest")},
        "dataset_identity": dataset_identity,
        "missing_source_files": missing,
        "logical_cpus": cpus,
        "memory": {
            name: memory.get(name, 0)
            for name in ("MemTotal", "MemAvailable", "Cached", "Dirty", "Writeback")
        },
        "raid": {
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
        },
        "gpu": gpu,
        "ray_processes": ray_processes,
        "git": git,
        "campaign_root": campaign_value,
    }


def _runtime_capacity_snapshot() -> dict[str, Any]:
    memory = _meminfo()
    usage = shutil.disk_usage(RAID_ROOT)
    reasons: list[str] = []
    if int(memory.get("MemAvailable", 0)) < MIN_HOST_AVAILABLE_BYTES:
        reasons.append("host available memory is below the frozen runtime minimum")
    if usage.free < MIN_CAMPAIGN_FREE_BYTES:
        reasons.append("/raid free space is below the frozen runtime minimum")
    return {
        "valid": not reasons,
        "rejection_reasons": reasons,
        "minimum_host_available_bytes": MIN_HOST_AVAILABLE_BYTES,
        "minimum_raid_free_bytes": MIN_CAMPAIGN_FREE_BYTES,
        "memory": {
            name: memory.get(name, 0)
            for name in ("MemTotal", "MemAvailable", "Cached", "Dirty", "Writeback")
        },
        "raid": {
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
        },
    }


def _worker_command(
    *,
    python: Path,
    dataset_root: Path,
    campaign_root: Path,
    spec: TrialSpec,
    expected_head: str,
    expected_dataset_identity: Mapping[str, Any],
) -> tuple[list[str], Path, Path]:
    artifact = campaign_root / "artifacts" / spec.workload / f"{spec.name}.json"
    runtime = campaign_root / "runtime" / spec.name
    expected_digest = (
        TARGET_PLAN_DIGEST
        if spec.workload == "full"
        else frozen_campaign(dataset_root)["smoke"]["digest"]
    )
    command = [
        str(_python_invocation_path(python)),
        "-m",
        PACKAGE,
        "--output",
        str(artifact),
        "--runtime",
        str(runtime),
        "--dataset-root",
        str(dataset_root),
        "--trial-name",
        spec.name,
        "--backend",
        spec.backend,
        "--workload",
        spec.workload,
        "--expected-plan-digest",
        expected_digest,
        "--expected-git-head",
        expected_head,
        "--expected-dataset-identity-digest",
        str(expected_dataset_identity["identity_digest"]),
    ]
    return command, artifact, runtime


def _copy_compact_evidence(
    campaign_root: Path, spec: TrialSpec, artifact: Mapping[str, Any]
) -> dict[str, str]:
    evidence = campaign_root / "evidence" / spec.workload / spec.name
    evidence.mkdir(parents=True, exist_ok=False)
    copied: dict[str, str] = {}
    sink = artifact.get("sink", {})
    if not sink and isinstance(artifact.get("failure_evidence"), Mapping):
        sink = artifact["failure_evidence"].get("sink", {})
    for name, key in (("manifest.json", "manifest"), ("_SUCCESS", "success")):
        value = sink.get(key) if isinstance(sink, Mapping) else None
        if value is not None:
            destination = evidence / name
            destination.write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            copied[key] = str(destination)
    return copied


def _remove_trial_runtime(
    campaign_root: Path, runtime: Path, *, process_group_terminated: bool
) -> dict[str, Any]:
    started = time.perf_counter()
    root = campaign_root.resolve()
    resolved = runtime.resolve(strict=False)
    expected_parent = (root / "runtime").resolve(strict=False)
    result: dict[str, Any] = {
        "path": str(resolved),
        "outside_primary_timer": True,
        "removed": False,
        "error": None,
    }
    try:
        if resolved.parent != expected_parent or not resolved.name:
            raise RuntimeError(f"refusing unsafe runtime cleanup: {resolved}")
        if runtime.is_symlink():
            raise RuntimeError(f"refusing symlink runtime cleanup: {runtime}")
        if not process_group_terminated:
            raise RuntimeError("worker process group remains alive")
        files = logical_bytes = 0
        if resolved.is_dir():
            for directory, _, names in os.walk(resolved, followlinks=False):
                for name in names:
                    path = Path(directory) / name
                    files += 1
                    try:
                        logical_bytes += path.stat(follow_symlinks=False).st_size
                    except FileNotFoundError:
                        pass
            shutil.rmtree(resolved)
        token = hashlib.sha256(str(resolved).encode()).hexdigest()[:12]
        shared = Path("/dev/shm/rgs") / token
        if shared.parent == Path("/dev/shm/rgs") and len(shared.name) == 12:
            shutil.rmtree(shared, ignore_errors=True)
        result.update(
            {
                "removed": not resolved.exists(),
                "files_removed": files,
                "logical_file_bytes_removed": logical_bytes,
            }
        )
        if resolved.exists():
            raise RuntimeError(f"runtime still exists after cleanup: {resolved}")
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
    result["elapsed_s"] = time.perf_counter() - started
    return result


def _read_optional_json(path: Path) -> dict[str, Any]:
    try:
        if path.is_file():
            return read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    return {}


def _read_source_event_progress(path: Path) -> dict[str, Any]:
    started: dict[int, Mapping[str, Any]] = {}
    produced: dict[int, Mapping[str, Any]] = {}
    failed: set[int] = set()
    if not path.is_file():
        return {"available": False}
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.endswith("\n"):
                    break
                value = json.loads(line)
                ordinal = int(value["ordinal"])
                event = value.get("event")
                if event == "started":
                    started[ordinal] = value
                elif event == "produced":
                    produced[ordinal] = value
                elif event == "failed":
                    failed.add(ordinal)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {"available": False, "error": f"{type(error).__name__}: {error}"}
    return {
        "available": True,
        "started_tasks": len(started),
        "produced_tasks": len(produced),
        "failed_tasks": len(failed),
        "produced_rows": sum(int(item.get("rows", 0)) for item in produced.values()),
        "produced_decoded_bytes": sum(
            int(item.get("decoded_bytes", 0)) for item in produced.values()
        ),
        "planned_decoded_bytes": sum(
            int(item.get("planned_size_bytes", 0)) for item in produced.values()
        ),
        "first_started_wall_time_ns": min(
            (
                int(item.get("wall_time_ns", 0))
                for item in started.values()
                if item.get("wall_time_ns") is not None
            ),
            default=None,
        ),
        "last_produced_wall_time_ns": max(
            (
                int(item.get("wall_time_ns", 0))
                for item in produced.values()
                if item.get("wall_time_ns") is not None
            ),
            default=None,
        ),
    }


def _phase_e2e_elapsed_s(phase: Mapping[str, Any]) -> float | None:
    value = phase.get("e2e_elapsed_s")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _timed_pipeline_is_running(phase: Mapping[str, Any]) -> bool:
    return bool(
        phase.get("phase") == "pipeline-running"
        and phase.get("timed_pipeline_started") is True
    )


def _oom_kill_count() -> int | None:
    """Read the host kernel's monotonic OOM-kill counter, when available."""

    try:
        with Path("/proc/vmstat").open(encoding="utf-8") as stream:
            for line in stream:
                name, value = line.split()
                if name == "oom_kill":
                    return int(value)
    except (OSError, ValueError):
        return None
    return None


def _memory_cgroup_oom_snapshot() -> dict[str, Any] | None:
    """Read the current session cgroup's local OOM counters on cgroup v2."""

    try:
        unified = None
        for line in Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines():
            hierarchy, controllers, relative = line.split(":", 2)
            if hierarchy == "0" and controllers == "":
                unified = relative
                break
        if unified is None:
            return None
        root = Path("/sys/fs/cgroup").resolve()
        group = (root / unified.lstrip("/")).resolve()
        if group != root and not group.is_relative_to(root):
            return None
        local = group / "memory.events.local"
        events = local if local.is_file() else group / "memory.events"
        values: dict[str, int] = {}
        for line in events.read_text(encoding="utf-8").splitlines():
            name, raw = line.split()
            values[name] = int(raw)
        return {
            "source": str(events),
            "cgroup": unified,
            "oom": values.get("oom"),
            "oom_kill": values.get("oom_kill"),
        }
    except (OSError, ValueError):
        return None


def _oom_snapshot() -> dict[str, Any]:
    return {
        "host_vmstat_oom_kill": _oom_kill_count(),
        "session_cgroup": _memory_cgroup_oom_snapshot(),
    }


def _counter_delta(before: Any, after: Any) -> int | None:
    if (
        isinstance(before, int)
        and not isinstance(before, bool)
        and isinstance(after, int)
        and not isinstance(after, bool)
    ):
        return after - before
    return None


def _ray_process_crash_signature(artifact: Mapping[str, Any]) -> str | None:
    """Return the Ray process-crash marker present in a failed worker artifact."""

    # Source telemetry can contain one record per input block.  It is both
    # large and irrelevant to this process-crash proof, so inspect only the
    # worker's exception and durable failure summaries.
    fragments: list[Any] = [
        artifact.get("exception"),
        artifact.get("rejection_reasons"),
    ]
    phase = artifact.get("last_durable_phase")
    if isinstance(phase, Mapping):
        fragments.append(phase.get("pipeline_failure"))
    text = json.dumps(fragments, sort_keys=True, default=str).lower()
    for marker in (
        "workercrashederror",
        "worker died unexpectedly",
        "localrayletdiederror",
        "connection to the raylet",
        "raylet exited",
        "raylet has been terminated",
    ):
        if marker in text:
            return marker
    return None


def _cgroup_oom_evidence(
    before: Mapping[str, Any] | None,
    after: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Return a same-session cgroup OOM-kill delta, if it is trustworthy."""

    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        return None
    source = before.get("source")
    cgroup = before.get("cgroup")
    if (
        not isinstance(source, str)
        or Path(source).name not in {"memory.events", "memory.events.local"}
        or not Path(source).is_absolute()
        or not Path(source).is_relative_to(Path("/sys/fs/cgroup"))
        or after.get("source") != source
        or not isinstance(cgroup, str)
        or after.get("cgroup") != cgroup
    ):
        return None
    delta = _counter_delta(before.get("oom_kill"), after.get("oom_kill"))
    if delta is None:
        return None
    return {
        "source": source,
        "cgroup": cgroup,
        "before": before.get("oom_kill"),
        "after": after.get("oom_kill"),
        "delta": delta,
        "oom_before": before.get("oom"),
        "oom_after": after.get("oom"),
    }


def _wait_for_worker(
    process: subprocess.Popen[Any],
    *,
    runtime: Path,
    pipeline_timeout_s: int,
    post_pipeline_timeout_s: int,
) -> tuple[int, bool, str | None, dict[str, Any]]:
    """Apply the 24-hour limit only to the durable timed pipeline phase."""

    launched = time.monotonic()
    phase_path = runtime / "telemetry" / "phase.json"
    pipeline_deadline: float | None = None
    post_deadline: float | None = None
    last_phase: dict[str, Any] = {}
    while True:
        returncode = process.poll()
        if returncode is not None:
            return returncode, False, None, last_phase

        observed = _read_optional_json(phase_path)
        if observed:
            last_phase = observed
        phase = str(last_phase.get("phase", "preparing"))
        now = time.monotonic()
        if phase == "pipeline-running" and pipeline_deadline is None:
            started_wall_ns = int(last_phase.get("wall_time_ns", time.time_ns()))
            already_elapsed = max(0.0, (time.time_ns() - started_wall_ns) / 1e9)
            pipeline_deadline = now + max(0.0, pipeline_timeout_s - already_elapsed)
        terminal_pipeline_phase = phase in {
            "pipeline-finished",
            "validation-running",
            "validation-finished",
            "worker-complete",
        }
        persisted_elapsed = _phase_e2e_elapsed_s(last_phase)

        timeout_phase: str | None = None
        # Check the persisted primary interval before changing to the
        # post-pipeline watchdog.  This catches a worker that publishes
        # pipeline-finished just after its deadline between controller polls.
        if persisted_elapsed is not None and persisted_elapsed > pipeline_timeout_s:
            timeout_phase = "pipeline"
        elif terminal_pipeline_phase:
            pipeline_deadline = None
            if post_deadline is None:
                post_deadline = now + post_pipeline_timeout_s

        if (
            timeout_phase is None
            and pipeline_deadline is not None
            and now >= pipeline_deadline
        ):
            timeout_phase = "pipeline"
        elif (
            timeout_phase is None and post_deadline is not None and now >= post_deadline
        ):
            timeout_phase = "post-pipeline"
        elif timeout_phase is None and (
            pipeline_deadline is None
            and post_deadline is None
            and (now - launched >= PRE_PIPELINE_TIMEOUT_S)
        ):
            timeout_phase = "pre-pipeline"
        if timeout_phase is not None:
            _terminate_process_group(process)
            returncode = process.poll()
            if returncode is None:
                returncode = 124
            return returncode, True, timeout_phase, last_phase
        time.sleep(WORKER_POLL_S)


def _artifact_pipeline_elapsed_s(
    artifact: Mapping[str, Any], phase: Mapping[str, Any]
) -> float | None:
    timings = artifact.get("timings_s")
    if isinstance(timings, Mapping):
        for key in ("e2e", "e2e_until_failure"):
            value = timings.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
    return _phase_e2e_elapsed_s(phase)


def _artifact_capacity_phase(
    artifact: Mapping[str, Any], last_phase: Mapping[str, Any]
) -> Mapping[str, Any]:
    durable = artifact.get("last_durable_phase")
    if isinstance(durable, Mapping):
        return durable
    return last_phase


def _mark_rejected(artifact: dict[str, Any], reason: str) -> None:
    artifact["valid"] = False
    artifact["status"] = "rejected"
    artifact.setdefault("rejection_reasons", []).append(reason)


def _apply_controller_outcome(
    artifact: dict[str, Any],
    *,
    spec: TrialSpec,
    returncode: int,
    timed_out: bool,
    timeout_phase: str | None,
    last_phase: Mapping[str, Any],
    timeout_s: int,
    oom_before: int | None,
    oom_after: int | None,
    cgroup_oom_before: Mapping[str, Any] | None = None,
    cgroup_oom_after: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reconcile worker claims with watchdog, phase, and kernel evidence."""

    phase = _artifact_capacity_phase(artifact, last_phase)
    elapsed = _artifact_pipeline_elapsed_s(artifact, phase)
    cpu_full = spec.backend == "pyarrow" and spec.workload == "full"
    timed_oom_baseline = phase.get("kernel_oom_baseline")
    timed_host_before = (
        timed_oom_baseline.get("host_vmstat_oom_kill")
        if isinstance(timed_oom_baseline, Mapping)
        else None
    )
    timed_cgroup_before = (
        timed_oom_baseline.get("session_cgroup")
        if isinstance(timed_oom_baseline, Mapping)
        else None
    )
    timed_oom_finished = phase.get("kernel_oom_finished")
    timed_cgroup_finished = (
        timed_oom_finished.get("session_cgroup")
        if isinstance(timed_oom_finished, Mapping)
        else None
    )
    host_oom_delta = _counter_delta(timed_host_before, oom_after)
    cgroup_oom = _cgroup_oom_evidence(timed_cgroup_before, cgroup_oom_after)
    terminal_cgroup_oom = _cgroup_oom_evidence(
        timed_cgroup_before, timed_cgroup_finished
    )
    # Production DGX runs expose a session cgroup.  Host-global vmstat is only
    # a portability fallback when the timed phase and controller both prove
    # that cgroup counters were unavailable.  The pre-launch snapshots are
    # diagnostic only because they include Ray startup and graph construction.
    oom_evidence = cgroup_oom
    if (
        isinstance(timed_oom_baseline, Mapping)
        and timed_cgroup_before is None
        and cgroup_oom_after is None
    ):
        if host_oom_delta is not None:
            oom_evidence = {
                "source": "/proc/vmstat:oom_kill",
                "before": timed_host_before,
                "after": oom_after,
                "delta": host_oom_delta,
            }
    capacity_evidence: dict[str, Any] | None = None

    # A persisted primary interval is authoritative even when the worker
    # crossed its deadline and changed phase between controller polls.
    if spec.workload == "full" and elapsed is not None and elapsed > timeout_s:
        if cpu_full:
            artifact["valid"] = False
            artifact["status"] = "capacity-timeout"
            artifact.setdefault("rejection_reasons", []).append(
                f"CPU timed pipeline took {elapsed:.9f}s, exceeding the frozen "
                f"{timeout_s}-second limit"
            )
            capacity_evidence = {
                "kind": "persisted-e2e-deadline",
                "e2e_elapsed_s": elapsed,
                "limit_s": timeout_s,
                "phase": dict(phase),
            }
        else:
            _mark_rejected(
                artifact,
                f"timed pipeline took {elapsed:.9f}s, exceeding the frozen "
                f"{timeout_s}-second limit",
            )
    elif timed_out:
        phase_proves_timed_cpu = bool(
            cpu_full
            and timeout_phase == "pipeline"
            and (
                _timed_pipeline_is_running(phase)
                or (
                    phase.get("phase") == "pipeline-finished"
                    and phase.get("timed_pipeline_finished") is True
                    and _phase_e2e_elapsed_s(phase) is not None
                    and float(phase["e2e_elapsed_s"]) > timeout_s
                )
            )
        )
        if phase_proves_timed_cpu:
            artifact["valid"] = False
            artifact["status"] = "capacity-timeout"
            artifact.setdefault("rejection_reasons", []).append(
                f"CPU timed pipeline exceeded the frozen {timeout_s}-second limit"
            )
            capacity_evidence = {
                "kind": "controller-pipeline-deadline",
                "limit_s": timeout_s,
                "phase": dict(phase),
            }
        else:
            _mark_rejected(
                artifact,
                f"worker exceeded the {timeout_phase} watchdog; this is not a "
                "phase-proven CPU pipeline capacity result",
            )
    elif (
        cpu_full
        and returncode != 0
        and returncode != -9
        and isinstance(terminal_cgroup_oom, Mapping)
        and int(terminal_cgroup_oom.get("delta", 0)) > 0
        and phase.get("phase") in {"pipeline-finished", "worker-complete"}
        and phase.get("timed_pipeline_started") is True
        and phase.get("timed_pipeline_finished") is True
        and phase.get("pipeline_succeeded") is False
        and (crash_marker := _ray_process_crash_signature(artifact)) is not None
    ):
        artifact["valid"] = False
        artifact["status"] = "capacity-oom"
        artifact.setdefault("rejection_reasons", []).append(
            "a Ray worker or raylet failed during the CPU timed pipeline while "
            "the trial session cgroup OOM-kill counter increased"
        )
        capacity_evidence = {
            "kind": "session-cgroup-ray-process-oom",
            **terminal_cgroup_oom,
            "ray_process_crash_marker": crash_marker,
            "phase": dict(phase),
        }
    elif returncode == -9:
        kernel_oom_proved = bool(
            cpu_full
            and _timed_pipeline_is_running(phase)
            and isinstance(oom_evidence, Mapping)
            and int(oom_evidence.get("delta", 0)) > 0
        )
        if kernel_oom_proved:
            artifact["valid"] = False
            artifact["status"] = "capacity-oom"
            artifact.setdefault("rejection_reasons", []).append(
                "CPU timed pipeline received SIGKILL while its available "
                "kernel OOM-kill counter increased"
            )
            capacity_evidence = {
                "kind": "kernel-oom-kill",
                **dict(oom_evidence),
                "phase": dict(phase),
            }
        else:
            _mark_rejected(
                artifact,
                "worker received unexplained SIGKILL; acceptance requires a "
                "durable pipeline-running phase and an increased kernel "
                "OOM-kill counter",
            )
    elif artifact.get("valid") and returncode != 0:
        _mark_rejected(
            artifact,
            f"worker claimed success but exited with status {returncode}",
        )

    if artifact.get("status") == "capacity-failure" and returncode != 2:
        _mark_rejected(
            artifact,
            "worker capacity-failure did not use its frozen exit status 2",
        )
    if capacity_evidence is not None:
        artifact["controller_capacity_evidence"] = capacity_evidence
    return artifact


def _run_worker(
    *,
    root: Path,
    campaign_root: Path,
    spec: TrialSpec,
    command: list[str],
    artifact_path: Path,
    runtime: Path,
    timeout_s: int,
    expected_dataset_identity: Mapping[str, Any],
) -> dict[str, Any]:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    log = campaign_root / "logs" / spec.workload / f"{spec.name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    if artifact_path.exists() or log.exists() or runtime.exists():
        raise FileExistsError(f"refusing to overwrite observation {spec.name}")

    started = time.time()
    timed_out = False
    timeout_phase: str | None = None
    last_phase: dict[str, Any] = {}
    process_group_terminated = True
    oom_before_snapshot: dict[str, Any] = {}
    oom_after_snapshot: dict[str, Any] = {}
    with log.open("x", encoding="utf-8") as stream:
        # Capture this immediately before process creation so a SIGKILL can be
        # retained as CPU capacity evidence only when the kernel OOM counter
        # actually advances during this worker's timed lifetime.
        oom_before_snapshot = _oom_snapshot()
        process = subprocess.Popen(
            command,
            cwd=root,
            env={**os.environ, "PYTHONPATH": str(root)},
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        returncode, timed_out, timeout_phase, last_phase = _wait_for_worker(
            process,
            runtime=runtime,
            pipeline_timeout_s=timeout_s,
            post_pipeline_timeout_s=(
                POST_PIPELINE_TIMEOUT_S if spec.workload == "full" else SMOKE_TIMEOUT_S
            ),
        )
        oom_after_snapshot = _oom_snapshot()
        if _process_group_alive(process.pid):
            process_group_terminated = _terminate_process_group(process)

    if artifact_path.is_file():
        artifact = read_json(artifact_path)
    else:
        progress = _read_optional_json(runtime / "telemetry" / "progress.json")
        artifact = {
            "valid": False,
            "status": "missing-artifact",
            "trial_name": spec.name,
            "backend": spec.backend,
            "workload": spec.workload,
            "dataset_identity": dict(expected_dataset_identity),
            "timed_pipeline_started": bool(last_phase.get("timed_pipeline_started")),
            "last_durable_phase": last_phase,
            "external_failure_evidence": {
                "progress_checkpoint": progress,
                "phase": last_phase,
                "source": _read_source_event_progress(
                    runtime / "telemetry" / "source" / "source-events.jsonl"
                ),
            },
            "rejection_reasons": [
                f"worker produced no artifact (exit {returncode}); inspect {log}"
            ],
        }
    _apply_controller_outcome(
        artifact,
        spec=spec,
        returncode=returncode,
        timed_out=timed_out,
        timeout_phase=timeout_phase,
        last_phase=last_phase,
        timeout_s=timeout_s,
        oom_before=oom_before_snapshot.get("host_vmstat_oom_kill"),
        oom_after=oom_after_snapshot.get("host_vmstat_oom_kill"),
        cgroup_oom_before=oom_before_snapshot.get("session_cgroup"),
        cgroup_oom_after=oom_after_snapshot.get("session_cgroup"),
    )
    if artifact.get("dataset_identity") != dict(expected_dataset_identity):
        _mark_rejected(
            artifact,
            "worker dataset identity differs from the controller's immediately "
            "pre-launch verification",
        )
    if not process_group_terminated:
        _mark_rejected(
            artifact, "worker process group remained alive after the observation"
        )
    capacity_evidence = artifact.pop("controller_capacity_evidence", None)
    artifact["controller"] = {
        "command": command,
        "returncode": returncode,
        "log": str(log),
        "wall_started_unix_s": started,
        "wall_finished_unix_s": time.time(),
        "timed_out": timed_out,
        "timeout_phase": timeout_phase,
        "last_durable_phase": last_phase,
        "process_group_terminated": process_group_terminated,
        "kernel_oom_kill_counter": {
            "source": "/proc/vmstat:oom_kill",
            "before": oom_before_snapshot.get("host_vmstat_oom_kill"),
            "after": oom_after_snapshot.get("host_vmstat_oom_kill"),
            "delta": _counter_delta(
                oom_before_snapshot.get("host_vmstat_oom_kill"),
                oom_after_snapshot.get("host_vmstat_oom_kill"),
            ),
        },
        "session_cgroup_oom_counter": {
            "before": oom_before_snapshot.get("session_cgroup"),
            "after": oom_after_snapshot.get("session_cgroup"),
            "evidence": _cgroup_oom_evidence(
                oom_before_snapshot.get("session_cgroup"),
                oom_after_snapshot.get("session_cgroup"),
            ),
        },
        "capacity_evidence": capacity_evidence,
    }
    sink_evidence = artifact.get("sink") or artifact.get("failure_evidence", {}).get(
        "sink"
    )
    if sink_evidence:
        artifact["controller"]["compact_evidence"] = _copy_compact_evidence(
            campaign_root, spec, artifact
        )
    cleanup = _remove_trial_runtime(
        campaign_root, runtime, process_group_terminated=process_group_terminated
    )
    artifact["controller"]["runtime_cleanup"] = cleanup
    if cleanup.get("error"):
        _mark_rejected(
            artifact,
            f"trial-owned runtime cleanup failed: {cleanup['error']}",
        )
    write_json(artifact_path, artifact)
    return artifact


def _assert_frozen_checkout(root: Path, expected_head: str) -> None:
    identity = _git_identity(root)
    if not identity.get("clean"):
        raise RuntimeError(
            f"worktree changed during campaign: {identity.get('status')}"
        )
    if identity.get("head") != expected_head:
        raise RuntimeError(
            f"campaign commit changed from {expected_head} to {identity.get('head')}"
        )
    if identity.get("branch") != IMPLEMENTATION_BRANCH:
        raise RuntimeError(f"campaign branch changed to {identity.get('branch')!r}")


def _install_frozen_overlay(root: Path) -> dict[str, Any]:
    """Install and fingerprint the clean checkout over the exact Ray wheel."""

    from .runner import _install_ray_data_overlay, _wheel_identity

    wheel = _wheel_identity(root)
    manifest = _install_ray_data_overlay(root, wheel)
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return {
        **manifest,
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        "installed_from_clean_checkout": True,
    }


def _assert_frozen_overlay(root: Path, expected: Mapping[str, Any]) -> None:
    source = Path(str(expected.get("source", ""))).resolve()
    target = Path(str(expected.get("target", ""))).resolve()
    if source != (root / "python/ray/data").resolve():
        raise RuntimeError("overlay source changed during campaign")
    if not target.is_relative_to((root / ".venv").resolve()):
        raise RuntimeError("overlay target changed during campaign")
    files = expected.get("files")
    if not isinstance(files, Mapping):
        raise RuntimeError("overlay manifest has no file hashes")
    for relative, digest in files.items():
        source_path = source / str(relative)
        target_path = target / str(relative)
        if not source_path.is_file() or not target_path.is_file():
            raise RuntimeError(f"overlay file disappeared: {relative}")
        source_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        target_digest = hashlib.sha256(target_path.read_bytes()).hexdigest()
        if source_digest != digest or target_digest != digest:
            raise RuntimeError(f"overlay file changed during campaign: {relative}")
    target_python = {
        str(path.relative_to(target))
        for path in target.rglob("*.py")
        if "__pycache__" not in path.parts
    }
    if target_python != set(files):
        raise RuntimeError("overlay target file set changed during campaign")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _python_invocation_path(python: Path) -> Path:
    """Return an absolute executable path without resolving its venv symlink."""

    return Path(os.path.abspath(python))


def _assert_frozen_python(python: Path) -> Path:
    root = _repo_root()
    expected = _python_invocation_path(root / ".venv" / "bin" / "python")
    actual = _python_invocation_path(python)
    if actual != expected or not actual.is_file():
        raise RuntimeError(f"benchmark Python is {actual}, expected {expected}")
    if Path(sys.prefix).resolve() != (root / ".venv").resolve():
        raise RuntimeError(
            f"controller prefix is {Path(sys.prefix).resolve()}, expected "
            f"{(root / '.venv').resolve()}"
        )
    return actual


def _canonical_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _gate_command(python: Path, gate: GateSpec) -> list[str]:
    return [str(_python_invocation_path(python)), "-m", "pytest", "-sv", gate.nodeid]


def _run_gate_process(
    *,
    root: Path,
    gate_root: Path,
    python: Path,
    gate: GateSpec,
    expected_head: str,
    overlay: Mapping[str, Any],
) -> dict[str, Any]:
    log = gate_root / "logs" / f"{gate.name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = _gate_command(python, gate)
    environment = dict(gate.environment)
    pre_gpu = _wait_for_gpu_idle()
    pre_ray = _ray_process_inventory()
    started = time.time()
    returncode = 125
    timed_out = False
    process_group_terminated = True
    error: str | None = None
    if not pre_ray.get("valid") or not pre_gpu.get("valid"):
        error = "resource-idle precheck failed"
        log.write_text(error + "\n", encoding="utf-8")
    else:
        try:
            with log.open("x", encoding="utf-8") as stream:
                process = subprocess.Popen(
                    command,
                    cwd=root,
                    env={
                        **os.environ,
                        "PYTHONPATH": str(root),
                        **environment,
                    },
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    returncode = process.wait(timeout=SMOKE_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    process_group_terminated = _terminate_process_group(process)
                    returncode = process.poll() if process.poll() is not None else 124
                if _process_group_alive(process.pid):
                    process_group_terminated = _terminate_process_group(process)
        except BaseException as caught:
            error = f"{type(caught).__name__}: {caught}"

    post_gpu = _wait_for_gpu_idle()
    post_ray = _ray_process_inventory()
    try:
        _assert_frozen_checkout(root, expected_head)
        _assert_frozen_overlay(root, overlay)
        identity_valid = True
        identity_error = None
    except BaseException as caught:
        identity_valid = False
        identity_error = f"{type(caught).__name__}: {caught}"
    valid = bool(
        error is None
        and returncode == 0
        and not timed_out
        and process_group_terminated
        and pre_ray.get("valid")
        and post_ray.get("valid")
        and pre_gpu.get("valid")
        and post_gpu.get("valid")
        and identity_valid
    )
    return {
        "name": gate.name,
        "valid": valid,
        "nodeid": gate.nodeid,
        "command": command,
        "environment": environment,
        "uses_ray": gate.uses_ray,
        "returncode": returncode,
        "timed_out": timed_out,
        "timeout_s": SMOKE_TIMEOUT_S,
        "process_group_terminated": process_group_terminated,
        "wall_started_unix_s": started,
        "wall_finished_unix_s": time.time(),
        "log": str(log.resolve()),
        "log_bytes": log.stat().st_size,
        "log_sha256": _sha256_file(log),
        "resource_checks": {
            "before": {"ray": pre_ray, "gpu": pre_gpu},
            "after": {"ray": post_ray, "gpu": post_gpu},
        },
        "identity_valid": identity_valid,
        "identity_error": identity_error,
        "error": error,
    }


def run_prerequisite_gates(
    dataset_root: Path, gate_root: Path, python: Path
) -> dict[str, Any]:
    """Run and durably bind all pre-1-TB proof tests to one code identity."""

    root = _repo_root()
    python = _assert_frozen_python(python)
    gate_root = _safe_campaign_root(gate_root, must_not_exist=True)
    check = preflight(
        dataset_root,
        campaign_root=gate_root,
        require_clean_git=True,
    )
    if not check["valid"]:
        raise RuntimeError(f"gate preflight rejected: {check['rejection_reasons']}")
    expected_head = str(check["git"]["head"])
    _assert_frozen_checkout(root, expected_head)
    overlay = _install_frozen_overlay(root)
    _assert_frozen_checkout(root, expected_head)
    _assert_frozen_overlay(root, overlay)

    gate_root.mkdir(parents=True, exist_ok=False)
    write_json(gate_root / "PREFLIGHT.json", check)
    write_json(gate_root / "RAY_DATA_OVERLAY.json", overlay)
    results = [
        _run_gate_process(
            root=root,
            gate_root=gate_root,
            python=python,
            gate=gate,
            expected_head=expected_head,
            overlay=overlay,
        )
        for gate in GATE_SEQUENCE
    ]
    body: dict[str, Any] = {
        "schema_version": GATE_SCHEMA_VERSION,
        "valid": all(item["valid"] for item in results),
        "git_head": expected_head,
        "git_branch": check["git"]["branch"],
        "overlay_manifest_sha256": overlay["manifest_sha256"],
        "python": str(_python_invocation_path(python)),
        "campaign_digest": frozen_campaign(dataset_root)["campaign_digest"],
        "dataset_root": str(dataset_root.resolve()),
        "dataset_identity": check["dataset_identity"],
        "gate_root": str(gate_root),
        "gates": results,
    }
    manifest = {**body, "manifest_digest": _canonical_digest(body)}
    write_json(gate_root / "GATES.json", manifest)
    return manifest


def _validate_prerequisite_gates(
    manifest_path: Path,
    *,
    dataset_root: Path,
    expected_head: str,
    overlay: Mapping[str, Any],
    python: Path,
) -> dict[str, Any]:
    resolved = manifest_path.resolve()
    raid = RAID_ROOT.resolve()
    if not resolved.is_file() or not resolved.is_relative_to(raid):
        raise RuntimeError("gate manifest must be an existing file below /raid")
    value = read_json(resolved)
    if value.get("schema_version") != GATE_SCHEMA_VERSION:
        raise RuntimeError("unsupported prerequisite gate schema")
    digest = value.get("manifest_digest")
    body = {key: item for key, item in value.items() if key != "manifest_digest"}
    if not isinstance(digest, str) or digest != _canonical_digest(body):
        raise RuntimeError("prerequisite gate manifest digest mismatch")
    if value.get("valid") is not True:
        raise RuntimeError("prerequisite gate suite was not valid")
    if value.get("git_head") != expected_head:
        raise RuntimeError("prerequisite gates used a different git commit")
    if value.get("git_branch") != IMPLEMENTATION_BRANCH:
        raise RuntimeError("prerequisite gates used a different branch")
    if value.get("overlay_manifest_sha256") != overlay.get("manifest_sha256"):
        raise RuntimeError("prerequisite gates used a different Ray Data overlay")
    if value.get("python") != str(_python_invocation_path(python)):
        raise RuntimeError("prerequisite gates used a different Python environment")
    if value.get("campaign_digest") != frozen_campaign(dataset_root)["campaign_digest"]:
        raise RuntimeError("prerequisite gates used a different frozen campaign")
    if value.get("dataset_root") != str(dataset_root.resolve()):
        raise RuntimeError("prerequisite gates used a different dataset root")
    current_dataset_identity = _dataset_identity(
        dataset_root, verify_parquet_files=True
    )
    if value.get("dataset_identity") != current_dataset_identity:
        raise RuntimeError("prerequisite gates used different Parquet source bytes")
    if Path(str(value.get("gate_root", ""))).resolve() != resolved.parent:
        raise RuntimeError("prerequisite gate root does not own its manifest")

    gates = value.get("gates")
    if not isinstance(gates, list) or len(gates) != len(GATE_SEQUENCE):
        raise RuntimeError("prerequisite gate set is incomplete")
    by_name = {item.get("name"): item for item in gates if isinstance(item, Mapping)}
    if set(by_name) != {gate.name for gate in GATE_SEQUENCE}:
        raise RuntimeError("prerequisite gate names changed")
    for expected in GATE_SEQUENCE:
        gate = by_name[expected.name]
        if not (
            gate.get("valid") is True
            and gate.get("nodeid") == expected.nodeid
            and gate.get("command") == _gate_command(python, expected)
            and gate.get("environment") == dict(expected.environment)
            and gate.get("uses_ray") is expected.uses_ray
            and gate.get("returncode") == 0
            and gate.get("timed_out") is False
            and gate.get("process_group_terminated") is True
            and gate.get("identity_valid") is True
            and gate.get("error") is None
        ):
            raise RuntimeError(f"prerequisite gate is not accepted: {expected.name}")
        checks = gate.get("resource_checks")
        if not isinstance(checks, Mapping) or any(
            not isinstance(checks.get(point), Mapping)
            or not isinstance(checks[point].get(resource), Mapping)
            or checks[point][resource].get("valid") is not True
            for point in ("before", "after")
            for resource in ("ray", "gpu")
        ):
            raise RuntimeError(
                f"prerequisite gate lacks resource isolation: {expected.name}"
            )
        raw_log = Path(str(gate.get("log", "")))
        log = raw_log.resolve()
        expected_log = (resolved.parent / "logs" / f"{expected.name}.log").resolve()
        if log != expected_log or not log.is_file() or raw_log.is_symlink():
            raise RuntimeError(f"prerequisite gate log is not owned: {expected.name}")
        if gate.get("log_bytes") != log.stat().st_size:
            raise RuntimeError(f"prerequisite gate log size changed: {expected.name}")
        if gate.get("log_sha256") != _sha256_file(log):
            raise RuntimeError(f"prerequisite gate log changed: {expected.name}")
    return {
        **value,
        "source_manifest": str(resolved),
        "source_manifest_sha256": _sha256_file(resolved),
    }


def _copy_prerequisite_gate_evidence(
    campaign_root: Path, validated: Mapping[str, Any]
) -> dict[str, Any]:
    destination = campaign_root / "evidence" / "prerequisite-gates"
    destination.mkdir(parents=True, exist_ok=False)
    source_manifest = Path(str(validated["source_manifest"]))
    copied_manifest = destination / "GATES.source.json"
    shutil.copyfile(source_manifest, copied_manifest)
    logs: dict[str, Any] = {}
    for gate in validated["gates"]:
        source = Path(str(gate["log"]))
        target = destination / f"{gate['name']}.log"
        shutil.copyfile(source, target)
        digest = _sha256_file(target)
        if digest != gate["log_sha256"]:
            raise RuntimeError(f"gate log changed while copying: {gate['name']}")
        logs[str(gate["name"])] = {
            "path": str(target),
            "bytes": target.stat().st_size,
            "sha256": digest,
        }
    return {
        "manifest": {
            "path": str(copied_manifest),
            "bytes": copied_manifest.stat().st_size,
            "sha256": _sha256_file(copied_manifest),
        },
        "logs": logs,
    }


def _accepted_capacity_result(artifact: Mapping[str, Any]) -> bool:
    if not (
        artifact.get("backend") == "pyarrow"
        and artifact.get("workload") == "full"
        and artifact.get("timed_pipeline_started") is True
    ):
        return False
    status = artifact.get("status")
    phase = _artifact_capacity_phase(artifact, {})
    controller = artifact.get("controller")
    if not isinstance(controller, Mapping):
        return False
    if status == "capacity-failure":
        return bool(
            controller.get("returncode") == 2
            and controller.get("timed_out") is False
            and phase.get("phase") == "pipeline-finished"
            and phase.get("timed_pipeline_started") is True
            and phase.get("timed_pipeline_finished") is True
            and phase.get("pipeline_succeeded") is False
            and phase.get("pipeline_failure_is_capacity") is True
        )

    evidence = controller.get("capacity_evidence")
    if not isinstance(evidence, Mapping):
        return False
    if status == "capacity-timeout":
        limit = evidence.get("limit_s")
        if not (
            isinstance(limit, (int, float))
            and not isinstance(limit, bool)
            and float(limit) == TRIAL_TIMEOUT_S
        ):
            return False
        if evidence.get("kind") == "persisted-e2e-deadline":
            elapsed = evidence.get("e2e_elapsed_s")
            return bool(
                isinstance(elapsed, (int, float))
                and not isinstance(elapsed, bool)
                and float(elapsed) > float(limit)
            )
        if evidence.get("kind") == "controller-pipeline-deadline":
            evidence_phase = evidence.get("phase")
            if not isinstance(evidence_phase, Mapping):
                return False
            phase_elapsed = _phase_e2e_elapsed_s(evidence_phase)
            return bool(
                _timed_pipeline_is_running(evidence_phase)
                or (
                    evidence_phase.get("phase") == "pipeline-finished"
                    and phase_elapsed is not None
                    and phase_elapsed > float(limit)
                )
            )
        return False
    if status == "capacity-oom":
        source = evidence.get("source")
        cgroup_source = bool(
            isinstance(source, str)
            and Path(source).is_absolute()
            and Path(source).is_relative_to(Path("/sys/fs/cgroup"))
            and Path(source).name in {"memory.events", "memory.events.local"}
        )
        counter_proved = bool(
            isinstance(evidence.get("delta"), int)
            and not isinstance(evidence.get("delta"), bool)
            and int(evidence["delta"]) > 0
        )
        evidence_phase = (
            evidence.get("phase") if isinstance(evidence.get("phase"), Mapping) else {}
        )
        if evidence.get("kind") == "kernel-oom-kill":
            return bool(
                controller.get("returncode") == -9
                and controller.get("timed_out") is False
                and counter_proved
                and (cgroup_source or source == "/proc/vmstat:oom_kill")
                and _timed_pipeline_is_running(evidence_phase)
            )
        if evidence.get("kind") == "session-cgroup-ray-process-oom":
            return bool(
                isinstance(controller.get("returncode"), int)
                and controller.get("returncode") not in {0, -9}
                and controller.get("timed_out") is False
                and counter_proved
                and cgroup_source
                and isinstance(evidence.get("ray_process_crash_marker"), str)
                and bool(evidence.get("ray_process_crash_marker"))
                and evidence_phase.get("phase")
                in {"pipeline-finished", "worker-complete"}
                and evidence_phase.get("timed_pipeline_started") is True
                and evidence_phase.get("timed_pipeline_finished") is True
                and evidence_phase.get("pipeline_succeeded") is False
            )
        return False
    return False


def summarize(artifacts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_name = {str(item.get("trial_name")): item for item in artifacts}
    gpu = by_name.get(RUN_SEQUENCE[0].name)
    cpu = by_name.get(RUN_SEQUENCE[1].name)
    reasons: list[str] = []
    if gpu is None or not gpu.get("valid"):
        reasons.append("the one GPU performance observation is missing or rejected")
    if cpu is None:
        reasons.append("the one CPU performance observation is missing")
    elif not cpu.get("valid") and not _accepted_capacity_result(cpu):
        reasons.append("the CPU observation failed for a non-capacity reason")

    gpu_s = None if gpu is None else gpu.get("timings_s", {}).get("e2e")
    cpu_s = None if cpu is None else cpu.get("timings_s", {}).get("e2e")
    ratio = None
    if isinstance(gpu_s, (int, float)) and isinstance(cpu_s, (int, float)) and gpu_s:
        ratio = float(cpu_s) / float(gpu_s)
    return {
        "valid": not reasons,
        "rejection_reasons": reasons,
        "directional": True,
        "observations_per_arm": 1,
        "gpu_e2e_s": gpu_s,
        "gpu_e2e_minutes": None if gpu_s is None else float(gpu_s) / 60,
        "cpu_e2e_s": cpu_s,
        "cpu_e2e_minutes": None if cpu_s is None else float(cpu_s) / 60,
        "cpu_capacity_result": (
            None if cpu is None or cpu.get("valid") else cpu.get("status")
        ),
        "cpu_over_gpu": ratio,
        "gpu_reached_2x": None if ratio is None else ratio >= 2,
        "historical_gpu_local_ipc_s": 2829.415758983,
        "historical_context_warning": (
            "cross-format and cross-implementation context only; not an "
            "apples-to-apples speedup"
        ),
    }


def run_campaign(
    dataset_root: Path,
    campaign_root: Path,
    python: Path,
    *,
    gate_manifest: Path | None = None,
    smoke_only: bool = False,
) -> dict[str, Any]:
    root = _repo_root()
    python = _assert_frozen_python(python)
    campaign_root = _safe_campaign_root(campaign_root, must_not_exist=True)
    check = preflight(
        dataset_root,
        campaign_root=campaign_root,
        require_clean_git=True,
    )
    if not check["valid"]:
        raise RuntimeError(f"preflight rejected campaign: {check['rejection_reasons']}")
    expected_dataset_identity = check["dataset_identity"]
    expected_head = str(check["git"]["head"])
    _assert_frozen_checkout(root, expected_head)
    overlay = _install_frozen_overlay(root)
    _assert_frozen_checkout(root, expected_head)
    prerequisite_gates = None
    if not smoke_only:
        if gate_manifest is None:
            raise RuntimeError(
                "full 1 TB arms require --gate-manifest from the gates command"
            )
        prerequisite_gates = _validate_prerequisite_gates(
            gate_manifest,
            dataset_root=dataset_root,
            expected_head=expected_head,
            overlay=overlay,
            python=python,
        )
    campaign_root.mkdir(parents=True, exist_ok=False)
    write_json(campaign_root / "CAMPAIGN.json", frozen_campaign(dataset_root))
    write_json(campaign_root / "PREFLIGHT.json", check)
    write_json(campaign_root / "RAY_DATA_OVERLAY.json", overlay)
    write_json(
        campaign_root / "EXECUTION.json",
        {
            "mode": "smoke-only" if smoke_only else "full",
            "sequence": [
                asdict(item)
                for item in (
                    SMOKE_SEQUENCE if smoke_only else (*SMOKE_SEQUENCE, *RUN_SEQUENCE)
                )
            ],
        },
    )
    if prerequisite_gates is not None:
        prerequisite_gates = {
            **prerequisite_gates,
            "campaign_evidence": _copy_prerequisite_gate_evidence(
                campaign_root, prerequisite_gates
            ),
        }
        write_json(campaign_root / "PREREQUISITE_GATES.json", prerequisite_gates)

    _, full_slices = exact_plan(dataset_root)
    smoke_plan = smoke_slices(load_manifest(dataset_root))
    artifacts: list[dict[str, Any]] = []
    sequence = SMOKE_SEQUENCE if smoke_only else (*SMOKE_SEQUENCE, *RUN_SEQUENCE)
    try:
        for spec in sequence:
            _assert_frozen_checkout(root, expected_head)
            _assert_frozen_overlay(root, overlay)
            active = _ray_process_inventory()
            if not active.get("valid"):
                raise RuntimeError(
                    f"Ray became active before {spec.name}; refusing interference"
                )
            idle = _wait_for_gpu_idle()
            write_json(campaign_root / "preflight" / f"{spec.name}-gpu-idle.json", idle)
            if not idle.get("valid"):
                raise RuntimeError(f"GPU idle check rejected {spec.name}")

            warm = _warm_source_cache(
                full_slices if spec.workload == "full" else smoke_plan
            )
            write_json(
                campaign_root / "warmups" / f"{spec.name}.json",
                {**warm, "outside_primary_timer": True},
            )
            if spec.workload == "full":
                assert gate_manifest is not None
                prerequisite_gates = _validate_prerequisite_gates(
                    gate_manifest,
                    dataset_root=dataset_root,
                    expected_head=expected_head,
                    overlay=overlay,
                    python=python,
                )
            # Warm-up can take minutes. Recheck both isolation predicates after
            # it; a final launch check follows the source-identity hash below.
            post_warmup_gpu = _wait_for_gpu_idle()
            post_warmup_ray = _ray_process_inventory()
            post_warmup = {
                "ray": post_warmup_ray,
                "gpu": post_warmup_gpu,
                "checked_after_source_warmup": True,
                "before_source_identity_hash": True,
            }
            write_json(
                campaign_root / "preflight" / f"{spec.name}-post-warmup-idle.json",
                post_warmup,
            )
            if not post_warmup_ray.get("valid"):
                raise RuntimeError(
                    f"Ray became active during warm-up before {spec.name}"
                )
            if not post_warmup_gpu.get("valid"):
                raise RuntimeError(
                    f"GPUs became active during warm-up before {spec.name}"
                )
            current_dataset_identity = _dataset_identity(
                dataset_root, verify_parquet_files=True
            )
            write_json(
                campaign_root / "preflight" / f"{spec.name}-dataset-identity.json",
                current_dataset_identity,
            )
            if current_dataset_identity != expected_dataset_identity:
                raise RuntimeError(f"frozen Parquet source changed before {spec.name}")
            launch_capacity = _runtime_capacity_snapshot()
            launch_ray = _ray_process_inventory()
            launch_gpu = _wait_for_gpu_idle()
            launch_readiness = {
                "capacity": launch_capacity,
                "ray": launch_ray,
                "gpu": launch_gpu,
                "checked_after_source_hash": True,
                "immediately_before_worker_launch": True,
            }
            write_json(
                campaign_root / "preflight" / f"{spec.name}-launch-readiness.json",
                launch_readiness,
            )
            if not launch_capacity["valid"]:
                raise RuntimeError(
                    f"runtime capacity check rejected {spec.name}: "
                    f"{launch_capacity['rejection_reasons']}"
                )
            if not launch_ray.get("valid"):
                raise RuntimeError(f"Ray became active before launching {spec.name}")
            if not launch_gpu.get("valid"):
                raise RuntimeError(f"GPUs became active before launching {spec.name}")
            command, artifact_path, runtime = _worker_command(
                python=python,
                dataset_root=dataset_root,
                campaign_root=campaign_root,
                spec=spec,
                expected_head=expected_head,
                expected_dataset_identity=current_dataset_identity,
            )
            artifact = _run_worker(
                root=root,
                campaign_root=campaign_root,
                spec=spec,
                command=command,
                artifact_path=artifact_path,
                runtime=runtime,
                timeout_s=(
                    TRIAL_TIMEOUT_S if spec.workload == "full" else SMOKE_TIMEOUT_S
                ),
                expected_dataset_identity=current_dataset_identity,
            )
            try:
                _assert_frozen_checkout(root, expected_head)
                _assert_frozen_overlay(root, overlay)
            except BaseException as error:
                artifact["valid"] = False
                artifact["status"] = "invalidated-code-identity"
                artifact.setdefault("rejection_reasons", []).append(
                    f"post-observation code identity failed: "
                    f"{type(error).__name__}: {error}"
                )
                write_json(artifact_path, artifact)
            artifacts.append(artifact)
            if spec.workload == "smoke" and not artifact.get("valid"):
                raise RuntimeError(
                    f"correctness smoke {spec.name} failed: "
                    f"{artifact.get('rejection_reasons')}"
                )
            if spec == RUN_SEQUENCE[0] and not artifact.get("valid"):
                raise RuntimeError(
                    f"GPU performance arm failed: {artifact.get('rejection_reasons')}"
                )
            if spec == RUN_SEQUENCE[1] and not (
                artifact.get("valid") or _accepted_capacity_result(artifact)
            ):
                raise RuntimeError(
                    f"CPU arm had a harness/non-capacity failure: "
                    f"{artifact.get('rejection_reasons')}"
                )
        _assert_frozen_checkout(root, expected_head)
        _assert_frozen_overlay(root, overlay)
    finally:
        write_json(campaign_root / "ARTIFACT_INDEX.json", artifacts)

    if smoke_only:
        report = {
            "valid": len(artifacts) == len(SMOKE_SEQUENCE)
            and all(item.get("valid") for item in artifacts),
            "mode": "smoke-only",
            "smokes": [
                {
                    "trial_name": item.get("trial_name"),
                    "backend": item.get("backend"),
                    "valid": item.get("valid"),
                    "e2e_s": item.get("timings_s", {}).get("e2e"),
                }
                for item in artifacts
            ],
        }
    else:
        report = summarize(artifacts)
        report["prerequisite_gate_manifest"] = str(Path(str(gate_manifest)).resolve())
    report.update({"git_head": expected_head, "campaign_root": str(campaign_root)})
    write_json(campaign_root / "RESULTS.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("self-check", "preflight", "gates", "smoke", "run", "report"),
    )
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--campaign-root", type=Path)
    parser.add_argument("--gate-manifest", type=Path)
    parser.add_argument(
        "--python", type=Path, default=_repo_root() / ".venv" / "bin" / "python"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "self-check":
        print(json.dumps(frozen_campaign(args.dataset_root), indent=2, sort_keys=True))
        return 0
    if args.command == "preflight":
        result = preflight(args.dataset_root, campaign_root=args.campaign_root)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["valid"] else 1
    if args.command == "gates":
        if args.campaign_root is None:
            raise SystemExit("--campaign-root is required for gates")
        result = run_prerequisite_gates(
            args.dataset_root, args.campaign_root, args.python
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["valid"] else 1
    if args.command in {"smoke", "run"}:
        if args.campaign_root is None:
            raise SystemExit(f"--campaign-root is required for {args.command}")
        if args.command == "run" and args.gate_manifest is None:
            raise SystemExit("--gate-manifest is required for run")
        result = run_campaign(
            args.dataset_root,
            args.campaign_root,
            args.python,
            gate_manifest=args.gate_manifest,
            smoke_only=args.command == "smoke",
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["valid"] else 1
    if args.campaign_root is None:
        raise SystemExit("--campaign-root is required for report")
    artifacts = json.loads(
        (args.campaign_root / "ARTIFACT_INDEX.json").read_text(encoding="utf-8")
    )
    if not isinstance(artifacts, list):
        raise RuntimeError("ARTIFACT_INDEX.json is not a list")
    result = summarize(artifacts)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
