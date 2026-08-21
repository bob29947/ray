"""Shared frozen-workload and host helpers for the streaming Parquet study.

This module intentionally has no import-time dependency on Ray.  It contains
only the constants and controller-side helpers shared by the Parquet source,
controller, worker, and their focused tests.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

from .data import plan_dict, scaled_cohort_slices
from .spec import EXPECTED_ROWS, load_manifest


PLAN_KIND = "runstore-1tb"
TARGET_ROWS = 1_177_097_812
TARGET_BLOCKS = 9_151
TARGET_DECODED_BYTES = 999_999_999_762
TARGET_PLAN_DIGEST = "e80163bbc681bc8b04b34d273a4bcdb58b52ef4cbe794b92f07f01bb4d00226c"
TARGET_ROW_ID_SUM = 692_779_628_919_044_766
ORIGIN_AIRPORT_ID_KEY = "OriginAirportID"
SORT_KEY_STATS = {
    ORIGIN_AIRPORT_ID_KEY: {
        "arrow_type": "int64",
        "null_rows": 0,
        "cardinality": 401,
        "min": 10_135,
        "max": 16_869,
        "frequency_digest": (
            "b42271899ab4eb56c768e8d6843ffb893f0f5f4978861933b82ac8db8ee50132"
        ),
        "smoke": {
            "null_rows": 0,
            "cardinality": 285,
            "frequency_digest": (
                "a5f2889e1c5992c82de4bd8651653b2331c0bbcb9a7df446cbd9e839ccbb7ccc"
            ),
        },
    },
}
SUPPORTED_SORT_KEYS = tuple(SORT_KEY_STATS)
OBJECT_STORE_BYTES = 200_000_000_000
MIN_CAMPAIGN_FREE_BYTES = 6 << 40
LOCAL_RUN_MIN_FREE_BYTES = 4 << 40
MIN_HOST_AVAILABLE_BYTES = int(1.2 * (1 << 40))
EXPECTED_CPUS = 96
EXPECTED_GPUS = 16
GPU_IDLE_SETTLE_TIMEOUT_S = 60.0
GPU_IDLE_STABLE_SAMPLES = 2
GPU_IDLE_POLL_S = 1.0
GPU_COMMUNICATION_ENVIRONMENT = {
    "CUDF_SPILL": "0",
    "RAPIDSMPF_LOG": "WARN",
    "RAPIDSMPF_UCXX_PROGRESS_MODE": "polling",
    "UCX_LOG_LEVEL": "warn",
    "UCX_MEMTYPE_CACHE": "n",
    "UCX_SOCKADDR_TLS_PRIORITY": "tcp",
    "UCX_TLS": "cuda_copy,cuda_ipc,sm,tcp",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def exact_plan(dataset_root: Path) -> tuple[dict[str, Any], tuple[Any, ...]]:
    """Regenerate and prove the frozen 1 TB row-group plan."""

    manifest = load_manifest(dataset_root)
    native_columns = tuple(manifest["schema_names"])
    if len(native_columns) != 109 or not set(SUPPORTED_SORT_KEYS).issubset(
        native_columns
    ):
        raise RuntimeError("The frozen full-payload integer-key schema changed")
    slices = scaled_cohort_slices(manifest, TARGET_ROWS, EXPECTED_ROWS)
    plan = plan_dict(slices, kind=PLAN_KIND)
    if (
        plan["rows"] != TARGET_ROWS
        or plan["blocks"] != TARGET_BLOCKS
        or plan["digest"] != TARGET_PLAN_DIGEST
    ):
        raise RuntimeError(
            "The frozen 1 TB plan changed: "
            f"rows={plan['rows']}, blocks={plan['blocks']}, digest={plan['digest']}"
        )
    last = slices[-1]
    if (
        last.copy,
        last.year,
        last.month,
        last.row_group,
        last.rows,
        last.row_id_start,
    ) != (14, 2020, 12, 0, 40_855, 1_177_056_957):
        raise RuntimeError(f"The frozen terminal slice changed: {last}")
    return plan, slices


def _meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    with Path("/proc/meminfo").open(encoding="utf-8") as stream:
        for line in stream:
            name, amount, *units = line.replace(":", "").split()
            multiplier = 1024 if units == ["kB"] else 1
            values[name] = int(amount) * multiplier
    return values


def _nvidia_inventory() -> dict[str, Any]:
    command = (
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    )
    try:
        completed = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "valid": False,
            "error": repr(error),
            "command": list(command),
            "gpus": [],
            "rejection_reasons": [repr(error)],
        }
    rows = []
    for line in completed.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 5:
            continue
        try:
            rows.append(
                {
                    "index": int(fields[0]),
                    "name": fields[1],
                    "memory_total_mib": int(fields[2]),
                    "memory_used_mib": int(fields[3]),
                    "utilization_percent": int(fields[4]),
                }
            )
        except ValueError:
            continue
    reasons = []
    if completed.returncode:
        reasons.append(completed.stderr.strip() or f"exit code {completed.returncode}")
    if len(rows) != EXPECTED_GPUS:
        reasons.append(f"found {len(rows)} GPUs, expected {EXPECTED_GPUS}")
    if rows and any("V100" not in row["name"] for row in rows):
        reasons.append("one or more GPUs are not V100s")
    if rows and any(row["utilization_percent"] > 5 for row in rows):
        reasons.append("one or more GPUs exceed 5% utilization")
    if rows and any(row["memory_used_mib"] > 512 for row in rows):
        reasons.append("one or more GPUs use over 512 MiB before the campaign")
    return {
        "valid": not reasons,
        "command": list(command),
        "returncode": completed.returncode,
        "gpus": rows,
        "rejection_reasons": reasons,
    }


def _wait_for_gpu_idle(
    *,
    timeout_s: float = GPU_IDLE_SETTLE_TIMEOUT_S,
    stable_samples: int = GPU_IDLE_STABLE_SAMPLES,
    poll_s: float = GPU_IDLE_POLL_S,
) -> dict[str, Any]:
    """Require a bounded run of stable idle samples before an observation.

    ``nvidia-smi`` utilization is a rolling sample and can remain at 100% for
    one query after all GPU processes and allocations have exited.  Waiting
    here is outside the primary timer and starts no benchmark work.
    """
    if timeout_s < 0:
        raise ValueError("timeout_s must be nonnegative")
    if stable_samples <= 0:
        raise ValueError("stable_samples must be positive")
    if poll_s < 0:
        raise ValueError("poll_s must be nonnegative")

    started = time.perf_counter()
    deadline = started + timeout_s
    streak = 0
    samples: list[dict[str, Any]] = []
    while True:
        inventory = _nvidia_inventory()
        sampled = time.perf_counter()
        samples.append(
            {
                "elapsed_s": sampled - started,
                "inventory": inventory,
            }
        )
        streak = streak + 1 if inventory["valid"] else 0
        if streak >= stable_samples:
            return {
                "valid": True,
                "elapsed_s": sampled - started,
                "timeout_s": timeout_s,
                "required_stable_samples": stable_samples,
                "final_stable_samples": streak,
                "poll_s": poll_s,
                "samples": samples,
                "rejection_reasons": [],
            }
        if sampled >= deadline:
            reasons = [
                f"GPUs did not remain idle for {stable_samples} consecutive "
                f"samples within {timeout_s:.1f}s"
            ]
            reasons.extend(inventory.get("rejection_reasons", []))
            return {
                "valid": False,
                "elapsed_s": sampled - started,
                "timeout_s": timeout_s,
                "required_stable_samples": stable_samples,
                "final_stable_samples": streak,
                "poll_s": poll_s,
                "samples": samples,
                "rejection_reasons": reasons,
            }
        time.sleep(min(poll_s, max(0.0, deadline - sampled)))


def _ray_process_inventory() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ("ps", "-eo", "pid=,comm=,args="),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        return {"valid": False, "processes": [], "error": repr(error)}
    names = ("raylet", "gcs_server", "plasma_store")
    processes = [
        line.strip()
        for line in completed.stdout.splitlines()
        if any(name in line for name in names)
    ]
    return {"valid": not processes, "processes": processes}


def _warm_source_cache(slices: Iterable[Any]) -> dict[str, Any]:
    paths = sorted({Path(item.path) for item in slices})
    started = time.perf_counter()
    total = 0
    payload = bytearray(8 << 20)
    view = memoryview(payload)
    for path in paths:
        with path.open("rb", buffering=0) as stream:
            while True:
                size = stream.readinto(view)
                if not size:
                    break
                total += size
    return {
        "unique_files": len(paths),
        "physical_bytes_read": total,
        "elapsed_s": time.perf_counter() - started,
    }


def _process_group_alive(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_process_group(
    process: subprocess.Popen[Any], *, grace_s: float = 30.0
) -> bool:
    """Terminate every child in a worker's dedicated session/process group."""

    process_group = process.pid
    if _process_group_alive(process_group):
        try:
            os.killpg(process_group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace_s
    while _process_group_alive(process_group) and time.monotonic() < deadline:
        process.poll()
        time.sleep(0.05)
    if _process_group_alive(process_group):
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + grace_s
        while _process_group_alive(process_group) and time.monotonic() < deadline:
            process.poll()
            time.sleep(0.05)
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass
    return not _process_group_alive(process_group)
