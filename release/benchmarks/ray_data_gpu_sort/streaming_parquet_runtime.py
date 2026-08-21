"""Worker-runtime helpers for the streaming Parquet benchmark."""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .backend_stats import rank_stats, required_fields_missing, value
from .spec import RAY_COMMIT, RAY_VERSION
from .streaming_parquet_common import (
    EXPECTED_CPUS,
    EXPECTED_GPUS,
    GPU_COMMUNICATION_ENVIRONMENT,
    OBJECT_STORE_BYTES,
)
from .worker import SHM_PARENT, _verify_overlay


GPU_STAT_ZERO_FIELDS = (
    ("cpu_sort_rows", "CPU sorting"),
    ("cpu_merge_rows", "CPU merging"),
    ("fallback_count", "output-conversion fallback"),
    ("mpf_host_spill_bytes", "MPF host spill"),
)


def _read_meminfo() -> dict[str, int]:
    result: dict[str, int] = {}
    with Path("/proc/meminfo").open(encoding="utf-8") as stream:
        for line in stream:
            name, amount, *units = line.replace(":", "").split()
            result[name] = int(amount) * (1024 if units == ["kB"] else 1)
    return result


def _cpu_times() -> tuple[int, ...]:
    fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()
    if fields[0] != "cpu":
        raise RuntimeError("/proc/stat has no aggregate CPU row")
    return tuple(int(item) for item in fields[1:])


def _raid_leaf_devices() -> tuple[str, ...]:
    stat = os.stat("/raid")
    sys_device = Path(f"/sys/dev/block/{os.major(stat.st_dev)}:{os.minor(stat.st_dev)}")
    block = sys_device.resolve().name
    slaves = Path("/sys/class/block") / block / "slaves"
    leaves = (
        tuple(sorted(path.name for path in slaves.iterdir())) if slaves.is_dir() else ()
    )
    return leaves or (block,)


def _disk_stats(devices: Sequence[str]) -> dict[str, dict[str, int]]:
    wanted = set(devices)
    result: dict[str, dict[str, int]] = {}
    for line in Path("/proc/diskstats").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 14 or fields[2] not in wanted:
            continue
        # Linux reports sectors in 512-byte units, independent of logical block size.
        result[fields[2]] = {
            "reads_completed": int(fields[3]),
            "read_bytes": int(fields[5]) * 512,
            "read_time_ms": int(fields[6]),
            "writes_completed": int(fields[7]),
            "write_bytes": int(fields[9]) * 512,
            "write_time_ms": int(fields[10]),
            "io_time_ms": int(fields[12]),
            "weighted_io_time_ms": int(fields[13]),
        }
    missing = wanted - result.keys()
    if missing:
        raise RuntimeError(f"Missing /proc/diskstats rows for {sorted(missing)}")
    return result


class ResourceMonitor:
    """Bounded process/RAID/GPU telemetry for one timed end-to-end interval."""

    def __init__(self, sample_s: float = 1.0) -> None:
        self.sample_s = sample_s
        self.devices = _raid_leaf_devices()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: str | None = None
        self._samples = 0
        self._first_cpu: tuple[int, ...] | None = None
        self._last_cpu: tuple[int, ...] | None = None
        self._first_disk: dict[str, dict[str, int]] | None = None
        self._last_disk: dict[str, dict[str, int]] | None = None
        self._min_available = 1 << 63
        self._first_cached: int | None = None
        self._last_cached: int | None = None
        self._min_cached = 1 << 63
        self._max_cached = 0
        self._max_dirty = 0
        self._max_writeback = 0
        self._gpu_peak_memory: list[int] = []
        self._gpu_peak_utilization: list[int] = []
        self._gpu_total_memory: list[int] = []
        self._nvml: Any | None = None
        self._gpu_handles: list[Any] = []

    def _initialize_gpu(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._gpu_handles = [
                pynvml.nvmlDeviceGetHandleByIndex(index)
                for index in range(pynvml.nvmlDeviceGetCount())
            ]
            self._gpu_total_memory = [
                int(pynvml.nvmlDeviceGetMemoryInfo(handle).total)
                for handle in self._gpu_handles
            ]
            self._gpu_peak_memory = [0] * len(self._gpu_handles)
            self._gpu_peak_utilization = [0] * len(self._gpu_handles)
        except BaseException as error:
            self._error = f"NVML: {type(error).__name__}: {error}"

    def _sample(self) -> None:
        memory = _read_meminfo()
        cpu = _cpu_times()
        disk = _disk_stats(self.devices)
        if self._first_cpu is None:
            self._first_cpu = cpu
            self._first_disk = disk
        self._last_cpu = cpu
        self._last_disk = disk
        self._min_available = min(self._min_available, memory.get("MemAvailable", 0))
        cached = memory.get("Cached", 0)
        if self._first_cached is None:
            self._first_cached = cached
        self._last_cached = cached
        self._min_cached = min(self._min_cached, cached)
        self._max_cached = max(self._max_cached, cached)
        self._max_dirty = max(self._max_dirty, memory.get("Dirty", 0))
        self._max_writeback = max(self._max_writeback, memory.get("Writeback", 0))
        if self._nvml is not None:
            for index, handle in enumerate(self._gpu_handles):
                used = int(self._nvml.nvmlDeviceGetMemoryInfo(handle).used)
                utilization = int(self._nvml.nvmlDeviceGetUtilizationRates(handle).gpu)
                self._gpu_peak_memory[index] = max(self._gpu_peak_memory[index], used)
                self._gpu_peak_utilization[index] = max(
                    self._gpu_peak_utilization[index], utilization
                )
        self._samples += 1

    def __enter__(self) -> "ResourceMonitor":
        self._initialize_gpu()
        self._sample()

        def loop() -> None:
            try:
                while not self._stop.wait(self.sample_s):
                    self._sample()
            except BaseException as error:
                self._error = f"monitor: {type(error).__name__}: {error}"

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
            self._error = (
                self._error or f"final sample: {type(error).__name__}: {error}"
            )
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except BaseException:
                pass

    def to_dict(self) -> dict[str, Any]:
        first_disk = self._first_disk or {}
        last_disk = self._last_disk or {}
        per_device = {}
        for device in self.devices:
            first, last = first_disk.get(device, {}), last_disk.get(device, {})
            per_device[device] = {
                name: int(last.get(name, 0)) - int(first.get(name, 0))
                for name in (
                    "reads_completed",
                    "read_bytes",
                    "read_time_ms",
                    "writes_completed",
                    "write_bytes",
                    "write_time_ms",
                    "io_time_ms",
                    "weighted_io_time_ms",
                )
            }
        totals = {
            name: sum(item[name] for item in per_device.values())
            for name in (
                "reads_completed",
                "read_bytes",
                "read_time_ms",
                "writes_completed",
                "write_bytes",
                "write_time_ms",
                "io_time_ms",
                "weighted_io_time_ms",
            )
        }
        cpu_percent = None
        iowait_percent = None
        if self._first_cpu is not None and self._last_cpu is not None:
            delta = [end - start for start, end in zip(self._first_cpu, self._last_cpu)]
            total = sum(delta)
            if total:
                idle = delta[3] + (delta[4] if len(delta) > 4 else 0)
                cpu_percent = 100 * (total - idle) / total
                iowait_percent = 100 * (delta[4] if len(delta) > 4 else 0) / total
        return {
            "sample_interval_s": self.sample_s,
            "samples": self._samples,
            "error": self._error,
            "host": {
                "min_mem_available_bytes": self._min_available,
                "page_cache_start_bytes": self._first_cached,
                "page_cache_end_bytes": self._last_cached,
                "page_cache_min_bytes": self._min_cached,
                "page_cache_max_bytes": self._max_cached,
                "max_dirty_bytes": self._max_dirty,
                "max_writeback_bytes": self._max_writeback,
                "mean_cpu_busy_percent": cpu_percent,
                "mean_iowait_percent": iowait_percent,
            },
            "raid": {
                "leaf_devices": list(self.devices),
                "per_device": per_device,
                "totals": totals,
                "scope": "aggregate physical traffic on all /raid RAID leaf devices",
            },
            "gpu": {
                "peak_memory_used_bytes": self._gpu_peak_memory,
                "total_memory_bytes": self._gpu_total_memory,
                "peak_utilization_percent": self._gpu_peak_utilization,
            },
        }


def _sync_filesystem(path: Path) -> dict[str, Any]:
    """Synchronously drain dirty data for the filesystem containing ``path``."""

    before = _read_meminfo()
    started = time.perf_counter()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        syncfs = getattr(libc, "syncfs", None)
        if syncfs is None:
            raise RuntimeError(
                "libc does not provide syncfs; refusing an unscoped drain"
            )
        syncfs.argtypes = [ctypes.c_int]
        syncfs.restype = ctypes.c_int
        if syncfs(descriptor) != 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code), str(path))
    finally:
        os.close(descriptor)
    elapsed = time.perf_counter() - started
    after = _read_meminfo()
    return {
        "elapsed_s": elapsed,
        "method": "libc.syncfs on the /raid output directory file descriptor",
        "dirty_bytes_before": before.get("Dirty", 0),
        "writeback_bytes_before": before.get("Writeback", 0),
        "dirty_bytes_after": after.get("Dirty", 0),
        "writeback_bytes_after": after.get("Writeback", 0),
    }


def _start_ray(runtime: Path, backend: str) -> tuple[Any, Path]:
    overlay = _verify_overlay()
    import ray

    actual = Path(ray.__file__).resolve()
    expected = Path(overlay["target"]).parent.resolve()
    if actual.parent != expected:
        raise RuntimeError(
            f"Imported {actual}, expected wheel-backed Ray at {expected}"
        )
    if ray.__version__ != RAY_VERSION or ray.__commit__ != RAY_COMMIT:
        raise RuntimeError(
            f"Expected Ray {RAY_VERSION}/{RAY_COMMIT}, got "
            f"{ray.__version__}/{ray.__commit__}"
        )
    token = hashlib.sha256(str(runtime).encode()).hexdigest()[:12]
    owned = (SHM_PARENT / token).resolve()
    if owned.parent != SHM_PARENT or len(owned.name) != 12:
        raise RuntimeError(f"Unsafe shared-memory Ray root: {owned}")
    ray_root = owned / "ray"
    plasma_root = owned / "plasma"
    try:
        plasma_root.mkdir(parents=True, exist_ok=False)
        spill = (runtime / "ray-spill").resolve()
        if not spill.is_relative_to(Path("/raid").resolve()):
            raise RuntimeError("Ray spill directory must be under /raid")
        options: dict[str, Any] = {
            "address": "local",
            "num_cpus": EXPECTED_CPUS,
            "num_gpus": EXPECTED_GPUS if backend == "gpu" else 0,
            "object_store_memory": OBJECT_STORE_BYTES,
            "object_spilling_directory": str(spill),
            "include_dashboard": False,
            "log_to_driver": True,
            "_temp_dir": str(ray_root),
            "_plasma_directory": str(plasma_root),
            "_system_config": {"max_direct_call_object_size": 0},
        }
        ray.init(**options)
        resources = ray.cluster_resources()
        if int(resources.get("CPU", 0)) != EXPECTED_CPUS:
            raise RuntimeError(f"Ray advertises {resources.get('CPU')} CPUs")
        expected_gpus = EXPECTED_GPUS if backend == "gpu" else 0
        if int(resources.get("GPU", 0)) != expected_gpus:
            raise RuntimeError(f"Ray advertises {resources.get('GPU', 0)} GPUs")
        if int(resources.get("object_store_memory", 0)) != OBJECT_STORE_BYTES:
            raise RuntimeError(
                f"Ray object store is {resources.get('object_store_memory')} bytes"
            )
        return ray, owned
    except BaseException:
        # Startup is transactional: callers cannot receive ``owned`` if any
        # post-init resource assertion fails, so rollback must happen here.
        try:
            ray.shutdown()
        except BaseException:
            pass
        shutil.rmtree(owned, ignore_errors=True)
        raise


def _metric(stats: Mapping[str, Any], names: Sequence[str]) -> tuple[Any, str | None]:
    for name in names:
        if name in stats:
            return stats[name], name
    return None, None


def _gpu_reasons(stats: dict[str, Any], run_store: str) -> list[str]:
    reasons: list[str] = []
    missing = required_fields_missing(stats)
    if missing:
        reasons.append(f"missing GPU telemetry: {missing}")
    ranks = rank_stats(stats)
    if len(ranks) != EXPECTED_GPUS:
        reasons.append(
            f"GPU telemetry has {len(ranks)} ranks, expected {EXPECTED_GPUS}"
        )
    for rank, details in enumerate(ranks):
        observed_environment = details.get("communication_environment")
        if observed_environment != GPU_COMMUNICATION_ENVIRONMENT:
            reasons.append(
                f"GPU rank {rank} communication environment is "
                f"{observed_environment!r}, expected "
                f"{GPU_COMMUNICATION_ENVIRONMENT!r}"
            )
    for field, label in GPU_STAT_ZERO_FIELDS:
        if int(value(stats, field, default=-1)) != 0:
            reasons.append(f"GPU backend used {label}")
    observed_store = value(stats, "run_store", "external_run_store", default=None)
    if observed_store != run_store:
        reasons.append(
            f"run_store telemetry is {observed_store!r}, expected {run_store!r}"
        )
    for name in (
        "plasma_intermediate_write_bytes",
        "plasma_intermediate_read_bytes",
        "plasma_intermediate_write_calls",
        "plasma_intermediate_read_calls",
        "plasma_intermediate_write_s",
        "plasma_intermediate_read_s",
        "plasma_output_write_bytes",
        "plasma_output_write_calls",
        "plasma_output_write_s",
    ):
        if name not in stats:
            reasons.append(f"missing GPU I/O telemetry: {name}")
    if run_store == "local_disk":
        required_local = (
            "local_run_write_bytes",
            "local_run_read_bytes",
            "local_run_physical_write_bytes",
            "local_run_physical_read_bytes",
            "local_run_write_calls",
            "local_run_read_calls",
            "local_run_write_s",
            "local_run_read_s",
            "local_run_restore_s",
            "local_run_live_bytes",
            "local_run_peak_bytes",
            "local_run_live_files",
            "local_run_peak_files",
            "local_run_write_errors",
            "local_run_read_errors",
            "local_run_cleanup_errors",
            "local_run_cleanup_pending_files",
            "plasma_intermediate_write_bytes",
            "plasma_intermediate_read_bytes",
            "plasma_output_write_bytes",
            "plasma_output_write_calls",
            "plasma_output_write_s",
        )
        absent = [name for name in required_local if name not in stats]
        if absent:
            reasons.append(f"missing local-run telemetry: {absent}")
        writes = int(value(stats, "local_run_write_bytes", default=0) or 0)
        reads = int(value(stats, "local_run_read_bytes", default=0) or 0)
        if writes <= 0 or reads <= 0:
            reasons.append("local run store did not report positive write/read bytes")
        for aliases, label in (
            (
                ("intermediate_plasma_write_bytes", "plasma_intermediate_write_bytes"),
                "intermediate Plasma writes",
            ),
            (
                ("intermediate_plasma_read_bytes", "plasma_intermediate_read_bytes"),
                "intermediate Plasma reads",
            ),
        ):
            amount, field = _metric(stats, aliases)
            if field is None:
                reasons.append(f"missing telemetry for {label}")
            elif int(amount or 0) != 0:
                reasons.append(f"local run store reported {amount} bytes of {label}")
        if int(value(stats, "local_run_live_files", default=-1)) != 0:
            reasons.append("local run telemetry reports live files after sort")
        if int(value(stats, "local_run_live_bytes", default=-1)) != 0:
            reasons.append("local run telemetry reports live bytes after sort")
        for name in (
            "local_run_write_errors",
            "local_run_read_errors",
            "local_run_cleanup_errors",
            "local_run_cleanup_pending_files",
        ):
            if int(value(stats, name, default=-1)) != 0:
                reasons.append(f"{name}={value(stats, name, default=None)!r}")
        if int(value(stats, "plasma_write_bytes", default=-1)) != int(
            value(stats, "plasma_output_write_bytes", default=-2)
        ):
            reasons.append(
                "local mode Plasma total differs from final-output Plasma writes"
            )
    else:
        if int(value(stats, "local_run_write_bytes", default=0) or 0) != 0:
            reasons.append("Plasma mode unexpectedly reported local-run writes")
        for name in (
            "plasma_intermediate_write_bytes",
            "plasma_intermediate_read_bytes",
            "plasma_output_write_bytes",
        ):
            if name not in stats:
                reasons.append(f"missing Plasma telemetry: {name}")
        if int(value(stats, "externalized_bytes", default=0)) > 0 and (
            int(value(stats, "plasma_intermediate_write_bytes", default=0)) <= 0
            or int(value(stats, "plasma_intermediate_read_bytes", default=0)) <= 0
        ):
            reasons.append("external Plasma mode lacks positive intermediate I/O")
    return reasons


def _remaining_files(path: Path) -> list[str]:
    return sorted(
        str(item.relative_to(path)) for item in path.rglob("*") if item.is_file()
    )
