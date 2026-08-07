"""One fresh-Ray-runtime BTS sort observation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from .backend_stats import (
    get_last_run_stats,
    peak_device_bytes,
    rank_stats,
    required_fields_missing,
    value,
)
from .data import cohort_slices, plan_dict, read_projection, smoke_slices
from .spec import (
    DATASET_ROOT,
    EXPECTED_BLOCKS,
    EXPECTED_FULL_BYTES,
    EXPECTED_ROWS,
    LARGE_COPIES,
    LARGE_ROWS,
    RAY_COMMIT,
    RAY_VERSION,
    RAY_WHEEL_SHA256,
    SMOKE_KEYS,
    cell_by_name,
    load_manifest,
)

GIB = 1 << 30
OBJECT_STORE_BYTES = 512 * GIB
# The restricted-budget study must exercise cluster-realistic Ray spill, not
# quietly retain every external GPU run in RAM. 132 GiB still holds the
# 63.881-GiB materialized input and final output together, but forces
# intermediate immutable runs/replacements onto the configured RAID path.
SPILL_OBJECT_STORE_BYTES = 132 * GIB
LARGE_OBJECT_STORE_BYTES = 256 * GIB
_SPILL_FIELDS = (
    "spilled_bytes_total",
    "spilled_objects_total",
    "restored_bytes_total",
    "restored_objects_total",
    "spill_time_total_s",
    "restore_time_total_s",
)
# Keep enough room for Ray's timestamped session and Unix socket names. Linux
# limits AF_UNIX paths to 107 bytes.
SHM_PARENT = Path("/dev/shm/rgs")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _verify_overlay() -> dict[str, Any]:
    manifest_path = _repo_root() / ".venv/gpu-sort-ray-data-overlay.json"
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    source = Path(manifest["source"]).resolve()
    target = Path(manifest["target"]).resolve()
    if source != (_repo_root() / "python/ray/data").resolve():
        raise RuntimeError("Ray Data overlay manifest names the wrong source")
    if not target.is_relative_to((_repo_root() / ".venv").resolve()):
        raise RuntimeError("Ray Data overlay manifest names a target outside .venv")
    for relative, expected in manifest["files"].items():
        source_digest = hashlib.sha256((source / relative).read_bytes()).hexdigest()
        target_digest = hashlib.sha256((target / relative).read_bytes()).hexdigest()
        if source_digest != expected or target_digest != expected:
            raise RuntimeError(f"Stale Ray Data overlay file: {relative}")
    target_python = {
        str(path.relative_to(target))
        for path in target.rglob("*.py")
        if "__pycache__" not in path.parts
    }
    if target_python != set(manifest["files"]):
        raise RuntimeError("Ray Data overlay contains unexpected Python files")
    wheel = manifest.get("wheel", {})
    wheel_path = Path(wheel.get("path", "")).resolve()
    if (
        not wheel_path.is_file()
        or not wheel_path.is_relative_to((_repo_root() / ".venv").resolve())
        or hashlib.sha256(wheel_path.read_bytes()).hexdigest() != RAY_WHEEL_SHA256
    ):
        raise RuntimeError("Worker does not have the exact benchmark Ray wheel")
    return manifest


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _spill_snapshot(ray: Any) -> dict[str, int | float]:
    """Read cumulative Ray object spill/restore counters cluster-wide."""

    from ray._private.internal_api import get_memory_info_reply, get_state_from_address

    state = get_state_from_address(ray.get_runtime_context().gcs_address)
    stats = get_memory_info_reply(state).store_stats
    return {name: getattr(stats, name) for name in _SPILL_FIELDS}


def _spill_delta(
    before: dict[str, int | float], after: dict[str, int | float]
) -> dict[str, int | float]:
    return {name: after[name] - before[name] for name in _SPILL_FIELDS}


def _stable_spill_snapshot(ray: Any) -> dict[str, int | float]:
    """Wait outside the sort timer for asynchronous spill counters to settle."""

    prior = _spill_snapshot(ray)
    stable = 0
    deadline = time.monotonic() + 5.0
    while stable < 2 and time.monotonic() < deadline:
        time.sleep(0.2)
        current = _spill_snapshot(ray)
        stable = stable + 1 if current == prior else 0
        prior = current
    return prior


def _metadata(dataset: Any) -> dict[str, int]:
    blocks = rows = size = 0
    for bundle in dataset.iter_internal_ref_bundles():
        blocks += len(bundle.block_refs)
        for metadata in bundle.metadata:
            if metadata.num_rows is None or metadata.size_bytes is None:
                raise RuntimeError("Materialized Ray block metadata is incomplete")
            rows += int(metadata.num_rows)
            size += int(metadata.size_bytes)
    return {"blocks": blocks, "rows": rows, "decoded_bytes": size}


def _refs(dataset: Any) -> list[Any]:
    return [
        ref
        for bundle in dataset.iter_internal_ref_bundles()
        for ref in bundle.block_refs
    ]


def _residency(ray: Any, dataset: Any, *, timeout_s: float = 5.0) -> dict[str, Any]:
    refs = _refs(dataset)
    deadline = time.monotonic() + timeout_s
    while True:
        locations = ray.experimental.get_object_locations(refs)
        resident = sum(bool(locations.get(ref, {}).get("node_ids")) for ref in refs)
        if resident == len(refs) or time.monotonic() >= deadline:
            break
        # Object-location publication can lag a just-completed ray.put by a
        # few scheduler heartbeats. This wait is outside every sort timer.
        time.sleep(0.05)
    return {
        "objects": len(refs),
        "resident_objects": resident,
        "all_resident": resident == len(refs),
    }


def _schema(dataset: Any) -> Any:
    schema = dataset.schema()
    return getattr(schema, "base_schema", schema)


class GpuMonitor:
    def __init__(self) -> None:
        self.peaks: list[int] = []
        self.totals: list[int] = []
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "GpuMonitor":
        try:
            import pynvml

            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            handles = [
                pynvml.nvmlDeviceGetHandleByIndex(index) for index in range(count)
            ]
            self.totals = [
                int(pynvml.nvmlDeviceGetMemoryInfo(handle).total) for handle in handles
            ]
            self.peaks = [0] * count

            def sample() -> None:
                try:
                    while not self._stop.wait(0.1):
                        for index, handle in enumerate(handles):
                            used = int(pynvml.nvmlDeviceGetMemoryInfo(handle).used)
                            self.peaks[index] = max(self.peaks[index], used)
                except BaseException as error:  # telemetry must invalidate, not kill, a trial
                    self.error = repr(error)

            self._thread = threading.Thread(target=sample, daemon=True)
            self._thread.start()
            self._pynvml = pynvml
        except BaseException as error:
            self.error = repr(error)
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        module = getattr(self, "_pynvml", None)
        if module is not None:
            module.nvmlShutdown()

    def to_dict(self) -> dict[str, Any]:
        return {
            "peak_used_bytes": self.peaks,
            "total_bytes": self.totals,
            "error": self.error,
        }


def _start_ray(runtime: Path, kind: str, backend: str) -> tuple[Any, Path]:
    overlay = _verify_overlay()
    import ray

    expected = Path(overlay["target"]).parent.resolve()
    actual = Path(ray.__file__).resolve()
    if actual.parent != expected:
        raise RuntimeError(
            f"Worker imported {actual}, not the wheel-backed Ray package at {expected}"
        )
    if ray.__version__ != RAY_VERSION or ray.__commit__ != RAY_COMMIT:
        raise RuntimeError(
            f"Expected Ray {RAY_VERSION} at {RAY_COMMIT}, got "
            f"{ray.__version__} at {ray.__commit__}"
        )

    token = hashlib.sha256(str(runtime).encode()).hexdigest()[:12]
    owned = (SHM_PARENT / token).resolve()
    if owned.parent != SHM_PARENT or len(owned.name) != 12:
        raise RuntimeError(f"Unsafe shared-memory path: {owned}")
    ray_root = owned / "ray"
    plasma_root = owned / "plasma"
    plasma_root.mkdir(parents=True, exist_ok=False)
    spill = runtime / "ray-spill"
    spill.mkdir(parents=True, exist_ok=True)
    if spill.resolve().is_relative_to(Path("/dev/shm").resolve()):
        raise RuntimeError(
            "Ray disk spill must use local disk/RAID, never the RAM-backed /dev/shm"
        )
    options: dict[str, Any] = dict(
        address="local",
        object_spilling_directory=str(spill),
        include_dashboard=False,
        log_to_driver=True,
        _temp_dir=str(ray_root),
        _plasma_directory=str(plasma_root),
    )
    if backend == "gpu":
        options.update(
            num_gpus=16,
            object_store_memory={
                "spill": SPILL_OBJECT_STORE_BYTES,
                "large": LARGE_OBJECT_STORE_BYTES,
            }.get(kind, OBJECT_STORE_BYTES),
            _system_config={
                # The timed boundary is explicitly object-store to object-store,
                # including the smallest narrow-projection blocks.
                "max_direct_call_object_size": 0,
            },
        )
    ray.init(**options)
    return ray, owned


def _materialize(
    ray: Any, plan: dict[str, Any], columns: tuple[str, ...]
) -> tuple[Any, dict[str, Any]]:
    def read_and_seal(item: dict[str, Any], selected: tuple[str, ...]) -> Any:
        # A task's direct return may stay in a worker heap under default Ray
        # settings. The benchmark boundary requires Plasma, so explicitly put
        # the table and return its nested ObjectRef without changing any Ray
        # startup or CPU sort configuration.
        import ray as worker_ray

        return worker_ray.put(read_projection(item, selected))

    reader = ray.remote(num_cpus=1)(read_and_seal)
    started = time.perf_counter()
    nested = [reader.remote(item, columns) for item in plan["slices"]]
    refs = ray.get(nested)
    if not all(isinstance(ref, ray.ObjectRef) for ref in refs):
        raise RuntimeError("Projection readers did not return sealed Plasma ObjectRefs")
    dataset = ray.data.from_arrow_refs(refs).materialize()
    elapsed = time.perf_counter() - started
    stats = _metadata(dataset)
    stats.update(
        {
            "read_materialize_s": elapsed,
            "residency": _residency(ray, dataset),
            "schema": str(_schema(dataset)),
        }
    )
    if stats["rows"] != plan["rows"] or stats["blocks"] != plan["blocks"]:
        raise RuntimeError(f"Materialized input changed from its plan: {stats}")
    if not stats["residency"]["all_resident"]:
        raise RuntimeError("Input is not resident in Plasma before the timed sort")
    return dataset, stats


def _seal_inline_output(ray: Any, dataset: Any) -> Any:
    """Seal any default-Ray inline result blocks before ending the timer."""

    refs = _refs(dataset)
    locations = ray.experimental.get_object_locations(refs)
    missing = [
        index
        for index, ref in enumerate(refs)
        if not locations.get(ref, {}).get("node_ids")
    ]
    if not missing:
        return dataset
    replacements = dict(zip(missing, ray.get([refs[index] for index in missing])))
    sealed = [
        ray.put(replacements[index]) if index in replacements else ref
        for index, ref in enumerate(refs)
    ]
    return ray.data.from_arrow_refs(sealed).materialize()


def _sort(
    ray: Any, dataset: Any, keys: tuple[str, ...], backend: str
) -> tuple[Any, float]:
    kwargs: dict[str, Any] = {"key": list(keys), "descending": [False] * len(keys)}
    if backend == "gpu":
        kwargs["backend"] = "gpu"
    started = time.perf_counter_ns()
    output = dataset.sort(**kwargs).materialize()
    output = _seal_inline_output(ray, output)
    elapsed = (time.perf_counter_ns() - started) / 1_000_000_000
    return output, elapsed


def _exact_snapshot(ray: Any, output: Any, path: Path) -> dict[str, str]:
    import pyarrow as pa

    table = pa.concat_tables(ray.get(_refs(output))).combine_chunks()
    path.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_file(
        sink, table.schema
    ) as writer:
        writer.write_table(table)
    return {
        "snapshot_path": str(path.resolve()),
        "physical_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _validate_origin(ray: Any, output: Any) -> dict[str, Any]:
    """Distributed linear orderedness check plus exact row_id sum; no CPU sort."""

    def check(table: Any) -> dict[str, Any]:
        import pyarrow as pa
        import pyarrow.compute as pc

        if isinstance(table, pa.RecordBatch):
            table = pa.Table.from_batches([table])
        origin = table.column("Origin").combine_chunks()
        clean = pc.fill_null(origin, "\U0010ffff")
        ordered = True
        if len(clean) > 1:
            ordered = bool(
                pc.all(
                    pc.greater_equal(clean.slice(1), clean.slice(0, len(clean) - 1))
                ).as_py()
            )
        return {
            "ordered": ordered,
            "first": clean[0].as_py() if len(clean) else None,
            "last": clean[-1].as_py() if len(clean) else None,
            "rows": len(clean),
            "row_id_sum": int(pc.sum(table.column("row_id")).as_py() or 0),
        }

    checker = ray.remote(num_cpus=1)(check)
    parts = ray.get([checker.remote(ref) for ref in _refs(output)])
    boundaries = all(
        left["last"] is None or right["first"] is None or left["last"] <= right["first"]
        for left, right in zip(parts, parts[1:])
    )
    total_rows = sum(item["rows"] for item in parts)
    total_sum = sum(item["row_id_sum"] for item in parts)
    return {
        "ordered": boundaries and all(item["ordered"] for item in parts),
        "rows": total_rows,
        "row_id_sum": total_sum,
        "expected_row_id_sum": LARGE_ROWS * (LARGE_ROWS - 1) // 2,
        "blocks_checked": len(parts),
    }


def run(args: argparse.Namespace, ray: Any) -> dict[str, Any]:
    root = args.dataset_root.resolve()
    manifest = load_manifest(root)
    cell = cell_by_name(manifest, args.cell)
    if args.kind == "smoke":
        slices = smoke_slices(manifest)
        columns = tuple(manifest["schema_names"]) + ("row_id",)
        keys = SMOKE_KEYS
    else:
        copies = LARGE_COPIES if args.kind == "large" else 1
        slices = cohort_slices(manifest, copies=copies)
        columns = cell.columns
        keys = cell.keys
    plan = plan_dict(slices, kind=args.kind)
    before_input_spill = _spill_snapshot(ray)
    dataset, input_stats = _materialize(ray, plan, columns)
    after_input_spill = _stable_spill_snapshot(ray)
    input_spill = _spill_delta(before_input_spill, after_input_spill)
    input_stats["ray_object_store_io"] = input_spill
    if args.kind != "smoke" and args.kind != "large":
        if (
            input_stats["rows"] != EXPECTED_ROWS
            or input_stats["blocks"] != EXPECTED_BLOCKS
        ):
            raise RuntimeError("Performance trial does not use the exact fixed cohort")
    spill = args.runtime / "ray-spill"
    before_spill = _spill_snapshot(ray)
    monitor = GpuMonitor()
    with monitor:
        output, cold_s = _sort(ray, dataset, keys, args.backend)
    after_spill = _stable_spill_snapshot(ray)
    output_stats = _metadata(output)
    output_stats["residency"] = _residency(ray, output)
    output_stats["schema"] = str(_schema(output))
    backend_stats = get_last_run_stats(output) if args.backend == "gpu" else {}
    ray_object_store_io = _spill_delta(before_spill, after_spill)
    ray_disk_spill = int(ray_object_store_io["spilled_bytes_total"])
    reasons: list[str] = []
    if output_stats["rows"] != input_stats["rows"]:
        reasons.append("output row count differs from input")
    if output_stats["schema"] != input_stats["schema"]:
        reasons.append("output schema differs from input")
    if args.kind != "large" and not output_stats["residency"]["all_resident"]:
        reasons.append("output is not resident in Plasma")
    # Default Ray/PyArrow may use Ray's normal disk-backed object spilling while
    # executing a wide sort.  That is part of the CPU baseline, so record it
    # instead of overriding or rejecting Ray's default behavior.  Resident GPU
    # trend cells must still complete without externalization or disk spill.
    if args.kind == "trend" and args.backend == "gpu" and ray_disk_spill != 0:
        reasons.append(
            f"64 GiB trial wrote {ray_disk_spill} bytes to Ray's disk spill directory"
        )
    if int(input_spill["spilled_bytes_total"]) != 0:
        reasons.append("input materialization spilled before the timed sort")
    if args.kind in ("trend", "spill") and cell.name == "full":
        if input_stats["decoded_bytes"] != EXPECTED_FULL_BYTES:
            reasons.append(
                "full-cell decoded bytes changed: "
                f"{input_stats['decoded_bytes']} != {EXPECTED_FULL_BYTES}"
            )
    if (
        args.kind == "large"
        and input_stats["decoded_bytes"] != LARGE_COPIES * EXPECTED_FULL_BYTES
    ):
        reasons.append(
            "large proof is not exactly three copies of the full decoded cohort"
        )
    if args.backend == "gpu":
        missing = required_fields_missing(backend_stats)
        if missing:
            reasons.append(f"missing GPU telemetry: {missing}")
        if len(rank_stats(backend_stats)) != 16:
            reasons.append(
                f"GPU telemetry has {len(rank_stats(backend_stats))} ranks, expected 16"
            )
        if int(value(backend_stats, "cpu_sort_rows", default=-1)) != 0:
            reasons.append("GPU backend performed CPU sorting")
        if int(value(backend_stats, "cpu_merge_rows", default=-1)) != 0:
            reasons.append("GPU backend performed CPU merging")
        if (
            args.budget_bytes is not None
            and int(value(backend_stats, "memory_budget_bytes", default=0))
            != args.budget_bytes
        ):
            reasons.append("backend did not report the requested memory budget")
    validation: dict[str, Any] = {}
    if args.kind == "smoke":
        validation.update(
            _exact_snapshot(ray, output, args.output.with_suffix(".arrow"))
        )
    elif args.kind == "large":
        validation = _validate_origin(ray, output)
        if not validation["ordered"]:
            reasons.append("large output is not globally ordered")
        if validation["row_id_sum"] != validation["expected_row_id_sum"]:
            reasons.append("large output row_id checksum differs from input")
    return {
        "valid": not reasons,
        "status": "accepted" if not reasons else "rejected",
        "kind": args.kind,
        "backend": args.backend,
        "cell": cell.to_dict(),
        "repetition": args.repetition,
        "budget_bytes": args.budget_bytes,
        "plan": {key: value for key, value in plan.items() if key != "slices"},
        "timing_boundary": "materialized Plasma input through output sealed in Plasma",
        "cold_sort_s": cold_s,
        "input": input_stats,
        "output": output_stats,
        "throughput_rows_s": input_stats["rows"] / cold_s,
        "throughput_gib_s": input_stats["decoded_bytes"] / GIB / cold_s,
        "gpu_stats": backend_stats,
        "gpu_peak_device_bytes": peak_device_bytes(backend_stats),
        "nvml": monitor.to_dict(),
        "ray_disk_spill_bytes": ray_disk_spill,
        "ray_object_store_io": ray_object_store_io,
        "object_store_memory_bytes": int(
            ray.cluster_resources().get("object_store_memory", 0)
        ),
        "ray_startup_mode": "default" if args.backend == "pyarrow" else "gpu_benchmark",
        "ray_disk_spill_directory": str(spill.resolve()),
        "validation": validation,
        "rejection_reasons": reasons,
        "ray_module": str(Path(ray.__file__).resolve()),
        "ray_version": ray.__version__,
        "ray_commit": ray.__commit__,
        "wheel_sha256": RAY_WHEEL_SHA256,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--runtime", type=Path, required=True)
    result.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    result.add_argument(
        "--kind", choices=("trend", "spill", "smoke", "large"), required=True
    )
    result.add_argument("--backend", choices=("pyarrow", "gpu"), required=True)
    result.add_argument("--cell", default="full")
    result.add_argument("--repetition", type=int, default=1)
    result.add_argument("--budget-bytes", type=int)
    return result


def main() -> int:
    args = parser().parse_args()
    args.runtime = args.runtime.resolve()
    args.runtime.mkdir(parents=True, exist_ok=False)
    if args.budget_bytes is None:
        os.environ.pop("RAY_DATA_GPU_SORT_MEMORY_BUDGET_BYTES", None)
    else:
        os.environ["RAY_DATA_GPU_SORT_MEMORY_BUDGET_BYTES"] = str(args.budget_bytes)
    ray = None
    owned = None
    started = time.perf_counter()
    try:
        ray, owned = _start_ray(args.runtime, args.kind, args.backend)
        startup_s = time.perf_counter() - started
        result = run(args, ray)
        result["ray_startup_s"] = startup_s
    except BaseException as error:
        result = {
            "valid": False,
            "status": "rejected",
            "kind": args.kind,
            "backend": args.backend,
            "cell": args.cell,
            "repetition": args.repetition,
            "budget_bytes": args.budget_bytes,
            "rejection_reasons": [f"{type(error).__name__}: {error}"],
            "exception": traceback.format_exc(),
        }
    finally:
        if ray is not None:
            try:
                ray.shutdown()
            except BaseException:
                pass
        if owned is not None:
            resolved = owned.resolve()
            if resolved.parent == SHM_PARENT and len(resolved.name) == 12:
                shutil.rmtree(resolved, ignore_errors=True)
        write_json(args.output, result)
    return 0 if result.get("valid") else 1


if __name__ == "__main__":
    raise SystemExit(main())
