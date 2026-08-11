"""Execute one BTS observation against an already-running 16-node cluster."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

from release.benchmarks.ray_data_gpu_sort.backend_stats import (
    get_last_run_stats,
    peak_device_bytes,
    rank_stats,
    required_fields_missing,
    value,
)
from release.benchmarks.ray_data_gpu_sort.data import (
    cohort_slices,
    plan_dict,
    read_projection,
    scaled_cohort_slices,
    smoke_slices,
)
from release.benchmarks.ray_data_gpu_sort.spec import (
    EXPECTED_BLOCKS,
    EXPECTED_FULL_BYTES,
    EXPECTED_ROWS,
    RAY_COMMIT,
    RAY_VERSION,
    RAY_WHEEL_SHA256,
    SMOKE_KEYS,
    cell_by_name,
    load_manifest,
)

from .common import artifact_sentinel, atomic_json, digest, file_sha256, read_json

GIB = 1 << 30
LOCATION_LOOKUP_TIMEOUT_MS = 10_000
SPILL_FIELDS = (
    "spilled_bytes_total",
    "spilled_objects_total",
    "restored_bytes_total",
    "restored_objects_total",
    "spill_time_total_s",
    "restore_time_total_s",
)
SAMPLING_MODE = "cpu_sampled_arrow"
SAMPLING_SCHEME = "deterministic_stratified_random"
SAMPLING_SCHEME_VERSION = 1
MIN_SAMPLE_ROWS = 65_536
MAX_PLANNING_H2D_BYTES = 1 << 20
SAMPLING_SUBPHASES = (
    "cpu_sample_construction",
    "boundary_sort",
    "orchestration_remainder",
)
SHA256 = re.compile(r"[0-9a-f]{64}")


def _refs(dataset: Any) -> list[Any]:
    return [
        ref
        for bundle in dataset.iter_internal_ref_bundles()
        for ref in bundle.block_refs
    ]


def _metadata(dataset: Any) -> dict[str, int]:
    blocks = rows = decoded = 0
    for bundle in dataset.iter_internal_ref_bundles():
        blocks += len(bundle.block_refs)
        for item in bundle.metadata:
            if item.num_rows is None or item.size_bytes is None:
                raise RuntimeError("Ray block metadata is incomplete")
            rows += int(item.num_rows)
            decoded += int(item.size_bytes)
    return {"blocks": blocks, "rows": rows, "decoded_bytes": decoded}


def _schema(dataset: Any) -> Any:
    schema = dataset.schema(fetch_if_missing=True)
    return getattr(schema, "base_schema", schema)


def _alive_nodes(ray: Any) -> list[dict[str, Any]]:
    return sorted(
        (dict(node) for node in ray.nodes() if node.get("Alive")),
        key=lambda item: str(item.get("NodeID", "")),
    )


def _affinity(node_id: str) -> Any:
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    return NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)


def _spill_snapshot(ray: Any) -> dict[str, Any]:
    from ray._private.internal_api import get_state_from_address, node_stats

    state = get_state_from_address(ray.get_runtime_context().gcs_address)
    alive = sorted(
        (node for node in state.node_table() if node["Alive"]),
        key=lambda item: str(item.get("NodeID", "")),
    )
    per_node = []
    totals: dict[str, int | float] = {name: 0 for name in SPILL_FIELDS}
    for node in alive:
        stats = node_stats(
            node_manager_address=node["NodeManagerAddress"],
            node_manager_port=node["NodeManagerPort"],
            include_memory_info=False,
        ).store_stats
        fields = {name: getattr(stats, name) for name in SPILL_FIELDS}
        for name, amount in fields.items():
            totals[name] += amount
        per_node.append(
            {
                "node_id": str(node.get("NodeID", "")),
                "node_ip": str(node.get("NodeManagerAddress", "")),
                **fields,
            }
        )
    return {"nodes": per_node, "totals": totals}


def _spill_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    before_nodes = {item["node_id"]: item for item in before["nodes"]}
    per_node = []
    for current in after["nodes"]:
        prior = before_nodes.get(current["node_id"], {})
        per_node.append(
            {
                "node_id": current["node_id"],
                "node_ip": current["node_ip"],
                **{name: current[name] - prior.get(name, 0) for name in SPILL_FIELDS},
            }
        )
    return {
        "nodes": per_node,
        "totals": {
            name: after["totals"][name] - before["totals"][name]
            for name in SPILL_FIELDS
        },
    }


def _stable_spill(ray: Any) -> dict[str, Any]:
    prior = _spill_snapshot(ray)
    stable = 0
    deadline = time.monotonic() + 8
    while stable < 2 and time.monotonic() < deadline:
        time.sleep(0.25)
        current = _spill_snapshot(ray)
        stable = stable + 1 if current["totals"] == prior["totals"] else 0
        prior = current
    return prior


def _locations(ray: Any, dataset: Any) -> dict[str, Any]:
    refs = _refs(dataset)
    nodes: dict[str, int] = {}
    located = 0
    batches = 0
    timeout_batches = 0
    error_batches = 0
    timed_out_objects = 0
    error_objects = 0
    failures = []
    for start in range(0, len(refs), 128):
        batch = refs[start : start + 128]
        batches += 1
        try:
            records = ray.experimental.get_object_locations(
                batch, timeout_ms=LOCATION_LOOKUP_TIMEOUT_MS
            )
        except TimeoutError as error:
            timeout_batches += 1
            timed_out_objects += len(batch)
            failures.append(
                {
                    "start": start,
                    "objects": len(batch),
                    "kind": "timeout",
                    "type": type(error).__name__,
                    "message": str(error)[:240],
                }
            )
            continue
        except Exception as error:
            error_batches += 1
            error_objects += len(batch)
            failures.append(
                {
                    "start": start,
                    "objects": len(batch),
                    "kind": "error",
                    "type": type(error).__name__,
                    "message": str(error)[:240],
                }
            )
            continue
        for ref in batch:
            ids = list(records.get(ref, {}).get("node_ids", ()))
            located += int(bool(ids))
            for node_id in ids:
                nodes[str(node_id)] = nodes.get(str(node_id), 0) + 1
    return {
        "objects": len(refs),
        "located_objects": located,
        "all_locatable": located == len(refs),
        "objects_by_node": nodes,
        "lookup_timeout_ms": LOCATION_LOOKUP_TIMEOUT_MS,
        "lookup_batches": batches,
        "lookup_timeout_batches": timeout_batches,
        "lookup_error_batches": error_batches,
        "lookup_timed_out_objects": timed_out_objects,
        "lookup_error_objects": error_objects,
        "lookup_failures": failures,
    }


def _read_and_seal(item: dict[str, Any], columns: tuple[str, ...]) -> Any:
    # A normal task return is sealed in the worker's local Plasma store, just
    # like an explicit ``ray.put``.  It is owned by the submitting driver and
    # can be reconstructed by rerunning this task if its local copy is lost.
    # Do not return a nested ``ray.put`` ObjectRef: that object is owned by the
    # temporary reader worker and becomes unreconstructable if Ray retires or
    # kills that worker under host-memory pressure.
    return read_projection(item, columns)


def _materialize(
    ray: Any,
    plan: Mapping[str, Any],
    columns: tuple[str, ...],
    nodes: Sequence[Mapping[str, Any]],
) -> tuple[Any, dict[str, Any]]:
    reader = ray.remote(num_cpus=1)(_read_and_seal)
    started = time.perf_counter()
    refs = []
    placements = []
    assigned_rows = [0] * len(nodes)
    for item in plan["slices"]:
        node_index = min(
            range(len(nodes)), key=lambda index: (assigned_rows[index], index)
        )
        node = nodes[node_index]
        node_id = str(node["NodeID"])
        refs.append(
            reader.options(scheduling_strategy=_affinity(node_id)).remote(item, columns)
        )
        placements.append(node_id)
        assigned_rows[node_index] += int(item["rows"])
    if not all(isinstance(ref, ray.ObjectRef) for ref in refs):
        raise RuntimeError("projection readers did not return Plasma ObjectRefs")
    dataset = ray.data.from_arrow_refs(refs).materialize()
    elapsed = time.perf_counter() - started
    result = _metadata(dataset)
    result.update(
        {
            "read_materialize_s": elapsed,
            "schema": str(_schema(dataset)),
            "locations": _locations(ray, dataset),
            "placement_digest": digest(placements),
            "planned_blocks_per_node": {
                node_id: placements.count(node_id)
                for node_id in sorted(set(placements))
            },
        }
    )
    if result["rows"] != int(plan["rows"]) or result["blocks"] != int(plan["blocks"]):
        raise RuntimeError("materialized input differs from its exact plan")
    return dataset, result


def _monitor_actor_class(ray: Any) -> Any:
    class Monitor:
        def __init__(self, spill_dir: str, sample_s: float = 0.1) -> None:
            self.spill_dir = Path(spill_dir)
            self.sample_s = sample_s
            self.stop_event = threading.Event()
            self.samples: list[dict[str, Any]] = []
            self.thread: threading.Thread | None = None

        def _sample(self) -> dict[str, Any]:
            import psutil
            import ray as worker_ray

            memory = psutil.virtual_memory()
            disk = psutil.disk_io_counters()
            network = psutil.net_io_counters()
            physical = 0
            if self.spill_dir.is_dir():
                for root, _, files in os.walk(self.spill_dir):
                    for name in files:
                        try:
                            physical += (Path(root) / name).stat().st_size
                        except FileNotFoundError:
                            pass
            gpu = None
            initialized = False
            try:
                import pynvml

                pynvml.nvmlInit()
                initialized = True
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu = {
                    "memory_used_bytes": int(info.used),
                    "memory_total_bytes": int(info.total),
                    "utilization_percent": int(utilization.gpu),
                }
            except BaseException:
                pass
            finally:
                if initialized:
                    pynvml.nvmlShutdown()
            return {
                "time_ns": time.time_ns(),
                "node_id": str(worker_ray.get_runtime_context().get_node_id()),
                "host_memory_used_bytes": int(memory.used),
                "host_memory_available_bytes": int(memory.available),
                "disk_read_bytes": int(disk.read_bytes) if disk else None,
                "disk_write_bytes": int(disk.write_bytes) if disk else None,
                "network_recv_bytes": int(network.bytes_recv),
                "network_sent_bytes": int(network.bytes_sent),
                "spill_directory_bytes": physical,
                "gpu": gpu,
            }

        def _run(self) -> None:
            while not self.stop_event.wait(self.sample_s):
                self.samples.append(self._sample())

        def start(self) -> str:
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
            gpu_values = [item["gpu"] for item in self.samples if item["gpu"]]
            return {
                "node_id": first["node_id"],
                "samples": len(self.samples),
                "peak_host_memory_used_bytes": max(
                    item["host_memory_used_bytes"] for item in self.samples
                ),
                "minimum_host_memory_available_bytes": min(
                    item["host_memory_available_bytes"] for item in self.samples
                ),
                "disk_read_bytes_delta": None
                if first["disk_read_bytes"] is None
                else last["disk_read_bytes"] - first["disk_read_bytes"],
                "disk_write_bytes_delta": None
                if first["disk_write_bytes"] is None
                else last["disk_write_bytes"] - first["disk_write_bytes"],
                "network_recv_bytes_delta": last["network_recv_bytes"]
                - first["network_recv_bytes"],
                "network_sent_bytes_delta": last["network_sent_bytes"]
                - first["network_sent_bytes"],
                "peak_spill_directory_bytes": max(
                    item["spill_directory_bytes"] for item in self.samples
                ),
                "gpu_peak_memory_used_bytes": max(
                    (item["memory_used_bytes"] for item in gpu_values), default=None
                ),
                "gpu_total_memory_bytes": max(
                    (item["memory_total_bytes"] for item in gpu_values), default=None
                ),
                "gpu_peak_utilization_percent": max(
                    (item["utilization_percent"] for item in gpu_values), default=None
                ),
            }

    return ray.remote(num_cpus=0)(Monitor)


def _start_monitors(
    ray: Any, nodes: Sequence[Mapping[str, Any]], spill_directory: Path
) -> list[Any]:
    actor = _monitor_actor_class(ray)
    handles = [
        actor.options(scheduling_strategy=_affinity(str(node["NodeID"]))).remote(
            str(spill_directory)
        )
        for node in nodes
    ]
    ids = ray.get([handle.start.remote() for handle in handles])
    if len(set(ids)) != len(nodes):
        raise RuntimeError("resource monitors were not placed one per node")
    return handles


def _stop_monitors(ray: Any, handles: Sequence[Any]) -> dict[str, Any]:
    nodes = ray.get([handle.stop.remote() for handle in handles])
    return {
        "nodes": nodes,
        "totals": {
            name: sum(int(item.get(name, 0) or 0) for item in nodes)
            for name in (
                "disk_read_bytes_delta",
                "disk_write_bytes_delta",
                "network_recv_bytes_delta",
                "network_sent_bytes_delta",
            )
        },
    }


def _sort(dataset: Any, keys: tuple[str, ...], backend: str) -> tuple[Any, float]:
    kwargs: dict[str, Any] = {"key": list(keys), "descending": [False] * len(keys)}
    if backend == "gpu":
        kwargs["backend"] = "gpu"
    started = time.perf_counter_ns()
    output = dataset.sort(**kwargs).materialize()
    return output, (time.perf_counter_ns() - started) / 1e9


def _configure_gpu_sort(backend: str, wave_fraction: float) -> None:
    """Set GPU controls before a Dataset captures the current DataContext."""

    if backend != "gpu":
        return
    from ray.data import DataContext

    context = DataContext.get_current()
    context.gpu_shuffle_num_actors = 16
    context.set_config("gpu_sort_auto_wave_fraction", float(wave_fraction))


def _sampling_telemetry_reasons(
    stats: Mapping[str, Any],
    *,
    input_rows: int | None = None,
    input_blocks: int | None = None,
) -> list[str]:
    """Validate proof that the production stratified CPU planner ran."""

    reasons = []
    exact = {
        "sampling_mode": SAMPLING_MODE,
        "sampling_scheme": SAMPLING_SCHEME,
        "sampling_scheme_version": SAMPLING_SCHEME_VERSION,
    }
    for name, expected in exact.items():
        if stats.get(name) != expected:
            reasons.append(
                f"GPU planner telemetry {name}={stats.get(name)!r}, expected {expected!r}"
            )

    seed = stats.get("sample_seed")
    if seed != 0 or isinstance(seed, bool):
        reasons.append("GPU planner telemetry sample_seed is not the fixed seed 0")
    for name in ("sample_target_rows", "sample_rows"):
        raw = stats.get(name)
        if not isinstance(raw, int) or isinstance(raw, bool) or raw < MIN_SAMPLE_ROWS:
            reasons.append(f"GPU planner telemetry {name} is below {MIN_SAMPLE_ROWS:,}")
    raw_bytes = stats.get("sample_bytes")
    if not isinstance(raw_bytes, int) or isinstance(raw_bytes, bool) or raw_bytes <= 0:
        reasons.append("GPU planner telemetry sample_bytes is not positive")
    planning_bytes = stats.get("planning_sample_bytes")
    if (
        not isinstance(planning_bytes, int)
        or isinstance(planning_bytes, bool)
        or planning_bytes <= 0
        or (isinstance(raw_bytes, int) and planning_bytes > raw_bytes)
    ):
        reasons.append("GPU planner telemetry planning_sample_bytes is invalid")
    sampled_blocks = stats.get("sampled_block_count")
    if (
        not isinstance(sampled_blocks, int)
        or isinstance(sampled_blocks, bool)
        or sampled_blocks <= 0
    ):
        reasons.append("GPU planner telemetry sampled_block_count is not positive")

    target = stats.get("sample_target_rows")
    actual = stats.get("sample_rows")
    if isinstance(target, int) and isinstance(actual, int) and target != actual:
        reasons.append("GPU planner telemetry target and actual sample rows differ")
    if input_rows is not None and input_blocks is not None:
        expected_target = min(int(input_rows), max(MIN_SAMPLE_ROWS, int(input_blocks)))
        if target != expected_target:
            reasons.append(
                "GPU planner telemetry sample target differs from the exact global budget"
            )
        if sampled_blocks != int(input_blocks):
            reasons.append(
                "GPU planner telemetry sampled block count differs from input blocks"
            )

    quotas = stats.get("sample_quota_rows")
    if not isinstance(quotas, Mapping) or set(quotas) != {"min", "median", "max"}:
        reasons.append("GPU planner telemetry sample_quota_rows is incomplete")
    elif (
        any(
            not isinstance(quotas[name], (int, float))
            or isinstance(quotas[name], bool)
            or quotas[name] <= 0
            for name in ("min", "median", "max")
        )
        or not quotas["min"] <= quotas["median"] <= quotas["max"]
    ):
        reasons.append("GPU planner telemetry sample_quota_rows is invalid")

    for name in ("sample_plan_digest", "sample_index_digest", "boundary_digest"):
        raw = stats.get(name)
        if not isinstance(raw, str) or SHA256.fullmatch(raw) is None:
            reasons.append(f"GPU planner telemetry {name} is not a SHA-256 digest")

    planning_h2d = stats.get("planning_h2d_bytes")
    if (
        not isinstance(planning_h2d, int)
        or isinstance(planning_h2d, bool)
        or planning_h2d < 0
        or planning_h2d > MAX_PLANNING_H2D_BYTES
    ):
        reasons.append(
            "GPU planner telemetry planning_h2d_bytes exceeds the 1 MiB gate"
        )
    subphases = stats.get("sampling_subphases_s")
    if not isinstance(subphases, Mapping):
        reasons.append("GPU planner telemetry sampling_subphases_s is missing")
    else:
        for name in SAMPLING_SUBPHASES:
            raw = subphases.get(name)
            if not isinstance(raw, (int, float)) or isinstance(raw, bool) or raw < 0:
                reasons.append(f"GPU planner telemetry subphase {name} is invalid")
    return reasons


def _ordered_partition(table: Any, keys: tuple[str, ...]) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.compute as pc

    if isinstance(table, pa.RecordBatch):
        table = pa.Table.from_batches([table])
    rows = table.num_rows
    ordered = True
    if rows > 1:
        prefix = pa.array([True] * (rows - 1))
        less_any = pa.array([False] * (rows - 1))
        for key in keys:
            column = table.column(key).combine_chunks()
            left, right = column.slice(0, rows - 1), column.slice(1)
            left_null, right_null = pc.is_null(left), pc.is_null(right)
            both_null = pc.and_(left_null, right_null)
            less = pc.if_else(
                left_null, False, pc.if_else(right_null, True, pc.less(left, right))
            )
            equal = pc.if_else(
                both_null,
                True,
                pc.if_else(pc.or_(left_null, right_null), False, pc.equal(left, right)),
            )
            less_any = pc.or_(less_any, pc.and_(prefix, less))
            prefix = pc.and_(prefix, equal)
        ordered = bool(pc.all(pc.or_(less_any, prefix)).as_py())
    return {
        "ordered": ordered,
        "rows": rows,
        "first": [table.column(key)[0].as_py() for key in keys] if rows else None,
        "last": [table.column(key)[-1].as_py() for key in keys] if rows else None,
        "row_id_sum": 0
        if rows == 0
        else int(pc.sum(table.column("row_id")).as_py() or 0),
    }


def _tuple_leq(left: Sequence[Any], right: Sequence[Any]) -> bool:
    for a, b in zip(left, right):
        if a is None and b is None:
            continue
        if a is None:
            return False
        if b is None:
            return True
        if a < b:
            return True
        if a > b:
            return False
    return True


def _validate_order(ray: Any, output: Any, keys: tuple[str, ...]) -> dict[str, Any]:
    check = ray.remote(num_cpus=1)(_ordered_partition)
    parts = ray.get([check.remote(ref, keys) for ref in _refs(output)])
    boundaries = all(
        left["last"] is None
        or right["first"] is None
        or _tuple_leq(left["last"], right["first"])
        for left, right in zip(parts, parts[1:])
    )
    rows = sum(item["rows"] for item in parts)
    row_sum = sum(item["row_id_sum"] for item in parts)
    return {
        "ordered": boundaries and all(item["ordered"] for item in parts),
        "rows": rows,
        "row_id_sum": row_sum,
        "expected_row_id_sum": rows * (rows - 1) // 2,
        "blocks_checked": len(parts),
    }


def _exact_table(ray: Any, dataset: Any) -> Any:
    import pyarrow as pa

    return pa.concat_tables(ray.get(_refs(dataset))).combine_chunks()


def _verified_bundle_digest(root: Path) -> str:
    bundle_path = root / "BUNDLE.json"
    bundle = read_json(bundle_path)
    expected = bundle.get("files")
    actual = {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path != bundle_path
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    }
    if (
        bundle.get("schema_version") != 1
        or bundle.get("kind") != "gpu_sort_cloud_bundle"
        or not isinstance(expected, dict)
        or actual != expected
        or bundle.get("bundle_digest") != digest(expected)
    ):
        raise RuntimeError("worker source differs from its immutable bundle identity")
    return str(bundle["bundle_digest"])


def _identity(args: argparse.Namespace, plan: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[4]
    cloud = Path(__file__).resolve().parent
    files = {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(cloud.glob("*.py"))
    }
    overlay = hashlib.sha256()
    overlay_count = 0
    for path in sorted((root / "python/ray/data").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        overlay.update(
            str(path.relative_to(root)).encode()
            + b"\0"
            + bytes.fromhex(file_sha256(path))
        )
        overlay_count += 1
    value = {
        "trial": {
            "kind": args.kind,
            "backend": args.backend,
            "cell": args.cell,
            "repetition": args.repetition,
            "scale_numerator": args.scale_numerator,
            "scale_denominator": args.scale_denominator,
            "wave_fraction": args.wave_fraction,
        },
        "bundle_digest": _verified_bundle_digest(root),
        "input_plan_digest": plan["digest"],
        "dataset_manifest_sha256": file_sha256(args.dataset_root / "manifest.json"),
        "wheel_sha256": RAY_WHEEL_SHA256,
        "harness_files": files,
        "ray_data_overlay": {
            "file_count": overlay_count,
            "sha256": overlay.hexdigest(),
        },
    }
    return {**value, "digest": digest(value)}


def _run_smoke(
    ray: Any,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    nodes: list[dict[str, Any]],
) -> dict[str, Any]:
    _configure_gpu_sort("gpu", args.wave_fraction)
    slices = smoke_slices(manifest)
    plan = plan_dict(slices, kind="smoke")
    columns = tuple(manifest["schema_names"]) + ("row_id",)
    dataset, input_stats = _materialize(ray, plan, columns, nodes)
    before = _stable_spill(ray)
    monitors = _start_monitors(ray, nodes, args.spill_directory)
    cpu, cpu_s = _sort(dataset, SMOKE_KEYS, "pyarrow")
    gpu, gpu_s = _sort(dataset, SMOKE_KEYS, "gpu")
    resources = _stop_monitors(ray, monitors)
    after = _stable_spill(ray)
    cpu_table, gpu_table = _exact_table(ray, cpu), _exact_table(ray, gpu)
    gpu_stats = get_last_run_stats(gpu)
    reasons = []
    exact_schema = cpu_table.schema.equals(gpu_table.schema, check_metadata=True)
    exact_values = cpu_table.equals(gpu_table)
    if not exact_schema:
        reasons.append("CPU and GPU smoke schemas differ")
    if not exact_values:
        reasons.append("CPU and GPU smoke rows/values differ")
    ranks = rank_stats(gpu_stats)
    if len(ranks) != 16:
        reasons.append("smoke did not use all 16 GPU ranks")
    if len({str(item.get("node_id", "")) for item in ranks}) != 16:
        reasons.append("smoke GPU ranks do not cover 16 distinct Ray nodes")
    for field, label in (
        ("cpu_sort_rows", "CPU sorting"),
        ("cpu_merge_rows", "CPU merging"),
        ("fallback_count", "output fallback"),
        ("mpf_host_spill_bytes", "MPF host spill"),
    ):
        if int(value(gpu_stats, field, default=-1)) != 0:
            reasons.append(f"smoke GPU backend used {label}")
    reasons.extend(
        _sampling_telemetry_reasons(
            gpu_stats,
            input_rows=int(input_stats["rows"]),
            input_blocks=int(input_stats["blocks"]),
        )
    )
    identity = _identity(args, plan)
    return {
        "valid": not reasons,
        "status": "accepted" if not reasons else "rejected",
        "kind": "smoke",
        "backend": "smoke",
        "cell": args.cell,
        "repetition": args.repetition,
        "scale_numerator": args.scale_numerator,
        "scale_denominator": args.scale_denominator,
        "wave_fraction": args.wave_fraction,
        "cpu_sort_s": cpu_s,
        "gpu_sort_s": gpu_s,
        "cold_sort_s": gpu_s,
        "input": input_stats,
        "output": _metadata(gpu),
        "gpu_stats": gpu_stats,
        "resources": resources,
        "ray_object_store_io": _spill_delta(before, after),
        "validation": {
            "exact_every_row_value": exact_schema and exact_values,
            "rows": cpu_table.num_rows,
        },
        "rejection_reasons": reasons,
        "plan": {key: value for key, value in plan.items() if key != "slices"},
        "artifact_identity": identity,
        "ray_version": ray.__version__,
        "ray_commit": ray.__commit__,
    }


def _run_performance(
    ray: Any,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    nodes: list[dict[str, Any]],
) -> dict[str, Any]:
    _configure_gpu_sort(args.backend, args.wave_fraction)
    cell = cell_by_name(manifest, args.cell)
    slices = (
        scaled_cohort_slices(manifest, args.scale_numerator, args.scale_denominator)
        if args.kind == "natural"
        else cohort_slices(manifest)
    )
    plan = plan_dict(slices, kind=args.kind)
    before_input = _stable_spill(ray)
    dataset, input_stats = _materialize(ray, plan, cell.columns, nodes)
    after_input = _stable_spill(ray)
    input_io = _spill_delta(before_input, after_input)
    before_sort = after_input
    monitors = _start_monitors(ray, nodes, args.spill_directory)
    output = None
    elapsed = None
    exception = None
    try:
        output, elapsed = _sort(dataset, cell.keys, args.backend)
    except BaseException as error:
        exception = f"{type(error).__name__}: {error}"
        exception_traceback = traceback.format_exc()
    resources = _stop_monitors(ray, monitors)
    after_sort = _stable_spill(ray)
    reasons = []
    input_spilled = int(input_io["totals"]["spilled_bytes_total"] or 0)
    input_restored = int(input_io["totals"]["restored_bytes_total"] or 0)
    if input_spilled or input_restored:
        reasons.append(
            "input was not fully resident at timer start: "
            f"spilled={input_spilled}, restored={input_restored}"
        )
    if not input_stats["locations"]["all_locatable"]:
        reasons.append("one or more materialized input ObjectRefs are not locatable")
    planned = input_stats.get("planned_blocks_per_node") or {}
    if len(planned) != 16 or any(int(count) <= 0 for count in planned.values()):
        reasons.append("input placement receipts do not cover all 16 Ray nodes")
    gpu_stats: dict[str, Any] = {}
    output_stats: dict[str, Any] = {}
    validation: dict[str, Any] = {}
    if exception is not None:
        reasons.append(f"sort failed: {exception}")
    else:
        assert output is not None and elapsed is not None
        output_stats = {
            **_metadata(output),
            "schema": str(_schema(output)),
            "locations": _locations(ray, output),
        }
        validation = _validate_order(ray, output, cell.keys)
        if output_stats["rows"] != input_stats["rows"]:
            reasons.append("output row count differs from input")
        if output_stats["schema"] != input_stats["schema"]:
            reasons.append("output schema differs from input")
        if not output_stats["locations"]["all_locatable"]:
            reasons.append("one or more output ObjectRefs are not Ray-locatable")
        if not validation["ordered"]:
            reasons.append("output is not globally ordered")
        if validation["row_id_sum"] != validation["expected_row_id_sum"]:
            reasons.append("output row_id checksum differs from the exact cohort")
        if args.backend == "gpu":
            gpu_stats = get_last_run_stats(output)
            missing = required_fields_missing(gpu_stats)
            if missing:
                reasons.append(f"missing GPU telemetry: {missing}")
            ranks = rank_stats(gpu_stats)
            if len(ranks) != 16:
                reasons.append(f"GPU telemetry has {len(ranks)} ranks, expected 16")
            if len({str(item.get("node_id", "")) for item in ranks}) != 16:
                reasons.append("GPU ranks do not cover 16 distinct Ray nodes")
            if int(value(gpu_stats, "cpu_sort_rows", default=-1)) != 0:
                reasons.append("GPU backend performed CPU sorting")
            if int(value(gpu_stats, "cpu_merge_rows", default=-1)) != 0:
                reasons.append("GPU backend performed CPU merging")
            if int(value(gpu_stats, "fallback_count", default=-1)) != 0:
                reasons.append("GPU backend used an output-conversion fallback")
            if int(value(gpu_stats, "mpf_host_spill_bytes", default=-1)) != 0:
                reasons.append("GPU backend used MPF host spill")
            reasons.extend(
                _sampling_telemetry_reasons(
                    gpu_stats,
                    input_rows=int(input_stats["rows"]),
                    input_blocks=int(input_stats["blocks"]),
                )
            )
    if args.kind == "trend" and (
        input_stats["rows"] != EXPECTED_ROWS or input_stats["blocks"] != EXPECTED_BLOCKS
    ):
        reasons.append("trend cell does not use the fixed 80,738,761-row cohort")
    if (
        cell.name == "full"
        and args.scale_numerator == args.scale_denominator
        and input_stats["decoded_bytes"] != EXPECTED_FULL_BYTES
    ):
        reasons.append(
            "the 1x full payload changed decoded size: "
            f"{input_stats['decoded_bytes']} != {EXPECTED_FULL_BYTES}"
        )
    identity = _identity(args, plan)
    result = {
        "valid": not reasons,
        "status": "accepted" if not reasons else "rejected",
        "kind": args.kind,
        "backend": args.backend,
        "cell": cell.to_dict(),
        "repetition": args.repetition,
        "scale_numerator": args.scale_numerator,
        "scale_denominator": args.scale_denominator,
        "wave_fraction": args.wave_fraction,
        "timing_boundary": "materialized projected input in Plasma to sorted output sealed and Ray-locatable",
        "cold_sort_s": elapsed,
        "input": {**input_stats, "ray_object_store_io": input_io},
        "output": output_stats,
        "throughput_rows_s": None if elapsed is None else input_stats["rows"] / elapsed,
        "throughput_gib_s": None
        if elapsed is None
        else input_stats["decoded_bytes"] / GIB / elapsed,
        "gpu_stats": gpu_stats,
        "gpu_peak_device_bytes": peak_device_bytes(gpu_stats),
        "resources": resources,
        "ray_object_store_io": _spill_delta(before_sort, after_sort),
        "object_store_memory_bytes": int(
            ray.cluster_resources().get("object_store_memory", 0)
        ),
        "ray_disk_spill_directory": str(args.spill_directory),
        "validation": validation,
        "rejection_reasons": reasons,
        "plan": {key: value for key, value in plan.items() if key != "slices"},
        "artifact_identity": identity,
        "ray_version": ray.__version__,
        "ray_commit": ray.__commit__,
    }
    if exception is not None:
        result["exception"] = exception_traceback
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--dataset-root", type=Path, required=True)
    value.add_argument("--spill-directory", type=Path, required=True)
    value.add_argument("--kind", choices=("smoke", "trend", "natural"), required=True)
    value.add_argument("--backend", choices=("smoke", "gpu", "pyarrow"), required=True)
    value.add_argument("--cell", default="full")
    value.add_argument("--repetition", type=int, default=1)
    value.add_argument("--scale-numerator", type=int, default=1)
    value.add_argument("--scale-denominator", type=int, default=1)
    value.add_argument("--wave-fraction", type=float, default=0.50)
    value.add_argument("--selected-wave", type=Path)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.selected_wave is not None:
        args.wave_fraction = float(
            read_json(args.selected_wave)["selected_wave_fraction"]
        )
    os.environ.pop("RAY_DATA_GPU_SORT_MEMORY_BUDGET_BYTES", None)
    import ray

    result: dict[str, Any]
    try:
        ray.init(address="auto", logging_level="ERROR")
        if ray.__version__ != RAY_VERSION or ray.__commit__ != RAY_COMMIT:
            raise RuntimeError("worker did not import the exact stock Ray wheel")
        nodes = _alive_nodes(ray)
        if len(nodes) != 16:
            raise RuntimeError(f"expected 16 live Ray nodes, found {len(nodes)}")
        manifest = load_manifest(args.dataset_root)
        if args.kind == "smoke":
            result = _run_smoke(ray, args, manifest, nodes)
        else:
            result = _run_performance(ray, args, manifest, nodes)
    except BaseException as error:
        result = {
            "valid": False,
            "status": "rejected",
            "kind": args.kind,
            "backend": args.backend,
            "cell": args.cell,
            "repetition": args.repetition,
            "scale_numerator": args.scale_numerator,
            "scale_denominator": args.scale_denominator,
            "wave_fraction": args.wave_fraction,
            "rejection_reasons": [f"{type(error).__name__}: {error}"],
            "exception": traceback.format_exc(),
        }
    finally:
        try:
            ray.shutdown()
        except BaseException:
            pass
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(artifact_sentinel(args.output, result), flush=True)
    return 0 if result.get("valid") else 1


if __name__ == "__main__":
    raise SystemExit(main())
