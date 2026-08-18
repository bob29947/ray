"""Run one BTS sort observation on an already-running Ray cluster.

The cluster lifecycle is deliberately external.  Restart Ray before invoking
this worker when measuring a cold observation; the worker only materializes an
input, times one sort, validates its output, and writes a receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

from .data import plan_dict, read_projection, scaled_slices, smoke_slices
from .spec import (
    EXPECTED_BLOCKS,
    EXPECTED_FULL_BYTES,
    EXPECTED_ROWS,
    SMOKE_KEYS,
    cell_by_name,
    load_manifest,
)

GIB = 1 << 30
SPILL_FIELDS = (
    "spilled_bytes_total",
    "spilled_objects_total",
    "restored_bytes_total",
    "restored_objects_total",
    "spill_time_total_s",
    "restore_time_total_s",
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    """Collect Ray Core spill counters from every live raylet."""

    from ray._private.internal_api import get_state_from_address, node_stats

    state = get_state_from_address(ray.get_runtime_context().gcs_address)
    nodes = sorted(
        (node for node in state.node_table() if node["Alive"]),
        key=lambda item: str(item.get("NodeID", "")),
    )
    records = []
    totals: dict[str, int | float] = dict.fromkeys(SPILL_FIELDS, 0)
    for node in nodes:
        store = node_stats(
            node_manager_address=node["NodeManagerAddress"],
            node_manager_port=node["NodeManagerPort"],
            include_memory_info=False,
        ).store_stats
        fields = {name: getattr(store, name) for name in SPILL_FIELDS}
        records.append({"node_id": str(node["NodeID"]), **fields})
        for name, amount in fields.items():
            totals[name] += amount
    return {"nodes": records, "totals": totals}


def _stable_spill(ray: Any) -> dict[str, Any]:
    """Wait outside the sort timer for asynchronous counters to settle."""

    prior = _spill_snapshot(ray)
    stable = 0
    deadline = time.monotonic() + 8
    while stable < 2 and time.monotonic() < deadline:
        time.sleep(0.25)
        current = _spill_snapshot(ray)
        stable = stable + 1 if current["totals"] == prior["totals"] else 0
        prior = current
    return prior


def _spill_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    previous = {item["node_id"]: item for item in before["nodes"]}
    records = []
    for item in after["nodes"]:
        old = previous.get(item["node_id"], {})
        records.append(
            {
                "node_id": item["node_id"],
                **{name: item[name] - old.get(name, 0) for name in SPILL_FIELDS},
            }
        )
    return {
        "nodes": records,
        "totals": {
            name: after["totals"][name] - before["totals"][name]
            for name in SPILL_FIELDS
        },
    }


def _read_and_seal(item: dict[str, Any], columns: tuple[str, ...]) -> Any:
    # A task return is sealed in the worker's local Plasma store.  Returning a
    # nested ray.put() ref would instead tie ownership to this temporary task.
    return read_projection(item, columns)


def _materialize(
    ray: Any,
    plan: Mapping[str, Any],
    columns: tuple[str, ...],
    nodes: Sequence[Mapping[str, Any]],
) -> tuple[Any, dict[str, Any]]:
    reader = ray.remote(num_cpus=1)(_read_and_seal)
    assigned_rows = [0] * len(nodes)
    placements = []
    refs = []
    started = time.perf_counter()
    for item in plan["slices"]:
        index = min(range(len(nodes)), key=lambda value: (assigned_rows[value], value))
        node_id = str(nodes[index]["NodeID"])
        refs.append(
            reader.options(scheduling_strategy=_affinity(node_id)).remote(item, columns)
        )
        placements.append(node_id)
        assigned_rows[index] += int(item["rows"])
    dataset = ray.data.from_arrow_refs(refs).materialize()
    stats = _metadata(dataset)
    stats.update(
        {
            "read_materialize_s": time.perf_counter() - started,
            "schema": str(_schema(dataset)),
            "planned_blocks_per_node": {
                node_id: placements.count(node_id)
                for node_id in sorted(set(placements))
            },
        }
    )
    if stats["rows"] != plan["rows"] or stats["blocks"] != plan["blocks"]:
        raise RuntimeError("Materialized input differs from its deterministic plan")
    return dataset, stats


def _monitor_actor(ray: Any) -> Any:
    class Monitor:
        def __init__(self, spill_directory: str) -> None:
            self.spill_directory = Path(spill_directory)
            self.stop_event = threading.Event()
            self.thread: threading.Thread | None = None
            self.samples: list[dict[str, Any]] = []

        def _sample(self) -> dict[str, Any]:
            import psutil
            import ray as worker_ray

            network = psutil.net_io_counters()
            memory = psutil.virtual_memory()
            physical_spill = 0
            if self.spill_directory.is_dir():
                for root, _, files in os.walk(self.spill_directory):
                    for name in files:
                        try:
                            physical_spill += (Path(root) / name).stat().st_size
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
                gpu = {"used_bytes": int(info.used), "total_bytes": int(info.total)}
            except BaseException:
                pass
            finally:
                if initialized:
                    pynvml.nvmlShutdown()
            return {
                "node_id": str(worker_ray.get_runtime_context().get_node_id()),
                "network_received_bytes": int(network.bytes_recv),
                "network_sent_bytes": int(network.bytes_sent),
                "host_memory_used_bytes": int(memory.used),
                "spill_directory_bytes": physical_spill,
                "gpu": gpu,
            }

        def _run(self) -> None:
            while not self.stop_event.wait(0.2):
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
            gpu = [item["gpu"] for item in self.samples if item["gpu"]]
            return {
                "node_id": first["node_id"],
                "samples": len(self.samples),
                "network_received_bytes": last["network_received_bytes"]
                - first["network_received_bytes"],
                "network_sent_bytes": last["network_sent_bytes"]
                - first["network_sent_bytes"],
                "peak_host_memory_used_bytes": max(
                    item["host_memory_used_bytes"] for item in self.samples
                ),
                "peak_spill_directory_bytes": max(
                    item["spill_directory_bytes"] for item in self.samples
                ),
                "gpu_peak_memory_used_bytes": max(
                    (item["used_bytes"] for item in gpu), default=None
                ),
                "gpu_total_memory_bytes": max(
                    (item["total_bytes"] for item in gpu), default=None
                ),
            }

    return ray.remote(num_cpus=0)(Monitor)


def _start_monitors(
    ray: Any, nodes: Sequence[Mapping[str, Any]], spill_directory: Path
) -> list[Any]:
    actor = _monitor_actor(ray)
    handles = [
        actor.options(scheduling_strategy=_affinity(str(node["NodeID"]))).remote(
            str(spill_directory)
        )
        for node in nodes
    ]
    observed = ray.get([handle.start.remote() for handle in handles])
    if set(observed) != {str(node["NodeID"]) for node in nodes}:
        raise RuntimeError("Resource monitors did not cover every Ray node")
    return handles


def _stop_monitors(ray: Any, handles: Sequence[Any]) -> dict[str, Any]:
    nodes = ray.get([handle.stop.remote() for handle in handles])
    return {
        "nodes": nodes,
        "network_received_bytes": sum(item["network_received_bytes"] for item in nodes),
        "network_sent_bytes": sum(item["network_sent_bytes"] for item in nodes),
    }


def _configure_gpu_sort(wave_fraction: float, actors: int) -> None:
    from ray.data import DataContext

    context = DataContext.get_current()
    context.gpu_shuffle_num_actors = actors
    context.set_config("gpu_sort_auto_wave_fraction", wave_fraction)


def _sort(dataset: Any, keys: tuple[str, ...], backend: str) -> tuple[Any, float]:
    kwargs: dict[str, Any] = {"key": list(keys), "descending": [False] * len(keys)}
    if backend == "gpu":
        kwargs["backend"] = "gpu"
    started = time.perf_counter_ns()
    output = dataset.sort(**kwargs).materialize()
    return output, (time.perf_counter_ns() - started) / 1e9


def _ordered_partition(table: Any, keys: tuple[str, ...]) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.compute as pc

    if isinstance(table, pa.RecordBatch):
        table = pa.Table.from_batches([table])
    rows = table.num_rows
    ordered = True
    if rows > 1:
        prefix = pa.array([True] * (rows - 1))
        less = pa.array([False] * (rows - 1))
        for key in keys:
            column = table.column(key).combine_chunks()
            left, right = column.slice(0, rows - 1), column.slice(1)
            left_null, right_null = pc.is_null(left), pc.is_null(right)
            equal_null = pc.and_(left_null, right_null)
            key_less = pc.if_else(
                left_null, False, pc.if_else(right_null, True, pc.less(left, right))
            )
            key_equal = pc.if_else(
                equal_null,
                True,
                pc.if_else(pc.or_(left_null, right_null), False, pc.equal(left, right)),
            )
            less = pc.or_(less, pc.and_(prefix, key_less))
            prefix = pc.and_(prefix, key_equal)
        ordered = bool(pc.all(pc.or_(less, prefix)).as_py())
    return {
        "ordered": ordered,
        "rows": rows,
        "first": [table.column(key)[0].as_py() for key in keys] if rows else None,
        "last": [table.column(key)[-1].as_py() for key in keys] if rows else None,
        "row_id_sum": int(pc.sum(table.column("row_id")).as_py() or 0) if rows else 0,
    }


def _tuple_leq(left: Sequence[Any], right: Sequence[Any]) -> bool:
    """Compare key tuples with the benchmark's ascending, null-last order."""

    for a, b in zip(left, right):
        if a is None and b is None:
            continue
        if a is None:
            return False
        if b is None or a < b:
            return True
        if a > b:
            return False
    return True


def _validate_output(ray: Any, output: Any, keys: tuple[str, ...]) -> dict[str, Any]:
    check = ray.remote(num_cpus=1)(_ordered_partition)
    parts = ray.get([check.remote(ref, keys) for ref in _refs(output)])
    boundaries = all(
        left["last"] is None
        or right["first"] is None
        or _tuple_leq(left["last"], right["first"])
        for left, right in zip(parts, parts[1:])
    )
    rows = sum(item["rows"] for item in parts)
    row_id_sum = sum(item["row_id_sum"] for item in parts)
    return {
        "ordered": boundaries and all(item["ordered"] for item in parts),
        "rows": rows,
        "row_id_sum": row_id_sum,
        "expected_row_id_sum": rows * (rows - 1) // 2,
        "blocks_checked": len(parts),
    }


def _exact_table(ray: Any, dataset: Any) -> Any:
    import pyarrow as pa

    return pa.concat_tables(ray.get(_refs(dataset))).combine_chunks()


def _artifact_identity(
    args: argparse.Namespace, plan: Mapping[str, Any]
) -> dict[str, Any]:
    directory = Path(__file__).resolve().parent
    harness = {
        path.name: _file_sha256(path)
        for path in sorted(directory.glob("*.py"))
        if path.name != "test_benchmark.py"
    }
    value = {
        "kind": args.kind,
        "backend": args.backend,
        "cell": args.cell,
        "repetition": args.repetition,
        "scale": [args.scale_numerator, args.scale_denominator],
        "wave_fraction": args.wave_fraction,
        "input_plan_digest": plan["digest"],
        "manifest_sha256": _file_sha256(args.dataset_root / "manifest.json"),
        "harness": harness,
    }
    value["digest"] = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return value


def _run_smoke(
    ray: Any,
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    nodes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    _configure_gpu_sort(args.wave_fraction, args.nodes)
    plan = plan_dict(smoke_slices(args.dataset_root, manifest), kind="smoke")
    columns = tuple(manifest["schema_names"]) + ("row_id",)
    dataset, input_stats = _materialize(ray, plan, columns, nodes)
    cpu, cpu_s = _sort(dataset, SMOKE_KEYS, "pyarrow")
    gpu, gpu_s = _sort(dataset, SMOKE_KEYS, "gpu")
    cpu_table, gpu_table = _exact_table(ray, cpu), _exact_table(ray, gpu)
    exact = cpu_table.schema.equals(
        gpu_table.schema, check_metadata=True
    ) and cpu_table.equals(gpu_table)
    return {
        "valid": exact,
        "kind": "smoke",
        "cpu_sort_s": cpu_s,
        "gpu_sort_s": gpu_s,
        "input": input_stats,
        "validation": {"exact_schema_and_every_row_value": exact},
        "artifact_identity": _artifact_identity(args, plan),
    }


def _run_observation(
    ray: Any,
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    nodes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    cell = cell_by_name(manifest, args.cell)
    if args.backend == "gpu":
        _configure_gpu_sort(args.wave_fraction, args.nodes)
    plan = plan_dict(
        scaled_slices(
            args.dataset_root,
            manifest,
            args.scale_numerator,
            args.scale_denominator,
        ),
        kind=args.kind,
    )
    before_input = _stable_spill(ray)
    dataset, input_stats = _materialize(ray, plan, cell.columns, nodes)
    after_input = _stable_spill(ray)
    input_spill = _spill_delta(before_input, after_input)
    monitors = _start_monitors(ray, nodes, args.spill_directory)
    output = None
    elapsed = None
    exception = None
    try:
        output, elapsed = _sort(dataset, cell.keys, args.backend)
    except BaseException as error:
        exception = f"{type(error).__name__}: {error}"
    resources = _stop_monitors(ray, monitors)
    after_sort = _stable_spill(ray)
    reasons = []
    if any(input_spill["totals"][name] for name in SPILL_FIELDS):
        reasons.append("Input materialization spilled before the timed boundary")
    output_stats: dict[str, Any] = {}
    validation: dict[str, Any] = {}
    if output is None:
        reasons.append(f"Sort failed: {exception}")
    else:
        output_stats = {**_metadata(output), "schema": str(_schema(output))}
        validation = _validate_output(ray, output, cell.keys)
        if output_stats["rows"] != input_stats["rows"]:
            reasons.append("Output row count differs from input")
        if output_stats["schema"] != input_stats["schema"]:
            reasons.append("Output schema differs from input")
        if not validation["ordered"]:
            reasons.append("Output is not globally ordered")
        if validation["row_id_sum"] != validation["expected_row_id_sum"]:
            reasons.append("Output row_id checksum differs from the exact cohort")
    if args.kind == "trend" and (
        input_stats["rows"] != EXPECTED_ROWS or input_stats["blocks"] != EXPECTED_BLOCKS
    ):
        reasons.append("Trend observation does not use the frozen 1x cohort")
    if (
        cell.name == "full"
        and args.scale_numerator == args.scale_denominator
        and input_stats["decoded_bytes"] != EXPECTED_FULL_BYTES
    ):
        reasons.append("The decoded 1x full payload changed")
    return {
        "valid": not reasons,
        "kind": args.kind,
        "backend": args.backend,
        "cell": cell.to_dict(),
        "repetition": args.repetition,
        "scale": [args.scale_numerator, args.scale_denominator],
        "wave_fraction": args.wave_fraction,
        "timing_boundary": (
            "materialized projected Plasma input through sorted output materialized "
            "and sealed in Plasma"
        ),
        "cold_sort_s": elapsed,
        "input": {**input_stats, "ray_spill": input_spill},
        "output": output_stats,
        "validation": validation,
        "throughput_rows_s": None if elapsed is None else input_stats["rows"] / elapsed,
        "throughput_gib_s": (
            None if elapsed is None else input_stats["decoded_bytes"] / GIB / elapsed
        ),
        "resources": resources,
        "ray_sort_spill": _spill_delta(after_input, after_sort),
        "rejection_reasons": reasons,
        "artifact_identity": _artifact_identity(args, plan),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--spill-directory", type=Path, required=True)
    parser.add_argument("--kind", choices=("smoke", "trend", "natural"), required=True)
    parser.add_argument("--backend", choices=("gpu", "pyarrow"), default="gpu")
    parser.add_argument("--cell", default="full")
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument("--scale-numerator", type=int, default=1)
    parser.add_argument("--scale-denominator", type=int, default=1)
    parser.add_argument("--wave-fraction", type=float, default=0.375)
    parser.add_argument("--nodes", type=int, default=16)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.spill_directory = args.spill_directory.expanduser().resolve()
    ray = None
    try:
        if args.spill_directory.is_relative_to(Path("/dev/shm")):
            raise ValueError("Ray filesystem spill must use local disk, not /dev/shm")
        import ray as ray_module

        ray = ray_module
        ray.init(address="auto", logging_level="ERROR")
        nodes = _alive_nodes(ray)
        if len(nodes) != args.nodes:
            raise RuntimeError(f"Expected {args.nodes} Ray nodes, found {len(nodes)}")
        if args.kind == "smoke" or args.backend == "gpu":
            gpu_nodes = [
                node
                for node in nodes
                if float((node.get("Resources") or {}).get("GPU", 0)) >= 1
            ]
            if len(gpu_nodes) != args.nodes:
                raise RuntimeError(
                    f"Expected one GPU-capable Ray node per rank, found {len(gpu_nodes)}"
                )
        manifest = load_manifest(args.dataset_root)
        result = (
            _run_smoke(ray, args, manifest, nodes)
            if args.kind == "smoke"
            else _run_observation(ray, args, manifest, nodes)
        )
        result.update({"ray_version": ray.__version__, "ray_commit": ray.__commit__})
    except BaseException as error:
        result = {
            "valid": False,
            "kind": args.kind,
            "backend": args.backend,
            "cell": args.cell,
            "repetition": args.repetition,
            "rejection_reasons": [f"{type(error).__name__}: {error}"],
            "exception": traceback.format_exc(),
        }
    finally:
        if ray is not None:
            ray.shutdown()
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result.get("valid") else 1


if __name__ == "__main__":
    raise SystemExit(main())
