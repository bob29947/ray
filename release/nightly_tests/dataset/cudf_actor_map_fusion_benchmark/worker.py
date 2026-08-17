#!/usr/bin/env python3
"""Run one cold arm of the eight-node cuDF actor-map fusion benchmark.

The lifecycle controller starts a fresh Ray runtime for every invocation.  Source
creation and validation are deliberately outside the measured interval; the timer
contains only construction and materialization of the five-map chain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

from benchmark_core import (
    STAGE_NAMES,
    TrialRejected,
    atomic_write_json,
    batch_probe_reasons,
    categorical_cardinalities,
    categorical_columns,
    collect_ray_data_metrics,
    combine_block_summaries,
    correctness_reasons,
    expected_schema_pairs,
    make_arrow_block,
    normalize_schema,
    numeric_columns,
    numeric_means,
    numeric_scales,
    validate_actor_plan,
    validate_schema,
)


EXPECTED_SOURCE_COMMIT = "aa17c53e462889bc835ab7f95446c3c6b80b24c4"
EXPECTED_COMPILED_RAY_COMMIT = "2741c6461d2bd3e5ff114af67be7a1190453dadd"
EXPECTED_FUSION_MODULE_SHA256 = (
    "270fbc4a89ec5b6302d88f03e3fdefd83919c484007ae710a6fc4def23b27916"
)
RESULT_MARKER = "__MAP_FUSION_RESULT__="
HASH_SEEDS = (0, 0x9E3779B1)


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
            return sectors_read * 512, sectors_written * 512, fields[2]
    return None


def _node_affinity(node_id: str) -> Any:
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    return NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)


def _monitor_actor_class(ray_module: Any) -> Any:
    """Create the proven low-frequency, zero-CPU per-node monitor actor."""

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


class BatchProbe:
    """Small side-channel actor for batch totals and transform GPU participation."""

    def __init__(self):
        self._stages: dict[str, dict[str, Any]] = {}
        self._gpu_slots: set[str] = set()
        self._actors: set[str] = set()
        self._gpu_memory_peak_bytes = 0
        self._gpu_memory_total_bytes = 0

    def record(
        self,
        stage: str,
        rows: int,
        start_ns: int,
        end_ns: int,
        actor_id: str,
        node_id: str,
        gpu_ids: tuple[str, ...],
        gpu_memory_used: int,
        gpu_memory_total: int,
    ) -> None:
        metrics = self._stages.setdefault(
            stage,
            {
                "batches": 0,
                "rows": 0,
                "first_start_ns": start_ns,
                "last_end_ns": end_ns,
                "actors": set(),
                "gpu_slots": set(),
            },
        )
        metrics["batches"] += 1
        metrics["rows"] += rows
        metrics["first_start_ns"] = min(metrics["first_start_ns"], start_ns)
        metrics["last_end_ns"] = max(metrics["last_end_ns"], end_ns)
        metrics["actors"].add(actor_id)
        self._actors.add(actor_id)
        for gpu_id in gpu_ids:
            slot = f"{node_id}:{gpu_id}"
            metrics["gpu_slots"].add(slot)
            self._gpu_slots.add(slot)
        self._gpu_memory_peak_bytes = max(self._gpu_memory_peak_bytes, gpu_memory_used)
        self._gpu_memory_total_bytes = max(
            self._gpu_memory_total_bytes, gpu_memory_total
        )

    def snapshot(self) -> dict[str, Any]:
        stages = {}
        for name, metrics in self._stages.items():
            stages[name] = {
                **metrics,
                "actors": sorted(metrics["actors"]),
                "gpu_slots": sorted(metrics["gpu_slots"]),
            }
        total = self._gpu_memory_total_bytes
        return {
            "stages": stages,
            "actors": sorted(self._actors),
            "gpu_slots": sorted(self._gpu_slots),
            "gpu_memory_peak_bytes_per_process": self._gpu_memory_peak_bytes,
            "gpu_memory_total_bytes": total,
            "gpu_memory_peak_fraction_per_process": (
                self._gpu_memory_peak_bytes / total if total else None
            ),
        }


class _MeasuredStage:
    stage_name = ""

    def __init__(self, probe):
        import ray

        context = ray.get_runtime_context()
        self._probe = probe
        self._actor_id = str(context.get_actor_id())
        self._node_id = str(context.get_node_id())
        self._gpu_ids = tuple(str(gpu_id) for gpu_id in ray.get_gpu_ids())

    def _record(self, rows: int, start_ns: int, end_ns: int) -> None:
        import cupy as cp

        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        self._probe.record.remote(
            self.stage_name,
            rows,
            start_ns,
            end_ns,
            self._actor_id,
            self._node_id,
            self._gpu_ids,
            int(total_bytes - free_bytes),
            int(total_bytes),
        )


class NumericLog1p(_MeasuredStage):
    stage_name = "NumericLog1p"

    def __init__(self, columns, probe):
        super().__init__(probe)
        self._columns = columns

    def __call__(self, batch):
        import numpy as np

        started = time.time_ns()
        output = batch.copy(deep=False)
        # Apply per-column because cuDF 25.12's DataFrame-wide path accumulates
        # masks. Restore the input mask because its NumPy dispatch turns NaN into 0.
        for column in self._columns:
            series = output[column]
            output[column] = np.log1p(series).mask(series.isna())
        ended = time.time_ns()
        self._record(len(output), started, ended)
        return output


class NumericStandardize(_MeasuredStage):
    stage_name = "NumericStandardize"

    def __init__(self, columns, means, scales, probe):
        super().__init__(probe)
        self._columns = columns
        self._means = means
        self._scales = scales

    def __call__(self, batch):
        started = time.time_ns()
        output = batch.copy(deep=False)
        for column, mean, scale in zip(self._columns, self._means, self._scales):
            output[column] = ((output[column] - mean) / scale).astype("float32")
        ended = time.time_ns()
        self._record(len(output), started, ended)
        return output


class NumericFillNulls(_MeasuredStage):
    stage_name = "NumericFillNulls"

    def __init__(self, columns, probe):
        super().__init__(probe)
        self._columns = columns

    def __call__(self, batch):
        started = time.time_ns()
        output = batch.copy(deep=False)
        for column in self._columns:
            output[column] = output[column].fillna(0.0).astype("float32")
        ended = time.time_ns()
        self._record(len(output), started, ended)
        return output


class CategoricalToString(_MeasuredStage):
    stage_name = "CategoricalToString"

    def __init__(self, columns, probe):
        super().__init__(probe)
        self._columns = columns

    def __call__(self, batch):
        started = time.time_ns()
        output = batch.copy(deep=False)
        for column in self._columns:
            output[column] = output[column].astype("str")
        ended = time.time_ns()
        self._record(len(output), started, ended)
        return output


class CategoricalVocabularyLookup(_MeasuredStage):
    stage_name = "CategoricalVocabularyLookup"

    def __init__(self, columns, cardinalities, probe):
        super().__init__(probe)
        self._columns = columns
        self._vocabularies = tuple(
            {str(value): value + 1 for value in range(cardinality)}
            for cardinality in cardinalities
        )

    def __call__(self, batch):
        started = time.time_ns()
        output = batch.copy(deep=False)
        for column, vocabulary in zip(self._columns, self._vocabularies):
            output[column] = output[column].map(vocabulary).fillna(0).astype("int32")
        ended = time.time_ns()
        self._record(len(output), started, ended)
        return output


def summarize_output_block(table, block_index: int) -> dict[str, Any]:
    """Validate one final Arrow block on GPU and hash every row twice."""

    import cudf

    from benchmark_core import (
        CATEGORICAL_COLUMN_COUNT,
        NUMERIC_COLUMN_COUNT,
        categorical_cardinalities,
        categorical_columns,
        normalize_schema,
        numeric_columns,
        validate_schema,
    )

    schema_pairs = normalize_schema(table.schema)
    validate_schema(schema_pairs)
    frame = cudf.DataFrame.from_arrow(table)
    hash_sums = []
    for seed in HASH_SEEDS:
        row_hashes = frame.hash_values(method="xxhash64", seed=seed)
        hash_sums.append(int(row_hashes.sum()) % (1 << 64))

    null_count = sum(
        table.column(index).null_count for index in range(table.num_columns)
    )
    numeric = frame[list(numeric_columns())]
    numeric_missing_count = int(numeric.isna().sum().sum())
    numeric_zero_counts = [
        int(value) for value in (numeric == 0).sum().to_arrow().to_pylist()
    ]

    categorical_zero_counts = []
    categorical_range_violations = 0
    for column, cardinality in zip(categorical_columns(), categorical_cardinalities()):
        series = frame[column]
        categorical_zero_counts.append(int((series == 0).sum()))
        categorical_range_violations += int(
            ((series < 0) | (series > cardinality)).sum()
        )

    assert len(numeric_zero_counts) == NUMERIC_COLUMN_COUNT
    assert len(categorical_zero_counts) == CATEGORICAL_COLUMN_COUNT
    return {
        "block_index": block_index,
        "rows": len(frame),
        "bytes": table.nbytes,
        "hash_sums": hash_sums,
        "null_count": null_count,
        "nan_count": max(0, numeric_missing_count - null_count),
        "numeric_zero_counts": numeric_zero_counts,
        "categorical_zero_counts": categorical_zero_counts,
        "categorical_range_violations": categorical_range_violations,
    }


def _cluster_shape(ray, expected_nodes: int) -> dict[str, Any]:
    alive = [node for node in ray.nodes() if node.get("Alive")]
    per_node = []
    for node in alive:
        resources = node.get("Resources", {})
        per_node.append(
            {
                "node_id": node.get("NodeID"),
                "address": node.get("NodeManagerAddress"),
                "cpu": float(resources.get("CPU", 0)),
                "gpu": float(resources.get("GPU", 0)),
                "object_store_memory": float(resources.get("object_store_memory", 0)),
            }
        )
    shape = {
        "alive_nodes": len(alive),
        "total_cpu": sum(node["cpu"] for node in per_node),
        "total_gpu": sum(node["gpu"] for node in per_node),
        "per_node": per_node,
    }
    if expected_nodes != 8:
        raise TrialRejected(
            f"benchmark contract requires --nodes 8, got {expected_nodes}"
        )
    if (
        shape["alive_nodes"] != 8
        or shape["total_cpu"] != 128
        or shape["total_gpu"] != 8
        or any(node["cpu"] != 16 or node["gpu"] != 1 for node in per_node)
    ):
        raise TrialRejected(f"cluster is not the required 8/128/8 shape: {shape!r}")
    return shape


def _object_store_snapshot(ray) -> dict[str, int | float]:
    from ray._private.internal_api import get_memory_info_reply, get_state_from_address

    reply = get_memory_info_reply(
        get_state_from_address(ray.get_runtime_context().gcs_address)
    )
    stats = reply.store_stats
    fields = (
        "object_store_bytes_used",
        "object_store_bytes_avail",
        "object_store_bytes_primary_copy",
        "object_store_bytes_fallback",
        "spilled_bytes_total",
        "restored_bytes_total",
        "cumulative_created_bytes",
        "cumulative_created_objects",
    )
    return {field: getattr(stats, field, 0) for field in fields}


def _object_store_delta(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    return {
        key: after.get(key, 0) - before.get(key, 0)
        for key in ("cumulative_created_bytes", "cumulative_created_objects")
    }


def _build_chain(dataset, stage_specs, map_options):
    result = dataset
    for stage_class, constructor_args in stage_specs:
        result = result.map_batches(
            stage_class,
            fn_constructor_args=constructor_args,
            **map_options,
        )
    return result


def _assert_physical_plan(dataset, mode: str) -> dict[str, Any]:
    from ray.data._internal.execution.operators.actor_pool_map_operator import (
        ActorPoolMapOperator,
    )
    from ray.data._internal.logical.optimizers import get_execution_plan

    physical_plan, _ = get_execution_plan(dataset._logical_plan)
    actor_names = [
        operator.name
        for operator in physical_plan.dag.post_order_iter()
        if isinstance(operator, ActorPoolMapOperator)
    ]
    return validate_actor_plan(mode, actor_names)


def _wait_for_probe(ray, probe, rows: int, timeout_s: float = 30.0):
    deadline = time.monotonic() + timeout_s
    snapshot = {}
    while time.monotonic() < deadline:
        snapshot = ray.get(probe.snapshot.remote())
        stages = snapshot.get("stages", {})
        if all(
            int(stages.get(stage, {}).get("rows", 0)) >= rows for stage in STAGE_NAMES
        ):
            return snapshot
        time.sleep(0.05)
    return snapshot


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("isolated", "fused"), required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--trial-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-block-rows", type=int, default=62_500)
    return parser.parse_args(argv)


def _validate_args(args) -> None:
    if args.rows <= 0:
        raise TrialRejected("--rows must be positive")
    if args.batch_size <= 0:
        raise TrialRejected("--batch-size must be positive")
    if args.source_block_rows <= 0:
        raise TrialRejected("--source-block-rows must be positive")
    if not args.trial_root.is_dir():
        raise TrialRejected(f"trial root does not exist: {args.trial_root}")
    root = args.trial_root.resolve()
    output = args.output.resolve()
    if root != output and root not in output.parents:
        raise TrialRejected(f"output {output} is outside trial root {root}")


def _identity(args) -> dict[str, Any]:
    return {
        "mode": args.mode,
        "rows": args.rows,
        "batch_size": args.batch_size,
        "nodes": args.nodes,
        "source_block_rows": args.source_block_rows,
    }


def run_trial(args) -> dict[str, Any]:
    import ray
    import ray.data
    from ray.data import ActorPoolStrategy
    from ray.data._internal.logical.rules import cudf_actor_fusion
    from ray.data.context import DataContext

    _validate_args(args)
    ray.init(address="auto")
    result = {
        **_identity(args),
        "status": "running",
        "valid": False,
        "benchmark_label": "wide synthetic tabular transform",
        "timing_scope": "materialized Arrow object store -> five map_batches -> materialized object store",
        "ray_version": ray.__version__,
        "ray_compiled_commit": getattr(ray, "__commit__", None),
        "benchmark_source_commit": EXPECTED_SOURCE_COMMIT,
    }
    if getattr(ray, "__commit__", None) != EXPECTED_COMPILED_RAY_COMMIT:
        raise TrialRejected(
            f"compiled Ray commit is {getattr(ray, '__commit__', None)!r}, expected "
            f"{EXPECTED_COMPILED_RAY_COMMIT!r}"
        )
    fusion_module_path = Path(cudf_actor_fusion.__file__)
    fusion_module_sha256 = hashlib.sha256(fusion_module_path.read_bytes()).hexdigest()
    result["fusion_module"] = {
        "path": str(fusion_module_path),
        "sha256": fusion_module_sha256,
        "expected_sha256": EXPECTED_FUSION_MODULE_SHA256,
    }
    if fusion_module_sha256 != EXPECTED_FUSION_MODULE_SHA256:
        raise TrialRejected(
            f"installed fusion module SHA-256 is {fusion_module_sha256!r}, expected "
            f"{EXPECTED_FUSION_MODULE_SHA256!r}"
        )
    result["cluster"] = _cluster_shape(ray, args.nodes)

    context = DataContext.get_current()
    context.enable_cudf_actor_fusion = args.mode == "fused"
    context.raise_original_map_exception = False
    context.actor_init_retry_on_errors = False
    context.actor_task_retry_on_errors = False
    context.retried_map_errors = False
    context.max_tasks_in_flight_per_actor = None

    source_started = time.perf_counter()
    generate = ray.remote(num_cpus=1, max_retries=0)(make_arrow_block)
    source_refs = []
    for start in range(0, args.rows, args.source_block_rows):
        stop = min(args.rows, start + args.source_block_rows)
        source_refs.append(
            generate.options(scheduling_strategy="SPREAD").remote(start, stop)
        )
    ready, _ = ray.wait(source_refs, num_returns=len(source_refs), fetch_local=False)
    if len(ready) != len(source_refs):
        raise TrialRejected("not all source Arrow blocks materialized")
    source = ray.data.from_arrow_refs(source_refs)
    source_schema = normalize_schema(source.schema())
    if source_schema != expected_schema_pairs():
        raise TrialRejected("materialized Arrow source schema does not match contract")
    if source.count() != args.rows:
        raise TrialRejected(
            "materialized Arrow source row count does not match contract"
        )
    source_prep_seconds = time.perf_counter() - source_started
    result["source"] = {
        "materialized": True,
        "blocks": len(source_refs),
        "rows": args.rows,
        "bytes": source.size_bytes(),
        "prep_seconds": source_prep_seconds,
        "schema_columns": len(source_schema),
    }

    before_store = _object_store_snapshot(ray)
    if before_store["spilled_bytes_total"] or before_store["restored_bytes_total"]:
        raise TrialRejected(
            "source preparation spilled or restored object-store data: "
            f"{before_store!r}"
        )

    ProbeActor = ray.remote(num_cpus=0, max_restarts=0)(BatchProbe)
    probe = ProbeActor.remote()
    numbers = numeric_columns()
    categories = categorical_columns()
    cardinalities = categorical_cardinalities()
    stage_specs = (
        (NumericLog1p, (numbers, probe)),
        (
            NumericStandardize,
            (numbers, numeric_means(), numeric_scales(), probe),
        ),
        (NumericFillNulls, (numbers, probe)),
        (CategoricalToString, (categories, probe)),
        (CategoricalVocabularyLookup, (categories, cardinalities, probe)),
    )
    map_options = {
        "batch_format": "cudf",
        "batch_size": args.batch_size,
        "zero_copy_batch": True,
        "compute": ActorPoolStrategy(min_size=1, max_size=8),
        "num_gpus": 1,
        "max_restarts": 0,
        "max_task_retries": 0,
    }

    # Planning proof is deliberately outside the interval.  The measured chain is
    # constructed afresh below, so this cannot materialize or cache its output.
    probe_chain = _build_chain(source, stage_specs, map_options)
    result["physical_plan"] = _assert_physical_plan(probe_chain, args.mode)

    alive_nodes = [node for node in ray.nodes() if node.get("Alive")]
    monitors = _start_monitors(ray, alive_nodes, args.trial_root)
    telemetry = None
    try:
        timer_start_wall_ns = time.time_ns()
        timer_start_ns = time.perf_counter_ns()
        transformed = _build_chain(source, stage_specs, map_options)
        output = transformed.materialize()
        timer_end_ns = time.perf_counter_ns()
        timer_end_wall_ns = time.time_ns()
    finally:
        # Stop sampling immediately after materialize(), outside the measured scope.
        telemetry = _stop_monitors(ray, monitors)
    elapsed_seconds = (timer_end_ns - timer_start_ns) / 1e9

    after_store = _object_store_snapshot(ray)
    probe_snapshot = _wait_for_probe(ray, probe, args.rows)
    stats = output._raw_stats()
    ray_data_metrics = collect_ray_data_metrics(stats)
    output_schema = normalize_schema(output.schema())
    validate_schema(output_schema)

    output_refs = output.to_arrow_refs()
    validate_block = ray.remote(num_cpus=1, num_gpus=1, max_retries=0)(
        summarize_output_block
    )
    summaries = ray.get(
        [
            validate_block.remote(block_ref, block_index)
            for block_index, block_ref in enumerate(output_refs)
        ]
    )
    correctness = combine_block_summaries(summaries)

    reasons = []
    reasons.extend(correctness_reasons(correctness, args.rows))
    reasons.extend(batch_probe_reasons(probe_snapshot, args.rows, args.mode))
    totals = ray_data_metrics["totals"]
    if totals["num_tasks_failed"]:
        reasons.append(f"Ray Data reported {totals['num_tasks_failed']} failed tasks")
    for key in (
        "global_bytes_spilled",
        "global_bytes_restored",
        "dataset_bytes_spilled",
        "obj_store_mem_spilled",
    ):
        if totals[key]:
            reasons.append(f"Ray Data reported {key}={totals[key]}")
    if after_store["spilled_bytes_total"] or after_store["restored_bytes_total"]:
        reasons.append(f"object store reported spill/restore: {after_store!r}")
    if len(telemetry["nodes"]) != args.nodes:
        reasons.append(
            f"resource telemetry covered {len(telemetry['nodes'])} nodes, expected "
            f"{args.nodes}"
        )
    if any(node["gpu_samples"] == 0 for node in telemetry["nodes"]):
        reasons.append("resource telemetry missed GPU samples on one or more nodes")
    if any(
        not node["gpu_total_memory_bytes"] or node["gpu_peak_memory_used_bytes"] is None
        for node in telemetry["nodes"]
    ):
        reasons.append("resource telemetry missed GPU memory on one or more nodes")
    elif any(
        node["gpu_peak_memory_used_bytes"] / node["gpu_total_memory_bytes"] >= 0.8
        for node in telemetry["nodes"]
    ):
        reasons.append("one or more GPUs reached at least 80% memory usage")

    probe_memory_fraction = probe_snapshot.get("gpu_memory_peak_fraction_per_process")
    if probe_memory_fraction is not None and probe_memory_fraction >= 0.8:
        reasons.append("a transform process reached at least 80% GPU memory usage")

    probe_stages = probe_snapshot.get("stages", {})
    first_start_ns = min(
        (metrics["first_start_ns"] for metrics in probe_stages.values()),
        default=None,
    )
    last_end_ns = max(
        (metrics["last_end_ns"] for metrics in probe_stages.values()),
        default=None,
    )
    result.update(
        {
            "status": "accepted" if not reasons else "rejected",
            "valid": not reasons,
            "rejection_reasons": reasons,
            "timing": {
                "elapsed_seconds": elapsed_seconds,
                "rows_per_second": args.rows / elapsed_seconds,
                "start_unix_ns": timer_start_wall_ns,
                "end_unix_ns": timer_end_wall_ns,
                "actor_startup_seconds": (
                    None
                    if first_start_ns is None
                    else max(0.0, (first_start_ns - timer_start_wall_ns) / 1e9)
                ),
                "tail_seconds": (
                    None
                    if last_end_ns is None
                    else max(0.0, (timer_end_wall_ns - last_end_ns) / 1e9)
                ),
            },
            "correctness": correctness,
            "output": {
                "materialized": True,
                "blocks": len(output_refs),
                "rows": correctness["rows"],
                "bytes": correctness["bytes"],
                "schema_columns": len(output_schema),
            },
            "batch_probe": probe_snapshot,
            "telemetry": telemetry,
            "ray_data_metrics": ray_data_metrics,
            "object_store": {
                "before": before_store,
                "after": after_store,
                "timed_created": _object_store_delta(before_store, after_store),
            },
            "execution_contract": {
                "fusion_enabled": context.enable_cudf_actor_fusion,
                "actor_pool": {"min_size": 1, "max_size": 8},
                "num_gpus_per_actor": 1,
                "serial_concurrency": True,
                "default_queue_depth": True,
                "stage_names": list(STAGE_NAMES),
            },
        }
    )
    return result


def main(argv=None) -> int:
    args = _parse_args(argv)
    result = {
        **_identity(args),
        "status": "failed",
        "valid": False,
    }
    exit_code = 1
    try:
        result = run_trial(args)
        exit_code = 0 if result["valid"] else 2
    except TrialRejected as error:
        result.update(
            {
                "status": "rejected",
                "valid": False,
                "rejection_reasons": [str(error)],
            }
        )
        exit_code = 2
    except Exception as error:  # A failed observation must still leave a receipt.
        result.update(
            {
                "status": "failed",
                "valid": False,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )
        exit_code = 1

    try:
        atomic_write_json(args.output, result)
    finally:
        print(
            RESULT_MARKER + json.dumps(result, sort_keys=True, separators=(",", ":")),
            flush=True,
        )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
