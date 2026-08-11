"""Physical Ray Data operator for the spillable distributed GPU sort.

The driver deliberately keeps only Plasma references until all input is known.
GPU ranks first make a bounded sampling pass, then process synchronized shuffle
waves.  This is the key distinction from the original prototype: an input no
longer has to fit in aggregate VRAM before range boundaries can be selected.
"""

from __future__ import annotations

import functools
import json
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import ray
from ray.actor import ActorHandle
from ray.data import ExecutionOptions
from ray.data._internal.execution.bundle_queue import ReorderingBundleQueue
from ray.data._internal.execution.interfaces import (
    ExecutionResources,
    PhysicalOperator,
    RefBundle,
)
from ray.data._internal.execution.interfaces.physical_operator import DataOpTask, OpTask
from ray.data._internal.execution.operators.hash_shuffle import (
    _get_total_cluster_resources,
)
from ray.data._internal.execution.operators.sub_progress import SubProgressBarMixin
from ray.data._internal.gpu_sort.config import GPUSortConfig
from ray.data.block import BlockStats, to_stats
from ray.data.context import DataContext

GPU_SORT_PARTITION_ID_KEY = b"ray-data-gpu-sort-partition"
GPU_SORT_DIAGNOSTICS_KEY = b"ray-data-gpu-sort-diagnostics"

# Driver-local benchmark hook.  A JSON round trip in the accessor prevents a
# caller from mutating the record while another Dataset is being constructed.
LAST_RUN_STATS: Dict[str, Any] = {}


def get_last_run_stats() -> Dict[str, Any]:
    return json.loads(json.dumps(LAST_RUN_STATS))


def _validate_gpu_schema(schema: Any, key_columns: Sequence[str]) -> None:
    """Validate the intentionally small, Arrow-only first-PR type surface."""

    import pyarrow as pa

    schema = getattr(schema, "base_schema", schema)
    if not isinstance(schema, pa.Schema):
        raise NotImplementedError("GPU sort currently requires Arrow-backed blocks.")
    missing = [name for name in key_columns if name not in schema.names]
    if missing:
        raise ValueError(
            f"GPU sort keys {missing} are absent from schema columns {schema.names}."
        )
    unsupported = [
        field.name
        for field in schema
        if (
            pa.types.is_nested(field.type)
            or pa.types.is_union(field.type)
            or pa.types.is_dictionary(field.type)
            or isinstance(field.type, pa.ExtensionType)
        )
    ]
    if unsupported:
        raise NotImplementedError(
            "GPU sort supports flat Arrow scalar columns only; unsupported "
            f"columns: {unsupported}."
        )

    key_types = [schema.field(name).type for name in key_columns]
    bad_keys = [
        name
        for name, typ in zip(key_columns, key_types)
        if not (
            pa.types.is_boolean(typ)
            or pa.types.is_string(typ)
            or pa.types.is_large_string(typ)
            or pa.types.is_integer(typ)
            or pa.types.is_floating(typ)
            or pa.types.is_date(typ)
            or pa.types.is_time(typ)
            or pa.types.is_timestamp(typ)
        )
    ]
    if bad_keys:
        raise NotImplementedError(
            "GPU sort keys must be boolean, string, integer, float, date, "
            f"time, or timestamp columns; unsupported keys: {bad_keys}."
        )


@dataclass(frozen=True)
class _InputBlock:
    value: Any
    size_bytes: int
    num_rows: int


def _underlying_object_ref(block: _InputBlock) -> Any:
    value = block.value
    return value[0] if isinstance(value, tuple) and len(value) == 3 else value


def _assign_blocks_by_locality(
    blocks: Sequence[_InputBlock],
    actor_node_ids: Sequence[str],
    object_locations: Mapping[Any, Mapping[str, Any]],
) -> Tuple[
    List[List[_InputBlock]],
    List[int],
    List[int],
    List[int],
    List[int],
]:
    """Assign blocks locally, with deterministic decoded-byte balancing.

    An object may have multiple replicas, no reported location (for example an
    inline object), or a location on a node without a GPU-sort actor.  The same
    size-first rule is used within the local candidates and as the global
    fallback, so location lookup is never a correctness dependency.
    """

    nranks = len(actor_node_ids)
    if nranks < 1:
        raise ValueError("GPU sort requires at least one actor for block assignment.")
    blocks_by_rank: List[List[_InputBlock]] = [[] for _ in range(nranks)]
    assigned_bytes = [0] * nranks
    assigned_blocks = [0] * nranks
    local_bytes = [0] * nranks
    local_blocks = [0] * nranks

    for block in blocks:
        ref = _underlying_object_ref(block)
        try:
            location = object_locations.get(ref, {})
        except TypeError:
            location = {}
        node_ids = set(location.get("node_ids", ()) or ())
        candidates = [
            rank
            for rank, node_id in enumerate(actor_node_ids)
            if node_id and node_id in node_ids
        ]
        is_local = bool(candidates)
        if not candidates:
            candidates = list(range(nranks))
        rank = min(
            candidates,
            key=lambda item: (
                assigned_bytes[item],
                assigned_blocks[item],
                item,
            ),
        )
        blocks_by_rank[rank].append(block)
        assigned_bytes[rank] += int(block.size_bytes)
        assigned_blocks[rank] += 1
        if is_local:
            local_bytes[rank] += int(block.size_bytes)
            local_blocks[rank] += 1

    return (
        blocks_by_rank,
        assigned_bytes,
        assigned_blocks,
        local_bytes,
        local_blocks,
    )


def _make_waves(
    blocks_by_rank: Sequence[Sequence[_InputBlock]],
    target_bytes_per_rank: Optional[int],
) -> List[List[List[Any]]]:
    """Create the same number of deterministic, bounded waves for every rank."""

    rank_waves: List[List[List[Any]]] = []
    for blocks in blocks_by_rank:
        if not blocks:
            rank_waves.append([])
            continue
        if target_bytes_per_rank is None:
            rank_waves.append([[block.value for block in blocks]])
            continue
        waves: List[List[Any]] = []
        current: List[Any] = []
        current_bytes = 0
        for block in blocks:
            if current and current_bytes + block.size_bytes > target_bytes_per_rank:
                waves.append(current)
                current = []
                current_bytes = 0
            current.append(block.value)
            current_bytes += block.size_bytes
        if current:
            waves.append(current)
        rank_waves.append(waves)

    count = max((len(waves) for waves in rank_waves), default=0)
    return [
        [waves[index] if index < len(waves) else [] for waves in rank_waves]
        for index in range(count)
    ]


def _wave_target_bytes(
    blocks_by_rank: Sequence[Sequence[_InputBlock]],
    *,
    explicit_residency_budget_bytes: Optional[int],
    actor_usable_budgets: Sequence[int],
    auto_wave_fraction: float,
) -> Optional[int]:
    """Return a bounded-wave target, or ``None`` for the resident fast path."""

    if explicit_residency_budget_bytes is not None:
        # Preserve the original capacity-study behavior exactly.
        return max(256 << 20, int(explicit_residency_budget_bytes) // 2)

    budgets = [int(value) for value in actor_usable_budgets if int(value) > 0]
    if not budgets:
        raise RuntimeError("GPU sort actors did not report usable memory budgets.")
    target = max(256 << 20, int(min(budgets) * float(auto_wave_fraction)))
    largest_rank_input = max(
        (sum(block.size_bytes for block in blocks) for blocks in blocks_by_rank),
        default=0,
    )
    return None if largest_rank_input <= target else target


def _derive_num_ranks(data_context: DataContext) -> int:
    configured = data_context.gpu_shuffle_num_actors
    if configured is not None:
        if configured < 1:
            raise ValueError("gpu_shuffle_num_actors must be positive.")
        return int(configured)
    ranks = int(_get_total_cluster_resources().gpu or 0)
    if ranks < 1:
        raise RuntimeError(
            "GPU sort requires at least one Ray GPU resource. Set "
            "DataContext.gpu_shuffle_num_actors to override detection."
        )
    return ranks


def _operator_config(data_context: DataContext) -> Dict[str, Any]:
    env_budget = os.environ.get("RAY_DATA_GPU_SORT_MEMORY_BUDGET_BYTES")
    context_budget = data_context.get_config("gpu_sort_memory_budget_bytes", None)
    sample_size = max(
        65_536, int(data_context.get_config("gpu_sort_sample_size", 65_536))
    )
    return GPUSortConfig(
        sample_size=sample_size,
        sample_seed=int(data_context.get_config("gpu_sort_sample_seed", 0)),
        residency_budget_bytes=(
            env_budget if env_budget is not None else context_budget
        ),
        auto_wave_fraction=float(
            data_context.get_config("gpu_sort_auto_wave_fraction", 0.50)
        ),
        exchange_batch_bytes=int(
            data_context.get_config("gpu_sort_exchange_batch_bytes", 512 << 20)
        ),
        run_chunk_bytes=int(
            data_context.get_config("gpu_sort_run_chunk_bytes", 512 << 20)
        ),
        setup_timeout_s=float(
            data_context.get_config("gpu_sort_setup_timeout_s", 300.0)
        ),
    ).to_actor_dict()


class _RankPool:
    """Fresh, non-detached one-actor-per-GPU rank pool."""

    def __init__(
        self,
        nranks: int,
        key_columns: List[str],
        ascending: List[bool],
        config: Dict[str, Any],
    ) -> None:
        self.nranks = nranks
        self.key_columns = key_columns
        self.ascending = ascending
        self.config = config
        self.actors: List[ActorHandle] = []
        self.rank_infos: List[Dict[str, Any]] = []
        self._shutdown_lock = threading.Lock()

    def start(self) -> None:
        from ray.data._internal.gpu_sort.actor import GPUSortActor

        self.actors = [
            GPUSortActor.options(
                num_gpus=1,
                num_cpus=1,
                scheduling_strategy="SPREAD",
            ).remote(
                nranks=self.nranks,
                index=rank,
                key_columns=self.key_columns,
                ascending=self.ascending,
                num_partitions=self.nranks,
                config=self.config,
            )
            for rank in range(self.nranks)
        ]
        timeout = self.config["setup_timeout_s"]
        root_rank, address = ray.get(
            self.actors[0].setup_root.remote(), timeout=timeout
        )
        if int(root_rank) != 0:
            raise RuntimeError(
                f"GPU sort communicator assigned root rank {root_rank}, expected 0."
            )
        setup = ray.get(
            [actor.setup_worker.remote(address) for actor in self.actors],
            timeout=timeout,
        )
        actors_by_rank: List[Optional[ActorHandle]] = [None] * self.nranks
        infos_by_rank: List[Optional[Dict[str, Any]]] = [None] * self.nranks
        for actor, result in zip(self.actors, setup):
            rank = int(result["rank"])
            if not 0 <= rank < self.nranks or actors_by_rank[rank] is not None:
                raise RuntimeError(
                    f"GPU sort communicator returned invalid rank {rank}."
                )
            actors_by_rank[rank] = actor
            infos_by_rank[rank] = dict(result)
        if any(actor is None for actor in actors_by_rank):
            raise RuntimeError("GPU sort communicator rank assignment is incomplete.")
        self.actors = [actor for actor in actors_by_rank if actor is not None]
        self.rank_infos = [info for info in infos_by_rank if info is not None]
        ready = ray.get(
            [actor.is_ready.remote() for actor in self.actors], timeout=timeout
        )
        if not all(ready):
            raise RuntimeError("One or more GPU sort ranks failed initialization.")

    def shutdown(self) -> None:
        with self._shutdown_lock:
            actors, self.actors = self.actors, []
        if not actors:
            return
        try:
            ray.get([actor.release.remote() for actor in actors], timeout=10)
        except Exception:
            pass
        finally:
            for actor in actors:
                ray.kill(actor, no_restart=True)

    def shutdown_async(self) -> None:
        """Reap successful one-shot ranks without extending output sealing time."""

        thread = threading.Thread(
            target=self.shutdown,
            name="gpu-sort-rank-reaper",
            daemon=True,
        )
        thread.start()


class GPUSortOperator(PhysicalOperator, SubProgressBarMixin):
    """Unified resident/spillable distributed GPU range sort."""

    def __init__(
        self,
        input_op: PhysicalOperator,
        data_context: DataContext,
        *,
        sort_key: Any,
    ) -> None:
        if sort_key.boundaries is not None:
            raise ValueError("GPU sort does not support explicit `boundaries`.")
        key_columns = list(sort_key.get_columns())
        if not key_columns:
            raise ValueError("GPU sort requires at least one sort key.")
        nranks = _derive_num_ranks(data_context)
        descending = list(sort_key.get_descending())
        config = _operator_config(data_context)
        super().__init__(
            name=f"GPUSort(keys={key_columns}, ranks={nranks})",
            input_dependencies=[input_op],
            data_context=data_context,
        )
        self._sort_key = sort_key
        self._key_columns = key_columns
        self._config = config
        self._rank_pool = _RankPool(
            nranks, key_columns, [not value for value in descending], config
        )
        self._input_blocks: List[_InputBlock] = []
        self._blocks_by_rank: List[List[_InputBlock]] = [[] for _ in range(nranks)]
        self._assigned_bytes = [0] * nranks
        self._assigned_blocks = [0] * nranks
        self._local_bytes = [0] * nranks
        self._local_blocks = [0] * nranks
        self._wave_target_bytes: Optional[int] = None
        self._wave_count = 0
        self._input_bundles: List[RefBundle] = []
        self._input_rows = 0
        self._input_bytes = 0
        self._input_schema = None
        self._input_stats: List[BlockStats] = []
        self._output_stats: List[BlockStats] = []
        self._output_queue = ReorderingBundleQueue()
        self._extraction_tasks: Dict[int, DataOpTask] = {}
        self._finalization_started = False
        self._finalization_succeeded = False
        self._run_started_at: Optional[float] = None
        self._controller_phases: Dict[str, float] = {}
        self._sample_manifests: List[Dict[str, Any]] = []
        self._sample_rows = 0
        self._sample_bytes = 0
        self._sampling_subphases = {
            "cpu_sample_construction": 0.0,
            "boundary_sort": 0.0,
            "orchestration_remainder": 0.0,
        }
        self._progress = {"GPU Sample": None, "GPU Sort/Merge": None}

    def start(self, options: ExecutionOptions) -> None:
        # Actor creation happens after all Plasma refs are collected, but before
        # sampling.  It remains inside the measured Dataset.sort boundary.
        super().start(options)

    def _add_input_inner(self, bundle: RefBundle, input_index: int) -> None:
        if input_index != 0:
            raise ValueError("GPU sort accepts exactly one input dependency.")
        self._input_bundles.append(bundle)
        self._input_stats.extend(to_stats(bundle.metadata))
        if self._input_schema is None and bundle.schema is not None:
            self._input_schema = bundle.schema

        for (block_ref, metadata), block_slice in zip(bundle.blocks, bundle.slices):
            if block_slice is None:
                value: Any = block_ref
                rows = int(metadata.num_rows or 0)
                size_bytes = int(metadata.size_bytes or 0)
            else:
                value = (
                    block_ref,
                    int(block_slice.start_offset),
                    int(block_slice.end_offset),
                )
                rows = int(block_slice.num_rows)
                full_rows = int(metadata.num_rows or 0)
                size_bytes = (
                    max(1, math.ceil((metadata.size_bytes or 0) * rows / full_rows))
                    if full_rows
                    else int(metadata.size_bytes or 0)
                )
            # Actor placement and communicator rank order are not known until
            # MPF bootstrap, so assignment is deliberately deferred.
            self._input_blocks.append(
                _InputBlock(value=value, size_bytes=size_bytes, num_rows=rows)
            )
            self._input_bytes += size_bytes
            self._input_rows += rows

    def _assign_input_blocks(self) -> None:
        actor_node_ids = [
            str(info.get("node_id", "")) for info in self._rank_pool.rank_infos
        ]
        refs = []
        seen = set()
        for block in self._input_blocks:
            ref = _underlying_object_ref(block)
            if isinstance(ref, ray.ObjectRef) and ref not in seen:
                seen.add(ref)
                refs.append(ref)
        try:
            locations = ray.experimental.get_object_locations(refs) if refs else {}
        except Exception:
            # The API is experimental and excludes some valid objects.  A
            # deterministic non-local assignment is always safe.
            locations = {}
        (
            self._blocks_by_rank,
            self._assigned_bytes,
            self._assigned_blocks,
            self._local_bytes,
            self._local_blocks,
        ) = _assign_blocks_by_locality(
            self._input_blocks, actor_node_ids, locations
        )

    def _sample(self) -> Tuple[Any, Any]:
        target = self._config["sample_size"]
        per_rank = max(1, math.ceil(target / self._rank_pool.nranks))
        seed = self._config["sample_seed"]
        construction_started = time.perf_counter()
        refs = [
            actor.sample_blocks.remote(
                [block.value for block in blocks], per_rank, seed + rank
            )
            for rank, (actor, blocks) in enumerate(
                zip(self._rank_pool.actors, self._blocks_by_rank)
            )
        ]
        manifests = ray.get(refs, timeout=self._config["setup_timeout_s"])
        self._sampling_subphases["cpu_sample_construction"] = (
            time.perf_counter() - construction_started
        )
        self._sample_manifests = [dict(item) for item in manifests]
        schema = self._input_schema or next(
            (
                item.get("schema")
                for item in manifests
                if item.get("schema") is not None
            ),
            None,
        )
        if schema is None:
            raise ValueError("GPU sort could not determine the input Arrow schema.")
        _validate_gpu_schema(schema, self._key_columns)
        self._sort_key.validate_schema(getattr(schema, "base_schema", schema))
        samples = [
            item["sample"]
            for item in manifests
            if item.get("sample") is not None and item["sample"].num_rows
        ]
        result = ray.get(
            self._rank_pool.actors[0].compute_boundaries.remote(samples, schema),
            timeout=self._config["setup_timeout_s"],
        )
        self._sample_rows = int(result.get("sample_rows", 0) or 0)
        self._sample_bytes = int(result.get("sample_bytes", 0) or 0)
        self._sampling_subphases["boundary_sort"] = float(
            result.get("boundary_sort_s", 0.0) or 0.0
        )
        return schema, result["boundaries"]

    def _plan_waves(self) -> List[List[List[Any]]]:
        budget = self._config["residency_budget_bytes"]
        self._wave_target_bytes = _wave_target_bytes(
            self._blocks_by_rank,
            explicit_residency_budget_bytes=budget,
            actor_usable_budgets=[
                int(info.get("usable_memory_budget_bytes", 0) or 0)
                for info in self._rank_pool.rank_infos
            ],
            auto_wave_fraction=float(self._config["auto_wave_fraction"]),
        )
        waves = _make_waves(self._blocks_by_rank, self._wave_target_bytes)
        self._wave_count = len(waves)
        return waves

    def _try_finalize(self) -> None:
        if self._finalization_started or not self._inputs_complete:
            return
        self._finalization_started = True
        self._run_started_at = time.perf_counter()
        if not self._input_blocks:
            self._finalization_succeeded = True
            self._publish_diagnostics([])
            return

        try:
            started = time.perf_counter()
            self._rank_pool.start()
            self._controller_phases["startup"] = time.perf_counter() - started

            started = time.perf_counter()
            self._assign_input_blocks()
            self._controller_phases["input_assignment"] = (
                time.perf_counter() - started
            )

            started = time.perf_counter()
            schema, boundaries = self._sample()
            self._controller_phases["sampling"] = time.perf_counter() - started
            self._sampling_subphases["orchestration_remainder"] = max(
                0.0,
                self._controller_phases["sampling"]
                - self._sampling_subphases["cpu_sample_construction"]
                - self._sampling_subphases["boundary_sort"],
            )

            started = time.perf_counter()
            ray.get(
                [
                    actor.install_plan.remote(schema, boundaries)
                    for actor in self._rank_pool.actors
                ],
                timeout=self._config["setup_timeout_s"],
            )
            waves = self._plan_waves()
            for wave_id, blocks_for_ranks in enumerate(waves):
                ray.get(
                    [
                        actor.process_wave.remote(wave_id, blocks)
                        for actor, blocks in zip(
                            self._rank_pool.actors, blocks_for_ranks
                        )
                    ]
                )
            self._controller_phases["partition_and_exchange"] = (
                time.perf_counter() - started
            )
            self._schedule_extraction()
        except Exception:
            self._rank_pool.shutdown()
            raise

    def _schedule_extraction(self) -> None:
        for rank, actor in enumerate(self._rank_pool.actors):

            def _on_bundle_ready(bundle: RefBundle, rank: int = rank) -> None:
                schema = bundle.schema
                metadata = schema.metadata if schema is not None else None
                partition_id = rank
                if metadata and GPU_SORT_PARTITION_ID_KEY in metadata:
                    partition_id = int(metadata[GPU_SORT_PARTITION_ID_KEY].decode())
                if partition_id != rank:
                    raise RuntimeError(
                        "GPU sort rank emitted an output for a nonlocal ordered "
                        f"partition: rank={rank}, partition={partition_id}."
                    )
                if metadata:
                    clean = {
                        key: value
                        for key, value in metadata.items()
                        if key
                        not in (GPU_SORT_PARTITION_ID_KEY, GPU_SORT_DIAGNOSTICS_KEY)
                    }
                    schema = schema.with_metadata(clean or None)
                    bundle = RefBundle(
                        bundle.blocks,
                        schema=schema,
                        owns_blocks=bundle.owns_blocks,
                        slices=bundle.slices,
                    )
                self._output_queue.add(bundle, key=rank)
                self._metrics.on_output_queued(bundle)
                self._metrics.on_task_output_generated(rank, bundle)
                progress = self._progress["GPU Sort/Merge"]
                if progress is not None:
                    progress.update(
                        increment=bundle.num_rows() or 0, total=self._input_rows
                    )

            def _on_done(
                exc: Optional[Exception],
                worker_stats=None,
                driver_stats=None,
                rank: int = rank,
            ) -> None:
                self._extraction_tasks.pop(rank, None)
                self._output_queue.finalize(key=rank)
                self._metrics.on_task_finished(
                    task_index=rank,
                    exception=exc,
                    task_exec_stats=worker_stats,
                    task_exec_driver_stats=driver_stats,
                )
                if exc is not None:
                    self._rank_pool.shutdown()
                    return
                if not self._extraction_tasks:
                    diagnostics = ray.get(
                        [actor.diagnostics.remote() for actor in self._rank_pool.actors]
                    )
                    self._publish_diagnostics(diagnostics)
                    self._finalization_succeeded = True
                    self._rank_pool.shutdown_async()

            generator = actor.finish_and_extract.options(
                num_returns="streaming"
            ).remote()
            task = DataOpTask(
                task_index=rank,
                streaming_gen=generator,
                output_ready_callback=_on_bundle_ready,
                task_done_callback=functools.partial(_on_done, rank=rank),
                operator_name=self.name,
            )
            self._extraction_tasks[rank] = task
            self._metrics.on_task_submitted(
                rank, RefBundle([], schema=None, owns_blocks=False), task.get_task_id()
            )

    def _publish_diagnostics(self, diagnostics: List[Dict[str, Any]]) -> None:
        global LAST_RUN_STATS

        ranks: List[Dict[str, Any]] = []
        for default_rank, raw in enumerate(diagnostics):
            item = dict(raw or {})
            item.setdefault("rank", default_rank)
            item.setdefault(
                "node_id",
                self._rank_pool.rank_infos[default_rank].get("node_id", "")
                if default_rank < len(self._rank_pool.rank_infos)
                else "",
            )
            item.setdefault(
                "usable_memory_budget_bytes",
                self._rank_pool.rank_infos[default_rank].get(
                    "usable_memory_budget_bytes", 0
                )
                if default_rank < len(self._rank_pool.rank_infos)
                else 0,
            )
            item.setdefault("peak_device_bytes", 0)
            item.setdefault("input_bytes", self._assigned_bytes[default_rank])
            item.setdefault("input_blocks", self._assigned_blocks[default_rank])
            item.setdefault("local_input_bytes", self._local_bytes[default_rank])
            item.setdefault("local_input_blocks", self._local_blocks[default_rank])
            for name in (
                "output_bytes",
                "externalized_bytes",
                "externalized_rows",
                "initial_run_count",
                "merge_pass_count",
                "replacement_run_count",
                "h2d_bytes",
                "planning_h2d_bytes",
                "d2h_bytes",
                "plasma_read_bytes",
                "plasma_write_bytes",
                "mpf_host_spill_bytes",
                "ray_disk_spill_bytes",
                "cpu_sort_rows",
                "cpu_merge_rows",
                "fallback_count",
            ):
                item.setdefault(name, 0)
            item.setdefault("phases_s", {})
            ranks.append(item)

        def total(name: str) -> int:
            return sum(int(item.get(name, 0) or 0) for item in ranks)

        phase_names = (
            "partition",
            "mpf_shuffle",
            "run_sort",
            "gpu_merge",
            "arrow_conversion",
            "plasma_seal",
        )
        phases = {
            name: max(
                (float(item.get("phases_s", {}).get(name, 0) or 0) for item in ranks),
                default=0.0,
            )
            for name in phase_names
        }
        phases["sampling"] = float(self._controller_phases.get("sampling", 0.0))
        elapsed = (
            time.perf_counter() - self._run_started_at
            if self._run_started_at is not None
            else 0.0
        )
        phases["orchestration"] = max(0.0, elapsed - sum(phases.values()))
        externalized_bytes = total("externalized_bytes")
        budgets = [int(item.get("memory_budget_bytes", 0) or 0) for item in ranks]
        configured_budget = int(self._config["residency_budget_bytes"] or 0)
        first_times = [
            float(item["first_externalize_s"])
            for item in ranks
            if item.get("first_externalize_s") is not None
        ]
        first_waves = [
            int(item["first_externalize_wave"])
            for item in ranks
            if item.get("first_externalize_wave") is not None
        ]
        LAST_RUN_STATS = {
            "mode": "external" if externalized_bytes else "resident",
            "sampling_mode": "cpu_sampled_arrow",
            "sample_rows": self._sample_rows,
            "sample_bytes": self._sample_bytes,
            "planning_h2d_bytes": total("planning_h2d_bytes"),
            "sampling_subphases_s": dict(self._sampling_subphases),
            "memory_budget_bytes": max(budgets, default=0) or configured_budget,
            "peak_device_bytes": max(
                (int(item["peak_device_bytes"]) for item in ranks), default=0
            ),
            "input_rows": self._input_rows,
            "input_bytes": self._input_bytes,
            "auto_wave_fraction": float(self._config["auto_wave_fraction"]),
            "wave_target_bytes": self._wave_target_bytes,
            "wave_count": self._wave_count,
            "ranks": ranks,
            "externalized_bytes": externalized_bytes,
            "externalized_rows": total("externalized_rows"),
            "first_externalize_s": min(first_times) if first_times else None,
            "first_externalize_wave": min(first_waves) if first_waves else None,
            "initial_run_count": total("initial_run_count"),
            "merge_pass_count": max(
                (int(item["merge_pass_count"]) for item in ranks), default=0
            ),
            "replacement_run_count": total("replacement_run_count"),
            "h2d_bytes": total("h2d_bytes"),
            "d2h_bytes": total("d2h_bytes"),
            "plasma_read_bytes": total("plasma_read_bytes"),
            "plasma_write_bytes": total("plasma_write_bytes"),
            "mpf_host_spill_bytes": total("mpf_host_spill_bytes"),
            "ray_disk_spill_bytes": total("ray_disk_spill_bytes"),
            "cpu_sort_rows": total("cpu_sort_rows"),
            "cpu_merge_rows": total("cpu_merge_rows"),
            "fallback_count": total("fallback_count"),
            "phases_s": phases,
            "controller_phases_s": dict(self._controller_phases),
            "total_s": elapsed,
        }

    def has_next(self) -> bool:
        self._try_finalize()
        return self._output_queue.has_next()

    def _get_next_inner(self) -> RefBundle:
        bundle = self._output_queue.get_next()
        self._metrics.on_output_dequeued(bundle)
        self._output_stats.extend(to_stats(bundle.metadata))
        return bundle

    def get_active_tasks(self) -> List[OpTask]:
        self._try_finalize()
        return list(self._extraction_tasks.values())

    def has_completed(self) -> bool:
        return (
            self._finalization_started
            and not self._extraction_tasks
            and super().has_completed()
        )

    def _do_shutdown(self, force: bool = False) -> None:
        self._rank_pool.shutdown()
        self._input_bundles.clear()
        self._extraction_tasks.clear()
        super()._do_shutdown(force)

    def current_logical_usage(self) -> ExecutionResources:
        return ExecutionResources(
            gpu=(len(self._rank_pool.actors) or self._rank_pool.nranks)
        )

    @property
    def base_resource_usage(self) -> ExecutionResources:
        return ExecutionResources(gpu=self._rank_pool.nranks)

    def incremental_resource_usage(self) -> ExecutionResources:
        return ExecutionResources(gpu=1)

    def get_sub_progress_bar_names(self) -> List[str]:
        return list(self._progress)

    def set_sub_progress_bar(self, name: str, pg: Any) -> None:
        if name in self._progress:
            self._progress[name] = pg

    def get_stats(self) -> Dict[str, List[BlockStats]]:
        return {
            f"{self.name}_input": self._input_stats,
            f"{self.name}_output": self._output_stats,
        }
