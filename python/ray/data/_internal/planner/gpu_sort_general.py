"""General end-to-end multi-GPU sort wired into Ray Data's ``ds.sort()`` path.

This backend is *general*: it represents blocks as columnar cuDF tables (rather
than a hardcoded dense int32 matrix) and therefore handles

    * arbitrary column dtypes (int / float / bool / string / datetime / ...),
    * multiple sort keys, each ascending or descending,
    * nulls (``na_position`` first/last),

while still sorting the whole dataset across every local GPU in a single pass.

Topology (mirrors RAPIDS' own ``rapidsmpf`` Ray example): instead of one actor
owning all GPUs, we use **N actors, one GPU each**, connected by a UCXX
communicator. The distributed all-to-all is done by **rapidsmpf's Shuffler**
over UCXX, which rides CUDA-IPC / NVLink intra-node (see ``_UCX_ENV``).

Per sort (all device-resident work overlaps across the N ranks):

    H2D     pull assigned input blocks (Arrow, object store) -> one cuDF table
            per rank (RAM -> VRAM), and emit a tiny key sample.
    SORT    (VRAM -> VRAM, fully on device)
              * local ``cudf.sort_values`` by the sort key(s),
              * range-partition by global quantile boundaries
                (``pylibcudf.search.lower_bound`` -> ``split_and_pack``),
              * rapidsmpf Shuffler all-to-all over NVLink,
              * ``unpack_and_concat`` the received range + final local sort.
    D2H     sorted cuDF table per rank -> Arrow block back in the object store.

Output partition ``p`` holds the p-th global key range in sorted order, so the
concatenation of output blocks (ordered by partition id) is globally sorted --
for any schema, key set, sort direction and null placement.

This backend is selected by ``ds.sort(..., gpu=True)`` (or ``backend="gpu"``) or
by setting ``RAY_DATA_GPU_SORT=1`` in the environment; the CPU sort remains the
default otherwise.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import ray
from ray.data._internal.execution.interfaces import RefBundle, TaskContext
from ray.data.block import BlockAccessor

logger = logging.getLogger(__name__)

# Populated on the driver after each general GPU sort so a benchmark/driver can
# read the detailed phase timings and the on-GPU correctness summary (mirrors
# ``gpu_sort.LAST_RUN_STATS``). The Ray Data AllToAll transform fn runs on the
# driver, so this module global is visible to a driver-side benchmark.
LAST_RUN_STATS: dict = {}

# Stable, ordered names for the detached one-GPU sorter actors.
_ACTOR_PREFIX = "__rmpf_gpu_sorter__"

# Incremented per sort call so concurrent shufflers never collide on op-id
# (every rank is handed the *same* value for a given shuffle).
_OP_COUNTER = [0]


def actor_names(num_gpus: int) -> List[str]:
    return [f"{_ACTOR_PREFIX}{i}" for i in range(num_gpus)]


# --------------------------------------------------------------------------- #
# Cluster topology + resolution helpers
#
# These let the sort adapt to a multi-node cluster instead of the original
# single-DGX-2 assumptions: the rank count is derived from the cluster's GPU
# total (not hardcoded to 16), and device->host spill is gated on topology so a
# single node keeps the in-VRAM fast path while a multi-node run is OOM-safe.
# --------------------------------------------------------------------------- #
def _cluster_gpu_count() -> int:
    """Total number of GPUs visible in the cluster (best-effort, 0 if unknown)."""
    try:
        from ray.data._internal.execution.operators.hash_shuffle import (
            _get_total_cluster_resources,
        )

        return int(_get_total_cluster_resources().gpu or 0)
    except Exception:
        try:
            return int(ray.cluster_resources().get("GPU", 0))
        except Exception:
            return 0


def _cluster_node_count() -> int:
    """Number of alive nodes exposing GPU resources (best-effort, >=1)."""
    try:
        nodes = [
            n
            for n in ray.nodes()
            if n.get("Alive", False)
            and float(n.get("Resources", {}).get("GPU", 0)) > 0
        ]
        return max(1, len(nodes))
    except Exception:
        return 1


def resolve_num_gpus() -> int:
    """Resolve the number of one-GPU sort ranks to use.

    Precedence: ``RAY_DATA_GPU_SORT_NUM_GPUS`` (explicit override) else the
    cluster's total GPU count. Every rank must join the shuffle, so requesting
    more ranks than the cluster has GPUs would hang forever -- we raise instead.
    """
    cluster = _cluster_gpu_count()
    env = os.environ.get("RAY_DATA_GPU_SORT_NUM_GPUS")
    if env is not None:
        want = max(1, int(env))
    else:
        want = max(1, cluster)
    if cluster > 0 and want > cluster:
        raise ValueError(
            f"GPU sort requested {want} rank(s) but the cluster only exposes "
            f"{cluster} GPU(s). Lower RAY_DATA_GPU_SORT_NUM_GPUS (or unset it to "
            f"auto-detect) or add GPUs to the cluster."
        )
    return want


# Default device-memory limit (as a fraction of total VRAM) when spill is
# auto-enabled on a multi-node cluster. Mirrors the gpu_shuffle backend's
# ``"auto"`` (spill once device use approaches the pool cap).
_AUTO_SPILL_FRAC = 0.80
_SPILL_OFF_TOKENS = {"", "none", "off", "0", "disable", "disabled", "false", "no"}


def _resolve_spill_frac() -> Optional[float]:
    """Topology-gated device->host spill threshold (fraction of VRAM).

    Returns ``None`` to disable spilling. Precedence:
      * ``RAY_DATA_GPU_SORT_SPILL_FRAC`` set -> explicit (a recognized "off"
        token disables; otherwise the float is used), regardless of topology.
      * unset -> spill OFF on a single node (preserve the in-VRAM fast path that
        the DGX-2 numbers were measured with), auto-ON across >1 node so a large
        per-rank share spills to host instead of a hard cudaMalloc OOM.
    """
    env = os.environ.get("RAY_DATA_GPU_SORT_SPILL_FRAC")
    if env is not None:
        if env.strip().lower() in _SPILL_OFF_TOKENS:
            return None
        return float(env)
    return _AUTO_SPILL_FRAC if _cluster_node_count() > 1 else None


def _plc_order_null(ascending, na_position):
    """Map (ascending, na_position) to per-column libcudf (Order, NullOrder).

    Chosen so null placement matches Ray's CPU sort, i.e. pyarrow
    ``sort_indices`` with ``null_placement="at_end"`` (the default): nulls land
    at the END of the final order regardless of the column's direction
    (``na_position="last"``), or at the START for ``"first"``. Empirically
    ASCENDING+AFTER and DESCENDING+BEFORE both put nulls last; the reverse puts
    them first. ``cudf.sort_values`` can only express one global ``na_position``,
    so we drive the cuDF sort *kernel* directly to get this per-column behavior.
    """
    import pylibcudf as plc

    order = [
        plc.types.Order.ASCENDING if a else plc.types.Order.DESCENDING
        for a in ascending
    ]
    if na_position == "first":
        null_prec = [
            plc.types.NullOrder.BEFORE if a else plc.types.NullOrder.AFTER
            for a in ascending
        ]
    else:
        null_prec = [
            plc.types.NullOrder.AFTER if a else plc.types.NullOrder.BEFORE
            for a in ascending
        ]
    return order, null_prec


# --------------------------------------------------------------------------- #
# UCX transport config: force the bulk shuffle data onto CUDA-IPC / NVLink.
#
# cuda_ipc  -> peer GPU<->GPU copies over the NVSwitch fabric (the fast path),
# cuda_copy -> register/stage CUDA buffers,
# sm        -> intra-node host shared memory,
# tcp       -> bootstrap / control plane (and a last-resort fallback).
# Without cuda_ipc the shuffle would fall back to host staging and crater
# toward the ~46 GiB/s PCIe wall instead of the NVLink fabric.
#
# Cluster / cross-node note: cuda_ipc and sm are INTRA-NODE only. Across nodes
# the data falls back to ``tcp`` (correct, but ENA/TCP-bound). To use RDMA on EC2
# EFA-capable instances, override the transport list, e.g.
#   RAY_DATA_GPU_SORT_UCX_TLS=cuda_copy,cuda_ipc,sm,rc,ud,tcp   (IB/RoCE verbs)
# plus the appropriate UCX/libfabric EFA provider env. This is left pluggable but
# is NOT validated here (requires real EFA hardware); the default keeps the
# intra-node fast path and a TCP cross-node fallback.
# --------------------------------------------------------------------------- #
def _ucx_env() -> Dict[str, str]:
    return {
        "UCX_TLS": os.environ.get("RAY_DATA_GPU_SORT_UCX_TLS", "cuda_copy,cuda_ipc,sm,tcp"),
        "UCX_SOCKADDR_TLS_PRIORITY": "tcp",
        # cudf/rmm under Ray: keep the memtype cache off (recommended for UCX +
        # CUDA so device-pointer type lookups stay correct after pool reuse).
        "UCX_MEMTYPE_CACHE": "n",
        "RAPIDSMPF_LOG": os.environ.get("RAPIDSMPF_LOG", "WARN"),
    }


def _build_actor_class():
    """Define the one-GPU sorter actor class (lazily, to keep heavy RAPIDS
    imports off the driver until the GPU path is actually used)."""
    from rapidsmpf.utils.ray_utils import BaseShufflingActor

    @ray.remote(num_gpus=1, num_cpus=4)
    class _GeneralGpuSorter(BaseShufflingActor):
        """Owns ONE GPU; holds its slice of the dataset as a cuDF table and
        cooperates in a rapidsmpf range-partition shuffle + local sort."""

        def __init__(self, nranks: int, index: int, spill_frac: Optional[float] = None):
            super().__init__(nranks)
            # Make sure UCX picks NVLink even if the actor's runtime_env didn't
            # carry the vars (defensive; runtime_env is the primary path).
            for k, v in _ucx_env().items():
                os.environ.setdefault(k, v)
            self.index = index
            # Topology-gated device->host spill threshold (fraction of VRAM),
            # resolved on the driver: None disables spilling (single-node fast
            # path), a float enables it (multi-node OOM-safety).
            self._spill_frac = spill_frac
            self.df = None
            self.col_names: Optional[List[str]] = None
            self.schema_in = None  # pyarrow schema of the input blocks
            self.br = None
            self.stats = None
            self._mr = None
            self.last_stats: dict = {}
            # Sorted output blocks (host Arrow) held here until the executor
            # pulls each one via emit_block(); kept off ray.put so the EXECUTOR,
            # not this GPU actor, owns the output ObjectRefs.
            self._out_blocks: dict = {}

        # -- one-time per-process device setup ---------------------------- #
        def setup_worker(self, root_address_bytes: bytes) -> None:
            super().setup_worker(root_address_bytes)  # UCXX comm + default br
            import rmm
            from rapidsmpf.memory.buffer_resource import (
                BufferResource,
                LimitAvailableMemory,
            )
            from rapidsmpf.memory.buffer import MemoryType
            from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor
            from rapidsmpf.statistics import Statistics

            # A pool keeps the copy-heavy partition/shuffle/concat peaks off the
            # cudaMalloc critical path. Cap below total VRAM and (optionally)
            # spill device->host past a threshold so a transient peak can't OOM.
            total = rmm.mr.available_device_memory()[1]
            frac_max = float(os.environ.get("RAY_DATA_GPU_SORT_POOL_FRAC", "0.80"))
            pool_max = (int(total * frac_max) // 256) * 256
            pool_init = (int(total * 0.5) // 256) * 256
            # Remember the device total + pool cap so mem_stats()/the per-sort
            # stats can report headroom (peak vs cap vs total) for the memory-
            # pressure analysis -- this is the biggest dataset we sort.
            self._total_vram = int(total)
            self._pool_max = int(pool_max)
            self._mr = RmmResourceAdaptor(
                rmm.mr.PoolMemoryResource(
                    rmm.mr.CudaMemoryResource(),
                    initial_pool_size=pool_init,
                    maximum_pool_size=pool_max,
                )
            )
            rmm.mr.set_current_device_resource(self._mr)

            # Spill is gated on topology by the driver (see _resolve_spill_frac):
            # off single-node, on across nodes. An explicit env still wins.
            spill_frac = self._spill_frac
            mem_available = None
            if spill_frac is not None:
                limit = int(total * float(spill_frac))
                mem_available = {
                    MemoryType.DEVICE: LimitAvailableMemory(self._mr, limit=limit)
                }
            self.stats = Statistics(enable=False, mr=self._mr)
            self.br = BufferResource(self._mr, memory_available=mem_available)

            # One-time, per-rank transport/placement log so cluster runs are
            # diagnosable (which UCX transports were selected, and where this
            # rank landed). cuda_ipc/sm are intra-node only; cross-node falls to
            # tcp unless an RDMA/EFA transport is added via RAY_DATA_GPU_SORT_UCX_TLS.
            try:
                logger.info(
                    "GPU sort rank %d ready on node %s: UCX_TLS=%s spill=%s",
                    self.index,
                    ray.util.get_node_ip_address(),
                    os.environ.get("UCX_TLS"),
                    "off" if spill_frac is None else f"{spill_frac:g} of VRAM",
                )
            except Exception:
                pass

        def is_ready(self) -> bool:
            return self.is_initialized() and self.br is not None

        # -- phase: H2D (RAM -> VRAM) + key sample ------------------------ #
        def load(self, block_refs, key_cols, sample_size, schema):
            import cudf
            import cupy as cp

            t0 = time.perf_counter()
            # Canonical input Arrow schema (broadcast from the driver). Used to
            # cast every output block back so all blocks agree -- e.g. an
            # all-null string partition would otherwise serialize as Arrow type
            # `null`. Also lets a rank with no assigned blocks build an empty
            # table of the right shape so it can still join the shuffle (every
            # one of the ``nranks`` ranks must participate).
            self.schema_in = getattr(schema, "base_schema", schema)
            tables = ray.get(list(block_refs))  # zero-copy from object store
            if len(tables) == 0:
                df = cudf.DataFrame.from_arrow(self.schema_in.empty_table())
            else:
                parts = [cudf.DataFrame.from_arrow(t) for t in tables]  # H2D
                df = (parts[0] if len(parts) == 1
                      else cudf.concat(parts, ignore_index=True))
                del parts
            self.df = df
            self.col_names = list(df.columns)
            n = len(df)
            stride = max(1, n // max(1, sample_size))
            # reset_index(drop=True): a strided ``iloc[::stride]`` leaves a
            # non-range cuDF index that ``to_arrow()`` would serialize as an
            # extra ``index`` column. Ranks with different strides (uneven row
            # counts) would then emit samples with mismatched schemas and the
            # driver's ``pa.concat_tables(samples)`` would fail. Dropping the
            # index keeps every rank's sample exactly ``key_cols``.
            sample = df[key_cols].iloc[::stride].reset_index(drop=True).to_arrow()
            cp.cuda.runtime.deviceSynchronize()
            return sample, n, time.perf_counter() - t0

        # -- global quantile boundaries (runs on one rank) --------------- #
        def compute_boundaries(self, sample_table, key_cols, ascending,
                               na_position, nparts):
            import cudf
            import pylibcudf as plc
            from rapidsmpf.utils.cudf import (
                cudf_to_pylibcudf_table,
                pylibcudf_to_cudf_dataframe,
            )

            # Sort the gathered key sample with the SAME per-column order/null
            # precedence the range partition uses, so the chosen boundary rows
            # are monotonic under that ordering (required for a correct range
            # partition -> globally sorted concatenation).
            s = cudf.DataFrame.from_arrow(sample_table)[key_cols]
            order, null_prec = _plc_order_null(ascending, na_position)
            keys = cudf_to_pylibcudf_table(s)
            sorted_tbl = plc.sorting.sort_by_key(keys, keys, order, null_prec)
            s = pylibcudf_to_cudf_dataframe(sorted_tbl, key_cols)
            m = len(s)
            if m == 0 or nparts <= 1:
                return s.iloc[0:0].to_arrow()
            # nparts-1 evenly spaced interior rows -> nparts key ranges.
            idx = [int((j * m) // nparts) for j in range(1, nparts)]
            idx = [min(max(i, 0), m - 1) for i in idx]
            return s.iloc[idx].reset_index(drop=True).to_arrow()

        # -- phase: SORT (VRAM -> VRAM) + D2H ----------------------------- #
        def sort_partition(self, boundaries_table, key_cols, ascending,
                           na_position, op_id):
            import cudf
            import cupy as cp
            import pylibcudf as plc
            from rmm.pylibrmm.stream import DEFAULT_STREAM
            from rapidsmpf.integrations.cudf.partition import (
                split_and_pack,
                unpack_and_concat,
                unspill_partitions,
            )
            from rapidsmpf.utils.cudf import (
                cudf_to_pylibcudf_table,
                pylibcudf_to_cudf_dataframe,
            )

            # Device bytes currently resident before any sort work == this
            # rank's H2D-loaded input partition (the "per-rank partition size").
            resident_bytes = (
                int(self._mr.current_allocated) if self._mr is not None else 0
            )

            nranks = self.nranks()
            names = self.col_names
            key_idx = [names.index(k) for k in key_cols]
            order, null_prec = _plc_order_null(ascending, na_position)

            def _sort_by_keys(table):
                keys = plc.Table([table.columns()[i] for i in key_idx])
                return plc.sorting.sort_by_key(table, keys, order, null_prec)

            # ---- local sort by the key(s) using cuDF's sort kernel ----
            cp.cuda.runtime.deviceSynchronize()
            ts0 = time.perf_counter()
            full = cudf_to_pylibcudf_table(self.df)
            ds = _sort_by_keys(full)  # sorted copy of the whole table
            del full
            self.df = None

            # ---- range-partition by global boundaries ----
            bnd = cudf.DataFrame.from_arrow(boundaries_table)
            n_rows = ds.num_rows()
            if len(bnd) == 0:
                splits = []
            else:
                ds_keys = plc.Table([ds.columns()[i] for i in key_idx])
                bnd_keys = cudf_to_pylibcudf_table(bnd[key_cols])
                splits_col = plc.search.lower_bound(
                    ds_keys, bnd_keys, order, null_prec
                )
                splits = (
                    cudf.Series.from_pylibcudf(splits_col).to_numpy().tolist()
                )
            # pad/truncate to exactly nranks-1 interior splits (empty parts ok)
            splits = [int(min(max(s, 0), n_rows)) for s in splits]
            while len(splits) < nranks - 1:
                splits.append(n_rows)
            splits = splits[: nranks - 1]
            packed = split_and_pack(ds, splits, DEFAULT_STREAM, self.br)
            del ds
            cp.cuda.runtime.deviceSynchronize()
            ts1 = time.perf_counter()

            # ---- rapidsmpf all-to-all shuffle (over UCXX / NVLink) ----
            shuffler = self.create_shuffler(
                op_id, total_num_partitions=nranks,
                buffer_resource=self.br, statistics=self.stats,
            )
            shuffler.insert_chunks(packed)
            shuffler.insert_finished(list(range(nranks)))
            outputs: Dict[int, "plc.Table"] = {}
            while not shuffler.finished():
                pid = shuffler.wait_any()
                chunks = shuffler.extract(pid)
                tbl = unpack_and_concat(
                    unspill_partitions(
                        chunks, br=self.br, allow_overbooking=True,
                        statistics=self.stats,
                    ),
                    DEFAULT_STREAM,
                    self.br,
                )
                outputs[pid] = tbl
            shuffler.shutdown()
            cp.cuda.runtime.deviceSynchronize()
            ts2 = time.perf_counter()

            # ---- final sort of each received range (merges the N sorted
            # runs that landed in this partition) ----
            out_meta = []
            sorted_dfs: Dict[int, "cudf.DataFrame"] = {}
            for pid, tbl in outputs.items():
                sorted_tbl = _sort_by_keys(tbl)
                sorted_dfs[pid] = pylibcudf_to_cudf_dataframe(sorted_tbl, names)
            cp.cuda.runtime.deviceSynchronize()
            ts3 = time.perf_counter()

            # ---- D2H: sorted cuDF -> Arrow, held on the actor ----
            # Crucially we do NOT ``ray.put`` here: that would make this GPU
            # actor the *owner* of the output blocks, so the actor could not be
            # torn down (to free its GPU) without invalidating them
            # (OwnerDiedError). Instead each block is returned later by
            # emit_block() as a task RESULT, so the executor (the caller) owns
            # it -- the Ray-native ownership for an all-to-all op.
            self._out_blocks = {}
            for pid in sorted(sorted_dfs):
                df = sorted_dfs[pid]
                tbl = df.to_arrow()
                # Normalize to the input schema so all output blocks agree
                # (e.g. an all-null string partition -> proper `string` type).
                if self.schema_in is not None and not tbl.schema.equals(self.schema_in):
                    tbl = tbl.cast(self.schema_in)
                meta = BlockAccessor.for_block(tbl).get_metadata()
                self._out_blocks[int(pid)] = tbl
                out_meta.append({
                    "pid": int(pid),
                    "meta": meta,
                    "nrows": int(len(df)),
                })
            cp.cuda.runtime.deviceSynchronize()
            ts4 = time.perf_counter()

            schema = None
            if out_meta and self.schema_in is not None:
                # Canonical (input) schema -- all output blocks are cast to it.
                schema = BlockAccessor.for_block(self.schema_in.empty_table()).schema()

            # Per-rank memory-pressure snapshot. ``peak_vram_bytes`` is the RMM
            # adaptor's lifetime high-water mark (resets only when the actor /
            # pool is recreated), so across reused trials it is the worst-case
            # peak -- the conservative number to report for "did it fit?".
            peak_bytes = (
                int(self._mr.get_main_record().peak()) if self._mr is not None else 0
            )
            cur_bytes = (
                int(self._mr.current_allocated) if self._mr is not None else 0
            )
            stats = {
                "local_sort_s": ts1 - ts0,
                "shuffle_s": ts2 - ts1,
                "final_sort_s": ts3 - ts2,
                "gpu_only_s": ts3 - ts0,   # VRAM->sorted-VRAM (excl. final D2H)
                "d2h_s": ts4 - ts3,
                "rows_out": sum(m["nrows"] for m in out_meta),
                "rows_in_rank": int(n_rows),
                "resident_bytes": resident_bytes,
                "peak_vram_bytes": peak_bytes,
                "current_vram_bytes": cur_bytes,
                "pool_max_bytes": int(getattr(self, "_pool_max", 0)),
                "total_vram_bytes": int(getattr(self, "_total_vram", 0)),
                "spill_frac": self._spill_frac,
            }
            self.last_stats = stats
            return out_meta, schema, stats

        def emit_block(self, pid: int):
            """Return the sorted output block for ``pid`` as this task's return
            value, so the *caller* (the executor running the sort transform fn)
            becomes the owner of the resulting ObjectRef. The bytes live in this
            node's object store and survive this actor being torn down -- which
            is what lets the sort release its GPUs while the sorted dataset stays
            valid. The block is dropped from the actor once emitted to free host
            memory.
            """
            return self._out_blocks.pop(int(pid))

        def get_last_stats(self) -> dict:
            return self.last_stats

        def mem_stats(self) -> dict:
            """Current device-memory snapshot for this rank (peak/current vs the
            RMM pool cap and the device total), plus the resolved spill setting.
            Used by the benchmark to report per-rank VRAM pressure / headroom and
            whether device spill is even enabled. Safe to call any time after
            ``setup_worker``; ``get_main_record().peak()`` is the lifetime
            high-water mark."""
            if self._mr is None:
                return {"index": self.index, "ready": False}
            rec = self._mr.get_main_record()
            return {
                "index": self.index,
                "ready": True,
                "peak_vram_bytes": int(rec.peak()),
                "current_vram_bytes": int(self._mr.current_allocated),
                "pool_max_bytes": int(getattr(self, "_pool_max", 0)),
                "total_vram_bytes": int(getattr(self, "_total_vram", 0)),
                "spill_frac": self._spill_frac,
            }

        def release(self) -> None:
            self.df = None
            self._out_blocks = {}

    return _GeneralGpuSorter


# Cache the actor class so repeated planning reuses one definition.
_ACTOR_CLASS = [None]


def _get_actor_class():
    if _ACTOR_CLASS[0] is None:
        _ACTOR_CLASS[0] = _build_actor_class()
    return _ACTOR_CLASS[0]


def _get_actor_pool(num_gpus: int):
    """Get-or-create the detached one-GPU sorter actors and ensure the UCXX
    cluster is set up exactly once (reused across trials within a process)."""
    cls = _get_actor_class()
    names = actor_names(num_gpus)
    env = {"env_vars": _ucx_env()}
    # Topology-gated spill, resolved here on the driver and passed into each
    # actor (so the actor process doesn't need to re-inspect the cluster).
    spill_frac = _resolve_spill_frac()
    actors = [
        cls.options(
            name=names[i],
            get_if_exists=True,
            lifetime="detached",
            namespace="rmpf_gpu_sort",
            runtime_env=env,
            num_gpus=1,
            # SPREAD so single-GPU nodes each get exactly one rank and
            # multi-GPU nodes pack one rank per GPU -- required for a cluster.
            scheduling_strategy="SPREAD",
        ).remote(num_gpus, i, spill_frac)
        for i in range(num_gpus)
    ]
    ready = ray.get([a.is_ready.remote() for a in actors])
    if not all(ready):
        # Fresh cluster: elect actors[0] as root, connect all workers. Bound the
        # UCXX handshake so a mis-sized cluster / bad network fails loudly
        # instead of hanging forever.
        timeout_s = float(os.environ.get("RAY_DATA_GPU_SORT_SETUP_TIMEOUT_S", "120"))
        try:
            _, root_addr = ray.get(
                actors[0].setup_root.remote(), timeout=timeout_s
            )
            ray.get(
                [a.setup_worker.remote(root_addr) for a in actors],
                timeout=timeout_s,
            )
        except ray.exceptions.GetTimeoutError as e:
            raise TimeoutError(
                f"GPU sort UCXX setup did not complete within {timeout_s}s "
                f"across {num_gpus} rank(s). Check that all {num_gpus} GPUs are "
                f"schedulable, GPU/network health, and UCX transport settings "
                f"(RAY_DATA_GPU_SORT_UCX_TLS). Set "
                f"RAY_DATA_GPU_SORT_SETUP_TIMEOUT_S to extend the deadline."
            ) from e
    return actors


def kill_actor_pool(num_gpus: int) -> None:
    for name in actor_names(num_gpus):
        try:
            ray.kill(ray.get_actor(name, namespace="rmpf_gpu_sort"))
        except Exception:
            pass


def release_gpu_sort_pool(num_gpus: int, timeout_s: float = 30.0) -> bool:
    """Tear down the detached GPU sorter actors and wait (bounded) for their
    GPUs to be reclaimed, so a downstream op can use them.

    This is only safe once the sort's output blocks have been emitted as task
    results (the executor owns them; the bytes live in node-local plasma and
    survive the actors going away). Returns ``True`` if the GPUs came back
    within ``timeout_s``.
    """
    before = ray.available_resources().get("GPU", 0.0)
    kill_actor_pool(num_gpus)
    # Killing is async: the raylet reclaims the GPUs once the actor processes
    # actually exit. Poll until availability rises by ~num_gpus (the pool size).
    target = before + num_gpus - 0.5
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if ray.available_resources().get("GPU", 0.0) >= target:
            return True
        time.sleep(0.1)
    return False


# --------------------------------------------------------------------------- #
# Ray Data transform fn (the ds.sort() hook), general backend.
# --------------------------------------------------------------------------- #
def generate_gpu_sort_general_fn(sort_key, data_context, num_gpus: Optional[int] = None):
    """Return an AllToAllTransformFn that sorts all input blocks across GPUs
    using cuDF (columnar) + a rapidsmpf range-partition shuffle.

    ``num_gpus`` is the number of one-GPU ranks. When ``None`` it is derived
    from the cluster's GPU total (see :func:`resolve_num_gpus`) so the sort
    scales from a single DGX-2 to a multi-node cluster without a hardcoded count.
    """
    import pyarrow as pa

    if num_gpus is None:
        num_gpus = resolve_num_gpus()

    key_cols = list(sort_key.get_columns())
    descending = list(sort_key.get_descending())
    ascending = [not d for d in descending]
    na_position = os.environ.get("RAY_DATA_GPU_SORT_NA_POSITION", "last")
    sample_size = int(os.environ.get("RAY_DATA_GPU_SORT_SAMPLE", str(1 << 16)))

    def fn(refs: List[RefBundle], ctx: TaskContext) -> Tuple[List[RefBundle], dict]:
        block_refs = []
        schema_in = None
        for rb in refs:
            schema_in = schema_in or rb.schema
            block_refs.extend(rb.block_refs)
        if len(block_refs) == 0:
            return refs, {"GPUSortGeneral": []}

        # Use ALL ranks: the UCXX communicator/Shuffler spans every one of the
        # ``num_gpus`` ranks, so each must participate even if it gets no input
        # blocks (it then sorts/partitions an empty table).
        n = num_gpus
        actors = _get_actor_pool(num_gpus)

        t0 = time.perf_counter()
        # H2D: assign blocks round-robin and load + sample on each rank.
        assign = [block_refs[i::n] for i in range(n)]
        load_res = ray.get(
            [actors[i].load.remote(assign[i], key_cols, sample_size, schema_in)
             for i in range(n)]
        )
        samples = [r[0] for r in load_res]
        in_rows = sum(r[1] for r in load_res)
        h2d_s = max(r[2] for r in load_res)
        t1 = time.perf_counter()

        # Global boundaries from the concatenated key sample (on one rank).
        sample_all = pa.concat_tables(samples) if len(samples) > 1 else samples[0]
        boundaries = ray.get(
            actors[0].compute_boundaries.remote(
                sample_all, key_cols, ascending, na_position, n
            )
        )
        t2 = time.perf_counter()

        # SORT + shuffle + D2H on every rank in parallel.
        _OP_COUNTER[0] += 1
        op_id = _OP_COUNTER[0] % 256
        part_res = ray.get(
            [
                actors[i].sort_partition.remote(
                    boundaries, key_cols, ascending, na_position, op_id
                )
                for i in range(n)
            ]
        )
        t3 = time.perf_counter()

        # Collect (rank, pid, meta) from all ranks; output blocks ordered by
        # partition id == global key-range order -> concatenation is sorted.
        entries = []
        schema_out = None
        for rank, (out_meta, schema, _stats) in enumerate(part_res):
            schema_out = schema_out or schema
            for m in out_meta:
                entries.append({"rank": rank, **m})
        entries.sort(key=lambda e: e["pid"])

        # Pull each sorted block out of its producing GPU actor as a task
        # RESULT, so the EXECUTOR (this process, the caller) owns the resulting
        # ObjectRef -- not the GPU actor. The bytes live in node-local plasma and
        # survive the actor being torn down, exactly like a normal Ray Data task
        # output. ``ray.wait`` (without fetching the bytes here) ensures every
        # block is materialized before the caller may release the GPU actors.
        block_refs = [actors[e["rank"]].emit_block.remote(e["pid"]) for e in entries]
        if block_refs:
            ray.wait(block_refs, num_returns=len(block_refs), fetch_local=False)
        t4 = time.perf_counter()

        out = []
        stats_list = []
        rows_out = 0
        block_rows = []
        for e, block_ref in zip(entries, block_refs):
            stats_list.append(e["meta"].to_stats())
            rows_out += e["nrows"]
            block_rows.append(e["nrows"])
            out.append(
                RefBundle(
                    [(block_ref, e["meta"])], owns_blocks=True, schema=schema_out
                )
            )

        # Aggregate per-phase timings (ranks barrier at the shuffle, so the
        # wall of each phase is ~max across ranks).
        def _max(key):
            return max(s[key] for _, _, s in part_res)

        def _collect(key):
            return [s.get(key) for _, _, s in part_res]

        def _first(key, default=None):
            for _, _, s in part_res:
                if key in s:
                    return s[key]
            return default

        peak_list = [p for p in _collect("peak_vram_bytes") if p is not None]
        LAST_RUN_STATS.clear()
        LAST_RUN_STATS.update({
            "h2d_s": h2d_s,
            "boundaries_s": t2 - t1,
            "local_sort_s": _max("local_sort_s"),
            "shuffle_s": _max("shuffle_s"),
            "final_sort_s": _max("final_sort_s"),
            "gpu_only_s": _max("gpu_only_s"),
            "d2h_s": _max("d2h_s"),
            "emit_s": t4 - t3,            # pull blocks to the executor + ray.wait
            "full_s": t4 - t0,            # RAM->VRAM->sort->VRAM->RAM (+emit)
            "wall_fn_s": t4 - t0,
            "rows_in": in_rows,
            "rows_out": rows_out,
            "num_output_blocks": len(out),
            "block_rows": block_rows,
            "num_gpus": n,
            # Memory-pressure snapshot (per-rank + aggregate), so a driver/bench
            # can report whether the per-rank partition fit in VRAM and the
            # headroom vs the RMM pool cap -- no extra ray.get of the actors,
            # these ride back on the sort_partition result before any release.
            "peak_vram_bytes_per_rank": _collect("peak_vram_bytes"),
            "peak_vram_bytes_max": max(peak_list, default=0),
            "resident_bytes_per_rank": _collect("resident_bytes"),
            "rows_in_per_rank": _collect("rows_in_rank"),
            "pool_max_bytes": _first("pool_max_bytes", 0),
            "total_vram_bytes": _first("total_vram_bytes", 0),
            "spill_frac": _first("spill_frac", None),
        })

        # Opt-in: release the GPU sorter actors now so a downstream GPU op
        # (e.g. an encoder) can use all the GPUs. Safe because the output
        # blocks were emitted as task results above and ray.wait confirmed they
        # are materialized -- they are owned by the executor, not these actors,
        # so they survive the teardown. Default OFF preserves the across-trials
        # pool reuse the sort microbenchmark relies on.
        if os.environ.get("RAY_DATA_GPU_SORT_RELEASE") == "1":
            release_gpu_sort_pool(n)

        return out, {"GPUSortGeneral": stats_list}

    return fn
