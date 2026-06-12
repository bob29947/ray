"""Shared GPU helpers for the experimental, opt-in GPU ordinal encoder.

This module is the foundation for
:class:`~ray.data.preprocessors.GpuOrdinalEncoder`. It mirrors the host-staged
design of the experimental GPU sort
(``ray.data._internal.planner.gpu_sort_general``): each batch is pulled from the
object store as an Arrow block (RAM), moved to a GPU as a cuDF frame, processed
on the device, and written back as an Arrow block (RAM). The CPU path remains
the default; everything here is reached only from ``GpuOrdinalEncoder``.

Design notes (these are the levers that make the host-staged path a win and not
a transfer-bound wash):

* **Only the needed columns cross the bus.** Transforms convert just the
  operator's input columns to cuDF and re-attach the outputs to the original
  Arrow table, so payload columns never round-trip host<->device.
* **The CUDA/cuDF context is initialized once per worker**, not per batch, via a
  stateful :class:`_GpuBatchActor` map_batches UDF. Per-batch GPU tasks would
  re-pay context/import cost on every block.
* **The heavy fit reduction (unique values) runs on GPU workers**; only the
  small per-block result is merged on the driver (with pyarrow, so no driver GPU
  is required).

All RAPIDS imports are deferred to call time so importing Ray (or this module)
never requires cuDF/CUDA on a CPU-only install.
"""

from __future__ import annotations

import math
import os
from collections import namedtuple
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import pyarrow as pa

    from ray.data.dataset import Dataset


# Per-column numeric moments produced by :func:`gpu_mean_std`. ``mean`` and
# ``std`` are the convenient population statistics over the **non-null** values
# (``std`` uses ``n - ddof`` in the denominator, matching the CPU
# ``StandardScaler``). The raw Welford components (``n`` non-null count, ``m2``
# non-null sum of squared deviations, ``n_total`` rows incl. nulls) are also
# returned so a fused fit can derive scaler-after-impute stats analytically
# (``std_imputed = sqrt(m2 / n_total)``) without materializing imputed data.
MeanStd = namedtuple("MeanStd", ["mean", "std", "n", "m2", "n_total"])


def env_num_gpus(default: int = 1) -> int:
    """Number of one-GPU workers to use for the GPU preprocessors.

    ``RAY_DATA_GPU_PREPROC_NUM_GPUS`` is an explicit override. When unset, the
    concurrency defaults to the cluster's total GPU count so the preprocessors
    scale across a multi-node cluster instead of pinning to a single GPU (the
    old default of 1). Falls back to ``default`` if the cluster size is unknown.
    """
    env = os.environ.get("RAY_DATA_GPU_PREPROC_NUM_GPUS")
    if env is not None:
        return max(1, int(env))
    try:
        import ray

        gpus = int(ray.cluster_resources().get("GPU", 0))
        if gpus > 0:
            return gpus
    except Exception:
        pass
    return default


def env_batch_size(default: int = 1 << 20) -> int:
    """Per-worker batch size (``RAY_DATA_GPU_PREPROC_BATCH_SIZE``).

    ``map_batches`` requires an explicit ``batch_size`` whenever ``num_gpus`` is
    set, so the GPU preprocessors always pass one.
    """
    return int(os.environ.get("RAY_DATA_GPU_PREPROC_BATCH_SIZE", default))


# Auto GPU block (device batch) sizing. The fused pass holds the input columns,
# the produced columns, and cuDF intermediates on the device at once, so the
# per-batch device working set scales with ``rows * bytes_per_row``. We pick a
# rows-per-batch that (a) keeps the working set to a safe fraction of per-GPU
# VRAM (headroom for the RMM pool + future transfer/compute overlap) and (b) is
# small enough that every GPU gets several batches (load balancing). All knobs
# are env-overridable; the user can always pin an explicit batch size.
_L4_VRAM_BYTES = 24 * 1024 ** 3
_AUTO_MIN_ROWS = 256 * 1024
_AUTO_MAX_ROWS = 4 * 1024 * 1024


def per_gpu_vram_bytes(default: int = _L4_VRAM_BYTES) -> int:
    """Total bytes of VRAM on the visible GPU; falls back to ``default`` (L4).

    Queried on whatever process calls this (the driver, which on the GPU cluster
    is itself a GPU node). Best-effort across cupy / rmm; never raises.

    ``RAY_DATA_GPU_PREPROC_VRAM_BYTES`` is an explicit override (bytes). It lets a
    box with large GPUs (e.g. a 32 GB V100) emulate a smaller cluster GPU (e.g. a
    24 GB L4) so :func:`auto_gpu_block_rows` derives a cluster-representative
    device batch during local benchmarking.
    """
    env = os.environ.get("RAY_DATA_GPU_PREPROC_VRAM_BYTES")
    if env:
        try:
            v = int(float(env))
            if v > 0:
                return v
        except (TypeError, ValueError):
            pass
    try:
        import cupy

        _free, total = cupy.cuda.Device().mem_info
        if total and total > 0:
            return int(total)
    except Exception:
        pass
    try:
        import rmm

        info = rmm.mr.available_device_memory()
        if isinstance(info, (tuple, list)) and len(info) == 2 and info[1]:
            return int(info[1])
    except Exception:
        pass
    return int(default)


# --------------------------------------------------------------------------- #
# Optional per-worker tuning knobs (opt-in; default off so behaviour is
# unchanged unless explicitly enabled). Each is read at call time so the
# benchmark / pipeline can A/B them via the environment.
#
#   RAY_DATA_GPU_PREPROC_RMM_POOL=1
#       Install a pooled RMM allocator on each GPU worker. By default cuDF
#       services every allocation (each intermediate column, each op output) via
#       cudaMalloc/cudaFree -- synchronous driver calls paid per op, per block.
#       A pool pre-allocates a slab once and sub-allocates from it, removing that
#       per-op overhead from BOTH the fit reductions and the fused transform.
#       The pool starts small and grows on demand (so it never grabs more VRAM
#       than the non-pooled peak), capped at RMM_MAX_FRACTION of the device.
# --------------------------------------------------------------------------- #
_RMM_POOL_READY = False


def env_rmm_pool() -> bool:
    """Whether to install a pooled RMM allocator on each GPU worker."""
    return os.environ.get("RAY_DATA_GPU_PREPROC_RMM_POOL", "0") == "1"


def env_fit_single_pass() -> bool:
    """Single-pass driver merge for the fit reductions (default ON).

    The per-block unique / value-count partials are concatenated and merged on
    the driver. The single-pass path collapses the old per-column ``filter +
    unique`` (one full scan of the concatenated partials *per column*) into ONE
    ``group_by`` over all columns at once -- strictly fewer passes for >= 2 fit
    columns, equal-or-lower peak memory, and bit-identical results (uniques keep
    nulls so the encoder can still reproduce raise-on-null; value counts already
    drop nulls per block). Set ``RAY_DATA_GPU_PREPROC_FIT_SINGLE_PASS=0`` to
    restore the per-column path -- a removable escape hatch for a one-time
    null-parity / pyarrow-version check, NOT a performance knob.
    """
    return os.environ.get("RAY_DATA_GPU_PREPROC_FIT_SINGLE_PASS", "1") != "0"


def ensure_rmm_pool() -> None:
    """Install a pooled RMM allocator once per worker process (idempotent).

    No-op when ``RAY_DATA_GPU_PREPROC_RMM_POOL`` is unset, when RMM is missing, or
    after the first call (a process-global flag keeps it ~free per block). Never
    raises -- a pool is a pure speedup, so any failure silently leaves the
    default (cudaMalloc-per-op) allocator in place.
    """
    global _RMM_POOL_READY
    if _RMM_POOL_READY:
        return
    # Set the flag first: even on failure we must not retry on every block.
    _RMM_POOL_READY = True
    if not env_rmm_pool():
        return
    def _envf(name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, default))
        except (TypeError, ValueError):
            return default

    try:
        import rmm

        total = per_gpu_vram_bytes()
        init_frac = _envf("RAY_DATA_GPU_PREPROC_RMM_INIT_FRACTION", 0.10)
        max_frac = _envf("RAY_DATA_GPU_PREPROC_RMM_MAX_FRACTION", 0.80)
        init_bytes = max(1 << 28, int(total * max(0.0, min(init_frac, max_frac))))
        max_bytes = int(total * max(init_frac, min(max_frac, 1.0)))
        # 256-byte aligned, as RMM expects.
        init_bytes -= init_bytes % 256
        max_bytes -= max_bytes % 256
        rmm.reinitialize(
            pool_allocator=True,
            initial_pool_size=init_bytes,
            maximum_pool_size=max_bytes,
        )
    except Exception:
        pass


def auto_gpu_block_rows(
    ds: "Dataset",
    *,
    num_gpus: Optional[int] = None,
    total_rows: Optional[int] = None,
    bytes_per_row: Optional[float] = None,
) -> int:
    """Pick a VRAM-aware rows-per-batch for the fused GPU op.

    The result is clamped to ``[_AUTO_MIN_ROWS, _AUTO_MAX_ROWS]`` and is the
    minimum of:

    * **VRAM budget** -- ``vram_fraction`` of per-GPU VRAM divided by the
      estimated per-row device working set (``peak_factor * bytes_per_row``).
    * **Load-balance cap** -- ``total_rows / (num_gpus * min_blocks_per_gpu)``.
      With the default ``min_blocks_per_gpu = 1`` this is ``rows / num_gpus`` (one
      block per GPU), which never *exceeds* what keeps every GPU busy. Local
      sweeps (DGX V100) show fused throughput improves monotonically with batch
      size up to ~one-block-per-GPU and then flattens (bigger only idles GPUs),
      so the win is to be VRAM-bounded, not to over-split: e.g. 14.7M rows / 8
      GPUs went 0.46M->2.0M batch = 3.0x->5.4x vs the CPU subtotal. Raise
      ``min_blocks_per_gpu`` only if you want more, smaller blocks (finer
      load-balancing across uneven nodes).

    Env overrides: ``RAY_DATA_GPU_PREPROC_VRAM_FRACTION`` (default 0.15),
    ``RAY_DATA_GPU_PREPROC_PEAK_FACTOR`` (default 3.0),
    ``RAY_DATA_GPU_PREPROC_MIN_BLOCKS_PER_GPU`` (default 1). The VRAM budget keeps
    it safe on a 24 GB L4 and shrinks automatically for the wider ``--feature-set
    wide`` frame (more bytes/row -> fewer rows/batch).
    """

    def _envf(name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, default))
        except (TypeError, ValueError):
            return default

    vram_fraction = _envf("RAY_DATA_GPU_PREPROC_VRAM_FRACTION", 0.15)
    peak_factor = max(1.0, _envf("RAY_DATA_GPU_PREPROC_PEAK_FACTOR", 3.0))
    min_blocks_per_gpu = max(1, int(_envf("RAY_DATA_GPU_PREPROC_MIN_BLOCKS_PER_GPU", 1)))

    if num_gpus is None:
        num_gpus = env_num_gpus()
    num_gpus = max(1, int(num_gpus))

    # bytes/row from the (materialized) dataset, best-effort.
    if bytes_per_row is None:
        try:
            rows = total_rows if total_rows is not None else ds.count()
            size = ds.size_bytes() or 0
            bytes_per_row = (size / rows) if rows else 0.0
        except Exception:
            bytes_per_row = 0.0
    if total_rows is None:
        try:
            total_rows = ds.count()
        except Exception:
            total_rows = 0

    # VRAM-budget rows.
    if bytes_per_row and bytes_per_row > 0:
        budget = per_gpu_vram_bytes() * vram_fraction
        rows_vram = int(budget / (peak_factor * bytes_per_row))
    else:
        rows_vram = _AUTO_MAX_ROWS

    # Load-balance rows (enough batches to feed every GPU).
    if total_rows and total_rows > 0:
        rows_balance = int(total_rows // (num_gpus * min_blocks_per_gpu)) or total_rows
    else:
        rows_balance = _AUTO_MAX_ROWS

    rows = min(rows_vram, rows_balance, _AUTO_MAX_ROWS)
    rows = max(_AUTO_MIN_ROWS, rows)
    # Round down to a clean 64K multiple for readability/stability.
    rows = max(_AUTO_MIN_ROWS, (rows // (64 * 1024)) * (64 * 1024))
    return int(rows)


def gpu_available() -> bool:
    """Return ``True`` if cuDF and a usable CUDA device are importable here.

    Used by the ``Gpu*`` preprocessors to transparently fall back to the CPU
    implementation (a faithful drop-in) when no GPU/RAPIDS stack is present.
    """
    try:
        import cudf  # noqa: F401
        import rmm

        # Raises if there is no CUDA device visible to this process.
        rmm.mr.available_device_memory()
        return True
    except Exception:
        return False


def attach_arrow_columns(
    table: "pa.Table", new_columns: Dict[str, "pa.Array"]
) -> "pa.Table":
    """Return ``table`` with ``new_columns`` added or replaced in place.

    Columns not in ``new_columns`` are left untouched on the host -- they are
    never converted to cuDF, so payload columns do not cross the bus.
    """
    import pyarrow as pa

    for name, arr in new_columns.items():
        if isinstance(arr, pa.ChunkedArray):
            arr = arr.combine_chunks()
        idx = table.schema.get_field_index(name)
        if idx == -1:
            table = table.append_column(name, arr)
        else:
            table = table.set_column(idx, name, arr)
    return table


class _GpuBatchActor:
    """Stateful ``map_batches`` UDF for host-staged GPU transforms.

    ``build_state`` runs once per worker (e.g. to move a fitted vocabulary to the
    device); ``apply_fn(state, arrow_batch) -> arrow_batch`` runs per batch. Both
    are cloudpickled with whatever (small) fitted state they close over.
    """

    def __init__(
        self,
        build_state: Callable[[], Any],
        apply_fn: Callable[[Any, "pa.Table"], "pa.Table"],
    ):
        # Touch cuDF here so the CUDA context + RMM initialization is paid once,
        # at actor construction, instead of on the first batch. Installing the
        # pooled allocator (if enabled) here means every subsequent device
        # allocation in this actor's transforms is served from the pool.
        import cudf  # noqa: F401

        ensure_rmm_pool()
        self._apply = apply_fn
        self._state = build_state()

    def __call__(self, batch: "pa.Table") -> "pa.Table":
        return self._apply(self._state, batch)


def gpu_transform(
    ds: "Dataset",
    *,
    build_state: Callable[[], Any],
    apply_fn: Callable[[Any, "pa.Table"], "pa.Table"],
    batch_size: Optional[int] = None,
    concurrency: Optional[int] = None,
) -> "Dataset":
    """Run a host-staged GPU transform over ``ds`` using a persistent actor pool.

    Each actor owns one GPU (``num_gpus=1``); ``concurrency`` actors run in
    parallel (defaults to :func:`env_num_gpus`).
    """
    bs = batch_size if batch_size is not None else env_batch_size()
    conc = concurrency if concurrency is not None else env_num_gpus()
    return ds.map_batches(
        _GpuBatchActor,
        fn_constructor_kwargs={"build_state": build_state, "apply_fn": apply_fn},
        batch_format="pyarrow",
        zero_copy_batch=True,
        num_gpus=1,
        batch_size=bs,
        concurrency=conc,
    )


def _group_columns_by_type(
    ds: "Dataset", columns: List[str]
) -> Dict[str, List[str]]:
    """Group columns by dtype so same-dtype columns can share one GPU pass.

    Fusing columns into a single ``map_batches`` pass avoids re-paying the
    actor-pool / CUDA-context startup once per column (the dominant cost of the
    fit on a typical handful of categorical columns).
    """
    try:
        schema = ds.schema()
        types = dict(zip(schema.names, schema.types))
    except Exception:
        types = {}
    groups: Dict[str, List[str]] = {}
    for col in columns:
        groups.setdefault(str(types.get(col, "unknown")), []).append(col)
    return groups


def _combine(array):
    import pyarrow as pa

    return array.combine_chunks() if isinstance(array, pa.ChunkedArray) else array


def from_arrow_robust(table: "pa.Table"):
    """``cudf.DataFrame.from_arrow`` with a retry for the cudf 26.02 host quirk.

    cudf 26.02's ``from_arrow_host`` can raise ``cudaErrorInvalidValue`` on
    certain **sliced / non-zero-offset** Arrow buffers -- which is exactly what
    Ray Data hands a zero-copy ``map_batches`` UDF when a block is a slice of a
    larger materialized block. An Arrow IPC round-trip rematerializes the values
    into contiguous, offset-0 buffers that convert cleanly, so we retry once that
    way (and fall back to a pandas round-trip as a last resort). The fast path is
    unchanged; the rematerialization only runs on the (intermittent) failure.
    """
    import cudf

    # Install the pooled allocator once per worker (no-op unless enabled). This
    # is the single H2D entry point for BOTH the fit reductions and the fused
    # transform, so it is the natural place to guarantee the pool is ready.
    ensure_rmm_pool()

    try:
        return cudf.DataFrame.from_arrow(table)
    except RuntimeError:
        try:
            compact = _ipc_from_bytes(_ipc_to_bytes(table))
            return cudf.DataFrame.from_arrow(compact)
        except RuntimeError:
            return cudf.DataFrame.from_pandas(table.to_pandas())


def _ordered_unique(items: List[str]) -> List[str]:
    """Stable de-duplication preserving first-seen order."""
    seen = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def gpu_unique_values(
    ds: "Dataset",
    columns: List[str],
    *,
    batch_size: Optional[int] = None,
    concurrency: Optional[int] = None,
) -> Dict[str, "pa.Array"]:
    """Global per-column set of unique values, computed on GPU.

    Same-dtype columns are processed in a **single** ``map_batches`` pass (the
    per-block distinct values are emitted in long ``(__col, value)`` form), then
    de-duplicated per column on the driver with pyarrow. Nulls are **kept** so
    callers can reproduce the CPU encoders' "raise on null" behavior; the
    returned order is unspecified (callers sort as needed).

    Returns ``{column: pyarrow.Array}`` of the global unique values.
    """
    import pyarrow as pa

    bs = batch_size if batch_size is not None else env_batch_size()
    conc = concurrency if concurrency is not None else env_num_gpus()
    out: Dict[str, "pa.Array"] = {}

    for group in _group_columns_by_type(ds, columns).values():

        def per_block(batch: "pa.Table", _group: List[str] = group) -> "pa.Table":
            import cudf
            import pyarrow as pa

            gdf = from_arrow_robust(batch.select(_group))
            names, values = [], []
            for col in _group:
                uniq = _combine(gdf[col].unique().to_arrow())
                names.append(pa.array([col] * len(uniq)))
                values.append(uniq)
            return pa.table(
                {"__col": pa.concat_arrays(names), "value": pa.concat_arrays(values)}
            )

        partials = ds.map_batches(
            per_block,
            batch_format="pyarrow",
            zero_copy_batch=True,
            num_gpus=1,
            batch_size=bs,
            concurrency=conc,
        )
        tables = list(
            partials.iter_batches(batch_format="pyarrow", batch_size=None)
        )
        merged = pa.concat_tables(tables) if tables else None
        out.update(_merge_uniques_table(merged, group))

    return out


def gpu_sum_count(
    ds: "Dataset",
    columns: List[str],
    *,
    batch_size: Optional[int] = None,
    concurrency: Optional[int] = None,
) -> Dict[str, Tuple[float, int]]:
    """Global per-column ``(sum, count)`` over **non-null** values, on GPU.

    Each block is converted to cuDF once and every column's partial
    ``(sum, count)`` is emitted in long ``(__col, sum, count)`` form; the driver
    sums the partials with pyarrow (no driver GPU required). Used by the
    standalone ``GpuSimpleImputer`` ``"mean"`` strategy, where the fill value is
    ``sum / count``.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    bs = batch_size if batch_size is not None else env_batch_size()
    conc = concurrency if concurrency is not None else env_num_gpus()
    cols = list(columns)

    def per_block(batch: "pa.Table", _cols: List[str] = cols) -> "pa.Table":
        import cudf
        import pyarrow as pa

        gdf = from_arrow_robust(batch.select(_cols))
        names, sums, counts = [], [], []
        for col in _cols:
            series = gdf[col]
            n = int(series.count())
            names.append(col)
            counts.append(n)
            sums.append(float(series.sum()) if n else 0.0)
        return pa.table(
            {
                "__col": pa.array(names, pa.string()),
                "sum": pa.array(sums, pa.float64()),
                "count": pa.array(counts, pa.int64()),
            }
        )

    partials = ds.map_batches(
        per_block,
        batch_format="pyarrow",
        zero_copy_batch=True,
        num_gpus=1,
        batch_size=bs,
        concurrency=conc,
    )
    tables = list(partials.iter_batches(batch_format="pyarrow", batch_size=None))
    merged = pa.concat_tables(tables) if tables else None

    out: Dict[str, Tuple[float, int]] = {}
    for col in columns:
        if merged is None or merged.num_rows == 0:
            out[col] = (0.0, 0)
            continue
        sub = merged.filter(pc.equal(merged.column("__col"), col))
        total_sum = float(pc.sum(sub.column("sum")).as_py() or 0.0)
        total_count = int(pc.sum(sub.column("count")).as_py() or 0)
        out[col] = (total_sum, total_count)
    return out


def gpu_mean_std(
    ds: "Dataset",
    columns: List[str],
    *,
    ddof: int = 0,
    batch_size: Optional[int] = None,
    concurrency: Optional[int] = None,
) -> Dict[str, "MeanStd"]:
    """Global per-column numeric moments, computed on GPU with a Welford merge.

    Each block emits ``(n, mean, m2, n_total)`` per column (``n``/``mean``/``m2``
    over non-null values via cuDF's stable per-block variance; ``n_total`` is the
    block row count incl. nulls). The driver combines block partials with the
    parallel (Chan) algorithm. Returns ``{col: MeanStd(mean, std, n, m2,
    n_total)}`` where ``std`` uses ``n - ddof`` (population std for ``ddof=0``,
    matching the CPU ``StandardScaler``). ``m2`` and ``n_total`` let a fused fit
    derive the scaler-after-impute std as ``sqrt(m2 / n_total)``.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    bs = batch_size if batch_size is not None else env_batch_size()
    conc = concurrency if concurrency is not None else env_num_gpus()
    cols = list(columns)

    def per_block(batch: "pa.Table", _cols: List[str] = cols) -> "pa.Table":
        import cudf
        import pyarrow as pa

        gdf = from_arrow_robust(batch.select(_cols))
        n_total = batch.num_rows
        names, ns, means, m2s, ntots = [], [], [], [], []
        for col in _cols:
            series = gdf[col]
            n = int(series.count())
            names.append(col)
            ns.append(n)
            ntots.append(n_total)
            if n > 0:
                mean = float(series.mean())
                # Population variance over non-null values; m2 = var * n.
                var = float(series.var(ddof=0)) if n > 1 else 0.0
                means.append(mean)
                m2s.append(var * n)
            else:
                means.append(float("nan"))
                m2s.append(0.0)
        return pa.table(
            {
                "__col": pa.array(names, pa.string()),
                "n": pa.array(ns, pa.int64()),
                "mean": pa.array(means, pa.float64()),
                "m2": pa.array(m2s, pa.float64()),
                "n_total": pa.array(ntots, pa.int64()),
            }
        )

    partials = ds.map_batches(
        per_block,
        batch_format="pyarrow",
        zero_copy_batch=True,
        num_gpus=1,
        batch_size=bs,
        concurrency=conc,
    )
    tables = list(partials.iter_batches(batch_format="pyarrow", batch_size=None))
    merged = pa.concat_tables(tables) if tables else None

    out: Dict[str, "MeanStd"] = {}
    for col in columns:
        if merged is None or merged.num_rows == 0:
            out[col] = MeanStd(float("nan"), float("nan"), 0, 0.0, 0)
            continue
        sub = merged.filter(pc.equal(merged.column("__col"), col))
        ns = sub.column("n").to_pylist()
        means = sub.column("mean").to_pylist()
        m2s = sub.column("m2").to_pylist()
        ntots = sub.column("n_total").to_pylist()

        n_agg, mean_agg, m2_agg, ntot_agg = 0, 0.0, 0.0, 0
        for n_b, mean_b, m2_b, nt_b in zip(ns, means, m2s, ntots):
            ntot_agg += int(nt_b or 0)
            if not n_b:
                continue
            if n_agg == 0:
                n_agg, mean_agg, m2_agg = int(n_b), float(mean_b), float(m2_b)
            else:
                delta = float(mean_b) - mean_agg
                new_n = n_agg + int(n_b)
                mean_agg += delta * int(n_b) / new_n
                m2_agg += float(m2_b) + delta * delta * n_agg * int(n_b) / new_n
                n_agg = new_n

        if n_agg - ddof > 0:
            std = math.sqrt(m2_agg / (n_agg - ddof))
        else:
            std = float("nan")
        mean_final = mean_agg if n_agg > 0 else float("nan")
        out[col] = MeanStd(mean_final, std, n_agg, m2_agg, ntot_agg)
    return out


def gpu_value_counts(
    ds: "Dataset",
    columns: List[str],
    *,
    batch_size: Optional[int] = None,
    concurrency: Optional[int] = None,
) -> Dict[str, "pa.Table"]:
    """Global per-column value counts (non-null), computed on GPU.

    Same-dtype columns share one ``map_batches`` pass; per-block cuDF
    ``value_counts`` partials are emitted in long ``(__col, value, count)`` form
    and summed per value on the driver with pyarrow. Returns
    ``{col: pyarrow.Table}`` with columns ``value`` and ``count`` (int64). Used
    by the ``GpuSimpleImputer`` ``"most_frequent"`` strategy (mode = max count,
    ties broken by the smallest value).
    """
    import pyarrow as pa

    bs = batch_size if batch_size is not None else env_batch_size()
    conc = concurrency if concurrency is not None else env_num_gpus()
    out: Dict[str, "pa.Table"] = {}

    for group in _group_columns_by_type(ds, columns).values():

        def per_block(batch: "pa.Table", _group: List[str] = group) -> "pa.Table":
            import cudf
            import pyarrow as pa

            gdf = from_arrow_robust(batch.select(_group))
            names, values, counts = [], [], []
            for col in _group:
                vc = gdf[col].value_counts(dropna=True)
                vals = _combine(vc.index.to_arrow())
                cnts = _combine(vc.to_arrow())
                names.append(pa.array([col] * len(vals), pa.string()))
                values.append(vals)
                counts.append(cnts)
            if values:
                value_arr = pa.concat_arrays([v.cast(values[0].type) for v in values])
                name_arr = pa.concat_arrays(names)
                count_arr = pa.concat_arrays([c.cast(pa.int64()) for c in counts])
            else:
                value_arr = pa.array([])
                name_arr = pa.array([], pa.string())
                count_arr = pa.array([], pa.int64())
            return pa.table(
                {"__col": name_arr, "value": value_arr, "count": count_arr}
            )

        partials = ds.map_batches(
            per_block,
            batch_format="pyarrow",
            zero_copy_batch=True,
            num_gpus=1,
            batch_size=bs,
            concurrency=conc,
        )
        tables = list(
            partials.iter_batches(batch_format="pyarrow", batch_size=None)
        )
        merged = pa.concat_tables(tables) if tables else None
        out.update(_merge_value_counts_table(merged, group))
    return out


# --------------------------------------------------------------------------- #
# One-scan fused fit reductions.
#
# ``fused_fit`` needs up to three global reductions (numeric moments, categorical
# unique values, and value counts for most_frequent imputation). Run separately
# (:func:`gpu_mean_std` / :func:`gpu_unique_values` / :func:`gpu_value_counts`)
# each is its own actor-pool ``map_batches`` pass that re-reads every block and
# re-pays ``cudf.DataFrame.from_arrow`` -- i.e. >=3 full GPU scans of the dataset.
# Empirically the fit dominates the fused stage (per the benchmark, ~80% of the
# fused wall), so collapsing those scans into ONE is the single biggest lever.
#
# :func:`gpu_fused_reductions` does exactly one pass: per block it converts the
# union of the needed columns to cuDF once, computes all three reductions, and
# emits each partial as an Arrow-IPC blob tagged by ``(kind, group)``. The driver
# concatenates per tag and runs the SAME merge helpers the per-kind functions use
# -- so the fitted ``stats_`` are identical (parity preserved); only the number
# of device scans changes. The IPC packing keeps heterogeneous value dtypes
# (a string vocabulary vs an int vocabulary) type-safe inside one output block.
# --------------------------------------------------------------------------- #
def _moments_partial(gdf, cols: List[str], n_total: int) -> "pa.Table":
    """Per-block numeric moments (matches :func:`gpu_mean_std`'s per_block)."""
    import pyarrow as pa

    names, ns, means, m2s, ntots = [], [], [], [], []
    for col in cols:
        series = gdf[col]
        n = int(series.count())
        names.append(col)
        ns.append(n)
        ntots.append(n_total)
        if n > 0:
            means.append(float(series.mean()))
            var = float(series.var(ddof=0)) if n > 1 else 0.0
            m2s.append(var * n)
        else:
            means.append(float("nan"))
            m2s.append(0.0)
    return pa.table(
        {
            "__col": pa.array(names, pa.string()),
            "n": pa.array(ns, pa.int64()),
            "mean": pa.array(means, pa.float64()),
            "m2": pa.array(m2s, pa.float64()),
            "n_total": pa.array(ntots, pa.int64()),
        }
    )


def _merge_moments_table(
    merged: Optional["pa.Table"], columns: List[str], ddof: int = 0
) -> Dict[str, "MeanStd"]:
    """Welford-merge per-block moment partials (matches :func:`gpu_mean_std`)."""
    import pyarrow.compute as pc

    out: Dict[str, "MeanStd"] = {}
    for col in columns:
        if merged is None or merged.num_rows == 0:
            out[col] = MeanStd(float("nan"), float("nan"), 0, 0.0, 0)
            continue
        sub = merged.filter(pc.equal(merged.column("__col"), col))
        ns = sub.column("n").to_pylist()
        means = sub.column("mean").to_pylist()
        m2s = sub.column("m2").to_pylist()
        ntots = sub.column("n_total").to_pylist()

        n_agg, mean_agg, m2_agg, ntot_agg = 0, 0.0, 0.0, 0
        for n_b, mean_b, m2_b, nt_b in zip(ns, means, m2s, ntots):
            ntot_agg += int(nt_b or 0)
            if not n_b:
                continue
            if n_agg == 0:
                n_agg, mean_agg, m2_agg = int(n_b), float(mean_b), float(m2_b)
            else:
                delta = float(mean_b) - mean_agg
                new_n = n_agg + int(n_b)
                mean_agg += delta * int(n_b) / new_n
                m2_agg += float(m2_b) + delta * delta * n_agg * int(n_b) / new_n
                n_agg = new_n

        if n_agg - ddof > 0:
            std = math.sqrt(m2_agg / (n_agg - ddof))
        else:
            std = float("nan")
        mean_final = mean_agg if n_agg > 0 else float("nan")
        out[col] = MeanStd(mean_final, std, n_agg, m2_agg, ntot_agg)
    return out


def _uniques_partial(gdf, group: List[str]) -> "pa.Table":
    """Per-block unique values (matches :func:`gpu_unique_values`'s per_block)."""
    import pyarrow as pa

    names, values = [], []
    for col in group:
        uniq = _combine(gdf[col].unique().to_arrow())
        names.append(pa.array([col] * len(uniq)))
        values.append(uniq)
    return pa.table(
        {"__col": pa.concat_arrays(names), "value": pa.concat_arrays(values)}
    )


def _merge_uniques_table(
    merged: Optional["pa.Table"], group: List[str]
) -> Dict[str, "pa.Array"]:
    """De-duplicate per-block unique partials (matches :func:`gpu_unique_values`).

    Single pass (default): one ``group_by(["__col", "value"])`` over the
    concatenated partials yields the distinct ``(col, value)`` pairs in a single
    scan -- a null value forms its own group, kept exactly as ``pc.unique`` would
    (the encoder needs it to reproduce raise-on-null). The per-column
    ``filter + pc.unique`` (one full scan per column) is the escape hatch under
    ``RAY_DATA_GPU_PREPROC_FIT_SINGLE_PASS=0``. Both produce identical sets.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    if merged is None or merged.num_rows == 0:
        return {col: pa.array([]) for col in group}

    if env_fit_single_pass():
        distinct = merged.group_by(["__col", "value"]).aggregate([])
        col_arr = distinct.column("__col")
        val_arr = distinct.column("value")
        return {
            col: _combine(val_arr.filter(pc.equal(col_arr, col))) for col in group
        }

    out: Dict[str, "pa.Array"] = {}
    for col in group:
        sub = merged.filter(pc.equal(merged.column("__col"), col))
        out[col] = pc.unique(_combine(sub.column("value")))
    return out


def _value_counts_partial(gdf, group: List[str]) -> "pa.Table":
    """Per-block value counts (matches :func:`gpu_value_counts`'s per_block)."""
    import pyarrow as pa

    names, values, counts = [], [], []
    for col in group:
        vc = gdf[col].value_counts(dropna=True)
        vals = _combine(vc.index.to_arrow())
        cnts = _combine(vc.to_arrow())
        names.append(pa.array([col] * len(vals), pa.string()))
        values.append(vals)
        counts.append(cnts)
    if values:
        value_arr = pa.concat_arrays([v.cast(values[0].type) for v in values])
        name_arr = pa.concat_arrays(names)
        count_arr = pa.concat_arrays([c.cast(pa.int64()) for c in counts])
    else:
        value_arr = pa.array([])
        name_arr = pa.array([], pa.string())
        count_arr = pa.array([], pa.int64())
    return pa.table({"__col": name_arr, "value": value_arr, "count": count_arr})


def _merge_value_counts_table(
    merged: Optional["pa.Table"], group: List[str]
) -> Dict[str, "pa.Table"]:
    """Sum per-block value-count partials (matches :func:`gpu_value_counts`).

    Single pass (default): one ``group_by(["__col", "value"]).sum(count)`` over
    the concatenated partials sums counts for every column at once (nulls were
    already dropped per block via ``value_counts(dropna=True)``). The per-column
    path (one scan per column) is the escape hatch under
    ``RAY_DATA_GPU_PREPROC_FIT_SINGLE_PASS=0``. Both produce identical counts.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    empty = pa.table({"value": pa.array([]), "count": pa.array([], pa.int64())})
    if merged is None or merged.num_rows == 0:
        return {col: empty for col in group}

    if env_fit_single_pass():
        summed = merged.group_by(["__col", "value"]).aggregate([("count", "sum")])
        col_arr = summed.column("__col")
        val_arr = summed.column("value")
        cnt_arr = summed.column("count_sum")
        out: Dict[str, "pa.Table"] = {}
        for col in group:
            mask = pc.equal(col_arr, col)
            vals = _combine(val_arr.filter(mask))
            if len(vals) == 0:
                out[col] = empty
                continue
            out[col] = pa.table(
                {
                    "value": vals,
                    "count": _combine(cnt_arr.filter(mask)).cast(pa.int64()),
                }
            )
        return out

    out: Dict[str, "pa.Table"] = {}
    for col in group:
        sub = merged.filter(pc.equal(merged.column("__col"), col))
        if sub.num_rows == 0:
            out[col] = empty
            continue
        grouped = (
            pa.table(
                {
                    "value": _combine(sub.column("value")),
                    "count": _combine(sub.column("count")),
                }
            )
            .group_by("value")
            .aggregate([("count", "sum")])
        )
        out[col] = pa.table(
            {
                "value": grouped.column("value"),
                "count": grouped.column("count_sum").cast(pa.int64()),
            }
        )
    return out


def _ipc_to_bytes(table: "pa.Table") -> bytes:
    """Serialize a (small) Arrow table to IPC stream bytes."""
    import pyarrow as pa

    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def _ipc_from_bytes(blob: bytes) -> "pa.Table":
    """Deserialize an Arrow table from IPC stream bytes."""
    import pyarrow as pa

    with pa.ipc.open_stream(pa.BufferReader(blob)) as reader:
        return reader.read_all()


def gpu_fused_reductions(
    ds: "Dataset",
    *,
    moment_cols: Optional[List[str]] = None,
    unique_cols: Optional[List[str]] = None,
    vc_cols: Optional[List[str]] = None,
    ddof: int = 0,
    batch_size: Optional[int] = None,
    concurrency: Optional[int] = None,
) -> Tuple[Dict[str, "MeanStd"], Dict[str, "pa.Array"], Dict[str, "pa.Table"]]:
    """Compute moments + uniques + value counts in ONE GPU pass over ``ds``.

    Returns ``(moments, uniques, value_counts)`` with the SAME contents the
    separate :func:`gpu_mean_std` / :func:`gpu_unique_values` /
    :func:`gpu_value_counts` would produce, but converting each block to cuDF
    exactly once. Same-dtype unique / value-count columns share a group (as in
    the per-kind functions); each per-block partial is shipped as an Arrow-IPC
    blob tagged ``(kind, group)`` so heterogeneous value dtypes stay type-safe.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    bs = batch_size if batch_size is not None else env_batch_size()
    conc = concurrency if concurrency is not None else env_num_gpus()

    moment_cols = _ordered_unique(list(moment_cols or []))
    unique_groups = (
        list(_group_columns_by_type(ds, list(unique_cols)).values())
        if unique_cols
        else []
    )
    vc_groups = (
        list(_group_columns_by_type(ds, list(vc_cols)).values()) if vc_cols else []
    )

    all_cols = _ordered_unique(
        list(moment_cols)
        + [c for g in unique_groups for c in g]
        + [c for g in vc_groups for c in g]
    )
    if not all_cols:
        return {}, {}, {}

    def per_block(
        batch: "pa.Table",
        _mom: List[str] = moment_cols,
        _ug: List[List[str]] = unique_groups,
        _vg: List[List[str]] = vc_groups,
        _all: List[str] = all_cols,
    ) -> "pa.Table":
        import cudf
        import pyarrow as pa

        present = [c for c in _all if c in batch.column_names]
        gdf = from_arrow_robust(batch.select(present))
        n_total = batch.num_rows
        kinds, groups, payloads = [], [], []
        if _mom:
            kinds.append("moments")
            groups.append(-1)
            payloads.append(_ipc_to_bytes(_moments_partial(gdf, _mom, n_total)))
        for gi, grp in enumerate(_ug):
            kinds.append("uniques")
            groups.append(gi)
            payloads.append(_ipc_to_bytes(_uniques_partial(gdf, grp)))
        for gi, grp in enumerate(_vg):
            kinds.append("value_counts")
            groups.append(gi)
            payloads.append(_ipc_to_bytes(_value_counts_partial(gdf, grp)))
        return pa.table(
            {
                "__kind": pa.array(kinds, pa.string()),
                "__group": pa.array(groups, pa.int32()),
                "payload": pa.array(payloads, pa.binary()),
            }
        )

    partials = ds.map_batches(
        per_block,
        batch_format="pyarrow",
        zero_copy_batch=True,
        num_gpus=1,
        batch_size=bs,
        concurrency=conc,
    )
    tables = list(partials.iter_batches(batch_format="pyarrow", batch_size=None))
    merged = pa.concat_tables(tables) if tables else None

    def _collect(kind: str, group_idx: int) -> Optional["pa.Table"]:
        if merged is None or merged.num_rows == 0:
            return None
        sub = merged.filter(
            pc.and_(
                pc.equal(merged.column("__kind"), kind),
                pc.equal(merged.column("__group"), group_idx),
            )
        )
        if sub.num_rows == 0:
            return None
        subtables = [_ipc_from_bytes(p.as_py()) for p in sub.column("payload")]
        return pa.concat_tables(subtables) if subtables else None

    moments = (
        _merge_moments_table(_collect("moments", -1), moment_cols, ddof)
        if moment_cols
        else {}
    )
    uniques: Dict[str, "pa.Array"] = {}
    for gi, grp in enumerate(unique_groups):
        uniques.update(_merge_uniques_table(_collect("uniques", gi), grp))
    value_counts: Dict[str, "pa.Table"] = {}
    for gi, grp in enumerate(vc_groups):
        value_counts.update(_merge_value_counts_table(_collect("value_counts", gi), grp))

    return moments, uniques, value_counts
