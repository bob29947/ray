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

import os
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    import pyarrow as pa

    from ray.data.dataset import Dataset


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
        # at actor construction, instead of on the first batch.
        import cudf  # noqa: F401

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
    import pyarrow.compute as pc

    bs = batch_size if batch_size is not None else env_batch_size()
    conc = concurrency if concurrency is not None else env_num_gpus()
    out: Dict[str, "pa.Array"] = {}

    for group in _group_columns_by_type(ds, columns).values():

        def per_block(batch: "pa.Table", _group: List[str] = group) -> "pa.Table":
            import cudf
            import pyarrow as pa

            gdf = cudf.DataFrame.from_arrow(batch.select(_group))
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
        for col in group:
            if merged is None or merged.num_rows == 0:
                out[col] = pa.array([])
                continue
            sub = merged.filter(pc.equal(merged.column("__col"), col))
            out[col] = pc.unique(_combine(sub.column("value")))

    return out
