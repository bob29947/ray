"""Device-resident fusion for GPU preprocessors (experimental, opt-in).

This module is the fusion seam for the GPU preprocessors. The standalone
``Gpu*`` operators (see ``gpu_encoder.py``, ``gpu_scaler.py``,
``gpu_imputer.py``) are *host-staged*: each one pulls a block as Arrow (RAM),
moves its input columns to the GPU, computes, and writes the result back as
Arrow. Run back-to-back, that pays the PCIe round trip once **per operator**.

The win here is **device residency**: when several fusable operators run in a
row (e.g. ``Chain(SimpleImputer, OrdinalEncoder, StandardScaler)``), a single
``map_batches`` pass crosses PCIe **once** -- H2D the union of the needed input
columns, thread the resident cuDF frame through each operator's device step in
order, then D2H the produced columns once. Impute and scale "ride" encode's
residency instead of each paying their own transfer (standalone they lose to a
vectorized CPU pass; the transfer exceeds the in-RAM work).

The extensibility contract is the :class:`DeviceFusable` mixin. A preprocessor
becomes fusable by implementing a handful of small hooks; the fused runner owns
all H2D/D2H. Adding a new GPU op is "implement the device hook + one registry
line" -- it then composes in any order inside a fused run.

All RAPIDS imports are deferred to call time, so importing this module never
requires cuDF/CUDA on a CPU-only install.
"""

from __future__ import annotations

import logging
import os
import time
from collections import namedtuple
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Opt-in coarse profiling of the fused transform's per-block H2D / compute / D2H
# split. When ``RAY_DATA_GPU_PREPROC_PROFILE`` is set, each worker accumulates
# wall time per phase and logs running totals (throttled) -- a lightweight aid
# for the device-residency benchmark; zero overhead when unset.
_PROFILE_ENV = "RAY_DATA_GPU_PREPROC_PROFILE"
# When ``RAY_DATA_GPU_PREPROC_PROFILE_DIR`` is also set, each worker additionally
# writes its cumulative ``{h2d, compute, d2h, n}`` totals to a per-worker JSON
# file in that directory (overwritten each block). A local benchmark driver can
# then read + sum the files after the pass to recover the device-time breakdown
# without scraping logs (works on a single node; per-node on a cluster).
_PROFILE_DIR_ENV = "RAY_DATA_GPU_PREPROC_PROFILE_DIR"

if TYPE_CHECKING:
    import cudf
    import pyarrow as pa

    from ray.data.dataset import Dataset
    from ray.data.preprocessor import Preprocessor


# A reduction a fusable op needs at fit time. ``kind`` is one of
# ``"moments"`` (numeric mean/std/M2 via :func:`_gpu.gpu_mean_std`),
# ``"uniques"`` (:func:`_gpu.gpu_unique_values`) or ``"value_counts"``
# (:func:`_gpu.gpu_value_counts`). The fused fit unions requests by kind and
# runs each shared reduction once.
FitRequest = namedtuple("FitRequest", ["kind", "columns"])


def _ordered_unique(items: List[str]) -> List[str]:
    """Stable de-duplication preserving first-seen order."""
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _safe_schema(ds: "Dataset"):
    """Best-effort ``ds.schema()`` (returns ``None`` if it can't be resolved)."""
    try:
        return ds.schema()
    except Exception:
        return None


class DeviceFusable:
    """Mixin contract for preprocessors that can run inside a fused device chain.

    A fusable operator declares the columns it reads/writes on the resident cuDF
    frame, how to build its device-side state once per worker, and how to apply
    itself to the frame in place. For fitting, it declares which shared
    reductions it needs and how to populate its ``stats_`` from the results
    (using the SAME ``stats_`` keys as its CPU base class, for drop-in parity).

    Concrete ``Gpu*`` operators implement the core hooks; the two
    dependency/capability hooks have sensible defaults.
    """

    # --- transform hooks ---------------------------------------------------- #
    def _device_input_columns(self) -> List[str]:
        """Columns this op needs present on-device (read)."""
        raise NotImplementedError

    def _device_output_columns(self) -> List[str]:
        """Columns this op writes/updates on-device."""
        raise NotImplementedError

    def _device_build_state(self) -> Any:
        """Build and return this op's device-side state.

        Runs **once per worker** (after the CUDA/cuDF context is initialized),
        e.g. to move a fitted vocabulary to the device as cuDF categoricals or
        to return scalar fitted stats. The result is passed to
        :meth:`_device_step` for every batch.
        """
        raise NotImplementedError

    def _device_step(self, state: Any, gdf: "cudf.DataFrame") -> "cudf.DataFrame":
        """Apply this op to the resident cuDF frame in place; return it.

        No H2D/D2H here -- the fused runner owns the transfers.
        """
        raise NotImplementedError

    # --- fit hooks ---------------------------------------------------------- #
    def _device_fit_requests(self) -> List[FitRequest]:
        """Declare the shared reductions this op needs at fit time."""
        return []

    def _device_set_fitted(self, ctx: "FusedFitContext") -> None:
        """Populate ``self.stats_`` from the fused-fit reductions in ``ctx``."""
        raise NotImplementedError

    # --- optional dependency / capability hooks ----------------------------- #
    def _device_null_fill_columns(self) -> Dict[str, str]:
        """Columns this op fills nulls for, mapped to the fill strategy.

        Imputers override this so downstream ops in the same fused run can ask
        (via :class:`FusedFitContext`) whether a column was already imputed --
        the basis for the impute/scale shared-stats fold. Non-imputers fill
        nothing.
        """
        return {}

    def _device_can_fuse(self, schema) -> bool:
        """Whether this op can run on-device for the given (input) schema.

        Defaults to ``True``. Ops that must fall back to CPU for certain
        schemas (e.g. the encoder on list-typed columns) override this; a
        ``False`` here makes :class:`~ray.data.preprocessors.Chain` break the
        fused run and execute the op via its normal (CPU-falling-back) path.
        """
        return True


def is_device_fusable(op: "Preprocessor") -> bool:
    """True if ``op`` implements the :class:`DeviceFusable` contract."""
    return isinstance(op, DeviceFusable)


class FusedFitContext:
    """Holds the shared fit reductions and answers cross-op dependency questions.

    Built once per fused run from the computed reductions and the ordered list
    of operators. Each op's :meth:`DeviceFusable._device_set_fitted` reads what
    it needs and may ask whether a column was imputed by an earlier op in the
    run (so a scaler can use the imputed-aware std denominator and an encoder
    can drop the now-absent null from its vocabulary).
    """

    def __init__(
        self,
        ops: List["DeviceFusable"],
        moments: Dict[str, Any],
        value_counts: Dict[str, "pa.Table"],
        uniques: Dict[str, "pa.Array"],
    ):
        self._ops = list(ops)
        self._moments = moments
        self._value_counts = value_counts
        self._uniques = uniques

    def moments(self, col: str):
        """``MeanStd`` for ``col`` (mean, std, n, m2, n_total)."""
        return self._moments[col]

    def uniques(self, col: str) -> "pa.Array":
        """Global unique values for ``col`` (nulls kept)."""
        return self._uniques[col]

    def mode(self, col: str):
        """Most frequent value of ``col`` (ties broken by the smallest value)."""
        import pyarrow.compute as pc

        tbl = self._value_counts.get(col)
        if tbl is None or tbl.num_rows == 0:
            return None
        counts = tbl.column("count")
        max_count = pc.max(counts).as_py()
        candidates = pc.filter(tbl.column("value"), pc.equal(counts, max_count))
        if len(candidates) == 0:
            return None
        return pc.min(candidates).as_py()

    def _fill_cols_before(self, op: "DeviceFusable", strategy: str) -> set:
        cols: set = set()
        for other in self._ops:
            if other is op:
                break
            if not is_device_fusable(other):
                continue
            for col, strat in other._device_null_fill_columns().items():
                if strat == strategy:
                    cols.add(col)
        return cols

    def mean_imputed_before(self, op: "DeviceFusable", col: str) -> bool:
        """True if ``col`` is mean-imputed by an op positioned before ``op``."""
        return col in self._fill_cols_before(op, "mean")

    def mode_imputed_before(self, op: "DeviceFusable", col: str) -> bool:
        """True if ``col`` is most_frequent-imputed by an op before ``op``."""
        return col in self._fill_cols_before(op, "most_frequent")


def fused_fit(
    ds: "Dataset",
    ops: List["DeviceFusable"],
    *,
    batch_size: Optional[int] = None,
    concurrency: Optional[int] = None,
) -> None:
    """Fit a run of fusable ops, sharing reductions across them.

    Unions every op's :meth:`DeviceFusable._device_fit_requests` by kind and
    runs each shared reduction once (so a mean-imputer and a scaler over the
    same column share a single moments scan -- the impute-mean = scaler-mean
    fold). Then each op populates its own ``stats_`` from the results, with
    dependency-aware resolution via :class:`FusedFitContext`.
    """
    from ray.data.preprocessors import _gpu

    moment_cols: List[str] = []
    unique_cols: List[str] = []
    vc_cols: List[str] = []
    for op in ops:
        for req in op._device_fit_requests():
            if req.kind == "moments":
                moment_cols += list(req.columns)
            elif req.kind == "uniques":
                unique_cols += list(req.columns)
            elif req.kind == "value_counts":
                vc_cols += list(req.columns)
            else:
                raise ValueError(f"Unknown fused-fit reduction kind: {req.kind!r}")

    moment_cols = _ordered_unique(moment_cols)
    unique_cols = _ordered_unique(unique_cols)
    vc_cols = _ordered_unique(vc_cols)

    # One-scan fit (default): compute all three reductions in a single device
    # pass (one ``from_arrow`` per block) instead of one full GPU scan per kind.
    # ``RAY_DATA_GPU_PREPROC_FUSED_FIT=0`` restores the legacy per-kind scans
    # (used by the benchmark harness to A/B the two paths). Both produce
    # identical ``stats_``.
    one_scan = os.environ.get("RAY_DATA_GPU_PREPROC_FUSED_FIT", "1") != "0"
    if one_scan and (moment_cols or unique_cols or vc_cols):
        moments, uniques, value_counts = _gpu.gpu_fused_reductions(
            ds,
            moment_cols=moment_cols,
            unique_cols=unique_cols,
            vc_cols=vc_cols,
            batch_size=batch_size,
            concurrency=concurrency,
        )
    else:
        moments = (
            _gpu.gpu_mean_std(
                ds, moment_cols, batch_size=batch_size, concurrency=concurrency
            )
            if moment_cols
            else {}
        )
        uniques = (
            _gpu.gpu_unique_values(
                ds, unique_cols, batch_size=batch_size, concurrency=concurrency
            )
            if unique_cols
            else {}
        )
        value_counts = (
            _gpu.gpu_value_counts(
                ds, vc_cols, batch_size=batch_size, concurrency=concurrency
            )
            if vc_cols
            else {}
        )

    ctx = FusedFitContext(ops, moments, value_counts, uniques)
    for op in ops:
        op._device_set_fitted(ctx)


def run_fused_device_transform(
    ds: "Dataset",
    ops: List["DeviceFusable"],
    *,
    batch_size: Optional[int] = None,
    concurrency: Optional[int] = None,
) -> "Dataset":
    """Transform ``ds`` by threading a resident cuDF frame through ``ops``.

    One persistent-actor ``map_batches`` pass: per block, H2D the union of the
    ops' input columns **once**, run each op's device step in order on the
    resident frame, then D2H the union of produced columns **once**. Payload
    columns (everything not produced by an op) never cross the bus.

    Works for a single op too -- the standalone ``Gpu*`` operators route their
    GPU transform through here with ``ops=[self]``.
    """
    from ray.data.preprocessors._gpu import attach_arrow_columns, gpu_transform

    ops = list(ops)
    in_cols = _ordered_unique([c for op in ops for c in op._device_input_columns()])
    out_cols = _ordered_unique(
        [c for op in ops for c in op._device_output_columns()]
    )

    profile = bool(os.environ.get(_PROFILE_ENV))
    profile_dir = os.environ.get(_PROFILE_DIR_ENV) if profile else None

    def build_state():
        # Runs once per worker (after CUDA init): build each op's device state.
        state = [op._device_build_state() for op in ops]
        if profile:
            # Per-worker cumulative phase wall (seconds) and block count.
            prof = {"h2d": 0.0, "compute": 0.0, "d2h": 0.0, "n": 0}
            if profile_dir:
                import uuid

                prof["_file"] = os.path.join(
                    profile_dir, f"prof_{os.getpid()}_{uuid.uuid4().hex[:8]}.json"
                )
            state.append(prof)
        return state

    def apply(states, batch: "pa.Table") -> "pa.Table":
        from ray.data.preprocessors._gpu import from_arrow_robust

        present = [c for c in in_cols if c in batch.column_names]
        if not profile:
            gdf = from_arrow_robust(batch.select(present))
            for op, state in zip(ops, states):
                gdf = op._device_step(state, gdf)
            new_columns = {c: gdf[c].to_arrow() for c in out_cols if c in gdf.columns}
            return attach_arrow_columns(batch, new_columns)

        prof = states[-1]
        op_states = states[:-1]
        t0 = time.perf_counter()
        gdf = from_arrow_robust(batch.select(present))
        t1 = time.perf_counter()
        for op, state in zip(ops, op_states):
            gdf = op._device_step(state, gdf)
        t2 = time.perf_counter()
        new_columns = {c: gdf[c].to_arrow() for c in out_cols if c in gdf.columns}
        t3 = time.perf_counter()
        prof["h2d"] += t1 - t0
        prof["compute"] += t2 - t1
        prof["d2h"] += t3 - t2
        prof["n"] += 1
        if prof["n"] % 16 == 0:
            logger.info(
                "fused device transform profile (worker cumulative, %d blocks): "
                "H2D=%.3fs compute=%.3fs D2H=%.3fs",
                prof["n"],
                prof["h2d"],
                prof["compute"],
                prof["d2h"],
            )
        prof_file = prof.get("_file")
        if prof_file:
            # Overwrite each block so the last write holds the full totals; the
            # driver reads these after the pass (cheap: one tiny JSON per worker).
            try:
                import json

                tmp = prof_file + ".tmp"
                with open(tmp, "w") as fh:
                    json.dump(
                        {
                            "h2d": prof["h2d"],
                            "compute": prof["compute"],
                            "d2h": prof["d2h"],
                            "n": prof["n"],
                        },
                        fh,
                    )
                os.replace(tmp, prof_file)
            except Exception:
                pass
        return attach_arrow_columns(batch, new_columns)

    return gpu_transform(
        ds,
        build_state=build_state,
        apply_fn=apply,
        batch_size=batch_size,
        concurrency=concurrency,
    )


def _upgrade_registry() -> Dict[type, Callable[["Preprocessor"], "DeviceFusable"]]:
    """Map a CPU preprocessor class -> a builder for its fusable GPU counterpart.

    Imports are deferred to call time so this module stays importable on a
    CPU-only install and to avoid import cycles with ``chain.py`` /
    ``__init__.py``. To make a new CPU preprocessor fusable, add a one-line
    entry here returning the matching ``Gpu*`` op built from the CPU op's
    public config.
    """
    from ray.data.preprocessors.encoder import OrdinalEncoder
    from ray.data.preprocessors.gpu_encoder import GpuOrdinalEncoder
    from ray.data.preprocessors.gpu_imputer import GpuSimpleImputer
    from ray.data.preprocessors.gpu_scaler import GpuStandardScaler
    from ray.data.preprocessors.imputer import SimpleImputer
    from ray.data.preprocessors.scaler import StandardScaler

    return {
        SimpleImputer: lambda o: GpuSimpleImputer(
            columns=o.columns,
            strategy=o.strategy,
            fill_value=o.fill_value,
            output_columns=o.output_columns,
        ),
        OrdinalEncoder: lambda o: GpuOrdinalEncoder(
            columns=o.columns,
            encode_lists=o.encode_lists,
            output_columns=o.output_columns,
        ),
        StandardScaler: lambda o: GpuStandardScaler(
            columns=o.columns,
            output_columns=o.output_columns,
        ),
    }


def upgrade_to_device_op(op: "Preprocessor") -> "Preprocessor":
    """Return the fusable GPU counterpart of ``op`` (or ``op`` unchanged).

    Already-fusable ops (the user passed a ``Gpu*`` op directly) are returned
    as-is. CPU ops with a registered GPU equivalent are rebuilt from their
    public config. Anything else is returned unchanged (it will break a fused
    run and execute on its normal path).
    """
    if is_device_fusable(op):
        return op
    builder = _upgrade_registry().get(type(op))
    if builder is None:
        return op
    return builder(op)
