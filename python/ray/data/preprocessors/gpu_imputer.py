"""GPU-accelerated imputer for Ray Data (experimental, opt-in).

Host-staged drop-in for :class:`SimpleImputer`, mirroring
:mod:`ray.data.preprocessors.gpu_encoder` / ``gpu_scaler``: blocks stay
host-resident (Arrow); only the operator's columns move to the GPU (cuDF) for
the fit reduction / fill, then results are re-attached to the Arrow block. The
CPU imputer remains the default; opt in by using ``GpuSimpleImputer`` directly
or via a ``Chain(..., backend="gpu")`` that fuses it with neighbouring GPU ops.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List

from ray.data.preprocessors._gpu_fused import DeviceFusable, FitRequest
from ray.data.preprocessors.imputer import SimpleImputer
from ray.data.preprocessors.version_support import (
    SerializablePreprocessor as Serializable,
)
from ray.util.annotations import PublicAPI

if TYPE_CHECKING:
    import cudf

    from ray.data.dataset import Dataset

logger = logging.getLogger(__name__)


@PublicAPI(stability="alpha")
@Serializable(version=1, identifier="io.ray.preprocessors.gpu_simple_imputer")
class GpuSimpleImputer(SimpleImputer, DeviceFusable):
    """GPU-accelerated, host-staged drop-in for :class:`SimpleImputer`.

    Computes the fill value (``fit``) and fills missing values (``transform``)
    on the GPU using cuDF, keeping blocks host-resident. Supports the
    ``"mean"``, ``"most_frequent"`` and ``"constant"`` strategies and writes the
    same ``stats_`` keys as :class:`SimpleImputer` (``mean({col})`` /
    ``most_frequent({col})``), so it is a faithful drop-in.

    Falls back transparently to the CPU :class:`SimpleImputer` when no GPU /
    RAPIDS stack is available.

    When composed in a :class:`~ray.data.preprocessors.Chain` with
    ``backend="gpu"``, it implements the device-fusion contract so impute runs
    in the same device-resident pass as the following encode / scale, and it
    advertises its filled columns so a downstream scaler can apply the
    impute-mean == scaler-mean fold.

    .. note::
        For ``"most_frequent"``, ties are broken by the **smallest** value
        (deterministic), which can differ from the CPU imputer's first-seen
        ``Counter`` order when two values share the top count.

    .. seealso::
        :class:`SimpleImputer`
            The CPU implementation this mirrors and falls back to.
    """

    def _fit(self, dataset: "Dataset") -> "SimpleImputer":
        from ray.data.preprocessors import _gpu

        if not _gpu.gpu_available():
            return super()._fit(dataset)

        if self._strategy == "mean":
            sums = _gpu.gpu_sum_count(dataset, self._columns)
            for col in self._columns:
                total, count = sums[col]
                self.stats_[f"mean({col})"] = (total / count) if count else None
        elif self._strategy == "most_frequent":
            counts = _gpu.gpu_value_counts(dataset, self._columns)
            for col in self._columns:
                self.stats_[f"most_frequent({col})"] = _mode_from_counts(counts[col])
        else:
            # "constant" is not fittable; nothing to compute.
            return super()._fit(dataset)
        return self

    def _transform(
        self,
        ds: "Dataset",
        batch_size,
        num_cpus=None,
        memory=None,
        concurrency=None,
    ) -> "Dataset":
        from ray.data.preprocessors import _gpu
        from ray.data.preprocessors._gpu_fused import run_fused_device_transform

        if not _gpu.gpu_available():
            return super()._transform(
                ds,
                batch_size=batch_size,
                num_cpus=num_cpus,
                memory=memory,
                concurrency=concurrency,
            )
        return run_fused_device_transform(
            ds, [self], batch_size=batch_size, concurrency=concurrency
        )

    # --- device-fusion contract -------------------------------------------- #
    def _device_input_columns(self) -> List[str]:
        return list(self._columns)

    def _device_output_columns(self) -> List[str]:
        return list(self._output_columns)

    def _device_build_state(self) -> Dict[str, Any]:
        return {col: self._get_fill_value(col) for col in self._columns}

    def _device_step(self, state: Dict[str, Any], gdf: "cudf.DataFrame") -> "cudf.DataFrame":
        import cudf

        for input_col, output_col in zip(self._columns, self._output_columns):
            value = state[input_col]
            if value is None:
                raise ValueError(
                    f"Column {input_col} has no fill value. "
                    "Check the data used to fit the SimpleImputer."
                )
            if input_col in gdf.columns:
                series = gdf[input_col]
            else:
                # Missing column: create it filled (matches CPU SimpleImputer).
                series = cudf.Series([value] * len(gdf))
            # A float fill on an integer column must widen the column first.
            if isinstance(value, float) and series.dtype.kind in ("i", "u"):
                series = series.astype("float64")
            gdf[output_col] = series.fillna(value)
        return gdf

    def _device_fit_requests(self) -> List[FitRequest]:
        if self._strategy == "mean":
            return [FitRequest("moments", list(self._columns))]
        if self._strategy == "most_frequent":
            return [FitRequest("value_counts", list(self._columns))]
        return []

    def _device_set_fitted(self, ctx) -> None:
        if self._strategy == "mean":
            for col in self._columns:
                m = ctx.moments(col)
                self.stats_[f"mean({col})"] = m.mean if m.n > 0 else None
        elif self._strategy == "most_frequent":
            for col in self._columns:
                self.stats_[f"most_frequent({col})"] = ctx.mode(col)
        # "constant": nothing to fit.

    def _device_null_fill_columns(self) -> Dict[str, str]:
        # Report the OUTPUT columns (the ones actually filled) so a downstream
        # op reading them can detect the imputation.
        return {out: self._strategy for out in self._output_columns}


def _mode_from_counts(tbl: "Any"):
    """Most frequent value from a ``(value, count)`` table; ties -> smallest."""
    import pyarrow.compute as pc

    if tbl is None or tbl.num_rows == 0:
        return None
    counts = tbl.column("count")
    max_count = pc.max(counts).as_py()
    candidates = pc.filter(tbl.column("value"), pc.equal(counts, max_count))
    if len(candidates) == 0:
        return None
    return pc.min(candidates).as_py()
