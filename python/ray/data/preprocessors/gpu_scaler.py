"""GPU-accelerated scalers for Ray Data (experimental, opt-in).

Mirrors the host-staged design of :mod:`ray.data.preprocessors.gpu_encoder`:
blocks start and end in RAM (Arrow); only the operator's input columns move to
the GPU (as cuDF) for the fit reduction / transform, then results are
re-attached to the original Arrow block. The CPU scalers remain the default;
you opt in by using the ``Gpu*`` class explicitly, or transparently when a
``Chain(..., backend="gpu")`` fuses it with neighbouring GPU ops.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List

from ray.data.preprocessors._gpu_fused import DeviceFusable, FitRequest
from ray.data.preprocessors.scaler import _EPSILON, StandardScaler
from ray.data.preprocessors.version_support import SerializablePreprocessor
from ray.util.annotations import PublicAPI

if TYPE_CHECKING:
    import cudf

    from ray.data.dataset import Dataset

logger = logging.getLogger(__name__)


@PublicAPI(stability="alpha")
@SerializablePreprocessor(
    version=1, identifier="io.ray.preprocessors.gpu_standard_scaler"
)
class GpuStandardScaler(StandardScaler, DeviceFusable):
    r"""GPU-accelerated, host-staged drop-in for :class:`StandardScaler`.

    Computes per-column mean / standard deviation (``fit``) and applies the
    z-score :math:`x' = (x - \bar{x}) / s` (``transform``) on the GPU using cuDF,
    while keeping blocks host-resident. Output values and the fitted
    ``stats_`` keys (``mean({col})`` / ``std({col})``, population std with
    ``ddof=0``) are identical to :class:`StandardScaler`.

    This is an experimental, opt-in operator. If no GPU / RAPIDS stack is
    available it transparently falls back to the CPU :class:`StandardScaler`, so
    it stays a faithful drop-in.

    When composed in a :class:`~ray.data.preprocessors.Chain` with
    ``backend="gpu"``, it implements the device-fusion contract
    (:class:`~ray.data.preprocessors._gpu_fused.DeviceFusable`) so impute +
    encode + scale run in a single device-resident pass. If a column was
    mean-imputed by an earlier op in the same fused run, the scaler reuses that
    reduction (impute-mean == scaler-mean) and computes the post-impute std
    analytically (``sqrt(M2 / N_total)``) -- a null then resolves to ``0`` after
    the z-score, with no imputed intermediate materialized.

    .. seealso::
        :class:`StandardScaler`
            The CPU implementation this mirrors and falls back to.
    """

    def _fit(self, dataset: "Dataset") -> "StandardScaler":
        from ray.data.preprocessors import _gpu

        if not _gpu.gpu_available():
            return super()._fit(dataset)

        moments = _gpu.gpu_mean_std(dataset, self._columns, ddof=0)
        for col in self._columns:
            m = moments[col]
            self.stats_[f"mean({col})"] = m.mean if m.n > 0 else None
            self.stats_[f"std({col})"] = m.std if m.n > 0 else None
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
        return {
            col: (self.stats_[f"mean({col})"], self.stats_[f"std({col})"])
            for col in self._columns
        }

    def _device_step(self, state: Dict[str, Any], gdf: "cudf.DataFrame") -> "cudf.DataFrame":
        for input_col, output_col in zip(self._columns, self._output_columns):
            mean, std = state[input_col]
            if mean is None or std is None:
                # Degenerate (empty-at-fit) column: emit an all-null float column.
                gdf[output_col] = _null_float_like(gdf, input_col)
                continue
            # Match CPU StandardScaler numerical guard: near-zero std -> divide by 1.
            denom = 1.0 if std < _EPSILON else float(std)
            gdf[output_col] = (gdf[input_col] - float(mean)) / denom
        return gdf

    def _device_fit_requests(self) -> List[FitRequest]:
        return [FitRequest("moments", list(self._columns))]

    def _device_set_fitted(self, ctx) -> None:
        import math

        for col in self._columns:
            m = ctx.moments(col)
            if m.n <= 0:
                self.stats_[f"mean({col})"] = None
                self.stats_[f"std({col})"] = None
                continue
            self.stats_[f"mean({col})"] = m.mean
            if ctx.mean_imputed_before(self, col) and m.n_total > 0:
                # Scaler fits on the POST-(mean-)impute column: nulls became the
                # mean (0 contribution to M2) but raised the count to N_total.
                self.stats_[f"std({col})"] = math.sqrt(m.m2 / m.n_total)
            else:
                self.stats_[f"std({col})"] = m.std


def _null_float_like(gdf: "cudf.DataFrame", col: str):
    """All-null float64 column matching the length of ``gdf`` (degenerate path)."""
    import cudf

    return cudf.Series([None] * len(gdf), dtype="float64")
