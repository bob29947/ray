"""GPU-accelerated standard scaling for Ray Data (experimental, opt-in).

Host-staged drop-in for :class:`StandardScaler`: the per-column mean and
(population) standard deviation are computed on the GPU with cuDF, and the
per-batch scale runs on the GPU, while blocks start and end in RAM as Arrow.
The CPU scaler remains the default; you opt in by using
:class:`GpuStandardScaler` explicitly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List

from ray.data.preprocessors.scaler import _EPSILON, StandardScaler
from ray.data.preprocessors.version_support import SerializablePreprocessor
from ray.util.annotations import PublicAPI

if TYPE_CHECKING:
    import pyarrow as pa

    from ray.data.dataset import Dataset


def _make_build_state(stats_by_col: Dict[str, tuple]) -> Callable[[], Dict]:
    def build() -> Dict:
        # mean/std are small scalars; no device-resident state is needed.
        return stats_by_col

    return build


def _make_apply(
    columns: List[str], output_columns: List[str]
) -> Callable[[Dict, "pa.Table"], "pa.Table"]:
    def apply(stats_by_col: Dict, batch: "pa.Table") -> "pa.Table":
        import cudf
        import pyarrow as pa

        from ray.data.preprocessors._gpu import attach_arrow_columns

        # Snapshot all input columns to the device once, so writing an output
        # never reads an already-scaled input (the read-after-write hazard the
        # CPU _transform_arrow guards against when output_columns overlap).
        gdf = cudf.DataFrame.from_arrow(batch.select(columns))
        new_columns = {}
        for input_col, output_col in zip(columns, output_columns):
            mean, std = stats_by_col[input_col]
            # Mirror StandardScaler._transform_arrow: an undefined statistic
            # yields a null column of the original type. A NaN statistic flows
            # through below and produces NaN, exactly like the CPU path.
            if mean is None or std is None:
                new_columns[output_col] = pa.nulls(
                    batch.num_rows, type=batch.schema.field(input_col).type
                )
                continue
            # Match _scale_column: a (near-)constant column scales to zeros.
            denom = 1.0 if std < _EPSILON else std
            new_columns[output_col] = ((gdf[input_col] - mean) / denom).to_arrow()
        return attach_arrow_columns(batch, new_columns)

    return apply


@PublicAPI(stability="alpha")
@SerializablePreprocessor(
    version=1, identifier="io.ray.preprocessors.gpu_standard_scaler"
)
class GpuStandardScaler(StandardScaler):
    r"""GPU-accelerated, host-staged drop-in for :class:`StandardScaler`.

    Computes the per-column mean and **population** standard deviation
    (``ddof=0``) on the GPU via a cuDF mean / sum-of-squared-deviations
    reduction, and applies :math:`x' = (x - \bar{x}) / s` on the GPU, while
    keeping blocks host-resident (Arrow in -> GPU -> Arrow out). Output is
    identical to :class:`StandardScaler`: constant-valued columns (where
    :math:`s < 10^{-8}`) scale to zeros, and integer columns widen to float.

    This is an experimental, opt-in operator that transparently falls back to
    the CPU :class:`StandardScaler` when no GPU / RAPIDS stack is available, so
    it stays a faithful drop-in.

    Concurrency and batch size are controlled by the
    ``RAY_DATA_GPU_PREPROC_NUM_GPUS`` and ``RAY_DATA_GPU_PREPROC_BATCH_SIZE``
    environment variables (or the ``transform_concurrency`` / batch arguments).

    Examples:
        >>> import pandas as pd
        >>> import ray
        >>> from ray.data.preprocessors import GpuStandardScaler
        >>>
        >>> df = pd.DataFrame({"X1": [-2, 0, 2], "X2": [-3, -3, 3]})
        >>> ds = ray.data.from_pandas(df)  # doctest: +SKIP
        >>> scaler = GpuStandardScaler(columns=["X1", "X2"])
        >>> scaler.fit_transform(ds).to_pandas()  # doctest: +SKIP
                 X1        X2
        0 -1.224745 -0.707107
        1  0.000000 -0.707107
        2  1.224745  1.414214

    Args:
        columns: The columns to separately scale.
        output_columns: The names of the transformed columns. If None, the
            transformed columns will be the same as the input columns. If not
            None, the length of ``output_columns`` must match the length of
            ``columns``, otherwise an error will be raised.

    .. seealso::
        :class:`StandardScaler`
            The CPU implementation this mirrors and falls back to.
    """

    def _fit(self, dataset: "Dataset") -> "StandardScaler":
        from ray.data.preprocessors import _gpu

        if not _gpu.gpu_available():
            return super()._fit(dataset)

        # ddof=0 matches StandardScaler's Std(col, ddof=0) (population std).
        stats = _gpu.gpu_mean_std(dataset, self._columns, ddof=0)
        for col in self._columns:
            mean, std = stats[col]
            self.stats_[f"mean({col})"] = mean
            self.stats_[f"std({col})"] = std
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

        if not _gpu.gpu_available():
            return super()._transform(
                ds,
                batch_size=batch_size,
                num_cpus=num_cpus,
                memory=memory,
                concurrency=concurrency,
            )

        stats_by_col = {
            col: (self.stats_[f"mean({col})"], self.stats_[f"std({col})"])
            for col in self._columns
        }
        build_state = _make_build_state(stats_by_col)
        apply_fn = _make_apply(list(self._columns), list(self._output_columns))
        return _gpu.gpu_transform(
            ds,
            build_state=build_state,
            apply_fn=apply_fn,
            batch_size=batch_size,
            concurrency=concurrency,
        )
