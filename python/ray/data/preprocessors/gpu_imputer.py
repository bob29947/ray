"""GPU-accelerated imputation for Ray Data (experimental, opt-in).

Host-staged drop-in for :class:`SimpleImputer`: the fit statistics (mean /
most-frequent value) are computed on the GPU with cuDF, and the per-batch fill
runs on the GPU, while blocks start and end in RAM as Arrow. The CPU imputer
remains the default; you opt in by using :class:`GpuSimpleImputer` explicitly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List

from ray.data.preprocessors.imputer import SimpleImputer
from ray.data.preprocessors.version_support import (
    SerializablePreprocessor as Serializable,
)
from ray.util.annotations import PublicAPI

if TYPE_CHECKING:
    import pyarrow as pa

    from ray.data.dataset import Dataset


def _make_build_state(fill_by_col: Dict[str, object]) -> Callable[[], Dict]:
    def build() -> Dict:
        # Fill values are small scalars; no device-resident state is needed.
        return fill_by_col

    return build


def _make_apply(
    columns: List[str], output_columns: List[str]
) -> Callable[[Dict, "pa.Table"], "pa.Table"]:
    def apply(fill_by_col: Dict, batch: "pa.Table") -> "pa.Table":
        import cudf
        import pyarrow as pa

        from ray.data.preprocessors._gpu import attach_arrow_columns

        present = [c for c in columns if c in batch.column_names]
        gdf = cudf.DataFrame.from_arrow(batch.select(present)) if present else None

        new_columns = {}
        num_rows = batch.num_rows
        for input_col, output_col in zip(columns, output_columns):
            value = fill_by_col[input_col]
            if input_col not in batch.column_names:
                # Match CPU SimpleImputer: a missing column is created filled.
                new_columns[output_col] = pa.array([value] * num_rows)
                continue
            series = gdf[input_col]
            # A float fill (mean) on an integer column should widen to float, as
            # pandas does once the column contains NaN.
            if isinstance(value, float) and series.dtype.kind in ("i", "u"):
                series = series.astype("float64")
            new_columns[output_col] = series.fillna(value).to_arrow()
        return attach_arrow_columns(batch, new_columns)

    return apply


@PublicAPI(stability="alpha")
@Serializable(version=1, identifier="io.ray.preprocessors.gpu_simple_imputer")
class GpuSimpleImputer(SimpleImputer):
    r"""GPU-accelerated, host-staged drop-in for :class:`SimpleImputer`.

    Computes the fill statistic on the GPU (``"mean"`` via a GPU sum/count
    reduction, ``"most_frequent"`` via cuDF value counts) and applies the fill on
    the GPU with cuDF, while keeping blocks host-resident (Arrow in -> GPU ->
    Arrow out). The ``"most_frequent"`` strategy over string columns is where the
    GPU win is largest, since the CPU path counts string values with Python
    ``Counter`` objects.

    This is an experimental, opt-in operator that transparently falls back to the
    CPU :class:`SimpleImputer` when no GPU / RAPIDS stack is available.

    .. note::
        For ``"most_frequent"``, ties are broken **deterministically** by the
        smallest value. The CPU :class:`SimpleImputer` breaks ties by insertion
        order (``Counter.most_common``), so results can differ only when two
        values share the maximum count.

    Args:
        columns: The columns to apply imputation to.
        strategy: One of ``"mean"``, ``"most_frequent"``, or ``"constant"``.
        fill_value: The value to use when ``strategy="constant"``.
        output_columns: The names of the transformed columns. If None, the
            transformed columns will be the same as the input columns.

    .. seealso::
        :class:`SimpleImputer`
            The CPU implementation this mirrors and falls back to.
    """

    def _fit(self, dataset: "Dataset") -> "SimpleImputer":
        from ray.data.preprocessors import _gpu

        if not _gpu.gpu_available():
            return super()._fit(dataset)

        import pyarrow.compute as pc

        if self._strategy == "mean":
            sums = _gpu.gpu_sum_count(dataset, self._columns)
            for col in self._columns:
                total, count = sums[col]
                self.stats_[f"mean({col})"] = (total / count) if count else None
            return self

        if self._strategy == "most_frequent":
            counts = _gpu.gpu_value_counts(dataset, self._columns)
            for col in self._columns:
                table = counts[col]
                if table.num_rows == 0:
                    self.stats_[f"most_frequent({col})"] = None
                    continue
                count_col = table.column("count")
                value_col = table.column("value")
                max_count = pc.max(count_col).as_py()
                candidates = pc.filter(value_col, pc.equal(count_col, max_count))
                # Deterministic tie-break: smallest value.
                self.stats_[f"most_frequent({col})"] = pc.min(candidates).as_py()
            return self

        # "constant" is not fittable and never reaches here.
        return super()._fit(dataset)

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

        fill_by_col = {}
        for col in self._columns:
            value = self._get_fill_value(col)
            if value is None:
                raise ValueError(
                    f"Column {col} has no fill value. "
                    "Check the data used to fit the SimpleImputer."
                )
            fill_by_col[col] = value

        build_state = _make_build_state(fill_by_col)
        apply_fn = _make_apply(list(self._columns), list(self._output_columns))
        return _gpu.gpu_transform(
            ds,
            build_state=build_state,
            apply_fn=apply_fn,
            batch_size=batch_size,
            concurrency=concurrency,
        )
