"""GPU-accelerated encoders for Ray Data (experimental, opt-in).

These mirror the host-staged design of the experimental GPU sort: blocks start
and end in RAM (Arrow), and only the operator's input columns are moved to a GPU
(as cuDF) for the encode, then re-attached to the original Arrow block. The CPU
encoders remain the default; you opt in by using the ``Gpu*`` class explicitly.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Callable, Dict, List

from ray.data.preprocessors.encoder import OrdinalEncoder
from ray.data.preprocessors.version_support import SerializablePreprocessor
from ray.util.annotations import PublicAPI

if TYPE_CHECKING:
    import pyarrow as pa

    from ray.data.dataset import Dataset

logger = logging.getLogger(__name__)

# Soft guardrail: the fitted vocabulary is replicated as a cuDF categorical on
# *every* GPU rank (see ``_make_build_state``), so a very large cardinality can
# OOM each GPU and bloat the driver-side merge. We warn past this many distinct
# categories (override with ``RAY_DATA_GPU_PREPROC_MAX_VOCAB_WARN``); the encode
# still proceeds. A future device-side / sharded-broadcast path would lift this.
_DEFAULT_MAX_VOCAB_WARN = 5_000_000


def _make_build_state(
    columns: List[str], keys_by_col: Dict[str, "pa.Array"]
) -> Callable[[], Dict]:
    """Build a closure that moves the fitted vocabularies to the device once."""

    def build() -> Dict:
        import cudf

        dtypes = {}
        for col in columns:
            cats = cudf.Series.from_arrow(keys_by_col[col])
            dtypes[col] = cudf.CategoricalDtype(categories=cats, ordered=True)
        return dtypes

    return build


def _make_apply(
    columns: List[str], output_columns: List[str]
) -> Callable[[Dict, "pa.Table"], "pa.Table"]:
    """Build the per-batch encode closure (Arrow in -> Arrow out)."""

    def apply(dtypes: Dict, batch: "pa.Table") -> "pa.Table":
        import cudf

        from ray.data.preprocessors._gpu import attach_arrow_columns

        # Move only the input columns across the bus (payload columns stay host).
        gdf = cudf.DataFrame.from_arrow(batch.select(columns))
        new_columns = {}
        for input_col, output_col in zip(columns, output_columns):
            series = gdf[input_col]
            # Match the CPU encoder: null *inputs* raise; unseen (non-null)
            # categories map to null (-> NaN in pandas), via cat.codes == -1.
            if series.null_count:
                raise ValueError(
                    f"Unable to transform column {input_col!r} because it "
                    "contains null values. Consider imputing missing values "
                    "first."
                )
            dtype = dtypes[input_col]
            n_categories = len(dtype.categories)
            codes = series.astype(dtype).cat.codes.astype("int64")
            # Unseen categories map to a sentinel that cuDF stores in an unsigned
            # code type (so -1 reads back as e.g. 255). Mask anything outside the
            # valid code range to null -> NaN, matching the CPU encoder.
            codes = codes.mask((codes < 0) | (codes >= n_categories))
            new_columns[output_col] = codes.to_arrow()
        return attach_arrow_columns(batch, new_columns)

    return apply


@PublicAPI(stability="alpha")
@SerializablePreprocessor(
    version=1, identifier="io.ray.preprocessors.gpu_ordinal_encoder"
)
class GpuOrdinalEncoder(OrdinalEncoder):
    r"""GPU-accelerated, host-staged drop-in for :class:`OrdinalEncoder`.

    Computes category vocabularies (``fit``) and maps categories to ordinal
    integer codes (``transform``) on the GPU using cuDF, while keeping blocks
    host-resident: each batch is pulled as Arrow (RAM), the input columns are
    moved to a GPU, encoded, and written back as Arrow (RAM). Output codes are
    identical to :class:`OrdinalEncoder` (sorted-vocabulary order; unseen
    categories become ``NaN``).

    This is an experimental, opt-in operator. If no GPU / RAPIDS stack is
    available, or if a target column is list-typed, it transparently falls back
    to the CPU :class:`OrdinalEncoder` implementation, so it stays a faithful
    drop-in.

    Concurrency and batch size are controlled by the
    ``RAY_DATA_GPU_PREPROC_NUM_GPUS`` and ``RAY_DATA_GPU_PREPROC_BATCH_SIZE``
    environment variables (or the ``transform_concurrency`` / batch arguments).

    Examples:
        >>> import pandas as pd
        >>> import ray
        >>> from ray.data.preprocessors import GpuOrdinalEncoder
        >>> df = pd.DataFrame({"item": ["i20", "i10", "i30", "i10"]})
        >>> ds = ray.data.from_pandas(df)  # doctest: +SKIP
        >>> enc = GpuOrdinalEncoder(columns=["item"])
        >>> enc.fit_transform(ds).to_pandas()  # doctest: +SKIP
           item
        0     1
        1     0
        2     2
        3     0

    Args:
        columns: The columns to separately encode.
        encode_lists: If ``True``, encode list elements (CPU fallback path).
            ``True`` by default.
        output_columns: The names of the transformed columns. If None, the
            transformed columns will be the same as the input columns.

    .. seealso::
        :class:`OrdinalEncoder`
            The CPU implementation this mirrors and falls back to.
    """

    def _fit(self, dataset: "Dataset") -> "OrdinalEncoder":
        from ray.data.preprocessors import _gpu

        if not _gpu.gpu_available() or self._has_list_columns(dataset):
            return super()._fit(dataset)

        import pyarrow as pa
        import pyarrow.compute as pc

        uniques = _gpu.gpu_unique_values(dataset, self._columns)
        for col in self._columns:
            values = uniques[col]
            if len(values) and pc.any(
                pc.is_null(values, nan_is_null=True)
            ).as_py():
                raise ValueError(
                    "Unable to fit column because it contains null values. "
                    "Consider imputing missing values first."
                )
            # Match CPU OrdinalEncoder: codes assigned by sorted value order.
            sorted_values = pc.take(values, pc.sort_indices(values))
            codes = pa.array(range(len(sorted_values)), type=pa.int64())
            self.stats_[f"unique_values({col})"] = (sorted_values, codes)
            self._warn_if_vocab_large(col, len(sorted_values))
        return self

    @staticmethod
    def _warn_if_vocab_large(column: str, n_categories: int) -> None:
        """Warn when a column's vocabulary is large enough to risk per-GPU OOM.

        The vocabulary is broadcast to and materialized as a categorical on every
        GPU rank, so cardinality (not row count) is the scaling limit here.
        """
        try:
            threshold = int(
                os.environ.get(
                    "RAY_DATA_GPU_PREPROC_MAX_VOCAB_WARN",
                    _DEFAULT_MAX_VOCAB_WARN,
                )
            )
        except (TypeError, ValueError):
            threshold = _DEFAULT_MAX_VOCAB_WARN
        if n_categories > threshold:
            logger.warning(
                "GpuOrdinalEncoder: column %r has %d distinct categories, which "
                "is replicated on every GPU rank and merged on the driver; this "
                "may OOM at high cardinality. Consider hash encoding, or raise "
                "RAY_DATA_GPU_PREPROC_MAX_VOCAB_WARN to silence.",
                column,
                n_categories,
            )

    def _transform(
        self,
        ds: "Dataset",
        batch_size,
        num_cpus=None,
        memory=None,
        concurrency=None,
    ) -> "Dataset":
        from ray.data.preprocessors import _gpu

        if not _gpu.gpu_available() or self._has_list_columns(ds):
            return super()._transform(
                ds,
                batch_size=batch_size,
                num_cpus=num_cpus,
                memory=memory,
                concurrency=concurrency,
            )

        keys_by_col = {
            col: self._get_arrow_arrays(col)[0] for col in self._columns
        }
        build_state = _make_build_state(list(self._columns), keys_by_col)
        apply_fn = _make_apply(
            list(self._columns), list(self._output_columns)
        )
        return _gpu.gpu_transform(
            ds,
            build_state=build_state,
            apply_fn=apply_fn,
            batch_size=batch_size,
            concurrency=concurrency,
        )

    def _has_list_columns(self, ds: "Dataset") -> bool:
        """True if any target column is list-typed (-> CPU fallback path)."""
        import pyarrow as pa

        try:
            schema = ds.schema()
            types = dict(zip(schema.names, schema.types))
        except Exception:
            return False
        for col in self._columns:
            dtype = types.get(col)
            # ``schema().types`` yields pyarrow types for Arrow-backed datasets
            # and numpy dtypes for pandas-backed ones; only the former can be a
            # list type we must route to the CPU fallback.
            if isinstance(dtype, pa.DataType) and (
                pa.types.is_list(dtype) or pa.types.is_large_list(dtype)
            ):
                return True
        return False
