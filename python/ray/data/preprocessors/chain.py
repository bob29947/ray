from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from ray.data.preprocessor import Preprocessor, SerializablePreprocessorBase
from ray.data.preprocessors.utils import (
    _PublicField,
    migrate_private_fields,
)
from ray.data.preprocessors.version_support import SerializablePreprocessor
from ray.data.util.data_batch_conversion import BatchFormat

if TYPE_CHECKING:
    from ray.air.data_batch_type import DataBatchType
    from ray.data.dataset import Dataset


@SerializablePreprocessor(version=1, identifier="io.ray.preprocessors.chain")
class Chain(SerializablePreprocessorBase):
    """Combine multiple preprocessors into a single :py:class:`Preprocessor`.

    When you call ``fit``, each preprocessor is fit on the dataset produced by the
    preceeding preprocessor's ``fit_transform``.

    Example:
        >>> import pandas as pd
        >>> import ray
        >>> from ray.data.preprocessors import *
        >>>
        >>> df = pd.DataFrame({
        ...     "X0": [0, 1, 2],
        ...     "X1": [3, 4, 5],
        ...     "Y": ["orange", "blue", "orange"],
        ... })
        >>> ds = ray.data.from_pandas(df)  # doctest: +SKIP
        >>>
        >>> preprocessor = Chain(
        ...     StandardScaler(columns=["X0", "X1"]),
        ...     Concatenator(columns=["X0", "X1"], output_column_name="X"),
        ...     LabelEncoder(label_column="Y")
        ... )
        >>> preprocessor.fit_transform(ds).to_pandas()  # doctest: +SKIP
           Y                                         X
        0  1  [-1.224744871391589, -1.224744871391589]
        1  0                                [0.0, 0.0]
        2  1    [1.224744871391589, 1.224744871391589]

    Args:
        *preprocessors: The preprocessors to sequentially compose.
    """

    def fit_status(self):
        fittable_count = 0
        fitted_count = 0

        for p in self._preprocessors:
            if p.fit_status() == Preprocessor.FitStatus.FITTED:
                fittable_count += 1
                fitted_count += 1
            elif p.fit_status() in (
                Preprocessor.FitStatus.NOT_FITTED,
                Preprocessor.FitStatus.PARTIALLY_FITTED,
            ):
                fittable_count += 1
            else:
                assert p.fit_status() == Preprocessor.FitStatus.NOT_FITTABLE
        if fittable_count > 0:
            if fitted_count == fittable_count:
                return Preprocessor.FitStatus.FITTED
            elif fitted_count > 0:
                return Preprocessor.FitStatus.PARTIALLY_FITTED
            else:
                return Preprocessor.FitStatus.NOT_FITTED
        else:
            return Preprocessor.FitStatus.NOT_FITTABLE

    _VALID_BACKENDS = (None, "cpu", "gpu", "auto")

    def __init__(
        self,
        *preprocessors: SerializablePreprocessorBase,
        backend: Optional[str] = None,
    ):
        super().__init__()
        if backend not in self._VALID_BACKENDS:
            raise ValueError(
                f"Invalid backend {backend!r}. Expected one of "
                f"{self._VALID_BACKENDS}."
            )
        self._preprocessors = preprocessors
        # Fusion backend. ``None``/``"cpu"`` -> today's per-child CPU behavior
        # (no change). ``"gpu"``/``"auto"`` -> when a GPU is available, fuse
        # contiguous runs of device-fusable children into a single
        # device-resident pass (impute+encode+scale crossing PCIe once); falls
        # back to the CPU path when no GPU/RAPIDS stack is present.
        self._backend = backend

    @property
    def preprocessors(self) -> Tuple[SerializablePreprocessorBase, ...]:
        return self._preprocessors

    @property
    def backend(self) -> Optional[str]:
        return self._backend

    def _gpu_enabled(self) -> bool:
        """True if this chain should use the fused device-resident GPU path."""
        if self._backend in (None, "cpu"):
            return False
        from ray.data.preprocessors import _gpu

        return _gpu.gpu_available()

    def _upgrade_ops(self) -> Tuple[SerializablePreprocessorBase, ...]:
        """Replace registered CPU children with their fusable GPU counterparts."""
        from ray.data.preprocessors._gpu_fused import upgrade_to_device_op

        return tuple(upgrade_to_device_op(op) for op in self._preprocessors)

    @staticmethod
    def _build_segments(ops, schema):
        """Group ``ops`` into maximal contiguous fusable runs vs single ops.

        Returns a list of ``("fused", [ops])`` / ``("single", op)`` segments. A
        run is a maximal span of device-fusable ops that can fuse for the given
        (input) schema; anything else breaks the run and runs on its own path.
        """
        from ray.data.preprocessors._gpu_fused import is_device_fusable

        segments = []
        run = []
        for op in ops:
            if is_device_fusable(op) and op._device_can_fuse(schema):
                run.append(op)
            else:
                if run:
                    segments.append(("fused", run))
                    run = []
                segments.append(("single", op))
        if run:
            segments.append(("fused", run))
        return segments

    def _fit(self, ds: "Dataset") -> SerializablePreprocessorBase:
        if not self._gpu_enabled():
            for preprocessor in self._preprocessors[:-1]:
                ds = preprocessor.fit_transform(ds)
            self._preprocessors[-1].fit(ds)
            return self

        from ray.data.preprocessors._gpu_fused import (
            _safe_schema,
            fused_fit,
            run_fused_device_transform,
        )

        ops = self._upgrade_ops()
        self._preprocessors = ops
        segments = self._build_segments(ops, _safe_schema(ds))
        for i, (kind, payload) in enumerate(segments):
            is_last = i == len(segments) - 1
            if kind == "fused":
                fused_fit(ds, payload)
                for op in payload:
                    op._fitted = True
                if not is_last:
                    ds = run_fused_device_transform(ds, payload)
            else:
                if is_last:
                    payload.fit(ds)
                else:
                    ds = payload.fit_transform(ds)
        return self

    def fit_transform(self, ds: "Dataset") -> "Dataset":
        if not self._gpu_enabled():
            for preprocessor in self._preprocessors:
                ds = preprocessor.fit_transform(ds)
            return ds

        from ray.data.preprocessors._gpu_fused import (
            _safe_schema,
            fused_fit,
            run_fused_device_transform,
        )

        ops = self._upgrade_ops()
        self._preprocessors = ops
        for kind, payload in self._build_segments(ops, _safe_schema(ds)):
            if kind == "fused":
                fused_fit(ds, payload)
                for op in payload:
                    op._fitted = True
                ds = run_fused_device_transform(ds, payload)
            else:
                ds = payload.fit_transform(ds)
        return ds

    def _transform(
        self,
        ds: "Dataset",
        batch_size: Optional[int],
        num_cpus: Optional[float] = None,
        memory: Optional[float] = None,
        concurrency: Optional[int] = None,
    ) -> "Dataset":
        if not self._gpu_enabled():
            for preprocessor in self._preprocessors:
                ds = preprocessor.transform(
                    ds,
                    batch_size=batch_size,
                    num_cpus=num_cpus,
                    memory=memory,
                    concurrency=concurrency,
                )
            return ds

        from ray.data.preprocessors._gpu_fused import (
            _safe_schema,
            run_fused_device_transform,
        )

        for kind, payload in self._build_segments(
            self._preprocessors, _safe_schema(ds)
        ):
            if kind == "fused":
                ds = run_fused_device_transform(
                    ds, payload, batch_size=batch_size, concurrency=concurrency
                )
            else:
                ds = payload.transform(
                    ds,
                    batch_size=batch_size,
                    num_cpus=num_cpus,
                    memory=memory,
                    concurrency=concurrency,
                )
        return ds

    def _transform_batch(self, df: "DataBatchType") -> "DataBatchType":
        for preprocessor in self._preprocessors:
            df = preprocessor.transform_batch(df)
        return df

    def __repr__(self):
        arguments = ", ".join(
            repr(preprocessor) for preprocessor in self._preprocessors
        )
        return f"{self.__class__.__name__}({arguments})"

    def _determine_transform_to_use(self) -> BatchFormat:
        # This is relevant for BatchPrediction.
        # For Chain preprocessor, we picked the first one as entry point.
        # TODO (jiaodong): We should revisit if our Chain preprocessor is
        # still optimal with context of lazy execution.
        return self._preprocessors[0]._determine_transform_to_use()

    def _get_serializable_fields(self) -> Dict[str, Any]:
        return {
            "preprocessors": self._preprocessors,
            "backend": self._backend,
        }

    def _set_serializable_fields(self, fields: Dict[str, Any], version: int):
        # required fields
        self._preprocessors = fields["preprocessors"]
        # optional fields
        self._backend = fields.get("backend")

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Handle backwards compatibility for old pickled objects."""
        super().__setstate__(state)
        migrate_private_fields(
            self,
            fields={
                "_preprocessors": _PublicField(public_field="preprocessors"),
            },
        )
        # ``backend`` was added later; default to CPU for old pickles.
        if not hasattr(self, "_backend"):
            self._backend = None
