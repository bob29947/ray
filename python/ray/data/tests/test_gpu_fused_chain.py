"""Tests for the fused device-resident GPU path in Chain.

Covers the CPU->GPU upgrade registry, contiguous-run segmentation, CPU fallback
parity, end-to-end fused parity vs the CPU chain (GPU-gated), and serialization
of a fitted fused chain.
"""

import os

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

import ray
from ray.data.preprocessors import (
    Chain,
    Concatenator,
    OrdinalEncoder,
    SimpleImputer,
    StandardScaler,
)
from ray.data.preprocessors import _gpu
from ray.data.preprocessors._gpu_fused import is_device_fusable, upgrade_to_device_op
from ray.data.preprocessors.gpu_encoder import GpuOrdinalEncoder
from ray.data.preprocessors.gpu_imputer import GpuSimpleImputer
from ray.data.preprocessors.gpu_scaler import GpuStandardScaler

requires_gpu = pytest.mark.skipif(
    not _gpu.gpu_available(),
    reason="requires a GPU + RAPIDS (cudf/rmm)",
)


@pytest.fixture(scope="module", autouse=True)
def _ray_init():
    os.environ.setdefault("RAY_DATA_GPU_PREPROC_NUM_GPUS", "1")
    os.environ.setdefault("RAY_DATA_GPU_PREPROC_BATCH_SIZE", "1048576")
    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
    ray.data.DataContext.get_current().execution_options.preserve_order = True
    yield
    ray.shutdown()


def _df():
    return pd.DataFrame(
        {
            "id": range(8),
            "num1": [1.0, 2.0, None, 4.0, 5.0, None, 7.0, 8.0],   # mean-impute + scale
            "num2": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],  # scale only
            "cat1": ["a", "b", None, "a", "a", "b", None, "c"],   # most_freq + encode
            "cat2": ["x", "y", "x", "z", "x", "y", "z", "x"],     # encode only
        }
    )


def _make_chain(backend):
    return Chain(
        SimpleImputer(columns=["num1"], strategy="mean"),
        SimpleImputer(columns=["cat1"], strategy="most_frequent"),
        OrdinalEncoder(columns=["cat1", "cat2"]),
        StandardScaler(columns=["num1", "num2"]),
        backend=backend,
    )


def _sorted(ds):
    return ds.to_pandas().sort_values("id").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Upgrade registry + segmentation (no GPU required)
# --------------------------------------------------------------------------- #
def test_upgrade_registry_maps_cpu_to_gpu():
    imp = upgrade_to_device_op(SimpleImputer(columns=["a"], strategy="mean"))
    assert isinstance(imp, GpuSimpleImputer)
    assert imp.columns == ["a"] and imp.strategy == "mean"

    enc = upgrade_to_device_op(OrdinalEncoder(columns=["c"], output_columns=["c_e"]))
    assert isinstance(enc, GpuOrdinalEncoder)
    assert enc.columns == ["c"] and enc.output_columns == ["c_e"]

    sc = upgrade_to_device_op(StandardScaler(columns=["x"]))
    assert isinstance(sc, GpuStandardScaler)


def test_upgrade_passthrough():
    # Already-fusable op is returned unchanged.
    g = GpuStandardScaler(columns=["x"])
    assert upgrade_to_device_op(g) is g
    # Unregistered op is returned unchanged (will break a fused run).
    c = Concatenator(columns=["x", "y"], output_column_name="z")
    assert upgrade_to_device_op(c) is c
    assert not is_device_fusable(c)


def test_build_segments_contiguous_runs():
    ops = [
        GpuStandardScaler(columns=["x"]),
        Concatenator(columns=["x", "y"], output_column_name="z"),
        GpuStandardScaler(columns=["y"]),
        GpuStandardScaler(columns=["z"]),
    ]
    segs = Chain._build_segments(ops, None)
    kinds = [k for k, _ in segs]
    assert kinds == ["fused", "single", "fused"]
    assert len(segs[0][1]) == 1  # first fused run: one op
    assert len(segs[2][1]) == 2  # last fused run: two contiguous ops


# --------------------------------------------------------------------------- #
# CPU fallback parity (no GPU required)
# --------------------------------------------------------------------------- #
def test_cpu_fallback_matches_cpu_chain():
    df = _df()
    with patch.object(_gpu, "gpu_available", return_value=False):
        cpu = _make_chain("cpu").fit_transform(ray.data.from_pandas(df))
        gpu = _make_chain("gpu").fit_transform(ray.data.from_pandas(df))
    a, b = _sorted(cpu), _sorted(gpu)
    for c in ["num1", "num2", "cat1", "cat2"]:
        np.testing.assert_allclose(
            pd.to_numeric(a[c]).to_numpy(float),
            pd.to_numeric(b[c]).to_numpy(float),
            rtol=1e-6,
            atol=1e-6,
            equal_nan=True,
        )


def test_cpu_fallback_leaves_children_unupgraded():
    df = _df()
    chain = _make_chain("gpu")
    with patch.object(_gpu, "gpu_available", return_value=False):
        chain.fit_transform(ray.data.from_pandas(df))
    # No GPU -> CPU path -> children stay the original CPU classes.
    assert not any(is_device_fusable(op) for op in chain.preprocessors)


# --------------------------------------------------------------------------- #
# End-to-end fused parity (GPU required)
# --------------------------------------------------------------------------- #
@requires_gpu
def test_gpu_fused_matches_cpu_chain():
    df = _df()
    cpu = _sorted(_make_chain("cpu").fit_transform(ray.data.from_pandas(df)))
    gpu = _sorted(_make_chain("gpu").fit_transform(ray.data.from_pandas(df)))
    for c in ["num1", "num2", "cat1", "cat2"]:
        np.testing.assert_allclose(
            pd.to_numeric(cpu[c]).to_numpy(float),
            pd.to_numeric(gpu[c]).to_numpy(float),
            rtol=1e-6,
            atol=1e-6,
            equal_nan=True,
        )
    # A mean-imputed-then-scaled null resolves to exactly 0 (rows id 2 and 5).
    assert gpu["num1"].to_numpy(float)[2] == pytest.approx(0.0)
    assert gpu["num1"].to_numpy(float)[5] == pytest.approx(0.0)


@requires_gpu
def test_gpu_fused_fitted_stats_match_cpu():
    df = _df()
    cpu_chain = _make_chain("cpu")
    cpu_chain.fit(ray.data.from_pandas(df))
    gpu_chain = _make_chain("gpu")
    gpu_chain.fit(ray.data.from_pandas(df))

    def _find(chain, cls, **attrs):
        for op in chain.preprocessors:
            if isinstance(op, cls) and all(
                getattr(op, k) == v for k, v in attrs.items()
            ):
                return op
        raise AssertionError(f"{cls} not found")

    cpu_sc = _find(cpu_chain, StandardScaler)
    gpu_sc = _find(gpu_chain, StandardScaler)
    # num1 is mean-imputed before scaling -> scaler std uses N_total (8), so it
    # must match the CPU scaler fit on the imputed column (NOT the non-null std).
    for c in ["num1", "num2"]:
        assert gpu_sc.stats_[f"mean({c})"] == pytest.approx(cpu_sc.stats_[f"mean({c})"])
        assert gpu_sc.stats_[f"std({c})"] == pytest.approx(cpu_sc.stats_[f"std({c})"])

    gpu_imp = _find(gpu_chain, SimpleImputer, strategy="mean")
    cpu_imp = _find(cpu_chain, SimpleImputer, strategy="mean")
    assert gpu_imp.stats_["mean(num1)"] == pytest.approx(cpu_imp.stats_["mean(num1)"])
    # impute-mean == scaler-mean (the shared-stats fold).
    assert gpu_imp.stats_["mean(num1)"] == pytest.approx(gpu_sc.stats_["mean(num1)"])


@requires_gpu
def test_gpu_fused_serialization_roundtrip():
    df = _df()
    chain = _make_chain("gpu")
    chain.fit(ray.data.from_pandas(df))
    # After a GPU fit, children are the fitted device-fusable ops.
    assert any(is_device_fusable(op) for op in chain.preprocessors)

    restored = Chain.deserialize(chain.serialize())
    assert restored.backend == "gpu"
    out = _sorted(restored.transform(ray.data.from_pandas(df)))
    ref = _sorted(_make_chain("cpu").fit_transform(ray.data.from_pandas(df)))
    for c in ["num1", "num2", "cat1", "cat2"]:
        np.testing.assert_allclose(
            pd.to_numeric(out[c]).to_numpy(float),
            pd.to_numeric(ref[c]).to_numpy(float),
            rtol=1e-6,
            atol=1e-6,
            equal_nan=True,
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
