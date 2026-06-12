"""Parity / fallback / serialization tests for GpuSimpleImputer.

GPU parity tests skip automatically without a GPU + RAPIDS stack.
"""

import os

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

import ray
from ray.data.preprocessors import SimpleImputer
from ray.data.preprocessors import _gpu
from ray.data.preprocessors.gpu_imputer import GpuSimpleImputer

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


def _num_df():
    return pd.DataFrame(
        {
            "id": range(8),
            "x": [1.0, 2.0, None, 4.0, 5.0, None, 7.0, 8.0],
        }
    )


def _cat_df():
    # 'a' is the unambiguous mode (no tie), so CPU/GPU agree.
    return pd.DataFrame(
        {
            "id": range(8),
            "c": ["a", "a", "a", "b", None, "b", None, "c"],
        }
    )


def _sorted(ds):
    return ds.to_pandas().sort_values("id").reset_index(drop=True)


def test_cpu_fallback_mean_matches_cpu():
    df = _num_df()
    with patch.object(_gpu, "gpu_available", return_value=False):
        cpu = SimpleImputer(columns=["x"], strategy="mean").fit_transform(
            ray.data.from_pandas(df)
        )
        gpu = GpuSimpleImputer(columns=["x"], strategy="mean").fit_transform(
            ray.data.from_pandas(df)
        )
    np.testing.assert_allclose(
        _sorted(cpu)["x"].to_numpy(float), _sorted(gpu)["x"].to_numpy(float)
    )


@requires_gpu
def test_gpu_mean_matches_cpu():
    df = _num_df()
    cpu = SimpleImputer(columns=["x"], strategy="mean")
    cpu_ds = cpu.fit_transform(ray.data.from_pandas(df))
    gpu = GpuSimpleImputer(columns=["x"], strategy="mean")
    gpu_ds = gpu.fit_transform(ray.data.from_pandas(df))
    # Mean over non-null {1,2,4,5,7,8} = 4.5; nulls become 4.5.
    assert gpu.stats_["mean(x)"] == pytest.approx(cpu.stats_["mean(x)"])
    assert gpu.stats_["mean(x)"] == pytest.approx(4.5)
    np.testing.assert_allclose(
        _sorted(cpu_ds)["x"].to_numpy(float), _sorted(gpu_ds)["x"].to_numpy(float)
    )


@requires_gpu
def test_gpu_most_frequent_matches_cpu():
    df = _cat_df()
    cpu = SimpleImputer(columns=["c"], strategy="most_frequent")
    cpu_ds = cpu.fit_transform(ray.data.from_pandas(df))
    gpu = GpuSimpleImputer(columns=["c"], strategy="most_frequent")
    gpu_ds = gpu.fit_transform(ray.data.from_pandas(df))
    assert gpu.stats_["most_frequent(c)"] == "a"
    assert gpu.stats_["most_frequent(c)"] == cpu.stats_["most_frequent(c)"]
    assert list(_sorted(cpu_ds)["c"]) == list(_sorted(gpu_ds)["c"])


@requires_gpu
def test_gpu_constant_fill():
    df = _num_df()
    gpu = GpuSimpleImputer(columns=["x"], strategy="constant", fill_value=-1.0)
    out = _sorted(gpu.fit_transform(ray.data.from_pandas(df)))
    assert out["x"].isnull().sum() == 0
    assert (out["x"].to_numpy(float)[[2, 5]] == -1.0).all()


def test_serialization_roundtrip():
    df = _num_df()
    with patch.object(_gpu, "gpu_available", return_value=False):
        imp = GpuSimpleImputer(columns=["x"], strategy="mean")
        imp.fit(ray.data.from_pandas(df))
    restored = GpuSimpleImputer.deserialize(imp.serialize())
    assert isinstance(restored, GpuSimpleImputer)
    assert restored.strategy == "mean"
    assert restored.stats_["mean(x)"] == pytest.approx(imp.stats_["mean(x)"])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
