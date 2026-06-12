"""Parity / fallback / serialization tests for GpuStandardScaler.

The GPU parity tests are skipped automatically when no GPU + RAPIDS stack is
available; the CPU-fallback and serialization tests run anywhere.
"""

import os

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

import ray
from ray.data.preprocessors import StandardScaler
from ray.data.preprocessors import _gpu
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
            "id": range(10),
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "y": [-2.0, 0.0, 2.0, -2.0, 0.0, 2.0, -2.0, 0.0, 2.0, 0.0],
        }
    )


def _sorted(ds):
    return ds.to_pandas().sort_values("id").reset_index(drop=True)


def _assert_close(a, b, cols):
    for c in cols:
        np.testing.assert_allclose(
            pd.to_numeric(a[c]).to_numpy(float),
            pd.to_numeric(b[c]).to_numpy(float),
            rtol=1e-6,
            atol=1e-6,
            equal_nan=True,
        )


def test_cpu_fallback_matches_cpu():
    """With no GPU, GpuStandardScaler is byte-identical to StandardScaler."""
    df = _df()
    with patch.object(_gpu, "gpu_available", return_value=False):
        cpu = StandardScaler(columns=["x", "y"]).fit_transform(ray.data.from_pandas(df))
        gpu = GpuStandardScaler(columns=["x", "y"]).fit_transform(
            ray.data.from_pandas(df)
        )
    _assert_close(_sorted(cpu), _sorted(gpu), ["x", "y"])


def test_cpu_fallback_stats_match():
    df = _df()
    cpu = StandardScaler(columns=["x", "y"])
    cpu.fit(ray.data.from_pandas(df))
    with patch.object(_gpu, "gpu_available", return_value=False):
        gpu = GpuStandardScaler(columns=["x", "y"])
        gpu.fit(ray.data.from_pandas(df))
    for c in ["x", "y"]:
        assert gpu.stats_[f"mean({c})"] == pytest.approx(cpu.stats_[f"mean({c})"])
        assert gpu.stats_[f"std({c})"] == pytest.approx(cpu.stats_[f"std({c})"])


@requires_gpu
def test_gpu_matches_cpu():
    df = _df()
    cpu = StandardScaler(columns=["x", "y"]).fit_transform(ray.data.from_pandas(df))
    gpu = GpuStandardScaler(columns=["x", "y"]).fit_transform(ray.data.from_pandas(df))
    _assert_close(_sorted(cpu), _sorted(gpu), ["x", "y"])


@requires_gpu
def test_gpu_stats_match_cpu():
    df = _df()
    cpu = StandardScaler(columns=["x", "y"])
    cpu.fit(ray.data.from_pandas(df))
    gpu = GpuStandardScaler(columns=["x", "y"])
    gpu.fit(ray.data.from_pandas(df))
    for c in ["x", "y"]:
        assert gpu.stats_[f"mean({c})"] == pytest.approx(cpu.stats_[f"mean({c})"])
        assert gpu.stats_[f"std({c})"] == pytest.approx(cpu.stats_[f"std({c})"])


@requires_gpu
def test_gpu_output_columns_append():
    df = _df()
    gpu = GpuStandardScaler(columns=["x"], output_columns=["x_scaled"]).fit_transform(
        ray.data.from_pandas(df)
    )
    out = _sorted(gpu)
    assert "x_scaled" in out.columns
    # Original column is preserved unchanged in append mode.
    np.testing.assert_allclose(out["x"].to_numpy(float), df["x"].to_numpy(float))


def test_serialization_roundtrip():
    df = _df()
    with patch.object(_gpu, "gpu_available", return_value=False):
        sc = GpuStandardScaler(columns=["x", "y"])
        sc.fit(ray.data.from_pandas(df))
    restored = GpuStandardScaler.deserialize(sc.serialize())
    assert isinstance(restored, GpuStandardScaler)
    assert restored.columns == ["x", "y"]
    for c in ["x", "y"]:
        assert restored.stats_[f"mean({c})"] == pytest.approx(sc.stats_[f"mean({c})"])
        assert restored.stats_[f"std({c})"] == pytest.approx(sc.stats_[f"std({c})"])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
