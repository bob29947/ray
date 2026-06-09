"""Tests for the experimental GPU standard scaler (``GpuStandardScaler``).

Two tiers:

* **Wiring / fallback** (no GPU): the operator is exported, serializes, and -
  when no GPU is available - is byte-identical to the CPU ``StandardScaler``.
  These run on a CPU-only host / CI.
* **GPU correctness** (``cudf`` + a visible CUDA device): ``GpuStandardScaler``
  is a faithful drop-in for ``StandardScaler`` -- per-column mean / population
  std fit, ``(x - mean) / std`` transform, constant columns -> zeros, integer
  columns widen to float. The oracle is the CPU scaler.

The GPU tests skip cleanly when the RAPIDS stack or a GPU is unavailable.
"""

import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import ray
from ray.data.preprocessors import GpuStandardScaler, StandardScaler, _gpu


def _gpu_stack_available() -> bool:
    try:
        import cudf  # noqa: F401
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


GPU = _gpu_stack_available()
requires_gpu = pytest.mark.skipif(not GPU, reason="needs cudf + a visible CUDA device")
NGPU = int(os.environ.get("RAY_DATA_TEST_GPU_PREPROC_NGPUS", "2"))


@pytest.fixture(scope="module")
def ray_cluster():
    os.environ["RAY_DATA_GPU_PREPROC_NUM_GPUS"] = str(NGPU)
    os.environ["RAY_DATA_GPU_PREPROC_BATCH_SIZE"] = "8192"
    ray.init(
        num_cpus=16,
        num_gpus=(NGPU if GPU else 0),
        include_dashboard=False,
        logging_level="ERROR",
    )
    try:
        yield
    finally:
        ray.shutdown()


def _numeric_df(n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "a": rng.normal(5.0, 2.0, n),  # float, non-trivial mean/std
            "b": rng.normal(-3.0, 10.0, n),  # float, larger spread
            "c": rng.integers(0, 100, n),  # integer -> widens to float
            "k": np.full(n, 7.0),  # constant -> scales to zeros
            "id": np.arange(n),
        }
    )


def _aligned(df):
    return df.sort_values("id").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Tier 1: wiring + CPU fallback (no GPU required)
# --------------------------------------------------------------------------- #
def test_exported():
    assert issubclass(GpuStandardScaler, StandardScaler)


def test_serialization_roundtrip():
    sc = GpuStandardScaler(columns=["a", "b"], output_columns=["a_s", "b_s"])
    restored = GpuStandardScaler.deserialize(sc.serialize())
    assert isinstance(restored, GpuStandardScaler)
    assert restored.columns == ["a", "b"]
    assert restored.output_columns == ["a_s", "b_s"]


def test_cpu_fallback_parity(ray_cluster, monkeypatch):
    # Force the CPU path and prove it equals the stock StandardScaler.
    monkeypatch.setattr(_gpu, "gpu_available", lambda: False)
    cols = ["a", "b", "c", "k"]
    ds = ray.data.from_pandas(_numeric_df()).repartition(8)
    cpu = _aligned(StandardScaler(columns=cols).fit_transform(ds).to_pandas())
    gpu = _aligned(GpuStandardScaler(columns=cols).fit_transform(ds).to_pandas())
    for c in cols:
        assert np.allclose(cpu[c].values, gpu[c].values, rtol=1e-6, atol=1e-8), c


# --------------------------------------------------------------------------- #
# Tier 2: GPU correctness vs the CPU StandardScaler oracle
# --------------------------------------------------------------------------- #
@requires_gpu
def test_gpu_parity_fit_transform(ray_cluster):
    cols = ["a", "b", "c", "k"]
    ds = ray.data.from_pandas(_numeric_df()).repartition(8)
    cpu = _aligned(StandardScaler(columns=cols).fit_transform(ds).to_pandas())
    gpu = _aligned(GpuStandardScaler(columns=cols).fit_transform(ds).to_pandas())
    for c in cols:
        assert np.allclose(cpu[c].values, gpu[c].values, rtol=1e-6, atol=1e-8), c


@requires_gpu
def test_gpu_stats_match_cpu(ray_cluster):
    # The fitted mean / (population) std must match the CPU aggregators.
    cols = ["a", "b", "c"]
    ds = ray.data.from_pandas(_numeric_df()).repartition(8)
    cpu = StandardScaler(columns=cols).fit(ds)
    gpu = GpuStandardScaler(columns=cols).fit(ds)
    for c in cols:
        assert np.isclose(
            cpu.stats_[f"mean({c})"], gpu.stats_[f"mean({c})"], rtol=1e-6, atol=1e-8
        ), c
        assert np.isclose(
            cpu.stats_[f"std({c})"], gpu.stats_[f"std({c})"], rtol=1e-6, atol=1e-8
        ), c


@requires_gpu
def test_gpu_constant_column_zeros(ray_cluster):
    # A constant column (std < epsilon) scales to zeros, like the CPU scaler.
    ds = ray.data.from_pandas(_numeric_df()).repartition(4)
    gpu = GpuStandardScaler(columns=["k"]).fit_transform(ds).to_pandas()
    assert np.allclose(gpu["k"].values, 0.0)


@requires_gpu
def test_gpu_output_columns_append(ray_cluster):
    cols = ["a", "b"]
    out = ["a_s", "b_s"]
    ds = ray.data.from_pandas(_numeric_df()).repartition(4)
    cpu = _aligned(
        StandardScaler(columns=cols, output_columns=out).fit_transform(ds).to_pandas()
    )
    gpu = _aligned(
        GpuStandardScaler(columns=cols, output_columns=out)
        .fit_transform(ds)
        .to_pandas()
    )
    # Original columns are preserved unchanged; new scaled columns match CPU.
    for c in cols:
        assert np.allclose(cpu[c].values, gpu[c].values, rtol=1e-6, atol=1e-8), c
    for c in out:
        assert np.allclose(cpu[c].values, gpu[c].values, rtol=1e-6, atol=1e-8), c


@requires_gpu
def test_gpu_integer_widens_to_float(ray_cluster):
    # Scaling an integer column yields float64 output (matches the Arrow path).
    ds = ray.data.from_pandas(_numeric_df()).repartition(4)
    gpu = (
        GpuStandardScaler(columns=["c"]).fit_transform(ds).to_pandas().sort_values("id")
    )
    assert gpu["c"].dtype.kind == "f"
    assert abs(gpu["c"].mean()) < 1e-6
    assert abs(gpu["c"].std(ddof=0) - 1.0) < 1e-6


@requires_gpu
def test_gpu_compose_cpu_and_gpu_ops(ray_cluster):
    # op-before (CPU map) -> GPU scale -> op-after (CPU map): verifies the GPU
    # scaler composes with other operators and that payload columns survive.
    ds = ray.data.from_pandas(_numeric_df()).repartition(8)

    def add_pre(b: pd.DataFrame) -> pd.DataFrame:
        b = b.copy()
        b["pre"] = b["id"] % 7
        return b

    pre = ds.map_batches(add_pre, batch_format="pandas")
    scaled = GpuStandardScaler(columns=["a", "b"]).fit_transform(pre)

    def add_post(b: pd.DataFrame) -> pd.DataFrame:
        b = b.copy()
        b["ok"] = b["a"].notna() & b["b"].notna()
        return b

    out = scaled.map_batches(add_post, batch_format="pandas").to_pandas()
    assert "pre" in out.columns  # before-op column survived the GPU stage
    assert out["ok"].all()
    for c in ["a", "b"]:
        assert abs(out[c].mean()) < 1e-6, c
        assert abs(out[c].std(ddof=0) - 1.0) < 1e-6, c


@requires_gpu
def test_gpu_arrow_input_parity(ray_cluster):
    # Arrow-backed dataset (real nulls absent) scales identically to CPU.
    n = 20_000
    rng = np.random.default_rng(3)
    tbl = pa.table(
        {
            "x": pa.array(rng.normal(2.0, 5.0, n)),
            "y": pa.array(rng.integers(-50, 50, n)),
            "id": pa.array(np.arange(n, dtype=np.int64)),
        }
    )
    ds = ray.data.from_arrow(tbl).repartition(8)
    cpu = _aligned(StandardScaler(columns=["x", "y"]).fit_transform(ds).to_pandas())
    gpu = _aligned(GpuStandardScaler(columns=["x", "y"]).fit_transform(ds).to_pandas())
    for c in ["x", "y"]:
        assert np.allclose(cpu[c].values, gpu[c].values, rtol=1e-6, atol=1e-8), c


def test_zero_regression_standard_scaler_unchanged(ray_cluster):
    # The presence of the GPU subclass must not change default CPU behavior.
    ds = ray.data.from_pandas(_numeric_df()).repartition(4)
    sc = StandardScaler(columns=["a"]).fit(ds)
    out = sc.transform(ds).to_pandas()
    assert abs(out["a"].mean()) < 1e-6
    assert abs(out["a"].std(ddof=0) - 1.0) < 1e-6


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
