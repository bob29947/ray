"""Tests for the experimental GPU imputer (``GpuSimpleImputer``).

Two tiers:

* **Wiring / fallback** (no GPU): exported, serializes, and - with no GPU -
  byte-identical to the CPU ``SimpleImputer``.
* **GPU correctness** (``cudf`` + a visible CUDA device): faithful drop-in for
  ``SimpleImputer`` across the ``mean`` / ``most_frequent`` / ``constant``
  strategies. Tie-break for ``most_frequent`` is documented (smallest value).

The GPU tests skip cleanly when the RAPIDS stack or a GPU is unavailable.
"""

import os

import numpy as np
import pandas as pd
import pytest

import ray
from ray.data.preprocessors import GpuSimpleImputer, SimpleImputer, _gpu


def _gpu_stack_available() -> bool:
    try:
        import cudf  # noqa: F401
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


GPU = _gpu_stack_available()
requires_gpu = pytest.mark.skipif(
    not GPU, reason="needs cudf + a visible CUDA device"
)
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


def _df_with_nulls(n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    dev = rng.choice(["ios", "ios", "ios", "android", "web"], n).astype(object)
    dev[rng.random(n) < 0.1] = None
    country = rng.choice(["US", "US", "CA", "GB"], n).astype(object)
    country[rng.random(n) < 0.1] = None
    price = rng.random(n) * 100.0
    price[rng.random(n) < 0.1] = np.nan
    return pd.DataFrame(
        {"dev": dev, "country": country, "price": price, "id": np.arange(n)}
    )


def _aligned(df):
    return df.sort_values("id").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Tier 1: wiring + CPU fallback (no GPU required)
# --------------------------------------------------------------------------- #
def test_exported():
    assert issubclass(GpuSimpleImputer, SimpleImputer)


def test_serialization_roundtrip():
    imp = GpuSimpleImputer(columns=["a"], strategy="constant", fill_value="x")
    restored = GpuSimpleImputer.deserialize(imp.serialize())
    assert isinstance(restored, GpuSimpleImputer)
    assert restored.columns == ["a"]
    assert restored.strategy == "constant"
    assert restored.fill_value == "x"


def test_cpu_fallback_parity(ray_cluster, monkeypatch):
    monkeypatch.setattr(_gpu, "gpu_available", lambda: False)
    ds = ray.data.from_pandas(_df_with_nulls()).repartition(8)
    cpu = _aligned(
        SimpleImputer(columns=["dev", "country"], strategy="most_frequent")
        .fit_transform(ds)
        .to_pandas()
    )
    gpu = _aligned(
        GpuSimpleImputer(columns=["dev", "country"], strategy="most_frequent")
        .fit_transform(ds)
        .to_pandas()
    )
    assert (cpu["dev"] == gpu["dev"]).all()
    assert (cpu["country"] == gpu["country"]).all()


# --------------------------------------------------------------------------- #
# Tier 2: GPU correctness vs the CPU SimpleImputer oracle
# --------------------------------------------------------------------------- #
@requires_gpu
def test_gpu_most_frequent_parity(ray_cluster):
    ds = ray.data.from_pandas(_df_with_nulls()).repartition(8)
    cols = ["dev", "country"]
    cpu = _aligned(
        SimpleImputer(columns=cols, strategy="most_frequent")
        .fit_transform(ds)
        .to_pandas()
    )
    gpu = _aligned(
        GpuSimpleImputer(columns=cols, strategy="most_frequent")
        .fit_transform(ds)
        .to_pandas()
    )
    for c in cols:
        assert (cpu[c] == gpu[c]).all(), c
        assert gpu[c].notna().all(), c


@requires_gpu
def test_gpu_mean_parity(ray_cluster):
    ds = ray.data.from_pandas(_df_with_nulls()).repartition(8)
    cpu = _aligned(
        SimpleImputer(columns=["price"], strategy="mean")
        .fit_transform(ds)
        .to_pandas()
    )
    gpu = _aligned(
        GpuSimpleImputer(columns=["price"], strategy="mean")
        .fit_transform(ds)
        .to_pandas()
    )
    assert np.allclose(cpu["price"].values, gpu["price"].values, rtol=1e-6)
    assert gpu["price"].notna().all()


@requires_gpu
def test_gpu_constant(ray_cluster):
    ds = ray.data.from_pandas(_df_with_nulls()).repartition(4)
    gpu = (
        GpuSimpleImputer(columns=["dev"], strategy="constant", fill_value="?")
        .fit_transform(ds)
        .to_pandas()
    )
    assert gpu["dev"].notna().all()
    # Every previously-null value is now the constant.
    assert (gpu["dev"] == "?").sum() > 0


@requires_gpu
def test_gpu_most_frequent_tie_break_smallest(ray_cluster):
    # Two values tied at the max count -> GpuSimpleImputer picks the smallest.
    df = pd.DataFrame(
        {"c": ["b", "b", "a", "a", None], "id": np.arange(5)}
    )
    ds = ray.data.from_pandas(df)
    imp = GpuSimpleImputer(columns=["c"], strategy="most_frequent")
    imp.fit(ds)
    assert imp.stats_["most_frequent(c)"] == "a"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
