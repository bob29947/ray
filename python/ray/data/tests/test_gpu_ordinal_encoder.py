"""Tests for the experimental GPU ordinal encoder (``GpuOrdinalEncoder``).

Two tiers:

* **Wiring / fallback** (no GPU): the operator is exported, serializes, and -
  when no GPU is available - is byte-identical to the CPU ``OrdinalEncoder``.
  These run on a CPU-only host / CI.
* **GPU correctness** (``cudf`` + a visible CUDA device): ``GpuOrdinalEncoder``
  is a faithful drop-in for ``OrdinalEncoder`` (sorted-vocabulary codes,
  unseen -> NaN, append mode, multi-block). The oracle is the CPU encoder.

The GPU tests skip cleanly when the RAPIDS stack or a GPU is unavailable.
"""

import os

import numpy as np
import pandas as pd
import pytest

import ray
from ray.data.preprocessors import GpuOrdinalEncoder, OrdinalEncoder, _gpu


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


def _categorical_df(n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "item": rng.choice([f"i{i}" for i in range(50)], n),
            "dev": rng.choice(["ios", "android", "web"], n),
            "country": rng.choice(["US", "CA", "GB", "DE"], n),
            "id": np.arange(n),
        }
    )


def _aligned(df):
    return df.sort_values("id").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Tier 1: wiring + CPU fallback (no GPU required)
# --------------------------------------------------------------------------- #
def test_exported():
    from ray.data.preprocessors import GpuOrdinalEncoder as G

    assert issubclass(G, OrdinalEncoder)


def test_serialization_roundtrip():
    enc = GpuOrdinalEncoder(columns=["a", "b"], output_columns=["a_e", "b_e"])
    restored = GpuOrdinalEncoder.deserialize(enc.serialize())
    assert isinstance(restored, GpuOrdinalEncoder)
    assert restored.columns == ["a", "b"]
    assert restored.output_columns == ["a_e", "b_e"]


def test_cpu_fallback_parity(ray_cluster, monkeypatch):
    # Force the CPU path and prove it equals the stock OrdinalEncoder.
    monkeypatch.setattr(_gpu, "gpu_available", lambda: False)
    cols = ["item", "dev", "country"]
    ds = ray.data.from_pandas(_categorical_df()).repartition(8)
    cpu = _aligned(OrdinalEncoder(columns=cols).fit_transform(ds).to_pandas())
    gpu = _aligned(GpuOrdinalEncoder(columns=cols).fit_transform(ds).to_pandas())
    for c in cols:
        assert (cpu[c] == gpu[c]).all()


# --------------------------------------------------------------------------- #
# Tier 2: GPU correctness vs the CPU OrdinalEncoder oracle
# --------------------------------------------------------------------------- #
@requires_gpu
def test_gpu_parity_fit_transform(ray_cluster):
    cols = ["item", "dev", "country"]
    ds = ray.data.from_pandas(_categorical_df()).repartition(8)
    cpu = _aligned(OrdinalEncoder(columns=cols).fit_transform(ds).to_pandas())
    gpu = _aligned(GpuOrdinalEncoder(columns=cols).fit_transform(ds).to_pandas())
    for c in cols:
        assert (cpu[c] == gpu[c]).all(), c


@requires_gpu
def test_gpu_unseen_becomes_nan(ray_cluster):
    cols = ["item", "dev", "country"]
    train = ray.data.from_pandas(_categorical_df()).repartition(8)
    enc_cpu = OrdinalEncoder(columns=cols)
    enc_cpu.fit(train)
    enc_gpu = GpuOrdinalEncoder(columns=cols)
    enc_gpu.fit(train)
    new = pd.DataFrame(
        {
            "item": ["i1", "UNSEEN"],
            "dev": ["ios", "ZZ"],
            "country": ["US", "CA"],
            "id": [0, 1],
        }
    )
    ds = ray.data.from_pandas(new)
    cpu = _aligned(enc_cpu.transform(ds).to_pandas())
    gpu = _aligned(enc_gpu.transform(ds).to_pandas())
    for c in cols:
        # Compare with NaN-safe fill.
        assert (cpu[c].fillna(-999) == gpu[c].fillna(-999)).all(), c
    # The unseen row's codes must be NaN.
    assert gpu.loc[gpu["id"] == 1, ["item", "dev"]].isna().all().all()


@requires_gpu
def test_gpu_output_columns_append(ray_cluster):
    cols = ["item", "dev"]
    out = ["item_e", "dev_e"]
    ds = ray.data.from_pandas(_categorical_df()).repartition(4)
    cpu = _aligned(
        OrdinalEncoder(columns=cols, output_columns=out)
        .fit_transform(ds)
        .to_pandas()
    )
    gpu = _aligned(
        GpuOrdinalEncoder(columns=cols, output_columns=out)
        .fit_transform(ds)
        .to_pandas()
    )
    # Original columns preserved, new code columns equal the CPU result.
    for c in cols:
        assert (cpu[c] == gpu[c]).all()
    for c in out:
        assert (cpu[c] == gpu[c]).all()


@requires_gpu
def test_gpu_integer_categorical(ray_cluster):
    # Integer categorical keys must encode by sorted numeric order, like CPU.
    rng = np.random.default_rng(7)
    n = 30_000
    df = pd.DataFrame({"k": rng.integers(0, 100, n), "id": np.arange(n)})
    ds = ray.data.from_pandas(df).repartition(4)
    cpu = _aligned(OrdinalEncoder(columns=["k"]).fit_transform(ds).to_pandas())
    gpu = _aligned(GpuOrdinalEncoder(columns=["k"]).fit_transform(ds).to_pandas())
    assert (cpu["k"] == gpu["k"]).all()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
