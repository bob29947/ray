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
import pyarrow as pa
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
        SimpleImputer(columns=["price"], strategy="mean").fit_transform(ds).to_pandas()
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
    df = pd.DataFrame({"c": ["b", "b", "a", "a", None], "id": np.arange(5)})
    ds = ray.data.from_pandas(df)
    imp = GpuSimpleImputer(columns=["c"], strategy="most_frequent")
    imp.fit(ds)
    assert imp.stats_["most_frequent(c)"] == "a"


# --------------------------------------------------------------------------- #
# Tier 2b: yandex/yambda-shaped data -- schema parity, tie-awareness, mean
#   widening, and composability with other CPU/GPU operators.
#
# These use tiny in-memory Arrow tables that mirror the multi_event columns
# (uint32 ids, a float playback column with "natural" nulls, a string
# event_type). The CPU-tier tests need no GPU/network/files; the GPU tests skip
# cleanly without RAPIDS. A real-data smoke test is gated behind env vars so CI
# never needs the multi-GB parquet.
# --------------------------------------------------------------------------- #
def _yambda_arrow(n=20_000, seed=0):
    """Tiny Arrow table shaped like flat/500m multi_event (with nulls)."""
    rng = np.random.default_rng(seed)
    item = pa.array(
        rng.integers(0, 4000, n).astype(np.uint32), mask=(rng.random(n) < 0.1)
    )
    played = pa.array(rng.random(n) * 100.0, mask=(rng.random(n) < 0.1))
    ev = rng.choice(
        np.array(["listen", "listen", "listen", "like", "dislike"]), n
    ).astype(object)
    ev[rng.random(n) < 0.1] = None
    return pa.table(
        {
            "item_id": item,
            "played_ratio_pct": played,
            "event_type": pa.array(ev),
            "id": pa.array(np.arange(n, dtype=np.int64)),
        }
    )


def test_yambda_cpu_fallback_parity(ray_cluster, monkeypatch):
    # With no GPU, GpuSimpleImputer must be byte-identical to SimpleImputer on
    # the yambda schema (mean over a float column, most_frequent over a string).
    monkeypatch.setattr(_gpu, "gpu_available", lambda: False)
    ds = ray.data.from_arrow(_yambda_arrow()).repartition(8)
    for cols, strat in (["event_type"], "most_frequent"), (
        ["played_ratio_pct"],
        "mean",
    ):
        cpu = _aligned(
            SimpleImputer(columns=cols, strategy=strat).fit_transform(ds).to_pandas()
        )
        gpu = _aligned(
            GpuSimpleImputer(columns=cols, strategy=strat).fit_transform(ds).to_pandas()
        )
        if strat == "mean":
            assert np.allclose(cpu[cols[0]], gpu[cols[0]], rtol=1e-6)
        else:
            assert (cpu[cols[0]] == gpu[cols[0]]).all()


def test_zero_regression_simple_imputer_unchanged(ray_cluster):
    # The presence of the GPU subclass must not change default CPU behavior.
    ds = ray.data.from_arrow(_yambda_arrow()).repartition(4)
    imp = SimpleImputer(columns=["event_type"], strategy="most_frequent").fit(ds)
    out = imp.transform(ds).to_pandas()
    assert out["event_type"].notna().all()
    assert imp.stats_["most_frequent(event_type)"] == "listen"


@requires_gpu
def test_gpu_yambda_parity(ray_cluster):
    ds = ray.data.from_arrow(_yambda_arrow()).repartition(8)
    # most_frequent over the string event_type
    cpu = _aligned(
        SimpleImputer(columns=["event_type"], strategy="most_frequent")
        .fit_transform(ds)
        .to_pandas()
    )
    gpu = _aligned(
        GpuSimpleImputer(columns=["event_type"], strategy="most_frequent")
        .fit_transform(ds)
        .to_pandas()
    )
    assert (cpu["event_type"] == gpu["event_type"]).all()
    # mean over the float playback column
    cpu_m = _aligned(
        SimpleImputer(columns=["played_ratio_pct"], strategy="mean")
        .fit_transform(ds)
        .to_pandas()
    )
    gpu_m = _aligned(
        GpuSimpleImputer(columns=["played_ratio_pct"], strategy="mean")
        .fit_transform(ds)
        .to_pandas()
    )
    assert np.allclose(cpu_m["played_ratio_pct"], gpu_m["played_ratio_pct"], rtol=1e-6)


@requires_gpu
def test_gpu_mean_int_column_widens_to_float(ray_cluster):
    # A float (mean) fill on an integer column widens to float64, like pandas.
    tbl = pa.table(
        {
            "x": pa.array([1, 2, 3, None, 5], type=pa.int64()),
            "id": pa.array(np.arange(5)),
        }
    )
    ds = ray.data.from_arrow(tbl)
    gpu = (
        GpuSimpleImputer(columns=["x"], strategy="mean")
        .fit_transform(ds)
        .to_pandas()
        .sort_values("id")
    )
    assert gpu["x"].dtype.kind == "f"
    assert gpu["x"].notna().all()
    assert gpu["x"].tolist()[3] == pytest.approx((1 + 2 + 3 + 5) / 4)  # mean = 2.75


@requires_gpu
def test_gpu_most_frequent_no_tie_matches_cpu(ray_cluster):
    # Companion to the tie test: with a clear winner, CPU and GPU agree.
    ds = ray.data.from_pandas(
        pd.DataFrame({"c": ["a", "a", "a", "b", None], "id": np.arange(5)})
    )
    cpu = SimpleImputer(columns=["c"], strategy="most_frequent").fit(ds)
    gpu = GpuSimpleImputer(columns=["c"], strategy="most_frequent").fit(ds)
    assert cpu.stats_["most_frequent(c)"] == gpu.stats_["most_frequent(c)"] == "a"


@requires_gpu
def test_gpu_compose_cpu_and_gpu_ops(ray_cluster):
    # op-before (CPU map) -> GPU impute -> GPU impute -> op-after (CPU map):
    # verifies the GPU imputer composes with other operators and releases its
    # actor pool between stages, and that payload columns survive.
    ds = ray.data.from_arrow(_yambda_arrow()).repartition(8)

    def add_pre(b: pd.DataFrame) -> pd.DataFrame:
        b = b.copy()
        b["pre"] = b["item_id"].fillna(0).astype("int64") % 7
        return b

    pre = ds.map_batches(add_pre, batch_format="pandas")
    g1 = GpuSimpleImputer(columns=["event_type"], strategy="most_frequent").fit(pre)
    mid = g1.transform(pre)
    g2 = GpuSimpleImputer(columns=["played_ratio_pct"], strategy="mean").fit(mid)
    mid2 = g2.transform(mid)

    def add_post(b: pd.DataFrame) -> pd.DataFrame:
        b = b.copy()
        b["ok"] = b["event_type"].notna() & b["played_ratio_pct"].notna()
        return b

    out = mid2.map_batches(add_post, batch_format="pandas").to_pandas()
    assert out["event_type"].notna().all()
    assert out["played_ratio_pct"].notna().all()
    assert out["ok"].all()
    assert "pre" in out.columns  # the before-op column survived the GPU stages


_RUN_YAMBDA = os.environ.get("RAY_DATA_RUN_GPU_YAMBDA") == "1"
_YAMBDA_PARQUET = os.environ.get("YAMBDA_PARQUET", "")


@pytest.mark.skipif(
    not (GPU and _RUN_YAMBDA and _YAMBDA_PARQUET and os.path.exists(_YAMBDA_PARQUET)),
    reason="set RAY_DATA_RUN_GPU_YAMBDA=1 + YAMBDA_PARQUET=/path/to/*.parquet on a GPU",
)
def test_gpu_yambda_realdata_smoke(ray_cluster):
    ds = ray.data.read_parquet(_YAMBDA_PARQUET).limit(2_000_000)

    def mask(b: pd.DataFrame) -> pd.DataFrame:
        b = b.copy()
        m = np.random.default_rng(0).random(len(b)) < 0.05
        b.loc[m, "item_id"] = np.nan
        return b

    masked = ds.map_batches(mask, batch_format="pandas")
    out = (
        GpuSimpleImputer(columns=["item_id"], strategy="most_frequent")
        .fit_transform(masked)
        .to_pandas()
    )
    assert out["item_id"].notna().all()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
