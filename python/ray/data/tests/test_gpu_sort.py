"""Tests for the experimental multi-GPU sort backend (``ds.sort(gpu=True)``).

Two tiers:

* **Policy / wiring / zero-regression** (no GPU): exercise ``_resolve_gpu_impl``,
  the ``Dataset.sort(gpu=, backend=)`` argument validation, and prove the
  default CPU sort path is byte-identical whether or not the GPU plumbing is
  present. These run on a CPU-only host / CI.
* **GPU correctness** (``cudf`` + ``rapidsmpf`` + a visible GPU): prove
  ``ds.sort(gpu=True)`` (the cuDF + rapidsmpf "general" engine) is a faithful
  drop-in for the CPU sort across dtypes, multiple keys with mixed directions,
  nulls, NaN, strings, datetime, empty/0-row and many-block inputs. The oracle
  is an independent pandas sort with ``na_position="last"`` (Ray's default), so
  the test never compares the engine to itself.

The GPU tests skip cleanly when the RAPIDS stack or a GPU is unavailable.
"""

import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import ray
from ray.data._internal.planner.sort import _resolve_gpu_impl


# --------------------------------------------------------------------------- #
# GPU availability detection
# --------------------------------------------------------------------------- #
def _gpu_stack_available() -> bool:
    try:
        import cudf  # noqa: F401
        import cupy  # noqa: F401
        import rapidsmpf  # noqa: F401

        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


GPU = _gpu_stack_available()
requires_gpu = pytest.mark.skipif(
    not GPU, reason="needs cudf + rapidsmpf + a visible CUDA device"
)
# Small rank count keeps the per-test UCXX cluster cheap; the engine is correct
# for any number of ranks.
NGPU = int(os.environ.get("RAY_DATA_TEST_GPU_SORT_NGPUS", "2"))


# --------------------------------------------------------------------------- #
# Tier 1: routing policy (pure, no Ray / no GPU)
# --------------------------------------------------------------------------- #
class TestResolveGpuImpl:
    @pytest.fixture(autouse=True)
    def _clear_env(self, monkeypatch):
        monkeypatch.delenv("RAY_DATA_GPU_SORT", raising=False)
        monkeypatch.delenv("RAY_DATA_GPU_SORT_IMPL", raising=False)

    def test_default_is_cpu(self):
        assert _resolve_gpu_impl(None) is None

    def test_flag_true_defaults_to_general(self):
        assert _resolve_gpu_impl(True) == "general"

    def test_flag_false_forces_cpu_even_with_env(self, monkeypatch):
        monkeypatch.setenv("RAY_DATA_GPU_SORT", "1")
        monkeypatch.setenv("RAY_DATA_GPU_SORT_IMPL", "general")
        assert _resolve_gpu_impl(False) is None

    def test_flag_true_honors_impl_env(self, monkeypatch):
        monkeypatch.setenv("RAY_DATA_GPU_SORT_IMPL", "tuned")
        assert _resolve_gpu_impl(True) == "tuned"

    def test_legacy_env_on_defaults_to_tuned(self, monkeypatch):
        monkeypatch.setenv("RAY_DATA_GPU_SORT", "1")
        assert _resolve_gpu_impl(None) == "tuned"

    def test_impl_env_alone_selects_backend(self, monkeypatch):
        monkeypatch.setenv("RAY_DATA_GPU_SORT_IMPL", "general")
        assert _resolve_gpu_impl(None) == "general"

    def test_garbage_impl_env_is_ignored(self, monkeypatch):
        monkeypatch.setenv("RAY_DATA_GPU_SORT_IMPL", "bogus")
        assert _resolve_gpu_impl(None) is None
        assert _resolve_gpu_impl(True) == "general"


# --------------------------------------------------------------------------- #
# Shared Ray fixture: a GPU cluster when available, else CPU-only so the
# policy / regression tests still run on CI.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def ray_cluster():
    os.environ["RAY_DATA_GPU_SORT_NUM_GPUS"] = str(NGPU)
    ray.init(
        num_cpus=16,
        num_gpus=(NGPU if GPU else 0),
        include_dashboard=False,
        logging_level="ERROR",
    )
    try:
        yield
    finally:
        if GPU:
            try:
                from ray.data._internal.planner.gpu_sort_general import (
                    kill_actor_pool,
                )

                kill_actor_pool(NGPU)
            except Exception:
                pass
        ray.shutdown()


# --------------------------------------------------------------------------- #
# Tier 1b: Dataset.sort argument validation + zero-regression (no GPU needed)
# --------------------------------------------------------------------------- #
def test_sort_backend_arg_validation(ray_cluster):
    ds = ray.data.range(16)
    with pytest.raises(ValueError, match="must be 'cpu' or 'gpu'"):
        ds.sort("id", backend="cuda")
    with pytest.raises(ValueError, match="only one of"):
        ds.sort("id", gpu=True, backend="gpu")


def test_default_sort_unchanged_zero_regression(ray_cluster):
    # gpu unset and gpu=False must produce the identical CPU result; the
    # default path must be untouched by the GPU plumbing.
    ds = ray.data.range(10_000).map(lambda r: {"id": (r["id"] * 7919) % 10_007})
    want = sorted(r["id"] for r in ds.take_all())
    got_default = [r["id"] for r in ds.sort("id").take_all()]
    got_cpu = [r["id"] for r in ds.sort("id", gpu=False).take_all()]
    assert got_default == want
    assert got_cpu == want


def test_sort_op_carries_gpu_flag(ray_cluster):
    # The gpu flag must be plumbed onto the Sort logical operator.
    from ray.data._internal.logical.operators.all_to_all_operator import Sort

    dag_true = ray.data.range(16).sort("id", gpu=True)._logical_plan.dag
    dag_def = ray.data.range(16).sort("id")._logical_plan.dag
    assert isinstance(dag_true, Sort) and dag_true.gpu is True
    assert isinstance(dag_def, Sort) and dag_def.gpu is None


# --------------------------------------------------------------------------- #
# Tier 2: GPU correctness vs an independent pandas oracle
# --------------------------------------------------------------------------- #
def _oracle(pdf: pd.DataFrame, keys, ascending):
    """Independent pandas sort with Ray's default null placement (last)."""
    return pdf.sort_values(
        by=list(keys), ascending=list(ascending), na_position="last",
        kind="stable",
    ).reset_index(drop=True)


def _gpu_sort_to_pandas(blocks, keys, descending):
    ds = ray.data.from_arrow(blocks)
    out = ds.sort(keys if len(keys) > 1 else keys[0],
                  descending=descending if len(keys) > 1 else descending[0],
                  gpu=True)
    rows = out.take_all()
    # Preserve column order from the first block's schema.
    cols = blocks[0].schema.names
    return pd.DataFrame(rows)[cols]


def _assert_keys_match(got: pd.DataFrame, oracle: pd.DataFrame, keys):
    # Compare the key columns in order (payload row-integrity is checked
    # separately via a unique id where relevant).
    g = got[list(keys)].reset_index(drop=True)
    o = oracle[list(keys)].reset_index(drop=True)
    pd.testing.assert_frame_equal(g, o, check_dtype=False)


@requires_gpu
def test_int_single_key_ascending(ray_cluster):
    rng = np.random.default_rng(0)
    n = 2_000_000
    c0 = rng.integers(0, 2**31 - 1, n, dtype=np.int32)
    ids = np.arange(n, dtype=np.int64)
    tbl = pa.table({"c0": c0, "id": ids})
    blocks = [tbl.slice(i * (n // 8), n // 8) for i in range(8)]
    got = _gpu_sort_to_pandas(blocks, ["c0"], [False])
    oracle = _oracle(tbl.to_pandas(), ["c0"], [True])
    _assert_keys_match(got, oracle, ["c0"])
    # Row integrity: the (c0, id) multiset is preserved.
    assert sorted(map(tuple, got[["c0", "id"]].values.tolist())) == \
        sorted(map(tuple, tbl.to_pandas()[["c0", "id"]].values.tolist()))


@requires_gpu
def test_int_single_key_descending(ray_cluster):
    rng = np.random.default_rng(1)
    n = 500_000
    tbl = pa.table({"k": rng.integers(-(2**20), 2**20, n, dtype=np.int64)})
    blocks = [tbl.slice(i * (n // 4), n // 4) for i in range(4)]
    got = _gpu_sort_to_pandas(blocks, ["k"], [True])
    oracle = _oracle(tbl.to_pandas(), ["k"], [False])
    _assert_keys_match(got, oracle, ["k"])


@requires_gpu
def test_multi_key_mixed_directions(ray_cluster):
    rng = np.random.default_rng(2)
    n = 400_000
    tbl = pa.table({
        "a": rng.integers(0, 10, n, dtype=np.int32),
        "b": rng.integers(0, 1000, n, dtype=np.int32),
        "c": rng.random(n),
    })
    blocks = [tbl.slice(i * (n // 4), n // 4) for i in range(4)]
    got = _gpu_sort_to_pandas(blocks, ["a", "b", "c"], [False, True, False])
    oracle = _oracle(tbl.to_pandas(), ["a", "b", "c"], [True, False, True])
    _assert_keys_match(got, oracle, ["a", "b", "c"])


@requires_gpu
def test_floats_with_nan_sorted_last(ray_cluster):
    # NaN must be ordered like null (at the end), matching pyarrow's default.
    rng = np.random.default_rng(3)
    n = 200_000
    f = rng.random(n)
    f[rng.integers(0, n, n // 20)] = np.nan
    tbl = pa.table({"f": f})
    blocks = [tbl.slice(i * (n // 4), n // 4) for i in range(4)]
    ds = ray.data.from_arrow(blocks)
    got = [r["f"] for r in ds.sort("f", gpu=True).take_all()]
    # All non-NaN ascending, then NaNs at the end.
    non_nan = [x for x in got if not (x != x)]
    n_nan = sum(1 for x in got if x != x)
    assert non_nan == sorted(non_nan)
    assert [x != x for x in got][-n_nan:] == [True] * n_nan if n_nan else True


@requires_gpu
def test_null_int_key_at_end(ray_cluster):
    n = 300_000
    rng = np.random.default_rng(4)
    vals = rng.integers(0, 1000, n).astype(object)
    mask = rng.random(n) < 0.1
    arr = pa.array([None if m else int(v) for v, m in zip(vals, mask)], type=pa.int64())
    tbl = pa.table({"k": arr})
    blocks = [tbl.slice(i * (n // 4), n // 4) for i in range(4)]
    ds = ray.data.from_arrow(blocks)
    got = [r["k"] for r in ds.sort("k", gpu=True).take_all()]
    n_null = sum(1 for x in got if x is None)
    non_null = [x for x in got if x is not None]
    assert non_null == sorted(non_null)
    assert got[-n_null:] == [None] * n_null if n_null else True


@requires_gpu
def test_string_multikey_with_nulls(ray_cluster):
    rng = np.random.default_rng(5)
    n = 200_000
    words = ["apple", "banana", "cherry", "date", "fig", "grape", None]
    s = [words[i] for i in rng.integers(0, len(words), n)]
    f = rng.random(n)
    f[rng.integers(0, n, n // 20)] = np.nan
    # Use a real Arrow null for the float key too (NaN handled separately).
    f_arr = pa.array([None if x != x else x for x in f], type=pa.float64())
    tbl = pa.table({"s": pa.array(s, type=pa.string()), "f": f_arr})
    blocks = [tbl.slice(i * (n // 4), n // 4) for i in range(4)]
    got = _gpu_sort_to_pandas(blocks, ["s", "f"], [False, True])
    oracle = _oracle(tbl.to_pandas(), ["s", "f"], [True, False])
    _assert_keys_match(got, oracle, ["s", "f"])


@requires_gpu
def test_datetime_key(ray_cluster):
    rng = np.random.default_rng(6)
    n = 300_000
    base = np.datetime64("2026-01-03")
    days = rng.integers(-3000, 3000, n)
    ts = (base + days.astype("timedelta64[D]")).astype("datetime64[ns]")
    tbl = pa.table({"t": pa.array(ts)})
    blocks = [tbl.slice(i * (n // 6), n // 6) for i in range(6)]
    ds = ray.data.from_arrow(blocks)
    got = [r["t"] for r in ds.sort("t", gpu=True).take_all()]
    assert got == sorted(got)
    assert len(got) == tbl.num_rows


@requires_gpu
def test_empty_and_single_row(ray_cluster):
    # 0-row block.
    empty = pa.table({"c0": pa.array([], type=pa.int64())})
    ds = ray.data.from_arrow([empty])
    assert ds.sort("c0", gpu=True).count() == 0
    # 1-row block.
    one = pa.table({"c0": pa.array([42], type=pa.int64())})
    ds = ray.data.from_arrow([one])
    assert [r["c0"] for r in ds.sort("c0", gpu=True).take_all()] == [42]


@requires_gpu
def test_matches_ray_cpu_sort_order(ray_cluster):
    # Direct cross-check vs Ray's own pyarrow CPU sort (no nulls -> well-defined).
    rng = np.random.default_rng(7)
    n = 1_000_000
    tbl = pa.table({"c0": rng.integers(0, 2**31 - 1, n, dtype=np.int32),
                    "p": np.arange(n, dtype=np.int64)})
    blocks = [tbl.slice(i * (n // 8), n // 8) for i in range(8)]
    cpu = [r["c0"] for r in ray.data.from_arrow(blocks).sort("c0").take_all()]
    gpu = [r["c0"] for r in ray.data.from_arrow(blocks).sort("c0", gpu=True).take_all()]
    assert gpu == cpu


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", "-s", __file__]))
