"""Tier B/C: real-GPU *pseudo-cluster* tests for the GPU sort cluster refactor.

These spin a multi-node Ray cluster on a single physical box with
``ray.cluster_utils.Cluster`` and partition the physical GPUs across the
simulated nodes (each node's raylet gets a distinct ``CUDA_VISIBLE_DEVICES``
slice, verified to land each rank on a distinct physical GPU). This exercises
the *Ray-level* multi-node path that the refactor added:

  * rank count auto-derived from the cluster GPU total (no hardcoded 16),
  * ``scheduling_strategy="SPREAD"`` placing one rank per GPU/node,
  * topology-gated device->host spill auto-enabled across >1 node,
  * executor-owned output blocks surviving the post-sort actor teardown,
  * the GPU preprocessors fanning out across nodes (Tier C end-to-end).

Caveat (documented in the plan): on one physical box the GPUs are peer-accessible
via NVSwitch, so the UCXX shuffle may still ride cuda_ipc even across simulated
nodes -- this validates correctness/placement/ownership/resource behavior, NOT
the real cross-node network transport (that needs actual EC2/EFA hardware).

These tests are heavy (each spins a fresh multi-raylet cluster), so they are
*opt-in*: set ``RAY_DATA_RUN_GPU_PSEUDOCLUSTER=1`` (and have cudf + rapidsmpf +
enough GPUs). They skip cleanly otherwise.

Run:
    RAY_DATA_RUN_GPU_PSEUDOCLUSTER=1 \
        .venv/bin/python -m pytest \
        python/ray/data/tests/test_gpu_sort_pseudocluster.py -v -s
"""

import contextlib
import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import ray


# --------------------------------------------------------------------------- #
# Availability / opt-in gating
# --------------------------------------------------------------------------- #
def _physical_gpu_count() -> int:
    try:
        import cupy as cp

        return int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        return 0


def _gpu_stack_available() -> bool:
    try:
        import cudf  # noqa: F401
        import cupy  # noqa: F401
        import rapidsmpf  # noqa: F401

        return _physical_gpu_count() > 0
    except Exception:
        return False


OPT_IN = os.environ.get("RAY_DATA_RUN_GPU_PSEUDOCLUSTER") == "1"
PHYS_GPUS = _physical_gpu_count()
requires_pseudocluster = pytest.mark.skipif(
    not (OPT_IN and _gpu_stack_available()),
    reason="set RAY_DATA_RUN_GPU_PSEUDOCLUSTER=1 with cudf+rapidsmpf+GPUs to run",
)


# --------------------------------------------------------------------------- #
# Pseudo-cluster builder: N nodes x G GPUs, physical GPUs partitioned per node.
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def pseudo_cluster(n_nodes: int, gpus_per_node: int, object_store_gib: float = 2.0):
    from ray.cluster_utils import Cluster

    total = n_nodes * gpus_per_node
    osm = int(object_store_gib * 1024**3)
    # Must be set before raylets spawn: silences the cosmetic Ray 3.0 OpenTelemetry
    # teardown crash ("pure virtual method called") seen when a GPU worker exits.
    os.environ.setdefault("RAY_enable_open_telemetry", "0")
    saved_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")

    cluster = Cluster(
        initialize_head=True,
        head_node_args={
            "num_cpus": 4,
            "num_gpus": 0,
            "object_store_memory": osm,
            "include_dashboard": False,
        },
    )
    try:
        for i in range(n_nodes):
            lo = i * gpus_per_node
            # Each node's raylet sees a distinct physical-GPU slice, so Ray hands
            # each rank a unique physical device (verified by the dev probe).
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                str(lo + j) for j in range(gpus_per_node)
            )
            cluster.add_node(
                num_cpus=8, num_gpus=gpus_per_node, object_store_memory=osm
            )
        # Restore the driver's own CUDA_VISIBLE_DEVICES (driver does no GPU work).
        if saved_cvd is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved_cvd

        cluster.wait_for_nodes()
        ray.init(address=cluster.address, logging_level="ERROR")
        yield total
    finally:
        with contextlib.suppress(Exception):
            from ray.data._internal.planner.gpu_sort_general import kill_actor_pool

            kill_actor_pool(total)
        with contextlib.suppress(Exception):
            ray.shutdown()
        with contextlib.suppress(Exception):
            cluster.shutdown()


def _make_sort_blocks(n_rows: int, n_blocks: int, seed: int = 0):
    """Small mixed int/string/float dataset with a UNIQUE composite key (so the
    GPU and pandas orders match row-for-row), split into ``n_blocks`` blocks."""
    rng = np.random.default_rng(seed)
    k_int = rng.integers(0, 1000, n_rows, dtype=np.int64)
    k_str = pa.array([f"s{x}" for x in rng.integers(0, 5000, n_rows)])
    # Unique tiebreaker -> deterministic total order across the 3-key sort.
    k_uniq = rng.permutation(n_rows).astype(np.int64)
    val = np.arange(n_rows, dtype=np.int64)  # payload + row-integrity key
    tbl = pa.table({"k_int": k_int, "k_str": k_str, "k_uniq": k_uniq, "val": val})
    step = max(1, n_rows // n_blocks)
    return [tbl.slice(i, min(step, n_rows - i)) for i in range(0, n_rows, step)]


def _oracle_sorted(tbl: pa.Table, keys) -> pd.DataFrame:
    return (
        tbl.to_pandas()
        .sort_values(by=list(keys), na_position="last", kind="stable")
        .reset_index(drop=True)
    )


def _wait_for_gpus(total: int, timeout_s: float = 90.0) -> None:
    """Block until >= ``total`` GPUs are free again.

    Each GPU stage (impute / sort / encode) uses all GPUs and frees them when it
    finishes, but the teardown is async. Waiting between stages gives a clean
    sequential hand-off so the next stage isn't starved (the same pattern the
    e2e benchmark uses); without it the stages contend for the same GPUs and
    deadlock.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if ray.available_resources().get("GPU", 0) >= total - 0.5:
            return
        time.sleep(0.2)


# --------------------------------------------------------------------------- #
# Tier B: correctness + resource behavior across simulated nodes.
# --------------------------------------------------------------------------- #
@requires_pseudocluster
@pytest.mark.parametrize(
    "n_nodes,gpus_per_node",
    [
        pytest.param(4, 1, id="4nodes_x_1gpu"),  # the g6.4xlarge single-GPU case
        pytest.param(2, 2, id="2nodes_x_2gpu"),  # mixed intra+inter node
    ],
)
def test_pseudocluster_sort_correct_and_releases(n_nodes, gpus_per_node, monkeypatch):
    if n_nodes * gpus_per_node > PHYS_GPUS:
        pytest.skip(f"needs {n_nodes * gpus_per_node} GPUs, have {PHYS_GPUS}")

    # Auto-derive rank count from the cluster (do NOT pin it), and release the
    # sort GPUs afterwards so we can assert they come back.
    monkeypatch.delenv("RAY_DATA_GPU_SORT_NUM_GPUS", raising=False)
    monkeypatch.delenv("RAY_DATA_GPU_SORT_SPILL_FRAC", raising=False)
    monkeypatch.setenv("RAY_DATA_GPU_SORT_RELEASE", "1")

    keys = ["k_int", "k_str", "k_uniq"]
    with pseudo_cluster(n_nodes, gpus_per_node) as total:
        from ray.data._internal.planner import gpu_sort_general as g

        # The refactor's topology detection sees the real simulated cluster.
        assert g._cluster_gpu_count() == total
        assert g._cluster_node_count() == n_nodes
        assert g.resolve_num_gpus() == total
        # >1 node -> spill auto-enabled (OOM-safety) without an explicit env.
        assert g._resolve_spill_frac() == g._AUTO_SPILL_FRAC

        ctx = ray.data.DataContext.get_current()
        ctx.execution_options.preserve_order = True

        blocks = _make_sort_blocks(n_rows=200_000, n_blocks=total * 2)
        full = pa.concat_tables(blocks)
        out = ray.data.from_arrow(blocks).sort(keys, gpu=True).materialize()

        # (a) correctness: global key order matches an independent pandas oracle.
        got = pd.DataFrame(out.take_all())[keys + ["val"]].reset_index(drop=True)
        oracle = _oracle_sorted(full, keys)[keys + ["val"]].reset_index(drop=True)
        pd.testing.assert_frame_equal(got[keys], oracle[keys], check_dtype=False)
        # (b) row integrity: every input row preserved (compare by unique val).
        assert sorted(got["val"].tolist()) == list(range(full.num_rows))

        # (c) resource behavior: the sort released its GPUs (RELEASE=1) and the
        # output survived the actor-pool teardown (we just read it above).
        deadline = time.time() + 30
        while time.time() < deadline:
            if ray.available_resources().get("GPU", 0) >= total - 0.5:
                break
            time.sleep(0.2)
        assert ray.available_resources().get("GPU", 0) >= total - 0.5


# --------------------------------------------------------------------------- #
# Tier C: end-to-end impute -> sort -> encode across simulated nodes, with the
# GPU operators vs their CPU counterparts (faithful drop-in parity).
# --------------------------------------------------------------------------- #
@requires_pseudocluster
def test_pseudocluster_e2e_impute_sort_encode(monkeypatch):
    if 2 * 2 > PHYS_GPUS:
        pytest.skip(f"needs 4 GPUs, have {PHYS_GPUS}")

    monkeypatch.delenv("RAY_DATA_GPU_SORT_NUM_GPUS", raising=False)
    monkeypatch.delenv("RAY_DATA_GPU_PREPROC_NUM_GPUS", raising=False)
    monkeypatch.setenv("RAY_DATA_GPU_SORT_RELEASE", "1")
    monkeypatch.setenv("RAY_DATA_GPU_PREPROC_BATCH_SIZE", "100000")

    from ray.data.preprocessors import (
        GpuOrdinalEncoder,
        GpuSimpleImputer,
        OrdinalEncoder,
        SimpleImputer,
    )

    sort_key = ["k_int", "k_uniq"]
    impute_cols = ["cat"]
    encode_cols = ["item", "cat"]

    def make_ds(n_rows, n_blocks, seed):
        rng = np.random.default_rng(seed)
        cats = np.array(["a", "a", "a", "b", "c"], dtype=object)
        cat = cats[rng.integers(0, len(cats), n_rows)].astype(object)
        cat[rng.random(n_rows) < 0.1] = None  # nulls for the imputer
        tbl = pa.table(
            {
                "k_int": rng.integers(0, 500, n_rows, dtype=np.int64),
                "k_uniq": rng.permutation(n_rows).astype(np.int64),
                "item": pa.array([f"i{x}" for x in rng.integers(0, 2000, n_rows)]),
                "cat": pa.array(cat),
                "id": np.arange(n_rows, dtype=np.int64),
            }
        )
        return ray.data.from_arrow(
            [
                tbl.slice(i, n_rows // n_blocks)
                for i in range(0, n_rows, n_rows // n_blocks)
            ]
        ).materialize()

    def run(ds, gpu: bool, total: int):
        Imp = GpuSimpleImputer if gpu else SimpleImputer
        Enc = GpuOrdinalEncoder if gpu else OrdinalEncoder
        # Materialize between stages (RAM->RAM) and wait for GPUs to free, so the
        # impute/sort/encode pools hand the GPUs off sequentially instead of
        # contending for them in one fused execution (which would deadlock).
        ds = (
            Imp(columns=impute_cols, strategy="most_frequent")
            .fit_transform(ds)
            .materialize()
        )
        if gpu:
            _wait_for_gpus(total)
        ds = ds.sort(sort_key, backend="gpu" if gpu else "cpu").materialize()
        if gpu:
            _wait_for_gpus(total)
        ds = Enc(columns=encode_cols).fit_transform(ds).materialize()
        return ds

    with pseudo_cluster(2, 2) as total:
        ctx = ray.data.DataContext.get_current()
        ctx.execution_options.preserve_order = True

        base = make_ds(80_000, 8, seed=1)
        cpu = run(base, gpu=False, total=total).to_pandas()
        gpu = run(base, gpu=True, total=total).to_pandas()

        # Content parity, order-independent (re-key by the unique id).
        c = cpu.sort_values("id").reset_index(drop=True)
        g_ = gpu.sort_values("id").reset_index(drop=True)
        for col in ["k_int", "k_uniq", "item", "cat"]:
            assert (c[col].fillna(-1) == g_[col].fillna(-1)).all(), col

        # The GPU result is globally sorted by the key (in materialized order).
        k = gpu[sort_key].reset_index(drop=True)
        assert k.equals(k.sort_values(sort_key, kind="stable").reset_index(drop=True))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", "-s", __file__]))
