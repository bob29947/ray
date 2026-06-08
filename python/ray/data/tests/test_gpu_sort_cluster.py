"""Tier A (CPU-only) tests for the GPU sort *cluster* refactor.

These tests do NOT require a GPU or the cudf/rapidsmpf/ucxx stack. They exercise
the pure cluster-adaptation logic that was added so ``ds.sort(gpu=True)`` works
across a multi-node cluster instead of the original single-DGX-2 assumptions:

* rank count derived from the cluster's GPU total (no hardcoded 16), and a clear
  error -- not a hang -- when more ranks are requested than there are GPUs;
* topology-gated device->host spill (OFF on a single node to preserve the
  in-VRAM fast path, auto-ON across >1 node for OOM-safety), with an env override;
* the sort actor pool requesting ``scheduling_strategy="SPREAD"`` and passing the
  resolved spill setting into each rank;
* the GPU preprocessors defaulting their concurrency to the cluster GPU total.

The heavy GPU correctness/resource behavior is covered by the GPU tier in
``test_gpu_sort.py`` (and the pseudo-cluster harness), which skips without a GPU.
"""

import pytest

from ray.data._internal.planner import gpu_sort_general as g
from ray.data.preprocessors import _gpu


# --------------------------------------------------------------------------- #
# Import isolation: the cluster logic must be importable + runnable CPU-only.
# --------------------------------------------------------------------------- #
def test_module_importable_and_helpers_present():
    for name in (
        "resolve_num_gpus",
        "_resolve_spill_frac",
        "_cluster_gpu_count",
        "_cluster_node_count",
        "_get_actor_pool",
    ):
        assert hasattr(g, name), name


# --------------------------------------------------------------------------- #
# resolve_num_gpus: derive from cluster, honor override, fail loudly if too big.
# --------------------------------------------------------------------------- #
class TestResolveNumGpus:
    def test_auto_uses_cluster_total(self, monkeypatch):
        monkeypatch.delenv("RAY_DATA_GPU_SORT_NUM_GPUS", raising=False)
        monkeypatch.setattr(g, "_cluster_gpu_count", lambda: 8)
        assert g.resolve_num_gpus() == 8

    def test_env_override_within_cluster(self, monkeypatch):
        monkeypatch.setenv("RAY_DATA_GPU_SORT_NUM_GPUS", "4")
        monkeypatch.setattr(g, "_cluster_gpu_count", lambda: 16)
        assert g.resolve_num_gpus() == 4

    def test_env_override_exceeding_cluster_raises(self, monkeypatch):
        # The old default (16) on a 4-GPU cluster would hang waiting for ranks
        # that can never be scheduled; we must raise instead.
        monkeypatch.setenv("RAY_DATA_GPU_SORT_NUM_GPUS", "16")
        monkeypatch.setattr(g, "_cluster_gpu_count", lambda: 4)
        with pytest.raises(ValueError, match="only exposes 4 GPU"):
            g.resolve_num_gpus()

    def test_unknown_cluster_size_does_not_raise(self, monkeypatch):
        # cluster size unknown (0) -> fall back to >=1, never raise.
        monkeypatch.delenv("RAY_DATA_GPU_SORT_NUM_GPUS", raising=False)
        monkeypatch.setattr(g, "_cluster_gpu_count", lambda: 0)
        assert g.resolve_num_gpus() == 1


# --------------------------------------------------------------------------- #
# _resolve_spill_frac: topology-gated default + explicit override.
# --------------------------------------------------------------------------- #
class TestResolveSpillFrac:
    def test_single_node_default_off(self, monkeypatch):
        monkeypatch.delenv("RAY_DATA_GPU_SORT_SPILL_FRAC", raising=False)
        monkeypatch.setattr(g, "_cluster_node_count", lambda: 1)
        assert g._resolve_spill_frac() is None

    def test_multi_node_default_auto_on(self, monkeypatch):
        monkeypatch.delenv("RAY_DATA_GPU_SORT_SPILL_FRAC", raising=False)
        monkeypatch.setattr(g, "_cluster_node_count", lambda: 3)
        assert g._resolve_spill_frac() == g._AUTO_SPILL_FRAC

    @pytest.mark.parametrize("token", ["off", "none", "0", "", "disabled", "FALSE"])
    def test_env_off_tokens_disable(self, monkeypatch, token):
        monkeypatch.setenv("RAY_DATA_GPU_SORT_SPILL_FRAC", token)
        # Even on a multi-node cluster, an explicit "off" wins.
        monkeypatch.setattr(g, "_cluster_node_count", lambda: 8)
        assert g._resolve_spill_frac() is None

    def test_env_float_overrides_topology(self, monkeypatch):
        monkeypatch.setenv("RAY_DATA_GPU_SORT_SPILL_FRAC", "0.5")
        monkeypatch.setattr(g, "_cluster_node_count", lambda: 1)  # single node
        assert g._resolve_spill_frac() == 0.5


# --------------------------------------------------------------------------- #
# _get_actor_pool: SPREAD placement + spill plumbing (actor class mocked, so no
# GPU / rapidsmpf needed).
# --------------------------------------------------------------------------- #
def test_actor_pool_requests_spread_and_passes_spill(monkeypatch):
    recorded = {"options": None, "remote_args": []}

    class _FakeMethod:
        def remote(self, *args, **kwargs):
            return ("objref",)

    class _FakeHandle:
        # Any actor method (.is_ready/.setup_root/.setup_worker) -> remote().
        def __getattr__(self, _name):
            return _FakeMethod()

    class _FakeBound:
        def remote(self, *args, **kwargs):
            recorded["remote_args"].append(args)
            return _FakeHandle()

    class _FakeCls:
        def options(self, **kwargs):
            recorded["options"] = kwargs
            return _FakeBound()

    monkeypatch.setattr(g, "_get_actor_class", lambda: _FakeCls())
    monkeypatch.setattr(g, "_resolve_spill_frac", lambda: 0.8)
    # Readiness probe returns all-ready so the UCXX setup path is skipped.
    monkeypatch.setattr(g.ray, "get", lambda refs, **kw: [True for _ in refs])

    actors = g._get_actor_pool(3)

    assert len(actors) == 3
    assert recorded["options"]["scheduling_strategy"] == "SPREAD"
    assert recorded["options"]["num_gpus"] == 1
    assert recorded["options"]["lifetime"] == "detached"
    # spill_frac is the 3rd positional constructor arg: (num_gpus, index, spill).
    assert [a[2] for a in recorded["remote_args"]] == [0.8, 0.8, 0.8]
    assert [a[1] for a in recorded["remote_args"]] == [0, 1, 2]


# --------------------------------------------------------------------------- #
# GPU preprocessor concurrency derives from the cluster GPU total.
# --------------------------------------------------------------------------- #
class TestPreprocConcurrency:
    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("RAY_DATA_GPU_PREPROC_NUM_GPUS", "6")
        assert _gpu.env_num_gpus() == 6

    def test_auto_uses_cluster_gpus(self, monkeypatch):
        # env_num_gpus does a lazy ``import ray`` and calls cluster_resources(),
        # so patch the global ray module.
        import ray

        monkeypatch.delenv("RAY_DATA_GPU_PREPROC_NUM_GPUS", raising=False)
        monkeypatch.setattr(ray, "cluster_resources", lambda: {"GPU": 8})
        assert _gpu.env_num_gpus() == 8

    def test_auto_falls_back_to_default_without_gpus(self, monkeypatch):
        import ray

        monkeypatch.delenv("RAY_DATA_GPU_PREPROC_NUM_GPUS", raising=False)
        monkeypatch.setattr(ray, "cluster_resources", lambda: {})
        assert _gpu.env_num_gpus(default=1) == 1


# --------------------------------------------------------------------------- #
# Pseudo-cluster topology: validate detection/derivation/spill-gating against a
# REAL multi-node Ray cluster (no physical GPUs needed -- nodes only *declare*
# GPU resources, and the helpers just read ray.nodes()/cluster_resources()).
# This is the fast, reliable core of the "works on a cluster" claim; the heavy
# real-GPU correctness across topologies is the GPU tier / pseudo-cluster job.
# --------------------------------------------------------------------------- #
@pytest.fixture
def multi_node_cluster():
    import ray

    try:
        from ray.cluster_utils import Cluster
    except Exception:  # pragma: no cover - environment without cluster_utils
        pytest.skip("ray.cluster_utils.Cluster unavailable")

    cluster = Cluster(
        initialize_head=True,
        head_node_args={"num_cpus": 2, "num_gpus": 0},
    )
    # Two GPU "nodes" x 2 declared GPUs each = 4 GPUs across 2 nodes.
    cluster.add_node(num_cpus=2, num_gpus=2)
    cluster.add_node(num_cpus=2, num_gpus=2)
    cluster.wait_for_nodes()
    ray.init(address=cluster.address)
    try:
        yield cluster
    finally:
        ray.shutdown()
        cluster.shutdown()


def test_pseudo_cluster_topology_drives_nranks_and_spill(
    multi_node_cluster, monkeypatch
):
    monkeypatch.delenv("RAY_DATA_GPU_SORT_NUM_GPUS", raising=False)
    monkeypatch.delenv("RAY_DATA_GPU_SORT_SPILL_FRAC", raising=False)

    # Topology is read from the live cluster (not monkeypatched).
    assert g._cluster_gpu_count() == 4
    assert g._cluster_node_count() == 2  # head has no GPU; 2 GPU nodes

    # Rank count auto-derives to the cluster GPU total.
    assert g.resolve_num_gpus() == 4
    # >1 GPU node -> spill auto-enabled for OOM-safety.
    assert g._resolve_spill_frac() == g._AUTO_SPILL_FRAC

    # Asking for more ranks than the cluster has GPUs fails loudly (no hang).
    monkeypatch.setenv("RAY_DATA_GPU_SORT_NUM_GPUS", "8")
    with pytest.raises(ValueError, match="only exposes 4 GPU"):
        g.resolve_num_gpus()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", "-s", __file__]))
