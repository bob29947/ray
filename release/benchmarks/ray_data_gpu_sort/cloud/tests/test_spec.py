import sys
import types

import pytest

from release.benchmarks.ray_data_gpu_sort.backend_stats import required_fields_missing
from release.benchmarks.ray_data_gpu_sort.cloud import worker as cloud_worker
from release.benchmarks.ray_data_gpu_sort.cloud.spec import cpu_trials, gpu_trials
from release.benchmarks.ray_data_gpu_sort.cloud.study import (
    GPU_ONLY_TRIAL_MODE,
    _trial_values,
)
from release.benchmarks.ray_data_gpu_sort.cloud.worker import (
    _configure_gpu_sort,
    _sampling_telemetry_reasons,
)
from release.benchmarks.ray_data_gpu_sort.cloud.workflow import select


def test_exact_matrix() -> None:
    gpu = gpu_trials()
    cpu = cpu_trials()
    assert sum(item.kind == "smoke" for item in gpu) == 1
    assert sum(item.kind == "trend" for item in gpu) == 12
    assert sum(item.kind == "trend" for item in cpu) == 6
    assert sum(item.kind == "natural" for item in cpu) == 2
    assert sum(item.kind == "natural" for item in gpu) == 5


def test_gpu_only_mode_reuses_the_exact_full_gpu_matrix() -> None:
    assert _trial_values("gpu", GPU_ONLY_TRIAL_MODE) == gpu_trials()
    with pytest.raises(ValueError, match="has no CPU arm"):
        _trial_values("cpu", GPU_ONLY_TRIAL_MODE)


def _sampling_stats() -> dict:
    return {
        "sampling_mode": "cpu_sampled_arrow",
        "sampling_scheme": "deterministic_stratified_random",
        "sampling_scheme_version": 1,
        "sample_seed": 0,
        "sample_target_rows": 65_536,
        "sample_rows": 65_536,
        "sample_bytes": 1024,
        "planning_sample_bytes": 512,
        "sampled_block_count": 627,
        "sample_quota_rows": {"min": 104, "median": 105.0, "max": 105},
        "sample_plan_digest": "a" * 64,
        "sample_index_digest": "b" * 64,
        "boundary_digest": "c" * 64,
        "planning_h2d_bytes": 4096,
        "sampling_subphases_s": {
            "cpu_sample_construction": 0.5,
            "boundary_sort": 0.1,
            "orchestration_remainder": 0.05,
        },
    }


def test_sampling_acceptance_proves_stratified_cpu_planner() -> None:
    stats = _sampling_stats()
    assert (
        _sampling_telemetry_reasons(stats, input_rows=80_738_761, input_blocks=627)
        == []
    )

    stats["sample_index_digest"] = "not-a-digest"
    stats["planning_h2d_bytes"] = (1 << 20) + 1
    reasons = _sampling_telemetry_reasons(stats)
    assert any("sample_index_digest" in reason for reason in reasons)
    assert any("1 MiB" in reason for reason in reasons)

    stats = _sampling_stats()
    stats["sample_rows"] -= 1
    reasons = _sampling_telemetry_reasons(
        stats, input_rows=80_738_761, input_blocks=627
    )
    assert any("target and actual" in reason for reason in reasons)


def test_gpu_acceptance_requires_fallback_and_mpf_spill_telemetry() -> None:
    stats = {
        "peak_device_bytes": 1,
        "externalized_bytes": 0,
        "initial_run_count": 0,
        "merge_pass_count": 0,
        "cpu_sort_rows": 0,
        "cpu_merge_rows": 0,
        "phases_s": {},
    }
    assert required_fields_missing(stats) == [
        "fallback_count",
        "mpf_host_spill_bytes",
    ]
    stats.update(fallback_count=0, mpf_host_spill_bytes=0)
    assert required_fields_missing(stats) == []


def test_wave_gate(tmp_path) -> None:
    import json

    default = tmp_path / "default.json"
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "selected.json"
    default.write_text(json.dumps({"valid": True, "cold_sort_s": 100.0}))
    candidate.write_text(json.dumps({"valid": True, "cold_sort_s": 98.0}))
    assert select(default, candidate, output)["selected_wave_fraction"] == 0.50
    candidate.write_text(json.dumps({"valid": True, "cold_sort_s": 96.0}))
    assert select(default, candidate, output)["selected_wave_fraction"] == 0.375
    default.write_text(json.dumps({"valid": False, "cold_sort_s": 90.0}))
    with pytest.raises(RuntimeError, match="default 0.50"):
        select(default, candidate, output)


def test_gpu_wave_fraction_is_captured_before_materialization(monkeypatch) -> None:
    from ray.data import DataContext
    from ray.data._internal.gpu_sort.operator import _operator_config

    context = DataContext.get_current()
    previous_ranks = context.gpu_shuffle_num_actors
    previous_fraction = context.get_config("gpu_sort_auto_wave_fraction", None)
    captured = {}

    class ContextCaptured(Exception):
        pass

    def capture_context(*_args, **_kwargs):
        # ``from_arrow_refs`` takes the same copy when it creates the Dataset.
        dataset_context = DataContext.get_current().copy()
        captured.update(_operator_config(dataset_context))
        raise ContextCaptured

    monkeypatch.setattr(
        cloud_worker,
        "cell_by_name",
        lambda *_: types.SimpleNamespace(columns=()),
    )
    monkeypatch.setattr(cloud_worker, "cohort_slices", lambda *_: [])
    monkeypatch.setattr(
        cloud_worker,
        "plan_dict",
        lambda *_args, **_kwargs: {"slices": []},
    )
    monkeypatch.setattr(cloud_worker, "_stable_spill", lambda *_: {})
    monkeypatch.setattr(cloud_worker, "_materialize", capture_context)
    args = types.SimpleNamespace(
        backend="gpu", wave_fraction=0.375, cell="full", kind="trend"
    )
    try:
        with pytest.raises(ContextCaptured):
            cloud_worker._run_performance(None, args, {}, [])
        # Change the global value after capture to prove this assertion reads
        # the Dataset-style copy, not the mutable current context.
        _configure_gpu_sort("gpu", 0.50)
        assert captured["auto_wave_fraction"] == 0.375

        captured.clear()
        monkeypatch.setattr(cloud_worker, "smoke_slices", lambda *_: [])
        with pytest.raises(ContextCaptured):
            cloud_worker._run_smoke(
                None,
                types.SimpleNamespace(wave_fraction=0.375),
                {"schema_names": []},
                [],
            )
        assert captured["auto_wave_fraction"] == 0.375
    finally:
        context.gpu_shuffle_num_actors = previous_ranks
        if previous_fraction is None:
            context.remove_config("gpu_sort_auto_wave_fraction")
        else:
            context.set_config("gpu_sort_auto_wave_fraction", previous_fraction)


def test_projection_reader_returns_reconstructable_task_output(monkeypatch) -> None:
    table = object()
    fake_ray = types.ModuleType("ray")
    fake_ray.put = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("projection readers must not create worker-owned nested refs")
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(cloud_worker, "read_projection", lambda *_: table)

    assert cloud_worker._read_and_seal({}, ()) is table


def test_materialize_keeps_reader_task_output_refs(monkeypatch) -> None:
    class ObjectRef:
        pass

    refs = [ObjectRef(), ObjectRef()]
    submitted = []

    class Reader:
        def options(self, **_kwargs):
            return self

        def remote(self, item, columns):
            submitted.append((item, columns))
            return refs[len(submitted) - 1]

    class Dataset:
        def materialize(self):
            return self

    seen = {}

    def from_arrow_refs(values):
        seen["refs"] = list(values)
        return Dataset()

    fake_ray = types.SimpleNamespace(
        ObjectRef=ObjectRef,
        remote=lambda **_kwargs: lambda _fn: Reader(),
        get=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("materialization must not unwrap nested ObjectRefs")
        ),
        data=types.SimpleNamespace(from_arrow_refs=from_arrow_refs),
    )
    monkeypatch.setattr(cloud_worker, "_affinity", lambda node_id: node_id)
    monkeypatch.setattr(
        cloud_worker,
        "_metadata",
        lambda _dataset: {"blocks": 2, "rows": 5, "decoded_bytes": 10},
    )
    monkeypatch.setattr(cloud_worker, "_schema", lambda _dataset: "schema")
    monkeypatch.setattr(
        cloud_worker,
        "_locations",
        lambda *_: {"all_locatable": True, "objects_by_node": {"node-0": 2}},
    )
    plan = {
        "rows": 5,
        "blocks": 2,
        "slices": [{"rows": 2}, {"rows": 3}],
    }

    _, result = cloud_worker._materialize(
        fake_ray, plan, ("Origin",), [{"NodeID": "node-0"}]
    )

    assert seen["refs"] == refs
    assert len(submitted) == 2
    assert result["rows"] == 5
