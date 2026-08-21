import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pytest

from ray.data._internal.execution.interfaces import TaskContext
from ray.data.datasource.datasink import WriteResult

from release.benchmarks.ray_data_gpu_sort.streaming_parquet_sink import (
    StreamingParquetDatasink,
)
from release.benchmarks.ray_data_gpu_sort import streaming_parquet_e2e as controller
from release.benchmarks.ray_data_gpu_sort.streaming_parquet_e2e import (
    GATE_SCHEMA_VERSION,
    GATE_SEQUENCE,
    IMPLEMENTATION_BRANCH,
    ORIGIN_AIRPORT_ID_KEY,
    RUN_SEQUENCE,
    SMOKE_SEQUENCE,
    SORT_KEY_STATS,
    TARGET_BLOCKS,
    TARGET_DECODED_BYTES,
    TARGET_PLAN_DIGEST,
    TARGET_ROWS,
    TRIAL_TIMEOUT_S,
    _accepted_capacity_result,
    _apply_controller_outcome,
    _canonical_digest,
    _safe_campaign_root,
    _validate_prerequisite_gates,
    _wait_for_worker,
    _worker_command,
    summarize,
)
from release.benchmarks.ray_data_gpu_sort.streaming_parquet_e2e_worker import (
    _capacity_failure,
    _configure_context,
    _frequency_digest,
    _safe_runtime,
    _timed_pipeline_capacity_failure,
    _timeline,
    _validate_durable_parquet,
)

GPU_ORIGIN_ID_SPEC = next(
    item
    for item in RUN_SEQUENCE
    if item.backend == "gpu" and item.sort_key == ORIGIN_AIRPORT_ID_KEY
)
CPU_ORIGIN_ID_SPEC = next(
    item
    for item in RUN_SEQUENCE
    if item.backend == "pyarrow" and item.sort_key == ORIGIN_AIRPORT_ID_KEY
)


def _artifact(
    name: str, backend: str, seconds: float, sort_key: str = ORIGIN_AIRPORT_ID_KEY
) -> dict:
    return {
        "valid": True,
        "status": "accepted",
        "trial_name": name,
        "backend": backend,
        "workload": "full",
        "sort_key": sort_key,
        "timings_s": {"e2e": seconds},
    }


def _accepted_matrix(*, gpu_s: float = 100.0, cpu_s: float = 250.0) -> list[dict]:
    return [
        _artifact(GPU_ORIGIN_ID_SPEC.name, "gpu", gpu_s),
        _artifact(CPU_ORIGIN_ID_SPEC.name, "pyarrow", cpu_s),
    ]


def test_frozen_contract_and_unprefixed_branch_name():
    assert IMPLEMENTATION_BRANCH == "origin-airport-id-1tb-parquet-e2e"
    assert "codex" not in IMPLEMENTATION_BRANCH
    assert TARGET_ROWS == 1_177_097_812
    assert TARGET_BLOCKS == 9_151
    assert TARGET_DECODED_BYTES == 999_999_999_762
    assert TARGET_PLAN_DIGEST == (
        "e80163bbc681bc8b04b34d273a4bcdb58b52ef4cbe794b92f07f01bb4d00226c"
    )
    assert [item.backend for item in RUN_SEQUENCE] == ["gpu", "pyarrow"]
    assert [item.backend for item in SMOKE_SEQUENCE] == ["gpu", "pyarrow"]
    assert {item.sort_key for item in (*SMOKE_SEQUENCE, *RUN_SEQUENCE)} == {
        ORIGIN_AIRPORT_ID_KEY,
    }
    assert tuple(SORT_KEY_STATS) == (ORIGIN_AIRPORT_ID_KEY,)
    assert SORT_KEY_STATS[ORIGIN_AIRPORT_ID_KEY]["cardinality"] == 401


def test_controller_has_separate_gate_smoke_and_full_commands():
    parser = controller._parser()
    assert parser.parse_args(["gates"]).command == "gates"
    assert parser.parse_args(["smoke"]).command == "smoke"
    assert parser.parse_args(["run"]).command == "run"


def test_cpu_context_overrides_only_push_shuffle(tmp_path):
    from ray.data import DataContext
    from ray.data.context import ShuffleStrategy

    context = DataContext.get_current()
    old_preserve_order = context.execution_options.preserve_order
    old_target_max_block_size = context.target_max_block_size
    old_shuffle_strategy = context.shuffle_strategy
    old_use_polars = context.use_polars
    old_use_polars_sort = context.use_polars_sort
    try:
        context.execution_options.preserve_order = False
        context.target_max_block_size = 128 << 20
        context.shuffle_strategy = ShuffleStrategy.HASH_SHUFFLE
        context.use_polars = False
        context.use_polars_sort = False
        configured = _configure_context("pyarrow", tmp_path)
        assert configured == {
            "backend": "pyarrow",
            "shuffle_strategy": ShuffleStrategy.SORT_SHUFFLE_PUSH_BASED.value,
            "preserve_order": False,
            "target_max_block_size": 128 << 20,
            "use_polars": False,
            "use_polars_sort": False,
            "pre_override_shuffle_strategy": ShuffleStrategy.HASH_SHUFFLE.value,
            "cpu_sort_overrides": ["shuffle_strategy"],
        }
    finally:
        context.execution_options.preserve_order = old_preserve_order
        context.target_max_block_size = old_target_max_block_size
        context.shuffle_strategy = old_shuffle_strategy
        context.use_polars = old_use_polars
        context.use_polars_sort = old_use_polars_sort


def test_gate_command_preserves_virtualenv_invocation_symlink(tmp_path):
    python = tmp_path / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(Path("/usr/bin/python3.10"))

    command = controller._gate_command(python, GATE_SEQUENCE[0])

    assert command[0] == str(python.absolute())
    assert command[0] != str(python.resolve())


def test_worker_command_binds_the_sort_key(tmp_path):
    command, _, _ = _worker_command(
        python=tmp_path / "venv" / "bin" / "python",
        dataset_root=tmp_path / "dataset",
        campaign_root=tmp_path / "campaign",
        spec=GPU_ORIGIN_ID_SPEC,
        expected_head="a" * 40,
        expected_dataset_identity={"identity_digest": "b" * 64},
    )
    assert command[command.index("--sort-key") + 1] == ORIGIN_AIRPORT_ID_KEY


def test_dataset_identity_binds_and_verifies_selected_parquet_bytes(
    tmp_path, monkeypatch
):
    parquet = tmp_path / "parquet" / "data.parquet"
    parquet.parent.mkdir()
    parquet.write_bytes(b"frozen parquet payload")
    parquet_sha = hashlib.sha256(parquet.read_bytes()).hexdigest()
    manifest = {
        "schema_names": [f"c{index}" for index in range(109)],
        "manifest_sha256": "a" * 64,
        "months": [
            {
                "parquet_path": str(parquet),
                "parquet_sha256": parquet_sha,
            }
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(controller, "load_manifest", lambda _root: manifest)
    monkeypatch.setattr(
        controller,
        "exact_plan",
        lambda _root: ({"digest": "plan"}, (SimpleNamespace(path=str(parquet)),)),
    )

    identity = controller._dataset_identity(tmp_path, verify_parquet_files=True)
    assert identity["selected_parquet_files"] == 1
    assert identity["selected_parquet_sha256_digest"]

    parquet.write_bytes(b"mutated parquet payload")
    with pytest.raises(RuntimeError, match="SHA-256 changed"):
        controller._dataset_identity(tmp_path, verify_parquet_files=True)


def test_runtime_capacity_snapshot_enforces_memory_and_raid_bounds(monkeypatch):
    monkeypatch.setattr(
        controller,
        "_meminfo",
        lambda: {
            "MemTotal": controller.MIN_HOST_AVAILABLE_BYTES * 2,
            "MemAvailable": controller.MIN_HOST_AVAILABLE_BYTES - 1,
        },
    )
    monkeypatch.setattr(
        controller.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(
            total=controller.MIN_CAMPAIGN_FREE_BYTES * 2,
            used=controller.MIN_CAMPAIGN_FREE_BYTES + 1,
            free=controller.MIN_CAMPAIGN_FREE_BYTES - 1,
        ),
    )
    value = controller._runtime_capacity_snapshot()
    assert not value["valid"]
    assert len(value["rejection_reasons"]) == 2


@pytest.mark.parametrize(
    "value",
    (
        Path("relative"),
        Path("/raid"),
        Path("/raid/one-component"),
        Path("/tmp/not-raid/campaign"),
    ),
)
def test_campaign_root_rejects_broad_or_out_of_scope_paths(value):
    with pytest.raises(ValueError):
        _safe_campaign_root(value, must_not_exist=False)


def test_campaign_root_refuses_overwrite(tmp_path, monkeypatch):
    # Keep the safety predicate under test without creating files on /raid.
    candidate = Path("/raid/benchmarks/already-present")
    monkeypatch.setattr(Path, "exists", lambda self: self == candidate)
    with pytest.raises(FileExistsError):
        _safe_campaign_root(candidate, must_not_exist=True)


@pytest.mark.parametrize(
    "value",
    (Path("relative"), Path("/raid"), Path("/raid/one/two"), Path("/tmp/a/b/c")),
)
def test_worker_runtime_rejects_broad_or_out_of_scope_paths(value):
    with pytest.raises(ValueError):
        _safe_runtime(value, must_not_exist=False)


def test_directional_summary_reports_cpu_over_gpu():
    artifacts = _accepted_matrix()
    gpu = next(
        item for item in artifacts if item["trial_name"] == GPU_ORIGIN_ID_SPEC.name
    )
    gpu["gpu_stats"] = {
        "ranks": [{"rank": rank} for rank in range(16)],
        "input_object_refs_received": 9_166,
        "initial_run_count": 144,
        "source_run_count": 128,
    }
    cpu = next(
        item for item in artifacts if item["trial_name"] == CPU_ORIGIN_ID_SPEC.name
    )
    cpu["cpu_stats"] = {"blocking_all_to_all": {"input_blocks_received": 9_166}}
    result = summarize(artifacts)
    assert result["valid"]
    assert result["observations_per_backend_per_key"] == 1
    assert result["cpu_over_gpu"][ORIGIN_AIRPORT_ID_KEY] == 2.5
    assert result["results_by_sort_key"][ORIGIN_AIRPORT_ID_KEY]["gpu_reached_2x"]
    geometry = result["results_by_sort_key"][ORIGIN_AIRPORT_ID_KEY][
        "execution_geometry"
    ]
    assert geometry == {
        "planned_source_slices": TARGET_BLOCKS,
        "gpu": {
            "ranks": 16,
            "input_object_refs_received": 9_166,
            "initial_run_count": 144,
            "source_run_count": 128,
        },
        "cpu": {"input_blocks_received": 9_166},
    }


def test_directional_summary_retains_cpu_capacity_result():
    cpu = {
        "valid": False,
        "status": "capacity-failure",
        "trial_name": CPU_ORIGIN_ID_SPEC.name,
        "backend": "pyarrow",
        "workload": "full",
        "sort_key": ORIGIN_AIRPORT_ID_KEY,
        "timed_pipeline_started": True,
        "last_durable_phase": {
            "phase": "pipeline-finished",
            "timed_pipeline_started": True,
            "timed_pipeline_finished": True,
            "pipeline_succeeded": False,
            "pipeline_failure_is_capacity": True,
        },
        "controller": {"returncode": 2, "timed_out": False},
        "timings_s": {},
    }
    artifacts = [
        item for item in _accepted_matrix() if item["trial_name"] != cpu["trial_name"]
    ]
    result = summarize([*artifacts, cpu])
    assert result["valid"]
    origin_id = result["results_by_sort_key"][ORIGIN_AIRPORT_ID_KEY]
    assert origin_id["cpu_capacity_result"] == "capacity-failure"
    assert origin_id["cpu_over_gpu"] is None


def test_directional_summary_never_uses_invalid_capacity_elapsed_time():
    artifacts = _accepted_matrix()
    cpu = next(
        item for item in artifacts if item["trial_name"] == CPU_ORIGIN_ID_SPEC.name
    )
    cpu.update(
        {
            "valid": False,
            "status": "capacity-failure",
            "timed_pipeline_started": True,
            "last_durable_phase": {
                "phase": "pipeline-finished",
                "timed_pipeline_started": True,
                "timed_pipeline_finished": True,
                "pipeline_succeeded": False,
                "pipeline_failure_is_capacity": True,
            },
            "controller": {"returncode": 2, "timed_out": False},
            "timings_s": {"e2e": 999.0},
        }
    )
    result = summarize(artifacts)
    origin_id = result["results_by_sort_key"][ORIGIN_AIRPORT_ID_KEY]
    assert result["valid"]
    assert origin_id["cpu_e2e_s"] is None
    assert origin_id["cpu_over_gpu"] is None


def test_directional_summary_rejects_mislabeled_artifact():
    artifacts = _accepted_matrix()
    gpu = next(
        item for item in artifacts if item["trial_name"] == GPU_ORIGIN_ID_SPEC.name
    )
    gpu["sort_key"] = "unexpected_key"

    result = summarize(artifacts)

    assert not result["valid"]
    assert any(
        "GPU artifact identity mismatch" in item for item in result["rejection_reasons"]
    )
    origin_id = result["results_by_sort_key"][ORIGIN_AIRPORT_ID_KEY]
    assert origin_id["gpu_e2e_s"] is None
    assert origin_id["cpu_over_gpu"] is None


def test_capacity_classification_is_conservative():
    assert _capacity_failure(MemoryError("out of memory"))
    assert _capacity_failure(OSError(28, "No space left on device"))
    assert not _capacity_failure(RuntimeError("schema mismatch"))


def test_capacity_result_requires_failure_inside_timed_cpu_terminal_call():
    args = argparse.Namespace(backend="pyarrow", workload="full")
    proved = {
        "timed_pipeline_started": True,
        "timed_pipeline_finished": True,
        "pipeline_succeeded": False,
        "pipeline_failure_is_capacity": True,
    }
    assert _timed_pipeline_capacity_failure(args, proved)
    assert not _timed_pipeline_capacity_failure(
        args, {**proved, "pipeline_succeeded": True}
    )
    assert not _timed_pipeline_capacity_failure(
        args, {**proved, "timed_pipeline_finished": False}
    )
    assert not _timed_pipeline_capacity_failure(
        args, {**proved, "pipeline_failure_is_capacity": False}
    )
    assert not _timed_pipeline_capacity_failure(
        argparse.Namespace(backend="gpu", workload="full"), proved
    )


def test_controller_capacity_acceptance_requires_phase_specific_proof():
    worker_failure = {
        "backend": "pyarrow",
        "workload": "full",
        "status": "capacity-failure",
        "timed_pipeline_started": True,
        "last_durable_phase": {
            "phase": "pipeline-finished",
            "timed_pipeline_started": True,
            "timed_pipeline_finished": True,
            "pipeline_succeeded": False,
            "pipeline_failure_is_capacity": True,
        },
        "controller": {"returncode": 2, "timed_out": False},
    }
    assert _accepted_capacity_result(worker_failure)
    assert not _accepted_capacity_result(
        {
            **worker_failure,
            "last_durable_phase": {
                **worker_failure["last_durable_phase"],
                "phase": "validation-running",
            },
        }
    )
    assert not _accepted_capacity_result(
        {**worker_failure, "status": "oom", "last_durable_phase": {}}
    )


def test_persisted_e2e_deadline_is_enforced_after_phase_flip():
    artifact = _artifact(CPU_ORIGIN_ID_SPEC.name, "pyarrow", TRIAL_TIMEOUT_S + 0.25)
    _apply_controller_outcome(
        artifact,
        spec=CPU_ORIGIN_ID_SPEC,
        returncode=0,
        timed_out=False,
        timeout_phase=None,
        last_phase={
            "phase": "validation-running",
            "timed_pipeline_started": True,
        },
        timeout_s=TRIAL_TIMEOUT_S,
        oom_before=10,
        oom_after=10,
    )
    assert artifact["status"] == "capacity-timeout"
    assert artifact["valid"] is False
    assert artifact["controller_capacity_evidence"]["kind"] == (
        "persisted-e2e-deadline"
    )

    gpu = _artifact(GPU_ORIGIN_ID_SPEC.name, "gpu", TRIAL_TIMEOUT_S + 0.25)
    _apply_controller_outcome(
        gpu,
        spec=GPU_ORIGIN_ID_SPEC,
        returncode=0,
        timed_out=False,
        timeout_phase=None,
        last_phase={"phase": "worker-complete"},
        timeout_s=TRIAL_TIMEOUT_S,
        oom_before=10,
        oom_after=10,
    )
    assert gpu["status"] == "rejected"


def test_waiter_checks_persisted_elapsed_before_post_pipeline_watchdog(
    monkeypatch, tmp_path
):
    class Process:
        pid = 123
        returncode = None

        def poll(self):
            return self.returncode

    process = Process()
    monkeypatch.setattr(
        controller,
        "_read_optional_json",
        lambda _: {
            "phase": "pipeline-finished",
            "timed_pipeline_started": True,
            "timed_pipeline_finished": True,
            "e2e_elapsed_s": 10.01,
        },
    )

    def terminate(value):
        value.returncode = -15
        return True

    monkeypatch.setattr(controller, "_terminate_process_group", terminate)
    returncode, timed_out, timeout_phase, phase = _wait_for_worker(
        process,
        runtime=tmp_path,
        pipeline_timeout_s=10,
        post_pipeline_timeout_s=20,
    )
    assert (returncode, timed_out, timeout_phase) == (-15, True, "pipeline")
    assert phase["e2e_elapsed_s"] == 10.01


def test_sigkill_needs_running_phase_and_kernel_oom_delta():
    phase = {
        "phase": "pipeline-running",
        "timed_pipeline_started": True,
        "kernel_oom_baseline": {
            "host_vmstat_oom_kill": 50,
            "session_cgroup": None,
        },
    }
    proved = {
        "valid": False,
        "status": "missing-artifact",
        "backend": "pyarrow",
        "workload": "full",
        "timed_pipeline_started": True,
        "last_durable_phase": phase,
        "rejection_reasons": [],
    }
    _apply_controller_outcome(
        proved,
        spec=CPU_ORIGIN_ID_SPEC,
        returncode=-9,
        timed_out=False,
        timeout_phase=None,
        last_phase=phase,
        timeout_s=TRIAL_TIMEOUT_S,
        oom_before=50,
        oom_after=51,
    )
    assert proved["status"] == "capacity-oom"
    evidence = proved.pop("controller_capacity_evidence")
    proved["controller"] = {
        "returncode": -9,
        "timed_out": False,
        "capacity_evidence": evidence,
    }
    assert _accepted_capacity_result(proved)

    unexplained_phase = {
        **phase,
        "kernel_oom_baseline": {
            "host_vmstat_oom_kill": 51,
            "session_cgroup": None,
        },
    }
    unexplained = {
        **proved,
        "controller": {},
        "last_durable_phase": unexplained_phase,
        "rejection_reasons": [],
    }
    _apply_controller_outcome(
        unexplained,
        spec=CPU_ORIGIN_ID_SPEC,
        returncode=-9,
        timed_out=False,
        timeout_phase=None,
        last_phase=unexplained_phase,
        timeout_s=TRIAL_TIMEOUT_S,
        oom_before=51,
        oom_after=51,
    )
    assert unexplained["status"] == "rejected"
    assert not _accepted_capacity_result(unexplained)


def test_ray_child_oom_requires_session_cgroup_and_terminal_failure_phase():
    phase = {
        "phase": "pipeline-finished",
        "timed_pipeline_started": True,
        "timed_pipeline_finished": True,
        "pipeline_succeeded": False,
        "pipeline_failure": "WorkerCrashedError: worker died unexpectedly",
    }
    artifact = {
        "valid": False,
        "status": "rejected",
        "backend": "pyarrow",
        "workload": "full",
        "timed_pipeline_started": True,
        "timed_pipeline_finished": True,
        "pipeline_succeeded": False,
        "last_durable_phase": phase,
        "exception": "ray.exceptions.WorkerCrashedError: worker died unexpectedly",
        "rejection_reasons": [],
    }
    before = {
        "source": "/sys/fs/cgroup/user.slice/session.scope/memory.events.local",
        "cgroup": "/user.slice/session.scope",
        "oom": 2,
        "oom_kill": 4,
    }
    after = {**before, "oom": 3, "oom_kill": 5}
    phase["kernel_oom_baseline"] = {
        "host_vmstat_oom_kill": 100,
        "session_cgroup": before,
    }
    phase["kernel_oom_finished"] = {
        "host_vmstat_oom_kill": 101,
        "session_cgroup": after,
    }
    _apply_controller_outcome(
        artifact,
        spec=CPU_ORIGIN_ID_SPEC,
        returncode=1,
        timed_out=False,
        timeout_phase=None,
        last_phase=phase,
        timeout_s=TRIAL_TIMEOUT_S,
        oom_before=100,
        oom_after=101,
        cgroup_oom_before=before,
        cgroup_oom_after=after,
    )
    assert artifact["status"] == "capacity-oom"
    evidence = artifact.pop("controller_capacity_evidence")
    assert evidence["kind"] == "session-cgroup-ray-process-oom"
    artifact["controller"] = {
        "returncode": 1,
        "timed_out": False,
        "capacity_evidence": evidence,
    }
    assert _accepted_capacity_result(artifact)

    host_only_phase = {
        **phase,
        "kernel_oom_baseline": {
            "host_vmstat_oom_kill": 200,
            "session_cgroup": None,
        },
        "kernel_oom_finished": {
            "host_vmstat_oom_kill": 201,
            "session_cgroup": None,
        },
    }
    host_only = {
        **artifact,
        "status": "rejected",
        "controller": {},
        "last_durable_phase": host_only_phase,
        "rejection_reasons": [],
    }
    _apply_controller_outcome(
        host_only,
        spec=CPU_ORIGIN_ID_SPEC,
        returncode=1,
        timed_out=False,
        timeout_phase=None,
        last_phase=host_only_phase,
        timeout_s=TRIAL_TIMEOUT_S,
        oom_before=200,
        oom_after=201,
    )
    assert host_only["status"] == "rejected"
    assert not _accepted_capacity_result(host_only)


def test_post_timer_oom_cannot_validate_an_earlier_ray_child_crash():
    before = {
        "source": "/sys/fs/cgroup/user.slice/session.scope/memory.events.local",
        "cgroup": "/user.slice/session.scope",
        "oom": 0,
        "oom_kill": 0,
    }
    controller_after = {**before, "oom": 1, "oom_kill": 1}
    phase = {
        "phase": "pipeline-finished",
        "timed_pipeline_started": True,
        "timed_pipeline_finished": True,
        "pipeline_succeeded": False,
        "pipeline_failure": "WorkerCrashedError: worker died unexpectedly",
        "kernel_oom_baseline": {
            "host_vmstat_oom_kill": 0,
            "session_cgroup": before,
        },
        "kernel_oom_finished": {
            "host_vmstat_oom_kill": 0,
            "session_cgroup": before,
        },
    }
    artifact = {
        "valid": False,
        "status": "rejected",
        "backend": "pyarrow",
        "workload": "full",
        "timed_pipeline_started": True,
        "timed_pipeline_finished": True,
        "pipeline_succeeded": False,
        "last_durable_phase": phase,
        "exception": "ray.exceptions.WorkerCrashedError: worker died unexpectedly",
        "rejection_reasons": [],
    }
    _apply_controller_outcome(
        artifact,
        spec=CPU_ORIGIN_ID_SPEC,
        returncode=1,
        timed_out=False,
        timeout_phase=None,
        last_phase=phase,
        timeout_s=TRIAL_TIMEOUT_S,
        oom_before=0,
        oom_after=1,
        cgroup_oom_before=before,
        cgroup_oom_after=controller_after,
    )
    assert artifact["status"] == "rejected"
    assert not _accepted_capacity_result(artifact)


def test_sigkill_before_timed_pipeline_is_never_capacity_oom():
    phase = {
        "phase": "pipeline-arming",
        "timed_pipeline_started": False,
        "kernel_oom_baseline": {
            "host_vmstat_oom_kill": 0,
            "session_cgroup": None,
        },
    }
    artifact = {
        "valid": False,
        "status": "missing-artifact",
        "backend": "pyarrow",
        "workload": "full",
        "timed_pipeline_started": False,
        "last_durable_phase": phase,
        "rejection_reasons": [],
    }
    before = {
        "source": "/sys/fs/cgroup/user.slice/session.scope/memory.events.local",
        "cgroup": "/user.slice/session.scope",
        "oom": 0,
        "oom_kill": 0,
    }
    phase["kernel_oom_baseline"]["session_cgroup"] = before
    _apply_controller_outcome(
        artifact,
        spec=CPU_ORIGIN_ID_SPEC,
        returncode=-9,
        timed_out=False,
        timeout_phase=None,
        last_phase=phase,
        timeout_s=TRIAL_TIMEOUT_S,
        oom_before=0,
        oom_after=1,
        cgroup_oom_before=before,
        cgroup_oom_after={**before, "oom": 1, "oom_kill": 1},
    )
    assert artifact["status"] == "rejected"
    assert not _accepted_capacity_result(artifact)


def test_prelaunch_oom_increment_cannot_prove_timed_pipeline_oom():
    prelaunch = {
        "source": "/sys/fs/cgroup/user.slice/session.scope/memory.events.local",
        "cgroup": "/user.slice/session.scope",
        "oom": 0,
        "oom_kill": 0,
    }
    timed = {**prelaunch, "oom": 1, "oom_kill": 1}
    phase = {
        "phase": "pipeline-running",
        "timed_pipeline_started": True,
        "kernel_oom_baseline": {
            "host_vmstat_oom_kill": 10,
            "session_cgroup": timed,
        },
    }
    artifact = {
        "valid": False,
        "status": "missing-artifact",
        "backend": "pyarrow",
        "workload": "full",
        "timed_pipeline_started": True,
        "last_durable_phase": phase,
        "rejection_reasons": [],
    }
    _apply_controller_outcome(
        artifact,
        spec=CPU_ORIGIN_ID_SPEC,
        returncode=-9,
        timed_out=False,
        timeout_phase=None,
        last_phase=phase,
        timeout_s=TRIAL_TIMEOUT_S,
        oom_before=9,
        oom_after=10,
        cgroup_oom_before=prelaunch,
        cgroup_oom_after=timed,
    )
    assert artifact["status"] == "rejected"
    assert not _accepted_capacity_result(artifact)


def test_post_pipeline_timeout_is_not_a_cpu_capacity_result():
    artifact = {
        "valid": False,
        "status": "missing-artifact",
        "backend": "pyarrow",
        "workload": "full",
        "timed_pipeline_started": True,
        "last_durable_phase": {
            "phase": "validation-running",
            "timed_pipeline_started": True,
        },
        "rejection_reasons": [],
    }
    _apply_controller_outcome(
        artifact,
        spec=CPU_ORIGIN_ID_SPEC,
        returncode=-15,
        timed_out=True,
        timeout_phase="post-pipeline",
        last_phase=artifact["last_durable_phase"],
        timeout_s=TRIAL_TIMEOUT_S,
        oom_before=1,
        oom_after=1,
    )
    assert artifact["status"] == "rejected"
    assert not _accepted_capacity_result(artifact)


def test_gate_manifest_binds_commit_overlay_and_hashed_logs(tmp_path, monkeypatch):
    gate_root = tmp_path / "gate-root"
    logs = gate_root / "logs"
    logs.mkdir(parents=True)
    results = []
    for gate in GATE_SEQUENCE:
        log = logs / f"{gate.name}.log"
        log.write_text(f"{gate.name}: passed\n", encoding="utf-8")
        results.append(
            {
                "name": gate.name,
                "valid": True,
                "nodeid": gate.nodeid,
                "command": controller._gate_command(Path("/venv/python"), gate),
                "environment": dict(gate.environment),
                "uses_ray": gate.uses_ray,
                "returncode": 0,
                "timed_out": False,
                "process_group_terminated": True,
                "identity_valid": True,
                "error": None,
                "resource_checks": {
                    "before": {"ray": {"valid": True}, "gpu": {"valid": True}},
                    "after": {"ray": {"valid": True}, "gpu": {"valid": True}},
                },
                "log": str(log),
                "log_bytes": log.stat().st_size,
                "log_sha256": controller._sha256_file(log),
            }
        )
    monkeypatch.setattr(controller, "RAID_ROOT", tmp_path)
    monkeypatch.setattr(
        controller,
        "frozen_campaign",
        lambda _: {"campaign_digest": "frozen-campaign"},
    )
    dataset_identity = {
        "dataset_root": str(tmp_path.resolve()),
        "manifest_file_sha256": "c" * 64,
        "manifest_declared_sha256": "d" * 64,
        "selected_parquet_files": 153,
        "selected_parquet_physical_bytes": 123,
        "selected_parquet_sha256_digest": "e" * 64,
        "identity_digest": "f" * 64,
    }
    monkeypatch.setattr(
        controller,
        "_dataset_identity",
        lambda *_args, **_kwargs: dataset_identity,
    )
    body = {
        "schema_version": GATE_SCHEMA_VERSION,
        "valid": True,
        "git_head": "a" * 40,
        "git_branch": IMPLEMENTATION_BRANCH,
        "overlay_manifest_sha256": "b" * 64,
        "python": str(Path("/venv/python").resolve()),
        "campaign_digest": "frozen-campaign",
        "dataset_root": str(tmp_path.resolve()),
        "dataset_identity": dataset_identity,
        "gate_root": str(gate_root.resolve()),
        "gates": results,
    }
    manifest = {**body, "manifest_digest": _canonical_digest(body)}
    path = gate_root / "GATES.json"
    controller.write_json(path, manifest)
    validated = _validate_prerequisite_gates(
        path,
        dataset_root=tmp_path,
        expected_head="a" * 40,
        overlay={"manifest_sha256": "b" * 64},
        python=Path("/venv/python"),
    )
    assert validated["source_manifest"] == str(path.resolve())

    (logs / f"{GATE_SEQUENCE[0].name}.log").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="log (size )?changed"):
        _validate_prerequisite_gates(
            path,
            dataset_root=tmp_path,
            expected_head="a" * 40,
            overlay={"manifest_sha256": "b" * 64},
            python=Path("/venv/python"),
        )


def test_timeline_proves_gpu_overlap_and_blocking_cpu():
    timeline = _timeline(
        e2e_started_wall_ns=1,
        e2e_finished_wall_ns=100,
        source={
            "first_read_started_wall_time_ns": 2,
            "last_input_produced_wall_time_ns": 50,
        },
        sink={
            "telemetry": {
                "first_writer_work_started_at_ns": 70,
                "drain_completed_at_ns": 99,
            }
        },
        gpu={
            "first_gpu_run_committed_at_ns": 20,
            "finalization_complete_at_ns": 80,
        },
        cpu={
            "blocking_all_to_all": {"inputs_complete_at_ns": 50},
            "push_based_shuffle": {"first_map_task_submitted_at_ns": 51},
        },
    )
    assert timeline["overlap_proofs"] == {
        "read_and_gpu_run_creation": True,
        "gpu_merge_or_extraction_and_parquet_write": True,
        "cpu_shuffle_started_before_eos": False,
    }


def test_bounded_durable_validator_reopens_committed_parquet(tmp_path):
    output = tmp_path / "output"
    table = pa.table(
        {
            ORIGIN_AIRPORT_ID_KEY: pa.array([10135, 10135, 10257], type=pa.int64()),
            "row_id": pa.array([2, 0, 1], type=pa.int64()),
        }
    )
    sink = StreamingParquetDatasink(output)
    sink.on_write_start(table.schema)
    receipt = sink.write([table], TaskContext(task_idx=0, op_name="test"))
    sink.on_write_complete(
        WriteResult(num_rows=3, size_bytes=table.nbytes, write_returns=[receipt])
    )
    assert sink.result is not None
    validation = _validate_durable_parquet(
        output,
        sink.result,
        table.schema,
        sort_key=ORIGIN_AIRPORT_ID_KEY,
        expected_rows=3,
        expected_row_id_sum=3,
        expected_sort_key_stats={
            "null_rows": 0,
            "cardinality": 2,
            "min": 10135,
            "max": 10257,
            "frequency_digest": _frequency_digest({"10135": 2, "10257": 1}),
        },
    )
    assert validation["valid"], validation["rejection_reasons"]
    assert validation["file_count"] == 1
    assert validation["row_group_count"] == 1


@pytest.mark.parametrize(
    ("values", "valid"),
    (([1, 2, None], True), ([1, None, 2], False)),
)
def test_durable_validator_enforces_numeric_nulls_last(tmp_path, values, valid):
    output = tmp_path / "output"
    table = pa.table(
        {
            ORIGIN_AIRPORT_ID_KEY: pa.array(values, type=pa.int64()),
            "row_id": pa.array([0, 1, 2], type=pa.int64()),
        }
    )
    sink = StreamingParquetDatasink(output)
    sink.on_write_start(table.schema)
    receipt = sink.write([table], TaskContext(task_idx=0, op_name="test"))
    sink.on_write_complete(
        WriteResult(num_rows=3, size_bytes=table.nbytes, write_returns=[receipt])
    )
    assert sink.result is not None
    validation = _validate_durable_parquet(
        output,
        sink.result,
        table.schema,
        sort_key=ORIGIN_AIRPORT_ID_KEY,
        expected_rows=3,
        expected_row_id_sum=3,
        expected_sort_key_stats={
            "null_rows": 1,
            "cardinality": 2,
            "min": 1,
            "max": 2,
            "frequency_digest": _frequency_digest({"1": 1, "2": 1}),
        },
    )
    assert validation["valid"] is valid
    assert validation["nulls_last"] is valid


def test_worker_contains_no_intermediate_dataset_actions():
    source = (Path(__file__).parents[1] / "streaming_parquet_e2e_worker.py").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        ".materialize(",
        ".count(",
        ".take(",
        ".schema(",
        ".to_arrow_refs(",
    ):
        assert forbidden not in source
