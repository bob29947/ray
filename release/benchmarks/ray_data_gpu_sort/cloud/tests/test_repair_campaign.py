import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from release.benchmarks.ray_data_gpu_sort.cloud import (
    lifecycle,
    report as report_module,
    study,
    worker,
)
from release.benchmarks.ray_data_gpu_sort.cloud.common import digest, file_sha256
from release.benchmarks.ray_data_gpu_sort.cloud.report import (
    _bundle_artifact_provenance,
    _results,
    build,
)
from release.benchmarks.ray_data_gpu_sort.cloud.study import (
    CPU_REPAIR_TRIAL_MODE,
    CPU_REPAIR_TRIAL_NAMES,
    GPU_245X_REPAIR_TRIAL_MODE,
    GPU_245X_REPAIR_TRIAL_NAMES,
    GPU_REPAIR_TRIAL_MODE,
    GPU_REPAIR_TRIAL_NAMES,
    _commands,
    _trial_values,
    _validate_prior_cpu_teardown,
    _validate_prior_gpu_teardown,
    execute_arm,
)


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _write_plan(path: Path, body: dict[str, Any]) -> dict[str, Any]:
    value = {**body, "plan_sha256": digest(body)}
    _write(path, value)
    return value


def _bundle(label: str) -> dict[str, Any]:
    wheel = _sha("wheel")
    files = {
        "generated/dataset-manifest.json": _sha("portable-manifest"),
        "wheelhouse/ray.whl": wheel,
        "python/ray/data/a.py": _sha(f"overlay-{label}"),
        "release/benchmarks/ray_data_gpu_sort/cloud/worker.py": _sha(f"worker-{label}"),
        "release/benchmarks/ray_data_gpu_sort/cloud/report.py": _sha(f"report-{label}"),
    }
    return {
        "bundle_digest": digest(files),
        "files": files,
        "ray_version": "2.55.1",
        "ray_commit": "commit",
        "wheel_sha256": wheel,
    }


def _full_study(path: Path) -> dict[str, Any]:
    campaign = "original"
    return _write_plan(
        path,
        {
            "schema_version": 1,
            "kind": "gpu_sort_bts_cloud_study",
            "campaign": campaign,
            "dataset_digest": _sha("dataset"),
            "dataset_manifest_sha256": _sha("portable-manifest"),
            "bundle": _bundle("original"),
            "arms": {
                "gpu": {
                    "cluster_name": f"ray-gpu-sort-{campaign}-gpu",
                    "trials": [
                        trial.to_dict() for trial in _trial_values("gpu", "full")
                    ],
                    "topology": {
                        "nodes": 16,
                        "vcpus_per_node": 16,
                        "gpus_per_node": 1,
                        "instance_type": "g6.4xlarge",
                    },
                },
                "cpu": {
                    "cluster_name": f"ray-gpu-sort-{campaign}-cpu",
                    "trials": [
                        trial.to_dict() for trial in _trial_values("cpu", "full")
                    ],
                    "topology": {
                        "nodes": 16,
                        "vcpus_per_node": 16,
                        "gpus_per_node": 0,
                        "instance_type": "m5dn.4xlarge",
                    },
                },
            },
        },
    )


def _legacy_execution(study_root: Path, arm: str) -> tuple[Path, dict[str, Any]]:
    study = _full_study(study_root / "study.json")
    execution = study_root / "executions" / arm
    is_gpu = arm == "gpu"
    ids = [f"i-{index + (0 if is_gpu else 100):016x}" for index in range(16)]
    lifecycle_body = {
        "schema_version": 1,
        "campaign": study["campaign"],
        "arm": arm,
        "cluster_name": study["arms"][arm]["cluster_name"],
        "cluster_yaml_sha256": _sha("cluster"),
        "command_plan_sha256": _sha("commands"),
        "steps": [],
        "expected": {
            "nodes": 16,
            "cpus": 256.0,
            "gpus": 16.0 if is_gpu else 0.0,
            "instance_type": "g6.4xlarge" if is_gpu else "m5dn.4xlarge",
            "single_availability_zone": True,
        },
        "same_ec2_ids": True,
        "fresh_ray_node_ids_each_trial": True,
        "ray_default_plasma": True,
    }
    lifecycle_plan = _write_plan(execution / "lifecycle-plan.json", lifecycle_body)
    teardown = {"verified_empty": True, "terminated_instance_ids": ids}
    _write(execution / "instances.json", {"instance_ids": ids})
    _write(execution / "teardown.json", teardown)
    _write(execution / "result.json", {"status": "success", "teardown": teardown})
    assert lifecycle_plan["arm"] == arm
    return execution / "teardown.json", study


def _legacy_gpu_execution(study_root: Path) -> tuple[Path, dict[str, Any]]:
    return _legacy_execution(study_root, "gpu")


def _legacy_cpu_execution(study_root: Path) -> tuple[Path, dict[str, Any]]:
    return _legacy_execution(study_root, "cpu")


def _result(
    backend: str,
    elapsed: float | None,
    *,
    kind: str,
    repetition: int = 1,
    scale: tuple[int, int] = (1, 1),
    wave_fraction: float | None = None,
    plan: dict[str, Any] | None = None,
    cell: dict[str, Any] | str = "full",
) -> dict[str, Any]:
    source = {
        "rows": 100,
        "blocks": 2,
        "decoded_bytes": 1000,
        "schema": "a: int64",
    }
    return {
        "valid": elapsed is not None,
        "kind": kind,
        "backend": backend,
        "cell": cell,
        "repetition": repetition,
        "wave_fraction": wave_fraction,
        "cold_sort_s": elapsed,
        "input": source,
        "output": source if elapsed is not None else {},
        "validation": {
            "ordered": elapsed is not None,
            "row_id_sum": 10,
            "expected_row_id_sum": 10,
        },
        "rejection_reasons": [] if elapsed is not None else ["failed"],
        "scale_numerator": scale[0],
        "scale_denominator": scale[1],
        "throughput_rows_s": None if elapsed is None else 100 / elapsed,
        "throughput_gib_s": None if elapsed is None else 1000 / (1 << 30) / elapsed,
        "ray_object_store_io": {
            "totals": {"spilled_bytes_total": 0, "restored_bytes_total": 0}
        },
        "gpu_stats": {},
        "resources": {},
        "plan": plan or {"kind": kind, "rows": 100, "blocks": 2, "digest": _sha(kind)},
    }


def test_repair_trial_mode_is_the_minimal_ordered_gpu_campaign(tmp_path: Path) -> None:
    values = _trial_values("gpu", GPU_REPAIR_TRIAL_MODE)
    assert tuple(value.name for value in values) == GPU_REPAIR_TRIAL_NAMES
    resets = {value.name: tmp_path / f"{value.name}.yaml" for value in values}
    runtimes = {value.name: Path("/mnt/nvme/runtime") / value.name for value in values}
    commands = _commands(
        arm="gpu",
        python="/opt/ray/bin/python",
        generated=tmp_path,
        reset_yamls=resets,
        reset_runtimes=runtimes,
        dataset_root=Path("/mnt/nvme/dataset"),
        results_root=Path("/mnt/nvme/results"),
        trial_values=values,
    )
    assert [item["name"] for item in commands] == [
        "mkdir-results",
        "inventory",
        "stage-dataset",
        "transport-smoke",
        "tune-2x-wave-0500",
        "tune-2x-wave-0375",
        "select-wave",
        "natural-2x-selected-r2",
        "natural-245x-r1",
        "natural-245x-r2",
    ]
    assert all(
        command.get("restart_yaml") == str(resets[command["name"]])
        for command in commands
        if command["name"] in GPU_REPAIR_TRIAL_NAMES
    )
    assert all(
        command["timeout_seconds"] == 14400
        for command in commands
        if command["name"] in GPU_REPAIR_TRIAL_NAMES
        and command["name"] != "transport-smoke"
    )


def test_gpu_245x_repair_mode_runs_only_fixed_wave_observations(
    tmp_path: Path,
) -> None:
    values = _trial_values("gpu", GPU_245X_REPAIR_TRIAL_MODE)
    assert tuple(value.name for value in values) == GPU_245X_REPAIR_TRIAL_NAMES
    assert all(
        value.wave_fraction == 0.50 and value.selected_wave_file is None
        for value in values
        if value.kind == "natural"
    )
    resets = {value.name: tmp_path / f"{value.name}.yaml" for value in values}
    runtimes = {value.name: Path("/mnt/nvme/runtime") / value.name for value in values}
    commands = _commands(
        arm="gpu",
        python="/opt/ray/bin/python",
        generated=tmp_path,
        reset_yamls=resets,
        reset_runtimes=runtimes,
        dataset_root=Path("/mnt/nvme/dataset"),
        results_root=Path("/mnt/nvme/results"),
        trial_values=values,
    )
    assert [item["name"] for item in commands] == [
        "mkdir-results",
        "inventory",
        "stage-dataset",
        *GPU_245X_REPAIR_TRIAL_NAMES,
    ]
    assert "select-wave" not in {item["name"] for item in commands}
    trial_commands = {
        command["name"]: command
        for command in commands
        if command["name"] in GPU_245X_REPAIR_TRIAL_NAMES
    }
    assert all(
        command["restart_yaml"] == str(resets[name])
        for name, command in trial_commands.items()
    )
    for name in ("natural-245x-r1", "natural-245x-r2"):
        assert "--wave-fraction 0.5" in trial_commands[name]["command"]
        assert "--selected-wave" not in trial_commands[name]["command"]
        assert trial_commands[name]["timeout_seconds"] == 1800


def test_cpu_repair_mode_runs_only_missing_cpu_observations(tmp_path: Path) -> None:
    values = _trial_values("cpu", CPU_REPAIR_TRIAL_MODE)
    assert tuple(value.name for value in values) == CPU_REPAIR_TRIAL_NAMES
    resets = {value.name: tmp_path / f"{value.name}.yaml" for value in values}
    runtimes = {value.name: Path("/mnt/nvme/runtime") / value.name for value in values}
    commands = _commands(
        arm="cpu",
        python="/opt/ray/bin/python",
        generated=tmp_path,
        reset_yamls=resets,
        reset_runtimes=runtimes,
        dataset_root=Path("/mnt/nvme/dataset"),
        results_root=Path("/mnt/nvme/results"),
        trial_values=values,
    )
    assert [item["name"] for item in commands] == [
        "mkdir-results",
        "inventory",
        "stage-dataset",
        *CPU_REPAIR_TRIAL_NAMES,
    ]
    assert "select-wave" not in {item["name"] for item in commands}
    assert all(
        command.get("restart_yaml") == str(resets[command["name"]])
        for command in commands
        if command["name"] in CPU_REPAIR_TRIAL_NAMES
    )


def test_smoke_result_has_the_same_strict_artifact_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root = tmp_path / "dataset"
    _write(dataset_root / "manifest.json", {})
    args = SimpleNamespace(
        dataset_root=dataset_root,
        kind="smoke",
        backend="smoke",
        cell="full",
        repetition=1,
        scale_numerator=1,
        scale_denominator=1,
        wave_fraction=0.5,
        spill_directory=tmp_path / "spill",
    )

    class Table:
        num_rows = 100
        schema = None

        def equals(self, other: Any) -> bool:
            return isinstance(other, Table)

    class Schema:
        def equals(self, other: Any, check_metadata: bool = False) -> bool:
            return isinstance(other, Schema) and check_metadata

    table = Table()
    table.schema = Schema()
    fake_ray = SimpleNamespace(__version__="2.55.1", __commit__="commit")
    plan = {"kind": "smoke", "rows": 100, "blocks": 16, "digest": _sha("smoke")}
    monkeypatch.setattr(worker, "_configure_gpu_sort", lambda *args: None)
    monkeypatch.setattr(worker, "_verified_bundle_digest", lambda root: _sha("bundle"))
    monkeypatch.setattr(worker, "smoke_slices", lambda manifest: ())
    monkeypatch.setattr(
        worker, "plan_dict", lambda slices, kind: {**plan, "slices": []}
    )
    monkeypatch.setattr(worker, "_materialize", lambda *args: (object(), {"rows": 100}))
    monkeypatch.setattr(worker, "_stable_spill", lambda ray: {})
    monkeypatch.setattr(worker, "_start_monitors", lambda *args: None)
    monkeypatch.setattr(worker, "_stop_monitors", lambda *args: {})
    monkeypatch.setattr(worker, "_spill_delta", lambda *args: {})
    monkeypatch.setattr(worker, "_sort", lambda *args: (object(), 1.0))
    monkeypatch.setattr(worker, "_exact_table", lambda *args: table)
    monkeypatch.setattr(worker, "_metadata", lambda dataset: {"rows": 100})
    monkeypatch.setattr(
        worker, "get_last_run_stats", lambda dataset: {"ranks": [{}] * 16}
    )
    monkeypatch.setattr(worker, "rank_stats", lambda stats: stats["ranks"])
    result = worker._run_smoke(fake_ray, args, {"schema_names": []}, [{}] * 16)
    identity = result["artifact_identity"]
    assert identity["digest"] == digest(
        {key: item for key, item in identity.items() if key != "digest"}
    )
    assert identity["trial"] == {
        "kind": "smoke",
        "backend": "smoke",
        "cell": "full",
        "repetition": 1,
        "scale_numerator": 1,
        "scale_denominator": 1,
        "wave_fraction": 0.5,
    }
    assert result["ray_version"] == "2.55.1"
    assert result["ray_commit"] == "commit"


def test_location_lookup_timeout_is_bounded_and_nonfatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refs = [object() for _ in range(129)]
    observed_timeouts = []

    def timeout(batch: list[Any], *, timeout_ms: int) -> dict[Any, Any]:
        observed_timeouts.append((len(batch), timeout_ms))
        raise TimeoutError("location lookup deadline")

    ray = SimpleNamespace(experimental=SimpleNamespace(get_object_locations=timeout))
    monkeypatch.setattr(worker, "_refs", lambda dataset: refs)
    result = worker._locations(ray, object())
    assert observed_timeouts == [
        (128, worker.LOCATION_LOOKUP_TIMEOUT_MS),
        (1, worker.LOCATION_LOOKUP_TIMEOUT_MS),
    ]
    assert result["located_objects"] == 0
    assert result["all_locatable"] is False
    assert result["lookup_timeout_batches"] == 2
    assert result["lookup_timed_out_objects"] == 129
    assert result["lookup_error_batches"] == 0


def test_empty_ordered_partition_does_not_require_row_id_column() -> None:
    import pyarrow as pa

    result = worker._ordered_partition(
        pa.table({"Origin": pa.array([], type=pa.string())}), ("Origin",)
    )
    assert result == {
        "ordered": True,
        "rows": 0,
        "first": None,
        "last": None,
        "row_id_sum": 0,
    }


@pytest.mark.parametrize(
    "trial_mode", (GPU_REPAIR_TRIAL_MODE, GPU_245X_REPAIR_TRIAL_MODE)
)
def test_repair_execution_rechecks_prior_teardown_receipt(
    trial_mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, _ = _legacy_gpu_execution(tmp_path / "original")
    provenance = _validate_prior_gpu_teardown(receipt)
    plan = tmp_path / "repair-study.json"
    plan_value = _write_plan(
        plan,
        {
            "kind": "gpu_sort_bts_cloud_study",
            "trial_mode": trial_mode,
            "prior_gpu_teardown": provenance,
            "arms": {"gpu": {"lifecycle_args": ["--campaign", "repair"]}},
        },
    )
    called: list[list[str]] = []
    monkeypatch.setattr(lifecycle, "main", lambda args: called.append(args) or 0)
    assert (
        execute_arm(plan, "gpu", tmp_path / "live", execute=True, confirm_plan_sha="x")
        == 0
    )
    assert called
    tampered_plan = {**plan_value, "trial_mode": "full"}
    _write(plan, tampered_plan)
    with pytest.raises(ValueError, match="study plan identity/digest"):
        execute_arm(
            plan,
            "gpu",
            tmp_path / "tampered-plan",
            execute=True,
            confirm_plan_sha="x",
        )
    _write(plan, plan_value)
    instances = Path(provenance["files"]["instances"]["path"])
    _write(instances, {"instance_ids": ["i-tampered"] * 16})
    with pytest.raises(RuntimeError, match="instance IDs"):
        execute_arm(
            plan, "gpu", tmp_path / "live-2", execute=True, confirm_plan_sha="x"
        )


def test_cpu_repair_binds_prior_cpu_teardown_without_gpu_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, _ = _legacy_cpu_execution(tmp_path / "original")
    provenance = _validate_prior_cpu_teardown(receipt)
    plan = tmp_path / "cpu-repair-study.json"
    _write_plan(
        plan,
        {
            "kind": "gpu_sort_bts_cloud_study",
            "trial_mode": CPU_REPAIR_TRIAL_MODE,
            "prior_cpu_teardown": provenance,
            "arms": {"cpu": {"lifecycle_args": ["--campaign", "cpu-repair"]}},
        },
    )
    called: list[list[str]] = []
    monkeypatch.setattr(lifecycle, "main", lambda args: called.append(args) or 0)
    artifact_root = tmp_path / "live/cpu"
    assert (
        execute_arm(
            plan,
            "cpu",
            artifact_root,
            execute=True,
            confirm_plan_sha="x",
        )
        == 0
    )
    assert called
    assert not (artifact_root.parent / "gpu/teardown.json").exists()

    instances = Path(provenance["files"]["instances"]["path"])
    _write(instances, {"instance_ids": ["i-tampered"] * 16})
    with pytest.raises(RuntimeError, match="CPU instance IDs"):
        execute_arm(
            plan,
            "cpu",
            tmp_path / "live-2/cpu",
            execute=True,
            confirm_plan_sha="x",
        )


def test_full_cpu_mode_still_requires_sibling_gpu_teardown(tmp_path: Path) -> None:
    plan = tmp_path / "full-study.json"
    _write_plan(
        plan,
        {
            "kind": "gpu_sort_bts_cloud_study",
            "trial_mode": "full",
            "arms": {"cpu": {"lifecycle_args": ["--campaign", "full"]}},
        },
    )
    with pytest.raises(RuntimeError, match="GPU arm's verified-empty teardown"):
        execute_arm(
            plan,
            "cpu",
            tmp_path / "executions/cpu",
            execute=True,
            confirm_plan_sha="x",
        )


def test_cpu_repair_cli_forwards_prior_cpu_teardown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: dict[str, Any] = {}
    monkeypatch.setattr(study, "prepare", lambda **kwargs: called.update(kwargs))
    receipt = tmp_path / "prior/teardown.json"
    assert (
        study.main(
            [
                "prepare",
                "--campaign-id",
                "cpu-repair",
                "--trial-mode",
                CPU_REPAIR_TRIAL_MODE,
                "--prior-cpu-teardown",
                str(receipt),
                "--local-config",
                str(tmp_path / "local.json"),
                "--dataset-manifest",
                str(tmp_path / "manifest.json"),
                "--output-root",
                str(tmp_path / "output"),
            ]
        )
        == 0
    )
    assert called["trial_mode"] == CPU_REPAIR_TRIAL_MODE
    assert called["prior_cpu_teardown"] == receipt
    assert called["prior_gpu_teardown"] is None


def test_gpu_245x_repair_cli_forwards_prior_gpu_teardown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: dict[str, Any] = {}
    monkeypatch.setattr(study, "prepare", lambda **kwargs: called.update(kwargs))
    receipt = tmp_path / "prior/teardown.json"
    assert (
        study.main(
            [
                "prepare",
                "--campaign-id",
                "gpu-245x-repair",
                "--trial-mode",
                GPU_245X_REPAIR_TRIAL_MODE,
                "--prior-gpu-teardown",
                str(receipt),
                "--local-config",
                str(tmp_path / "local.json"),
                "--dataset-manifest",
                str(tmp_path / "manifest.json"),
                "--output-root",
                str(tmp_path / "output"),
            ]
        )
        == 0
    )
    assert called["trial_mode"] == GPU_245X_REPAIR_TRIAL_MODE
    assert called["prior_gpu_teardown"] == receipt
    assert called["prior_cpu_teardown"] is None


def _report_fixture(tmp_path: Path) -> dict[str, Path]:
    original_root = tmp_path / "original"
    prior_teardown, original_plan = _legacy_gpu_execution(original_root)
    study = original_root / "study.json"
    gpu, cpu = tmp_path / "gpu", tmp_path / "cpu"
    cell = {"name": "full", "columns": ["a"], "keys": ["a"]}
    plans = {
        "smoke": {
            "kind": "smoke",
            "rows": 100,
            "blocks": 2,
            "digest": _sha("smoke-plan"),
        },
        "2x": {"kind": "natural", "rows": 200, "blocks": 4, "digest": _sha("2x-plan")},
        "245x": {
            "kind": "natural",
            "rows": 245,
            "blocks": 5,
            "digest": _sha("245x-plan"),
        },
    }
    for repetition, elapsed in ((1, 50.0), (2, 52.0)):
        _write(
            gpu / f"trend-full-r{repetition}.json",
            _result("gpu", elapsed, kind="trend", repetition=repetition, cell=cell),
        )
    _write(
        cpu / "trend-full-r1.json", _result("pyarrow", 200.0, kind="trend", cell=cell)
    )
    _write(
        cpu / "natural-2x-r1.json",
        _result("pyarrow", 300.0, kind="natural", scale=(2, 1), cell=cell),
    )
    _write(
        cpu / "natural-245x-r1.json",
        _result("pyarrow", 400.0, kind="natural", scale=(245, 100), cell=cell),
    )

    original_specs = {
        "transport-smoke.json": ("smoke", "smoke", 2.0, 1, (1, 1), 0.5),
        "tune-2x-wave-0500.json": ("gpu", "natural", 100.0, 1, (2, 1), 0.5),
        "tune-2x-wave-0375.json": ("gpu", "natural", 90.0, 1, (2, 1), 0.375),
        "natural-2x-selected-r2.json": ("gpu", "natural", 91.0, 2, (2, 1), 0.375),
        "natural-245x-r1.json": ("gpu", "natural", None, 1, (245, 100), 0.375),
        "natural-245x-r2.json": ("gpu", "natural", None, 2, (245, 100), 0.375),
    }
    for filename, (
        backend,
        kind,
        elapsed,
        repetition,
        scale,
        wave,
    ) in original_specs.items():
        plan = plans[
            "smoke" if kind == "smoke" else "2x" if scale == (2, 1) else "245x"
        ]
        value = _result(
            backend,
            elapsed,
            kind=kind,
            repetition=repetition,
            scale=scale,
            wave_fraction=wave,
            plan=plan,
            cell="full" if kind == "smoke" else cell,
        )
        if kind == "smoke":
            value.update(
                {
                    "cpu_sort_s": 1.0,
                    "gpu_sort_s": 2.0,
                    "validation": {"exact_every_row_value": True, "rows": 100},
                }
            )
        _write(gpu / filename, value)
    _write(
        gpu / "configs/selected-wave.json",
        {"selected_wave_fraction": 0.375, "reason": "old"},
    )

    repair_root = tmp_path / "repair"
    execution = repair_root / "executions/gpu"
    repair = execution / "remote-results"
    prepared = repair_root / "prepared/gpu"
    launch, commands = prepared / "launch.yaml", prepared / "commands.json"
    launch.parent.mkdir(parents=True, exist_ok=True)
    launch.write_text("cluster", encoding="utf-8")
    commands.write_text("commands", encoding="utf-8")
    repair_bundle = _bundle("repair")
    repair_body = {
        "schema_version": 1,
        "kind": "gpu_sort_bts_cloud_study",
        "campaign": "repair",
        "trial_mode": "gpu-repair",
        "dataset_digest": original_plan["dataset_digest"],
        "dataset_manifest_sha256": original_plan["dataset_manifest_sha256"],
        "bundle": repair_bundle,
        "prior_gpu_teardown": _validate_prior_gpu_teardown(prior_teardown),
        "arms": {
            "gpu": {
                "cluster_name": "ray-gpu-sort-repair-gpu",
                "launch_yaml": str(launch),
                "command_plan": str(commands),
                "topology": {
                    "nodes": 16,
                    "vcpus_per_node": 16,
                    "gpus_per_node": 1,
                    "instance_type": "g6.4xlarge",
                },
                "trials": [
                    trial.to_dict() for trial in _trial_values("gpu", "gpu-repair")
                ],
            }
        },
    }
    repair_study = repair_root / "study.json"
    repair_plan = _write_plan(repair_study, repair_body)
    ids = [f"i-{index + 100:016x}" for index in range(16)]
    lifecycle_body = {
        "schema_version": 1,
        "campaign": "repair",
        "arm": "gpu",
        "cluster_name": "ray-gpu-sort-repair-gpu",
        "cluster_yaml_sha256": file_sha256(launch),
        "command_plan_sha256": file_sha256(commands),
        "steps": [],
        "expected": {
            "nodes": 16,
            "cpus": 256.0,
            "gpus": 16.0,
            "instance_type": "g6.4xlarge",
            "single_availability_zone": True,
        },
        "same_ec2_ids": True,
        "fresh_ray_node_ids_each_trial": True,
        "ray_default_plasma": True,
    }
    lifecycle_plan = _write_plan(execution / "lifecycle-plan.json", lifecycle_body)
    teardown = {
        "schema_version": 2,
        "verified_empty": True,
        "terminated_instance_ids": ids,
        "campaign": "repair",
        "arm": "gpu",
        "cluster_name": "ray-gpu-sort-repair-gpu",
        "lifecycle_plan_sha256": lifecycle_plan["plan_sha256"],
        "instance_ids_sha256": digest(ids),
    }
    _write(execution / "instances.json", {"instance_ids": ids})
    _write(execution / "teardown.json", teardown)
    _write(execution / "result.json", {"status": "success", "teardown": teardown})

    staged_manifest = _sha("staged-manifest")
    staged_nodes = [
        {
            "instance_id": ids[index],
            "ray_node_id": f"gpu-node-{index}",
            "dataset_digest": repair_plan["dataset_digest"],
            "manifest_sha256": staged_manifest,
        }
        for index in range(16)
    ]
    _write(
        repair / "admin/dataset-stage.json",
        {
            "dataset_digest": repair_plan["dataset_digest"],
            "nodes": staged_nodes,
            "digest": digest(staged_nodes),
        },
    )
    _write(
        repair / "configs/selected-wave.json",
        {
            "selected_wave_fraction": 0.5,
            "reason": "candidate improved by less than 3%",
            "observed_relative_improvement": 0.0125,
            "minimum_relative_improvement": 0.03,
        },
    )
    repair_times = {
        "transport-smoke": 2.0,
        "tune-2x-wave-0500": 80.0,
        "tune-2x-wave-0375": 79.0,
        "natural-2x-selected-r2": 82.0,
        "natural-245x-r1": 120.0,
        "natural-245x-r2": 122.0,
    }
    provenance = _bundle_artifact_provenance(repair_plan)
    for trial in _trial_values("gpu", "gpu-repair"):
        wave = trial.wave_fraction
        if wave is None:
            wave = 0.5
        scale = (trial.scale_numerator, trial.scale_denominator)
        plan = plans[
            "smoke" if trial.kind == "smoke" else "2x" if scale == (2, 1) else "245x"
        ]
        value = _result(
            trial.backend,
            repair_times[trial.name],
            kind=trial.kind,
            repetition=trial.repetition,
            scale=scale,
            wave_fraction=wave,
            plan=plan,
            cell="full" if trial.kind == "smoke" else cell,
        )
        identity_body = {
            "trial": {
                "kind": trial.kind,
                "backend": trial.backend,
                "cell": trial.cell,
                "repetition": trial.repetition,
                "scale_numerator": trial.scale_numerator,
                "scale_denominator": trial.scale_denominator,
                "wave_fraction": wave,
            },
            "input_plan_digest": plan["digest"],
            "dataset_manifest_sha256": staged_manifest,
            **provenance,
        }
        value.update(
            {
                "artifact_identity": {**identity_body, "digest": digest(identity_body)},
                "ray_version": repair_bundle["ray_version"],
                "ray_commit": repair_bundle["ray_commit"],
            }
        )
        if trial.kind == "smoke":
            value.update(
                {
                    "cpu_sort_s": 1.0,
                    "gpu_sort_s": 2.0,
                    "validation": {"exact_every_row_value": True, "rows": 100},
                }
            )
        _write(repair / "results" / f"{trial.name}.json", value)
    return {
        "study": study,
        "gpu": gpu,
        "cpu": cpu,
        "repair": repair,
        "repair_study": repair_study,
    }


def _add_gpu_245x_report_fixture(
    tmp_path: Path, paths: dict[str, Path]
) -> dict[str, Path]:
    original_plan = json.loads(paths["study"].read_text(encoding="utf-8"))
    prior_teardown = paths["study"].parent / "executions/gpu/teardown.json"
    root = tmp_path / "gpu-245x-repair"
    execution = root / "executions/gpu"
    results = execution / "remote-results"
    prepared = root / "prepared/gpu"
    launch, commands = prepared / "launch.yaml", prepared / "commands.json"
    launch.parent.mkdir(parents=True, exist_ok=True)
    launch.write_text("cluster-245x", encoding="utf-8")
    reset_root = prepared / "resets"
    reset_root.mkdir(parents=True, exist_ok=True)
    reset_paths = {
        name: reset_root / f"{name}.yaml"
        for name in ("transport-smoke", "natural-245x-r1", "natural-245x-r2")
    }
    for reset in reset_paths.values():
        reset.write_text("cluster-reset", encoding="utf-8")
    command_steps = [
        {
            "name": "mkdir-results",
            "command": "mkdir -p /mnt/nvme/results",
            "restart_yaml": None,
            "allow_failure": False,
            "timeout_seconds": 300,
        },
        {
            "name": "inventory",
            "command": "python -m cloud.admin inventory",
            "restart_yaml": None,
            "allow_failure": False,
            "timeout_seconds": 1800,
        },
        {
            "name": "stage-dataset",
            "command": "python -m cloud.admin stage",
            "restart_yaml": None,
            "allow_failure": False,
            "timeout_seconds": 7200,
        },
        {
            "name": "transport-smoke",
            "command": (
                "python -m release.benchmarks.ray_data_gpu_sort.cloud.worker "
                "--output /mnt/nvme/results/transport-smoke.json --kind smoke "
                "--backend smoke --cell full --repetition 1 --scale-numerator 1 "
                "--scale-denominator 1"
            ),
            "restart_yaml": str(reset_paths["transport-smoke"]),
            "allow_failure": False,
            "timeout_seconds": 7200,
        },
        {
            "name": "natural-245x-r1",
            "command": (
                "python -m release.benchmarks.ray_data_gpu_sort.cloud.worker "
                "--output /mnt/nvme/results/natural-245x-r1.json --kind natural "
                "--backend gpu --cell full --repetition 1 --scale-numerator 245 "
                "--scale-denominator 100 --wave-fraction 0.5"
            ),
            "restart_yaml": str(reset_paths["natural-245x-r1"]),
            "allow_failure": True,
            "timeout_seconds": 1800,
        },
        {
            "name": "natural-245x-r2",
            "command": (
                "python -m release.benchmarks.ray_data_gpu_sort.cloud.worker "
                "--output /mnt/nvme/results/natural-245x-r2.json --kind natural "
                "--backend gpu --cell full --repetition 2 --scale-numerator 245 "
                "--scale-denominator 100 --wave-fraction 0.5"
            ),
            "restart_yaml": str(reset_paths["natural-245x-r2"]),
            "allow_failure": True,
            "timeout_seconds": 1800,
        },
    ]
    _write(commands, {"schema_version": 1, "commands": command_steps})
    bundle = _bundle("245x-repair")
    study_body = {
        "schema_version": 1,
        "kind": "gpu_sort_bts_cloud_study",
        "campaign": "gpu-245x-repair",
        "trial_mode": GPU_245X_REPAIR_TRIAL_MODE,
        "dataset_digest": original_plan["dataset_digest"],
        "dataset_manifest_sha256": original_plan["dataset_manifest_sha256"],
        "bundle": bundle,
        "prior_gpu_teardown": _validate_prior_gpu_teardown(prior_teardown),
        "arms": {
            "gpu": {
                "cluster_name": "ray-gpu-sort-gpu-245x-repair-gpu",
                "launch_yaml": str(launch),
                "command_plan": str(commands),
                "topology": {
                    "nodes": 16,
                    "vcpus_per_node": 16,
                    "gpus_per_node": 1,
                    "instance_type": "g6.4xlarge",
                },
                "trials": [
                    trial.to_dict()
                    for trial in _trial_values("gpu", GPU_245X_REPAIR_TRIAL_MODE)
                ],
            }
        },
    }
    repair_study = root / "study.json"
    repair_plan = _write_plan(repair_study, study_body)
    ids = [f"i-{index + 300:016x}" for index in range(16)]
    lifecycle_body = {
        "schema_version": 1,
        "campaign": repair_plan["campaign"],
        "arm": "gpu",
        "cluster_name": repair_plan["arms"]["gpu"]["cluster_name"],
        "cluster_yaml_sha256": file_sha256(launch),
        "command_plan_sha256": file_sha256(commands),
        "steps": command_steps,
        "expected": {
            "nodes": 16,
            "cpus": 256.0,
            "gpus": 16.0,
            "instance_type": "g6.4xlarge",
            "single_availability_zone": True,
        },
        "same_ec2_ids": True,
        "fresh_ray_node_ids_each_trial": True,
        "ray_default_plasma": True,
    }
    lifecycle_plan = _write_plan(execution / "lifecycle-plan.json", lifecycle_body)
    teardown = {
        "schema_version": 2,
        "verified_empty": True,
        "terminated_instance_ids": ids,
        "campaign": repair_plan["campaign"],
        "arm": "gpu",
        "cluster_name": repair_plan["arms"]["gpu"]["cluster_name"],
        "lifecycle_plan_sha256": lifecycle_plan["plan_sha256"],
        "instance_ids_sha256": digest(ids),
    }
    _write(execution / "instances.json", {"instance_ids": ids})
    _write(execution / "teardown.json", teardown)
    receipts = []
    for ordinal, step in enumerate(command_steps, 1):
        receipt = {
            "name": step["name"],
            "returncode": 0,
            "rsync_returncode": 0,
            "allow_failure": step["allow_failure"],
            "artifact_recovered_from_stdout": False,
        }
        _write(execution / f"step-{ordinal:02d}.json", receipt)
        receipts.append(receipt)
    _write(
        execution / "result.json",
        {"status": "success", "steps": receipts, "teardown": teardown},
    )

    staged_manifest = _sha("245x-staged-manifest")
    staged_nodes = [
        {
            "instance_id": instance_id,
            "ray_node_id": f"gpu-245x-node-{index}",
            "dataset_digest": repair_plan["dataset_digest"],
            "manifest_sha256": staged_manifest,
        }
        for index, instance_id in enumerate(ids)
    ]
    _write(
        results / "admin/dataset-stage.json",
        {
            "dataset_digest": repair_plan["dataset_digest"],
            "nodes": staged_nodes,
            "digest": digest(staged_nodes),
        },
    )
    provenance = _bundle_artifact_provenance(repair_plan)
    elapsed = {
        "transport-smoke": 3.0,
        "natural-245x-r1": 130.0,
        "natural-245x-r2": 132.0,
    }
    for trial in _trial_values("gpu", GPU_245X_REPAIR_TRIAL_MODE):
        original = json.loads(
            (paths["gpu"] / f"{trial.name}.json").read_text(encoding="utf-8")
        )
        wave = 0.50
        value = _result(
            trial.backend,
            elapsed[trial.name],
            kind=trial.kind,
            repetition=trial.repetition,
            scale=(trial.scale_numerator, trial.scale_denominator),
            wave_fraction=wave,
            plan=original["plan"],
            cell="full" if trial.kind == "smoke" else original["cell"],
        )
        identity_body = {
            "trial": {
                "kind": trial.kind,
                "backend": trial.backend,
                "cell": trial.cell,
                "repetition": trial.repetition,
                "scale_numerator": trial.scale_numerator,
                "scale_denominator": trial.scale_denominator,
                "wave_fraction": wave,
            },
            "input_plan_digest": original["plan"]["digest"],
            "dataset_manifest_sha256": staged_manifest,
            **provenance,
        }
        value.update(
            {
                "artifact_identity": {
                    **identity_body,
                    "digest": digest(identity_body),
                },
                "ray_version": bundle["ray_version"],
                "ray_commit": bundle["ray_commit"],
            }
        )
        if trial.kind == "smoke":
            value.update(
                {
                    "cpu_sort_s": 1.0,
                    "gpu_sort_s": elapsed[trial.name],
                    "validation": {"exact_every_row_value": True, "rows": 100},
                }
            )
        _write(results / "results" / f"{trial.name}.json", value)
    paths.update(
        {
            "gpu_245x_repair": results,
            "gpu_245x_repair_study": repair_study,
        }
    )
    return paths


def _make_gpu_repair_245x_handoff(paths: dict[str, Path]) -> dict[str, Path]:
    for name in ("natural-245x-r1.json", "natural-245x-r2.json"):
        (paths["repair"] / "results" / name).unlink()
    execution = paths["repair"].parent
    lifecycle_path = execution / "lifecycle-plan.json"
    lifecycle = json.loads(lifecycle_path.read_text(encoding="utf-8"))
    lifecycle.pop("plan_sha256")
    names = [
        "mkdir-results",
        "inventory",
        "stage-dataset",
        "transport-smoke",
        "tune-2x-wave-0500",
        "tune-2x-wave-0375",
        "select-wave",
        "natural-2x-selected-r2",
        "natural-245x-r1",
        "natural-245x-r2",
    ]
    allow_failure = [False, False, False, False, True, True, False, True, True, True]
    lifecycle["steps"] = [
        {"name": name, "allow_failure": allowed}
        for name, allowed in zip(names, allow_failure)
    ]
    lifecycle = _write_plan(lifecycle_path, lifecycle)
    receipts = []
    for ordinal, (name, allowed) in enumerate(zip(names[:9], allow_failure[:9]), 1):
        receipt = {
            "name": name,
            "allow_failure": allowed,
            "returncode": 0,
            "rsync_returncode": 0,
            "artifact_recovered_from_stdout": False,
        }
        if ordinal == 9:
            receipt.update(
                {
                    "returncode": 1,
                    "rsync_returncode": 1,
                    "artifact_recovery_error": (
                        "ValueError: expected exactly one artifact sentinel, found 0"
                    ),
                }
            )
        _write(execution / f"step-{ordinal:02d}.json", receipt)
        receipts.append(receipt)
    teardown = json.loads((execution / "teardown.json").read_text(encoding="utf-8"))
    teardown["lifecycle_plan_sha256"] = lifecycle["plan_sha256"]
    _write(execution / "teardown.json", teardown)
    _write(
        execution / "failure.json",
        {
            "type": "RuntimeError",
            "message": (
                "artifact sync failed after natural-245x-r1; stdout recovery failed: "
                "expected exactly one artifact sentinel, found 0"
            ),
        },
    )
    _write(
        execution / "result.json",
        {"status": "failed", "steps": receipts, "teardown": teardown},
    )
    return paths


def _cpu_report_fixture(tmp_path: Path) -> dict[str, Path]:
    paths = _report_fixture(tmp_path)
    prior_teardown, original_plan = _legacy_cpu_execution(paths["study"].parent)
    gpu, cpu = paths["gpu"], paths["cpu"]
    cells = {
        name: {"name": name, "columns": ["a"], "keys": ["a"]}
        for name in (
            "narrow",
            "core",
            "full",
            "origin-string",
            "origin-integer",
            "route",
        )
    }
    original_provenance = _bundle_artifact_provenance(original_plan)
    original_staged_manifest = _sha("original-staged-manifest")

    def identified(
        value: dict[str, Any], *, arm: str, trial_name: str
    ) -> dict[str, Any]:
        trial = next(
            item
            for item in original_plan["arms"][arm]["trials"]
            if item["name"] == trial_name
        )
        identity_body = {
            "trial": {
                "kind": trial["kind"],
                "backend": trial["backend"],
                "cell": trial["cell"],
                "repetition": trial["repetition"],
                "scale_numerator": trial["scale_numerator"],
                "scale_denominator": trial["scale_denominator"],
                "wave_fraction": 0.5,
            },
            "input_plan_digest": value["plan"]["digest"],
            "dataset_manifest_sha256": original_staged_manifest,
            **original_provenance,
        }
        return {
            **value,
            "artifact_identity": {
                **identity_body,
                "digest": digest(identity_body),
            },
            "ray_version": original_plan["bundle"]["ray_version"],
            "ray_commit": original_plan["bundle"]["ray_commit"],
        }

    # The two valid observations are intentionally outside the repair subset.
    for cell, elapsed in (("narrow", 111.0), ("core", 112.0)):
        plan = {
            "kind": "trend",
            "rows": 100,
            "blocks": 2,
            "digest": _sha(f"{cell}-plan"),
        }
        for repetition in (1, 2):
            gpu_value = _result(
                "gpu",
                40.0 + repetition,
                kind="trend",
                repetition=repetition,
                plan=plan,
                cell=cells[cell],
            )
            _write(
                gpu / f"trend-{cell}-r{repetition}.json",
                identified(
                    gpu_value,
                    arm="gpu",
                    trial_name=f"trend-{cell}-r{repetition}",
                ),
            )
        cpu_value = _result(
            "pyarrow",
            elapsed,
            kind="trend",
            wave_fraction=0.5,
            plan=plan,
            cell=cells[cell],
        )
        _write(
            cpu / f"trend-{cell}-r1.json",
            identified(
                cpu_value,
                arm="cpu",
                trial_name=f"trend-{cell}-r1",
            ),
        )

    repaired_trends = ("full", "origin-string", "origin-integer", "route")
    for cell in repaired_trends:
        filename = f"trend-{cell}-r1.json"
        existing = (
            json.loads((gpu / filename).read_text())
            if (gpu / filename).is_file()
            else None
        )
        plan = (
            existing["plan"]
            if existing is not None
            else {
                "kind": "trend",
                "rows": 100,
                "blocks": 2,
                "digest": _sha(f"{cell}-plan"),
            }
        )
        for repetition in (1, 2):
            if not (gpu / f"trend-{cell}-r{repetition}.json").is_file():
                _write(
                    gpu / f"trend-{cell}-r{repetition}.json",
                    _result(
                        "gpu",
                        50.0 + repetition,
                        kind="trend",
                        repetition=repetition,
                        plan=plan,
                        cell=cells[cell],
                    ),
                )
        original = _result("pyarrow", None, kind="trend", plan=plan, cell=cells[cell])
        _write(cpu / filename, original)

    gpu_2x = json.loads((gpu / "tune-2x-wave-0500.json").read_text())
    gpu_245x = json.loads((gpu / "natural-245x-r1.json").read_text())
    for filename, scale, reference in (
        ("natural-2x-r1.json", (2, 1), gpu_2x),
        ("natural-245x-r1.json", (245, 100), gpu_245x),
    ):
        _write(
            cpu / filename,
            _result(
                "pyarrow",
                None,
                kind="natural",
                scale=scale,
                plan=reference["plan"],
                cell=cells["full"],
            ),
        )

    repair_root = tmp_path / "cpu-repair"
    execution = repair_root / "executions/cpu"
    repair = execution / "remote-results"
    prepared = repair_root / "prepared/cpu"
    launch, commands = prepared / "launch.yaml", prepared / "commands.json"
    launch.parent.mkdir(parents=True, exist_ok=True)
    launch.write_text("cluster", encoding="utf-8")
    commands.write_text("commands", encoding="utf-8")
    repair_bundle = _bundle("cpu-repair")
    repair_body = {
        "schema_version": 1,
        "kind": "gpu_sort_bts_cloud_study",
        "campaign": "cpu-repair",
        "trial_mode": "cpu-repair",
        "dataset_digest": original_plan["dataset_digest"],
        "dataset_manifest_sha256": original_plan["dataset_manifest_sha256"],
        "bundle": repair_bundle,
        "prior_cpu_teardown": _validate_prior_cpu_teardown(prior_teardown),
        "arms": {
            "cpu": {
                "cluster_name": "ray-gpu-sort-cpu-repair-cpu",
                "launch_yaml": str(launch),
                "command_plan": str(commands),
                "topology": {
                    "nodes": 16,
                    "vcpus_per_node": 16,
                    "gpus_per_node": 0,
                    "instance_type": "m5dn.4xlarge",
                },
                "trials": [
                    trial.to_dict()
                    for trial in _trial_values("cpu", CPU_REPAIR_TRIAL_MODE)
                ],
            }
        },
    }
    repair_study = repair_root / "study.json"
    repair_plan = _write_plan(repair_study, repair_body)
    ids = [f"i-{index + 200:016x}" for index in range(16)]
    lifecycle_body = {
        "schema_version": 1,
        "campaign": "cpu-repair",
        "arm": "cpu",
        "cluster_name": "ray-gpu-sort-cpu-repair-cpu",
        "cluster_yaml_sha256": file_sha256(launch),
        "command_plan_sha256": file_sha256(commands),
        "steps": [],
        "expected": {
            "nodes": 16,
            "cpus": 256.0,
            "gpus": 0.0,
            "instance_type": "m5dn.4xlarge",
            "single_availability_zone": True,
        },
        "same_ec2_ids": True,
        "fresh_ray_node_ids_each_trial": True,
        "ray_default_plasma": True,
    }
    lifecycle_plan = _write_plan(execution / "lifecycle-plan.json", lifecycle_body)
    teardown = {
        "schema_version": 2,
        "verified_empty": True,
        "terminated_instance_ids": ids,
        "campaign": "cpu-repair",
        "arm": "cpu",
        "cluster_name": "ray-gpu-sort-cpu-repair-cpu",
        "lifecycle_plan_sha256": lifecycle_plan["plan_sha256"],
        "instance_ids_sha256": digest(ids),
    }
    _write(execution / "instances.json", {"instance_ids": ids})
    _write(execution / "teardown.json", teardown)
    _write(execution / "result.json", {"status": "success", "teardown": teardown})

    staged_manifest = _sha("cpu-staged-manifest")
    staged_nodes = [
        {
            "instance_id": ids[index],
            "ray_node_id": f"cpu-node-{index}",
            "dataset_digest": repair_plan["dataset_digest"],
            "manifest_sha256": staged_manifest,
        }
        for index in range(16)
    ]
    _write(
        repair / "admin/dataset-stage.json",
        {
            "dataset_digest": repair_plan["dataset_digest"],
            "nodes": staged_nodes,
            "digest": digest(staged_nodes),
        },
    )
    provenance = _bundle_artifact_provenance(repair_plan)
    repair_times = {
        "trend-full-r1": 201.0,
        "trend-origin-string-r1": 202.0,
        "trend-origin-integer-r1": 203.0,
        "trend-route-r1": 204.0,
        "natural-2x-r1": 301.0,
        "natural-245x-r1": 401.0,
    }
    for trial in _trial_values("cpu", CPU_REPAIR_TRIAL_MODE):
        filename = f"{trial.name}.json"
        reference_name = (
            "tune-2x-wave-0500.json" if filename == "natural-2x-r1.json" else filename
        )
        reference = json.loads((gpu / reference_name).read_text())
        cell = reference["cell"]
        plan = reference["plan"]
        value = _result(
            "pyarrow",
            repair_times[trial.name],
            kind=trial.kind,
            repetition=trial.repetition,
            scale=(trial.scale_numerator, trial.scale_denominator),
            wave_fraction=0.5,
            plan=plan,
            cell=cell,
        )
        identity_body = {
            "trial": {
                "kind": trial.kind,
                "backend": trial.backend,
                "cell": trial.cell,
                "repetition": trial.repetition,
                "scale_numerator": trial.scale_numerator,
                "scale_denominator": trial.scale_denominator,
                "wave_fraction": 0.5,
            },
            "input_plan_digest": plan["digest"],
            "dataset_manifest_sha256": staged_manifest,
            **provenance,
        }
        value.update(
            {
                "artifact_identity": {
                    **identity_body,
                    "digest": digest(identity_body),
                },
                "ray_version": repair_bundle["ray_version"],
                "ray_commit": repair_bundle["ray_commit"],
            }
        )
        _write(repair / "results" / filename, value)
    return {
        **paths,
        "cpu_repair": repair,
        "cpu_repair_study": repair_study,
    }


def _resource_failure_log(seconds: int) -> str:
    minutes, remainder = divmod(seconds, 60)
    return "\n".join(
        [
            "Execution plan: InputDataBuffer[Input] -> AllToAllOperator[Sort]",
            "Shuffle Reduce:   0% 0.00/1.00 [00:00<?, ? row/s]",
            f"Shuffle Map: 100% 198M/198M [{minutes:02d}:{remainder:02d}<00:00, 1 row/s]",
            "More than 16GB of driver memory used to store Ray Data block data and metadata.",
            "Shared connection to 192.0.2.1 closed.",
        ]
    )


def _configure_resource_failure_execution(
    *,
    study_path: Path,
    remote_results: Path,
    seconds: int,
    instance_offset: int,
    with_teardown: bool,
    stdout_recovery_evidence: bool,
) -> dict[str, Any]:
    plan = json.loads(study_path.read_text(encoding="utf-8"))
    execution = remote_results.parent
    arm = plan["arms"]["cpu"]
    names = [
        "mkdir-results",
        "inventory",
        "stage-dataset",
        *CPU_REPAIR_TRIAL_NAMES,
    ]
    lifecycle_body = {
        "schema_version": 1,
        "campaign": plan["campaign"],
        "arm": "cpu",
        "cluster_name": arm["cluster_name"],
        "cluster_yaml_sha256": file_sha256(Path(arm["launch_yaml"])),
        "command_plan_sha256": file_sha256(Path(arm["command_plan"])),
        "steps": [
            {
                "name": name,
                "command": f"run {name}",
                "restart_yaml": None,
                "timeout_seconds": 300,
                "allow_failure": index >= 3,
            }
            for index, name in enumerate(names)
        ],
        "expected": {
            "nodes": 16,
            "cpus": 256.0,
            "gpus": 0.0,
            "instance_type": "m5dn.4xlarge",
            "single_availability_zone": True,
        },
        "same_ec2_ids": True,
        "fresh_ray_node_ids_each_trial": True,
        "ray_default_plasma": True,
    }
    lifecycle_plan = _write_plan(execution / "lifecycle-plan.json", lifecycle_body)
    for ordinal, name in enumerate(names, 1):
        final = ordinal == len(names)
        receipt: dict[str, Any] = {
            "name": name,
            "allow_failure": ordinal >= 4,
            "returncode": 1 if ordinal >= 4 else 0,
            "rsync_returncode": 1 if final else 0,
        }
        if stdout_recovery_evidence:
            receipt["artifact_recovered_from_stdout"] = False
        if final and stdout_recovery_evidence:
            receipt["artifact_recovery_error"] = (
                "ValueError: expected exactly one artifact sentinel, found 0"
            )
        _write(execution / f"step-{ordinal:02d}.json", receipt)

    suffix = (
        "; stdout recovery failed: expected exactly one artifact sentinel, found 0"
        if stdout_recovery_evidence
        else ""
    )
    failure_message = f"artifact sync failed after natural-245x-r1{suffix}"
    _write(
        execution / "failure.json",
        {"type": "RuntimeError", "message": failure_message},
    )
    prior_ray_ids = [f"prior-{instance_offset}-{index}" for index in range(16)]
    failed_ray_ids = [f"failed-{instance_offset}-{index}" for index in range(16)]
    for ordinal, name, ray_ids in (
        (8, "natural-2x-r1", prior_ray_ids),
        (9, "natural-245x-r1", failed_ray_ids),
    ):
        _write(
            execution / f"shape-{ordinal:02d}-{name}.json",
            {
                "attempts": 1,
                "observed": {
                    "nodes": 16,
                    "cpus": 256,
                    "gpus": 0,
                    "ray_node_ids": ray_ids,
                },
            },
        )
    (execution / "command-09-natural-245x-r1.log").write_text(
        _resource_failure_log(seconds), encoding="utf-8"
    )
    (execution / "rsync-09-natural-245x-r1.log").write_text(
        "Connection timed out during banner exchange\nError: SSH command failed.\n",
        encoding="utf-8",
    )
    instances_path = execution / "instances.json"
    if instances_path.is_file():
        instance_ids = json.loads(instances_path.read_text(encoding="utf-8"))[
            "instance_ids"
        ]
    else:
        instance_ids = [f"i-{index + instance_offset:016x}" for index in range(16)]
    _write(execution / "instances.json", {"instance_ids": instance_ids})
    result = execution / "result.json"
    if result.exists():
        result.unlink()
    if with_teardown:
        _write(
            execution / "teardown.json",
            {
                "schema_version": 2,
                "verified_empty": True,
                "terminated_instance_ids": instance_ids,
                "campaign": plan["campaign"],
                "arm": "cpu",
                "cluster_name": arm["cluster_name"],
                "lifecycle_plan_sha256": lifecycle_plan["plan_sha256"],
                "instance_ids_sha256": digest(sorted(instance_ids)),
            },
        )
    return {
        "lifecycle_plan": lifecycle_plan,
        "failure_message": failure_message,
    }


def _resource_profile_files(study_root: Path, *, with_teardown: bool) -> dict[str, str]:
    names = [
        "study.json",
        "executions/cpu/lifecycle-plan.json",
        "executions/cpu/instances.json",
        "executions/cpu/failure.json",
        *(f"executions/cpu/step-{ordinal:02d}.json" for ordinal in range(1, 10)),
        "executions/cpu/shape-08-natural-2x-r1.json",
        "executions/cpu/shape-09-natural-245x-r1.json",
        "executions/cpu/command-09-natural-245x-r1.log",
        "executions/cpu/rsync-09-natural-245x-r1.log",
    ]
    if with_teardown:
        names.append("executions/cpu/teardown.json")
    return {name: file_sha256(study_root / name) for name in names}


def _cpu_resource_rejection_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    paths: dict[str, Any] = _cpu_report_fixture(tmp_path)
    final_worker = paths["cpu_repair"] / "results/natural-245x-r1.json"
    final_worker.unlink()
    primary_study = paths["cpu_repair_study"]
    primary_root = primary_study.parent
    primary_execution = _configure_resource_failure_execution(
        study_path=primary_study,
        remote_results=paths["cpu_repair"],
        seconds=380,
        instance_offset=400,
        with_teardown=True,
        stdout_recovery_evidence=True,
    )

    corroborating_campaign = "cpu-repair-corroborating"
    corroborating_root = tmp_path / corroborating_campaign
    corroborating_remote = corroborating_root / "executions/cpu/remote-results"
    shutil.copytree(paths["cpu_repair"], corroborating_remote)
    primary_plan = json.loads(primary_study.read_text(encoding="utf-8"))
    corroborating_launch = corroborating_root / "prepared/cpu/launch.yaml"
    corroborating_commands = corroborating_root / "prepared/cpu/commands.json"
    corroborating_launch.parent.mkdir(parents=True, exist_ok=True)
    corroborating_launch.write_text("cluster", encoding="utf-8")
    corroborating_commands.write_text("commands", encoding="utf-8")
    corroborating_body = {
        key: value for key, value in primary_plan.items() if key != "plan_sha256"
    }
    corroborating_body["campaign"] = corroborating_campaign
    corroborating_body["arms"]["cpu"]["cluster_name"] = (
        "ray-gpu-sort-cpu-repair-corroborating-cpu"
    )
    corroborating_body["arms"]["cpu"]["launch_yaml"] = str(corroborating_launch)
    corroborating_body["arms"]["cpu"]["command_plan"] = str(corroborating_commands)
    corroborating_study = corroborating_root / "study.json"
    corroborating_plan = _write_plan(corroborating_study, corroborating_body)
    corroborating_execution = _configure_resource_failure_execution(
        study_path=corroborating_study,
        remote_results=corroborating_remote,
        seconds=374,
        instance_offset=500,
        with_teardown=False,
        stdout_recovery_evidence=False,
    )

    profile = {
        "campaign": primary_plan["campaign"],
        "study_plan_sha256": primary_plan["plan_sha256"],
        "lifecycle_plan_sha256": primary_execution["lifecycle_plan"]["plan_sha256"],
        "trial": "natural-245x-r1",
        "step_ordinal": 9,
        "diagnostic_time_to_failure_s": 380,
        "failure_message": primary_execution["failure_message"],
        "stdout_recovery_evidence": True,
        "files": _resource_profile_files(primary_root, with_teardown=True),
        "corroborating": {
            "campaign": corroborating_campaign,
            "study_plan_sha256": corroborating_plan["plan_sha256"],
            "lifecycle_plan_sha256": corroborating_execution["lifecycle_plan"][
                "plan_sha256"
            ],
            "diagnostic_time_to_failure_s": 374,
            "failure_message": corroborating_execution["failure_message"],
            "stdout_recovery_evidence": False,
            "files": _resource_profile_files(corroborating_root, with_teardown=False),
        },
    }
    monkeypatch.setattr(report_module, "CPU_RESOURCE_REJECTION_PROFILE", profile)
    paths.update(
        {
            "resource_profile": profile,
            "primary_root": primary_root,
            "corroborating_root": corroborating_root,
        }
    )
    return paths


def test_report_uses_repair_only_for_tuning_and_natural_results(tmp_path: Path) -> None:
    paths = _report_fixture(tmp_path)
    output = tmp_path / "report.md"
    result = build(
        paths["study"],
        paths["gpu"],
        paths["cpu"],
        output,
        paths["repair"],
        paths["repair_study"],
    )
    full = next(row for row in result["trend"]["rows"] if row["cell"] == "full")
    natural = {row["scale"]: row for row in result["natural_spill"]["rows"]}
    assert full["gpu_s"] == [50.0, 52.0]
    assert result["tuning"]["screens"]["0.50"]["cold_sort_s"] == 80.0
    assert natural["2x"]["gpu_s"] == [80.0, 82.0]
    assert natural["2.45x"]["gpu_s"] == [120.0, 122.0]
    history = result["gpu_repair_overlay"]["superseded_original_observations"]
    assert history["tune-2x-wave-0500.json"]["cold_sort_s"] == 100.0
    assert "Superseded original GPU attempts" in output.read_text(encoding="utf-8")


def test_report_gpu_245x_overlay_replaces_only_245x_results(tmp_path: Path) -> None:
    paths = _make_gpu_repair_245x_handoff(
        _add_gpu_245x_report_fixture(tmp_path, _report_fixture(tmp_path))
    )
    output = tmp_path / "report-245x.md"
    result = build(
        paths["study"],
        paths["gpu"],
        paths["cpu"],
        output,
        gpu_repair_root=paths["repair"],
        gpu_repair_study=paths["repair_study"],
        gpu_245x_repair_root=paths["gpu_245x_repair"],
        gpu_245x_repair_study=paths["gpu_245x_repair_study"],
    )
    natural = {row["scale"]: row for row in result["natural_spill"]["rows"]}
    assert result["smoke"]["cold_sort_s"] == 2.0
    assert result["tuning"]["screens"]["0.50"]["cold_sort_s"] == 80.0
    assert natural["2x"]["gpu_s"] == [80.0, 82.0]
    assert natural["2.45x"]["gpu_s"] == [130.0, 132.0]
    overlay = result["gpu_245x_repair_overlay"]
    assert overlay["gate_smoke"]["cold_sort_s"] == 3.0
    assert overlay["replacement_names"] == [
        "natural-245x-r1.json",
        "natural-245x-r2.json",
    ]
    assert not overlay["superseded_full_repair_observations"][
        "natural-245x-r1.json"
    ]
    assert (
        result["gpu_repair_overlay"]["validated_provenance"]["execution"][
            "completed_prefix_handoff"
        ]
        is True
    )
    markdown = output.read_text(encoding="utf-8")
    assert "GPU 2.45× repair overlay" in markdown
    assert "full repair's 2× tuning and observations remain authoritative" in markdown


def test_report_gpu_245x_overlay_requires_full_repair(tmp_path: Path) -> None:
    paths = _add_gpu_245x_report_fixture(tmp_path, _report_fixture(tmp_path))
    with pytest.raises(ValueError, match="requires the full GPU repair overlay"):
        build(
            paths["study"],
            paths["gpu"],
            paths["cpu"],
            tmp_path / "missing-full-repair.md",
            gpu_245x_repair_root=paths["gpu_245x_repair"],
            gpu_245x_repair_study=paths["gpu_245x_repair_study"],
        )


def test_report_rejects_full_repair_prefix_without_245x_handoff(
    tmp_path: Path,
) -> None:
    paths = _make_gpu_repair_245x_handoff(_report_fixture(tmp_path))
    with pytest.raises(ValueError, match="exact full repair trial set"):
        build(
            paths["study"],
            paths["gpu"],
            paths["cpu"],
            tmp_path / "prefix-without-handoff.md",
            gpu_repair_root=paths["repair"],
            gpu_repair_study=paths["repair_study"],
        )


def test_report_rejects_nonfixed_gpu_245x_repair_trial(tmp_path: Path) -> None:
    paths = _add_gpu_245x_report_fixture(tmp_path, _report_fixture(tmp_path))
    study_path = paths["gpu_245x_repair_study"]
    value = json.loads(study_path.read_text(encoding="utf-8"))
    value.pop("plan_sha256")
    trial = value["arms"]["gpu"]["trials"][1]
    trial["wave_fraction"] = 0.375
    body = {key: item for key, item in trial.items() if key != "identity"}
    trial["identity"] = digest(body)
    _write_plan(study_path, value)
    with pytest.raises(ValueError, match="trial is not exact"):
        build(
            paths["study"],
            paths["gpu"],
            paths["cpu"],
            tmp_path / "nonfixed-wave.md",
            gpu_repair_root=paths["repair"],
            gpu_repair_study=paths["repair_study"],
            gpu_245x_repair_root=paths["gpu_245x_repair"],
            gpu_245x_repair_study=study_path,
        )


def test_report_rejects_unsuccessful_gpu_245x_repair_lifecycle(
    tmp_path: Path,
) -> None:
    paths = _add_gpu_245x_report_fixture(tmp_path, _report_fixture(tmp_path))
    result_path = paths["gpu_245x_repair"].parent / "result.json"
    value = json.loads(result_path.read_text(encoding="utf-8"))
    value["status"] = "failed"
    _write(result_path, value)
    with pytest.raises(ValueError, match="lifecycle did not complete"):
        build(
            paths["study"],
            paths["gpu"],
            paths["cpu"],
            tmp_path / "failed-lifecycle.md",
            gpu_repair_root=paths["repair"],
            gpu_repair_study=paths["repair_study"],
            gpu_245x_repair_root=paths["gpu_245x_repair"],
            gpu_245x_repair_study=paths["gpu_245x_repair_study"],
        )


def test_report_rejects_gpu_245x_repair_missing_step_receipt(tmp_path: Path) -> None:
    paths = _add_gpu_245x_report_fixture(tmp_path, _report_fixture(tmp_path))
    (paths["gpu_245x_repair"].parent / "step-06.json").unlink()
    with pytest.raises(ValueError, match="missing step receipt 6"):
        build(
            paths["study"],
            paths["gpu"],
            paths["cpu"],
            tmp_path / "missing-step.md",
            gpu_repair_root=paths["repair"],
            gpu_repair_study=paths["repair_study"],
            gpu_245x_repair_root=paths["gpu_245x_repair"],
            gpu_245x_repair_study=paths["gpu_245x_repair_study"],
        )


def test_report_rejects_gpu_245x_without_fresh_runtime_invariant(
    tmp_path: Path,
) -> None:
    paths = _add_gpu_245x_report_fixture(tmp_path, _report_fixture(tmp_path))
    execution = paths["gpu_245x_repair"].parent
    lifecycle_path = execution / "lifecycle-plan.json"
    lifecycle_value = json.loads(lifecycle_path.read_text(encoding="utf-8"))
    lifecycle_value.pop("plan_sha256")
    lifecycle_value["fresh_ray_node_ids_each_trial"] = False
    lifecycle_value = _write_plan(lifecycle_path, lifecycle_value)
    teardown_path = execution / "teardown.json"
    teardown = json.loads(teardown_path.read_text(encoding="utf-8"))
    teardown["lifecycle_plan_sha256"] = lifecycle_value["plan_sha256"]
    _write(teardown_path, teardown)
    result_path = execution / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["teardown"] = teardown
    _write(result_path, result)
    with pytest.raises(ValueError, match="lifecycle identity differs"):
        build(
            paths["study"],
            paths["gpu"],
            paths["cpu"],
            tmp_path / "not-fresh.md",
            gpu_repair_root=paths["repair"],
            gpu_repair_study=paths["repair_study"],
            gpu_245x_repair_root=paths["gpu_245x_repair"],
            gpu_245x_repair_study=paths["gpu_245x_repair_study"],
        )


def test_report_rejects_full_gpu_repair_trial_drift(tmp_path: Path) -> None:
    paths = _report_fixture(tmp_path)
    study_path = paths["repair_study"]
    value = json.loads(study_path.read_text(encoding="utf-8"))
    value.pop("plan_sha256")
    trial = value["arms"]["gpu"]["trials"][1]
    trial["scale_numerator"] = 3
    body = {key: item for key, item in trial.items() if key != "identity"}
    trial["identity"] = digest(body)
    _write_plan(study_path, value)
    with pytest.raises(ValueError, match="trial differs from the original study"):
        build(
            paths["study"],
            paths["gpu"],
            paths["cpu"],
            tmp_path / "full-trial-drift.md",
            gpu_repair_root=paths["repair"],
            gpu_repair_study=study_path,
        )


def test_report_rejects_gpu_245x_input_plan_drift(tmp_path: Path) -> None:
    paths = _add_gpu_245x_report_fixture(tmp_path, _report_fixture(tmp_path))
    artifact = paths["gpu_245x_repair"] / "results/natural-245x-r1.json"
    value = json.loads(artifact.read_text(encoding="utf-8"))
    value["plan"]["rows"] = 999
    _write(artifact, value)
    with pytest.raises(ValueError, match="input plan differs from original"):
        build(
            paths["study"],
            paths["gpu"],
            paths["cpu"],
            tmp_path / "plan-drift.md",
            gpu_repair_root=paths["repair"],
            gpu_repair_study=paths["repair_study"],
            gpu_245x_repair_root=paths["gpu_245x_repair"],
            gpu_245x_repair_study=paths["gpu_245x_repair_study"],
        )


def test_report_rejects_unpaired_or_tampered_repair_provenance(tmp_path: Path) -> None:
    paths = _report_fixture(tmp_path)
    with pytest.raises(ValueError, match="both repair study and results"):
        build(
            paths["study"],
            paths["gpu"],
            paths["cpu"],
            tmp_path / "unpaired.md",
            paths["repair"],
        )
    artifact = paths["repair"] / "results/natural-245x-r1.json"
    value = json.loads(artifact.read_text(encoding="utf-8"))
    value["artifact_identity"]["wheel_sha256"] = _sha("wrong-wheel")
    _write(artifact, value)
    with pytest.raises(ValueError, match="artifact identity digest"):
        build(
            paths["study"],
            paths["gpu"],
            paths["cpu"],
            tmp_path / "tampered.md",
            paths["repair"],
            paths["repair_study"],
        )


def test_result_loader_rejects_identityless_or_same_identity_differences(
    tmp_path: Path,
) -> None:
    first = _result("gpu", 10.0, kind="natural")
    second = {**first, "cold_sort_s": 11.0}
    _write(tmp_path / "first/result.json", first)
    _write(tmp_path / "second/result.json", second)
    with pytest.raises(ValueError, match="conflicting duplicate contents"):
        _results(tmp_path)
    same_identity = tmp_path / "same-identity"
    first["artifact_identity"] = {"digest": _sha("shared-identity")}
    second["artifact_identity"] = first["artifact_identity"]
    _write(same_identity / "first/result.json", first)
    _write(same_identity / "second/result.json", second)
    with pytest.raises(ValueError, match="conflicting duplicate contents"):
        _results(same_identity)


def test_report_rejects_repair_study_from_another_dataset(tmp_path: Path) -> None:
    paths = _report_fixture(tmp_path)
    value = json.loads(paths["repair_study"].read_text(encoding="utf-8"))
    value.pop("plan_sha256")
    value["dataset_digest"] = _sha("other-dataset")
    _write_plan(paths["repair_study"], value)
    with pytest.raises(ValueError, match="differ in dataset_digest"):
        build(
            paths["study"],
            paths["gpu"],
            paths["cpu"],
            tmp_path / "wrong-dataset.md",
            paths["repair"],
            paths["repair_study"],
        )


def test_report_cpu_repair_replaces_exact_subset_and_preserves_narrow_core(
    tmp_path: Path,
) -> None:
    paths = _cpu_report_fixture(tmp_path)
    output = tmp_path / "cpu-report.md"
    result = build(
        paths["study"],
        paths["gpu"],
        paths["cpu"],
        output,
        cpu_repair_root=paths["cpu_repair"],
        cpu_repair_study=paths["cpu_repair_study"],
    )
    trends = {row["cell"]: row for row in result["trend"]["rows"]}
    natural = {row["scale"]: row for row in result["natural_spill"]["rows"]}
    assert trends["narrow"]["cpu_s"] == 111.0
    assert trends["core"]["cpu_s"] == 112.0
    assert trends["full"]["cpu_s"] == 201.0
    assert trends["origin-string"]["cpu_s"] == 202.0
    assert trends["origin-integer"]["cpu_s"] == 203.0
    assert trends["route"]["cpu_s"] == 204.0
    assert natural["2x"]["cpu_s"] == 301.0
    assert natural["2.45x"]["cpu_s"] == 401.0
    overlay = result["cpu_repair_overlay"]
    assert overlay["preserved_original_names"] == [
        "trend-narrow-r1.json",
        "trend-core-r1.json",
    ]
    assert set(overlay["replacement_names"]) == {
        f"{name}.json" for name in CPU_REPAIR_TRIAL_NAMES
    }
    assert (
        overlay["superseded_original_observations"]["trend-full-r1.json"]["cold_sort_s"]
        is None
    )
    markdown = output.read_text(encoding="utf-8")
    assert "Superseded original CPU attempts" in markdown
    assert "original narrow/core observations are preserved" in markdown


def test_report_represents_exact_final_cpu_resource_rejection_without_worker_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _cpu_resource_rejection_fixture(tmp_path, monkeypatch)
    output = tmp_path / "resource-rejection.md"
    result = build(
        paths["study"],
        paths["gpu"],
        paths["cpu"],
        output,
        paths["repair"],
        paths["repair_study"],
        paths["cpu_repair"],
        paths["cpu_repair_study"],
    )
    large = next(
        row for row in result["natural_spill"]["rows"] if row["scale"] == "2.45x"
    )
    assert large["cpu_s"] is None
    assert large["observed_speedup"] is None
    assert large["cpu_classification"] == "resource-rejection"
    assert large["comparison_status"] == "resource-rejection"
    assert large["cpu_observations"] == []
    rejection = large["cpu_lifecycle_resource_rejection"]
    assert large["cpu_observation_summary"]["classification"] == (
        "resource-rejection"
    )
    assert "lifecycle_resource_rejection" in large["cpu_observation_summary"]
    assert rejection["reproduced"] is True
    assert rejection["reproduction_count"] == 2
    assert [
        attempt["diagnostic_time_to_failure_s"] for attempt in rejection["attempts"]
    ] == [380, 374]
    overlay = result["cpu_repair_overlay"]
    assert overlay["replacement_names"] == [
        f"{name}.json" for name in CPU_REPAIR_TRIAL_NAMES[:-1]
    ]
    assert overlay["resource_rejection_names"] == ["natural-245x-r1.json"]
    assert overlay["unexplained_missing_names"] == []
    assert not (paths["cpu_repair"] / "results/natural-245x-r1.json").exists()

    markdown = output.read_text(encoding="utf-8")
    assert "n/a (resource rejection)" in markdown
    assert "head/driver resource failure at the map-to-reduce boundary" in markdown
    assert "reproduced twice" in markdown
    assert "diagnostic time-to-failure only" in markdown
    persisted = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    persisted_large = next(
        row for row in persisted["natural_spill"]["rows"] if row["scale"] == "2.45x"
    )
    assert persisted_large["cpu_observations"] == []


def test_report_rejects_malformed_cpu_resource_failure_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _cpu_resource_rejection_fixture(tmp_path, monkeypatch)
    relative = "executions/cpu/command-09-natural-245x-r1.log"
    log = paths["primary_root"] / relative
    log.write_text(
        _resource_failure_log(380) + "\nShuffle Reduce: 10% 1/10\n",
        encoding="utf-8",
    )
    paths["resource_profile"]["files"][relative] = file_sha256(log)
    with pytest.raises(ValueError, match="exact map-to-reduce head/driver failure"):
        build(
            paths["study"],
            paths["gpu"],
            paths["cpu"],
            tmp_path / "malformed-log.md",
            cpu_repair_root=paths["cpu_repair"],
            cpu_repair_study=paths["cpu_repair_study"],
        )


def test_report_rejects_earlier_cpu_failure_as_final_resource_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _cpu_resource_rejection_fixture(tmp_path, monkeypatch)
    relative = "executions/cpu/step-09.json"
    receipt = paths["primary_root"] / relative
    value = json.loads(receipt.read_text(encoding="utf-8"))
    value["name"] = "natural-2x-r1"
    _write(receipt, value)
    paths["resource_profile"]["files"][relative] = file_sha256(receipt)
    with pytest.raises(ValueError, match="step receipt differs at 9"):
        build(
            paths["study"],
            paths["gpu"],
            paths["cpu"],
            tmp_path / "earlier-step.md",
            cpu_repair_root=paths["cpu_repair"],
            cpu_repair_study=paths["cpu_repair_study"],
        )


def test_report_rejects_missing_cpu_resource_rejection_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _cpu_resource_rejection_fixture(tmp_path, monkeypatch)
    missing = (
        paths["corroborating_root"] / "executions/cpu/rsync-09-natural-245x-r1.log"
    )
    missing.unlink()
    with pytest.raises(ValueError, match="resource-rejection evidence is missing"):
        build(
            paths["study"],
            paths["gpu"],
            paths["cpu"],
            tmp_path / "missing-evidence.md",
            cpu_repair_root=paths["cpu_repair"],
            cpu_repair_study=paths["cpu_repair_study"],
        )


def test_report_rejects_unpaired_or_tampered_cpu_repair(tmp_path: Path) -> None:
    paths = _cpu_report_fixture(tmp_path)
    with pytest.raises(ValueError, match="CPU repair overlay requires both"):
        build(
            paths["study"],
            paths["gpu"],
            paths["cpu"],
            tmp_path / "unpaired-cpu.md",
            cpu_repair_root=paths["cpu_repair"],
        )
    artifact = paths["cpu_repair"] / "results/trend-route-r1.json"
    value = json.loads(artifact.read_text(encoding="utf-8"))
    value["artifact_identity"]["input_plan_digest"] = _sha("wrong-plan")
    _write(artifact, value)
    with pytest.raises(ValueError, match="CPU repair artifact identity digest"):
        build(
            paths["study"],
            paths["gpu"],
            paths["cpu"],
            tmp_path / "tampered-cpu.md",
            cpu_repair_root=paths["cpu_repair"],
            cpu_repair_study=paths["cpu_repair_study"],
        )


def test_report_rejects_changed_cpu_prior_provenance_or_cell(tmp_path: Path) -> None:
    paths = _cpu_report_fixture(tmp_path)
    plan = json.loads(paths["cpu_repair_study"].read_text(encoding="utf-8"))
    instances = Path(plan["prior_cpu_teardown"]["files"]["instances"]["path"])
    original_instances = instances.read_text(encoding="utf-8")
    _write(instances, {"instance_ids": ["i-changed"] * 16})
    with pytest.raises(ValueError, match="recorded original provenance changed"):
        build(
            paths["study"],
            paths["gpu"],
            paths["cpu"],
            tmp_path / "changed-prior.md",
            cpu_repair_root=paths["cpu_repair"],
            cpu_repair_study=paths["cpu_repair_study"],
        )
    instances.write_text(original_instances, encoding="utf-8")

    artifact = paths["cpu_repair"] / "results/trend-route-r1.json"
    value = json.loads(artifact.read_text(encoding="utf-8"))
    value["cell"]["keys"] = ["wrong-key"]
    _write(artifact, value)
    with pytest.raises(ValueError, match="cell/columns/keys differ from original"):
        build(
            paths["study"],
            paths["gpu"],
            paths["cpu"],
            tmp_path / "changed-cell.md",
            cpu_repair_root=paths["cpu_repair"],
            cpu_repair_study=paths["cpu_repair_study"],
        )


def test_report_binds_preserved_cpu_artifacts_and_repair_topology(
    tmp_path: Path,
) -> None:
    paths = _cpu_report_fixture(tmp_path)
    narrow = paths["cpu"] / "trend-narrow-r1.json"
    value = json.loads(narrow.read_text(encoding="utf-8"))
    value["artifact_identity"]["wheel_sha256"] = _sha("foreign-wheel")
    identity_body = {
        key: item for key, item in value["artifact_identity"].items() if key != "digest"
    }
    value["artifact_identity"]["digest"] = digest(identity_body)
    _write(narrow, value)
    with pytest.raises(ValueError, match="preserved CPU provenance differs"):
        build(
            paths["study"],
            paths["gpu"],
            paths["cpu"],
            tmp_path / "foreign-narrow.md",
            cpu_repair_root=paths["cpu_repair"],
            cpu_repair_study=paths["cpu_repair_study"],
        )

    paths = _cpu_report_fixture(tmp_path / "topology")
    value = json.loads(paths["cpu_repair_study"].read_text(encoding="utf-8"))
    value.pop("plan_sha256")
    value["arms"]["cpu"]["topology"]["instance_type"] = "c7i.4xlarge"
    _write_plan(paths["cpu_repair_study"], value)
    with pytest.raises(ValueError, match="differs from the frozen original topology"):
        build(
            paths["study"],
            paths["gpu"],
            paths["cpu"],
            tmp_path / "foreign-topology.md",
            cpu_repair_root=paths["cpu_repair"],
            cpu_repair_study=paths["cpu_repair_study"],
        )
