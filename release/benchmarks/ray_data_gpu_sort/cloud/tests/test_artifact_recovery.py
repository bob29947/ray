from __future__ import annotations

import base64
import hashlib
import json
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from release.benchmarks.ray_data_gpu_sort.cloud import lifecycle, worker
from release.benchmarks.ray_data_gpu_sort.cloud.common import (
    ARTIFACT_SENTINEL,
    artifact_sentinel,
    canonical_bytes,
    digest,
    file_sha256,
)
from release.benchmarks.ray_data_gpu_sort.cloud.lifecycle import (
    WORKER_MODULE,
    Step,
    _bundle_provenance,
    _recover_artifact_from_stdout,
)
from release.benchmarks.ray_data_gpu_sort.spec import (
    RAY_COMMIT,
    RAY_VERSION,
    RAY_WHEEL_SHA256,
)

REMOTE_RESULTS = "/mnt/nvme/gpu-sort-results"
PLAN_SHA256 = "a" * 64
MANIFEST_SHA256 = "b" * 64
DATASET_SHA256 = "c" * 64
BUNDLE_SHA256 = "f" * 64
HARNESS_FILES = {
    "release/benchmarks/ray_data_gpu_sort/cloud/worker.py": "d" * 64
}
OVERLAY_FILE = ("python/ray/data/a.py", "e" * 64)
OVERLAY_HASH = hashlib.sha256(
    OVERLAY_FILE[0].encode() + b"\0" + bytes.fromhex(OVERLAY_FILE[1])
).hexdigest()
EXPECTED_PROVENANCE = {
    "bundle_digest": BUNDLE_SHA256,
    "wheel_sha256": RAY_WHEEL_SHA256,
    "harness_files": HARNESS_FILES,
    "ray_data_overlay": {"file_count": 1, "sha256": OVERLAY_HASH},
    "ray_version": RAY_VERSION,
    "ray_commit": RAY_COMMIT,
}


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _prior_sync(tmp_path: Path) -> Path:
    root = tmp_path / "prior"
    inventory_nodes = [
        {"instance_id": f"i-{index:02d}"} for index in range(16)
    ]
    stage_nodes = [
        {
            "instance_id": f"i-{index:02d}",
            "dataset_digest": DATASET_SHA256,
            "manifest_sha256": MANIFEST_SHA256,
        }
        for index in range(16)
    ]
    _write(
        root / "admin/inventory.json",
        {
            "schema_version": 1,
            "kind": "gpu_sort_cloud_inventory",
            "nodes": inventory_nodes,
        },
    )
    _write(
        root / "admin/dataset-stage.json",
        {
            "schema_version": 1,
            "kind": "gpu_sort_cloud_dataset_stage",
            "dataset_digest": DATASET_SHA256,
            "nodes": stage_nodes,
            "digest": digest(stage_nodes),
        },
    )
    return root


def _command(
    name: str,
    *,
    output: str | None = None,
    kind: str = "natural",
    selected_wave: str | None = None,
) -> str:
    arguments = [
        "python",
        "-m",
        WORKER_MODULE,
        "--output",
        output or f"{REMOTE_RESULTS}/results/{name}.json",
        "--dataset-root",
        "/mnt/nvme/dataset",
        "--spill-directory",
        "/mnt/nvme/runtime/spill",
        "--kind",
        kind,
        "--backend",
        "pyarrow",
        "--cell",
        "full",
        "--repetition",
        "1",
        "--scale-numerator",
        "245",
        "--scale-denominator",
        "100",
    ]
    if selected_wave is None:
        arguments.extend(("--wave-fraction", "0.5"))
    else:
        arguments.extend(("--selected-wave", selected_wave))
    return shlex.join(arguments)


def _artifact(*, repetition: int = 1, wave_fraction: float = 0.5) -> dict[str, Any]:
    trial = {
        "kind": "natural",
        "backend": "pyarrow",
        "cell": "full",
        "repetition": repetition,
        "scale_numerator": 245,
        "scale_denominator": 100,
        "wave_fraction": wave_fraction,
    }
    identity = {
        "trial": trial,
        "bundle_digest": BUNDLE_SHA256,
        "input_plan_digest": PLAN_SHA256,
        "dataset_manifest_sha256": MANIFEST_SHA256,
        "wheel_sha256": RAY_WHEEL_SHA256,
        "harness_files": HARNESS_FILES,
        "ray_data_overlay": {"file_count": 1, "sha256": OVERLAY_HASH},
    }
    return {
        "valid": False,
        "status": "rejected",
        **trial,
        "cell": {"name": "full"},
        "rejection_reasons": ["one or more output ObjectRefs are not Ray-locatable"],
        "plan": {
            "kind": "natural",
            "blocks": 1536,
            "rows": 197809964,
            "digest": PLAN_SHA256,
        },
        "artifact_identity": {**identity, "digest": digest(identity)},
        "ray_version": RAY_VERSION,
        "ray_commit": RAY_COMMIT,
    }


def _recover(
    tmp_path: Path,
    *,
    step: Step,
    stdout: str,
    prior: tuple[Path, ...] | None = None,
) -> dict[str, Any]:
    return _recover_artifact_from_stdout(
        step=step,
        stdout=stdout,
        remote_results=REMOTE_RESULTS,
        local=tmp_path / "recovered-step",
        prior_successful_syncs=prior or (_prior_sync(tmp_path),),
        expected_provenance=EXPECTED_PROVENANCE,
    )


def test_recovers_one_rejected_performance_artifact_with_full_provenance(
    tmp_path: Path,
) -> None:
    name = "natural-245x-r1"
    output = Path(f"{REMOTE_RESULTS}/results/{name}.json")
    artifact = _artifact()
    step = Step(name, _command(name), allow_failure=True)

    recovery = _recover(
        tmp_path,
        step=step,
        stdout=f"ordinary worker output\n{artifact_sentinel(output, artifact)}\n",
    )

    recovered = tmp_path / "recovered-step/results/natural-245x-r1.json"
    assert json.loads(recovered.read_text(encoding="utf-8")) == artifact
    assert recovery == {
        "artifact_recovered_from_stdout": True,
        "recovered_artifact": str(recovered),
        "recovered_artifact_sha256": digest(artifact),
    }


def test_worker_emits_sentinel_after_writing_a_rejected_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import ray

    output = tmp_path / "rejected.json"
    artifact = _artifact()
    wrote = False
    real_atomic_json = worker.atomic_json
    real_artifact_sentinel = worker.artifact_sentinel

    def tracked_write(path: Path, value: dict[str, Any]) -> None:
        nonlocal wrote
        real_atomic_json(path, value)
        wrote = True

    def tracked_sentinel(path: Path, value: dict[str, Any]) -> str:
        assert wrote and path == output and value == artifact
        return real_artifact_sentinel(path, value)

    monkeypatch.setattr(ray, "init", lambda **kwargs: None)
    monkeypatch.setattr(ray, "shutdown", lambda: None)
    monkeypatch.setattr(ray, "__version__", worker.RAY_VERSION)
    monkeypatch.setattr(ray, "__commit__", worker.RAY_COMMIT)
    monkeypatch.setattr(worker, "_alive_nodes", lambda ray_module: [{}] * 16)
    monkeypatch.setattr(worker, "load_manifest", lambda root: {})
    monkeypatch.setattr(worker, "_run_performance", lambda *args: artifact)
    monkeypatch.setattr(worker, "atomic_json", tracked_write)
    monkeypatch.setattr(worker, "artifact_sentinel", tracked_sentinel)

    returncode = worker.main(
        [
            "--output",
            str(output),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--spill-directory",
            str(tmp_path / "spill"),
            "--kind",
            "natural",
            "--backend",
            "pyarrow",
            "--scale-numerator",
            "245",
            "--scale-denominator",
            "100",
        ]
    )

    assert returncode == 1
    lines = capsys.readouterr().out.splitlines()
    assert sum(ARTIFACT_SENTINEL in line for line in lines) == 1
    assert json.loads(output.read_text(encoding="utf-8")) == artifact


def test_bundle_digest_verifies_every_mounted_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_root = tmp_path / "verified-bundle"
    paths = [
        "wheelhouse/ray.whl",
        "release/benchmarks/ray_data_gpu_sort/cloud/worker.py",
        "release/benchmarks/ray_data_gpu_sort/data.py",
        "python/ray/data/a.py",
    ]
    for index, name in enumerate(paths):
        path = bundle_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"file-{index}", encoding="utf-8")
    files = {name: file_sha256(bundle_root / name) for name in paths}
    bundle = {
        "schema_version": 1,
        "kind": "gpu_sort_cloud_bundle",
        "bundle_digest": digest(files),
        "files": files,
        "ray_version": RAY_VERSION,
        "ray_commit": RAY_COMMIT,
        "wheel_sha256": files["wheelhouse/ray.whl"],
    }
    _write(bundle_root / "BUNDLE.json", bundle)
    cluster = tmp_path / "bundle-cluster.yaml"
    cluster.write_text(
        yaml.safe_dump(
            {"file_mounts": {"/home/ray/gpu-sort-src": str(bundle_root)}}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        lifecycle, "RAY_WHEEL_SHA256", files["wheelhouse/ray.whl"]
    )

    provenance = _bundle_provenance(cluster)
    assert provenance["bundle_digest"] == digest(files)
    assert worker._verified_bundle_digest(bundle_root) == digest(files)

    (bundle_root / "release/benchmarks/ray_data_gpu_sort/data.py").write_text(
        "tampered", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="bundle identity is invalid"):
        _bundle_provenance(cluster)
    with pytest.raises(RuntimeError, match="immutable bundle identity"):
        worker._verified_bundle_digest(bundle_root)


def test_lifecycle_succeeds_but_preserves_failed_rsync_after_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import boto3

    name = "natural-245x-r1"
    artifact = _artifact()
    stdout = artifact_sentinel(
        Path(f"{REMOTE_RESULTS}/results/{name}.json"), artifact
    )
    bundle_root = tmp_path / "bundle"
    bundle_files = {
        f"wheelhouse/ray-{RAY_VERSION}.whl": RAY_WHEEL_SHA256,
        **HARNESS_FILES,
        OVERLAY_FILE[0]: OVERLAY_FILE[1],
    }
    _write(
        bundle_root / "BUNDLE.json",
        {
            "schema_version": 1,
            "kind": "gpu_sort_cloud_bundle",
            "bundle_digest": digest(bundle_files),
            "files": bundle_files,
            "ray_version": RAY_VERSION,
            "ray_commit": RAY_COMMIT,
            "wheel_sha256": RAY_WHEEL_SHA256,
        },
    )
    cluster = tmp_path / "cluster.yaml"
    cluster.write_text(
        yaml.safe_dump(
            {
                "cluster_name": "recovery-test",
                "provider": {"region": "us-west-2"},
                "file_mounts": {"/home/ray/gpu-sort-src": str(bundle_root)},
            }
        ),
        encoding="utf-8",
    )
    plan = tmp_path / "commands.json"
    _write(
        plan,
        {
            "schema_version": 1,
            "commands": [
                {"name": "mkdir-results", "command": "true"},
                {"name": "inventory", "command": "true"},
                {"name": "stage-dataset", "command": "true"},
                {
                    "name": name,
                    "command": _command(name),
                    "allow_failure": True,
                },
            ],
        },
    )
    artifact_root = tmp_path / "execution"
    artifact_root.mkdir()
    args = SimpleNamespace(
        command_plan=plan,
        cluster_yaml=cluster,
        cluster_name="recovery-test",
        region="us-west-2",
        campaign="campaign",
        arm="cpu",
        nodes=16,
        cpus=256.0,
        gpus=0.0,
        instance_type="m5dn.4xlarge",
        single_availability_zone=True,
        artifact_root=artifact_root,
        execute=False,
        confirm_plan_sha=None,
        aws_profile=None,
        ray_cli=tmp_path / "ray",
        remote_python="/venv/bin/python",
        remote_results=REMOTE_RESULTS,
        bundle_digest=BUNDLE_SHA256,
    )
    monkeypatch.setattr(
        lifecycle, "_bundle_provenance", lambda cluster_yaml: EXPECTED_PROVENANCE
    )
    assert lifecycle.execute(args) == 0
    args.execute = True
    args.confirm_plan_sha = json.loads(
        (artifact_root / "lifecycle-plan.json").read_text(encoding="utf-8")
    )["plan_sha256"]

    class Session:
        def client(self, name: str) -> object:
            assert name == "ec2"
            return object()

    sync_count = 0

    def fake_run(
        argv: list[object], *, log: Path, timeout: float, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        nonlocal sync_count
        values = [str(value) for value in argv]
        if len(values) > 1 and values[1] == "rsync-down":
            sync_count += 1
            local = Path(values[-1])
            if sync_count >= 2:
                inventory_nodes = [
                    {"instance_id": f"i-{index:02d}"} for index in range(16)
                ]
                _write(
                    local / "admin/inventory.json",
                    {
                        "schema_version": 1,
                        "kind": "gpu_sort_cloud_inventory",
                        "nodes": inventory_nodes,
                    },
                )
            if sync_count >= 3:
                stage_nodes = [
                    {
                        "instance_id": f"i-{index:02d}",
                        "dataset_digest": DATASET_SHA256,
                        "manifest_sha256": MANIFEST_SHA256,
                    }
                    for index in range(16)
                ]
                _write(
                    local / "admin/dataset-stage.json",
                    {
                        "schema_version": 1,
                        "kind": "gpu_sort_cloud_dataset_stage",
                        "dataset_digest": DATASET_SHA256,
                        "nodes": stage_nodes,
                        "digest": digest(stage_nodes),
                    },
                )
            return subprocess.CompletedProcess(
                values, 23 if sync_count == 4 else 0, stdout="rsync output"
            )
        if len(values) > 1 and values[1] == "exec" and WORKER_MODULE in values[-1]:
            return subprocess.CompletedProcess(values, 1, stdout=stdout)
        return subprocess.CompletedProcess(values, 0, stdout="")

    ids = tuple(f"i-{index:02d}" for index in range(16))
    monkeypatch.setattr(boto3, "Session", lambda **kwargs: Session())
    monkeypatch.setattr(lifecycle, "_scoped_instances", lambda *args: ())
    monkeypatch.setattr(lifecycle, "_run", fake_run)
    monkeypatch.setattr(lifecycle, "_wait_shape", lambda **kwargs: ids)
    monkeypatch.setattr(lifecycle, "_instance_ids", lambda *args, **kwargs: ids)
    monkeypatch.setattr(
        lifecycle,
        "_terminate",
        lambda *args, **kwargs: {
            "verified_empty": True,
            "terminated_instance_ids": list(ids),
        },
    )

    assert lifecycle.execute(args) == 0
    result = json.loads((artifact_root / "result.json").read_text(encoding="utf-8"))
    record = result["steps"][-1]
    assert result["status"] == "success"
    assert record["returncode"] == 1
    assert record["rsync_returncode"] == 23
    assert record["allow_failure"] is True
    assert record["artifact_recovered_from_stdout"] is True
    recovered = artifact_root / f"remote-results/04-{name}/results/{name}.json"
    assert json.loads(recovered.read_text(encoding="utf-8")) == artifact


@pytest.mark.parametrize(
    "stdout,match",
    [
        ("no sentinel here", "exactly one"),
        (ARTIFACT_SENTINEL + "not-base64!", "valid base64"),
        (
            ARTIFACT_SENTINEL
            + base64.b64encode(canonical_bytes({"schema_version": 1})).decode(),
            "envelope fields differ",
        ),
    ],
)
def test_recovery_rejects_missing_or_malformed_sentinel(
    tmp_path: Path, stdout: str, match: str
) -> None:
    name = "natural-245x-r1"
    with pytest.raises(ValueError, match=match):
        _recover(
            tmp_path,
            step=Step(name, _command(name), allow_failure=True),
            stdout=stdout,
        )


def test_recovery_rejects_mismatched_trial_and_duplicate_sentinel(
    tmp_path: Path,
) -> None:
    name = "natural-245x-r1"
    output = Path(f"{REMOTE_RESULTS}/results/{name}.json")
    mismatched = artifact_sentinel(output, _artifact(repetition=2))
    step = Step(name, _command(name), allow_failure=True)
    with pytest.raises(ValueError, match="identity differs"):
        _recover(tmp_path, step=step, stdout=mismatched)
    valid = artifact_sentinel(output, _artifact())
    with pytest.raises(ValueError, match="exactly one"):
        _recover(tmp_path, step=step, stdout=f"{valid}\n{valid}\n")


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("identity-schema", "identity schema"),
        ("dataset", "dataset identity"),
        ("plan", "input-plan identity"),
        ("harness", "harness_files differs"),
        ("ray-commit", "Ray commit differs"),
    ],
)
def test_recovery_binds_identity_to_admin_plan_and_bundle(
    tmp_path: Path, mutation: str, match: str
) -> None:
    name = "natural-245x-r1"
    output = Path(f"{REMOTE_RESULTS}/results/{name}.json")
    artifact = json.loads(json.dumps(_artifact()))
    identity = artifact["artifact_identity"]
    if mutation == "identity-schema":
        identity.pop("wheel_sha256")
    elif mutation == "dataset":
        identity["dataset_manifest_sha256"] = "f" * 64
    elif mutation == "plan":
        artifact["plan"]["digest"] = "f" * 64
    elif mutation == "harness":
        identity["harness_files"] = {
            "release/benchmarks/ray_data_gpu_sort/cloud/worker.py": "f" * 64
        }
    else:
        artifact["ray_commit"] = "f" * 40
    identity_body = {key: value for key, value in identity.items() if key != "digest"}
    identity["digest"] = digest(identity_body)

    with pytest.raises(ValueError, match=match):
        _recover(
            tmp_path,
            step=Step(name, _command(name), allow_failure=True),
            stdout=artifact_sentinel(output, artifact),
        )


def test_recovery_rejects_required_and_nonperformance_steps(tmp_path: Path) -> None:
    name = "natural-245x-r1"
    output = Path(f"{REMOTE_RESULTS}/results/{name}.json")
    stdout = artifact_sentinel(output, _artifact())
    with pytest.raises(ValueError, match="required steps"):
        _recover(tmp_path, step=Step(name, _command(name)), stdout=stdout)
    with pytest.raises(ValueError, match="only performance"):
        _recover(
            tmp_path,
            step=Step(name, _command(name, kind="smoke"), allow_failure=True),
            stdout=stdout,
        )


@pytest.mark.parametrize(
    ("name", "output"),
    [
        ("../escape", f"{REMOTE_RESULTS}/results/escape.json"),
        ("natural-245x-r1", f"{REMOTE_RESULTS}/results/../escape.json"),
        ("natural-245x-r1", f"{REMOTE_RESULTS}/results/another-step.json"),
    ],
)
def test_recovery_rejects_filename_mismatch_and_path_traversal(
    tmp_path: Path, name: str, output: str
) -> None:
    sentinel = artifact_sentinel(Path(output), _artifact())
    with pytest.raises(ValueError):
        _recover(
            tmp_path,
            step=Step(name, _command(name, output=output), allow_failure=True),
            stdout=sentinel,
        )
    assert not (tmp_path / "escape.json").exists()


def test_recovery_uses_only_a_prior_synced_selected_wave(tmp_path: Path) -> None:
    name = "natural-245x-r1"
    selected = f"{REMOTE_RESULTS}/configs/selected-wave.json"
    output = Path(f"{REMOTE_RESULTS}/results/{name}.json")
    prior = _prior_sync(tmp_path)
    _write(prior / "configs/selected-wave.json", {"selected_wave_fraction": 0.375})
    artifact = _artifact(wave_fraction=0.375)

    _recover(
        tmp_path,
        step=Step(
            name,
            _command(name, selected_wave=selected),
            allow_failure=True,
        ),
        stdout=artifact_sentinel(output, artifact),
        prior=(prior,),
    )

    (prior / "configs/selected-wave.json").unlink()
    with pytest.raises(FileNotFoundError):
        _recover(
            tmp_path,
            step=Step(
                name,
                _command(name, selected_wave=selected),
                allow_failure=True,
            ),
            stdout=artifact_sentinel(output, artifact),
            prior=(prior,),
        )
