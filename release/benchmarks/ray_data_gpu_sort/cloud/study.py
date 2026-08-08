"""Prepare or execute the full AWS BTS study or one-arm repair subsets."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import lifecycle
from .common import atomic_json, digest, file_sha256, read_json
from .render import (
    REMOTE_BASE,
    REMOTE_DATASET_MANIFEST,
    REMOTE_SOURCE,
    make_bundle,
    render_cluster,
    repo_root,
    write_yaml,
)
from .spec import CPUS, NODES, Trial, trials

FULL_TRIAL_MODE = "full"
GPU_REPAIR_TRIAL_MODE = "gpu-repair"
GPU_245X_REPAIR_TRIAL_MODE = "gpu-245x-repair"
CPU_REPAIR_TRIAL_MODE = "cpu-repair"
TRIAL_MODES = (
    FULL_TRIAL_MODE,
    GPU_REPAIR_TRIAL_MODE,
    GPU_245X_REPAIR_TRIAL_MODE,
    CPU_REPAIR_TRIAL_MODE,
)
GPU_REPAIR_TRIAL_NAMES = (
    "transport-smoke",
    "tune-2x-wave-0500",
    "tune-2x-wave-0375",
    "natural-2x-selected-r2",
    "natural-245x-r1",
    "natural-245x-r2",
)
GPU_245X_REPAIR_TRIAL_NAMES = (
    "transport-smoke",
    "natural-245x-r1",
    "natural-245x-r2",
)
CPU_REPAIR_TRIAL_NAMES = (
    "trend-full-r1",
    "trend-origin-string-r1",
    "trend-origin-integer-r1",
    "trend-route-r1",
    "natural-2x-r1",
    "natural-245x-r1",
)
REPAIR_TOPOLOGIES = {
    "gpu": {
        "nodes": 16,
        "cpus": 256.0,
        "gpus": 16.0,
        "instance_type": "g6.4xlarge",
        "single_availability_zone": True,
    },
    "cpu": {
        "nodes": 16,
        "cpus": 256.0,
        "gpus": 0.0,
        "instance_type": "m5dn.4xlarge",
        "single_availability_zone": True,
    },
}


def _plan_digest_matches(value: Mapping[str, Any]) -> bool:
    claimed = value.get("plan_sha256")
    body = {key: item for key, item in value.items() if key != "plan_sha256"}
    return isinstance(claimed, str) and claimed == digest(body)


def _validate_prior_arm_teardown(
    teardown_path: Path,
    arm: str,
    frozen: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if arm not in REPAIR_TOPOLOGIES:
        raise ValueError(f"unsupported prior arm: {arm}")
    teardown_path = teardown_path.expanduser().resolve()
    execution_root = teardown_path.parent
    paths = {
        "teardown": teardown_path,
        "lifecycle_plan": execution_root / "lifecycle-plan.json",
        "instances": execution_root / "instances.json",
        "result": execution_root / "result.json",
        "original_study": execution_root.parent.parent / "study.json",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"prior {arm.upper()} execution provenance is missing: {missing}"
        )

    teardown = read_json(paths["teardown"])
    lifecycle_plan = read_json(paths["lifecycle_plan"])
    instances = read_json(paths["instances"])
    result = read_json(paths["result"])
    original_study = read_json(paths["original_study"])
    if teardown.get("verified_empty") is not True:
        raise RuntimeError(
            f"prior {arm.upper()} teardown receipt is not verified empty"
        )
    if lifecycle_plan.get("arm") != arm:
        raise RuntimeError(f"prior lifecycle receipt is not a {arm.upper()} arm")
    if not _plan_digest_matches(lifecycle_plan):
        raise RuntimeError(f"prior {arm.upper()} lifecycle plan digest is invalid")
    if not _plan_digest_matches(original_study):
        raise RuntimeError("original cloud study plan digest is invalid")
    if original_study.get("trial_mode", FULL_TRIAL_MODE) != FULL_TRIAL_MODE:
        raise RuntimeError(
            f"prior {arm.upper()} execution is not from the original full study"
        )

    campaign = lifecycle_plan.get("campaign")
    cluster_name = lifecycle_plan.get("cluster_name")
    prior_arm = (original_study.get("arms") or {}).get(arm) or {}
    topology = prior_arm.get("topology") or {}
    if not isinstance(campaign, str) or not campaign:
        raise RuntimeError(f"prior {arm.upper()} campaign identity is missing")
    if (
        original_study.get("campaign") != campaign
        or prior_arm.get("cluster_name") != cluster_name
        or cluster_name != f"ray-gpu-sort-{campaign}-{arm}"
    ):
        raise RuntimeError(
            f"prior {arm.upper()} campaign/cluster identity is inconsistent"
        )
    expected = {
        "nodes": int(topology.get("nodes", -1)),
        "cpus": float(topology.get("nodes", -1))
        * float(topology.get("vcpus_per_node", -1)),
        "gpus": float(topology.get("nodes", -1))
        * float(topology.get("gpus_per_node", -1)),
        "instance_type": topology.get("instance_type"),
        "single_availability_zone": True,
    }
    observed_expected = lifecycle_plan.get("expected") or {}
    if any(observed_expected.get(key) != value for key, value in expected.items()):
        raise RuntimeError(
            f"prior {arm.upper()} lifecycle topology differs from its study plan"
        )
    if expected != REPAIR_TOPOLOGIES[arm]:
        raise RuntimeError(
            f"prior execution is not the frozen 16-node {arm.upper()} topology"
        )

    instance_ids = instances.get("instance_ids")
    terminated_ids = teardown.get("terminated_instance_ids")
    if (
        not isinstance(instance_ids, list)
        or not isinstance(terminated_ids, list)
        or len(instance_ids) != 16
        or len(set(instance_ids)) != 16
        or sorted(instance_ids) != sorted(terminated_ids)
    ):
        raise RuntimeError(
            f"prior {arm.upper()} instance IDs do not exactly match terminated IDs"
        )
    if result.get("status") != "success" or result.get("teardown") != teardown:
        raise RuntimeError(
            f"prior {arm.upper()} lifecycle result does not contain the exact "
            "successful teardown"
        )

    # Schema-v2 receipts are self-identifying. Older completed campaigns are
    # accepted through the sibling lifecycle/study receipts above.
    future_fields = {
        "campaign": campaign,
        "arm": arm,
        "cluster_name": cluster_name,
        "lifecycle_plan_sha256": lifecycle_plan["plan_sha256"],
        "instance_ids_sha256": digest(sorted(instance_ids)),
    }
    for key, value in future_fields.items():
        if key in teardown and teardown[key] != value:
            raise RuntimeError(
                f"prior {arm.upper()} teardown identity field {key!r} is inconsistent"
            )

    provenance = {
        "schema_version": 2,
        "campaign": campaign,
        "cluster_name": cluster_name,
        "arm": arm,
        "topology": expected,
        "instance_ids": sorted(instance_ids),
        "instance_ids_sha256": digest(sorted(instance_ids)),
        "original_study_plan_sha256": original_study["plan_sha256"],
        "lifecycle_plan_sha256": lifecycle_plan["plan_sha256"],
        "files": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in paths.items()
        },
    }
    if frozen is not None and provenance != dict(frozen):
        raise RuntimeError(
            f"prior {arm.upper()} execution provenance changed after repair preparation"
        )
    return provenance


def _validate_prior_gpu_teardown(
    teardown_path: Path, frozen: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    return _validate_prior_arm_teardown(teardown_path, "gpu", frozen)


def _validate_prior_cpu_teardown(
    teardown_path: Path, frozen: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    return _validate_prior_arm_teardown(teardown_path, "cpu", frozen)


def _trial_values(arm: str, trial_mode: str) -> tuple[Trial, ...]:
    if trial_mode == FULL_TRIAL_MODE:
        return trials(arm)
    if trial_mode == GPU_245X_REPAIR_TRIAL_MODE:
        if arm != "gpu":
            raise ValueError(f"{trial_mode} trial mode has no {arm.upper()} arm")
        return (
            Trial("transport-smoke", "gpu", "smoke", "smoke", "full", 1),
            Trial(
                "natural-245x-r1",
                "gpu",
                "natural",
                "gpu",
                "full",
                1,
                245,
                100,
                wave_fraction=0.50,
                allow_failure=True,
            ),
            Trial(
                "natural-245x-r2",
                "gpu",
                "natural",
                "gpu",
                "full",
                2,
                245,
                100,
                wave_fraction=0.50,
                allow_failure=True,
            ),
        )
    repair = {
        GPU_REPAIR_TRIAL_MODE: ("gpu", GPU_REPAIR_TRIAL_NAMES),
        CPU_REPAIR_TRIAL_MODE: ("cpu", CPU_REPAIR_TRIAL_NAMES),
    }
    if trial_mode not in repair:
        raise ValueError(f"unknown trial mode: {trial_mode}")
    expected_arm, names = repair[trial_mode]
    if arm != expected_arm:
        raise ValueError(f"{trial_mode} trial mode has no {arm.upper()} arm")
    by_name = {trial.name: trial for trial in trials(arm)}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise RuntimeError(f"{arm.upper()} repair trials are missing: {missing}")
    return tuple(by_name[name] for name in names)


def _module(python: str, module: str, *arguments: object) -> str:
    return (
        "cd "
        + shlex.quote(str(REMOTE_SOURCE))
        + " && "
        + shlex.join(
            [
                "env",
                f"PYTHONPATH={REMOTE_SOURCE}",
                python,
                "-m",
                module,
                *(str(item) for item in arguments),
            ]
        )
    )


def _worker_command(
    python: str,
    trial: Trial,
    *,
    dataset_root: Path,
    results_root: Path,
    spill_directory: Path,
) -> str:
    arguments: list[object] = [
        "--output",
        results_root / "results" / f"{trial.name}.json",
        "--dataset-root",
        dataset_root,
        "--spill-directory",
        spill_directory,
        "--kind",
        trial.kind,
        "--backend",
        trial.backend,
        "--cell",
        trial.cell,
        "--repetition",
        trial.repetition,
        "--scale-numerator",
        trial.scale_numerator,
        "--scale-denominator",
        trial.scale_denominator,
    ]
    if trial.wave_fraction is not None:
        arguments.extend(("--wave-fraction", trial.wave_fraction))
    if trial.selected_wave_file is not None:
        arguments.extend(("--selected-wave", results_root / trial.selected_wave_file))
    return _module(
        python,
        "release.benchmarks.ray_data_gpu_sort.cloud.worker",
        *arguments,
    )


def _commands(
    *,
    arm: str,
    python: str,
    generated: Path,
    reset_yamls: Mapping[str, Path],
    reset_runtimes: Mapping[str, Path],
    dataset_root: Path,
    results_root: Path,
    trial_values: Sequence[Trial],
) -> list[dict[str, Any]]:
    admin = "release.benchmarks.ray_data_gpu_sort.cloud.admin"
    value: list[dict[str, Any]] = [
        {
            "name": "mkdir-results",
            "command": f"mkdir -p {shlex.quote(str(results_root / 'admin'))} {shlex.quote(str(results_root / 'configs'))} {shlex.quote(str(results_root / 'results'))}",
            "timeout_seconds": 300,
        },
        {
            "name": "inventory",
            "command": _module(
                python,
                admin,
                "inventory",
                "--nodes",
                NODES,
                "--cpus",
                CPUS,
                "--gpus",
                16 if arm == "gpu" else 0,
                "--output",
                results_root / "admin/inventory.json",
            ),
            "timeout_seconds": 1800,
        },
        {
            "name": "stage-dataset",
            "command": _module(
                python,
                admin,
                "stage",
                "--nodes",
                NODES,
                "--manifest",
                REMOTE_DATASET_MANIFEST,
                "--dataset-root",
                dataset_root,
                "--output",
                results_root / "admin/dataset-stage.json",
            ),
            "timeout_seconds": 7200,
        },
    ]
    for trial in trial_values:
        if arm == "gpu" and trial.name == "natural-2x-selected-r2":
            value.append(
                {
                    "name": "select-wave",
                    "command": _module(
                        python,
                        "release.benchmarks.ray_data_gpu_sort.cloud.workflow",
                        "--default",
                        results_root / "results/tune-2x-wave-0500.json",
                        "--candidate",
                        results_root / "results/tune-2x-wave-0375.json",
                        "--output",
                        results_root / "configs/selected-wave.json",
                    ),
                    "timeout_seconds": 300,
                }
            )
        value.append(
            {
                "name": trial.name,
                "command": _worker_command(
                    python,
                    trial,
                    dataset_root=dataset_root,
                    results_root=results_root,
                    spill_directory=reset_runtimes[trial.name] / "spill",
                ),
                "restart_yaml": str(reset_yamls[trial.name]),
                "allow_failure": trial.allow_failure,
                "timeout_seconds": (
                    1800
                    if (
                        trial.kind == "natural"
                        and trial.name.startswith("natural-245x-")
                        and trial.wave_fraction == 0.50
                        and trial.selected_wave_file is None
                    )
                    else 14400
                    if trial.kind == "natural"
                    else 7200
                ),
            }
        )
    return value


def _validate_inputs(local: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    required = {
        "region",
        "availability_zone",
        "ami_id",
        "security_group_ids",
        "iam_instance_profile_arn",
        "ec2_key_name",
        "ssh_private_key_path",
        "ssh_user",
    }
    missing = sorted(required - set(local))
    if missing:
        raise ValueError(f"local config is missing: {missing}")
    raw_subnets = local.get("subnet_ids") or (
        [local["subnet_id"]] if local.get("subnet_id") else []
    )
    if not raw_subnets or not all(
        isinstance(item, str) and item for item in raw_subnets
    ):
        raise ValueError(
            "local config requires subnet_id or a non-empty subnet_ids list"
        )
    if local.get("docker_image") and "@sha256:" not in str(local["docker_image"]):
        raise ValueError("cloud Docker image must use an immutable digest")
    if int(manifest.get("file_count", -1)) != 153:
        raise ValueError("portable BTS manifest does not contain 153 Parquet files")
    cohort = manifest.get("cohort") or {}
    if (
        int(cohort.get("rows", -1)) != 80_738_761
        or int(cohort.get("blocks", -1)) != 627
    ):
        raise ValueError("portable BTS manifest does not describe the fixed cohort")


def prepare(
    *,
    campaign: str,
    local_config: Path,
    dataset_manifest: Path,
    output_root: Path,
    trial_mode: str = FULL_TRIAL_MODE,
    prior_gpu_teardown: Path | None = None,
    prior_cpu_teardown: Path | None = None,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"study output is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    root = repo_root()
    local = read_json(local_config)
    manifest = read_json(dataset_manifest)
    _validate_inputs(local, manifest)
    prior_gpu: dict[str, Any] | None = None
    prior_cpu: dict[str, Any] | None = None
    if trial_mode in (GPU_REPAIR_TRIAL_MODE, GPU_245X_REPAIR_TRIAL_MODE):
        if prior_gpu_teardown is None:
            raise ValueError(f"{trial_mode} preparation requires --prior-gpu-teardown")
        if prior_cpu_teardown is not None:
            raise ValueError(
                "--prior-cpu-teardown is only valid with --trial-mode cpu-repair"
            )
        prior_gpu = _validate_prior_gpu_teardown(prior_gpu_teardown)
    elif trial_mode == CPU_REPAIR_TRIAL_MODE:
        if prior_cpu_teardown is None:
            raise ValueError("cpu-repair preparation requires --prior-cpu-teardown")
        if prior_gpu_teardown is not None:
            raise ValueError(
                "--prior-gpu-teardown is only valid with --trial-mode gpu-repair"
            )
        prior_cpu = _validate_prior_cpu_teardown(prior_cpu_teardown)
    elif trial_mode == FULL_TRIAL_MODE:
        if prior_gpu_teardown is not None or prior_cpu_teardown is not None:
            raise ValueError(
                "prior teardown flags are only valid with a matching repair mode"
            )
    else:
        raise ValueError(f"unknown trial mode: {trial_mode}")
    topology = read_json(Path(__file__).with_name("topologies.json"))
    bundle_root = output_root / "bundle"
    bundle = make_bundle(
        root=root, output=bundle_root, dataset_manifest=dataset_manifest
    )
    arms: dict[str, Any] = {}
    if trial_mode in (GPU_REPAIR_TRIAL_MODE, GPU_245X_REPAIR_TRIAL_MODE):
        arm_names = ("gpu",)
    elif trial_mode == CPU_REPAIR_TRIAL_MODE:
        arm_names = ("cpu",)
    else:
        arm_names = ("gpu", "cpu")
    for arm in arm_names:
        arm_trials = _trial_values(arm, trial_mode)
        arm_root = output_root / "prepared" / arm
        launch, remote = render_cluster(
            campaign=campaign,
            arm=arm,
            local=local,
            topology=topology[arm],
            bundle=bundle_root,
            bundle_digest=str(bundle["bundle_digest"]),
            trial="bootstrap",
        )
        launch_path = arm_root / "launch.yaml"
        write_yaml(launch_path, launch)
        resets: dict[str, Path] = {}
        reset_runtimes: dict[str, Path] = {}
        for trial in arm_trials:
            reset, reset_remote = render_cluster(
                campaign=campaign,
                arm=arm,
                local=local,
                topology=topology[arm],
                bundle=bundle_root,
                bundle_digest=str(bundle["bundle_digest"]),
                trial=trial.name,
            )
            if reset_remote["remote_python"] != remote["remote_python"]:
                raise RuntimeError("remote environment changed between reset YAMLs")
            path = arm_root / "resets" / f"{trial.name}.yaml"
            write_yaml(path, reset)
            resets[trial.name] = path.resolve()
            reset_runtimes[trial.name] = Path(reset_remote["runtime"])
        dataset_root = REMOTE_BASE / "dataset" / str(manifest["dataset_digest"])
        results_root = REMOTE_BASE / "results" / campaign / arm
        commands = _commands(
            arm=arm,
            python=remote["remote_python"],
            generated=output_root,
            reset_yamls=resets,
            reset_runtimes=reset_runtimes,
            dataset_root=dataset_root,
            results_root=results_root,
            trial_values=arm_trials,
        )
        command_path = arm_root / "commands.json"
        atomic_json(command_path, {"schema_version": 1, "commands": commands})
        lifecycle.load_steps(command_path)
        lifecycle_args = [
            "--campaign",
            campaign,
            "--arm",
            arm,
            "--cluster-name",
            remote["cluster_name"],
            "--cluster-yaml",
            str(launch_path.resolve()),
            "--command-plan",
            str(command_path.resolve()),
            "--remote-results",
            str(results_root),
            "--bundle-digest",
            str(bundle["bundle_digest"]),
            "--remote-python",
            remote["remote_python"],
            "--region",
            str(local["region"]),
            "--nodes",
            str(topology[arm]["nodes"]),
            "--cpus",
            str(int(topology[arm]["nodes"]) * int(topology[arm]["vcpus_per_node"])),
            "--gpus",
            str(int(topology[arm]["nodes"]) * int(topology[arm]["gpus_per_node"])),
            "--instance-type",
            str(topology[arm]["instance_type"]),
            "--single-availability-zone",
        ]
        if local.get("aws_profile"):
            lifecycle_args.extend(("--aws-profile", str(local["aws_profile"])))
        arms[arm] = {
            "topology": topology[arm],
            "cluster_name": remote["cluster_name"],
            "launch_yaml": str(launch_path.resolve()),
            "command_plan": str(command_path.resolve()),
            "remote_results": str(results_root),
            "remote_python": remote["remote_python"],
            "lifecycle_args": lifecycle_args,
            "trials": [trial.to_dict() for trial in arm_trials],
        }
    body = {
        "schema_version": 1,
        "kind": "gpu_sort_bts_cloud_study",
        "campaign": campaign,
        "trial_mode": trial_mode,
        "execution_order": (
            ["verified_prior_gpu_teardown", trial_mode, "verified_teardown"]
            if trial_mode in (GPU_REPAIR_TRIAL_MODE, GPU_245X_REPAIR_TRIAL_MODE)
            else ["verified_prior_cpu_teardown", "cpu-repair", "verified_teardown"]
            if trial_mode == CPU_REPAIR_TRIAL_MODE
            else ["gpu", "verified_teardown", "cpu"]
        ),
        "prior_gpu_teardown": prior_gpu,
        "prior_cpu_teardown": prior_cpu,
        "fresh_ray_runtime_per_trial": True,
        "retained_ec2_fleet_per_arm": True,
        "ray_default_plasma_both_arms": True,
        "dataset_digest": manifest["dataset_digest"],
        "dataset_manifest_sha256": file_sha256(dataset_manifest),
        "bundle": bundle,
        "arms": arms,
    }
    result = {**body, "plan_sha256": digest(body)}
    atomic_json(output_root / "study.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def execute_arm(
    plan_path: Path,
    arm: str,
    artifact_root: Path,
    *,
    execute: bool,
    confirm_plan_sha: str | None,
) -> int:
    plan = read_json(plan_path)
    if plan.get("kind") != "gpu_sort_bts_cloud_study" or not _plan_digest_matches(plan):
        raise ValueError("study plan identity/digest is invalid")
    if arm not in plan.get("arms", {}):
        raise ValueError(
            f"arm {arm!r} is not present in this {plan.get('trial_mode', 'full')!r} plan"
        )
    if execute and plan.get("trial_mode") in (
        GPU_REPAIR_TRIAL_MODE,
        GPU_245X_REPAIR_TRIAL_MODE,
    ):
        prior = plan.get("prior_gpu_teardown") or {}
        teardown_file = (prior.get("files") or {}).get("teardown") or {}
        _validate_prior_gpu_teardown(
            Path(str(teardown_file.get("path", ""))), frozen=prior
        )
    if execute and plan.get("trial_mode") == CPU_REPAIR_TRIAL_MODE:
        prior = plan.get("prior_cpu_teardown") or {}
        teardown_file = (prior.get("files") or {}).get("teardown") or {}
        _validate_prior_cpu_teardown(
            Path(str(teardown_file.get("path", ""))), frozen=prior
        )
    if arm == "cpu" and plan.get("trial_mode", FULL_TRIAL_MODE) == FULL_TRIAL_MODE:
        gpu_teardown = artifact_root.parent / "gpu" / "teardown.json"
        if execute and (
            not gpu_teardown.is_file()
            or read_json(gpu_teardown).get("verified_empty") is not True
        ):
            raise RuntimeError(
                "CPU launch requires the GPU arm's verified-empty teardown receipt"
            )
    arguments = [
        *plan["arms"][arm]["lifecycle_args"],
        "--artifact-root",
        str(artifact_root),
    ]
    if execute:
        arguments.append("--execute")
    if confirm_plan_sha:
        arguments.extend(("--confirm-plan-sha", confirm_plan_sha))
    return lifecycle.main(arguments)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("prepare")
    create.add_argument("--campaign-id", required=True)
    create.add_argument("--local-config", type=Path, required=True)
    create.add_argument("--dataset-manifest", type=Path, required=True)
    create.add_argument("--output-root", type=Path, required=True)
    create.add_argument("--trial-mode", choices=TRIAL_MODES, default=FULL_TRIAL_MODE)
    create.add_argument("--prior-gpu-teardown", type=Path)
    create.add_argument("--prior-cpu-teardown", type=Path)
    run = sub.add_parser("execute-arm")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--arm", choices=("gpu", "cpu"), required=True)
    run.add_argument("--artifact-root", type=Path, required=True)
    run.add_argument("--execute", action="store_true")
    run.add_argument("--confirm-plan-sha")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(
            campaign=args.campaign_id,
            local_config=args.local_config.expanduser().resolve(),
            dataset_manifest=args.dataset_manifest.expanduser().resolve(),
            output_root=args.output_root.expanduser().resolve(),
            trial_mode=args.trial_mode,
            prior_gpu_teardown=args.prior_gpu_teardown,
            prior_cpu_teardown=args.prior_cpu_teardown,
        )
        return 0
    return execute_arm(
        args.plan.expanduser().resolve(),
        args.arm,
        args.artifact_root.expanduser().resolve(),
        execute=args.execute,
        confirm_plan_sha=args.confirm_plan_sha,
    )


if __name__ == "__main__":
    raise SystemExit(main())
