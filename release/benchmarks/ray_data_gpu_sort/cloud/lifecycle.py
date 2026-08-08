"""One launch, fresh Ray per trial, one teardown for a cloud arm."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

import yaml

from .common import (
    ARTIFACT_SENTINEL,
    atomic_json,
    canonical_bytes,
    digest,
    file_sha256,
    read_json,
)
from ..spec import RAY_COMMIT, RAY_VERSION, RAY_WHEEL_SHA256

SHAPE_MARKER = "__GPU_SORT_CLOUD_SHAPE__="
NONTERMINAL = ("pending", "running", "stopping", "stopped", "shutting-down")
WORKER_MODULE = "release.benchmarks.ray_data_gpu_sort.cloud.worker"
SAFE_STEP_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
SHA256 = re.compile(r"[0-9a-f]{64}")
IDENTITY_KEYS = {
    "trial",
    "bundle_digest",
    "input_plan_digest",
    "dataset_manifest_sha256",
    "wheel_sha256",
    "harness_files",
    "ray_data_overlay",
    "digest",
}


@dataclass(frozen=True)
class Step:
    name: str
    command: str
    restart_yaml: str | None = None
    allow_failure: bool = False
    timeout_seconds: float = 7200


def load_steps(path: Path) -> tuple[Step, ...]:
    value = read_json(path)
    if value.get("schema_version") != 1 or not isinstance(value.get("commands"), list):
        raise ValueError("invalid command plan")
    result = []
    for raw in value["commands"]:
        step = Step(**raw)
        if not step.name or not step.command or step.timeout_seconds <= 0:
            raise ValueError(f"invalid command step: {raw}")
        if step.restart_yaml is not None and not Path(step.restart_yaml).is_file():
            raise ValueError(f"missing reset YAML: {step.restart_yaml}")
        result.append(step)
    if len({step.name for step in result}) != len(result):
        raise ValueError("command names are not unique")
    return tuple(result)


def _ray_args(ray_cli: Path, operation: str) -> list[str]:
    value = [str(ray_cli), operation]
    if operation in ("up", "exec"):
        value.append("--no-config-cache")
    return [*value, "--log-style", "record", "--log-color", "false"]


def _exec_args(ray_cli: Path, cluster: Path, command: str) -> list[str]:
    return [*_ray_args(ray_cli, "exec"), "--run-env", "docker", str(cluster), command]


def _run(
    argv: Sequence[object], *, log: Path, timeout: float, check: bool = True
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [str(item) for item in argv],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        timeout=timeout,
        check=False,
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(result.stdout or "", encoding="utf-8")
    if check and result.returncode:
        raise RuntimeError(f"command exited {result.returncode}; see {log}")
    return result


def _single_option(arguments: Sequence[str], name: str, *, required: bool = True) -> str | None:
    positions = [index for index, value in enumerate(arguments) if value == name]
    if not positions:
        if required:
            raise ValueError(f"worker command is missing {name}")
        return None
    if len(positions) != 1 or positions[0] + 1 >= len(arguments):
        raise ValueError(f"worker command has an ambiguous {name}")
    value = arguments[positions[0] + 1]
    if value.startswith("--"):
        raise ValueError(f"worker command has no value for {name}")
    return value


def _worker_arguments(command: str) -> tuple[str, ...]:
    tokens = shlex.split(command)
    positions = [
        index
        for index in range(len(tokens) - 1)
        if tokens[index] == "-m" and tokens[index + 1] == WORKER_MODULE
    ]
    if len(positions) != 1:
        raise ValueError("step is not exactly one cloud worker command")
    arguments = tuple(tokens[positions[0] + 2 :])
    if any(value in {"&&", "||", ";", "|"} for value in arguments):
        raise ValueError("worker command contains trailing shell control")
    return arguments


def _remote_child(root: str, child: str) -> PurePosixPath:
    raw_root = PurePosixPath(root)
    raw_child = PurePosixPath(child)
    if not raw_root.is_absolute() or not raw_child.is_absolute():
        raise ValueError("remote result paths must be absolute")
    if ".." in raw_root.parts or ".." in raw_child.parts:
        raise ValueError("remote result paths may not traverse parents")
    try:
        raw_child.relative_to(raw_root)
    except ValueError as error:
        raise ValueError("worker output is outside the remote result root") from error
    return raw_child


def _decode_artifact_sentinel(stdout: str) -> dict[str, Any]:
    lines = [line for line in stdout.splitlines() if ARTIFACT_SENTINEL in line]
    if len(lines) != 1:
        raise ValueError(f"expected exactly one artifact sentinel, found {len(lines)}")
    payload = lines[0].split(ARTIFACT_SENTINEL, 1)[1].strip()
    try:
        encoded = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("artifact sentinel is not valid base64") from error
    try:
        envelope = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("artifact sentinel is not valid JSON") from error
    if not isinstance(envelope, dict) or canonical_bytes(envelope) != encoded:
        raise ValueError("artifact sentinel is not one canonical JSON object")
    if set(envelope) != {"schema_version", "output", "artifact_sha256", "artifact"}:
        raise ValueError("artifact sentinel envelope fields differ")
    artifact = envelope.get("artifact")
    if envelope.get("schema_version") != 1 or not isinstance(artifact, dict):
        raise ValueError("artifact sentinel envelope is invalid")
    if envelope.get("artifact_sha256") != digest(artifact):
        raise ValueError("artifact sentinel digest differs from its payload")
    return envelope


def _bundle_provenance(cluster_yaml: Path) -> dict[str, Any]:
    config = yaml.safe_load(cluster_yaml.read_text(encoding="utf-8"))
    mounts = config.get("file_mounts") if isinstance(config, dict) else None
    if not isinstance(mounts, dict):
        raise ValueError("cluster YAML has no file mounts")
    candidates = [Path(str(value)) / "BUNDLE.json" for value in mounts.values()]
    candidates = [path for path in candidates if path.is_file()]
    if len(candidates) != 1:
        raise ValueError("cluster YAML does not identify exactly one local bundle")
    bundle_path = candidates[0]
    bundle_root = bundle_path.parent
    bundle = read_json(bundle_path)
    files = bundle.get("files")
    actual_files = {
        str(path.relative_to(bundle_root)): file_sha256(path)
        for path in sorted(bundle_root.rglob("*"))
        if path.is_file()
        and path != bundle_path
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    }
    if (
        bundle.get("schema_version") != 1
        or bundle.get("kind") != "gpu_sort_cloud_bundle"
        or not isinstance(files, dict)
        or actual_files != files
        or bundle.get("bundle_digest") != digest(files)
        or bundle.get("ray_version") != RAY_VERSION
        or bundle.get("ray_commit") != RAY_COMMIT
        or bundle.get("wheel_sha256") != RAY_WHEEL_SHA256
        or any(
            not isinstance(name, str)
            or not isinstance(sha, str)
            or SHA256.fullmatch(sha) is None
            for name, sha in files.items()
        )
    ):
        raise ValueError("local benchmark bundle identity is invalid")
    wheel_hashes = [
        sha
        for name, sha in files.items()
        if name.startswith("wheelhouse/") and name.endswith(".whl")
    ]
    if wheel_hashes != [RAY_WHEEL_SHA256]:
        raise ValueError("local benchmark bundle has the wrong Ray wheel")
    harness_prefix = "release/benchmarks/ray_data_gpu_sort/cloud/"
    harness = {
        name: sha
        for name, sha in files.items()
        if name.startswith(harness_prefix)
        and name.endswith(".py")
        and "/" not in name[len(harness_prefix) :]
    }
    overlay = hashlib.sha256()
    overlay_files = sorted(
        (name, sha)
        for name, sha in files.items()
        if name.startswith("python/ray/data/")
        and name.endswith(".py")
        and "__pycache__" not in PurePosixPath(name).parts
    )
    if not harness or not overlay_files:
        raise ValueError("local benchmark bundle is missing code provenance")
    for name, sha in overlay_files:
        overlay.update(name.encode() + b"\0" + bytes.fromhex(sha))
    return {
        "bundle_digest": bundle["bundle_digest"],
        "wheel_sha256": RAY_WHEEL_SHA256,
        "harness_files": harness,
        "ray_data_overlay": {
            "file_count": len(overlay_files),
            "sha256": overlay.hexdigest(),
        },
        "ray_version": RAY_VERSION,
        "ray_commit": RAY_COMMIT,
    }


def _latest_admin_sync(prior_syncs: Sequence[Path]) -> tuple[Path, str]:
    for root in reversed(prior_syncs):
        inventory = root / "admin/inventory.json"
        staging = root / "admin/dataset-stage.json"
        if inventory.is_file() and staging.is_file():
            inventory_value = read_json(inventory)
            stage_value = read_json(staging)
            inventory_nodes = inventory_value.get("nodes")
            stage_nodes = stage_value.get("nodes")
            if (
                inventory_value.get("schema_version") != 1
                or inventory_value.get("kind") != "gpu_sort_cloud_inventory"
                or not isinstance(inventory_nodes, list)
                or len(inventory_nodes) != 16
                or stage_value.get("schema_version") != 1
                or stage_value.get("kind") != "gpu_sort_cloud_dataset_stage"
                or not isinstance(stage_nodes, list)
                or len(stage_nodes) != 16
                or stage_value.get("digest") != digest(stage_nodes)
            ):
                continue
            inventory_instance_values = [
                str(item.get("instance_id"))
                for item in inventory_nodes
                if isinstance(item, dict) and item.get("instance_id")
            ]
            stage_instance_values = [
                str(item.get("instance_id"))
                for item in stage_nodes
                if isinstance(item, dict) and item.get("instance_id")
            ]
            stage_manifest_values = [
                str(item.get("manifest_sha256"))
                for item in stage_nodes
                if isinstance(item, dict) and item.get("manifest_sha256")
            ]
            stage_dataset_values = [
                str(item.get("dataset_digest"))
                for item in stage_nodes
                if isinstance(item, dict) and item.get("dataset_digest")
            ]
            inventory_instances = set(inventory_instance_values)
            stage_instances = set(stage_instance_values)
            stage_manifests = set(stage_manifest_values)
            stage_datasets = set(stage_dataset_values)
            if (
                len(inventory_instance_values) != 16
                or len(inventory_instances) != 16
                or len(stage_instance_values) != 16
                or stage_instances != inventory_instances
                or len(stage_manifest_values) != 16
                or len(stage_manifests) != 1
                or SHA256.fullmatch(next(iter(stage_manifests))) is None
                or len(stage_dataset_values) != 16
                or stage_datasets != {str(stage_value.get("dataset_digest"))}
                or SHA256.fullmatch(next(iter(stage_datasets), "")) is None
            ):
                continue
            return root, next(iter(stage_manifests))
    raise ValueError("no prior successful sync contains required admin artifacts")


def _expected_trial(
    arguments: Sequence[str], *, remote_results: str, prior_sync: Path
) -> dict[str, Any]:
    expected: dict[str, Any] = {
        "kind": _single_option(arguments, "--kind"),
        "backend": _single_option(arguments, "--backend"),
        "cell": _single_option(arguments, "--cell"),
        "repetition": int(_single_option(arguments, "--repetition")),
        "scale_numerator": int(_single_option(arguments, "--scale-numerator")),
        "scale_denominator": int(_single_option(arguments, "--scale-denominator")),
    }
    selected = _single_option(arguments, "--selected-wave", required=False)
    explicit = _single_option(arguments, "--wave-fraction", required=False)
    if selected is None:
        expected["wave_fraction"] = 0.50 if explicit is None else float(explicit)
    else:
        selected_path = _remote_child(remote_results, selected)
        relative = selected_path.relative_to(PurePosixPath(remote_results))
        local_selected = prior_sync.joinpath(*relative.parts)
        selected_value = read_json(local_selected).get("selected_wave_fraction")
        if not isinstance(selected_value, (int, float)):
            raise ValueError("prior selected-wave artifact is invalid")
        expected["wave_fraction"] = float(selected_value)
    return expected


def _recover_artifact_from_stdout(
    *,
    step: Step,
    stdout: str,
    remote_results: str,
    local: Path,
    prior_successful_syncs: Sequence[Path],
    expected_provenance: dict[str, Any],
) -> dict[str, Any]:
    """Recover only a provenance-complete performance result after failed rsync."""
    if not step.allow_failure:
        raise ValueError("required steps may not recover artifacts from stdout")
    if SAFE_STEP_NAME.fullmatch(step.name) is None:
        raise ValueError("step name is not one safe path component")
    arguments = _worker_arguments(step.command)
    kind = _single_option(arguments, "--kind")
    backend = _single_option(arguments, "--backend")
    if kind not in ("trend", "natural") or backend not in ("gpu", "pyarrow"):
        raise ValueError("only performance steps may recover artifacts from stdout")
    output_value = _single_option(arguments, "--output")
    output = _remote_child(remote_results, output_value)
    expected_output = PurePosixPath(remote_results) / "results" / f"{step.name}.json"
    if output != expected_output:
        raise ValueError(f"worker output does not match the step name: {output}")
    prior_sync, dataset_manifest_sha256 = _latest_admin_sync(
        prior_successful_syncs
    )
    expected_trial = _expected_trial(
        arguments, remote_results=remote_results, prior_sync=prior_sync
    )
    envelope = _decode_artifact_sentinel(stdout)
    if envelope["output"] != output_value:
        raise ValueError("sentinel output differs from the worker command")
    artifact = envelope["artifact"]
    valid = artifact.get("valid")
    if not isinstance(valid, bool) or artifact.get("status") != (
        "accepted" if valid else "rejected"
    ):
        raise ValueError("recovered artifact has inconsistent status")
    identity = artifact.get("artifact_identity")
    if not isinstance(identity, dict) or set(identity) != IDENTITY_KEYS:
        raise ValueError("recovered artifact identity schema differs")
    identity_body = {key: value for key, value in identity.items() if key != "digest"}
    if identity.get("digest") != digest(identity_body):
        raise ValueError("recovered artifact identity digest differs")
    if identity.get("trial") != expected_trial:
        raise ValueError("recovered artifact identity differs from the step command")
    plan = artifact.get("plan")
    if (
        not isinstance(plan, dict)
        or set(plan) != {"kind", "blocks", "rows", "digest"}
        or plan.get("kind") != kind
        or not isinstance(plan.get("blocks"), int)
        or not isinstance(plan.get("rows"), int)
        or int(plan["blocks"]) < 1
        or int(plan["rows"]) < 1
        or SHA256.fullmatch(str(plan.get("digest", ""))) is None
        or identity.get("input_plan_digest") != plan.get("digest")
    ):
        raise ValueError("recovered artifact input-plan identity differs")
    if identity.get("dataset_manifest_sha256") != dataset_manifest_sha256:
        raise ValueError("recovered artifact dataset identity differs")
    if identity.get("bundle_digest") != expected_provenance.get("bundle_digest"):
        raise ValueError("recovered artifact bundle digest differs")
    for field in ("wheel_sha256", "harness_files", "ray_data_overlay"):
        if identity.get(field) != expected_provenance.get(field):
            raise ValueError(f"recovered artifact {field} differs from the bundle")
    if artifact.get("ray_version") != expected_provenance.get("ray_version"):
        raise ValueError("recovered artifact Ray version differs from the bundle")
    if artifact.get("ray_commit") != expected_provenance.get("ray_commit"):
        raise ValueError("recovered artifact Ray commit differs from the bundle")
    artifact_cell = artifact.get("cell")
    artifact_cell = (
        artifact_cell.get("name")
        if isinstance(artifact_cell, dict)
        else artifact_cell
    )
    top_level_trial = {
        "kind": artifact.get("kind"),
        "backend": artifact.get("backend"),
        "cell": artifact_cell,
        "repetition": artifact.get("repetition"),
        "scale_numerator": artifact.get("scale_numerator"),
        "scale_denominator": artifact.get("scale_denominator"),
        "wave_fraction": artifact.get("wave_fraction"),
    }
    if top_level_trial != expected_trial:
        raise ValueError("recovered artifact fields differ from the step command")
    destination = local / "results" / output.name
    atomic_json(destination, artifact)
    return {
        "artifact_recovered_from_stdout": True,
        "recovered_artifact": str(destination),
        "recovered_artifact_sha256": envelope["artifact_sha256"],
    }


def _identity(path: Path) -> tuple[str, str]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return str(value["cluster_name"]), str(value["provider"]["region"])


def _shape_command(remote_python: str) -> str:
    source = f"""import json,ray
ray.init(address='auto',logging_level='ERROR')
nodes=[n for n in ray.nodes() if n.get('Alive')]
r=ray.cluster_resources()
print({SHAPE_MARKER!r}+json.dumps({{'nodes':len(nodes),'cpus':float(r.get('CPU',0)),'gpus':float(r.get('GPU',0)),'ray_node_ids':sorted(str(n.get('NodeID')) for n in nodes)}},sort_keys=True),flush=True)
ray.shutdown()"""
    encoded = base64.b64encode(source.encode()).decode()
    return f"{shlex.quote(remote_python)} -c \"import base64;exec(base64.b64decode('{encoded}'))\""


def _wait_shape(
    *,
    ray_cli: Path,
    cluster: Path,
    remote_python: str,
    nodes: int,
    cpus: float,
    gpus: float,
    artifact: Path,
    previous_ids: Sequence[str] = (),
    timeout: float = 1800,
) -> tuple[str, ...]:
    deadline = time.monotonic() + timeout
    previous = set(previous_ids)
    attempt = 0
    last = None
    while time.monotonic() < deadline:
        attempt += 1
        result = _run(
            _exec_args(ray_cli, cluster, _shape_command(remote_python)),
            log=artifact.with_name(f"{artifact.stem}-{attempt:03d}.log"),
            timeout=180,
            check=False,
        )
        for line in reversed((result.stdout or "").splitlines()):
            if SHAPE_MARKER in line:
                try:
                    last = json.loads(line.split(SHAPE_MARKER, 1)[1])
                except json.JSONDecodeError:
                    pass
                break
        if result.returncode == 0 and isinstance(last, dict):
            ids = tuple(str(value) for value in last.get("ray_node_ids", ()))
            if (
                int(last.get("nodes", -1)) == nodes
                and float(last.get("cpus", -1)) == cpus
                and float(last.get("gpus", -1)) == gpus
                and len(ids) == nodes
                and len(set(ids)) == nodes
                and (not previous or previous.isdisjoint(ids))
            ):
                atomic_json(artifact, {"observed": last, "attempts": attempt})
                return ids
        time.sleep(5)
    raise RuntimeError(f"Ray did not reach the exact fresh shape; last={last}")


def _scoped_instance_records(client: Any, cluster_name: str) -> tuple[dict[str, Any], ...]:
    response = client.describe_instances(
        Filters=[
            {"Name": "tag:ray-cluster-name", "Values": [cluster_name]},
            {"Name": "instance-state-name", "Values": list(NONTERMINAL)},
        ]
    )
    return tuple(sorted(
        (dict(instance)
        for reservation in response.get("Reservations", ())
        for instance in reservation.get("Instances", ())),
        key=lambda item: str(item["InstanceId"]),
    ))


def _scoped_instances(client: Any, cluster_name: str) -> tuple[str, ...]:
    return tuple(
        str(instance["InstanceId"])
        for instance in _scoped_instance_records(client, cluster_name)
    )


def _instance_ids(
    client: Any,
    cluster_name: str,
    expected: int,
    *,
    instance_type: str,
    single_availability_zone: bool,
) -> tuple[str, ...]:
    records = _scoped_instance_records(client, cluster_name)
    ids = tuple(str(item["InstanceId"]) for item in records)
    if len(ids) != expected:
        raise RuntimeError(f"expected {expected} scoped EC2 instances, found {ids}")
    if {str(item.get("InstanceType")) for item in records} != {instance_type}:
        raise RuntimeError("EC2 instance types differ from the frozen topology")
    if {str((item.get("State") or {}).get("Name")) for item in records} != {"running"}:
        raise RuntimeError("not every benchmark EC2 instance is running")
    zones = {str((item.get("Placement") or {}).get("AvailabilityZone")) for item in records}
    if single_availability_zone and len(zones) != 1:
        raise RuntimeError(f"benchmark fleet spans availability zones: {sorted(zones)}")
    return ids


def _terminate(client: Any, cluster_name: str, timeout: float = 900) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    terminated: set[str] = set()
    empty = 0
    while time.monotonic() < deadline:
        ids = _scoped_instances(client, cluster_name)
        if ids:
            empty = 0
            new = sorted(set(ids) - terminated)
            if new:
                client.terminate_instances(InstanceIds=new)
                terminated.update(new)
        else:
            empty += 1
            if empty >= 3:
                return {"verified_empty": True, "terminated_instance_ids": sorted(terminated)}
        time.sleep(5)
    raise RuntimeError("exact-tag EC2 teardown did not become stably empty")


def execute(args: argparse.Namespace) -> int:
    import boto3

    steps = load_steps(args.command_plan)
    cluster_name, yaml_region = _identity(args.cluster_yaml)
    expected_provenance = _bundle_provenance(args.cluster_yaml)
    if expected_provenance["bundle_digest"] != args.bundle_digest:
        raise ValueError("mounted bundle differs from the prepared study bundle")
    if cluster_name != args.cluster_name or yaml_region != args.region:
        raise ValueError("cluster identity differs from lifecycle arguments")
    plan_body = {
        "schema_version": 1,
        "campaign": args.campaign,
        "arm": args.arm,
        "cluster_name": args.cluster_name,
        "cluster_yaml_sha256": file_sha256(args.cluster_yaml),
        "command_plan_sha256": file_sha256(args.command_plan),
        "steps": [asdict(step) for step in steps],
        "expected": {
            "nodes": args.nodes,
            "cpus": args.cpus,
            "gpus": args.gpus,
            "instance_type": args.instance_type,
            "single_availability_zone": args.single_availability_zone,
        },
        "same_ec2_ids": True,
        "fresh_ray_node_ids_each_trial": True,
        "ray_default_plasma": True,
        "artifact_provenance": expected_provenance,
    }
    plan_sha = hashlib.sha256(
        json.dumps(plan_body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    atomic_json(args.artifact_root / "lifecycle-plan.json", {**plan_body, "plan_sha256": plan_sha})
    if not args.execute:
        print(json.dumps({"dry_run": True, "plan_sha256": plan_sha}, indent=2))
        return 0
    if args.confirm_plan_sha != plan_sha:
        raise RuntimeError(f"billable execution requires --confirm-plan-sha {plan_sha}")
    session = boto3.Session(profile_name=args.aws_profile, region_name=args.region)
    client = session.client("ec2")
    if _scoped_instances(client, args.cluster_name):
        raise RuntimeError("refusing launch into a non-empty exact resource scope")
    failure: BaseException | None = None
    teardown: dict[str, Any] | None = None
    results = []
    successful_syncs: list[Path] = []
    try:
        _run(
                [*_ray_args(args.ray_cli, "up"), "-y", args.cluster_yaml],
                log=args.artifact_root / "ray-up.log",
                timeout=1800,
        )
        ray_ids = _wait_shape(
                ray_cli=args.ray_cli,
                cluster=args.cluster_yaml,
                remote_python=args.remote_python,
                nodes=args.nodes,
                cpus=args.cpus,
                gpus=args.gpus,
                artifact=args.artifact_root / "shape-initial.json",
        )
        ec2_ids = _instance_ids(
            client,
            args.cluster_name,
            args.nodes,
            instance_type=args.instance_type,
            single_availability_zone=args.single_availability_zone,
        )
        atomic_json(args.artifact_root / "instances.json", {"instance_ids": ec2_ids})
        for ordinal, step in enumerate(steps, 1):
            cluster = Path(step.restart_yaml) if step.restart_yaml else args.cluster_yaml
            if step.restart_yaml:
                    before = _instance_ids(
                        client,
                        args.cluster_name,
                        args.nodes,
                        instance_type=args.instance_type,
                        single_availability_zone=args.single_availability_zone,
                    )
                    _run(
                        [*_ray_args(args.ray_cli, "up"), "-y", "--restart-only", cluster],
                        log=args.artifact_root / f"restart-{ordinal:02d}-{step.name}.log",
                        timeout=1800,
                    )
                    ray_ids = _wait_shape(
                        ray_cli=args.ray_cli,
                        cluster=cluster,
                        remote_python=args.remote_python,
                        nodes=args.nodes,
                        cpus=args.cpus,
                        gpus=args.gpus,
                        previous_ids=ray_ids,
                        artifact=args.artifact_root / f"shape-{ordinal:02d}-{step.name}.json",
                    )
                    after = _instance_ids(
                        client,
                        args.cluster_name,
                        args.nodes,
                        instance_type=args.instance_type,
                        single_availability_zone=args.single_availability_zone,
                    )
                    if before != ec2_ids or after != ec2_ids:
                        raise RuntimeError("EC2 identities changed during fresh-Ray restart")
            completed = _run(
                    _exec_args(args.ray_cli, cluster, step.command),
                    log=args.artifact_root / f"command-{ordinal:02d}-{step.name}.log",
                    timeout=step.timeout_seconds,
                    check=False,
                )
            local = args.artifact_root / "remote-results" / f"{ordinal:02d}-{step.name}"
            local.mkdir(parents=True, exist_ok=True)
            sync = _run(
                    [
                        *_ray_args(args.ray_cli, "rsync-down"),
                        str(cluster),
                        args.remote_results.rstrip("/") + "/",
                        str(local) + "/",
                    ],
                    log=args.artifact_root / f"rsync-{ordinal:02d}-{step.name}.log",
                    timeout=900,
                    check=False,
                )
            recovery: dict[str, Any] = {"artifact_recovered_from_stdout": False}
            recovery_error: BaseException | None = None
            if sync.returncode:
                try:
                    recovery = _recover_artifact_from_stdout(
                        step=step,
                        stdout=completed.stdout or "",
                        remote_results=args.remote_results,
                        local=local,
                        prior_successful_syncs=successful_syncs,
                        expected_provenance=expected_provenance,
                    )
                except BaseException as error:
                    recovery_error = error
            else:
                successful_syncs.append(local)
            record = {
                    "name": step.name,
                    "returncode": completed.returncode,
                    "rsync_returncode": sync.returncode,
                    "allow_failure": step.allow_failure,
                    **recovery,
                }
            if recovery_error is not None:
                record["artifact_recovery_error"] = (
                    f"{type(recovery_error).__name__}: {recovery_error}"
                )
            results.append(record)
            atomic_json(args.artifact_root / f"step-{ordinal:02d}.json", record)
            if sync.returncode and not recovery["artifact_recovered_from_stdout"]:
                raise RuntimeError(
                    f"artifact sync failed after {step.name}; stdout recovery failed: "
                    f"{recovery_error}"
                )
            if completed.returncode and not step.allow_failure:
                raise RuntimeError(f"required step failed: {step.name}")
    except BaseException as error:
        failure = error
        atomic_json(
            args.artifact_root / "failure.json",
            {"type": type(error).__name__, "message": str(error)},
        )
    finally:
        try:
            _run(
                [*_ray_args(args.ray_cli, "down"), "-y", args.cluster_yaml],
                log=args.artifact_root / "ray-down.log",
                timeout=300,
                check=False,
            )
        finally:
            try:
                terminated = _terminate(client, args.cluster_name)
                instance_ids = sorted(terminated.get("terminated_instance_ids", ()))
                teardown = {
                    **terminated,
                    "schema_version": 2,
                    "campaign": args.campaign,
                    "arm": args.arm,
                    "cluster_name": args.cluster_name,
                    "lifecycle_plan_sha256": plan_sha,
                    "instance_ids_sha256": digest(instance_ids),
                }
                atomic_json(args.artifact_root / "teardown.json", teardown)
            except BaseException as error:
                if failure is None:
                    failure = error
    atomic_json(
        args.artifact_root / "result.json",
        {
            "status": "success" if failure is None else "failed",
            "steps": results,
            "teardown": teardown,
        },
    )
    if failure is not None:
        raise failure
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--campaign", required=True)
    value.add_argument("--arm", choices=("gpu", "cpu"), required=True)
    value.add_argument("--cluster-name", required=True)
    value.add_argument("--cluster-yaml", type=Path, required=True)
    value.add_argument("--command-plan", type=Path, required=True)
    value.add_argument("--artifact-root", type=Path, required=True)
    value.add_argument("--remote-results", required=True)
    value.add_argument("--bundle-digest", required=True)
    value.add_argument("--remote-python", required=True)
    value.add_argument("--region", required=True)
    value.add_argument("--aws-profile")
    value.add_argument("--nodes", type=int, required=True)
    value.add_argument("--cpus", type=float, required=True)
    value.add_argument("--gpus", type=float, required=True)
    value.add_argument("--instance-type", required=True)
    value.add_argument("--single-availability-zone", action="store_true")
    value.add_argument("--ray-cli", type=Path, default=Path(sys.executable).with_name("ray"))
    value.add_argument("--execute", action="store_true")
    value.add_argument("--confirm-plan-sha")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    for name in ("cluster_yaml", "command_plan", "artifact_root", "ray_cli"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.artifact_root.mkdir(parents=True, exist_ok=False)
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
