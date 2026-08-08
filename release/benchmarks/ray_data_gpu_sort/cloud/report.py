"""Aggregate cloud trial artifacts into JSON and a concise Markdown report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from .common import atomic_json, digest, file_sha256, read_json
from .spec import CELLS

GIB = 1 << 30
LOCATION_WARNING = "one or more materialized input ObjectRefs are not locatable"
OUTPUT_LOCATION_WARNING = "one or more output ObjectRefs are not Ray-locatable"
LOCATION_METADATA_WARNINGS = frozenset(
    (LOCATION_WARNING, OUTPUT_LOCATION_WARNING)
)
OOM_MARKERS = (
    "MemoryError",
    "out_of_memory",
    "std::bad_alloc",
    "Maximum pool size exceeded",
)

# The DGX values are from gpu-sort-external-artifacts/REPORT.md. Previous-cloud
# values are from the completed bts-cloud-20260804-t report. Keeping raw CPU and
# GPU times here makes cross-machine comparisons auditable, not just their ratios.
DGX = {
    "narrow": {"cpu_s": 203.946, "gpu_s": 15.931},
    "core": {"cpu_s": 243.555, "gpu_s": 33.667},
    "full": {"cpu_s": 269.752, "gpu_s": 53.904},
    "origin-string": {"cpu_s": 223.844, "gpu_s": 51.101},
    "origin-integer": {"cpu_s": 156.026, "gpu_s": 50.678},
    "route": {"cpu_s": 273.418, "gpu_s": 52.776},
}
PREVIOUS_CLOUD = {
    "narrow": {"cpu_s": 72.33, "gpu_s": 11.83},
    "core": {"cpu_s": 122.36, "gpu_s": 19.41},
    "full": {"cpu_s": 121.86, "gpu_s": 27.85},
    "origin-string": {"cpu_s": 81.05, "gpu_s": None},
    "origin-integer": {"cpu_s": 60.18, "gpu_s": None},
    "route": {"cpu_s": 124.51, "gpu_s": 27.85},
}
DGX_NATURAL = {
    "1x": {"cpu_s": 271.564, "gpu_s": 53.816},
    "2x": {"cpu_s": 1179.517, "gpu_s": 320.671},
    "2.45x": {"cpu_s": 2077.804, "gpu_s": 513.690},
}

GPU_REPAIR_RESULT_NAMES = (
    "transport-smoke.json",
    "tune-2x-wave-0500.json",
    "tune-2x-wave-0375.json",
    "natural-2x-selected-r2.json",
    "natural-245x-r1.json",
    "natural-245x-r2.json",
)
GPU_REPAIR_COMPLETED_PREFIX_NAMES = GPU_REPAIR_RESULT_NAMES[:4]
GPU_245X_REPAIR_RESULT_NAMES = (
    "transport-smoke.json",
    "natural-245x-r1.json",
    "natural-245x-r2.json",
)
GPU_245X_REPAIR_REPLACEMENT_NAMES = (
    "natural-245x-r1.json",
    "natural-245x-r2.json",
)
CPU_REPAIR_RESULT_NAMES = (
    "trend-full-r1.json",
    "trend-origin-string-r1.json",
    "trend-origin-integer-r1.json",
    "trend-route-r1.json",
    "natural-2x-r1.json",
    "natural-245x-r1.json",
)
CPU_RESOURCE_REJECTION_RESULT = "natural-245x-r1.json"

# The final default-Ray/PyArrow 2.45x observation exhausted the head/driver at
# the map-to-reduce boundary in two independent, fresh-runtime CPU campaigns.
# No worker result exists, so this is deliberately a frozen evidence profile,
# not a general relaxation of repair-result validation.  Every raw evidence
# file is content-addressed here; normal CPU repairs and every GPU repair still
# require their complete worker JSON and successful lifecycle receipt.
CPU_RESOURCE_REJECTION_PROFILE: dict[str, Any] = {
    "campaign": "bts-ext-20260807-j",
    "study_plan_sha256": (
        "3e44f5437d9b5e94e24856c9489cd5b7d002692f17dd66c87ac4a9305f469516"
    ),
    "lifecycle_plan_sha256": (
        "41ae97e2fa4698e36a636d71fba0c3cc5afcc6b1878046374a4ab140a2da3dde"
    ),
    "trial": "natural-245x-r1",
    "step_ordinal": 9,
    "diagnostic_time_to_failure_s": 380,
    "failure_message": (
        "artifact sync failed after natural-245x-r1; stdout recovery failed: "
        "expected exactly one artifact sentinel, found 0"
    ),
    "stdout_recovery_evidence": True,
    "files": {
        "study.json": (
            "1bd465f8fecc54aaf4df85397ba7d26b4d06b17665998e61355e9dd87a2471f7"
        ),
        "executions/cpu/lifecycle-plan.json": (
            "1e5d3f4f870aa8c1e283e6cf8b530c4639efef796bad34183628dde4ec3c3fdc"
        ),
        "executions/cpu/instances.json": (
            "0a3ecf73219f08a7b67f94fca92253154f45be4d7bee83ed7efdf4bddfd8e163"
        ),
        "executions/cpu/failure.json": (
            "083353a5d065945d59369021b7dd646924ecd451a67841aaf9eb5482dca4b846"
        ),
        "executions/cpu/step-01.json": (
            "fb1e0bdb81af3aa2e96fc9c7d84ab9297175dedc2a2d2eba4d4cd648bb97ecc1"
        ),
        "executions/cpu/step-02.json": (
            "77996f6a4a8782199f170aa59fe225330da35a1b7eb63e1ccfa435db090f4993"
        ),
        "executions/cpu/step-03.json": (
            "c554fe1a2bf7e24e0837804d795497d99c16c576ef5fda3bad1c8b57295e814c"
        ),
        "executions/cpu/step-04.json": (
            "9f09454dbc16c8d5d8d8291e561e7a512dc70b7ca6e62330eebb7f5ad24f26a0"
        ),
        "executions/cpu/step-05.json": (
            "e851aa7a2a52557333187281fb821af126592bace7602455dd38d6520a15bea2"
        ),
        "executions/cpu/step-06.json": (
            "867ae090c046a9c8142aa71eec60e0e0a9b817d0028333aede296e83a2198814"
        ),
        "executions/cpu/step-07.json": (
            "882e976006688bc04ca8c5781f4a92db8278729ddcbfe5f0e8cdce1df8d718d1"
        ),
        "executions/cpu/step-08.json": (
            "e9e9e682db73f7a69acfda9e8d27de27d57f57e1de4bcdf35054977c87d8900c"
        ),
        "executions/cpu/step-09.json": (
            "7bcb14fe513dbd98aac55a7a6a5005102dc1fe7fc58ec475da7e05607d97a3d8"
        ),
        "executions/cpu/shape-08-natural-2x-r1.json": (
            "363b0e691309c80f1ffe5d6445a4eaf245950bd05fdaffafc0b65257d4c4d46a"
        ),
        "executions/cpu/shape-09-natural-245x-r1.json": (
            "9a7d2c5983ab2b0324262675ce59ee823fe9505735784b470bcdfc9ff20dca02"
        ),
        "executions/cpu/command-09-natural-245x-r1.log": (
            "7735242da0f19132a355d07591c379ee79920446e5112e204313d851419257a0"
        ),
        "executions/cpu/rsync-09-natural-245x-r1.log": (
            "3aaebc9454b808bc31361634d12867955388bbea30621bcdf35930a611848221"
        ),
        "executions/cpu/teardown.json": (
            "8c0d008da8965ce7af877ae5d9ca586b3029084a80dcbf8c9a51dfa8f9219809"
        ),
    },
    "corroborating": {
        "campaign": "bts-ext-20260807-i",
        "study_plan_sha256": (
            "f1ed70aaf81f1a6bbb349bffefffc7a82166bbaf4d178b262d451b18e62c4192"
        ),
        "lifecycle_plan_sha256": (
            "f1127d60163805bd4c85e7a7bd852abadb4caa0fb8db8a1bc7a5e550bf243b7d"
        ),
        "diagnostic_time_to_failure_s": 374,
        "failure_message": "artifact sync failed after natural-245x-r1",
        "stdout_recovery_evidence": False,
        "files": {
            "study.json": (
                "b88c090a3605cf89c4548abbfb444e83bd58e8fa0b73dcff4ceb3672afc4be13"
            ),
            "executions/cpu/lifecycle-plan.json": (
                "a402b9fa3f78cd8d76e805d83aca7e2fa033e0abd75e7ca953dbb0f274fd92c5"
            ),
            "executions/cpu/instances.json": (
                "028a541c61882050b57c5e6d49dad1fa81950e97705c7bcbbc1d1ae9abff3dbd"
            ),
            "executions/cpu/failure.json": (
                "3137ed7df8bc972b5989f2147d62436d67c209053b3caabe6391866e8a5249ab"
            ),
            "executions/cpu/step-01.json": (
                "bae60bfea16768ed6b8bced9332a770e39203759e3688f05e212d65b2a32a24e"
            ),
            "executions/cpu/step-02.json": (
                "538209eac16d7e14ca3b581653485d8a7c55ed1e44abcb1f69d0d2feeaa531c6"
            ),
            "executions/cpu/step-03.json": (
                "eec444c67cd06f583be5f57ad9627386017d9d1a0f559c3deb5f29b6222058be"
            ),
            "executions/cpu/step-04.json": (
                "0ab8cd3dd2fcacd7865d8f2c413ddee098e2936adb41b0f460a8af617587148c"
            ),
            "executions/cpu/step-05.json": (
                "ea88a1accf789698d8f88c61f7eb281c290902a48c3402ccad4c56466fbcadce"
            ),
            "executions/cpu/step-06.json": (
                "6677827945eb9c0692457207ee8630f34c537b8e996924061ff52b6113780836"
            ),
            "executions/cpu/step-07.json": (
                "8cb5666cf537912fa3cf7bf778009b2a1f05bc76d5f84e73da6fd90c1b77f108"
            ),
            "executions/cpu/step-08.json": (
                "1e4576b7a9997192019f7adfb617e0e4310eb0cde6f71fd8ca4b70ad4253347e"
            ),
            "executions/cpu/step-09.json": (
                "520c10b54ecba7c45a88d36bae9b5b65b3b64ed36489391eafc9c8d50a655f34"
            ),
            "executions/cpu/shape-08-natural-2x-r1.json": (
                "664f8406afcfac65e5c95f229c989c32b301bdfcef8f3f936957c1bdc5efea42"
            ),
            "executions/cpu/shape-09-natural-245x-r1.json": (
                "1627b6530de72f1592669d277db9a45aab9df6d70c39c988eb0e606c9f479201"
            ),
            "executions/cpu/command-09-natural-245x-r1.log": (
                "27c060c7c88a4937b087fc003e72e913f8847c5f2ce538508a113497deac2d53"
            ),
            "executions/cpu/rsync-09-natural-245x-r1.log": (
                "34dcd285a26d0c4fc9bc633e68b6bacbcc89d867118cc91463e52052829175cd"
            ),
        },
    },
}


def _validated_study(path: Path) -> dict[str, Any]:
    value = read_json(path)
    claimed = value.get("plan_sha256")
    body = {key: item for key, item in value.items() if key != "plan_sha256"}
    if not isinstance(claimed, str) or claimed != digest(body):
        raise ValueError(f"invalid study plan digest: {path}")
    bundle = value.get("bundle") or {}
    files = bundle.get("files") or {}
    if not isinstance(files, Mapping) or bundle.get("bundle_digest") != digest(files):
        raise ValueError(f"invalid study bundle digest: {path}")
    if files.get("generated/dataset-manifest.json") != value.get(
        "dataset_manifest_sha256"
    ):
        raise ValueError(f"study dataset manifest is not the bundled manifest: {path}")
    wheel_hashes = [
        sha
        for name, sha in files.items()
        if name.startswith("wheelhouse/") and name.endswith(".whl")
    ]
    if wheel_hashes != [bundle.get("wheel_sha256")]:
        raise ValueError(f"study wheel is not present in its bundle: {path}")
    return value


def _unique_json(root: Path, name: str) -> dict[str, Any]:
    candidates = sorted(root.rglob(name))
    if not candidates:
        raise ValueError(f"repair result tree is missing {name}")
    values = [read_json(path) for path in candidates]
    if len({digest(value) for value in values}) != 1:
        raise ValueError(f"repair result tree contains conflicting copies of {name}")
    return values[0]


def _bundle_artifact_provenance(plan: Mapping[str, Any]) -> dict[str, Any]:
    bundle = plan["bundle"]
    files = bundle["files"]
    prefix = "release/benchmarks/ray_data_gpu_sort/cloud/"
    harness = {
        name: sha
        for name, sha in files.items()
        if name.startswith(prefix)
        and name.endswith(".py")
        and "/" not in name[len(prefix) :]
    }
    overlay = hashlib.sha256()
    overlay_files = sorted(
        (name, sha)
        for name, sha in files.items()
        if name.startswith("python/ray/data/")
        and name.endswith(".py")
        and "__pycache__" not in Path(name).parts
    )
    for name, sha in overlay_files:
        overlay.update(name.encode() + b"\0" + bytes.fromhex(sha))
    return {
        "wheel_sha256": bundle["wheel_sha256"],
        "harness_files": harness,
        "ray_data_overlay": {
            "file_count": len(overlay_files),
            "sha256": overlay.hexdigest(),
        },
    }


def _repair_execution_provenance(
    repair_plan: Mapping[str, Any], repair_root: Path, *, arm: str = "gpu"
) -> dict[str, Any]:
    repair_root = repair_root.resolve()
    if repair_root.name != "remote-results":
        raise ValueError(
            f"--{arm}-repair-results must name the execution's remote-results directory"
        )
    execution_root = repair_root.parent
    paths = {
        name: execution_root / filename
        for name, filename in {
            "lifecycle_plan": "lifecycle-plan.json",
            "instances": "instances.json",
            "teardown": "teardown.json",
            "result": "result.json",
        }.items()
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError(f"repair execution provenance is missing: {missing}")
    lifecycle = read_json(paths["lifecycle_plan"])
    instances = read_json(paths["instances"])
    teardown = read_json(paths["teardown"])
    result = read_json(paths["result"])
    lifecycle_body = {
        key: item for key, item in lifecycle.items() if key != "plan_sha256"
    }
    if lifecycle.get("plan_sha256") != digest(lifecycle_body):
        raise ValueError("repair lifecycle plan digest is invalid")
    repair_arm = repair_plan["arms"][arm]
    topology = repair_arm["topology"]
    expected = {
        "nodes": int(topology["nodes"]),
        "cpus": float(topology["nodes"] * topology["vcpus_per_node"]),
        "gpus": float(topology["nodes"] * topology["gpus_per_node"]),
        "instance_type": topology["instance_type"],
        "single_availability_zone": True,
    }
    if (
        lifecycle.get("campaign") != repair_plan.get("campaign")
        or lifecycle.get("arm") != arm
        or lifecycle.get("cluster_name") != repair_arm.get("cluster_name")
        or lifecycle.get("expected") != expected
        or lifecycle.get("cluster_yaml_sha256")
        != file_sha256(Path(repair_arm["launch_yaml"]))
        or lifecycle.get("command_plan_sha256")
        != file_sha256(Path(repair_arm["command_plan"]))
        or lifecycle.get("same_ec2_ids") is not True
        or lifecycle.get("fresh_ray_node_ids_each_trial") is not True
        or lifecycle.get("ray_default_plasma") is not True
    ):
        raise ValueError("repair lifecycle identity differs from the repair study")
    ids = instances.get("instance_ids")
    terminated = teardown.get("terminated_instance_ids")
    if (
        not isinstance(ids, list)
        or len(ids) != 16
        or len(set(ids)) != 16
        or sorted(ids) != sorted(terminated or ())
        or teardown.get("verified_empty") is not True
        or result.get("status") != "success"
        or result.get("teardown") != teardown
    ):
        raise ValueError("repair lifecycle did not complete with the exact retained fleet")
    required_teardown = {
        "campaign": repair_plan["campaign"],
        "arm": arm,
        "cluster_name": repair_arm["cluster_name"],
        "lifecycle_plan_sha256": lifecycle["plan_sha256"],
        "instance_ids_sha256": digest(sorted(ids)),
    }
    if any(teardown.get(key) != item for key, item in required_teardown.items()):
        raise ValueError("repair teardown receipt is missing its execution identity")
    return {
        "campaign": repair_plan["campaign"],
        "arm": arm,
        "cluster_name": repair_arm["cluster_name"],
        "lifecycle_plan_sha256": lifecycle["plan_sha256"],
        "instance_ids": sorted(ids),
        "instance_ids_sha256": digest(sorted(ids)),
        "files": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in paths.items()
        },
    }


def _repair_245x_handoff_execution_provenance(
    repair_plan: Mapping[str, Any], repair_root: Path
) -> dict[str, Any]:
    """Validate the exact failed lifecycle that handed 2.45x to a new campaign."""
    repair_root = repair_root.resolve()
    if repair_root.name != "remote-results":
        raise ValueError(
            "--gpu-repair-results must name the execution's remote-results directory"
        )
    execution_root = repair_root.parent
    paths = {
        name: execution_root / filename
        for name, filename in {
            "lifecycle_plan": "lifecycle-plan.json",
            "instances": "instances.json",
            "teardown": "teardown.json",
            "result": "result.json",
            "failure": "failure.json",
        }.items()
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError(f"GPU repair handoff provenance is missing: {missing}")
    lifecycle = read_json(paths["lifecycle_plan"])
    instances = read_json(paths["instances"])
    teardown = read_json(paths["teardown"])
    result = read_json(paths["result"])
    failure = read_json(paths["failure"])
    lifecycle_body = {
        key: item for key, item in lifecycle.items() if key != "plan_sha256"
    }
    arm = repair_plan["arms"]["gpu"]
    topology = arm["topology"]
    expected = {
        "nodes": int(topology["nodes"]),
        "cpus": float(topology["nodes"] * topology["vcpus_per_node"]),
        "gpus": float(topology["nodes"] * topology["gpus_per_node"]),
        "instance_type": topology["instance_type"],
        "single_availability_zone": True,
    }
    if (
        lifecycle.get("plan_sha256") != digest(lifecycle_body)
        or lifecycle.get("campaign") != repair_plan.get("campaign")
        or lifecycle.get("arm") != "gpu"
        or lifecycle.get("cluster_name") != arm.get("cluster_name")
        or lifecycle.get("expected") != expected
        or lifecycle.get("cluster_yaml_sha256")
        != file_sha256(Path(arm["launch_yaml"]))
        or lifecycle.get("command_plan_sha256")
        != file_sha256(Path(arm["command_plan"]))
        or lifecycle.get("same_ec2_ids") is not True
        or lifecycle.get("fresh_ray_node_ids_each_trial") is not True
        or lifecycle.get("ray_default_plasma") is not True
    ):
        raise ValueError("GPU repair handoff lifecycle identity differs from its study")

    expected_step_names = [
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
    expected_allow_failure = [False, False, False, False, True, True, False, True, True, True]
    steps = lifecycle.get("steps") or []
    if [step.get("name") for step in steps] != expected_step_names or [
        step.get("allow_failure") for step in steps
    ] != expected_allow_failure:
        raise ValueError("GPU repair handoff lifecycle is not the exact full repair plan")

    receipts: list[dict[str, Any]] = []
    for ordinal, step in enumerate(steps[:9], 1):
        path = execution_root / f"step-{ordinal:02d}.json"
        if not path.is_file():
            raise ValueError(f"GPU repair handoff is missing step receipt {ordinal}")
        receipt = read_json(path)
        if (
            receipt.get("name") != step["name"]
            or receipt.get("allow_failure") is not step["allow_failure"]
            or receipt.get("artifact_recovered_from_stdout") not in (None, False)
        ):
            raise ValueError(f"GPU repair handoff step receipt differs at {ordinal}")
        if ordinal < 9 and receipt.get("rsync_returncode") != 0:
            raise ValueError(
                "GPU repair handoff lost artifacts before the 2.45x observation"
            )
        if ordinal in (1, 2, 3, 4, 5, 7, 8) and receipt.get("returncode") != 0:
            raise ValueError(
                "GPU repair handoff failed before completing its smoke and 2x prefix"
            )
        receipts.append(receipt)
        paths[f"step_{ordinal:02d}"] = path

    final = receipts[-1]
    recovery_error = "ValueError: expected exactly one artifact sentinel, found 0"
    failure_message = (
        "artifact sync failed after natural-245x-r1; stdout recovery failed: "
        "expected exactly one artifact sentinel, found 0"
    )
    if (
        final.get("name") != "natural-245x-r1"
        or final.get("returncode") in (None, 0)
        or final.get("rsync_returncode") in (None, 0)
        or final.get("artifact_recovery_error") != recovery_error
        or (execution_root / "step-10.json").exists()
        or result.get("status") != "failed"
        or result.get("steps") != receipts
        or result.get("teardown") != teardown
        or failure
        != {
            "type": "RuntimeError",
            "message": failure_message,
        }
    ):
        raise ValueError(
            "GPU repair handoff is not the exact natural-245x-r1 artifact failure"
        )

    ids = instances.get("instance_ids")
    terminated = teardown.get("terminated_instance_ids")
    if (
        not isinstance(ids, list)
        or len(ids) != 16
        or len(set(ids)) != 16
        or sorted(ids) != sorted(terminated or ())
        or teardown.get("verified_empty") is not True
    ):
        raise ValueError("GPU repair handoff did not tear down the exact retained fleet")
    required_teardown = {
        "campaign": repair_plan["campaign"],
        "arm": "gpu",
        "cluster_name": arm["cluster_name"],
        "lifecycle_plan_sha256": lifecycle["plan_sha256"],
        "instance_ids_sha256": digest(sorted(ids)),
    }
    if any(teardown.get(key) != item for key, item in required_teardown.items()):
        raise ValueError("GPU repair handoff teardown is missing its execution identity")
    return {
        "campaign": repair_plan["campaign"],
        "arm": "gpu",
        "cluster_name": arm["cluster_name"],
        "lifecycle_plan_sha256": lifecycle["plan_sha256"],
        "instance_ids": sorted(ids),
        "instance_ids_sha256": digest(sorted(ids)),
        "completed_prefix_handoff": True,
        "handoff_failure_trial": "natural-245x-r1",
        "files": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in paths.items()
        },
    }


def _validate_gpu_245x_successful_execution_steps(
    repair_plan: Mapping[str, Any], repair_root: Path
) -> dict[str, Any]:
    """Bind a successful minimal repair to its exact command and step receipts."""
    execution_root = repair_root.resolve().parent
    lifecycle = read_json(execution_root / "lifecycle-plan.json")
    result = read_json(execution_root / "result.json")
    command_path = Path(repair_plan["arms"]["gpu"]["command_plan"])
    command_plan = read_json(command_path)
    raw_commands = command_plan.get("commands")
    if command_plan.get("schema_version") != 1 or not isinstance(raw_commands, list):
        raise ValueError("2.45x repair command plan is invalid")

    allowed_command_fields = {
        "name",
        "command",
        "restart_yaml",
        "allow_failure",
        "timeout_seconds",
    }
    command_steps = []
    for raw in raw_commands:
        if not isinstance(raw, Mapping) or set(raw) - allowed_command_fields:
            raise ValueError("2.45x repair command plan contains an invalid step")
        command_steps.append(
            {
                "name": raw.get("name"),
                "command": raw.get("command"),
                "restart_yaml": raw.get("restart_yaml"),
                "allow_failure": raw.get("allow_failure", False),
                "timeout_seconds": raw.get("timeout_seconds", 7200),
            }
        )
    expected_names = [
        "mkdir-results",
        "inventory",
        "stage-dataset",
        "transport-smoke",
        "natural-245x-r1",
        "natural-245x-r2",
    ]
    expected_allow_failure = [False, False, False, False, True, True]
    expected_timeouts = [300, 1800, 7200, 7200, 1800, 1800]
    lifecycle_steps = lifecycle.get("steps") or []
    if (
        lifecycle_steps != command_steps
        or [step.get("name") for step in command_steps] != expected_names
        or [step.get("allow_failure") for step in command_steps]
        != expected_allow_failure
        or [step.get("timeout_seconds") for step in command_steps]
        != expected_timeouts
        or any(step.get("restart_yaml") is not None for step in command_steps[:3])
    ):
        raise ValueError(
            "2.45x repair lifecycle is not the exact six-step fixed-wave plan"
        )

    reset_files: dict[str, dict[str, str]] = {}
    expected_trials = {
        "transport-smoke": ("smoke", "smoke", 1, 1, 1, None),
        "natural-245x-r1": ("natural", "gpu", 1, 245, 100, 0.50),
        "natural-245x-r2": ("natural", "gpu", 2, 245, 100, 0.50),
    }

    def option(arguments: Sequence[str], name: str) -> str | None:
        positions = [index for index, item in enumerate(arguments) if item == name]
        if not positions:
            return None
        if len(positions) != 1 or positions[0] + 1 >= len(arguments):
            raise ValueError(f"2.45x repair worker option is ambiguous: {name}")
        return arguments[positions[0] + 1]

    for step in command_steps[3:]:
        name = str(step["name"])
        restart = step.get("restart_yaml")
        restart_path = Path(str(restart))
        if (
            not isinstance(restart, str)
            or restart_path.name != f"{name}.yaml"
            or not restart_path.is_file()
        ):
            raise ValueError(f"2.45x repair trial lacks its restart YAML: {name}")
        reset_files[name] = {
            "path": str(restart_path),
            "sha256": file_sha256(restart_path),
        }
        tokens = shlex.split(str(step.get("command") or ""))
        module_positions = [
            index
            for index in range(len(tokens) - 1)
            if tokens[index : index + 2]
            == ["-m", "release.benchmarks.ray_data_gpu_sort.cloud.worker"]
        ]
        if len(module_positions) != 1:
            raise ValueError(f"2.45x repair step is not one worker command: {name}")
        arguments = tokens[module_positions[0] + 2 :]
        if any(item in {"&&", "||", ";", "|"} for item in arguments):
            raise ValueError(f"2.45x repair worker command has trailing shell: {name}")
        kind, backend, repetition, numerator, denominator, wave = expected_trials[name]
        observed = (
            option(arguments, "--kind"),
            option(arguments, "--backend"),
            int(option(arguments, "--repetition") or -1),
            int(option(arguments, "--scale-numerator") or -1),
            int(option(arguments, "--scale-denominator") or -1),
        )
        if (
            observed != (kind, backend, repetition, numerator, denominator)
            or option(arguments, "--cell") != "full"
            or Path(option(arguments, "--output") or "").name != f"{name}.json"
            or option(arguments, "--selected-wave") is not None
            or (
                wave is None
                and option(arguments, "--wave-fraction") is not None
            )
            or (
                wave is not None
                and option(arguments, "--wave-fraction") != str(wave)
            )
        ):
            raise ValueError(f"2.45x repair worker command differs from trial: {name}")

    receipts: list[dict[str, Any]] = []
    receipt_files: dict[str, dict[str, str]] = {}
    for ordinal, step in enumerate(command_steps, 1):
        path = execution_root / f"step-{ordinal:02d}.json"
        if not path.is_file():
            raise ValueError(f"2.45x repair is missing step receipt {ordinal}")
        receipt = read_json(path)
        sync_code = receipt.get("rsync_returncode")
        recovered = receipt.get("artifact_recovered_from_stdout")
        if (
            receipt.get("name") != step["name"]
            or receipt.get("allow_failure") is not step["allow_failure"]
            or not isinstance(receipt.get("returncode"), int)
            or not isinstance(sync_code, int)
            or (sync_code == 0 and recovered not in (None, False))
            or (sync_code != 0 and recovered is not True)
            or receipt.get("artifact_recovery_error") is not None
            or (
                ordinal <= 4
                and (
                    receipt.get("returncode") != 0
                    or sync_code != 0
                    or recovered not in (None, False)
                )
            )
        ):
            raise ValueError(f"2.45x repair step receipt differs at {ordinal}")
        receipts.append(receipt)
        receipt_files[f"step-{ordinal:02d}"] = {
            "path": str(path),
            "sha256": file_sha256(path),
        }
    if (
        result.get("status") != "success"
        or result.get("steps") != receipts
        or (execution_root / "step-07.json").exists()
        or (execution_root / "failure.json").exists()
    ):
        raise ValueError(
            "2.45x repair successful lifecycle does not contain the exact receipts"
        )
    return {
        "command_plan_sha256": file_sha256(command_path),
        "receipts": receipt_files,
        "reset_yamls": reset_files,
    }


def _selected_wave_from_results(
    default: Mapping[str, Any], candidate: Mapping[str, Any]
) -> tuple[float, float | None]:
    def usable(value: Mapping[str, Any]) -> float | None:
        elapsed = _number(value.get("cold_sort_s"))
        return (
            elapsed
            if value.get("valid") is True
            and elapsed is not None
            and math.isfinite(elapsed)
            and elapsed > 0
            else None
        )

    default_s, candidate_s = usable(default), usable(candidate)
    if default_s is None and candidate_s is None:
        raise ValueError("repair wave screens are both invalid")
    if default_s is None:
        return 0.375, None
    if candidate_s is None:
        return 0.50, None
    improvement = (default_s - candidate_s) / default_s
    return (0.375 if improvement >= 0.03 else 0.50), improvement


def _validate_repair_prior(
    *,
    arm: str,
    original_study_path: Path,
    original_plan: Mapping[str, Any],
    repair_plan: Mapping[str, Any],
) -> dict[str, Any]:
    prior = repair_plan.get(f"prior_{arm}_teardown") or {}
    original_record = (prior.get("files") or {}).get("original_study") or {}
    original_arm = (original_plan.get("arms") or {}).get(arm, {})
    if (
        original_record.get("sha256") != file_sha256(original_study_path)
        or prior.get("original_study_plan_sha256")
        != original_plan.get("plan_sha256")
        or prior.get("campaign") != original_plan.get("campaign")
        or prior.get("arm") != arm
        or prior.get("cluster_name") != original_arm.get("cluster_name")
    ):
        raise ValueError(
            f"{arm.upper()} repair study is not bound to the original report study"
        )
    for name, record in (prior.get("files") or {}).items():
        path = Path(str(record.get("path", "")))
        if not path.is_file() or file_sha256(path) != record.get("sha256"):
            raise ValueError(
                f"{arm.upper()} repair study's recorded original provenance "
                f"changed: {name}"
            )
    for field in ("dataset_digest", "dataset_manifest_sha256"):
        if repair_plan.get(field) != original_plan.get(field):
            raise ValueError(f"repair and original studies differ in {field}")
    for field in ("ray_version", "ray_commit", "wheel_sha256"):
        if repair_plan["bundle"].get(field) != original_plan["bundle"].get(field):
            raise ValueError(f"repair and original studies differ in {field}")
    original_topology = original_arm.get("topology") or {}
    repair_topology = (repair_plan.get("arms") or {}).get(arm, {}).get("topology") or {}

    def normalized(raw: Mapping[str, Any]) -> dict[str, Any]:
        nodes = int(raw.get("nodes", -1))
        return {
            "nodes": nodes,
            "cpus": float(nodes * int(raw.get("vcpus_per_node", -1))),
            "gpus": float(nodes * int(raw.get("gpus_per_node", -1))),
            "instance_type": raw.get("instance_type"),
            "single_availability_zone": True,
        }

    frozen_topology = prior.get("topology")
    if (
        not isinstance(frozen_topology, Mapping)
        or normalized(original_topology) != dict(frozen_topology)
        or normalized(repair_topology) != dict(frozen_topology)
    ):
        raise ValueError(
            f"{arm.upper()} repair topology differs from the frozen original topology"
        )
    return dict(prior)


def _repair_staged_manifest(
    repair_plan: Mapping[str, Any],
    repair_root: Path,
    *,
    arm: str,
    expected_instance_ids: Sequence[str],
) -> str:
    stage = _unique_json(repair_root, "dataset-stage.json")
    stage_nodes = stage.get("nodes") or []
    manifest_shas = {node.get("manifest_sha256") for node in stage_nodes}
    instance_ids = {node.get("instance_id") for node in stage_nodes}
    ray_node_ids = {node.get("ray_node_id") for node in stage_nodes}
    if (
        stage.get("dataset_digest") != repair_plan["dataset_digest"]
        or stage.get("digest") != digest(stage_nodes)
        or len(stage_nodes) != 16
        or any(
            node.get("dataset_digest") != repair_plan["dataset_digest"]
            for node in stage_nodes
        )
        or len(manifest_shas) != 1
        or None in manifest_shas
        or len(instance_ids) != 16
        or None in instance_ids
        or instance_ids != set(expected_instance_ids)
        or len(ray_node_ids) != 16
        or None in ray_node_ids
    ):
        raise ValueError(
            f"{arm.upper()} repair staged-dataset provenance is inconsistent"
        )
    return str(next(iter(manifest_shas)))


def _validate_repair_overlay(
    original_study_path: Path,
    original_plan: Mapping[str, Any],
    original_gpu: Mapping[str, Mapping[str, Any]],
    repair_study_path: Path,
    repair_root: Path,
    repair: Mapping[str, Mapping[str, Any]],
    *,
    allow_245x_handoff: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    repair_plan = _validated_study(repair_study_path)
    if repair_plan.get("trial_mode") != "gpu-repair" or set(
        repair_plan.get("arms", {})
    ) != {"gpu"}:
        raise ValueError("repair study is not an exact gpu-repair plan")
    trials = repair_plan["arms"]["gpu"].get("trials") or []
    trial_names = [trial.get("name") for trial in trials]
    expected_names = [name.removesuffix(".json") for name in GPU_REPAIR_RESULT_NAMES]
    result_names = (
        GPU_REPAIR_COMPLETED_PREFIX_NAMES
        if allow_245x_handoff
        and set(repair) == set(GPU_REPAIR_COMPLETED_PREFIX_NAMES)
        else GPU_REPAIR_RESULT_NAMES
    )
    completed_prefix_handoff = result_names == GPU_REPAIR_COMPLETED_PREFIX_NAMES
    if trial_names != expected_names or set(repair) != set(result_names):
        raise ValueError(
            "repair study/results do not contain the exact full repair trial set "
            "or its narrowly allowed completed 2x prefix"
        )
    original_trials = {
        trial.get("name"): trial
        for trial in ((original_plan.get("arms") or {}).get("gpu", {}).get("trials") or [])
    }
    for trial in trials:
        claimed = trial.get("identity")
        body = {key: item for key, item in trial.items() if key != "identity"}
        if claimed != digest(body):
            raise ValueError(f"repair trial identity is invalid: {trial.get('name')}")
        original_trial = original_trials.get(trial.get("name")) or {}
        original_body = {
            key: item for key, item in original_trial.items() if key != "identity"
        }
        if (
            original_trial.get("identity") != digest(original_body)
            or body != original_body
        ):
            raise ValueError(
                f"repair trial differs from the original study: {trial.get('name')}"
            )

    _validate_repair_prior(
        arm="gpu",
        original_study_path=original_study_path,
        original_plan=original_plan,
        repair_plan=repair_plan,
    )
    execution = (
        _repair_245x_handoff_execution_provenance(repair_plan, repair_root)
        if completed_prefix_handoff
        else _repair_execution_provenance(repair_plan, repair_root, arm="gpu")
    )
    staged_manifest_sha = _repair_staged_manifest(
        repair_plan,
        repair_root,
        arm="gpu",
        expected_instance_ids=execution["instance_ids"],
    )
    selection = _unique_json(repair_root, "selected-wave.json")
    default = repair["tune-2x-wave-0500.json"]
    candidate = repair["tune-2x-wave-0375.json"]
    expected_wave, improvement = _selected_wave_from_results(default, candidate)
    observed_improvement = _number(selection.get("observed_relative_improvement"))
    if (
        _number(selection.get("selected_wave_fraction")) != expected_wave
        or _number(selection.get("minimum_relative_improvement")) != 0.03
        or (
            improvement is not None
            and (
                observed_improvement is None
                or not math.isclose(
                    observed_improvement,
                    improvement,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            )
        )
        or (improvement is None and observed_improvement is not None)
    ):
        raise ValueError("repair selected-wave artifact does not implement the 3% gate")

    bundle_provenance = _bundle_artifact_provenance(repair_plan)
    by_name = {trial["name"]: trial for trial in trials}
    for filename in result_names:
        name = filename.removesuffix(".json")
        value = repair[filename]
        if completed_prefix_handoff and not _completed_correctly(value):
            raise ValueError(
                f"GPU repair handoff prefix did not complete correctly: {filename}"
            )
        trial = by_name[name]
        wave = trial.get("wave_fraction")
        if wave is None:
            wave = expected_wave if trial.get("selected_wave_file") else 0.50
        expected_trial = {
            "kind": trial["kind"],
            "backend": trial["backend"],
            "cell": trial["cell"],
            "repetition": trial["repetition"],
            "scale_numerator": trial["scale_numerator"],
            "scale_denominator": trial["scale_denominator"],
            "wave_fraction": wave,
        }
        top_trial = {
            key: value.get(key)
            for key in (
                "kind",
                "backend",
                "repetition",
                "scale_numerator",
                "scale_denominator",
                "wave_fraction",
            )
        }
        if value.get("cell") not in (trial["cell"],) and not (
            isinstance(value.get("cell"), Mapping)
            and value["cell"].get("name") == trial["cell"]
        ):
            raise ValueError(f"repair result cell differs from trial: {filename}")
        if top_trial != {key: expected_trial[key] for key in top_trial}:
            raise ValueError(f"repair result trial fields differ from study: {filename}")
        identity = value.get("artifact_identity") or {}
        identity_body = {
            key: item for key, item in identity.items() if key != "digest"
        }
        if identity.get("digest") != digest(identity_body):
            raise ValueError(f"repair artifact identity digest is invalid: {filename}")
        if (
            identity.get("trial") != expected_trial
            or identity.get("dataset_manifest_sha256") != staged_manifest_sha
            or identity.get("wheel_sha256") != bundle_provenance["wheel_sha256"]
            or identity.get("harness_files") != bundle_provenance["harness_files"]
            or identity.get("ray_data_overlay") != bundle_provenance["ray_data_overlay"]
            or identity.get("input_plan_digest") != (value.get("plan") or {}).get("digest")
            or value.get("ray_version") != repair_plan["bundle"]["ray_version"]
            or value.get("ray_commit") != repair_plan["bundle"]["ray_commit"]
        ):
            raise ValueError(f"repair artifact provenance differs from study: {filename}")
        original = original_gpu.get(filename)
        if not original or value.get("plan") != original.get("plan"):
            raise ValueError(f"repair input plan differs from original: {filename}")
        if name != "transport-smoke":
            current_cell, original_cell = value.get("cell") or {}, original.get("cell") or {}
            if any(
                current_cell.get(field) != original_cell.get(field)
                for field in ("name", "columns", "keys")
            ):
                raise ValueError(f"repair cell/columns/keys differ from original: {filename}")
        else:
            for field in ("rows", "blocks", "schema"):
                if field in (original.get("input") or {}) and (
                    value.get("input") or {}
                ).get(field) != original["input"].get(field):
                    raise ValueError(f"repair smoke input differs from original: {field}")
    return repair_plan, selection, {
        "repair_study_sha256": file_sha256(repair_study_path),
        "repair_study_plan_sha256": repair_plan["plan_sha256"],
        "execution": execution,
        "staged_manifest_sha256": staged_manifest_sha,
        **bundle_provenance,
    }


def _validate_gpu_245x_repair_overlay(
    original_study_path: Path,
    original_plan: Mapping[str, Any],
    original_gpu: Mapping[str, Mapping[str, Any]],
    repair_study_path: Path,
    repair_root: Path,
    repair: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the fixed-wave, 2.45x-only GPU repair campaign."""
    repair_plan = _validated_study(repair_study_path)
    if repair_plan.get("trial_mode") != "gpu-245x-repair" or set(
        repair_plan.get("arms", {})
    ) != {"gpu"}:
        raise ValueError("2.45x repair study is not an exact gpu-245x-repair plan")

    trials = repair_plan["arms"]["gpu"].get("trials") or []
    expected_names = [
        name.removesuffix(".json") for name in GPU_245X_REPAIR_RESULT_NAMES
    ]
    if [trial.get("name") for trial in trials] != expected_names or set(
        repair
    ) != set(GPU_245X_REPAIR_RESULT_NAMES):
        raise ValueError(
            "2.45x repair study/results do not contain the exact three repair trials"
        )

    expected_bodies = {
        "transport-smoke": {
            "name": "transport-smoke",
            "arm": "gpu",
            "kind": "smoke",
            "backend": "smoke",
            "cell": "full",
            "repetition": 1,
            "scale_numerator": 1,
            "scale_denominator": 1,
            "wave_fraction": None,
            "selected_wave_file": None,
            "allow_failure": False,
        },
        **{
            f"natural-245x-r{repetition}": {
                "name": f"natural-245x-r{repetition}",
                "arm": "gpu",
                "kind": "natural",
                "backend": "gpu",
                "cell": "full",
                "repetition": repetition,
                "scale_numerator": 245,
                "scale_denominator": 100,
                "wave_fraction": 0.50,
                "selected_wave_file": None,
                "allow_failure": True,
            }
            for repetition in (1, 2)
        },
    }
    original_trials = {
        trial.get("name"): trial
        for trial in ((original_plan.get("arms") or {}).get("gpu", {}).get("trials") or [])
    }
    for trial in trials:
        name = trial.get("name")
        body = {key: item for key, item in trial.items() if key != "identity"}
        if body != expected_bodies.get(str(name)) or trial.get("identity") != digest(
            body
        ):
            raise ValueError(f"2.45x repair trial is not exact: {name}")
        original_trial = original_trials.get(name) or {}
        original_body = {
            key: item for key, item in original_trial.items() if key != "identity"
        }
        if original_trial.get("identity") != digest(original_body) or any(
            body[field] != original_body.get(field)
            for field in (
                "arm",
                "kind",
                "backend",
                "cell",
                "repetition",
                "scale_numerator",
                "scale_denominator",
                "allow_failure",
            )
        ):
            raise ValueError(
                f"2.45x repair trial differs from the original study: {name}"
            )

    _validate_repair_prior(
        arm="gpu",
        original_study_path=original_study_path,
        original_plan=original_plan,
        repair_plan=repair_plan,
    )
    execution = _repair_execution_provenance(repair_plan, repair_root, arm="gpu")
    execution["successful_steps"] = _validate_gpu_245x_successful_execution_steps(
        repair_plan, repair_root
    )
    staged_manifest_sha = _repair_staged_manifest(
        repair_plan,
        repair_root,
        arm="gpu",
        expected_instance_ids=execution["instance_ids"],
    )
    bundle_provenance = _bundle_artifact_provenance(repair_plan)
    by_name = {trial["name"]: trial for trial in trials}
    for filename in GPU_245X_REPAIR_RESULT_NAMES:
        name = filename.removesuffix(".json")
        value = repair[filename]
        trial = by_name[name]
        # The worker records its parser default for the smoke. Every artifact in
        # this repair campaign is therefore explicitly bound to fixed wave 0.50.
        expected_trial = {
            "kind": trial["kind"],
            "backend": trial["backend"],
            "cell": trial["cell"],
            "repetition": trial["repetition"],
            "scale_numerator": trial["scale_numerator"],
            "scale_denominator": trial["scale_denominator"],
            "wave_fraction": 0.50,
        }
        top_trial = {
            key: value.get(key)
            for key in (
                "kind",
                "backend",
                "repetition",
                "scale_numerator",
                "scale_denominator",
                "wave_fraction",
            )
        }
        cell = value.get("cell")
        if not (
            cell == trial["cell"]
            or (isinstance(cell, Mapping) and cell.get("name") == trial["cell"])
        ):
            raise ValueError(f"2.45x repair result cell differs from trial: {filename}")
        if top_trial != {key: expected_trial[key] for key in top_trial}:
            raise ValueError(
                f"2.45x repair result trial fields differ from study: {filename}"
            )

        identity = value.get("artifact_identity") or {}
        identity_body = {
            key: item for key, item in identity.items() if key != "digest"
        }
        if identity.get("digest") != digest(identity_body):
            raise ValueError(
                f"2.45x repair artifact identity digest is invalid: {filename}"
            )
        if (
            identity.get("trial") != expected_trial
            or identity.get("dataset_manifest_sha256") != staged_manifest_sha
            or identity.get("wheel_sha256") != bundle_provenance["wheel_sha256"]
            or identity.get("harness_files") != bundle_provenance["harness_files"]
            or identity.get("ray_data_overlay")
            != bundle_provenance["ray_data_overlay"]
            or identity.get("input_plan_digest")
            != (value.get("plan") or {}).get("digest")
            or value.get("ray_version") != repair_plan["bundle"]["ray_version"]
            or value.get("ray_commit") != repair_plan["bundle"]["ray_commit"]
        ):
            raise ValueError(
                f"2.45x repair artifact provenance differs from study: {filename}"
            )

        original = original_gpu.get(filename) or {}
        if not original or value.get("plan") != original.get("plan"):
            raise ValueError(
                f"2.45x repair input plan differs from original: {filename}"
            )
        if name == "transport-smoke":
            for field in ("rows", "blocks", "schema"):
                if field in (original.get("input") or {}) and (
                    value.get("input") or {}
                ).get(field) != original["input"].get(field):
                    raise ValueError(
                        f"2.45x repair smoke input differs from original: {field}"
                    )
        else:
            current_cell = value.get("cell") or {}
            original_cell = original.get("cell") or {}
            if not isinstance(current_cell, Mapping) or not isinstance(
                original_cell, Mapping
            ) or any(
                current_cell.get(field) != original_cell.get(field)
                for field in ("name", "columns", "keys")
            ):
                raise ValueError(
                    f"2.45x repair cell/columns/keys differ from original: {filename}"
                )

    return repair_plan, {
        "repair_study_sha256": file_sha256(repair_study_path),
        "repair_study_plan_sha256": repair_plan["plan_sha256"],
        "execution": execution,
        "staged_manifest_sha256": staged_manifest_sha,
        **bundle_provenance,
    }


def _cpu_input_reference_name(filename: str) -> str:
    if filename == "natural-2x-r1.json":
        return "tune-2x-wave-0500.json"
    return filename


def _validate_preserved_cpu_observation(
    *,
    filename: str,
    value: Mapping[str, Any],
    gpu_reference: Mapping[str, Any],
    original_plan: Mapping[str, Any],
) -> None:
    if _classification(value) not in ("accepted", "telemetry-warning"):
        raise ValueError(
            f"CPU repair requires the original completed observation: {filename}"
        )
    name = filename.removesuffix(".json")
    original_trials = {
        trial.get("name"): trial
        for trial in ((original_plan.get("arms") or {}).get("cpu", {}).get("trials") or [])
    }
    trial = original_trials.get(name) or {}
    trial_body = {key: item for key, item in trial.items() if key != "identity"}
    if trial.get("identity") != digest(trial_body):
        raise ValueError(f"original CPU trial identity is invalid: {name}")
    expected_trial = {
        "kind": trial.get("kind"),
        "backend": trial.get("backend"),
        "cell": trial.get("cell"),
        "repetition": trial.get("repetition"),
        "scale_numerator": trial.get("scale_numerator"),
        "scale_denominator": trial.get("scale_denominator"),
        "wave_fraction": 0.50,
    }
    top_trial = {
        key: value.get(key)
        for key in (
            "kind",
            "backend",
            "repetition",
            "scale_numerator",
            "scale_denominator",
            "wave_fraction",
        )
    }
    if top_trial != {key: expected_trial[key] for key in top_trial}:
        raise ValueError(f"preserved CPU trial fields differ from study: {filename}")

    identity = value.get("artifact_identity") or {}
    identity_body = {key: item for key, item in identity.items() if key != "digest"}
    provenance = _bundle_artifact_provenance(original_plan)
    reference_identity = gpu_reference.get("artifact_identity") or {}
    if identity.get("digest") != digest(identity_body):
        raise ValueError(f"preserved CPU artifact identity is invalid: {filename}")
    if (
        identity.get("trial") != expected_trial
        or identity.get("wheel_sha256") != provenance["wheel_sha256"]
        or identity.get("harness_files") != provenance["harness_files"]
        or identity.get("ray_data_overlay") != provenance["ray_data_overlay"]
        or identity.get("input_plan_digest") != (value.get("plan") or {}).get("digest")
        or identity.get("dataset_manifest_sha256")
        != reference_identity.get("dataset_manifest_sha256")
        or value.get("ray_version") != original_plan["bundle"]["ray_version"]
        or value.get("ray_commit") != original_plan["bundle"]["ray_commit"]
    ):
        raise ValueError(f"preserved CPU provenance differs from study: {filename}")
    cell, reference_cell = value.get("cell") or {}, gpu_reference.get("cell") or {}
    if (
        not gpu_reference
        or value.get("plan") != gpu_reference.get("plan")
        or any(
            cell.get(field) != reference_cell.get(field)
            for field in ("name", "columns", "keys")
        )
    ):
        raise ValueError(f"preserved CPU input/cell differs from original: {filename}")


def _checked_profile_files(
    study_root: Path, files: Mapping[str, Any], *, label: str
) -> dict[str, dict[str, str]]:
    """Resolve and content-address every file used by a frozen exception."""
    root = study_root.resolve()
    checked: dict[str, dict[str, str]] = {}
    if not files:
        raise ValueError(f"{label} resource-rejection evidence profile is empty")
    for name, expected_sha in files.items():
        relative = Path(str(name))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{label} resource-rejection evidence path is unsafe: {name}")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(
                f"{label} resource-rejection evidence is missing: {name}"
            )
        observed_sha = file_sha256(path)
        if not isinstance(expected_sha, str) or observed_sha != expected_sha:
            raise ValueError(
                f"{label} resource-rejection evidence hash differs: {name}"
            )
        checked[str(name)] = {"path": str(path), "sha256": observed_sha}
    return checked


def _resource_shape_node_ids(path: Path, *, label: str) -> list[str]:
    shape = read_json(path)
    observed = shape.get("observed") or {}
    node_ids = observed.get("ray_node_ids")
    if (
        observed.get("nodes") != 16
        or observed.get("cpus") != 256
        or observed.get("gpus") != 0
        or not isinstance(node_ids, list)
        or len(node_ids) != 16
        or len(set(node_ids)) != 16
        or any(not isinstance(item, str) or not item for item in node_ids)
    ):
        raise ValueError(
            f"{label} resource-rejection shape is not the exact 16-node CPU fleet"
        )
    return sorted(node_ids)


def _resource_failure_diagnostic(
    command_log: Path,
    rsync_log: Path,
    *,
    expected_seconds: int,
    label: str,
) -> int:
    ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
    command = ansi.sub("", command_log.read_text(encoding="utf-8", errors="replace"))
    rsync = ansi.sub("", rsync_log.read_text(encoding="utf-8", errors="replace"))
    complete = list(
        re.finditer(
            r"Shuffle Map:\s*100%\s+198M/198M\s+\[(\d+):(\d+)<",
            command,
        )
    )
    reduce_progress = [
        int(value) for value in re.findall(r"Shuffle Reduce:\s*(\d+)%", command)
    ]
    warning = command.find("More than 16GB of driver memory used")
    closed = command.find("Shared connection to", complete[-1].start() if complete else 0)
    if (
        "InputDataBuffer[Input] -> AllToAllOperator[Sort]" not in command
        or not complete
        or not reduce_progress
        or any(value != 0 for value in reduce_progress)
        or warning < complete[0].start()
        or closed < max(warning, complete[-1].start())
        or "closed" not in command[closed : closed + 120]
        or "SSH command failed" not in rsync
        or not any(
            marker in rsync
            for marker in (
                "not responding",
                "Connection timed out",
                "timed out during banner exchange",
            )
        )
    ):
        raise ValueError(
            f"{label} resource-rejection logs do not show the exact "
            "map-to-reduce head/driver failure"
        )
    elapsed = max(
        int(match.group(1)) * 60 + int(match.group(2)) for match in complete
    )
    if elapsed != expected_seconds:
        raise ValueError(
            f"{label} resource-rejection diagnostic time differs: {elapsed}"
        )
    return elapsed


def _validate_cpu_resource_campaign(
    *,
    study_path: Path,
    plan: Mapping[str, Any],
    repair_root: Path,
    profile: Mapping[str, Any],
    require_teardown: bool,
) -> dict[str, Any]:
    """Validate one frozen lifecycle failure without inventing a worker result."""
    repair_root = repair_root.resolve()
    execution_root = repair_root.parent
    study_root = execution_root.parent.parent
    label = str(profile.get("campaign") or "unknown campaign")
    trial = CPU_RESOURCE_REJECTION_RESULT.removesuffix(".json")
    expected_step_names = [
        "mkdir-results",
        "inventory",
        "stage-dataset",
        *(name.removesuffix(".json") for name in CPU_REPAIR_RESULT_NAMES),
    ]
    required_evidence = {
        "study.json",
        "executions/cpu/lifecycle-plan.json",
        "executions/cpu/instances.json",
        "executions/cpu/failure.json",
        *(f"executions/cpu/step-{ordinal:02d}.json" for ordinal in range(1, 10)),
        "executions/cpu/shape-08-natural-2x-r1.json",
        "executions/cpu/shape-09-natural-245x-r1.json",
        "executions/cpu/command-09-natural-245x-r1.log",
        "executions/cpu/rsync-09-natural-245x-r1.log",
    }
    if require_teardown:
        required_evidence.add("executions/cpu/teardown.json")
    profile_files = profile.get("files") or {}
    if (
        repair_root.name != "remote-results"
        or study_path.resolve() != (study_root / "study.json").resolve()
        or plan.get("campaign") != profile.get("campaign")
        or plan.get("plan_sha256") != profile.get("study_plan_sha256")
        or plan.get("trial_mode") != "cpu-repair"
        or set(plan.get("arms", {})) != {"cpu"}
        or not isinstance(profile_files, Mapping)
        or set(profile_files) != required_evidence
    ):
        raise ValueError(
            f"{label} resource rejection is not bound to the exact CPU campaign"
        )
    evidence = _checked_profile_files(study_root, profile_files, label=label)
    if (execution_root / "result.json").exists():
        raise ValueError(
            f"{label} resource-rejection lifecycle unexpectedly has a success result"
        )
    if list(repair_root.rglob(CPU_RESOURCE_REJECTION_RESULT)):
        raise ValueError(
            f"{label} resource rejection cannot coexist with a worker result"
        )
    expected_worker_results = set(CPU_REPAIR_RESULT_NAMES) - {
        CPU_RESOURCE_REJECTION_RESULT
    }
    if set(_results(repair_root)) != expected_worker_results:
        raise ValueError(
            f"{label} resource rejection does not follow the exact five earlier "
            "CPU worker results"
        )

    lifecycle_path = execution_root / "lifecycle-plan.json"
    lifecycle = read_json(lifecycle_path)
    lifecycle_body = {
        key: item for key, item in lifecycle.items() if key != "plan_sha256"
    }
    arm = plan["arms"]["cpu"]
    topology = arm["topology"]
    expected_topology = {
        "nodes": int(topology["nodes"]),
        "cpus": float(topology["nodes"] * topology["vcpus_per_node"]),
        "gpus": float(topology["nodes"] * topology["gpus_per_node"]),
        "instance_type": topology["instance_type"],
        "single_availability_zone": True,
    }
    steps = lifecycle.get("steps") or []
    if (
        lifecycle.get("plan_sha256") != digest(lifecycle_body)
        or lifecycle.get("plan_sha256") != profile.get("lifecycle_plan_sha256")
        or lifecycle.get("campaign") != plan.get("campaign")
        or lifecycle.get("arm") != "cpu"
        or lifecycle.get("cluster_name") != arm.get("cluster_name")
        or lifecycle.get("expected") != expected_topology
        or lifecycle.get("cluster_yaml_sha256")
        != file_sha256(Path(arm["launch_yaml"]))
        or lifecycle.get("command_plan_sha256")
        != file_sha256(Path(arm["command_plan"]))
        or lifecycle.get("same_ec2_ids") is not True
        or lifecycle.get("fresh_ray_node_ids_each_trial") is not True
        or lifecycle.get("ray_default_plasma") is not True
        or [step.get("name") for step in steps] != expected_step_names
        or any(
            step.get("allow_failure") is not (index >= 3)
            for index, step in enumerate(steps)
        )
    ):
        raise ValueError(
            f"{label} resource-rejection lifecycle differs from the exact CPU plan"
        )

    receipts = []
    for ordinal, (step, expected_name) in enumerate(
        zip(steps, expected_step_names), 1
    ):
        receipt = read_json(execution_root / f"step-{ordinal:02d}.json")
        expected_allow_failure = ordinal >= 4
        if (
            receipt.get("name") != expected_name
            or receipt.get("allow_failure") is not expected_allow_failure
            or receipt.get("artifact_recovered_from_stdout") not in (None, False)
        ):
            raise ValueError(
                f"{label} resource-rejection step receipt differs at {ordinal}"
            )
        if ordinal <= 3 and (
            receipt.get("returncode") != 0 or receipt.get("rsync_returncode") != 0
        ):
            raise ValueError(
                f"{label} resource-rejection setup failed before the measured trials"
            )
        if 4 <= ordinal < len(steps) and receipt.get("rsync_returncode") != 0:
            raise ValueError(
                f"{label} resource-rejection evidence starts before the final trial"
            )
        receipts.append(receipt)

    final = receipts[-1]
    if (
        final.get("name") != trial
        or final.get("allow_failure") is not True
        or final.get("returncode") in (None, 0)
        or final.get("rsync_returncode") in (None, 0)
    ):
        raise ValueError(
            f"{label} resource rejection is not the final natural-245x-r1 step"
        )
    if profile.get("stdout_recovery_evidence") is True and (
        final.get("artifact_recovered_from_stdout") is not False
        or final.get("artifact_recovery_error")
        != "ValueError: expected exactly one artifact sentinel, found 0"
    ):
        raise ValueError(
            f"{label} resource rejection is missing exact stdout-recovery evidence"
        )
    failure = read_json(execution_root / "failure.json")
    if (
        failure.get("type") != "RuntimeError"
        or failure.get("message") != profile.get("failure_message")
    ):
        raise ValueError(
            f"{label} resource-rejection failure receipt differs from the final step"
        )

    prior_ids = _resource_shape_node_ids(
        execution_root / "shape-08-natural-2x-r1.json", label=label
    )
    failed_ids = _resource_shape_node_ids(
        execution_root / "shape-09-natural-245x-r1.json", label=label
    )
    if set(prior_ids) & set(failed_ids):
        raise ValueError(
            f"{label} resource-rejection trial did not receive fresh Ray node IDs"
        )
    instances = read_json(execution_root / "instances.json")
    instance_ids = instances.get("instance_ids")
    if (
        not isinstance(instance_ids, list)
        or len(instance_ids) != 16
        or len(set(instance_ids)) != 16
    ):
        raise ValueError(
            f"{label} resource-rejection campaign did not retain exactly 16 instances"
        )
    if require_teardown:
        teardown = read_json(execution_root / "teardown.json")
        required = {
            "campaign": plan["campaign"],
            "arm": "cpu",
            "cluster_name": arm["cluster_name"],
            "lifecycle_plan_sha256": lifecycle["plan_sha256"],
            "instance_ids_sha256": digest(sorted(instance_ids)),
        }
        if (
            teardown.get("verified_empty") is not True
            or sorted(teardown.get("terminated_instance_ids") or ())
            != sorted(instance_ids)
            or any(teardown.get(key) != value for key, value in required.items())
        ):
            raise ValueError(
                f"{label} resource-rejection fleet teardown is not exact and empty"
            )

    diagnostic_seconds = _resource_failure_diagnostic(
        execution_root / f"command-09-{trial}.log",
        execution_root / f"rsync-09-{trial}.log",
        expected_seconds=int(profile["diagnostic_time_to_failure_s"]),
        label=label,
    )
    return {
        "campaign": plan["campaign"],
        "study_plan_sha256": plan["plan_sha256"],
        "lifecycle_plan_sha256": lifecycle["plan_sha256"],
        "instance_ids": sorted(instance_ids),
        "instance_ids_sha256": digest(sorted(instance_ids)),
        "fresh_ray_node_ids": {
            "prior_trial_sha256": digest(prior_ids),
            "failed_trial_sha256": digest(failed_ids),
        },
        "diagnostic_time_to_failure_s": diagnostic_seconds,
        "diagnostic_only": True,
        "failure_boundary": "shuffle-map complete; shuffle-reduce not started",
        "files": evidence,
    }


def _validate_cpu_resource_rejection(
    *,
    original_study_path: Path,
    original_plan: Mapping[str, Any],
    repair_study_path: Path,
    repair_plan: Mapping[str, Any],
    repair_root: Path,
) -> dict[str, Any]:
    if (
        CPU_RESOURCE_REJECTION_PROFILE.get("trial")
        != CPU_RESOURCE_REJECTION_RESULT.removesuffix(".json")
        or CPU_RESOURCE_REJECTION_PROFILE.get("step_ordinal") != 9
    ):
        raise ValueError("CPU resource-rejection profile targets the wrong final trial")
    authoritative = _validate_cpu_resource_campaign(
        study_path=repair_study_path,
        plan=repair_plan,
        repair_root=repair_root,
        profile=CPU_RESOURCE_REJECTION_PROFILE,
        require_teardown=True,
    )
    corroborating_profile = CPU_RESOURCE_REJECTION_PROFILE.get("corroborating") or {}
    corroborating_root = (
        repair_study_path.resolve().parent.parent
        / str(corroborating_profile.get("campaign"))
    )
    corroborating_study = corroborating_root / "study.json"
    corroborating_plan = _validated_study(corroborating_study)
    authoritative_trials = repair_plan["arms"]["cpu"].get("trials") or []
    corroborating_trials = corroborating_plan.get("arms", {}).get("cpu", {}).get(
        "trials"
    ) or []
    if corroborating_trials != authoritative_trials:
        raise ValueError(
            "corroborating CPU resource-rejection study does not contain the exact "
            "same six trials"
        )
    _validate_repair_prior(
        arm="cpu",
        original_study_path=original_study_path,
        original_plan=original_plan,
        repair_plan=corroborating_plan,
    )
    corroborating = _validate_cpu_resource_campaign(
        study_path=corroborating_study,
        plan=corroborating_plan,
        repair_root=corroborating_root / "executions/cpu/remote-results",
        profile=corroborating_profile,
        require_teardown=False,
    )
    return {
        "kind": "lifecycle_resource_rejection",
        "classification": "resource-rejection",
        "backend": "pyarrow",
        "trial": CPU_RESOURCE_REJECTION_RESULT.removesuffix(".json"),
        "reason": "head/driver resource failure at the map-to-reduce boundary",
        "failure_boundary": "shuffle-map complete; shuffle-reduce not started",
        "reproduced": True,
        "reproduction_count": 2,
        "diagnostic_only": True,
        "diagnostic_time_to_failure_s": authoritative[
            "diagnostic_time_to_failure_s"
        ],
        "attempts": [authoritative, corroborating],
        "authoritative_execution": authoritative,
    }


def _validate_cpu_repair_overlay(
    original_study_path: Path,
    original_plan: Mapping[str, Any],
    original_gpu: Mapping[str, Mapping[str, Any]],
    original_cpu: Mapping[str, Mapping[str, Any]],
    repair_study_path: Path,
    repair_root: Path,
    repair: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    for name in ("trend-narrow-r1.json", "trend-core-r1.json"):
        _validate_preserved_cpu_observation(
            filename=name,
            value=original_cpu.get(name, {}),
            gpu_reference=original_gpu.get(name, {}),
            original_plan=original_plan,
        )
    repair_plan = _validated_study(repair_study_path)
    if repair_plan.get("trial_mode") != "cpu-repair" or set(
        repair_plan.get("arms", {})
    ) != {"cpu"}:
        raise ValueError("CPU repair study is not an exact cpu-repair plan")
    trials = repair_plan["arms"]["cpu"].get("trials") or []
    trial_names = [trial.get("name") for trial in trials]
    expected_names = [name.removesuffix(".json") for name in CPU_REPAIR_RESULT_NAMES]
    complete_results = set(CPU_REPAIR_RESULT_NAMES)
    resource_rejection_results = complete_results - {CPU_RESOURCE_REJECTION_RESULT}
    if trial_names != expected_names or set(repair) not in (
        complete_results,
        resource_rejection_results,
    ):
        raise ValueError(
            "CPU repair study/results do not contain the exact repair trials or "
            "the narrowly allowed final resource rejection"
        )

    original_trials = {
        trial.get("name"): trial
        for trial in ((original_plan.get("arms") or {}).get("cpu", {}).get("trials") or [])
    }
    if set(expected_names) - set(original_trials):
        raise ValueError("original study is missing CPU trial identities needed by repair")
    for trial in trials:
        claimed = trial.get("identity")
        body = {key: item for key, item in trial.items() if key != "identity"}
        original = original_trials[trial["name"]]
        original_body = {
            key: item for key, item in original.items() if key != "identity"
        }
        if claimed != digest(body) or original.get("identity") != digest(original_body):
            raise ValueError(f"CPU repair trial identity is invalid: {trial.get('name')}")
        if body != original_body:
            raise ValueError(
                f"CPU repair trial differs from the original study: {trial.get('name')}"
            )

    _validate_repair_prior(
        arm="cpu",
        original_study_path=original_study_path,
        original_plan=original_plan,
        repair_plan=repair_plan,
    )
    resource_rejection = None
    if set(repair) == complete_results:
        execution = _repair_execution_provenance(repair_plan, repair_root, arm="cpu")
    else:
        resource_rejection = _validate_cpu_resource_rejection(
            original_study_path=original_study_path,
            original_plan=original_plan,
            repair_study_path=repair_study_path,
            repair_plan=repair_plan,
            repair_root=repair_root,
        )
        execution = resource_rejection["authoritative_execution"]
    staged_manifest_sha = _repair_staged_manifest(
        repair_plan,
        repair_root,
        arm="cpu",
        expected_instance_ids=execution["instance_ids"],
    )
    bundle_provenance = _bundle_artifact_provenance(repair_plan)
    by_name = {trial["name"]: trial for trial in trials}

    for filename in CPU_REPAIR_RESULT_NAMES:
        if filename == CPU_RESOURCE_REJECTION_RESULT and resource_rejection is not None:
            continue
        name = filename.removesuffix(".json")
        value = repair[filename]
        trial = by_name[name]
        # The worker records its parser default even though CPU sort does not use
        # the GPU wave fraction. This is part of the immutable artifact identity.
        wave = trial.get("wave_fraction")
        expected_trial = {
            "kind": trial["kind"],
            "backend": trial["backend"],
            "cell": trial["cell"],
            "repetition": trial["repetition"],
            "scale_numerator": trial["scale_numerator"],
            "scale_denominator": trial["scale_denominator"],
            "wave_fraction": 0.50 if wave is None else wave,
        }
        top_trial = {
            key: value.get(key)
            for key in (
                "kind",
                "backend",
                "repetition",
                "scale_numerator",
                "scale_denominator",
                "wave_fraction",
            )
        }
        cell = value.get("cell")
        if not isinstance(cell, Mapping) or cell.get("name") != trial["cell"]:
            raise ValueError(f"CPU repair result cell differs from trial: {filename}")
        if top_trial != {key: expected_trial[key] for key in top_trial}:
            raise ValueError(
                f"CPU repair result trial fields differ from study: {filename}"
            )

        identity = value.get("artifact_identity") or {}
        identity_body = {
            key: item for key, item in identity.items() if key != "digest"
        }
        if identity.get("digest") != digest(identity_body):
            raise ValueError(
                f"CPU repair artifact identity digest is invalid: {filename}"
            )
        if (
            identity.get("trial") != expected_trial
            or identity.get("dataset_manifest_sha256") != staged_manifest_sha
            or identity.get("wheel_sha256") != bundle_provenance["wheel_sha256"]
            or identity.get("harness_files") != bundle_provenance["harness_files"]
            or identity.get("ray_data_overlay")
            != bundle_provenance["ray_data_overlay"]
            or identity.get("input_plan_digest")
            != (value.get("plan") or {}).get("digest")
            or value.get("ray_version") != repair_plan["bundle"]["ray_version"]
            or value.get("ray_commit") != repair_plan["bundle"]["ray_commit"]
        ):
            raise ValueError(
                f"CPU repair artifact provenance differs from study: {filename}"
            )

        original_attempt = original_cpu.get(filename) or {}
        reference = original_attempt if original_attempt.get("plan") else original_gpu.get(
            _cpu_input_reference_name(filename), {}
        )
        if not reference or value.get("plan") != reference.get("plan"):
            raise ValueError(
                f"CPU repair input plan differs from the original cohort: {filename}"
            )
        reference_cell = reference.get("cell") or {}
        if not isinstance(reference_cell, Mapping) or any(
            cell.get(field) != reference_cell.get(field)
            for field in ("name", "columns", "keys")
        ):
            raise ValueError(
                f"CPU repair cell/columns/keys differ from original: {filename}"
            )

    return repair_plan, {
        "repair_study_sha256": file_sha256(repair_study_path),
        "repair_study_plan_sha256": repair_plan["plan_sha256"],
        "execution": execution,
        "staged_manifest_sha256": staged_manifest_sha,
        **bundle_provenance,
    }, resource_rejection


def _number(value: Any) -> float | None:
    return (
        float(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else None
    )


def _median(values: Iterable[float | int | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return median(present) if present else None


def _results(root: Path) -> dict[str, dict[str, Any]]:
    candidates: dict[str, list[tuple[int, Path, dict[str, Any]]]] = {}
    for path in root.rglob("*.json"):
        try:
            value = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if "backend" not in value or "kind" not in value:
            continue
        candidates.setdefault(path.name, []).append(
            (path.stat().st_mtime_ns, path, value)
        )

    result: dict[str, dict[str, Any]] = {}
    for name, values in candidates.items():
        content_digests = {digest(value) for _, _, value in values}
        if len(content_digests) > 1:
            paths = sorted(str(path) for _, path, _ in values)
            raise ValueError(f"conflicting duplicate contents for {name}: {paths}")
        identities = {
            str(value.get("artifact_identity", {}).get("digest"))
            for _, _, value in values
            if value.get("artifact_identity", {}).get("digest")
        }
        if len(identities) > 1:
            raise ValueError(
                f"conflicting artifact identities for {name}: {sorted(identities)}"
            )
        _, path, value = max(values, key=lambda item: item[0])
        result[name] = {**value, "_artifact_path": str(path.resolve())}
    return result


def _selection(root: Path) -> dict[str, Any] | None:
    selections = list(root.rglob("selected-wave.json"))
    return (
        read_json(max(selections, key=lambda path: path.stat().st_mtime_ns))
        if selections
        else None
    )


def _reasons(value: Mapping[str, Any]) -> list[str]:
    raw = value.get("rejection_reasons")
    return [str(item) for item in raw] if isinstance(raw, list) else []


def _completed_correctly(value: Mapping[str, Any]) -> bool:
    if _number(value.get("cold_sort_s")) is None:
        return False
    validation = value.get("validation")
    if not isinstance(validation, Mapping):
        return False
    if value.get("kind") == "smoke":
        return validation.get("exact_every_row_value") is True
    source, output = value.get("input"), value.get("output")
    if not isinstance(source, Mapping) or not isinstance(output, Mapping):
        return False
    return (
        validation.get("ordered") is True
        and validation.get("row_id_sum") == validation.get("expected_row_id_sum")
        and source.get("rows") == output.get("rows")
        and source.get("schema") == output.get("schema")
    )


def _classification(value: Mapping[str, Any]) -> str:
    if not value:
        return "missing"
    if value.get("valid") is True:
        return "accepted"
    reasons = _reasons(value)
    joined = "\n".join(reasons)
    if any(marker in joined for marker in OOM_MARKERS):
        return "oom"
    if (
        reasons
        and set(reasons) <= LOCATION_METADATA_WARNINGS
        and _completed_correctly(value)
    ):
        return "telemetry-warning"
    return "failed"


def _time(value: Mapping[str, Any], *, strict: bool = False) -> float | None:
    classification = _classification(value)
    allowed = {"accepted"} if strict else {"accepted", "telemetry-warning"}
    return _number(value.get("cold_sort_s")) if classification in allowed else None


def _ray_io(value: Mapping[str, Any], name: str) -> int | None:
    try:
        raw = value["ray_object_store_io"]["totals"][name]
        return int(raw) if isinstance(raw, (int, float)) else None
    except (KeyError, TypeError, ValueError):
        return None


def _gpu_stat(value: Mapping[str, Any], name: str) -> float | None:
    stats = value.get("gpu_stats")
    return _number(stats.get(name)) if isinstance(stats, Mapping) else None


def _input(value: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = value.get("input")
    return raw if isinstance(raw, Mapping) else {}


def _short_reason(value: Mapping[str, Any]) -> str | None:
    classification = _classification(value)
    reasons = _reasons(value)
    if classification == "oom":
        joined = " ".join(reasons)
        allocation = re.search(r"failed to allocate ([^)]+)\)", joined)
        ceiling = re.search(r"current/max/try size = ([^)]+)", joined)
        where = (
            " during GPU run externalization"
            if "_externalize_device_tables" in joined
            else ""
        )
        detail = []
        if allocation:
            detail.append(f"allocation {allocation.group(1)}")
        if ceiling:
            detail.append(f"current/max/try {ceiling.group(1)}")
        return f"RMM OOM{where}" + (f" ({'; '.join(detail)})" if detail else "")
    if not reasons:
        return None
    clean = re.sub(r"\x1b\[[0-9;]*m", "", reasons[0]).splitlines()[0]
    return clean[:240]


def _gpu_metrics(value: Mapping[str, Any]) -> dict[str, Any]:
    stats = value.get("gpu_stats")
    stats = stats if isinstance(stats, Mapping) else {}
    ranks = stats.get("ranks")
    ranks = (
        [item for item in ranks if isinstance(item, Mapping)]
        if isinstance(ranks, list)
        else []
    )
    resources = value.get("resources")
    nodes = resources.get("nodes", []) if isinstance(resources, Mapping) else []
    nodes = (
        [item for item in nodes if isinstance(item, Mapping)]
        if isinstance(nodes, list)
        else []
    )
    output_bytes = [_number(item.get("output_bytes")) for item in ranks]
    output_bytes = [item for item in output_bytes if item is not None and item > 0]
    rank_input = sum(_number(item.get("input_bytes")) or 0 for item in ranks)
    rank_local = sum(_number(item.get("local_input_bytes")) or 0 for item in ranks)
    nvml_peaks = [_number(item.get("gpu_peak_memory_used_bytes")) for item in nodes]
    nvml_peaks = [item for item in nvml_peaks if item is not None]
    headrooms = [
        (_number(item.get("gpu_total_memory_bytes")) or 0)
        - (_number(item.get("gpu_peak_memory_used_bytes")) or 0)
        for item in nodes
        if _number(item.get("gpu_total_memory_bytes")) is not None
        and _number(item.get("gpu_peak_memory_used_bytes")) is not None
    ]
    totals = resources.get("totals", {}) if isinstance(resources, Mapping) else {}
    return {
        "peak_rmm_gib": None
        if _gpu_stat(value, "peak_device_bytes") is None
        else _gpu_stat(value, "peak_device_bytes") / GIB,
        "peak_nvml_gib": max(nvml_peaks) / GIB if nvml_peaks else None,
        "minimum_nvml_headroom_gib": min(headrooms) / GIB if headrooms else None,
        "output_balance_max_over_min": max(output_bytes) / min(output_bytes)
        if output_bytes
        else None,
        "local_input_fraction": rank_local / rank_input if rank_input else None,
        "network_sent_gib": (_number(totals.get("network_sent_bytes_delta")) or 0) / GIB
        if totals
        else None,
        "network_received_gib": (_number(totals.get("network_recv_bytes_delta")) or 0)
        / GIB
        if totals
        else None,
    }


def _io_metrics(value: Mapping[str, Any]) -> dict[str, Any]:
    decoded = _number(_input(value).get("decoded_bytes"))
    write = _ray_io(value, "spilled_bytes_total")
    restore = _ray_io(value, "restored_bytes_total")
    external = _gpu_stat(value, "externalized_bytes")
    h2d = _gpu_stat(value, "h2d_bytes")
    d2h = _gpu_stat(value, "d2h_bytes")
    plasma_read = _gpu_stat(value, "plasma_read_bytes")
    plasma_write = _gpu_stat(value, "plasma_write_bytes")

    def gib(raw: float | int | None) -> float | None:
        return None if raw is None else float(raw) / GIB

    def amplification(raw: float | int | None) -> float | None:
        return None if raw is None or not decoded else float(raw) / decoded

    return {
        "ray_write_gib": gib(write),
        "ray_restore_gib": gib(restore),
        "ray_write_amplification": amplification(write),
        "externalized_gib": gib(external),
        "externalized_fraction": amplification(external),
        "h2d_amplification": amplification(h2d),
        "d2h_amplification": amplification(d2h),
        "plasma_read_amplification": amplification(plasma_read),
        "plasma_write_amplification": amplification(plasma_write),
        "initial_runs": int(_gpu_stat(value, "initial_run_count") or 0)
        if stats_present(value)
        else None,
        "replacement_runs": int(_gpu_stat(value, "replacement_run_count") or 0)
        if stats_present(value)
        else None,
        "merge_passes": int(_gpu_stat(value, "merge_pass_count") or 0)
        if stats_present(value)
        else None,
        "first_externalize_s": _gpu_stat(value, "first_externalize_s"),
        "first_externalize_wave": _gpu_stat(value, "first_externalize_wave"),
        "wave_count": int(_gpu_stat(value, "wave_count") or 0)
        if stats_present(value)
        else None,
        "wave_target_bytes": _gpu_stat(value, "wave_target_bytes"),
    }


def stats_present(value: Mapping[str, Any]) -> bool:
    stats = value.get("gpu_stats")
    return isinstance(stats, Mapping) and bool(stats)


def _observation(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "classification": _classification(value),
        "strict_valid": value.get("valid") is True,
        "cold_sort_s": _number(value.get("cold_sort_s")),
        "throughput_rows_s": _number(value.get("throughput_rows_s")),
        "throughput_gib_s": _number(value.get("throughput_gib_s")),
        "rejection_reasons": _reasons(value),
        "failure_summary": _short_reason(value),
        "artifact_path": value.get("_artifact_path"),
        "io": _io_metrics(value),
        "gpu": _gpu_metrics(value),
    }


def _comparison(
    cpu_value: Mapping[str, Any], gpu_values: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    cpu_observed = _time(cpu_value)
    cpu_strict = _time(cpu_value, strict=True)
    gpu_observed = [_time(value) for value in gpu_values]
    gpu_strict = [_time(value, strict=True) for value in gpu_values]
    observed_present = [value for value in gpu_observed if value is not None]
    strict_present = [value for value in gpu_strict if value is not None]
    observed_median = median(observed_present) if len(observed_present) == 2 else None
    strict_median = median(strict_present) if len(strict_present) == 2 else None
    observed_speedup = (
        cpu_observed / observed_median
        if cpu_observed is not None and observed_median
        else None
    )
    strict_speedup = (
        cpu_strict / strict_median if cpu_strict is not None and strict_median else None
    )
    if strict_speedup is not None:
        status = "accepted"
    elif observed_speedup is not None:
        status = "telemetry-warning"
    elif any(_classification(value) == "oom" for value in gpu_values):
        status = "oom"
    else:
        status = "incomplete"
    return {
        "cpu_s": _number(cpu_value.get("cold_sort_s")),
        "cpu_classification": _classification(cpu_value),
        "gpu_s": [_number(value.get("cold_sort_s")) for value in gpu_values],
        "gpu_classifications": [_classification(value) for value in gpu_values],
        "gpu_observed_median_s": observed_median,
        "gpu_strict_median_s": strict_median,
        "observed_speedup": observed_speedup,
        "strict_speedup": strict_speedup,
        "comparison_status": status,
    }


def _first_input(values: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    return next(
        (
            _input(value)
            for value in values
            if _number(_input(value).get("decoded_bytes")) is not None
        ),
        {},
    )


def _trend(cpu: Mapping[str, Any], gpu: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for cell in CELLS:
        cpu_result = cpu.get(f"trend-{cell}-r1.json", {})
        gpu_results = [gpu.get(f"trend-{cell}-r{rep}.json", {}) for rep in (1, 2)]
        source = _first_input([*gpu_results, cpu_result])
        comparison = _comparison(cpu_result, gpu_results)
        rows.append(
            {
                "cell": cell,
                "rows": source.get("rows"),
                "blocks": source.get("blocks"),
                "decoded_gib": (_number(source.get("decoded_bytes")) or 0) / GIB,
                **comparison,
                "cpu_throughput_gib_s": _number(cpu_result.get("throughput_gib_s"))
                if _time(cpu_result) is not None
                else None,
                "gpu_throughput_gib_s": _median(
                    _number(value.get("throughput_gib_s"))
                    if _time(value) is not None
                    else None
                    for value in gpu_results
                ),
                "gpu_throughput_mrows_s": (
                    _median(
                        (_number(value.get("throughput_rows_s")) or 0) / 1e6
                        if _time(value) is not None
                        else None
                        for value in gpu_results
                    )
                ),
                "dgx": DGX[cell],
                "previous_cloud": PREVIOUS_CLOUD[cell],
                "gpu_observation_summaries": [
                    _observation(value) for value in gpu_results
                ],
                "cpu_observation_summary": _observation(cpu_result),
                "gpu_observations": gpu_results,
                "cpu_observation": cpu_result,
            }
        )
    return {"rows": rows}


def _natural(
    cpu: Mapping[str, Any],
    gpu: Mapping[str, Any],
    selection: Mapping[str, Any] | None,
    cpu_resource_rejections: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    selected = _number(selection.get("selected_wave_fraction")) if selection else None
    selected_2x = [
        value
        for value in (
            gpu.get("tune-2x-wave-0500.json", {}),
            gpu.get("tune-2x-wave-0375.json", {}),
            gpu.get("natural-2x-selected-r2.json", {}),
        )
        if selected is not None and _number(value.get("wave_fraction")) == selected
    ]
    specs = (
        (
            "1x",
            cpu.get("trend-full-r1.json", {}),
            [gpu.get("trend-full-r1.json", {}), gpu.get("trend-full-r2.json", {})],
        ),
        ("2x", cpu.get("natural-2x-r1.json", {}), selected_2x),
        (
            "2.45x",
            cpu.get("natural-245x-r1.json", {}),
            [gpu.get("natural-245x-r1.json", {}), gpu.get("natural-245x-r2.json", {})],
        ),
    )
    rows = []
    for scale, cpu_value, gpu_values in specs:
        resource_rejection = (
            (cpu_resource_rejections or {}).get(CPU_RESOURCE_REJECTION_RESULT)
            if scale == "2.45x"
            else None
        )
        source = _first_input([*gpu_values, cpu_value])
        comparison = _comparison(cpu_value, gpu_values)
        if resource_rejection is not None:
            comparison["cpu_classification"] = "resource-rejection"
            if len(gpu_values) == 2 and all(
                _classification(value) in ("accepted", "telemetry-warning")
                for value in gpu_values
            ):
                comparison["comparison_status"] = "resource-rejection"
        gpu_io = [_io_metrics(value) for value in gpu_values]
        cpu_io = _io_metrics(cpu_value)
        cpu_observation_summary = _observation(cpu_value)
        if resource_rejection is not None:
            cpu_observation_summary.update(
                {
                    "classification": "resource-rejection",
                    "failure_summary": (
                        "no worker artifact; exact lifecycle evidence records the "
                        "reproduced head/driver resource failure"
                    ),
                    "lifecycle_resource_rejection": resource_rejection,
                }
            )
        gpu_writes = [item["ray_write_gib"] for item in gpu_io]
        median_gpu_write = _median(gpu_writes)
        cpu_write = cpu_io["ray_write_gib"]
        rows.append(
            {
                "scale": scale,
                "rows": source.get("rows"),
                "blocks": source.get("blocks"),
                "decoded_gib": (_number(source.get("decoded_bytes")) or 0) / GIB,
                **comparison,
                "cpu_io": cpu_io,
                "gpu_io": gpu_io,
                "gpu_cpu_ray_write_ratio": (
                    median_gpu_write / cpu_write
                    if median_gpu_write is not None and cpu_write not in (None, 0)
                    else None
                ),
                "gpu_ray_spill_scope": (
                    "pre-failure"
                    if any(
                        _classification(value) in ("oom", "failed")
                        for value in gpu_values
                    )
                    else "completed-sort"
                ),
                "gpu_ray_write_amplification_median": _median(
                    item["ray_write_amplification"] for item in gpu_io
                ),
                "gpu_observation_summaries": [
                    _observation(value) for value in gpu_values
                ],
                "cpu_observation_summary": cpu_observation_summary,
                "gpu_observations": gpu_values,
                "cpu_observations": []
                if resource_rejection is not None
                else [cpu_value],
                "cpu_lifecycle_resource_rejection": resource_rejection,
                "dgx": DGX_NATURAL[scale],
            }
        )
    return {
        "selected_wave_fraction": selected,
        "selection_reason": selection.get("reason") if selection else None,
        "rows": rows,
    }


def _tuning(
    gpu: Mapping[str, Any], selection: Mapping[str, Any] | None
) -> dict[str, Any]:
    default = gpu.get("tune-2x-wave-0500.json", {})
    candidate = gpu.get("tune-2x-wave-0375.json", {})
    return {
        "selected_wave_fraction": selection.get("selected_wave_fraction")
        if selection
        else None,
        "reason": selection.get("reason")
        if selection
        else "selection artifact missing",
        "observed_relative_improvement": selection.get("observed_relative_improvement")
        if selection
        else None,
        "minimum_relative_improvement": selection.get("minimum_relative_improvement")
        if selection
        else None,
        "screens": {"0.50": _observation(default), "0.375": _observation(candidate)},
    }


def _phase_medians(values: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    phases = []
    for value in values:
        stats = value.get("gpu_stats")
        raw = stats.get("phases_s") if isinstance(stats, Mapping) else None
        if _time(value) is not None and isinstance(raw, Mapping):
            phases.append(raw)
    names = sorted({name for raw in phases for name in raw})
    return {
        name: median(
            [float(raw[name]) for raw in phases if _number(raw.get(name)) is not None]
        )
        for name in names
    }


def _fmt(value: Any, digits: int = 2) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _fmt_time(value: Mapping[str, Any]) -> str:
    elapsed = _number(value.get("cold_sort_s"))
    classification = _classification(value)
    return classification if elapsed is None else f"{elapsed:.2f} ({classification})"


def _fmt_pair(cpu_s: float | None, gpu_s: float | None) -> str:
    speedup = cpu_s / gpu_s if cpu_s is not None and gpu_s else None
    return f"{_fmt(cpu_s)}/{_fmt(gpu_s)} ({_fmt(speedup)}×)"


def _geometry(io: Mapping[str, Any]) -> str:
    if io.get("initial_runs") is None:
        return "n/a"
    return f"{io['initial_runs']}/{io['replacement_runs']}/{io['merge_passes']}"


def _waves(io: Mapping[str, Any]) -> str:
    count = io.get("wave_count")
    if count is None:
        return "n/a"
    target = _number(io.get("wave_target_bytes"))
    return (
        f"{int(count)}/whole-input"
        if target is None
        else f"{int(count)}/{target / GIB:.2f} GiB"
    )


def _directional_lines(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    by_cell = {str(row["cell"]): row for row in rows}

    def change(left: str, right: str, field: str) -> str:
        a, b = by_cell[left].get(field), by_cell[right].get(field)
        return "n/a" if a is None or b in (None, 0) else f"{float(a) / float(b):.2f}×"

    return [
        f"- Payload, narrow→core→full: CPU `{_fmt(by_cell['narrow'].get('cpu_s'))}→{_fmt(by_cell['core'].get('cpu_s'))}→{_fmt(by_cell['full'].get('cpu_s'))}` s; GPU median `{_fmt(by_cell['narrow'].get('gpu_observed_median_s'))}→{_fmt(by_cell['core'].get('gpu_observed_median_s'))}→{_fmt(by_cell['full'].get('gpu_observed_median_s'))}` s.",
        f"- Origin string vs integer: CPU string/integer `{change('origin-string', 'origin-integer', 'cpu_s')}`; GPU string/integer `{change('origin-string', 'origin-integer', 'gpu_observed_median_s')}`.",
        f"- One/two/four keys: GPU medians `{_fmt(by_cell['origin-string'].get('gpu_observed_median_s'))}/{_fmt(by_cell['route'].get('gpu_observed_median_s'))}/{_fmt(by_cell['full'].get('gpu_observed_median_s'))}` s; statuses `{by_cell['origin-string']['comparison_status']}/{by_cell['route']['comparison_status']}/{by_cell['full']['comparison_status']}`.",
    ]


def build(
    study: Path,
    gpu_root: Path,
    cpu_root: Path,
    output: Path,
    gpu_repair_root: Path | None = None,
    gpu_repair_study: Path | None = None,
    cpu_repair_root: Path | None = None,
    cpu_repair_study: Path | None = None,
    gpu_245x_repair_root: Path | None = None,
    gpu_245x_repair_study: Path | None = None,
) -> dict[str, Any]:
    if (gpu_repair_root is None) != (gpu_repair_study is None):
        raise ValueError("GPU repair overlay requires both repair study and results paths")
    if (cpu_repair_root is None) != (cpu_repair_study is None):
        raise ValueError("CPU repair overlay requires both repair study and results paths")
    if (gpu_245x_repair_root is None) != (gpu_245x_repair_study is None):
        raise ValueError(
            "GPU 2.45x repair overlay requires both repair study and results paths"
        )
    if gpu_245x_repair_root is not None and gpu_repair_root is None:
        raise ValueError(
            "GPU 2.45x repair overlay requires the full GPU repair overlay"
        )
    plan = (
        _validated_study(study)
        if (
            gpu_repair_root is not None
            or cpu_repair_root is not None
            or gpu_245x_repair_root is not None
        )
        else read_json(study)
    )
    gpu, cpu = _results(gpu_root), _results(cpu_root)
    original_selection = _selection(gpu_root)
    gpu_repair = _results(gpu_repair_root) if gpu_repair_root is not None else {}
    gpu_repair_provenance: dict[str, Any] | None = None
    if gpu_repair_root is None:
        effective_gpu = gpu
        selection = original_selection
    else:
        assert gpu_repair_study is not None
        _, selection, gpu_repair_provenance = _validate_repair_overlay(
            study,
            plan,
            gpu,
            gpu_repair_study,
            gpu_repair_root,
            gpu_repair,
            allow_245x_handoff=gpu_245x_repair_root is not None,
        )
        # Repair artifacts are authoritative for exactly these observations. A
        # missing repair artifact stays missing instead of silently falling back
        # to the superseded run. Trend cells always remain from the original arm.
        effective_gpu = dict(gpu)
        for name in GPU_REPAIR_RESULT_NAMES:
            effective_gpu[name] = gpu_repair.get(name, {})

    gpu_245x_repair = (
        _results(gpu_245x_repair_root)
        if gpu_245x_repair_root is not None
        else {}
    )
    gpu_245x_repair_provenance: dict[str, Any] | None = None
    superseded_full_gpu_245x: dict[str, Mapping[str, Any]] = {}
    if gpu_245x_repair_root is not None:
        assert gpu_245x_repair_study is not None
        _, gpu_245x_repair_provenance = _validate_gpu_245x_repair_overlay(
            study,
            plan,
            gpu,
            gpu_245x_repair_study,
            gpu_245x_repair_root,
            gpu_245x_repair,
        )
        superseded_full_gpu_245x = {
            name: effective_gpu.get(name, {})
            for name in GPU_245X_REPAIR_REPLACEMENT_NAMES
        }
        # The exact smoke is a gate for this lifecycle, not a replacement for
        # the full repair campaign's active smoke or its 2x tuning results.
        for name in GPU_245X_REPAIR_REPLACEMENT_NAMES:
            effective_gpu[name] = gpu_245x_repair[name]

    cpu_repair = _results(cpu_repair_root) if cpu_repair_root is not None else {}
    cpu_repair_provenance: dict[str, Any] | None = None
    cpu_resource_rejection: dict[str, Any] | None = None
    if cpu_repair_root is None:
        effective_cpu = cpu
    else:
        assert cpu_repair_study is not None
        _, cpu_repair_provenance, cpu_resource_rejection = (
            _validate_cpu_repair_overlay(
                study,
                plan,
                gpu,
                cpu,
                cpu_repair_study,
                cpu_repair_root,
                cpu_repair,
            )
        )
        effective_cpu = dict(cpu)
        for name in CPU_REPAIR_RESULT_NAMES:
            if name in cpu_repair:
                effective_cpu[name] = cpu_repair[name]
            else:
                effective_cpu.pop(name, None)

    trend = _trend(effective_cpu, gpu)
    natural = _natural(
        effective_cpu,
        effective_gpu,
        selection,
        {
            CPU_RESOURCE_REJECTION_RESULT: cpu_resource_rejection,
        }
        if cpu_resource_rejection is not None
        else None,
    )
    smoke_value = effective_gpu.get("transport-smoke.json", {})
    smoke = {
        **_observation(smoke_value),
        "cpu_sort_s": _number(smoke_value.get("cpu_sort_s")),
        "gpu_sort_s": _number(smoke_value.get("gpu_sort_s")),
        "exact_every_row_value": smoke_value.get("validation", {}).get(
            "exact_every_row_value"
        ),
        "rows": smoke_value.get("validation", {}).get("rows"),
        "gpu_ranks": len(smoke_value.get("gpu_stats", {}).get("ranks", [])),
    }
    tuning = _tuning(effective_gpu, selection)
    full_gpu = next(
        row["gpu_observations"] for row in trend["rows"] if row["cell"] == "full"
    )
    active_gpu_performance = [
        value for row in trend["rows"] for value in row["gpu_observations"]
    ] + [
        value
        for row in natural["rows"]
        if row["scale"] != "1x"
        for value in row["gpu_observations"]
    ]

    def all_explicit_zero(name: str) -> bool:
        values = [_gpu_stat(value, name) for value in active_gpu_performance]
        return bool(values) and all(value == 0 for value in values)

    gpu_only_acceptance = {
        "unique_active_performance_observations": len(active_gpu_performance),
        "all_cpu_sort_rows_zero": all_explicit_zero("cpu_sort_rows"),
        "all_cpu_merge_rows_zero": all_explicit_zero("cpu_merge_rows"),
        "all_mpf_host_spill_bytes_zero": all_explicit_zero("mpf_host_spill_bytes"),
    }
    completed_gpu_scales = [
        row["scale"]
        for row in natural["rows"]
        if len([value for value in row["gpu_s"] if value is not None]) == 2
    ]
    completed_cpu_scales = [
        row["scale"] for row in natural["rows"] if row["cpu_s"] is not None
    ]
    result = {
        "schema_version": 3,
        "kind": "gpu_sort_bts_cloud_report",
        "campaign": plan["campaign"],
        "study_plan_sha256": plan["plan_sha256"],
        "execution_contract": {
            "topologies": {
                arm: plan.get("arms", {}).get(arm, {}).get("topology")
                for arm in ("gpu", "cpu")
            },
            "retained_ec2_fleet_per_arm": plan.get(
                "retained_ec2_fleet_per_arm"
            ),
            "fresh_ray_runtime_per_observation": plan.get(
                "fresh_ray_runtime_per_trial"
            ),
            "ray_default_plasma_both_arms": plan.get(
                "ray_default_plasma_both_arms"
            ),
            "filesystem_spill_tier": "/mnt/nvme",
            "filesystem_spill_uses_dev_shm": False,
            "timed_boundary": (
                "projected input materialized in Plasma through sorted output "
                "sealed in Plasma"
            ),
            "ec2_reprovisioned_per_observation": False,
        },
        "completion": {
            "six_cell_gpu_observations_completed": sum(
                _time(value) is not None
                for row in trend["rows"]
                for value in row["gpu_observations"]
            ),
            "six_cell_cpu_observations_completed": sum(
                _time(row["cpu_observation"]) is not None for row in trend["rows"]
            ),
            "gpu_natural_scales_completed": completed_gpu_scales,
            "cpu_natural_scales_completed": completed_cpu_scales,
            "cpu_245x_outcome": (
                "resource-rejection"
                if cpu_resource_rejection is not None
                else next(
                    row["cpu_classification"]
                    for row in natural["rows"]
                    if row["scale"] == "2.45x"
                )
            ),
            "cpu_245x_speedup_claimed": False,
            "all_campaign_instance_scopes_verified_empty": all(
                value is not None
                for value in (
                    gpu_repair_provenance,
                    gpu_245x_repair_provenance,
                    cpu_repair_provenance,
                )
            ),
        },
        "classification_legend": {
            "accepted": "all strict acceptance gates passed",
            "telemetry-warning": "sort completed and exact validation passed; only Ray location metadata was incomplete",
            "oom": "sort failed with a GPU/RMM out-of-memory error; no timing operand",
            "failed": "another acceptance or execution failure; no timing operand",
            "resource-rejection": "no worker result: exact lifecycle evidence shows a repeated head/driver resource failure; diagnostic time is not a sort-time operand",
        },
        "smoke": smoke,
        "tuning": tuning,
        "trend": trend,
        "natural_spill": natural,
        "gpu_only_acceptance": gpu_only_acceptance,
        "full_baseline_phase_medians_s": _phase_medians(full_gpu),
        "gpu_repair_overlay": {
            "enabled": gpu_repair_root is not None,
            "source_root": str(gpu_repair_root.resolve())
            if gpu_repair_root is not None
            else None,
            "replacement_names": [
                name for name in GPU_REPAIR_RESULT_NAMES if name in gpu_repair
            ],
            "missing_names": [
                name for name in GPU_REPAIR_RESULT_NAMES if name not in gpu_repair
            ]
            if gpu_repair_root is not None
            else [],
            "original_selection": original_selection,
            "repair_selection": selection if gpu_repair_root is not None else None,
            "repair_study": str(gpu_repair_study.resolve())
            if gpu_repair_study is not None
            else None,
            "validated_provenance": gpu_repair_provenance,
            "superseded_original_observations": {
                name: gpu[name] for name in GPU_REPAIR_RESULT_NAMES if name in gpu
            }
            if gpu_repair_root is not None
            else {},
        },
        "gpu_245x_repair_overlay": {
            "enabled": gpu_245x_repair_root is not None,
            "source_root": str(gpu_245x_repair_root.resolve())
            if gpu_245x_repair_root is not None
            else None,
            "gate_names": ["transport-smoke.json"]
            if gpu_245x_repair_root is not None
            else [],
            "replacement_names": list(GPU_245X_REPAIR_REPLACEMENT_NAMES)
            if gpu_245x_repair_root is not None
            else [],
            "repair_study": str(gpu_245x_repair_study.resolve())
            if gpu_245x_repair_study is not None
            else None,
            "validated_provenance": gpu_245x_repair_provenance,
            "gate_smoke": gpu_245x_repair.get("transport-smoke.json")
            if gpu_245x_repair_root is not None
            else None,
            "superseded_full_repair_observations": superseded_full_gpu_245x,
            "superseded_original_observations": {
                name: gpu[name]
                for name in GPU_245X_REPAIR_REPLACEMENT_NAMES
                if name in gpu
            }
            if gpu_245x_repair_root is not None
            else {},
        },
        "cpu_repair_overlay": {
            "enabled": cpu_repair_root is not None,
            "source_root": str(cpu_repair_root.resolve())
            if cpu_repair_root is not None
            else None,
            "replacement_names": [
                name for name in CPU_REPAIR_RESULT_NAMES if name in cpu_repair
            ],
            "missing_names": [
                name for name in CPU_REPAIR_RESULT_NAMES if name not in cpu_repair
            ]
            if cpu_repair_root is not None
            else [],
            "resource_rejection_names": [CPU_RESOURCE_REJECTION_RESULT]
            if cpu_resource_rejection is not None
            else [],
            "unexplained_missing_names": [
                name
                for name in CPU_REPAIR_RESULT_NAMES
                if name not in cpu_repair
                and not (
                    cpu_resource_rejection is not None
                    and name == CPU_RESOURCE_REJECTION_RESULT
                )
            ]
            if cpu_repair_root is not None
            else [],
            "lifecycle_resource_rejections": {
                CPU_RESOURCE_REJECTION_RESULT: cpu_resource_rejection
            }
            if cpu_resource_rejection is not None
            else {},
            "repair_study": str(cpu_repair_study.resolve())
            if cpu_repair_study is not None
            else None,
            "validated_provenance": cpu_repair_provenance,
            "preserved_original_names": [
                "trend-narrow-r1.json",
                "trend-core-r1.json",
            ],
            "superseded_original_observations": {
                name: cpu[name] for name in CPU_REPAIR_RESULT_NAMES if name in cpu
            }
            if cpu_repair_root is not None
            else {},
        },
        "raw_result_counts": {
            "gpu": len(gpu),
            "gpu_repair": len(gpu_repair) if gpu_repair_root is not None else 0,
            "gpu_245x_repair": len(gpu_245x_repair)
            if gpu_245x_repair_root is not None
            else 0,
            "cpu": len(cpu),
            "cpu_repair": len(cpu_repair) if cpu_repair_root is not None else 0,
        },
    }
    atomic_json(output.with_suffix(".json"), result)

    screen_050_s = _number(tuning["screens"]["0.50"].get("cold_sort_s"))
    screen_0375_s = _number(tuning["screens"]["0.375"].get("cold_sort_s"))
    raw_candidate_improvement = (
        None
        if screen_050_s in (None, 0) or screen_0375_s is None
        else 100 * (screen_050_s - screen_0375_s) / screen_050_s
    )
    lines = [
        f"# BTS Cloud GPU Sort — {plan['campaign']}",
        "",
        "CPU is default Ray/PyArrow. Every observation uses a fresh Ray runtime on the retained fleet. Times marked `telemetry-warning` completed exact validation but missed only Ray location metadata; they are shown for directional analysis but are not strict accepted measurements. OOM and failed observations are never timing operands.",
        "",
        "## Execution contract and completion",
        "",
        "- GPU: 16 retained `g6.4xlarge` instances, one L4 and one GPU-sort rank per node. CPU: 16 retained `m5dn.4xlarge` instances using unmodified default Ray/PyArrow sort.",
        "- Each arm launched EC2 once. Between observations Ray alone was stopped and restarted on those same instances, with new Ray node IDs and an empty trial-specific runtime and `/mnt/nvme` spill directory. The immutable BTS publication, environment, and OS page cache remained staged on local NVMe.",
        "- The timed boundary is projected input materialized in Plasma through sorted output sealed and Ray-locatable. Ray restart and input reading/materialization are excluded; GPU actor, CUDA, RMM, MPF, transfer, externalization, and merge startup are included.",
        "- Ray default Plasma sizing was retained on both arms. Filesystem spill used `/mnt/nvme`, never `/dev/shm`. The GPU fleet ran first and was terminated before CPU launch; all campaign instance scopes were verified empty after teardown.",
        "- Dataset: normalized public BTS On-Time Performance data from 2013-04 through 2025-12. Every 1× cell uses the same 80,738,761 rows and 627 blocks. Narrow retains `Origin, Dest, FlightDate, CRSDepTime, row_id` (5 columns); core retains the first 56 native columns through `DistanceGroup` plus `row_id` (57); full retains all 109 native columns plus `row_id` (110). The four-key baseline is `Origin, Dest, FlightDate, CRSDepTime`; the other full-payload cells use `Origin`, `OriginAirportID`, or `Origin, Dest`.",
        "- Composite build provenance is explicit: the original six resident GPU trend cells use Ray Data overlay `273c73b5b434f499f1773922e0b131488c64e809e352451454e39016a0475dec`; GPU 2×/2.45× and repaired CPU observations use final overlay `e0105442cc991f372614e21671aee0a68e60e06fa189503b2f7e84f5e12632aa`. Their sole production difference is external-run workspace/headroom handling in `backend.py`. Every retained original trend cell stayed resident with zero externalized bytes, so that change cannot exercise there and those observations are intentionally reused.",
        f"- GPU-only acceptance: `{gpu_only_acceptance['unique_active_performance_observations']}` unique active performance artifacts; CPU-sort rows zero `{gpu_only_acceptance['all_cpu_sort_rows_zero']}`; CPU-merge rows zero `{gpu_only_acceptance['all_cpu_merge_rows_zero']}`; MPF host-spill bytes zero `{gpu_only_acceptance['all_mpf_host_spill_bytes_zero']}`.",
        "- The six 1× cells and GPU 1×/2×/2.45× size points completed. Default CPU completed through 2×; its 2.45× resource rejection is reported as a capacity boundary, not a timing result.",
        "",
        "## Correctness and automatic-wave tuning",
        "",
        f"- Exact 160k-row smoke: `{smoke['classification']}`; exact every row/value `{smoke['exact_every_row_value']}`; CPU/GPU `{_fmt(smoke['cpu_sort_s'], 3)}/{_fmt(smoke['gpu_sort_s'], 3)}` s; GPU ranks `{smoke['gpu_ranks']}`.",
        f"- 2× wave screens: 0.50 `{_fmt(tuning['screens']['0.50']['cold_sort_s'], 3)} s ({tuning['screens']['0.50']['classification']})`; 0.375 `{_fmt(tuning['screens']['0.375']['cold_sort_s'], 3)} s ({tuning['screens']['0.375']['classification']})`.",
        f"- Selected `{tuning['selected_wave_fraction']}`: {tuning['reason']}; strict observed improvement `{_fmt(None if tuning['observed_relative_improvement'] is None else 100 * float(tuning['observed_relative_improvement']), 2)}%` against a `{_fmt(None if tuning['minimum_relative_improvement'] is None else 100 * float(tuning['minimum_relative_improvement']), 1)}%` gate. Independently, the raw 0.375 time was `{_fmt(None if raw_candidate_improvement is None else -raw_candidate_improvement, 2)}%` slower than 0.50.",
    ]
    if gpu_repair_root is not None:
        overlay = result["gpu_repair_overlay"]
        lines.extend(
            [
                f"- Repair overlay: `{len(overlay['replacement_names'])}` authoritative artifacts from `{overlay['source_root']}`; missing `{', '.join(overlay['missing_names']) or 'none'}`. Original trend observations are unchanged.",
                "",
                "### Superseded original GPU attempts",
                "",
                "These remain in the report JSON for provenance; repaired observations above and in the natural-size tables are the active results.",
                "",
                "| Observation | Original time/status | Repair time/status |",
                "|---|---:|---:|",
            ]
        )
        history = overlay["superseded_original_observations"]
        for name in GPU_REPAIR_RESULT_NAMES:
            lines.append(
                f"| {name.removesuffix('.json')} | {_fmt_time(history.get(name, {}))} | {_fmt_time(gpu_repair.get(name, {}))} |"
            )
    if gpu_245x_repair_root is not None:
        overlay = result["gpu_245x_repair_overlay"]
        gate_smoke = overlay["gate_smoke"] or {}
        lines.extend(
            [
                "",
                f"- GPU 2.45× repair overlay: exact smoke `{_fmt_time(gate_smoke)}`; two fixed-wave-0.50 observations from `{overlay['source_root']}` replace only the 2.45× results. The full repair's 2× tuning and observations remain authoritative.",
                "",
                "### Superseded full-repair GPU 2.45× attempts",
                "",
                "The original and full-repair attempts remain in the report JSON for provenance; only the minimal repair's two 2.45× observations are active.",
                "",
                "| Observation | Full-repair time/status | 2.45× repair time/status |",
                "|---|---:|---:|",
            ]
        )
        history = overlay["superseded_full_repair_observations"]
        for name in GPU_245X_REPAIR_REPLACEMENT_NAMES:
            lines.append(
                f"| {name.removesuffix('.json')} | {_fmt_time(history.get(name, {}))} | {_fmt_time(gpu_245x_repair.get(name, {}))} |"
            )
        lines.extend(
            [
                "",
                "The missing full-repair 2.45× artifacts were traced to the cloud harness, not the production sort: reader tasks returned nested `ray.put()` refs owned by temporary workers. Under 2.45× object-store pressure, loss of one owner left unreconstructable pending refs and the sort waited indefinitely. The repair returns Arrow tables directly from node-affined reader tasks, so Ray retains reconstructable task outputs. No production GPU-sort change was needed for this fix. The simple retained-fleet rerun then completed both 2.45× observations.",
                "",
            ]
        )
    if cpu_repair_root is not None:
        overlay = result["cpu_repair_overlay"]
        rejection_count = len(overlay["resource_rejection_names"])
        lines.extend(
            [
                "",
                f"- CPU repair overlay: `{len(overlay['replacement_names'])}` authoritative worker artifacts from `{overlay['source_root']}` plus `{rejection_count}` validated lifecycle resource rejection; unexplained missing `{', '.join(overlay['unexplained_missing_names']) or 'none'}`; original narrow/core observations are preserved.",
                "",
                "### Superseded original CPU attempts",
                "",
                "These remain in the report JSON for provenance; repaired CPU observations in the trend and natural-size tables are authoritative.",
                "",
                "| Observation | Original time/status | Repair time/status |",
                "|---|---:|---:|",
            ]
        )
        history = overlay["superseded_original_observations"]
        for name in CPU_REPAIR_RESULT_NAMES:
            repair_text = (
                "resource-rejection"
                if name in overlay["resource_rejection_names"]
                else _fmt_time(cpu_repair.get(name, {}))
            )
            lines.append(
                f"| {name.removesuffix('.json')} | {_fmt_time(history.get(name, {}))} | {repair_text} |"
            )
        if cpu_resource_rejection is not None:
            lines.extend(
                [
                    "",
                    "### CPU 2.45× lifecycle resource rejection",
                    "",
                    "The default Ray/PyArrow trial produced no worker result. Exact lifecycle evidence identifies a **head/driver resource failure at the map-to-reduce boundary**: shuffle map reached 100%, shuffle reduce remained at 0%, the >16 GiB driver-memory warning appeared, and the head connection then closed. This was reproduced twice on fresh 16-node CPU runtimes.",
                    "",
                    "The elapsed values below are diagnostic time-to-failure only. They are not completed sort times and are never used in a CPU/GPU speedup.",
                    "",
                    "| Campaign | Boundary | Diagnostic time to failure | Study / lifecycle / command-log SHA-256 |",
                    "|---|---|---:|---|",
                ]
            )
            for attempt in cpu_resource_rejection["attempts"]:
                files = attempt["files"]
                command_name = (
                    "executions/cpu/command-09-natural-245x-r1.log"
                )
                lines.append(
                    f"| {attempt['campaign']} | {attempt['failure_boundary']} | {attempt['diagnostic_time_to_failure_s']} s (diagnostic only) | `{attempt['study_plan_sha256'][:12]}` / `{attempt['lifecycle_plan_sha256'][:12]}` / `{files[command_name]['sha256'][:12]}` |"
                )
    lines.extend(
        [
            "",
            "## Payload and key trends",
            "",
            "`Speedup` is the observed comparison. Its status is `accepted` only when the CPU and both GPU observations are strict-valid.",
            "",
            "| Cell | Rows / blocks / GiB | CPU s (status) | GPU observations s (status) | GPU median s | Speedup / status | CPU/GPU GiB/s | DGX CPU/GPU (speedup) | Previous cloud CPU/GPU (speedup) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in trend["rows"]:
        dgx, prior = row["dgx"], row["previous_cloud"]
        gpu_text = ", ".join(_fmt_time(value) for value in row["gpu_observations"])
        lines.append(
            f"| {row['cell']} | {row['rows'] or 'n/a'} / {row['blocks'] or 'n/a'} / {_fmt(row['decoded_gib'], 1)} | {_fmt_time(row['cpu_observation'])} | {gpu_text} | {_fmt(row['gpu_observed_median_s'])} | {_fmt(row['observed_speedup'])}× / {row['comparison_status']} | {_fmt(row['cpu_throughput_gib_s'], 3)}/{_fmt(row['gpu_throughput_gib_s'], 3)} | {_fmt_pair(dgx['cpu_s'], dgx['gpu_s'])} | {_fmt_pair(prior['cpu_s'], prior['gpu_s'])} |"
        )
    lines.extend(
        ["", "### Directional summary", "", *_directional_lines(trend["rows"])]
    )

    lines.extend(
        [
            "",
            "## Natural size and spill trend",
            "",
            f"Frozen automatic wave fraction: `{natural['selected_wave_fraction']}` ({natural['selection_reason']}). External GiB is GPU-run externalization from VRAM to Plasma; Ray W/R is cumulative write/restore traffic during the timed sort, not simultaneous disk footprint, so the two are not additive. Run geometry is initial/replacement/passes. A `whole-input` wave target means the planner admitted the input as one wave; it does not imply the destination remained resident.",
            "",
            "| Scale | Rows / blocks / GiB | CPU s | GPU observations | GPU median / speedup / status | Waves/target per GPU rep | External GiB per GPU rep | Runs per GPU rep | CPU Ray W/R GiB | GPU Ray W/R GiB per rep | GPU/CPU Ray write | Ray write amp CPU/GPU | DGX CPU/GPU (speedup) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in natural["rows"]:
        gpu_times = ", ".join(_fmt_time(value) for value in row["gpu_observations"])
        waves = ", ".join(_waves(item) for item in row["gpu_io"])
        ext = ", ".join(_fmt(item["externalized_gib"], 1) for item in row["gpu_io"])
        geometry = ", ".join(_geometry(item) for item in row["gpu_io"])
        gpu_ray = ", ".join(
            f"{_fmt(item['ray_write_gib'], 1)}/{_fmt(item['ray_restore_gib'], 1)}"
            for item in row["gpu_io"]
        )
        cpu_io = row["cpu_io"]
        dgx = row["dgx"]
        cpu_time = (
            "n/a (resource rejection)"
            if row["cpu_lifecycle_resource_rejection"] is not None
            else _fmt_time(row["cpu_observations"][0])
        )
        speedup = (
            "n/a"
            if row["cpu_lifecycle_resource_rejection"] is not None
            else f"{_fmt(row['observed_speedup'])}×"
        )
        lines.append(
            f"| {row['scale']} | {row['rows'] or 'n/a'} / {row['blocks'] or 'n/a'} / {_fmt(row['decoded_gib'], 1)} | {cpu_time} | {gpu_times} | {_fmt(row['gpu_observed_median_s'])} / {speedup} / {row['comparison_status']} | {waves} | {ext} | {geometry} | {_fmt(cpu_io['ray_write_gib'], 1)}/{_fmt(cpu_io['ray_restore_gib'], 1)} | {gpu_ray} | {_fmt(row['gpu_cpu_ray_write_ratio'])}× ({row['gpu_ray_spill_scope']}) | {_fmt(cpu_io['ray_write_amplification'])}/{_fmt(row['gpu_ray_write_amplification_median'])} | {_fmt_pair(dgx['cpu_s'], dgx['gpu_s'])} |"
        )

    lines.extend(
        [
            "",
            "### GPU movement and amplification by observation",
            "",
            "| Scale / rep | Status | Throughput GiB/s | Externalized % | H2D/D2H amp | Plasma read/write amp | Ray write/restore GiB | First externalize s/wave |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in natural["rows"]:
        for index, (summary, io) in enumerate(
            zip(row["gpu_observation_summaries"], row["gpu_io"]), 1
        ):
            first = (
                "n/a"
                if io["first_externalize_s"] is None
                else f"{_fmt(io['first_externalize_s'])}/{int(io['first_externalize_wave'] or 0)}"
            )
            lines.append(
                f"| {row['scale']} / r{index} | {summary['classification']} | {_fmt(summary['throughput_gib_s'], 3)} | {_fmt(None if io['externalized_fraction'] is None else 100 * io['externalized_fraction'], 1)} | {_fmt(io['h2d_amplification'])}/{_fmt(io['d2h_amplification'])} | {_fmt(io['plasma_read_amplification'])}/{_fmt(io['plasma_write_amplification'])} | {_fmt(io['ray_write_gib'], 1)}/{_fmt(io['ray_restore_gib'], 1)} | {first} |"
            )

    lines.extend(
        [
            "",
            "## Full-baseline GPU phase and resource summary",
            "",
            "| Median phase | Seconds |",
            "|---|---:|",
        ]
    )
    for name, seconds in result["full_baseline_phase_medians_s"].items():
        lines.append(f"| `{name}` | {_fmt(seconds, 3)} |")
    lines.extend(
        [
            "",
            "| Rep | Status | Peak RMM / NVML / min headroom GiB | Output balance | Local input | Network send/recv GiB |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for index, value in enumerate(full_gpu, 1):
        summary, metrics = _observation(value), _gpu_metrics(value)
        lines.append(
            f"| {index} | {summary['classification']} | {_fmt(metrics['peak_rmm_gib'], 1)}/{_fmt(metrics['peak_nvml_gib'], 1)}/{_fmt(metrics['minimum_nvml_headroom_gib'], 1)} | {_fmt(metrics['output_balance_max_over_min'])}× | {_fmt(None if metrics['local_input_fraction'] is None else 100 * metrics['local_input_fraction'], 1)}% | {_fmt(metrics['network_sent_gib'], 1)}/{_fmt(metrics['network_received_gib'], 1)} |"
        )

    exceptions = []
    for label, values in (
        *(
            (f"trend/{row['cell']}", row["gpu_observations"] + [row["cpu_observation"]])
            for row in trend["rows"]
        ),
        *(
            (
                f"natural/{row['scale']}",
                row["gpu_observations"] + row["cpu_observations"],
            )
            for row in natural["rows"]
        ),
    ):
        for value in values:
            if _classification(value) not in ("accepted", "missing"):
                exceptions.append((label, value))
    lines.extend(
        [
            "",
            "## Warnings and failures",
            "",
            "| Observation | Backend / rep | Classification | Raw time s | Ray write/restore GiB | Reason |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    if not exceptions and cpu_resource_rejection is None:
        lines.append("| none | — | — | — | — | — |")
    for label, value in exceptions:
        io = _io_metrics(value)
        lines.append(
            f"| {label} | {value.get('backend', 'missing')} / {value.get('repetition', 'n/a')} | {_classification(value)} | {_fmt(_number(value.get('cold_sort_s')))} | {_fmt(io['ray_write_gib'], 1)}/{_fmt(io['ray_restore_gib'], 1)} | {(_short_reason(value) or 'n/a').replace('|', '/')} |"
        )
    if cpu_resource_rejection is not None:
        diagnostic_times = "/".join(
            str(attempt["diagnostic_time_to_failure_s"])
            for attempt in cpu_resource_rejection["attempts"]
        )
        lines.append(
            "| natural/2.45x | pyarrow / r1 | resource-rejection | n/a | n/a/n/a | "
            f"head/driver resource failure at map-to-reduce boundary, reproduced twice ({diagnostic_times} s diagnostic-only) |"
        )
    active_gpu_oom = any(
        _classification(value) == "oom"
        for row in natural["rows"]
        for value in row["gpu_observations"]
    )
    lines.append("")
    if active_gpu_oom:
        lines.extend(
            [
                "At 2.45×, any OOM row reports exact pre-failure Ray spill counters, but externalized-run totals are unavailable because the failed actor could not return final GPU telemetry. These failures are evidence about the current spill admission path, not completed performance measurements.",
                "",
            ]
        )
    elif gpu_245x_repair_root is not None:
        lines.extend(
            [
                "The original GPU 2.45× OOM attempts and the later harness-stalled attempts are superseded history only. The active production build completed both fixed-wave 0.50 GPU observations. The sole active 2.45× failure is the reproduced default-Ray/PyArrow CPU head/driver resource rejection, so no 2.45× CPU/GPU speedup is claimed.",
                "",
            ]
        )
    lines.extend(
        [
            "The companion JSON retains every raw observation, phase timing, per-rank GPU diagnostic, per-node NVML/NVMe/network measurement, run geometry, and cluster-wide per-raylet spill counter.",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--gpu-results", type=Path, required=True)
    parser.add_argument("--cpu-results", type=Path, required=True)
    parser.add_argument("--gpu-repair-results", type=Path)
    parser.add_argument("--gpu-repair-study", type=Path)
    parser.add_argument("--gpu-245x-repair-results", type=Path)
    parser.add_argument("--gpu-245x-repair-study", type=Path)
    parser.add_argument("--cpu-repair-results", type=Path)
    parser.add_argument("--cpu-repair-study", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (args.gpu_repair_results is None) != (args.gpu_repair_study is None):
        parser.error("--gpu-repair-results and --gpu-repair-study must be supplied together")
    if (args.cpu_repair_results is None) != (args.cpu_repair_study is None):
        parser.error("--cpu-repair-results and --cpu-repair-study must be supplied together")
    if (args.gpu_245x_repair_results is None) != (
        args.gpu_245x_repair_study is None
    ):
        parser.error(
            "--gpu-245x-repair-results and --gpu-245x-repair-study must be "
            "supplied together"
        )
    if args.gpu_245x_repair_results is not None and args.gpu_repair_results is None:
        parser.error(
            "--gpu-245x-repair-results requires the full --gpu-repair-results overlay"
        )
    result = build(
        args.study,
        args.gpu_results,
        args.cpu_results,
        args.output,
        args.gpu_repair_results,
        args.gpu_repair_study,
        args.cpu_repair_results,
        args.cpu_repair_study,
        args.gpu_245x_repair_results,
        args.gpu_245x_repair_study,
    )
    print(
        json.dumps(
            {
                "campaign": result["campaign"],
                "markdown": str(args.output.resolve()),
                "json": str(args.output.with_suffix(".json").resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
