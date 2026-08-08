"""Create the exact autoscaler YAMLs and immutable remote source bundle."""

from __future__ import annotations

import hashlib
import re
import shlex
import shutil
from pathlib import Path
from typing import Any, Mapping

import yaml

from release.benchmarks.ray_data_gpu_sort.spec import (
    RAY_COMMIT,
    RAY_VERSION,
    RAY_WHEEL,
    RAY_WHEEL_SHA256,
)

from .common import atomic_json, digest, file_sha256

SAFE_CAMPAIGN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,47}$")
REMOTE_SOURCE = Path("/home/ray/gpu-sort-src")
REMOTE_WHEEL = REMOTE_SOURCE / "wheelhouse" / RAY_WHEEL
REMOTE_DATASET_MANIFEST = REMOTE_SOURCE / "generated/dataset-manifest.json"
REMOTE_BASE = Path("/mnt/nvme/ray-gpu-sort")
GPU_IMAGE = "rayproject/ray:2.55.1-py310-gpu@sha256:a1240f249968c8b5e6d4c2b957f824c415fa470b2720a7898e02e90d12247053"
CPU_IMAGE = "rayproject/ray:2.55.1-py310@sha256:bddf81a43131ff29c2371cff648e4125941f8d19cedf0fd97fd9d4d9bbc80372"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _copy_python_tree(source: Path, target: Path) -> None:
    for path in sorted(source.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)


def make_bundle(
    *, root: Path, output: Path, dataset_manifest: Path
) -> dict[str, Any]:
    if output.exists():
        if any(output.iterdir()):
            raise RuntimeError(f"bundle is not empty: {output}")
    else:
        output.mkdir(parents=True)
    _copy_python_tree(root / "python/ray/data", output / "python/ray/data")
    benchmark = root / "release/benchmarks/ray_data_gpu_sort"
    for name in ("__init__.py", "spec.py", "data.py", "backend_stats.py"):
        destination = output / "release/benchmarks/ray_data_gpu_sort" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(benchmark / name, destination)
    _copy_python_tree(benchmark / "cloud", output / "release/benchmarks/ray_data_gpu_sort/cloud")
    for name in (
        "requirements-gpu.lock",
        "requirements-cpu.lock",
        "aws-runtime.requirements.txt",
        "topologies.json",
    ):
        shutil.copy2(
            benchmark / "cloud" / name,
            output / "release/benchmarks/ray_data_gpu_sort/cloud" / name,
        )
    for package in (
        output / "release",
        output / "release/benchmarks",
    ):
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").touch()
    generated = output / "generated"
    generated.mkdir()
    shutil.copy2(dataset_manifest, generated / "dataset-manifest.json")
    wheel = root / ".venv/wheelhouse" / RAY_WHEEL
    if file_sha256(wheel) != RAY_WHEEL_SHA256:
        raise RuntimeError("the stock Ray wheel digest does not match the frozen spec")
    (output / "wheelhouse").mkdir()
    shutil.copy2(wheel, output / "wheelhouse" / RAY_WHEEL)
    files = {
        str(path.relative_to(output)): file_sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    bundle_digest = digest(files)
    identity = {
        "schema_version": 1,
        "kind": "gpu_sort_cloud_bundle",
        "bundle_digest": bundle_digest,
        "files": files,
        "ray_version": RAY_VERSION,
        "ray_commit": RAY_COMMIT,
        "wheel_sha256": RAY_WHEEL_SHA256,
    }
    atomic_json(output / "BUNDLE.json", identity)
    return identity


def _nvme_init(expected_devices: int) -> str:
    return f"""set -euo pipefail
MOUNT=/opt/dlami/nvme
sudo mkdir -p "$MOUNT"
if mountpoint -q "$MOUNT"; then sudo chmod 1777 "$MOUNT"; exit 0; fi
ROOT=$(readlink -f "$(findmnt -nro SOURCE --target /)")
DEVICES=()
for SYS in /sys/block/nvme*n1; do
  [ -r "$SYS/device/model" ] || continue
  [ "$(tr -d '\\000' < "$SYS/device/model" | xargs)" = 'Amazon EC2 NVMe Instance Storage' ] || continue
  DEVICE=/dev/$(basename "$SYS")
  [ "$(readlink -f "$DEVICE")" != "$ROOT" ] || {{ echo 'refusing root device' >&2; exit 1; }}
  DEVICES+=("$DEVICE")
done
[ "${{#DEVICES[@]}}" -eq {expected_devices} ] || {{ echo 'unexpected instance-store count' >&2; exit 1; }}
for DEVICE in "${{DEVICES[@]}}"; do [ -z "$(lsblk -nrpo MOUNTPOINT "$DEVICE" | tr -d '[:space:]')" ] || exit 1; done
if [ "${{#DEVICES[@]}}" -eq 1 ]; then
  TARGET="${{DEVICES[0]}}"
else
  sudo mkdir -p /dev/md
  TARGET=/dev/md/ray_gpu_sort
  sudo mdadm --create "$TARGET" --run --level=0 --raid-devices="${{#DEVICES[@]}}" "${{DEVICES[@]}}"
fi
[ -z "$(lsblk -dnro FSTYPE "$TARGET" || true)" ] && sudo mkfs.ext4 -F -E nodiscard "$TARGET"
sudo mount -t ext4 -o noatime,nodiratime "$TARGET" "$MOUNT"
sudo chmod 1777 "$MOUNT"
test -w "$MOUNT"
""".strip()


def _setup(bundle_digest: str, arm: str) -> tuple[str, str, str]:
    environment = REMOTE_BASE / "envs" / bundle_digest[:24]
    python = environment / ".venv/bin/python"
    ray = environment / ".venv/bin/ray"
    lock = REMOTE_SOURCE / f"release/benchmarks/ray_data_gpu_sort/cloud/requirements-{arm}.lock"
    aws_lock = REMOTE_SOURCE / "release/benchmarks/ray_data_gpu_sort/cloud/aws-runtime.requirements.txt"
    imports = (
        "boto3,cudf,cupy,numpy,pandas,pyarrow,pylibcudf,rapidsmpf,ray,rmm,ucxx"
        if arm == "gpu"
        else "boto3,numpy,pandas,pyarrow,ray"
    )
    command = f"""set -euo pipefail
ENV={shlex.quote(str(environment))}
MARKER="$ENV/READY"
if [ ! -f "$MARKER" ]; then
  [ ! -e "$ENV" ] || {{ echo 'incomplete environment exists' >&2; exit 1; }}
  mkdir -p "$ENV"
  python3.10 -m venv --system-site-packages "$ENV/.venv"
  "$ENV/.venv/bin/python" -m pip install --disable-pip-version-check --no-cache-dir -r {shlex.quote(str(lock))} -r {shlex.quote(str(aws_lock))}
  "$ENV/.venv/bin/python" -m pip install --disable-pip-version-check --no-cache-dir --no-deps --force-reinstall {shlex.quote(str(REMOTE_WHEEL))}
  RAY_FILE=$("$ENV/.venv/bin/python" -c 'import pathlib,ray; print(pathlib.Path(ray.__file__).resolve())')
  case "$RAY_FILE" in "$ENV"/.venv/*) ;; *) echo "Ray resolved outside benchmark venv: $RAY_FILE" >&2; exit 1;; esac
  TARGET=$("$ENV/.venv/bin/python" -c 'import pathlib,ray; print(pathlib.Path(ray.__file__).parent / "data")')
  rm -rf "$TARGET"
  cp -a {shlex.quote(str(REMOTE_SOURCE / 'python/ray/data'))} "$TARGET"
  "$ENV/.venv/bin/python" -c {shlex.quote('import ' + imports)}
  "$ENV/.venv/bin/python" -c 'import ray; assert ray.__version__ == "{RAY_VERSION}"; assert ray.__commit__ == "{RAY_COMMIT}"'
  printf '%s\\n' {shlex.quote(bundle_digest)} > "$MARKER"
fi
test "$(cat "$MARKER")" = {shlex.quote(bundle_digest)}""".strip()
    return command, str(python), str(ray)


def _prepare_runtime(campaign: str, trial: str, gpus: int) -> str:
    short_campaign = hashlib.sha256(campaign.encode()).hexdigest()[:10]
    short_trial = hashlib.sha256(trial.encode()).hexdigest()[:12]
    runtime = Path(f"/mnt/nvme/rgs/{short_campaign}/{short_trial}")
    gpu_check = (
        "test -z \"$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')\""
        if gpus
        else "true"
    )
    return f"""set -euo pipefail
test -z "$(swapon --show --noheadings)"
{gpu_check}
ROOT={shlex.quote(str(runtime))}
test ! -e "$ROOT"
mkdir -p "$ROOT"/ray "$ROOT"/spill "$ROOT"/tmp "$ROOT"/cache
test -d /mnt/nvme
case "$ROOT/spill" in /dev/shm/*) exit 1;; esac""".strip()


def render_cluster(
    *,
    campaign: str,
    arm: str,
    local: Mapping[str, Any],
    topology: Mapping[str, Any],
    bundle: Path,
    bundle_digest: str,
    trial: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    if SAFE_CAMPAIGN.fullmatch(campaign) is None:
        raise ValueError(f"unsafe campaign ID: {campaign!r}")
    nodes = int(topology["nodes"])
    cpus = int(topology["vcpus_per_node"])
    gpus = int(topology["gpus_per_node"])
    cluster_name = f"ray-gpu-sort-{campaign}-{arm}"
    subnet_ids = list(local.get("subnet_ids") or [local["subnet_id"]])
    setup, remote_python, remote_ray = _setup(bundle_digest, arm)
    short_campaign = hashlib.sha256(campaign.encode()).hexdigest()[:10]
    short_trial = hashlib.sha256(trial.encode()).hexdigest()[:12]
    runtime = Path(f"/mnt/nvme/rgs/{short_campaign}/{short_trial}")
    common_env = (
        f"env PYTHONPATH={REMOTE_SOURCE} TMPDIR={runtime}/tmp "
        f"XDG_CACHE_HOME={runtime}/cache RAY_GPU_SORT_SPILL_DIR={runtime}/spill "
        "UCX_TLS=cuda_copy,sm,tcp UCX_MEMTYPE_CACHE=n "
    )
    head_start = (
        f"{common_env}{remote_ray} start --head --port=6379 --include-dashboard=false "
        f"--autoscaling-config=~/ray_bootstrap_config.yaml --temp-dir={runtime}/ray "
        f"--object-spilling-directory={runtime}/spill --num-cpus={cpus} --num-gpus={gpus}"
    )
    worker_start = (
        f"{common_env}{remote_ray} start --address=$RAY_HEAD_IP:6379 "
        f"--temp-dir={runtime}/ray --object-spilling-directory={runtime}/spill "
        f"--num-cpus={cpus} --num-gpus={gpus}"
    )
    node_config = {
        "InstanceType": str(topology["instance_type"]),
        "ImageId": str(local["ami_id"]),
        "DisableApiTermination": False,
        "InstanceInitiatedShutdownBehavior": "terminate",
        "SubnetIds": [str(item) for item in subnet_ids],
        "SecurityGroupIds": list(local["security_group_ids"]),
        "IamInstanceProfile": {"Arn": str(local["iam_instance_profile_arn"])},
        "KeyName": str(local["ec2_key_name"]),
        "MetadataOptions": {
            "HttpEndpoint": "enabled",
            "HttpPutResponseHopLimit": 2,
            "HttpTokens": "required",
        },
        "BlockDeviceMappings": [
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {
                    "DeleteOnTermination": True,
                    "Encrypted": True,
                    "VolumeSize": int(local.get("root_volume_gib", 100)),
                    "VolumeType": "gp3",
                },
            }
        ],
        "TagSpecifications": [
            {
                "ResourceType": kind,
                "Tags": [
                    {"Key": "ray-gpu-sort-project", "Value": "ray-data-gpu-external-sort"},
                    {"Key": "ray-gpu-sort-campaign", "Value": campaign},
                    {"Key": "ray-gpu-sort-arm", "Value": arm},
                    {"Key": "ray-gpu-sort-cluster", "Value": cluster_name},
                    {"Key": "ray-gpu-sort-topology", "Value": str(topology["name"])},
                ],
            }
            for kind in ("instance", "volume")
        ],
    }
    node_type = {
        "min_workers": 0,
        "max_workers": 0,
        "resources": {},
        "node_config": node_config,
    }
    worker_type = {
        **node_type,
        "min_workers": nodes - 1,
        "max_workers": nodes - 1,
    }
    docker_options = [
        f"--shm-size={int(topology['docker_shm_gib'])}gb",
        "--ulimit=nofile=65536:65536",
        f"--env=AWS_DEFAULT_REGION={local['region']}",
        "--volume=/opt/dlami/nvme:/mnt/nvme",
    ]
    if gpus:
        docker_options.append("--gpus=all")
    value = {
        "cluster_name": cluster_name,
        "max_workers": nodes - 1,
        "upscaling_speed": 1.0,
        "idle_timeout_minutes": 0,
        "provider": {
            "type": "aws",
            "region": str(local["region"]),
            "availability_zone": str(local["availability_zone"]),
            "cache_stopped_nodes": False,
        },
        "auth": {
            "ssh_user": str(local["ssh_user"]),
            "ssh_private_key": str(local["ssh_private_key_path"]),
        },
        "docker": {
            "image": str(local.get("docker_image") or (GPU_IMAGE if arm == "gpu" else CPU_IMAGE)),
            "container_name": "ray_gpu_sort_cloud",
            "pull_before_run": True,
            "run_options": docker_options,
        },
        "available_node_types": {
            "ray.head.default": node_type,
            "ray.worker.default": worker_type,
        },
        "head_node_type": "ray.head.default",
        "file_mounts": {str(REMOTE_SOURCE): str(bundle)},
        "rsync_exclude": ["__pycache__", "*.pyc"],
        "initialization_commands": [
            "sudo cloud-init status --wait",
            _nvme_init(int(topology["instance_store_devices"])),
        ],
        "setup_commands": [setup],
        "head_start_ray_commands": [
            f"{remote_ray} stop --force",
            _prepare_runtime(campaign, trial, gpus),
            head_start,
        ],
        "worker_start_ray_commands": [
            f"{remote_ray} stop --force",
            _prepare_runtime(campaign, trial, gpus),
            worker_start,
        ],
    }
    return value, {
        "cluster_name": cluster_name,
        "remote_python": remote_python,
        "remote_ray": remote_ray,
        "runtime": str(runtime),
    }


def write_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(value), sort_keys=False), encoding="utf-8")
