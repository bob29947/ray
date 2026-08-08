"""One-per-node inventory and immutable BTS S3 staging."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import atomic_json, digest, file_sha256, read_json


def _node_affinity(node_id: str) -> Any:
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    return NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)


def _alive(ray: Any) -> list[dict[str, Any]]:
    return sorted(
        (dict(node) for node in ray.nodes() if node.get("Alive")),
        key=lambda node: str(node.get("NodeID", "")),
    )


def _instance_id() -> str:
    token_request = urllib.request.Request(
        "http://169.254.169.254/latest/api/token",
        method="PUT",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
    )
    with urllib.request.urlopen(token_request, timeout=2) as response:
        token = response.read().decode()
    request = urllib.request.Request(
        "http://169.254.169.254/latest/meta-data/instance-id",
        headers={"X-aws-ec2-metadata-token": token},
    )
    with urllib.request.urlopen(request, timeout=2) as response:
        return response.read().decode()


def _node_inventory(expect_gpu: bool) -> dict[str, Any]:
    import importlib.metadata
    import psutil
    import ray

    nvme = shutil.disk_usage("/mnt/nvme")
    versions = {}
    for package in ("ray", "pyarrow", "cudf-cu12", "rmm-cu12", "rapidsmpf-cu12", "ucxx-cu12"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    gpu = None
    if expect_gpu:
        import pynvml

        pynvml.nvmlInit()
        try:
            if pynvml.nvmlDeviceGetCount() != 1:
                raise RuntimeError("GPU node does not expose exactly one L4")
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu = {
                "name": name.decode() if isinstance(name, bytes) else str(name),
                "memory_total_bytes": int(memory.total),
                "memory_used_bytes": int(memory.used),
            }
        finally:
            pynvml.nvmlShutdown()
    return {
        "hostname": socket.gethostname(),
        "instance_id": _instance_id(),
        "ray_node_id": str(ray.get_runtime_context().get_node_id()),
        "ray_node_ip": ray.util.get_node_ip_address(),
        "versions": versions,
        "gpu": gpu,
        "swap_total_bytes": int(psutil.swap_memory().total),
        "nvme_total_bytes": int(nvme.total),
        "nvme_free_bytes": int(nvme.free),
    }


def inventory(ray: Any, *, nodes: int, cpus: int, gpus: int) -> dict[str, Any]:
    alive = _alive(ray)
    resources = ray.cluster_resources()
    if (
        len(alive) != nodes
        or int(resources.get("CPU", 0)) != cpus
        or int(resources.get("GPU", 0)) != gpus
    ):
        raise RuntimeError(f"unexpected Ray shape: nodes={len(alive)}, resources={resources}")
    task = ray.remote(num_cpus=0.01)(_node_inventory)
    records = ray.get(
        [
            task.options(scheduling_strategy=_node_affinity(str(node["NodeID"]))).remote(gpus > 0)
            for node in alive
        ]
    )
    if len({record["instance_id"] for record in records}) != nodes:
        raise RuntimeError("inventory did not cover distinct EC2 instances")
    if any(record["swap_total_bytes"] for record in records):
        raise RuntimeError("swap is enabled on one or more benchmark nodes")
    return {
        "schema_version": 1,
        "kind": "gpu_sort_cloud_inventory",
        "shape": {"nodes": nodes, "cpus": cpus, "gpus": gpus},
        "ray_default_object_store_bytes": int(resources.get("object_store_memory", 0)),
        "nodes": records,
    }


def _stage_node(manifest: Mapping[str, Any], root: str) -> dict[str, Any]:
    import boto3
    import pyarrow.parquet as pq
    import ray

    destination = Path(root)
    expected_digest = str(manifest["dataset_digest"])
    owner = destination / "OWNER.json"
    expected_owner = {"dataset_digest": expected_digest, "schema_version": 1}
    if destination.exists() and owner.exists():
        if read_json(owner) != expected_owner:
            raise RuntimeError("staged dataset belongs to a different publication")
    else:
        if destination.exists() and any(destination.iterdir()):
            raise RuntimeError("unowned staged dataset directory is not empty")
        destination.mkdir(parents=True, exist_ok=True)
        atomic_json(owner, expected_owner)
    s3 = boto3.client("s3", region_name=os.environ.get("AWS_DEFAULT_REGION"))
    downloaded = reused = 0
    for item in manifest["files"]:
        path = destination / str(item["relative_path"])
        expected_size = int(item["bytes"])
        expected_sha = str(item["sha256"])
        if path.is_file() and path.stat().st_size == expected_size and file_sha256(path) == expected_sha:
            reused += 1
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
        s3.download_file(str(manifest["s3"]["bucket"]), str(item["object_key"]), str(temporary))
        if temporary.stat().st_size != expected_size or file_sha256(temporary) != expected_sha:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"download verification failed: {item['object_key']}")
        os.replace(temporary, path)
        downloaded += 1
    slices = list(manifest["portable_plan"]["slices"])
    first = destination / str(slices[0]["path"])
    schema_names = pq.ParquetFile(first).schema_arrow.names
    local_manifest = {
        "schema_names": schema_names,
        "row_groups": [
            {
                "path": str((destination / str(item["path"])).resolve()),
                "row_group": int(item["row_group"]),
                "rows": int(item["rows"]),
                "year": int(item["year"]),
                "month": int(item["month"]),
            }
            for item in slices
        ],
        "cloud_dataset_digest": expected_digest,
    }
    atomic_json(destination / "manifest.json", local_manifest)
    atomic_json(destination / "cloud-manifest.json", dict(manifest))
    return {
        "instance_id": _instance_id(),
        "ray_node_id": str(ray.get_runtime_context().get_node_id()),
        "dataset_root": str(destination),
        "dataset_digest": expected_digest,
        "manifest_sha256": file_sha256(destination / "manifest.json"),
        "downloaded_files": downloaded,
        "reused_files": reused,
    }


def stage(ray: Any, manifest: Mapping[str, Any], root: Path, nodes: int) -> dict[str, Any]:
    alive = _alive(ray)
    if len(alive) != nodes:
        raise RuntimeError("cluster changed before dataset staging")
    task = ray.remote(num_cpus=0.01)(_stage_node)
    records = ray.get(
        [
            task.options(scheduling_strategy=_node_affinity(str(node["NodeID"]))).remote(
                dict(manifest), str(root)
            )
            for node in alive
        ]
    )
    if len({item["instance_id"] for item in records}) != nodes:
        raise RuntimeError("dataset was not staged on every EC2 instance")
    if len({item["manifest_sha256"] for item in records}) != 1:
        raise RuntimeError("staged manifests differ between nodes")
    return {
        "schema_version": 1,
        "kind": "gpu_sort_cloud_dataset_stage",
        "dataset_digest": manifest["dataset_digest"],
        "nodes": records,
        "digest": digest(records),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inv = sub.add_parser("inventory")
    inv.add_argument("--nodes", type=int, required=True)
    inv.add_argument("--cpus", type=int, required=True)
    inv.add_argument("--gpus", type=int, required=True)
    inv.add_argument("--output", type=Path, required=True)
    staged = sub.add_parser("stage")
    staged.add_argument("--nodes", type=int, required=True)
    staged.add_argument("--manifest", type=Path, required=True)
    staged.add_argument("--dataset-root", type=Path, required=True)
    staged.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    import ray

    ray.init(address="auto", logging_level="ERROR")
    try:
        if args.command == "inventory":
            result = inventory(ray, nodes=args.nodes, cpus=args.cpus, gpus=args.gpus)
        else:
            result = stage(ray, read_json(args.manifest), args.dataset_root, args.nodes)
        atomic_json(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        ray.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
