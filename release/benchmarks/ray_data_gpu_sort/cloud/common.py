"""Small shared primitives used by the cloud controller and workers."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

ARTIFACT_SENTINEL = "__RAY_DATA_GPU_SORT_ARTIFACT_V1__="


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def artifact_sentinel(output: Path, artifact: Mapping[str, Any]) -> str:
    """Encode one complete result for last-resort controller recovery."""
    value = dict(artifact)
    envelope = {
        "schema_version": 1,
        "output": str(output),
        "artifact_sha256": digest(value),
        "artifact": value,
    }
    return ARTIFACT_SENTINEL + base64.b64encode(canonical_bytes(envelope)).decode(
        "ascii"
    )


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
