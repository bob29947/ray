"""Small compatibility layer around the experimental GPU-sort telemetry API."""

from __future__ import annotations

import dataclasses
import importlib
import json
from typing import Any


def _plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    elif hasattr(value, "to_dict"):
        value = value.to_dict()
    # Detach any proxy/custom mappings and prove JSON serializability.
    return json.loads(json.dumps(value))


def get_last_run_stats(output: Any | None = None) -> dict[str, Any]:
    """Return backend stats while tolerating their final pre-PR module location."""

    for module_name in (
        "ray.data._internal.gpu_sort.operator",
        "ray.data._internal.gpu_sort",
        "ray.data._internal.gpu_sort.backend",
    ):
        try:
            module = importlib.import_module(module_name)
            getter = getattr(module, "get_last_run_stats", None)
            if getter is not None:
                value = _plain(getter())
                if isinstance(value, dict) and value:
                    return value
        except (ImportError, AttributeError):
            continue
    if output is not None:
        value = getattr(output, "_gpu_sort_stats", None)
        if value is not None:
            value = _plain(value)
            if isinstance(value, dict) and value:
                return value
    return {}


def value(stats: dict[str, Any], *paths: str, default: Any = 0) -> Any:
    """Read a preferred field plus a few harmless legacy/nested aliases."""

    for path in paths:
        current: Any = stats
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                break
            current = current[part]
        else:
            if current is not None:
                return current
    return default


def rank_stats(stats: dict[str, Any]) -> list[dict[str, Any]]:
    ranks = value(stats, "ranks", "per_rank", "backend.ranks", default=[])
    if isinstance(ranks, dict):
        ranks = [dict(payload, rank=rank) for rank, payload in ranks.items()]
    return [dict(item) for item in ranks if isinstance(item, dict)]


def peak_device_bytes(stats: dict[str, Any]) -> int:
    direct = int(
        value(
            stats,
            "peak_device_bytes",
            "memory.peak_device_bytes",
            "max_peak_rmm_bytes",
            default=0,
        )
        or 0
    )
    per_rank = [
        int(value(item, "peak_device_bytes", "peak_rmm_bytes", default=0) or 0)
        for item in rank_stats(stats)
    ]
    return max([direct, *per_rank])


def spill_geometry(stats: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(
            value(stats, "externalized_bytes", "spill.externalized_bytes", default=0)
            or 0
        ),
        int(value(stats, "initial_run_count", "runs.initial", default=0) or 0),
        int(value(stats, "merge_pass_count", "runs.merge_passes", default=0) or 0),
    )


def required_fields_missing(stats: dict[str, Any]) -> list[str]:
    required = (
        "peak_device_bytes",
        "externalized_bytes",
        "initial_run_count",
        "merge_pass_count",
        "cpu_sort_rows",
        "cpu_merge_rows",
        "phases_s",
    )
    return [name for name in required if name not in stats]
