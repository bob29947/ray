"""Freeze the automatic wave fraction after the two 2x screens."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Sequence

from .common import atomic_json, read_json


def _usable(path: Path) -> tuple[dict[str, Any], float] | None:
    value = read_json(path)
    elapsed = value.get("cold_sort_s")
    if (
        value.get("valid") is not True
        or not isinstance(elapsed, (int, float))
        or isinstance(elapsed, bool)
        or not math.isfinite(float(elapsed))
        or elapsed <= 0
    ):
        return None
    return value, float(elapsed)


def select(default_path: Path, candidate_path: Path, output: Path) -> dict[str, Any]:
    default = _usable(default_path)
    candidate = _usable(candidate_path)
    if default is None:
        raise RuntimeError(
            "default 0.50 automatic-wave screen was invalid; the 0.375 "
            "candidate cannot clear a relative-speed gate without it"
        )
    if candidate is None:
        selected = 0.50
        reason = "0.375 screen was invalid"
        improvement = None
    else:
        improvement = (default[1] - candidate[1]) / default[1]
        if improvement >= 0.03:
            selected = 0.375
            reason = "0.375 was valid and at least 3% faster"
        else:
            selected = 0.50
            reason = "0.375 did not clear the 3% gate"
    result = {
        "schema_version": 1,
        "kind": "gpu_sort_cloud_wave_selection",
        "selected_wave_fraction": selected,
        "minimum_relative_improvement": 0.03,
        "observed_relative_improvement": improvement,
        "reason": reason,
        "screens": {
            "0.50": str(default_path),
            "0.375": str(candidate_path),
        },
    }
    atomic_json(output, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--default", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = select(args.default, args.candidate, args.output)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
