"""Run the focused CPU-sampled boundary-planning A/B on the local DGX."""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path
from statistics import median
from typing import Any

from .backend_stats import rank_stats, value
from .data import cohort_slices, plan_dict
from .runner import Study, _repo_root, _sha256_files, read_json, write_json
from .spec import (
    DATASET_ROOT,
    EXPECTED_BLOCKS,
    EXPECTED_FULL_BYTES,
    EXPECTED_ROWS,
    FOUR_KEYS,
    GIB,
    RAY_WHEEL_SHA256,
    cells,
    load_manifest,
)

BASELINE_GPU_S = (54.47087706, 53.336513943)
BASELINE_PYARROW_S = 269.752122802
MAX_H2D_BYTES = 66 * GIB
MIN_TOTAL_REDUCTION = 0.03
PLANNER_MODE = "cpu_sampled_arrow"
REQUIRED_SUBPHASES = (
    "cpu_sample_construction",
    "boundary_sort",
    "orchestration_remainder",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_ref(path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "valid": artifact.get("valid"),
        "provenance": artifact.get("provenance", {}),
    }


def _full_cell(manifest: dict[str, Any]) -> dict[str, Any]:
    return next(cell.to_dict() for cell in cells(manifest) if cell.name == "full")


def _require_archived_trial(
    path: Path,
    *,
    backend: str,
    repetition: int,
    expected_s: float,
    plan_digest: str,
    cell: dict[str, Any],
) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Missing archived {backend} observation: {path}")
    artifact = read_json(path)
    mismatches = []
    expected = {
        "valid": True,
        "kind": "trend",
        "backend": backend,
        "repetition": repetition,
        "wheel_sha256": RAY_WHEEL_SHA256,
    }
    for key, wanted in expected.items():
        if artifact.get(key) != wanted:
            mismatches.append(key)
    if artifact.get("plan", {}).get("digest") != plan_digest:
        mismatches.append("input plan digest")
    if artifact.get("cell") != cell:
        mismatches.append("full-cell specification")
    input_stats = artifact.get("input", {})
    if (
        input_stats.get("rows") != EXPECTED_ROWS
        or input_stats.get("blocks") != EXPECTED_BLOCKS
        or input_stats.get("decoded_bytes") != EXPECTED_FULL_BYTES
    ):
        mismatches.append("fixed cohort")
    if abs(float(artifact.get("cold_sort_s", -1)) - expected_s) > 1e-6:
        mismatches.append("accepted timing anchor")
    if mismatches:
        raise RuntimeError(
            f"Archived observation {path} is not the accepted baseline: "
            + ", ".join(mismatches)
        )
    return artifact


def _load_archived(
    root: Path, manifest: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    plan = plan_dict(cohort_slices(manifest), kind="trend")
    cell = _full_cell(manifest)
    directory = root / "trials/trend"
    pyarrow = _require_archived_trial(
        directory / "full-pyarrow-r1.json",
        backend="pyarrow",
        repetition=1,
        expected_s=BASELINE_PYARROW_S,
        plan_digest=plan["digest"],
        cell=cell,
    )
    gpu = [
        _require_archived_trial(
            directory / f"full-gpu-r{repetition}.json",
            backend="gpu",
            repetition=repetition,
            expected_s=expected_s,
            plan_digest=plan["digest"],
            cell=cell,
        )
        for repetition, expected_s in enumerate(BASELINE_GPU_S, start=1)
    ]
    return pyarrow, gpu, plan


def _semantic_table(path: Path) -> Any:
    import pyarrow as pa
    import pyarrow.compute as pc

    with pa.memory_map(str(path), "r") as source:
        table = pa.ipc.open_file(source).read_all().combine_chunks()
    arrays = []
    names = []
    for field, column in zip(table.schema, table.columns):
        arrays.append(column)
        names.append(field.name)
        if pa.types.is_floating(field.type):
            nan_mask = pc.fill_null(pc.is_nan(column), False)
            arrays[-1] = pc.if_else(nan_mask, pa.scalar(0, type=field.type), column)
            arrays.append(nan_mask)
            names.append(f"__nan__{field.name}")
    return table, pa.Table.from_arrays(arrays, names=names)


def _resident_reasons(trial: dict[str, Any]) -> list[str]:
    stats = trial.get("gpu_stats", {})
    ranks = rank_stats(stats)
    reasons = []
    checks = {
        "GPU externalization": int(value(stats, "externalized_bytes", default=-1)),
        "Ray disk spill": int(trial.get("ray_disk_spill_bytes", -1)),
        "MPF host spill": int(value(stats, "mpf_host_spill_bytes", default=-1)),
        "CPU sort rows": int(value(stats, "cpu_sort_rows", default=-1)),
        "CPU merge rows": int(value(stats, "cpu_merge_rows", default=-1)),
        "fallback count": int(value(stats, "fallback_count", default=-1)),
    }
    reasons.extend(
        f"{name} was {amount}, expected zero"
        for name, amount in checks.items()
        if amount
    )
    if len(ranks) != 16:
        reasons.append(f"GPU rank count was {len(ranks)}, expected 16")
    if len(ranks) == 16 and any(
        int(item.get("output_bytes", 0)) <= 0 for item in ranks
    ):
        reasons.append("one or more GPU ranks produced an empty output partition")
    locations = trial.get("output", {}).get("locations", {})
    if not locations.get("all_locatable"):
        reasons.append("not every output ObjectRef is Ray-locatable")
    if trial.get("output", {}).get("rows") != EXPECTED_ROWS:
        reasons.append("output row count differs from the fixed cohort")
    if trial.get("output", {}).get("schema") != trial.get("input", {}).get("schema"):
        reasons.append("output schema differs from input")
    return reasons


def _run_smoke(study: Study) -> dict[str, Any]:
    trials = [
        study.trial(
            kind="smoke",
            backend="pyarrow",
            cell="full",
            repetition=1,
            name="pyarrow",
        ),
        study.trial(
            kind="smoke",
            backend="gpu",
            cell="full",
            repetition=1,
            name="gpu-resident",
        ),
    ]
    baseline, baseline_semantic = _semantic_table(
        Path(trials[0]["validation"]["snapshot_path"])
    )
    candidate, candidate_semantic = _semantic_table(
        Path(trials[1]["validation"]["snapshot_path"])
    )
    if not baseline.schema.equals(candidate.schema, check_metadata=True):
        raise RuntimeError("Resident GPU smoke schema differs from PyArrow")
    if not baseline_semantic.equals(candidate_semantic):
        mismatch = next(
            (
                name
                for name in baseline_semantic.column_names
                if not baseline_semantic[name].equals(candidate_semantic[name])
            ),
            "unknown",
        )
        raise RuntimeError(
            "Resident GPU smoke rows/values differ from PyArrow; first "
            f"mismatching column is {mismatch}"
        )
    stats = trials[1].get("gpu_stats", {})
    smoke_counters = {
        "externalized bytes": int(value(stats, "externalized_bytes", default=-1)),
        "Ray disk spill bytes": int(trials[1].get("ray_disk_spill_bytes", -1)),
        "MPF host spill bytes": int(value(stats, "mpf_host_spill_bytes", default=-1)),
        "CPU sort rows": int(value(stats, "cpu_sort_rows", default=-1)),
        "CPU merge rows": int(value(stats, "cpu_merge_rows", default=-1)),
        "fallback count": int(value(stats, "fallback_count", default=-1)),
    }
    nonzero = {name: amount for name, amount in smoke_counters.items() if amount}
    if nonzero:
        raise RuntimeError(f"Resident GPU smoke used a spill/fallback path: {nonzero}")
    if len(rank_stats(stats)) != 16:
        raise RuntimeError("Resident GPU smoke did not use all 16 GPU ranks")
    summary = {
        "valid": True,
        "exact_schema_rows_values": True,
        "forced_spill_run": False,
        "rows": trials[0]["output"]["rows"],
        "trials": trials,
    }
    write_json(study.artifacts / "smoke.json", summary)
    return summary


def _nvml_summary(trial: dict[str, Any]) -> dict[str, int]:
    peaks = [int(item) for item in trial.get("nvml", {}).get("peak_used_bytes", [])]
    totals = [int(item) for item in trial.get("nvml", {}).get("total_bytes", [])]
    return {
        "peak_bytes": max(peaks, default=0),
        "minimum_headroom_bytes": min(
            (total - peak for peak, total in zip(peaks, totals)), default=0
        ),
    }


def _observation(trial: dict[str, Any]) -> dict[str, Any]:
    stats = trial["gpu_stats"]
    outputs = [int(item.get("output_bytes", 0)) for item in rank_stats(stats)]
    balanced_outputs = len(outputs) == 16 and all(item > 0 for item in outputs)
    subphases = stats.get("sampling_subphases_s", {})
    return {
        "repetition": trial["repetition"],
        "cold_sort_s": float(trial["cold_sort_s"]),
        "throughput_rows_s": float(trial["throughput_rows_s"]),
        "throughput_gib_s": float(trial["throughput_gib_s"]),
        "phases_s": stats.get("phases_s", {}),
        "controller_phases_s": stats.get("controller_phases_s", {}),
        "sampling_s": float(value(stats, "phases_s.sampling", default=-1)),
        "sampling_mode": stats.get("sampling_mode"),
        "sampling_subphases_s": subphases,
        "sample_rows": int(stats.get("sample_rows", -1)),
        "sample_bytes": int(stats.get("sample_bytes", -1)),
        "planning_h2d_bytes": int(stats.get("planning_h2d_bytes", -1)),
        "h2d_bytes": int(value(stats, "h2d_bytes", default=-1)),
        "d2h_bytes": int(value(stats, "d2h_bytes", default=-1)),
        "peak_device_bytes": int(value(stats, "peak_device_bytes", default=0)),
        "nvml": _nvml_summary(trial),
        "output_balance": {
            "minimum_bytes": min(outputs, default=0),
            "maximum_bytes": max(outputs, default=0),
            "max_over_min": max(outputs) / min(outputs) if balanced_outputs else None,
        },
        "externalized_bytes": int(value(stats, "externalized_bytes", default=-1)),
        "ray_disk_spill_bytes": int(trial.get("ray_disk_spill_bytes", -1)),
        "mpf_host_spill_bytes": int(value(stats, "mpf_host_spill_bytes", default=-1)),
        "cpu_sort_rows": int(value(stats, "cpu_sort_rows", default=-1)),
        "cpu_merge_rows": int(value(stats, "cpu_merge_rows", default=-1)),
        "fallback_count": int(value(stats, "fallback_count", default=-1)),
        "artifact_identity": trial.get("provenance", {}).get("trial_identity", {}),
    }


def _median_map(values: list[dict[str, Any]]) -> dict[str, float]:
    names = sorted(set.intersection(*(set(item) for item in values))) if values else []
    return {
        name: median(float(item[name]) for item in values)
        for name in names
        if all(isinstance(item[name], (int, float)) for item in values)
    }


def _build_report(
    *,
    study: Study,
    archived_root: Path,
    pyarrow: dict[str, Any],
    baseline: list[dict[str, Any]],
    plan: dict[str, Any],
    smoke: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    observations = [_observation(item) for item in candidates]
    baseline_median = median(float(item["cold_sort_s"]) for item in baseline)
    candidate_median = median(item["cold_sort_s"] for item in observations)
    baseline_sampling = median(
        float(value(item["gpu_stats"], "phases_s.sampling", default=-1))
        for item in baseline
    )
    candidate_sampling = median(item["sampling_s"] for item in observations)
    health_reasons = [
        reason for item in candidates for reason in _resident_reasons(item)
    ]
    telemetry_reasons = []
    for item in observations:
        rep = item["repetition"]
        if item["sampling_mode"] != PLANNER_MODE:
            telemetry_reasons.append(f"r{rep} sampling_mode is not {PLANNER_MODE}")
        if item["sample_rows"] < 65_536 or item["sample_bytes"] <= 0:
            telemetry_reasons.append(f"r{rep} sample row/byte telemetry is incomplete")
        missing = [
            name
            for name in REQUIRED_SUBPHASES
            if name not in item["sampling_subphases_s"]
        ]
        if missing:
            telemetry_reasons.append(f"r{rep} is missing sampling subphases: {missing}")
        if item["planning_h2d_bytes"] < 0 or item["h2d_bytes"] < 0:
            telemetry_reasons.append(f"r{rep} is missing H2D telemetry")
    gates = {
        "both_observations_valid_and_resident": not health_reasons,
        "complete_candidate_planner_telemetry": not telemetry_reasons,
        "median_at_least_3_percent_faster": candidate_median
        <= baseline_median * (1 - MIN_TOTAL_REDUCTION),
        "median_sampling_improved": candidate_sampling < baseline_sampling,
        "each_h2d_at_most_66_gib": all(
            item["h2d_bytes"] <= MAX_H2D_BYTES for item in observations
        ),
    }
    worthwhile = all(gates.values())
    baseline_paths = [
        archived_root / "trials/trend/full-gpu-r1.json",
        archived_root / "trials/trend/full-gpu-r2.json",
    ]
    pyarrow_path = archived_root / "trials/trend/full-pyarrow-r1.json"
    candidate_paths = [
        study.path("trend", f"candidate-full-gpu-r{repetition}")
        for repetition in (1, 2)
    ]
    smoke_paths = [
        study.path("smoke", "pyarrow"),
        study.path("smoke", "gpu-resident"),
    ]
    summary = {
        "schema_version": 1,
        "kind": "cpu_sampled_boundary_planning_ab",
        "status": "accepted"
        if not health_reasons and not telemetry_reasons
        else "rejected",
        "worthwhile": worthwhile,
        "workload": {
            "rows": EXPECTED_ROWS,
            "blocks": EXPECTED_BLOCKS,
            "decoded_bytes": EXPECTED_FULL_BYTES,
            "columns": len(candidates[0]["cell"]["columns"]),
            "keys": list(FOUR_KEYS),
            "input_plan_digest": plan["digest"],
            "timing_boundary": candidates[0]["timing_boundary"],
        },
        "thresholds": {
            "minimum_total_time_reduction_fraction": MIN_TOTAL_REDUCTION,
            "maximum_h2d_bytes_each_observation": MAX_H2D_BYTES,
            "sampling_must_improve": True,
        },
        "smoke": {
            "valid": smoke["valid"],
            "exact_schema_rows_values": smoke["exact_schema_rows_values"],
            "forced_spill_run": False,
            "rows": smoke["rows"],
            "artifacts": [
                _artifact_ref(path, item)
                for path, item in zip(smoke_paths, smoke["trials"])
            ],
        },
        "archived": {
            "pyarrow": {
                "cold_sort_s": float(pyarrow["cold_sort_s"]),
                "artifact": _artifact_ref(pyarrow_path, pyarrow),
            },
            "gpu": {
                "cold_sort_s": [float(item["cold_sort_s"]) for item in baseline],
                "median_s": baseline_median,
                "sampling_s": [
                    float(value(item["gpu_stats"], "phases_s.sampling", default=-1))
                    for item in baseline
                ],
                "sampling_median_s": baseline_sampling,
                "h2d_bytes": [
                    int(value(item["gpu_stats"], "h2d_bytes", default=-1))
                    for item in baseline
                ],
                "artifacts": [
                    _artifact_ref(path, item)
                    for path, item in zip(baseline_paths, baseline)
                ],
            },
        },
        "candidate": {
            "observations": observations,
            "median_s": candidate_median,
            "sampling_median_s": candidate_sampling,
            "phase_medians_s": _median_map([item["phases_s"] for item in observations]),
            "sampling_subphase_medians_s": _median_map(
                [item["sampling_subphases_s"] for item in observations]
            ),
            "speedup_vs_pyarrow": float(pyarrow["cold_sort_s"]) / candidate_median,
            "artifacts": [
                _artifact_ref(path, item)
                for path, item in zip(candidate_paths, candidates)
            ],
        },
        "comparison": {
            "seconds_saved": baseline_median - candidate_median,
            "total_time_reduction_fraction": 1 - candidate_median / baseline_median,
            "sampling_seconds_saved": baseline_sampling - candidate_sampling,
            "baseline_speedup_vs_pyarrow": float(pyarrow["cold_sort_s"])
            / baseline_median,
            "candidate_speedup_vs_pyarrow": float(pyarrow["cold_sort_s"])
            / candidate_median,
        },
        "acceptance_gates": gates,
        "rejection_reasons": health_reasons + telemetry_reasons,
        "artifact_identity": {
            "generated_unix_s": time.time(),
            "dataset_manifest_sha256": study.dataset_manifest_sha256,
            "harness_sha256": _sha256_files(study.root),
            "wheel_sha256": study.wheel["sha256"],
            "candidate_ray_overlay_sha256": candidates[0]["provenance"][
                "ray_overlay_sha256"
            ],
            "candidate_trial_identities": [
                item["provenance"]["trial_identity"] for item in candidates
            ],
        },
    }
    return summary


def _render(summary: dict[str, Any]) -> str:
    archived = summary["archived"]
    candidate = summary["candidate"]
    comparison = summary["comparison"]
    observations = candidate["observations"]
    lines = [
        "# CPU-Sampled Boundary Planning A/B",
        "",
        f"**Verdict: {'worthwhile' if summary['worthwhile'] else 'not worthwhile'}.**",
        "",
        "The CPU handles only the approximately 65K-row planning sample. The "
        "80,738,761-row distributed dataset is still partitioned, shuffled, and sorted "
        "on the 16 GPUs; no dataset rows are CPU-sorted or CPU-merged.",
        "",
        "## Result",
        "",
        "| Metric | Archived GPU planner | CPU-sampled candidate |",
        "|---|---:|---:|",
        f"| Cold r1 | {archived['gpu']['cold_sort_s'][0]:.3f}s | {observations[0]['cold_sort_s']:.3f}s |",
        f"| Cold r2 | {archived['gpu']['cold_sort_s'][1]:.3f}s | {observations[1]['cold_sort_s']:.3f}s |",
        f"| Cold median | {archived['gpu']['median_s']:.3f}s | {candidate['median_s']:.3f}s |",
        f"| Sampling median | {archived['gpu']['sampling_median_s']:.3f}s | {candidate['sampling_median_s']:.3f}s |",
        f"| H2D r1 / r2 | {archived['gpu']['h2d_bytes'][0] / GIB:.3f} / {archived['gpu']['h2d_bytes'][1] / GIB:.3f} GiB | {observations[0]['h2d_bytes'] / GIB:.3f} / {observations[1]['h2d_bytes'] / GIB:.3f} GiB |",
        f"| Speedup vs archived PyArrow ({archived['pyarrow']['cold_sort_s']:.3f}s) | {comparison['baseline_speedup_vs_pyarrow']:.2f}x | {comparison['candidate_speedup_vs_pyarrow']:.2f}x |",
        "",
        f"The candidate saved {comparison['seconds_saved']:.3f}s "
        f"({100 * comparison['total_time_reduction_fraction']:.1f}%) at the median.",
        "",
        "## Candidate planning telemetry",
        "",
        "| Observation | Sampling | CPU sample construction | Boundary sort | Remainder | Sample rows | Sample bytes | Planning H2D |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in observations:
        sub = item["sampling_subphases_s"]
        lines.append(
            f"| r{item['repetition']} | {item['sampling_s']:.3f}s | "
            f"{float(sub.get('cpu_sample_construction', -1)):.3f}s | "
            f"{float(sub.get('boundary_sort', -1)):.3f}s | "
            f"{float(sub.get('orchestration_remainder', -1)):.3f}s | "
            f"{item['sample_rows']:,} | {item['sample_bytes'] / (1 << 20):.1f} MiB | "
            f"{item['planning_h2d_bytes'] / GIB:.3f} GiB |"
        )
    lines.extend(
        [
            "",
            "## Candidate health",
            "",
            "| Observation | Throughput | Peak RMM | Peak NVML | Min headroom | Output max/min | External / Ray / MPF spill | CPU sort / merge / fallback |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in observations:
        lines.append(
            f"| r{item['repetition']} | {item['throughput_gib_s']:.3f} GiB/s | "
            f"{item['peak_device_bytes'] / GIB:.2f} GiB | "
            f"{item['nvml']['peak_bytes'] / GIB:.2f} GiB | "
            f"{item['nvml']['minimum_headroom_bytes'] / GIB:.2f} GiB | "
            f"{float(item['output_balance']['max_over_min'] or 0):.2f}x | "
            f"{item['externalized_bytes']} / {item['ray_disk_spill_bytes']} / "
            f"{item['mpf_host_spill_bytes']} B | {item['cpu_sort_rows']} / "
            f"{item['cpu_merge_rows']} / {item['fallback_count']} |"
        )
    lines.extend(
        [
            "",
            "## Candidate cold-phase attribution",
            "",
            "| Phase | r1 | r2 | Median |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in sorted(candidate["phase_medians_s"]):
        lines.append(
            f"| {name.replace('_', ' ')} | "
            f"{float(observations[0]['phases_s'].get(name, 0)):.3f}s | "
            f"{float(observations[1]['phases_s'].get(name, 0)):.3f}s | "
            f"{candidate['phase_medians_s'][name]:.3f}s |"
        )
    lines.extend(["", "## Acceptance", ""])
    for name, passed in summary["acceptance_gates"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — {name.replace('_', ' ')}")
    lines.extend(
        [
            "",
            "Each performance observation used the exact 63.881-GiB, 110-column, "
            "627-block BTS cohort and four natural keys. Input materialization was "
            "excluded; GPU actor/CUDA/RMM/MPF startup and output Plasma sealing were "
            "included. The exact smoke ran PyArrow once and resident GPU once, with no "
            "forced-spill trial.",
            "",
        ]
    )
    return "\n".join(lines)


def parser() -> argparse.ArgumentParser:
    root = _repo_root()
    result = argparse.ArgumentParser(
        description="Run the fixed resident CPU-sampled boundary-planning A/B."
    )
    result.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    result.add_argument(
        "--artifact-root",
        type=Path,
        default=root / ".venv/gpu-sort-sampling-ab-artifacts",
    )
    result.add_argument(
        "--runtime-root",
        type=Path,
        default=root / ".venv/gpu-sort-sampling-ab-runtime",
    )
    result.add_argument(
        "--archived-root",
        type=Path,
        default=root / ".venv/gpu-sort-external-artifacts",
    )
    result.add_argument("--timeout-s", type=int, default=14_400)
    return result


def main() -> int:
    args = parser().parse_args()
    manifest = load_manifest(args.dataset_root.resolve())
    pyarrow, baseline, plan = _load_archived(args.archived_root.resolve(), manifest)
    study = Study(args)
    smoke = _run_smoke(study)
    candidates = [
        study.trial(
            kind="trend",
            backend="gpu",
            cell="full",
            repetition=repetition,
            name=f"candidate-full-gpu-r{repetition}",
            reuse=False,
        )
        for repetition in (1, 2)
    ]
    summary = _build_report(
        study=study,
        archived_root=args.archived_root.resolve(),
        pyarrow=pyarrow,
        baseline=baseline,
        plan=plan,
        smoke=smoke,
        candidates=candidates,
    )
    write_json(study.artifacts / "sampling-ab.json", summary)
    report = study.artifacts / "SAMPLING_AB_REPORT.md"
    report.write_text(_render(summary), encoding="utf-8")
    print(report)
    return 0 if summary["status"] == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
