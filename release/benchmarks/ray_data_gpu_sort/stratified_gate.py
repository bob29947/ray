"""Run the focused deterministic-stratified sampling gate on the local DGX."""

from __future__ import annotations

import argparse
import hashlib
import re
import time
from pathlib import Path
from statistics import median
from typing import Any

from .backend_stats import rank_stats
from .runner import Study, _repo_root, _sha256_files, read_json, write_json
from .sampling_ab import (
    REQUIRED_SUBPHASES,
    _artifact_ref,
    _observation,
    _resident_reasons,
    _run_smoke,
)
from .spec import (
    DATASET_ROOT,
    EXPECTED_BLOCKS,
    EXPECTED_FULL_BYTES,
    EXPECTED_ROWS,
    FOUR_KEYS,
    GIB,
)

BASELINE_MEDIAN_S = 48.8007693485
MAX_COLD_MEDIAN_S = 50.27
MAX_SAMPLING_MEDIAN_S = 1.0
MAX_H2D_BYTES = 66 * GIB
MAX_OUTPUT_BALANCE = 1.10
EXPECTED_SAMPLE_SEED = 0
MIN_SAMPLE_TARGET = 65_536
SAMPLING_MODE = "cpu_sampled_arrow"
SAMPLING_SCHEME = "deterministic_stratified_random"
SAMPLING_SCHEME_VERSION = 1
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_baseline(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Missing accepted CPU-sampled baseline: {path}")
    summary = read_json(path)
    candidate = summary.get("candidate", {})
    workload = summary.get("workload", {})
    mismatches = []
    if summary.get("status") != "accepted" or not summary.get("worthwhile"):
        mismatches.append("accepted/worthwhile status")
    if abs(float(candidate.get("median_s", -1)) - BASELINE_MEDIAN_S) > 1e-6:
        mismatches.append("48.801-second median")
    expected_workload = {
        "rows": EXPECTED_ROWS,
        "blocks": EXPECTED_BLOCKS,
        "decoded_bytes": EXPECTED_FULL_BYTES,
        "keys": list(FOUR_KEYS),
    }
    for name, expected in expected_workload.items():
        if workload.get(name) != expected:
            mismatches.append(f"workload {name}")
    if len(candidate.get("observations", ())) != 2:
        mismatches.append("two baseline GPU observations")
    if mismatches:
        raise RuntimeError(
            f"Archived baseline {path} is not the accepted CPU-sampled result: "
            + ", ".join(mismatches)
        )
    return summary


def _digest(value_: Any) -> bool:
    return isinstance(value_, str) and SHA256_PATTERN.fullmatch(value_) is not None


def _sampling_telemetry(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "sampling_mode": stats.get("sampling_mode"),
        "sampling_scheme": stats.get("sampling_scheme"),
        "sampling_scheme_version": stats.get("sampling_scheme_version"),
        "sample_seed": stats.get("sample_seed"),
        "sample_target_rows": stats.get("sample_target_rows"),
        "sample_rows": stats.get("sample_rows"),
        "sample_bytes": stats.get("sample_bytes"),
        "planning_sample_bytes": stats.get("planning_sample_bytes"),
        "sampled_block_count": stats.get("sampled_block_count"),
        "sample_quota_rows": stats.get("sample_quota_rows"),
        "sample_plan_digest": stats.get("sample_plan_digest"),
        "sample_index_digest": stats.get("sample_index_digest"),
        "boundary_digest": stats.get("boundary_digest"),
        "planning_h2d_bytes": stats.get("planning_h2d_bytes"),
        "sampling_subphases_s": stats.get("sampling_subphases_s"),
    }


def _sampling_reasons(
    stats: dict[str, Any], *, expected_block_count: int | None
) -> list[str]:
    telemetry = _sampling_telemetry(stats)
    reasons = []
    exact = {
        "sampling_mode": SAMPLING_MODE,
        "sampling_scheme": SAMPLING_SCHEME,
        "sampling_scheme_version": SAMPLING_SCHEME_VERSION,
        "sample_seed": EXPECTED_SAMPLE_SEED,
    }
    for name, expected in exact.items():
        if telemetry[name] != expected:
            reasons.append(f"{name} was {telemetry[name]!r}, expected {expected!r}")

    integer_fields = (
        "sample_target_rows",
        "sample_rows",
        "sample_bytes",
        "planning_sample_bytes",
        "sampled_block_count",
        "planning_h2d_bytes",
    )
    for name in integer_fields:
        if not isinstance(telemetry[name], int) or telemetry[name] < 0:
            reasons.append(f"{name} is not a nonnegative integer")
    target = telemetry["sample_target_rows"]
    rows = telemetry["sample_rows"]
    if isinstance(target, int) and target < MIN_SAMPLE_TARGET:
        reasons.append(f"sample_target_rows was {target}, expected at least 65,536")
    if isinstance(target, int) and isinstance(rows, int) and rows != target:
        reasons.append(f"sample_rows was {rows}, expected target {target}")
    if telemetry["sample_bytes"] == 0:
        reasons.append("sample_bytes was zero")
    if (
        isinstance(telemetry["sample_bytes"], int)
        and isinstance(telemetry["planning_sample_bytes"], int)
        and telemetry["planning_sample_bytes"] > telemetry["sample_bytes"]
    ):
        reasons.append("planning_sample_bytes exceeded physical sample_bytes")
    if (
        expected_block_count is not None
        and telemetry["sampled_block_count"] != expected_block_count
    ):
        reasons.append(
            f"sampled_block_count was {telemetry['sampled_block_count']}, "
            f"expected {expected_block_count}"
        )

    quotas = telemetry["sample_quota_rows"]
    if not isinstance(quotas, dict):
        reasons.append("sample_quota_rows is not an object")
    else:
        ordered = [quotas.get(name) for name in ("min", "median", "max")]
        if not all(isinstance(item, (int, float)) and item >= 1 for item in ordered):
            reasons.append("sample_quota_rows min/median/max are not positive numbers")
        elif ordered != sorted(ordered):
            reasons.append("sample_quota_rows is not ordered min <= median <= max")

    for name in ("sample_plan_digest", "sample_index_digest", "boundary_digest"):
        if not _digest(telemetry[name]):
            reasons.append(f"{name} is not a lowercase SHA-256 digest")

    subphases = telemetry["sampling_subphases_s"]
    if not isinstance(subphases, dict):
        reasons.append("sampling_subphases_s is not an object")
    else:
        missing = [name for name in REQUIRED_SUBPHASES if name not in subphases]
        if missing:
            reasons.append(f"sampling_subphases_s is missing {missing}")
        for name in REQUIRED_SUBPHASES:
            amount = subphases.get(name)
            if not isinstance(amount, (int, float)) or amount < 0:
                reasons.append(f"sampling subphase {name} is not nonnegative")
    return reasons


def _observation_with_sampling(trial: dict[str, Any]) -> dict[str, Any]:
    observation = _observation(trial)
    stats = trial.get("gpu_stats", {})
    observation["sampling"] = _sampling_telemetry(stats)
    observation["rank_count"] = len(rank_stats(stats))
    return observation


def _gate_results(
    observations: list[dict[str, Any]],
    *,
    health_reasons: list[str],
    telemetry_reasons: list[str],
) -> dict[str, bool]:
    cold_median = median(item["cold_sort_s"] for item in observations)
    sampling_median = median(item["sampling_s"] for item in observations)
    digests = (
        "sample_plan_digest",
        "sample_index_digest",
        "boundary_digest",
    )
    return {
        "both_observations_valid_and_resident": not health_reasons,
        "complete_stratified_sampling_telemetry": not telemetry_reasons,
        "deterministic_digests_across_fresh_runtimes": all(
            len({item["sampling"].get(name) for item in observations}) == 1
            for name in digests
        ),
        "cold_median_at_most_50_27_s": cold_median <= MAX_COLD_MEDIAN_S,
        "sampling_median_at_most_1_s": sampling_median <= MAX_SAMPLING_MEDIAN_S,
        "each_h2d_at_most_66_gib": all(
            item["h2d_bytes"] <= MAX_H2D_BYTES for item in observations
        ),
        "each_output_balance_at_most_1_10": all(
            item["output_balance"]["max_over_min"] is not None
            and item["output_balance"]["max_over_min"] <= MAX_OUTPUT_BALANCE
            for item in observations
        ),
    }


def _build_summary(
    *,
    study: Study,
    baseline_path: Path,
    baseline: dict[str, Any],
    smoke: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    observations = [_observation_with_sampling(item) for item in candidates]
    health_reasons = []
    telemetry_reasons = []
    for trial in candidates:
        repetition = trial["repetition"]
        health_reasons.extend(
            f"r{repetition}: {reason}" for reason in _resident_reasons(trial)
        )
        telemetry_reasons.extend(
            f"r{repetition}: {reason}"
            for reason in _sampling_reasons(
                trial.get("gpu_stats", {}), expected_block_count=EXPECTED_BLOCKS
            )
        )
    smoke_gpu = smoke["trials"][1]
    telemetry_reasons.extend(
        f"smoke: {reason}"
        for reason in _sampling_reasons(
            smoke_gpu.get("gpu_stats", {}),
            expected_block_count=smoke_gpu.get("input", {}).get("blocks"),
        )
    )
    gates = _gate_results(
        observations,
        health_reasons=health_reasons,
        telemetry_reasons=telemetry_reasons,
    )
    accepted = all(gates.values())
    cold_median = median(item["cold_sort_s"] for item in observations)
    sampling_median = median(item["sampling_s"] for item in observations)
    candidate_paths = [
        study.path("trend", f"stratified-full-gpu-r{repetition}")
        for repetition in (1, 2)
    ]
    smoke_paths = [
        study.path("smoke", "pyarrow"),
        study.path("smoke", "gpu-resident"),
    ]
    return {
        "schema_version": 1,
        "kind": "deterministic_stratified_sampling_dgx_gate",
        "status": "accepted" if accepted else "rejected",
        "aws_gate_passed": accepted,
        "workload": {
            "rows": EXPECTED_ROWS,
            "blocks": EXPECTED_BLOCKS,
            "decoded_bytes": EXPECTED_FULL_BYTES,
            "columns": len(candidates[0]["cell"]["columns"]),
            "keys": list(FOUR_KEYS),
            "input_plan_digest": candidates[0]["plan"]["digest"],
            "timing_boundary": candidates[0]["timing_boundary"],
        },
        "thresholds": {
            "baseline_cpu_sampled_median_s": BASELINE_MEDIAN_S,
            "maximum_cold_median_s": MAX_COLD_MEDIAN_S,
            "maximum_sampling_median_s": MAX_SAMPLING_MEDIAN_S,
            "maximum_h2d_bytes_each": MAX_H2D_BYTES,
            "maximum_output_balance_each": MAX_OUTPUT_BALANCE,
        },
        "baseline": {
            "path": str(baseline_path.resolve()),
            "sha256": _sha256(baseline_path),
            "median_s": float(baseline["candidate"]["median_s"]),
            "cold_sort_s": [
                float(item["cold_sort_s"])
                for item in baseline["candidate"]["observations"]
            ],
        },
        "smoke": {
            "valid": smoke["valid"],
            "exact_schema_rows_values": smoke["exact_schema_rows_values"],
            "rows": smoke["rows"],
            "gpu_sampling": _sampling_telemetry(smoke_gpu.get("gpu_stats", {})),
            "artifacts": [
                _artifact_ref(path, item)
                for path, item in zip(smoke_paths, smoke["trials"])
            ],
        },
        "candidate": {
            "observations": observations,
            "median_s": cold_median,
            "sampling_median_s": sampling_median,
            "change_vs_cpu_sampled_baseline_fraction": (
                cold_median / BASELINE_MEDIAN_S - 1
            ),
            "artifacts": [
                _artifact_ref(path, item)
                for path, item in zip(candidate_paths, candidates)
            ],
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


def _render(summary: dict[str, Any]) -> str:
    candidate = summary["candidate"]
    observations = candidate["observations"]
    lines = [
        "# Deterministic Stratified Sampling: DGX Gate",
        "",
        f"**Verdict: {summary['status']}.** "
        + (
            "The AWS benchmark campaign is unblocked."
            if summary["aws_gate_passed"]
            else "Stop before AWS."
        ),
        "",
        "The CPU orders only the deterministic 65K-row planning sample. All "
        "80,738,761 dataset rows are still partitioned, shuffled, and sorted on GPU.",
        f"Sampling scheme: `{SAMPLING_SCHEME}` v{SAMPLING_SCHEME_VERSION}, "
        f"seed {EXPECTED_SAMPLE_SEED}.",
        "",
        "## Performance",
        "",
        "| Metric | CPU-sampled baseline | Stratified r1 | Stratified r2 | Stratified median |",
        "|---|---:|---:|---:|---:|",
        f"| Cold time | {summary['baseline']['median_s']:.3f}s | "
        f"{observations[0]['cold_sort_s']:.3f}s | {observations[1]['cold_sort_s']:.3f}s | "
        f"{candidate['median_s']:.3f}s |",
        f"| Sampling | — | {observations[0]['sampling_s']:.3f}s | "
        f"{observations[1]['sampling_s']:.3f}s | {candidate['sampling_median_s']:.3f}s |",
        f"| H2D | — | {observations[0]['h2d_bytes'] / GIB:.3f} GiB | "
        f"{observations[1]['h2d_bytes'] / GIB:.3f} GiB | — |",
        f"| Output max/min | — | "
        f"{_display_number(observations[0]['output_balance']['max_over_min'], 'x')} | "
        f"{_display_number(observations[1]['output_balance']['max_over_min'], 'x')} | — |",
        "",
        f"Median change from the accepted CPU-sampled result: "
        f"{100 * candidate['change_vs_cpu_sampled_baseline_fraction']:+.1f}%.",
        "",
        "## Reproducibility and sampling",
        "",
        "| Observation | Target / actual | Blocks | Quota min / median / max | Planning H2D | Plan digest | Index digest | Boundary digest |",
        "|---|---:|---:|---:|---:|---|---|---|",
    ]
    for item in observations:
        sampling = item["sampling"]
        quota = sampling["sample_quota_rows"]
        quota = quota if isinstance(quota, dict) else {}
        lines.append(
            f"| r{item['repetition']} | "
            f"{_display_number(sampling['sample_target_rows'])} / "
            f"{_display_number(sampling['sample_rows'])} | "
            f"{_display_number(sampling['sampled_block_count'])} | "
            f"{_display_number(quota.get('min'))} / "
            f"{_display_number(quota.get('median'))} / "
            f"{_display_number(quota.get('max'))} | "
            f"{_display_number(sampling['planning_h2d_bytes'], ' B')} | "
            f"{_display_digest(sampling['sample_plan_digest'])} | "
            f"{_display_digest(sampling['sample_index_digest'])} | "
            f"{_display_digest(sampling['boundary_digest'])} |"
        )
    lines.extend(
        [
            "",
            "## Resident GPU health",
            "",
            "| Observation | Ranks | Peak RMM | Peak NVML | Output max/min | GPU / Ray / MPF spill | CPU sort / merge / fallback |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in observations:
        lines.append(
            f"| r{item['repetition']} | {item['rank_count']} | "
            f"{item['peak_device_bytes'] / GIB:.2f} GiB | "
            f"{item['nvml']['peak_bytes'] / GIB:.2f} GiB | "
            f"{_display_number(item['output_balance']['max_over_min'], 'x')} | "
            f"{item['externalized_bytes']} / {item['ray_disk_spill_bytes']} / "
            f"{item['mpf_host_spill_bytes']} B | {item['cpu_sort_rows']} / "
            f"{item['cpu_merge_rows']} / {item['fallback_count']} |"
        )
    lines.extend(["", "## Acceptance", ""])
    for name, passed in summary["acceptance_gates"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — {name.replace('_', ' ')}")
    if summary["rejection_reasons"]:
        lines.extend(["", "## Rejection reasons", ""])
        lines.extend(f"- {reason}" for reason in summary["rejection_reasons"])
    lines.extend(
        [
            "",
            "The exact smoke ran PyArrow once and GPU once. The performance gate "
            "ran exactly two fresh-Ray-runtime observations of the fixed 63.881-GiB, "
            "110-column, four-key BTS workload. No forced-spill or trend matrix was run.",
            "",
        ]
    )
    return "\n".join(lines)


def _failure_report(stage: str, error: BaseException) -> str:
    return "\n".join(
        [
            "# Deterministic Stratified Sampling: DGX Gate",
            "",
            "**Verdict: rejected. Stop before AWS.**",
            "",
            f"The gate stopped during `{stage}`: {type(error).__name__}: {error}",
            "",
        ]
    )


def _display_number(value_: Any, suffix: str = "") -> str:
    if isinstance(value_, int):
        return f"{value_:,}{suffix}"
    if isinstance(value_, float):
        return f"{value_:,.3f}{suffix}"
    return "missing"


def _display_digest(value_: Any) -> str:
    return f"`{value_[:12]}…`" if isinstance(value_, str) else "missing"


def _self_check() -> None:
    digest = "a" * 64
    telemetry = {
        "sampling_mode": SAMPLING_MODE,
        "sampling_scheme": SAMPLING_SCHEME,
        "sampling_scheme_version": SAMPLING_SCHEME_VERSION,
        "sample_seed": EXPECTED_SAMPLE_SEED,
        "sample_target_rows": MIN_SAMPLE_TARGET,
        "sample_rows": MIN_SAMPLE_TARGET,
        "sample_bytes": 123,
        "planning_sample_bytes": 99,
        "sampled_block_count": EXPECTED_BLOCKS,
        "sample_quota_rows": {"min": 104, "median": 105.0, "max": 105},
        "sample_plan_digest": digest,
        "sample_index_digest": digest,
        "boundary_digest": digest,
        "planning_h2d_bytes": 10,
        "sampling_subphases_s": {name: 0.1 for name in REQUIRED_SUBPHASES},
    }
    assert not _sampling_reasons(telemetry, expected_block_count=EXPECTED_BLOCKS)
    broken = dict(telemetry, sample_index_digest="not-a-digest")
    assert _sampling_reasons(broken, expected_block_count=EXPECTED_BLOCKS)
    observation = {
        "cold_sort_s": 49.0,
        "sampling_s": 0.8,
        "h2d_bytes": 64 * GIB,
        "output_balance": {"max_over_min": 1.05},
        "sampling": telemetry,
    }
    assert all(
        _gate_results(
            [observation, dict(observation)],
            health_reasons=[],
            telemetry_reasons=[],
        ).values()
    )
    changed = dict(observation, sampling=dict(telemetry, boundary_digest="b" * 64))
    gates = _gate_results(
        [observation, changed], health_reasons=[], telemetry_reasons=[]
    )
    assert not gates["deterministic_digests_across_fresh_runtimes"]
    rendered_observation = {
        **observation,
        "repetition": 1,
        "rank_count": 16,
        "peak_device_bytes": 10 * GIB,
        "nvml": {"peak_bytes": 11 * GIB},
        "externalized_bytes": 0,
        "ray_disk_spill_bytes": 0,
        "mpf_host_spill_bytes": 0,
        "cpu_sort_rows": 0,
        "cpu_merge_rows": 0,
        "fallback_count": 0,
    }
    report_summary = {
        "status": "accepted",
        "aws_gate_passed": True,
        "baseline": {"median_s": BASELINE_MEDIAN_S},
        "candidate": {
            "observations": [
                rendered_observation,
                dict(rendered_observation, repetition=2),
            ],
            "median_s": 49.0,
            "sampling_median_s": 0.8,
            "change_vs_cpu_sampled_baseline_fraction": 49.0 / BASELINE_MEDIAN_S - 1,
        },
        "acceptance_gates": gates,
        "rejection_reasons": [],
    }
    assert "Deterministic Stratified Sampling" in _render(report_summary)
    print("stratified gate self-check passed")


def parser() -> argparse.ArgumentParser:
    root = _repo_root()
    result = argparse.ArgumentParser(
        description="Run the deterministic-stratified sampling DGX gate."
    )
    result.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    result.add_argument(
        "--artifact-root",
        type=Path,
        default=root / ".venv/gpu-sort-stratified-gate-artifacts",
    )
    result.add_argument(
        "--runtime-root",
        type=Path,
        default=root / ".venv/gpu-sort-stratified-gate-runtime",
    )
    result.add_argument(
        "--baseline-summary",
        type=Path,
        default=root / ".venv/gpu-sort-sampling-ab-artifacts/sampling-ab.json",
    )
    result.add_argument("--timeout-s", type=int, default=14_400)
    result.add_argument("--self-check", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.self_check:
        _self_check()
        return 0

    stage = "baseline validation"
    report = args.artifact_root.resolve() / "STRATIFIED_SAMPLING_DGX_GATE.md"
    summary_path = args.artifact_root.resolve() / "stratified-sampling-dgx-gate.json"
    try:
        baseline_path = args.baseline_summary.resolve()
        baseline = _load_baseline(baseline_path)
        stage = "harness initialization"
        study = Study(args)
        stage = "exact PyArrow/GPU smoke"
        smoke = _run_smoke(study)
        stage = "two fresh-runtime 64-GiB observations"
        candidates = [
            study.trial(
                kind="trend",
                backend="gpu",
                cell="full",
                repetition=repetition,
                name=f"stratified-full-gpu-r{repetition}",
                reuse=False,
            )
            for repetition in (1, 2)
        ]
        stage = "acceptance evaluation"
        summary = _build_summary(
            study=study,
            baseline_path=baseline_path,
            baseline=baseline,
            smoke=smoke,
            candidates=candidates,
        )
        write_json(summary_path, summary)
        report.write_text(_render(summary), encoding="utf-8")
        print(report)
        return 0 if summary["aws_gate_passed"] else 1
    except BaseException as error:
        report.parent.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema_version": 1,
            "kind": "deterministic_stratified_sampling_dgx_gate",
            "status": "rejected",
            "aws_gate_passed": False,
            "failed_stage": stage,
            "rejection_reasons": [f"{type(error).__name__}: {error}"],
        }
        write_json(summary_path, failure)
        report.write_text(_failure_report(stage, error), encoding="utf-8")
        print(report)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
