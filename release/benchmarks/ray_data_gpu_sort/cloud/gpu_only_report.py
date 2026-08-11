"""Report a GPU-only AWS rerun against the immutable finalized AWS campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from .common import atomic_json, digest, file_sha256, read_json
from .report import (
    GIB,
    _bundle_artifact_provenance,
    _classification,
    _gpu_metrics,
    _io_metrics,
    _number,
    _phase_medians,
    _repair_execution_provenance,
    _repair_staged_manifest,
    _results,
    _selection,
    _time,
    _validated_study,
)
from .spec import CELLS, gpu_trials
from .worker import _sampling_telemetry_reasons

ARCHIVED_REPORT_SHA256 = (
    "96a08625bc6709a68e085f1c090614d18ad229635e5bfd1ba1b51412912cf534"
)
EXPECTED_ARCHIVED_CAMPAIGN = "bts-external-cloud-20260807-g"


def _archived(path: Path) -> dict[str, Any]:
    if file_sha256(path) != ARCHIVED_REPORT_SHA256:
        raise ValueError("archived AWS report differs from its frozen SHA-256")
    value = read_json(path)
    if (
        value.get("kind") != "gpu_sort_bts_cloud_report"
        or value.get("campaign") != EXPECTED_ARCHIVED_CAMPAIGN
        or [row.get("cell") for row in value.get("trend", {}).get("rows", [])]
        != list(CELLS)
        or [row.get("scale") for row in value.get("natural_spill", {}).get("rows", [])]
        != ["1x", "2x", "2.45x"]
    ):
        raise ValueError("archived AWS report is not the finalized BTS campaign")
    return value


def _validate_plan(path: Path) -> dict[str, Any]:
    plan = _validated_study(path)
    if plan.get("trial_mode") != "gpu-only" or set(plan.get("arms", {})) != {"gpu"}:
        raise ValueError("GPU-only report requires an exact gpu-only study")
    expected = [trial.to_dict() for trial in gpu_trials()]
    if plan["arms"]["gpu"].get("trials") != expected:
        raise ValueError("GPU-only study does not contain the exact frozen GPU matrix")
    return plan


def _expected_wave(trial: Mapping[str, Any], selected: float) -> float:
    if trial.get("wave_fraction") is not None:
        return float(trial["wave_fraction"])
    if trial.get("selected_wave_file") is not None:
        return selected
    return 0.50


def _validate_artifacts(
    plan: Mapping[str, Any], root: Path, results: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    expected_names = {f"{trial.name}.json" for trial in gpu_trials()}
    if set(results) != expected_names:
        raise ValueError(
            "GPU-only result set differs from the frozen matrix: "
            f"missing={sorted(expected_names - set(results))}, "
            f"unexpected={sorted(set(results) - expected_names)}"
        )
    selection = _selection(root)
    if not isinstance(selection, Mapping):
        raise ValueError("GPU-only result tree is missing selected-wave.json")
    selected = _number(selection.get("selected_wave_fraction"))
    if selected not in (0.50, 0.375):
        raise ValueError("GPU-only selected wave is neither 0.50 nor 0.375")

    execution = _repair_execution_provenance(plan, root, arm="gpu")
    staged_manifest = _repair_staged_manifest(
        plan,
        root,
        arm="gpu",
        expected_instance_ids=execution["instance_ids"],
    )
    provenance = _bundle_artifact_provenance(plan)
    by_name = {trial.name: trial.to_dict() for trial in gpu_trials()}
    for filename, artifact in results.items():
        name = filename.removesuffix(".json")
        trial = by_name[name]
        cell = artifact.get("cell")
        cell_name = cell.get("name") if isinstance(cell, Mapping) else cell
        expected_trial = {
            "kind": trial["kind"],
            "backend": trial["backend"],
            "cell": trial["cell"],
            "repetition": trial["repetition"],
            "scale_numerator": trial["scale_numerator"],
            "scale_denominator": trial["scale_denominator"],
            "wave_fraction": _expected_wave(trial, float(selected)),
        }
        observed_trial = {
            "kind": artifact.get("kind"),
            "backend": artifact.get("backend"),
            "cell": cell_name,
            "repetition": artifact.get("repetition"),
            "scale_numerator": artifact.get("scale_numerator"),
            "scale_denominator": artifact.get("scale_denominator"),
            "wave_fraction": artifact.get("wave_fraction"),
        }
        if observed_trial != expected_trial:
            raise ValueError(f"GPU-only artifact differs from its trial: {filename}")
        identity = artifact.get("artifact_identity") or {}
        identity_body = {key: item for key, item in identity.items() if key != "digest"}
        identity_trial = {
            key: expected_trial[key] for key in expected_trial if key != "cell"
        }
        identity_trial["cell"] = expected_trial["cell"]
        if (
            identity.get("digest") != digest(identity_body)
            or identity.get("trial") != identity_trial
            or identity.get("bundle_digest") != plan["bundle"]["bundle_digest"]
            or identity.get("dataset_manifest_sha256") != staged_manifest
            or identity.get("wheel_sha256") != provenance["wheel_sha256"]
            or identity.get("harness_files") != provenance["harness_files"]
            or identity.get("ray_data_overlay") != provenance["ray_data_overlay"]
            or identity.get("input_plan_digest")
            != (artifact.get("plan") or {}).get("digest")
            or artifact.get("ray_version") != plan["bundle"]["ray_version"]
            or artifact.get("ray_commit") != plan["bundle"]["ray_commit"]
        ):
            raise ValueError(f"GPU-only artifact provenance differs: {filename}")
    return {
        "selection": dict(selection),
        "execution": execution,
        "staged_manifest_sha256": staged_manifest,
        **provenance,
    }


def _sampling(value: Mapping[str, Any]) -> dict[str, Any]:
    stats = value.get("gpu_stats")
    stats = stats if isinstance(stats, Mapping) else {}
    source = value.get("input")
    source = source if isinstance(source, Mapping) else {}
    sub = stats.get("sampling_subphases_s")
    sub = sub if isinstance(sub, Mapping) else {}
    return {
        "mode": stats.get("sampling_mode"),
        "scheme": stats.get("sampling_scheme"),
        "scheme_version": stats.get("sampling_scheme_version"),
        "seed": stats.get("sample_seed"),
        "target_rows": stats.get("sample_target_rows"),
        "rows": stats.get("sample_rows"),
        "bytes": stats.get("sample_bytes"),
        "planning_bytes": stats.get("planning_sample_bytes"),
        "sampled_blocks": stats.get("sampled_block_count"),
        "quota_rows": stats.get("sample_quota_rows"),
        "plan_digest": stats.get("sample_plan_digest"),
        "index_digest": stats.get("sample_index_digest"),
        "boundary_digest": stats.get("boundary_digest"),
        "planning_h2d_bytes": stats.get("planning_h2d_bytes"),
        "subphases_s": dict(sub),
        "validation_reasons": _sampling_telemetry_reasons(
            stats,
            input_rows=source.get("rows"),
            input_blocks=source.get("blocks"),
        ),
    }


def _summary(value: Mapping[str, Any]) -> dict[str, Any]:
    stats = value.get("gpu_stats")
    stats = stats if isinstance(stats, Mapping) else {}
    return {
        "classification": _classification(value),
        "cold_sort_s": _number(value.get("cold_sort_s")),
        "throughput_rows_s": _number(value.get("throughput_rows_s")),
        "throughput_gib_s": _number(value.get("throughput_gib_s")),
        "rejection_reasons": list(value.get("rejection_reasons") or []),
        "sampling": _sampling(value),
        "io": _io_metrics(value),
        "gpu": _gpu_metrics(value),
        "phases_s": dict(stats.get("phases_s") or {}),
        "artifact_path": value.get("_artifact_path"),
    }


def _usable_median(values: Sequence[Mapping[str, Any]]) -> float | None:
    times = [_time(value) for value in values]
    return (
        median([float(item) for item in times if item is not None])
        if len([item for item in times if item is not None]) == 2
        else None
    )


def _complete_median(values: Sequence[Mapping[str, Any]], getter: Any) -> float | None:
    measured = [_number(getter(value)) for value in values]
    return (
        median(float(item) for item in measured if item is not None)
        if measured and all(item is not None for item in measured)
        else None
    )


def _repeat_digests(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = ("plan_digest", "index_digest", "boundary_digest")
    sampled = [_sampling(value) for value in values]
    return {name: [item[name] for item in sampled] for name in fields} | {
        "identical": all(len({item[name] for item in sampled}) == 1 for name in fields)
    }


def _sampling_comparison(
    values: Sequence[Mapping[str, Any]],
    archived_values: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    def phase(value: Mapping[str, Any], name: str) -> Any:
        return (value.get("gpu_stats") or {}).get("phases_s", {}).get(name)

    def subphase(value: Mapping[str, Any], name: str) -> Any:
        return (value.get("gpu_stats") or {}).get("sampling_subphases_s", {}).get(name)

    planning_h2d = _complete_median(
        values,
        lambda value: (value.get("gpu_stats") or {}).get("planning_h2d_bytes"),
    )
    return {
        "archived_fixed_stride_median_s": _complete_median(
            archived_values, lambda value: phase(value, "sampling")
        ),
        "stratified_median_s": _complete_median(
            values, lambda value: phase(value, "sampling")
        ),
        "stratified_subphase_medians_s": {
            name: _complete_median(
                values, lambda value, phase_name=name: subphase(value, phase_name)
            )
            for name in (
                "cpu_sample_construction",
                "boundary_sort",
                "orchestration_remainder",
            )
        },
        "stratified_planning_h2d_mib": (
            None if planning_h2d is None else planning_h2d / (1 << 20)
        ),
        "archived_total_h2d_amplification": _complete_median(
            archived_values,
            lambda value: _io_metrics(value).get("h2d_amplification"),
        ),
        "stratified_total_h2d_amplification": _complete_median(
            values, lambda value: _io_metrics(value).get("h2d_amplification")
        ),
        "archived_output_balance": _complete_median(
            archived_values,
            lambda value: _gpu_metrics(value).get("output_balance_max_over_min"),
        ),
        "stratified_output_balance": _complete_median(
            values,
            lambda value: _gpu_metrics(value).get("output_balance_max_over_min"),
        ),
    }


def _trend_rows(
    current: Mapping[str, Mapping[str, Any]], archived: Mapping[str, Any]
) -> list[dict[str, Any]]:
    old = {row["cell"]: row for row in archived["trend"]["rows"]}
    rows = []
    for cell in CELLS:
        values = [current[f"trend-{cell}-r{rep}.json"] for rep in (1, 2)]
        archived_values = old[cell].get("gpu_observations") or []
        source = values[0].get("input") or {}
        new_median = _usable_median(values)
        old_gpu = _number(old[cell].get("gpu_observed_median_s"))
        cpu = _number(old[cell].get("cpu_s"))
        rows.append(
            {
                "cell": cell,
                "rows": source.get("rows"),
                "blocks": source.get("blocks"),
                "decoded_gib": (_number(source.get("decoded_bytes")) or 0) / GIB,
                "gpu_s": [_number(value.get("cold_sort_s")) for value in values],
                "gpu_median_s": new_median,
                "archived_gpu_median_s": old_gpu,
                "gpu_change_percent": None
                if new_median is None or not old_gpu
                else 100 * (new_median - old_gpu) / old_gpu,
                "archived_cpu_s": cpu,
                "archived_cpu_classification": old[cell].get("cpu_classification"),
                "directional_cpu_gpu_speedup": None
                if new_median is None or not cpu
                else cpu / new_median,
                "throughput_gib_s": None
                if new_median is None
                else ((_number(source.get("decoded_bytes")) or 0) / GIB) / new_median,
                "repeat_digests": _repeat_digests(values),
                "sampling_comparison": _sampling_comparison(values, archived_values),
                "phase_medians_s": _phase_medians(values),
                "observations": values,
                "observation_summaries": [_summary(value) for value in values],
            }
        )
    return rows


def _natural_values(
    current: Mapping[str, Mapping[str, Any]], selected: float, scale: str
) -> list[Mapping[str, Any]]:
    if scale == "1x":
        return [current[f"trend-full-r{rep}.json"] for rep in (1, 2)]
    if scale == "2x":
        screen = (
            "tune-2x-wave-0500.json" if selected == 0.50 else "tune-2x-wave-0375.json"
        )
        return [current[screen], current["natural-2x-selected-r2.json"]]
    return [current[f"natural-245x-r{rep}.json"] for rep in (1, 2)]


def _natural_rows(
    current: Mapping[str, Mapping[str, Any]],
    archived: Mapping[str, Any],
    selected: float,
) -> list[dict[str, Any]]:
    old = {row["scale"]: row for row in archived["natural_spill"]["rows"]}
    rows = []
    for scale in ("1x", "2x", "2.45x"):
        values = _natural_values(current, selected, scale)
        source = values[0].get("input") or {}
        new_median = _usable_median(values)
        old_gpu = _number(old[scale].get("gpu_observed_median_s"))
        cpu = _number(old[scale].get("cpu_s"))
        io = [_io_metrics(value) for value in values]
        archived_cpu_io = old[scale].get("cpu_io") or {}
        median_ray_write_gib = median(
            [float(item["ray_write_gib"] or 0) for item in io]
        )
        median_ray_restore_gib = median(
            [float(item["ray_restore_gib"] or 0) for item in io]
        )
        archived_cpu_ray_write_gib = _number(archived_cpu_io.get("ray_write_gib"))
        archived_cpu_ray_restore_gib = _number(archived_cpu_io.get("ray_restore_gib"))
        rows.append(
            {
                "scale": scale,
                "rows": source.get("rows"),
                "blocks": source.get("blocks"),
                "decoded_gib": (_number(source.get("decoded_bytes")) or 0) / GIB,
                "gpu_s": [_number(value.get("cold_sort_s")) for value in values],
                "gpu_median_s": new_median,
                "archived_gpu_median_s": old_gpu,
                "gpu_change_percent": None
                if new_median is None or not old_gpu
                else 100 * (new_median - old_gpu) / old_gpu,
                "archived_cpu_s": cpu,
                "directional_cpu_gpu_speedup": None
                if new_median is None or not cpu
                else cpu / new_median,
                "archived_cpu_classification": old[scale].get("cpu_classification"),
                "archived_cpu_io": archived_cpu_io,
                "io": io,
                "median_externalized_gib": median(
                    [float(item["externalized_gib"] or 0) for item in io]
                ),
                "median_ray_write_gib": median_ray_write_gib,
                "median_ray_restore_gib": median_ray_restore_gib,
                "gpu_cpu_ray_write_ratio": None
                if not archived_cpu_ray_write_gib
                else median_ray_write_gib / archived_cpu_ray_write_gib,
                "gpu_cpu_ray_restore_ratio": None
                if not archived_cpu_ray_restore_gib
                else median_ray_restore_gib / archived_cpu_ray_restore_gib,
                "repeat_digests": _repeat_digests(values),
                "phase_medians_s": _phase_medians(values),
                "observations": values,
                "observation_summaries": [_summary(value) for value in values],
            }
        )
    return rows


def _ratio(
    rows: Mapping[str, Mapping[str, Any]], left: str, right: str
) -> float | None:
    a = _number(rows[left].get("gpu_median_s"))
    b = _number(rows[right].get("gpu_median_s"))
    return a / b if a is not None and b else None


def build(
    study: Path, gpu_root: Path, archived_report: Path, output: Path
) -> dict[str, Any]:
    plan = _validate_plan(study)
    archived = _archived(archived_report)
    current = _results(gpu_root)
    provenance = _validate_artifacts(plan, gpu_root, current)
    selected = float(provenance["selection"]["selected_wave_fraction"])
    trend = _trend_rows(current, archived)
    natural = _natural_rows(current, archived, selected)
    by_cell = {row["cell"]: row for row in trend}
    active_names = {
        "transport-smoke.json",
        "tune-2x-wave-0500.json",
        "tune-2x-wave-0375.json",
        *(f"trend-{cell}-r{rep}.json" for cell in CELLS for rep in (1, 2)),
        *(
            value.get("_artifact_path", "") and Path(str(value["_artifact_path"])).name
            for scale in ("2x", "2.45x")
            for value in _natural_values(current, selected, scale)
        ),
    }
    active = [current[name] for name in sorted(active_names) if name in current]
    performance = [value for value in active if value.get("backend") == "gpu"]
    telemetry_reasons = {
        Path(str(value.get("_artifact_path"))).name: _sampling(value)[
            "validation_reasons"
        ]
        for value in active
        if _sampling(value)["validation_reasons"]
    }
    zero_fields = (
        "cpu_sort_rows",
        "cpu_merge_rows",
        "fallback_count",
        "mpf_host_spill_bytes",
    )
    zero_gates = {
        name: bool(performance)
        and all(
            _number((value.get("gpu_stats") or {}).get(name)) == 0
            for value in performance
        )
        for name in zero_fields
    }
    smoke = current["transport-smoke.json"]
    repeat_groups = {row["cell"]: row["repeat_digests"] for row in trend} | {
        f"natural-{row['scale']}": row["repeat_digests"] for row in natural
    }
    repeat_groups["tuning-2x-both-waves-and-selected-r2"] = _repeat_digests(
        [
            current["tune-2x-wave-0500.json"],
            current["tune-2x-wave-0375.json"],
            current["natural-2x-selected-r2.json"],
        ]
    )
    acceptance = {
        "smoke_accepted": _classification(smoke) == "accepted",
        "smoke_exact": smoke.get("validation", {}).get("exact_every_row_value") is True,
        "smoke_ranks": len((smoke.get("gpu_stats") or {}).get("ranks", [])),
        "usable_performance_observations": sum(
            _classification(value) in ("accepted", "telemetry-warning")
            for value in performance
        ),
        "active_performance_observations": len(performance),
        "only_location_metadata_warnings_are_usable": all(
            _classification(value) in ("accepted", "telemetry-warning")
            for value in performance
        ),
        "sampling_telemetry_reasons": telemetry_reasons,
        "repeat_digests_identical": all(
            group["identical"] for group in repeat_groups.values()
        ),
        **{f"all_{name}_zero": passed for name, passed in zero_gates.items()},
    }
    acceptance["all_gates_passed"] = (
        all(
            value
            for key, value in acceptance.items()
            if key
            not in (
                "smoke_ranks",
                "usable_performance_observations",
                "active_performance_observations",
                "sampling_telemetry_reasons",
            )
        )
        and not telemetry_reasons
        and acceptance["smoke_ranks"] == 16
        and (
            acceptance["usable_performance_observations"]
            == acceptance["active_performance_observations"]
        )
    )

    result = {
        "schema_version": 1,
        "kind": "gpu_sort_bts_cloud_gpu_only_report",
        "campaign": plan["campaign"],
        "study_plan_sha256": plan["plan_sha256"],
        "archived_comparison": {
            "campaign": archived["campaign"],
            "path": str(archived_report.resolve()),
            "sha256": ARCHIVED_REPORT_SHA256,
            "cpu_scope": "archived cross-campaign directional denominator; CPU was not rerun",
        },
        "execution": {
            "topology": plan["arms"]["gpu"]["topology"],
            "fresh_ray_runtime_per_observation": True,
            "retained_ec2_fleet": True,
            "ray_default_plasma": True,
            "filesystem_spill_tier": "/mnt/nvme",
            "filesystem_spill_uses_dev_shm": False,
            "validated_provenance": provenance,
        },
        "acceptance": acceptance,
        "smoke": _summary(smoke),
        "tuning": {
            **provenance["selection"],
            "screens": {
                "0.50": _summary(current["tune-2x-wave-0500.json"]),
                "0.375": _summary(current["tune-2x-wave-0375.json"]),
            },
        },
        "trend": {
            "rows": trend,
            "directional": {
                "payload_narrow_to_core": _ratio(by_cell, "core", "narrow"),
                "payload_core_to_full": _ratio(by_cell, "full", "core"),
                "origin_string_over_integer": _ratio(
                    by_cell, "origin-string", "origin-integer"
                ),
                "route_over_one_key": _ratio(by_cell, "route", "origin-string"),
                "four_keys_over_one_key": _ratio(by_cell, "full", "origin-string"),
            },
        },
        "natural_spill": {"rows": natural},
        "repeat_digest_groups": repeat_groups,
    }
    atomic_json(output.with_suffix(".json"), result)

    def fmt(value: Any, digits: int = 2) -> str:
        return "n/a" if value is None else f"{float(value):.{digits}f}"

    warnings = [
        f"{row['cell']} r{index}"
        for row in trend
        for index, observation in enumerate(row["observations"], start=1)
        if _classification(observation) == "telemetry-warning"
    ]
    archived_cpu_statuses = ", ".join(
        f"{row['cell']}={row['archived_cpu_classification']}" for row in trend
    )

    lines = [
        f"# BTS AWS GPU Sort — {plan['campaign']}",
        "",
        "This is a GPU-only rerun of the finalized AWS BTS matrix with deterministic stratified-random CPU boundary planning. CPU values below are immutable results from the prior campaign and are cross-campaign directional denominators, not fresh paired observations.",
        "",
        "## Correctness, reproducibility, and tuning",
        "",
        f"- Exact 160k-row smoke: `{_classification(smoke)}`; exact rows/values `{acceptance['smoke_exact']}`; GPU ranks `{acceptance['smoke_ranks']}`.",
        f"- Planner telemetry complete: `{not telemetry_reasons}`; all repeat sample-plan/index/boundary digests identical: `{acceptance['repeat_digests_identical']}`.",
        f"- CPU sort/merge, output fallback, and MPF host spill are all zero: `{all(zero_gates.values())}`.",
        f"- Selected automatic wave fraction `{selected}`: {provenance['selection'].get('reason')}.",
        f"- Overall acceptance: `{acceptance['all_gates_passed']}`. Location metadata gaps are usable only when they are the sole warning and the completed result passes exact validation.",
        f"- Retained metadata-warning GPU observations: `{', '.join(warnings) if warnings else 'none'}`. These timings passed placement-plan, zero-pre-timer-spill, row/schema/order/checksum, and output-residency checks; only the experimental input ObjectRef-location lookup was incomplete.",
        "- Performance-cell checksum means the deterministic `row_id` sum. Exact every-column/every-value equality was checked in the 160k-row smoke, not repeated for each large cell.",
        "",
        "## Payload, datatype, and key trends",
        "",
        "| Cell | GiB | New GPU runs (s) | New median | Prior GPU | Change | Archived CPU | Directional speedup | H2D amp |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in trend:
        h2d = [item["io"]["h2d_amplification"] for item in row["observation_summaries"]]
        lines.append(
            f"| {row['cell']} | {row['decoded_gib']:.3f} | {' / '.join(fmt(v, 3) for v in row['gpu_s'])} | {fmt(row['gpu_median_s'], 3)} | {fmt(row['archived_gpu_median_s'], 3)} | {fmt(row['gpu_change_percent'], 1)}% | {fmt(row['archived_cpu_s'], 3)} | {fmt(row['directional_cpu_gpu_speedup'])}× | {' / '.join(fmt(v) for v in h2d)}× |"
        )
    directional = result["trend"]["directional"]
    lines.extend(
        [
            "",
            f"- GPU payload cost: core/narrow `{fmt(directional['payload_narrow_to_core'])}×`; full/core `{fmt(directional['payload_core_to_full'])}×`.",
            f"- Full-payload Origin string/integer GPU ratio: `{fmt(directional['origin_string_over_integer'])}×`.",
            f"- Key-count GPU ratios versus one Origin key: route `{fmt(directional['route_over_one_key'])}×`; four keys `{fmt(directional['four_keys_over_one_key'])}×`.",
            f"- Archived CPU classifications: {archived_cpu_statuses}. These remain directional cross-campaign denominators; metadata warnings are not fresh paired measurements.",
            "",
            "## Fixed-stride versus stratified planning",
            "",
            "| Cell | Old fixed-stride sampling | New stratified sampling | New CPU sample / boundary / orchestration | Planning H2D | Total H2D old / new | Output balance old / new |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in trend:
        comparison = row["sampling_comparison"]
        subphases = comparison["stratified_subphase_medians_s"]
        lines.append(
            f"| {row['cell']} | {fmt(comparison['archived_fixed_stride_median_s'], 3)}s | "
            f"{fmt(comparison['stratified_median_s'], 3)}s | "
            f"{fmt(subphases['cpu_sample_construction'], 3)} / "
            f"{fmt(subphases['boundary_sort'], 3)} / "
            f"{fmt(subphases['orchestration_remainder'], 3)}s | "
            f"{fmt(comparison['stratified_planning_h2d_mib'], 3)} MiB | "
            f"{fmt(comparison['archived_total_h2d_amplification'])} / "
            f"{fmt(comparison['stratified_total_h2d_amplification'])}× | "
            f"{fmt(comparison['archived_output_balance'])} / "
            f"{fmt(comparison['stratified_output_balance'])}× |"
        )
    lines.extend(
        [
            "",
            "The identical plan/index/boundary digests prove reproducibility, not sampling quality. Robustness comes from selecting one seeded pseudorandom row per stratum; the focused periodic-input test demonstrates the aliasing pattern that fixed strides can miss. On BTS, output balance was comparable rather than uniformly better (the single-key cells changed from about 2.04× to 2.10×).",
            "",
            "The AWS implementation change also includes the previously accepted sample-first CPU planner. The H2D and timing gains should be attributed primarily to avoiding a full GPU planning pass, not to randomization alone: on DGX, stratification was 0.7% slower than the 48.801-second CPU-sampled fixed-stride baseline (49.136 seconds), while passing the gate.",
            "",
            "## Natural spill trend",
            "",
            "| Size | New GPU runs (s) | Median | Prior GPU | Change | Archived CPU | Directional speedup | VRAM externalized | Ray NVMe writes | H2D amp | Runs / replacements / passes |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in natural:
        summaries = row["observation_summaries"]
        h2d = [item["io"]["h2d_amplification"] for item in summaries]
        geometry = [
            f"{item['io']['initial_runs']}/{item['io']['replacement_runs']}/{item['io']['merge_passes']}"
            for item in summaries
        ]
        lines.append(
            f"| {row['scale']} ({row['decoded_gib']:.3f} GiB) | {' / '.join(fmt(v, 3) for v in row['gpu_s'])} | {fmt(row['gpu_median_s'], 3)} | {fmt(row['archived_gpu_median_s'], 3)} | {fmt(row['gpu_change_percent'], 1)}% | {fmt(row['archived_cpu_s'], 3)} | {fmt(row['directional_cpu_gpu_speedup'])}× | {fmt(row['median_externalized_gib'], 2)} GiB | {fmt(row['median_ray_write_gib'], 2)} GiB | {' / '.join(fmt(v) for v in h2d)}× | {' / '.join(geometry)} |"
        )
    lines.extend(
        [
            "",
            f"At 2×, GPU Ray spill wrote `{fmt(natural[1]['median_ray_write_gib'])}` GiB versus archived CPU `{fmt(natural[1]['archived_cpu_io'].get('ray_write_gib'))}` GiB (`{fmt(natural[1]['gpu_cpu_ray_write_ratio'])}×` as much), and restored `{fmt(natural[1]['median_ray_restore_gib'])}` versus `{fmt(natural[1]['archived_cpu_io'].get('ray_restore_gib'))}` GiB (`{fmt(natural[1]['gpu_cpu_ray_restore_ratio'], 1)}×`). GPU still finished `{fmt(natural[1]['directional_cpu_gpu_speedup'])}×` faster, so additional object-store churn did not erase the compute advantage.",
            "",
            "`VRAM externalized` is cumulative sorted-run output, not simultaneous excess VRAM or final disk footprint. Every row was externalized once at 2× and 2.45×; Ray NVMe writes are cumulative object-store writes and therefore include amplification.",
            "",
            "The phase maps are inner component timings and are not an additive reconstruction of cold wall time; the measured cold time is authoritative and includes the remaining outer scheduling/orchestration gap.",
            "",
            "No CPU speedup is claimed at 2.45× because the archived default-Ray CPU observation did not complete. The companion JSON retains every raw GPU observation, sampling digest and subphase, full phase map, run geometry, per-rank memory/locality/output balance, per-node NVML/network data, and Ray spill counters.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--gpu-results", type=Path, required=True)
    parser.add_argument("--archived-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build(
        args.study.resolve(),
        args.gpu_results.resolve(),
        args.archived_report.resolve(),
        args.output.resolve(),
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
