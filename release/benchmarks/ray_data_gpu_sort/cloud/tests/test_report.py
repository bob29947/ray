import json
from pathlib import Path
from typing import Any

from release.benchmarks.ray_data_gpu_sort.cloud.report import (
    LOCATION_WARNING,
    OUTPUT_LOCATION_WARNING,
    _classification,
    build,
)
from release.benchmarks.ray_data_gpu_sort.cloud.gpu_only_report import (
    _repeat_digests,
    _sampling_comparison,
)


def _result(
    backend: str,
    *,
    cell: str = "full",
    repetition: int = 1,
    elapsed: float | None = 10.0,
    valid: bool = True,
    reasons: list[str] | None = None,
    scale: tuple[int, int] = (1, 1),
    ray_write: int = 0,
    externalized: int = 0,
) -> dict[str, Any]:
    source = {"rows": 100, "blocks": 2, "decoded_bytes": 1000, "schema": "a: int64"}
    completed = elapsed is not None
    gpu = backend == "gpu"
    return {
        "valid": valid,
        "status": "accepted" if valid else "rejected",
        "kind": "trend" if scale == (1, 1) else "natural",
        "backend": backend,
        "cell": {"name": cell, "label": cell, "columns": ["a"], "keys": ["a"]},
        "repetition": repetition,
        "scale_numerator": scale[0],
        "scale_denominator": scale[1],
        "wave_fraction": 0.5 if gpu else None,
        "cold_sort_s": elapsed,
        "input": source,
        "output": source if completed else {},
        "throughput_rows_s": None if elapsed is None else 100 / elapsed,
        "throughput_gib_s": None if elapsed is None else 1000 / (1 << 30) / elapsed,
        "validation": (
            {"ordered": True, "row_id_sum": 4950, "expected_row_id_sum": 4950}
            if completed
            else {}
        ),
        "rejection_reasons": reasons or [],
        "ray_object_store_io": {
            "totals": {"spilled_bytes_total": ray_write, "restored_bytes_total": 0}
        },
        "gpu_stats": (
            {
                "externalized_bytes": externalized,
                "initial_run_count": 2 if externalized else 0,
                "replacement_run_count": 1 if externalized else 0,
                "merge_pass_count": 1 if externalized else 0,
                "h2d_bytes": 2000,
                "d2h_bytes": 1000,
                "plasma_read_bytes": externalized,
                "plasma_write_bytes": 1000 + externalized,
                "peak_device_bytes": 500,
                "phases_s": {"sampling": 1.0, "partition": 2.0},
                "ranks": [
                    {
                        "rank": 0,
                        "input_bytes": 1000,
                        "local_input_bytes": 1000,
                        "output_bytes": 1000,
                    }
                ],
            }
            if gpu and completed
            else {}
        ),
        "resources": {
            "nodes": [
                {
                    "gpu_peak_memory_used_bytes": 600,
                    "gpu_total_memory_bytes": 1000,
                }
            ],
            "totals": {"network_sent_bytes_delta": 100, "network_recv_bytes_delta": 90},
        },
    }


def _write(root: Path, name: str, value: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(json.dumps(value), encoding="utf-8")


def test_gpu_only_report_requires_repeatable_sampling_digests() -> None:
    first = _result("gpu")
    second = _result("gpu", repetition=2)
    sampling = {
        "sample_plan_digest": "a" * 64,
        "sample_index_digest": "b" * 64,
        "boundary_digest": "c" * 64,
    }
    first["gpu_stats"].update(sampling)
    second["gpu_stats"].update(sampling)
    assert _repeat_digests([first, second])["identical"] is True

    second["gpu_stats"]["boundary_digest"] = "d" * 64
    assert _repeat_digests([first, second])["identical"] is False


def test_gpu_only_report_compares_old_and_new_sampling() -> None:
    old = [_result("gpu"), _result("gpu", repetition=2)]
    new = [_result("gpu"), _result("gpu", repetition=2)]
    for value in new:
        value["gpu_stats"]["phases_s"]["sampling"] = 0.5
        value["gpu_stats"]["planning_h2d_bytes"] = 4096
        value["gpu_stats"]["sampling_subphases_s"] = {
            "cpu_sample_construction": 0.3,
            "boundary_sort": 0.1,
            "orchestration_remainder": 0.1,
        }
    comparison = _sampling_comparison(new, old)
    assert comparison["archived_fixed_stride_median_s"] == 1.0
    assert comparison["stratified_median_s"] == 0.5
    assert comparison["stratified_planning_h2d_mib"] == 4096 / (1 << 20)


def test_classification_preserves_location_warning_but_not_oom() -> None:
    warning = _result("gpu", valid=False, reasons=[LOCATION_WARNING])
    assert _classification(warning) == "telemetry-warning"
    oom = _result(
        "gpu",
        elapsed=None,
        valid=False,
        reasons=[
            "sort failed: MemoryError: std::bad_alloc: Maximum pool size exceeded"
        ],
    )
    assert _classification(oom) == "oom"


def test_classification_accepts_only_completed_location_metadata_warnings() -> None:
    output_only = _result("pyarrow", valid=False, reasons=[OUTPUT_LOCATION_WARNING])
    both = _result(
        "pyarrow",
        valid=False,
        reasons=[LOCATION_WARNING, OUTPUT_LOCATION_WARNING],
    )
    extra = _result(
        "pyarrow",
        valid=False,
        reasons=[OUTPUT_LOCATION_WARNING, "output row count differs from input"],
    )
    incomplete = _result(
        "pyarrow",
        elapsed=None,
        valid=False,
        reasons=[OUTPUT_LOCATION_WARNING],
    )
    assert _classification(output_only) == "telemetry-warning"
    assert _classification(both) == "telemetry-warning"
    assert _classification(extra) == "failed"
    assert _classification(incomplete) == "failed"


def test_report_keeps_raw_warning_times_and_prefailure_spill(tmp_path: Path) -> None:
    gpu, cpu = tmp_path / "gpu", tmp_path / "cpu"
    accepted = _result("gpu")
    warning = _result("gpu", valid=False, reasons=[LOCATION_WARNING])
    _write(gpu, "trend-narrow-r1.json", warning)
    _write(gpu, "trend-narrow-r2.json", accepted)
    _write(cpu, "trend-narrow-r1.json", _result("pyarrow", elapsed=40.0))
    _write(gpu, "trend-full-r1.json", accepted)
    _write(gpu, "trend-full-r2.json", _result("gpu", repetition=2, elapsed=12.0))
    _write(cpu, "trend-full-r1.json", _result("pyarrow", elapsed=50.0))

    tune = _result("gpu", elapsed=20.0, scale=(2, 1), ray_write=300, externalized=1000)
    tune["wave_fraction"] = 0.5
    selected = _result(
        "gpu",
        repetition=2,
        elapsed=22.0,
        scale=(2, 1),
        ray_write=320,
        externalized=1000,
    )
    _write(gpu, "tune-2x-wave-0500.json", tune)
    rejected_tune = _result("gpu", elapsed=21.0, scale=(2, 1))
    rejected_tune["wave_fraction"] = 0.375
    _write(gpu, "tune-2x-wave-0375.json", rejected_tune)
    _write(gpu, "natural-2x-selected-r2.json", selected)
    _write(
        cpu,
        "natural-2x-r1.json",
        _result("pyarrow", elapsed=80.0, scale=(2, 1), ray_write=200),
    )

    oom_reason = (
        "sort failed: MemoryError in _externalize_device_tables: std::bad_alloc: "
        "Maximum pool size exceeded (failed to allocate 180 MiB): "
        "current/max/try size = 18 GiB, 19 GiB, 180 MiB"
    )
    for repetition, spill in ((1, 400), (2, 600)):
        _write(
            gpu,
            f"natural-245x-r{repetition}.json",
            _result(
                "gpu",
                repetition=repetition,
                elapsed=None,
                valid=False,
                reasons=[oom_reason],
                scale=(245, 100),
                ray_write=spill,
            ),
        )
    _write(
        cpu,
        "natural-245x-r1.json",
        _result("pyarrow", elapsed=100.0, scale=(245, 100), ray_write=250),
    )

    smoke = _result("gpu", elapsed=2.0)
    smoke.update(
        {
            "kind": "smoke",
            "backend": "smoke",
            "cpu_sort_s": 1.0,
            "gpu_sort_s": 2.0,
            "validation": {"exact_every_row_value": True, "rows": 100},
        }
    )
    _write(gpu, "transport-smoke.json", smoke)
    (gpu / "selected-wave.json").write_text(
        json.dumps(
            {
                "selected_wave_fraction": 0.5,
                "reason": "candidate missed gate",
                "observed_relative_improvement": -0.05,
                "minimum_relative_improvement": 0.03,
            }
        ),
        encoding="utf-8",
    )
    study = tmp_path / "study.json"
    study.write_text(
        json.dumps({"campaign": "test", "plan_sha256": "abc"}), encoding="utf-8"
    )

    output = tmp_path / "report.md"
    report = build(study, gpu, cpu, output)
    narrow = next(row for row in report["trend"]["rows"] if row["cell"] == "narrow")
    assert narrow["gpu_s"] == [10.0, 10.0]
    assert narrow["gpu_observed_median_s"] == 10.0
    assert narrow["gpu_strict_median_s"] is None
    assert narrow["comparison_status"] == "telemetry-warning"

    large = next(
        row for row in report["natural_spill"]["rows"] if row["scale"] == "2.45x"
    )
    assert large["comparison_status"] == "oom"
    assert large["gpu_s"] == [None, None]
    assert [item["ray_write_gib"] for item in large["gpu_io"]] == [
        400 / (1 << 30),
        600 / (1 << 30),
    ]
    markdown = output.read_text(encoding="utf-8")
    assert "RMM OOM during GPU run externalization" in markdown
    assert "203.95/15.93 (12.80×)" in markdown
    assert "telemetry-warning" in markdown
