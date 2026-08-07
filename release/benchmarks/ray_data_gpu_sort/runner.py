"""DGX-only orchestration and reporting for the compact BTS study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from statistics import median
from typing import Any

from .backend_stats import spill_geometry, value
from .spec import (
    DATASET_ROOT,
    GIB,
    OLD_DGX,
    RAY_COMMIT,
    RAY_VERSION,
    RAY_WHEEL,
    RAY_WHEEL_SHA256,
    cells,
    load_manifest,
)

PACKAGE = "release.benchmarks.ray_data_gpu_sort.worker"
ALLOWLIST = ("python/ray/data/", "release/benchmarks/ray_data_gpu_sort/")


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256_files(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(Path(__file__).parent.glob("*.py")):
        result[str(path.relative_to(root))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    return result


def _overlay_hashes(root: Path) -> dict[str, Any]:
    """Return one compact identity covering every overlaid Ray Data source file."""

    digest = hashlib.sha256()
    count = 0
    for path in sorted((root / "python/ray/data").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative = str(path.relative_to(root))
        payload = path.read_bytes()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(payload).digest())
        count += 1
    return {
        "algorithm": "sha256(path\\0sha256(contents))",
        "file_count": count,
        "sha256": digest.hexdigest(),
    }


def _wheel_identity(root: Path) -> dict[str, Any]:
    wheel = (root / ".venv/wheelhouse" / RAY_WHEEL).resolve()
    if not wheel.is_file() or not wheel.is_relative_to((root / ".venv").resolve()):
        raise RuntimeError(f"Missing exact Ray wheel: {wheel}")
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    if digest != RAY_WHEEL_SHA256:
        raise RuntimeError(f"Ray wheel digest changed: {digest}")
    return {"path": str(wheel), "sha256": digest, "bytes": wheel.stat().st_size}


def _install_ray_data_overlay(root: Path, wheel: dict[str, Any]) -> dict[str, Any]:
    """Copy only Ray Data into the wheel-backed venv, as used by the prior DGX work."""

    source = (root / "python/ray/data").resolve()
    matches = list((root / ".venv/lib").glob("python*/site-packages/ray/data"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one wheel-backed Ray Data package, found {matches}"
        )
    target = matches[0].resolve()
    venv = (root / ".venv").resolve()
    if not target.is_relative_to(venv) or target.name != "data":
        raise RuntimeError(f"Unsafe Ray Data overlay target: {target}")
    for cache in sorted(
        target.rglob("__pycache__"), key=lambda path: len(path.parts), reverse=True
    ):
        resolved = cache.resolve()
        if not resolved.is_relative_to(target) or cache.is_symlink():
            raise RuntimeError(f"Unsafe Ray Data bytecode cache: {cache}")
        shutil.rmtree(cache)
    for bytecode in target.rglob("*.pyc"):
        resolved = bytecode.resolve()
        if not resolved.is_relative_to(target) or bytecode.is_symlink():
            raise RuntimeError(f"Unsafe Ray Data bytecode file: {bytecode}")
        bytecode.unlink()
    source_python = {
        path.relative_to(source)
        for path in source.rglob("*.py")
        if "__pycache__" not in path.parts
    }
    for path in sorted(target.rglob("*.py")):
        relative = path.relative_to(target)
        if relative not in source_python:
            path.unlink()
    shutil.copytree(
        source,
        target,
        dirs_exist_ok=True,
        copy_function=shutil.copy2,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    files = {
        str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source.rglob("*.py"))
        if "__pycache__" not in path.parts
    }
    target_python = {
        path.relative_to(target)
        for path in target.rglob("*.py")
        if "__pycache__" not in path.parts
    }
    if target_python != source_python:
        raise RuntimeError(
            "Ray Data overlay Python file set differs from the source tree"
        )
    manifest = {
        "source": str(source),
        "target": str(target),
        "files": files,
        "wheel": wheel,
    }
    write_json(root / ".venv/gpu-sort-ray-data-overlay.json", manifest)
    return manifest


def _change_group(path: str) -> str:
    if path.startswith("python/ray/data/tests/"):
        return "tests"
    if path.startswith("python/ray/data/"):
        return "production"
    return "benchmarks"


def _implementation_delta(root: Path, base: str = RAY_COMMIT) -> dict[str, Any]:
    """Describe the exact allowlisted working tree relative to an explicit base."""

    pathspecs = ("python/ray/data", "release/benchmarks/ray_data_gpu_sort")

    def command(*parts: str) -> str:
        return subprocess.check_output(parts, cwd=root, text=True)

    head = command("git", "rev-parse", "HEAD").strip()
    tracked_paths = set(
        filter(
            None,
            command(
                "git",
                "diff",
                "--name-only",
                "--no-renames",
                base,
                "--",
                *pathspecs,
            ).splitlines(),
        )
    )
    untracked_paths = set(
        filter(
            None,
            command(
                "git",
                "ls-files",
                "--others",
                "--exclude-standard",
                "--",
                *pathspecs,
            ).splitlines(),
        )
    )
    changed_paths = sorted(tracked_paths | untracked_paths)

    numstat: dict[str, tuple[int, int, bool]] = {}
    for line in command(
        "git", "diff", "--numstat", "--no-renames", base, "--", *pathspecs
    ).splitlines():
        added, deleted, path = line.split("\t", 2)
        binary = added == "-" or deleted == "-"
        numstat[path] = (
            0 if binary else int(added),
            0 if binary else int(deleted),
            binary,
        )

    manifest = []
    counts = {
        name: {"files": 0, "added_lines": 0, "deleted_lines": 0, "binary_files": 0}
        for name in ("production", "tests", "benchmarks", "total")
    }
    for path in changed_paths:
        current = root / path
        payload = current.read_bytes() if current.is_file() else None
        if path in untracked_paths and path not in numstat:
            binary = payload is not None and b"\0" in payload
            added = 0 if binary or payload is None else len(payload.splitlines())
            deleted = 0
            numstat[path] = (added, deleted, binary)
        added, deleted, binary = numstat.get(path, (0, 0, False))
        group = _change_group(path)
        for name in (group, "total"):
            counts[name]["files"] += 1
            counts[name]["added_lines"] += added
            counts[name]["deleted_lines"] += deleted
            counts[name]["binary_files"] += int(binary)
        manifest.append(
            {
                "path": path,
                "state": "untracked" if path in untracked_paths else "tracked",
                "bytes": len(payload) if payload is not None else None,
                "sha256": hashlib.sha256(payload).hexdigest()
                if payload is not None
                else None,
            }
        )

    status = command("git", "status", "--short", "--untracked-files=all").splitlines()
    status_paths = []
    for line in status:
        path = line[3:]
        if " -> " in path:
            path = path.rsplit(" -> ", 1)[1]
        status_paths.append(path.strip('"'))
    outside = sorted(
        path
        for path in status_paths
        if not any(path.startswith(prefix) for prefix in ALLOWLIST)
    )
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return {
        "base": base,
        "head": head,
        "current_diff_sha256": hashlib.sha256(encoded).hexdigest(),
        "changed_paths": changed_paths,
        "untracked_paths": sorted(untracked_paths),
        "counts": counts,
        "worktree_status": status,
        "allowlist_valid": not outside,
        "out_of_allowlist_paths": outside,
    }


def _git_identity(root: Path) -> dict[str, Any]:
    try:
        return _implementation_delta(root)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        return {"base": RAY_COMMIT, "error": repr(error)}


def _implementation_identity(root: Path) -> dict[str, Any]:
    result = _git_identity(root)
    try:
        result["branch"] = subprocess.check_output(
            ("git", "branch", "--show-current"), cwd=root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        result["inspection_error"] = repr(error)
    return result


def _require_valid(artifact: dict[str, Any], label: str) -> dict[str, Any]:
    if not artifact.get("valid"):
        raise RuntimeError(
            f"Required {label} artifact is rejected: {artifact.get('rejection_reasons')}"
        )
    return artifact


def _accepted(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Missing required {label} artifact: {path}")
    return _require_valid(read_json(path), label)


def _memory_summary(trials: list[dict[str, Any]]) -> tuple[int, int, int]:
    device_peak = max(
        (int(item.get("gpu_peak_device_bytes", 0)) for item in trials), default=0
    )
    nvml_peak = 0
    headroom: list[int] = []
    for item in trials:
        nvml = item.get("nvml", {})
        peaks = [int(value) for value in nvml.get("peak_used_bytes", ())]
        totals = [int(value) for value in nvml.get("total_bytes", ())]
        nvml_peak = max([nvml_peak, *peaks])
        headroom.extend(total - peak for peak, total in zip(peaks, totals))
    return device_peak, nvml_peak, min(headroom, default=0)


def _output_balance(trials: list[dict[str, Any]]) -> tuple[int, int, float]:
    output_bytes = [
        int(rank.get("output_bytes", 0))
        for trial in trials
        for rank in value(trial.get("gpu_stats", {}), "ranks", default=[])
        if int(rank.get("output_bytes", 0)) > 0
    ]
    if not output_bytes:
        return 0, 0, 0.0
    low, high = min(output_bytes), max(output_bytes)
    return low, high, high / low


def _fallback_count(trials: list[dict[str, Any]]) -> int:
    return sum(
        int(rank.get("fallback_count", 0))
        for trial in trials
        for rank in value(trial.get("gpu_stats", {}), "ranks", default=[])
    )


def _plasma_amplification(trial: dict[str, Any]) -> float:
    stats = trial.get("gpu_stats", {})
    io_bytes = int(value(stats, "plasma_read_bytes", default=0)) + int(
        value(stats, "plasma_write_bytes", default=0)
    )
    return io_bytes / max(1, int(trial["input"]["decoded_bytes"]))


def _ray_io(trial: dict[str, Any]) -> tuple[int, int]:
    counters = trial.get("ray_object_store_io") or {}
    return (
        int(counters.get("spilled_bytes_total", trial.get("ray_disk_spill_bytes", 0))),
        int(counters.get("restored_bytes_total", 0)),
    )


def _file_count(value: int) -> str:
    return f"{value} {'file' if value == 1 else 'files'}"


def _wait_for_gpu_quiescence(*, count: int = 16, timeout_s: float = 120.0) -> None:
    """Ensure a simulated cold run does not inherit retiring CUDA contexts."""

    import pynvml

    pynvml.nvmlInit()
    try:
        if pynvml.nvmlDeviceGetCount() < count:
            raise RuntimeError(f"DGX benchmark requires {count} visible GPUs")
        deadline = time.monotonic() + timeout_s
        while True:
            used = [
                int(
                    pynvml.nvmlDeviceGetMemoryInfo(
                        pynvml.nvmlDeviceGetHandleByIndex(index)
                    ).used
                )
                for index in range(count)
            ]
            if max(used, default=0) <= 1 << 30:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "GPU memory did not quiesce before the next cold run: "
                    f"max used={max(used)} bytes"
                )
            time.sleep(1.0)
    finally:
        pynvml.nvmlShutdown()


class Study:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = _repo_root()
        self.python = self.root / ".venv/bin/python"
        if Path(sys.prefix).resolve() != (self.root / ".venv").resolve():
            raise RuntimeError(f"Run this harness with {self.python}")
        if not self.python.is_file():
            raise FileNotFoundError(self.python)
        self.wheel = _wheel_identity(self.root)
        self.overlay = _install_ray_data_overlay(self.root, self.wheel)
        import ray

        if ray.__version__ != RAY_VERSION or ray.__commit__ != RAY_COMMIT:
            raise RuntimeError(
                f"Expected Ray {RAY_VERSION} at {RAY_COMMIT}, got "
                f"{ray.__version__} at {ray.__commit__}"
            )
        self.artifacts = args.artifact_root.resolve()
        self.runtime = args.runtime_root.resolve()
        self.logs = self.artifacts / "logs"
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.manifest = load_manifest(args.dataset_root.resolve())
        self.cells = {cell.name: cell for cell in cells(self.manifest)}

    def path(self, kind: str, name: str) -> Path:
        return self.artifacts / "trials" / kind / f"{name}.json"

    def trial(
        self,
        *,
        kind: str,
        backend: str,
        cell: str,
        repetition: int,
        name: str,
        budget_bytes: int | None = None,
        reuse: bool = False,
    ) -> dict[str, Any]:
        output = self.path(kind, name)
        if reuse and output.is_file():
            existing = read_json(output)
            if existing.get("valid") and existing.get("provenance", {}).get(
                "ray_overlay_sha256"
            ) == _overlay_hashes(self.root):
                print(f"[{kind}] {name}: reusing accepted observation", flush=True)
                return existing
        log = self.logs / kind / f"{name}.log"
        runtime = self.runtime / f"{kind}-{name}-{uuid.uuid4().hex[:8]}"
        output.parent.mkdir(parents=True, exist_ok=True)
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.python),
            "-m",
            PACKAGE,
            "--output",
            str(output),
            "--runtime",
            str(runtime),
            "--dataset-root",
            str(self.args.dataset_root),
            "--kind",
            kind,
            "--backend",
            backend,
            "--cell",
            cell,
            "--repetition",
            str(repetition),
        ]
        if budget_bytes is not None:
            command.extend(("--budget-bytes", str(budget_bytes)))
        print(
            f"[{kind}] {name}: {backend}, budget={budget_bytes or 'default'}",
            flush=True,
        )
        if backend == "gpu":
            _wait_for_gpu_quiescence()
        started = time.time()
        try:
            with log.open("w", encoding="utf-8") as stream:
                result = subprocess.run(
                    command,
                    cwd=self.root,
                    env={
                        **os.environ,
                        # The repository root exposes ``release``. Do not add
                        # root/python: the compiled ray._raylet remains wheel-backed.
                        "PYTHONPATH": str(self.root),
                    },
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=self.args.timeout_s,
                )
            if not output.is_file():
                raise RuntimeError(f"Worker produced no artifact; see {log}")
            artifact = read_json(output)
            artifact["provenance"] = {
                "command": command,
                "worker_returncode": result.returncode,
                "wheel": self.wheel,
                "wall_started_unix_s": started,
                "git": _git_identity(self.root),
                "harness_sha256": _sha256_files(self.root),
                "ray_overlay_sha256": _overlay_hashes(self.root),
                "overlay_manifest_sha256": hashlib.sha256(
                    (self.root / ".venv/gpu-sort-ray-data-overlay.json").read_bytes()
                ).hexdigest(),
            }
            write_json(output, artifact)
            if result.returncode or not artifact.get("valid"):
                raise RuntimeError(
                    f"Rejected {name} (worker return code {result.returncode}): "
                    f"{artifact.get('rejection_reasons')}; see {log}"
                )
            print(f"  accepted in {artifact['cold_sort_s']:.3f}s", flush=True)
            return artifact
        finally:
            resolved = runtime.resolve()
            if resolved.parent == self.runtime and resolved.name.startswith(
                f"{kind}-{name}-"
            ):
                shutil.rmtree(resolved, ignore_errors=True)

    def smoke(self) -> None:
        results = [
            self.trial(
                kind="smoke",
                backend="pyarrow",
                cell="full",
                repetition=1,
                name="pyarrow",
            ),
            self.trial(
                kind="smoke",
                backend="gpu",
                cell="full",
                repetition=1,
                name="gpu-resident",
            ),
            self.trial(
                kind="smoke",
                backend="gpu",
                cell="full",
                repetition=1,
                name="gpu-spill",
                budget_bytes=self.args.smoke_budget_mib << 20,
            ),
        ]
        import pyarrow as pa
        import pyarrow.compute as pc

        def semantic_table(result: dict[str, Any]) -> Any:
            path = result["validation"]["snapshot_path"]
            with pa.memory_map(path, "r") as source:
                table = pa.ipc.open_file(source).read_all().combine_chunks()
            arrays = []
            names = []
            for field, column in zip(table.schema, table.columns):
                arrays.append(column)
                names.append(field.name)
                if pa.types.is_floating(field.type):
                    nan_mask = pc.fill_null(pc.is_nan(column), False)
                    arrays[-1] = pc.if_else(
                        nan_mask, pa.scalar(0, type=field.type), column
                    )
                    arrays.append(nan_mask)
                    names.append(f"__nan__{field.name}")
            return table, pa.Table.from_arrays(arrays, names=names)

        baseline, baseline_semantic = semantic_table(results[0])
        for result in results[1:]:
            candidate, candidate_semantic = semantic_table(result)
            if not baseline.schema.equals(candidate.schema, check_metadata=True):
                raise RuntimeError("Smoke output schema differs from PyArrow")
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
                    "Smoke output row/value differs from PyArrow; first "
                    f"mismatching column is {mismatch}"
                )
        externalized = int(
            value(results[-1]["gpu_stats"], "externalized_bytes", default=0)
        )
        if externalized <= 0:
            raise RuntimeError(
                "The constrained GPU smoke did not externalize a run; lower --smoke-budget-mib"
            )
        write_json(self.artifacts / "smoke.json", {"valid": True, "trials": results})

    def trend(self) -> None:
        for cell in self.cells.values():
            self.trial(
                kind="trend",
                backend="pyarrow",
                cell=cell.name,
                repetition=1,
                name=f"{cell.name}-pyarrow-r1",
                reuse=True,
            )
            for repetition in (1, 2):
                self.trial(
                    kind="trend",
                    backend="gpu",
                    cell=cell.name,
                    repetition=repetition,
                    name=f"{cell.name}-gpu-r{repetition}",
                    reuse=True,
                )

    def gpu_trends(self) -> None:
        """Collect the six-cell GPU matrix without repeating CPU baselines."""

        for cell in self.cells.values():
            for repetition in (1, 2):
                self.trial(
                    kind="trend",
                    backend="gpu",
                    cell=cell.name,
                    repetition=repetition,
                    name=f"{cell.name}-gpu-r{repetition}",
                    reuse=True,
                )

    def baseline(self) -> None:
        self.trial(
            kind="trend",
            backend="gpu",
            cell="full",
            repetition=1,
            name="full-gpu-r1",
            reuse=True,
        )

    def _full_gpu(self) -> list[dict[str, Any]]:
        results = []
        for repetition in (1, 2):
            path = self.path("trend", f"full-gpu-r{repetition}")
            if path.is_file() and read_json(path).get("valid"):
                results.append(read_json(path))
            else:
                results.append(
                    self.trial(
                        kind="trend",
                        backend="gpu",
                        cell="full",
                        repetition=repetition,
                        name=f"full-gpu-r{repetition}",
                    )
                )
        return results

    def spill(self) -> None:
        resident = self._full_gpu()
        peak = max(int(item["gpu_peak_device_bytes"]) for item in resident)
        if peak <= 0:
            raise RuntimeError(
                "Resident runs did not report a usable peak_device_bytes B"
            )
        requested = (
            ("light", int(peak * 0.75)),
            ("medium", int(peak * 0.50)),
            ("heavy", max(4 * GIB, int(peak * 0.25))),
        )
        points: list[dict[str, Any]] = []
        prior_geometry = spill_geometry(resident[0]["gpu_stats"])
        replacement_used = False
        for label, initial_budget in requested:
            budget = initial_budget
            first = self.trial(
                kind="spill",
                backend="gpu",
                cell="full",
                repetition=1,
                name=f"{label}-r1",
                budget_bytes=budget,
            )
            geometry = spill_geometry(first["gpu_stats"])
            replaced: dict[str, Any] | None = None
            if not replacement_used and (
                geometry[0] <= 0 or geometry == prior_geometry
            ):
                replaced = first
                replacement_used = True
                budget = max(256 << 20, int(budget * 0.80))
                first = self.trial(
                    kind="spill",
                    backend="gpu",
                    cell="full",
                    repetition=1,
                    name=f"{label}-replacement-r1",
                    budget_bytes=budget,
                )
                geometry = spill_geometry(first["gpu_stats"])
            second = self.trial(
                kind="spill",
                backend="gpu",
                cell="full",
                repetition=2,
                name=f"{label}-r2",
                budget_bytes=budget,
            )
            points.append(
                {
                    "label": label,
                    "derived_budget_bytes": initial_budget,
                    "final_budget_bytes": budget,
                    "replacement_candidate": replaced,
                    "trials": [first, second],
                }
            )
            prior_geometry = geometry
        summary = {"resident": resident, "peak_B_bytes": peak, "points": points}
        write_json(self.artifacts / "spill-study.json", summary)

    def large(self) -> None:
        spill_path = self.artifacts / "spill-study.json"
        if not spill_path.is_file():
            raise RuntimeError("Run the spill study before the large proof")
        heavy = read_json(spill_path)["points"][-1]
        result = self.trial(
            kind="large",
            backend="gpu",
            cell="origin-string",
            repetition=1,
            name="origin-3x-heavy",
            budget_bytes=int(heavy["final_budget_bytes"]),
        )
        stats = result["gpu_stats"]
        if int(value(stats, "externalized_bytes", default=0)) <= 0:
            raise RuntimeError("Large proof did not externalize GPU runs")
        if int(value(stats, "merge_pass_count", default=0)) <= 0:
            raise RuntimeError("Large proof did not execute a GPU merge pass")

    def report(self) -> None:
        implementation = _implementation_identity(self.root)
        if implementation.get("error") or implementation.get("inspection_error"):
            raise RuntimeError(
                f"Could not identify the implementation: {implementation}"
            )
        if not implementation.get("allowlist_valid"):
            raise RuntimeError(
                "Changes outside the report allowlist: "
                f"{implementation.get('out_of_allowlist_paths')}"
            )

        smoke = _accepted(self.artifacts / "smoke.json", "exact smoke summary")
        smoke_trials = smoke.get("trials", [])
        if len(smoke_trials) != 3:
            raise RuntimeError(
                f"Exact smoke has {len(smoke_trials)} trials, expected 3"
            )
        for index, trial in enumerate(smoke_trials, start=1):
            _require_valid(trial, f"exact smoke trial {index}")

        spill_path = self.artifacts / "spill-study.json"
        if not spill_path.is_file():
            raise RuntimeError(f"Missing required spill study: {spill_path}")
        spill = read_json(spill_path)
        resident = spill.get("resident", [])
        points = spill.get("points", [])
        if len(resident) != 2 or len(points) != 3:
            raise RuntimeError(
                "Spill study must contain two resident trials and three points"
            )
        for index, trial in enumerate(resident, start=1):
            _require_valid(trial, f"resident spill baseline {index}")
        for point in points:
            trials = point.get("trials", [])
            if len(trials) != 2:
                raise RuntimeError(
                    f"Spill point {point.get('label')} does not have two trials"
                )
            for index, trial in enumerate(trials, start=1):
                _require_valid(trial, f"{point.get('label')} spill trial {index}")

        large = _accepted(
            self.path("large", "origin-3x-heavy"), "three-copy external/skew proof"
        )
        disk_trials = [trial for point in points for trial in point["trials"]] + [large]
        if any(not trial.get("ray_disk_spill_directory") for trial in disk_trials):
            raise RuntimeError(
                "A spilling trial did not record its filesystem directory"
            )
        spill_directories = sorted(
            {
                str(Path(trial["ray_disk_spill_directory"]).resolve())
                for trial in disk_trials
            }
        )
        if any(
            Path(directory).is_relative_to(Path("/dev/shm").resolve())
            or not Path(directory).is_relative_to(Path("/raid").resolve())
            for directory in spill_directories
        ):
            raise RuntimeError(
                f"A filesystem spill directory is not on RAID: {spill_directories}"
            )

        overlay_manifest = self.root / ".venv/gpu-sort-ray-data-overlay.json"
        environment_verification = {
            "verified_at_report_generation_unix_s": time.time(),
            "scope": (
                "Current report-generation environment only; historical trial provenance "
                "is retained unchanged."
            ),
            "ray_version": RAY_VERSION,
            "ray_commit": RAY_COMMIT,
            "wheel": self.wheel,
            "ray_data_overlay": _overlay_hashes(self.root),
            "overlay_manifest_sha256": hashlib.sha256(
                overlay_manifest.read_bytes()
            ).hexdigest(),
        }
        summary: dict[str, Any] = {
            "smoke": smoke,
            "trends": {},
            "spill": spill,
            "spill_directories": spill_directories,
            "large": large,
            "implementation": implementation,
            "current_environment_verification": environment_verification,
        }
        smoke_times = [float(item["cold_sort_s"]) for item in smoke_trials]
        lines = [
            "# DGX Unified Spillable GPU Sort",
            "",
            "Timed boundary: materialized Plasma input through sorted output sealed in Plasma. Reading and Ray startup are excluded.",
            "",
            "The report-generation environment was independently verified as Ray "
            f"{RAY_VERSION} at `{RAY_COMMIT}` using wheel `{self.wheel['sha256']}`. "
            "This is a current-environment check, not a retroactive rewrite of historical "
            "trial provenance.",
            "",
            "## Exact correctness smoke",
            "",
            f"PyArrow, resident GPU, and forced-external GPU completed in "
            f"{smoke_times[0]:.3f}/{smoke_times[1]:.3f}/{smoke_times[2]:.3f}s. "
            "The accepted smoke compared schema and every row/value exactly.",
            "",
            "## Reproduced BTS trends",
            "",
            "| Cell | GiB | Old CPU | New CPU | Old GPU | New GPU r1/r2 | New median | Speedup |",
            "|:--|--:|--:|--:|--:|:--|--:|--:|",
        ]
        health_rows = []
        for cell in self.cells.values():
            cpu_path = self.path("trend", f"{cell.name}-pyarrow-r1")
            gpu_paths = [
                self.path("trend", f"{cell.name}-gpu-r{rep}") for rep in (1, 2)
            ]
            cpu = _accepted(cpu_path, f"{cell.name} PyArrow trend")
            gpu = [
                _accepted(path, f"{cell.name} GPU trend r{repetition}")
                for repetition, path in enumerate(gpu_paths, start=1)
            ]
            if cpu.get("cell") != cell.to_dict() or any(
                item.get("cell") != cell.to_dict() for item in gpu
            ):
                raise RuntimeError(
                    f"Trend artifact identity differs from cell {cell.name}"
                )
            gpu_times = [float(item["cold_sort_s"]) for item in gpu]
            gpu_median = median(gpu_times)
            speedup = float(cpu["cold_sort_s"]) / gpu_median
            old = OLD_DGX[cell.name]
            old_median = median(old["gpu_s"])
            actual_gib = float(gpu[0]["input"]["decoded_bytes"]) / GIB
            device_peak, nvml_peak, headroom = _memory_summary(gpu)
            low, high, balance = _output_balance(gpu)
            externalized = max(
                int(value(item["gpu_stats"], "externalized_bytes", default=0))
                for item in gpu
            )
            runs = max(
                int(value(item["gpu_stats"], "initial_run_count", default=0))
                for item in gpu
            )
            pyarrow_ray_write, pyarrow_ray_restore = _ray_io(cpu)
            health = {
                "pyarrow_gib_s": float(cpu["throughput_gib_s"]),
                "gpu_median_gib_s": median(
                    float(item["throughput_gib_s"]) for item in gpu
                ),
                "gpu_median_rows_s": median(
                    float(item["throughput_rows_s"]) for item in gpu
                ),
                "input_materialize_pyarrow_s": float(
                    cpu["input"]["read_materialize_s"]
                ),
                "input_materialize_gpu_median_s": median(
                    float(item["input"]["read_materialize_s"]) for item in gpu
                ),
                "output_min_bytes": low,
                "output_max_bytes": high,
                "output_balance": balance,
                "peak_device_bytes": device_peak,
                "peak_nvml_bytes": nvml_peak,
                "minimum_nvml_headroom_bytes": headroom,
                "externalized_bytes": externalized,
                "initial_runs": runs,
                "fallback_count": _fallback_count(gpu),
                "pyarrow_ray_write_bytes": pyarrow_ray_write,
                "pyarrow_ray_restore_bytes": pyarrow_ray_restore,
            }
            row = {
                "cell": cell.to_dict(),
                "pyarrow": cpu,
                "gpu": gpu,
                "gpu_median_s": gpu_median,
                "speedup": speedup,
                "old": old,
                "health": health,
            }
            summary["trends"][cell.name] = row
            lines.append(
                f"| {cell.label} | {actual_gib:.3f} | {old['pyarrow_s']:.3f}s | "
                f"{cpu['cold_sort_s']:.3f}s | {old_median:.3f}s | "
                f"{gpu_times[0]:.3f}/{gpu_times[1]:.3f}s | {gpu_median:.3f}s | {speedup:.2f}x |"
            )
            health_rows.append(
                f"| {cell.label} | {health['pyarrow_gib_s']:.3f}/"
                f"{health['gpu_median_gib_s']:.3f} | "
                f"{health['gpu_median_rows_s'] / 1_000_000:.3f} | "
                f"{health['input_materialize_pyarrow_s']:.1f}/"
                f"{health['input_materialize_gpu_median_s']:.1f} | "
                f"{pyarrow_ray_write / GIB:.2f}/{pyarrow_ray_restore / GIB:.2f} | "
                f"{low / GIB:.2f}-{high / GIB:.2f} ({balance:.2f}x) | "
                f"{device_peak / GIB:.2f}/{nvml_peak / GIB:.2f}/{headroom / GIB:.2f} | "
                f"{externalized / GIB:.2f}/{runs}/{health['fallback_count']} |"
            )
        lines.extend(
            [
                "",
                "Every trend cell uses 80,738,761 rows and 627 input blocks. Read/materialize "
                "times below remain outside the cold-sort timer. CPU Ray write/restore is "
                "measured inside the default-Ray cold-sort window and is therefore included "
                "in the CPU time.",
                "",
                "### Trend health",
                "",
                "| Cell | CPU/GPU GiB/s | GPU Mrows/s | Read CPU/GPU s | CPU Ray write/restore GiB | Output GiB/rank (balance) | Device/NVML/headroom GiB | External GiB/runs/fallbacks |",
                "|:--|--:|--:|--:|--:|:--|:--|:--|",
                *health_rows,
            ]
        )

        origin_balance = summary["trends"]["origin-string"]["health"]["output_balance"]
        full_balance = summary["trends"]["full"]["health"]["output_balance"]
        _, _, large_natural_balance = _output_balance([large])
        skew_limitation = (
            "These measurements cover natural BTS skew, not an adversarial unseen-skew "
            "safety guarantee. Wave admission is sender-bounded rather than controlled by "
            "receiver-issued credits, so an unseen all-to-one range can aggregate traffic "
            "from multiple senders before the destination externalizes it. A hard bound for "
            "that case requires receiver-side credit/backpressure."
        )
        summary["skew_scope"] = {
            "natural_bts_output_balance": {
                "origin_string_64gib": origin_balance,
                "four_key_full_64gib": full_balance,
                "origin_string_3x_large": large_natural_balance,
            },
            "adversarial_unseen_skew_limitation": skew_limitation,
        }
        lines.extend(
            [
                "",
                f"Measured natural BTS range balance was {origin_balance:.2f}x for the "
                f"64-GiB single-Origin cell, {full_balance:.2f}x for the four-key full cell, "
                f"and {large_natural_balance:.2f}x for the three-copy Origin proof. "
                + skew_limitation,
            ]
        )

        resident_times = [float(item["cold_sort_s"]) for item in resident]
        resident_median = median(resident_times)
        resident_amplification = median(
            _plasma_amplification(item) for item in resident
        )
        lines.extend(
            [
                "",
                "## GPU spill cost",
                "",
                "Ray filesystem spill was directed to per-trial directories under the project "
                "runtime root on `/raid`, not RAM-backed `/dev/shm`. Absolute directories are "
                "retained in `study.json`.",
                "",
                "| Pressure | Budget/GPU | GPU r1/r2 | Median | Slowdown | Externalized | Runs | Merge passes | Plasma amplification |",
                "|:--|--:|:--|--:|--:|--:|--:|--:|--:|",
                f"| Resident | default | {resident_times[0]:.3f}/{resident_times[1]:.3f}s | "
                f"{resident_median:.3f}s | 1.00x | 0 GiB | 0 | 0 | "
                f"{resident_amplification:.2f}x |",
            ]
        )
        for point in points:
            trials = point["trials"]
            times = [float(item["cold_sort_s"]) for item in trials]
            med = median(times)
            stats = trials[0]["gpu_stats"]
            externalized = int(value(stats, "externalized_bytes", default=0))
            runs = int(value(stats, "initial_run_count", default=0))
            passes = int(value(stats, "merge_pass_count", default=0))
            amplification = median(_plasma_amplification(item) for item in trials)
            lines.append(
                f"| {point['label'].title()} | {point['final_budget_bytes'] / GIB:.2f} GiB | "
                f"{times[0]:.3f}/{times[1]:.3f}s | {med:.3f}s | "
                f"{med / resident_median:.2f}x | {externalized / GIB:.2f} GiB | "
                f"{runs} | {passes} | {amplification:.2f}x |"
            )
        observations = [
            ("resident", index + 1, trial) for index, trial in enumerate(resident)
        ] + [
            (point["label"], index + 1, trial)
            for point in points
            for index, trial in enumerate(point["trials"])
        ]
        lines.extend(
            [
                "",
                "### Per-observation spill telemetry",
                "",
                "| Point | Rep | Budget GiB | Ext GiB/rows | First s/wave | Runs/passes/replacements | H2D/D2H GiB | Plasma read/write GiB | MPF/Ray write/restore GiB | Device/NVML/headroom GiB |",
                "|:--|--:|--:|:--|:--|:--|:--|:--|:--|:--|",
            ]
        )
        for label, repetition, trial in observations:
            stats = trial["gpu_stats"]
            first_s = value(stats, "first_externalize_s", default=None)
            first_wave = value(stats, "first_externalize_wave", default=None)
            first = "-/-" if first_s is None else f"{float(first_s):.3f}/{first_wave}"
            ray_write, ray_restore = _ray_io(trial)
            device_peak, nvml_peak, headroom = _memory_summary([trial])
            budget = (
                "default"
                if trial.get("budget_bytes") is None
                else f"{int(trial['budget_bytes']) / GIB:.2f}"
            )
            lines.append(
                f"| {label} | {repetition} | {budget} | "
                f"{int(value(stats, 'externalized_bytes', default=0)) / GIB:.2f}/"
                f"{int(value(stats, 'externalized_rows', default=0)):,} | {first} | "
                f"{int(value(stats, 'initial_run_count', default=0))}/"
                f"{int(value(stats, 'merge_pass_count', default=0))}/"
                f"{int(value(stats, 'replacement_run_count', default=0))} | "
                f"{int(value(stats, 'h2d_bytes', default=0)) / GIB:.2f}/"
                f"{int(value(stats, 'd2h_bytes', default=0)) / GIB:.2f} | "
                f"{int(value(stats, 'plasma_read_bytes', default=0)) / GIB:.2f}/"
                f"{int(value(stats, 'plasma_write_bytes', default=0)) / GIB:.2f} | "
                f"{int(value(stats, 'mpf_host_spill_bytes', default=0)) / GIB:.2f}/"
                f"{ray_write / GIB:.2f}/{ray_restore / GIB:.2f} | "
                f"{device_peak / GIB:.2f}/{nvml_peak / GIB:.2f}/{headroom / GIB:.2f} |"
            )

        phase_names = (
            "sampling",
            "partition",
            "mpf_shuffle",
            "run_sort",
            "gpu_merge",
            "arrow_conversion",
            "plasma_seal",
            "orchestration",
        )
        lines.extend(
            [
                "",
                "### Per-observation phase seconds",
                "",
                "| Point/rep | " + " | ".join(phase_names) + " |",
                "|:--|" + "--:|" * len(phase_names),
            ]
        )
        for label, repetition, trial in observations:
            phases = value(trial["gpu_stats"], "phases_s", default={})
            lines.append(
                f"| {label}/r{repetition} | "
                + " | ".join(
                    f"{float(phases.get(name, 0)):.3f}" for name in phase_names
                )
                + " |"
            )

        stats = large["gpu_stats"]
        large_write, large_restore = _ray_io(large)
        large_device, large_nvml, large_headroom = _memory_summary([large])
        large_low, large_high, large_balance = _output_balance([large])
        large_phases = value(stats, "phases_s", default={})
        large_validation = large.get("validation", {})
        lines.extend(
            [
                "",
                "## 3x large/skew proof",
                "",
                "| Metric | Result |",
                "|:--|:--|",
                f"| Input | {large['input']['decoded_bytes'] / GIB:.3f} GiB; "
                f"{large['input']['rows']:,} rows; {large['input']['blocks']:,} blocks |",
                f"| Cold sort / throughput | {large['cold_sort_s']:.3f}s; "
                f"{large['throughput_gib_s']:.3f} GiB/s; "
                f"{large['throughput_rows_s']:,.0f} rows/s |",
                f"| Input materialize (excluded) | "
                f"{large['input']['read_materialize_s']:.3f}s |",
                f"| External runs | "
                f"{int(value(stats, 'externalized_bytes', default=0)) / GIB:.3f} GiB; "
                f"{int(value(stats, 'initial_run_count', default=0))} initial; "
                f"{int(value(stats, 'replacement_run_count', default=0))} replacements; "
                f"{int(value(stats, 'merge_pass_count', default=0))} passes |",
                f"| Ray RAID write / restore | {large_write / GIB:.3f} / "
                f"{large_restore / GIB:.3f} GiB |",
                f"| Plasma read / write | "
                f"{int(value(stats, 'plasma_read_bytes', default=0)) / GIB:.3f} / "
                f"{int(value(stats, 'plasma_write_bytes', default=0)) / GIB:.3f} GiB |",
                f"| H2D / D2H | {int(value(stats, 'h2d_bytes', default=0)) / GIB:.3f} / "
                f"{int(value(stats, 'd2h_bytes', default=0)) / GIB:.3f} GiB |",
                f"| Device / NVML / headroom | {large_device / GIB:.3f} / "
                f"{large_nvml / GIB:.3f} / {large_headroom / GIB:.3f} GiB |",
                f"| Output range balance | {large_low / GIB:.3f}-{large_high / GIB:.3f} "
                f"GiB/rank; {large_balance:.2f}x |",
                f"| CPU sort / merge / fallback | "
                f"{int(value(stats, 'cpu_sort_rows', default=0))} / "
                f"{int(value(stats, 'cpu_merge_rows', default=0))} / "
                f"{_fallback_count([large])} |",
                f"| Validation | ordered={large_validation.get('ordered')}; "
                f"rows={large_validation.get('rows'):,}; exact row_id checksum="
                f"{large_validation.get('row_id_sum') == large_validation.get('expected_row_id_sum')} |",
                "| Phase seconds | "
                + "; ".join(
                    f"{name}={float(large_phases.get(name, 0)):.3f}"
                    for name in phase_names
                )
                + " |",
            ]
        )

        narrow = summary["trends"]["narrow"]
        core = summary["trends"]["core"]
        full = summary["trends"]["full"]
        origin_string = summary["trends"]["origin-string"]
        origin_integer = summary["trends"]["origin-integer"]
        route = summary["trends"]["route"]
        production = implementation["counts"]["production"]
        tests = implementation["counts"]["tests"]
        benchmarks = implementation["counts"]["benchmarks"]
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                f"Payload width moves the GPU median from {narrow['gpu_median_s']:.3f}s to "
                f"{core['gpu_median_s']:.3f}s to {full['gpu_median_s']:.3f}s, while the "
                f"corresponding default-Ray CPU observations are "
                f"{narrow['pyarrow']['cold_sort_s']:.3f}s, "
                f"{core['pyarrow']['cold_sort_s']:.3f}s, and "
                f"{full['pyarrow']['cold_sort_s']:.3f}s.",
                f"For the full payload, string and integer single keys take "
                f"{origin_string['pyarrow']['cold_sort_s']:.3f}s and "
                f"{origin_integer['pyarrow']['cold_sort_s']:.3f}s on CPU versus GPU medians "
                f"of {origin_string['gpu_median_s']:.3f}s and "
                f"{origin_integer['gpu_median_s']:.3f}s. One/two/four natural keys have GPU "
                f"medians of {origin_string['gpu_median_s']:.3f}/"
                f"{route['gpu_median_s']:.3f}/{full['gpu_median_s']:.3f}s.",
                "The four pressure points show directional externalization and RAID-I/O cost; "
                "two GPU observations are not a statistical study. No cloud result is included.",
                "",
                "## Implementation identity",
                "",
                f"Branch: `{implementation['branch']}`; explicit base: "
                f"`{implementation['base']}`; HEAD: `{implementation['head']}`; current "
                f"allowlisted-diff digest: `{implementation['current_diff_sha256']}`.",
                f"Production: {_file_count(production['files'])}, "
                f"+{production['added_lines']}/-{production['deleted_lines']} lines. "
                f"Focused tests: {_file_count(tests['files'])}, "
                f"+{tests['added_lines']}/-{tests['deleted_lines']} lines. Benchmark/report: "
                f"{_file_count(benchmarks['files'])}, +{benchmarks['added_lines']}/"
                f"-{benchmarks['deleted_lines']} lines.",
                f"All {len(implementation['changed_paths'])} changed paths are within "
                "`python/ray/data/**` or `release/benchmarks/ray_data_gpu_sort/**`.",
                "",
                "`study.json` records this current identity separately from the unchanged raw "
                "trial provenance, including historical harness and overlay hashes.",
                "",
            ]
        )
        write_json(self.artifacts / "study.json", summary)
        (self.artifacts / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
        print(self.artifacts / "REPORT.md")


def parser() -> argparse.ArgumentParser:
    root = _repo_root()
    result = argparse.ArgumentParser()
    result.add_argument(
        "command",
        choices=(
            "smoke",
            "baseline",
            "trends",
            "gpu-trends",
            "spill",
            "large",
            "report",
            "all",
        ),
    )
    result.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    result.add_argument(
        "--artifact-root", type=Path, default=root / ".venv/gpu-sort-external-artifacts"
    )
    result.add_argument(
        "--runtime-root", type=Path, default=root / ".venv/gpu-sort-external-runtime"
    )
    result.add_argument("--smoke-budget-mib", type=int, default=16)
    result.add_argument("--timeout-s", type=int, default=14_400)
    return result


def main() -> int:
    args = parser().parse_args()
    study = Study(args)
    commands = (
        ("smoke", "trends", "spill", "large")
        if args.command == "all"
        else (args.command,)
    )
    for command in commands:
        method = {"trends": "trend", "gpu-trends": "gpu_trends"}.get(command, command)
        getattr(study, method)()
    if args.command == "all":
        study.report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
