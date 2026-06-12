"""Compare a GPU fused run against the CPU baseline from their manifests.

Reads the ``manifest.json`` written by ``gpu_pipeline.py`` and ``cpu_pipeline.py``
(local path or ``s3://`` URI) and reports the only apples-to-apples number: the
GPU **fused** stage wall vs the CPU **impute + encode + scale** subtotal (sort is
CPU in both pipelines, so it cancels and is intentionally excluded -- do NOT use
the CPU manifest's ``gpu_target_subtotal_s``, which is sort+encode+scale). Also
checks fitted-stat parity (scaler mean/std, encoder vocab sizes, imputer fills)
so the speedup is only trusted when the two pipelines produced the same model
inputs.

Usage:
    python compare_manifests.py \
        --gpu s3://.../outputs/gpu_days1_30_8node/manifest.json \
        --cpu s3://.../outputs/cpu_days1_30_8node/manifest.json

    # local dirs work too (pass the dir or the manifest.json):
    python compare_manifests.py --gpu data/criteo_days1_30_gpu --cpu data/criteo_days1_30_cpu
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, Tuple

_SUBTOTAL_STAGES = ("impute", "encode", "scale")


def _read_bytes(path: str) -> bytes:
    """Read a file's bytes from a local path or an ``s3://`` URI (pyarrow.fs)."""
    if path.startswith("s3://"):
        import pyarrow.fs as pafs

        fs, p = pafs.FileSystem.from_uri(path)
        with fs.open_input_stream(p) as f:
            return f.read()
    with open(path, "rb") as f:
        return f.read()


def load_manifest(path: str) -> Dict[str, Any]:
    """Load a manifest from a manifest.json file, or a dir containing one."""
    if not path.endswith(".json"):
        path = path.rstrip("/") + "/manifest.json"
    return json.loads(_read_bytes(path).decode("utf-8"))


def cpu_subtotal_s(cpu: Dict[str, Any]) -> float:
    st = cpu.get("stage_timings_s", {})
    return float(sum(st.get(k, 0.0) for k in _SUBTOTAL_STAGES))


def gpu_fused_s(gpu: Dict[str, Any]) -> float:
    if gpu.get("fused_stage_s") is not None:
        return float(gpu["fused_stage_s"])
    return float(gpu.get("stage_timings_s", {}).get("fused", 0.0))


def _rel_err(a: float, b: float) -> float:
    if a is None or b is None:
        return float("inf")
    if isinstance(a, float) and (math.isnan(a) or math.isnan(b)):
        return 0.0 if (math.isnan(a) and math.isnan(b)) else float("inf")
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom


def _check_scaler(gpu_f: Dict, cpu_f: Dict, rtol: float) -> Tuple[bool, str]:
    g, c = gpu_f.get("scaler", {}), cpu_f.get("scaler", {})
    cols = sorted(set(g) & set(c))
    worst, worst_col = 0.0, None
    for col in cols:
        for stat in ("mean", "std"):
            e = _rel_err(g[col].get(stat), c[col].get(stat))
            if e > worst:
                worst, worst_col = e, f"{col}.{stat}"
    ok = worst <= rtol
    return ok, f"scaler: {len(cols)} cols, worst rel-err {worst:.2e} ({worst_col})"


def _check_vocab(gpu_f: Dict, cpu_f: Dict) -> Tuple[bool, str]:
    g, c = gpu_f.get("encoder_vocab_size", {}), cpu_f.get("encoder_vocab_size", {})
    cols = sorted(set(g) & set(c))
    mism = [(col, g[col], c[col]) for col in cols if g[col] != c[col]]
    ok = not mism
    detail = f"encoder vocab: {len(cols)} cols, {len(mism)} mismatch"
    if mism:
        detail += " e.g. " + ", ".join(f"{col}(gpu={a},cpu={b})" for col, a, b in mism[:3])
    return ok, detail


def _check_imputer(gpu_f: Dict, cpu_f: Dict, rtol: float) -> Tuple[bool, str]:
    g, c = gpu_f.get("imputer_mean", {}), cpu_f.get("imputer_mean", {})
    cols = sorted(set(g) & set(c))
    worst, worst_col = 0.0, None
    for col in cols:
        e = _rel_err(g[col], c[col])
        if e > worst:
            worst, worst_col = e, col
    ok = worst <= rtol
    return ok, f"imputer mean: {len(cols)} cols, worst rel-err {worst:.2e} ({worst_col})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", required=True, help="GPU manifest.json (or its dir), local or s3://")
    ap.add_argument("--cpu", required=True, help="CPU manifest.json (or its dir), local or s3://")
    ap.add_argument("--rtol", type=float, default=1e-6, help="relative tolerance for fitted floats")
    args = ap.parse_args()

    gpu = load_manifest(args.gpu)
    cpu = load_manifest(args.cpu)

    g_rows, c_rows = gpu.get("rows"), cpu.get("rows")
    g_fs = gpu.get("feature_set")
    c_fs = cpu.get("feature_set")

    fused = gpu_fused_s(gpu)
    subtotal = cpu_subtotal_s(cpu)
    speedup = (subtotal / fused) if fused else float("nan")

    P = lambda *a: print(*a, flush=True)  # noqa: E731
    P("=" * 70)
    P("CriteoPrivateAd fused-GPU vs CPU baseline")
    P("=" * 70)
    P(f"rows           : gpu={g_rows:,}  cpu={c_rows:,}" if g_rows and c_rows else
      f"rows           : gpu={g_rows}  cpu={c_rows}")
    P(f"feature_set    : gpu={g_fs}  cpu={c_fs}")
    P(f"gpu n_gpus     : {gpu.get('n_gpus')}  batch={gpu.get('preproc_batch_size')}  "
      f"num_gpus={gpu.get('preproc_num_gpus')}")
    if g_rows != c_rows:
        P("WARNING: row counts differ -- not an apples-to-apples comparison.")
    if g_fs != c_fs:
        P("WARNING: feature_set differs -- rerun both with the same --feature-set.")

    P("-" * 70)
    P("headline (the fair comparison; sort is CPU in both and excluded):")
    P(f"  CPU impute+encode+scale : {subtotal:8.2f} s")
    P(f"  GPU fused stage         : {fused:8.2f} s")
    P(f"  speedup                 : {speedup:8.2f}x")

    g_total = sum(gpu.get("stage_timings_s", {}).values())
    c_total = sum(cpu.get("stage_timings_s", {}).values())
    P("-" * 70)
    P("end-to-end wall (read+prep+sort dominate at 30 days; same work both sides):")
    P(f"  GPU total : {g_total:8.2f} s   stages={gpu.get('stage_timings_s')}")
    P(f"  CPU total : {c_total:8.2f} s   stages={cpu.get('stage_timings_s')}")

    P("-" * 70)
    P("saved-output global sortedness:")
    gv = gpu.get("verification", {}) or {}
    cv = cpu.get("verification", {}) or {}
    P(f"  GPU globally_sorted = {gv.get('globally_sorted')}   "
      f"CPU globally_sorted = {cv.get('globally_sorted')}")

    P("-" * 70)
    P("fitted-stat parity (GPU vs CPU):")
    gpu_f, cpu_f = gpu.get("fitted", {}), cpu.get("fitted", {})
    results = [
        _check_scaler(gpu_f, cpu_f, args.rtol),
        _check_vocab(gpu_f, cpu_f),
        _check_imputer(gpu_f, cpu_f, args.rtol),
    ]
    all_ok = True
    for ok, detail in results:
        all_ok = all_ok and ok
        P(f"  {'MATCH' if ok else 'MISMATCH'}  {detail}")
    P("-" * 70)
    P(f"PARITY: {'MATCH' if all_ok else 'MISMATCH'}   "
      f"SORTED: gpu={gv.get('globally_sorted')}   "
      f"SPEEDUP: {speedup:.2f}x")
    P("=" * 70)


if __name__ == "__main__":
    main()
