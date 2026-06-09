"""GPU sort + GPU OrdinalEncoder pipeline for CriteoPrivateAd (days 1..30).

This swaps EXACTLY TWO operators of the CPU baseline
(``bench_criteo_cpu_baseline.py``) to GPU and leaves everything else identical:

    stage    CPU baseline              this run
    -----    ------------              --------
    read     read_parquet              (same, CPU)
    prep     prep_batch                (same, CPU -- reused verbatim)
    sort     ds.sort(backend="cpu")    ds.sort(backend="gpu")     <-- GPU
    impute   SimpleImputer (CPU)       (same, CPU)
    encode   OrdinalEncoder (CPU)      GpuOrdinalEncoder          <-- GPU
    scale    StandardScaler (CPU)      (same, CPU -- NOT gpu scale)
    write    write_parquet             (same, CPU)

Scale stays on CPU: this run is NOT "GPU scale". A GPU scaler would be a
separate future GpuStandardScaler run.

Sort and encode are sized INDEPENDENTLY (they have different optimal
block/partition sizing):

  * GPU sort sizing  : RAY_DATA_GPU_SORT_NUM_GPUS (default 16),
                       RAY_DATA_GPU_SORT_POOL_FRAC, RAY_DATA_GPU_SORT_SPILL_FRAC,
                       + the input block count from read/prep. Per-rank partition
                       = pruned dataset / num_gpus. We report rows/bytes, block
                       count, per-rank estimated vs measured size, RMM pool cap,
                       peak VRAM/rank, headroom, and RMM/Ray spill status.
  * GPU encoder sizing: RAY_DATA_GPU_PREPROC_BATCH_SIZE / NUM_GPUS, tuned by
                       scan_encoder_blocksize.py (independent of the sort).

Two repartitions are timed and reported SEPARATELY (never folded into the
sort/encode numbers):
  * post_sort_repartition  : GPU sort emits num_gpus blocks -> back to the CPU
                             baseline block count before CPU impute.
  * post_encode_repartition: GpuOrdinalEncoder's map_batches may emit far fewer
                             blocks (driven by the batch size); if so, repartition
                             back before CPU scale + write.
preserve_order=True throughout; the saved output is verified globally sorted by
[user_id, day_int, display_order] exactly like the CPU baseline.

Run:
    RAY_DATA_GPU_SORT_NUM_GPUS=16 RAY_DATA_GPU_PREPROC_NUM_GPUS=16 \
      RAY_DATA_GPU_PREPROC_BATCH_SIZE=<winner-from-scan> \
      .venv/bin/python gpu_preproc_bench/bench_criteo_gpu_sort_encode.py --days 1-30 \
      --out gpu_preproc_bench/data/criteo_days1_30_gpu_sort_encode
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------------------------------- #
# Parse args + set GPU env BEFORE importing ray (the engines read env at import
# / actor-construction time).
# --------------------------------------------------------------------------- #
_ap = argparse.ArgumentParser()
_ap.add_argument("--days", default="1-30", help="day(s): '1' | '1-5' | '1-30' | 'all'")
_ap.add_argument("--gpus", type=int, default=16, help="GPUs for the GPU sort + encoder")
_ap.add_argument("--batch-size", type=int, default=2_000_000,
                 help="GpuOrdinalEncoder per-worker batch size (RAY_DATA_GPU_PREPROC_BATCH_SIZE)")
_ap.add_argument("--pool-frac", type=float, default=None,
                 help="RAY_DATA_GPU_SORT_POOL_FRAC (RMM pool cap fraction of VRAM); default 0.80")
_ap.add_argument("--spill-frac", default=None,
                 help="RAY_DATA_GPU_SORT_SPILL_FRAC (device->host spill); default off (single node)")
_ap.add_argument("--blocks", type=int, default=None,
                 help="post-sort repartition target block count; default = read block count")
_ap.add_argument("--rows", type=int, default=0, help="row cap for a quick smoke run (0 = full)")
_ap.add_argument("--null-indicator-threshold", type=float, default=0.01)
_ap.add_argument("--out", default=None, help="output dir (default: data/criteo_days<lo>_<hi>_gpu_sort_encode)")
_ap.add_argument("--no-write", action="store_true", help="skip writing parquet (RAM->RAM only)")
_ap.add_argument("--object-store-gb", type=float, default=None,
                 help="Ray object-store size GiB; default ~85%% of /dev/shm (CPU-baseline policy)")
_ap.add_argument("--cpu-baseline", default=None,
                 help="path to the CPU baseline manifest.json for the side-by-side table")
_ap.add_argument("--demo-oom", action="store_true",
                 help="stress mode: force a per-rank partition OOM then show spill mitigation")
ARGS = _ap.parse_args()


def _set_gpu_env() -> None:
    os.environ["RAY_DATA_GPU_SORT_NUM_GPUS"] = str(ARGS.gpus)
    os.environ["RAY_DATA_GPU_PREPROC_NUM_GPUS"] = str(ARGS.gpus)
    os.environ["RAY_DATA_GPU_SORT_RELEASE"] = "1"  # free sort pool for the encoder
    os.environ.setdefault("RAY_DATA_GPU_PREPROC_BATCH_SIZE", str(ARGS.batch_size))
    # Profile the GPU encode transform (H2D/compute/D2H) in-line; the per-batch
    # CUDA sync overhead is negligible vs the encode work and lets us report the
    # transfer-vs-compute split for the device-resident projection.
    os.environ.setdefault("RAY_DATA_GPU_PREPROC_PROFILE", "1")
    os.environ.setdefault("RAY_enable_open_telemetry", "0")
    if ARGS.pool_frac is not None:
        os.environ["RAY_DATA_GPU_SORT_POOL_FRAC"] = str(ARGS.pool_frac)
    if ARGS.spill_frac is not None:
        os.environ["RAY_DATA_GPU_SORT_SPILL_FRAC"] = str(ARGS.spill_frac)


_set_gpu_env()

import pyarrow as pa  # noqa: E402

import criteo  # noqa: E402
import ray  # noqa: E402

# Reuse the CPU baseline's stage code verbatim so prep / impute / scale / verify
# are byte-for-byte identical (the comparability contract).
import bench_criteo_cpu_baseline as cpu_base  # noqa: E402
from bench_criteo_cpu_baseline import (  # noqa: E402
    Metrics,
    add_indicators,
    ds_stats,
    prep_batch,
    _jsonable,
    _resolve_object_store_bytes,
    _verify_dataset_sorted,
    _verify_saved_sorted,
    _vocab_size,
)

from ray.data.preprocessors import (  # noqa: E402
    GpuOrdinalEncoder,
    SimpleImputer,
    StandardScaler,
    _gpu,
)
from ray.data._internal.planner import gpu_sort_general as G  # noqa: E402

P = lambda *a: print(*a, flush=True)  # noqa: E731
GIB = 1024 ** 3
# This run's GPU-accelerated stages are ONLY sort + encode (scale stays CPU).
GPU_STAGES = {"sort", "encode"}


# --------------------------------------------------------------------------- #
# Memory / spill probes
# --------------------------------------------------------------------------- #
class GpuMemSampler:
    """Background nvidia-smi poller: records the peak single-GPU used MiB over a
    window. Independent cross-check of the RMM per-rank peak."""

    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.peak_mib = 0.0

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    text=True, stderr=subprocess.DEVNULL,
                )
                vals = [float(x) for x in out.split() if x.strip()]
                if vals:
                    self.peak_mib = max(self.peak_mib, max(vals))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self) -> "GpuMemSampler":
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> float:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        return self.peak_mib


def ray_spill_summary() -> Dict[str, Any]:
    """Ray object-store spill: spilled/restored MiB parsed from the cluster
    memory summary, plus whether the on-disk spill dir holds any objects."""
    out: Dict[str, Any] = {"spilled_mib": 0, "restored_mib": 0, "spill_dir": None,
                           "spill_dir_bytes": 0, "ok": False}
    try:
        from ray._private.internal_api import memory_summary

        s = memory_summary(stats_only=True)
        m = re.search(r"Spilled (\d+) MiB", s)
        out["spilled_mib"] = int(m.group(1)) if m else 0
        m = re.search(r"Restored (\d+) MiB", s)
        out["restored_mib"] = int(m.group(1)) if m else 0
        out["ok"] = True
    except Exception as e:
        out["error"] = str(e)
    # Inspect the actual spill directory (default under the session dir, i.e.
    # /tmp/ray/session_*/...). Report bytes so "did /tmp get touched?" is measured.
    try:
        sess = ray._private.worker._global_node.get_session_dir_path()
        spill_dir = os.path.join(sess, "ray_spilled_objects")
        out["spill_dir"] = spill_dir
        total = 0
        if os.path.isdir(spill_dir):
            for root, _dirs, files in os.walk(spill_dir):
                for f in files:
                    try:
                        total += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
        out["spill_dir_bytes"] = total
    except Exception:
        pass
    return out


def _wait_for_gpus(gpus: int, timeout_s: float = 90.0) -> None:
    """Block (best-effort) until >= ``gpus`` GPUs are free, so the next GPU
    stage starts with all GPUs (the sort releases its pool when RELEASE=1)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if ray.available_resources().get("GPU", 0.0) >= gpus - 0.5:
            return
        time.sleep(0.25)


def fmt_gib(nbytes: Optional[float]) -> str:
    if not nbytes:
        return "   n/a"
    return f"{nbytes / GIB:6.2f}"


# --------------------------------------------------------------------------- #
# The GPU pipeline (only sort + encode are GPU; scale stays CPU)
# --------------------------------------------------------------------------- #
def run_pipeline(ds, roles, m: Metrics, target_blocks: int, gpus: int,
                 extra: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    numeric_raw = tuple(roles.numeric_raw)
    categorical = tuple(roles.categorical)
    list_features = tuple(roles.list_features)
    indicator_cols = tuple(roles.indicator_cols)
    fitted: Dict[str, Any] = {}

    # ---- prep (CPU, reused verbatim): label engineering + prune ----------- #
    with m.stage("prep", extra["read_cols"]) as rec:
        ds = ds.map_batches(
            lambda t: prep_batch(t, numeric_raw, categorical, list_features),
            batch_format="pyarrow",
            batch_size=None,
        ).materialize()
        prune_rows, ncols, gib = ds_stats(ds)
        rec.update(rows=prune_rows, out_cols=ncols, gib=gib)
    # The pruned dataset IS the GPU sort input -> record its size for sizing.
    extra["sort_input_rows"] = prune_rows
    extra["sort_input_bytes"] = int(ds.size_bytes() or 0)
    extra["sort_input_blocks"] = ds.num_blocks()

    # ---- sort (GPU): ds.sort(backend="gpu") ------------------------------- #
    _wait_for_gpus(gpus)
    sampler = GpuMemSampler().start()
    with m.stage("sort", ncols) as rec:
        ds = ds.sort(roles.sort_key, backend="gpu").materialize()
        _, ncols, gib = ds_stats(ds)
        rec.update(rows=ds.count(), out_cols=ncols, gib=gib)
    rec["gpu"] = dict(G.LAST_RUN_STATS)
    rec["nvsmi_peak_mib"] = sampler.stop()
    extra["sort_gpu_stats"] = rec["gpu"]
    extra["sort_nvsmi_peak_mib"] = rec["nvsmi_peak_mib"]
    extra["sort_blocks_out"] = ds.num_blocks()

    # ---- post-sort repartition (TIMED SEPARATELY) ------------------------- #
    # GPU sort emits num_gpus blocks; repartition back to the CPU-baseline block
    # count so impute/scale/write match the baseline's parallelism. Not folded
    # into the sort time.
    with m.stage("post_sort_repartition", ds.num_blocks()) as rec:
        ds = ds.repartition(target_blocks, shuffle=False).materialize()
        _, ncols, gib = ds_stats(ds)
        rec.update(rows=ds.count(), out_cols=ncols, gib=gib,
                   note=f"{extra['sort_blocks_out']}->{ds.num_blocks()} blocks")

    # ---- impute (CPU, reused verbatim) ------------------------------------ #
    with m.stage("impute", ncols) as rec:
        if indicator_cols:
            ds = ds.map_batches(
                lambda t: add_indicators(t, indicator_cols),
                batch_format="pyarrow",
                batch_size=None,
            )
        num_imputer = SimpleImputer(columns=roles.impute_numeric, strategy="mean")
        ds = num_imputer.fit_transform(ds)
        cat_imputer = None
        if roles.impute_categorical:
            cat_imputer = SimpleImputer(
                columns=roles.impute_categorical, strategy="most_frequent"
            )
            ds = cat_imputer.fit_transform(ds)
        ds = ds.materialize()
        _, ncols, gib = ds_stats(ds)
        rec.update(rows=ds.count(), out_cols=ncols, gib=gib)
    fitted["num_imputer"] = num_imputer
    fitted["cat_imputer"] = cat_imputer

    # ---- encode (GPU): GpuOrdinalEncoder ---------------------------------- #
    _wait_for_gpus(gpus)
    _gpu.reset_phase_stats()
    encoder = GpuOrdinalEncoder(columns=roles.categorical)
    with m.stage("encode", ncols) as rec:
        ds = encoder.fit_transform(ds).materialize()
        _, ncols, gib = ds_stats(ds)
        rec.update(rows=ds.count(), out_cols=ncols, gib=gib)
    enc_out_blocks = ds.num_blocks()
    rec["gpu"] = _gpu.collect_phase_stats()
    rec["out_blocks"] = enc_out_blocks
    extra["encode_phase_stats"] = rec["gpu"]
    extra["encode_out_blocks"] = enc_out_blocks
    fitted["encoder"] = encoder

    # ---- post-encode repartition (CONDITIONAL, TIMED SEPARATELY) ---------- #
    # GpuOrdinalEncoder's map_batches can emit far fewer blocks than the baseline
    # (one per batch-size chunk). If so, repartition back before CPU scale/write.
    used_post_encode = enc_out_blocks < 0.5 * target_blocks
    extra["post_encode_repartition_used"] = bool(used_post_encode)
    with m.stage("post_encode_repartition", enc_out_blocks) as rec:
        if used_post_encode:
            ds = ds.repartition(target_blocks, shuffle=False).materialize()
            _, ncols, gib = ds_stats(ds)
            rec.update(rows=ds.count(), out_cols=ncols, gib=gib,
                       note=f"{enc_out_blocks}->{ds.num_blocks()} blocks")
        else:
            rec.update(rows=ds.count(), out_cols=ncols, gib=0.0,
                       note=f"skipped ({enc_out_blocks} blocks >= target/2)")

    # ---- scale (CPU StandardScaler -- NOT a GPU stage) -------------------- #
    with m.stage("scale", ncols) as rec:
        scaler = StandardScaler(columns=roles.numeric_features)
        ds = scaler.fit_transform(ds).materialize()
        _, ncols, gib = ds_stats(ds)
        rec.update(rows=ds.count(), out_cols=ncols, gib=gib)
    fitted["scaler"] = scaler

    return ds, fitted


def _load_json(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    import logging

    available = criteo.discover_days()
    days = criteo.parse_days(ARGS.days, available)
    days_label = f"{days[0]}" if len(days) == 1 else f"{days[0]}-{days[-1]}"

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = ARGS.out or os.path.join(
        here, "data", f"criteo_days{days[0]}_{days[-1]}_gpu_sort_encode"
    )
    cpu_baseline_path = ARGS.cpu_baseline or os.path.join(
        here, "data", f"criteo_days{days[0]}_{days[-1]}_cpu_baseline", "manifest.json"
    )
    cpu_manifest = _load_json(cpu_baseline_path)

    osm_bytes = _resolve_object_store_bytes(ARGS.object_store_gb)
    ray.init(logging_level="ERROR", include_dashboard=False,
             object_store_memory=osm_bytes)
    logging.getLogger("ray.data").setLevel(logging.ERROR)
    ctx = ray.data.DataContext.get_current()
    ctx.enable_rich_progress_bars = False
    ctx.use_ray_tqdm = False
    ctx.execution_options.preserve_order = True

    roles = criteo.column_roles_multi(
        days, null_indicator_threshold=ARGS.null_indicator_threshold
    )

    P("=" * 80)
    P(f"CriteoPrivateAd GPU sort + GPU encode  days={days_label} "
      f"({len(days)} day_int partition(s))")
    P("=" * 80)
    P("scope: ONLY sort(CPU->GPU) and encode(CPU->GPU) change; "
      "impute/scale/read/prep/write stay CPU (scale is NOT GPU here)")
    P(f"GPU sort sizing : num_gpus={ARGS.gpus}  "
      f"POOL_FRAC={os.environ.get('RAY_DATA_GPU_SORT_POOL_FRAC', '0.80 (default)')}  "
      f"SPILL_FRAC={os.environ.get('RAY_DATA_GPU_SORT_SPILL_FRAC', 'off (default, single-node)')}")
    P(f"GPU encode sizing: batch_size={os.environ['RAY_DATA_GPU_PREPROC_BATCH_SIZE']}  "
      f"num_gpus={ARGS.gpus}  (independent of the sort)")
    P(f"object store: {osm_bytes / GIB:.0f} GiB (in-RAM /dev/shm; CPU-baseline policy)")
    P(f"sort key: {roles.sort_key}   (kept-raw metadata keys: {roles.metadata_keys})")
    if cpu_manifest:
        P(f"CPU baseline manifest: {cpu_baseline_path}")
    else:
        P(f"CPU baseline manifest NOT found at {cpu_baseline_path} "
          f"(side-by-side will show GPU only)")

    if ARGS.demo_oom:
        demo_oom(days, roles, ARGS.gpus)
        ray.shutdown()
        return

    m = Metrics()
    extra: Dict[str, Any] = {}

    # ---- read (CPU, reused policy) ---------------------------------------- #
    with m.stage("read", 0) as rec:
        ds = criteo.read_ray_days(days)
        if ARGS.rows:
            ds = ds.limit(ARGS.rows)
        if ARGS.blocks is not None:
            ds = ds.repartition(ARGS.blocks, shuffle=False)
        ds = ds.materialize()
        n_blocks = ds.num_blocks()
        rows_in, ncols, gib = ds_stats(ds)
        rec.update(rows=rows_in, out_cols=ncols, gib=gib)
    if criteo.SORT_DAY not in ds.schema().names:
        raise RuntimeError(f"{criteo.SORT_DAY!r} missing after read")
    extra["read_cols"] = ncols
    extra["read_blocks"] = n_blocks
    target_blocks = ARGS.blocks or n_blocks
    extra["target_blocks"] = target_blocks
    P(f"read produced {n_blocks} blocks; post-GPU-stage repartition target = "
      f"{target_blocks} blocks")

    # ---- pipeline (prep -> GPU sort -> repart -> impute -> GPU encode ->
    #      repart -> CPU scale) --------------------------------------------- #
    ds, fitted = run_pipeline(ds, roles, m, target_blocks, ARGS.gpus, extra)
    final_rows, final_cols, final_gib = ds_stats(ds)

    # ---- write (CPU) ------------------------------------------------------ #
    if not ARGS.no_write:
        with m.stage("write", final_cols) as rec:
            if os.path.isdir(out_dir):
                shutil.rmtree(out_dir)
            os.makedirs(out_dir, exist_ok=True)
            ds.write_parquet(out_dir)
            rec.update(rows=rows_in, out_cols=final_cols, gib=final_gib)

    # ---- verify saved-output global sortedness (same check as CPU run) ---- #
    if not ARGS.no_write:
        P(f"\nverifying saved-output global sortedness over {roles.sort_key} ...")
        verify = _verify_saved_sorted(out_dir, roles.sort_key, rows_in)
    else:
        verify = _verify_dataset_sorted(ds, roles.sort_key, rows_in)

    spill = ray_spill_summary()
    checks = cpu_base._sanity_checks(ds, roles, rows_in, verify, out_dir, ARGS.no_write)

    _print_report(m, roles, extra, verify, spill, checks, cpu_manifest, fitted,
                  rows_in, ARGS.gpus)

    if not ARGS.no_write:
        manifest = _build_manifest(days, roles, rows_in, m, extra, verify, spill,
                                   fitted, osm_bytes, target_blocks)
        with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=2, default=_jsonable)
        P(f"\nwrote output parquet + manifest.json -> {out_dir}")

    G.kill_actor_pool(ARGS.gpus)
    _gpu.kill_phase_stats_actor()
    ray.shutdown()


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
_STAGE_ORDER = ["read", "prep", "sort", "post_sort_repartition", "impute",
                "encode", "post_encode_repartition", "scale", "write"]
_CPU_LABEL = {  # which CPU baseline stage each GPU stage compares against
    "read": "read", "prep": "prep", "sort": "sort", "impute": "impute",
    "encode": "encode", "scale": "scale", "write": "write",
}


def _print_report(m, roles, extra, verify, spill, checks, cpu_manifest, fitted,
                  rows_in, gpus) -> None:
    gpu = {rec["name"]: rec["secs"] for rec in m.stages}
    gpu_total = sum(rec["secs"] for rec in m.stages)
    cpu_t = (cpu_manifest or {}).get("stage_timings_s", {}) if cpu_manifest else {}
    cpu_total = sum(cpu_t.values()) if cpu_t else 0.0

    P("\n" + "-" * 80)
    P("CPU baseline vs GPU (sort+encode) -- per-stage wall, RAM -> RAM")
    P(f"  {'stage':<24} {'CPU (s)':>10} {'GPU (s)':>10} {'speedup':>9}   note")
    for name in _STAGE_ORDER:
        g = gpu.get(name)
        if g is None:
            continue
        cpu_name = _CPU_LABEL.get(name)
        c = cpu_t.get(cpu_name) if cpu_name else None
        is_gpu = name in GPU_STAGES
        if c is not None:
            spd = f"{c / g:7.2f}x" if g else "    n/a"
            cpu_s = f"{c:>10.2f}"
        else:
            spd, cpu_s = "    n/a", f"{'n/a':>10}"
        note = "GPU" if is_gpu else ("GPU-overhead" if "repartition" in name else "CPU")
        rec = next(r for r in m.stages if r["name"] == name)
        if rec.get("note"):
            note += f" [{rec['note']}]"
        P(f"  {name:<24} {cpu_s} {g:>10.2f} {spd:>9}   {note}")
    P("  " + "-" * 70)
    if cpu_total:
        P(f"  {'TOTAL':<24} {cpu_total:>10.2f} {gpu_total:>10.2f} "
          f"{cpu_total / gpu_total:>8.2f}x   end-to-end")
    else:
        P(f"  {'TOTAL':<24} {'n/a':>10} {gpu_total:>10.2f}")

    # ---- the two required subtotals -------------------------------------- #
    g_sort = gpu.get("sort", 0.0)
    g_enc = gpu.get("encode", 0.0)
    g_rps = gpu.get("post_sort_repartition", 0.0)
    g_rpe = gpu.get("post_encode_repartition", 0.0)
    c_sort = cpu_t.get("sort")
    c_enc = cpu_t.get("encode")
    P("\nreplaced-stage subtotals (the honest accounting):")
    if c_sort is not None and c_enc is not None:
        pure_cpu = c_sort + c_enc
        pure_gpu = g_sort + g_enc
        P(f"  1) PURE replaced operators:")
        P(f"       CPU sort+encode                                   = "
          f"{pure_cpu:9.2f} s")
        P(f"       GPU sort+encode                                   = "
          f"{pure_gpu:9.2f} s   ({pure_cpu / pure_gpu:.2f}x)")
        pipe_gpu = g_sort + g_rps + g_enc + g_rpe
        P(f"  2) HONEST pipeline replacement (incl. repartitions):")
        P(f"       CPU sort+encode                                   = "
          f"{pure_cpu:9.2f} s")
        P(f"       GPU sort+post_sort_rep+encode+post_encode_rep     = "
          f"{pipe_gpu:9.2f} s   ({pure_cpu / pipe_gpu:.2f}x)")
        P(f"       (post_sort_repartition={g_rps:.2f}s, "
          f"post_encode_repartition={g_rpe:.2f}s -- NOT hidden in sort/encode)")
    else:
        P("  (CPU baseline manifest missing -> subtotals unavailable)")

    _print_sort_sizing(extra, spill, gpus)
    _print_resident_projection(gpu, extra, gpus)

    P("\nsaved-output sortedness verification:")
    P(f"  globally_sorted = {verify['globally_sorted']}  "
      f"(in_block_sorted={verify['all_blocks_internally_sorted']}, "
      f"boundaries_ok={verify['adjacent_boundaries_ok']})")
    P(f"  rows_counted = {verify['rows_counted']:,} / {verify['expected_rows']:,} "
      f"(match={verify['rows_match']}), blocks={verify['n_blocks']}, "
      f"key={verify['sort_key']}")

    P("\nsanity checks (same as CPU baseline):")
    for name, ok in checks.items():
        P(f"  {'PASS' if ok else 'FAIL'}  {name}")

    # Encoder vocab parity vs the CPU baseline manifest.
    if cpu_manifest:
        _print_vocab_parity(roles, fitted["encoder"], cpu_manifest)


def _print_sort_sizing(extra, spill, gpus) -> None:
    s = extra.get("sort_gpu_stats", {}) or {}
    in_bytes = extra.get("sort_input_bytes", 0)
    in_rows = extra.get("sort_input_rows", 0)
    in_blocks = extra.get("sort_input_blocks", 0)
    pool_max = s.get("pool_max_bytes", 0)
    total_vram = s.get("total_vram_bytes", 0)
    peak_per_rank = s.get("peak_vram_bytes_per_rank", []) or []
    peak_max = s.get("peak_vram_bytes_max", 0)
    resident = s.get("resident_bytes_per_rank", []) or []
    rows_rank = s.get("rows_in_per_rank", []) or []
    spill_frac = s.get("spill_frac", None)
    est_per_rank = in_bytes / gpus if gpus else 0

    P("\n" + "=" * 80)
    P("GPU SORT block/partition sizing + memory pressure  (biggest dataset)")
    P("=" * 80)
    P(f"  input rows (after prep)      : {in_rows:,}")
    P(f"  input bytes (after prep)     : {fmt_gib(in_bytes)} GiB")
    P(f"  input block count            : {in_blocks}")
    P(f"  num_gpus (ranks)             : {gpus}")
    P(f"  per-rank ESTIMATED rows/bytes: {in_rows // gpus if gpus else 0:,} rows / "
      f"{fmt_gib(est_per_rank)} GiB   (= pruned / num_gpus)")
    if resident:
        P(f"  per-rank MEASURED resident   : "
          f"rows={rows_rank}  bytes={[round(b / GIB, 2) for b in resident]} GiB")
    P(f"  RMM pool cap per GPU         : {fmt_gib(pool_max)} GiB "
      f"(of {fmt_gib(total_vram)} GiB total VRAM)")
    if peak_per_rank:
        P(f"  peak VRAM per rank (RMM)     : "
          f"{[round(b / GIB, 2) for b in peak_per_rank]} GiB")
    P(f"  peak VRAM max (RMM)          : {fmt_gib(peak_max)} GiB")
    P(f"  peak VRAM (nvidia-smi, 1 GPU): {extra.get('sort_nvsmi_peak_mib', 0) / 1024:6.2f} GiB")
    if pool_max:
        P(f"  headroom vs pool cap         : {fmt_gib(pool_max - peak_max)} GiB")
    if total_vram:
        P(f"  headroom vs total VRAM       : {fmt_gib(total_vram - peak_max)} GiB")

    P("\n  spill / OOM status -- EXPECTED vs MEASURED (not claimed from expectation):")
    P(f"    Expected : pruned sort input ~{fmt_gib(in_bytes)} GiB over {gpus} GPUs "
      f"= ~{fmt_gib(est_per_rank)} GiB/GPU before overhead; pool cap ~{fmt_gib(pool_max)} "
      f"GiB -> should fit with spill OFF.")
    dev_spill = "disabled (None)" if spill_frac is None else f"{spill_frac}"
    P(f"    Measured :")
    P(f"      RAY_DATA_GPU_SORT_SPILL_FRAC          = {dev_spill}")
    P(f"      GPU/RMM device spill engaged          = "
      f"{'no (path disabled)' if spill_frac is None else 'see peak vs limit'}")
    fit_ok = bool(pool_max) and peak_max < pool_max
    P(f"      peak VRAM < RMM pool cap (no OOM)      = {fit_ok} "
      f"({fmt_gib(peak_max)} < {fmt_gib(pool_max)} GiB)")
    P(f"      Ray object-store spilled bytes        = "
      f"{spill.get('spilled_mib', 0)} MiB")
    P(f"      Ray object-store restored bytes       = "
      f"{spill.get('restored_mib', 0)} MiB")
    P(f"      spill dir ({spill.get('spill_dir')}) bytes = "
      f"{spill.get('spill_dir_bytes', 0)}")
    # Explicit, unambiguous statements (only when actually true).
    if spill.get("spilled_mib", -1) == 0 and spill.get("spill_dir_bytes", 1) == 0:
        P("      -> Ray object-store spilled bytes = 0 (no /tmp spill)")
    if spill_frac is None:
        P("      -> GPU/RMM device spill = disabled / not engaged")
    if fit_ok:
        P("      -> No OOM (peak VRAM fit under the RMM pool cap)")


def _print_resident_projection(gpu, extra, gpus) -> None:
    s = extra.get("sort_gpu_stats", {}) or {}
    enc = (extra.get("encode_phase_stats", {}) or {}).get("phases", {}) or {}
    # The sort phase timers are per-rank walls (max across ranks). The encode
    # phase timers are SUMMED across all transform batches AND all GPU workers
    # (GPU-seconds), so divide by concurrency for a wall-equivalent.
    g = max(1, gpus)
    enc_h2d_w = enc.get("h2d", 0) / g
    enc_cmp_w = enc.get("compute", 0) / g
    enc_d2h_w = enc.get("d2h", 0) / g
    P("\n" + "=" * 80)
    P("Current (cold start + transfers) vs theoretical device-resident block")
    P("=" * 80)
    P("  GPU SORT (per-rank wall; transfers dominate FULL):")
    P(f"    stage wall (incl. actor cold start + Ray) : {gpu.get('sort', 0):7.2f} s")
    P(f"    in-fn FULL (H2D+sort+shuffle+D2H+emit)    : {s.get('full_s', 0):7.2f} s")
    P(f"      H2D {s.get('h2d_s', 0):.2f}  gpu_only {s.get('gpu_only_s', 0):.2f}  "
      f"shuffle {s.get('shuffle_s', 0):.2f}  D2H {s.get('d2h_s', 0):.2f}  "
      f"emit(D2H->obj store+wait) {s.get('emit_s', 0):.2f}")
    transfer = (s.get('h2d_s', 0) + s.get('d2h_s', 0) + s.get('emit_s', 0))
    P(f"    transfer/move (H2D+D2H+emit) {transfer:.2f}s vs compute (gpu_only) "
      f"{s.get('gpu_only_s', 0):.2f}s  -> sort is TRANSFER-bound")
    P(f"    theoretical resident (gpu_only, paid once) : {s.get('gpu_only_s', 0):7.3f} s")
    P(f"  GPU ENCODE transform phases (GPU-seconds summed over {gpus} workers; "
      f"wall-equiv = /{gpus}):")
    P(f"    stage wall (fit + transform + cold start) : {gpu.get('encode', 0):7.2f} s")
    P(f"      aggregate GPU-s : H2D {enc.get('h2d', 0):.2f}  "
      f"compute {enc.get('compute', 0):.2f}  D2H {enc.get('d2h', 0):.2f}")
    P(f"      wall-equiv      : H2D {enc_h2d_w:.2f}  compute {enc_cmp_w:.2f}  "
      f"D2H {enc_d2h_w:.2f}  -> encode is COMPUTE-bound (transfers small)")
    # Chain, in consistent wall-equivalent units: a device-resident sort->encode
    # pays H2D once at ingest, runs sort gpu_only + encode compute on-device, and
    # D2H once at output -- removing the sort's emit (D2H->object store) and the
    # inter-stage round-trip entirely.
    h2d_once = s.get("h2d_s", 0)
    sort_only = s.get("gpu_only_s", 0)
    d2h_once = max(enc_d2h_w, s.get("d2h_s", 0))
    resident_chain = h2d_once + sort_only + enc_cmp_w + d2h_once
    current_chain = gpu.get("sort", 0) + gpu.get("encode", 0)
    P("  CHAIN sort->encode (device-resident projection, wall-equivalent):")
    P(f"    current (two host-staged GPU stages, incl. cold start) : "
      f"{current_chain:7.2f} s")
    P(f"    resident lower bound (H2D once + sort_gpu_only + encode_compute_wall "
      f"+ D2H once) = {resident_chain:6.2f} s")
    if resident_chain > 0:
        P(f"    -> device residency removes the sort's emit/D2H ({s.get('emit_s',0):.1f}+"
          f"{s.get('d2h_s',0):.1f}s) and the encode's fixed actor/fit overhead; "
          f"projected ~{current_chain / resident_chain:.1f}x the current GPU "
          f"sort+encode wall")


def _print_vocab_parity(roles, encoder, cpu_manifest) -> None:
    cpu_vocab = (cpu_manifest.get("fitted", {}) or {}).get("encoder_vocab_size", {})
    if not cpu_vocab:
        return
    mism = []
    for col in roles.categorical:
        try:
            gpu_n = _vocab_size(encoder.stats_[f"unique_values({col})"])
        except Exception:
            continue
        cpu_n = cpu_vocab.get(col)
        if cpu_n is not None and int(cpu_n) != int(gpu_n):
            mism.append((col, cpu_n, gpu_n))
    P("\nencoder vocab parity vs CPU baseline:")
    if not mism:
        P(f"  PASS  all {len(roles.categorical)} categorical vocab sizes match CPU")
    else:
        P(f"  FAIL  {len(mism)} mismatched: " +
          ", ".join(f"{c}(cpu={a},gpu={b})" for c, a, b in mism[:6]))


def _build_manifest(days, roles, rows_in, m, extra, verify, spill, fitted,
                    osm_bytes, target_blocks) -> Dict[str, Any]:
    scaler = fitted["scaler"]
    encoder = fitted["encoder"]
    num_imputer = fitted["num_imputer"]
    cat_imputer = fitted.get("cat_imputer")
    s = extra.get("sort_gpu_stats", {}) or {}
    return {
        "dataset": "CriteoPrivateAd",
        "run": "gpu_sort_encode",
        "gpu_stages": sorted(GPU_STAGES),
        "scale_backend": "cpu",
        "days": days,
        "rows": rows_in,
        "sort_key": roles.sort_key,
        "target_blocks": target_blocks,
        "object_store_gib": round(osm_bytes / GIB, 1),
        "stage_timings_s": {rec["name"]: round(rec["secs"], 3) for rec in m.stages},
        "gpu_sort_sizing": {
            "num_gpus": s.get("num_gpus"),
            "input_rows": extra.get("sort_input_rows"),
            "input_bytes": extra.get("sort_input_bytes"),
            "input_blocks": extra.get("sort_input_blocks"),
            "blocks_out": extra.get("sort_blocks_out"),
            "per_rank_resident_bytes": s.get("resident_bytes_per_rank"),
            "per_rank_rows": s.get("rows_in_per_rank"),
            "rmm_pool_cap_bytes": s.get("pool_max_bytes"),
            "total_vram_bytes": s.get("total_vram_bytes"),
            "peak_vram_per_rank_bytes": s.get("peak_vram_bytes_per_rank"),
            "peak_vram_max_bytes": s.get("peak_vram_bytes_max"),
            "nvsmi_peak_mib": extra.get("sort_nvsmi_peak_mib"),
            "spill_frac": s.get("spill_frac"),
        },
        "gpu_sort_phase_s": {k: s.get(k) for k in
                             ("h2d_s", "gpu_only_s", "shuffle_s", "d2h_s",
                              "emit_s", "full_s")},
        "gpu_encode_sizing": {
            "batch_size": int(os.environ["RAY_DATA_GPU_PREPROC_BATCH_SIZE"]),
            "num_gpus": ARGS.gpus,
            "out_blocks": extra.get("encode_out_blocks"),
            "post_encode_repartition_used": extra.get("post_encode_repartition_used"),
        },
        "gpu_encode_phase_s": (extra.get("encode_phase_stats", {}) or {}).get("phases"),
        "spill": {
            "ray_object_store_spilled_mib": spill.get("spilled_mib"),
            "ray_object_store_restored_mib": spill.get("restored_mib"),
            "ray_spill_dir": spill.get("spill_dir"),
            "ray_spill_dir_bytes": spill.get("spill_dir_bytes"),
            "gpu_rmm_device_spill_frac": s.get("spill_frac"),
        },
        "fitted": {
            "encoder_vocab_size": {
                c: _vocab_size(encoder.stats_[f"unique_values({c})"])
                for c in roles.categorical
            },
            "imputer_mean": {
                c: num_imputer.stats_[f"mean({c})"] for c in roles.impute_numeric
            },
            "imputer_most_frequent": (
                {c: cat_imputer.stats_[f"most_frequent({c})"]
                 for c in roles.impute_categorical}
                if cat_imputer is not None else {}
            ),
            "scaler": {
                c: {"mean": scaler.stats_[f"mean({c})"],
                    "std": scaler.stats_[f"std({c})"]}
                for c in roles.numeric_features
            },
        },
        "verification": verify,
    }


# --------------------------------------------------------------------------- #
# Stress / demo: a per-rank partition that does not fit in VRAM
# --------------------------------------------------------------------------- #
def demo_oom(days, roles, gpus) -> None:
    """Force a per-rank partition OOM (too few GPUs and/or low pool frac, single-
    node spill off), then show device spill mitigates it. Reads + preps the full
    selected days, then sorts."""
    P("\n" + "=" * 80)
    P(f"--demo-oom: per-rank partition that does not fit in VRAM "
      f"(num_gpus={gpus}, POOL_FRAC={os.environ.get('RAY_DATA_GPU_SORT_POOL_FRAC','0.80')})")
    P("=" * 80)
    numeric_raw = tuple(roles.numeric_raw)
    categorical = tuple(roles.categorical)
    list_features = tuple(roles.list_features)
    ds = criteo.read_ray_days(days)
    if ARGS.rows:
        ds = ds.limit(ARGS.rows)
    ds = ds.map_batches(
        lambda t: prep_batch(t, numeric_raw, categorical, list_features),
        batch_format="pyarrow", batch_size=None,
    ).materialize()
    in_bytes = int(ds.size_bytes() or 0)
    P(f"  pruned input: {ds.count():,} rows / {fmt_gib(in_bytes)} GiB; "
      f"per-rank estimate = {fmt_gib(in_bytes / gpus)} GiB over {gpus} GPUs")

    P("\n  [1] sort with spill OFF (single-node default) -- expect OOM if a rank "
      "exceeds the pool cap:")
    sampler = GpuMemSampler().start()
    try:
        out = ds.sort(roles.sort_key, backend="gpu").materialize()
        out.count()
        sampler.stop()
        P("      RESULT: completed (no OOM). Peak VRAM (nvidia-smi): "
          f"{sampler.peak_mib / 1024:.2f} GiB/GPU. "
          "Lower --gpus or --pool-frac to force the OOM.")
        G.kill_actor_pool(gpus)
        return
    except Exception as e:
        sampler.stop()
        msg = str(e).strip().splitlines()[-1] if str(e).strip() else repr(e)
        P(f"      RESULT: OOM / failure as expected -> {type(e).__name__}: {msg[:300]}")
    finally:
        G.kill_actor_pool(gpus)
        time.sleep(3)

    P("\n  [2] mitigation: enable device->host spill "
      "(RAY_DATA_GPU_SORT_SPILL_FRAC=0.70) and re-sort the SAME data:")
    os.environ["RAY_DATA_GPU_SORT_SPILL_FRAC"] = "0.70"
    sampler = GpuMemSampler().start()
    try:
        t0 = time.perf_counter()
        out = ds.sort(roles.sort_key, backend="gpu").materialize()
        n = out.count()
        dt = time.perf_counter() - t0
        sampler.stop()
        P(f"      RESULT: completed with spill ON in {dt:.1f}s, rows={n:,}, "
          f"peak VRAM (nvidia-smi) {sampler.peak_mib / 1024:.2f} GiB/GPU "
          "(slower but OOM-safe).")
    except Exception as e:
        sampler.stop()
        msg = str(e).strip().splitlines()[-1] if str(e).strip() else repr(e)
        P(f"      RESULT: still failed -> {type(e).__name__}: {msg[:300]}  "
          "(try more GPUs: per-rank shrinks as num_gpus grows)")
    finally:
        G.kill_actor_pool(gpus)
    P("\n  mitigations: (a) add GPUs -> per-rank = pruned/num_gpus shrinks; "
      "(b) enable RAY_DATA_GPU_SORT_SPILL_FRAC device->host spill; "
      "(c) raise RAY_DATA_GPU_SORT_POOL_FRAC if VRAM allows.")


if __name__ == "__main__":
    main()




