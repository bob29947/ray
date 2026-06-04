"""End-to-end sort benchmark: pyarrow-CPU vs polars-CPU vs general-GPU.

Each backend runs in its OWN fresh process + ``ray.init()``, the dataset is
rebuilt from seed 0, a warmup is measured but NOT counted, then ``--trials``
timed trials are run, followed by an independent full-scan correctness check of
the result (row count / key sum / min / max / global monotonicity).

Backends (all sort the SAME dataset; only the sort path differs):

    cpu          ds.sort("c0").materialize()              # default pyarrow CPU sort
    polars       ds.sort("c0").materialize()              # ctx.use_polars_sort=True
    gpu_general  ds.sort("c0", gpu=True).materialize()    # cuDF + rapidsmpf engine
    gpu_tuned    ds.sort("c0", gpu=True).materialize()    # RAY_DATA_GPU_SORT_IMPL=tuned

This is the Ray-3.0 / forked-Ray port of ``sort_iso_bench/cpu_vs_gpu_general.py``.
The only behavioral change vs that harness is how the polars Ray-CPU baseline is
enabled: Ray 3.0 dropped the ``RAY_DATA_USE_POLARS_SORT`` env var, so we set
``DataContext.use_polars_sort`` directly (it is propagated to the shuffle tasks).

Run (against the forked source Ray, NOT a pip install):

    PYTHONPATH=<worktree>/python \\
    <venv>/bin/python gpu_sort_bench/benchmark.py --gpus 16 --trials 3
    ... --quick                 # tiny 1 GiB sanity check
    ... --backends cpu,gpu_general
"""

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time

RESULT_PREFIX = "RESULT_JSON:"
ALL_BACKENDS = ("cpu", "polars", "gpu_general", "gpu_tuned")
DEFAULT_BACKENDS = ("cpu", "polars", "gpu_general")
TAG = {"cpu": "CPU", "polars": "POLARS", "gpu_general": "GPU-GENERAL",
       "gpu_tuned": "GPU-TUNED"}
# Documented Ray Data pyarrow CPU baseline on the 16xV100 reference box.
CPU_BASELINE_S = 45.691


def object_store_bytes():
    free = shutil.disk_usage("/dev/shm").free
    return int(min(320 * 2**30, free * 0.55))


# =========================================================================
# WORKER: runs ONE backend in its own process, with its own Ray.
# =========================================================================
def run_worker(backend, rows, cols, blocks, trials, gpus):
    use_gpu = backend in ("gpu_tuned", "gpu_general")
    # Select the GPU engine BEFORE importing ray so shuffle worker processes
    # inherit it (the engine reads these env vars at runtime).
    if use_gpu:
        os.environ["RAY_DATA_GPU_SORT_IMPL"] = (
            "tuned" if backend == "gpu_tuned" else "general"
        )
        os.environ["RAY_DATA_GPU_SORT_NUM_GPUS"] = str(gpus)
    os.environ["RAY_DATA_GPU_SORT"] = "0"  # use the ds.sort(gpu=True) flag
    if backend == "polars":
        # One polars thread per Ray task: Ray supplies the cross-block
        # parallelism, polars the faster per-block kernel. Without this polars
        # grabs all cores in every task and oversubscribes (slower than pyarrow).
        os.environ["POLARS_MAX_THREADS"] = "1"

    import logging
    import numpy as np
    import pyarrow as pa
    import ray
    from ray.data import DataContext

    tag = TAG[backend]
    ray.init(object_store_memory=object_store_bytes(), include_dashboard=False,
             logging_level="ERROR")
    logging.getLogger("ray.data").setLevel(logging.ERROR)
    ctx = DataContext.get_current()
    ctx.enable_rich_progress_bars = False
    ctx.use_ray_tqdm = False
    # Ray 3.0: enable polars sort via the DataContext (propagated to tasks).
    ctx.use_polars_sort = backend == "polars"
    assert ctx.use_polars_sort == (backend == "polars"), "polars flag wiring broke"

    # --- build the dataset and remember the input's key fingerprint ---
    rpb = rows // blocks
    rng = np.random.default_rng(0)
    in_count, in_sum, in_min, in_max = 0, 0, None, None
    refs = []
    for _ in range(blocks):
        c0 = rng.integers(0, 2**31 - 1, rpb, dtype=np.int32)
        in_count += int(c0.shape[0])
        in_sum += int(c0.astype(np.int64).sum())
        lo, hi = int(c0.min()), int(c0.max())
        in_min = lo if in_min is None else min(in_min, lo)
        in_max = hi if in_max is None else max(in_max, hi)
        data = {"c0": c0}
        for j in range(1, cols):
            data[f"c{j}"] = np.zeros(rpb, dtype=np.int32)
        refs.append(ray.put(pa.table(data)))
    ds = ray.data.from_arrow_refs(refs)

    def do_sort():
        if use_gpu:
            return ds.sort("c0", gpu=True).materialize()
        return ds.sort("c0").materialize()

    def phase_stats():
        if backend == "gpu_tuned":
            from ray.data._internal.planner.gpu_sort import LAST_RUN_STATS
            return dict(LAST_RUN_STATS)
        if backend == "gpu_general":
            from ray.data._internal.planner.gpu_sort_general import LAST_RUN_STATS
            return dict(LAST_RUN_STATS)
        return {}

    print(f"[{tag}] dataset ready ({rows:,} rows, {rows * cols * 4 / 2**30:.0f} GiB). "
          f"warming up...", flush=True)
    w0 = time.perf_counter()
    do_sort()
    warmup_s = time.perf_counter() - w0
    print(f"[{tag}] warmup: {warmup_s:8.3f} s (not counted)", flush=True)

    times = []
    last = None
    phase_best = None
    for t in range(trials):
        t0 = time.perf_counter()
        sorted_ds = do_sort()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        ph = phase_stats()
        if phase_best is None or (t1 - t0) <= min(times):
            phase_best = ph
        extra = ""
        if ph:
            extra = (f"  [full={ph.get('full_s', 0):.3f} h2d={ph.get('h2d_s', 0):.3f} "
                     f"gpu={ph.get('gpu_sort_s', ph.get('gpu_only_s', 0)):.3f} "
                     f"shuf={ph.get('shuffle_s', 0):.3f} d2h={ph.get('d2h_s', 0):.3f}]")
        print(f"[{tag}] run {t + 1}/{trials}: {t1 - t0:8.3f} s{extra}", flush=True)
        if last is not None:
            del last
        last = sorted_ds

    # --- prove it is ACTUALLY sorted (independent scan, not timed) ---
    print(f"[{tag}] checking the result is actually sorted...", flush=True)
    prev, rows_seen, ksum, kmin, kmax, monotonic = None, 0, 0, None, None, True
    for batch in last.iter_batches(batch_size=8_000_000, batch_format="numpy"):
        c0 = batch["c0"]
        if c0.size == 0:
            continue
        if not bool(np.all(c0[1:] >= c0[:-1])):
            monotonic = False
        if prev is not None and int(c0[0]) < prev:
            monotonic = False
        prev = int(c0[-1])
        rows_seen += int(c0.size)
        ksum += int(c0.astype(np.int64).sum())
        lo, hi = int(c0.min()), int(c0.max())
        kmin = lo if kmin is None else min(kmin, lo)
        kmax = hi if kmax is None else max(kmax, hi)

    rows_ok = rows_seen == in_count
    sum_ok = ksum == in_sum
    minmax_ok = (kmin == in_min and kmax == in_max)
    ok = monotonic and rows_ok and sum_ok and minmax_ok
    print(f"[{tag}] sorted={monotonic}  rows={rows_seen:,}({'ok' if rows_ok else 'BAD'})  "
          f"key_sum={'ok' if sum_ok else 'BAD'}  min/max={'ok' if minmax_ok else 'BAD'}  "
          f"-> {'PASS' if ok else 'FAIL'}", flush=True)

    print(RESULT_PREFIX + json.dumps(
        {"backend": backend, "best": min(times),
         "median": statistics.median(times), "mean": sum(times) / len(times),
         "warmup": warmup_s, "trials": times, "ok": bool(ok),
         "phase": phase_best or {}}), flush=True)

    # Release the GPUs / detached actors promptly.
    try:
        from ray.data._internal.planner.gpu_sort import _SORTER_NAME
        ray.kill(ray.get_actor(_SORTER_NAME))
    except Exception:
        pass
    try:
        from ray.data._internal.planner.gpu_sort_general import kill_actor_pool
        kill_actor_pool(gpus)
    except Exception:
        pass
    ray.shutdown()


# =========================================================================
# PARENT: launches each backend in a fresh subprocess and summarizes.
# =========================================================================
_NOISE = ("INFO ", "WARNING ", "warnings.warn", "FutureWarning", "(raylet)",
          "Tip:", "namespace=", "Started a local Ray", "logging_progress",
          "streaming_executor", "resource_manager", "Registered dataset",
          "FrontendNotFoundError", "dashboard", "Dashboard")


def _interesting(line):
    return line.strip() and not any(n in line for n in _NOISE)


def run_backend_in_subprocess(backend, trials, gpus, quick):
    print(f"\n=== {TAG[backend]}: fresh process + fresh ray.init() ===", flush=True)
    cmd = [sys.executable, os.path.abspath(__file__), "--backend", backend,
           "--trials", str(trials), "--gpus", str(gpus)]
    if quick:
        cmd.append("--quick")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    result = None
    for line in proc.stdout:
        line = line.rstrip("\n")
        if line.startswith(RESULT_PREFIX):
            result = json.loads(line[len(RESULT_PREFIX):])
        elif _interesting(line):
            print("   " + line, flush=True)
    proc.wait()
    if result is None:
        # Non-fatal: a single backend failure shouldn't abort the comparison.
        print(f"   !! {backend} run FAILED (exit {proc.returncode})", flush=True)
        return {"backend": backend, "best": None, "ok": False, "failed": True}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--gpus", type=int, default=16)
    parser.add_argument("--quick", action="store_true", help="tiny 1 GiB smoke test")
    parser.add_argument("--backends", default=",".join(DEFAULT_BACKENDS),
                        help="comma-separated subset of: " + ",".join(ALL_BACKENDS))
    parser.add_argument("--backend", choices=ALL_BACKENDS,
                        help="(internal) run a single backend in this process")
    args = parser.parse_args()

    rows = (16 * 1024 * 1024) if args.quick else (1024 * 1024 * 1024)
    cols, blocks = 16, (64 if args.quick else 256)

    if args.backend:
        run_worker(args.backend, rows, cols, blocks, args.trials, args.gpus)
        return

    backends = [b for b in args.backends.split(",") if b]
    results = {}
    for backend in backends:
        results[backend] = run_backend_in_subprocess(backend, args.trials, args.gpus,
                                                      args.quick)
        time.sleep(5)

    cpu_best = (results.get("cpu") or {}).get("best")
    pol_best = (results.get("polars") or {}).get("best")
    print("\n" + "=" * 90)
    print(f"  dataset: {rows * cols * 4 / 2**30:.0f} GiB ({rows:,} rows x {cols} int32)")
    base = f"  CPU baseline (documented): {CPU_BASELINE_S:.3f} s"
    if cpu_best:
        base += f"   (this run: pyarrow {cpu_best:.3f} s"
        if pol_best:
            base += f", polars {pol_best:.3f} s"
        base += ")"
    print(base)
    print("-" * 90)
    for b in backends:
        r = results[b]
        if r.get("failed"):
            print(f"  {TAG[b]:<12} FAILED")
            continue
        med = r.get("median", r.get("mean"))
        vs_cpu = f"{cpu_best / r['best']:5.1f}x vs pyarrow" if (cpu_best and r["best"]) else ""
        vs_pol = f"{pol_best / r['best']:5.1f}x vs polars" if (pol_best and r["best"]) else ""
        print(f"  {TAG[b]:<12} best {r['best']:8.3f} s  median {med:8.3f} s  "
              f"warmup {r.get('warmup', 0):7.3f} s  "
              f"{'PASS' if r['ok'] else 'FAIL'}   {vs_cpu}  {vs_pol}")
        ph = r.get("phase") or {}
        if ph:
            gpu_only = ph.get("gpu_sort_s", ph.get("gpu_only_s"))
            line = f"               FULL {ph.get('full_s', 0):.3f}s  (h2d {ph.get('h2d_s', 0):.3f}"
            if ph.get("shuffle_s"):
                line += f", shuffle {ph['shuffle_s']:.3f}"
            line += f", d2h {ph.get('d2h_s', 0):.3f})"
            if gpu_only is not None:
                line += f"   GPU-only {gpu_only:.3f}s"
            print(line)
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
