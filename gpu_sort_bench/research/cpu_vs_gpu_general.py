"""
Four-way side-by-side sort benchmark: pyarrow-CPU vs polars-CPU vs tuned-GPU vs
general-GPU, same harness as ``cpu_vs_gpu.py`` (each backend in its OWN fresh
process + ray.init, the dataset rebuilt from seed 0, a warmup that is measured
but NOT counted, then >=3 timed trials), plus an independent full-scan
correctness check of every result. Reports BEST and MEDIAN per backend, and the
GPU speedups vs BOTH CPU baselines (pyarrow and polars).

Backends (all run the SAME 64 GiB dataset; the timed line differs only in how
the sort kernel / GPU path is selected):

    cpu          ds.sort("c0").materialize()              # default pyarrow CPU sort
    polars       ds.sort("c0").materialize()              # RAY_DATA_USE_POLARS_SORT=1
                 -> same Ray object-store shuffle, polars kernels per block
                    (POLARS_MAX_THREADS=1: one polars thread per Ray task)
    gpu_tuned    ds.sort("c0", gpu=True).materialize()    # RAY_DATA_GPU_SORT_IMPL=tuned
                 -> hand-tuned int32 matrix engine (gpu_sort.py)
    gpu_general  ds.sort("c0", gpu=True).materialize()    # RAY_DATA_GPU_SORT_IMPL=general
                 -> general cuDF + rapidsmpf engine (gpu_sort_general.py)

A/B NOTE: the tuned and general engines both patch the SAME Ray file
(`planner/sort.py`). They are toggled WITHOUT swapping files via the
``RAY_DATA_GPU_SORT_IMPL`` env var (set per-process by this harness), so both can
be measured in one run. ``gpu=True`` defaults to the general engine; the harness
sets IMPL=tuned for the tuned backend. ``polars`` is the *Ray* CPU option: the
polars env wiring is copied from ``cpu_vs_gpu.py`` and set BEFORE importing ray.

Run:
    .venv/bin/python cpu_vs_gpu_general.py            # full 64 GiB, 3 trials each
    .venv/bin/python cpu_vs_gpu_general.py --quick    # tiny 1 GiB sanity check
    .venv/bin/python cpu_vs_gpu_general.py --trials 5
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
BACKENDS = ("cpu", "polars", "gpu_tuned", "gpu_general")
TAG = {"cpu": "CPU", "polars": "POLARS", "gpu_tuned": "GPU-TUNED",
       "gpu_general": "GPU-GENERAL"}
CPU_BASELINE_S = 45.691  # documented Ray Data pyarrow CPU baseline on this box


def object_store_bytes():
    free = shutil.disk_usage("/dev/shm").free
    return int(min(320 * 2**30, free * 0.55))


# =========================================================================
# WORKER: runs ONE backend in its own process, with its own Ray.
# =========================================================================
def run_worker(backend, rows, cols, blocks, trials, gpus):
    use_gpu = backend in ("gpu_tuned", "gpu_general")
    # Select engine BEFORE importing ray so shuffle worker processes inherit it.
    if use_gpu:
        os.environ["RAY_DATA_GPU_SORT_IMPL"] = (
            "tuned" if backend == "gpu_tuned" else "general"
        )
        os.environ["RAY_DATA_GPU_SORT_NUM_GPUS"] = str(gpus)
    os.environ["RAY_DATA_GPU_SORT"] = "0"  # use the flag, not the legacy env on
    # The "Ray option" for Polars: Ray runs the SAME object-store shuffle but
    # sorts/merges each block with polars kernels instead of pyarrow. Set BEFORE
    # importing ray so the shuffle worker processes (which don't inherit the
    # driver's propagated DataContext) read these env-driven defaults.
    os.environ["RAY_DATA_USE_POLARS_SORT"] = "1" if backend == "polars" else "0"
    if backend == "polars":
        # Critical: one polars thread per Ray task. Polars otherwise grabs ALL
        # cores inside every block task, and Ray runs many block tasks at once,
        # so the default oversubscribes (~tasks x 96 threads on 96 cores) and is
        # SLOWER than pyarrow. With 1 thread/task Ray supplies the cross-block
        # parallelism and polars' faster kernel wins. (See cpu_vs_gpu.py.)
        os.environ["POLARS_MAX_THREADS"] = "1"

    import logging
    import numpy as np
    import pyarrow as pa
    import ray
    from ray.data import DataContext

    tag = TAG[backend]
    ray.init(object_store_memory=object_store_bytes())
    logging.getLogger("ray.data").setLevel(logging.WARNING)
    ctx = DataContext.get_current()
    ctx.enable_rich_progress_bars = False
    ctx.use_ray_tqdm = False
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
        # The single timed line. The GPU backends exercise the user-facing flag.
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

    # --- timed region (identical line for every backend) ---
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

    # release the GPUs / detached actors promptly
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
          "streaming_executor", "resource_manager", "Registered dataset")


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
        raise RuntimeError(f"{backend} run failed (exit {proc.returncode})")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--gpus", type=int, default=16)
    parser.add_argument("--quick", action="store_true", help="tiny 1 GiB smoke test")
    parser.add_argument("--backend", choices=BACKENDS,
                        help="(internal) run a single backend in this process")
    args = parser.parse_args()

    rows = (16 * 1024 * 1024) if args.quick else (1024 * 1024 * 1024)
    cols, blocks = 16, (64 if args.quick else 256)

    if args.backend:
        run_worker(args.backend, rows, cols, blocks, args.trials, args.gpus)
        return

    results = {}
    for backend in BACKENDS:
        results[backend] = run_backend_in_subprocess(backend, args.trials, args.gpus,
                                                      args.quick)
        time.sleep(5)

    cpu_best = results["cpu"]["best"]
    pol_best = results["polars"]["best"]
    print("\n" + "=" * 86)
    print(f"  dataset: {rows * cols * 4 / 2**30:.0f} GiB ({rows:,} rows x {cols} int32)")
    print(f"  CPU baseline (documented): {CPU_BASELINE_S:.3f} s   "
          f"(this run: pyarrow {cpu_best:.3f} s, polars {pol_best:.3f} s)")
    print("-" * 86)
    for b in BACKENDS:
        r = results[b]
        med = r.get("median", r["mean"])
        vs_cpu = f"{cpu_best / r['best']:5.1f}x vs pyarrow" if r["best"] else ""
        vs_pol = f"{pol_best / r['best']:5.1f}x vs polars" if r["best"] else ""
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
    print("=" * 86 + "\n")


if __name__ == "__main__":
    main()
