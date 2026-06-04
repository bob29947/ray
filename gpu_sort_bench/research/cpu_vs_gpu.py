"""
Time the SAME Ray sort on three backends -- cleanly and verifiably.

Backends (all run the EXACT same line `ds.sort("c0").materialize()`):

    cpu     normal Ray Data sort (pyarrow kernels)        RAY_DATA_GPU_SORT=0
    polars  Ray Data sort using Polars kernels            RAY_DATA_GPU_SORT=0 + use_polars_sort=True
    gpu     our multi-GPU sort                            RAY_DATA_GPU_SORT=1

`polars` is the *Ray* option: same Ray object-store shuffle, but Ray sorts each
block / merges with Polars instead of pyarrow (DataContext.use_polars_sort=True).

Two things this script is careful about:

1. CLEAN ISOLATION. Each backend runs in its OWN fresh process with its own
   ray.init() / ray.shutdown(). Ray is fully restarted between backends, the
   GPUs and /dev/shm object store are released between them, and no run inherits
   another's warm caches or scheduler state. The dataset is rebuilt each time
   from the same seed, so the input is byte-identical (apples to apples).

2. PROOF IT'S ACTUALLY SORTED. After timing, each run does an independent full
   scan of the result and checks it is globally non-decreasing across every
   block, and that row-count / key sum / min / max match the input.

Run it:
    .venv/bin/python cpu_vs_gpu.py            # full 64 GiB, 3 trials each
    .venv/bin/python cpu_vs_gpu.py --quick    # tiny 1 GiB  (fast sanity check)
    .venv/bin/python cpu_vs_gpu.py --trials 5
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

RESULT_PREFIX = "RESULT_JSON:"          # how the child hands numbers to the parent
BACKENDS = ("cpu", "polars", "gpu")
TAG = {"cpu": "CPU", "polars": "POLARS", "gpu": "GPU"}


def object_store_bytes():
    """A safe object-store size: big enough for a 64 GiB sort, but well under
    /dev/shm so a just-exited run's lingering shm can't trip the size check."""
    free = shutil.disk_usage("/dev/shm").free
    return int(min(320 * 2**30, free * 0.55))


# =========================================================================
# WORKER: runs ONE backend in its own process, with its own Ray.
# =========================================================================
def run_worker(backend, rows, cols, blocks, trials, gpus):
    # --- pick the backend (set BEFORE importing ray so the settings reach the
    # shuffle worker processes, which don't get the driver's propagated
    # DataContext -- they read these env-driven defaults) ---
    if backend == "gpu":
        os.environ["RAY_DATA_GPU_SORT"] = "1"
        os.environ["RAY_DATA_GPU_SORT_NUM_GPUS"] = str(gpus)
    else:
        os.environ["RAY_DATA_GPU_SORT"] = "0"
    # The "Ray option" for Polars: Ray sorts/merges each block with Polars.
    os.environ["RAY_DATA_USE_POLARS_SORT"] = "1" if backend == "polars" else "0"
    if backend == "polars":
        # Critical: one Polars thread per Ray task. Polars otherwise grabs ALL
        # cores inside every block task, and Ray runs many block tasks at once,
        # so the default oversubscribes (~tasks x 96 threads on 96 cores) and is
        # SLOWER than pyarrow. With 1 thread/task, Ray supplies the cross-block
        # parallelism and Polars' faster kernel wins. (Measured at 4 GiB: polars
        # default 3.14 s, polars 1-thread 1.65 s, pyarrow 2.09 s.)
        os.environ["POLARS_MAX_THREADS"] = "1"

    import logging
    import numpy as np
    import pyarrow as pa
    import ray
    from ray.data import DataContext

    tag = TAG[backend]

    ray.init(object_store_memory=object_store_bytes())
    logging.getLogger("ray.data").setLevel(logging.WARNING)  # quiet per-run logs
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

    print(f"[{tag}] dataset ready ({rows:,} rows, {rows * cols * 4 / 2**30:.0f} GiB). "
          f"warming up...", flush=True)
    w0 = time.perf_counter()
    ds.sort("c0").materialize()              # warmup: measured, but NOT counted
    warmup_s = time.perf_counter() - w0
    print(f"[{tag}] warmup: {warmup_s:8.3f} s (not counted)", flush=True)

    # --- the timed region (identical line for every backend) ---
    times = []
    last = None
    for t in range(trials):
        t0 = time.perf_counter()
        sorted_ds = ds.sort("c0").materialize()
        t1 = time.perf_counter()

        times.append(t1 - t0)
        print(f"[{tag}] run {t + 1}/{trials}: {t1 - t0:8.3f} s", flush=True)
        if last is not None:
            del last                          # free the previous result
        last = sorted_ds

    # --- prove it is ACTUALLY sorted (independent scan, not timed) ---
    print(f"[{tag}] checking the result is actually sorted...", flush=True)
    prev, rows_seen, ksum, kmin, kmax, monotonic = None, 0, 0, None, None, True
    for batch in last.iter_batches(batch_size=8_000_000, batch_format="numpy"):
        c0 = batch["c0"]
        if c0.size == 0:
            continue
        if not bool(np.all(c0[1:] >= c0[:-1])):      # ordered within the batch
            monotonic = False
        if prev is not None and int(c0[0]) < prev:   # ordered across the boundary
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
        {"backend": backend, "best": min(times), "mean": sum(times) / len(times),
         "warmup": warmup_s, "trials": times, "ok": bool(ok)}), flush=True)

    # release the GPUs promptly (also happens on process exit)
    try:
        from ray.data._internal.planner.gpu_sort import _SORTER_NAME
        ray.kill(ray.get_actor(_SORTER_NAME))
    except Exception:
        pass
    ray.shutdown()


# =========================================================================
# PARENT: launches each backend in a fresh subprocess and summarizes.
# =========================================================================
_NOISE = ("INFO ", "WARNING ", "warnings.warn", "FutureWarning", "(raylet)",
          "Tip:", "namespace=", "Started a local Ray")


def _interesting(line):
    return line.strip() and not any(n in line for n in _NOISE)


def run_backend_in_subprocess(backend, trials, gpus, quick):
    print(f"\n=== {TAG[backend]}: fresh process + fresh ray.init() "
          f"(released afterwards) ===", flush=True)
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
    parser.add_argument("--trials", type=int, default=3, help="timed runs per backend")
    parser.add_argument("--gpus", type=int, default=16)
    parser.add_argument("--quick", action="store_true", help="tiny 1 GiB smoke test")
    parser.add_argument("--backend", choices=BACKENDS,
                        help="(internal) run a single backend in this process")
    args = parser.parse_args()

    rows = (16 * 1024 * 1024) if args.quick else (1024 * 1024 * 1024)
    cols, blocks = 16, (64 if args.quick else 256)

    # Worker mode: run exactly one backend in this (fresh) process.
    if args.backend:
        run_worker(args.backend, rows, cols, blocks, args.trials, args.gpus)
        return

    # Parent mode: run each backend in its own fresh process, in turn.
    results = {}
    for backend in BACKENDS:
        results[backend] = run_backend_in_subprocess(backend, args.trials, args.gpus,
                                                      args.quick)
        time.sleep(5)  # let /dev/shm + GPUs fully release before the next fresh ray

    cpu_best = results["cpu"]["best"]
    print("\n" + "=" * 60)
    print(f"  dataset: {rows * cols * 4 / 2**30:.0f} GiB ({rows:,} rows x {cols} int32)")
    for b in BACKENDS:
        r = results[b]
        speed = f"{cpu_best / r['best']:5.1f}x vs CPU" if r["best"] else ""
        print(f"  {TAG[b]:<7} best {r['best']:8.3f} s   mean {r['mean']:8.3f} s   "
              f"warmup {r.get('warmup', 0):7.3f} s (not counted)   "
              f"sorted={'PASS' if r['ok'] else 'FAIL'}   {speed}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
