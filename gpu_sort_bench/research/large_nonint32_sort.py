"""Large (>=16 GiB) NON-int32 end-to-end GPU sort -- generality AT SCALE.

The small ``test_general_sort.py`` proves the general engine is a faithful
drop-in across dtypes/keys/nulls on 4M rows. This proves it ALSO holds at scale
on a non-int32 key. Two key types are supported (``--key``):

    datetime  c0 = timestamp[s] (random calendar dates centered on 2026-01-03,
              spanning ~2016..2036) -- a real "sort by time" workload (DEFAULT).
    float64   c0 = random float64 spread over many orders of magnitude.

Either way it's a ~17 GiB dataset sorted ascending end-to-end with
``ds.sort("c0", gpu=True)``.

Verification (the GPU result must match Ray's pyarrow sort):
  * ORACLE = ``numpy.sort`` of the input keys. For a single ascending key with no
    nulls this is, element-for-element, exactly what Ray's pyarrow sort emits
    (timestamps sort by their int64 representation, same as pyarrow).
  * GPU: stream the sorted output in block order and assert it equals the oracle
    element-by-element (plus globally monotonic, row count / key sum / min / max).
  * CPU (pyarrow, Ray's default): same check against the SAME oracle -- so GPU
    and Ray-pyarrow are proven identical (both equal the oracle exactly).

Run:
    .venv/bin/python large_nonint32_sort.py                       # datetime key, ~17 GiB
    .venv/bin/python large_nonint32_sort.py --key float64
    .venv/bin/python large_nonint32_sort.py --rows $((256*1024*1024)) --gpus 16
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

os.environ.setdefault("RAY_DATA_GPU_SORT_IMPL", "general")
os.environ.setdefault("RAY_DATA_GPU_SORT", "0")


def object_store_bytes():
    free = shutil.disk_usage("/dev/shm").free
    return int(min(320 * 2**30, free * 0.55))


def _as_repr(arr):
    """Canonical int/float representation for ordering checks (datetime -> int64
    nanos/seconds-since-epoch; everything else unchanged)."""
    if arr.dtype.kind == "M":
        return arr.view("int64")
    return arr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=256 * 1024 * 1024,
                   help="total rows (default 256Mi -> ~17 GiB)")
    p.add_argument("--cols", type=int, default=16)
    p.add_argument("--blocks", type=int, default=256)
    p.add_argument("--gpus", type=int, default=16)
    p.add_argument("--key", choices=["datetime", "float64"], default="datetime")
    args = p.parse_args()
    os.environ["RAY_DATA_GPU_SORT_NUM_GPUS"] = str(args.gpus)

    import logging

    import numpy as np
    import pyarrow as pa
    import ray
    from ray.data import DataContext

    # key (8B: timestamp[s] or float64) + (cols-1) int32 (4B each)
    row_bytes = 8 + (args.cols - 1) * 4
    raw_gib = args.rows * row_bytes / 2**30
    rpb = args.rows // args.blocks

    ray.init(object_store_memory=object_store_bytes())
    logging.getLogger("ray.data").setLevel(logging.WARNING)
    ctx = DataContext.get_current()
    ctx.enable_rich_progress_bars = False
    ctx.use_ray_tqdm = False

    # ---- build dataset; keep ALL keys on host to form the oracle --------- #
    rng = np.random.default_rng(0)
    base = np.datetime64("2026-01-03", "s")
    span_s = 10 * 365 * 24 * 3600          # +/- ~10 years around 2026-01-03

    def make_key(n):
        if args.key == "datetime":
            off = rng.integers(-span_s, span_s, n, dtype=np.int64)
            return (base + off.astype("timedelta64[s]"))      # datetime64[s]
        return (rng.standard_normal(n) * 1e6).astype(np.float64)

    sample = make_key(4)  # to fix the host oracle dtype
    keys_all = np.empty(args.rows, dtype=sample.dtype)
    refs = []
    pos = 0
    for _ in range(args.blocks):
        c0 = make_key(rpb)
        keys_all[pos:pos + rpb] = c0
        pos += rpb
        data = {"c0": c0}
        for j in range(1, args.cols):
            data[f"c{j}"] = np.zeros(rpb, dtype=np.int32)
        refs.append(ray.put(pa.table(data)))
    n_rows = pos
    keys_all = keys_all[:n_rows]

    rep = _as_repr(keys_all)
    in_sum = int(rep.astype(np.int64).sum()) if rep.dtype.kind != "f" else float(rep.sum())
    in_min, in_max = rep.min(), rep.max()
    oracle = np.sort(keys_all, kind="stable")  # == pyarrow ascending order
    oracle_rep = _as_repr(oracle)
    sample_dates = oracle[:: max(1, n_rows // 6)][:6]
    del keys_all

    ds = ray.data.from_arrow_refs(refs)
    print(f"[large] {n_rows:,} rows x {args.cols} cols  (~{raw_gib:.1f} GiB), "
          f"key c0={args.key} ascending, {args.gpus} GPUs", flush=True)
    if args.key == "datetime":
        print(f"[large] sample sorted dates: "
              f"{', '.join(str(d) for d in sample_dates)}", flush=True)

    def stream_check(sorted_ds, label):
        """Stream output in block order; compare to the oracle element-wise."""
        prev, idx, rows_seen = None, 0, 0
        ksum = 0 if oracle_rep.dtype.kind != "f" else 0.0
        kmin, kmax, monotonic, matches = None, None, True, True
        for batch in sorted_ds.iter_batches(batch_size=8_000_000,
                                            batch_format="numpy"):
            c0 = batch["c0"]
            if c0.size == 0:
                continue
            r = _as_repr(c0)
            if not bool(np.all(r[1:] >= r[:-1])):
                monotonic = False
            if prev is not None and r[0] < prev:
                monotonic = False
            prev = r[-1]
            seg = oracle_rep[idx:idx + r.size]
            if seg.size != r.size or not np.array_equal(r, seg):
                matches = False
            idx += int(r.size)
            rows_seen += int(r.size)
            if oracle_rep.dtype.kind == "f":
                ksum += float(r.sum())
            else:
                ksum += int(r.astype(np.int64).sum())
            lo, hi = r.min(), r.max()
            kmin = lo if kmin is None else min(kmin, lo)
            kmax = hi if kmax is None else max(kmax, hi)
        rows_ok = rows_seen == n_rows and idx == n_rows
        if oracle_rep.dtype.kind == "f":
            sum_ok = abs(ksum - in_sum) <= max(1.0, abs(in_sum) * 1e-9)
        else:
            sum_ok = ksum == in_sum
        minmax_ok = (kmin == in_min and kmax == in_max)
        ok = monotonic and rows_ok and sum_ok and minmax_ok and matches
        print(f"[large:{label}] rows={rows_seen:,}({'ok' if rows_ok else 'BAD'})  "
              f"monotonic={monotonic}  ==oracle(Ray-pyarrow order)={matches}  "
              f"sum={'ok' if sum_ok else 'BAD'}  min/max={'ok' if minmax_ok else 'BAD'}  "
              f"-> {'PASS' if ok else 'FAIL'}", flush=True)
        return ok

    # ---- GPU sort (the path under test) --------------------------------- #
    print("[large] GPU warmup (not counted)...", flush=True)
    ds.sort("c0", backend="gpu").materialize()
    t0 = time.perf_counter()
    gpu_sorted = ds.sort("c0", backend="gpu").materialize()
    gpu_wall = time.perf_counter() - t0
    try:
        from ray.data._internal.planner.gpu_sort_general import LAST_RUN_STATS
        ph = dict(LAST_RUN_STATS)
    except Exception:
        ph = {}
    print(f"[large] GPU ds.sort('c0', gpu=True).materialize(): {gpu_wall:.3f} s "
          f"(full={ph.get('full_s', 0):.3f} h2d={ph.get('h2d_s', 0):.3f} "
          f"gpu_only={ph.get('gpu_only_s', 0):.3f} shuffle={ph.get('shuffle_s', 0):.3f} "
          f"d2h={ph.get('d2h_s', 0):.3f})", flush=True)
    gpu_ok = stream_check(gpu_sorted, "GPU")
    del gpu_sorted

    # release the GPU actor pool before the CPU oracle run
    try:
        from ray.data._internal.planner.gpu_sort_general import kill_actor_pool
        kill_actor_pool(args.gpus)
    except Exception:
        pass
    time.sleep(3)

    # ---- Ray pyarrow CPU sort (oracle confirmation) --------------------- #
    # MUST pass backend="cpu". With RAY_DATA_GPU_SORT_IMPL=general set in the env
    # (above), a plain ds.sort() / gpu=False resolves to op_gpu=None -> the GPU
    # engine; only backend="cpu" forces Ray's real pyarrow CPU sort. A CPU warmup
    # is run (not counted) so the timed CPU number excludes worker startup, matching
    # the GPU side's warmup.
    print("[large] Ray pyarrow CPU sort warmup (not counted)...", flush=True)
    ds.sort("c0", backend="cpu").materialize()
    print("[large] Ray pyarrow CPU sort (confirms oracle == Ray default)...",
          flush=True)
    t0 = time.perf_counter()
    cpu_sorted = ds.sort("c0", backend="cpu").materialize()
    cpu_wall = time.perf_counter() - t0
    print(f"[large] CPU ds.sort('c0').materialize(): {cpu_wall:.3f} s", flush=True)
    cpu_ok = stream_check(cpu_sorted, "CPU")

    ok = gpu_ok and cpu_ok
    print(f"\n[large] key={args.key}, ~{raw_gib:.1f} GiB:  "
          f"GPU==Ray-pyarrow sort -> {'PASS' if ok else 'FAIL'}  "
          f"(GPU {gpu_wall:.2f}s vs CPU {cpu_wall:.2f}s = "
          f"{cpu_wall / gpu_wall:.1f}x)", flush=True)

    try:
        from ray.data._internal.planner.gpu_sort_general import kill_actor_pool
        kill_actor_pool(args.gpus)
    except Exception:
        pass
    ray.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
