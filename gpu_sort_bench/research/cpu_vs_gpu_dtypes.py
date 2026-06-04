"""Fair CPU-vs-GPU sort benchmark for NON-int32 dtypes at a size/shape that
shows the GPU win -- and documents WHY size + width matter.

Two datasets (``--dataset``), both at a configurable width (``--cols``):

    datetime  c0 = timestamp[s] (random dates ~2016..2036), ascending.        (B)
              c1..c{cols-1} = int32 payload (zeros).
    strings   s = string key, 8 words, ~12% NULL          (asc, nulls last)    (C)
              f = float64 key, ~12% NaN                    (desc)
              g = int32 key in [0,1000)                    (asc)
              id,pay = int64 payload; c5..c{cols-1} = int32 payload (zeros).

Why this script exists / why width matters
------------------------------------------
The end-to-end GPU win is *data-movement bound*: the CPU sort exchanges every
block through Ray's shared-memory object store (~2-5 GiB/s), while the GPU sort
exchanges over NVLink (~800 GiB/s) and only pays the host<->device transfer
once. The cost the GPU avoids (the object-store shuffle) scales with the TOTAL
BYTES moved == rows x columns. So:

    * a NARROW table (few columns) is cheap for the CPU to shuffle -> the GPU,
      which still pays a fixed H2D/D2H + actor overhead, only TIES
      (measured: 5-col strings ties at both 9 GiB and 37 GiB), whereas
    * a WIDE table (here 16 columns, matching the 64 GiB int32 headline) makes
      the CPU object-store shuffle the bottleneck -> the GPU wins clearly.

Same timing contract as the other benchmarks: warmup measured but NOT counted,
>=3 timed trials, report BEST and MEDIAN, for BOTH the GPU general engine and
Ray's CPU pyarrow sort. Correctness (streamed, memory-safe): row count
preserved + GPU output globally non-decreasing on the primary key + an
order-sensitive digest over the full key tuple identical for GPU and CPU
(i.e. the GPU global order == Ray's pyarrow order exactly).

Run:
    .venv/bin/python cpu_vs_gpu_dtypes.py --dataset datetime --rows $((512*1024*1024)) --cols 16
    .venv/bin/python cpu_vs_gpu_dtypes.py --dataset strings  --rows $((512*1024*1024)) --cols 16
    .venv/bin/python cpu_vs_gpu_dtypes.py --dataset strings --quick
"""

from __future__ import annotations

import argparse
import os
import shutil
import statistics
import sys
import time

os.environ.setdefault("RAY_DATA_GPU_SORT_IMPL", "general")
os.environ.setdefault("RAY_DATA_GPU_SORT", "0")

WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
VOCAB_SORTED = sorted(WORDS)              # index_in -> ascending lexical rank
NAN_SENT = 0xFFFFFFFFFFFFFFFF             # canonical bits for any NaN f value
NULL_SRANK = len(VOCAB_SORTED)           # s rank for NULL (sorts last, ascending)

KEYS = {"datetime": ["c0"], "strings": ["s", "f", "g"]}
DESC = {"datetime": [False], "strings": [False, True, False]}


def object_store_bytes():
    free = shutil.disk_usage("/dev/shm").free
    return int(min(320 * 2**30, free * 0.55))


def build_inputs(dataset, rows, blocks, cols, nulls=True, seed=0):
    """Build the dataset block-by-block; return (arrow_tables, total_bytes, n)."""
    import numpy as np
    import pyarrow as pa

    rng = np.random.default_rng(seed)
    words = np.array(WORDS, dtype=object)
    base = np.datetime64("2026-01-03", "s")
    span_s = 10 * 365 * 24 * 3600
    rpb = rows // blocks
    tables, total_bytes, next_id = [], 0, 0

    for _ in range(blocks):
        n = rpb
        if dataset == "datetime":
            off = rng.integers(-span_s, span_s, n, dtype=np.int64)
            data = {"c0": base + off.astype("timedelta64[s]")}
            for j in range(1, cols):
                data[f"c{j}"] = np.zeros(n, dtype=np.int32)
        else:
            s = words[rng.integers(0, len(words), n)].astype(object)
            if nulls:
                s[rng.random(n) < 0.12] = None       # ~12% null string key
            f = rng.normal(size=n).astype("float64")
            if nulls:
                f[rng.random(n) < 0.12] = np.nan      # ~12% NaN float key
            g = rng.integers(0, 1000, n).astype("int32")
            ids = np.arange(next_id, next_id + n, dtype="int64")
            next_id += n
            pay = (ids * 2654435761 % 1_000_003).astype("int64")
            data = {"s": pa.array(s, type=pa.string()), "f": f, "g": g,
                    "id": ids, "pay": pay}
            for j in range(5, cols):           # int32 payload up to `cols`
                data[f"c{j}"] = np.zeros(n, dtype=np.int32)
        tbl = pa.table(data)
        total_bytes += tbl.nbytes
        tables.append(tbl)
    return tables, total_bytes, rpb * blocks


def _key_components(dataset, tbl):
    """Return (primary_rank int64 [ascending], [component uint64 arrays]) for
    the sort key(s). Equal keys -> equal components, so two datasets with the
    same key-at-position order produce the same digest."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc

    if dataset == "datetime":
        c0 = tbl.column("c0").combine_chunks().to_numpy(zero_copy_only=False)
        r = c0.view(np.int64)
        return r.astype(np.int64), [r.view(np.uint64).astype(np.uint64)]

    s_col = tbl.column("s").combine_chunks()
    s_idx = pc.index_in(s_col, value_set=pa.array(VOCAB_SORTED))
    s_rank = pc.fill_null(s_idx, NULL_SRANK).to_numpy(zero_copy_only=False).astype(np.int64)
    f = tbl.column("f").combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64)
    g = tbl.column("g").combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64)
    fbits = f.view(np.uint64).copy()
    fbits[np.isnan(f)] = np.uint64(NAN_SENT)
    return s_rank, [s_rank.astype(np.uint64), fbits, g.astype(np.uint64)]


def check_sorted_and_digest(dataset, ds):
    """Stream once; return (rows, primary_monotonic, order_sensitive_digest)."""
    import numpy as np

    primes = [np.uint64(1000003), np.uint64(2654435761), np.uint64(40503)]
    rows_seen, prev, monotonic, acc = 0, None, True, np.uint64(0)
    with np.errstate(over="ignore"):
        for batch in ds.iter_batches(batch_size=8_000_000, batch_format="pyarrow"):
            if batch.num_rows == 0:
                continue
            primary, comps = _key_components(dataset, batch)
            n = primary.shape[0]
            if n >= 2 and not bool(np.all(primary[1:] >= primary[:-1])):
                monotonic = False
            if prev is not None and n >= 1 and primary[0] < prev:
                monotonic = False
            prev = int(primary[-1])
            idx = np.arange(rows_seen, rows_seen + n, dtype=np.uint64)
            rowval = np.zeros(n, dtype=np.uint64)
            for p, comp in zip(primes, comps):
                rowval = rowval + p * comp
            acc = acc + (rowval * (np.uint64(2) * idx + np.uint64(1))).sum(dtype=np.uint64)
            rows_seen += n
    return rows_seen, monotonic, int(acc)


def timed_sort(dataset, ds, gpu, trials, tag):
    keys, desc = KEYS[dataset], DESC[dataset]

    def do():
        # Use backend= (NOT gpu=). Dataset.sort maps gpu=False -> op_gpu=None,
        # which, with RAY_DATA_GPU_SORT_IMPL=general set in the env, still routes
        # to the GPU engine. Only backend="cpu" (-> op_gpu=False) forces Ray's
        # real pyarrow CPU sort; backend="gpu" forces the general GPU engine.
        return ds.sort(keys, descending=desc,
                       backend=("gpu" if gpu else "cpu")).materialize()

    print(f"[{tag}] warmup (not counted)...", flush=True)
    w0 = time.perf_counter()
    do()
    print(f"[{tag}] warmup: {time.perf_counter() - w0:8.3f} s", flush=True)

    times, last, phase_best = [], None, None
    for t in range(trials):
        t0 = time.perf_counter()
        sorted_ds = do()
        dt = time.perf_counter() - t0
        times.append(dt)
        ph = {}
        if gpu:
            try:
                from ray.data._internal.planner.gpu_sort_general import LAST_RUN_STATS
                ph = dict(LAST_RUN_STATS)
            except Exception:
                ph = {}
        if phase_best is None or dt <= min(times):
            phase_best = ph
        extra = ""
        if ph:
            extra = (f"  [full={ph.get('full_s', 0):.3f} h2d={ph.get('h2d_s', 0):.3f} "
                     f"gpu={ph.get('gpu_only_s', 0):.3f} shuf={ph.get('shuffle_s', 0):.3f} "
                     f"d2h={ph.get('d2h_s', 0):.3f}]")
        print(f"[{tag}] run {t + 1}/{trials}: {dt:8.3f} s{extra}", flush=True)
        if last is not None:
            del last
        last = sorted_ds
    return times, last, (phase_best or {})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["datetime", "strings"], required=True)
    p.add_argument("--rows", type=int, default=512 * 1024 * 1024)
    p.add_argument("--cols", type=int, default=16)
    p.add_argument("--blocks", type=int, default=256)
    p.add_argument("--gpus", type=int, default=16)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--no-nulls", action="store_true",
                   help="strings: omit NULL string / NaN float keys. Ray's CPU "
                        "sort RAISES on null string keys (np.searchsorted: "
                        "None < str), so a CPU timing baseline needs this.")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.rows, args.blocks, args.trials = 8 * 1024 * 1024, 32, 2
    if args.dataset == "strings" and args.cols < 5:
        args.cols = 5
    os.environ["RAY_DATA_GPU_SORT_NUM_GPUS"] = str(args.gpus)

    import logging
    import ray
    from ray.data import DataContext

    ray.init(object_store_memory=object_store_bytes())
    logging.getLogger("ray.data").setLevel(logging.WARNING)
    ctx = DataContext.get_current()
    ctx.enable_rich_progress_bars = False
    ctx.use_ray_tqdm = False

    tables, total_bytes, n_rows = build_inputs(
        args.dataset, args.rows, args.blocks, args.cols, nulls=not args.no_nulls)
    gib = total_bytes / 2**30
    per_row = total_bytes / max(1, n_rows)
    refs = [ray.put(t) for t in tables]
    del tables
    ds = ray.data.from_arrow_refs(refs)

    print(f"[{args.dataset}] {n_rows:,} rows x {args.cols} cols = {gib:.2f} GiB "
          f"({total_bytes:,} bytes, {per_row:.1f} B/row), {args.blocks} blocks, "
          f"{args.gpus} GPUs", flush=True)
    nulls_state = "no-nulls" if args.no_nulls else "with-nulls"
    print(f"[{args.dataset}] keys={KEYS[args.dataset]} "
          f"descending={DESC[args.dataset]} ({nulls_state})", flush=True)

    g_times, g_last, g_phase = timed_sort(args.dataset, ds, True, args.trials, "GPU")
    g_rows, g_monotonic, g_digest = check_sorted_and_digest(args.dataset, g_last)
    del g_last
    try:
        from ray.data._internal.planner.gpu_sort_general import kill_actor_pool
        kill_actor_pool(args.gpus)
    except Exception:
        pass
    time.sleep(3)

    c_times, c_last, _ = timed_sort(args.dataset, ds, False, args.trials, "CPU")
    c_rows, _cm, c_digest = check_sorted_and_digest(args.dataset, c_last)
    del c_last

    rows_ok = (g_rows == n_rows == c_rows)
    digest_ok = (g_digest == c_digest)
    ok = rows_ok and g_monotonic and digest_ok
    g_best, g_med = min(g_times), statistics.median(g_times)
    c_best, c_med = min(c_times), statistics.median(c_times)

    print("\n" + "=" * 84)
    print(f"  {args.dataset}  {n_rows:,} rows x {args.cols} cols = {gib:.2f} GiB  "
          f"({args.gpus} GPUs)")
    print(f"  keys={KEYS[args.dataset]} descending={DESC[args.dataset]}")
    print("-" * 84)
    print(f"  CPU (pyarrow)  best {c_best:8.3f} s   median {c_med:8.3f} s")
    print(f"  GPU (general)  best {g_best:8.3f} s   median {g_med:8.3f} s   "
          f"-> {c_best / g_best:.1f}x vs CPU (best)")
    if g_phase:
        print(f"                 FULL {g_phase.get('full_s', 0):.3f}s  "
              f"(h2d {g_phase.get('h2d_s', 0):.3f}, shuffle {g_phase.get('shuffle_s', 0):.3f}, "
              f"d2h {g_phase.get('d2h_s', 0):.3f})   GPU-only {g_phase.get('gpu_only_s', 0):.3f}s")
    print("-" * 84)
    print(f"  correctness: rows={g_rows:,}({'ok' if rows_ok else 'BAD'})  "
          f"GPU primary-key sorted={g_monotonic}  "
          f"GPU order==CPU(pyarrow)={'ok' if digest_ok else 'BAD'}  "
          f"-> {'PASS' if ok else 'FAIL'}")
    print("=" * 84 + "\n")

    try:
        from ray.data._internal.planner.gpu_sort_general import kill_actor_pool
        kill_actor_pool(args.gpus)
    except Exception:
        pass
    ray.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
