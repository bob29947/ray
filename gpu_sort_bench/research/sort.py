import os
import sys
import time
import subprocess
from collections import Counter

import numpy as np
import pyarrow as pa
import ray
from ray.data import DataContext


def gb(n):
    return f"{n / 2**30:.2f} GiB"


def print_ray_runtime(label):
    print(f"\n=== {label} ===")
    print(f"ray version: {ray.__version__}")
    print(f"python: {sys.version.split()[0]}")
    print(f"os cpu_count: {os.cpu_count()}")

    total = ray.cluster_resources()
    avail = ray.available_resources()
    nodes = ray.nodes()
    alive_nodes = [n for n in nodes if n.get("Alive")]

    print(f"ray nodes alive/total: {len(alive_nodes)}/{len(nodes)}")
    print(f"ray total CPUs: {total.get('CPU', 0)}")
    print(f"ray available CPUs now: {avail.get('CPU', 0)}")
    print(f"ray total GPUs: {total.get('GPU', 0)}")
    print(f"ray object store: {gb(total.get('object_store_memory', 0))}")

    for i, n in enumerate(alive_nodes):
        r = n.get("Resources", {})
        print(
            f"node {i}: "
            f"addr={n.get('NodeManagerAddress')} "
            f"CPU={r.get('CPU', 0)} "
            f"GPU={r.get('GPU', 0)} "
            f"object_store={gb(r.get('object_store_memory', 0))}"
        )

    try:
        from ray.util.state import list_workers

        workers = list_workers(limit=10000, detail=False)
        types = Counter(w.get("worker_type") or w.get("type") or "unknown" for w in workers)
        alive = sum(1 for w in workers if w.get("is_alive", True))
        print(f"ray workers listed: {len(workers)}, alive: {alive}, by_type: {dict(types)}")
    except Exception as e:
        print(f"ray workers listed: unavailable ({type(e).__name__}: {e})")


def print_dataset_runtime(label, ds, fallback_blocks=None):
    print(f"\n=== {label} ===")
    print(f"schema: {ds.schema()}")
    try:
        print(f"num_blocks: {ds.num_blocks()}")
    except Exception as e:
        if fallback_blocks is not None:
            print(f"num_blocks: {fallback_blocks} input refs")
        else:
            print(f"num_blocks: unavailable ({type(e).__name__}: {e})")


ray.init(object_store_memory=512 * 2**30)

ctx = DataContext.get_current()
ctx.enable_rich_progress_bars = True
ctx.use_ray_tqdm = False

ROWS = 1024 * 1024 * 1024   # 1 Gi rows
COLS = 16
BLOCKS = 256
ROWS_PER_BLOCK = ROWS // BLOCKS
RAW_BYTES = ROWS * COLS * 4

print_ray_runtime("ray runtime before data build")
print(f"\nplanned rows: {ROWS:,}")
print(f"planned cols: {COLS}")
print(f"planned input blocks: {BLOCKS}")
print(f"rows per block: {ROWS_PER_BLOCK:,}")
print(f"raw dataset size: {gb(RAW_BYTES)}")
print(f"raw block size: {gb(ROWS_PER_BLOCK * COLS * 4)}")

cols = [f"c{i}" for i in range(COLS)]
rng = np.random.default_rng(0)

refs = []
for i in range(BLOCKS):
    data = {"c0": rng.integers(0, 2**31 - 1, ROWS_PER_BLOCK, dtype=np.int32)}
    for c in cols[1:]:
        data[c] = np.zeros(ROWS_PER_BLOCK, dtype=np.int32)

    refs.append(ray.put(pa.table(data)))
    print(f"put block {i + 1}/{BLOCKS}", flush=True)

ds = ray.data.from_arrow_refs(refs)

print_ray_runtime("ray runtime after input materialized")
print_dataset_runtime("input dataset", ds, fallback_blocks=len(refs))

t0 = time.perf_counter()
sorted_ds = ds.sort("c0").materialize()
t1 = time.perf_counter()

print(f"\nsort+materialize seconds: {t1 - t0:.3f}")

print_ray_runtime("ray runtime after sort")
print_dataset_runtime("sorted dataset", sorted_ds)

print("\n=== ray data stats ===")
print(sorted_ds.stats())

print("\n=== ray memory stats ===")
subprocess.run(["ray", "memory", "--stats-only"], check=False)