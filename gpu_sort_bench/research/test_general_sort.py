"""Generality correctness test for the cuDF + rapidsmpf GPU sort backend.

The hand-tuned engine (`gpu_sort.py`) only sorts a dense int32 matrix on a
single ascending key with no nulls. The general engine (`gpu_sort_general.py`)
must instead handle real columnar data. This test proves it produces the
correct GLOBAL order for:

    * string + numeric columns of different dtypes,
    * MULTIPLE sort keys with mixed ascending / descending,
    * NULLs in both a string key and a float key (na_position="last").

It runs the public API a user would write:

    ds.sort(["s", "f", "g"], descending=[False, True, False], backend="gpu")

and checks, against an INDEPENDENT pandas reference (NOT Ray's own sort), that

    1. the emitted key sequence equals the reference sorted key sequence
       (i.e. the data is globally ordered across all output blocks), and
    2. every row's payload travels with its keys (no rows lost / scrambled):
       the full result, re-keyed by a unique id, matches the input exactly.

The oracle is pandas (`na_position="last"`), not Ray's CPU sort, because:
  * a plain `ds.sort()` resolves to the GPU engine when RAY_DATA_GPU_SORT_IMPL
    is set (so it would compare the engine to itself), and
  * Ray's pyarrow CPU sort RAISES on null string keys (`None < str` in its
    np.searchsorted boundary search), so it cannot sort this input at all.
Float nulls are real Arrow nulls (not NaN), since cuDF orders NaN as the largest
float (it would sort first under a descending key, unlike a true null).

Run:
    .venv/bin/python test_general_sort.py
    .venv/bin/python test_general_sort.py --rows 4000000 --blocks 32 --gpus 16
"""

from __future__ import annotations

import argparse
import math
import os
import sys

# Select the general backend explicitly so a stray RAY_DATA_GPU_SORT_IMPL in the
# environment can't silently route this generality test to the tuned engine
# (which cannot represent strings/nulls). `gpu=True` already defaults to general.
os.environ.setdefault("RAY_DATA_GPU_SORT_IMPL", "general")


def _norm(v):
    """Canonicalize a cell for null-aware, dtype-tolerant comparison."""
    if v is None:
        return None
    if isinstance(v, float):
        if math.isnan(v):
            return None
        if v.is_integer():
            return int(v)
    return v


def _col(df, name):
    return [_norm(v) for v in df[name].tolist()]


def _rows(df, cols):
    return list(zip(*[_col(df, c) for c in cols]))


def build_inputs(rows, blocks, seed=0):
    import numpy as np
    import pyarrow as pa

    rng = np.random.default_rng(seed)
    words = np.array(["alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
                      "golf", "hotel"])
    rpb = rows // blocks
    refs_data = []
    next_id = 0
    for _ in range(blocks):
        n = rpb
        s = words[rng.integers(0, len(words), n)].astype(object)
        # ~12% nulls in the string key
        s[rng.random(n) < 0.12] = None
        f = rng.normal(size=n).astype("float64")
        # ~12% nulls in the float key. Use a real Arrow NULL (validity bitmap),
        # not NaN: cuDF/pylibcudf treat NaN as the largest float (so it would sort
        # FIRST for a descending key), whereas a true null obeys na_position. Using
        # a null keeps "nulls last" unambiguous and identical between the engine and
        # the pandas oracle.
        f_null = rng.random(n) < 0.12
        g = rng.integers(0, 1000, n).astype("int32")  # a third (numeric) key
        ids = np.arange(next_id, next_id + n, dtype="int64")
        next_id += n
        pay = (ids * 2654435761 % 1_000_003).astype("int64")  # payload to track
        refs_data.append(pa.table({"s": pa.array(s, type=pa.string()),
                                    "f": pa.array(f, mask=f_null),
                                    "g": g, "id": ids, "pay": pay}))
    return refs_data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=4_000_000)
    p.add_argument("--blocks", type=int, default=32)
    p.add_argument("--gpus", type=int, default=16)
    args = p.parse_args()

    os.environ["RAY_DATA_GPU_SORT_NUM_GPUS"] = str(args.gpus)

    import numpy as np  # noqa: F401
    import pyarrow as pa
    import ray
    from ray.data import DataContext

    ray.init(object_store_memory=32 * 2**30)
    ctx = DataContext.get_current()
    ctx.enable_rich_progress_bars = False
    ctx.use_ray_tqdm = False

    keys = ["s", "f", "g"]
    descending = [False, True, False]
    ascending = [not d for d in descending]

    tables = build_inputs(args.rows, args.blocks)
    full = pa.concat_tables(tables).to_pandas()
    refs = [ray.put(t) for t in tables]
    ds = ray.data.from_arrow_refs(refs)

    print(f"[general-test] {len(full):,} rows x {full.shape[1]} cols, "
          f"keys={keys} descending={descending}, {args.gpus} GPUs", flush=True)

    def concat(ds_):
        return pa.concat_tables(ray.get(ds_.to_arrow_refs())).to_pandas()

    # ---- the public API a user writes (the GPU path under test) ----
    # Use backend="gpu" to force the general engine explicitly.
    gpu_out = concat(ds.sort(keys, descending=descending, backend="gpu").materialize())

    # ---- ORACLE: an INDEPENDENT pandas sort, NOT Ray's CPU sort. Two reasons:
    #   (a) with RAY_DATA_GPU_SORT_IMPL set, a plain ds.sort() (gpu unset) resolves
    #       to the GPU engine -- so it cannot serve as an independent oracle; and
    #   (b) Ray's pyarrow CPU sort RAISES on null string keys (it does
    #       np.searchsorted on the string column -> "None < str"), so it cannot
    #       sort this input at all.
    # pandas' na_position="last" matches the engine's na_position="last" (nulls
    # sort last regardless of per-key direction), so the global key order must match.
    oracle = full.sort_values(by=keys, ascending=ascending, na_position="last",
                              kind="stable").reset_index(drop=True)

    # 1) GLOBAL ORDER: emitted key sequence == oracle key sequence.
    order_ok = _rows(gpu_out, keys) == _rows(oracle, keys)

    # 2) ROW INTEGRITY: re-key by the unique id and compare every column,
    #    proving each payload stayed attached to its keys and nothing was lost.
    rows_ok = len(gpu_out) == len(full)
    out_by_id = gpu_out.sort_values("id").reset_index(drop=True)
    in_by_id = full.sort_values("id").reset_index(drop=True)
    integrity_ok = rows_ok and all(
        _col(out_by_id, c) == _col(in_by_id, c) for c in full.columns
    )

    ok = order_ok and integrity_ok
    print(f"[general-test] rows: gpu_out={len(gpu_out):,} in={len(full):,} "
          f"({'ok' if rows_ok else 'BAD'})", flush=True)
    print(f"[general-test] global key order == independent pandas oracle: "
          f"{'PASS' if order_ok else 'FAIL'}", flush=True)
    print(f"[general-test] payload/row integrity preserved: "
          f"{'PASS' if integrity_ok else 'FAIL'}", flush=True)
    print(f"\n[general-test] RESULT: {'PASS' if ok else 'FAIL'}", flush=True)

    try:
        from ray.data._internal.planner.gpu_sort_general import kill_actor_pool
        kill_actor_pool(args.gpus)
    except Exception:
        pass
    ray.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
