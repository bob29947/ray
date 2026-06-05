# Experimental multi-GPU sort for Ray Data (`ds.sort(gpu=True)`)

An **opt-in, end-to-end multi-GPU sort backend** for `Dataset.sort()`, built on
**cuDF + rapidsmpf** over UCXX/NVLink. The CPU sort remains the default; the GPU
path is selected only with `ds.sort(..., gpu=True)` (or `backend="gpu"`).

## What it is

`ds.sort("key", gpu=True)` runs the whole sort across the local GPUs in one pass:

```
H2D    pull input Arrow blocks -> one cuDF table per rank (RAM -> VRAM) + key sample
SORT   local cudf sort -> range-partition by global quantile boundaries
       -> rapidsmpf Shuffler all-to-all over NVLink -> unpack + final local sort
D2H    sorted cuDF table per rank -> Arrow block back in the object store
```

It uses **N actors, one GPU each**, connected by a UCXX communicator (the RAPIDS
`rapidsmpf` Ray integration). Output partition `p` holds the p-th global key
range, so concatenating output blocks in partition order is globally sorted —
for any schema, key set, sort direction, and null placement.

The engine lives in `python/ray/data/_internal/planner/gpu_sort_general.py`
(the `"general"` backend), selected by `ds.sort(..., gpu=True)` /
`backend="gpu"` or `RAY_DATA_GPU_SORT=1`.

## Surface area (the whole diff)

* `Dataset.sort(..., gpu=None, backend=None)` — the user-facing opt-in
  (`dataset.py`).
* `Sort.gpu` logical-op field (`all_to_all_operator.py`).
* `generate_sort_fn(..., gpu=...)` + `_resolve_gpu_impl()` routing
  (`planner/sort.py`); `plan_all_to_all_op.py` forwards `op.gpu`.
* `planner/gpu_sort_general.py` — the GPU sort engine.

CPU sort control flow is **untouched** when `gpu` is unset/`False` (see
zero-regression below).

## Running the forked Ray (this worktree's source)

Each worktree has its own self-contained `.venv` with every dependency **except
Ray**, plus this worktree's Ray installed **editable** (`pip install -e .` with
`SKIP_BAZEL_BUILD=1`, reusing the prebuilt native core). So `import ray` resolves
to this worktree's `python/ray` from any directory — no `PYTHONPATH` needed:

```bash
.venv/bin/python -c "import ray, cudf, rapidsmpf; print(ray.__version__, ray.__file__)"
# 3.0.0.dev0  <worktree>/python/ray/__init__.py
```

## Tests (accuracy + zero-regression)

```bash
.venv/bin/python -m pytest python/ray/data/tests/test_gpu_sort.py -v
```

* **Policy / wiring / zero-regression** tests run on any host (no GPU): they pin
  `_resolve_gpu_impl` precedence, `sort(gpu=, backend=)` validation, and that the
  default `ds.sort()` is unchanged.
* **GPU correctness** tests (cuDF + rapidsmpf + a GPU) verify `ds.sort(gpu=True)`
  matches an **independent pandas oracle** (`na_position="last"`, Ray's default)
  across int / float+NaN / nulls / multi-key asc+desc / strings / datetime /
  empty / single-row / many-block inputs, and matches Ray's own pyarrow CPU sort
  order on null-free data. They **skip cleanly** when the RAPIDS stack is absent.

## Benchmark (reproduce the numbers)

```bash
.venv/bin/python gpu_sort_bench/benchmark.py --gpus 16 --trials 3        # 64 GiB
.venv/bin/python gpu_sort_bench/benchmark.py --quick                     # 1 GiB sanity
.venv/bin/python gpu_sort_bench/benchmark.py --backends cpu,gpu_general  # subset
```

Dataset: **1Gi rows (1,073,741,824) × 16 int32 = 64 GiB**, key `c0` = random
int32, seed 0. Each backend runs in a fresh process + `ray.init()`; a warmup is
measured but not counted; **best/median of N trials** (a one-off run-2
object-store high-water-mark spike hits every backend, so mean is not
representative). Every result is independently full-scan verified (row count /
key sum / min / max / global monotonicity).

### Results (16× Tesla V100-SXM3-32GB, cuDF 26.02, rapidsmpf 26.02; forked Ray 3.0.0.dev0)

Measured on the forked source Ray (best/median of 3 trials;
the run-2 spike that hits every backend is excluded by best/median):

| backend | best | median | FULL | h2d | shuffle | GPU-only | d2h | vs pyarrow | vs polars | sorted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| pyarrow (CPU) | 56.183 s | 118.43 s | — | — | — | — | — | 1.0× | — | PASS |
| polars (CPU)  | 61.800 s | 122.90 s | — | — | — | — | — | 0.9× | 1.0× | PASS |
| **gpu_general** | **6.514 s** | **6.710 s** | 6.199 s | 2.045 | 0.099 | **0.361 s** | 3.731 | **8.6×** | **9.5×** | **PASS** |

**The numbers hold.** The engine-intrinsic costs reproduce the prior study
(`research/MERGE_JUSTIFICATION.md`) essentially exactly — **GPU-only 0.361 s vs
0.372 s**, **rapidsmpf shuffle 0.099 s vs 0.110 s**. End-to-end wall is 6.5 s vs
the documented 5.6 s (host transfers ran a touch slower this session: H2D 2.0 s,
D2H 3.7 s — the documented largest-cost phase). The speedup **exceeds** the
documented 6.1×/4.9× here because this session's CPU baselines were slower than
the documented 34.7 s/27.5 s (vs the documented 45.691 s baseline this is
**7.0×**). Correctness PASS: all 1,073,741,824 rows, globally sorted, key
sum/min/max identical to the input.

Other documented figures this engine carries (see `research/`): datetime key
**6.4×**, strings multi-key **~58×**, rapidsmpf all-to-all **~818 GiB/s** in
isolation (2.2× a hand-rolled P2P loop).

## Zero regression on the default path

With `gpu` unset/`False`, the only added work is a single `getattr(op, "gpu",
None)` and the `_resolve_gpu_impl(...) is None` early return — the CPU sort path
is otherwise byte-for-byte unchanged. The in-tree `python/ray/data/tests/
test_sort.py` is the regression gate (same pass/fail with the feature present
and `gpu` unset).

## Dependencies (opt-in, lazy-imported)

cuDF, rmm, cupy, **rapidsmpf**, **ucxx** (CUDA 12.x). All heavy imports are
deferred until the GPU path actually runs, so CPU-only Ray installs are
unaffected. Install on the same RAPIDS release line, e.g.:

```bash
pip install --extra-index-url=https://pypi.nvidia.com \
  "cudf-cu12==26.2.*" "rmm-cu12==26.2.*" "cupy-cuda12x==14.*" \
  "rapidsmpf-cu12==26.2.*" "ucxx-cu12==0.48.*"
```

## Status / limitations

Experimental, single-node (one UCXX communicator over local NVLink; `num_gpus`
from `RAY_DATA_GPU_SORT_NUM_GPUS`, default 16). The pageable `cudf.to_arrow` D2H
is the largest remaining phase. See `gpu_sort_bench/research/MERGE_JUSTIFICATION.md`
for the full merge analysis, weaknesses, and the device-resident follow-up.
