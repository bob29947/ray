# Ray Data spillable GPU sort: DGX benchmark

This is the compact, local-only harness for the 16-V100 DGX. It reuses the
normalized BTS corpus at
`/raid/spark-team/bobbwang/datasets/bts-airline-on-time`; it does not download
data, create cloud resources, or include dataset preparation code.

Run everything from the project virtual environment:

```bash
cd /raid/spark-team/bobbwang/projects/ray-data-gpu-external-sort
.venv/bin/python -m release.benchmarks.ray_data_gpu_sort.runner all
```

The stages can also be run separately:

```bash
.venv/bin/python -m release.benchmarks.ray_data_gpu_sort.runner smoke
.venv/bin/python -m release.benchmarks.ray_data_gpu_sort.runner trends
.venv/bin/python -m release.benchmarks.ray_data_gpu_sort.runner gpu-trends
.venv/bin/python -m release.benchmarks.ray_data_gpu_sort.runner spill
.venv/bin/python -m release.benchmarks.ray_data_gpu_sort.runner large
.venv/bin/python -m release.benchmarks.ray_data_gpu_sort.runner report
```

`smoke` compares exact schema/order/values for PyArrow, resident GPU, and a
memory-constrained GPU run. `trends` runs the exact prior six-cell cohort with
one CPU and two GPU observations per cell. `gpu-trends` refreshes only those
twelve GPU observations after a GPU implementation change, preserving the
existing default-Ray CPU baselines. `spill` reuses the full-cell GPU baseline
and adds two observations at 0.75B, 0.50B, and max(4 GiB, 0.25B), where B is
the measured maximum per-rank resident allocation. At most one uninformative
spill point is replaced. `large` runs the 3x cohort sorted by `Origin`, checks
global order and the exact `row_id` sum without a CPU sort, and allows Ray to
spill Plasma objects to RAID.

The PyArrow worker uses literal Ray defaults for sizing: it passes no
`num_cpus`, `num_gpus`, `object_store_memory`, or `_system_config`. On this DGX,
Ray selected 96 CPUs and a 200,000,000,000-byte (186.3-GiB) object store; each
trial records the selected value. GPU trials explicitly request 16 GPUs and
the study-specific Plasma size.

Plasma is RAM-backed at `/dev/shm/rgs/<token>/plasma`. GPU-externalized runs
are first sealed there as Ray ObjectRefs. Actual filesystem spill is written
to `.venv/gpu-sort-external-runtime/<trial>/ray-spill` on `/raid`, matching a
cluster's local-disk/NVMe spill tier; workers reject any filesystem-spill path
under `/dev/shm`. Restricted 64-GiB trials use 132 GiB of Plasma and the
192-GiB proof uses 256 GiB. Cumulative Ray Core spill/restore counters retain
the disk-traffic total even after restored spill files are deleted.

Every worker starts and stops a fresh local Ray runtime. Parquet projection and
Plasma materialization happen before the sort timer; it stops after the output
is materialized and sealed. Before each GPU worker launch, the runner waits
outside the measured interval for all 16 GPUs to fall below 1 GiB used, so a
simulated cold run cannot inherit retiring CUDA contexts. Runtime data is
isolated under `.venv` and removed after each worker.

The minimal local validation and report refresh are:

```bash
.venv/bin/ruff check python/ray/data/_internal/gpu_sort \
  python/ray/data/tests/test_gpu_sort_engine.py \
  release/benchmarks/ray_data_gpu_sort
.venv/bin/python -m release.benchmarks.ray_data_gpu_sort.runner smoke
.venv/bin/python -m release.benchmarks.ray_data_gpu_sort.runner report
```

Results are written to:

```text
.venv/gpu-sort-external-artifacts/
  REPORT.md
  study.json
  spill-study.json
  trials/
  logs/
```

The GPU backend must honor `RAY_DATA_GPU_SORT_MEMORY_BUDGET_BYTES` and expose
`ray.data._internal.gpu_sort.operator.get_last_run_stats()`. Missing telemetry,
CPU sort/merge fallback, wrong rows/schema, or an unsuccessful exact smoke
invalidates the run.
