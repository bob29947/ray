# BTS cloud replication

This package is the self-contained AWS counterpart of the DGX BTS study. It
uses one retained 16-node fleet per arm and a completely fresh Ray runtime for
every observation. It never recreates EC2 instances between observations.

The prepared YAML deliberately omits `--object-store-memory`, so Ray chooses
the Plasma size on both CPU and GPU. Every Ray filesystem-spill directory is a
trial-owned path below `/mnt/nvme`; `/dev/shm` is used only by Plasma. GPU run
externalization, Ray Plasma-to-NVMe spill, and MPF/backend telemetry remain
separate in result JSON.

## Prepare

```bash
.venv/bin/python -m release.benchmarks.ray_data_gpu_sort.cloud.study prepare \
  --campaign-id bts-external-YYYYMMDD-a \
  --local-config .venv/gpu-sort-bts-cloud-artifacts/local-config.json \
  --dataset-manifest .venv/gpu-sort-bts-cloud-artifacts/dataset/controller-manifest.portable.json \
  --output-root .venv/gpu-sort-bts-cloud-artifacts/studies/bts-external-YYYYMMDD-a
```

Preparation verifies the stock Ray 2.55.1 wheel, creates a content-addressed
Ray Data/harness bundle, and renders launch YAML, one reset YAML per trial, and
the command plans. It does not contact AWS.

## Execute

First run `execute-arm` without `--execute`, using a disposable dry-run artifact
root such as `STUDY/dry-run/gpu`. The resulting lifecycle plan prints the exact
SHA that must be supplied to authorize billable execution. Use a new artifact
root for the billable run:

```bash
.venv/bin/python -m release.benchmarks.ray_data_gpu_sort.cloud.study execute-arm \
  --plan STUDY/study.json --arm gpu --artifact-root STUDY/executions/gpu \
  --execute --confirm-plan-sha SHA_FROM_DRY_RUN
```

The lifecycle performs one `ray up`, latches all 16 EC2 IDs, uses
`ray up --restart-only` before every trial, requires all Ray node IDs to change,
and synchronizes artifacts after every command. `ray down`, exact-tag EC2
termination, and three stable-empty polls occur once in `finally`.

If rsync-down fails after an allow-failure trend or natural observation, the
lifecycle can recover only that observation's self-hashed artifact from the
worker's final stdout sentinel. Recovery requires a prior successful cumulative
sync containing the inventory and dataset-staging receipts, validates the exact
trial identity and `results/<step>.json` path, and binds the dataset, input plan,
Ray wheel, harness, and Ray Data overlay to the staged receipts and prepared
bundle. It records both the nonzero rsync code and
`artifact_recovered_from_stdout: true`. Setup, wave selection, and the required
exact smoke remain fail-closed on any sync failure.

Only after the GPU teardown receipt says `verified_empty: true`, execute the CPU
arm using `STUDY/executions/cpu`. A failed 16-rank exact smoke stops the GPU arm
before performance work. Ordinary performance failures are retained as results
and do not prevent independent later cells.

## GPU-only full campaign

Use `gpu-only` to rerun the complete GPU matrix without provisioning a CPU
fleet. It uses the same exact smoke, both 2× wave screens, selected 2× second
observation, twelve six-cell trend observations, and two 2.45× observations as
the full study.

```bash
.venv/bin/python -m release.benchmarks.ray_data_gpu_sort.cloud.study prepare \
  --campaign-id bts-external-YYYYMMDD-gpu-only-a \
  --trial-mode gpu-only \
  --local-config .venv/gpu-sort-bts-cloud-artifacts/local-config.json \
  --dataset-manifest .venv/gpu-sort-bts-cloud-artifacts/dataset/controller-manifest.portable.json \
  --output-root GPU_ONLY_STUDY

.venv/bin/python -m release.benchmarks.ray_data_gpu_sort.cloud.study execute-arm \
  --plan GPU_ONLY_STUDY/study.json --arm gpu \
  --artifact-root GPU_ONLY_STUDY/dry-run/gpu

.venv/bin/python -m release.benchmarks.ray_data_gpu_sort.cloud.study execute-arm \
  --plan GPU_ONLY_STUDY/study.json --arm gpu \
  --artifact-root GPU_ONLY_STUDY/executions/gpu \
  --execute --confirm-plan-sha SHA_FROM_DRY_RUN
```

Every GPU result must prove the deterministic stratified CPU planner ran,
including its mode/version, fixed seed, sample counts and bytes, quota summary,
sample-plan/index/boundary digests, CPU planning subphases, and at most 1 MiB of
planning H2D. CPU dataset sort/merge, output fallback, and MPF host spill must
remain zero.

## GPU repair-only campaign

When a production fix needs only the GPU spill-path observations repeated, use
the explicit `gpu-repair` trial mode with a new campaign ID and output root. It
requires the original GPU fleet's unchanged verified-empty teardown receipt and
prepares only: exact smoke, both 2× wave screens, wave selection, selected 2×
r2, and 2.45× r1/r2.

For legacy executions, preparation binds that receipt to the sibling original
study, lifecycle plan, latched instance list, and lifecycle result. Campaign,
cluster, frozen topology, and all 16 terminated instance IDs must agree. Every
file hash is frozen in the repair plan and rechecked immediately before launch.
New lifecycle receipts also carry this campaign, cluster, plan, and instance-set
identity directly.

```bash
.venv/bin/python -m release.benchmarks.ray_data_gpu_sort.cloud.study prepare \
  --campaign-id bts-external-YYYYMMDD-repair-a \
  --trial-mode gpu-repair \
  --prior-gpu-teardown ORIGINAL_STUDY/executions/gpu/teardown.json \
  --local-config .venv/gpu-sort-bts-cloud-artifacts/local-config.json \
  --dataset-manifest .venv/gpu-sort-bts-cloud-artifacts/dataset/controller-manifest.portable.json \
  --output-root REPAIR_STUDY

.venv/bin/python -m release.benchmarks.ray_data_gpu_sort.cloud.study execute-arm \
  --plan REPAIR_STUDY/study.json --arm gpu \
  --artifact-root REPAIR_STUDY/dry-run/gpu

.venv/bin/python -m release.benchmarks.ray_data_gpu_sort.cloud.study execute-arm \
  --plan REPAIR_STUDY/study.json --arm gpu \
  --artifact-root REPAIR_STUDY/executions/gpu \
  --execute --confirm-plan-sha SHA_FROM_REPAIR_DRY_RUN
```

The simple lifecycle still performs one launch and one teardown. It restarts
Ray on the retained 16-node fleet before each observation and rsyncs the full
remote result tree after every command. The repair plan has no CPU arm and
cannot be used to launch one.

## GPU 2.45x capacity-only repair

Use `gpu-245x-repair` when an earlier repair already produced the authoritative
smoke, 2x wave selection, and 2x observations, but only the two 2.45x results
must be recovered. It binds the failed full repair's exact lifecycle/teardown
provenance, which in turn is bound to the original GPU teardown. The command
plan is deliberately small: inventory, immutable dataset staging, one exact
smoke, and fixed-wave 0.50 2.45x r1/r2. It still launches EC2 once, retains the
same 16 instances, restarts only Ray between observations, then tears the fleet
down once.

```bash
.venv/bin/python -m release.benchmarks.ray_data_gpu_sort.cloud.study prepare \
  --campaign-id bts-external-YYYYMMDD-245x-repair-a \
  --trial-mode gpu-245x-repair \
  --prior-gpu-teardown FULL_REPAIR_STUDY/executions/gpu/teardown.json \
  --local-config .venv/gpu-sort-bts-cloud-artifacts/local-config.json \
  --dataset-manifest .venv/gpu-sort-bts-cloud-artifacts/dataset/controller-manifest.portable.json \
  --output-root CAPACITY_REPAIR_STUDY

.venv/bin/python -m release.benchmarks.ray_data_gpu_sort.cloud.study execute-arm \
  --plan CAPACITY_REPAIR_STUDY/study.json --arm gpu \
  --artifact-root CAPACITY_REPAIR_STUDY/dry-run/gpu

.venv/bin/python -m release.benchmarks.ray_data_gpu_sort.cloud.study execute-arm \
  --plan CAPACITY_REPAIR_STUDY/study.json --arm gpu \
  --artifact-root CAPACITY_REPAIR_STUDY/executions/gpu \
  --execute --confirm-plan-sha SHA_FROM_CAPACITY_DRY_RUN
```

## CPU repair-only campaign

Use `cpu-repair` after the original CPU fleet has a verified-empty teardown
receipt. The repair plan binds and rechecks the original study, lifecycle,
instance-set, result, and teardown hashes in the same way as GPU repair. It
launches one retained 16-node `m5dn.4xlarge` fleet and runs only the six missing
observations: full, Origin string, Origin integer, route, natural 2×, and
natural 2.45×. Each observation still receives a fresh Ray runtime.

```bash
.venv/bin/python -m release.benchmarks.ray_data_gpu_sort.cloud.study prepare \
  --campaign-id bts-external-YYYYMMDD-cpu-repair-a \
  --trial-mode cpu-repair \
  --prior-cpu-teardown ORIGINAL_STUDY/executions/cpu/teardown.json \
  --local-config .venv/gpu-sort-bts-cloud-artifacts/local-config.json \
  --dataset-manifest .venv/gpu-sort-bts-cloud-artifacts/dataset/controller-manifest.portable.json \
  --output-root CPU_REPAIR_STUDY

.venv/bin/python -m release.benchmarks.ray_data_gpu_sort.cloud.study execute-arm \
  --plan CPU_REPAIR_STUDY/study.json --arm cpu \
  --artifact-root CPU_REPAIR_STUDY/dry-run/cpu

.venv/bin/python -m release.benchmarks.ray_data_gpu_sort.cloud.study execute-arm \
  --plan CPU_REPAIR_STUDY/study.json --arm cpu \
  --artifact-root CPU_REPAIR_STUDY/executions/cpu \
  --execute --confirm-plan-sha SHA_FROM_REPAIR_DRY_RUN
```

The CPU-repair plan has no GPU arm and does not require a sibling GPU teardown;
its prerequisite is the bound teardown of the prior CPU execution.

## Report

```bash
.venv/bin/python -m release.benchmarks.ray_data_gpu_sort.cloud.report \
  --study STUDY/study.json \
  --gpu-results STUDY/executions/gpu/remote-results \
  --cpu-results STUDY/executions/cpu/remote-results \
  --output STUDY/BTS_CLOUD_EXTERNAL_SORT.md
```

To make a repaired GPU campaign authoritative for tuning, 2×, and 2.45× while
retaining the original six trend cells and superseded GPU attempts, add:

```bash
  --gpu-repair-study REPAIR_STUDY/study.json \
  --gpu-repair-results REPAIR_STUDY/executions/gpu/remote-results
```

To replace only the six failed or missing CPU observations while preserving the
original valid narrow and core observations, add:

```bash
  --cpu-repair-study CPU_REPAIR_STUDY/study.json \
  --cpu-repair-results CPU_REPAIR_STUDY/executions/cpu/remote-results
```

GPU and CPU repair options may be supplied together. Each study/results pair is
required as a pair; repaired values become authoritative in the tables while
the superseded original attempts remain in the report JSON and Markdown.

When a `gpu-245x-repair` campaign replaces only the two capacity observations,
also supply:

```bash
  --gpu-245x-repair-study CAPACITY_REPAIR_STUDY/study.json \
  --gpu-245x-repair-results CAPACITY_REPAIR_STUDY/executions/gpu/remote-results
```

This overlay is accepted only together with the full GPU repair overlay whose
validated prefix supplies the smoke, selected wave, and 2x results.

The report refuses a repair overlay unless its study, completed lifecycle,
dataset staging receipt, exact trial identities, Ray wheel, Ray Data overlay,
harness hashes, input plans, cells, keys, and columns all reconcile with the
original study and the supplied artifacts.

For a `gpu-only` campaign, compare against the immutable finalized AWS report
instead of supplying a new CPU result tree:

```bash
.venv/bin/python -m release.benchmarks.ray_data_gpu_sort.cloud.gpu_only_report \
  --study GPU_ONLY_STUDY/study.json \
  --gpu-results GPU_ONLY_STUDY/executions/gpu/remote-results \
  --archived-report .venv/gpu-sort-cloud-artifacts/studies/bts-external-cloud-20260807-g/report/BTS_CLOUD_EXTERNAL_SORT_FINAL.json \
  --output GPU_ONLY_STUDY/report/BTS_CLOUD_STRATIFIED_GPU_ONLY.md
```

The archived report is accepted only at SHA-256
`96a08625bc6709a68e085f1c090614d18ad229635e5bfd1ba1b51412912cf534`.
Its CPU values are labeled cross-campaign directional denominators; no 2.45×
speedup is claimed because that archived CPU observation did not complete.
