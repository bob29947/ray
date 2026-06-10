# CriteoPrivateAd CPU preprocessing E2E benchmark

One benchmark, two execution targets — **no separate local/cloud copies**:

1. **Locally** on the DGX/dev box with the worktree `.venv`.
2. On an **AWS CPU Ray cluster** launched from this worktree (`cluster/ray-cpu.yaml`).

The same `cpu_pipeline.py` + `criteo.py` drive both. The execution target is
selected entirely by flags:

| Flag | Local | Cloud |
|------|-------|-------|
| `--ray-address` | `local` (starts a local Ray) | `auto` (attaches to the cluster) |
| `--data-root` | a local path | an `s3://…` URI |
| `--out` | a local dir | an `s3://…` URI |

Pipeline stages (each materialized RAM→RAM and timed): `read, prep, sort,
impute, encode, scale, write` (+ `TOTAL`). `sort/encode/scale` are flagged as
GPU-acceleration targets for a later comparison; this baseline runs all on CPU.

> The worktree on this box is `/bobbwang/projects/clean/gpu_recommender_preprocess`.
> Run local commands from that directory.

---

## 1. Local (DGX `.venv`)

Single day (the canonical local command):

```bash
cd /bobbwang/projects/clean/gpu_recommender_preprocess

.venv/bin/python benchmarks/criteo/cpu_pipeline.py \
  --days 1 \
  --data-root /bobbwang/datasets/CriteoPrivateAd/data \
  --out benchmarks/criteo/data/criteo_days1_cpu_baseline \
  --ray-address local
```

Quick smoke (row-capped, RAM→RAM, no write):

```bash
.venv/bin/python benchmarks/criteo/cpu_pipeline.py \
  --days 1 --rows 100000 \
  --data-root /bobbwang/datasets/CriteoPrivateAd/data \
  --ray-address local \
  --no-write
```

`--days` accepts `1`, a range like `1-5` / `1-30`, or `all`. Omitting
`--data-root` keeps the local default (`/bobbwang/datasets/CriteoPrivateAd/data`);
omitting `--ray-address` defaults to `local` — so prior local behavior is
unchanged.

---

## 2. AWS CPU Ray cluster

The cloud cluster installs the **Ray wheel built from this worktree** (mounted
from `./dist`) — it does not mount the Ray source tree. Run all commands from
the worktree root so the relative `file_mounts` (`dist`, `benchmarks`) resolve.

### Build the wheel

```bash
./build-wheel.sh 3.10 ./dist
```

### Launch the cluster

```bash
ray up cluster/ray-cpu.yaml -y
```

### Sanity checks

```bash
ray exec cluster/ray-cpu.yaml 'python -c "import ray; print(ray.__version__); print(ray.__file__)"'
ray exec cluster/ray-cpu.yaml 'python -c "import ray; ray.init(address=\"auto\"); print(ray.cluster_resources())"'
```

### Run the benchmark on the cluster

> Replace the `s3://TODO_REQUIRED/...` placeholders with your real private
> CriteoPrivateAd dataset/output buckets. They are intentionally **not** filled
> in here.

Smoke (row-capped; skips S3 sort-verify while you confirm S3 wiring):

```bash
ray exec cluster/ray-cpu.yaml \
  'cd /home/ray/benchmarks/criteo && python cpu_pipeline.py \
    --days 1 --rows 100000 \
    --data-root s3://TODO_REQUIRED/criteo-private-ad/data \
    --out s3://TODO_REQUIRED/criteo-private-ad/outputs/cpu_smoke_days1 \
    --ray-address auto \
    --skip-saved-sort-verify'
```

Full single day:

```bash
ray exec cluster/ray-cpu.yaml \
  'cd /home/ray/benchmarks/criteo && python cpu_pipeline.py \
    --days 1 \
    --data-root s3://TODO_REQUIRED/criteo-private-ad/data \
    --out s3://TODO_REQUIRED/criteo-private-ad/outputs/cpu_baseline_days1 \
    --ray-address auto'
```

### Tear down

```bash
ray down cluster/ray-cpu.yaml -y
```

---

## Notes

- **S3 paths** are `s3://TODO_REQUIRED/...` placeholders. Fill in the real
  private dataset (`--data-root`) and output (`--out`) buckets before running.
- **S3 output safety**: an existing `s3://` `--out` prefix is an error (never a
  silent overwrite). Pass `--overwrite` to delete it first, or choose a new
  `--out`. Local output always overwrites (current behavior).
- **`--skip-saved-sort-verify`** skips only the saved-output global-sortedness
  read-back (an escape hatch for first S3 smoke runs); the write still happens
  unless `--no-write` is also given.
- **Object store**: `--object-store-gb` applies only to `--ray-address local`.
  When attached to a cluster (`auto`) the cluster owns the object store.
- **Cluster scaling**: total nodes = 1 head + N workers. For X total nodes set
  top-level `max_workers` and `ray.worker.default` `min_workers`/`max_workers`
  all to `X - 1` (see `cluster/ray-cpu.yaml`).
- **Volume**: the YAML uses a 300 GiB gp3 root; for full multi-day runs or heavy
  shuffle/write spill, raise it (e.g. 1000 GiB gp3, 80000 IOPS, 2000 MB/s).
- Generated outputs under `benchmarks/criteo/data/` and the built root `dist/`
  wheel are git-ignored — do not commit them.
```

