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
GPU-acceleration targets; `cpu_pipeline.py` runs all on CPU.

`gpu_pipeline.py` is the GPU counterpart: identical dataset, roles, sort key,
verification and manifest, but the `impute → encode → scale` steps are composed
into a single `Chain(..., backend="gpu")` and run as **one device-resident
pass** (each block crosses PCIe once: H2D → impute+encode+scale on the resident
cuDF frame → D2H). Sort stays on CPU this iteration (cluster-aware GPU sort is a
documented follow-up). The fused stage is timed as one unit so it compares
directly against the CPU baseline's `impute + encode + scale` subtotal. See
section 3 below and `cluster/ray-gpu.yaml`.

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

### Stage the dataset to S3 (one-time)

The benchmark reads from `s3://bobbwang-ray-e2e-criteo/criteo-private-ad/data`
(us-west-2). Upload the local dataset there once. **Run this from a shell with
AWS write access** — the agent/automation sandbox has read-only network egress
(GET allowed, PUT/POST blocked), so it cannot create buckets or upload:

```bash
aws s3 mb s3://bobbwang-ray-e2e-criteo --region us-west-2
aws s3api put-public-access-block --bucket bobbwang-ray-e2e-criteo \
  --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
aws s3 sync /bobbwang/datasets/CriteoPrivateAd/data \
  s3://bobbwang-ray-e2e-criteo/criteo-private-ad/data
# verify:
aws s3 ls s3://bobbwang-ray-e2e-criteo/criteo-private-ad/data/
```

### Run the benchmark on the cluster

> Outputs are written under
> `s3://bobbwang-ray-e2e-criteo/criteo-private-ad/outputs/...`. Make sure the
> dataset has been staged to S3 (see above) first.

Smoke (row-capped; skips S3 sort-verify while you confirm S3 wiring):

```bash
ray exec cluster/ray-cpu.yaml \
  'cd /home/ray/benchmarks/criteo && python cpu_pipeline.py \
    --days 1 --rows 100000 \
    --data-root s3://bobbwang-ray-e2e-criteo/criteo-private-ad/data \
    --out s3://bobbwang-ray-e2e-criteo/criteo-private-ad/outputs/cpu_smoke_days1 \
    --ray-address auto \
    --skip-saved-sort-verify'
```

Full single day:

```bash
ray exec cluster/ray-cpu.yaml \
  'cd /home/ray/benchmarks/criteo && python cpu_pipeline.py \
    --days 1 \
    --data-root s3://bobbwang-ray-e2e-criteo/criteo-private-ad/data \
    --out s3://bobbwang-ray-e2e-criteo/criteo-private-ad/outputs/cpu_baseline_days1 \
    --ray-address auto'
```

### Tear down

```bash
ray down cluster/ray-cpu.yaml -y
```

---

## 3. GPU pipeline (fused, device-resident)

`gpu_pipeline.py` takes the same flags as `cpu_pipeline.py` (plus a few GPU
knobs) and produces an output that matches the CPU baseline. The only change is
that `impute + encode + scale` run as one fused GPU pass. Requires a GPU with
the RAPIDS stack (cudf/rmm/cupy); with no GPU it transparently falls back to
the CPU path.

Local (single day, on a GPU box):

```bash
.venv/bin/python benchmarks/criteo/gpu_pipeline.py \
  --days 1 \
  --data-root /bobbwang/datasets/CriteoPrivateAd/data \
  --out benchmarks/criteo/data/criteo_days1_gpu \
  --ray-address local
```

Flags shared with the CPU pipeline worth calling out:

- `--feature-set lean|wide` — `lean` (default) is the inference-realistic recipe
  (drops the 80 `features_not_available_*`). `wide` keeps that bucket as features
  (≈2.5x more feature columns) for a much wider fused frame. Run the CPU baseline
  with the **same** `--feature-set` for a fair compare. (See section 4 for what
  this does to the GPU-vs-CPU ratio.)

GPU-only flags:

- `--gpu-batch-size N` — per-worker rows per fused batch (sets
  `RAY_DATA_GPU_PREPROC_BATCH_SIZE`). **Default (recommended): leave it off.** The
  auto sizer now targets the VRAM-bounded "≈one block per GPU" point that the
  local sweeps found optimal (see section 4), and shrinks automatically for the
  wider `--feature-set wide` frame. Pass this only to pin an explicit size.
- `--gpu-num-gpus N` — number of one-GPU fused workers (sets
  `RAY_DATA_GPU_PREPROC_NUM_GPUS`; default = the cluster's total GPU count, so it
  scales across nodes with no cross-node GPU traffic).
- `--profile` — log each worker's fused-transform H2D / compute / D2H wall split
  (sets `RAY_DATA_GPU_PREPROC_PROFILE`). The fused fit/transform device-time split
  is also collectable per-worker via `RAY_DATA_GPU_PREPROC_PROFILE_DIR`.

AWS GPU Ray cluster (8× g6.4xlarge, 1× L4 each — see `cluster/ray-gpu.yaml`):

```bash
./build-wheel.sh 3.10 ./dist
ray up cluster/ray-gpu.yaml -y

ray exec cluster/ray-gpu.yaml \
  'cd /home/ray/benchmarks/criteo && python gpu_pipeline.py \
    --days 1 \
    --data-root s3://bobbwang-ray-e2e-criteo/criteo-private-ad/data \
    --out s3://bobbwang-ray-e2e-criteo/criteo-private-ad/outputs/gpu_days1 \
    --ray-address auto'

ray down cluster/ray-gpu.yaml -y
```

Leave `--gpu-batch-size` unset (above): the fused device batch is VRAM-auto
(section 4), which is what a real user runs and what the local sweep found
optimal. See section 5 for the full end-to-end cloud run (GPU + CPU baseline).

To benchmark: run `cpu_pipeline.py` on the 8× m5dn.4xlarge CPU cluster and
`gpu_pipeline.py` on the 8× g6.4xlarge GPU cluster over the same `--days`, then
compare the GPU `fused` stage wall against the CPU `impute + encode + scale`
subtotal (both manifests record per-stage timings and fitted stats for parity).

---

## 4. Local fused GPU-vs-CPU profiling (`bench_fused_local.py`)

`bench_fused_local.py` is a fast, repeatable harness for iterating on the fused
`impute + encode + scale` device stage **without** re-running the whole pipeline
each time. It materializes `read → prep → sort → indicators` **once** (Ray-auto
blocks — it never repartitions), caches that sorted dataset in-process, then
sweeps GPU fused variants against the CPU `impute+encode+scale` subtotal on the
identical cached input. For each variant it times the fused **fit** and
**transform** separately, reads the per-worker H2D / compute / D2H split, and
checks fitted-stat parity vs the CPU ops.

```bash
# quick loop (a few days, default sweep), runs on the DGX box's GPUs
.venv/bin/python benchmarks/criteo/bench_fused_local.py --days 1-3

# sweep device batch x overlap, emulate the 8x L4 cluster shape
.venv/bin/python benchmarks/criteo/bench_fused_local.py \
  --days 1-3 --emulate-cluster --cluster-gpus 8 \
  --gpu-batch-sizes auto,1000000,2000000 --overlap-modes off,on --no-cpu
```

`--emulate-cluster` pins the worker count to `--cluster-gpus` and caps per-GPU
VRAM to `--cluster-vram-gb` (default 24, via `RAY_DATA_GPU_PREPROC_VRAM_BYTES`)
so the auto device batch matches what an L4 would pick. **The DGX box's V100s
(32 GB, NVLink) are not the deploy target (L4, 24 GB, PCIe), so treat local
numbers as directional and confirm the final ranking on AWS.**

### What the iteration found (days 1-3 ≈ 14.7M rows, emulating 8× L4)

Fused stage vs the CPU `impute+encode+scale` subtotal (≈115 s):

| change | fused wall | vs CPU |
|--------|-----------:|-------:|
| baseline: per-kind fit scans + old auto batch (0.46M) | 83.8 s | 1.38x |
| **one-scan fit** (combine the 3 reductions into 1 GPU pass) | 37.6 s | 3.0x |
| **+ VRAM-bounded auto batch** (≈1 block/GPU, 1.84M) | **21.2 s** | **5.4x** |

Both changes are **on by default** and parity-validated (GPU fitted stats match
the CPU baseline, lean and wide). Ranked levers, largest first:

1. **One-scan fit** is the big one. The fit dominates the fused wall, and it was
   doing ≥3 full GPU scans (one per reduction kind); collapsing them into a
   single `from_arrow`-per-block pass cut fit ~2.9x (lean) / ~2.4x (wide).
2. **Device batch size** — bigger is better up to ≈one block per GPU (then extra
   size just idles GPUs). The old auto sizer's load-balance cap forced tiny
   batches; the sizer now targets the VRAM-bounded `rows / num_gpus` point. On
   the 8-GPU shape: 0.46M → 1.0M → 2.0M batch = 3.0x → 4.7x → 5.4x.
3. **GPU count** — 8 is the sweet spot for this dataset (g4 5.1x, g8 5.4x, g16
   4.2x: past ≈one block/GPU more workers just shrink the batch).
4. **Transfer/compute overlap** (pack 2 fused actors per GPU,
   `RAY_DATA_GPU_PREPROC_GPU_FRACTION<1`) — **not worth it here.** The transform
   is compute-bound (~70-85% compute, only ~15% transfer), so packing slightly
   *hurt* in every case. Left off by default.

### Lean vs wide

`--feature-set wide` (keep the `features_not_available_*` bucket → 228 vs 97
columns) still wins big on GPU (**4.0x** vs the wide CPU subtotal), but the
*ratio* is smaller than lean (5.4x). The CPU baseline here is dominated by a
fixed-cost most-frequent imputation (a pandas `value_counts` pickle fallback on a
couple of columns) that barely grows with column count, while the GPU does real
per-column work — so widening the frame adds GPU work faster than CPU work. Net:
the GPU advantage is largest on the lean recipe; wide mainly demonstrates the
fused stage scales gracefully to much wider frames.

### Recommended config (how a real user runs it)

Everything auto — no manual block or batch sizing:

```bash
# GPU (8x L4 cluster): one-scan fit + VRAM-auto device batch are the defaults
python gpu_pipeline.py --days 1-30 --ray-address auto --data-root s3://… --out s3://…
# CPU baseline, same days/feature-set
python cpu_pipeline.py --days 1-30 --ray-address auto --data-root s3://… --out s3://…
```

Leave `--blocks`, `--gpu-batch-size`, and `--gpu-num-gpus` unset; Ray sizes
read/sort blocks, the fused device batch is VRAM-auto, and the fused worker count
defaults to the cluster's GPU total. The result JSONs from the runs above live
in `benchmarks/criteo/data/` (git-ignored).

---

## 5. End-to-end cloud run (GPU + matched CPU baseline)

The realistic, all-auto cloud benchmark: a cheap GPU smoke, one full day to
de-risk (confirms saved output is globally sorted and parity holds *before* the
long run), then the full 30-day GPU run, then the matched 8-node CPU baseline.
Clusters are launched **sequentially** and torn down right after to control cost
(the 8x g6.4xlarge GPU cluster is ~\$12-16/hr). Everything is auto -- no
`--blocks`, `--gpu-batch-size`, or `--gpu-num-gpus`.

```bash
# 0. one-time: build the wheel + stage the dataset to S3 (skip if already done)
./build-wheel.sh 3.10 ./dist
aws s3 ls s3://bobbwang-ray-e2e-criteo/criteo-private-ad/data/ >/dev/null 2>&1 || \
  aws s3 sync /bobbwang/datasets/CriteoPrivateAd/data \
    s3://bobbwang-ray-e2e-criteo/criteo-private-ad/data

# 1. GPU cluster: up + verify GPUs/RAPIDS/overlay
ray up cluster/ray-gpu.yaml -y
ray exec cluster/ray-gpu.yaml 'python -c "from ray.data.preprocessors import _gpu; print(\"gpu_available\", _gpu.gpu_available())"'

# 1a. smoke (cheap wiring check)
ray exec cluster/ray-gpu.yaml 'cd /home/ray/benchmarks/criteo && python gpu_pipeline.py --days 1 --rows 100000 --data-root s3://bobbwang-ray-e2e-criteo/criteo-private-ad/data --out s3://bobbwang-ray-e2e-criteo/criteo-private-ad/outputs/gpu_smoke_days1 --ray-address auto --skip-saved-sort-verify'

# 1b. de-risk: one real full day -- confirm globally_sorted=True + parity first
ray exec cluster/ray-gpu.yaml 'cd /home/ray/benchmarks/criteo && python gpu_pipeline.py --days 1 --data-root s3://bobbwang-ray-e2e-criteo/criteo-private-ad/data --out s3://bobbwang-ray-e2e-criteo/criteo-private-ad/outputs/gpu_days1 --ray-address auto --overwrite'

# 1c. full 30-day, everything auto (the headline GPU run)
ray exec cluster/ray-gpu.yaml 'cd /home/ray/benchmarks/criteo && python gpu_pipeline.py --days 1-30 --data-root s3://bobbwang-ray-e2e-criteo/criteo-private-ad/data --out s3://bobbwang-ray-e2e-criteo/criteo-private-ad/outputs/gpu_days1_30_8node --ray-address auto --overwrite'
ray down cluster/ray-gpu.yaml -y

# 2. matched 8-node CPU baseline, same 30 days, all-auto
ray up cluster/ray-cpu.yaml -y
ray exec cluster/ray-cpu.yaml 'cd /home/ray/benchmarks/criteo && python cpu_pipeline.py --days 1-30 --data-root s3://bobbwang-ray-e2e-criteo/criteo-private-ad/data --out s3://bobbwang-ray-e2e-criteo/criteo-private-ad/outputs/cpu_days1_30_8node --ray-address auto --overwrite'
ray down cluster/ray-cpu.yaml -y

# 3. compare: GPU fused stage vs CPU impute+encode+scale subtotal + parity
python benchmarks/criteo/compare_manifests.py \
  --gpu s3://bobbwang-ray-e2e-criteo/criteo-private-ad/outputs/gpu_days1_30_8node/manifest.json \
  --cpu s3://bobbwang-ray-e2e-criteo/criteo-private-ad/outputs/cpu_days1_30_8node/manifest.json
```

`compare_manifests.py` reports the only fair number -- GPU `fused_stage_s` vs the
CPU `impute + encode + scale` subtotal (sort is CPU in both and cancels; this is
NOT the CPU manifest's `gpu_target_subtotal_s`, which is sort+encode+scale) --
plus fitted-stat parity and the saved-output `globally_sorted` flag. For a
WIDE-frame comparison, add `--feature-set wide` to **both** the GPU and CPU
30-day commands and a fresh `..._wide` `--out`.

---

## Notes

- **S3 paths** use the bucket `s3://bobbwang-ray-e2e-criteo` (us-west-2): data at
  `/criteo-private-ad/data`, outputs at `/criteo-private-ad/outputs`. Stage the
  dataset once (see "Stage the dataset to S3") before any cloud run.
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
- **GPU fused tuning knobs** (env; sane defaults, override only to experiment):
  `RAY_DATA_GPU_PREPROC_FUSED_FIT` (default `1` = one-scan fit; `0` = legacy
  per-kind scans, for A/B), `RAY_DATA_GPU_PREPROC_VRAM_FRACTION` (0.15) /
  `RAY_DATA_GPU_PREPROC_PEAK_FACTOR` (3.0) / `RAY_DATA_GPU_PREPROC_MIN_BLOCKS_PER_GPU`
  (1) drive the auto device-batch sizer, `RAY_DATA_GPU_PREPROC_GPU_FRACTION`
  (1.0) packs multiple fused transform actors per GPU, and
  `RAY_DATA_GPU_PREPROC_VRAM_BYTES` overrides detected per-GPU VRAM (used by
  `bench_fused_local.py --emulate-cluster`).
- **cuDF 26.02 `from_arrow_host`**: this version intermittently raises
  `cudaErrorInvalidValue` on sliced/offset Arrow buffers; the device conversion
  retries via an Arrow-IPC round-trip (offset-0 buffers), so local runs are
  robust. The AWS cluster's RAPIDS build does not hit this.
- Generated outputs under `benchmarks/criteo/data/` and the built root `dist/`
  wheel are git-ignored — do not commit them.
```

