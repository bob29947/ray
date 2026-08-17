# cuDF Parquet read-to-map fusion benchmark

This directory contains the byte-exact worker and a Ray-formatted copy of the
tokenizer workload used to validate the cuDF Parquet read fusion proposed in
[leewyang/ray#5](https://github.com/leewyang/ray/pull/5).

The tokenizer source is the measured implementation with only mechanical
Black and Ruff formatting applied for the Ray repository. Its logic is
unchanged.

The benchmark compares two ordinary Ray Data pipelines:

- **Isolated:** CPU/PyArrow Parquet read, Arrow object-store materialization,
  Arrow-to-cuDF conversion, then the GPU tokenizer.
- **Fused:** direct cuDF Parquet decode and the same tokenizer in one downstream
  GPU actor.

Both arms call `read_parquet(...).map_batches(...).materialize()`. Fusion is
the only physical-plan difference in the strict concurrency-1 comparison.

## Workload and timing

The scaled synthetic transaction cohort has 960 million rows in 12 immutable S3
Parquet objects. The benchmark selects 13 columns (9.902 GB compressed) and runs
a callable-class GPU tokenizer that produces four output columns:
`uc_key`, `ts`, a 12-element `token_ids` tensor, and `label`.

Timing starts immediately before `read_parquet()` and ends when
`map_batches()` output is materialized in Ray's object store. S3 listing,
Parquet decode, tokenizer execution, and output materialization are included.
Cluster startup, post-run plan inspection, schema validation, and deterministic
full-output checksum validation are excluded.

## Exact provenance

- Published feature head: `6c7a858b3116d01085984eddbcde7cc17736fe71`.
- Measured direct-S3 candidate: `dc54f6fd4988ad9ce88ebbf005e4368ad7f78f8d`.
- Lee base: `3605699192aca4b82231dde65a5136175d978481`.
- Compiled Ray core: `2741c6461d2bd3e5ff114af67be7a1190453dadd`.
- Derived wheel SHA-256:
  `99f262b983ecfdc3a90db9b570ffaa01d797a6101242aae1fdaeaff567bbea27`.
- Worker SHA-256:
  `a5cff1646d7dfd82acc8cb29bf202199c4a7ae189dae3f82b5556c43462e1486`.
- Measured full campaign manifest SHA-256:
  `f340c0facf33e1be7b4fe976627459cf33aec8de171a721fff36129960956080`.
- Published three-cohort manifest SHA-256:
  `653e063897bd382746cb5bd6995dcac2dff52b852cb193155c4ef89a1161d05a`.
- Published manifest body digest:
  `8b08496a34fdfba7ea378b741f2f4f2765d20be948afeb8f3d8ba02d88fba5c2`.
- Optional profiler helper SHA-256:
  `b7f54cfa2dd82836b11032f072364d0ec33f1ea90c7faa26d711b4d105084b0e`.
- Python 3.11.15, CUDA 12.2, cuDF/KvikIO/RMM 25.12.

The current feature head adds one Arrow-fallback guard after the measured
candidate. That one-line follow-up is not on the measured direct-S3 path.

## Cluster and transport

The final comparison used eight `g6.4xlarge` instances in `us-west-2`.
Every node supplied 16 CPUs and one L4 GPU; the head also ran one worker. Each
arm used eight one-GPU actors, a 16,777,216-row batch size, default Ray object
store settings, and the same NVMe spill configuration.

```bash
export CUDF_KVIKIO_REMOTE_IO=1
export KVIKIO_NTHREADS=32
export KVIKIO_TASK_SIZE=16777216
```

The strict comparison passes `max_concurrency=1` in both arms and leaves the
actor-pool queue unset, so Ray derives a queue depth of 2. The tuned fused run
passes the public `max_concurrency=4` option and still leaves the queue unset,
so Ray derives a queue depth of 8. The optimizer does not inject hidden
concurrency.

## Running one arm

Start a fresh eight-node Ray runtime and run from this directory so the included
`src` tokenizer package is importable. The S3 objects require appropriate
read credentials.

```bash
mkdir -p /mnt/nvme/read-fusion/isolated-c1

python worker.py \
  --manifest cohorts.json \
  --cohort tune-b16-t16 \
  --mode isolated \
  --nodes 8 \
  --batch-size 16777216 \
  --map-batches-max-concurrency 1 \
  --spill-directory /mnt/nvme/read-fusion/isolated-c1 \
  --output /mnt/nvme/read-fusion/isolated-c1/result.json
```

Restart Ray before the fused observation, use an equal-content cohort with
disjoint object keys, change `--mode` to `fused`, and write to a new trial
directory. For the tuned result, set
`--map-batches-max-concurrency 4`. Do not pass the worker's benchmark-only
physical-concurrency or fractional-GPU probes.

The worker records physical operators, resolved actor geometry, row count,
schema, deterministic checksum, Ray spill/restore, network and NVMe traffic,
GPU utilization and memory, transport realization, retries, and elapsed rows/s.
It writes a self-hashed JSON result even when an observation fails.

## Validation and result

```bash
pytest -q test_worker.py
```

The strict concurrency-1 comparison reached **1.822x**. With user-selected
`max_concurrency=4`, the fused arm reached **2.898x** relative to the ordinary
concurrency-1 isolated pipeline.

See [RESULTS.md](RESULTS.md) for the full comparison, costs, correctness,
traffic, spill evidence, and limitations.
