# cuDF actor-map fusion benchmark

This directory contains the exact transform worker used to validate the cuDF
actor-map fusion proposed in
[leewyang/ray#3](https://github.com/leewyang/ray/pull/3). The workload is a
wide synthetic tabular transform, not measured Stripe performance.

The materialized Arrow source has 681 `float32` numerical columns and 74
`int32` categorical columns. The timed pipeline contains five ordinary
callable-class `map_batches()` stages:

1. Numeric `log1p`.
2. Numeric standardization with fixed pre-fitted state.
3. Numeric null and NaN filling.
4. Categorical string conversion.
5. Categorical vocabulary lookup with null and OOV values mapped to zero.

Timing starts immediately before constructing the five-map chain and ends when
the final `materialize()` returns. Source creation, fitting, reading, writing,
physical-plan inspection, and post-timing digest validation are excluded.

## Exact provenance

- Fusion PR head: `aa17c53e462889bc835ab7f95446c3c6b80b24c4`.
- Lee base: `3605699192aca4b82231dde65a5136175d978481`.
- Worker SHA-256:
  `f5b20ca2966a690d24210ea4d38391446bbf988c579e5cb662c0553473b25f18`.
- Installed fusion-module SHA-256:
  `270fbc4a89ec5b6302d88f03e3fdefd83919c484007ae710a6fc4def23b27916`.
- Derived wheel SHA-256:
  `4106870d1edd3d1cdff04ab4ae7c0ae072272113186e5cc61a620afda13e3b20`.

The worker intentionally checks these identities before measuring anything.

## Running one arm

Start a fresh eight-node Ray runtime before each observation. Each node must
provide 16 CPUs and one GPU. The head participates as a GPU worker. Run from
this directory so `benchmark_core.py` is importable.

```bash
mkdir -p /mnt/nvme/map-fusion/isolated

python worker.py \
  --mode isolated \
  --rows 24000000 \
  --batch-size 250000 \
  --nodes 8 \
  --trial-root /mnt/nvme/map-fusion/isolated \
  --output /mnt/nvme/map-fusion/isolated/result.json
```

Restart Ray, then repeat with `--mode fused` and a different trial directory.
Both arms use `ActorPoolStrategy(min_size=1, max_size=8)`, one GPU per actor,
serial/default actor concurrency, default queue depth, and the same default Ray
object-store and spill behavior. Only `enable_cudf_actor_fusion` changes.

The worker records row count, schema, full-data digest, callable invocation
counts, physical operators, task failures, object-store spill/restore, GPU and
host memory, network traffic, NVMe traffic, and elapsed rows/s. It leaves a JSON
result even when an observation fails its contract.

The original worker treated any spill as a rejection. The final campaign policy
allowed default Ray spilling and qualified a receipt only when spill was its
sole rejection reason; every correctness, retry, physical-plan, GPU-coverage,
and callable-boundary check remained mandatory.

## Validation and result

```bash
pytest -q test_benchmark_core.py
```

The largest qualified pair used 24 million rows and 250,000-row batches:

- Isolated: 406.956 seconds.
- Fused: 100.607 seconds.
- Speedup: 4.045x.
- Same-setting execution-cost savings: 75.28%.

See [RESULTS.md](RESULTS.md) for the full trend, methodology, correctness
evidence, spill telemetry, costs, and limitations.
