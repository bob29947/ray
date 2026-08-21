# Eight-node cuDF map-fusion benchmark

## Result

The largest qualified same-setting pair reached **4.045×** on the wide
synthetic tabular transform:

| | Isolated | Fused |
|---|---:|---:|
| Elapsed | 406.956s | 100.607s |
| Throughput | 58,974 rows/s | 238,553 rows/s |
| Physical actor maps | 5 | 1 |
| Ray tasks | 480 | 96 |
| Calls per stage | 96 | 96 |
| Natural Ray spill | 81.12 GB | 144.49 GB |
| Ray restore | 0 | 0 |
| Aggregate network receive + send | 737.79 GB | 137.61 GB |
| Peak GPU memory per process | 7.27% | 10.31% |
| Execution cost | $1.1966 | $0.2958 |
| Normalized cost / billion rows | $49.86 | $12.33 |

This is **75.28% less transform time and execution cost**, or $0.9008 saved
for this 24M-row observation at the receipted eight-node rate of $10.5856/hour.

## Workload and timing

The deterministic materialized Arrow input contains 24M rows, 681 `float32`
numerical columns, and 74 `int32` categorical columns (74.745 GB). The five
ordinary callable-class `map_batches()` stages are:

1. Numeric `log1p`.
2. Numeric standardization from fixed pre-fitted state.
3. Numeric null and NaN filling.
4. Categorical conversion to strings.
5. Categorical vocabulary lookup with null and OOV values mapped to zero.

Timing starts immediately before constructing this five-map chain and stops
when the final `materialize()` returns. Input generation/materialization,
physical-plan proof, fitting, reading, writing, and post-timing digest checks
are excluded.

Both arms use eight retained `g6.4xlarge` nodes, eight one-GPU actors,
`ActorPoolStrategy(min_size=1, max_size=8)`, serial/default actor concurrency,
default queue depth, `batch_size=250_000`, `batch_format="cudf"`, and
`zero_copy_batch=True`. Object-store capacity, spill thresholds, and the NVMe
spill path are identical. Only `enable_cudf_actor_fusion` changes.

## Qualified trend

| Rows | Batch | Isolated | Fused | Speedup | Savings |
|---:|---:|---:|---:|---:|---:|
| 8M | 125K | 200.823s | 59.602s | 3.369× | 70.3% |
| 8M | 250K | 135.561s | 38.662s | 3.506× | 71.5% |
| 16M | 125K | 392.005s | 113.351s | 3.458× | 71.1% |
| 16M | 250K | 257.184s | 68.925s | 3.731× | 73.2% |
| 24M | 250K | 406.956s | 100.607s | **4.045×** | **75.3%** |

The lowest fused elapsed time is 38.662s at 8M/250K. The highest fused
throughput is 238,553 rows/s at 24M/250K. The lowest isolated elapsed time is
135.561s at 8M/250K; the highest isolated throughput is 62,212 rows/s at
16M/250K.

The 400K pairs are excluded even though their final digests match: Ray's
isolated stages received different callable invocation counts from one another,
so those pairs do not preserve stateful callable boundaries. The 1M run is a
physical-plan/correctness smoke test, not a performance result, because only
four fused GPU slots had enough tasks.

## Correctness and interpretation

The headline pair has the required five-map isolated plan and one named fused
plan, all eight GPU slots, identical 96-call/24M-row totals at every stage,
matching schema and full-data deterministic digests, no task failures or
retries, no OOM, no fallback, and zero restored bytes. Fused peak process VRAM
is 10.31%, well below the 80% limit.

Natural spilling is not treated as a failure. At 24M rows the fused arm spills
more output-store data, but still finishes four times faster because it removes
four actor boundaries and their intermediate task/object/network traffic. The
task count falls 5× and measured aggregate node network traffic falls about
5.36×.

This result is a **wide synthetic tabular transform** (a linearized
combined-transform surrogate). It is not measured Stripe performance and does
not cover fitting, Parquet/S3 reading, writing, CPU comparison, or end-to-end
preprocessing.

## Code, validation, and cleanup

The benchmark uses the existing map-fusion candidate
`aa17c53e462889bc835ab7f95446c3c6b80b24c4`, with Lee base
`3605699192aca4b82231dde65a5136175d978481`. The wheel, bundle, and installed
fusion-module hashes are recorded in [PROVENANCE.json](PROVENANCE.json), and the\nper-pair metrics and correctness digests are in [RESULTS.json](RESULTS.json). This\ncampaign changed
**zero tracked Ray or Ray Data production files**; benchmark and lifecycle
artifacts live under `.codex-work/`.

All 62 benchmark-infrastructure tests pass; Ruff and `py_compile` pass.

The exact eight retained instances were terminated with the dry-run plan hash
`e9e9111...`. Their eight root volumes were automatically deleted; no network
resources were created or removed. Three stable-empty checks passed at
08:02:09Z, 08:02:18Z, and 08:02:28Z.

The complete campaign used 15 unique instances because an early autoscaler
reset replaced seven workers. IAM denied `cloudtrail:LookupEvents`, and those
seven workers had already aged out of `DescribeInstances`, so the fail-closed
ledger does not publish a guessed all-campaign total. The per-observation
execution costs above are fully determined from measured time and the current
price receipt; a later AWS Cost and Usage Report is needed for authoritative
campaign spend.
