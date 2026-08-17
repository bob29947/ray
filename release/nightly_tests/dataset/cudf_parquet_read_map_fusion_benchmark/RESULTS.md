# Eight-node direct-S3 read-fusion benchmark

## Result

The strict same-setting comparison reached **1.822x**:

| | Isolated c1 | Fused c1 |
|---|---:|---:|
| Elapsed | 53.909s | 29.589s |
| Throughput | 17.808M rows/s | 32.445M rows/s |
| Speedup | 1.000x | **1.822x** |
| Time-derived savings | - | **45.11%** |
| Execution cost | $0.15852 | $0.08700 |
| Physical read/map operators | 2 | 1 |
| Actor concurrency / derived queue | 1 / 2 | 1 / 2 |
| Ray object-store spill | 20.91 GB | 0 |
| Aggregate network receive + send | 204.13 GB | 10.24 GB |
| Peak device GPU memory | 33.94% | 36.18% |

The tuned fused arm used the public `max_concurrency=4` option:

| | Ordinary isolated c1 | Tuned fused c4 |
|---|---:|---:|
| Elapsed | 53.909s | 18.599s |
| Throughput | 17.808M rows/s | 51.614M rows/s |
| Relative speedup | 1.000x | **2.898x** |
| Time-derived savings | - | **65.50%** |
| Execution cost | $0.15852 | $0.05469 |
| Actor concurrency / derived queue | 1 / 2 | 4 / 8 |
| Ray object-store spill | 20.91 GB | 0 |
| Aggregate network receive + send | 204.13 GB | 10.24 GB |
| Peak device GPU memory | 33.94% | 60.91% |

The tuned row is intentionally labeled asymmetric. An isolated
`max_concurrency=4` observation failed during Arrow-to-cuDF conversion with
`cudaErrorInvalidValue`, so it is not presented as a same-concurrency result.

At the receipted price of $1.3232 per node-hour, the eight-node cluster costs
$10.5856/hour. Execution cost is elapsed time multiplied by that same cluster
rate; the savings therefore equal the elapsed-time savings.

## Workload and timing

Each valid observation processes 960 million rows from 12 immutable direct-S3
Parquet objects. The full cohort is 11.702 GB compressed; the selected 13
columns are 9.902 GB compressed. The tokenizer materializes 76.8 GB of output.

Timing starts immediately before `read_parquet()` and stops after the mapped
output is materialized in Ray's object store. Listing, Parquet I/O and decode,
the GPU tokenizer, and output publication are inside the timer. Runtime startup,
plan inspection, and correctness checks are outside it.

All valid runs used eight `g6.4xlarge` nodes, eight one-GPU actors, 128 total
CPUs, 16,777,216-row batches, default object-store sizing, an omitted
`ActorPoolStrategy` queue option, and:

```text
CUDF_KVIKIO_REMOTE_IO=1
KVIKIO_NTHREADS=32
KVIKIO_TASK_SIZE=16777216
```

The isolated and fused arms used disjoint S3 keys with the same content digest.
Across the two fused observations, measured network receive was 20.353 GB for
19.803 GB of selected Parquet input, 2.77% overhead. No local dataset cache was
used.

## Correctness and interpretation

Every accepted observation processed exactly 960 million rows and produced the
same ordered schema:

- `uc_key: int64`
- `ts: int64`
- `token_ids: int32[12]`
- `label: int64`

The full-output, block-boundary-independent checksum matched in all three valid
observations:

```text
tfm-tokenizer-uint64-commutative-v1
sum1=f733d30d437c3580
sum2=d4a48d7c20121308
```

The physical-plan check found separate `ReadFilesParquetV2` and
`MapBatches(GPUTokenizer)` operators for the isolated arm. The fused arms had
one `ReadFilesParquetV2->MapBatches(GPUTokenizer)` actor operator. All eight
GPU slots were used, transport settings matched on every node, no output was
restored from spill, and no valid observation reported fallback, retries, OOM,
or task failure.

The strict c1 gain comes from removing CPU/PyArrow decode, decoded Arrow
object-store materialization, cluster transfer, and Arrow-to-cuDF conversion.
The c4 run additionally overlaps actor tasks while preserving the same tokenizer
and one-GPU-per-actor geometry. The optimizer itself preserves Ray's default
concurrency when the user omits the option.

## Scope and limitations

These are single cold observations on a scaled synthetic transaction-tokenizer
workload. They are not Stripe measurements and do not include preprocessing
fit, downstream training, or writing. The c4 figure compares a tuned fused arm
with the ordinary isolated c1 baseline and must not be described as a
same-setting result.

The measured wheel used candidate
`dc54f6fd4988ad9ce88ebbf005e4368ad7f78f8d` on Lee base
`3605699192aca4b82231dde65a5136175d978481`. The published feature head
`6c7a858b3116d01085984eddbcde7cc17736fe71` adds only a one-line
direct-S3-unavailable fallback guard after that candidate.

The byte-exact worker, mechanically Ray-formatted tokenizer source, and exact
S3 object inventories for the three reported cohorts are included here. Cloud
provisioning, credentials, instance identities, and lifecycle logs are
deliberately excluded.
