# AWS BTS GPU sort: 255.5 GiB scaling

This report extends the [original AWS results](RESULTS.md) with a strong-scaling
campaign over 322,955,044 rows (255.523 GiB decoded). At the same active node
count, the GPU external sort was 3.46-6.40x faster and 64.8-81.0% cheaper per
completed sort than configured push-based PyArrow. Default pull-based PyArrow
did not complete at any tested scale, so it is not used as the speedup or cost
denominator.

Machine-readable values, including unrounded comparison ratios, are in
[AWS_255G_SCALING.json](AWS_255G_SCALING.json).

## Headline comparison

| Dataset | Active nodes | Default pull CPU | Configured push CPU | GPU pair | GPU median | GPU speedup vs push | Logical cost saving vs push |
|---:|---:|---|---:|---:|---:|---:|---:|
| 255.5 GiB | 2 | Failed: head-memory/OOM pressure | 8761.10s | 2307.55 / 2345.75s | 2326.65s | 3.77x | 67.7% |
| 255.5 GiB | 4 | Failed: head-memory/OOM pressure | 4409.71s | 1287.90 / 1262.43s | 1275.16s | 3.46x | 64.8% |
| 255.5 GiB | 8 | Failed: head-memory/OOM pressure | 2280.62s | 500.57 / 511.94s | 506.25s | 4.50x | 73.0% |
| 255.5 GiB | 16 | Failed: head-memory/OOM pressure | 1354.90s | 212.45 / 210.67s | 211.56s | 6.40x | 81.0% |

Speedup is configured-push CPU time divided by the GPU pair median. Logical
cost saving is `1 - GPU cost / configured-push CPU cost`, using the active-node
on-demand rates below. These claims do not use an estimated completion time for
the failed default-pull runs.

## Dataset, method, and hardware

- The four-copy normalized U.S. DOT BTS workload has 322,955,044 rows, 2,508
  Arrow blocks, 110 columns, and 274,365,695,588 decoded bytes (255.523 GiB).
  The ascending keys are `Origin, Dest, FlightDate, CRSDepTime`.
- The projected input was materialized before timing. Cold time starts with
  materialized input, including naturally spilled ObjectRefs, and ends when
  `Dataset.sort(...).materialize()` has sealed a Ray-locatable sorted output.
- GPU points are pairs; each CPU variant has one observation per active-node
  count. Ray was restarted between observations. A retained 16-node fleet was
  sliced to 2, 4, 8, or 16 active nodes, and every completed run verified that
  inactive nodes did no dataset work or spill I/O.
- GPU nodes were AWS `g6.4xlarge` instances with one NVIDIA L4 24 GB GPU per
  node. CPU nodes were AWS `m5dn.4xlarge` instances. Ray object spilling used
  local NVMe.
- The CPU baseline is default pull-based Ray/PyArrow. The completed CPU
  alternative explicitly selected `sort_shuffle_push_based`. The final GPU
  campaign used a 0.25 wave fraction and 512 MiB exchange batches.
- The stock Ray base was 2.55.1 commit
  `237c2455ebb1ea15a32dd9e1fdeb2d617badc37f`. The measured candidate applied a
  shared 551-file Ray Data overlay (`df1e614913bfd6c61151e617a05c70c7c2d42fe4e5feea004b77e197090df817`)
  and wheel (`bb49fbbe53a1d931e1f92d17f9271338f0b738885f8f70b7f531aa33f019d8af`)
  to the common input plan (`1d6b93af0b77485775ab7766d4e64771b260ffa07b5132c2fe9e31ec6593574d`).
  It is historical algorithm evidence, not a claim that the cleaned production
  branch was benchmarked byte-for-byte.

## CPU capacity and configured-push scaling

All four default-pull observations are certified terminal capacity failures.
They encountered head-memory or kernel-OOM pressure followed by task retry,
lost-object, or no-progress states. None produced a completed output, so no
default-pull latency, logical per-sort cost, or matched speedup is reported.

Configured push completed at every scale and passed row-count, schema,
checksum, global-order, output-locatability, and inactive-node-isolation checks.

| Nodes | Cold | GiB/s | Scaling from 2 nodes | Parallel efficiency | Logical cost |
|---:|---:|---:|---:|---:|---:|
| 2 | 8761.10s | 0.029 | 1.00x | 100.0% | $5.296 |
| 4 | 4409.71s | 0.058 | 1.99x | 99.3% | $5.331 |
| 8 | 2280.62s | 0.112 | 3.84x | 96.0% | $5.514 |
| 16 | 1354.90s | 0.189 | 6.47x | 80.8% | $6.552 |

## GPU pairs and scaling

Only the final 0.25-wave observations contribute to these medians.

| GPUs | Cold pair | Median | GiB/s | Scaling from 2 GPUs | Parallel efficiency | Logical cost | Correctness |
|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | 2307.55 / 2345.75s | 2326.65s | 0.110 | 1.00x | 100.0% | $1.710 | 2/2 PASS |
| 4 | 1287.90 / 1262.43s | 1275.16s | 0.200 | 1.82x | 91.2% | $1.875 | 2/2 PASS |
| 8 | 500.57 / 511.94s | 506.25s | 0.505 | 4.60x | 114.9% | $1.489 | 2/2 PASS |
| 16 | 212.45 / 210.67s | 211.56s | 1.208 | 11.00x | 137.5% | $1.244 | 2/2 PASS |

The superlinear 8- and 16-GPU efficiencies coincide with fewer waves and a
drop from three merge passes to two; they should not be extrapolated beyond
the measured 2-16 GPU range.

## Spill and external-sort telemetry

GPU values below are pair medians. Externalized bytes are cumulative GPU-sorted
run output, not simultaneous VRAM excess.

| GPUs | Input/GPU | Waves | Pre-timer spill | Timed write / restore | Peak spill dir | GPU externalized | Initial + replacement runs | Merge passes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 127.76 GiB | 28 | 235.76 GiB | 1018.30 / 1251.29 GiB | 762.74 GiB | 255.54 GiB | 82.5 + 26.5 | 3 |
| 4 | 63.88 GiB | 14 | 244.19 GiB | 999.38 / 1227.23 GiB | 775.42 GiB | 255.54 GiB | 83.5 + 27.5 | 3 |
| 8 | 31.94 GiB | 7 | 255.59 GiB | 725.30 / 863.83 GiB | 765.83 GiB | 255.55 GiB | 89.0 + 24.0 | 2 |
| 16 | 15.97 GiB | 4 | 255.59 GiB | 623.27 / 372.60 GiB | 760.05 GiB | 255.54 GiB | 81.5 + 17.5 | 2 |

Configured-push CPU spill also decreased with scale:

| CPU nodes | Pre-timer spill | Timed write / restore | Peak spill dir | Peak head memory |
|---:|---:|---:|---:|---:|
| 2 | 242.27 GiB | 536.17 / 736.27 GiB | 765.89 GiB | 50.99 GiB |
| 4 | 249.89 GiB | 533.26 / 686.19 GiB | 774.11 GiB | 49.88 GiB |
| 8 | 255.59 GiB | 528.41 / 613.48 GiB | 695.87 GiB | 47.97 GiB |
| 16 | 255.59 GiB | 420.12 / 419.40 GiB | 643.59 GiB | 44.33 GiB |

The failed 2-node default-pull run recorded 242.74 GiB of pre-timer spill,
233.93 GiB written and 2033.89 GiB restored during the timed attempt, with a
475.33 GiB peak spill directory. Atomic memory/spill counters did not survive
the 4-, 8-, and 16-node pull failures, so those fields remain unavailable.

## Why 0.375 became 0.25

The original 0.375 wave fraction was not repeatable at this size. One 2-GPU
observation completed in 2200.47s, but its pair failed with a confirmed
RAPIDS-MPF/RMM transient OOM while requesting 236.113 MiB against an
18.729 GiB pool maximum. Two 4-GPU attempts independently showed the same
signature for 117.130 MiB and 127.180 MiB requests; a separate preflight
failure was classified as infrastructure and excluded from algorithm claims.

Reducing the wave fraction to 0.25 lowered device pressure and produced eight
consecutive accepted observations: two each at 2, 4, 8, and 16 GPUs. All eight
passed correctness and placement checks. Against the one accepted 0.375
2-GPU observation, the two 0.25 runs were 4.87% and 6.60% slower; that bounded
cost bought a complete, internally consistent scaling series.

## Cost basis

Logical per-sort cost is `active nodes * cold seconds / 3600 * node-hour
price`: $1.088 per `m5dn.4xlarge` CPU node-hour and $1.3232 per `g6.4xlarge`
GPU node-hour. It excludes retained inactive nodes and applies only to
completed sorts.

Campaign cost is a separate retained-fleet view. The 16-node rates were
$17.408 per CPU-fleet-hour and $21.1712 per GPU-fleet-hour. CPU retained-fleet
cost was $188.124 over 10.807 hours. GPU cost is a conservative upper bound of
$119.362 over 5.638 hours because per-instance termination records were no
longer queryable; the combined conservative upper bound is $307.485.

## Validation and caveats

- Every final GPU observation and every configured-push CPU observation passed
  exact rows, schema, `row_id` checksum, global ordering, output locatability,
  and inactive-node isolation. All expected artifacts were present and valid.
- GPU medians contain two observations; each CPU point contains one. The report
  shows observed values, not confidence intervals.
- Default pull capacity failures are measured outcomes, but they provide no
  completion-time or cost denominator. The headline comparison therefore uses
  configured push explicitly.
- Dashboard summary requests failed, so peak/final Ray task and object counts
  are unavailable. This does not change the cold-time boundary.
- Retained-fleet cost describes the supplied campaign fleets; logical cost
  describes only the active nodes for one completed sort. The two must not be
  mixed.
