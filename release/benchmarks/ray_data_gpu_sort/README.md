# Ray Data distributed GPU sort benchmark

This directory contains the portable worker and the historical AWS results for
Ray Data's spillable distributed GPU sort.  It intentionally contains no EC2
lifecycle code, account identifiers, cluster YAML, dependency locks, or local
filesystem defaults.

The workload uses U.S. DOT Bureau of Transportation Statistics flight records:
80,738,761 rows, 627 Arrow blocks, and 63.881 GiB decoded for the full 1x
projection.  Each row is one flight segment from April 2013 through December
2025.  The full projection has 109 native columns plus a generated, unique
`row_id`.

The six trend cells hold rows and blockization fixed while varying payload and
keys:

| Cell | Projection | Ascending sort keys |
|---|---:|---|
| Narrow | 5 columns | `Origin, Dest, FlightDate, CRSDepTime` |
| Core | 57 columns | `Origin, Dest, FlightDate, CRSDepTime` |
| Full | 110 columns | `Origin, Dest, FlightDate, CRSDepTime` |
| Origin string | 110 columns | `Origin` |
| Origin integer | 110 columns | `OriginAirportID` |
| Route | 110 columns | `Origin, Dest` |

The natural-size study repeats the full/four-key cohort at 1x, 2x, and 2.45x
without lowering the GPU residency budget.  Ray's filesystem spill directory
must be on local disk (NVMe in the AWS study), never `/dev/shm`.

## Run one observation

Stage the normalized BTS publication at the same absolute path on every node.
The directory must contain `manifest.json` plus its `parquet/` tree.  Start Ray
with its filesystem spill path on local disk, then run one worker from the head:

```bash
python -m release.benchmarks.ray_data_gpu_sort.worker \
  --dataset-root /path/to/bts-airline-on-time \
  --spill-directory /path/to/trial/ray-spill \
  --output /path/to/trial/result.json \
  --kind trend \
  --backend gpu \
  --cell full \
  --repetition 1 \
  --wave-fraction 0.375
```

The example selects GPU sort. For the two CPU modes, `--backend pyarrow` uses
the default pull-based shuffle; add `--shuffle-strategy push` to select the
configured push-based shuffle.

For a 2x natural-spill observation, add `--kind natural
--scale-numerator 2`.  For the 2.45x point use `--scale-numerator 245
--scale-denominator 100`.  Run the exact 160,000-row CPU/GPU smoke first with
`--kind smoke`.

Restart Ray before every performance invocation and use a new, empty spill
directory.  The worker connects to the existing cluster; it never starts,
restarts, provisions, or terminates nodes.  Input projection and Plasma
materialization happen before the timer.  The timed interval is exactly
`Dataset.sort(...).materialize()`, ending after the output is sealed in Plasma.

The result receipt includes input shape, official cold wall time, throughput,
schema/order/row-count checks, the exact `row_id` checksum, Ray Core spill and
restore counters, per-node peak memory, network traffic, and a digest of the
input plan and harness.  The smoke additionally compares every PyArrow and GPU
output row and value.

Run the CPU-only harness checks with:

```bash
python -m pytest -q release/benchmarks/ray_data_gpu_sort/test_benchmark.py
```

## Result provenance

- [RESULTS.md](RESULTS.md) ([JSON](RESULTS.json)) contains the original payload,
  key, natural-spill, and optimization results through 156.5 GiB.
- [AWS_255G_SCALING.md](AWS_255G_SCALING.md)
  ([JSON](AWS_255G_SCALING.json)) contains the 255.5 GiB, 2-16 node comparison
  of GPU sort, default pull-based PyArrow, and configured push-based PyArrow.

Those measurements came from the instrumented Ray 2.55.1 candidate from which
the production implementation was distilled. They are historical algorithm
evidence, not a claim that the cleaned PR head was benchmarked byte-for-byte.
The benchmark branch inherits the current production code from its parent
commit; rerunning this worker produces receipts for that exact checkout.
