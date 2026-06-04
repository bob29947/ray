"""Isolated H2D / D2H transfer microbenchmark.

Goal: figure out which levers actually move host<->device bandwidth for the
64 GiB benchmark payload, before committing to a rewrite of the Ray GPU sort.

We compare the two axes that the analysis says matter most:

    process model :  one 16-GPU actor driving 16 threads (current design)
                     vs 16 single-GPU actors (one process per PCIe link)
    host memory   :  pageable numpy  vs  pinned (page-locked) cudaHostAlloc

Plus the host-side *marshalling* cost that the current code pays on top of the
DMA: assembling a row-major (rows, 16) int32 array from 16 separate Arrow
columns (a 64 GiB host transpose), versus a plain contiguous column copy.

Raw DMA is measured on flat int32 buffers (layout is irrelevant to PCIe
bandwidth); marshalling is measured separately so we can attribute the cost.

Timing contract mirrors the real path: host buffer resident -> device buffer
resident (H2D) and the reverse (D2H), warm, with a per-device sync.
"""

import argparse
import os
import time

# Let every actor see every GPU; each actor picks its device by rank. Required
# later for cross-process P2P/IPC, and harmless for the pure-transfer test.
os.environ.setdefault("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "1")

import numpy as np
import ray

GiB = 2 ** 30

# NUMA placement of the 16 V100s on this box (from `nvidia-smi topo -m`):
#   GPU 0-7  -> NUMA node 0 (CPUs 0-23, 48-71)
#   GPU 8-15 -> NUMA node 1 (CPUs 24-47, 72-95)
NUMA0 = list(range(0, 24)) + list(range(48, 72))
NUMA1 = list(range(24, 48)) + list(range(72, 96))


def cpus_for_gpu(dev):
    return NUMA0 if dev < 8 else NUMA1


def _pinned(cp, n, dtype):
    # Allocate page-locked host memory via cudaHostAlloc and view it as numpy.
    # (Avoids cupyx.empty_pinned, which has a concurrent-import race when called
    # from many threads at once.)
    nbytes = int(n) * np.dtype(dtype).itemsize
    mem = cp.cuda.alloc_pinned_memory(nbytes)
    a = np.frombuffer(mem, dtype=dtype, count=int(n))
    a[:] = 0
    return a


def _one_gpu_buffers(cp, dev, n, numa_bind):
    if numa_bind:
        try:
            os.sched_setaffinity(0, set(cpus_for_gpu(dev)))
        except Exception:
            pass
    with cp.cuda.Device(dev):
        d = cp.zeros(n, dtype=cp.int32)
        h_page = np.zeros(n, dtype=np.int32)
        h_pin = _pinned(cp, n, np.int32)
        stream = cp.cuda.Stream(non_blocking=True)
        cp.cuda.runtime.deviceSynchronize()
    return d, h_page, h_pin, stream


def _h2d(cp, dev, d, src, stream, pinned):
    with cp.cuda.Device(dev):
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        if pinned:
            d.set(src, stream=stream)
            stream.synchronize()
        else:
            d.set(src)  # pageable -> synchronous staged copy
            cp.cuda.runtime.deviceSynchronize()
        return time.perf_counter() - t0


def _d2h(cp, dev, d, dst, stream, pinned):
    with cp.cuda.Device(dev):
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        if pinned:
            d.get(out=dst, stream=stream)
            stream.synchronize()
        else:
            dst[...] = d.get()  # pageable
            cp.cuda.runtime.deviceSynchronize()
        return time.perf_counter() - t0


# --------------------------------------------------------------------------- #
# 16 single-GPU actors (one process per PCIe link)
# --------------------------------------------------------------------------- #
@ray.remote(num_gpus=1)
class OneGpu:
    def __init__(self, dev, n, numa_bind):
        import cupy as cp
        self.cp = cp
        self.dev = dev
        self.d, self.h_page, self.h_pin, self.stream = _one_gpu_buffers(cp, dev, n, numa_bind)

    def h2d(self, pinned):
        src = self.h_pin if pinned else self.h_page
        return _h2d(self.cp, self.dev, self.d, src, self.stream, pinned)

    def d2h(self, pinned):
        dst = self.h_pin if pinned else self.h_page
        return _d2h(self.cp, self.dev, self.d, dst, self.stream, pinned)

    def ping(self):
        return self.dev


# --------------------------------------------------------------------------- #
# 1 actor owning all 16 GPUs, threaded (the current design)
# --------------------------------------------------------------------------- #
@ray.remote(num_gpus=16)
class AllGpu:
    def __init__(self, ndev, n, numa_bind):
        import cupy as cp
        from concurrent.futures import ThreadPoolExecutor
        self.cp = cp
        self.ndev = ndev
        self.n = n
        self.numa_bind = numa_bind
        self.pool = ThreadPoolExecutor(max_workers=ndev)
        self.d = [None] * ndev
        self.h_page = [None] * ndev
        self.h_pin = [None] * ndev
        self.stream = [None] * ndev

        def build(dev):
            self.d[dev], self.h_page[dev], self.h_pin[dev], self.stream[dev] = \
                _one_gpu_buffers(cp, dev, n, numa_bind)

        list(self.pool.map(build, range(ndev)))

    def _par(self, fn):
        return list(self.pool.map(fn, range(self.ndev)))

    def h2d(self, pinned):
        cp = self.cp

        def go(dev):
            src = self.h_pin[dev] if pinned else self.h_page[dev]
            return _h2d(cp, dev, self.d[dev], src, self.stream[dev], pinned)

        # wall across all devices (concurrency-limited), measured here so we
        # don't pay Ray dispatch overhead in the number.
        t0 = time.perf_counter()
        self._par(go)
        return time.perf_counter() - t0

    def d2h(self, pinned):
        cp = self.cp

        def go(dev):
            dst = self.h_pin[dev] if pinned else self.h_page[dev]
            return _d2h(cp, dev, self.d[dev], dst, self.stream[dev], pinned)

        t0 = time.perf_counter()
        self._par(go)
        return time.perf_counter() - t0

    def ping(self):
        return self.ndev


# --------------------------------------------------------------------------- #
# marshalling cost: 16 Arrow columns -> host array the DMA can consume
# --------------------------------------------------------------------------- #
def marshalling_bench(rows_per_gpu, cols, ndev):
    """How long to turn 16 Arrow columns into a transferable host buffer."""
    import pyarrow as pa
    print("\n=== host marshalling cost (single process, one GPU's share) ===")
    # one GPU's worth of Arrow blocks: emulate as a pyarrow Table.
    data = {f"c{j}": np.zeros(rows_per_gpu, dtype=np.int32) for j in range(cols)}
    data["c0"] = np.random.default_rng(0).integers(0, 2 ** 31 - 1, rows_per_gpu, dtype=np.int32)
    tbl = pa.table(data)
    nbytes = rows_per_gpu * cols * 4

    # (a) current: build row-major (rows, cols) via per-column strided writes
    reps = 3
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        host = np.empty((rows_per_gpu, cols), dtype=np.int32)
        for j in range(cols):
            host[:, j] = tbl.column(j).to_numpy(zero_copy_only=False)
        best = min(best, time.perf_counter() - t0)
    print(f"row-major transpose assembly : {best * 1e3:8.1f} ms  ({nbytes / GiB / best:6.2f} GiB/s)")

    # (b) contiguous column copy into a column-major (cols, rows) buffer
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        host = np.empty((cols, rows_per_gpu), dtype=np.int32)
        for j in range(cols):
            host[j] = tbl.column(j).to_numpy(zero_copy_only=False)
        best = min(best, time.perf_counter() - t0)
    print(f"column-major contiguous copy : {best * 1e3:8.1f} ms  ({nbytes / GiB / best:6.2f} GiB/s)")

    # (c) zero-copy numpy views of the Arrow buffers (no copy at all)
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        views = [tbl.column(j).to_numpy(zero_copy_only=False) for j in range(cols)]
        best = min(best, time.perf_counter() - t0)
    print(f"zero-copy column views       : {best * 1e3:8.1f} ms  (just .to_numpy handles)")


def run(make_actors, label, n, ndev, trials):
    actors = make_actors()
    ray.get([a.ping.remote() for a in actors])
    total = n * 4 * ndev  # bytes across all GPUs

    def measure(op, pinned):
        # warmup
        ray.get([getattr(a, op).remote(pinned) for a in actors])
        wall_best = 1e9
        for _ in range(trials):
            t0 = time.perf_counter()
            per = ray.get([getattr(a, op).remote(pinned) for a in actors])
            wall = time.perf_counter() - t0
            wall_best = min(wall_best, wall)
        return wall_best, total / GiB / wall_best

    for pinned in (False, True):
        tag = "pinned " if pinned else "pageable"
        hb, hbw = measure("h2d", pinned)
        db, dbw = measure("d2h", pinned)
        print(f"{label:28s} {tag}  H2D {hb*1e3:8.1f} ms {hbw:7.1f} GiB/s   "
              f"D2H {db*1e3:8.1f} ms {dbw:7.1f} GiB/s")
    for a in actors:
        ray.kill(a)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=1024 * 1024 * 1024)
    p.add_argument("--cols", type=int, default=16)
    p.add_argument("--gpus", type=int, default=16)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--no-numa", action="store_true")
    args = p.parse_args()

    ndev = args.gpus
    rows_per_gpu = args.rows // ndev
    n = rows_per_gpu * args.cols  # int32 elements per GPU
    numa_bind = not args.no_numa

    ray.init(num_gpus=ndev)
    print(f"=== transfer_bench: {args.rows:,} rows x {args.cols} int32 = "
          f"{args.rows * args.cols * 4 / GiB:.1f} GiB across {ndev} GPUs ===")
    print(f"per-GPU payload: {n * 4 / GiB:.2f} GiB   numa_bind={numa_bind}")
    print(f"{'design':28s} {'host':8s}  {'H2D':>21s}   {'D2H':>21s}")

    run(lambda: [OneGpu.remote(0, n, numa_bind)],
        "1x single-GPU (no contention)", n, 1, args.trials)

    run(lambda: [OneGpu.remote(d, n, numa_bind) for d in range(ndev)],
        "16x single-GPU actors", n, ndev, args.trials)

    run(lambda: [AllGpu.remote(ndev, n, numa_bind)],
        "1x 16-GPU actor (threads)", n, ndev, args.trials)

    marshalling_bench(rows_per_gpu, args.cols, ndev)

    ray.shutdown()


if __name__ == "__main__":
    main()
