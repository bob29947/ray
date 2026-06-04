"""Fast H2D strategy probe (no full Ray sort, no dataset rebuild each time).

The full benchmark showed D2H is at the hardware ceiling (~48 GiB/s) but H2D
sits at ~30 GiB/s. Input arrives as many separate Arrow column buffers (16
blocks x 16 cols per GPU), so the question is how to feed them to the device
fastest. Compare, inside one 16-GPU actor (the real design), threaded:

  a) pageable .set() per (block,col) into a column-major device array + GPU
     transpose to row-major   [what gpu_sort.py does today]
  b) stage Arrow -> reused pinned (column-major) host buffer, one big async
     pinned H2D + GPU transpose
  c) same as (b) but skip the transpose (isolates the GPU transpose cost)
  d) pageable .set() per (block,col), NO transpose (isolates transpose for (a))

Reports effective GiB/s for the whole payload across all GPUs.
"""

import argparse
import os
import time

os.environ.setdefault("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "1")

import numpy as np
import ray

GiB = 2 ** 30
NUMA0 = list(range(0, 24)) + list(range(48, 72))
NUMA1 = list(range(24, 48)) + list(range(72, 96))


@ray.remote(num_gpus=16)
class Probe:
    def __init__(self, ndev, blocks_per_gpu, rows_per_block, cols):
        import cupy as cp
        import pyarrow as pa
        from concurrent.futures import ThreadPoolExecutor

        self.cp = cp
        self.pa = pa
        self.ndev = ndev
        self.cols = cols
        self.bpg = blocks_per_gpu
        self.rpb = rows_per_block
        self.rows = blocks_per_gpu * rows_per_block
        self.pool = ThreadPoolExecutor(max_workers=ndev)
        self.streams = [None] * ndev
        self.Xc = [None] * ndev          # column-major device buffer
        self.pin = [None] * ndev         # pinned host staging (cols, rows)
        self.blocks = [None] * ndev      # list of arrow tables per gpu

        def setup(dev):
            cpus = NUMA0 if dev < 8 else NUMA1
            try:
                os.sched_setaffinity(0, set(cpus))
            except Exception:
                pass
            with cp.cuda.Device(dev):
                rng = np.random.default_rng(dev)
                tbls = []
                for _ in range(blocks_per_gpu):
                    data = {"c0": rng.integers(0, 2 ** 31 - 1, rows_per_block, dtype=np.int32)}
                    for j in range(1, cols):
                        data[f"c{j}"] = np.zeros(rows_per_block, dtype=np.int32)
                    tbls.append(pa.table(data))
                self.blocks[dev] = tbls
                self.Xc[dev] = cp.empty((cols, self.rows), dtype=cp.int32)
                mem = cp.cuda.alloc_pinned_memory(cols * self.rows * 4)
                self.pin[dev] = np.frombuffer(mem, dtype=np.int32,
                                              count=cols * self.rows).reshape(cols, self.rows)
                self.pin[dev][:] = 0  # fault in the pages now
                self.streams[dev] = cp.cuda.Stream(non_blocking=True)
                cp.cuda.runtime.deviceSynchronize()

        list(self.pool.map(setup, range(ndev)))

    def _par(self, fn):
        return list(self.pool.map(fn, range(self.ndev)))

    def _run(self, fn):
        # warm
        self._par(fn)
        self._par(lambda d: self.cp.cuda.Device(d).synchronize())
        best = 1e9
        for _ in range(4):
            self._par(lambda d: self.cp.cuda.Device(d).synchronize())
            t0 = time.perf_counter()
            self._par(fn)
            self._par(lambda d: self.cp.cuda.Device(d).synchronize())
            best = min(best, time.perf_counter() - t0)
        total = self.rows * self.cols * 4 * self.ndev
        return best, total / GiB / best

    def pageable_set(self, transpose):
        cp = self.cp

        def go(dev):
            with cp.cuda.Device(dev):
                Xc = self.Xc[dev]
                off = 0
                for t in self.blocks[dev]:
                    n = t.num_rows
                    for j in range(self.cols):
                        Xc[j, off:off + n].set(t.column(j).to_numpy(zero_copy_only=False))
                    off += n
                if transpose:
                    _ = cp.ascontiguousarray(Xc.T)
        return self._run(go)

    def pinned_stage(self, transpose):
        cp = self.cp

        def go(dev):
            with cp.cuda.Device(dev):
                Xc = self.Xc[dev]
                pin = self.pin[dev]
                st = self.streams[dev]
                off = 0
                for t in self.blocks[dev]:
                    n = t.num_rows
                    for j in range(self.cols):
                        pin[j, off:off + n] = t.column(j).to_numpy(zero_copy_only=False)
                    off += n
                Xc.set(pin, stream=st)  # one big async pinned H2D
                st.synchronize()
                if transpose:
                    _ = cp.ascontiguousarray(Xc.T)
        return self._run(go)

    def ping(self):
        return self.ndev


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", type=int, default=16)
    p.add_argument("--blocks-per-gpu", type=int, default=16)
    p.add_argument("--rows-per-block", type=int, default=4 * 1024 * 1024)
    p.add_argument("--cols", type=int, default=16)
    args = p.parse_args()

    ray.init(num_gpus=args.gpus)
    total = args.gpus * args.blocks_per_gpu * args.rows_per_block * args.cols * 4
    print(f"=== h2d_probe: {args.gpus} GPUs x {args.blocks_per_gpu} blocks x "
          f"{args.rows_per_block:,} rows x {args.cols} cols = {total / GiB:.1f} GiB ===")
    probe = Probe.remote(args.gpus, args.blocks_per_gpu, args.rows_per_block, args.cols)
    ray.get(probe.ping.remote())

    for name, ref in [
        ("a) pageable .set() per-col + transpose", probe.pageable_set.remote(True)),
        ("d) pageable .set() per-col, no transpose", probe.pageable_set.remote(False)),
        ("b) pinned stage + big async + transpose", probe.pinned_stage.remote(True)),
        ("c) pinned stage + big async, no transpose", probe.pinned_stage.remote(False)),
    ]:
        best, bw = ray.get(ref)
        print(f"{name:46s}  {best * 1e3:8.1f} ms   {bw:7.1f} GiB/s")

    ray.shutdown()


if __name__ == "__main__":
    main()
