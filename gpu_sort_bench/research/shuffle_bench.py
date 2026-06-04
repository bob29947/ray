"""
Isolated SHUFFLE microbenchmark: Ray object-store all-to-all vs GPU P2P all-to-all.

This strips the sort algorithm away and measures only the redistribution of the
same 64 GiB across 16 workers (the all-to-all data movement), so we compare the
two interconnect mechanisms directly:

    ray  : object-store shuffle -- map tasks write per-destination partitions to
           the shared-memory object store, reduce tasks read them back.
           Measured with ds.random_shuffle().materialize() (a pure full shuffle,
           no sort comparisons).

    gpu  : P2P all-to-all over NVLink/NVSwitch. Each of 16 GPUs holds 1/16 of the
           data, pre-split into 16 equal contiguous chunks; the timed region is
           only the cudaMemcpyPeerAsync all-to-all (+ sync). No partition compute,
           no local sort -- just the transfer.

Timing contract (both): data already resident in memory -> reshuffled data
resident in memory. Warm, startup/build excluded.
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def gib(n):
    return f"{n / 2**30:.2f} GiB"


# --------------------------------------------------------------------------- #
# Ray object-store shuffle
# --------------------------------------------------------------------------- #
def run_ray(args):
    import pyarrow as pa
    import ray
    from ray.data import DataContext

    ray.init(object_store_memory=512 * 2**30)
    ctx = DataContext.get_current()
    ctx.enable_rich_progress_bars = False
    ctx.use_ray_tqdm = False

    rows_per_block = args.rows // args.blocks
    raw_bytes = args.rows * args.cols * 4
    cols = [f"c{i}" for i in range(args.cols)]
    rng = np.random.default_rng(0)

    print(f"building {args.blocks} blocks ({gib(raw_bytes)}) in object store...")
    refs = []
    for i in range(args.blocks):
        data = {"c0": rng.integers(0, 2**31 - 1, rows_per_block, dtype=np.int32)}
        for c in cols[1:]:
            data[c] = np.zeros(rows_per_block, dtype=np.int32)
        refs.append(ray.put(pa.table(data)))
    # Materialize the input once so each trial shuffles already-resident blocks
    # (matches the timing contract) and avoids map/shuffle fusion + its OOM risk.
    ds = ray.data.from_arrow_refs(refs).materialize()

    # Warm up: first shuffle pays worker spin-up / object-store first-touch.
    print("warmup shuffle...", flush=True)
    import gc
    sh = ds.random_shuffle().materialize()
    del sh
    gc.collect()

    times = []
    for t in range(args.trials):
        t0 = time.perf_counter()
        sh = ds.random_shuffle().materialize()
        t1 = time.perf_counter()
        dt = t1 - t0
        times.append(dt)
        print(f"trial {t}: {dt:.3f} s   {(raw_bytes / 2**30) / dt:.2f} GiB/s (dataset)", flush=True)
        del sh
        gc.collect()

    report("ray object-store shuffle", times, raw_bytes, args)


# --------------------------------------------------------------------------- #
# GPU P2P all-to-all shuffle
# --------------------------------------------------------------------------- #
def run_gpu(args):
    import cupy as cp

    devices = list(range(args.gpus))
    nd = len(devices)
    assert args.rows % nd == 0
    rows_per_gpu = args.rows // nd
    assert rows_per_gpu % nd == 0, "rows_per_gpu must split evenly into nd chunks"
    chunk = rows_per_gpu // nd          # rows each src sends to each dst
    cols = args.cols
    itemsize = 4 * cols
    raw_bytes = args.rows * cols * 4
    # bytes that actually cross the fabric (everything except the local 1/16)
    moved_bytes = raw_bytes * (nd - 1) // nd

    pool = ThreadPoolExecutor(max_workers=nd)

    def par(fn):
        return [f.result() for f in [pool.submit(fn, s, d) for s, d in enumerate(devices)]]

    # peer access
    def peer(slot, dev):
        with cp.cuda.Device(dev):
            for p in devices:
                if p == dev:
                    continue
                try:
                    cp.cuda.runtime.deviceEnablePeerAccess(p)
                except cp.cuda.runtime.CUDARuntimeError as e:
                    if "AlreadyEnabled" not in str(e):
                        raise
    par(peer)

    # input: each GPU holds rows_per_gpu x cols int32, already "partitioned" into
    # nd equal contiguous chunks (chunk j -> destination j).
    X = [None] * nd
    R = [None] * nd
    copy_streams = [None] * nd

    def build(slot, dev):
        with cp.cuda.Device(dev):
            rs = cp.random.RandomState(dev)
            x = cp.zeros((rows_per_gpu, cols), dtype=cp.int32)
            x[:, 0] = rs.randint(0, 2**31 - 1, size=rows_per_gpu, dtype=cp.int32)
            X[slot] = x
            R[slot] = cp.empty((rows_per_gpu, cols), dtype=cp.int32)
            copy_streams[slot] = cp.cuda.Stream(non_blocking=True)
            cp.cuda.runtime.deviceSynchronize()
    par(build)

    print(f"gpu input resident: {nd} x {gib(rows_per_gpu * cols * 4)} = {gib(raw_bytes)}")
    print(f"all-to-all chunk: {chunk:,} rows ({gib(chunk * itemsize)}) per (src,dst) pair")

    def exchange(slot, dev):
        i = slot
        with cp.cuda.Device(dev):
            st = copy_streams[i]
            base_src = X[i].data.ptr
            for j in range(nd):
                nbytes = chunk * itemsize
                src_ptr = base_src + j * chunk * itemsize          # chunk for dst j
                dst_ptr = R[j].data.ptr + i * chunk * itemsize     # land at src-i slot
                cp.cuda.runtime.memcpyPeerAsync(dst_ptr, devices[j], src_ptr, dev, nbytes, st.ptr)
            st.synchronize()

    def sync(slot, dev):
        with cp.cuda.Device(dev):
            cp.cuda.runtime.deviceSynchronize()

    # warmup
    par(exchange)
    par(sync)

    times = []
    for t in range(args.trials):
        par(sync)
        t0 = time.perf_counter()
        par(exchange)
        par(sync)
        t1 = time.perf_counter()
        dt = t1 - t0
        times.append(dt)
        print(f"trial {t}: {dt * 1e3:8.3f} ms   dataset={ (raw_bytes / 2**30) / dt:7.1f} GiB/s   "
              f"fabric={ (moved_bytes / 2**30) / dt:7.1f} GiB/s")

    report("gpu P2P all-to-all (NVLink/NVSwitch)", times, raw_bytes, args, moved_bytes)
    pool.shutdown()


# --------------------------------------------------------------------------- #
# rapidsmpf Shuffler all-to-all (over UCXX / NVLink) -- the SAME mechanism the
# general GPU sort uses, measured in isolation. N one-GPU actors connected by a
# UCXX communicator; each holds rows_per_gpu x cols int32 pre-split into N equal
# contiguous chunks (chunk j -> destination j), exactly like the `gpu` backend.
# The timed region is the rapidsmpf insert -> all-to-all -> extract (packed),
# excluding the local pre-pack and the post-unpack, to isolate the transfer.
# --------------------------------------------------------------------------- #
def _build_shuffle_actor():
    import ray
    from rapidsmpf.utils.ray_utils import BaseShufflingActor

    @ray.remote(num_gpus=1, num_cpus=4)
    class _ShuffleActor(BaseShufflingActor):
        def __init__(self, nranks):
            super().__init__(nranks)
            import os
            try:
                from ray.data._internal.planner.gpu_sort_general import _ucx_env
                for k, v in _ucx_env().items():
                    os.environ.setdefault(k, v)
            except Exception:
                os.environ.setdefault("UCX_TLS", "cuda_copy,cuda_ipc,sm,tcp")
            self.df = None
            self.splits = None

        def setup_worker(self, root_address_bytes):
            super().setup_worker(root_address_bytes)
            import rmm
            from rapidsmpf.memory.buffer_resource import BufferResource
            from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor

            total = rmm.mr.available_device_memory()[1]
            mr = RmmResourceAdaptor(
                rmm.mr.PoolMemoryResource(
                    rmm.mr.CudaMemoryResource(),
                    initial_pool_size=(int(total * 0.5) // 256) * 256,
                    maximum_pool_size=(int(total * 0.8) // 256) * 256,
                )
            )
            rmm.mr.set_current_device_resource(mr)
            self.br = BufferResource(mr)

        def prepare(self, rows_per_gpu, cols):
            import cudf
            import cupy as cp

            nd = self.nranks()
            assert rows_per_gpu % nd == 0
            chunk = rows_per_gpu // nd
            data = {"c0": cp.random.randint(0, 2**31 - 1, rows_per_gpu, dtype=cp.int32)}
            for j in range(1, cols):
                data[f"c{j}"] = cp.zeros(rows_per_gpu, dtype=cp.int32)
            self.df = cudf.DataFrame(data)
            self.splits = [chunk * (j + 1) for j in range(nd - 1)]
            cp.cuda.runtime.deviceSynchronize()

        def pack(self):
            from rmm.pylibrmm.stream import DEFAULT_STREAM
            from rapidsmpf.integrations.cudf.partition import split_and_pack
            from rapidsmpf.utils.cudf import cudf_to_pylibcudf_table

            self.packed = split_and_pack(
                cudf_to_pylibcudf_table(self.df), self.splits, DEFAULT_STREAM, self.br
            )

        def run_shuffle(self, op_id):
            import cupy as cp

            nd = self.nranks()
            shuffler = self.create_shuffler(
                op_id, total_num_partitions=nd, buffer_resource=self.br
            )
            shuffler.insert_chunks(self.packed)
            shuffler.insert_finished(list(range(nd)))
            received = 0
            while not shuffler.finished():
                pid = shuffler.wait_any()
                received += len(shuffler.extract(pid))  # keep packed (no unpack)
            shuffler.shutdown()
            cp.cuda.runtime.deviceSynchronize()
            return received

    return _ShuffleActor


def run_rapidsmpf(args):
    import ray
    from rapidsmpf.integrations.ray import setup_ray_ucxx_cluster

    devices = args.gpus
    rows_per_gpu = args.rows // devices
    assert args.rows % devices == 0
    assert rows_per_gpu % devices == 0, "rows_per_gpu must split evenly into nd chunks"
    cols = args.cols
    raw_bytes = args.rows * cols * 4
    moved_bytes = raw_bytes * (devices - 1) // devices

    ray.init(object_store_memory=64 * 2**30)

    cls = _build_shuffle_actor()
    actors = setup_ray_ucxx_cluster(cls, devices)
    ray.get([a.prepare.remote(rows_per_gpu, cols) for a in actors])
    print(f"rapidsmpf input resident: {devices} x "
          f"{gib(rows_per_gpu * cols * 4)} = {gib(raw_bytes)}")
    print(f"all-to-all chunk: {rows_per_gpu // devices:,} rows per (src,dst) pair")

    op = [0]

    def one_trial():
        op[0] += 1
        ray.get([a.pack.remote() for a in actors])  # local pre-pack (untimed)
        t0 = time.perf_counter()
        ray.get([a.run_shuffle.remote(op[0]) for a in actors])  # timed all-to-all
        return time.perf_counter() - t0

    print("warmup shuffle...", flush=True)
    one_trial()

    times = []
    for t in range(args.trials):
        dt = one_trial()
        times.append(dt)
        print(f"trial {t}: {dt * 1e3:8.3f} ms   dataset={(raw_bytes / 2**30) / dt:7.1f} "
              f"GiB/s   fabric={(moved_bytes / 2**30) / dt:7.1f} GiB/s", flush=True)

    report("rapidsmpf Shuffler all-to-all (UCXX/NVLink)", times, raw_bytes, args,
           moved_bytes)
    try:
        from ray.data._internal.planner.gpu_sort_general import kill_actor_pool  # noqa
    except Exception:
        pass
    ray.shutdown()


def report(label, times, raw_bytes, args, moved_bytes=None):
    best = min(times)
    mean = sum(times) / len(times)
    print(f"\n=== {label} ===")
    print(f"dataset: {gib(raw_bytes)}")
    print(f"best: {best * 1e3:.3f} ms   mean: {mean * 1e3:.3f} ms")
    print(f"dataset throughput (best): {(raw_bytes / 2**30) / best:.2f} GiB/s")
    if moved_bytes is not None:
        print(f"fabric bytes moved: {gib(moved_bytes)}")
        print(f"fabric throughput (best): {(moved_bytes / 2**30) / best:.2f} GiB/s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", required=True, choices=["ray", "gpu", "rapidsmpf"])
    p.add_argument("--rows", type=int, default=1024 * 1024 * 1024)
    p.add_argument("--cols", type=int, default=16)
    p.add_argument("--blocks", type=int, default=256, help="ray input blocks")
    p.add_argument("--gpus", type=int, default=16)
    p.add_argument("--trials", type=int, default=3)
    args = p.parse_args()

    print(f"=== shuffle_bench backend={args.backend} ===")
    print(f"rows={args.rows:,} cols={args.cols} -> {gib(args.rows * args.cols * 4)}")
    if args.backend == "ray":
        run_ray(args)
    elif args.backend == "rapidsmpf":
        run_rapidsmpf(args)
    else:
        run_gpu(args)


if __name__ == "__main__":
    main()
