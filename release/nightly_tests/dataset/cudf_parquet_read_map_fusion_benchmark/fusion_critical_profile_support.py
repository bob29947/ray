"""Benchmark-only timing hooks for the committed cuDF read fusion path.

The hooks aggregate timings inside each worker and print one JSON record per
actor initialization and fused task.  They do not change data, scheduling, or
the public Dataset call shape.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Dict, Iterator


MARKER = "FUSION_CRITICAL_PROFILE="
PROFILE_ENV = "RAY_DATA_FUSION_CRITICAL_PROFILE"
LAZY_PROFILE_ENV = "RAY_DATA_FUSION_CRITICAL_PROFILE_LAZY"
RUN_ID_ENV = "RAY_DATA_FUSION_CRITICAL_PROFILE_RUN_ID"

_INSTALLED = False
_TASK_HOOKS_INSTALLED = False
_PROFILE_INSTALLED_S = time.perf_counter()
_PROCESS_CREATED_WALL_S = None
_TASKS: Dict[tuple[str, int], Dict[str, Any]] = {}


def _process_created_wall_s() -> float | None:
    global _PROCESS_CREATED_WALL_S
    if _PROCESS_CREATED_WALL_S is None:
        try:
            import psutil

            _PROCESS_CREATED_WALL_S = psutil.Process(os.getpid()).create_time()
        except Exception:
            _PROCESS_CREATED_WALL_S = 0.0
    return _PROCESS_CREATED_WALL_S or None


def _runtime_fields() -> Dict[str, Any]:
    fields: Dict[str, Any] = {
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "profile_run_id": os.environ.get(RUN_ID_ENV),
    }
    try:
        import ray

        context = ray.get_runtime_context()
        fields["node_id"] = context.get_node_id()
        actor_id = context.get_actor_id()
        fields["actor_id"] = str(actor_id) if actor_id is not None else None
    except Exception:
        fields["node_id"] = None
        fields["actor_id"] = None
    return fields


def _emit(event: str, **fields: Any) -> None:
    record = {
        "event": event,
        "wall_s": time.time(),
        "perf_s": time.perf_counter(),
        **_runtime_fields(),
        **fields,
    }
    print(MARKER + json.dumps(record, sort_keys=True), flush=True)


def _task_key() -> tuple[str, int] | None:
    try:
        from ray.data._internal.execution.interfaces import TaskContext

        context = TaskContext.get_current()
    except Exception:
        return None
    if context is None:
        return None
    return context.op_name, context.task_idx


def _new_task(op_name: str, task_idx: int) -> Dict[str, Any]:
    task = {
        "op_name": op_name,
        "task_idx": task_idx,
        "start_perf_s": time.perf_counter(),
        "stages": defaultdict(
            lambda: {"calls": 0, "elapsed_s": 0.0, "rows": 0, "bytes": 0}
        ),
    }
    _TASKS[(op_name, task_idx)] = task
    return task


def _record(
    stage: str,
    elapsed_s: float,
    *,
    calls: int = 1,
    rows: int = 0,
    size_bytes: int = 0,
) -> None:
    key = _task_key()
    if key is None:
        # Ray's single-threaded callable-class wrapper executes the user method
        # on a helper thread, where TaskContext's thread-local value is absent.
        # This profiler is only valid for the required one-task-at-a-time actor,
        # so the process's sole in-flight task is unambiguous.
        if len(_TASKS) != 1:
            return
        key = next(iter(_TASKS))
    task = _TASKS.get(key)
    if task is None:
        task = _new_task(*key)
    aggregate = task["stages"][stage]
    aggregate["calls"] += calls
    aggregate["elapsed_s"] += elapsed_s
    aggregate["rows"] += rows
    aggregate["bytes"] += size_bytes


def _batch_nbytes(batch: Any) -> int:
    if not isinstance(batch, Mapping):
        return 0
    total = 0
    for value in batch.values():
        size = getattr(value, "nbytes", 0)
        if isinstance(size, int):
            total += size
    return total


def _install_actor_hooks() -> None:
    from ray.data._internal.execution.operators.actor_pool_map_operator import (
        _MapWorker,
    )

    original_init = _MapWorker.__init__
    original_submit = _MapWorker.submit

    def profiled_init(self, *args, **kwargs):
        # A cloud profiling job applies the setup hook to every Ray worker.
        # Delay CUDA-related imports until this process is actually a map actor.
        if os.environ.get(LAZY_PROFILE_ENV) == "1":
            _install_task_hooks()
        started = time.perf_counter()
        success = False
        try:
            original_init(self, *args, **kwargs)
            success = True
        finally:
            finished = time.perf_counter()
            created = _process_created_wall_s()
            _emit(
                "actor_ready",
                success=success,
                init_start_perf_s=started,
                ready_perf_s=finished,
                init_elapsed_s=finished - started,
                profile_install_to_init_s=started - _PROFILE_INSTALLED_S,
                process_age_at_ready_s=(
                    time.time() - created if created is not None else None
                ),
            )

    def profiled_submit(
        self, data_context, ctx, *blocks, slices=None, **kwargs
    ) -> Iterator[Any]:
        task = _new_task(ctx.op_name, ctx.task_idx)
        inner = original_submit(
            self,
            data_context,
            ctx,
            *blocks,
            slices=slices,
            **kwargs,
        )
        value_to_send = None
        first = True
        success = False
        try:
            while True:
                try:
                    if first:
                        item = next(inner)
                        first = False
                    else:
                        item = inner.send(value_to_send)
                except StopIteration:
                    success = True
                    break

                stage = (
                    "metadata_yield_suspended"
                    if isinstance(item, (bytes, bytearray))
                    else "block_yield_suspended"
                )
                suspended = time.perf_counter()
                value_to_send = yield item
                _record(stage, time.perf_counter() - suspended)
        finally:
            finished = time.perf_counter()
            stages = {name: dict(values) for name, values in task["stages"].items()}
            _TASKS.pop((ctx.op_name, ctx.task_idx), None)
            _emit(
                "task_complete",
                success=success,
                op_name=ctx.op_name,
                task_idx=ctx.task_idx,
                task_start_perf_s=task["start_perf_s"],
                task_end_perf_s=finished,
                task_elapsed_s=finished - task["start_perf_s"],
                stages=stages,
            )

    _MapWorker.__init__ = profiled_init
    _MapWorker.submit = profiled_submit


def _install_fusion_hooks() -> None:
    from ray.data._internal.logical.rules.cudf_parquet_read_fusion import (
        _CudfBatchMapTransformFn,
        _CudfParquetReader,
    )

    original_read = _CudfParquetReader.__call__
    original_pre_process = _CudfBatchMapTransformFn._pre_process

    def profiled_read(self, *args, **kwargs):
        iterator = iter(original_read(self, *args, **kwargs))
        while True:
            started = time.perf_counter()
            try:
                frame = next(iterator)
            except StopIteration:
                return
            _record("cudf_read", time.perf_counter() - started, rows=len(frame))
            yield frame

    def profiled_pre_process(self, *args, **kwargs):
        iterator = iter(original_pre_process(self, *args, **kwargs))
        while True:
            started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                return
            _record(
                "batch_ready_including_read",
                time.perf_counter() - started,
                rows=len(batch),
            )
            yield batch

    _CudfParquetReader.__call__ = profiled_read
    _CudfBatchMapTransformFn._pre_process = profiled_pre_process


def _install_udf_hook() -> None:
    from src.ray_tokenize import GPUTokenizer

    original_init = GPUTokenizer.__init__
    original_call = GPUTokenizer.__call__

    def profiled_init(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return original_init(self, *args, **kwargs)
        finally:
            _emit(
                "tokenizer_initialized",
                init_elapsed_s=time.perf_counter() - started,
            )

    def profiled_call(self, batch, *args, **kwargs):
        started = time.perf_counter()
        result = original_call(self, batch, *args, **kwargs)
        _record(
            "udf",
            time.perf_counter() - started,
            rows=len(batch),
            size_bytes=_batch_nbytes(result),
        )
        return result

    GPUTokenizer.__init__ = profiled_init
    GPUTokenizer.__call__ = profiled_call


def _install_output_hooks() -> None:
    from ray.data._internal.output_buffer import BlockOutputBuffer
    from ray.data.block import BlockAccessor

    original_batch_to_block = BlockAccessor.batch_to_block.__func__
    original_add_batch = BlockOutputBuffer.add_batch
    original_next = BlockOutputBuffer.next

    def profiled_batch_to_block(cls, batch, block_type=None):
        started = time.perf_counter()
        block = original_batch_to_block(cls, batch, block_type)
        _record(
            "batch_to_block",
            time.perf_counter() - started,
            rows=(len(next(iter(batch.values()))) if isinstance(batch, Mapping) else 0),
            size_bytes=_batch_nbytes(batch),
        )
        return block

    def profiled_add_batch(self, batch):
        started = time.perf_counter()
        result = original_add_batch(self, batch)
        _record(
            "output_buffer_add_batch",
            time.perf_counter() - started,
            rows=(len(next(iter(batch.values()))) if isinstance(batch, Mapping) else 0),
            size_bytes=_batch_nbytes(batch),
        )
        return result

    def profiled_next(self):
        started = time.perf_counter()
        block = original_next(self)
        elapsed_s = time.perf_counter() - started
        try:
            accessor = BlockAccessor.for_block(block)
            rows = accessor.num_rows()
            size_bytes = accessor.size_bytes()
        except Exception:
            rows = size_bytes = 0
        _record(
            "output_buffer_next",
            elapsed_s,
            rows=rows,
            size_bytes=size_bytes,
        )
        return block

    BlockAccessor.batch_to_block = classmethod(profiled_batch_to_block)
    BlockOutputBuffer.add_batch = profiled_add_batch
    BlockOutputBuffer.next = profiled_next


def _install_task_hooks() -> None:
    """Install hooks needed only by a fused map actor."""

    global _TASK_HOOKS_INSTALLED
    if _TASK_HOOKS_INSTALLED:
        return
    _TASK_HOOKS_INSTALLED = True
    _install_fusion_hooks()
    _install_udf_hook()
    _install_output_hooks()


def install_profile_hooks() -> None:
    """Install the timing wrappers once in the current Python process."""

    global _INSTALLED
    if _INSTALLED or os.environ.get(PROFILE_ENV) != "1":
        return
    _INSTALLED = True
    _install_actor_hooks()
    if os.environ.get(LAZY_PROFILE_ENV) != "1":
        _install_task_hooks()
