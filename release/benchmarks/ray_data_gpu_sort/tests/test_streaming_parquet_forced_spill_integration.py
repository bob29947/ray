"""Opt-in CPU push-sort spill proof for the streaming Parquet benchmark.

This test is intentionally excluded from ordinary CPU CI.  On a DGX, it runs
one lazy datasource -> blocking push-based global sort -> transactional Parquet
sink pipeline with an object store much smaller than the input.  The test only
accepts the run when Ray's raylet counters report physical object spilling and
the committed Parquet output passes the benchmark's bounded durable validator.

Run it from an exact benchmark checkout with a short, RAID-backed test root::

    RAY_DATA_RUN_CPU_PUSH_FORCED_SPILL_INTEGRATION=1 \
    RAY_DATA_FORCED_SPILL_TEST_ROOT=/raid/rgs-tests \
      PYTHONPATH=. .venv/bin/python -m pytest -sv \
      release/benchmarks/ray_data_gpu_sort/tests/\
test_streaming_parquet_forced_spill_integration.py
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import time
from typing import Iterator, Optional
import uuid

import numpy as np
import pyarrow as pa
import pytest

import ray
from ray.data import DataContext
from ray.data._internal.execution.operators.base_physical_operator import (
    get_last_all_to_all_stats,
    reset_last_all_to_all_stats,
)
from ray.data._internal.planner.exchange.push_based_shuffle_task_scheduler import (
    get_last_push_based_shuffle_stats,
    reset_last_push_based_shuffle_stats,
)
from ray.data.block import BlockMetadata
from ray.data.context import ShuffleStrategy
from ray.data.datasource import Datasource, ReadTask

from release.benchmarks.ray_data_gpu_sort.streaming_parquet_e2e_worker import (
    _frequency_digest,
    _validate_durable_parquet,
)
from release.benchmarks.ray_data_gpu_sort.streaming_parquet_sink import (
    StreamingParquetDatasink,
)


_RUN_INTEGRATION = (
    os.environ.get("RAY_DATA_RUN_CPU_PUSH_FORCED_SPILL_INTEGRATION", "0") == "1"
)
_INTEGRATION_ONLY = pytest.mark.skipif(
    not _RUN_INTEGRATION,
    reason=("set RAY_DATA_RUN_CPU_PUSH_FORCED_SPILL_INTEGRATION=1 on the DGX"),
)

_OBJECT_STORE_BYTES = 96 << 20
_BLOCKS = 24
_ROWS_PER_BLOCK = 8_192
_PAYLOAD_BYTES_PER_ROW = 2_048
_TOTAL_ROWS = _BLOCKS * _ROWS_PER_BLOCK
_ORIGINS = (
    "ATL",
    "BOS",
    "DEN",
    "DFW",
    "JFK",
    "LAX",
    "MIA",
    "ORD",
    "PHX",
    "SEA",
    "SFO",
)
_SCHEMA = pa.schema(
    [
        pa.field("Origin", pa.string(), nullable=False),
        pa.field("row_id", pa.int64(), nullable=False),
        pa.field("payload", pa.string(), nullable=True),
    ],
    metadata={b"ray-data-benchmark": b"cpu-push-forced-spill"},
)


def _payload(block_index: int) -> str:
    prefix = f"block-{block_index:02d}-"
    return prefix + "x" * (_PAYLOAD_BYTES_PER_ROW - len(prefix))


class _ClaimedOriginDatasource(Datasource):
    """A lazy, non-replayable synthetic source with per-reader disk claims."""

    def __init__(self, claim_directory: Path) -> None:
        self._claim_directory = str(claim_directory)

    def estimate_inmemory_data_size(self) -> Optional[int]:
        # The actual Arrow size is recorded by Ray after each read.  This
        # conservative estimate is only used for planning before execution.
        return _TOTAL_ROWS * (_PAYLOAD_BYTES_PER_ROW + 32)

    def get_read_tasks(
        self,
        parallelism: int,
        per_task_row_limit: Optional[int] = None,
        data_context: Optional[DataContext] = None,
    ) -> list[ReadTask]:
        del parallelism, data_context
        tasks: list[ReadTask] = []
        estimated_block_bytes = self.estimate_inmemory_data_size() // _BLOCKS

        for block_index in range(_BLOCKS):

            def read(index: int = block_index) -> Iterator[pa.Table]:
                claims = Path(self._claim_directory)
                claims.mkdir(parents=True, exist_ok=True)
                claim = claims / f"block-{index:05d}.claim"
                descriptor = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.close(descriptor)

                first = index * _ROWS_PER_BLOCK
                row_ids = np.arange(first, first + _ROWS_PER_BLOCK, dtype=np.int64)
                origins = [
                    _ORIGINS[(int(row_id) * 17 + index * 3) % len(_ORIGINS)]
                    for row_id in row_ids
                ]
                table = pa.Table.from_arrays(
                    [
                        pa.array(origins, type=pa.string()),
                        pa.array(row_ids, type=pa.int64()),
                        pa.array(
                            [_payload(index)] * _ROWS_PER_BLOCK,
                            type=pa.string(),
                        ),
                    ],
                    schema=_SCHEMA,
                )
                done = claims / f"block-{index:05d}.done"
                done.write_text(
                    json.dumps(
                        {"rows": table.num_rows, "bytes": table.nbytes},
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                yield table

            tasks.append(
                ReadTask(
                    read,
                    BlockMetadata(
                        num_rows=_ROWS_PER_BLOCK,
                        size_bytes=estimated_block_bytes,
                        input_files=None,
                        exec_stats=None,
                    ),
                    per_task_row_limit=per_task_row_limit,
                )
            )
        return tasks

    def num_rows(self) -> int:
        return _TOTAL_ROWS


def _spill_snapshot() -> dict[str, int | float]:
    """Read cumulative spill counters from the one local raylet."""

    from ray._private.internal_api import get_state_from_address, node_stats

    state = get_state_from_address(ray.get_runtime_context().gcs_address)
    alive = [node for node in state.node_table() if node["Alive"]]
    assert len(alive) == 1, alive
    node = alive[0]
    stats = node_stats(
        node_manager_address=node["NodeManagerAddress"],
        node_manager_port=node["NodeManagerPort"],
        include_memory_info=False,
    ).store_stats
    fields = (
        "spilled_bytes_total",
        "spilled_objects_total",
        "restored_bytes_total",
        "restored_objects_total",
        "spill_time_total_s",
        "restore_time_total_s",
    )
    return {name: getattr(stats, name) for name in fields}


def _stable_spill_snapshot() -> dict[str, int | float]:
    prior = _spill_snapshot()
    stable = 0
    deadline = time.monotonic() + 5.0
    while stable < 2 and time.monotonic() < deadline:
        time.sleep(0.2)
        current = _spill_snapshot()
        stable = stable + 1 if current == prior else 0
        prior = current
    return prior


@contextmanager
def _push_sort_context() -> Iterator[None]:
    context = DataContext.get_current()
    old_shuffle = context.shuffle_strategy
    old_preserve_order = context.execution_options.preserve_order
    old_min_block = context.target_min_block_size
    old_max_block = context.target_max_block_size
    context.shuffle_strategy = ShuffleStrategy.SORT_SHUFFLE_PUSH_BASED
    context.execution_options.preserve_order = True
    context.target_min_block_size = 1 << 20
    context.target_max_block_size = 32 << 20
    try:
        yield
    finally:
        context.shuffle_strategy = old_shuffle
        context.execution_options.preserve_order = old_preserve_order
        context.target_min_block_size = old_min_block
        context.target_max_block_size = old_max_block


def _expected_frequencies() -> Counter[str]:
    result: Counter[str] = Counter()
    for block_index in range(_BLOCKS):
        first = block_index * _ROWS_PER_BLOCK
        result.update(
            _ORIGINS[(row_id * 17 + block_index * 3) % len(_ORIGINS)]
            for row_id in range(first, first + _ROWS_PER_BLOCK)
        )
    return result


def _validated_test_root() -> Path:
    configured = os.environ.get("RAY_DATA_FORCED_SPILL_TEST_ROOT")
    if not configured:
        pytest.fail(
            "RAY_DATA_FORCED_SPILL_TEST_ROOT must name a short directory under "
            "/raid (for example /raid/rgs-tests)"
        )
    root = Path(configured).resolve()
    raid = Path("/raid").resolve()
    if root == raid or not root.is_relative_to(raid):
        pytest.fail("RAY_DATA_FORCED_SPILL_TEST_ROOT must be a child of /raid")
    if len(str(root)) > 24:
        pytest.fail(
            "RAY_DATA_FORCED_SPILL_TEST_ROOT is too long for Ray Unix sockets; "
            "use a short path such as /raid/rgs-tests"
        )
    root.mkdir(parents=True, exist_ok=True)
    return root


@_INTEGRATION_ONLY
def test_cpu_push_sort_forces_ray_spill_and_commits_sorted_parquet() -> None:
    base = _validated_test_root()
    token = uuid.uuid4().hex[:10]
    runtime = base / f"cpu-spill-{token}"
    socket_root = Path("/dev/shm/rgs") / token
    runtime.mkdir(exist_ok=False)
    socket_root.parent.mkdir(parents=True, exist_ok=True)
    spill = runtime / "ray-spill"
    output = runtime / "output"
    claims = runtime / "source-claims"
    spill.mkdir()

    ray.shutdown()
    prior_pythonpath = os.environ.get("PYTHONPATH")
    test_module_root = str(Path(__file__).resolve().parent)
    os.environ["PYTHONPATH"] = os.pathsep.join(
        value for value in (test_module_root, prior_pythonpath) if value
    )
    try:
        ray.init(
            num_cpus=4,
            num_gpus=0,
            object_store_memory=_OBJECT_STORE_BYTES,
            object_spilling_directory=str(spill),
            include_dashboard=False,
            log_to_driver=True,
            _temp_dir=str(socket_root),
            _system_config={
                "max_direct_call_object_size": 0,
                "object_spilling_threshold": 0.60,
            },
        )
        resources = ray.cluster_resources()
        assert int(resources["CPU"]) == 4
        assert int(resources.get("GPU", 0)) == 0
        assert int(resources["object_store_memory"]) == _OBJECT_STORE_BYTES

        reset_last_all_to_all_stats()
        reset_last_push_based_shuffle_stats()
        source = _ClaimedOriginDatasource(claims)
        sink = StreamingParquetDatasink(output)

        with _push_sort_context():
            dataset = ray.data.read_datasource(
                source, override_num_blocks=_BLOCKS, concurrency=4
            )
            sorted_dataset = dataset.sort(
                key=["Origin"], descending=[False], backend="cpu"
            )
            assert not claims.exists()
            assert not output.exists()
            before = _spill_snapshot()
            # This is the pipeline's only terminal Dataset action.
            sorted_dataset.write_datasink(
                sink,
                concurrency=2,
                ray_remote_args={"max_retries": 0},
            )
            after = _stable_spill_snapshot()

        delta = {name: after[name] - before[name] for name in before}
        all_to_all = get_last_all_to_all_stats()
        push = get_last_push_based_shuffle_stats()
        assert all_to_all is not None
        assert push is not None
        assert all_to_all["status"] == "succeeded"
        assert push["status"] == "succeeded"
        assert all_to_all["input_blocks"] == _BLOCKS
        assert all_to_all["peak_retained_bytes"] > _OBJECT_STORE_BYTES
        assert all_to_all["retained_blocks_at_eos"] == _BLOCKS
        assert all_to_all["bulk_started_at_ns"] >= all_to_all["eos_received_at_ns"]
        assert push["submission_code_path_requires_eos"] is True
        assert push["push_map_tasks_submitted"] == _BLOCKS
        assert push["first_push_map_task_preceded_eos"] is False
        assert delta["spilled_bytes_total"] > 0, delta
        assert delta["spilled_objects_total"] > 0, delta

        assert len(list(claims.glob("*.claim"))) == _BLOCKS
        assert len(list(claims.glob("*.done"))) == _BLOCKS
        expected = _expected_frequencies()
        result = sink.result
        assert result is not None
        validation = _validate_durable_parquet(
            output,
            result,
            _SCHEMA,
            sort_key="Origin",
            expected_rows=_TOTAL_ROWS,
            expected_row_id_sum=_TOTAL_ROWS * (_TOTAL_ROWS - 1) // 2,
            expected_sort_key_stats={
                "arrow_type": "string",
                "null_rows": 0,
                "cardinality": len(_ORIGINS),
                "frequency_digest": _frequency_digest(expected),
            },
        )
        assert validation["valid"], validation["rejection_reasons"]
        assert result["telemetry"]["transaction_committed"] is True

        print(
            "FORCED_SPILL_PROOF="
            + json.dumps(
                {
                    "object_store_bytes": _OBJECT_STORE_BYTES,
                    "input_blocks": _BLOCKS,
                    "input_rows": _TOTAL_ROWS,
                    "peak_retained_bytes": all_to_all["peak_retained_bytes"],
                    "ray_object_store_io": delta,
                    "push_map_tasks": push["push_map_tasks_submitted"],
                    "durable_validation": validation,
                },
                sort_keys=True,
            )
        )
    finally:
        ray.shutdown()
        if prior_pythonpath is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = prior_pythonpath
        shutil.rmtree(runtime, ignore_errors=True)
        shutil.rmtree(socket_root, ignore_errors=True)
