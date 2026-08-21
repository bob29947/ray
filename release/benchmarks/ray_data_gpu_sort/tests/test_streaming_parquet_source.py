from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
import importlib
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import ray
from ray.cloudpickle import dumps
from ray.data._internal.execution.interfaces import ExecutionOptions, RefBundle
from ray.data._internal.logical.operators import Read
from ray.data.block import BlockMetadata
from ray.data.context import DataContext
from ray.data.datasource import ReadTask

from release.benchmarks.ray_data_gpu_sort.streaming_parquet_common import (
    TARGET_BLOCKS,
    TARGET_DECODED_BYTES,
    TARGET_PLAN_DIGEST,
    TARGET_ROWS,
    exact_plan,
)
from release.benchmarks.ray_data_gpu_sort.spec import (
    DATASET_ROOT,
    cell_by_name,
    digest,
    load_manifest,
)
from release.benchmarks.ray_data_gpu_sort.data import read_projection
from release.benchmarks.ray_data_gpu_sort.streaming_parquet_source import (
    FrozenBTSPlanDatasource,
    FrozenPlanError,
    NonReplayableReadError,
    exact_block_size_bytes,
    read_source_telemetry,
    schema_from_manifest,
)


@pytest.fixture
def frozen_input(tmp_path: Path) -> dict:
    schema = pa.schema(
        [
            pa.field("Origin", pa.string()),
            pa.field("payload", pa.int64()),
        ],
        metadata={b"frozen-schema": b"v1"},
    )
    table = pa.Table.from_arrays(
        [
            pa.array(["SFO", "ATL", "BOS", "LAX", "ATL"]),
            pa.array([10, None, 30, 40, 50], type=pa.int64()),
        ],
        schema=schema,
    )
    path = tmp_path / "input.parquet"
    pq.write_table(table, path, row_group_size=3, store_schema=True)
    slices = [
        {
            "path": str(path),
            "row_group": 0,
            "rows": 2,
            "row_id_start": 0,
            "year": 2020,
            "month": 1,
            "copy": 0,
        },
        {
            "path": str(path),
            "row_group": 1,
            "rows": 2,
            "row_id_start": 2,
            "year": 2020,
            "month": 1,
            "copy": 1,
        },
    ]
    unsigned = {
        "kind": "test-frozen-plan",
        "slices": slices,
        "blocks": 2,
        "rows": 4,
    }
    plan = {**unsigned, "digest": digest(unsigned)}
    return {"schema": schema, "path": path, "plan": plan}


def _source(frozen_input: dict, telemetry: Path) -> FrozenBTSPlanDatasource:
    return FrozenBTSPlanDatasource(
        frozen_input["plan"],
        schema=frozen_input["schema"].append(pa.field("row_id", pa.int64())),
        telemetry_directory=telemetry,
        expected_plan_digest=frozen_input["plan"]["digest"],
        expected_rows=4,
        expected_blocks=2,
        expected_decoded_bytes=400,
        block_size_bytes=(175, 225),
    )


def test_construction_and_task_planning_are_lazy(
    frozen_input: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    telemetry = tmp_path / "telemetry"

    def unexpected_open(*_args, **_kwargs):
        raise AssertionError("planning must not open a Parquet reader")

    monkeypatch.setattr(pq, "ParquetFile", unexpected_open)
    source = _source(frozen_input, telemetry)
    tasks = source.get_read_tasks(parallelism=96)

    assert len(tasks) == 2
    assert not telemetry.exists()
    assert [task.metadata.num_rows for task in tasks] == [2, 2]
    assert [task.metadata.size_bytes for task in tasks] == [175, 225]
    assert [item.ordinal for item in source.block_descriptors] == [0, 1]
    assert sum(task.metadata.size_bytes for task in tasks) == 400
    assert all(
        task.schema.equals(source.output_schema, check_metadata=True) for task in tasks
    )


def test_physical_reader_descriptors_are_submitted_lazily_and_once(
    frozen_input: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_module = importlib.import_module(
        "release.benchmarks.ray_data_gpu_sort.streaming_parquet_source"
    )
    read_planner = importlib.import_module("ray.data._internal.planner.plan_read_op")
    source = _source(frozen_input, tmp_path / "telemetry")
    monkeypatch.setattr(source_module, "_READ_TASK_SUBMISSION_WINDOW", 1)

    def unexpected_open(*_args, **_kwargs):
        raise AssertionError("descriptor submission must not open Parquet")

    monkeypatch.setattr(pq, "ParquetFile", unexpected_open)
    submitted_tasks = []

    def fake_read_task_ref_bundle(read_task: ReadTask) -> RefBundle:
        submitted_tasks.append(read_task)
        object_id = bytes([len(submitted_tasks)]) * 28
        return RefBundle(
            (
                (
                    ray.ObjectRef(object_id),
                    BlockMetadata(1, 64, None, None),
                ),
            ),
            owns_blocks=False,
            schema=None,
        )

    monkeypatch.setattr(
        read_planner, "_read_task_ref_bundle", fake_read_task_ref_bundle
    )
    monkeypatch.setattr(read_planner, "_warn_on_high_parallelism", lambda *_args: None)
    logical_read = Read(source, source, parallelism=2)
    logical_read.set_detected_parallelism(2)
    physical_read = read_planner.plan_read_op(
        logical_read, [], DataContext.get_current()
    )
    input_buffer = physical_read.input_dependencies[0]

    # Graph construction and executor startup create no reader ObjectRefs.
    assert submitted_tasks == []
    input_buffer.start(ExecutionOptions())
    assert submitted_tasks == []

    # The first readiness probe creates one descriptor. Until that bundle is
    # dispatched downstream, the single-slot window cannot create the second.
    assert input_buffer.has_next()
    first_bundle = input_buffer.get_next()
    input_buffer.metrics.num_external_outqueue_blocks += len(first_bundle.blocks)
    assert len(submitted_tasks) == 1
    assert not input_buffer.has_next()
    assert not input_buffer.has_execution_finished()

    input_buffer.metrics.num_external_outqueue_blocks -= len(first_bundle.blocks)
    assert input_buffer.has_next()
    second_bundle = input_buffer.get_next()
    input_buffer.metrics.num_external_outqueue_blocks += len(second_bundle.blocks)
    assert len(submitted_tasks) == 2
    assert [task.metadata.num_rows for task in submitted_tasks] == [2, 2]
    assert [task.metadata.size_bytes for task in submitted_tasks] == [175, 225]

    input_buffer.metrics.num_external_outqueue_blocks -= len(second_bundle.blocks)
    assert not input_buffer.has_next()
    assert input_buffer.has_execution_finished()
    assert input_buffer._input_data == []


def test_read_tasks_select_exact_row_groups_and_construct_row_ids(
    frozen_input: dict, tmp_path: Path
) -> None:
    telemetry = tmp_path / "telemetry"
    source = _source(frozen_input, telemetry)
    first, second = source.get_read_tasks(parallelism=2)

    first_table = list(first())[0]
    second_table = list(second())[0]

    assert first_table.to_pydict() == {
        "Origin": ["SFO", "ATL"],
        "payload": [10, None],
        "row_id": [0, 1],
    }
    assert second_table.to_pydict() == {
        "Origin": ["LAX", "ATL"],
        "payload": [40, 50],
        "row_id": [2, 3],
    }
    assert first_table.schema.equals(source.output_schema, check_metadata=True)
    assert second_table.schema.equals(source.output_schema, check_metadata=True)
    assert first_table.schema.metadata == {b"frozen-schema": b"v1"}

    summary = read_source_telemetry(
        telemetry,
        plan_digest=frozen_input["plan"]["digest"],
        expected_blocks=2,
    )
    assert summary["events"] == 4
    assert summary["started_tasks"] == 2
    assert summary["produced_tasks"] == 2
    assert summary["failed_tasks"] == 0
    assert summary["exact_once_complete"] is True
    assert summary["completed_ordinals"] == [0, 1]
    assert summary["missing_ordinals"] == []
    assert list(summary["by_ordinal"]) == ["0", "1"]
    assert summary["produced_rows"] == 4
    assert summary["produced_decoded_bytes"] == first_table.nbytes + second_table.nbytes
    assert summary["planned_decoded_bytes"] == 400
    assert summary["produced_bytes_match_planned"] is False
    assert summary["byte_metadata_mismatch_ordinals"] == [0, 1]
    assert "exact_block_size_bytes" in summary["byte_metadata_semantics"]
    assert (
        summary["first_read_started_monotonic_ns"]
        <= summary["last_input_produced_monotonic_ns"]
    )


def test_read_callable_is_non_replayable(frozen_input: dict, tmp_path: Path) -> None:
    source = _source(frozen_input, tmp_path / "telemetry")
    task = source.get_read_tasks(parallelism=2)[0]

    assert len(list(task())) == 1
    with pytest.raises(NonReplayableReadError, match="already consumed"):
        list(task())


def test_replanning_is_deterministic_but_does_not_make_replay_safe(
    frozen_input: dict, tmp_path: Path
) -> None:
    source = _source(frozen_input, tmp_path / "telemetry")
    first_plan = source.get_read_tasks(parallelism=1)
    second_plan = source.get_read_tasks(parallelism=10_000)

    assert [task.metadata for task in first_plan] == [
        task.metadata for task in second_plan
    ]
    assert len(list(first_plan[1]())) == 1
    with pytest.raises(NonReplayableReadError):
        list(second_plan[1]())


def test_plan_digest_and_contiguous_row_ids_are_hard_contracts(
    frozen_input: dict, tmp_path: Path
) -> None:
    corrupted = copy.deepcopy(frozen_input["plan"])
    corrupted["slices"][0]["rows"] = 1
    with pytest.raises(FrozenPlanError, match="digest is invalid"):
        FrozenBTSPlanDatasource(
            corrupted,
            schema=frozen_input["schema"].append(pa.field("row_id", pa.int64())),
            telemetry_directory=tmp_path / "telemetry-a",
            expected_plan_digest=frozen_input["plan"]["digest"],
            expected_rows=4,
            expected_blocks=2,
            expected_decoded_bytes=400,
        )

    discontinuous = copy.deepcopy(frozen_input["plan"])
    discontinuous["slices"][1]["row_id_start"] = 3
    unsigned = {key: value for key, value in discontinuous.items() if key != "digest"}
    discontinuous["digest"] = digest(unsigned)
    with pytest.raises(FrozenPlanError, match="row_id_start"):
        FrozenBTSPlanDatasource(
            discontinuous,
            schema=frozen_input["schema"].append(pa.field("row_id", pa.int64())),
            telemetry_directory=tmp_path / "telemetry-b",
            expected_plan_digest=discontinuous["digest"],
            expected_rows=4,
            expected_blocks=2,
            expected_decoded_bytes=400,
        )


def test_source_requires_full_native_projection(
    frozen_input: dict, tmp_path: Path
) -> None:
    with pytest.raises(FrozenPlanError, match="all native columns"):
        FrozenBTSPlanDatasource(
            frozen_input["plan"],
            schema=frozen_input["schema"].append(pa.field("row_id", pa.int64())),
            telemetry_directory=tmp_path / "telemetry",
            columns=("Origin", "row_id"),
            expected_plan_digest=frozen_input["plan"]["digest"],
            expected_rows=4,
            expected_blocks=2,
            expected_decoded_bytes=400,
        )


def test_nonempty_smoke_plan_accepts_any_positive_scheduling_estimate(
    frozen_input: dict, tmp_path: Path
) -> None:
    schema = frozen_input["schema"].append(pa.field("row_id", pa.int64()))
    source = FrozenBTSPlanDatasource(
        frozen_input["plan"],
        schema=schema,
        telemetry_directory=tmp_path / "telemetry",
        expected_plan_digest=frozen_input["plan"]["digest"],
        expected_rows=4,
        expected_blocks=2,
        expected_decoded_bytes=1,
    )
    assert source.estimate_inmemory_data_size() == 1
    assert (
        sum(task.metadata.size_bytes for task in source.get_read_tasks(parallelism=2))
        == 1
    )

    with pytest.raises(FrozenPlanError, match="positive byte estimate"):
        FrozenBTSPlanDatasource(
            frozen_input["plan"],
            schema=schema,
            telemetry_directory=tmp_path / "telemetry-zero",
            expected_plan_digest=frozen_input["plan"]["digest"],
            expected_rows=4,
            expected_blocks=2,
            expected_decoded_bytes=0,
        )


def test_schema_is_reconstructed_from_manifest_without_parquet_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = {
        "schema": "Origin: string\nFlightDate: timestamp[ms]\nCancelled: bool",
        "schema_names": ["Origin", "FlightDate", "Cancelled"],
    }
    monkeypatch.setattr(
        pq,
        "read_schema",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("manifest schema parsing must not inspect Parquet")
        ),
    )

    schema = schema_from_manifest(manifest)

    assert schema == pa.schema(
        [
            ("Origin", pa.string()),
            ("FlightDate", pa.timestamp("ms")),
            ("Cancelled", pa.bool_()),
            ("row_id", pa.int64()),
        ]
    )


@pytest.mark.skipif(
    not (DATASET_ROOT / "manifest.json").is_file(),
    reason="local BTS corpus unavailable",
)
def test_exact_1tb_source_plans_9151_tasks_without_opening_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = load_manifest(DATASET_ROOT)
    plan, _ = exact_plan(DATASET_ROOT)
    output_schema = schema_from_manifest(manifest)

    def unexpected_open(*_args, **_kwargs):
        raise AssertionError("frozen source planning must not read Parquet")

    monkeypatch.setattr(pq, "ParquetFile", unexpected_open)
    telemetry = tmp_path / "telemetry"
    block_sizes = exact_block_size_bytes(plan, manifest)
    source = FrozenBTSPlanDatasource(
        plan,
        schema=output_schema,
        telemetry_directory=telemetry,
        block_size_bytes=block_sizes,
    )
    tasks = source.get_read_tasks(parallelism=TARGET_BLOCKS)

    assert source.plan_digest == TARGET_PLAN_DIGEST
    assert len(tasks) == TARGET_BLOCKS == 9_151
    assert sum(task.metadata.num_rows for task in tasks) == TARGET_ROWS
    assert sum(task.metadata.size_bytes for task in tasks) == TARGET_DECODED_BYTES
    assert block_sizes[-1] == 34_783_844
    assert [item.ordinal for item in source.block_descriptors] == list(
        range(TARGET_BLOCKS)
    )
    # One task captures one descriptor and the schema, not all 9,151 slices.
    assert len(dumps(tasks[0])) < 64 << 10
    assert not telemetry.exists()


@pytest.mark.skipif(
    os.environ.get("RAY_DATA_VERIFY_FROZEN_BTS_BYTES") != "1",
    reason="set RAY_DATA_VERIFY_FROZEN_BTS_BYTES=1 for the 68 GiB exact-byte proof",
)
def test_exact_block_metadata_matches_every_distinct_arrow_block() -> None:
    manifest = load_manifest(DATASET_ROOT)
    plan, slices = exact_plan(DATASET_ROOT)
    columns = cell_by_name(manifest, "origin-string").columns
    planned = exact_block_size_bytes(plan, manifest)
    distinct: dict[tuple[str, int, int], tuple[object, int]] = {}
    for item, size in zip(slices, planned):
        distinct.setdefault((item.path, item.row_group, item.rows), (item, size))

    def verify(value: tuple[object, int]) -> None:
        item, size = value
        table = read_projection(item.__dict__, columns)
        assert table.nbytes == size

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(verify, distinct.values()))
    assert sum(planned) == TARGET_DECODED_BYTES
