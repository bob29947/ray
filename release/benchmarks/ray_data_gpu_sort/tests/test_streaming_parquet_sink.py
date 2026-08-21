from __future__ import annotations

import errno
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ray.data._internal.execution.interfaces import TaskContext
from ray.data.datasource.datasink import WriteResult
from release.benchmarks.ray_data_gpu_sort import streaming_parquet_sink as sink_module
from release.benchmarks.ray_data_gpu_sort.streaming_parquet_sink import (
    StreamingParquetDatasink,
    StreamingParquetSinkCapacityError,
    StreamingParquetSinkError,
    _iter_slices_with_limit,
    _schema_from_base64,
)


def _context(task_index: int) -> TaskContext:
    return TaskContext(task_idx=task_index, op_name="Write")


def _result(*write_returns: dict) -> WriteResult[dict]:
    return WriteResult(num_rows=0, size_bytes=0, write_returns=list(write_returns))


def _table(start: int, stop: int) -> pa.Table:
    schema = pa.schema(
        [pa.field("Origin", pa.string()), pa.field("row_id", pa.int64())],
        metadata={b"benchmark": b"streaming-sort"},
    )
    return pa.Table.from_arrays(
        [
            pa.array([f"K{value:04d}" for value in range(start, stop)]),
            pa.array(range(start, stop), type=pa.int64()),
        ],
        schema=schema,
    )


def _committed_tables(output: Path) -> list[pa.Table]:
    return [pq.read_table(path) for path in sorted(output.glob("part-*.parquet"))]


def test_bounded_ordered_transaction_and_parquet_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    table_0 = _table(0, 24)
    table_1 = _table(24, 43)
    sink = StreamingParquetDatasink(
        output,
        max_decoded_file_bytes=120,
        row_group_decoded_bytes=55,
        data_page_bytes=1024,
        target_bytes_per_write=500,
    )
    sink.on_write_start(table_0.schema)

    result_0 = sink.write([table_0.slice(0, 10), table_0.slice(10)], _context(0))
    result_1 = sink.write([table_1], _context(1))

    sync_observations = []

    def observe_syncfs(path: Path) -> None:
        sync_observations.append(
            (
                (path / "manifest.json").is_file(),
                (path / "_SUCCESS").is_file(),
                not list(path.glob(".rollback-*")),
            )
        )

    monkeypatch.setattr(sink_module, "_syncfs", observe_syncfs)
    # Completion order must not affect final global order.
    sink.on_write_complete(_result(result_1, result_0))

    manifest_payload = (output / "manifest.json").read_bytes()
    manifest = json.loads(manifest_payload)
    assert sync_observations == [(True, True, True)]
    assert manifest["totals"]["num_rows"] == 43
    assert manifest["totals"]["file_count"] > 2
    assert [entry["file_index"] for entry in manifest["files"]] == list(
        range(len(manifest["files"]))
    )
    assert [entry["path"] for entry in manifest["files"]] == [
        f"part-{index:05d}.parquet" for index in range(len(manifest["files"]))
    ]
    assert [entry["task_index"] for entry in manifest["files"]] == sorted(
        entry["task_index"] for entry in manifest["files"]
    )
    assert all(entry["decoded_bytes"] <= 120 for entry in manifest["files"])
    assert all(
        entry["logical_to_physical_ratio"]
        == entry["decoded_bytes"] / entry["file_size_bytes"]
        for entry in manifest["files"]
    )
    assert all(
        entry["max_row_group_decoded_bytes"] <= 55 for entry in manifest["files"]
    )
    assert _schema_from_base64(manifest["schema_base64"]).equals(
        table_0.schema, check_metadata=True
    )

    committed = _committed_tables(output)
    assert pa.concat_tables(committed)["row_id"].to_pylist() == list(range(43))
    for path in sorted(output.glob("part-*.parquet")):
        parquet_file = pq.ParquetFile(path)
        assert parquet_file.schema_arrow.equals(table_0.schema, check_metadata=True)
        assert parquet_file.metadata.format_version == "2.6"
        for row_group_index in range(parquet_file.metadata.num_row_groups):
            row_group = parquet_file.metadata.row_group(row_group_index)
            for column_index in range(row_group.num_columns):
                assert row_group.column(column_index).compression == "ZSTD"
                assert row_group.column(column_index).statistics is not None
                assert any(
                    "DICTIONARY" in encoding
                    for encoding in row_group.column(column_index).encodings
                )

    success = json.loads((output / "_SUCCESS").read_text())
    assert success == {
        "manifest": "manifest.json",
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
    }
    assert not list(output.rglob("*.partial"))
    assert not list(output.glob(".rollback-*"))
    assert not (output / ".staging").exists()
    assert sink.target_bytes_per_write == 500
    assert sink.telemetry["transaction_committed"] is True
    assert sink.telemetry["peak_file_decoded_bytes"] <= 120
    assert sink.telemetry["peak_row_group_decoded_bytes"] <= 55
    assert sink.result is not None
    assert sink.result["manifest"] == manifest
    assert sink.result["success"] == success
    telemetry = sink.result["telemetry"]
    assert telemetry["logical_to_physical_ratio"] == (
        telemetry["decoded_bytes"] / telemetry["file_size_bytes"]
    )
    assert telemetry["writer_input_decoded_bytes_max"] <= 500
    assert telemetry["configured_peak_writer_input_bytes"] == 500 * 16
    assert telemetry["writer_input_target_overshoot_tasks"] == 0
    assert telemetry["parquet_close_s_sum"] >= 0
    assert telemetry["parquet_encoding_cpu_s_sum"] >= 0
    assert telemetry["split_count"] > 0
    assert telemetry["skipped_empty_blocks"] == 0
    assert telemetry["skipped_empty_partitions"] == 0
    assert len(sink.result["writer_tasks"]) == 2
    assert all(
        receipt["input_decoded_bytes"] <= 500 for receipt in sink.result["writer_tasks"]
    )
    assert (
        telemetry["write_started_at_ns"]
        <= telemetry["first_writer_work_started_at_ns"]
        <= telemetry["first_parquet_file_completed_at_ns"]
        <= telemetry["last_parquet_file_completed_at_ns"]
        <= telemetry["commit_started_at_ns"]
        <= telemetry["manifest_closed_at_ns"]
        <= telemetry["success_closed_at_ns"]
        <= telemetry["drain_completed_at_ns"]
    )


def test_empty_and_absent_schema_metadata_are_equivalent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "equivalent-empty-metadata"
    table_0 = _table(0, 8).replace_schema_metadata(None)
    table_1 = _table(8, 16).replace_schema_metadata({})
    declared_schema = table_0.schema.with_metadata({})
    sink = StreamingParquetDatasink(output, target_bytes_per_write=1_000)
    monkeypatch.setattr(sink_module, "_syncfs", lambda _path: None)

    sink.on_write_start(declared_schema)
    sink.on_write_start(table_0.schema)
    write_return_0 = sink.write([table_0], _context(0))
    write_return_1 = sink.write([table_1], _context(1))
    sink.on_write_complete(_result(write_return_1, write_return_0))

    manifest = json.loads((output / "manifest.json").read_bytes())
    assert _schema_from_base64(manifest["schema_base64"]).equals(
        declared_schema, check_metadata=True
    )
    assert pa.concat_tables(_committed_tables(output))["row_id"].to_pylist() == list(
        range(16)
    )


def test_non_equivalent_schema_metadata_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "different-metadata"
    table = _table(0, 8).replace_schema_metadata(None)
    sink = StreamingParquetDatasink(output)
    sink.on_write_start(table.schema.with_metadata({b"version": b"one"}))

    with pytest.raises(StreamingParquetSinkError, match="does not match"):
        sink.write([table.replace_schema_metadata({b"version": b"two"})], _context(0))

    sink.on_write_failed(RuntimeError("expected schema mismatch"))
    assert list(output.iterdir()) == []


def test_slice_helper_splits_oversized_block_without_concatenation() -> None:
    table = _table(0, 100)
    pieces = list(_iter_slices_with_limit(table, 96))

    assert len(pieces) > 1
    assert all(piece.nbytes <= 96 for piece in pieces)
    assert pa.concat_tables(pieces)["row_id"].to_pylist() == list(range(100))


def test_writer_rejects_aggregate_input_above_hard_bound(tmp_path: Path) -> None:
    output = tmp_path / "writer-bound"
    table = _table(0, 20)
    first = table.slice(0, 10)
    second = table.slice(10)
    bound = max(first.nbytes, second.nbytes)
    sink = StreamingParquetDatasink(
        output,
        max_decoded_file_bytes=bound,
        row_group_decoded_bytes=bound,
        target_bytes_per_write=bound,
    )
    sink.on_write_start(table.schema)

    with pytest.raises(StreamingParquetSinkError, match="writer input exceeded"):
        sink.write([first, second], _context(0))

    assert not any(sink._transaction_root.iterdir())
    sink.on_write_failed(RuntimeError("expected writer-bound test failure"))
    assert list(output.iterdir()) == []


def test_empty_schema_less_blocks_do_not_create_parquet_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "empty"
    sink = StreamingParquetDatasink(
        output, max_decoded_file_bytes=128, row_group_decoded_bytes=64
    )
    sink.on_write_start(None)
    write_return = sink.write([pa.table({}), pa.table({})], _context(0))
    monkeypatch.setattr(sink_module, "_syncfs", lambda _path: None)

    sink.on_write_complete(_result(write_return))

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["schema_base64"] is None
    assert manifest["totals"] == {
        "decoded_bytes": 0,
        "file_count": 0,
        "file_size_bytes": 0,
        "logical_to_physical_ratio": None,
        "num_rows": 0,
    }
    assert not list(output.glob("part-*.parquet"))
    assert (output / "_SUCCESS").is_file()
    assert sink.result["telemetry"]["skipped_empty_blocks"] == 2
    assert sink.result["telemetry"]["skipped_empty_partitions"] == 1


def test_variable_width_row_that_does_not_fit_remainder_starts_new_group_and_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "variable"
    table = pa.table(
        {
            "Origin": ["a" * 8, "b" * 8, "c" * 17, "d" * 8],
            "row_id": pa.array(range(4), type=pa.int64()),
        }
    )
    one_row_max = max(table.slice(index, 1).nbytes for index in range(4))
    sink = StreamingParquetDatasink(
        output,
        max_decoded_file_bytes=one_row_max + 8,
        row_group_decoded_bytes=one_row_max,
    )
    sink.on_write_start(table.schema)

    write_return = sink.write([table], _context(0))
    monkeypatch.setattr(sink_module, "_syncfs", lambda _path: None)
    sink.on_write_complete(_result(write_return))

    assert pa.concat_tables(_committed_tables(output))["row_id"].to_pylist() == [
        0,
        1,
        2,
        3,
    ]


def test_enospc_is_typed_and_writer_attempt_is_rolled_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "capacity"
    table = _table(0, 4)
    sink = StreamingParquetDatasink(
        output, max_decoded_file_bytes=128, row_group_decoded_bytes=64
    )
    sink.on_write_start(table.schema)

    def no_space(*_args, **_kwargs):
        raise OSError(errno.ENOSPC, "no space left on device")

    monkeypatch.setattr(sink_module, "_open_parquet_writer", no_space)
    with pytest.raises(StreamingParquetSinkCapacityError, match="capacity"):
        sink.write([table], _context(0))

    assert not any(sink._transaction_root.iterdir())
    sink.on_write_failed(RuntimeError("upstream failed"))
    assert list(output.iterdir()) == []


def test_setup_directory_enospc_is_typed_and_transactionally_cleaned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "setup-capacity"
    output.mkdir()
    sink = StreamingParquetDatasink(output)
    real_mkdir = Path.mkdir

    def no_space_for_rollback(path: Path, *args, **kwargs) -> None:
        if path == sink._rollback_root:
            raise OSError(errno.ENOSPC, "injected setup ENOSPC")
        real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", no_space_for_rollback)

    with pytest.raises(StreamingParquetSinkCapacityError, match="capacity"):
        sink.on_write_start(_table(0, 1).schema)

    assert list(output.iterdir()) == []
    assert sink.telemetry["rollback_errors"] == []


def test_writer_attempt_directory_enospc_is_typed_and_cleaned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "attempt-capacity"
    table = _table(0, 4)
    sink = StreamingParquetDatasink(output)
    sink.on_write_start(table.schema)
    real_mkdir = Path.mkdir

    def no_space_for_attempt(path: Path, *args, **kwargs) -> None:
        if path.parent == sink._transaction_root and path.name.startswith("task-"):
            raise OSError(errno.ENOSPC, "injected attempt ENOSPC")
        real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", no_space_for_attempt)

    with pytest.raises(StreamingParquetSinkCapacityError, match="capacity"):
        sink.write([table], _context(0))

    assert not any(sink._transaction_root.iterdir())
    sink.on_write_failed(RuntimeError("upstream failed"))
    assert list(output.iterdir()) == []


def test_writer_failure_removes_already_staged_task_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "writer-failure"
    table = _table(0, 30)
    sink = StreamingParquetDatasink(
        output, max_decoded_file_bytes=100, row_group_decoded_bytes=50
    )
    sink.on_write_start(table.schema)
    real_fsync = sink_module._fsync_file
    calls = 0

    def fail_second_file(path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected fsync failure")
        real_fsync(path)

    monkeypatch.setattr(sink_module, "_fsync_file", fail_second_file)
    with pytest.raises(StreamingParquetSinkError, match="injected fsync failure"):
        sink.write([table], _context(0))

    assert calls == 2
    assert not any(sink._transaction_root.iterdir())


def test_commit_enospc_rolls_back_final_and_staged_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "commit-failure"
    table = _table(0, 30)
    sink = StreamingParquetDatasink(
        output, max_decoded_file_bytes=100, row_group_decoded_bytes=50
    )
    sink.on_write_start(table.schema)
    write_return = sink.write([table], _context(0))
    assert len(write_return["files"]) > 1
    real_rename = sink_module._atomic_rename

    def fail_second_final(source: Path, destination: Path) -> None:
        if destination.name == "part-00001.parquet":
            raise OSError(errno.ENOSPC, "injected commit ENOSPC")
        real_rename(source, destination)

    monkeypatch.setattr(sink_module, "_atomic_rename", fail_second_final)
    with pytest.raises(StreamingParquetSinkCapacityError, match="capacity"):
        sink.on_write_complete(_result(write_return))

    assert list(output.iterdir()) == []
    assert sink.telemetry["transaction_committed"] is False
    assert sink.telemetry["rollback_errors"] == []


def test_syncfs_failure_rolls_back_files_manifest_and_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "drain-failure"
    table = _table(0, 6)
    sink = StreamingParquetDatasink(
        output, max_decoded_file_bytes=256, row_group_decoded_bytes=128
    )
    sink.on_write_start(table.schema)
    write_return = sink.write([table], _context(0))

    def fail_syncfs(_path: Path) -> None:
        raise OSError(errno.EIO, "injected drain failure")

    monkeypatch.setattr(sink_module, "_syncfs", fail_syncfs)
    with pytest.raises(StreamingParquetSinkError, match="drain failure"):
        sink.on_write_complete(_result(write_return))

    assert list(output.iterdir()) == []


def test_manifest_bytes_are_independent_of_task_completion_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sink_module, "_syncfs", lambda _path: None)
    manifests = []
    for run, reversed_completion in enumerate((False, True)):
        output = tmp_path / f"deterministic-{run}"
        table_0 = _table(0, 8)
        table_1 = _table(8, 16)
        sink = StreamingParquetDatasink(
            output, max_decoded_file_bytes=128, row_group_decoded_bytes=64
        )
        sink.on_write_start(table_0.schema)
        returns = [
            sink.write([table_0], _context(0)),
            sink.write([table_1], _context(1)),
        ]
        if reversed_completion:
            returns.reverse()
        sink.on_write_complete(_result(*returns))
        manifests.append((output / "manifest.json").read_bytes())

    assert manifests[0] == manifests[1]


def test_rollback_never_removes_unowned_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "ownership"
    sink = StreamingParquetDatasink(output)
    sink.on_write_start(None)
    sentinel = output / "user-sentinel"
    sentinel.write_text("keep")
    monkeypatch.setattr(sink_module, "_syncfs", lambda _path: None)

    sink.on_write_failed(RuntimeError("failure"))

    assert sentinel.read_text() == "keep"
    assert not (output / ".staging").exists()


def test_rejects_nonempty_output_directory(tmp_path: Path) -> None:
    output = tmp_path / "not-owned"
    output.mkdir()
    (output / "existing").write_text("do not overwrite")

    with pytest.raises(StreamingParquetSinkError, match="must be empty"):
        StreamingParquetDatasink(output).on_write_start(None)

    assert (output / "existing").read_text() == "do not overwrite"


@pytest.mark.parametrize(
    "interference_name",
    ["part-00000.parquet", "manifest.json", "_SUCCESS"],
)
def test_commit_rollback_preserves_preexisting_final_path_interference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interference_name: str,
) -> None:
    output = tmp_path / f"preexisting-{interference_name}"
    table = _table(0, 6)
    sink = StreamingParquetDatasink(
        output, max_decoded_file_bytes=256, row_group_decoded_bytes=128
    )
    sink.on_write_start(table.schema)
    write_return = sink.write([table], _context(0))
    external_payload = f"external {interference_name}".encode()
    interference_path = output / interference_name
    interference_path.write_bytes(external_payload)
    monkeypatch.setattr(sink_module, "_syncfs", lambda _path: None)

    with pytest.raises(StreamingParquetSinkError, match="commit"):
        sink.on_write_complete(_result(write_return))

    assert interference_path.read_bytes() == external_payload
    assert sink.telemetry["transaction_committed"] is False


@pytest.mark.parametrize(
    "interference_name", ["manifest.json.partial", "_SUCCESS.partial"]
)
def test_atomic_metadata_write_never_unlinks_preexisting_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interference_name: str,
) -> None:
    output = tmp_path / f"preexisting-{interference_name}"
    table = _table(0, 6)
    sink = StreamingParquetDatasink(
        output, max_decoded_file_bytes=256, row_group_decoded_bytes=128
    )
    sink.on_write_start(table.schema)
    write_return = sink.write([table], _context(0))
    external_payload = f"external {interference_name}".encode()
    assert sink._rollback_root.stat().st_mode & 0o777 == 0o700
    interference_path = sink._rollback_root / interference_name
    interference_path.write_bytes(external_payload)
    monkeypatch.setattr(sink_module, "_syncfs", lambda _path: None)

    with pytest.raises(StreamingParquetSinkError, match="commit"):
        sink.on_write_complete(_result(write_return))

    assert interference_path.read_bytes() == external_payload
    assert sink._rollback_root.is_dir()
    assert sink.telemetry["transaction_committed"] is False


@pytest.mark.parametrize(
    ("partial_name", "final_name"),
    [
        ("manifest.json.partial", "manifest.json"),
        ("_SUCCESS.partial", "_SUCCESS"),
    ],
)
def test_atomic_metadata_write_cleanup_preserves_replaced_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    partial_name: str,
    final_name: str,
) -> None:
    output = tmp_path / f"replaced-{final_name}"
    table = _table(0, 6)
    sink = StreamingParquetDatasink(
        output, max_decoded_file_bytes=256, row_group_decoded_bytes=128
    )
    sink.on_write_start(table.schema)
    write_return = sink.write([table], _context(0))
    external_payload = f"replacement {partial_name}".encode()
    real_atomic_rename = sink_module._atomic_rename

    def replace_partial_then_fail(source: Path, destination: Path) -> None:
        if source.name == partial_name and destination.name == final_name:
            source.unlink()
            source.write_bytes(external_payload)
            raise FileExistsError(errno.EEXIST, "injected interference", destination)
        real_atomic_rename(source, destination)

    monkeypatch.setattr(sink_module, "_atomic_rename", replace_partial_then_fail)
    monkeypatch.setattr(sink_module, "_syncfs", lambda _path: None)

    with pytest.raises(StreamingParquetSinkError, match="commit"):
        sink.on_write_complete(_result(write_return))

    assert (sink._rollback_root / partial_name).read_bytes() == external_payload
    assert sink._rollback_root.is_dir()
    assert sink.telemetry["transaction_committed"] is False


@pytest.mark.parametrize(
    ("partial_name", "final_name"),
    [
        ("manifest.json.partial", "manifest.json"),
        ("_SUCCESS.partial", "_SUCCESS"),
    ],
)
def test_atomic_metadata_publish_rejects_successfully_moved_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    partial_name: str,
    final_name: str,
) -> None:
    output = tmp_path / f"successful-metadata-replacement-{final_name}"
    table = _table(0, 6)
    sink = StreamingParquetDatasink(
        output, max_decoded_file_bytes=256, row_group_decoded_bytes=128
    )
    sink.on_write_start(table.schema)
    write_return = sink.write([table], _context(0))
    external_payload = f"successful replacement {partial_name}".encode()
    real_atomic_rename = sink_module._atomic_rename

    def replace_then_publish(source: Path, destination: Path) -> None:
        if source.name == partial_name and destination.name == final_name:
            source.unlink()
            source.write_bytes(external_payload)
        real_atomic_rename(source, destination)

    monkeypatch.setattr(sink_module, "_atomic_rename", replace_then_publish)
    monkeypatch.setattr(sink_module, "_syncfs", lambda _path: None)

    with pytest.raises(StreamingParquetSinkError, match="transaction-owned inode"):
        sink.on_write_complete(_result(write_return))

    assert (output / final_name).read_bytes() == external_payload
    assert sink.telemetry["transaction_committed"] is False


def test_staged_publish_rejects_successfully_moved_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "successful-staged-replacement"
    table = _table(0, 6)
    sink = StreamingParquetDatasink(
        output, max_decoded_file_bytes=256, row_group_decoded_bytes=128
    )
    sink.on_write_start(table.schema)
    write_return = sink.write([table], _context(0))
    external_payload = b"successful staged replacement"
    real_atomic_rename = sink_module._atomic_rename

    def replace_then_publish(source: Path, destination: Path) -> None:
        if source.suffix == ".staged" and destination.name == "part-00000.parquet":
            source.unlink()
            source.write_bytes(external_payload)
        real_atomic_rename(source, destination)

    monkeypatch.setattr(sink_module, "_atomic_rename", replace_then_publish)
    monkeypatch.setattr(sink_module, "_syncfs", lambda _path: None)

    with pytest.raises(StreamingParquetSinkError, match="transaction-owned inode"):
        sink.on_write_complete(_result(write_return))

    assert (output / "part-00000.parquet").read_bytes() == external_payload
    assert sink.telemetry["transaction_committed"] is False


@pytest.mark.parametrize(
    "interference_name",
    ["part-00000.parquet", "manifest.json", "_SUCCESS"],
)
def test_rollback_preserves_external_replacement_of_owned_final_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interference_name: str,
) -> None:
    output = tmp_path / f"replacement-{interference_name}"
    table = _table(0, 6)
    sink = StreamingParquetDatasink(
        output, max_decoded_file_bytes=256, row_group_decoded_bytes=128
    )
    sink.on_write_start(table.schema)
    write_return = sink.write([table], _context(0))
    external_payload = f"replacement {interference_name}".encode()

    def replace_owned_path_then_fail(_path: Path) -> None:
        replacement = output / f".external-{interference_name}"
        replacement.write_bytes(external_payload)
        replacement.replace(output / interference_name)
        raise OSError(errno.EIO, "injected drain failure")

    monkeypatch.setattr(sink_module, "_syncfs", replace_owned_path_then_fail)

    with pytest.raises(StreamingParquetSinkError, match="drain failure"):
        sink.on_write_complete(_result(write_return))

    assert (output / interference_name).read_bytes() == external_payload
    assert sink.telemetry["transaction_committed"] is False


def test_owned_cleanup_preserves_replacement_racing_atomic_quarantine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "owned.parquet"
    path.write_bytes(b"owned")
    owned = sink_module._owned_file(path)
    quarantine_root = tmp_path / "private-rollback"
    quarantine_root.mkdir(mode=0o700)
    external_payload = b"replacement in the old lstat-unlink window"
    real_atomic_rename = sink_module._atomic_rename
    injected = False

    def replace_immediately_before_quarantine(source: Path, destination: Path) -> None:
        nonlocal injected
        if source == path and not injected:
            injected = True
            source.unlink()
            source.write_bytes(external_payload)
        real_atomic_rename(source, destination)

    monkeypatch.setattr(
        sink_module, "_atomic_rename", replace_immediately_before_quarantine
    )

    assert sink_module._quarantine_if_owned(owned, quarantine_root) is None
    assert path.read_bytes() == external_payload
    assert list(quarantine_root.iterdir()) == []
    quarantine_root.rmdir()
    assert list(tmp_path.iterdir()) == [path]


def test_quarantine_preserves_unowned_files_when_restore_also_races(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "owned.parquet"
    path.write_bytes(b"owned")
    owned = sink_module._owned_file(path)
    quarantine_root = tmp_path / "private-rollback"
    quarantine_root.mkdir(mode=0o700)
    first_external = b"replacement moved into quarantine"
    second_external = b"replacement that blocks restoration"
    real_atomic_rename = sink_module._atomic_rename
    injected_source_replacement = False

    def race_both_moves(source: Path, destination: Path) -> None:
        nonlocal injected_source_replacement
        if source == path and not injected_source_replacement:
            injected_source_replacement = True
            source.unlink()
            source.write_bytes(first_external)
        elif source.parent == quarantine_root and destination == path:
            destination.write_bytes(second_external)
        real_atomic_rename(source, destination)

    monkeypatch.setattr(sink_module, "_atomic_rename", race_both_moves)

    error = sink_module._quarantine_if_owned(owned, quarantine_root)

    assert error is not None
    assert "unowned interference preserved" in error
    assert path.read_bytes() == second_external
    quarantined = list(quarantine_root.iterdir())
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == first_external


def test_syncfs_failure_survives_rollback_root_recreation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "rollback-root-recreation-failure"
    table = _table(0, 6)
    sink = StreamingParquetDatasink(
        output, max_decoded_file_bytes=256, row_group_decoded_bytes=128
    )
    sink.on_write_start(table.schema)
    write_return = sink.write([table], _context(0))

    def fail_syncfs(_path: Path) -> None:
        raise OSError(errno.EIO, "injected syncfs failure")

    def fail_recreate() -> None:
        raise OSError(errno.EIO, "injected rollback-root mkdir failure")

    monkeypatch.setattr(sink_module, "_syncfs", fail_syncfs)
    monkeypatch.setattr(sink, "_create_rollback_root", fail_recreate)

    with pytest.raises(
        sink_module.StreamingParquetSinkRollbackError,
        match="could not recreate the private rollback root",
    ):
        sink.on_write_complete(_result(write_return))

    telemetry = sink.telemetry
    assert telemetry["transaction_committed"] is False
    assert any(
        "injected rollback-root mkdir failure" in error
        for error in telemetry["rollback_errors"]
    )
    assert sink._owned_final_paths == []
    assert (output / "part-00000.parquet").is_file()
    assert (output / "manifest.json").is_file()
    assert (output / "_SUCCESS").is_file()
