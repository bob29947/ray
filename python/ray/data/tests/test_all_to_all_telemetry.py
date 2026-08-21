import json

import pytest

import ray
from ray.data._internal.execution.interfaces import ExecutionOptions, RefBundle
from ray.data._internal.execution.operators.base_physical_operator import (
    AllToAllOperator,
    get_last_all_to_all_stats,
    reset_last_all_to_all_stats,
)
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data.block import BlockMetadata
from ray.data.context import DataContext


def _ref_bundle(sizes, *, seed):
    blocks = []
    for offset, size in enumerate(sizes):
        object_id_byte = seed + offset
        blocks.append(
            (
                ray.ObjectRef(bytes([object_id_byte]) * 28),
                BlockMetadata(
                    num_rows=1,
                    size_bytes=size,
                    exec_stats=None,
                    input_files=None,
                ),
            )
        )
    return RefBundle(blocks, owns_blocks=False, schema=None)


def _operator(bulk_fn):
    context = DataContext.get_current()
    input_op = InputDataBuffer(context, [])
    operator = AllToAllOperator(bulk_fn, input_op, context, name="TelemetryTest")
    operator.start(ExecutionOptions())
    return operator


def test_all_to_all_telemetry_proves_blocking_input_retention():
    reset_last_all_to_all_stats()
    observed_during_bulk = {}

    def bulk_fn(bundles, task_ctx):
        observed_during_bulk.update(task_ctx.kwargs["_all_to_all_execution_telemetry"])
        assert len(bundles) == 2
        return [], {}

    operator = _operator(bulk_fn)
    operator.add_input(_ref_bundle([5, 7], seed=1), 0)
    live = get_last_all_to_all_stats()
    assert live is not None
    assert live["status"] == "collecting_inputs"
    assert live["current_retained_blocks"] == 2
    assert live["current_retained_bytes"] == 12
    operator.add_input(_ref_bundle([11], seed=3), 0)
    operator.all_inputs_done()

    stats = get_last_all_to_all_stats()
    assert stats is not None
    assert stats["operator_name"] == "TelemetryTest"
    assert stats["input_bundles"] == 2
    assert stats["input_blocks"] == 3
    assert stats["input_metadata_bytes"] == 23
    assert stats["peak_retained_blocks"] == 3
    assert stats["peak_retained_bytes"] == 23
    assert stats["retained_blocks_at_eos"] == 3
    assert stats["retained_bytes_at_eos"] == 23
    assert stats["current_retained_blocks"] == 0
    assert stats["current_retained_bytes"] == 0
    assert stats["first_input_received_at_ns"] <= stats["last_input_received_at_ns"]
    assert stats["last_input_received_at_ns"] <= stats["eos_received_at_ns"]
    assert stats["eos_received_at_ns"] <= stats["bulk_started_at_ns"]
    assert stats["bulk_started_at_ns"] <= stats["bulk_completed_at_ns"]
    assert stats["bulk_completed_at_ns"] <= stats["all_to_all_completed_at_ns"]
    assert stats["status"] == "succeeded"
    assert stats["bulk_failed_at_ns"] is None
    assert observed_during_bulk["retained_blocks_at_eos"] == 3
    assert observed_during_bulk["retained_bytes_at_eos"] == 23
    json.dumps(stats)

    # The public getter must not expose the mutable module-global dictionary.
    stats["input_blocks"] = -1
    assert get_last_all_to_all_stats()["input_blocks"] == 3


def test_all_to_all_telemetry_survives_bulk_failure():
    reset_last_all_to_all_stats()

    def bulk_fn(_bundles, _task_ctx):
        raise ValueError("expected telemetry failure")

    operator = _operator(bulk_fn)
    operator.add_input(_ref_bundle([13], seed=5), 0)

    with pytest.raises(ValueError, match="expected telemetry failure"):
        operator.all_inputs_done()

    stats = get_last_all_to_all_stats()
    assert stats is not None
    assert stats["status"] == "failed"
    assert stats["bulk_completed_at_ns"] is None
    assert stats["bulk_failed_at_ns"] is not None
    assert stats["all_to_all_failed_at_ns"] == stats["bulk_failed_at_ns"]
    assert stats["bulk_failure_type"] == "ValueError"
    assert stats["bulk_failure_message"] == "expected telemetry failure"
    # Preserve existing failure semantics: operator cleanup owns the retained input.
    assert stats["current_retained_blocks"] == 1
    assert stats["current_retained_bytes"] == 13
    json.dumps(stats)
