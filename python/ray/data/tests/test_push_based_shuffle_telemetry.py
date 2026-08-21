import json

import pytest

from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.planner.exchange.push_based_shuffle_task_scheduler import (
    PushBasedShuffleTaskScheduler,
    _MapStageIterator,
    get_last_push_based_shuffle_stats,
    reset_last_push_based_shuffle_stats,
)


class _TelemetryOnlyScheduler(PushBasedShuffleTaskScheduler):
    def __init__(self, submission_times, *, failure=None):
        super().__init__(exchange_spec=object())
        self._submission_times = submission_times
        self._failure = failure
        self.observed_live = None

    def _execute_impl(self, *args, **kwargs):
        for mapper_idx, submitted_at_ns in enumerate(self._submission_times):
            self._record_map_task_submission(submitted_at_ns, mapper_idx)
        self.observed_live = get_last_push_based_shuffle_stats()
        if self._failure is not None:
            raise self._failure
        return [], {}


def _task_context(parent_stats):
    return TaskContext(
        task_idx=0,
        op_name="Sort",
        kwargs={"_all_to_all_execution_telemetry": parent_stats},
    )


def test_push_shuffle_telemetry_records_exact_submissions_after_eos():
    reset_last_push_based_shuffle_stats()
    parent_stats = {"eos_received_at_ns": 100}
    scheduler = _TelemetryOnlyScheduler([101, 102, 103])

    assert scheduler.execute([], 1, _task_context(parent_stats)) == ([], {})
    assert scheduler.observed_live is not None
    assert scheduler.observed_live["status"] == "running"
    assert scheduler.observed_live["push_map_tasks_submitted"] == 3

    stats = get_last_push_based_shuffle_stats()
    assert stats is not None
    assert stats["status"] == "succeeded"
    assert stats["eos_received_at_ns"] == 100
    assert stats["first_push_map_task_submitted_at_ns"] == 101
    assert stats["last_push_map_task_submitted_at_ns"] == 103
    assert stats["first_push_map_task_index"] == 0
    assert stats["push_map_tasks_submitted"] == 3
    assert stats["first_push_map_task_preceded_eos"] is False
    assert stats["submission_code_path_requires_eos"] is True
    assert stats["shuffle_completed_at_ns"] is not None
    assert stats["shuffle_failed_at_ns"] is None
    assert parent_stats["first_push_map_task_submitted_at_ns"] == 101
    assert parent_stats["push_map_tasks_submitted"] == 3
    assert parent_stats["first_push_map_task_preceded_eos"] is False
    json.dumps(stats)


def test_push_shuffle_telemetry_survives_failure():
    reset_last_push_based_shuffle_stats()
    parent_stats = {"eos_received_at_ns": 200}
    scheduler = _TelemetryOnlyScheduler(
        [201], failure=RuntimeError("expected shuffle failure")
    )

    with pytest.raises(RuntimeError, match="expected shuffle failure"):
        scheduler.execute([], 1, _task_context(parent_stats))

    stats = get_last_push_based_shuffle_stats()
    assert stats is not None
    assert stats["status"] == "failed"
    assert stats["push_map_tasks_submitted"] == 1
    assert stats["shuffle_completed_at_ns"] is None
    assert stats["shuffle_failed_at_ns"] is not None
    assert stats["shuffle_failure_type"] == "RuntimeError"
    assert stats["shuffle_failure_message"] == "expected shuffle failure"
    assert parent_stats["push_shuffle_failed_at_ns"] == stats["shuffle_failed_at_ns"]
    json.dumps(stats)


class _FakeShuffleMap:
    def __init__(self, *, failure=None):
        self.calls = []
        self.failure = failure

    def remote(self, *args):
        self.calls.append(args)
        if self.failure is not None:
            raise self.failure
        return ["partition-ref", "metadata-ref"]


def test_map_iterator_records_only_successful_remote_submission(monkeypatch):
    monkeypatch.setattr(
        "ray.data._internal.planner.exchange."
        "push_based_shuffle_task_scheduler.time.time_ns",
        lambda: 456,
    )
    submissions = []
    shuffle_map = _FakeShuffleMap()
    iterator = _MapStageIterator(
        ["input-ref"],
        shuffle_map,
        ["map-arg"],
        on_map_task_submitted=lambda submitted_at_ns, mapper_idx: submissions.append(
            (submitted_at_ns, mapper_idx)
        ),
    )

    assert next(iterator) == "metadata-ref"
    assert shuffle_map.calls == [(0, "input-ref", "map-arg")]
    assert submissions == [(456, 0)]
    with pytest.raises(StopIteration):
        next(iterator)

    failed_submissions = []
    iterator = _MapStageIterator(
        ["input-ref"],
        _FakeShuffleMap(failure=RuntimeError("remote rejected")),
        [],
        on_map_task_submitted=lambda submitted_at_ns, mapper_idx: (
            failed_submissions.append((submitted_at_ns, mapper_idx))
        ),
    )
    with pytest.raises(RuntimeError, match="remote rejected"):
        next(iterator)
    assert failed_submissions == []
