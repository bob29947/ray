import inspect
import sys
import types
from dataclasses import FrozenInstanceError

import pytest

from ray.data import MapGroupPartitionsContext
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.logical.interfaces import LogicalPlan
from ray.data._internal.logical.operators import (
    InputData,
    MapBatches,
    MapGroupPartitions,
    Repartition,
)
from ray.data._internal.planner import create_planner
from ray.data._internal.planner.plan_map_group_partitions_op import (
    plan_map_group_partitions_op,
)
from ray.data.context import DataContext
from ray.data.dataset import Dataset
from ray.data.grouped_data import GroupedData


def identity(batch, context=None):
    return batch


class _FakeDataset:
    def __init__(self, logical_plan, context):
        self._logical_plan = logical_plan
        self._context = context

    @property
    def context(self):
        return self._context


@pytest.fixture
def source_dataset(monkeypatch):
    context = DataContext()

    def from_parent(parent, logical_plan):
        return _FakeDataset(logical_plan, parent.context)

    monkeypatch.setattr(Dataset, "_from_parent", staticmethod(from_parent))
    source_op = InputData(input_data=[])
    return _FakeDataset(LogicalPlan(source_op, context), context)


def test_map_group_partitions_has_function_only_api():
    signature = inspect.signature(GroupedData.map_group_partitions)
    assert list(signature.parameters) == ["self", "fn"]


@pytest.mark.parametrize("num_partitions", [None, 8])
def test_map_group_partitions_builds_logical_operator(source_dataset, num_partitions):
    dataset = GroupedData(
        source_dataset,
        ["User", "Card"],
        num_partitions=num_partitions,
    ).map_group_partitions(identity)

    op = dataset._logical_plan.dag
    assert isinstance(op, MapGroupPartitions)
    assert op.input_dependencies == [source_dataset._logical_plan.dag]
    assert op.key == ["User", "Card"]
    assert op.num_partitions == num_partitions
    assert op.fn is identity
    assert op.num_outputs == num_partitions
    assert op.name == "MapGroupPartitions"


@pytest.mark.parametrize(
    ("key", "error"),
    [
        (None, "one or more named"),
        ([], "one or more named"),
        (["User", ""], "non-empty string"),
        (["User", 1], "non-empty string"),
    ],
)
def test_map_group_partitions_rejects_invalid_keys(source_dataset, key, error):
    grouped = GroupedData(source_dataset, key, num_partitions=None)
    with pytest.raises(ValueError, match=error):
        grouped.map_group_partitions(identity)


def test_map_group_partitions_rejects_non_callable(source_dataset):
    grouped = GroupedData(source_dataset, "User", num_partitions=None)
    with pytest.raises(TypeError, match="fn must be callable"):
        grouped.map_group_partitions(None)


def test_planner_delegates_to_data_plane(monkeypatch):
    source = InputData(input_data=[])
    op = MapGroupPartitions(
        key=["User", "Card"],
        num_partitions=None,
        fn=identity,
        input_dependencies=[source],
    )
    context = DataContext()
    physical_child = object()
    expected = object()
    calls = []

    data_plane = types.ModuleType(
        "ray.data._internal.planner.parquet_cudf_map_group_partitions"
    )

    def plan(logical_op, data_context):
        calls.append((logical_op, data_context))
        return expected

    data_plane.plan_parquet_cudf_map_group_partitions = plan
    monkeypatch.setitem(sys.modules, data_plane.__name__, data_plane)

    result = plan_map_group_partitions_op(op, [physical_child], context)

    assert result is expected
    assert calls == [(op, context)]


def test_planner_discards_replaced_upstream_physical_plan(monkeypatch):
    source = InputData(input_data=[])
    op = MapGroupPartitions(
        key="User",
        num_partitions=None,
        fn=identity,
        input_dependencies=[source],
    )
    context = DataContext()
    replacement = InputDataBuffer(context, input_data=[])
    data_plane = types.ModuleType(
        "ray.data._internal.planner.parquet_cudf_map_group_partitions"
    )
    data_plane.plan_parquet_cudf_map_group_partitions = (
        lambda logical_op, data_context: replacement
    )
    monkeypatch.setitem(sys.modules, data_plane.__name__, data_plane)

    physical_plan, _ = create_planner().plan(LogicalPlan(op, context))

    assert physical_plan.dag is replacement
    assert set(physical_plan.op_map) == {replacement}
    assert physical_plan.op_map[replacement] is op


def test_partition_context_is_frozen_and_reports_group_count():
    context = MapGroupPartitionsContext(
        group_keys=("User", "Card"),
        input_group_boundaries=(0, 3, 8),
    )

    assert context.num_groups == 2
    with pytest.raises(FrozenInstanceError):
        context.group_keys = ("Other",)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"group_keys": (), "input_group_boundaries": (0,)},
        {"group_keys": ("key",), "input_group_boundaries": ()},
        {"group_keys": ("key",), "input_group_boundaries": (1, 2)},
        {"group_keys": ("key",), "input_group_boundaries": (0, 2, 2)},
        {"group_keys": ("key",), "input_group_boundaries": (0, True)},
    ],
)
def test_partition_context_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        MapGroupPartitionsContext(**kwargs)


def test_map_groups_still_builds_stock_logical_operators(monkeypatch, source_dataset):
    def repartition(num_outputs, *, keys=None, sort=False):
        assert num_outputs == 4
        assert keys == "User"
        assert sort
        op = Repartition(
            num_outputs=num_outputs,
            keys=[keys],
            sort=sort,
            input_dependencies=[source_dataset._logical_plan.dag],
        )
        shuffled = _FakeDataset(
            LogicalPlan(op, source_dataset.context), source_dataset.context
        )

        def map_batches(fn, **kwargs):
            map_op = MapBatches(
                fn,
                input_dependencies=[op],
            )
            return _FakeDataset(
                LogicalPlan(map_op, source_dataset.context), source_dataset.context
            )

        monkeypatch.setattr(
            shuffled,
            "_map_batches_without_batch_size_validation",
            map_batches,
            raising=False,
        )
        return shuffled

    monkeypatch.setattr(source_dataset, "repartition", repartition, raising=False)
    source_dataset.context.default_hash_shuffle_parallelism = 4

    result = GroupedData(source_dataset, "User", num_partitions=None).map_groups(
        lambda batch: batch
    )

    assert isinstance(result._logical_plan.dag, MapBatches)
    assert isinstance(result._logical_plan.dag.input_dependencies[0], Repartition)
    assert (
        "parquet_cudf_shuffle_elision"
        not in inspect.signature(GroupedData.map_groups).parameters
    )
