import math
import sys
import types

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

import ray
from ray.data._internal.gpu_sort.backend import (
    _ExternalRun,
    _scale_sample_weights,
    _workspace_bounded_payload_bytes,
    lazy_load_backend,
)
from ray.data._internal.gpu_sort.config import GPUSortConfig
from ray.data._internal.gpu_sort.operator import (
    _InputBlock,
    _assign_blocks_by_locality,
    _make_waves,
    _validate_gpu_schema,
    _wave_target_bytes,
)
from ray.data._internal.logical.rules.combine_shuffles import CombineShuffles
from ray.data.tests.conftest import *  # noqa
from ray.tests.conftest import *  # noqa


@pytest.fixture
def gpu_backend_class(monkeypatch):
    """Load the backend methods without importing optional RAPIDS packages."""

    ray_utils = types.ModuleType("rapidsmpf.utils.ray_utils")
    ray_utils.BaseShufflingActor = type("BaseShufflingActor", (), {})
    monkeypatch.setitem(sys.modules, "rapidsmpf.utils.ray_utils", ray_utils)
    return lazy_load_backend()


def test_gpu_sort_api_and_logical_backend(ray_start_regular):
    ds = ray.data.from_arrow(pa.table({"k": [2, 1], "payload": ["b", "a"]}))

    assert ds.sort("k")._logical_plan.dag.backend == "cpu"
    gpu_ds = ds.sort("k", backend="gpu")
    gpu_sort = gpu_ds._logical_plan.dag
    assert gpu_sort.backend == "gpu"
    assert (
        CombineShuffles._combine(
            gpu_ds.sort("payload", backend="gpu")._logical_plan.dag
        ).backend
        == "gpu"
    )

    with pytest.raises(ValueError, match="backend"):
        ds.sort("k", backend="cuda")
    with pytest.raises(ValueError, match="boundaries"):
        ds.sort("k", boundaries=[1], backend="gpu")
    with pytest.raises(NotImplementedError, match="flat Arrow scalar"):
        ray.data.from_arrow(pa.table({"k": [[1], [2]]})).sort("k", backend="gpu")


def test_gpu_sort_schema_and_wave_planning():
    schema = pa.schema(
        [
            pa.field("s", pa.string()),
            pa.field("i", pa.int64()),
            pa.field("f", pa.float64()),
            pa.field("flag", pa.bool_()),
            pa.field("ts", pa.timestamp("us")),
        ]
    )
    _validate_gpu_schema(schema, ["s", "i", "f", "flag", "ts"])
    with pytest.raises(NotImplementedError, match="unsupported keys"):
        _validate_gpu_schema(pa.schema([pa.field("blob", pa.binary())]), ["blob"])

    assert GPUSortConfig(residency_budget_bytes="4 GiB").residency_budget_bytes == (
        4 << 30
    )
    blocks = [
        [_InputBlock("a", 6, 1), _InputBlock("b", 6, 1)],
        [_InputBlock("c", 3, 1)],
    ]
    waves = _make_waves(blocks, target_bytes_per_rank=8)
    assert waves == [[["a"], ["c"]], [["b"], []]]
    assert _make_waves(blocks, target_bytes_per_rank=None) == [[["a", "b"], ["c"]]]


def test_gpu_sort_locality_assignment_balances_decoded_bytes():
    blocks = [
        _InputBlock("n0-large", 9, 1),
        _InputBlock("n0-small", 4, 1),
        _InputBlock("n1", 7, 1),
        _InputBlock("replicated", 7, 1),
        _InputBlock("unknown", 6, 1),
    ]
    locations = {
        "n0-large": {"node_ids": ["node-0"]},
        "n0-small": {"node_ids": ["node-0"]},
        "n1": {"node_ids": ["node-1"]},
        "replicated": {"node_ids": ["node-0", "node-1"]},
    }
    assigned, assigned_bytes, assigned_blocks, local_bytes, local_blocks = (
        _assign_blocks_by_locality(blocks, ["node-0", "node-1"], locations)
    )

    assert [[block.value for block in rank] for rank in assigned] == [
        ["n0-large", "n0-small", "unknown"],
        ["n1", "replicated"],
    ]
    assert assigned_bytes == [19, 14]
    assert assigned_blocks == [3, 2]
    assert local_bytes == [13, 14]
    assert local_blocks == [2, 2]


def test_gpu_sort_automatic_wave_target_uses_smallest_actor_budget():
    gib = 1 << 30
    small = [
        [_InputBlock("a", 2 * gib, 1)],
        [_InputBlock("b", 3 * gib, 1)],
    ]
    assert (
        _wave_target_bytes(
            small,
            explicit_residency_budget_bytes=None,
            actor_usable_budgets=[20 * gib, 16 * gib],
            auto_wave_fraction=0.50,
        )
        is None
    )

    large = [
        [_InputBlock("a", 5 * gib, 1), _InputBlock("b", 4 * gib, 1)],
        [_InputBlock("c", 2 * gib, 1)],
    ]
    assert _wave_target_bytes(
        large,
        explicit_residency_budget_bytes=None,
        actor_usable_budgets=[20 * gib, 16 * gib],
        auto_wave_fraction=0.50,
    ) == 8 * gib
    assert _wave_target_bytes(
        large,
        explicit_residency_budget_bytes=None,
        actor_usable_budgets=[20 * gib, 16 * gib],
        auto_wave_fraction=0.375,
    ) == 6 * gib
    assert _wave_target_bytes(
        large,
        explicit_residency_budget_bytes=4 * gib,
        actor_usable_budgets=[],
        auto_wave_fraction=0.50,
    ) == 2 * gib

    with pytest.raises(RuntimeError, match="usable memory budgets"):
        _wave_target_bytes(
            large,
            explicit_residency_budget_bytes=None,
            actor_usable_budgets=[],
            auto_wave_fraction=0.50,
        )
    with pytest.raises(ValueError, match="wave fraction"):
        GPUSortConfig(auto_wave_fraction=0)


def test_gpu_sort_inverse_inclusion_sample_weights():
    sampled = np.asarray([5, 5], dtype=np.uint64)
    small = _scale_sample_weights(sampled, population_rows=4, sample_rows=2)
    large = _scale_sample_weights(sampled, population_rows=12, sample_rows=2)
    assert small.tolist() == [10, 10]
    assert large.tolist() == [30, 30]
    assert small.dtype == large.dtype == np.dtype("uint64")

    uneven = _scale_sample_weights(
        np.asarray([5, 7], dtype=np.uint64), population_rows=5, sample_rows=2
    )
    assert uneven.tolist() == [12, 17]


def test_gpu_sort_comparator_matches_arrow_null_nan_order(
    gpu_backend_class, monkeypatch
):
    order = types.SimpleNamespace(ASCENDING="asc", DESCENDING="desc")
    null_order = types.SimpleNamespace(BEFORE="before", AFTER="after")
    plc = types.ModuleType("pylibcudf")
    plc.types = types.SimpleNamespace(Order=order, NullOrder=null_order)
    monkeypatch.setitem(sys.modules, "pylibcudf", plc)

    values = pa.array([None, float("nan"), -1.0, 2.0])
    expected = {
        True: [-1.0, 2.0, "nan", None],
        False: [2.0, -1.0, "nan", None],
    }
    for ascending in (True, False):
        indices = pc.sort_indices(
            pa.table({"key": values}),
            sort_keys=[("key", "ascending" if ascending else "descending")],
            null_placement="at_end",
        )
        actual = pc.take(values, indices).to_pylist()
        normalized = [
            "nan" if isinstance(value, float) and math.isnan(value) else value
            for value in actual
        ]
        assert normalized == expected[ascending]

        backend = object.__new__(gpu_backend_class)
        backend._key_columns = ["key"]
        backend._ascending = [ascending]
        backend._float_hidden = {"key": ("is_null", "is_nan")}
        backend._config = GPUSortConfig(null_position="last")
        orders, nulls = backend._order_and_nulls()
        assert orders == [
            order.ASCENDING,
            order.ASCENDING,
            order.ASCENDING if ascending else order.DESCENDING,
        ]
        assert nulls == [
            null_order.AFTER,
            null_order.AFTER,
            null_order.AFTER,
        ]

        backend._float_hidden = {}
        for null_position, expected_null_order in (
            ("first", null_order.BEFORE),
            ("last", null_order.AFTER),
        ):
            backend._config = GPUSortConfig(null_position=null_position)
            direct_orders, direct_nulls = backend._order_and_nulls()
            assert direct_orders == [
                order.ASCENDING if ascending else order.DESCENDING
            ]
            assert direct_nulls == [expected_null_order]


def test_gpu_sort_typed_all_null_arrow_output(gpu_backend_class):
    class Column:
        null_count = 3

        def to_arrow(self):
            raise AssertionError("all-null columns must use the typed Arrow path")

    class Series:
        _column = Column()

    class Frame:
        def __len__(self):
            return 3

        def __getitem__(self, name):
            return Series()

    schema = pa.schema(
        [
            pa.field("name", pa.string()),
            pa.field("count", pa.int64()),
            pa.field("ratio", pa.float64()),
            pa.field("when", pa.timestamp("us")),
        ]
    )
    table = gpu_backend_class._generic_frame_to_arrow(Frame(), schema)
    assert table.schema.equals(schema, check_metadata=True)
    assert table.num_rows == 3
    for field, column in zip(schema, table.columns):
        assert column.type.equals(field.type)
        assert column.null_count == 3


def test_gpu_sort_equal_keys_are_deterministically_balanced(
    gpu_backend_class, monkeypatch
):
    captured = {}

    class Table:
        def __init__(self, rows):
            self.rows = rows

        def num_rows(self):
            return self.rows

    class Frame:
        columns = ["key"]

        def __init__(self, rows):
            self.table = Table(rows)

    class Series:
        def __init__(self, values):
            self.values = list(values)

        @classmethod
        def from_pylibcudf(cls, values):
            return cls(values)

        def to_pylibcudf(self):
            return self.values, None

    def elementwise_kernel(_inputs, _output, operation, _name):
        assert "(base + static_cast<unsigned long long>(i)) % width" in operation

        def apply(lower, upper, base):
            return [
                low + (int(base) + index) % (high - low + 1)
                for index, (low, high) in enumerate(zip(lower, upper))
            ]

        return apply

    cp = types.ModuleType("cupy")
    cp.ElementwiseKernel = elementwise_kernel
    cp.uint64 = int
    monkeypatch.setitem(sys.modules, "cupy", cp)

    cudf = types.ModuleType("cudf")
    cudf.Series = Series
    monkeypatch.setitem(sys.modules, "cudf", cudf)

    plc = types.ModuleType("pylibcudf")
    plc.search = types.SimpleNamespace(
        lower_bound=lambda *_: [1] * captured["rows"],
        upper_bound=lambda *_: [3] * captured["rows"],
    )

    def partition(table, destinations, num_partitions):
        captured["destinations"] = list(destinations)
        counts = [
            captured["destinations"].count(index) for index in range(num_partitions)
        ]
        offsets = [0]
        for count in counts:
            offsets.append(offsets[-1] + count)
        return table, offsets

    plc.partitioning = types.SimpleNamespace(partition=partition)
    monkeypatch.setitem(sys.modules, "pylibcudf", plc)

    partition_module = types.ModuleType("rapidsmpf.integrations.cudf.partition")
    partition_module.split_and_pack = lambda *_: list(captured["destinations"])
    monkeypatch.setitem(
        sys.modules, "rapidsmpf.integrations.cudf.partition", partition_module
    )
    cudf_utils = types.ModuleType("rapidsmpf.utils.cudf")
    cudf_utils.cudf_to_pylibcudf_table = lambda frame: frame.table
    monkeypatch.setitem(sys.modules, "rapidsmpf.utils.cudf", cudf_utils)
    stream = types.ModuleType("rmm.pylibrmm.stream")
    stream.DEFAULT_STREAM = object()
    monkeypatch.setitem(sys.modules, "rmm.pylibrmm.stream", stream)

    backend = object.__new__(gpu_backend_class)
    backend._duplicate_kernel = None
    backend._row_ordinal = 0
    backend._boundary_keys = object()
    backend._num_partitions = 4
    backend._buffer_resource = object()
    backend._augment_table = lambda table, names: (table, names)
    backend._comparison_table = lambda table, names: table
    backend._order_and_nulls = lambda: ([], [])
    backend.rank = lambda: 0

    def distribute(rows):
        captured["rows"] = rows
        return backend._partition_and_pack(Frame(rows), wave_id=0)

    first = distribute(8)
    second = distribute(8)
    assert first == [1, 2, 3, 1, 2, 3, 1, 2]
    assert set(first + second) == {1, 2, 3}
    counts = [(first + second).count(partition) for partition in (1, 2, 3)]
    assert max(counts) - min(counts) <= 1


def test_gpu_sort_transitions_before_residency_overflow(gpu_backend_class):
    class Table:
        def __init__(self, rows, size_bytes):
            self.rows = rows
            self.size_bytes = size_bytes

        def num_rows(self):
            return self.rows

    backend = object.__new__(gpu_backend_class)
    backend._payload_limit_bytes = 10
    backend._device_tables = {0: []}
    backend._device_bytes = {0: 0}
    backend._stats = {"peak_live_bytes": 0}
    backend._table_bytes = lambda table: table.size_bytes
    backend._slice_table = lambda table, start, end: Table(
        end - start, max(1, table.size_bytes * (end - start) // table.rows)
    )
    transitions = []

    def externalize(partition, wave_id):
        transitions.append((partition, wave_id, backend._device_bytes[partition]))
        backend._device_tables[partition] = []
        backend._device_bytes[partition] = 0

    backend._externalize_device_tables = externalize
    backend._accept_received(0, Table(6, 6), wave_id=7)
    backend._accept_received(0, Table(6, 6), wave_id=7)
    assert transitions == [(0, 7, 6)]
    assert backend._device_bytes[0] == 6

    backend._accept_received(0, Table(4, 4), wave_id=8)
    assert transitions[-1] == (0, 8, 10)
    assert backend._device_bytes[0] == 0
    assert backend._stats["peak_live_bytes"] <= backend._payload_limit_bytes


def test_gpu_sort_externalization_reserves_live_allocator_workspace(
    gpu_backend_class, monkeypatch
):
    class Table:
        def __init__(self, rows, size_bytes):
            self.rows = rows
            self.size_bytes = size_bytes

        def num_rows(self):
            return self.rows

    cp = types.ModuleType("cupy")
    cp.cuda = types.SimpleNamespace(
        runtime=types.SimpleNamespace(deviceSynchronize=lambda: None)
    )
    monkeypatch.setitem(sys.modules, "cupy", cp)
    monkeypatch.setitem(sys.modules, "pylibcudf", types.ModuleType("pylibcudf"))

    assert _workspace_bounded_payload_bytes(100, 50, 3.0, 6) == 22

    backend = object.__new__(gpu_backend_class)
    backend._config = GPUSortConfig(
        exchange_batch_bytes=10,
        final_sort_workspace_factor=3.0,
    )
    backend._pool_max_bytes = 100
    backend._mr = types.SimpleNamespace(current_allocated=50)
    backend._payload_limit_bytes = 40
    backend._device_tables = {0: [Table(40, 40)]}
    backend._device_bytes = {0: 40}
    backend._runs = {0: []}
    backend._started_at = 0.0
    backend._stats = {
        "state": "DEVICE_ACCUMULATING",
        "mode": "resident",
        "first_externalize_s": None,
        "first_externalize_wave": None,
        "peak_device_bytes": 0,
        "phases_s": {"run_sort": 0.0},
    }
    backend._slice_table = lambda table, start, end: Table(
        end - start, table.size_bytes * (end - start) // table.rows
    )
    backend._table_bytes = lambda table: table.size_bytes
    sorted_sizes = []

    def sort_table(table):
        sorted_sizes.append(table.size_bytes)
        return table

    backend._sort_table = sort_table
    backend._store_table_as_run = lambda table, initial: _ExternalRun()
    backend._externalize_device_tables(0, wave_id=3)

    assert sorted_sizes == [25, 15]
    assert len(backend._runs[0]) == 2
    assert backend._stats["state"] == "EXTERNAL_RUNS"
    assert backend._stats["first_externalize_wave"] == 3


def test_gpu_sort_externalization_bounds_multitable_concat(
    gpu_backend_class, monkeypatch
):
    class MemoryResource:
        def __init__(self, current_allocated):
            self.current_allocated = current_allocated
            self.peak = current_allocated

        def allocate(self, size_bytes):
            self.current_allocated += size_bytes
            self.peak = max(self.peak, self.current_allocated)

        def free(self, size_bytes):
            self.current_allocated -= size_bytes

    memory = MemoryResource(current_allocated=10)

    class Table:
        def __init__(self, rows, size_bytes, *, owns_allocation=False):
            self.rows = rows
            self.size_bytes = size_bytes
            self.owns_allocation = owns_allocation
            if owns_allocation:
                memory.allocate(size_bytes)

        def num_rows(self):
            return self.rows

        def __del__(self):
            if self.owns_allocation:
                self.owns_allocation = False
                memory.free(self.size_bytes)

    cp = types.ModuleType("cupy")
    cp.cuda = types.SimpleNamespace(
        runtime=types.SimpleNamespace(deviceSynchronize=lambda: None)
    )
    monkeypatch.setitem(sys.modules, "cupy", cp)
    plc = types.ModuleType("pylibcudf")
    concat_sizes = []

    def bounded_concat(tables):
        size_bytes = sum(table.size_bytes for table in tables)
        if memory.current_allocated + size_bytes > 100:
            raise AssertionError("concatenate exceeded the RMM pool")
        concat_sizes.append(size_bytes)
        return Table(
            sum(table.rows for table in tables),
            size_bytes,
            owns_allocation=True,
        )

    plc.concatenate = types.SimpleNamespace(concatenate=bounded_concat)
    monkeypatch.setitem(sys.modules, "pylibcudf", plc)

    backend = object.__new__(gpu_backend_class)
    backend._config = GPUSortConfig(final_sort_workspace_factor=3.0)
    backend._pool_max_bytes = 100
    backend._mr = memory
    backend._payload_limit_bytes = 50
    backend._device_tables = {
        0: [Table(10, 10, owns_allocation=True) for _ in range(5)]
    }
    backend._device_bytes = {0: 50}
    backend._runs = {0: []}
    backend._started_at = 0.0
    backend._stats = {
        "state": "DEVICE_ACCUMULATING",
        "mode": "resident",
        "first_externalize_s": None,
        "first_externalize_wave": None,
        "peak_device_bytes": 0,
        "phases_s": {"run_sort": 0.0},
    }
    backend._slice_table = lambda table, start, end: (
        table
        if start == 0 and end == table.rows
        else Table(
            end - start,
            table.size_bytes * (end - start) // table.rows,
        )
    )
    backend._table_bytes = lambda table: table.size_bytes
    sorted_sizes = []

    def sort_table(table):
        assert (
            memory.current_allocated + 2 * table.size_bytes
            <= backend._pool_max_bytes
        )
        sorted_sizes.append(table.size_bytes)
        return table

    backend._sort_table = sort_table
    backend._store_table_as_run = lambda table, initial: _ExternalRun()
    backend._externalize_device_tables(0, wave_id=4)

    # Concatenating all five sources would require 60 + 50 > 100 bytes.
    # Bounded groups allocate and then release 20 and 30 source bytes.
    assert concat_sizes == [20, 30]
    assert sorted_sizes == [20, 30]
    assert len(backend._runs[0]) == 2
    assert backend._device_tables[0] == []
    assert memory.current_allocated == 10
    assert memory.peak == 80


def test_gpu_sort_merge_passes_have_bounded_fan_in(gpu_backend_class):
    backend = object.__new__(gpu_backend_class)
    backend._config = GPUSortConfig(merge_fan_in=4)
    backend._stats = {"merge_pass_count": 0}
    group_sizes = []

    def merge_group(group):
        group_sizes.append(len(group))
        return _ExternalRun()

    backend._merge_group = merge_group
    result = backend._merge_runs([_ExternalRun() for _ in range(17)])
    assert isinstance(result, _ExternalRun)
    assert backend._stats["merge_pass_count"] == 3
    assert group_sizes
    assert all(2 <= size <= 4 for size in group_sizes)
