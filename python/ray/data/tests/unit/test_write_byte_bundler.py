import pytest
import pyarrow as pa

import ray
from ray.data._internal.compute import TaskPoolStrategy
from ray.data._internal.execution.interfaces import RefBundle
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.execution.operators.map_operator import BytesRefBundler
from ray.data._internal.logical.operators import InputData, Write
from ray.data._internal.planner.plan_write_op import (
    _split_blocks_by_bytes,
    plan_write_op,
)
from ray.data.block import BlockAccessor, BlockMetadata
from ray.data.context import DataContext
from ray.data.datasource.datasink import Datasink


def _bundle(identifier: int, size_bytes: int) -> RefBundle:
    block_ref = ray.ObjectRef(bytes([identifier]) * 28)
    metadata = BlockMetadata(
        num_rows=1,
        size_bytes=size_bytes,
        exec_stats=None,
        input_files=None,
    )
    return RefBundle(((block_ref, metadata),), owns_blocks=False, schema=None)


def _drain_ready(bundler: BytesRefBundler):
    outputs = []
    while bundler.has_bundle():
        inputs, output = bundler.get_next_bundle()
        outputs.append((inputs, output))
    return outputs


def test_bytes_ref_bundler_preserves_consecutive_order_and_target():
    bundles = [_bundle(index, size) for index, size in enumerate([3, 4, 5, 2])]
    bundler = BytesRefBundler(target_bytes_per_bundle=7)

    outputs = []
    for bundle in bundles:
        bundler.add_bundle(bundle)
        outputs.extend(_drain_ready(bundler))
    bundler.done_adding_bundles()
    outputs.extend(_drain_ready(bundler))

    assert [inputs for inputs, _ in outputs] == [bundles[:2], bundles[2:]]
    assert [output.size_bytes() for _, output in outputs] == [7, 7]
    assert [block_ref for _, output in outputs for block_ref in output.block_refs] == [
        bundle.block_refs[0] for bundle in bundles
    ]
    assert bundler.num_blocks() == 0
    assert bundler.size_bytes() == 0


def test_bytes_ref_bundler_flushes_before_crossing_target():
    bundles = [_bundle(index, size) for index, size in enumerate([6, 5, 4])]
    bundler = BytesRefBundler(target_bytes_per_bundle=10)

    # The second input makes a complete prefix available, but isn't combined with
    # the first input because that would exceed the byte target.
    bundler.add_bundle(bundles[0])
    assert not bundler.has_bundle()
    bundler.add_bundle(bundles[1])
    inputs, output = bundler.get_next_bundle()
    assert inputs == [bundles[0]]
    assert output.size_bytes() == 6

    bundler.add_bundle(bundles[2])
    bundler.done_adding_bundles()
    inputs, output = bundler.get_next_bundle()
    assert inputs == bundles[1:]
    assert output.size_bytes() == 9


def test_bytes_ref_bundler_allows_only_indivisible_singleton_to_exceed_target():
    oversized = _bundle(0, 11)
    bundler = BytesRefBundler(target_bytes_per_bundle=10)

    bundler.add_bundle(oversized)
    inputs, output = bundler.get_next_bundle()

    assert inputs == [oversized]
    assert output.size_bytes() == 11
    assert bundler.size_bytes() == 0


def test_split_blocks_by_bytes_preserves_rows_schema_and_hard_bound():
    schema = pa.schema(
        [pa.field("key", pa.string()), pa.field("row_id", pa.int64())],
        metadata={b"source": b"write-rechunk-test"},
    )
    table = pa.Table.from_arrays(
        [
            pa.array(["x" * (5 + index % 7) for index in range(100)]),
            pa.array(range(100), type=pa.int64()),
        ],
        schema=schema,
    )
    target = table.slice(0, 9).nbytes

    pieces = list(_split_blocks_by_bytes([table], None, max_bytes_per_block=target))

    assert len(pieces) > 1
    assert all(
        BlockAccessor.for_block(piece).size_bytes() <= target for piece in pieces
    )
    combined = pa.concat_tables(pieces)
    assert combined.schema.equals(schema, check_metadata=True)
    assert combined["row_id"].to_pylist() == list(range(100))


def test_split_blocks_by_bytes_rejects_single_row_over_target():
    table = pa.table({"value": ["oversized-row"]})

    with pytest.raises(ValueError, match="single row exceeds"):
        list(_split_blocks_by_bytes([table], None, max_bytes_per_block=1))


@pytest.mark.parametrize("target", [0, -1, 1.5, True])
def test_bytes_ref_bundler_rejects_invalid_target(target):
    with pytest.raises(ValueError, match="positive integer"):
        BytesRefBundler(target)


def test_write_plan_uses_byte_bundler_and_preserves_concurrency_limit():
    class ByteBoundedDatasink(Datasink):
        @property
        def target_bytes_per_write(self):
            return 4 * 1024**3

        def write(self, blocks, ctx):
            raise AssertionError("The unit test doesn't execute write tasks")

    context = DataContext.get_current()
    logical_input = InputData([])
    physical_input = InputDataBuffer(context, input_data=[])
    write = Write(
        logical_input,
        ByteBoundedDatasink(),
        compute=TaskPoolStrategy(size=16),
    )

    physical_write = plan_write_op(write, [physical_input], context)

    assert isinstance(physical_write._block_ref_bundler, BytesRefBundler)
    assert physical_write._block_ref_bundler._target_bytes_per_bundle == 4 * 1024**3
    assert physical_write.get_max_concurrency_limit() == 16
    assert not physical_write.supports_fusion()
    split_write_input = physical_write.input_dependencies[0]
    assert split_write_input.name == "SplitWriteInput"
    assert split_write_input.input_dependencies == [physical_input]
    assert split_write_input.get_max_concurrency_limit() == 16
    assert not split_write_input.supports_fusion()

    # Already-bounded refs bypass the rechunk task and retain object identity.
    bounded = _bundle(7, 1024)
    split_write_input.start(context.execution_options)
    split_write_input.add_input(bounded, 0)
    output = split_write_input.get_next()
    assert output.block_refs == bounded.block_refs
    assert split_write_input.num_active_tasks() == 0
    assert split_write_input._extra_metrics()["write_input_passthrough_bundles"] == 1
    assert split_write_input._extra_metrics()["write_input_rechunked_bundles"] == 0


def test_write_rejects_row_and_byte_bundling_together():
    class ConflictingDatasink(Datasink):
        @property
        def min_rows_per_write(self):
            return 10

        @property
        def target_bytes_per_write(self):
            return 100

    with pytest.raises(ValueError, match="can't set both"):
        Write(InputData([]), ConflictingDatasink())
