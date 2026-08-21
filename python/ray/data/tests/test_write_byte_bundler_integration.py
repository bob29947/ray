import pyarrow as pa

import ray
from ray.data.block import BlockAccessor
from ray.data.context import DataContext
from ray.data.datasource.datasink import Datasink


def test_oversized_block_is_rechunked_before_write_task(ray_start_regular_shared):
    class RecordingSink(Datasink):
        def __init__(self, target):
            self._target = target
            self.receipts = None

        @property
        def target_bytes_per_write(self):
            return self._target

        def write(self, blocks, ctx):
            blocks = list(blocks)
            return {
                "task_index": ctx.task_idx,
                "input_bytes": sum(
                    BlockAccessor.for_block(block).size_bytes() for block in blocks
                ),
                "row_ids": [
                    row_id
                    for block in blocks
                    for row_id in BlockAccessor.for_block(block)
                    .to_arrow()["row_id"]
                    .to_pylist()
                ],
            }

        def on_write_complete(self, write_result):
            self.receipts = write_result.write_returns

    table = pa.table(
        {
            "key": ["same-duplicate-key"] * 200,
            "row_id": pa.array(range(200), type=pa.int64()),
        }
    )
    target = table.slice(0, 17).nbytes
    assert table.nbytes > target
    sink = RecordingSink(target)
    # Exercise pass-through -> asynchronous rechunk -> pass-through ordering.
    input_blocks = [table.slice(0, 10), table.slice(10, 180), table.slice(190)]
    assert input_blocks[0].nbytes <= target
    assert input_blocks[1].nbytes > target
    assert input_blocks[2].nbytes <= target
    context = DataContext.get_current()
    previous_preserve_order = context.execution_options.preserve_order
    context.execution_options.preserve_order = True
    try:
        ray.data.from_arrow(input_blocks).write_datasink(sink, concurrency=2)
    finally:
        context.execution_options.preserve_order = previous_preserve_order

    assert sink.receipts is not None
    assert len(sink.receipts) > 1
    assert all(receipt["input_bytes"] <= target for receipt in sink.receipts)
    ordered = sorted(sink.receipts, key=lambda receipt: receipt["task_index"])
    assert [row_id for receipt in ordered for row_id in receipt["row_ids"]] == list(
        range(200)
    )
