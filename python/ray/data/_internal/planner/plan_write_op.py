import functools
import itertools
import uuid
from typing import TYPE_CHECKING, Callable, Iterable, Iterator, List, Optional, Union

from ray.data._internal.compute import TaskPoolStrategy
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.execution.interfaces.ref_bundle import RefBundle
from ray.data._internal.execution.interfaces.task_context import TaskContext
from ray.data._internal.execution.operators.map_operator import (
    BytesRefBundler,
    MapOperator,
)
from ray.data._internal.execution.operators.map_transformer import (
    BlockMapTransformFn,
    MapTransformer,
)
from ray.data._internal.execution.operators.task_pool_map_operator import (
    TaskPoolMapOperator,
)
from ray.data.block import Block, BlockAccessor
from ray.data.context import DataContext
from ray.data.datasource.datasink import Datasink
from ray.data.datasource.datasource import Datasource

if TYPE_CHECKING:
    from ray.data._internal.logical.operators import Write

WRITE_UUID_KWARG_NAME = "write_uuid"
# Key for storing pending checkpoint paths for commit phase
PENDING_CHECKPOINTS_KWARG_NAME = "_pending_checkpoints"


class _SplitWriteInputTaskPoolMapOperator(TaskPoolMapOperator):
    """Only run the rechunk task when an input bundle crosses the hard cap."""

    def __init__(self, *args, max_bytes_per_bundle: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_bytes_per_bundle = max_bytes_per_bundle
        self._passthrough_bundles = 0
        self._rechunked_bundles = 0

    def _add_input_inner(self, refs: RefBundle, input_index: int):
        if refs.size_bytes() > self._max_bytes_per_bundle:
            self._rechunked_bundles += 1
            return super()._add_input_inner(refs, input_index)

        # Avoid serializing every already-bounded block through an identity map.
        # Allocate a completed queue key so pass-through and asynchronously split
        # bundles still share one contiguous ordered output sequence.
        assert input_index == 0, input_index
        assert self._output_queue is not None
        self._metrics.on_input_queued(refs, input_index=0)
        self._metrics.on_input_dequeued(refs, input_index=0)
        output_index = self._next_data_task_idx
        self._next_data_task_idx += 1
        self._output_queue.add(refs, key=output_index)
        self._output_queue.finalize(key=output_index)
        self._metrics.on_output_queued(refs)
        self._passthrough_bundles += 1

    def _extra_metrics(self):
        return {
            **super()._extra_metrics(),
            "write_input_passthrough_bundles": self._passthrough_bundles,
            "write_input_rechunked_bundles": self._rechunked_bundles,
            "write_input_max_bytes_per_bundle": self._max_bytes_per_bundle,
        }


def _split_blocks_by_bytes(
    blocks: Iterable[Block],
    _ctx: TaskContext,
    *,
    max_bytes_per_block: int,
) -> Iterator[Block]:
    """Split blocks into consecutive zero-copy row slices under a byte cap.

    This runs as its own streaming map stage immediately before a byte-bounded
    write.  Keeping it separate from the write map is important: the resulting
    object refs, rather than the original oversized ref, are what the write
    bundler uses as task inputs.
    """

    for block in blocks:
        accessor = BlockAccessor.for_block(block)
        block_bytes = accessor.size_bytes()
        num_rows = accessor.num_rows()
        if block_bytes <= max_bytes_per_block:
            yield block
            continue
        if num_rows <= 0:
            raise ValueError(
                "A zero-row block exceeds target_bytes_per_write and cannot be "
                f"split: block_bytes={block_bytes}, target={max_bytes_per_block}."
            )

        offset = 0
        while offset < num_rows:
            rows_remaining = num_rows - offset

            # Find the largest nonempty prefix that fits. Block sizes for row
            # slices are monotonic for the tabular blocks accepted by Ray Data.
            low = 1
            high = rows_remaining
            one_row = accessor.slice(offset, offset + 1, copy=False)
            one_row_bytes = BlockAccessor.for_block(one_row).size_bytes()
            if one_row_bytes > max_bytes_per_block:
                raise ValueError(
                    "A single row exceeds target_bytes_per_write and cannot be "
                    f"split: row_bytes={one_row_bytes}, "
                    f"target={max_bytes_per_block}."
                )
            while low < high:
                middle = (low + high + 1) // 2
                candidate = accessor.slice(offset, offset + middle, copy=False)
                candidate_bytes = BlockAccessor.for_block(candidate).size_bytes()
                if candidate_bytes <= max_bytes_per_block:
                    low = middle
                else:
                    high = middle - 1

            piece = accessor.slice(offset, offset + low, copy=False)
            piece_bytes = BlockAccessor.for_block(piece).size_bytes()
            assert piece_bytes <= max_bytes_per_block
            yield piece
            offset += low


def generate_write_fn(
    datasink_or_legacy_datasource: Union[Datasink, Datasource], **write_args
) -> Callable[[Iterator[Block], TaskContext], Iterator[Block]]:
    def fn(blocks: Iterator[Block], ctx: TaskContext) -> Iterator[Block]:
        """Writes the blocks to the given datasink or legacy datasource.

        Outputs the original blocks to be written."""
        # Create a copy of the iterator, so we can return the original blocks.
        it1, it2 = itertools.tee(blocks, 2)
        if isinstance(datasink_or_legacy_datasource, Datasink):
            ctx.kwargs["_datasink_write_return"] = datasink_or_legacy_datasource.write(
                it1, ctx
            )
        else:
            datasink_or_legacy_datasource.write(it1, ctx, **write_args)

        return it2

    return fn


def generate_collect_write_stats_fn() -> BlockMapTransformFn:
    # If the write op succeeds, the resulting Dataset is a list of
    # one Block which contain stats/metrics about the write.
    # Otherwise, an error will be raised. The Datasource can handle
    # execution outcomes with `on_write_complete()`` and `on_write_failed()``.
    def fn(blocks: Iterator[Block], ctx: TaskContext) -> Iterator[Block]:
        """Handles stats collection for block writes."""
        block_accessors = [BlockAccessor.for_block(block) for block in blocks]
        total_num_rows = sum(ba.num_rows() for ba in block_accessors)
        total_size_bytes = sum(ba.size_bytes() for ba in block_accessors)

        # NOTE: Write tasks can return anything, so we need to wrap it in a valid block
        # type.
        import pandas as pd

        block = pd.DataFrame(
            {
                "num_rows": [total_num_rows],
                "size_bytes": [total_size_bytes],
                "write_return": [ctx.kwargs.get("_datasink_write_return", None)],
            }
        )
        return iter([block])

    return BlockMapTransformFn(
        fn,
        is_udf=False,
        disable_block_shaping=True,
    )


def plan_write_op(
    op: "Write",
    physical_children: List[PhysicalOperator],
    data_context: DataContext,
) -> PhysicalOperator:
    collect_stats_fn = generate_collect_write_stats_fn()

    return _plan_write_op_internal(
        op,
        physical_children,
        data_context,
        post_transformations=[collect_stats_fn],
    )


def _plan_write_op_internal(
    op: "Write",
    physical_children: List[PhysicalOperator],
    data_context: DataContext,
    post_transformations: List[BlockMapTransformFn],
    pre_transformations: Optional[List[BlockMapTransformFn]] = None,
) -> PhysicalOperator:
    """Plan a write operation with optional pre and post write transformations.

    Args:
        op: The write operator.
        physical_children: The physical children operators.
        data_context: The data context.
        post_transformations: Transformations to run AFTER the write.
        pre_transformations: Transformations to run BEFORE the write.
            Useful for 2-phase commit where pending checkpoint is written first.

    Returns:
        The physical operator for the write operation.
    """
    assert len(physical_children) == 1
    input_physical_dag = physical_children[0]

    datasink = op.datasink_or_legacy_datasource
    write_fn = generate_write_fn(datasink, **op.write_args)

    # Build transform chain: pre_write -> write -> post_write
    pre_transforms = pre_transformations or []
    write_transform = BlockMapTransformFn(
        write_fn,
        is_udf=False,
        # NOTE: No need for block-shaping
        disable_block_shaping=True,
    )
    transform_fns = pre_transforms + [write_transform] + post_transformations

    map_transformer = MapTransformer(transform_fns)

    # Set up on_start callback for datasinks.
    # This allows on_write_start to receive the schema from the first input bundle,
    # enabling schema-dependent initialization (e.g., Iceberg schema evolution).
    on_start = None
    if isinstance(datasink, Datasink):
        on_start = datasink.on_write_start

    target_bytes_per_bundle = op.target_bytes_per_bundled_input
    if target_bytes_per_bundle is not None:
        split_transformer = MapTransformer(
            [
                BlockMapTransformFn(
                    functools.partial(
                        _split_blocks_by_bytes,
                        max_bytes_per_block=target_bytes_per_bundle,
                    ),
                    disable_block_shaping=True,
                )
            ]
        )
        # ``Dataset.write_datasink`` uses a task pool. Its selective operator
        # passes already-bounded refs through without a 1 TB identity copy, but
        # sends an oversized bundle through the streaming split transformer.
        if isinstance(op.compute, TaskPoolStrategy):
            input_physical_dag = _SplitWriteInputTaskPoolMapOperator(
                split_transformer,
                input_physical_dag,
                data_context,
                name="SplitWriteInput",
                max_concurrency=op.compute.size,
                max_bytes_per_bundle=target_bytes_per_bundle,
                supports_fusion=False,
                ray_remote_args=op.ray_remote_args,
            )
        else:
            input_physical_dag = MapOperator.create(
                split_transformer,
                input_physical_dag,
                data_context,
                name="SplitWriteInput",
                compute_strategy=op.compute,
                ray_remote_args=op.ray_remote_args,
                supports_fusion=False,
            )
    ref_bundler = (
        BytesRefBundler(target_bytes_per_bundle)
        if target_bytes_per_bundle is not None
        else None
    )

    map_op = MapOperator.create(
        map_transformer,
        input_physical_dag,
        data_context,
        name="Write",
        # Add a UUID to write tasks to prevent filename collisions. This a UUID for the
        # overall write operation, not the individual write tasks.
        map_task_kwargs={WRITE_UUID_KWARG_NAME: uuid.uuid4().hex},
        ray_remote_args=op.ray_remote_args,
        min_rows_per_bundle=op.min_rows_per_bundled_input,
        ref_bundler=ref_bundler,
        compute_strategy=op.compute,
        on_start=on_start,
        # Byte-bounded writes need a task boundary after the upstream operator so
        # that fusion can't replace the dedicated input bundler.
        supports_fusion=target_bytes_per_bundle is None,
    )

    return map_op
