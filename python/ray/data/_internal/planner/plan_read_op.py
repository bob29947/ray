import logging
import warnings
from typing import Iterable, Iterator, List, Optional, Protocol, runtime_checkable

import ray
from ray import ObjectRef
from ray.data._internal.execution.interfaces import PhysicalOperator, RefBundle
from ray.data._internal.execution.interfaces.task_context import TaskContext
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.execution.operators.map_operator import MapOperator
from ray.data._internal.execution.operators.map_transformer import (
    BlockMapTransformFn,
    MapTransformer,
)
from ray.data._internal.execution.util import memory_string
from ray.data._internal.logical.operators import Read
from ray.data._internal.output_buffer import OutputBlockSizeOption
from ray.data._internal.util import _warn_on_high_parallelism
from ray.data.block import Block, BlockMetadata
from ray.data.context import DataContext
from ray.data.datasource.datasource import ReadTask
from ray.experimental.locations import get_local_object_locations
from ray.util.debug import log_once

TASK_SIZE_WARN_THRESHOLD_BYTES = 1024 * 1024  # 1 MiB

logger = logging.getLogger(__name__)


@runtime_checkable
class _LazyReadTaskDatasource(Protocol):
    """Internal opt-in contract for one-shot, bounded read-task submission."""

    _supports_lazy_read_task_submission: bool

    @property
    def _read_task_count(self) -> int: ...

    @property
    def _read_task_submission_window(self) -> int: ...

    def _iter_read_tasks(
        self,
        parallelism: int,
        per_task_row_limit: Optional[int] = None,
        data_context: Optional[DataContext] = None,
    ) -> Iterator[ReadTask]: ...


def _derive_metadata(read_task: ReadTask, read_task_ref: ObjectRef) -> BlockMetadata:
    # NOTE: Use the `get_local_object_locations` API to get the size of the
    # serialized ReadTask, instead of pickling.
    # Because the ReadTask may capture ObjectRef objects, which cannot
    # be serialized out-of-band.
    locations = get_local_object_locations([read_task_ref])
    task_size = locations[read_task_ref]["object_size"]
    if task_size > TASK_SIZE_WARN_THRESHOLD_BYTES and log_once(
        f"large_read_task_{read_task.read_fn.__name__}"
    ):
        warnings.warn(
            "The serialized size of your read function named "
            f"'{read_task.read_fn.__name__}' is {memory_string(task_size)}. This size "
            "is relatively large. As a result, Ray might excessively "
            "spill objects during execution. To fix this issue, avoid accessing "
            f"`self` or other large objects in '{read_task.read_fn.__name__}'."
        )

    return BlockMetadata(
        num_rows=1,
        size_bytes=task_size,
        exec_stats=None,
        input_files=None,
    )


def _read_task_ref_bundle(read_task: ReadTask) -> RefBundle:
    read_task_ref = ray.put(read_task)
    return RefBundle(
        (
            (
                # TODO: figure out a better way to pass read tasks other than
                # ray.put().
                read_task_ref,
                _derive_metadata(read_task, read_task_ref),
            ),
        ),
        # These refs are roots of the execution DAG, so downstream operators must
        # not explicitly destroy them. Stock InputDataBuffer retains them for
        # replay; the opt-in one-shot buffer keeps them only while queued or used
        # by a submitted read task.
        owns_blocks=False,
        schema=None,
    )


def plan_read_op(
    op: Read,
    physical_children: List[PhysicalOperator],
    data_context: DataContext,
) -> PhysicalOperator:
    """Get the corresponding DAG of physical operators for Read.

    Note this method only converts the given `op`, but not its input dependencies.
    See Planner.plan() for more details.
    """
    assert len(physical_children) == 0

    read_task_source = op.datasource_or_legacy_reader

    def get_parallelism() -> int:
        parallelism = op.get_detected_parallelism()
        assert (
            parallelism is not None
        ), "Read parallelism must be set by the optimizer before execution"
        return parallelism

    def get_input_data(target_max_block_size) -> List[RefBundle]:
        del target_max_block_size
        parallelism = get_parallelism()

        # Get the original read tasks
        read_tasks = read_task_source.get_read_tasks(
            parallelism,
            per_task_row_limit=op.per_block_limit,
            data_context=data_context,
        )

        _warn_on_high_parallelism(parallelism, len(read_tasks))

        ret = []
        for read_task in read_tasks:
            ret.append(_read_task_ref_bundle(read_task))
        return ret

    lazy_read_tasks = (
        getattr(read_task_source, "_supports_lazy_read_task_submission", False) is True
    )
    if lazy_read_tasks:
        if not isinstance(read_task_source, _LazyReadTaskDatasource):
            raise TypeError("Lazy read-task datasource has an incomplete contract.")
        read_task_count = read_task_source._read_task_count
        submission_window = read_task_source._read_task_submission_window
        if read_task_count < 0:
            raise ValueError("Lazy read-task count must be nonnegative.")
        if submission_window <= 0:
            raise ValueError("Lazy read-task submission window must be positive.")

        def get_input_data_iterator(
            target_max_block_size: int,
        ) -> Iterator[RefBundle]:
            del target_max_block_size
            parallelism = get_parallelism()
            _warn_on_high_parallelism(parallelism, read_task_count)
            for read_task in read_task_source._iter_read_tasks(
                parallelism,
                per_task_row_limit=op.per_block_limit,
                data_context=data_context,
            ):
                yield _read_task_ref_bundle(read_task)

        inputs = InputDataBuffer(
            data_context,
            input_data_iterator_factory=get_input_data_iterator,
            estimated_num_output_bundles=read_task_count,
            max_pending_input_blocks=submission_window,
        )
    else:
        # Preserve the stock, replayable read-task behavior for all other
        # datasources.
        inputs = InputDataBuffer(data_context, input_data_factory=get_input_data)

    def do_read(blocks: Iterable[ReadTask], _: TaskContext) -> Iterable[Block]:
        for read_task in blocks:
            yield from read_task()

    # Create a MapTransformer for a read operator
    map_transformer = MapTransformer(
        [
            BlockMapTransformFn(
                do_read,
                is_udf=False,
                output_block_size_option=OutputBlockSizeOption.of(
                    target_max_block_size=data_context.target_max_block_size,
                ),
            ),
        ]
    )

    return MapOperator.create(
        map_transformer,
        inputs,
        data_context,
        name=op.name,
        compute_strategy=op.compute,
        ray_remote_args=op.ray_remote_args,
    )
