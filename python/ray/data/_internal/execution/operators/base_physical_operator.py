import abc
import threading
import time
import typing
from typing import Any, Dict, List, Optional

from ray.data._internal.execution.bundle_queue import FIFOBundleQueue
from ray.data._internal.execution.interfaces import (
    AllToAllTransformFn,
    PhysicalOperator,
    RefBundle,
    TaskContext,
)
from ray.data._internal.execution.operators.sub_progress import SubProgressBarMixin
from ray.data._internal.logical.interfaces import LogicalOperator
from ray.data._internal.stats import StatsDict
from ray.data.context import DataContext

if typing.TYPE_CHECKING:
    from ray.data._internal.progress.base_progress import BaseProgressBar


_ALL_TO_ALL_TELEMETRY_TASK_CONTEXT_KEY = "_all_to_all_execution_telemetry"
_last_all_to_all_stats: Optional[Dict[str, Any]] = None
_last_all_to_all_stats_lock = threading.Lock()


def reset_last_all_to_all_stats() -> None:
    """Clear the most recently completed all-to-all execution telemetry."""
    global _last_all_to_all_stats
    with _last_all_to_all_stats_lock:
        _last_all_to_all_stats = None


def get_last_all_to_all_stats() -> Optional[Dict[str, Any]]:
    """Return telemetry for the active or most recently finished all-to-all.

    This is intentionally driver-local. It is primarily useful for benchmarks that
    need to prove when a blocking all-to-all retained input and began its bulk work.
    Publishing bounded snapshots while input is collected also preserves capacity
    evidence if the driver is killed before the blocking bulk function returns.
    """
    with _last_all_to_all_stats_lock:
        if _last_all_to_all_stats is None:
            return None
        return dict(_last_all_to_all_stats)


def _publish_last_all_to_all_stats(stats: Dict[str, Any]) -> None:
    global _last_all_to_all_stats
    with _last_all_to_all_stats_lock:
        _last_all_to_all_stats = dict(stats)


class InternalQueueOperatorMixin(PhysicalOperator, abc.ABC):
    @abc.abstractmethod
    def internal_input_queue_num_blocks(self) -> int:
        """Returns Operator's internal input queue size (in blocks)"""
        ...

    @abc.abstractmethod
    def internal_input_queue_num_bytes(self) -> int:
        """Returns Operator's internal input queue size (in bytes)"""
        ...

    @abc.abstractmethod
    def internal_output_queue_num_blocks(self) -> int:
        """Returns Operator's internal output queue size (in blocks)"""
        ...

    @abc.abstractmethod
    def internal_output_queue_num_bytes(self) -> int:
        """Returns Operator's internal output queue size (in bytes)"""
        ...

    @abc.abstractmethod
    def clear_internal_input_queue(self) -> None:
        """Clear internal input queue(s).

        This should drain all buffered input bundles and update metrics appropriately
        by calling on_input_dequeued().
        """
        ...

    @abc.abstractmethod
    def clear_internal_output_queue(self) -> None:
        """Clear internal output queue(s).

        This should drain all buffered output bundles and update metrics appropriately
        by calling on_output_dequeued().
        """
        ...

    def mark_execution_finished(self) -> None:
        """Mark execution as finished and clear internal queues.

        This default implementation calls the parent's mark_execution_finished()
        and then clears internal input and output queues.
        """
        super().mark_execution_finished()
        self.clear_internal_input_queue()
        self.clear_internal_output_queue()


class OneToOneOperator(PhysicalOperator):
    """An operator that has one input and one output dependency.

    This operator serves as the base for map, filter, limit, etc.
    """

    def __init__(
        self,
        name: str,
        input_op: PhysicalOperator,
        data_context: DataContext,
        target_max_block_size_override: Optional[int] = None,
    ):
        """Create a OneToOneOperator.
        Args:
            input_op: Operator generating input data for this op.
            name: The name of this operator.
            target_max_block_size_override: The target maximum number of bytes to
                include in an output block.
        """
        super().__init__(name, [input_op], data_context, target_max_block_size_override)

    @property
    def input_dependency(self) -> PhysicalOperator:
        return self.input_dependencies[0]


class AllToAllOperator(
    InternalQueueOperatorMixin, SubProgressBarMixin, PhysicalOperator
):
    """A blocking operator that executes once its inputs are complete.

    This operator implements distributed sort / shuffle operations, etc.
    """

    def __init__(
        self,
        bulk_fn: AllToAllTransformFn,
        input_op: PhysicalOperator,
        data_context: DataContext,
        target_max_block_size_override: Optional[int] = None,
        num_outputs: Optional[int] = None,
        sub_progress_bar_names: Optional[List[str]] = None,
        name: str = "AllToAll",
    ):
        """Create an AllToAllOperator.
        Args:
            bulk_fn: The blocking transformation function to run. The inputs are the
                list of input ref bundles, and the outputs are the output ref bundles
                and a stats dict.
            input_op: Operator generating input data for this op.
            data_context: The DataContext instance containing configuration settings.
            target_max_block_size_override: The target maximum number of bytes to
                include in an output block.
            num_outputs: The number of expected output bundles for progress bar.
            sub_progress_bar_names: The names of internal sub progress bars.
            name: The name of this operator.
        """
        self._bulk_fn = bulk_fn
        self._next_task_index = 0
        self._num_outputs = num_outputs
        self._output_rows = 0
        self._sub_progress_bar_names = sub_progress_bar_names
        self._sub_progress_bar_dict = None
        self._input_buffer: FIFOBundleQueue = FIFOBundleQueue()
        self._output_buffer: FIFOBundleQueue = FIFOBundleQueue()
        self._stats: StatsDict = {}
        self._execution_telemetry: Dict[str, Any] = {
            "operator_name": name,
            "first_input_received_at_ns": None,
            "last_input_received_at_ns": None,
            "input_bundles": 0,
            "input_blocks": 0,
            "input_metadata_bytes": 0,
            "current_retained_blocks": 0,
            "current_retained_bytes": 0,
            "peak_retained_blocks": 0,
            "peak_retained_bytes": 0,
            "retained_blocks_at_eos": None,
            "retained_bytes_at_eos": None,
            "eos_received_at_ns": None,
            "bulk_started_at_ns": None,
            "bulk_completed_at_ns": None,
            "bulk_failed_at_ns": None,
            "bulk_failure_type": None,
            "bulk_failure_message": None,
            "all_to_all_completed_at_ns": None,
            "all_to_all_failed_at_ns": None,
            "status": "collecting_inputs",
            # These fields are populated by the push-based shuffle scheduler when
            # this all-to-all uses it. They stay null/zero for other bulk functions.
            "push_shuffle_started_at_ns": None,
            "first_push_map_task_submitted_at_ns": None,
            "last_push_map_task_submitted_at_ns": None,
            "first_push_map_task_index": None,
            "push_map_tasks_submitted": 0,
            "push_shuffle_completed_at_ns": None,
            "push_shuffle_failed_at_ns": None,
            "push_shuffle_failure_type": None,
            "push_shuffle_failure_message": None,
            "first_push_map_task_preceded_eos": None,
        }
        super().__init__(name, [input_op], data_context, target_max_block_size_override)

    def num_outputs_total(self) -> Optional[int]:
        return (
            self._num_outputs
            if self._num_outputs
            else self.input_dependencies[0].num_outputs_total()
        )

    def num_output_rows_total(self) -> Optional[int]:
        return (
            self._output_rows
            if self._output_rows
            else self.input_dependencies[0].num_output_rows_total()
        )

    def _add_input_inner(self, refs: RefBundle, input_index: int) -> None:
        assert not self.has_completed()
        assert input_index == 0, input_index
        received_at_ns = time.time_ns()
        self._input_buffer.add(refs)
        self._metrics.on_input_queued(refs, input_index=0)

        telemetry = self._execution_telemetry
        if telemetry["first_input_received_at_ns"] is None:
            telemetry["first_input_received_at_ns"] = received_at_ns
        telemetry["last_input_received_at_ns"] = received_at_ns
        telemetry["input_bundles"] += 1
        telemetry["input_blocks"] += len(refs.blocks)
        telemetry["input_metadata_bytes"] += sum(
            metadata.size_bytes for _, metadata in refs.blocks
        )
        retained_blocks = self._input_buffer.num_blocks()
        retained_bytes = self._input_buffer.estimate_size_bytes()
        telemetry["current_retained_blocks"] = retained_blocks
        telemetry["current_retained_bytes"] = retained_bytes
        telemetry["peak_retained_blocks"] = max(
            telemetry["peak_retained_blocks"], retained_blocks
        )
        telemetry["peak_retained_bytes"] = max(
            telemetry["peak_retained_bytes"], retained_bytes
        )
        _publish_last_all_to_all_stats(telemetry)

    def internal_input_queue_num_blocks(self) -> int:
        return self._input_buffer.num_blocks()

    def internal_input_queue_num_bytes(self) -> int:
        return self._input_buffer.estimate_size_bytes()

    def internal_output_queue_num_blocks(self) -> int:
        return self._output_buffer.num_blocks()

    def internal_output_queue_num_bytes(self) -> int:
        return self._output_buffer.estimate_size_bytes()

    def clear_internal_input_queue(self) -> None:
        """Clear internal input queue."""
        while self._input_buffer.has_next():
            bundle = self._input_buffer.get_next()
            self._metrics.on_input_dequeued(bundle, input_index=0)
        self._execution_telemetry["current_retained_blocks"] = 0
        self._execution_telemetry["current_retained_bytes"] = 0

    def clear_internal_output_queue(self) -> None:
        """Clear internal output queue."""
        while self._output_buffer.has_next():
            bundle = self._output_buffer.get_next()
            self._metrics.on_output_dequeued(bundle)

    def all_inputs_done(self) -> None:
        eos_received_at_ns = time.time_ns()
        telemetry = self._execution_telemetry
        telemetry["eos_received_at_ns"] = eos_received_at_ns
        telemetry["retained_blocks_at_eos"] = self._input_buffer.num_blocks()
        telemetry["retained_bytes_at_eos"] = self._input_buffer.estimate_size_bytes()
        telemetry["status"] = "running_bulk"
        ctx = TaskContext(
            task_idx=self._next_task_index,
            op_name=self.name,
            sub_progress_bar_dict=self._sub_progress_bar_dict,
            target_max_block_size_override=self.target_max_block_size_override,
        )
        ctx.kwargs[_ALL_TO_ALL_TELEMETRY_TASK_CONTEXT_KEY] = telemetry
        telemetry["bulk_started_at_ns"] = time.time_ns()
        _publish_last_all_to_all_stats(telemetry)
        try:
            # NOTE: We don't account object store memory use from intermediate
            # `bulk_fn` outputs (e.g., map outputs for map-reduce).
            output_buffer, self._stats = self._bulk_fn(
                self._input_buffer.to_list(), ctx
            )
            telemetry["bulk_completed_at_ns"] = time.time_ns()
            self._output_buffer = FIFOBundleQueue(output_buffer)

            while self._input_buffer.has_next():
                refs = self._input_buffer.get_next()
                self._metrics.on_input_dequeued(refs, input_index=0)
            telemetry["current_retained_blocks"] = 0
            telemetry["current_retained_bytes"] = 0

            for ref in self._output_buffer:
                self._metrics.on_output_queued(ref)

            self._next_task_index += 1

            super().all_inputs_done()
            telemetry["all_to_all_completed_at_ns"] = time.time_ns()
            telemetry["status"] = "succeeded"
        except BaseException as exc:
            failed_at_ns = time.time_ns()
            if telemetry["bulk_completed_at_ns"] is None:
                telemetry["bulk_failed_at_ns"] = failed_at_ns
            telemetry["all_to_all_failed_at_ns"] = failed_at_ns
            telemetry["bulk_failure_type"] = type(exc).__name__
            telemetry["bulk_failure_message"] = str(exc)
            telemetry["status"] = "failed"
            raise
        finally:
            ctx.kwargs.pop(_ALL_TO_ALL_TELEMETRY_TASK_CONTEXT_KEY, None)
            _publish_last_all_to_all_stats(telemetry)

    def has_next(self) -> bool:
        return len(self._output_buffer) > 0

    def _get_next_inner(self) -> RefBundle:
        bundle = self._output_buffer.get_next()
        self._metrics.on_output_dequeued(bundle)
        self._output_rows += bundle.num_rows()
        return bundle

    def get_stats(self) -> StatsDict:
        return self._stats

    def get_transformation_fn(self) -> AllToAllTransformFn:
        return self._bulk_fn

    def progress_str(self) -> str:
        return f"{self.num_output_rows_total() or 0} rows output"

    def get_sub_progress_bar_names(self) -> Optional[List[str]]:
        return self._sub_progress_bar_names

    def set_sub_progress_bar(self, name: str, pg: "BaseProgressBar"):
        if self._sub_progress_bar_dict is None:
            self._sub_progress_bar_dict = {}
        self._sub_progress_bar_dict[name] = pg

    def supports_fusion(self):
        return True

    def throttling_disabled(self) -> bool:
        # Disable resource allocation and throttling for the operator
        return True


class NAryOperator(PhysicalOperator):
    """An operator that has multiple input dependencies and one output.

    This operator serves as the base for union, zip, etc.
    """

    def __init__(
        self,
        data_context: DataContext,
        *input_ops: LogicalOperator,
    ):
        """Create a OneToOneOperator.
        Args:
            input_op: Operator generating input data for this op.
            name: The name of this operator.
        """
        input_names = ", ".join([op._name for op in input_ops])
        op_name = f"{self.__class__.__name__}({input_names})"
        super().__init__(
            op_name,
            list(input_ops),
            data_context,
        )
