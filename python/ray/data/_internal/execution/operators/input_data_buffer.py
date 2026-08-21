from typing import Callable, Iterator, List, Optional

from ray.data._internal.execution.interfaces import (
    ExecutionOptions,
    PhysicalOperator,
    RefBundle,
)
from ray.data._internal.stats import StatsDict
from ray.data.context import DataContext


class InputDataBuffer(PhysicalOperator):
    """Defines the input data for the operator DAG.

    For example, this may hold cached blocks from a previous Dataset execution, or
    the arguments for read tasks.
    """

    def __init__(
        self,
        data_context: DataContext,
        input_data: Optional[List[RefBundle]] = None,
        input_data_factory: Optional[Callable[[int], List[RefBundle]]] = None,
        *,
        input_data_iterator_factory: Optional[
            Callable[[int], Iterator[RefBundle]]
        ] = None,
        estimated_num_output_bundles: Optional[int] = None,
        max_pending_input_blocks: Optional[int] = None,
    ):
        """Create an InputDataBuffer.

        Args:
            data_context: :class:`~ray.data.context.DataContext`
                object to use injestion.
            input_data: The list of bundles to output from this operator.
            input_data_factory: The factory to get input data, if input_data is None.
            input_data_iterator_factory: A one-shot factory that creates input
                bundles incrementally. Unlike ``input_data_factory``, the iterator
                is not materialized when execution starts.
            estimated_num_output_bundles: The exact number of bundles yielded by
                ``input_data_iterator_factory``.
            max_pending_input_blocks: The maximum number of lazily created input
                blocks waiting in this operator's external output queue.
        """
        super().__init__("Input", [], data_context)
        self._is_lazy_input = input_data_iterator_factory is not None
        self._input_data_iterator: Optional[Iterator[RefBundle]] = None
        self._next_input_data: Optional[RefBundle] = None
        self._input_data_exhausted = False
        self._lazy_total_rows: Optional[int] = 0

        if self._is_lazy_input:
            assert input_data is None and input_data_factory is None
            assert estimated_num_output_bundles is not None
            assert estimated_num_output_bundles >= 0
            assert max_pending_input_blocks is not None
            assert max_pending_input_blocks > 0
            self._input_data = []
            self._input_data_iterator_factory = input_data_iterator_factory
            self._max_pending_input_blocks = max_pending_input_blocks
            self._estimated_num_output_bundles = estimated_num_output_bundles
            self._stats = {"input": []}
            self._is_input_initialized = False
        if input_data is not None:
            assert not self._is_lazy_input
            assert input_data_factory is None
            # Copy the input data to avoid mutating the original list.
            self._input_data = input_data[:]
            self._is_input_initialized = True
            self._initialize_metadata()
        elif not self._is_lazy_input:
            # Initialize input lazily when execution is started.
            assert input_data_factory is not None
            self._input_data_factory = input_data_factory
            self._is_input_initialized = False
        self._input_data_index = 0
        if not self._is_lazy_input:
            self.mark_execution_finished()

    def start(self, options: ExecutionOptions) -> None:
        if self._is_lazy_input:
            assert not self._is_input_initialized
            self._input_data_iterator = iter(
                self._input_data_iterator_factory(
                    self.target_max_block_size_override
                    or self.data_context.target_max_block_size
                )
            )
            self._is_input_initialized = True
        elif not self._is_input_initialized:
            self._input_data = self._input_data_factory(
                self.target_max_block_size_override
                or self.data_context.target_max_block_size
            )
            self._is_input_initialized = True
            self._initialize_metadata()
        # InputDataBuffer does not take inputs from other operators,
        # so we record input metrics here
        if not self._is_lazy_input:
            for bundle in self._input_data:
                self._metrics.on_input_received(bundle)
        super().start(options)

    def has_next(self) -> bool:
        if self._is_lazy_input:
            return self._has_next_lazy()
        return self._input_data_index < len(self._input_data)

    def _get_next_inner(self) -> RefBundle:
        if self._is_lazy_input:
            assert self._next_input_data is not None
            bundle = self._next_input_data
            self._next_input_data = None
            self._input_data_index += 1
            return bundle

        # We can't pop the input data. If we do, Ray might garbage collect the block
        # references, and Ray won't be able to reconstruct downstream objects.
        bundle = self._input_data[self._input_data_index]
        self._input_data_index += 1
        return bundle

    def get_stats(self) -> StatsDict:
        return {}

    def _add_input_inner(self, refs, input_index) -> None:
        raise ValueError("Inputs are not allowed for this operator.")

    def has_execution_finished(self) -> bool:
        if (
            self._is_lazy_input
            and not self._is_execution_marked_finished
            and not self._input_data_exhausted
        ):
            # A temporarily full lazy-output window is backpressure, not EOS.
            return False
        return super().has_execution_finished()

    def _has_next_lazy(self) -> bool:
        assert self._is_input_initialized
        if self._is_execution_marked_finished:
            self._close_input_data_iterator()
            self._next_input_data = None
            return False
        if self._next_input_data is not None:
            return True
        if self._input_data_exhausted:
            return False

        # ``process_completed_tasks()`` drains operator outputs in a loop. Bound
        # that loop by the number of descriptor blocks still waiting for the
        # downstream operator. The metric is decremented when the executor
        # dispatches a bundle, which opens exactly one slot for the next bundle.
        if self._metrics.num_external_outqueue_blocks >= self._max_pending_input_blocks:
            return False

        assert self._input_data_iterator is not None
        try:
            bundle = next(self._input_data_iterator)
        except StopIteration:
            self._input_data_exhausted = True
            self._input_data_iterator = None
            if self._input_data_index != self._estimated_num_output_bundles:
                raise RuntimeError(
                    "Lazy input iterator yielded "
                    f"{self._input_data_index} bundles, expected "
                    f"{self._estimated_num_output_bundles}."
                ) from None
            if self._lazy_total_rows:
                self._estimated_num_output_rows = self._lazy_total_rows
            return False

        if self._input_data_index >= self._estimated_num_output_bundles:
            self._close_input_data_iterator()
            raise RuntimeError(
                "Lazy input iterator yielded more than "
                f"{self._estimated_num_output_bundles} bundles."
            )
        if not bundle.blocks:
            self._close_input_data_iterator()
            raise RuntimeError("Lazy input iterator yielded an empty bundle.")

        # Lazy inputs are intentionally one-shot: don't retain every root bundle
        # in the driver. The external output queue and the submitted downstream
        # task own the live Python references while the descriptor is needed.
        self._stats["input"].extend(bundle.metadata)
        bundle_num_rows = bundle.num_rows()
        if self._lazy_total_rows is not None and bundle_num_rows is not None:
            self._lazy_total_rows += bundle_num_rows
        else:
            self._lazy_total_rows = None
        self._metrics.on_input_received(bundle)
        self._next_input_data = bundle
        return True

    def _close_input_data_iterator(self) -> None:
        iterator = self._input_data_iterator
        self._input_data_iterator = None
        if iterator is not None:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()

    def _do_shutdown(self, force: bool) -> None:
        self._close_input_data_iterator()
        self._next_input_data = None
        self._input_data_exhausted = True
        self.mark_execution_finished()
        super()._do_shutdown(force)

    def _initialize_metadata(self):
        assert self._input_data is not None and self._is_input_initialized
        self._estimated_num_output_bundles = len(self._input_data)

        block_metadata = []
        total_rows = 0
        for bundle in self._input_data:
            block_metadata.extend(bundle.metadata)
            bundle_num_rows = bundle.num_rows()
            if total_rows is not None and bundle_num_rows is not None:
                total_rows += bundle_num_rows
            else:
                # total row is unknown
                total_rows = None
        if total_rows:
            self._estimated_num_output_rows = total_rows
        self._stats = {
            "input": block_metadata,
        }
