from __future__ import annotations

import logging
import math
import sys
from typing import TYPE_CHECKING, Optional

from .autoscaling_actor_pool import ActorPoolScalingRequest, AutoscalingActorPool
from .base_actor_autoscaler import ActorAutoscaler
from ray.data._internal.execution.interfaces.execution_options import ExecutionResources
from ray.data.context import WARN_PREFIX, AutoscalingConfig

if TYPE_CHECKING:
    from ray.data._internal.execution.interfaces import PhysicalOperator
    from ray.data._internal.execution.resource_manager import (
        GPUActorAdmissionState,
        ResourceManager,
    )
    from ray.data._internal.execution.streaming_executor_state import OpState, Topology

logger = logging.getLogger(__name__)


class DefaultActorAutoscaler(ActorAutoscaler):
    def __init__(
        self,
        topology: "Topology",
        resource_manager: "ResourceManager",
        *,
        config: AutoscalingConfig,
    ):
        super().__init__(topology, resource_manager)

        self._actor_pool_scaling_up_threshold: float = (
            config.actor_pool_util_upscaling_threshold
        )
        self._actor_pool_scaling_down_threshold: float = (
            config.actor_pool_util_downscaling_threshold
        )
        self._actor_pool_max_upscaling_delta: Optional[
            int
        ] = config.actor_pool_max_upscaling_delta

        self._validate_autoscaling_config()

    def try_trigger_scaling(self):
        for op, state in self._topology.items():
            actor_pools = op.get_autoscaling_actor_pools()
            for actor_pool in actor_pools:
                # Trigger auto-scaling
                actor_pool.scale(
                    self._derive_target_scaling_config(actor_pool, op, state)
                )

    def _derive_target_scaling_config(
        self,
        actor_pool: AutoscalingActorPool,
        op: "PhysicalOperator",
        op_state: "OpState",
    ) -> ActorPoolScalingRequest:
        managed = getattr(op, "uses_gpu_actor_admission_control", lambda: False)()
        admission_state = (
            self._resource_manager.get_gpu_actor_admission_state(op)
            if managed
            else None
        )

        # If all inputs have been consumed, short-circuit. A managed actor op can
        # still hold its final rebundled input internally after the executor's
        # external queue is empty; treating that as complete would remove its
        # actors and prevent a later readmission from ever processing the bundle.
        execution_finished = (
            op.has_execution_finished() if managed else op.has_completed()
        )
        external_inputs_consumed = (
            op._inputs_complete and op_state.total_enqueued_input_blocks() == 0
        )
        internal_inputs_consumed = (
            not managed or op.internal_input_queue_num_blocks() == 0
        )
        if execution_finished or (
            external_inputs_consumed and internal_inputs_consumed
        ):
            num_to_scale_down = (
                actor_pool.current_size()
                if managed
                else self._compute_downscale_delta(actor_pool)
            )
            return (
                ActorPoolScalingRequest.downscale(
                    delta=-num_to_scale_down, force=True, reason="consumed all inputs"
                )
                if num_to_scale_down > 0
                else ActorPoolScalingRequest.no_op(
                    reason="consumed all inputs and pool is empty"
                )
            )

        if admission_state is not None:
            return self._derive_admission_controlled_scaling_request(
                actor_pool, op, op_state, admission_state
            )

        if actor_pool.current_size() < actor_pool.min_size():
            # Scale up, if the actor pool is below min size.
            return ActorPoolScalingRequest.upscale(
                delta=actor_pool.min_size() - actor_pool.current_size(),
                reason="pool below min size",
            )
        elif actor_pool.current_size() > actor_pool.max_size():
            return ActorPoolScalingRequest.downscale(
                delta=-(actor_pool.current_size() - actor_pool.max_size()),
                reason="pool exceeding max size",
            )

        allocation = self._resource_manager.get_allocation(op)
        op_usage = self._resource_manager.get_op_usage(op)
        if allocation is not None and op_usage is not None:
            over_budget_scale_down = _get_required_scale_down(
                actor_pool, allocation.subtract(op_usage)
            )
            if over_budget_scale_down > 0:
                max_can_release = actor_pool.current_size() - actor_pool.min_size()
                num_to_scale_down = min(over_budget_scale_down, max_can_release)
                if num_to_scale_down > 0:
                    return ActorPoolScalingRequest.downscale(
                        delta=-num_to_scale_down,
                        reason="actor pool exceeds resource allocation",
                    )
                return ActorPoolScalingRequest.no_op(
                    reason="actor pool exceeds resource allocation "
                    "but cannot scale below min size",
                )

        # To prevent unexpected downscaling from the initial size, short-circuit if
        # the operator hasn't received any inputs.
        if op.metrics.num_inputs_received == 0:
            return ActorPoolScalingRequest.no_op(reason="no inputs received")

        # Determine whether to scale up based on the actor pool utilization.
        util = actor_pool.get_pool_util()

        if util >= self._actor_pool_scaling_up_threshold:
            # Do not scale up if either
            #   - Actor Pool is at max size already
            #   - Op is throttled (ie exceeding allocated resource quota)
            if actor_pool.current_size() >= actor_pool.max_size():
                return ActorPoolScalingRequest.no_op(reason="reached max size")
            if not op_state._scheduling_status.under_resource_limits:
                return ActorPoolScalingRequest.no_op(
                    reason="operator exceeding resource quota"
                )

            budget = self._resource_manager.get_budget(op)
            budget_max_scale_up = (
                _get_max_scale_up(actor_pool, budget) if budget else sys.maxsize
            )

            # Determine maximum available scale up based on
            #   - Maximum available resource budget
            #   - Configured max scale-up delta (or "+inf" if not configured)
            #   - Total # of actors needed to reach `max_size`
            max_scale_up: int = min(
                budget_max_scale_up,
                self._get_actor_pool_max_upscaling_delta(),
                actor_pool.max_size() - actor_pool.current_size(),
            )

            if max_scale_up == 0:
                return ActorPoolScalingRequest.no_op(reason="exceeded resource limits")

            if util == float("inf"):
                return ActorPoolScalingRequest.upscale(
                    delta=1, reason="no running actors, scale up immediately"
                )

            delta = self._compute_upscale_delta(actor_pool, op_state)
            # At least scale up by 1
            delta = max(1, delta)
            # Cap delta
            delta = min(delta, max_scale_up)

            return ActorPoolScalingRequest.upscale(
                delta=delta,
                reason=(
                    f"utilization of {util} >= "
                    f"{self._actor_pool_scaling_up_threshold}"
                ),
            )
        elif util <= self._actor_pool_scaling_down_threshold:
            if actor_pool.num_pending_actors() > 0:
                return ActorPoolScalingRequest.no_op(
                    reason="no downscaling while actors are pending"
                )
            if actor_pool.current_size() <= actor_pool.min_size():
                return ActorPoolScalingRequest.no_op(reason="reached min size")

            max_can_release = actor_pool.current_size() - actor_pool.min_size()
            num_to_scale_down = min(
                self._compute_downscale_delta(actor_pool), max_can_release
            )

            return ActorPoolScalingRequest.downscale(
                delta=-num_to_scale_down,
                reason=(
                    f"utilization of {util} <= "
                    f"{self._actor_pool_scaling_down_threshold}"
                ),
            )
        else:
            return ActorPoolScalingRequest.no_op(
                reason=(
                    f"utilization of {util} w/in limits "
                    f"[{self._actor_pool_scaling_down_threshold}, "
                    f"{self._actor_pool_scaling_up_threshold}]"
                )
            )

    def _derive_admission_controlled_scaling_request(
        self,
        actor_pool: AutoscalingActorPool,
        op: "PhysicalOperator",
        op_state: "OpState",
        admission_state: "GPUActorAdmissionState",
    ) -> ActorPoolScalingRequest:
        from ray.data._internal.execution.resource_manager import (
            GPUActorAdmissionState,
        )

        actor_info = actor_pool.get_actor_info()
        current_size = actor_pool.current_size()

        if admission_state in (
            GPUActorAdmissionState.DORMANT,
            GPUActorAdmissionState.BLOCKED,
        ):
            inactive = actor_info.pending + actor_info.idle
            if inactive > 0:
                return ActorPoolScalingRequest.downscale(
                    delta=-inactive,
                    force=True,
                    reason=f"GPU actor admission state is {admission_state.value}",
                )
            return ActorPoolScalingRequest.no_op(
                reason=f"GPU actor admission state is {admission_state.value}"
            )

        if admission_state is GPUActorAdmissionState.FRONTIER:
            num_running = actor_pool.num_running_actors()
            # Let active work drain, but never retain an idle frontier actor: it
            # holds the scarce resource that an earlier admitted claimant needs.
            # Once no actor is running, retain exactly one pending request so Ray
            # Core can queue it for the eventual handoff.
            excess_inactive = (
                actor_info.pending + actor_info.idle
                if num_running > 0
                else max(actor_info.pending - 1, 0)
            )
            if excess_inactive > 0:
                return ActorPoolScalingRequest.downscale(
                    delta=-excess_inactive,
                    force=True,
                    reason="GPU actor admission frontier has excess actors",
                )
            if num_running == 0 and actor_info.pending == 0:
                return ActorPoolScalingRequest.upscale(
                    delta=1, reason="GPU actor admission frontier request"
                )
            return ActorPoolScalingRequest.no_op(
                reason="GPU actor admission frontier request already exists"
            )

        assert admission_state is GPUActorAdmissionState.ADMITTED

        allocation_target = self._resource_manager.get_allocation_target(op)
        target_actor_count = actor_pool.max_size()
        if allocation_target is not None:
            # The target is reservation + conceptual shared assignment. It is
            # independent of current usage, so an oversized pool cannot make its
            # own allocation grow merely by retaining actors. Admission always
            # protects one actor, and only pending/idle actors are force-released;
            # active excess actors drain naturally.
            target_actor_count = max(
                1,
                min(
                    _get_max_scale_up(actor_pool, allocation_target),
                    actor_pool.max_size(),
                ),
            )
            excess_actors = max(current_size - target_actor_count, 0)
            inactive_actors = actor_info.pending + actor_info.idle
            num_to_scale_down = min(excess_actors, inactive_actors)
            if num_to_scale_down > 0:
                return ActorPoolScalingRequest.downscale(
                    delta=-num_to_scale_down,
                    force=True,
                    reason="GPU actor pool exceeds allocation target",
                )

        budget = self._resource_manager.get_budget(op)
        budget_max_scale_up = (
            _get_max_scale_up(actor_pool, budget) if budget is not None else 0
        )
        max_scale_up = min(
            budget_max_scale_up,
            self._get_actor_pool_max_upscaling_delta(),
            actor_pool.max_size() - current_size,
            max(target_actor_count - current_size, 0),
        )

        if current_size == 0:
            if max_scale_up > 0:
                return ActorPoolScalingRequest.upscale(
                    delta=1, reason="GPU actor pool admitted"
                )
            return ActorPoolScalingRequest.no_op(
                reason="GPU actor pool admitted without actor allocation"
            )

        target_min_size = actor_pool.min_size()
        if op.metrics.num_inputs_received == 0:
            target_min_size = max(target_min_size, actor_pool.initial_size())
        if current_size < target_min_size:
            delta = min(target_min_size - current_size, max_scale_up)
            if delta > 0:
                return ActorPoolScalingRequest.upscale(
                    delta=delta,
                    reason="GPU actor pool below allocation-capped minimum",
                )

        util = actor_pool.get_pool_util()
        if util >= self._actor_pool_scaling_up_threshold:
            if max_scale_up == 0:
                return ActorPoolScalingRequest.no_op(
                    reason="GPU actor pool reached resource allocation"
                )
            delta = (
                1
                if util == float("inf")
                else max(1, self._compute_upscale_delta(actor_pool, op_state))
            )
            return ActorPoolScalingRequest.upscale(
                delta=min(delta, max_scale_up),
                reason=(
                    f"utilization of {util} >= "
                    f"{self._actor_pool_scaling_up_threshold}"
                ),
            )

        if util <= self._actor_pool_scaling_down_threshold:
            if actor_info.pending > 0 or current_size <= actor_pool.min_size():
                return ActorPoolScalingRequest.no_op(
                    reason="GPU actor pool reached configured minimum"
                )
            return ActorPoolScalingRequest.downscale(
                delta=-1,
                reason=(
                    f"utilization of {util} <= "
                    f"{self._actor_pool_scaling_down_threshold}"
                ),
            )

        return ActorPoolScalingRequest.no_op(
            reason=(
                f"utilization of {util} w/in limits "
                f"[{self._actor_pool_scaling_down_threshold}, "
                f"{self._actor_pool_scaling_up_threshold}]"
            )
        )

    def _get_actor_pool_max_upscaling_delta(self) -> int:
        return (
            self._actor_pool_max_upscaling_delta
            if self._actor_pool_max_upscaling_delta is not None
            else sys.maxsize
        )

    def _validate_autoscaling_config(self):
        # Validate that max upscaling delta is positive to prevent override by safeguard
        if (
            self._actor_pool_max_upscaling_delta is not None
            and self._actor_pool_max_upscaling_delta <= 0
        ):
            raise ValueError(
                f"actor_pool_max_upscaling_delta must be positive, "
                f"got {self._actor_pool_max_upscaling_delta}"
            )
        # Validate that upscaling threshold is positive to prevent division by zero
        # and incorrect scaling calculations
        if self._actor_pool_scaling_up_threshold <= 0:
            raise ValueError(
                f"actor_pool_util_upscaling_threshold must be positive, "
                f"got {self._actor_pool_scaling_up_threshold}"
            )

        for op, state in self._topology.items():
            for actor_pool in op.get_autoscaling_actor_pools():
                self._validate_actor_pool_autoscaling_config(actor_pool, op)

    def _validate_actor_pool_autoscaling_config(
        self,
        actor_pool: AutoscalingActorPool,
        op: "PhysicalOperator",
    ) -> None:
        """Validate autoscaling configuration.

        Args:
            actor_pool: Actor pool to validate configuration thereof.
            op: ``PhysicalOperator`` using target actor pool.
        """
        # Fixed-size pools don't autoscale by design
        if actor_pool.min_size() == actor_pool.max_size():
            return

        max_tasks_in_flight_per_actor = actor_pool.max_tasks_in_flight_per_actor()
        max_concurrency = actor_pool.max_actor_concurrency()

        if (
            max_tasks_in_flight_per_actor / max_concurrency
            < self._actor_pool_scaling_up_threshold
        ):
            logger.warning(
                f"{WARN_PREFIX} Actor Pool configuration of the {op} will not allow it to scale up: "
                f"configured utilization threshold ({self._actor_pool_scaling_up_threshold * 100}%) "
                f"couldn't be reached with configured max_concurrency={max_concurrency} "
                f"and max_tasks_in_flight_per_actor={max_tasks_in_flight_per_actor} "
                f"(max utilization will be max_tasks_in_flight_per_actor / max_concurrency = {(max_tasks_in_flight_per_actor / max_concurrency) * 100:g}%)"
            )

    def _compute_upscale_delta(
        self, actor_pool: AutoscalingActorPool, op_state: OpState
    ) -> int:
        # Calculate desired delta based on utilization
        return math.ceil(
            actor_pool.current_size()
            * (actor_pool.get_pool_util() / self._actor_pool_scaling_up_threshold - 1)
        )

    def _compute_downscale_delta(self, actor_pool: "AutoscalingActorPool") -> int:
        return 1


def _estimate_total_available_task_slots(actor_pool: "AutoscalingActorPool") -> int:
    # Estimates number of available task slots to schedule new tasks
    #
    # NOTE: This must include pending actors for estimation to make sure
    #       autoscaler appropriately accounts task slots that will be available
    #       once pending actors become running.
    return (
        actor_pool.max_tasks_in_flight_per_actor() * actor_pool.current_size()
        - actor_pool.num_tasks_in_flight()
    )


def _get_max_scale_up(
    actor_pool: AutoscalingActorPool,
    budget: ExecutionResources,
) -> int:
    """Get the maximum number of actors that can be scaled up.

    Args:
        actor_pool: The actor pool to scale up.
        budget: The budget to scale up.

    Returns:
        The maximum number of actors that can be scaled up, or `None` if you can
        scale up infinitely.
    """
    assert budget.cpu >= 0 and budget.gpu >= 0 and budget.memory >= 0

    per_actor = actor_pool.per_actor_resource_usage()
    assert per_actor.cpu >= 0 and per_actor.gpu >= 0 and per_actor.memory >= 0

    # floordiv handles per_actor.x == 0 → inf (no constraint from that resource)
    # and budget.x == inf → inf. We ignore object_store_memory since it is not
    # a per-actor declared resource.
    divisions = budget.floordiv(per_actor)
    max_scale_up = min(divisions.cpu, divisions.gpu, divisions.memory)
    if math.isinf(max_scale_up):
        return sys.maxsize
    return int(max_scale_up)


def _get_required_scale_down(
    actor_pool: AutoscalingActorPool,
    budget: ExecutionResources,
) -> int:
    """Get the number of actors that must be removed to fit within budget.

    Args:
        actor_pool: The actor pool to scale down.
        budget: The net remaining budget (allocation - usage). Can be negative
            if the operator is over its allocation.

    Returns:
        The number of actors that need to be removed, or 0 if the pool
        is within budget.
    """
    per_actor = actor_pool.per_actor_resource_usage()

    required_cpu_scale_down = 0
    if per_actor.cpu > 0 and budget.cpu < 0:
        required_cpu_scale_down = math.ceil(abs(budget.cpu) / per_actor.cpu)

    required_gpu_scale_down = 0
    if per_actor.gpu > 0 and budget.gpu < 0:
        required_gpu_scale_down = math.ceil(abs(budget.gpu) / per_actor.gpu)

    required_memory_scale_down = 0
    if per_actor.memory > 0 and budget.memory < 0:
        required_memory_scale_down = math.ceil(abs(budget.memory) / per_actor.memory)

    return max(
        required_cpu_scale_down, required_gpu_scale_down, required_memory_scale_down
    )
