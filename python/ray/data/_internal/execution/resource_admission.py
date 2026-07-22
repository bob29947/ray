from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ray.data._internal.execution.interfaces.execution_options import ExecutionResources


@dataclass(frozen=True)
class ResourceAdmissionSpec:
    minimum_resources: ExecutionResources
    unit_resources: Optional[ExecutionResources]
    min_units: int
    max_units: Optional[int]


@dataclass(frozen=True)
class ResourceAdmissionGrant:
    max_units: int
    may_submit: bool


def validate_resource_admission_spec(spec: ResourceAdmissionSpec) -> None:
    if spec.minimum_resources.is_zero() or not spec.minimum_resources.is_non_negative():
        raise ValueError("Resource admission minimum resources must be positive")
    if spec.min_units < 1 or (spec.unit_resources is None and spec.min_units != 1):
        raise ValueError("Resource admission min_units must be positive")
    if spec.unit_resources is not None and (
        not spec.unit_resources.is_non_negative()
        or spec.unit_resources.scale(spec.min_units) != spec.minimum_resources
    ):
        raise ValueError("Admission floor must equal unit resources times min_units")
    if spec.max_units is not None and spec.max_units < spec.min_units:
        raise ValueError(
            "Resource admission max_units cannot be smaller than the minimum floor"
        )
