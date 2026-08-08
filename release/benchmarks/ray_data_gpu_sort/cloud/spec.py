"""Frozen topology and trial matrix for the cloud replication."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .common import digest

NODES = 16
CPUS = 256
GPU_INSTANCE = "g6.4xlarge"
CPU_INSTANCE = "m5dn.4xlarge"
GPU_REPETITIONS = 2
CPU_REPETITIONS = 1
SCALES = ((1, 1), (2, 1), (245, 100))
CELLS = (
    "narrow",
    "core",
    "full",
    "origin-string",
    "origin-integer",
    "route",
)


@dataclass(frozen=True)
class Trial:
    name: str
    arm: str
    kind: str
    backend: str
    cell: str
    repetition: int
    scale_numerator: int = 1
    scale_denominator: int = 1
    wave_fraction: float | None = None
    selected_wave_file: str | None = None
    allow_failure: bool = False

    def __post_init__(self) -> None:
        if self.arm not in ("gpu", "cpu"):
            raise ValueError(f"invalid arm: {self.arm}")
        if self.backend not in ("gpu", "pyarrow", "smoke"):
            raise ValueError(f"invalid backend: {self.backend}")
        if self.repetition < 1 or self.scale_numerator < 1 or self.scale_denominator < 1:
            raise ValueError("trial repetition and scale must be positive")
        if self.wave_fraction is not None and not 0 < self.wave_fraction <= 1:
            raise ValueError("wave fraction must be in (0, 1]")

    @property
    def identity(self) -> str:
        return digest(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "identity": self.identity}


def gpu_trials() -> tuple[Trial, ...]:
    result = [
        Trial("transport-smoke", "gpu", "smoke", "smoke", "full", 1),
        Trial(
            "tune-2x-wave-0500",
            "gpu",
            "natural",
            "gpu",
            "full",
            1,
            2,
            1,
            0.50,
            allow_failure=True,
        ),
        Trial(
            "tune-2x-wave-0375",
            "gpu",
            "natural",
            "gpu",
            "full",
            1,
            2,
            1,
            0.375,
            allow_failure=True,
        ),
        Trial(
            "natural-2x-selected-r2",
            "gpu",
            "natural",
            "gpu",
            "full",
            2,
            2,
            1,
            selected_wave_file="configs/selected-wave.json",
            allow_failure=True,
        ),
    ]
    for cell in CELLS:
        for repetition in range(1, GPU_REPETITIONS + 1):
            result.append(
                Trial(
                    f"trend-{cell}-r{repetition}",
                    "gpu",
                    "trend",
                    "gpu",
                    cell,
                    repetition,
                    selected_wave_file="configs/selected-wave.json",
                    allow_failure=True,
                )
            )
    for repetition in range(1, GPU_REPETITIONS + 1):
        result.append(
            Trial(
                f"natural-245x-r{repetition}",
                "gpu",
                "natural",
                "gpu",
                "full",
                repetition,
                245,
                100,
                selected_wave_file="configs/selected-wave.json",
                allow_failure=True,
            )
        )
    return tuple(result)


def cpu_trials() -> tuple[Trial, ...]:
    result = [
        Trial(f"trend-{cell}-r1", "cpu", "trend", "pyarrow", cell, 1, allow_failure=True)
        for cell in CELLS
    ]
    for numerator, denominator in ((2, 1), (245, 100)):
        label = "2x" if (numerator, denominator) == (2, 1) else "245x"
        result.append(
            Trial(
                f"natural-{label}-r1",
                "cpu",
                "natural",
                "pyarrow",
                "full",
                1,
                numerator,
                denominator,
                allow_failure=True,
            )
        )
    return tuple(result)


def trials(arm: str) -> tuple[Trial, ...]:
    return gpu_trials() if arm == "gpu" else cpu_trials()


def validate_unique(values: Iterable[Trial]) -> None:
    names = [item.name for item in values]
    if len(names) != len(set(names)):
        raise ValueError("trial names must be unique")


validate_unique(gpu_trials())
validate_unique(cpu_trials())
