"""Opaque-owned symbolic population and cohort values."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Population:
    """A named IFED population used by Opaque's sampler and loader."""

    name: str

    def __post_init__(self) -> None:
        if not self.name.startswith("/"):
            raise ValueError(f"population name must start with '/', got {self.name!r}")


@dataclass(frozen=True)
class Cohort:
    """One symbolic Opaque batch, resolved by the federated gradient oracle."""

    round: int
    size: int
    separation: int = 0
    rounds: int | None = None
    population: Population | None = None
    origin: object | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.round < 0:
            raise ValueError(f"round must be >= 0, got {self.round}")
        if self.size < 1:
            raise ValueError(f"size must be >= 1, got {self.size}")
        if self.separation < 0:
            raise ValueError(f"separation must be >= 0, got {self.separation}")
        if self.rounds is not None and self.rounds < 1:
            raise ValueError(f"rounds must be >= 1, got {self.rounds}")