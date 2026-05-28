"""``WithRestarts`` composition wrapper + factory."""

from __future__ import annotations

from dataclasses import dataclass

from opaque.api.engine.scheduling._schedule import Schedule

__all__ = ["WithRestarts", "with_restarts"]


@dataclass(frozen=True, slots=True)
class WithRestarts:
    """Repeat ``schedule`` ``num_cycles`` times across
    ``[transition_begin, transition_begin + transition_steps)``.

    Each cycle has length ``transition_steps // num_cycles``; within a
    cycle, ``schedule`` is evaluated at the cycle-local integer step
    (``relative_step % cycle_length``).

    Before ``transition_begin`` returns ``schedule(0)``; after the
    final cycle, returns ``schedule(cycle_length)``.
    """

    schedule: Schedule
    transition_steps: int
    num_cycles: int
    transition_begin: int = 0

    def __call__(self, step: int) -> float:
        cycle_length = self.transition_steps // self.num_cycles
        relative = step - self.transition_begin
        if relative < 0:
            return self.schedule(0)
        if relative >= self.transition_steps:
            return self.schedule(cycle_length)
        return self.schedule(relative % cycle_length)


def with_restarts(
    schedule: Schedule,
    transition_steps: int,
    num_cycles: int,
    transition_begin: int = 0,
) -> WithRestarts:
    """Repeat ``schedule`` ``num_cycles`` times across
    ``[transition_begin, transition_begin + transition_steps)``.
    """
    if num_cycles <= 0:
        raise ValueError(f"with_restarts requires num_cycles > 0; got {num_cycles}.")
    if transition_steps <= 0:
        raise ValueError(
            f"with_restarts requires transition_steps > 0; got {transition_steps}."
        )
    if transition_steps % num_cycles != 0:
        raise ValueError(
            "with_restarts requires num_cycles to evenly divide transition_steps; "
            f"got transition_steps={transition_steps}, num_cycles={num_cycles} "
            f"(remainder={transition_steps % num_cycles})."
        )
    return WithRestarts(
        schedule=schedule,
        transition_steps=int(transition_steps),
        num_cycles=int(num_cycles),
        transition_begin=int(transition_begin),
    )
