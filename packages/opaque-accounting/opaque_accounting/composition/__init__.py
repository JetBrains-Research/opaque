"""Composition nodes and operators for combining DP processes.

This module provides:

- **Structural nodes**: Composed, Repeated, CachedProcess
- **Convenience functions**: repeat(), compose(), cached()

Most users should prefer the operator syntax: ``step * 1000`` or ``a | b``.
"""

from __future__ import annotations

from opaque_accounting.base import DpProcess
from opaque_accounting.composition.cached import CachedProcess, cached
from opaque_accounting.composition.composed import Composed
from opaque_accounting.composition.repeated import Repeated


def repeat(process: DpProcess, count: int) -> DpProcess:
    """Homogeneous k-fold composition (repeat a process ``count`` times).

    Equivalent to ``process * count``.

    Args:
        process: The process to repeat.
        count: Number of repetitions.

    Returns:
        Composed process.

    Example::

        step = acc.poisson(acc.gaussian(1.1), 0.01)
        training = acc.repeat(step, 1000)  # same as: step * 1000
        eps = training.epsilon_at(1e-5)
    """
    return process * count


def compose(left: DpProcess, right: DpProcess) -> DpProcess:
    """Heterogeneous composition of two processes.

    Equivalent to ``left | right``.

    Args:
        left: First process.
        right: Second process.

    Returns:
        Composed process.

    Example::

        # Multi-phase training with different noise
        phase1 = acc.poisson(acc.gaussian(0.9), 0.01) * 500
        phase2 = acc.poisson(acc.gaussian(0.7), 0.01) * 500
        total = acc.compose(phase1, phase2)  # same as: phase1 | phase2
        eps = total.epsilon_at(1e-5)
    """
    return left | right


__all__ = [
    # Structural nodes
    "Composed",
    "Repeated",
    "CachedProcess",
    # Convenience functions
    "repeat",
    "compose",
    "cached",
]
