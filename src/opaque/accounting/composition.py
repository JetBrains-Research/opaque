"""Composition operators for combining DP processes.

These are convenience functions wrapping the PLD composition operators.
Most users should prefer the operator syntax: ``step * 1000`` or ``a | b``.
"""

from __future__ import annotations

from opaque.accounting.base import DpProcess
from opaque.accounting.nodes import CachedProcess


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


def cached(process: DpProcess) -> CachedProcess:
    """Wrap a process so that its PLD is computed once and cached.

    Returns a :class:`~opaque.accounting.nodes.CachedProcess` that
    computes the PLD lazily on the first :meth:`pld` call and caches
    the result for all subsequent calls.

    ``CachedProcess`` also acts as an **opaque merge barrier**: the
    composition optimizer will not look through a cached node, so
    the cached PLD is reused as-is during further composition.

    Example::

        training = acc.cached(acc.poisson(acc.gaussian(1.1), 0.01) * 1000)
        eps = training.epsilon_at(1e-5)   # PLD computed here, cached
        adv = training.advantage()         # reuses cached PLD (free)

    Args:
        process: The process to cache.

    Returns:
        A :class:`~opaque.accounting.nodes.CachedProcess` wrapping *process*.
    """
    if isinstance(process, CachedProcess):
        return process
    return CachedProcess(process)
