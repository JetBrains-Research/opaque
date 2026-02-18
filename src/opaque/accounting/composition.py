"""Composition operators for combining DP processes."""

import opaque_accounting as _native

DpProcess = _native.DpProcess


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
    return _native.repeat(process, count)


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
    return _native.compose(left, right)


def cached(process: DpProcess) -> DpProcess:
    """Wrap a process in a PLD cache for efficient repeated queries.

    The returned process computes its Privacy Loss Distribution on first
    access and caches the result. Subsequent calls to ``epsilon_at()``,
    ``delta_at()``, etc. reuse the cached PLD instead of recomputing it.

    Useful in accounting loops where the same step process is composed many
    times — caching avoids redundant PLD computation.

    Note:
        Clones of a cached process share the same cache.

    Args:
        process: The process to cache.

    Returns:
        A new process that caches its PLD after first computation.

    Example::

        step = acc.cached(acc.poisson(acc.gaussian(1.1), 0.01))
        eps = step.epsilon_at(1e-5)   # computes PLD
        adv = step.advantage()         # reuses cached PLD
    """
    return _native.cached(process)
