"""Free-function ``at_step`` — alias for :meth:`DpFtrlProcess.at_step`.

Exposed for callers who prefer functional form (``at_step(p, K)``) over
method form (``p.at_step(K)``).  Both compute the same result; pick the
form that reads best at the call site.
"""

from __future__ import annotations

from opaque.api.accounting.core._base import DpProcess
from opaque.api.accounting.dpftrl._base import DpFtrlProcess

__all__ = ["at_step"]


def at_step(process: DpFtrlProcess, step: int) -> DpProcess:
    """Privacy-budget process after the first ``step`` of ``process.n_steps`` rounds.

    Wraps :meth:`DpFtrlProcess.at_step`.  See that method for the contract
    (sandwich form, monotonicity, endpoints).

    Args:
        process: A DP-FTRL whole-process accountant.
        step: Number of training rounds to consider (rounded up to the
            process's atomic unit; clamped to ``[0, process.n_steps]``).

    Returns:
        :class:`Identity` for ``step <= 0``; ``process`` itself for
        ``step >= process.n_steps``; otherwise a fresh accountant of
        the same concrete type with reduced ``n_steps``.

    Raises:
        TypeError: If ``process`` is not a :class:`DpFtrlProcess`.
        NotImplementedError: When the concrete amplification cannot
            safely truncate (e.g. ``BallsInBins`` with a correlated-MF
            inner whose Gram is sized for the original horizon).
    """
    if not isinstance(process, DpFtrlProcess):
        raise TypeError(
            f"at_step expects a DpFtrlProcess, got {type(process).__name__}. "
            "DP-SGD-style processes compose externally and have no "
            "intermediate-step accounting."
        )
    return process.at_step(step)
