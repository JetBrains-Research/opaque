"""Composition operators for combining DP processes."""

try:
    import opaque_accounting as _native
except ImportError as e:
    raise ImportError(
        "opaque-accounting native module not found. "
        "Install with: maturin develop -m crates/dp-accounting/Cargo.toml"
    ) from e

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

        step = acc.poisson(1.1, 0.01)
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
        phase1 = acc.poisson(0.9, 0.01) * 500
        phase2 = acc.poisson(0.7, 0.01) * 500
        total = acc.compose(phase1, phase2)  # same as: phase1 | phase2
        eps = total.epsilon_at(1e-5)
    """
    return _native.compose(left, right)
