"""Type stubs for the ``opaque_accounting`` Rust extension module.

``opaque_accounting`` is a PyO3-based native extension providing
Privacy Loss Distribution (PLD) accounting for differential privacy.

All mechanism constructors return typed :class:`DpProcess` subclasses.
Processes compose with ``*`` (repeat) and ``|`` (compose) operators.

Example::

    import opaque_accounting as dp

    step = dp.poisson(dp.gaussian(1.1), 0.01)
    training = step * 1000
    print(training.epsilon_at(1e-5))
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class DpProcess:
    """A differential privacy process that can be queried for privacy guarantees.

    ``DpProcess`` is the base class.  Every mechanism constructor returns a
    typed subclass, and composition operators produce new subclass instances.

    **Constructors** (module-level functions):

    - :func:`gaussian` → :class:`Gaussian`
    - :func:`poisson` → :class:`Poisson`
    - :func:`truncated_poisson` → :class:`TruncatedPoisson`
    - :func:`accumulate` → :class:`Accumulated`
    - :func:`eps_delta` → :class:`EpsDelta`
    - :func:`identity` → :class:`Identity`
    - :func:`adaclip` → :class:`AdaClip`

    **Composition**:

    - ``step * 1000`` or ``dp.repeat(step, 1000)`` → :class:`Repeated`
    - ``a | b`` or ``dp.compose(a, b)`` → :class:`Composed`

    **Privacy metrics** (all derived from the same PLD):

    - :meth:`epsilon_at` — (ε, δ)-DP
    - :meth:`delta_at` — (ε, δ)-DP (inverse direction)
    - :meth:`advantage` — f-DP total-variation advantage
    - :meth:`beta_at` — Type-II error at given Type-I error
    - :meth:`risk_at` — Bayes risk

    **Debugging**:

    - ``print(proc)`` — one-line summary with epsilon
    - :meth:`pld_info` — PLD grid diagnostics with timing
    - :meth:`summary` — formatted multi-line privacy report

    Example::

        import opaque_accounting as dp

        step = dp.poisson(dp.gaussian(1.1), 0.01)
        training = step * 1000
        print(training.epsilon_at(1e-5))    # ~1.52
        print(training.summary())            # full report
    """

    # -- metrics -------------------------------------------------------------

    def epsilon_at(self, delta: float) -> float:
        """Compute the smallest ε such that the mechanism satisfies (ε, δ)-DP.

        Args:
            delta: Failure probability (typically 1e-5 to 1e-7).
                Must be in [0, 1).

        Returns:
            The smallest ε achieving (ε, δ)-DP.

        Example::

            proc = dp.gaussian(1.1)
            eps = proc.epsilon_at(1e-5)
        """
        ...

    def delta_at(self, epsilon: float) -> float:
        """Compute the smallest δ such that the mechanism satisfies (ε, δ)-DP.

        Args:
            epsilon: Privacy budget.  Must be ≥ 0.

        Returns:
            The δ value at the given ε.

        Example::

            proc = dp.gaussian(1.1)
            delta = proc.delta_at(1.0)
        """
        ...

    def advantage(self) -> float:
        """Total-variation advantage: max probability of distinguishing neighbors.

        Returns:
            Advantage in [0, 1].  Lower → more private.

        Example::

            proc = dp.gaussian(1.0)
            adv = proc.advantage()
        """
        ...

    def beta_at(self, alpha: float) -> float:
        """Type-II error (β) at a given Type-I error (α).

        Args:
            alpha: Type-I error rate. Must be in [0, 1].

        Returns:
            Type-II error rate (β) in [0, 1].

        Example::

            proc = dp.gaussian(1.0)
            beta = proc.beta_at(0.05)
        """
        ...

    def risk_at(self, prior: float) -> float:
        """Bayes risk under an optimal adversary with a given prior.

        Args:
            prior: Prior probability. Typically 0.5.

        Returns:
            Bayes risk in [0, 0.5]. Higher → more private.

        Example::

            proc = dp.gaussian(1.0)
            risk = proc.risk_at(0.5)
        """
        ...

    # -- operators -----------------------------------------------------------

    def __mul__(self, count: int) -> Repeated:
        """``process * k`` is shorthand for ``repeat(process, k)``.

        Args:
            count: Number of repetitions.

        Returns:
            A :class:`Repeated` process.
        """
        ...

    def __rmul__(self, count: int) -> Repeated:
        """``k * process`` also works (reflected multiply).

        Args:
            count: Number of repetitions.

        Returns:
            A :class:`Repeated` process.
        """
        ...

    def __or__(self, other: DpProcess) -> Composed:
        """``a | b`` is shorthand for ``compose(a, b)``.

        Args:
            other: Second process.

        Returns:
            A :class:`Composed` process.
        """
        ...

    # -- introspection / debugging -------------------------------------------

    def pld_info(self) -> dict[str, Any]:
        """Compute the PLD and return diagnostic info about the internal grid.

        Returns:
            Dict with keys: ``grid_size``, ``discretization``,
            ``lower_index``, ``upper_index``, ``infinity_mass``,
            ``neg_infinity_mass``, ``pessimistic``, ``total_mass``,
            ``is_symmetric``, ``elapsed_ms``.
        """
        ...

    def summary(
        self,
        delta: float = 1e-5,
        epsilon: float = 1.0,
        alpha: float = 0.05,
        prior: float = 0.5,
    ) -> str:
        """Print a human-readable privacy summary.

        Args:
            delta: δ for ε computation.  Default: 1e-5.
            epsilon: ε for δ computation.  Default: 1.0.
            alpha: Type-I error for β computation.  Default: 0.05.
            prior: Prior for risk computation.  Default: 0.5.

        Returns:
            Formatted multi-line summary.
        """
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...


# ---------------------------------------------------------------------------
# Typed subclasses — mechanisms
# ---------------------------------------------------------------------------


class Gaussian(DpProcess):
    """Gaussian mechanism (typed :class:`DpProcess` subclass).

    Created by :func:`gaussian`.  Exposes the noise multiplier as a property.

    Example::

        g = dp.gaussian(1.1)
        g.noise_multiplier  # 1.1
        isinstance(g, dp.DpProcess)  # True
    """

    @property
    def noise_multiplier(self) -> float:
        """Ratio of noise std to sensitivity (σ / Δ)."""
        ...

    def __repr__(self) -> str: ...


class EpsDelta(DpProcess):
    """Fixed (ε, δ)-DP mechanism (typed :class:`DpProcess` subclass).

    Created by :func:`eps_delta`.

    Example::

        ed = dp.eps_delta(1.0, 1e-5)
        ed.epsilon  # 1.0
        ed.delta    # 1e-5
    """

    @property
    def epsilon(self) -> float:
        """Privacy parameter ε."""
        ...

    @property
    def delta(self) -> float:
        """Failure probability δ."""
        ...

    def __repr__(self) -> str: ...


class Identity(DpProcess):
    """Identity mechanism with zero privacy loss (typed :class:`DpProcess` subclass).

    Created by :func:`identity`.
    """

    def __repr__(self) -> str: ...


# ---------------------------------------------------------------------------
# Typed subclasses — amplification
# ---------------------------------------------------------------------------


class Poisson(DpProcess):
    """Poisson-subsampled mechanism (typed :class:`DpProcess` subclass).

    Created by :func:`poisson`.

    Example::

        p = dp.poisson(dp.gaussian(1.1), 0.01)
        p.inner        # Gaussian(noise_multiplier=1.1)
        p.sample_rate  # 0.01
    """

    @property
    def inner(self) -> DpProcess:
        """The inner (base) mechanism."""
        ...

    @property
    def sample_rate(self) -> float:
        """Poisson sampling probability."""
        ...

    def __repr__(self) -> str: ...


class TruncatedPoisson(DpProcess):
    """Truncated-Poisson-subsampled mechanism (typed :class:`DpProcess` subclass).

    Created by :func:`truncated_poisson`.

    Example::

        tp = dp.truncated_poisson(dp.gaussian(1.1), 0.01, 100, 10000)
        tp.inner           # Gaussian(noise_multiplier=1.1)
        tp.sample_rate     # 0.01
        tp.batch_size_cap  # 100
        tp.dataset_size    # 10000
    """

    @property
    def inner(self) -> DpProcess:
        """The inner (base) mechanism."""
        ...

    @property
    def sample_rate(self) -> float:
        """Poisson sampling probability."""
        ...

    @property
    def batch_size_cap(self) -> int:
        """Maximum batch size."""
        ...

    @property
    def dataset_size(self) -> int:
        """Total dataset size."""
        ...

    def __repr__(self) -> str: ...


class Accumulated(DpProcess):
    """Gradient-accumulated mechanism (typed :class:`DpProcess` subclass).

    Created by :func:`accumulate`.

    Example::

        a = dp.accumulate(dp.poisson(dp.gaussian(1.1), 0.01), 4)
        a.inner         # Poisson(...)
        a.microbatches  # 4
    """

    @property
    def inner(self) -> DpProcess:
        """The inner Poisson-subsampled mechanism."""
        ...

    @property
    def microbatches(self) -> int:
        """Number of micro-batches per step."""
        ...

    def __repr__(self) -> str: ...


# ---------------------------------------------------------------------------
# Typed subclasses — transforms
# ---------------------------------------------------------------------------


class AdaClip(DpProcess):
    """Adaptive clipping mechanism (typed :class:`DpProcess` subclass).

    Created by :func:`adaclip`.

    Example::

        ac = dp.adaclip(dp.gaussian(1.1), 50.0)
        ac.inner               # Gaussian(noise_multiplier=1.1)
        ac.quantile_noise_std  # 50.0
    """

    @property
    def inner(self) -> DpProcess:
        """The inner Gaussian mechanism."""
        ...

    @property
    def quantile_noise_std(self) -> float:
        """Noise std for quantile estimation."""
        ...

    def __repr__(self) -> str: ...


# ---------------------------------------------------------------------------
# Typed subclasses — composition
# ---------------------------------------------------------------------------


class Repeated(DpProcess):
    """Repeated process (typed :class:`DpProcess` subclass).

    Created by ``process * count`` or :func:`repeat`.

    Example::

        r = dp.gaussian(1.1) * 1000
        r.inner  # Gaussian(noise_multiplier=1.1)
        r.count  # 1000
    """

    @property
    def inner(self) -> DpProcess:
        """The repeated process."""
        ...

    @property
    def count(self) -> int:
        """Number of repetitions."""
        ...

    def __repr__(self) -> str: ...


class Composed(DpProcess):
    """Composed process (typed :class:`DpProcess` subclass).

    Created by ``a | b`` or :func:`compose`.

    Example::

        c = dp.gaussian(1.0) | dp.eps_delta(0.5)
        c.left   # Gaussian(noise_multiplier=1.0)
        c.right  # EpsDelta(epsilon=0.5, delta=0)
    """

    @property
    def left(self) -> DpProcess:
        """First process."""
        ...

    @property
    def right(self) -> DpProcess:
        """Second process."""
        ...

    def __repr__(self) -> str: ...


class Cached(DpProcess):
    """Cached process (typed :class:`DpProcess` subclass).

    Created by :func:`cached`.  PLD is computed once and reused.

    Example::

        c = dp.cached(dp.poisson(dp.gaussian(1.1), 0.01))
        c.inner  # Poisson(...)
    """

    @property
    def inner(self) -> DpProcess:
        """The wrapped process."""
        ...

    def __repr__(self) -> str: ...


# ---------------------------------------------------------------------------
# DiscretizationConfig
# ---------------------------------------------------------------------------


class DiscretizationConfig:
    """Configuration controlling PLD discretization precision.

    Args:
        discretization: Grid spacing for the PLD PMF. Default: 1e-4.
        log_mass_truncation_bound: Tails below exp(bound) are truncated. Default: -50.
        pessimistic_estimate: If True, upper bound on privacy loss. Default: True.
        max_grid_size: Max bins before automatic coarsening. Default: 10,000,000.

    Example::

        cfg = dp.DiscretizationConfig(discretization=1e-3)
        proc = dp.gaussian(1.1, discretization=cfg)
    """

    def __init__(
        self,
        discretization: float = 1e-4,
        log_mass_truncation_bound: float = -50.0,
        pessimistic_estimate: bool = True,
        max_grid_size: int = 10_000_000,
    ) -> None: ...

    @property
    def discretization(self) -> float:
        """Grid spacing for the PLD PMF."""
        ...

    @property
    def log_mass_truncation_bound(self) -> float:
        """Log-probability below which tails are truncated."""
        ...

    @property
    def pessimistic_estimate(self) -> bool:
        """Whether to round probabilities upward (upper bound on loss)."""
        ...

    @property
    def max_grid_size(self) -> int:
        """Maximum number of bins before automatic coarsening."""
        ...

    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...


# ---------------------------------------------------------------------------
# Module-level functions — mechanisms
# ---------------------------------------------------------------------------


def gaussian(
    noise_multiplier: float,
    discretization: DiscretizationConfig | None = None,
) -> Gaussian:
    """Create a Gaussian mechanism with sensitivity 1.

    Args:
        noise_multiplier: Ratio of noise std to sensitivity (σ / Δ).
        discretization: Override default PLD precision.

    Returns:
        A :class:`Gaussian` process.

    Example::

        proc = dp.gaussian(1.1)
        proc.epsilon_at(1e-5)    # ~3.92
        proc.noise_multiplier   # 1.1
    """
    ...


def eps_delta(
    epsilon: float,
    delta: float = 0.0,
    discretization: DiscretizationConfig | None = None,
) -> EpsDelta:
    """Create a mechanism with a fixed (ε, δ)-DP guarantee.

    Args:
        epsilon: Privacy parameter (must be ≥ 0).
        delta: Failure probability (default 0).
        discretization: Override default PLD precision.

    Returns:
        An :class:`EpsDelta` process.

    Example::

        proc = dp.eps_delta(1.0)
        combined = dp.gaussian(1.1) | dp.eps_delta(0.5)
    """
    ...


def identity(
    discretization: DiscretizationConfig | None = None,
) -> Identity:
    """Create an identity mechanism with zero privacy loss.

    Args:
        discretization: Override default PLD precision.

    Returns:
        An :class:`Identity` process (ε = 0, δ = 0).
    """
    ...


# ---------------------------------------------------------------------------
# Module-level functions — amplification
# ---------------------------------------------------------------------------


def poisson(
    inner: Gaussian | AdaClip,
    sample_rate: float,
) -> Poisson:
    """Poisson-subsampled mechanism (standard DP-SGD step).

    Wraps an inner process with Poisson subsampling for privacy amplification.

    Args:
        inner: The base mechanism (Gaussian or AdaClip).
        sample_rate: Poisson sampling probability q = batch_size / dataset_size.

    Returns:
        A :class:`Poisson` process.

    Example::

        step = dp.poisson(dp.gaussian(1.1), 0.01)
        training = step * 1000
        eps = training.epsilon_at(1e-5)

    See Also:
        :func:`truncated_poisson` for capped batch sizes (tighter bounds).
    """
    ...


def truncated_poisson(
    inner: Gaussian | AdaClip,
    sample_rate: float,
    batch_size_cap: int,
    dataset_size: int,
) -> TruncatedPoisson:
    """Truncated-Poisson-subsampled mechanism (production DP-SGD).

    Like :func:`poisson` but caps the batch at *batch_size_cap*.
    Provides tighter privacy bounds — up to 20% improvement in ε.

    Args:
        inner: The base mechanism (Gaussian or AdaClip).
        sample_rate: Expected sampling rate.
        batch_size_cap: Maximum batch size.
        dataset_size: Total dataset size.

    Returns:
        A :class:`TruncatedPoisson` process.

    Example::

        step = dp.truncated_poisson(dp.gaussian(1.1), 0.01, 100, 10000)
        training = step * 1000
        eps = training.epsilon_at(1e-5)
    """
    ...


def accumulate(
    inner: Poisson,
    microbatches: int,
) -> Accumulated:
    """Gradient-accumulated Poisson-subsampled mechanism.

    Models *microbatches* micro-batches accumulated before a single noise
    addition step.

    Args:
        inner: A Poisson-subsampled process (from :func:`poisson`).
        microbatches: Number of micro-batches per step.

    Returns:
        An :class:`Accumulated` process.

    Example::

        step = dp.accumulate(dp.poisson(dp.gaussian(1.1), 0.01), 4)
        training = step * 500
        eps = training.epsilon_at(1e-5)
    """
    ...


# ---------------------------------------------------------------------------
# Module-level functions — transforms
# ---------------------------------------------------------------------------


def adaclip(
    inner: Gaussian,
    quantile_noise_std: float,
) -> AdaClip:
    """Gaussian mechanism with adaptive clipping (Andrew et al. 2021).

    Args:
        inner: The base Gaussian mechanism.
        quantile_noise_std: Noise std for quantile estimation.

    Returns:
        An :class:`AdaClip` process.

    Example::

        step = dp.adaclip(dp.gaussian(1.1), 50.0)
        eps = step.epsilon_at(1e-5)
    """
    ...


# ---------------------------------------------------------------------------
# Module-level functions — caching
# ---------------------------------------------------------------------------


def cached(process: DpProcess) -> Cached:
    """Wrap a process in a PLD cache for efficient repeated queries.

    Args:
        process: The process to cache.

    Returns:
        A :class:`Cached` process.

    Example::

        step = dp.cached(dp.poisson(dp.gaussian(1.1), 0.01))
        eps = step.epsilon_at(1e-5)   # computes PLD
        adv = step.advantage()         # reuses cached PLD
    """
    ...


# ---------------------------------------------------------------------------
# Module-level functions — composition
# ---------------------------------------------------------------------------


def repeat(process: DpProcess, count: int) -> Repeated:
    """Homogeneous k-fold composition (repeat a process *count* times).

    Equivalent to ``process * count``.

    Args:
        process: The process to repeat.
        count: Number of repetitions.

    Returns:
        A :class:`Repeated` process.
    """
    ...


def compose(left: DpProcess, right: DpProcess) -> Composed:
    """Heterogeneous composition of two processes.

    Equivalent to ``left | right``.

    Args:
        left: First process.
        right: Second process.

    Returns:
        A :class:`Composed` process.
    """
    ...
