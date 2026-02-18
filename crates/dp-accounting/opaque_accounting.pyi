"""Type stubs for the ``opaque_accounting`` Rust extension module.

``opaque_accounting`` is a PyO3-based native extension providing
Privacy Loss Distribution (PLD) accounting for differential privacy.

All mechanism constructors return :class:`DpProcess` instances.
Processes compose with ``*`` (repeat) and ``|`` (compose) operators.

Example::

    import opaque_accounting as dp

    step = dp.poisson(1.1, 0.01)
    training = step * 1000
    print(training.epsilon_at(1e-5))
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------


class DpProcess:
    """A differential privacy process that can be queried for privacy guarantees.

    ``DpProcess`` is the central class in ``opaque_accounting``.  Every
    mechanism constructor (``gaussian``, ``poisson``, ``adaclip``, …) returns
    a ``DpProcess``, and composition operators produce new instances.

    **Constructors** (module-level functions):

    - :func:`gaussian` — Gaussian mechanism
    - :func:`poisson` — Poisson-subsampled Gaussian
    - :func:`truncated_poisson` — production DP-SGD with capped batch size
    - :func:`accumulate` — gradient accumulation (microbatching)
    - :func:`eps_delta` — fixed (ε, δ) guarantee
    - :func:`identity` — zero privacy loss
    - :func:`adaclip` — adaptive clipping (Andrew et al. 2021)

    **Composition**:

    - ``step * 1000`` or ``dp.repeat(step, 1000)`` — homogeneous k-fold
    - ``a | b`` or ``dp.compose(a, b)`` — heterogeneous two-process

    **Privacy metrics** (all derived from the same PLD):

    - :meth:`epsilon_at` — (ε, δ)-DP
    - :meth:`delta_at` — (ε, δ)-DP (inverse direction)
    - :meth:`advantage` — f-DP total-variation advantage
    - :meth:`beta_at` — Type-II error at given Type-I error
    - :meth:`risk_at` — Bayes risk

    **Debugging**:

    - ``print(proc)`` — one-line summary with epsilon
    - :meth:`describe` — constructor parameters as dict
    - :meth:`pld_info` — PLD grid diagnostics with timing
    - :meth:`summary` — formatted multi-line privacy report

    Example::

        import opaque_accounting as dp

        step = dp.poisson(1.1, 0.01)
        training = step * 1000
        print(training.epsilon_at(1e-5))    # ~3.73
        print(training.summary())            # full report
    """

    # -- metrics -------------------------------------------------------------

    def epsilon_at(self, delta: float) -> float:
        """Compute the smallest ε such that the mechanism satisfies (ε, δ)-DP.

        Solves: find min ε s.t.
        ``P[M(D) ∈ S] ≤ exp(ε) · P[M(D') ∈ S] + δ``
        for all neighboring datasets D, D' and all output sets S.

        Args:
            delta: Failure probability (typically 1e-5 to 1e-7).
                Must be in [0, 1).  Smaller δ → stricter guarantee.

        Returns:
            The smallest ε achieving (ε, δ)-DP.

        Example::

            proc = dp.gaussian(1.1)
            eps = proc.epsilon_at(1e-5)  # ~3.73
        """
        ...

    def delta_at(self, epsilon: float) -> float:
        """Compute the smallest δ such that the mechanism satisfies (ε, δ)-DP.

        Inverse of :meth:`epsilon_at`: given an ε budget, find δ.

        Args:
            epsilon: Privacy budget.  Must be ≥ 0.
                ε = 0 gives δ = advantage (worst-case distinguishing probability).

        Returns:
            The δ value at the given ε.

        Example::

            proc = dp.gaussian(1.1)
            delta = proc.delta_at(1.0)
        """
        ...

    def advantage(self) -> float:
        """Total-variation advantage: max probability of distinguishing neighbors.

        Equivalent to ``delta_at(0.0)`` — the hockey-stick divergence at ε = 0.
        This is the f-DP advantage metric from Dong et al. (2019).

        Returns:
            Advantage in [0, 1].  Lower → more private.

        Example::

            proc = dp.gaussian(1.0)
            adv = proc.advantage()  # ~0.31
        """
        ...

    def beta_at(self, alpha: float) -> float:
        """Type-II error (β) at a given Type-I error (α).

        In the hypothesis-testing interpretation of DP, an adversary tries to
        distinguish D from D'.  α is the false-positive rate and β is the
        false-negative rate.  Higher β → harder to detect → more private.

        Args:
            alpha: Type-I error rate (false positive).  Must be in [0, 1].

        Returns:
            Type-II error rate (β) in [0, 1].

        Example::

            proc = dp.gaussian(1.0)
            beta = proc.beta_at(0.05)
        """
        ...

    def risk_at(self, prior: float) -> float:
        """Bayes risk under an optimal adversary with a given prior.

        The risk is the minimum expected loss of any decision rule trying to
        distinguish D from D', weighted by the prior probability.
        ``risk = prior · β + (1 − prior) · α`` at the optimal threshold.

        Args:
            prior: Prior probability that the data came from D (vs D').
                Typically 0.5 for a balanced prior.

        Returns:
            Bayes risk in [0, 0.5].  Higher → more private.

        Example::

            proc = dp.gaussian(1.0)
            risk = proc.risk_at(0.5)
        """
        ...

    # -- operators -----------------------------------------------------------

    def __mul__(self, count: int) -> DpProcess:
        """``process * k`` is shorthand for ``repeat(process, k)``.

        Args:
            count: Number of repetitions.

        Returns:
            New DpProcess representing *k*-fold homogeneous composition.
        """
        ...

    def __rmul__(self, count: int) -> DpProcess:
        """``k * process`` also works (reflected multiply).

        Args:
            count: Number of repetitions.

        Returns:
            New DpProcess representing *k*-fold homogeneous composition.
        """
        ...

    def __or__(self, other: DpProcess) -> DpProcess:
        """``a | b`` is shorthand for ``compose(a, b)``.

        Args:
            other: Second process.

        Returns:
            New DpProcess representing heterogeneous composition.
        """
        ...

    # -- introspection / debugging -------------------------------------------

    def describe(self) -> dict[str, Any]:
        """Return constructor parameters as a dict.

        Returns:
            Dict with ``'type'`` key (label string) plus original constructor
            keyword arguments.

        Example::

            >>> dp.poisson(1.1, 0.01).describe()
            {'type': 'Poisson(...)', 'noise_multiplier': 1.1, 'sample_rate': 0.01}
        """
        ...

    def pld_info(self) -> dict[str, Any]:
        """Compute the PLD and return diagnostic info about the internal grid.

        Useful for understanding numerical precision and debugging unexpected
        results.

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


class DiscretizationConfig:
    """Configuration controlling PLD discretization precision.

    The PLD is represented as a discrete probability mass function (PMF) on a
    regular grid.  These parameters control the grid resolution, tail
    truncation, and rounding direction.

    **Defaults are chosen for high accuracy** (discretization=1e-4 gives ~1e-8
    error per composition step).  Coarser grids are faster but less precise;
    finer grids are more precise but use more memory.

    Args:
        discretization: Grid spacing for the PLD PMF.  Default: 1e-4.
            Smaller → more precise, larger grid.  Error scales as O(disc²).
        log_mass_truncation_bound: Tails with probability below exp(bound)
            are truncated.  Default: −50 (matching Google dp_accounting).
        pessimistic_estimate: If ``True`` (default), round probabilities
            to produce an **upper bound** on privacy loss.  Set to ``False``
            for an optimistic (lower-bound) estimate — useful for debugging
            but not safe for privacy guarantees.
        max_grid_size: If the grid exceeds this many bins, the discretization
            is automatically coarsened.  Default: 10,000,000.

    Example::

        cfg = dp.DiscretizationConfig(discretization=1e-3)
        proc = dp.gaussian(1.1, config=cfg)
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
    config: DiscretizationConfig | None = None,
) -> DpProcess:
    """Create a Gaussian mechanism with sensitivity 1.

    The Gaussian mechanism adds N(0, σ²) noise to a function with L2
    sensitivity 1.  This is the building block for DP-SGD: after clipping
    gradients to norm C, the effective noise std is ``noise_multiplier × C``.

    Args:
        noise_multiplier: Ratio of noise std to sensitivity (σ / Δ).
            Typical range for DP-SGD is [0.5, 2.0].  Higher → more private.
        config: Override default PLD precision.

    Returns:
        A process representing a single Gaussian mechanism application.

    Raises:
        ValueError: If *noise_multiplier* is out of the supported range.

    Example::

        proc = dp.gaussian(1.1)
        proc.epsilon_at(1e-5)  # ~3.73
    """
    ...


def eps_delta(
    epsilon: float,
    delta: float = 0.0,
    config: DiscretizationConfig | None = None,
) -> DpProcess:
    """Create a mechanism with a fixed (ε, δ)-DP guarantee.

    Represents a mechanism whose privacy loss is known analytically
    (e.g., randomized response, Laplace mechanism).  The resulting PLD is a
    two-point distribution capturing the worst-case privacy loss.

    Useful for composing non-Gaussian mechanisms with Gaussian ones.

    Args:
        epsilon: Privacy parameter (must be ≥ 0).
        delta: Failure probability (default 0, must be in [0, 1)).
        config: Override default PLD precision.

    Returns:
        A process with the given (ε, δ) guarantee.

    Example::

        # Pure ε-DP mechanism
        proc = dp.eps_delta(1.0)

        # Compose with a Gaussian
        combined = dp.gaussian(1.1) | dp.eps_delta(0.5)
    """
    ...


def identity(
    config: DiscretizationConfig | None = None,
) -> DpProcess:
    """Create an identity mechanism with zero privacy loss.

    The identity process represents a computation that reveals no information
    about the dataset (e.g., returning a constant).  Its PLD is a point mass
    at privacy loss = 0.  Composing with identity has no effect.

    Args:
        config: Override default PLD precision.

    Returns:
        A process with ε = 0, δ = 0 for all queries.
    """
    ...


# ---------------------------------------------------------------------------
# Module-level functions — amplification
# ---------------------------------------------------------------------------


def poisson(
    noise_multiplier: float,
    sample_rate: float,
    config: DiscretizationConfig | None = None,
) -> DpProcess:
    """Poisson-subsampled Gaussian mechanism (standard DP-SGD step).

    Each record is included independently with probability ``sample_rate``,
    providing **privacy amplification by subsampling**.  Standard model for
    DP-SGD: ``sample_rate = batch_size / dataset_size``.

    Args:
        noise_multiplier: Gaussian noise std / sensitivity.
        sample_rate: Poisson sampling probability q = batch_size / dataset_size.
            Must be in (0, 1].
        config: Override default PLD precision.

    Returns:
        A single Poisson-subsampled Gaussian step.

    Example::

        step = dp.poisson(1.1, 0.01)
        training = step * 1000
        eps = training.epsilon_at(1e-5)

    See Also:
        :func:`truncated_poisson` for capped batch sizes (tighter bounds).
    """
    ...


def truncated_poisson(
    noise_multiplier: float,
    sample_rate: float,
    batch_size_cap: int,
    dataset_size: int,
    config: DiscretizationConfig | None = None,
) -> DpProcess:
    """Truncated-Poisson-subsampled Gaussian (production DP-SGD).

    Like :func:`poisson` but caps the batch at *batch_size_cap*.  This models
    what production DP-SGD frameworks actually do: sample a random batch, but
    truncate if it exceeds a maximum.

    Provides **tighter privacy bounds** than standard (worst-case) Poisson —
    up to 20 % improvement in ε for the same noise level.

    Args:
        noise_multiplier: Gaussian noise std / sensitivity.
        sample_rate: Expected sampling rate q = batch_size / dataset_size.
        batch_size_cap: Maximum batch size B_max.
        dataset_size: Total dataset size n.
        config: Override default PLD precision.

    Returns:
        A single truncated-Poisson step.

    Example::

        step = dp.truncated_poisson(1.1, 0.01, batch_size_cap=100, dataset_size=10000)
        training = step * 1000
        eps = training.epsilon_at(1e-5)

    See Also:
        :func:`poisson` for standard (non-truncated) analysis.
    """
    ...


def accumulate(
    noise_multiplier: float,
    sample_rate: float,
    microbatches: int,
    config: DiscretizationConfig | None = None,
) -> DpProcess:
    """Gradient-accumulated Poisson-subsampled Gaussian.

    Models *microbatches* micro-batches accumulated before a single noise
    addition step.  Uses the Mixture-of-Gaussians framework: privacy analysis
    is exact over the possible numbers of records contributed by the
    micro-batches.

    Args:
        noise_multiplier: Gaussian noise std / sensitivity.
        sample_rate: Per-microbatch Poisson sampling rate.
        microbatches: Number of micro-batches accumulated per step.
        config: Override default PLD precision.

    Returns:
        A single accumulated step.

    Example::

        step = dp.accumulate(1.1, sample_rate=0.01, microbatches=4)
        training = step * 500
        eps = training.epsilon_at(1e-5)
    """
    ...


# ---------------------------------------------------------------------------
# Module-level functions — transforms
# ---------------------------------------------------------------------------


def adaclip(
    noise_multiplier: float,
    quantile_noise_std: float,
    config: DiscretizationConfig | None = None,
) -> DpProcess:
    """Gaussian mechanism with adaptive clipping (Andrew et al. 2021).

    Adaptive clipping adjusts the clipping threshold based on the empirical
    distribution of gradient norms.  The total privacy cost is the composition
    of the base Gaussian mechanism and the quantile-estimation mechanism.

    Args:
        noise_multiplier: Gradient noise multiplier for the main mechanism.
        quantile_noise_std: Noise std for the quantile estimation.
            Larger → more private quantile estimation, less accurate clipping.
        config: Override default PLD precision.

    Returns:
        A single AdaClip step.

    Example::

        step = dp.adaclip(1.1, quantile_noise_std=50.0)
        eps = step.epsilon_at(1e-5)
    """
    ...


# ---------------------------------------------------------------------------
# Module-level functions — caching
# ---------------------------------------------------------------------------


def cached(process: DpProcess) -> DpProcess:
    """Wrap a process in a PLD cache for efficient repeated queries.

    The returned process computes its Privacy Loss Distribution on first
    access and caches the result.  Subsequent calls to ``epsilon_at``,
    ``delta_at``, etc. reuse the cached PLD instead of recomputing it.

    Useful in accounting loops where the same step process is composed many
    times — caching avoids redundant PLD computation.

    Note:
        Clones of a cached process share the same cache.

    Args:
        process: The process to cache.

    Returns:
        A new DpProcess that caches its PLD after first computation.

    Example::

        step = dp.cached(dp.poisson(1.1, 0.01))
        eps = step.epsilon_at(1e-5)   # computes PLD
        adv = step.advantage()         # reuses cached PLD
    """
    ...


# ---------------------------------------------------------------------------
# Module-level functions — composition
# ---------------------------------------------------------------------------


def repeat(process: DpProcess, count: int) -> DpProcess:
    """Homogeneous k-fold composition (repeat a process *count* times).

    Equivalent to ``process * count``.

    Args:
        process: The process to repeat.
        count: Number of repetitions.

    Returns:
        Composed DpProcess.
    """
    ...


def compose(left: DpProcess, right: DpProcess) -> DpProcess:
    """Heterogeneous composition of two processes.

    Equivalent to ``left | right``.

    Args:
        left: First process.
        right: Second process.

    Returns:
        Composed DpProcess.
    """
    ...
