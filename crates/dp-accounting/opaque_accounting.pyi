"""Type stubs for the ``opaque_accounting`` Rust extension module.

``opaque_accounting`` provides flat PLD-based privacy accounting.
Rust computes PLDs; Python owns composition and dispatch.

The extension exports two classes (:class:`Pld` and :class:`DiscretizationConfig`)
and seven functions for creating PLDs from mechanism parameters.

Example::

    import opaque_accounting as dp

    pld = dp.gaussian_pld(1.1)
    composed = pld.self_compose(1000)
    print(composed.epsilon_at(1e-5))
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Pld — opaque PLD handle
# ---------------------------------------------------------------------------


class Pld:
    """An opaque Privacy Loss Distribution (PLD).

    Created by mechanism/amplification functions.  Supports privacy metric
    queries and composition.

    Example::

        import opaque_accounting as dp
        pld = dp.gaussian_pld(1.1)
        print(pld.epsilon_at(1e-5))
        composed = pld.self_compose(1000)
        print(composed.epsilon_at(1e-5))
    """

    def epsilon_at(self, delta: float) -> float:
        """Smallest ε achieving (ε, δ)-DP.

        Args:
            delta: Failure probability (typically 1e-5 to 1e-7).

        Returns:
            Epsilon value.
        """
        ...

    def delta_at(self, epsilon: float) -> float:
        """Smallest δ achieving (ε, δ)-DP.

        Args:
            epsilon: Privacy budget. Must be >= 0.

        Returns:
            Delta value.
        """
        ...

    def advantage(self) -> float:
        """Total-variation advantage.

        Returns:
            Advantage in [0, 1]. Lower is more private.
        """
        ...

    def beta_at(self, alpha: float) -> float:
        """Type-II error (β) at a given Type-I error (α).

        Args:
            alpha: Type-I error rate in [0, 1].

        Returns:
            Beta value in [0, 1].
        """
        ...

    def risk_at(self, prior: float) -> float:
        """Bayes risk under an optimal adversary.

        Args:
            prior: Prior probability (typically 0.5).

        Returns:
            Risk value in [0, 0.5].
        """
        ...

    def compose(self, other: Pld) -> Pld:
        """Compose this PLD with another (heterogeneous composition).

        Args:
            other: The other PLD to compose with.

        Returns:
            A new composed PLD.

        Raises:
            ValueError: If composition fails (mismatched discretization).
        """
        ...

    def self_compose(self, count: int) -> Pld:
        """Self-compose this PLD *count* times (homogeneous repetition).

        Args:
            count: Repetition count. Must be > 0.

        Returns:
            A new self-composed PLD.
        """
        ...

    def __mul__(self, count: int) -> Pld:
        """``pld * k`` is shorthand for ``pld.self_compose(k)``."""
        ...

    def __rmul__(self, count: int) -> Pld:
        """``k * pld`` also works."""
        ...

    def __or__(self, other: Pld) -> Pld:
        """``a | b`` is shorthand for ``a.compose(b)``."""
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...


# ---------------------------------------------------------------------------
# DiscretizationConfig — discretization configuration
# ---------------------------------------------------------------------------


class DiscretizationConfig:
    """Discretization configuration for PLD computation.

    Args:
        discretization: Grid spacing for the PLD PMF. Default: 1e-4.
            Smaller values give tighter bounds but use more memory.
        log_mass_truncation_bound: Log tail mass cutoff. Default: -50.0.
            Tails below exp(bound) are truncated.
        pessimistic_estimate: If True (default), round probabilities upward
            for an upper bound on privacy loss (safe for guarantees).
        max_grid_size: Maximum grid bins before automatic coarsening.
            Default: 10,000,000.

    Example::

        config = dp.DiscretizationConfig(discretization=0.001)
        pld = dp.gaussian_pld(1.1, config=config)
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
        """Whether to round upward (upper bound on loss)."""
        ...

    @property
    def max_grid_size(self) -> int:
        """Maximum bins before automatic coarsening."""
        ...

    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...


# ---------------------------------------------------------------------------
# Mechanism functions
# ---------------------------------------------------------------------------


def bounded_gaussian_pld(
    noise_multiplier: float,
    config: DiscretizationConfig | None = None,
) -> Pld:
    """Compute the PLD for the Bounded Gaussian mechanism (Replace adjacency).

    The Bounded Gaussian Mechanism (Chen & Hale, 2024) adds truncated Gaussian
    noise to keep outputs in a bounded domain.  Under Replace adjacency,
    sensitivity is 2Δ, so the PLD equals that of a Gaussian with
    ``effective_σ = noise_multiplier / 2``.

    Args:
        noise_multiplier: Ratio of noise std to sensitivity (σ/Δ), in [0.1, 2.4].
        config: Override default PLD precision.

    Returns:
        The privacy loss distribution.

    Example::

        pld = dp.bounded_gaussian_pld(1.1)
        pld.epsilon_at(1e-5)
    """
    ...


def gaussian_pld(
    noise_multiplier: float,
    config: DiscretizationConfig | None = None,
) -> Pld:
    """Compute the PLD for a Gaussian mechanism with sensitivity 1.

    Args:
        noise_multiplier: Ratio of noise std to sensitivity (σ/Δ).
        config: Override default PLD precision.

    Returns:
        The privacy loss distribution.

    Example::

        pld = dp.gaussian_pld(1.1)
        pld.epsilon_at(1e-5)  # ~3.92
    """
    ...


def eps_delta_pld(
    epsilon: float,
    delta: float,
    config: DiscretizationConfig | None = None,
) -> Pld:
    """Compute the PLD for a fixed (ε, δ)-mechanism.

    Args:
        epsilon: Privacy loss, >= 0.
        delta: Failure probability, in [0, 1].
        config: Override default PLD precision.

    Returns:
        The privacy loss distribution.
    """
    ...


def identity_pld(
    config: DiscretizationConfig | None = None,
) -> Pld:
    """Compute the PLD for the identity (zero-privacy-loss) mechanism.

    Args:
        config: Override default PLD precision.

    Returns:
        The identity PLD (neutral element for composition).
    """
    ...


# ---------------------------------------------------------------------------
# Amplification functions
# ---------------------------------------------------------------------------


def poisson_gaussian_pld(
    noise_multiplier: float,
    rate: float,
    config: DiscretizationConfig | None = None,
) -> Pld:
    """Compute the PLD for a Poisson-subsampled Gaussian mechanism.

    Args:
        noise_multiplier: σ/Δ ratio.
        rate: Poisson sampling probability, in (0, 1].
        config: Override default PLD precision.

    Returns:
        The amplified privacy loss distribution.

    Example::

        pld = dp.poisson_gaussian_pld(1.1, 0.01)
        training = pld.self_compose(1000)
        training.epsilon_at(1e-5)
    """
    ...


def truncated_poisson_gaussian_pld(
    noise_multiplier: float,
    rate: float,
    batch_size_max: int,
    dataset_size: int,
    config: DiscretizationConfig | None = None,
) -> Pld:
    """Compute the PLD for a truncated Poisson-subsampled Gaussian mechanism.

    Production DP-SGD sampling: caps batch at *batch_size_max* for
    predictable memory usage and tighter privacy bounds.

    Args:
        noise_multiplier: σ/Δ ratio.
        rate: Poisson sampling probability, in (0, 1].
        batch_size_max: Maximum batch size.
        dataset_size: Total dataset size.
        config: Override default PLD precision.

    Returns:
        The amplified privacy loss distribution.
    """
    ...


def accumulated_poisson_gaussian_pld(
    noise_multiplier: float,
    rate: float,
    microbatches: int,
    config: DiscretizationConfig | None = None,
) -> Pld:
    """Compute the PLD for an accumulated Poisson-subsampled Gaussian mechanism.

    Models gradient accumulation: *microbatches* micro-batches,
    Poisson-sampled, clipped gradients summed, noise added once.

    Args:
        noise_multiplier: σ/Δ ratio.
        rate: Poisson sampling probability, in (0, 1].
        microbatches: Number of microbatches, > 0.
        config: Override default PLD precision.

    Returns:
        The amplified privacy loss distribution.
    """
    ...


# ---------------------------------------------------------------------------
# AdaClip utility
# ---------------------------------------------------------------------------


def combined_sensitivity(
    noise_multiplier: float,
    quantile_noise_std: float,
) -> float:
    """Combined sensitivity for adaptive clipping (Andrew et al. 2021).

    Returns z̃ = sqrt(1/z² + 1/(4·σ_b²)).

    Args:
        noise_multiplier: Gradient noise multiplier z.
        quantile_noise_std: Std of quantile estimator noise σ_b.

    Returns:
        Combined sensitivity z̃.
    """
    ...
