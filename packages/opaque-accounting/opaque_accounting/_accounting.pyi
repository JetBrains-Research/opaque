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

def gaussian_pld(
    noise_multiplier: float,
    config: DiscretizationConfig,
) -> Pld:
    """Compute the PLD for a Gaussian mechanism with sensitivity 1.

    Args:
        noise_multiplier: Ratio of noise std to sensitivity (σ/Δ).
        config: PLD discretization configuration.

    Returns:
        The privacy loss distribution.

    Example::

        pld = dp.gaussian_pld(1.1, config)
        pld.epsilon_at(1e-5)  # ~3.92
    """
    ...

def eps_delta_pld(
    epsilon: float,
    delta: float,
    config: DiscretizationConfig,
) -> Pld:
    """Compute the PLD for a fixed (ε, δ)-mechanism.

    Args:
        epsilon: Privacy loss, >= 0.
        delta: Failure probability, in [0, 1].
        config: PLD discretization configuration.

    Returns:
        The privacy loss distribution.
    """
    ...

def identity_pld(
    config: DiscretizationConfig,
) -> Pld:
    """Compute the PLD for the identity (zero-privacy-loss) mechanism.

    Args:
        config: PLD discretization configuration.

    Returns:
        The identity PLD (neutral element for composition).
    """
    ...

def rectified_gaussian_pld(
    noise_multiplier: float,
    radius: float,
    config: DiscretizationConfig,
) -> Pld:
    """Compute the PLD for a rectified (clamped) Gaussian mechanism.

    Noise is sampled from a standard Gaussian and clamped to [−R·σ, R·σ].

    Args:
        noise_multiplier: Ratio of noise std to sensitivity (σ/Δ).
        radius: Support half-width in sigma units.
        config: PLD discretization configuration.

    Returns:
        The privacy loss distribution.
    """
    ...

def truncated_gaussian_pld(
    noise_multiplier: float,
    radius: float,
    config: DiscretizationConfig,
) -> Pld:
    """Compute the PLD for a truncated (renormalized) Gaussian mechanism.

    Noise is sampled from a Gaussian restricted to [−R·σ, R·σ] with renormalized density.

    Args:
        noise_multiplier: Ratio of noise std to sensitivity (σ/Δ).
        radius: Support half-width in sigma units.
        config: PLD discretization configuration.

    Returns:
        The privacy loss distribution.
    """
    ...

# ---------------------------------------------------------------------------
# Amplification functions
# ---------------------------------------------------------------------------

def poisson_gaussian_pld(
    noise_multiplier: float,
    rate: float,
    config: DiscretizationConfig,
) -> Pld:
    """Compute the PLD for a Poisson-subsampled Gaussian mechanism.

    Args:
        noise_multiplier: σ/Δ ratio.
        rate: Poisson sampling probability, in (0, 1].
        config: PLD discretization configuration.

    Returns:
        The amplified privacy loss distribution.

    Example::

        pld = dp.poisson_gaussian_pld(1.1, 0.01, config)
        training = pld.self_compose(1000)
        training.epsilon_at(1e-5)
    """
    ...

def truncated_poisson_gaussian_pld(
    noise_multiplier: float,
    rate: float,
    batch_size_max: int,
    dataset_size: int,
    config: DiscretizationConfig,
) -> Pld:
    """Compute the PLD for a truncated Poisson-subsampled Gaussian mechanism.

    Production DP-SGD sampling: caps batch at *batch_size_max* for
    predictable memory usage and tighter privacy bounds.

    Args:
        noise_multiplier: σ/Δ ratio.
        rate: Poisson sampling probability, in (0, 1].
        batch_size_max: Maximum batch size.
        dataset_size: Total dataset size.
        config: PLD discretization configuration.

    Returns:
        The amplified privacy loss distribution.
    """
    ...

def parallel_poisson_gaussian_pld(
    noise_multiplier: float,
    rate: float,
    microbatches: int,
    config: DiscretizationConfig,
) -> Pld:
    """Compute the PLD for a parallel Poisson-subsampled Gaussian mechanism.

    Models summing multiple independent Poisson samples before adding noise once.
    Use cases: gradient accumulation (m microbatches) or parallel workers (K workers).

    Args:
        noise_multiplier: σ/Δ ratio.
        rate: Poisson sampling probability, in (0, 1].
        microbatches: Number of independent samples, > 0.
        config: PLD discretization configuration.

    Returns:
        The amplified privacy loss distribution.
    """
    ...

def poisson_rectified_gaussian_pld(
    noise_multiplier: float,
    radius: float,
    rate: float,
    config: DiscretizationConfig,
) -> Pld:
    """Compute the PLD for a Poisson-subsampled rectified Gaussian mechanism.

    The rectified (clamped) Gaussian clips noise to [−R·σ, R·σ], giving tighter
    privacy bounds than the standard unbounded Gaussian.

    Args:
        noise_multiplier: σ/Δ ratio.
        radius: Support half-width in sigma units.
        rate: Poisson sampling probability, in (0, 1].
        config: PLD discretization configuration.

    Returns:
        The amplified privacy loss distribution.
    """
    ...

def poisson_truncated_gaussian_pld(
    noise_multiplier: float,
    radius: float,
    rate: float,
    config: DiscretizationConfig,
) -> Pld:
    """Compute the PLD for a Poisson-subsampled truncated Gaussian mechanism.

    The truncated (renormalized) Gaussian restricts noise to [−R·σ, R·σ]
    with renormalized density, giving even tighter privacy bounds.

    Args:
        noise_multiplier: σ/Δ ratio.
        radius: Support half-width in sigma units.
        rate: Poisson sampling probability, in (0, 1].
        config: PLD discretization configuration.

    Returns:
        The amplified privacy loss distribution.
    """
    ...

# ---------------------------------------------------------------------------
# Matrix factorization functions
# ---------------------------------------------------------------------------

def mf_gaussian_pld(
    noise_multiplier: float,
    sensitivity: float,
    config: DiscretizationConfig,
) -> Pld:
    """Compute the PLD for a matrix factorization Gaussian mechanism.

    Computes the privacy guarantee for the entire MF training run as a
    single Gaussian mechanism with effective noise multiplier σ/S.

    Args:
        noise_multiplier: Raw noise std σ (before MF). Must be positive.
        sensitivity: L2 sensitivity of the encoder matrix. Must be positive.
        config: PLD discretization configuration.

    Returns:
        The privacy loss distribution for the entire MF training run.

    Example::

        config = dp.DiscretizationConfig()
        pld = dp.mf_gaussian_pld(1.0, 2.5, config)
        pld.epsilon_at(1e-5)
    """
    ...

def max_participation_for_linear_fn(
    x: list[float],
    min_sep: int = 1,
    max_participations: int | None = None,
) -> float:
    """Solve max_u <x, u> where u respects min-sep participation.

    Uses dynamic programming (Algorithm 3, VecSens) from
    Choquette-Choo et al. (2023).

    Args:
        x: Vector of values to optimize over.
        min_sep: Minimum separation between selections (>= 1).
        max_participations: Optional upper bound on selections.

    Returns:
        The optimal inner product.
    """
    ...

def minsep_true_max_participations(
    n: int,
    min_sep: int,
    max_participations: int | None = None,
) -> int:
    """Maximum participations under a min-sep constraint.

    Args:
        n: Number of rounds.
        min_sep: Minimum separation between participations.
        max_participations: Optional upper bound.

    Returns:
        Effective maximum participations.
    """
    ...

def single_participation_sensitivity(
    column_norms: list[float],
) -> float:
    """L2 sensitivity under single participation.

    Args:
        column_norms: L2 norms of encoder matrix columns.

    Returns:
        Maximum column norm (the sensitivity).
    """
    ...

def banded_sensitivity(
    gram_diag: list[float],
    min_sep: int = 1,
    max_participations: int | None = None,
) -> float:
    """Exact L2 sensitivity for banded Gram matrices under min-sep participation.

    Args:
        gram_diag: Diagonal of Gram matrix X = C^T C.
        min_sep: Minimum separation between participations.
        max_participations: Optional upper bound.

    Returns:
        The exact L2 sensitivity.
    """
    ...

def general_sensitivity_upper_bound(
    gram_matrix: list[float],
    n: int,
    min_sep: int = 1,
    max_participations: int | None = None,
) -> float:
    """Upper bound on L2 sensitivity for general Gram matrices.

    Args:
        gram_matrix: Flattened row-major Gram matrix X = C^T C.
        n: Matrix dimension.
        min_sep: Minimum separation between participations.
        max_participations: Optional upper bound.

    Returns:
        An upper bound on the L2 sensitivity.
    """
    ...

def fixed_epoch_sensitivity(
    gram_matrix: list[float],
    n: int,
    epochs: int,
) -> float:
    """L2 sensitivity under fixed-epoch participation.

    Args:
        gram_matrix: Flattened row-major Gram matrix X = C^T C.
        n: Matrix dimension (total rounds).
        epochs: Number of epochs (must divide n).

    Returns:
        The L2 sensitivity under fixed-epoch participation.
    """
    ...

def blt_sensitivity_squared(
    buf_decay: list[float],
    output_scale: list[float],
    n: float,
) -> float:
    """Sensitivity squared for a BLT strategy matrix.

    Implements Lemma 5.3 of the BLT paper.

    Args:
        buf_decay: Decay factors for each buffer, each in (0, 1).
        output_scale: Scale factors for each buffer.
        n: Number of iterations (use float('inf') for asymptotic limit).

    Returns:
        The sensitivity squared.
    """
    ...

def toeplitz_minsep_sensitivity_squared(
    strategy_coef: list[float],
    n: int,
    min_sep: int = 1,
    max_participations: int | None = None,
) -> float:
    """Sensitivity squared for a Toeplitz matrix under min-sep participation.

    Implements BSR Theorem 2 closed-form for non-negative, non-increasing
    Toeplitz coefficients.

    Args:
        strategy_coef: Toeplitz coefficients (non-negative, non-increasing).
        n: Matrix dimension (total rounds).
        min_sep: Minimum separation between participations.
        max_participations: Optional upper bound.

    Returns:
        The sensitivity squared.
    """
    ...

# ---------------------------------------------------------------------------
# AdaClip utility
# ---------------------------------------------------------------------------

def adaclip_sensitivity(
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
