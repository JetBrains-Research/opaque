"""Type stubs for the ``opaque_accounting`` Rust extension module.

``opaque_accounting`` provides flat PLD-based privacy accounting.
Rust computes PLDs; Python owns composition and dispatch.

The extension exports two classes (:class:`Pld` and :class:`DiscretizationConfig`)
and functions for creating PLDs from mechanism parameters.

Example::

    import opaque_accounting as dp

    pld = dp.gaussian_pld(1.1)
    composed = pld.self_compose(1000)
    print(composed.epsilon_at(1e-5))
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Pld — opaque PLD handle (Rust-backed)
# ---------------------------------------------------------------------------

class Pld:
    """An opaque Privacy Loss Distribution (PLD) — Rust-backed.

    This is the low-level native type. Users should typically interact with
    :class:`~opaque_accounting.base.CgfPld` or
    :class:`~opaque_accounting.base.PmfPld` wrappers instead.
    """

    def epsilon_at(self, delta: float) -> float:
        """Smallest ε achieving (ε, δ)-DP."""
        ...

    def delta_at(self, epsilon: float) -> float:
        """Smallest δ achieving (ε, δ)-DP."""
        ...

    def advantage(self) -> float:
        """Total-variation advantage in [0, 1]."""
        ...

    def beta_at(self, alpha: float) -> float:
        """Type-II error (β) at a given Type-I error (α)."""
        ...

    def risk_at(self, prior: float) -> float:
        """Bayes risk under an optimal adversary."""
        ...

    def compose(self, other: Pld) -> Pld:
        """Compose this PLD with another (heterogeneous composition)."""
        ...

    def self_compose(self, count: int) -> Pld:
        """Self-compose this PLD *count* times (homogeneous repetition)."""
        ...

    def is_cgf(self) -> bool:
        """Whether this PLD is CGF-backed (saddle-point, no grid)."""
        ...

    def to_pmf(self, config: DiscretizationConfig) -> Pld:
        """Materialize a CGF-backed PLD to a PMF-backed PLD.

        If already PMF-backed, returns a clone unchanged.
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
    """Compute the PLD for a Gaussian mechanism with sensitivity 1."""
    ...

def cgf_gaussian_pld(
    noise_multiplier: float,
) -> Pld:
    """Create a CGF-backed PLD for a Gaussian mechanism."""
    ...

def cgf_poisson_gaussian_pld(
    noise_multiplier: float,
    rate: float,
) -> Pld:
    """Create a CGF-backed PLD for a Poisson-subsampled Gaussian."""
    ...

def eps_delta_pld(
    epsilon: float,
    delta: float,
    config: DiscretizationConfig,
) -> Pld:
    """Compute the PLD for a fixed (ε, δ)-mechanism."""
    ...

def identity_pld(
    config: DiscretizationConfig,
) -> Pld:
    """Compute the PLD for the identity (zero-privacy-loss) mechanism."""
    ...

def rectified_gaussian_pld(
    noise_multiplier: float,
    radius: float,
    config: DiscretizationConfig,
) -> Pld:
    """Compute the PLD for a rectified (clamped) Gaussian mechanism."""
    ...

def truncated_gaussian_pld(
    noise_multiplier: float,
    radius: float,
    config: DiscretizationConfig,
) -> Pld:
    """Compute the PLD for a truncated (renormalized) Gaussian mechanism."""
    ...

# ---------------------------------------------------------------------------
# Amplification functions
# ---------------------------------------------------------------------------

def poisson_gaussian_pld(
    noise_multiplier: float,
    rate: float,
    config: DiscretizationConfig,
) -> Pld:
    """Compute the PLD for a Poisson-subsampled Gaussian mechanism."""
    ...

def truncated_poisson_gaussian_pld(
    noise_multiplier: float,
    rate: float,
    batch_size_max: int,
    dataset_size: int,
    config: DiscretizationConfig,
) -> Pld:
    """Compute the PLD for a truncated Poisson-subsampled Gaussian mechanism."""
    ...

def parallel_poisson_gaussian_pld(
    noise_multiplier: float,
    rate: float,
    microbatches: int,
    config: DiscretizationConfig,
) -> Pld:
    """Compute the PLD for a parallel Poisson-subsampled Gaussian mechanism."""
    ...

def poisson_rectified_gaussian_pld(
    noise_multiplier: float,
    radius: float,
    rate: float,
    config: DiscretizationConfig,
) -> Pld:
    """Compute the PLD for a Poisson-subsampled rectified Gaussian mechanism."""
    ...

def poisson_truncated_gaussian_pld(
    noise_multiplier: float,
    radius: float,
    rate: float,
    config: DiscretizationConfig,
) -> Pld:
    """Compute the PLD for a Poisson-subsampled truncated Gaussian mechanism."""
    ...

# ---------------------------------------------------------------------------
# Matrix factorization functions
# ---------------------------------------------------------------------------

def mf_gaussian_pld(
    noise_multiplier: float,
    sensitivity: float,
    config: DiscretizationConfig,
) -> Pld:
    """Compute the PLD for a matrix factorization Gaussian mechanism."""
    ...

def max_participation_for_linear_fn(
    x: list[float],
    min_sep: int = 1,
    max_participations: int | None = None,
) -> float:
    """Solve max_u <x, u> where u respects min-sep participation."""
    ...

def minsep_true_max_participations(
    n: int,
    min_sep: int,
    max_participations: int | None = None,
) -> int:
    """Maximum participations under a min-sep constraint."""
    ...

def single_participation_sensitivity(
    column_norms: list[float],
) -> float:
    """L2 sensitivity under single participation."""
    ...

def banded_sensitivity(
    gram_diag: list[float],
    min_sep: int = 1,
    max_participations: int | None = None,
) -> float:
    """Exact L2 sensitivity for banded Gram matrices under min-sep participation."""
    ...

def general_sensitivity_upper_bound(
    gram_matrix: list[float],
    n: int,
    min_sep: int = 1,
    max_participations: int | None = None,
) -> float:
    """Upper bound on L2 sensitivity for general Gram matrices."""
    ...

def fixed_epoch_sensitivity(
    gram_matrix: list[float],
    n: int,
    epochs: int,
) -> float:
    """L2 sensitivity under fixed-epoch participation."""
    ...

def blt_sensitivity_squared(
    buf_decay: list[float],
    output_scale: list[float],
    n: float,
) -> float:
    """Sensitivity squared for a BLT strategy matrix."""
    ...

def toeplitz_minsep_sensitivity_squared(
    strategy_coef: list[float],
    n: int,
    min_sep: int = 1,
    max_participations: int | None = None,
) -> float:
    """Sensitivity squared for a Toeplitz matrix under min-sep participation."""
    ...

# ---------------------------------------------------------------------------
# AdaClip utility
# ---------------------------------------------------------------------------

def adaclip_sensitivity(
    noise_multiplier: float,
    quantile_noise_std: float,
) -> float:
    """Combined sensitivity for adaptive clipping (Andrew et al. 2021)."""
    ...
