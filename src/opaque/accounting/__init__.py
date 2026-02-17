"""Differential privacy accounting using Privacy Loss Distributions (PLD).

This module provides a compositional API for tracking privacy guarantees:

- **Mechanisms**: gaussian(), poisson(), truncated_poisson(), etc.
- **Composition**: Combine processes using ``*`` (repeat) or ``|`` (compose)
- **Metrics**: Query privacy with epsilon_at(), delta_at(), advantage(), etc.

The underlying implementation uses Google's PLD accounting via the
``opaque-accounting`` Rust crate (PyO3 bindings).

Example::

    import opaque.accounting as acc

    # Create a DP-SGD step
    step = acc.poisson(noise_multiplier=1.1, sample_rate=0.01)

    # Compose 1000 steps
    training = step * 1000

    # Query privacy at delta=1e-5
    epsilon = training.epsilon_at(1e-5)
    print(f"Privacy: (ε={epsilon:.2f}, δ=1e-5)")

For calibration (finding noise for target privacy budget), use the
:mod:`opaque.accounting.calibration` submodule.
"""

from typing import Optional, Union

try:
    import opaque_accounting as _native
except ImportError as e:
    raise ImportError(
        "opaque-accounting native module not found. "
        "Install with: maturin develop -m crates/dp-accounting/Cargo.toml"
    ) from e

# =============================================================================
# Module-level discretization defaults
# =============================================================================

_default_discretization: Optional["DiscretizationConfig"] = None


def set_discretization(
    discretization: float = 1e-4,
    log_mass_truncation_bound: float = -50.0,
    pessimistic_estimate: bool = True,
    max_grid_size: int = 10_000_000,
) -> None:
    """Set module-level default discretization parameters.

    These defaults are used when ``discretization=None`` is passed to mechanism
    constructors. By default, uses high-precision settings matching Google's
    ``dp_accounting`` library.

    Args:
        discretization: Grid spacing for PLD PMF. Smaller = more precise, larger = faster.
            Error scales as O(disc^2). Default: 1e-4.
        log_mass_truncation_bound: Tails with probability below exp(bound) are
            truncated. Default: -50 (matching Google).
        pessimistic_estimate: If True (default), round probabilities upward to produce
            an **upper bound** on privacy loss (safe for guarantees). If False, round
            downward (optimistic estimate, useful for debugging only).
        max_grid_size: If grid exceeds this many bins, coarsen discretization
            automatically. Default: 10,000,000.

    Example::

        # Use coarser discretization for faster computation
        acc.set_discretization(discretization=1e-3)

        # Use maximum precision
        acc.set_discretization(discretization=1e-5, max_grid_size=100_000_000)
    """
    global _default_discretization
    _default_discretization = DiscretizationConfig(
        discretization=discretization,
        log_mass_truncation_bound=log_mass_truncation_bound,
        pessimistic_estimate=pessimistic_estimate,
        max_grid_size=max_grid_size,
    )


def get_discretization() -> Optional["DiscretizationConfig"]:
    """Get the current module-level default discretization config.

    Returns:
        Current discretization config, or None if using native defaults.
    """
    return _default_discretization


def _resolve_discretization(
    config: Union[None, float, "DiscretizationConfig"]
) -> Optional["DiscretizationConfig"]:
    """Resolve discretization parameter to a config object.

    Args:
        config: None (use module default), float (use as discretization value),
            or DiscretizationConfig (use as-is).

    Returns:
        Resolved DiscretizationConfig or None (use Rust defaults).
    """
    if config is None:
        return _default_discretization
    elif isinstance(config, (int, float)):
        # Convert float to DiscretizationConfig using current defaults
        base = _default_discretization or DiscretizationConfig()
        return DiscretizationConfig(
            discretization=float(config),
            log_mass_truncation_bound=base.log_mass_truncation_bound,
            pessimistic_estimate=base.pessimistic_estimate,
            max_grid_size=base.max_grid_size,
        )
    else:
        return config


# =============================================================================
# Re-export types from native module
# =============================================================================

DpProcess = _native.DpProcess
"""A differential privacy process that can be queried for privacy guarantees.

This is the central class in ``opaque.accounting``. Every mechanism constructor
(``gaussian``, ``poisson``, etc.) returns a ``DpProcess``, and composition
operators produce new ``DpProcess`` instances.

All privacy metrics are derived from the same Privacy Loss Distribution (PLD):

- **epsilon_at(delta)**: Get epsilon for given delta (ε,δ-DP)
- **delta_at(epsilon)**: Get delta for given epsilon (ε,δ-DP)
- **advantage()**: Get f-DP total-variation advantage
- **beta_at(alpha)**: Get Type-II error at given Type-I error (hypothesis testing)
- **risk_at(prior)**: Get Bayes risk at given prior

Composition operators:

- **step * 1000**: Repeat a process 1000 times (homogeneous composition)
- **a | b**: Compose two different processes (heterogeneous composition)

Debugging:

- **print(proc)**: One-line summary with epsilon
- **describe()**: Constructor parameters as dict
- **pld_info()**: PLD grid diagnostics with timing
- **summary()**: Multi-line formatted privacy report

Example::

    step = acc.poisson(1.1, 0.01)
    training = step * 1000
    eps = training.epsilon_at(1e-5)
    print(training.summary())  # detailed report
"""

DiscretizationConfig = _native.DiscretizationConfig
"""Configuration controlling PLD discretization precision.

The PLD is represented as a discrete probability mass function (PMF) on a
regular grid. These parameters control grid resolution, tail truncation,
and rounding direction.

Defaults are chosen for high accuracy (discretization=1e-4 gives ~1e-8 error
per composition step). Coarser grids are faster but less precise.

Args:
    discretization: Grid spacing for PLD PMF. Default: 1e-4.
        Smaller = more precise, larger grid. Error scales as O(disc^2).
    log_mass_truncation_bound: Tails with probability below exp(bound) are
        truncated. Default: -50 (matching Google's dp_accounting).
    pessimistic_estimate: If True (default), round probabilities upward to
        produce an **upper bound** on privacy loss. If False, round downward
        (optimistic estimate - not safe for guarantees).
    max_grid_size: If grid exceeds this many bins, coarsen discretization
        automatically. Default: 10,000,000.

Example::

    # Faster but less precise
    cfg = acc.DiscretizationConfig(discretization=1e-3)

    # Maximum precision
    cfg = acc.DiscretizationConfig(
        discretization=1e-5,
        log_mass_truncation_bound=-50.0,
    )

    # Use with any mechanism
    proc = acc.gaussian(1.1, discretization=cfg)
"""


# =============================================================================
# Mechanism constructors
# =============================================================================


def gaussian(
    noise_multiplier: float,
    discretization: Union[None, float, DiscretizationConfig] = None,
) -> DpProcess:
    """Gaussian mechanism with noise multiplier σ.

    The Gaussian mechanism adds noise ~ N(0, σ²) to sensitivity-1 queries.
    This is the base mechanism for standard DP-SGD.

    Args:
        noise_multiplier: Noise standard deviation divided by sensitivity (σ/Δ).
            Larger values = more privacy, less utility.
        discretization: PLD precision config. Can be:
            - None: use module default (see :func:`set_discretization`)
            - float: use as grid spacing
            - DiscretizationConfig: full config

    Returns:
        DpProcess for a single Gaussian query.

    Example::

        # Single Gaussian query
        proc = acc.gaussian(1.1)
        eps = proc.epsilon_at(1e-5)

        # Composed 1000 times
        training = acc.gaussian(1.1) * 1000
        eps = training.epsilon_at(1e-5)
    """
    config = _resolve_discretization(discretization)
    return _native.gaussian(noise_multiplier, config=config)


def poisson(
    noise_multiplier: float,
    sample_rate: float,
    discretization: Union[None, float, DiscretizationConfig] = None,
) -> DpProcess:
    """Poisson-subsampled Gaussian mechanism (standard DP-SGD step).

    Each training step selects examples independently with probability ``sample_rate``,
    computes gradients, clips them, adds Gaussian noise with std ``noise_multiplier``,
    and updates the model.

    This is the **standard DP-SGD mechanism** used in most deep learning privacy work.

    Args:
        noise_multiplier: Gradient noise std (σ/Δ). Larger = more privacy.
        sample_rate: Probability of including each example (batch_size / dataset_size).
        discretization: PLD precision config (see :func:`gaussian`).

    Returns:
        DpProcess for one DP-SGD training step.

    Example::

        # One training step
        step = acc.poisson(noise_multiplier=1.1, sample_rate=0.01)

        # 1000 steps of training
        training = step * 1000
        eps = training.epsilon_at(1e-5)
    """
    config = _resolve_discretization(discretization)
    return _native.poisson(noise_multiplier, sample_rate, config=config)


def truncated_poisson(
    noise_multiplier: float,
    sample_rate: float,
    batch_size_cap: int,
    dataset_size: int,
    discretization: Union[None, float, DiscretizationConfig] = None,
) -> DpProcess:
    """Truncated Poisson sampling (production DP-SGD with capped batch size).

    In real systems, batch size is capped at ``batch_size_cap`` even though Poisson
    sampling can produce larger batches. This gives tighter privacy bounds than
    standard Poisson subsampling.

    **Use this for production DP-SGD** when you have a fixed batch size limit.

    Args:
        noise_multiplier: Gradient noise std (σ/Δ).
        sample_rate: Probability of including each example (batch_size / dataset_size).
        batch_size_cap: Maximum batch size (actual batches are capped at this value).
        dataset_size: Total number of examples in the dataset.
        discretization: PLD precision config (see :func:`gaussian`).

    Returns:
        DpProcess for one truncated Poisson DP-SGD step.

    Example::

        # CIFAR-10: n=50k, batch=250, sigma=0.8
        n = 50_000
        batch = 250
        step = acc.truncated_poisson(
            noise_multiplier=0.8,
            sample_rate=batch / n,
            batch_size_cap=batch,
            dataset_size=n,
        )
        training = step * 1000
        eps = training.epsilon_at(1e-5)
    """
    config = _resolve_discretization(discretization)
    return _native.truncated_poisson(
        noise_multiplier,
        sample_rate,
        batch_size_cap,
        dataset_size,
        config=config,
    )


def accumulate(
    noise_multiplier: float,
    sample_rate: float,
    microbatches: int,
    discretization: Union[None, float, DiscretizationConfig] = None,
) -> DpProcess:
    """Gradient accumulation (microbatching) mechanism.

    Process gradients in ``microbatches`` sub-batches, accumulate clipped gradients,
    then add noise once. This improves gradient quality compared to adding noise
    per microbatch while maintaining the same privacy guarantee.

    Args:
        noise_multiplier: Gradient noise std (σ/Δ) applied to accumulated gradient.
        sample_rate: Probability of including each example (batch_size / dataset_size).
        microbatches: Number of microbatches to accumulate before noising.
        discretization: PLD precision config (see :func:`gaussian`).

    Returns:
        DpProcess for one accumulation step.

    Example::

        # Accumulate 4 microbatches per step
        step = acc.accumulate(
            noise_multiplier=1.1,
            sample_rate=0.01,
            microbatches=4,
        )
        training = step * 500
        eps = training.epsilon_at(1e-5)
    """
    config = _resolve_discretization(discretization)
    return _native.accumulate(noise_multiplier, sample_rate, microbatches, config=config)


def adaclip(
    noise_multiplier: float,
    quantile_noise_std: float,
    discretization: Union[None, float, DiscretizationConfig] = None,
) -> DpProcess:
    """Adaptive clipping mechanism (Andrew et al. 2021).

    Adaptive clipping adjusts the clipping threshold based on the empirical
    distribution of gradient norms. The quantile estimation uses a noisy mechanism,
    adding extra privacy cost.

    The total privacy cost is the composition of:
    - Base Gaussian mechanism (noise_multiplier)
    - Quantile estimation mechanism (quantile_noise_std)

    Args:
        noise_multiplier: Gradient noise std for main mechanism (σ/Δ).
        quantile_noise_std: Noise std for quantile estimation.
            Larger = more private quantile, less accurate clipping.
        discretization: PLD precision config (see :func:`gaussian`).

    Returns:
        DpProcess for one AdaClip step.

    Example::

        step = acc.adaclip(
            noise_multiplier=1.1,
            quantile_noise_std=50.0,
        )
        eps = step.epsilon_at(1e-5)
    """
    config = _resolve_discretization(discretization)
    return _native.adaclip(noise_multiplier, quantile_noise_std, config=config)


def eps_delta(
    epsilon: float,
    delta: float = 0.0,
    discretization: Union[None, float, DiscretizationConfig] = None,
) -> DpProcess:
    """Fixed (ε, δ)-DP guarantee (for composition with other mechanisms).

    Useful when you have an external mechanism with known privacy parameters
    that you want to compose with other tracked processes.

    Args:
        epsilon: Privacy parameter ε.
        delta: Privacy parameter δ. Default: 0.0 (pure ε-DP).
        discretization: PLD precision config (see :func:`gaussian`).

    Returns:
        DpProcess representing the fixed guarantee.

    Example::

        # External mechanism with (3.0, 1e-5)-DP
        external = acc.eps_delta(3.0, 1e-5)

        # Compose with DP-SGD
        training = acc.poisson(1.1, 0.01) * 1000
        total = external | training
        eps = total.epsilon_at(1e-5)
    """
    config = _resolve_discretization(discretization)
    return _native.eps_delta(epsilon, delta, config=config)


def identity(
    discretization: Union[None, float, DiscretizationConfig] = None,
) -> DpProcess:
    """Identity mechanism (zero privacy loss).

    Useful as a placeholder or identity element in composition.

    Args:
        discretization: PLD precision config (see :func:`gaussian`).

    Returns:
        DpProcess with zero privacy loss.

    Example::

        # Identity has ε=0 for any δ
        proc = acc.identity()
        eps = proc.epsilon_at(1e-5)  # ~0
    """
    config = _resolve_discretization(discretization)
    return _native.identity(config=config)


# =============================================================================
# Composition functions
# =============================================================================


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


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Types
    "DpProcess",
    "DiscretizationConfig",
    # Module defaults
    "set_discretization",
    "get_discretization",
    # Mechanisms
    "gaussian",
    "poisson",
    "truncated_poisson",
    "accumulate",
    "adaclip",
    "eps_delta",
    "identity",
    # Composition
    "repeat",
    "compose",
]
