"""Mechanism constructors: convenience functions that build typed DpProcess nodes.

Each constructor validates inputs, resolves discretization config, and returns
the appropriate frozen dataclass from :mod:`opaque.accounting.types`.
"""

from __future__ import annotations

import opaque_accounting as _native

from opaque.accounting.base import DpProcess, DiscretizationConfig
from opaque.accounting.discretization import resolve_pld_config
from opaque.accounting.nodes import Identity
from opaque.accounting.types import (
    Accumulated,
    EpsDelta,
    Gaussian,
    Poisson,
    TruncatedPoisson,
)


def gaussian(
    noise_multiplier: float,
    *,
    discretization: None | float | DiscretizationConfig = None,
) -> Gaussian:
    """Gaussian mechanism with noise multiplier σ.

    The Gaussian mechanism adds noise ~ N(0, σ²) to sensitivity-1 queries.
    This is the base mechanism for standard DP-SGD.

    Args:
        noise_multiplier: Noise standard deviation divided by sensitivity (σ/Δ).
            Larger values = more privacy, less utility.
        discretization: PLD precision config (keyword-only). Can be:
            - None: use module default (see :func:`set_discretization`)
            - float: use as grid spacing
            - DiscretizationConfig: full config

    Returns:
        A :class:`Gaussian` process.

    Example::

        # Single Gaussian query
        proc = acc.gaussian(1.1)
        eps = proc.epsilon_at(1e-5)

        # Composed 1000 times
        training = acc.gaussian(1.1) * 1000
        eps = training.epsilon_at(1e-5)
    """
    config = resolve_pld_config(discretization)
    return Gaussian(noise_multiplier, config=config)


def poisson(
    inner: Gaussian,
    sample_rate: float,
) -> Poisson:
    """Poisson-subsampled Gaussian mechanism (standard DP-SGD step).

    Each training step selects examples independently with probability ``sample_rate``,
    computes gradients, clips them, adds Gaussian noise, and updates the model.

    This is the **standard DP-SGD mechanism** used in most deep learning privacy work.

    Args:
        inner: The base Gaussian mechanism (from :func:`gaussian`).
        sample_rate: Probability of including each example (batch_size / dataset_size).

    Returns:
        A :class:`Poisson` process.

    Example::

        # One training step
        step = acc.poisson(acc.gaussian(1.1), sample_rate=0.01)

        # 1000 steps of training
        training = step * 1000
        eps = training.epsilon_at(1e-5)
    """
    if not isinstance(inner, Gaussian):
        raise TypeError(
            f"poisson() requires a Gaussian inner mechanism, got {type(inner).__name__}. "
            "Use: acc.poisson(acc.gaussian(noise_multiplier), sample_rate)"
        )
    return Poisson(inner.noise_multiplier, sample_rate, config=inner.config)


def truncated_poisson(
    inner: Gaussian,
    sample_rate: float,
    batch_size_cap: int,
    dataset_size: int,
) -> DpProcess:
    """Truncated Poisson sampling (production DP-SGD with capped batch size).

    In real systems, batch size is capped at ``batch_size_cap`` even though Poisson
    sampling can produce larger batches. This gives tighter privacy bounds than
    standard Poisson subsampling.

    **Use this for production DP-SGD** when you have a fixed batch size limit.

    Args:
        inner: The base Gaussian mechanism (from :func:`gaussian`).
        sample_rate: Probability of including each example (batch_size / dataset_size).
        batch_size_cap: Maximum batch size (actual batches are capped at this value).
        dataset_size: Total number of examples in the dataset.

    Returns:
        A :class:`TruncatedPoisson` process.

    Example::

        # CIFAR-10: n=50k, batch=250, σ=0.8
        n = 50_000
        batch = 250
        g = acc.gaussian(0.8)
        step = acc.truncated_poisson(g, batch / n, batch, n)
        training = step * 1000
        eps = training.epsilon_at(1e-5)
    """
    if not isinstance(inner, Gaussian):
        raise TypeError(
            f"truncated_poisson() requires a Gaussian inner mechanism, got {type(inner).__name__}."
        )
    return TruncatedPoisson(
        inner.noise_multiplier,
        sample_rate,
        batch_size_cap,
        dataset_size,
        config=inner.config,
    )


def accumulate(
    inner: Poisson,
    microbatches: int,
) -> DpProcess:
    """Gradient accumulation (microbatching) mechanism.

    Process gradients in ``microbatches`` sub-batches, accumulate clipped gradients,
    then add noise once. This improves gradient quality compared to adding noise
    per microbatch while maintaining the same privacy guarantee.

    Args:
        inner: A Poisson process (from :func:`poisson`).
        microbatches: Number of microbatches to accumulate before noising.

    Returns:
        An :class:`Accumulated` process.

    Example::

        # Accumulate 4 microbatches per step
        step = acc.accumulate(
            acc.poisson(acc.gaussian(1.1), 0.01),
            microbatches=4,
        )
        training = step * 500
        eps = training.epsilon_at(1e-5)
    """
    if not isinstance(inner, Poisson):
        raise TypeError(
            f"accumulate() requires a Poisson inner mechanism, got {type(inner).__name__}. "
            "Use: acc.accumulate(acc.poisson(acc.gaussian(nm), rate), microbatches=k)"
        )
    return Accumulated(
        inner.noise_multiplier,
        inner.sample_rate,
        microbatches,
        config=inner.config,
    )


def adaclip(
    inner: Gaussian,
    quantile_noise_std: float,
) -> Gaussian:
    """Adaptive clipping mechanism (Andrew et al. 2021).

    Adaptive clipping adjusts the clipping threshold based on the empirical
    distribution of gradient norms. The quantile estimation uses a noisy mechanism,
    adding extra privacy cost.

    The total privacy cost uses the combined sensitivity formula:
    ``z_eff = 1 / sqrt(1/z² + 1/(4·σ_b²))``

    where z is the base noise multiplier and σ_b is the quantile noise std.

    The result is a Gaussian mechanism with the effective noise multiplier,
    so it can be composed with :func:`poisson` or :func:`truncated_poisson`::

        step = acc.poisson(acc.adaclip(acc.gaussian(1.1), 50.0), 0.01)

    Args:
        inner: The base Gaussian mechanism (from :func:`gaussian`).
        quantile_noise_std: Noise std for quantile estimation.
            Larger = more private quantile, less accurate clipping.

    Returns:
        A :class:`Gaussian` with the effective (reduced) noise multiplier.

    Example::

        step = acc.adaclip(acc.gaussian(1.1), quantile_noise_std=50.0)
        eps = step.epsilon_at(1e-5)
    """
    if not isinstance(inner, Gaussian):
        raise TypeError(
            f"adaclip() requires a Gaussian inner mechanism, got {type(inner).__name__}."
        )
    # Compute effective noise multiplier: z_eff = 1 / combined_sensitivity
    s = _native.combined_sensitivity(inner.noise_multiplier, quantile_noise_std)
    z_eff = 1.0 / s
    # Return Gaussian so the result can be fed into poisson()/truncated_poisson()
    return Gaussian(z_eff, config=inner.config)


def eps_delta(
    epsilon: float,
    delta: float = 0.0,
    *,
    discretization: None | float | DiscretizationConfig = None,
) -> DpProcess:
    """Fixed (ε, δ)-DP guarantee (for composition with other mechanisms).

    Useful when you have an external mechanism with known privacy parameters
    that you want to compose with other tracked processes.

    Args:
        epsilon: Privacy parameter ε.
        delta: Privacy parameter δ. Default: 0.0 (pure ε-DP).
        discretization: PLD precision config (keyword-only).

    Returns:
        A :class:`DpProcess` wrapping an (ε, δ) PLD.

    Example::

        # External mechanism with (3.0, 1e-5)-DP
        external = acc.eps_delta(3.0, 1e-5)

        # Compose with DP-SGD
        training = acc.poisson(acc.gaussian(1.1), 0.01) * 1000
        total = external | training
        eps = total.epsilon_at(1e-5)
    """
    config = resolve_pld_config(discretization)
    return EpsDelta(epsilon, delta, config=config)


def identity(
    *,
    discretization: None | float | DiscretizationConfig = None,
) -> DpProcess:
    """Identity mechanism (zero privacy loss).

    Useful as a placeholder or identity element in composition.

    Args:
        discretization: PLD precision config (keyword-only).

    Returns:
        An :class:`~opaque.accounting.nodes.Identity` process (ε=0 for any δ).

    Example::

        # Identity has ε=0 for any δ
        proc = acc.identity()
        eps = proc.epsilon_at(1e-5)  # ~0
    """
    config = resolve_pld_config(discretization)
    return Identity(config=config)
