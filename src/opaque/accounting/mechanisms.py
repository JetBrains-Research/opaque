"""DP-SGD mechanisms: composable typed subclasses of DpProcess."""

from typing import Union

import opaque_accounting as _native

from opaque.accounting._discretization import (
    DiscretizationConfig,
    resolve_discretization,
)

# Base class
DpProcess = _native.DpProcess

# Typed subclasses (re-exported from native module)
Gaussian = _native.Gaussian
EpsDelta = _native.EpsDelta
Identity = _native.Identity
Poisson = _native.Poisson
TruncatedPoisson = _native.TruncatedPoisson
Accumulated = _native.Accumulated
AdaClip = _native.AdaClip


def gaussian(
    noise_multiplier: float,
    *,
    discretization: Union[None, float, DiscretizationConfig] = None,
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
        A :class:`Gaussian` process (typed ``DpProcess`` subclass).

    Example::

        # Single Gaussian query
        proc = acc.gaussian(1.1)
        eps = proc.epsilon_at(1e-5)

        # Composed 1000 times
        training = acc.gaussian(1.1) * 1000
        eps = training.epsilon_at(1e-5)
    """
    config = resolve_discretization(discretization)
    return _native.gaussian(noise_multiplier, discretization=config)


def poisson(
    inner: Gaussian,
    sample_rate: float,
) -> Poisson:
    """Poisson-subsampled Gaussian mechanism (standard DP-SGD step).

    Each training step selects examples independently with probability ``sample_rate``,
    computes gradients, clips them, adds Gaussian noise, and updates the model.

    This is the **standard DP-SGD mechanism** used in most deep learning privacy work.

    Args:
        inner: The base :class:`Gaussian` mechanism (from :func:`gaussian`).
        sample_rate: Probability of including each example (batch_size / dataset_size).

    Returns:
        A :class:`Poisson` process (typed ``DpProcess`` subclass).

    Example::

        # One training step
        step = acc.poisson(acc.gaussian(1.1), sample_rate=0.01)

        # 1000 steps of training
        training = step * 1000
        eps = training.epsilon_at(1e-5)
    """
    return _native.poisson(inner, sample_rate)


def truncated_poisson(
    inner: Gaussian,
    sample_rate: float,
    batch_size_cap: int,
    dataset_size: int,
) -> TruncatedPoisson:
    """Truncated Poisson sampling (production DP-SGD with capped batch size).

    In real systems, batch size is capped at ``batch_size_cap`` even though Poisson
    sampling can produce larger batches. This gives tighter privacy bounds than
    standard Poisson subsampling.

    **Use this for production DP-SGD** when you have a fixed batch size limit.

    Args:
        inner: The base :class:`Gaussian` mechanism (from :func:`gaussian`).
        sample_rate: Probability of including each example (batch_size / dataset_size).
        batch_size_cap: Maximum batch size (actual batches are capped at this value).
        dataset_size: Total number of examples in the dataset.

    Returns:
        A :class:`TruncatedPoisson` process (typed ``DpProcess`` subclass).

    Example::

        # CIFAR-10: n=50k, batch=250, σ=0.8
        n = 50_000
        batch = 250
        g = acc.gaussian(0.8)
        step = acc.truncated_poisson(g, batch / n, batch, n)
        training = step * 1000
        eps = training.epsilon_at(1e-5)
    """
    return _native.truncated_poisson(inner, sample_rate, batch_size_cap, dataset_size)


def accumulate(
    inner: Poisson,
    microbatches: int,
) -> Accumulated:
    """Gradient accumulation (microbatching) mechanism.

    Process gradients in ``microbatches`` sub-batches, accumulate clipped gradients,
    then add noise once. This improves gradient quality compared to adding noise
    per microbatch while maintaining the same privacy guarantee.

    Args:
        inner: A :class:`Poisson` process (from :func:`poisson`).
        microbatches: Number of microbatches to accumulate before noising.

    Returns:
        An :class:`Accumulated` process (typed ``DpProcess`` subclass).

    Example::

        # Accumulate 4 microbatches per step
        step = acc.accumulate(
            acc.poisson(acc.gaussian(1.1), 0.01),
            microbatches=4,
        )
        training = step * 500
        eps = training.epsilon_at(1e-5)
    """
    return _native.accumulate(inner, microbatches)


def adaclip(
    inner: Gaussian,
    quantile_noise_std: float,
) -> AdaClip:
    """Adaptive clipping mechanism (Andrew et al. 2021).

    Adaptive clipping adjusts the clipping threshold based on the empirical
    distribution of gradient norms. The quantile estimation uses a noisy mechanism,
    adding extra privacy cost.

    The total privacy cost is the composition of:

    - Base Gaussian mechanism (``inner``)
    - Quantile estimation mechanism (``quantile_noise_std``)

    Args:
        inner: The base :class:`Gaussian` mechanism (from :func:`gaussian`).
        quantile_noise_std: Noise std for quantile estimation.
            Larger = more private quantile, less accurate clipping.

    Returns:
        An :class:`AdaClip` process (typed ``DpProcess`` subclass).

    Example::

        step = acc.adaclip(acc.gaussian(1.1), quantile_noise_std=50.0)
        eps = step.epsilon_at(1e-5)
    """
    return _native.adaclip(inner, quantile_noise_std)


def eps_delta(
    epsilon: float,
    delta: float = 0.0,
    *,
    discretization: Union[None, float, DiscretizationConfig] = None,
) -> EpsDelta:
    """Fixed (ε, δ)-DP guarantee (for composition with other mechanisms).

    Useful when you have an external mechanism with known privacy parameters
    that you want to compose with other tracked processes.

    Args:
        epsilon: Privacy parameter ε.
        delta: Privacy parameter δ. Default: 0.0 (pure ε-DP).
        discretization: PLD precision config (keyword-only).

    Returns:
        An :class:`EpsDelta` process (typed ``DpProcess`` subclass).

    Example::

        # External mechanism with (3.0, 1e-5)-DP
        external = acc.eps_delta(3.0, 1e-5)

        # Compose with DP-SGD
        training = acc.poisson(acc.gaussian(1.1), 0.01) * 1000
        total = external | training
        eps = total.epsilon_at(1e-5)
    """
    config = resolve_discretization(discretization)
    return _native.eps_delta(epsilon, delta, discretization=config)


def identity(
    *,
    discretization: Union[None, float, DiscretizationConfig] = None,
) -> Identity:
    """Identity mechanism (zero privacy loss).

    Useful as a placeholder or identity element in composition.

    Args:
        discretization: PLD precision config (keyword-only).

    Returns:
        An :class:`Identity` process (typed ``DpProcess`` subclass).

    Example::

        # Identity has ε=0 for any δ
        proc = acc.identity()
        eps = proc.epsilon_at(1e-5)  # ~0
    """
    config = resolve_discretization(discretization)
    return _native.identity(discretization=config)