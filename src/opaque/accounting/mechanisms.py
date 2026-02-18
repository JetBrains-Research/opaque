"""DP-SGD mechanisms: Gaussian, Poisson, etc."""

from typing import Union

import opaque_accounting as _native

from opaque.accounting._discretization import (
    DiscretizationConfig,
    resolve_discretization,
)

DpProcess = _native.DpProcess


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
    config = resolve_discretization(discretization)
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
    config = resolve_discretization(discretization)
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
    config = resolve_discretization(discretization)
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
    config = resolve_discretization(discretization)
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
    config = resolve_discretization(discretization)
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
    config = resolve_discretization(discretization)
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
    config = resolve_discretization(discretization)
    return _native.identity(config=config)
