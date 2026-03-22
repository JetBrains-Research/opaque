"""Per-sample trajectory composition for exact multi-step stochastic f-MIP.

Computes the true per-sample privacy by composing each sample's
Poisson-subsampled trajectory independently, then averaging.  This avoids
the cross-trajectory terms introduced by composing per-step mixture PLDs.

Each sample has a known sensitivity s_i (possibly varying per step).
The per-sample PLD at each step is PoissonSubsample(Gaussian(nm, s_i), q).
Composing T steps and averaging across samples gives the exact multi-step
stochastic f-MIP.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import opaque_accounting as acc
from opaque_accounting.base import DpProcess
from opaque_accounting.mechanisms.mip_gaussian import MipGaussian


def per_sample_composed_epsilon(
    noise_multiplier: float,
    sample_rate: float,
    sample_sensitivities: Sequence[float],
    num_steps: int,
    delta: float,
) -> float:
    """Average epsilon from per-sample trajectory composition.

    For each sample with known sensitivity s_i, composes T steps of
    PoissonSubsample(Gaussian(nm, s_i), q), then averages epsilon.

    Args:
        noise_multiplier: Noise multiplier σ/C.
        sample_rate: Poisson sampling rate q.
        sample_sensitivities: Per-sample normalized sensitivities.
            Each value is s_i = ||g_i|| / C for the sample (fixed across
            steps in this simplified model).
        num_steps: Total number of training steps T.
        delta: Target delta for epsilon computation.

    Returns:
        Average epsilon across all samples.

    Example::

        # 4 samples with different sensitivities, 100 steps
        sensitivities = [0.2, 0.5, 0.8, 1.0]
        avg_eps = per_sample_composed_epsilon(
            0.8, 0.01, sensitivities, num_steps=100, delta=1e-5
        )
    """
    if not sample_sensitivities:
        return 0.0

    total_eps = 0.0
    n = len(sample_sensitivities)

    # Group samples with identical sensitivities to avoid redundant PLD computation.
    sens_counts: Counter[float] = Counter()
    for s in sample_sensitivities:
        sens_counts[round(s, 6)] += 1

    for s, count in sens_counts.items():
        s_clamped = max(s, 1e-8)
        mip = MipGaussian(
            noise_multiplier=noise_multiplier,
            sensitivities=(s_clamped,),
            weights=(1.0,),
        )
        step = acc.poisson(mip, sample_rate=sample_rate)
        process = step * num_steps
        eps_i = process.epsilon_at(delta)
        total_eps += eps_i * count

    return total_eps / n


def per_sample_varying_composed_epsilon(
    noise_multiplier: float,
    sample_rate: float,
    sample_step_sensitivities: Sequence[Sequence[float]],
    delta: float,
) -> float:
    """Average epsilon with per-sample, per-step varying sensitivities.

    For the case where sensitivity changes per step (e.g., across epochs).
    Each sample has a list of T sensitivities, one per step.

    Args:
        noise_multiplier: Noise multiplier σ/C.
        sample_rate: Poisson sampling rate q.
        sample_step_sensitivities: For each sample, a list of T
            sensitivities (one per training step).
        delta: Target delta for epsilon computation.

    Returns:
        Average epsilon across all samples.
    """
    if not sample_step_sensitivities:
        return 0.0

    total_eps = 0.0
    n = len(sample_step_sensitivities)

    for sensitivities in sample_step_sensitivities:
        # Group identical sensitivities for Repeated merging
        sens_counts: Counter[float] = Counter()
        for s in sensitivities:
            sens_counts[round(s, 6)] += 1

        process: DpProcess = acc.identity()
        for s, count in sens_counts.items():
            s_clamped = max(s, 1e-8)
            mip = MipGaussian(
                noise_multiplier=noise_multiplier,
                sensitivities=(s_clamped,),
                weights=(1.0,),
            )
            step = acc.poisson(mip, sample_rate=sample_rate)
            process = process | (step * count)

        total_eps += process.epsilon_at(delta)

    return total_eps / n
