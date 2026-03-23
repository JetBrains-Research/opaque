"""Ex-post per-sample privacy accounting for DP-SGD.

Implements Formulation B (ex-post / conditional) from the individual privacy
accounting literature (Feldman & Zrnic NeurIPS 2021, Koskela et al. ICLR 2023,
Yu et al. arXiv:2206.02617).

Key insight: at step t, if sample i was NOT in the batch its privacy loss is
exactly zero (identity).  If it WAS in the batch, the privacy loss is that of
the **base Gaussian mechanism** at its observed sensitivity — no Poisson
subsampling mixture needed.  We only compose over the ~q·T steps where the
sample actually participated.

Note: the *average* ex-post epsilon across all samples is generally LOOSER
than the ex-ante formulation (composing PoissonSubsample(Gaussian) for all T
steps).  The ex-ante PLD equals the Binomial mixture of ex-post PLDs, and
finding a single ε threshold for the mixture is at least as good as averaging
individual ε thresholds (Jensen's inequality on the hockey-stick divergence).
The value of the ex-post approach is per-sample heterogeneity: individual
samples with few participations or small sensitivities get much lower epsilon.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import opaque_accounting as acc
from opaque_accounting.base import DpProcess
from opaque_accounting.mechanisms.mip_gaussian import MipGaussian


def per_sample_expost_epsilon(
    noise_multiplier: float,
    sample_step_sensitivities: Sequence[Sequence[float]],
    delta: float,
) -> float:
    """Average epsilon via ex-post per-sample composition.

    For each sample, composes the base Gaussian mechanism only over the steps
    where that sample was included in the batch (using the observed gradient
    norm at each such step).  Steps where the sample was absent contribute
    identity (zero privacy loss).

    Args:
        noise_multiplier: Noise multiplier σ/C.
        sample_step_sensitivities: For each sample, a list of sensitivities
            at the steps where it was included in the batch.  The length of
            each inner list equals the number of participations for that
            sample (NOT the total number of training steps).
        delta: Target delta for epsilon computation.

    Returns:
        Average epsilon across all samples.

    Example::

        # Sample 0 was in 3 batches with these norms; sample 1 in 5 batches
        participations = [
            [0.3, 0.4, 0.35],
            [0.9, 0.95, 0.88, 0.91, 0.87],
        ]
        avg_eps = per_sample_expost_epsilon(0.8, participations, delta=1e-5)
    """
    if not sample_step_sensitivities:
        return 0.0

    total_eps = 0.0
    n = len(sample_step_sensitivities)

    for sensitivities in sample_step_sensitivities:
        if not sensitivities:
            continue

        # Group identical sensitivities for Repeated merging
        sens_counts: Counter[float] = Counter()
        for s in sensitivities:
            sens_counts[round(s, 6)] += 1

        process: DpProcess = acc.identity()
        for s, count in sens_counts.items():
            s_clamped = max(s, 1e-8)
            # Base Gaussian mechanism — NO Poisson subsampling wrapper.
            mip = MipGaussian(
                noise_multiplier=noise_multiplier,
                sensitivities=(s_clamped,),
                weights=(1.0,),
            )
            process = process | (mip * count)

        total_eps += process.epsilon_at(delta)

    return total_eps / n


def per_sample_expost_epsilon_fixed(
    noise_multiplier: float,
    sample_sensitivities: Sequence[float],
    num_participations: int | Sequence[int],
    delta: float,
) -> float:
    """Average epsilon via ex-post composition with fixed per-sample sensitivity.

    Simplified variant: each sample has a fixed sensitivity and a known number
    of batch participations (either the same for all or specified per sample).

    Args:
        noise_multiplier: Noise multiplier σ/C.
        sample_sensitivities: Per-sample normalized sensitivities (s_i = ||g_i||/C).
        num_participations: Number of times each sample was in a batch.
            If a single int, the same count is used for all samples.
            If a sequence, must have the same length as sample_sensitivities.
        delta: Target delta for epsilon computation.

    Returns:
        Average epsilon across all samples.

    Example::

        sensitivities = [0.2, 0.5, 0.8, 1.0]
        # Each sample appeared in ~50 batches (q*T)
        avg_eps = per_sample_expost_epsilon_fixed(
            0.8, sensitivities, num_participations=50, delta=1e-5,
        )
    """
    if not sample_sensitivities:
        return 0.0

    n = len(sample_sensitivities)

    if isinstance(num_participations, int):
        counts = [num_participations] * n
    else:
        counts = list(num_participations)
        if len(counts) != n:
            raise ValueError(
                f"num_participations length ({len(counts)}) != "
                f"sample_sensitivities length ({n})"
            )

    total_eps = 0.0

    # Group by (sensitivity, count) to avoid redundant PLD computation.
    key_counts: Counter[tuple[float, int]] = Counter()
    for s, c in zip(sample_sensitivities, counts):
        key_counts[(round(s, 6), c)] += 1

    for (s, c), num_samples in key_counts.items():
        if c == 0:
            continue
        s_clamped = max(s, 1e-8)
        mip = MipGaussian(
            noise_multiplier=noise_multiplier,
            sensitivities=(s_clamped,),
            weights=(1.0,),
        )
        process = mip * c
        eps_i = process.epsilon_at(delta)
        total_eps += eps_i * num_samples

    return total_eps / n
