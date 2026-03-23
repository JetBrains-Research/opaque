"""Stochastic f-MIP privacy accounting for DP-SGD.

Implements per-sample privacy accounting via the stochastic f-MIP framework
(Leemann et al. NeurIPS 2023; Feldman & Zrnic NeurIPS 2021).

Key insight: at step t, if sample i was NOT in the batch its privacy loss is
exactly zero (identity).  If it WAS in the batch, the privacy loss is that of
the **base Gaussian mechanism** at its observed sensitivity — no Poisson
subsampling mixture needed.  We only compose over the ~q·T steps where the
sample actually participated.

The f-MIP epsilon is found by constructing the mixture PLD — the weighted
average of per-sample PLDs — and computing epsilon on it:

    mixture_pld = (1/N) Σ_i PLD_i
    ε = mixture_pld.epsilon_at(δ)

This is equivalent to finding the smallest ε such that
(1/N) Σ_i H_ε(PLD_i) ≤ δ, since hockey-stick is linear in the PLD.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import opaque_accounting as acc
from opaque_accounting.base import DpProcess, Pld
from opaque_accounting.mechanisms.mip_gaussian import MipGaussian


def _build_mixture_pld(
    processes_and_weights: list[tuple[DpProcess, int]],
    total_samples: int,
) -> Pld:
    """Build the mixture PLD: (1/N) Σ_i w_i · PLD_i."""
    mixture: Pld | None = None
    for process, weight in processes_and_weights:
        pld = process.pld()
        scaled = pld / (total_samples / weight)
        mixture = scaled if mixture is None else mixture + scaled
    return mixture


def per_sample_expost_epsilon(
    noise_multiplier: float,
    sample_step_sensitivities: Sequence[Sequence[float]],
    delta: float,
) -> float:
    """Stochastic f-MIP epsilon via per-sample ex-post composition.

    For each sample, composes the base Gaussian mechanism only over the steps
    where that sample was included in the batch (using the observed gradient
    norm at each such step).  Then constructs the mixture PLD (weighted average
    of per-sample PLDs) and computes epsilon on it.

    Args:
        noise_multiplier: Noise multiplier σ/C.
        sample_step_sensitivities: For each sample, a list of sensitivities
            at the steps where it was included in the batch.  The length of
            each inner list equals the number of participations for that
            sample (NOT the total number of training steps).
        delta: Target delta for epsilon computation.

    Returns:
        f-MIP epsilon (ε such that avg hockey-stick ≤ δ).

    Example::

        # Sample 0 was in 3 batches with these norms; sample 1 in 5 batches
        participations = [
            [0.3, 0.4, 0.35],
            [0.9, 0.95, 0.88, 0.91, 0.87],
        ]
        eps = per_sample_expost_epsilon(0.8, participations, delta=1e-5)
    """
    if not sample_step_sensitivities:
        return 0.0

    n = len(sample_step_sensitivities)

    # Group samples by their sensitivity profile for dedup
    profile_counts: Counter[tuple[float, ...]] = Counter()
    for sensitivities in sample_step_sensitivities:
        key = tuple(round(s, 6) for s in sensitivities)
        profile_counts[key] += 1

    processes: list[tuple[DpProcess, int]] = []
    for profile, count in profile_counts.items():
        if not profile:
            processes.append((acc.identity(), count))
            continue

        sens_counts: Counter[float] = Counter(profile)
        process: DpProcess = acc.identity()
        for s, k in sens_counts.items():
            s_clamped = max(s, 1e-8)
            mip = MipGaussian(
                noise_multiplier=noise_multiplier,
                sensitivities=(s_clamped,),
                weights=(1.0,),
            )
            process = process | (mip * k)
        processes.append((process, count))

    return _build_mixture_pld(processes, n).epsilon_at(delta)


def per_sample_expost_epsilon_fixed(
    noise_multiplier: float,
    sample_sensitivities: Sequence[float],
    num_participations: int | Sequence[int],
    delta: float,
) -> float:
    """Stochastic f-MIP epsilon with fixed per-sample sensitivity.

    Simplified variant: each sample has a fixed sensitivity and a known number
    of batch participations (either the same for all or specified per sample).
    Finds ε such that (1/N) Σ_i H_ε(composed_PLD_i) ≤ δ.

    Args:
        noise_multiplier: Noise multiplier σ/C.
        sample_sensitivities: Per-sample normalized sensitivities (s_i = ||g_i||/C).
        num_participations: Number of times each sample was in a batch.
            If a single int, the same count is used for all samples.
            If a sequence, must have the same length as sample_sensitivities.
        delta: Target delta for epsilon computation.

    Returns:
        f-MIP epsilon (ε such that avg hockey-stick ≤ δ).

    Example::

        sensitivities = [0.2, 0.5, 0.8, 1.0]
        # Each sample appeared in ~50 batches (q*T)
        eps = per_sample_expost_epsilon_fixed(
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

    # Group by (sensitivity, count) to avoid redundant PLD computation.
    key_counts: Counter[tuple[float, int]] = Counter()
    for s, c in zip(sample_sensitivities, counts):
        key_counts[(round(s, 6), c)] += 1

    processes: list[tuple[DpProcess, int]] = []
    for (s, c), num_samples in key_counts.items():
        if c == 0:
            processes.append((acc.identity(), num_samples))
            continue
        s_clamped = max(s, 1e-8)
        mip = MipGaussian(
            noise_multiplier=noise_multiplier,
            sensitivities=(s_clamped,),
            weights=(1.0,),
        )
        process = mip * c
        processes.append((process, num_samples))

    return _build_mixture_pld(processes, n).epsilon_at(delta)
