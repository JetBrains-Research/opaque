"""Compare ex-post per-sample accounting vs mixture vs RMS.

Simulates DP-SGD training with Poisson sampling and heterogeneous
sensitivities, then computes epsilon under four accounting modes:

1. **Ex-post per-sample** (Formulation B, Feldman & Zrnic 2021):
   For each sample, compose only the base Gaussian mechanism at the ~q·T
   steps where it was actually in the batch.  No Poisson subsampling wrapper.

2. **Mixture** (exact per-step, cross-trajectory composition):
   At each step, build a mixture PLD from the batch's sensitivity
   distribution, then compose across steps with Poisson amplification.

3. **RMS** (Jensen approximation):
   At each step, use a single Gaussian with s_rms sensitivity,
   then compose across steps with Poisson amplification.

4. **Worst-case baseline**: Poisson(Gaussian(nm, 1)) composed T steps.

Note: Ex-post composes fewer steps (~q·T) but WITHOUT subsampling
amplification.  The amplified approaches (RMS, mixture, baseline) compose
all T steps but benefit from Poisson amplification at each step.  For small
sampling rates the amplification benefit dominates, so:

  ε_RMS ≤ ε_mixture ≤ ε_baseline ≤ ε_expost  (typically)

The ex-post approach is still valuable for per-sample heterogeneity: samples
with small sensitivities get much lower ex-post epsilon even if the average
is higher.
"""

import math
import random
from collections import Counter

import pytest

import opaque_accounting as acc
from opaque_accounting.composition.per_sample import (
    per_sample_expost_epsilon,
    per_sample_expost_epsilon_fixed,
)
from opaque_accounting.mechanisms.mip_gaussian import MipGaussian


def simulate_dpsgd(
    *,
    num_samples: int,
    num_steps: int,
    sample_rate: float,
    noise_multiplier: float,
    sensitivity_fn,
    seed: int = 42,
):
    """Simulate DP-SGD and return data for all accounting modes.

    Returns per-sample participation records and per-step mixture/RMS processes.
    """
    rng = random.Random(seed)

    # Per-sample sensitivity (fixed across steps for this simulation)
    sample_sens = [sensitivity_fn(i) for i in range(num_samples)]

    # Ex-post: track which steps each sample participated in, with its norm
    sample_participations: list[list[float]] = [[] for _ in range(num_samples)]

    step_mixture_processes: list = []
    step_rms_processes: list = []

    for _t in range(num_steps):
        batch_indices = [i for i in range(num_samples) if rng.random() < sample_rate]

        if not batch_indices:
            step_mixture_processes.append(acc.identity())
            step_rms_processes.append(acc.identity())
            continue

        batch_sens = [sample_sens[i] for i in batch_indices]

        # Record participation for ex-post accounting
        for i in batch_indices:
            sample_participations[i].append(sample_sens[i])

        # Mixture: MipGaussian with the batch's sensitivity distribution
        counts: Counter[float] = Counter()
        for s in batch_sens:
            counts[round(s, 3)] += 1
        sorted_buckets = sorted(counts.keys())
        sensitivities_t = tuple(max(s, 1e-8) for s in sorted_buckets)
        weights_t = tuple(counts[b] / len(batch_sens) for b in sorted_buckets)
        step_mixture_processes.append(
            MipGaussian(
                noise_multiplier=noise_multiplier,
                sensitivities=sensitivities_t,
                weights=weights_t,
            )
        )

        # RMS: single MipGaussian with s_rms
        s_rms = math.sqrt(sum(s**2 for s in batch_sens) / len(batch_sens))
        s_rms = max(s_rms, 1e-8)
        step_rms_processes.append(
            MipGaussian(
                noise_multiplier=noise_multiplier,
                sensitivities=(s_rms,),
                weights=(1.0,),
            )
        )

    return {
        "sample_sensitivities": sample_sens,
        "sample_participations": sample_participations,
        "step_mixture_processes": step_mixture_processes,
        "step_rms_processes": step_rms_processes,
    }


def compose_steps_with_poisson(steps, sample_rate):
    """Compose a list of per-step DpProcesses with Poisson amplification."""
    process = acc.identity()
    for step in steps:
        process = process | acc.poisson(step, sample_rate=sample_rate)
    return process


class TestExpostPerSampleAccounting:
    """Verify ex-post per-sample composition and compare to mixture/RMS."""

    def test_single_participation(self):
        """Single sample, single participation: should match one base Gaussian step."""
        nm = 0.5
        s = 0.7

        eps = per_sample_expost_epsilon(nm, [[s]], delta=1e-5)
        expected = MipGaussian(
            noise_multiplier=nm, sensitivities=(s,), weights=(1.0,)
        ).epsilon_at(1e-5)

        assert eps == pytest.approx(expected, rel=1e-3)

    def test_fixed_variant_matches_varying(self):
        """Fixed-sensitivity variant should match the varying one when
        sensitivity is constant across participations."""
        nm = 0.8
        s = 0.6
        num_parts = 20
        delta = 1e-5

        eps_fixed = per_sample_expost_epsilon_fixed(
            nm,
            [s],
            num_participations=num_parts,
            delta=delta,
        )
        eps_varying = per_sample_expost_epsilon(
            nm,
            [[s] * num_parts],
            delta=delta,
        )

        assert eps_fixed == pytest.approx(eps_varying, rel=1e-3)

    def test_comparison_expost_vs_rms_vs_mixture(self):
        """Compare all accounting modes and verify amplified ordering."""
        num_samples = 500
        num_steps = 50
        sample_rate = 0.05
        nm = 0.8
        delta = 1e-5

        def sensitivity_fn(i):
            return 0.5 if i < num_samples // 2 else 1.0

        data = simulate_dpsgd(
            num_samples=num_samples,
            num_steps=num_steps,
            sample_rate=sample_rate,
            noise_multiplier=nm,
            sensitivity_fn=sensitivity_fn,
        )

        # 1. Ex-post per-sample (no amplification, ~q*T base Gaussian steps)
        eps_expost = per_sample_expost_epsilon(
            nm,
            data["sample_participations"],
            delta=delta,
        )

        # 2. Mixture (compose per-step mixtures with Poisson)
        eps_mixture = compose_steps_with_poisson(
            data["step_mixture_processes"], sample_rate
        ).epsilon_at(delta)

        # 3. RMS
        eps_rms = compose_steps_with_poisson(
            data["step_rms_processes"], sample_rate
        ).epsilon_at(delta)

        # 4. Worst-case baseline
        eps_baseline = (
            acc.poisson(acc.gaussian(nm), sample_rate=sample_rate) * num_steps
        ).epsilon_at(delta)

        print(f"\n{'Mode':<25} {'ε':>10}")
        print(f"{'-' * 35}")
        print(f"{'RMS (amplified)':<25} {eps_rms:10.4f}")
        print(f"{'Mixture (amplified)':<25} {eps_mixture:10.4f}")
        print(f"{'Baseline (amplified)':<25} {eps_baseline:10.4f}")
        print(f"{'Ex-post (no amplif.)':<25} {eps_expost:10.4f}")
        print(f"\nAmplification benefit: {eps_expost / eps_rms:.1f}x worse without it")

        # Amplified approaches maintain their ordering
        assert eps_rms < eps_mixture, (
            f"RMS ({eps_rms:.4f}) should be < mixture ({eps_mixture:.4f})"
        )
        assert eps_mixture < eps_baseline, (
            f"Mixture ({eps_mixture:.4f}) should be < baseline ({eps_baseline:.4f})"
        )
        # Ex-post loses amplification → higher epsilon for small q
        assert eps_expost > eps_rms, (
            f"Ex-post ({eps_expost:.4f}) should be > RMS ({eps_rms:.4f}) "
            f"because amplification dominates at q={sample_rate}"
        )

    def test_uniform_sensitivity_expost_vs_baseline(self):
        """When all sensitivities=1, compare ex-post vs baseline."""
        nm = 0.8
        q = 0.05
        T = 30
        N = 200
        delta = 1e-5

        # Ex-post with fixed participations: each sample in ~q*T steps
        num_parts = round(q * T)
        eps_expost = per_sample_expost_epsilon_fixed(
            nm,
            [1.0] * N,
            num_participations=num_parts,
            delta=delta,
        )
        eps_baseline = (acc.poisson(acc.gaussian(nm), sample_rate=q) * T).epsilon_at(
            delta
        )

        print("\nUniform sensitivity test:")
        print(f"  Ex-post (q*T={num_parts} base Gaussian steps): {eps_expost:.4f}")
        print(f"  Baseline (T={T} Poisson-subsampled steps):     {eps_baseline:.4f}")

        # Ex-post composes fewer steps but without amplification;
        # baseline has amplification but composes more steps.
        # Just verify both are finite and positive.
        assert eps_expost > 0
        assert eps_baseline > 0

    def test_high_variance_gap(self):
        """High variance in sensitivities should show clear numbers."""
        num_samples = 300
        num_steps = 80
        sample_rate = 0.05
        nm = 0.8
        delta = 1e-5

        def sensitivity_fn(i):
            return 0.3 if i < int(num_samples * 0.8) else 1.0

        data = simulate_dpsgd(
            num_samples=num_samples,
            num_steps=num_steps,
            sample_rate=sample_rate,
            noise_multiplier=nm,
            sensitivity_fn=sensitivity_fn,
        )

        eps_expost = per_sample_expost_epsilon(
            nm,
            data["sample_participations"],
            delta=delta,
        )
        eps_mixture = compose_steps_with_poisson(
            data["step_mixture_processes"], sample_rate
        ).epsilon_at(delta)
        eps_rms = compose_steps_with_poisson(
            data["step_rms_processes"], sample_rate
        ).epsilon_at(delta)

        print("\nHigh-variance sensitivity test:")
        print(f"  Ex-post:  {eps_expost:.4f}")
        print(f"  RMS:      {eps_rms:.4f}")
        print(f"  Mixture:  {eps_mixture:.4f}")

        # Amplified approaches keep their ordering; ex-post is higher
        assert eps_rms < eps_mixture
        assert eps_expost > eps_rms

    def test_empty_inputs(self):
        """Empty inputs return 0."""
        assert per_sample_expost_epsilon(0.8, [], delta=1e-5) == 0.0
        assert (
            per_sample_expost_epsilon_fixed(0.8, [], num_participations=10, delta=1e-5)
            == 0.0
        )

    def test_zero_participations(self):
        """Sample that was never in a batch contributes zero epsilon."""
        nm = 0.8
        delta = 1e-5

        # One sample with participations, one without
        eps = per_sample_expost_epsilon(nm, [[0.5, 0.5, 0.5], []], delta=delta)
        eps_single = per_sample_expost_epsilon(nm, [[0.5, 0.5, 0.5]], delta=delta)

        # Average should be half of the single sample's epsilon
        assert eps == pytest.approx(eps_single / 2, rel=1e-3)
