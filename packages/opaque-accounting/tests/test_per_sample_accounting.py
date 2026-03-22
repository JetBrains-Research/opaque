"""Compare per-sample trajectory accounting vs mixture vs RMS.

Simulates DP-SGD training with Poisson sampling and heterogeneous
sensitivities, then computes epsilon under three accounting modes:

1. **Per-sample trajectory** (exact multi-step stochastic f-MIP):
   For each sample with known sensitivity s_i, compose T steps of
   PoissonSubsample(Gaussian(nm, s_i), q), then average epsilon.

2. **Mixture** (exact per-step, cross-trajectory composition):
   At each step, build a mixture PLD from the batch's sensitivity
   distribution, then compose across steps.

3. **RMS** (Jensen approximation):
   At each step, use a single Gaussian with s_rms sensitivity,
   then compose across steps.

Expected ordering: ε_RMS ≤ ε_per_sample ≤ ε_mixture ≤ ε_baseline
"""

import math
import random
from collections import Counter

import pytest

import opaque_accounting as acc
from opaque_accounting.composition.per_sample import per_sample_composed_epsilon
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
    """Simulate DP-SGD and return data for all three accounting modes.

    Returns per-sample sensitivities and per-step mixture/RMS processes.
    """
    rng = random.Random(seed)

    # Per-sample sensitivity (fixed across steps)
    sample_sens = [sensitivity_fn(i) for i in range(num_samples)]

    step_mixture_processes: list = []
    step_rms_processes: list = []

    for _t in range(num_steps):
        batch_indices = [i for i in range(num_samples) if rng.random() < sample_rate]

        if not batch_indices:
            step_mixture_processes.append(acc.identity())
            step_rms_processes.append(acc.identity())
            continue

        batch_sens = [sample_sens[i] for i in batch_indices]

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
        "step_mixture_processes": step_mixture_processes,
        "step_rms_processes": step_rms_processes,
    }


def compose_steps_with_poisson(steps, sample_rate):
    """Compose a list of per-step DpProcesses with Poisson amplification."""
    process = acc.identity()
    for step in steps:
        process = process | acc.poisson(step, sample_rate=sample_rate)
    return process


class TestPerSampleAccounting:
    """Verify per-sample trajectory composition and compare to mixture/RMS."""

    def test_single_sensitivity(self):
        """Single sample: per-sample should match Poisson(MipGaussian) * T."""
        nm = 0.5
        s = 0.7
        q = 0.05
        T = 10

        eps = per_sample_composed_epsilon(nm, q, [s], num_steps=T, delta=1e-5)
        expected = (
            acc.poisson(
                MipGaussian(noise_multiplier=nm, sensitivities=(s,), weights=(1.0,)),
                sample_rate=q,
            )
            * T
        ).epsilon_at(1e-5)

        assert eps == pytest.approx(expected, rel=1e-3)

    def test_ordering_per_sample_vs_mixture_vs_rms(self):
        """Core test: verify ε_RMS ≤ ε_per_sample ≤ ε_mixture ≤ ε_baseline."""
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

        # 1. Per-sample trajectory (exact)
        eps_per_sample = per_sample_composed_epsilon(
            nm,
            sample_rate,
            data["sample_sensitivities"],
            num_steps=num_steps,
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
        print(f"{'RMS':<25} {eps_rms:10.4f}")
        print(f"{'Per-sample trajectory':<25} {eps_per_sample:10.4f}")
        print(f"{'Mixture':<25} {eps_mixture:10.4f}")
        print(f"{'Worst-case baseline':<25} {eps_baseline:10.4f}")

        assert eps_rms < eps_per_sample, (
            f"RMS ({eps_rms:.4f}) should be < per-sample ({eps_per_sample:.4f})"
        )
        assert eps_per_sample < eps_mixture, (
            f"Per-sample ({eps_per_sample:.4f}) should be < mixture ({eps_mixture:.4f})"
        )
        assert eps_mixture < eps_baseline, (
            f"Mixture ({eps_mixture:.4f}) should be < baseline ({eps_baseline:.4f})"
        )

    def test_uniform_sensitivity_all_match(self):
        """When all sensitivities=1, per-sample should match baseline."""
        nm = 0.8
        q = 0.05
        T = 30
        N = 200

        eps_per_sample = per_sample_composed_epsilon(
            nm, q, [1.0] * N, num_steps=T, delta=1e-5
        )
        eps_baseline = (acc.poisson(acc.gaussian(nm), sample_rate=q) * T).epsilon_at(
            1e-5
        )

        assert eps_per_sample == pytest.approx(eps_baseline, rel=0.05)

    def test_high_variance_maximizes_gap(self):
        """High variance in sensitivities should create a large gap
        between per-sample and mixture.
        """
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

        eps_per_sample = per_sample_composed_epsilon(
            nm,
            sample_rate,
            data["sample_sensitivities"],
            num_steps=num_steps,
            delta=delta,
        )
        eps_mixture = compose_steps_with_poisson(
            data["step_mixture_processes"], sample_rate
        ).epsilon_at(delta)
        eps_rms = compose_steps_with_poisson(
            data["step_rms_processes"], sample_rate
        ).epsilon_at(delta)

        print("\nHigh-variance sensitivity test:")
        print(f"  RMS:        {eps_rms:.4f}")
        print(f"  Per-sample: {eps_per_sample:.4f}")
        print(f"  Mixture:    {eps_mixture:.4f}")
        print(f"  Gap (mixture - per_sample): {eps_mixture - eps_per_sample:.4f}")

        assert eps_rms < eps_per_sample < eps_mixture
