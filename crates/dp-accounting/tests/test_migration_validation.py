"""Cross-validation tests ported from jbr-fed-privacy/packages/dp-accounting/.

These tests reproduce the test cases from the old Python binding (jbr.fed.accounting)
against the new Rust-based functional API (opaque_dp_accounting). The goal is to
verify that numbers match, ensuring nothing degraded during the migration.

The old API used PLDAccountant / EventAccountant classes. The new API uses a
functional style:  dp.poisson(nm, q) * steps  ->  .epsilon_at(delta).

Test categories:
  1. Gaussian mechanism -- detailed reference values
  2. Poisson subsampling -- 13 parameter combinations
  3. Truncated Poisson -- realistic workflows
  4. Composition sweep -- parameter sets vs dp_accounting
  5. Delta queries -- including high-epsilon regime
  6. Metrics -- beta, advantage, risk, monotonicity
  7. Calibration
  8. Numerical stability & regressions
  9. Composition properties
  10. Realistic end-to-end workflows
"""

import math

import pytest

import opaque_dp_accounting as dp

# Google's dp_accounting library -- skip tests if not installed.
pld_lib = pytest.importorskip(
    "dp_accounting.pld.privacy_loss_distribution",
    reason="dp_accounting not installed",
)

# riskcal library -- skip riskcal-specific tests if not installed.
riskcal = pytest.importorskip("riskcal", reason="riskcal not installed")
from riskcal.accountants import CTDAccountant as RiskcalAccountant
from riskcal.calibration.dpsgd import create_dpsgd_evaluator


# =============================================================================
# Helpers
# =============================================================================


def google_gaussian_pld(noise_multiplier, num_steps=1):
    """Create a PLD using Google's library for a Gaussian mechanism."""
    pld = pld_lib.from_gaussian_mechanism(
        standard_deviation=noise_multiplier,
        sensitivity=1.0,
        log_mass_truncation_bound=math.log(2) * (-32.0),
        value_discretization_interval=1e-4,
    )
    if num_steps > 1:
        pld = pld.self_compose(num_steps)
    return pld


def google_poisson_gaussian_pld(noise_multiplier, sample_rate, num_steps=1):
    """Create a PLD for Poisson-subsampled Gaussian using Google's library."""
    pld = pld_lib.from_gaussian_mechanism(
        standard_deviation=noise_multiplier,
        sensitivity=1.0,
        sampling_prob=sample_rate,
        log_mass_truncation_bound=math.log(2) * (-32.0),
        value_discretization_interval=1e-4,
        use_connect_dots=True,
    )
    if num_steps > 1:
        pld = pld.self_compose(num_steps)
    return pld


def rel_error(ours, ref):
    """Relative error, guarding against division by zero."""
    if ref == 0:
        return abs(ours)
    return abs(ours - ref) / abs(ref)


# =============================================================================
# 1. Gaussian mechanism -- detailed reference values
# =============================================================================

# Known reference deltas for sigma=1.0, sensitivity=1.0 from analytical Gaussian tail.
GAUSSIAN_REFERENCE_DELTAS = {
    # epsilon -> expected delta (from dp_accounting / scipy erfc)
    -2.0: 0.86749642,
    -1.0: 0.67881797,
    0.0: 0.38292492,
    1.0: 0.12693674,
    2.0: 0.020923636,
}


class TestGaussianMechanismDetailed:
    """Detailed Gaussian mechanism tests ported from mechanisms/gaussian/test_validation.py."""

    def test_known_reference_deltas(self):
        """delta_at for sigma=1.0 should match known analytical values."""
        proc = dp.gaussian(1.0)
        for eps, expected_delta in GAUSSIAN_REFERENCE_DELTAS.items():
            our_delta = proc.delta_at(eps)
            assert abs(our_delta - expected_delta) < 1e-4, (
                f"sigma=1.0, eps={eps}: ours={our_delta:.8f}, "
                f"expected={expected_delta:.8f}, "
                f"diff={abs(our_delta - expected_delta):.2e}"
            )

    @pytest.mark.parametrize("noise_multiplier", [0.3, 0.5, 0.8, 1.0])
    def test_delta_matches_dp_accounting(self, noise_multiplier):
        """delta_at various epsilons should match Google's dp_accounting."""
        proc = dp.gaussian(noise_multiplier)
        google_pld = google_gaussian_pld(noise_multiplier)

        for eps in [0.1, 0.5, 1.0, 2.0, 5.0]:
            our_delta = proc.delta_at(eps)
            ref_delta = google_pld.get_delta_for_epsilon(eps)
            assert abs(our_delta - ref_delta) < max(1e-6, 1e-4 * abs(ref_delta)), (
                f"sigma={noise_multiplier}, eps={eps}: ours={our_delta:.10f}, "
                f"ref={ref_delta:.10f}, diff={abs(our_delta - ref_delta):.2e}"
            )

    @pytest.mark.parametrize("noise_multiplier", [0.3, 0.5, 0.8, 1.0])
    def test_epsilon_matches_dp_accounting(self, noise_multiplier):
        """epsilon_at various deltas should match Google's dp_accounting."""
        proc = dp.gaussian(noise_multiplier)
        google_pld = google_gaussian_pld(noise_multiplier)

        for delta in [1e-3, 1e-4, 1e-5, 1e-6]:
            our_eps = proc.epsilon_at(delta)
            ref_eps = google_pld.get_epsilon_for_delta(delta)
            assert abs(our_eps - ref_eps) < max(1e-4, 1e-3 * abs(ref_eps)), (
                f"sigma={noise_multiplier}, delta={delta}: ours={our_eps:.8f}, "
                f"ref={ref_eps:.8f}, diff={abs(our_eps - ref_eps):.2e}"
            )

    def test_delta_monotonically_decreases_with_epsilon(self):
        """For fixed sigma, delta should decrease as epsilon increases."""
        proc = dp.gaussian(1.0)
        epsilons = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0]
        deltas = [proc.delta_at(e) for e in epsilons]
        for i in range(1, len(deltas)):
            assert deltas[i] <= deltas[i - 1] + 1e-15, (
                f"Non-monotonic: delta({epsilons[i-1]})={deltas[i-1]:.8f} "
                f"< delta({epsilons[i]})={deltas[i]:.8f}"
            )

    @pytest.mark.parametrize("noise_multiplier", [0.3, 0.5, 0.8, 1.0])
    def test_low_noise_gives_finite_delta(self, noise_multiplier):
        """Low noise should produce finite deltas in [0, 1]."""
        proc = dp.gaussian(noise_multiplier)
        for eps in [0.1, 0.5, 1.0, 2.0, 5.0]:
            d = proc.delta_at(eps)
            assert math.isfinite(d), (
                f"sigma={noise_multiplier}, eps={eps}: delta={d}"
            )
            assert 0.0 <= d <= 1.0, (
                f"sigma={noise_multiplier}, eps={eps}: delta={d}"
            )

    def test_edge_case_epsilons(self):
        """Extreme epsilon values should still yield valid deltas."""
        proc = dp.gaussian(1.0)
        for eps in [1e-6, 1e-3, 10.0, 20.0]:
            d = proc.delta_at(eps)
            assert math.isfinite(d), f"eps={eps}: delta={d}"
            assert 0.0 <= d <= 1.0, f"eps={eps}: delta={d}"


# =============================================================================
# 2. Poisson subsampling -- detailed validation
# =============================================================================

# 13 test cases covering a range of sigma, q from old tests.
# The new API always uses sensitivity=1, so noise_multiplier = sigma / sensitivity.
POISSON_TEST_CASES = [
    # (noise_multiplier, sample_rate, description)
    (0.3, 0.01, "low-noise small-q"),
    (0.5, 0.01, "mid-noise small-q"),
    (0.8, 0.01, "std-noise small-q"),
    (1.0, 0.01, "high-noise small-q"),
    (1.0, 0.001, "high-noise tiny-q"),
    (1.0, 0.1, "high-noise large-q"),
    (0.5, 0.1, "mid-noise large-q"),
    (0.5, 0.001, "mid-noise tiny-q"),
    (0.8, 0.005, "std-noise mid-q"),
    (1.1, 0.005, "higher-noise mid-q"),
    (1.2, 0.01, "max-noise small-q"),
    (0.3, 0.0001, "low-noise very-tiny-q"),
    (1.0, 1.0, "no-subsampling"),
]


class TestPoissonSubsampling:
    """Poisson subsampling tests ported from mechanisms/subsampling/poisson/test_validation.py."""

    @pytest.mark.parametrize(
        "noise_multiplier,sample_rate,desc",
        POISSON_TEST_CASES,
        ids=[c[2] for c in POISSON_TEST_CASES],
    )
    def test_single_step_delta_matches_dp_accounting(
        self, noise_multiplier, sample_rate, desc
    ):
        """Single Poisson step delta should match dp_accounting."""
        proc = dp.poisson(noise_multiplier, sample_rate)
        google_pld = google_poisson_gaussian_pld(noise_multiplier, sample_rate)

        for eps in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]:
            our_delta = proc.delta_at(eps)
            ref_delta = google_pld.get_delta_for_epsilon(eps)

            # For tiny deltas use absolute tolerance; otherwise relative.
            if abs(ref_delta) < 1e-12:
                assert abs(our_delta) < 1e-8, (
                    f"nm={noise_multiplier}, q={sample_rate}, eps={eps}: "
                    f"ours={our_delta:.2e}, ref={ref_delta:.2e}"
                )
            else:
                assert rel_error(our_delta, ref_delta) < 1e-3, (
                    f"nm={noise_multiplier}, q={sample_rate}, eps={eps}: "
                    f"ours={our_delta:.10f}, ref={ref_delta:.10f}, "
                    f"rel_err={rel_error(our_delta, ref_delta):.2e}"
                )

    @pytest.mark.parametrize(
        "noise_multiplier,sample_rate,desc",
        POISSON_TEST_CASES,
        ids=[c[2] for c in POISSON_TEST_CASES],
    )
    def test_single_step_epsilon_matches_dp_accounting(
        self, noise_multiplier, sample_rate, desc
    ):
        """Single Poisson step epsilon should match dp_accounting."""
        proc = dp.poisson(noise_multiplier, sample_rate)
        google_pld = google_poisson_gaussian_pld(noise_multiplier, sample_rate)

        for delta in [1e-3, 1e-4, 1e-5]:
            our_eps = proc.epsilon_at(delta)
            ref_eps = google_pld.get_epsilon_for_delta(delta)

            tol = max(1e-4, 5e-3 * abs(ref_eps))
            assert abs(our_eps - ref_eps) < tol, (
                f"nm={noise_multiplier}, q={sample_rate}, delta={delta}: "
                f"ours={our_eps:.8f}, ref={ref_eps:.8f}, "
                f"diff={abs(our_eps - ref_eps):.2e}"
            )

    def test_privacy_amplification(self):
        """Poisson subsampling should reduce privacy loss vs base Gaussian."""
        nm = 1.0
        eps = 1.0
        base = dp.gaussian(nm)
        subsampled = dp.poisson(nm, 0.01)

        base_delta = base.delta_at(eps)
        sub_delta = subsampled.delta_at(eps)

        assert sub_delta < base_delta, (
            f"Subsampled delta ({sub_delta}) should be < base delta ({base_delta})"
        )

    def test_no_amplification_when_q_one(self):
        """q=1 (full batch) should give approximately base Gaussian privacy."""
        nm = 0.5
        proc_full = dp.poisson(nm, 1.0)
        base = dp.gaussian(nm)

        delta = 1e-5
        eps_full = proc_full.epsilon_at(delta)
        eps_base = base.epsilon_at(delta)

        # Should be approximately equal (q=1 means no amplification).
        assert abs(eps_full - eps_base) < max(0.1, 0.01 * eps_base), (
            f"q=1: full={eps_full:.6f}, base={eps_base:.6f}"
        )

    @pytest.mark.parametrize(
        "noise_multiplier,sample_rate,num_steps",
        [
            (1.0, 0.01, 100),
            (1.0, 0.01, 1000),
            (1.1, 0.005, 500),
            (0.8, 0.01, 200),
            (0.5, 0.01, 500),
        ],
    )
    def test_composed_poisson_epsilon(self, noise_multiplier, sample_rate, num_steps):
        """Composed Poisson-Gaussian epsilon should match dp_accounting."""
        delta = 1e-5

        our_eps = dp.compute_epsilon(noise_multiplier, sample_rate, num_steps, delta)
        google_pld = google_poisson_gaussian_pld(
            noise_multiplier, sample_rate, num_steps
        )
        ref_eps = google_pld.get_epsilon_for_delta(delta)

        tol = max(1e-3, 0.01 * abs(ref_eps))
        assert abs(our_eps - ref_eps) < tol, (
            f"nm={noise_multiplier}, q={sample_rate}, k={num_steps}: "
            f"ours={our_eps:.6f}, ref={ref_eps:.6f}, "
            f"diff={abs(our_eps - ref_eps):.2e}"
        )

    def test_dpsgd_imagenet_scenario(self):
        """Realistic ImageNet DP-SGD: sigma=1.0, n=1.2M, batch=4096."""
        nm = 1.0
        n = 1_200_000
        batch = 4096
        q = batch / n

        proc = dp.poisson(nm, q)
        google_pld = google_poisson_gaussian_pld(nm, q)

        for eps in [1.0, 2.0, 5.0, 10.0]:
            our_delta = proc.delta_at(eps)
            ref_delta = google_pld.get_delta_for_epsilon(eps)

            if abs(ref_delta) < 1e-11:
                assert abs(our_delta) < 1e-8
            else:
                assert rel_error(our_delta, ref_delta) < 1e-3, (
                    f"ImageNet eps={eps}: ours={our_delta:.2e}, ref={ref_delta:.2e}"
                )


# =============================================================================
# 3. Truncated Poisson subsampling
# =============================================================================

TRUNCATED_TEST_CASES = [
    # (dataset_size, sample_rate, batch_size_cap, noise_multiplier, description)
    (10_000, 0.01, 128, 0.5, "small-dataset low-noise"),
    (10_000, 0.01, 128, 0.8, "small-dataset std-noise"),
    (50_000, 0.005, 256, 0.8, "cifar-scale"),
    (100_000, 0.001, 128, 0.8, "llm-finetune"),
    (1_280_000, 0.0032, 4096, 1.0, "imagenet-scale"),
    (10_000_000, 0.0001, 1024, 1.0, "large-dataset"),
    (100_000_000, 0.00001, 1024, 1.2, "huge-dataset"),
    (10_000, 0.01, 256, 1.0, "cap-above-expected"),
    (60_000, 0.00427, 256, 1.0, "opacus-default"),
]


class TestTruncatedPoisson:
    """Truncated Poisson tests ported from mechanisms/subsampling/truncated/test_validation.py."""

    @pytest.mark.parametrize(
        "n,q,b_max,nm,desc",
        TRUNCATED_TEST_CASES,
        ids=[c[4] for c in TRUNCATED_TEST_CASES],
    )
    def test_truncated_valid_output(self, n, q, b_max, nm, desc):
        """Truncated Poisson should produce valid finite deltas."""
        proc = dp.truncated_poisson(nm, q, batch_size_cap=b_max, dataset_size=n)
        for eps in [0.5, 1.0, 2.0, 5.0, 10.0]:
            d = proc.delta_at(eps)
            assert math.isfinite(d), f"{desc}, eps={eps}: delta={d}"
            assert 0.0 <= d <= 1.0, f"{desc}, eps={eps}: delta={d}"

    def test_truncated_matches_standard_when_no_truncation(self):
        """When B_max >= n (no truncation possible), should match standard Poisson."""
        nm, q, n = 0.8, 0.1, 100
        b_max = 100  # cap = dataset size -> no truncation

        trunc = dp.truncated_poisson(nm, q, batch_size_cap=b_max, dataset_size=n)
        standard = dp.poisson(nm, q)

        delta = 1e-5
        eps_trunc = trunc.epsilon_at(delta)
        eps_std = standard.epsilon_at(delta)

        assert abs(eps_trunc - eps_std) < max(0.01, 0.01 * eps_std), (
            f"No-truncation: trunc={eps_trunc:.6f}, std={eps_std:.6f}"
        )

    def test_truncated_differs_when_truncation_occurs(self):
        """When truncation occurs, truncated analysis may differ from standard."""
        nm, q, n = 0.8, 0.01, 10_000
        b_max = 128  # expected batch ~ 100, some truncation at 128

        trunc = dp.truncated_poisson(nm, q, batch_size_cap=b_max, dataset_size=n)

        delta = 1e-5
        eps_trunc = trunc.epsilon_at(delta)

        # Truncated should give finite valid result.
        assert math.isfinite(eps_trunc), f"eps_trunc={eps_trunc}"
        assert eps_trunc > 0, f"eps_trunc={eps_trunc}"

    def test_cifar10_workflow(self):
        """CIFAR-10: n=50k, batch=250, sigma=0.8, 100 epochs."""
        n = 50_000
        batch = 250
        q = batch / n
        nm = 0.8
        steps_per_epoch = n // batch
        total_steps = 100 * steps_per_epoch

        step = dp.truncated_poisson(nm, q, batch_size_cap=batch, dataset_size=n)
        training = step * total_steps

        eps = training.epsilon_at(1e-5)
        assert math.isfinite(eps), f"CIFAR-10: epsilon={eps}"
        assert eps > 0, f"CIFAR-10: epsilon={eps}"

    def test_imagenet_workflow(self):
        """ImageNet: n=1.28M, batch=4096, sigma=1.0."""
        n = 1_280_000
        batch = 4096
        q = batch / n
        nm = 1.0
        steps = 5 * (n // batch)  # 5 epochs

        step = dp.truncated_poisson(nm, q, batch_size_cap=batch, dataset_size=n)
        training = step * steps

        eps = training.epsilon_at(1e-6)
        assert math.isfinite(eps), f"ImageNet: epsilon={eps}"
        assert eps > 0, f"ImageNet: epsilon={eps}"

    def test_llm_finetuning_workflow(self):
        """LLM fine-tuning: n=100k, batch=128, sigma=0.8."""
        n = 100_000
        batch = 128
        q = batch / n
        nm = 0.8
        steps = 3 * (n // batch)  # 3 epochs

        step = dp.truncated_poisson(nm, q, batch_size_cap=batch, dataset_size=n)
        training = step * steps

        for delta in [1e-5, 1e-6]:
            eps = training.epsilon_at(delta)
            assert math.isfinite(eps), f"LLM: delta={delta}, epsilon={eps}"
            assert eps > 0, f"LLM: delta={delta}, epsilon={eps}"

    def test_very_large_dataset(self):
        """Numerical stability with n=100M, tiny q."""
        n = 100_000_000
        q = 0.00001
        b_max = 1024
        nm = 1.2

        proc = dp.truncated_poisson(nm, q, batch_size_cap=b_max, dataset_size=n)
        d = proc.delta_at(1.0)
        assert math.isfinite(d), f"100M dataset: delta={d}"
        assert 0.0 <= d <= 1.0, f"100M dataset: delta={d}"


# =============================================================================
# 4. Composition sweep -- ported from dual validation
# =============================================================================

VERY_LOW_NOISE_PARAMS = [
    # (sigma, q, steps, description)
    (0.1, 0.001, 20, "sigma=0.1 q=0.001 k=20"),
    (0.1, 0.01, 50, "sigma=0.1 q=0.01 k=50"),
    (0.1, 0.01, 100, "sigma=0.1 q=0.01 k=100"),
    (0.2, 0.01, 100, "sigma=0.2 q=0.01 k=100"),
    (0.2, 0.01, 200, "sigma=0.2 q=0.01 k=200"),
    (0.2, 0.05, 100, "sigma=0.2 q=0.05 k=100"),
    (0.3, 0.01, 200, "sigma=0.3 q=0.01 k=200"),
    (0.3, 0.01, 500, "sigma=0.3 q=0.01 k=500"),
    (0.3, 0.05, 200, "sigma=0.3 q=0.05 k=200"),
    (0.3, 0.05, 500, "sigma=0.3 q=0.05 k=500"),
]

PRODUCTION_NOISE_PARAMS = [
    (0.4, 0.01, 1000, "sigma=0.4 q=0.01 k=1000"),
    (0.4, 0.01, 2000, "sigma=0.4 q=0.01 k=2000"),
    (0.5, 0.01, 1000, "sigma=0.5 q=0.01 k=1000"),
    (0.5, 0.01, 2000, "sigma=0.5 q=0.01 k=2000"),
    (0.5, 0.05, 500, "sigma=0.5 q=0.05 k=500"),
    (0.6, 0.01, 2000, "sigma=0.6 q=0.01 k=2000"),
    (0.6, 0.01, 3000, "sigma=0.6 q=0.01 k=3000"),
    (0.7, 0.01, 3000, "sigma=0.7 q=0.01 k=3000"),
    (0.7, 0.01, 5000, "sigma=0.7 q=0.01 k=5000"),
]

HIGH_NOISE_PARAMS = [
    (0.8, 0.01, 5000, "sigma=0.8 q=0.01 k=5000"),
    (0.8, 0.001, 10000, "sigma=0.8 q=0.001 k=10000"),
    (1.0, 0.01, 5000, "sigma=1.0 q=0.01 k=5000"),
    (1.0, 0.01, 10000, "sigma=1.0 q=0.01 k=10000"),
    (1.0, 0.001, 10000, "sigma=1.0 q=0.001 k=10000"),
    (1.0, 0.001, 25000, "sigma=1.0 q=0.001 k=25000"),
    (1.2, 0.01, 10000, "sigma=1.2 q=0.01 k=10000"),
    (1.2, 0.01, 25000, "sigma=1.2 q=0.01 k=25000"),
]

ALL_COMPOSITION_PARAMS = (
    VERY_LOW_NOISE_PARAMS + PRODUCTION_NOISE_PARAMS + HIGH_NOISE_PARAMS
)


class TestCompositionSweep:
    """Parameter sweep ported from test_dual_validation.py & accountants/test_validation.py."""

    @pytest.mark.parametrize(
        "sigma,q,steps,desc",
        ALL_COMPOSITION_PARAMS,
        ids=[c[3] for c in ALL_COMPOSITION_PARAMS],
    )
    def test_epsilon_matches_dp_accounting(self, sigma, q, steps, desc):
        """Composed Poisson-Gaussian epsilon should match dp_accounting."""
        delta = 1e-5

        our_eps = dp.compute_epsilon(sigma, q, steps, delta)
        google_pld = google_poisson_gaussian_pld(sigma, q, steps)
        ref_eps = google_pld.get_epsilon_for_delta(delta)

        assert math.isfinite(our_eps), f"{desc}: epsilon={our_eps}"
        assert our_eps > 0, f"{desc}: epsilon={our_eps}"

        # Tolerance scales with composition depth.
        if steps > 10000:
            tol_rel = 1e-3
        elif steps > 5000:
            tol_rel = 5e-4
        else:
            tol_rel = 2e-4
        tol = max(1e-3, tol_rel * abs(ref_eps))

        assert abs(our_eps - ref_eps) < tol, (
            f"{desc}: ours={our_eps:.6f}, ref={ref_eps:.6f}, "
            f"rel_err={rel_error(our_eps, ref_eps):.2e}"
        )


# =============================================================================
# 5. Delta queries -- including high-epsilon regime
# =============================================================================


class TestDeltaQueries:
    """Delta query tests ported from accountants/test_validation.py."""

    def test_delta_values_match_dp_accounting(self):
        """Delta at various epsilons should match dp_accounting."""
        sigma, q, steps = 0.5, 0.1, 100
        proc = dp.poisson(sigma, q) * steps
        google_pld = google_poisson_gaussian_pld(sigma, q, steps)

        for eps in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
            our_delta = proc.delta_at(eps)
            ref_delta = google_pld.get_delta_for_epsilon(eps)

            assert 0.0 <= our_delta <= 1.0, (
                f"eps={eps}: delta={our_delta} out of range"
            )

            if abs(ref_delta) < 1e-10:
                assert abs(our_delta) < 1e-6, (
                    f"eps={eps}: ours={our_delta:.2e}, ref~0"
                )
            else:
                assert rel_error(our_delta, ref_delta) < 1e-3, (
                    f"eps={eps}: ours={our_delta:.10f}, ref={ref_delta:.10f}, "
                    f"rel_err={rel_error(our_delta, ref_delta):.2e}"
                )

    @pytest.mark.parametrize("epsilon", [30.0, 60.0, 90.0, 120.0, 150.0, 200.0])
    def test_high_epsilon_delta_queries(self, epsilon):
        """Very high epsilon values should return valid (tiny) deltas."""
        sigma, q, steps = 0.5, 0.01, 1000
        proc = dp.poisson(sigma, q) * steps
        google_pld = google_poisson_gaussian_pld(sigma, q, steps)

        our_delta = proc.delta_at(epsilon)
        ref_delta = google_pld.get_delta_for_epsilon(epsilon)

        assert 0.0 <= our_delta <= 1.0, f"eps={epsilon}: delta={our_delta}"

        if abs(ref_delta) > 1e-10:
            assert rel_error(our_delta, ref_delta) < 0.01, (
                f"eps={epsilon}: ours={our_delta:.2e}, ref={ref_delta:.2e}"
            )
        else:
            # Both should be very small.
            assert our_delta < 1e-8, f"eps={epsilon}: delta={our_delta}"

    def test_delta_in_valid_range_for_all_epsilons(self):
        """Delta should be in [0, 1] for a sweep of epsilons including negative."""
        proc = dp.poisson(0.5, 0.1) * 100
        for eps in [-10.0, -1.0, 0.0, 1.0, 5.0, 10.0, 50.0, 100.0]:
            d = proc.delta_at(eps)
            assert 0.0 <= d <= 1.0, f"eps={eps}: delta={d}"

    def test_boundary_epsilon_queries(self):
        """Epsilon at various deltas should be finite and monotonically decreasing."""
        proc = dp.poisson(0.5, 0.1) * 100
        deltas = [1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5]
        epsilons = []
        for d in deltas:
            eps = proc.epsilon_at(d)
            assert math.isfinite(eps), f"delta={d}: epsilon={eps}"
            assert eps >= 0, f"delta={d}: epsilon={eps}"
            epsilons.append(eps)

        # Epsilon should decrease as delta increases.
        for i in range(1, len(epsilons)):
            assert epsilons[i] <= epsilons[i - 1] + 1e-10, (
                f"Non-monotonic: eps(delta={deltas[i-1]})={epsilons[i-1]:.6f} "
                f"< eps(delta={deltas[i]})={epsilons[i]:.6f}"
            )


# =============================================================================
# 6. Metrics -- beta, advantage, risk, monotonicity
# =============================================================================


class TestMetricsBeta:
    """Beta metric tests ported from test_dual_validation.py & calibration/test_validation.py."""

    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1, 0.2, 0.5])
    def test_beta_in_valid_range(self, alpha):
        """Beta should be in [0, 1]."""
        proc = dp.poisson(0.5, 0.1) * 100
        beta = proc.beta_at(alpha)
        assert 0.0 <= beta <= 1.0, f"alpha={alpha}: beta={beta}"

    def test_beta_monotonic_across_alpha(self):
        """Beta should be non-increasing as alpha increases."""
        proc = dp.poisson(0.8, 0.01) * 500
        alphas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
        betas = [proc.beta_at(a) for a in alphas]
        for i in range(1, len(betas)):
            assert betas[i] <= betas[i - 1] + 1e-10, (
                f"Non-monotonic: beta(alpha={alphas[i-1]})={betas[i-1]:.8f} "
                f"< beta(alpha={alphas[i]})={betas[i]:.8f}"
            )

    @pytest.mark.parametrize("sigma", [0.3, 0.5, 0.8, 1.0])
    def test_beta_monotonic_across_alpha_different_sigma(self, sigma):
        """Beta monotonicity should hold for various noise levels."""
        proc = dp.poisson(sigma, 0.01) * 200
        alphas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
        betas = [proc.beta_at(a) for a in alphas]
        for i in range(1, len(betas)):
            assert betas[i] <= betas[i - 1] + 1e-10, (
                f"sigma={sigma}, non-monotonic: "
                f"beta(alpha={alphas[i-1]})={betas[i-1]:.8f} "
                f"< beta(alpha={alphas[i]})={betas[i]:.8f}"
            )

    @pytest.mark.parametrize("alpha", [0.80, 0.90, 0.95, 0.99])
    def test_beta_high_alpha_valid(self, alpha):
        """Beta should be valid even for high alpha values."""
        proc = dp.poisson(0.8, 0.01) * 100
        beta = proc.beta_at(alpha)
        assert 0.0 <= beta <= 1.0, f"alpha={alpha}: beta={beta}"
        assert math.isfinite(beta), f"alpha={alpha}: beta={beta}"


class TestMetricsAdvantage:
    """Advantage metric tests."""

    def test_advantage_in_valid_range(self):
        """Advantage should be in [0, 1]."""
        proc = dp.poisson(0.5, 0.1) * 100
        adv = proc.advantage()
        assert 0.0 <= adv <= 1.0, f"advantage={adv}"
        assert adv > 0, "advantage should be > 0 for non-trivial mechanism"

    def test_advantage_equals_delta_at_zero(self):
        """Advantage should equal delta_at(0)."""
        proc = dp.poisson(0.8, 0.01) * 500
        adv = proc.advantage()
        delta_0 = proc.delta_at(0.0)
        assert abs(adv - delta_0) < 1e-6, (
            f"advantage={adv}, delta_at(0)={delta_0}"
        )

    def test_advantage_decreases_with_noise(self):
        """More noise -> lower advantage (more private)."""
        advantages = []
        for sigma in [0.3, 0.5, 0.8, 1.0]:
            proc = dp.poisson(sigma, 0.01) * 500
            advantages.append(proc.advantage())

        for i in range(1, len(advantages)):
            assert advantages[i] <= advantages[i - 1] + 1e-10, (
                f"Advantage should decrease: sigma-sequence gave {advantages}"
            )


class TestMetricsRisk:
    """Bayes risk metric tests ported from pld/pld/test_validation.py."""

    def test_risk_basic(self):
        """Risk should be in [0, 0.5] for uniform prior."""
        proc = dp.gaussian(0.5)
        risk = proc.risk_at(0.5)
        assert 0.0 <= risk <= 0.5, f"risk={risk}"

    @pytest.mark.parametrize(
        "prior", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )
    def test_risk_bounds(self, prior):
        """Risk should be in [0, min(prior, 1-prior)]."""
        proc = dp.gaussian(0.5)
        risk = proc.risk_at(prior)
        upper = min(prior, 1.0 - prior)
        assert 0.0 <= risk <= upper + 1e-10, (
            f"prior={prior}: risk={risk}, bound={upper}"
        )

    def test_risk_symmetry_gaussian(self):
        """Gaussian PLD is symmetric, so risk(p) should equal risk(1-p)."""
        proc = dp.gaussian(0.5)
        for p1, p2 in [(0.2, 0.8), (0.3, 0.7), (0.4, 0.6)]:
            risk1 = proc.risk_at(p1)
            risk2 = proc.risk_at(p2)
            assert abs(risk1 - risk2) < 1e-4, (
                f"risk({p1})={risk1:.8f}, risk({p2})={risk2:.8f}"
            )

    def test_risk_monotonicity_with_noise(self):
        """Higher noise -> higher risk (more private, harder to distinguish)."""
        risk_low = dp.gaussian(0.3).risk_at(0.5)
        risk_high = dp.gaussian(0.8).risk_at(0.5)
        assert risk_high > risk_low, (
            f"risk(sigma=0.3)={risk_low}, risk(sigma=0.8)={risk_high}"
        )

    @pytest.mark.parametrize("sigma", [0.3, 0.5, 0.8])
    def test_risk_finite_and_valid(self, sigma):
        """Risk values should be finite and in valid range."""
        proc = dp.gaussian(sigma)
        for prior in [0.3, 0.5, 0.7]:
            our_risk = proc.risk_at(prior)
            assert math.isfinite(our_risk), (
                f"sigma={sigma}, prior={prior}: risk={our_risk}"
            )
            assert 0.0 <= our_risk <= 0.5, (
                f"sigma={sigma}, prior={prior}: risk={our_risk}"
            )


# =============================================================================
# 7. Calibration
# =============================================================================


class TestCalibration:
    """Calibration tests ported from calibration/test_validation.py."""

    def test_calibrate_basic(self):
        """Calibrated noise should achieve approximately the target epsilon."""
        target_eps = 8.0
        delta = 1e-5
        q = 0.01
        steps = 1000

        nm = dp.calibrate_noise(
            target_epsilon=target_eps,
            target_delta=delta,
            sample_rate=q,
            num_steps=steps,
        )

        actual_eps = dp.compute_epsilon(nm, q, steps, delta)
        assert abs(actual_eps - target_eps) < 0.1, (
            f"Calibrated nm={nm:.4f} gives eps={actual_eps:.6f}, target={target_eps}"
        )

    def test_calibrate_high_privacy(self):
        """Calibration for strict (low eps) privacy budget."""
        target_eps = 3.0
        delta = 1e-5
        q = 0.01
        steps = 1000

        nm = dp.calibrate_noise(
            target_epsilon=target_eps,
            target_delta=delta,
            sample_rate=q,
            num_steps=steps,
        )

        actual_eps = dp.compute_epsilon(nm, q, steps, delta)
        assert abs(actual_eps - target_eps) < 0.5, (
            f"High privacy: nm={nm:.4f}, eps={actual_eps:.6f}, target={target_eps}"
        )

    def test_calibrate_low_privacy(self):
        """Calibration for loose (high eps) privacy budget."""
        target_eps = 15.0
        delta = 1e-5
        q = 0.01
        steps = 1000

        nm = dp.calibrate_noise(
            target_epsilon=target_eps,
            target_delta=delta,
            sample_rate=q,
            num_steps=steps,
        )

        actual_eps = dp.compute_epsilon(nm, q, steps, delta)
        assert abs(actual_eps - target_eps) < 1.0, (
            f"Low privacy: nm={nm:.4f}, eps={actual_eps:.6f}, target={target_eps}"
        )

    def test_calibrate_monotonicity(self):
        """Stricter epsilon target should require more noise."""
        delta = 1e-5
        q = 0.01
        steps = 1000

        nm_loose = dp.calibrate_noise(
            target_epsilon=12.0, target_delta=delta,
            sample_rate=q, num_steps=steps,
        )
        nm_strict = dp.calibrate_noise(
            target_epsilon=6.0, target_delta=delta,
            sample_rate=q, num_steps=steps,
        )

        assert nm_strict > nm_loose, (
            f"Strict target should need more noise: "
            f"nm_strict={nm_strict:.4f}, nm_loose={nm_loose:.4f}"
        )

    @pytest.mark.parametrize(
        "batch_size,dataset_size",
        [(8, 10000), (16, 10000), (32, 10000)],
    )
    def test_calibrate_different_batch_sizes(self, batch_size, dataset_size):
        """Calibration should converge for different batch sizes."""
        q = batch_size / dataset_size
        steps = 5000
        target_eps = 10.0
        delta = 1e-4

        nm = dp.calibrate_noise(
            target_epsilon=target_eps,
            target_delta=delta,
            sample_rate=q,
            num_steps=steps,
        )

        actual_eps = dp.compute_epsilon(nm, q, steps, delta)
        assert abs(actual_eps - target_eps) < 1.0, (
            f"batch={batch_size}: nm={nm:.4f}, eps={actual_eps:.6f}"
        )

    def test_calibrate_validates_with_dp_accounting(self):
        """Calibrated noise checked against dp_accounting."""
        q = 0.01
        steps = 1000
        target_eps = 8.0
        delta = 1e-5

        nm = dp.calibrate_noise(
            target_epsilon=target_eps,
            target_delta=delta,
            sample_rate=q,
            num_steps=steps,
        )

        # Verify using dp_accounting.
        google_pld = google_poisson_gaussian_pld(nm, q, steps)
        eps_dp = google_pld.get_epsilon_for_delta(delta)

        assert eps_dp <= target_eps * 1.1, (
            f"dp_accounting check: nm={nm:.4f}, eps_dp={eps_dp:.6f}, "
            f"target={target_eps}"
        )


# =============================================================================
# 8. Numerical stability & regressions
# =============================================================================


class TestNumericalStability:
    """Numerical stability tests ported from accountants/test_validation.py."""

    @pytest.mark.parametrize("sigma", [0.3, 0.4, 0.5, 0.6, 0.8])
    def test_low_noise_finite_epsilon(self, sigma):
        """Low noise should give finite, positive epsilon (Bug #1 regression)."""
        proc = dp.poisson(sigma, 0.1) * 100
        eps = proc.epsilon_at(1e-5)
        assert math.isfinite(eps), f"sigma={sigma}: epsilon={eps} (should be finite)"
        assert eps > 0, f"sigma={sigma}: epsilon={eps} (should be positive)"

    @pytest.mark.parametrize("sigma", [0.3, 0.4, 0.5, 0.6, 0.8])
    def test_low_noise_matches_dp_accounting(self, sigma):
        """Low noise epsilon should match dp_accounting (regression check)."""
        q, steps = 0.1, 100
        proc = dp.poisson(sigma, q) * steps
        google_pld = google_poisson_gaussian_pld(sigma, q, steps)

        eps_ours = proc.epsilon_at(1e-5)
        eps_ref = google_pld.get_epsilon_for_delta(1e-5)

        assert rel_error(eps_ours, eps_ref) < 1e-3, (
            f"sigma={sigma}: ours={eps_ours:.6f}, ref={eps_ref:.6f}"
        )

    def test_regression_bug1_infinity_mass(self):
        """Bug #1: infinity mass should not cause infinite epsilon."""
        sigma, q = 0.5, 0.1
        for steps in [1, 5, 10, 20, 50, 100]:
            proc = dp.poisson(sigma, q) * steps
            eps = proc.epsilon_at(1e-5)
            assert math.isfinite(eps), (
                f"Infinity mass bug: sigma={sigma}, q={q}, k={steps}, epsilon={eps}"
            )

            google_pld = google_poisson_gaussian_pld(sigma, q, steps)
            ref_eps = google_pld.get_epsilon_for_delta(1e-5)
            assert rel_error(eps, ref_eps) < 1e-3, (
                f"k={steps}: ours={eps:.6f}, ref={ref_eps:.6f}"
            )

    def test_very_small_sampling_rate(self):
        """Very small q should give finite epsilon."""
        proc = dp.poisson(0.8, 0.0001) * 10000
        eps = proc.epsilon_at(1e-5)
        assert math.isfinite(eps), f"tiny q: epsilon={eps}"
        assert eps > 0, f"tiny q: epsilon={eps}"

    def test_near_full_batch(self):
        """q close to 1 should work."""
        proc = dp.poisson(0.5, 0.99) * 10
        eps = proc.epsilon_at(1e-5)
        assert math.isfinite(eps), f"near-full batch: epsilon={eps}"
        assert eps > 0, f"near-full batch: epsilon={eps}"

    def test_many_composition_steps(self):
        """Large k should give finite epsilon."""
        proc = dp.poisson(0.8, 0.001) * 50000
        eps = proc.epsilon_at(1e-5)
        assert math.isfinite(eps), f"50k steps: epsilon={eps}"
        assert eps > 0, f"50k steps: epsilon={eps}"

    def test_composition_growth_bounded(self):
        """Epsilon growth should be sublinear (not exponential)."""
        sigma, q = 0.5, 0.1
        epsilons = []
        for steps in [1, 10, 50, 100, 200]:
            proc = dp.poisson(sigma, q) * steps
            epsilons.append(proc.epsilon_at(1e-5))

        # Growth from 1->200 steps should be << 200x (sublinear).
        growth = epsilons[-1] / epsilons[0]
        assert growth < 100, (
            f"Growth factor 1->200 steps: {growth:.1f}x (should be << 200x)"
        )

    @pytest.mark.parametrize("delta", [1e-10, 1e-12])
    def test_extreme_small_delta(self, delta):
        """Very small delta should give finite epsilon."""
        proc = dp.poisson(1.0, 0.001) * 1000
        eps = proc.epsilon_at(delta)
        assert math.isfinite(eps), f"delta={delta}: epsilon={eps}"
        assert eps > 0, f"delta={delta}: epsilon={eps}"


# =============================================================================
# 9. Composition properties
# =============================================================================


class TestCompositionProperties:
    """Property-based composition tests ported from pld/pld/test_validation.py."""

    def test_epsilon_increases_with_composition(self):
        """More steps -> higher epsilon."""
        q, sigma, delta = 0.01, 0.8, 1e-5
        step_counts = [1, 5, 10, 50, 100]
        epsilons = [dp.compute_epsilon(sigma, q, k, delta) for k in step_counts]

        for i in range(1, len(epsilons)):
            assert epsilons[i] > epsilons[i - 1], (
                f"eps should increase: k={step_counts[i-1]}->{step_counts[i]}, "
                f"eps={epsilons[i-1]:.6f}->{epsilons[i]:.6f}"
            )

    def test_epsilon_decreases_with_noise(self):
        """More noise -> lower epsilon."""
        q, steps, delta = 0.01, 1000, 1e-5
        sigmas = [0.3, 0.5, 0.8, 1.0, 1.2]
        epsilons = [dp.compute_epsilon(s, q, steps, delta) for s in sigmas]

        for i in range(1, len(epsilons)):
            assert epsilons[i] < epsilons[i - 1], (
                f"eps should decrease: sigma={sigmas[i-1]}->{sigmas[i]}, "
                f"eps={epsilons[i-1]:.6f}->{epsilons[i]:.6f}"
            )

    def test_lower_sample_rate_gives_better_privacy(self):
        """Lower q -> lower epsilon (more privacy amplification)."""
        sigma, steps, delta = 0.5, 100, 1e-5
        sample_rates = [0.1, 0.01, 0.001]
        epsilons = [dp.compute_epsilon(sigma, q, steps, delta) for q in sample_rates]

        for i in range(1, len(epsilons)):
            assert epsilons[i] < epsilons[i - 1], (
                f"eps should decrease: q={sample_rates[i-1]}->{sample_rates[i]}, "
                f"eps={epsilons[i-1]:.6f}->{epsilons[i]:.6f}"
            )

    def test_heterogeneous_composition(self):
        """Compose phases with different noise levels."""
        delta = 1e-5

        # Three-phase training: different noise multipliers.
        phase1 = dp.poisson(0.5, 0.1) * 50
        phase2 = dp.poisson(0.8, 0.05) * 30
        phase3 = dp.poisson(1.0, 0.02) * 20

        combined = phase1 | phase2 | phase3
        eps_combined = combined.epsilon_at(delta)

        # Combined should be more than any single phase.
        eps1 = phase1.epsilon_at(delta)
        eps2 = phase2.epsilon_at(delta)
        eps3 = phase3.epsilon_at(delta)

        assert eps_combined > max(eps1, eps2, eps3), (
            f"Combined eps ({eps_combined}) should exceed max single-phase "
            f"({max(eps1, eps2, eps3)})"
        )

        # Cross-validate with dp_accounting.
        g_pld1 = google_poisson_gaussian_pld(0.5, 0.1, 50)
        g_pld2 = google_poisson_gaussian_pld(0.8, 0.05, 30)
        g_pld3 = google_poisson_gaussian_pld(1.0, 0.02, 20)
        g_combined = g_pld1.compose(g_pld2).compose(g_pld3)
        ref_eps = g_combined.get_epsilon_for_delta(delta)

        tol = max(0.01, 0.01 * abs(ref_eps))
        assert abs(eps_combined - ref_eps) < tol, (
            f"Hetero composition: ours={eps_combined:.6f}, ref={ref_eps:.6f}"
        )

    def test_operator_matches_function(self):
        """step * k should equal repeat(step, k)."""
        step = dp.poisson(1.0, 0.01)

        via_op = (step * 100).epsilon_at(1e-5)
        via_fn = dp.repeat(step, 100).epsilon_at(1e-5)

        assert via_op == pytest.approx(via_fn, rel=1e-10)

    def test_compose_operator_matches_function(self):
        """a | b should equal compose(a, b)."""
        a = dp.gaussian(1.0)
        b = dp.gaussian(0.8)

        via_op = (a | b).epsilon_at(1e-5)
        via_fn = dp.compose(a, b).epsilon_at(1e-5)

        assert via_op == pytest.approx(via_fn, rel=1e-10)

    def test_sublinear_composition_growth(self):
        """10x more steps should give < 10x more epsilon."""
        sigma, q, delta = 0.5, 0.01, 1e-5

        eps_1k = dp.compute_epsilon(sigma, q, 1000, delta)
        eps_10k = dp.compute_epsilon(sigma, q, 10000, delta)

        growth = eps_10k / eps_1k
        assert 1.5 < growth < 10.0, (
            f"Growth 1k->10k: {growth:.2f}x (expected sublinear)"
        )

    def test_gaussian_pld_symmetric(self):
        """Gaussian PLD should be symmetric (ADD = REMOVE)."""
        proc = dp.gaussian(0.5)
        info = proc.pld_info()
        assert info["is_symmetric"] is True, "Gaussian PLD should be symmetric"

    def test_subsampled_gaussian_pld_asymmetric(self):
        """Poisson-subsampled Gaussian PLD should be asymmetric."""
        proc = dp.poisson(0.5, 0.01)
        info = proc.pld_info()
        assert info["is_symmetric"] is False, (
            "Poisson-subsampled Gaussian PLD should be asymmetric"
        )


# =============================================================================
# 10. Realistic end-to-end workflows
# =============================================================================


class TestRealisticWorkflows:
    """Realistic training workflows ported from test_dual_validation.py."""

    def test_llm_finetuning(self):
        """LLM fine-tuning: 10k dataset, batch=16, sigma=0.5, 3 epochs."""
        dataset_size = 10_000
        batch_size = 16
        q = batch_size / dataset_size
        sigma = 0.5
        epochs = 3
        steps = epochs * (dataset_size // batch_size)
        delta = 1e-4

        our_eps = dp.compute_epsilon(sigma, q, steps, delta)
        ref_pld = google_poisson_gaussian_pld(sigma, q, steps)
        ref_eps = ref_pld.get_epsilon_for_delta(delta)

        assert math.isfinite(our_eps) and our_eps > 0
        assert rel_error(our_eps, ref_eps) < 1e-3, (
            f"LLM fine-tuning: ours={our_eps:.6f}, ref={ref_eps:.6f}"
        )

    def test_cifar10_training(self):
        """CIFAR-10: 50k dataset, batch=500, sigma=0.8, 10 epochs."""
        dataset_size = 50_000
        batch_size = 500
        q = batch_size / dataset_size
        sigma = 0.8
        epochs = 10
        steps = epochs * (dataset_size // batch_size)
        delta = 1e-5

        our_eps = dp.compute_epsilon(sigma, q, steps, delta)
        ref_pld = google_poisson_gaussian_pld(sigma, q, steps)
        ref_eps = ref_pld.get_epsilon_for_delta(delta)

        assert math.isfinite(our_eps) and our_eps > 0
        assert rel_error(our_eps, ref_eps) < 1e-3, (
            f"CIFAR-10: ours={our_eps:.6f}, ref={ref_eps:.6f}"
        )

    def test_imagenet_training(self):
        """ImageNet: 1.2M dataset, batch=4096, sigma=1.0, 5 epochs."""
        dataset_size = 1_200_000
        batch_size = 4096
        q = batch_size / dataset_size
        sigma = 1.0
        epochs = 5
        steps = epochs * (dataset_size // batch_size)
        delta = 1e-6

        our_eps = dp.compute_epsilon(sigma, q, steps, delta)
        ref_pld = google_poisson_gaussian_pld(sigma, q, steps)
        ref_eps = ref_pld.get_epsilon_for_delta(delta)

        assert math.isfinite(our_eps) and our_eps > 0
        assert rel_error(our_eps, ref_eps) < 1e-3, (
            f"ImageNet: ours={our_eps:.6f}, ref={ref_eps:.6f}"
        )

    def test_finetuning_benchmark_params(self):
        """Fine-tuning benchmark: sigma=0.5, 7812 steps (from old benchmarks)."""
        sigma = 0.5
        q = 0.00064
        steps = 7812
        delta = 1e-5

        our_eps = dp.compute_epsilon(sigma, q, steps, delta)
        ref_pld = google_poisson_gaussian_pld(sigma, q, steps)
        ref_eps = ref_pld.get_epsilon_for_delta(delta)

        assert math.isfinite(our_eps) and our_eps > 0
        tol = max(0.01, 0.01 * abs(ref_eps))
        assert abs(our_eps - ref_eps) < tol, (
            f"Fine-tuning benchmark: ours={our_eps:.6f}, ref={ref_eps:.6f}"
        )

    def test_alignment_benchmark_params(self):
        """Alignment benchmark: sigma=0.8, 3750 steps (from old benchmarks)."""
        sigma = 0.8
        q = 0.0008
        steps = 3750
        delta = 1e-5

        our_eps = dp.compute_epsilon(sigma, q, steps, delta)
        ref_pld = google_poisson_gaussian_pld(sigma, q, steps)
        ref_eps = ref_pld.get_epsilon_for_delta(delta)

        assert math.isfinite(our_eps) and our_eps > 0
        tol = max(0.01, 0.01 * abs(ref_eps))
        assert abs(our_eps - ref_eps) < tol, (
            f"Alignment benchmark: ours={our_eps:.6f}, ref={ref_eps:.6f}"
        )

    def test_all_metrics_consistent_workflow(self):
        """All metrics should be consistent for a single training run."""
        sigma, q, steps, delta_target = 0.5, 0.05, 200, 1e-5
        proc = dp.poisson(sigma, q) * steps

        eps = proc.epsilon_at(delta_target)
        delta = proc.delta_at(eps)
        adv = proc.advantage()
        beta = proc.beta_at(0.1)
        risk = proc.risk_at(0.5)

        # All should be finite.
        for name, val in [
            ("eps", eps), ("delta", delta), ("adv", adv),
            ("beta", beta), ("risk", risk),
        ]:
            assert math.isfinite(val), f"{name}={val}"

        # Delta round-trip.
        assert abs(delta - delta_target) < 0.01, (
            f"Round-trip: target delta={delta_target}, recovered delta={delta}"
        )

        # Advantage = delta_at(0).
        delta_0 = proc.delta_at(0.0)
        assert abs(adv - delta_0) < 1e-6, (
            f"advantage={adv}, delta_at(0)={delta_0}"
        )

        # Risk in valid range.
        assert 0.0 <= risk <= 0.5, f"risk={risk}"

        # Beta in valid range.
        assert 0.0 <= beta <= 1.0, f"beta={beta}"

    def test_multi_phase_heterogeneous_training(self):
        """Multi-phase DP-SGD: different noise at each phase, cross-validated."""
        delta = 1e-5

        # Phase 1: warm-up with high noise (1000 steps).
        # Phase 2: main training with medium noise (3000 steps).
        # Phase 3: fine-tuning with lower noise (1000 steps).
        phase1 = dp.poisson(1.0, 0.01) * 1000
        phase2 = dp.poisson(0.7, 0.01) * 3000
        phase3 = dp.poisson(0.5, 0.01) * 1000

        total = phase1 | phase2 | phase3
        our_eps = total.epsilon_at(delta)

        # Cross-validate with dp_accounting.
        g1 = google_poisson_gaussian_pld(1.0, 0.01, 1000)
        g2 = google_poisson_gaussian_pld(0.7, 0.01, 3000)
        g3 = google_poisson_gaussian_pld(0.5, 0.01, 1000)
        g_total = g1.compose(g2).compose(g3)
        ref_eps = g_total.get_epsilon_for_delta(delta)

        assert math.isfinite(our_eps) and our_eps > 0
        tol = max(0.05, 0.01 * abs(ref_eps))
        assert abs(our_eps - ref_eps) < tol, (
            f"Multi-phase: ours={our_eps:.6f}, ref={ref_eps:.6f}"
        )


# =============================================================================
# 11. Triple validation -- ours vs dp_accounting vs riskcal
# =============================================================================

# Riskcal helper: mirrors the old create_riskcal_accountant pattern.


def create_riskcal_accountant(sigma, q, steps):
    """Create a riskcal CTDAccountant for reference comparison."""
    acct = RiskcalAccountant()
    for _ in range(steps):
        acct.step(noise_multiplier=sigma, sample_rate=q)
    return acct


# Representative subset of composition params for triple-validation (fast).
TRIPLE_VALIDATION_PARAMS = [
    (0.3, 0.01, 200, "low-noise"),
    (0.5, 0.01, 1000, "production"),
    (0.5, 0.05, 200, "production-large-q"),
    (0.8, 0.01, 500, "high-noise"),
    (1.0, 0.01, 5000, "very-high-noise"),
    (1.0, 0.001, 10000, "high-composition"),
]


class TestTripleValidationEpsilon:
    """Validate epsilon against both dp_accounting AND riskcal.

    Ported from test_dual_validation.py::TestDualValidationBasic.
    """

    @pytest.mark.parametrize(
        "sigma,q,steps,desc",
        TRIPLE_VALIDATION_PARAMS,
        ids=[c[3] for c in TRIPLE_VALIDATION_PARAMS],
    )
    def test_epsilon_matches_riskcal(self, sigma, q, steps, desc):
        """Our epsilon should match riskcal."""
        delta = 1e-5

        our_eps = dp.compute_epsilon(sigma, q, steps, delta)

        acct_riskcal = create_riskcal_accountant(sigma, q, steps)
        eps_riskcal = acct_riskcal.get_epsilon(delta=delta)

        assert math.isfinite(our_eps), f"{desc}: our epsilon not finite"
        assert math.isfinite(eps_riskcal), f"{desc}: riskcal epsilon not finite"

        # Max observed error in old tests: 2.75e-05
        assert rel_error(our_eps, eps_riskcal) < 1e-3, (
            f"{desc}: ours={our_eps:.10f}, riskcal={eps_riskcal:.10f}, "
            f"rel_err={rel_error(our_eps, eps_riskcal):.2e}"
        )

    @pytest.mark.parametrize(
        "sigma,q,steps,desc",
        TRIPLE_VALIDATION_PARAMS,
        ids=[c[3] for c in TRIPLE_VALIDATION_PARAMS],
    )
    def test_all_three_epsilon_agree(self, sigma, q, steps, desc):
        """ours, dp_accounting, and riskcal should all agree on epsilon."""
        delta = 1e-5

        our_eps = dp.compute_epsilon(sigma, q, steps, delta)

        google_pld = google_poisson_gaussian_pld(sigma, q, steps)
        eps_dp = google_pld.get_epsilon_for_delta(delta)

        acct_riskcal = create_riskcal_accountant(sigma, q, steps)
        eps_riskcal = acct_riskcal.get_epsilon(delta=delta)

        epsilons = [our_eps, eps_dp, eps_riskcal]
        assert all(math.isfinite(e) for e in epsilons), (
            f"{desc}: not all finite: {epsilons}"
        )

        mean_eps = sum(epsilons) / len(epsilons)
        for label, eps in [
            ("ours", our_eps),
            ("dp_accounting", eps_dp),
            ("riskcal", eps_riskcal),
        ]:
            assert rel_error(eps, mean_eps) < 1e-3, (
                f"{desc}: {label}={eps:.10f} deviates from mean={mean_eps:.10f}"
            )


class TestTripleValidationBeta:
    """Validate beta (trade-off function) against riskcal.

    Ported from test_dual_validation.py::TestDualValidationMetrics
    and accountants/test_validation.py::TestBetaValidationVsRiskcal.
    """

    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1, 0.2, 0.5])
    def test_beta_matches_riskcal(self, alpha):
        """Beta at various alphas should match riskcal."""
        sigma, q, steps = 0.8, 0.01, 500

        proc = dp.poisson(sigma, q) * steps
        beta_ours = proc.beta_at(alpha)

        evaluator = create_dpsgd_evaluator(
            sample_rate=q,
            num_steps=steps,
            grid_step=1e-4,
            target_alpha=alpha,
        )
        metrics_ref = evaluator(sigma)
        beta_ref = metrics_ref.beta

        # Max observed error in old tests: 3.91e-05
        assert rel_error(beta_ours, beta_ref) < 1e-3, (
            f"alpha={alpha}: ours={beta_ours:.10f}, riskcal={beta_ref:.10f}, "
            f"rel_err={rel_error(beta_ours, beta_ref):.2e}"
        )

    @pytest.mark.parametrize("sigma", [0.3, 0.5, 0.8])
    def test_beta_low_noise_matches_riskcal(self, sigma):
        """Beta for low-noise scenarios should match riskcal."""
        q, steps = 0.05, 200

        proc = dp.poisson(sigma, q) * steps

        for alpha in [0.01, 0.05, 0.1, 0.3, 0.5]:
            beta_ours = proc.beta_at(alpha)

            evaluator = create_dpsgd_evaluator(
                sample_rate=q,
                num_steps=steps,
                grid_step=1e-4,
                target_alpha=alpha,
            )
            metrics_ref = evaluator(sigma)
            beta_ref = metrics_ref.beta

            assert rel_error(beta_ours, beta_ref) < 1e-3, (
                f"sigma={sigma}, alpha={alpha}: "
                f"ours={beta_ours:.10f}, riskcal={beta_ref:.10f}, "
                f"rel_err={rel_error(beta_ours, beta_ref):.2e}"
            )

    def test_regression_bug2_beta_high_alpha(self):
        """REGRESSION: Bug #2 - beta at high alpha was up to 33% wrong.

        Ported from test_dual_validation.py::TestDualValidationRegressions.
        """
        sigma, q, steps = 0.5, 0.0016, 1000

        proc = dp.poisson(sigma, q) * steps

        for alpha in [0.3, 0.4, 0.5, 0.6]:
            beta_ours = proc.beta_at(alpha)

            evaluator = create_dpsgd_evaluator(
                sample_rate=q,
                num_steps=steps,
                grid_step=1e-4,
                target_alpha=alpha,
            )
            metrics_ref = evaluator(sigma)
            beta_ref = metrics_ref.beta

            # Was up to 33% error before fix; now must be < 0.1%.
            assert rel_error(beta_ours, beta_ref) < 1e-3, (
                f"Bug#2 regression alpha={alpha}: "
                f"ours={beta_ours:.10f}, riskcal={beta_ref:.10f}, "
                f"rel_err={rel_error(beta_ours, beta_ref):.2e} "
                f"(was up to 33% before fix)"
            )


class TestTripleValidationAdvantage:
    """Validate advantage metric against riskcal.

    Ported from test_dual_validation.py::TestDualValidationMetrics.
    """

    def test_advantage_matches_riskcal(self):
        """Advantage should match riskcal."""
        sigma, q, steps = 0.5, 0.01, 1000

        proc = dp.poisson(sigma, q) * steps
        advantage_ours = proc.advantage()

        evaluator = create_dpsgd_evaluator(
            sample_rate=q,
            num_steps=steps,
            grid_step=1e-4,
        )
        metrics_ref = evaluator(sigma)
        advantage_ref = metrics_ref.advantage

        # Max observed error in old tests: 5.38e-10
        assert rel_error(advantage_ours, advantage_ref) < 1e-6, (
            f"advantage: ours={advantage_ours:.12f}, "
            f"riskcal={advantage_ref:.12f}, "
            f"rel_err={rel_error(advantage_ours, advantage_ref):.2e}"
        )

    @pytest.mark.parametrize("sigma", [0.3, 0.5, 0.8])
    def test_advantage_different_noise_matches_riskcal(self, sigma):
        """Advantage at different noise levels should match riskcal."""
        q, steps = 0.0016, 1000

        proc = dp.poisson(sigma, q) * steps
        advantage_ours = proc.advantage()

        evaluator = create_dpsgd_evaluator(
            sample_rate=q,
            num_steps=steps,
            grid_step=1e-4,
        )
        metrics_ref = evaluator(sigma)
        advantage_ref = metrics_ref.advantage

        assert rel_error(advantage_ours, advantage_ref) < 1e-4, (
            f"sigma={sigma}: advantage ours={advantage_ours:.10f}, "
            f"riskcal={advantage_ref:.10f}"
        )


class TestTripleValidationCalibration:
    """Cross-validate calibration results with riskcal.

    Ported from test_dual_validation.py::TestDualValidationCalibration.
    """

    def test_calibrate_epsilon_validates_with_riskcal(self):
        """Calibrated noise checked with riskcal epsilon."""
        q, steps = 0.01, 1000
        target_eps = 8.0
        delta = 1e-5

        nm = dp.calibrate_noise(
            target_epsilon=target_eps,
            target_delta=delta,
            sample_rate=q,
            num_steps=steps,
        )

        # Verify using riskcal.
        acct_riskcal = create_riskcal_accountant(nm, q, steps)
        eps_riskcal = acct_riskcal.get_epsilon(delta=delta)

        assert eps_riskcal <= target_eps * 1.1, (
            f"riskcal check: nm={nm:.4f}, eps_riskcal={eps_riskcal:.6f}, "
            f"target={target_eps}"
        )


class TestTripleValidationRealistic:
    """Realistic workflows validated against both references.

    Ported from test_dual_validation.py::TestDualValidationRealistic.
    """

    def test_llm_finetuning_triple(self):
        """LLM fine-tuning: validate epsilon vs dp_accounting, advantage vs riskcal."""
        dataset_size = 10_000
        batch_size = 16
        q = batch_size / dataset_size
        sigma = 0.5
        epochs = 3
        steps = epochs * (dataset_size // batch_size)
        delta = 1e-4

        # Our implementation.
        proc = dp.poisson(sigma, q) * steps
        our_eps = proc.epsilon_at(delta)
        our_advantage = proc.advantage()

        # dp_accounting.
        ref_pld = google_poisson_gaussian_pld(sigma, q, steps)
        eps_dp = ref_pld.get_epsilon_for_delta(delta)

        # riskcal.
        acct_riskcal = create_riskcal_accountant(sigma, q, steps)
        eps_riskcal = acct_riskcal.get_epsilon(delta=delta)

        evaluator = create_dpsgd_evaluator(
            sample_rate=q,
            num_steps=steps,
            grid_step=1e-4,
        )
        metrics_riskcal = evaluator(sigma)

        # All epsilons should agree.
        assert rel_error(our_eps, eps_dp) < 1e-3, (
            f"LLM: ours vs dp_accounting: {our_eps:.6f} vs {eps_dp:.6f}"
        )
        assert rel_error(our_eps, eps_riskcal) < 1e-3, (
            f"LLM: ours vs riskcal: {our_eps:.6f} vs {eps_riskcal:.6f}"
        )

        # Advantage should match riskcal.
        assert rel_error(our_advantage, metrics_riskcal.advantage) < 1e-4, (
            f"LLM: advantage ours={our_advantage:.10f}, "
            f"riskcal={metrics_riskcal.advantage:.10f}"
        )

    def test_cifar10_triple(self):
        """CIFAR-10: triple validation."""
        dataset_size = 50_000
        batch_size = 500
        q = batch_size / dataset_size
        sigma = 0.8
        epochs = 10
        steps = epochs * (dataset_size // batch_size)
        delta = 1e-5

        our_eps = dp.compute_epsilon(sigma, q, steps, delta)

        ref_pld = google_poisson_gaussian_pld(sigma, q, steps)
        eps_dp = ref_pld.get_epsilon_for_delta(delta)

        acct_riskcal = create_riskcal_accountant(sigma, q, steps)
        eps_riskcal = acct_riskcal.get_epsilon(delta=delta)

        assert rel_error(our_eps, eps_dp) < 1e-3
        assert rel_error(our_eps, eps_riskcal) < 1e-3

    def test_imagenet_triple(self):
        """ImageNet: triple validation."""
        dataset_size = 1_200_000
        batch_size = 4096
        q = batch_size / dataset_size
        sigma = 1.0
        epochs = 5
        steps = epochs * (dataset_size // batch_size)
        delta = 1e-6

        our_eps = dp.compute_epsilon(sigma, q, steps, delta)

        ref_pld = google_poisson_gaussian_pld(sigma, q, steps)
        eps_dp = ref_pld.get_epsilon_for_delta(delta)

        acct_riskcal = create_riskcal_accountant(sigma, q, steps)
        eps_riskcal = acct_riskcal.get_epsilon(delta=delta)

        assert rel_error(our_eps, eps_dp) < 1e-3
        assert rel_error(our_eps, eps_riskcal) < 1e-3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
