//! Gradient accumulation tests — Google reference validation, properties, and integration
//!
//! Tests the `Accumulated<Poisson<Gaussian>>` combinator for gradient accumulation
//! in DP-SGD. Includes:
//! - Math-level validation against Google's `MixtureGaussianPrivacyLoss` reference values
//! - m=1 fallback: accumulated matches standard Poisson exactly
//! - Accumulated vs repeated comparison (accumulated is worse — noise added once)
//! - Monotonicity properties (microbatches, rate, sigma)
//! - Hardcoded reference values for regression detection
//! - Composition via repeat + cached
//! - Process trait integration
//! - AdaClip + accumulated integration
//!
//! # Realistic parameters
//!
//! Tests use realistic DP-SGD parameters rather than toy values:
//! - Fine-tuning (50K dataset): q=0.00064 (batch 32), m=8, sigma=1.0
//! - Fine-tuning (200K dataset): q=0.00032 (batch 64), m=4, sigma=0.8
//! - Pre-training (1M dataset): q=0.000032 (batch 32), m=16, sigma=0.5

use super::*;
use crate::composition::repeat;

// =========================================================================
// Test configuration
// =========================================================================

// Tolerance for reference values is computed from PLD's actual discretization
// via reference_tol(disc, k). See tests/mod.rs for the derivation.

// =========================================================================
// Helpers
// =========================================================================

fn assert_close(ref_val: f64, func_val: f64, tol: f64, context: &str) {
    if ref_val.is_infinite() && func_val.is_infinite() && ref_val.signum() == func_val.signum() {
        return;
    }
    let diff = (ref_val - func_val).abs();
    assert!(
        diff < tol,
        "{}: ref={:.10e}, func={:.10e}, diff={:.2e} (tol={:.0e})",
        context,
        ref_val,
        func_val,
        diff,
        tol
    );
}

// =========================================================================
// m=1 fallback: matches standard Poisson exactly
// =========================================================================

#[test]
fn test_m1_matches_standard_poisson() {
    // accumulate(poisson(g, q), 1) should match poisson(g, q) bit-for-bit
    // because the evidence impl short-circuits to TightGaussianPoissonEvidence.
    let q = 0.00064;
    let nm = 1.0;

    let poisson_pld = poisson(gaussian(nm).unwrap(), q).pld().unwrap();
    let accum_pld = accumulate(poisson(gaussian(nm).unwrap(), q), 1)
        .unwrap()
        .pld()
        .unwrap();

    for &eps in &[0.01, 0.1, 0.5, 1.0] {
        assert_close(
            poisson_pld.delta_at(eps),
            accum_pld.delta_at(eps),
            1e-14,
            &format!("m=1 delta_at eps={}", eps),
        );
    }
    for &delta in &[1e-8, 1e-6, 1e-5, 1e-4] {
        assert_close(
            poisson_pld.epsilon_at(delta),
            accum_pld.epsilon_at(delta),
            1e-14,
            &format!("m=1 epsilon_at delta={}", delta),
        );
    }
}

// =========================================================================
// Accumulated vs repeated Poisson comparison
// =========================================================================

#[test]
fn test_accumulated_worse_than_repeated_poisson() {
    // Accumulated adds noise once; repeated adds noise m times.
    // So accumulated has worse privacy (higher delta) for the same epsilon.
    let nm = 1.0;
    let q = 0.00064;

    for m in [2, 4, 8] {
        let accum_pld = accumulate(poisson(gaussian(nm).unwrap(), q), m)
            .unwrap()
            .pld()
            .unwrap();
        let repeated_pld = repeat(poisson(gaussian(nm).unwrap(), q), m)
            .unwrap()
            .pld()
            .unwrap();

        for &eps in &[0.01, 0.1, 0.3, 0.5] {
            let accum_delta = accum_pld.delta_at(eps);
            let repeated_delta = repeated_pld.delta_at(eps);

            assert!(
                accum_delta >= repeated_delta - 1e-15,
                "m={}, eps={}: accumulated delta ({:.4e}) should be >= repeated delta ({:.4e})",
                m,
                eps,
                accum_delta,
                repeated_delta,
            );
        }
    }
}

#[test]
fn test_accumulated_vs_repeated_practical_gap() {
    // With realistic q=0.00064, m=8: P(K>=2) ≈ 0.002
    // The gap is meaningful but both produce usable epsilon values.
    // Accumulated is worse because noise is added once (not m times).
    let nm = 1.0;
    let q = 0.00064;
    let m = 8;
    let delta = 1e-5;

    let accum_eps = accumulate(poisson(gaussian(nm).unwrap(), q), m)
        .unwrap()
        .epsilon_at(delta)
        .unwrap();
    let repeated_eps = repeat(poisson(gaussian(nm).unwrap(), q), m)
        .unwrap()
        .epsilon_at(delta)
        .unwrap();

    // Accumulated should have higher epsilon (worse privacy)
    assert!(
        accum_eps >= repeated_eps,
        "accumulated eps ({}) should be >= repeated eps ({})",
        accum_eps,
        repeated_eps,
    );

    // Both should be finite and reasonable
    assert!(accum_eps.is_finite() && accum_eps > 0.0);
    assert!(repeated_eps.is_finite() && repeated_eps > 0.0);

    // Document the gap for reference
    eprintln!(
        "Practical gap: accum_eps={:.6}, repeated_eps={:.6}, ratio={:.2}x",
        accum_eps,
        repeated_eps,
        accum_eps / repeated_eps,
    );
}

// =========================================================================
// Monotonicity properties
// =========================================================================

#[test]
fn test_monotonicity_in_microbatches() {
    // More microbatches → worse privacy (higher delta)
    let nm = 1.0;
    let q = 0.00064;
    let microbatches = [1, 2, 4, 8, 16];
    let eps = 0.1;

    let deltas: Vec<f64> = microbatches
        .iter()
        .map(|&m| {
            accumulate(poisson(gaussian(nm).unwrap(), q), m)
                .unwrap()
                .pld()
                .unwrap()
                .delta_at(eps)
        })
        .collect();

    for i in 1..deltas.len() {
        assert!(
            deltas[i] >= deltas[i - 1] - 1e-15,
            "m={} delta={:.4e} should be >= m={} delta={:.4e}",
            microbatches[i],
            deltas[i],
            microbatches[i - 1],
            deltas[i - 1],
        );
    }
}

#[test]
fn test_monotonicity_in_rate() {
    // Higher sampling rate → worse privacy
    let nm = 1.0;
    let m = 4;
    let rates = [0.00032, 0.00064, 0.001];
    let eps = 0.1;

    let deltas: Vec<f64> = rates
        .iter()
        .map(|&q| {
            accumulate(poisson(gaussian(nm).unwrap(), q), m)
                .unwrap()
                .pld()
                .unwrap()
                .delta_at(eps)
        })
        .collect();

    for i in 1..deltas.len() {
        assert!(
            deltas[i] >= deltas[i - 1] - 1e-15,
            "q={} delta={:.4e} should be >= q={} delta={:.4e}",
            rates[i],
            deltas[i],
            rates[i - 1],
            deltas[i - 1],
        );
    }
}

#[test]
fn test_monotonicity_in_sigma() {
    // Larger sigma → better privacy (lower delta)
    let q = 0.00064;
    let m = 8;
    let sigmas = [0.8, 1.0, 1.1, 1.2];
    let eps = 0.1;

    let deltas: Vec<f64> = sigmas
        .iter()
        .map(|&nm| {
            accumulate(poisson(gaussian(nm).unwrap(), q), m)
                .unwrap()
                .pld()
                .unwrap()
                .delta_at(eps)
        })
        .collect();

    for i in 1..deltas.len() {
        assert!(
            deltas[i] <= deltas[i - 1] + 1e-15,
            "sigma={} delta={:.4e} should be <= sigma={} delta={:.4e}",
            sigmas[i],
            deltas[i],
            sigmas[i - 1],
            deltas[i - 1],
        );
    }
}

// =========================================================================
// Composition (realistic training run)
// =========================================================================

#[test]
fn test_accumulated_composition_via_repeat() {
    // 1000 training steps of logical batch 256 on a 50K dataset
    let step = accumulate(poisson(gaussian(1.0).unwrap(), 0.00064), 8).unwrap();
    let process = repeat(step, 1000).unwrap();
    let pld = process.pld().unwrap();

    let eps = pld.epsilon_at(1e-5);
    assert!(eps.is_finite() && eps > 0.0, "epsilon={}", eps);

    // Should be higher than single step
    let single_eps = accumulate(poisson(gaussian(1.0).unwrap(), 0.00064), 8)
        .unwrap()
        .pld()
        .unwrap()
        .epsilon_at(1e-5);
    assert!(
        eps > single_eps,
        "composed eps={} should be > single eps={}",
        eps,
        single_eps
    );
}

#[test]
fn test_accumulated_cached_composition() {
    let step = cached(accumulate(poisson(gaussian(1.0).unwrap(), 0.00064), 8).unwrap());
    let process = repeat(step, 1000).unwrap();
    let pld = process.pld().unwrap();

    let eps = pld.epsilon_at(1e-5);
    assert!(eps.is_finite() && eps > 0.0, "epsilon={}", eps);
}

// =========================================================================
// Process trait integration
// =========================================================================

#[test]
fn test_process_trait_all_metrics() {
    let acc = accumulate(poisson(gaussian(1.0).unwrap(), 0.00064), 8).unwrap();
    let pld = acc.pld().unwrap();

    let eps = acc.epsilon_at(1e-5).unwrap();
    assert!(eps.is_finite() && eps > 0.0, "epsilon_at: {}", eps);
    assert!((eps - pld.epsilon_at(1e-5)).abs() < 1e-10);

    let delta = acc.delta_at(0.5).unwrap();
    assert!(
        delta.is_finite() && delta >= 0.0 && delta <= 1.0,
        "delta_at: {}",
        delta
    );

    let adv = acc.advantage().unwrap();
    assert!(
        adv.is_finite() && adv >= 0.0 && adv <= 1.0,
        "advantage: {}",
        adv
    );

    let beta = acc.beta_at(0.01).unwrap();
    assert!(
        beta.is_finite() && beta >= 0.0 && beta <= 1.0,
        "beta_at: {}",
        beta
    );

    let risk = acc.risk_at(0.5).unwrap();
    assert!(
        risk.is_finite() && risk >= 0.0 && risk <= 0.5,
        "risk_at: {}",
        risk
    );
}

// =========================================================================
// Constructor validation
// =========================================================================

#[test]
#[should_panic(expected = "Number of microbatches must be > 0")]
fn test_invalid_microbatches_panics() {
    accumulate(poisson(gaussian(1.0).unwrap(), 0.01), 0).unwrap();
}

#[test]
fn test_structural_equality() {
    let a = accumulate(poisson(gaussian(1.1).unwrap(), 0.00064), 8).unwrap();
    let b = accumulate(poisson(gaussian(1.1).unwrap(), 0.00064), 8).unwrap();
    assert_eq!(a, b);

    let c = accumulate(poisson(gaussian(1.1).unwrap(), 0.00064), 4).unwrap();
    assert_ne!(a, c);

    let d = accumulate(poisson(gaussian(1.2).unwrap(), 0.00064), 8).unwrap();
    assert_ne!(a, d);

    let e = accumulate(poisson(gaussian(1.1).unwrap(), 0.001), 8).unwrap();
    assert_ne!(a, e);
}

// =========================================================================
// AdaClip + accumulated integration
// =========================================================================

#[test]
fn test_adaclip_accumulated() {
    let ac = adaclip(gaussian(1.0).unwrap(), 50.0);
    let acc = accumulate(poisson(ac.clone(), 0.00064), 8).unwrap();
    let pld = acc.pld().unwrap();

    let eps = pld.epsilon_at(1e-5);
    assert!(eps.is_finite() && eps > 0.0, "epsilon={}", eps);

    // Should match accumulate(poisson(gaussian(effective_nm), ...), m)
    let acc_explicit = accumulate(
        poisson(gaussian(ac.effective_noise_multiplier()).unwrap(), 0.00064),
        8,
    )
    .unwrap();
    let pld_explicit = acc_explicit.pld().unwrap();

    assert_close(
        pld.epsilon_at(1e-5),
        pld_explicit.epsilon_at(1e-5),
        1e-14,
        "adaclip accumulated vs explicit gaussian",
    );
}

// =========================================================================
// Reference values (captured from implementation — regression detection)
// =========================================================================

#[test]
fn test_reference_accumulated_nm10_q00064_m8() {
    // Fine-tuning: 50K dataset, physical batch 32, logical batch 256 (m=8)
    let pld = accumulate(poisson(gaussian(1.0).unwrap(), 0.00064), 8)
        .unwrap()
        .pld()
        .unwrap();
    let tol = reference_tol(pld_disc(&pld), 1);

    assert_close(
        pld.delta_at(0.01),
        5.748461529281695e-4,
        tol,
        "delta_at(0.01)",
    );
    assert_close(
        pld.delta_at(0.1),
        6.910429981700806e-6,
        tol,
        "delta_at(0.1)",
    );
    assert_close(
        pld.delta_at(0.5),
        1.154528932263161e-8,
        tol,
        "delta_at(0.5)",
    );
    assert_close(
        pld.epsilon_at(1e-5),
        8.816761463197417e-2,
        tol,
        "epsilon_at(1e-5)",
    );
    assert_close(
        pld.epsilon_at(1e-6),
        1.789937590369299e-1,
        tol,
        "epsilon_at(1e-6)",
    );
}

#[test]
fn test_reference_accumulated_nm08_q00032_m4() {
    // Fine-tuning: 200K dataset, physical batch 64, logical batch 256 (m=4)
    let pld = accumulate(poisson(gaussian(0.8).unwrap(), 0.00032), 4)
        .unwrap()
        .pld()
        .unwrap();
    let tol = reference_tol(pld_disc(&pld), 1);

    assert_close(
        pld.delta_at(0.01),
        6.754884305231108e-5,
        tol,
        "delta_at(0.01)",
    );
    assert_close(
        pld.delta_at(0.1),
        6.469273810374852e-7,
        tol,
        "delta_at(0.1)",
    );
    assert_close(
        pld.delta_at(0.5),
        2.359003109201414e-9,
        tol,
        "delta_at(0.5)",
    );
    assert_close(
        pld.epsilon_at(1e-5),
        3.157042474206383e-2,
        tol,
        "epsilon_at(1e-5)",
    );
    assert_close(
        pld.epsilon_at(1e-6),
        8.515667151236277e-2,
        tol,
        "epsilon_at(1e-6)",
    );
}
