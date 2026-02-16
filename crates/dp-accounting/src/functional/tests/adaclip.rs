//! AdaClip integration tests
//!
//! Tests the adaptive clipping mechanism combinator from Andrew et al. (2021),
//! "Differentially Private Learning with Adaptive Clipping." NeurIPS 2021.
//!
//! Key insights tested:
//! - Base mechanism delegates exactly to `gaussian(effective_nm)` (Theorem 1)
//! - Poisson amplification also delegates via `into_poisson_parts`
//! - Smaller `sigma_b` → worse privacy (more budget on quantile estimation)

use super::*;
use crate::functional::composition::repeat;
use crate::functional::discretization::DiscretizationConfig;

// =========================================================================
// Test configuration
// =========================================================================

/// Tolerance for base mechanism delegation (should be exact since delegation is literal).
const DELEGATION_TOL: f64 = 1e-14;

/// Helper to create AdaClip<Gaussian> for any noise multiplier (including > 1.2).
fn make_adaclip(nm: f64, sb: f64) -> AdaClip<Gaussian, GaussianAdaClipEvidence> {
    let g = Gaussian::new_unchecked(nm, DiscretizationConfig::default());
    adaclip(g, sb)
}

// Tolerance for reference values is computed from PLD's actual discretization
// via reference_tol(disc, k). See tests/mod.rs for the derivation.

// =========================================================================
// 1. Theorem 1 equivalence: base mechanism delegates to Gaussian
// =========================================================================

/// Base mechanism delegates exactly to `gaussian(effective_nm)`.
/// Since `Process::pld()` literally constructs `Gaussian { noise_multiplier: effective_nm }`,
/// the results should be bit-identical.
#[test]
fn test_base_delegation_exact() {
    for (nm, sb) in [
        (0.5, 10.0),
        (0.5, 20.0),
        (0.8, 30.0),
        (1.0, 50.0),
        (1.0, 100.0),
    ] {
        let ac = make_adaclip(nm, sb);
        let eff = ac.effective_noise_multiplier();
        let g = gaussian(eff).unwrap();

        let ac_pld = ac.pld().unwrap();
        let g_pld = g.pld().unwrap();

        let ac_eps = ac_pld.epsilon_at(1e-5);
        let g_eps = g_pld.epsilon_at(1e-5);
        let ac_delta = ac_pld.delta_at(1.0);
        let g_delta = g_pld.delta_at(1.0);

        assert!(
            (ac_eps - g_eps).abs() < DELEGATION_TOL,
            "nm={}, sb={}: epsilon_at diff = {:.2e}",
            nm,
            sb,
            (ac_eps - g_eps).abs()
        );
        assert!(
            (ac_delta - g_delta).abs() < DELEGATION_TOL,
            "nm={}, sb={}: delta_at diff = {:.2e}",
            nm,
            sb,
            (ac_delta - g_delta).abs()
        );
    }
}

// =========================================================================
// 2. Combined sensitivity and effective noise multiplier
// =========================================================================

/// As sigma_b → ∞, combined_sensitivity → 1/nm and effective_nm → nm.
/// The quantile query becomes "free" and the mechanism reduces to the base Gaussian.
#[test]
fn test_large_sigma_b_limit_sensitivity() {
    for nm in [0.5, 1.0, 1.5] {
        let ac = make_adaclip(nm, 1e10);
        let expected_s = 1.0 / nm;
        let expected_eff = nm; // z = 1/S = 1/(1/nm) = nm

        assert!(
            (ac.combined_sensitivity() - expected_s).abs() < 1e-10,
            "nm={}: S={:.15e}, expected={:.15e}",
            nm,
            ac.combined_sensitivity(),
            expected_s
        );
        assert!(
            (ac.effective_noise_multiplier() - expected_eff).abs() < 1e-10,
            "nm={}: eff_nm={:.15e}, expected={:.15e}",
            nm,
            ac.effective_noise_multiplier(),
            expected_eff
        );
    }
}

/// Combined sensitivity is always strictly greater than the base gradient
/// sensitivity `1/nm` (quantile query always adds to the total sensitivity).
#[test]
fn test_combined_sensitivity_exceeds_base() {
    for (nm, sb) in [(0.5, 20.0), (1.0, 50.0), (1.5, 100.0), (1.0, 1000.0)] {
        let ac = make_adaclip(nm, sb);
        let base_sensitivity = 1.0 / nm;
        assert!(
            ac.combined_sensitivity() > base_sensitivity,
            "nm={}, sb={}: S={} should be > base={}",
            nm,
            sb,
            ac.combined_sensitivity(),
            base_sensitivity
        );
    }
}

// =========================================================================
// 3. Poisson amplification: equivalence with delegated Gaussian
// =========================================================================

/// `poisson(adaclip(...))` converts to Gaussian internally via `into_poisson_parts`,
/// so it produces bit-identical results to `poisson(gaussian(effective_nm))`.
#[test]
fn test_poisson_converts_to_gaussian() {
    for (nm, sb) in [(0.5, 20.0), (1.0, 50.0), (1.0, 100.0)] {
        let rate = 0.01;
        let ac = make_adaclip(nm, sb);
        let eff_nm = ac.effective_noise_multiplier();
        let p_ac = poisson(ac, rate);
        let p_explicit = poisson(gaussian(eff_nm).unwrap(), rate);

        // Both produce Poisson<Gaussian, _> — inner Gaussians should be identical
        assert_eq!(
            p_ac.inner.noise_multiplier,
            p_explicit.inner.noise_multiplier
        );
        assert_eq!(p_ac, p_explicit);

        // PLDs should be bit-identical
        let p_ac_pld = p_ac.pld().unwrap();
        let p_exp_pld = p_explicit.pld().unwrap();

        assert!(
            (p_ac_pld.epsilon_at(1e-5) - p_exp_pld.epsilon_at(1e-5)).abs() < DELEGATION_TOL,
            "nm={}, sb={}: Poisson PLDs differ",
            nm,
            sb,
        );
    }
}

// =========================================================================
// 4. Poisson amplification tighter than base
// =========================================================================

/// Poisson-amplified epsilon should be strictly less than base epsilon.
#[test]
fn test_poisson_tighter_than_base() {
    for (nm, sb) in [(1.0, 50.0), (0.5, 20.0), (1.5, 100.0)] {
        let ac = make_adaclip(nm, sb);
        let base_pld = ac.pld().unwrap();
        let base_eps = base_pld.epsilon_at(1e-5);

        let p_ac = poisson(ac, 0.01);
        let p_pld = p_ac.pld().unwrap();
        let p_eps = p_pld.epsilon_at(1e-5);

        assert!(
            p_eps < base_eps,
            "nm={}, sb={}: Poisson eps={:.6e} should be < base eps={:.6e}",
            nm,
            sb,
            p_eps,
            base_eps
        );
    }
}

// =========================================================================
// 5. Monotonicity
// =========================================================================

/// Smaller sigma_b → worse privacy (more budget on quantile query).
#[test]
fn test_monotonicity_sigma_b() {
    let nm = 1.0;
    let sigma_bs = [10.0, 20.0, 50.0, 100.0, 1000.0];

    let epsilons: Vec<f64> = sigma_bs
        .iter()
        .map(|&sb| make_adaclip(nm, sb).pld().unwrap().epsilon_at(1e-5))
        .collect();

    for i in 1..epsilons.len() {
        assert!(
            epsilons[i] < epsilons[i - 1],
            "sigma_b={} -> eps={:.10e} should be < sigma_b={} -> eps={:.10e}",
            sigma_bs[i],
            epsilons[i],
            sigma_bs[i - 1],
            epsilons[i - 1]
        );
    }
}

/// Larger noise_multiplier → better privacy (epsilon decreases).
#[test]
fn test_monotonicity_noise_multiplier() {
    let sb = 50.0;
    let nms = [0.5, 0.8, 1.0, 1.5, 2.0];

    let epsilons: Vec<f64> = nms
        .iter()
        .map(|&nm| make_adaclip(nm, sb).pld().unwrap().epsilon_at(1e-5))
        .collect();

    for i in 1..epsilons.len() {
        assert!(
            epsilons[i] < epsilons[i - 1],
            "nm={} -> eps={:.10e} should be < nm={} -> eps={:.10e}",
            nms[i],
            epsilons[i],
            nms[i - 1],
            epsilons[i - 1]
        );
    }
}

// =========================================================================
// 6. All five metrics work
// =========================================================================

/// All five privacy metrics produce valid values.
#[test]
fn test_all_metrics() {
    let ac = adaclip(gaussian(1.0).unwrap(), 50.0);
    let pld = ac.pld().unwrap();

    let eps = pld.epsilon_at(1e-5);
    assert!(eps > 0.0 && eps.is_finite(), "epsilon_at: {}", eps);

    let delta = pld.delta_at(1.0);
    assert!(delta >= 0.0 && delta <= 1.0, "delta_at: {}", delta);

    let adv = pld.advantage();
    assert!(adv >= 0.0 && adv <= 1.0, "advantage: {}", adv);

    let beta = pld.beta_at(0.01);
    assert!(beta >= 0.0 && beta <= 1.0, "beta_at: {}", beta);

    let risk = pld.risk_at(0.5);
    assert!(risk >= 0.0 && risk <= 1.0, "risk_at: {}", risk);
}

// =========================================================================
// 7. Reference values — regression protection
//
// Coverage across nm ∈ {0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.2} to
// ensure the formula z_eff = 1/δ̃ is exercised at nm ≠ 1 (where the old
// incorrect z_Δ/S formula would diverge).
// =========================================================================

// -- 7a. Base mechanism (single step, no Poisson) -------------------------

/// Reference values for nm=0.1, sigma_b=50 (single step).
#[test]
fn test_reference_base_nm01() {
    let ac = make_adaclip(0.1, 50.0);
    let pld = ac.pld().unwrap();
    let tol = reference_tol(pld_disc(&pld), 1);

    assert_abs_eq(pld.epsilon_at(1e-5), 9.18173607148263073e1, tol, "nm=0.1 eps(1e-5)");
    assert_abs_eq(pld.epsilon_at(1e-8), 1.05327324941838626e2, tol, "nm=0.1 eps(1e-8)");
    assert_abs_eq(pld.delta_at(1.0), 9.99998983462454638e-1, tol, "nm=0.1 delta(1)");
}

/// Reference values for nm=0.3, sigma_b=50 (single step).
#[test]
fn test_reference_base_nm03() {
    let ac = make_adaclip(0.3, 50.0);
    let pld = ac.pld().unwrap();
    let tol = reference_tol(pld_disc(&pld), 1);

    assert_abs_eq(pld.epsilon_at(1e-5), 1.9130880882054647e1, tol, "nm=0.3 eps(1e-5)");
    assert_abs_eq(pld.epsilon_at(1e-8), 2.3677802943525734e1, tol, "nm=0.3 eps(1e-8)");
    assert_abs_eq(pld.delta_at(1.0), 8.4723828528328216e-1, tol, "nm=0.3 delta(1)");
}

/// Reference values for nm=0.5, sigma_b=50 (single step).
#[test]
fn test_reference_base_nm05() {
    let ac = make_adaclip(0.5, 50.0);
    let pld = ac.pld().unwrap();
    let tol = reference_tol(pld_disc(&pld), 1);

    assert_abs_eq(pld.epsilon_at(1e-5), 9.9974100758784594e0, tol, "nm=0.5 eps(1e-5)");
    assert_abs_eq(pld.epsilon_at(1e-8), 1.2749434044058306e1, tol, "nm=0.5 eps(1e-8)");
    assert_abs_eq(pld.delta_at(1.0), 5.0987046158374327e-1, tol, "nm=0.5 delta(1)");
}

/// Reference values for nm=0.7, sigma_b=50 (single step).
#[test]
fn test_reference_base_nm07() {
    let ac = make_adaclip(0.7, 50.0);
    let pld = ac.pld().unwrap();
    let tol = reference_tol(pld_disc(&pld), 1);

    assert_abs_eq(pld.epsilon_at(1e-5), 6.6526820048178896e0, tol, "nm=0.7 eps(1e-5)");
    assert_abs_eq(pld.epsilon_at(1e-8), 8.6329711681016210e0, tol, "nm=0.7 eps(1e-8)");
    assert_abs_eq(pld.delta_at(1.0), 2.9194993786402684e-1, tol, "nm=0.7 delta(1)");
}

/// Reference values for nm=0.9, sigma_b=50 (single step).
#[test]
fn test_reference_base_nm09() {
    let ac = make_adaclip(0.9, 50.0);
    let pld = ac.pld().unwrap();
    let tol = reference_tol(pld_disc(&pld), 1);

    assert_abs_eq(pld.epsilon_at(1e-5), 4.9475530331602435e0, tol, "nm=0.9 eps(1e-5)");
    assert_abs_eq(pld.epsilon_at(1e-8), 6.4975575522548166e0, tol, "nm=0.9 eps(1e-8)");
    assert_abs_eq(pld.delta_at(1.0), 1.6749685359195779e-1, tol, "nm=0.9 delta(1)");
}

/// Reference values for nm=1.0, sigma_b=50 (single step) — full metric coverage.
#[test]
fn test_reference_base_nm10() {
    let ac = adaclip(gaussian(1.0).unwrap(), 50.0);
    let pld = ac.pld().unwrap();
    let tol = reference_tol(pld_disc(&pld), 1);

    assert_abs_eq(pld.epsilon_at(1e-5), 4.3774314853281036e0, tol, "nm=1.0 eps(1e-5)");
    assert_abs_eq(pld.epsilon_at(1e-8), 5.7764197920330371e0, tol, "nm=1.0 eps(1e-8)");
    assert_abs_eq(pld.delta_at(1.0), 1.2695434065505284e-1, tol, "nm=1.0 delta(1)");
    assert_abs_eq(pld.advantage(), 3.8294252526436207e-1, tol, "nm=1.0 advantage");
    assert_abs_eq(pld.beta_at(0.01), 9.0762947479971801e-1, tol, "nm=1.0 beta(0.01)");
    assert_abs_eq(pld.risk_at(0.5), 3.0852873736826070e-1, tol, "nm=1.0 risk(0.5)");
}

/// Reference values for nm=1.1, sigma_b=50 (single step).
#[test]
fn test_reference_base_nm11() {
    let ac = make_adaclip(1.1, 50.0);
    let pld = ac.pld().unwrap();
    let tol = reference_tol(pld_disc(&pld), 1);

    assert_abs_eq(pld.epsilon_at(1e-5), 3.9215231786002911e0, tol, "nm=1.1 eps(1e-5)");
    assert_abs_eq(pld.epsilon_at(1e-8), 5.1967402620453615e0, tol, "nm=1.1 eps(1e-8)");
    assert_abs_eq(pld.delta_at(1.0), 9.6163351834875815e-2, tol, "nm=1.1 delta(1)");
}

/// Reference values for nm=1.2, sigma_b=50 (single step).
#[test]
fn test_reference_base_nm12() {
    let ac = make_adaclip(1.2, 50.0);
    let pld = ac.pld().unwrap();
    let tol = reference_tol(pld_disc(&pld), 1);

    assert_abs_eq(pld.epsilon_at(1e-5), 3.5489890458561901e0, tol, "nm=1.2 eps(1e-5)");
    assert_abs_eq(pld.epsilon_at(1e-8), 4.7208908521502231e0, tol, "nm=1.2 eps(1e-8)");
    assert_abs_eq(pld.delta_at(1.0), 7.2731985834799756e-2, tol, "nm=1.2 delta(1)");
}

// -- 7b. Poisson-amplified (single step) ----------------------------------

/// Poisson reference: nm=0.5, sigma_b=50, rate=0.01.
#[test]
fn test_reference_poisson_nm05() {
    let pld = poisson(make_adaclip(0.5, 50.0), 0.01).pld().unwrap();
    let tol = reference_tol(pld_disc(&pld), 1);

    assert_abs_eq(pld.epsilon_at(1e-5), 3.0254764255471960e0, tol, "poisson nm=0.5 eps(1e-5)");
    assert_abs_eq(pld.epsilon_at(1e-8), 6.3938037130479977e0, tol, "poisson nm=0.5 eps(1e-8)");
    assert_abs_eq(pld.delta_at(1.0), 2.7366629378142095e-4, tol, "poisson nm=0.5 delta(1)");
}

/// Poisson reference: nm=0.7, sigma_b=50, rate=0.01.
#[test]
fn test_reference_poisson_nm07() {
    let pld = poisson(make_adaclip(0.7, 50.0), 0.01).pld().unwrap();
    let tol = reference_tol(pld_disc(&pld), 1);

    assert_abs_eq(pld.epsilon_at(1e-5), 8.5173268085927856e-1, tol, "poisson nm=0.7 eps(1e-5)");
    assert_abs_eq(pld.epsilon_at(1e-8), 2.8280065061987227e0, tol, "poisson nm=0.7 eps(1e-8)");
    assert_abs_eq(pld.delta_at(1.0), 5.6937815911669056e-6, tol, "poisson nm=0.7 delta(1)");
}

/// Poisson reference: nm=1.0, sigma_b=50, rate=0.01.
#[test]
fn test_reference_poisson_nm10() {
    let pld = poisson(adaclip(gaussian(1.0).unwrap(), 50.0), 0.01).pld().unwrap();
    let tol = reference_tol(pld_disc(&pld), 1);

    assert_abs_eq(pld.epsilon_at(1e-5), 1.994871522153658e-1, tol, "poisson nm=1.0 eps(1e-5)");
    assert_abs_eq(pld.epsilon_at(1e-8), 8.395525243358211e-1, tol, "poisson nm=1.0 eps(1e-8)");
    assert_abs_eq(pld.delta_at(1.0), 2.735992596757459e-9, tol, "poisson nm=1.0 delta(1)");
}

/// Poisson reference: nm=1.1, sigma_b=50, rate=0.01.
#[test]
fn test_reference_poisson_nm11() {
    let pld = poisson(make_adaclip(1.1, 50.0), 0.01).pld().unwrap();
    let tol = reference_tol(pld_disc(&pld), 1);

    assert_abs_eq(pld.epsilon_at(1e-5), 1.4232004622065317e-1, tol, "poisson nm=1.1 eps(1e-5)");
    assert_abs_eq(pld.epsilon_at(1e-8), 5.8400727355587234e-1, tol, "poisson nm=1.1 eps(1e-8)");
    assert_abs_eq(pld.delta_at(1.0), 1.3125037409942916e-10, tol, "poisson nm=1.1 delta(1)");
}

// -- 7c. Composed Poisson (k=10) -----------------------------------------

/// Composed reference: nm=0.5, sigma_b=50, rate=0.01, k=10.
#[test]
fn test_reference_composed_nm05_k10() {
    let pld = repeat(poisson(make_adaclip(0.5, 50.0), 0.01), 10).unwrap().pld().unwrap();
    let tol = reference_tol(pld_disc(&pld), 10);

    assert_abs_eq(pld.epsilon_at(1e-5), 4.3607466714393572e0, tol, "composed nm=0.5 eps(1e-5)");
    assert_abs_eq(pld.epsilon_at(1e-8), 7.6684545529041754e0, tol, "composed nm=0.5 eps(1e-8)");
    assert_abs_eq(pld.delta_at(1.0), 2.8165948016593608e-3, tol, "composed nm=0.5 delta(1)");
}

/// Composed reference: nm=0.9, sigma_b=50, rate=0.01, k=10.
#[test]
fn test_reference_composed_nm09_k10() {
    let pld = repeat(poisson(make_adaclip(0.9, 50.0), 0.01), 10).unwrap().pld().unwrap();
    let tol = reference_tol(pld_disc(&pld), 10);

    assert_abs_eq(pld.epsilon_at(1e-5), 5.6691205185134763e-1, tol, "composed nm=0.9 eps(1e-5)");
    assert_abs_eq(pld.epsilon_at(1e-8), 1.6439921744126933e0, tol, "composed nm=0.9 eps(1e-8)");
    assert_abs_eq(pld.delta_at(1.0), 4.7915357202193893e-7, tol, "composed nm=0.9 delta(1)");
}

/// Composed reference: nm=1.0, sigma_b=50, rate=0.01, k=10.
#[test]
fn test_reference_composed_nm10_k10() {
    let pld = repeat(poisson(adaclip(gaussian(1.0).unwrap(), 50.0), 0.01), 10)
        .unwrap().pld().unwrap();
    let tol = reference_tol(pld_disc(&pld), 10);

    assert_abs_eq(pld.epsilon_at(1e-5), 3.7997203853781863e-1, tol, "composed nm=1.0 eps(1e-5)");
    assert_abs_eq(pld.epsilon_at(1e-8), 1.1387249148472243e0, tol, "composed nm=1.0 eps(1e-8)");
    assert_abs_eq(pld.delta_at(1.0), 2.9442359869864858e-8, tol, "composed nm=1.0 delta(1)");
}

/// Composed reference: nm=1.1, sigma_b=50, rate=0.01, k=10.
#[test]
fn test_reference_composed_nm11_k10() {
    let pld = repeat(poisson(make_adaclip(1.1, 50.0), 0.01), 10).unwrap().pld().unwrap();
    let tol = reference_tol(pld_disc(&pld), 10);

    assert_abs_eq(pld.epsilon_at(1e-5), 2.7179160942969072e-1, tol, "composed nm=1.1 eps(1e-5)");
    assert_abs_eq(pld.epsilon_at(1e-8), 8.0391211055515044e-1, tol, "composed nm=1.1 eps(1e-8)");
    assert_abs_eq(pld.delta_at(1.0), 1.4157909113663829e-9, tol, "composed nm=1.1 delta(1)");
}

// =========================================================================
// 8. Structural equality
// =========================================================================

#[test]
fn test_structural_equality() {
    let a = adaclip(gaussian(1.0).unwrap(), 50.0);
    let b = adaclip(gaussian(1.0).unwrap(), 50.0);
    assert_eq!(a, b);

    let c = adaclip(gaussian(1.0).unwrap(), 100.0);
    assert_ne!(a, c);

    // Poisson wrappers — poisson(adaclip) converts to Poisson<Gaussian, _>
    let pa = poisson(a.clone(), 0.01);
    let pb = poisson(b, 0.01);
    assert_eq!(pa, pb);

    let pc = poisson(a, 0.02);
    assert_ne!(pa, pc);
}

// =========================================================================
// 9. Process trait integration
// =========================================================================

/// Composition with `repeat()` works correctly.
#[test]
fn test_repeat_integration() {
    let ac = adaclip(gaussian(1.0).unwrap(), 50.0);
    let step = poisson(ac, 0.01);
    let composed = repeat(step, 5).unwrap();
    let pld = composed.pld().unwrap();

    // Should have higher epsilon than single step
    let single_pld = poisson(adaclip(gaussian(1.0).unwrap(), 50.0), 0.01)
        .pld()
        .unwrap();
    assert!(pld.epsilon_at(1e-5) > single_pld.epsilon_at(1e-5));
}

/// Cached wrapper works with AdaClip.
#[test]
fn test_cached_adaclip() {
    let ac = adaclip(gaussian(1.0).unwrap(), 50.0);
    let c = cached(ac);
    let pld1 = c.pld().unwrap();
    let pld2 = c.pld().unwrap();

    assert!(
        (pld1.epsilon_at(1e-5) - pld2.epsilon_at(1e-5)).abs() < 1e-15,
        "Cached should return identical results"
    );
}

/// Cached Poisson AdaClip works with repeat.
#[test]
fn test_cached_poisson_adaclip_repeat() {
    let step = cached(poisson(adaclip(gaussian(1.0).unwrap(), 50.0), 0.01));
    let composed = repeat(step, 10).unwrap();
    let pld = composed.pld().unwrap();

    // Should match non-cached version
    let expected = repeat(poisson(adaclip(gaussian(1.0).unwrap(), 50.0), 0.01), 10)
        .unwrap()
        .pld()
        .unwrap();

    assert_abs_eq(
        pld.epsilon_at(1e-5),
        expected.epsilon_at(1e-5),
        1e-14,
        "cached repeat epsilon_at",
    );
}
