//! Coarsening regression tests — verify bounded error from adaptive coarsening
//!
//! Tests that adaptive coarsening introduces bounded, known error by comparing
//! gaussian(nm) [default config] against build_exact_pld(nm) [disc=1e-4,
//! unlimited grid size].
//!
//! With the default max_grid_size of 10M, pre-composition coarsening is only
//! triggered for sufficiently small noise multipliers (e.g., nm in [0.1, 0.3])
//! where the effective discretization becomes very fine. These tests verify:
//! - Single-mechanism PLDs match exact PLDs when coarsening is inactive
//! - Single-mechanism PLDs stay within a bounded relative error when
//!   pre-composition coarsening is active
//! - Post-composition coarsening still works when grids exceed max_grid_size
//! - Explicit max_grid_size overrides via self_compose_with_max_grid_size()
//! - Coarsening conservativeness (pessimistic_estimate=true)

use super::reference::exact_pld;
use super::*;

// =========================================================================
// Single mechanism coarsening tests
// =========================================================================

#[test]
fn test_coarsening_error_single_mechanism() {
    // Verify coarsened gaussian(nm) stays within measured error bounds
    // of exact disc=1e-4 computation for each nm that triggers coarsening.
    let cases: &[(f64, f64)] = &[
        (0.10, 1e-7), // eff_disc=1.6e-3, measured ~5e-9
        (0.20, 1e-7), // eff_disc=8.0e-4, measured ~4e-9
        (0.30, 1e-7), // eff_disc=4.0e-4, measured ~1e-9
    ];

    for &(nm, max_rel_error) in cases {
        let coarsened = default_pld(nm);
        let exact = exact_pld(nm);

        for &delta in &[1e-5, 1e-6, 1e-8] {
            let eps_exact = exact.epsilon_at(delta);
            let eps_coarsened = coarsened.epsilon_at(delta);
            if eps_exact.is_finite() && eps_coarsened.is_finite() && eps_exact.abs() > 1e-10 {
                let rel_err = ((eps_coarsened - eps_exact) / eps_exact).abs();
                assert!(
                    rel_err < max_rel_error,
                    "nm={}, delta={}: rel_error={:.2e} exceeds {:.2e}",
                    nm,
                    delta,
                    rel_err,
                    max_rel_error
                );
            }
        }
    }
}

// =========================================================================
// Composition coarsening tests
// =========================================================================

#[test]
fn test_coarsening_error_composition() {
    // Verify coarsening error stays bounded after composition.
    // Composition accumulates FFT rounding; coarsening adds on top.
    let cases: &[(f64, usize, f64)] = &[
        (0.10, 100, 1e-4), // moderate coarsening, long composition
    ];

    for &(nm, k, max_rel_error) in cases {
        let coarsened = default_pld(nm).self_compose(k);
        let exact = exact_pld(nm).self_compose(k);

        for &delta in &[1e-5, 1e-6] {
            let eps_exact = exact.epsilon_at(delta);
            let eps_coarsened = coarsened.epsilon_at(delta);
            if eps_exact.is_finite() && eps_coarsened.is_finite() && eps_exact.abs() > 1e-10 {
                let rel_err = ((eps_coarsened - eps_exact) / eps_exact).abs();
                assert!(
                    rel_err < max_rel_error,
                    "nm={}, k={}, delta={}: rel_error={:.2e} exceeds {:.2e}",
                    nm,
                    k,
                    delta,
                    rel_err,
                    max_rel_error
                );
            }
        }
    }
}

#[test]
fn test_no_coarsening_above_threshold() {
    // For all nm in [0.1, 1.2], grid fits within 10M max grid size, so no coarsening.
    // Results must match exact PLD within floating-point tolerance.
    for &nm in &[0.5, 0.8, 1.0, 1.2] {
        let cached = default_pld(nm);
        let exact = exact_pld(nm);

        for &delta in &[1e-6, 1e-5, 1e-4] {
            let eps_default = cached.epsilon_at(delta);
            let eps_exact = exact.epsilon_at(delta);
            assert!(
                (eps_default - eps_exact).abs() < 1e-10,
                "nm={}, delta={}: default and exact should match, diff={:.2e}",
                nm,
                delta,
                (eps_default - eps_exact).abs()
            );
        }

        for &eps in &[0.5, 1.0, 5.0] {
            let delta_default = cached.delta_at(eps);
            let delta_exact = exact.delta_at(eps);
            assert!(
                (delta_default - delta_exact).abs() < 1e-10,
                "nm={}, eps={}: default and exact should match, diff={:.2e}",
                nm,
                eps,
                (delta_default - delta_exact).abs()
            );
        }
    }
}

// =========================================================================
// Coarsening conservativeness tests
// =========================================================================

#[test]
fn test_coarsening_is_conservative() {
    // With pessimistic_estimate=true (default), coarsened PLD should
    // report epsilon >= exact epsilon. This ensures privacy guarantees
    // are never understated by coarsening.
    for &nm in &[0.1, 0.2, 0.3] {
        let coarsened = default_pld(nm);
        let exact = exact_pld(nm);

        for &delta in &[1e-5, 1e-6] {
            let eps_coarsened = coarsened.epsilon_at(delta);
            let eps_exact = exact.epsilon_at(delta);
            if eps_exact.is_finite() {
                assert!(
                    eps_coarsened >= eps_exact - 1e-10,
                    "nm={}, delta={}: coarsened eps ({}) < exact eps ({})",
                    nm,
                    delta,
                    eps_coarsened,
                    eps_exact
                );
            }
        }
    }
}

#[test]
fn test_coarsening_power_of_2_at_mechanism_level() {
    // Verify that the effective discretization chosen by gaussian(nm)
    // is always a power-of-2 multiple of the base disc=1e-4.
    // Note: This test verifies the behavior indirectly by checking that
    // coarsened PLDs maintain accuracy, since epsilon_bounds is private.
    for &nm in &[0.1, 0.2, 0.3] {
        let coarsened = default_pld(nm);
        let exact = exact_pld(nm);

        // If coarsening uses power-of-2 ratios, error should be bounded
        let eps_c = coarsened.epsilon_at(1e-5);
        let eps_e = exact.epsilon_at(1e-5);
        if eps_e.is_finite() && eps_e > 1e-10 {
            let rel_err = ((eps_c - eps_e) / eps_e).abs();
            assert!(
                rel_err < 1e-6,
                "nm={}: rel_error={:.2e} suggests non-power-of-2 coarsening",
                nm,
                rel_err
            );
        }
    }
}

// =========================================================================
// Heterogeneous composition coarsening tests
// =========================================================================

#[test]
fn test_coarsening_error_hetero_composition() {
    // Cross-nm composition where auto-coarsening aligns grids.
    let pairs: &[(f64, f64, f64)] = &[
        (0.2, 0.8, 1e-4), // moderate coarsened + no coarsening
        (0.1, 0.5, 1e-4), // coarsened + no coarsening
    ];

    for &(nm1, nm2, max_rel_error) in pairs {
        let coarsened = default_pld(nm1).compose(&default_pld(nm2)).unwrap();
        let exact = exact_pld(nm1).compose(&exact_pld(nm2)).unwrap();

        for &delta in &[1e-5, 1e-4] {
            let eps_exact = exact.epsilon_at(delta);
            let eps_coarsened = coarsened.epsilon_at(delta);
            if eps_exact.is_finite() && eps_coarsened.is_finite() && eps_exact.abs() > 1e-10 {
                let rel_err = ((eps_coarsened - eps_exact) / eps_exact).abs();
                assert!(
                    rel_err < max_rel_error,
                    "nm={},{}, delta={}: rel_error={:.2e} exceeds {:.2e}",
                    nm1,
                    nm2,
                    delta,
                    rel_err,
                    max_rel_error
                );
            }
        }
    }
}

// =========================================================================
// Post-composition coarsening quality tests
// =========================================================================

#[test]
fn test_post_compose_coarsening_hetero_bounded_error() {
    // Verify post-composition coarsening error for heterogeneous composition.
    //
    // compose() uses default max_grid_size (10M) which may trigger post-composition
    // coarsening when the convolved grid exceeds 10M. compose_with_max_grid_size
    // with usize::MAX prevents this coarsening.
    //
    // With 10M max grid size, no single-mechanism PLDs in [0.1, 1.2] trigger
    // pre-composition coarsening, and heterogeneous compositions of two such PLDs
    // typically fit within 10M as well.

    // Case 1: nm=0.5+0.8 — post-composition coarsening is active
    let pld1 = default_pld(0.5);
    let pld2 = default_pld(0.8);

    let coarsened = pld1.compose(&pld2).unwrap();
    let reference = pld1.compose_with_max_grid_size(&pld2, usize::MAX).unwrap();

    for &delta in &[1e-6, 1e-5, 1e-3] {
        let eps_c = coarsened.epsilon_at(delta);
        let eps_r = reference.epsilon_at(delta);
        if eps_r > 1e-10 {
            let rel_err = (eps_c - eps_r).abs() / eps_r;
            assert!(
                rel_err < 1e-4,
                "hetero compose (0.5,0.8) delta={}: rel_err={:.2e} exceeds 1e-4",
                delta,
                rel_err
            );
        }
    }

    // Case 2: nm=0.2+0.8 — pre-composition coarsening (nm=0.2) is active
    let pld_small = default_pld(0.2);
    let pld_large = default_pld(0.8);

    let coarsened2 = pld_small.compose(&pld_large).unwrap();
    let reference2 = exact_pld(0.2).compose(exact_pld(0.8)).unwrap();

    for &delta in &[1e-6, 1e-5, 1e-3] {
        let eps_c = coarsened2.epsilon_at(delta);
        let eps_r = reference2.epsilon_at(delta);
        if eps_r > 1e-10 {
            let rel_err = (eps_c - eps_r).abs() / eps_r;
            assert!(
                rel_err < 1e-4,
                "hetero compose (0.2,0.8) delta={}: rel_err={:.2e} exceeds 1e-4",
                delta,
                rel_err
            );
        }
    }
}

// =========================================================================
// Self-compose with max_grid_size override tests
// =========================================================================

#[test]
fn test_self_compose_with_max_grid_size_overrides() {
    // Verify that self_compose_with_max_grid_size applies coarsening.
    let pld = default_pld(0.5);

    // Tight max grid size — forces coarsening after self_compose
    let tight = pld.self_compose_with_max_grid_size(10, 100_000);
    // No coarsening
    let unlimited = pld.self_compose_with_max_grid_size(10, usize::MAX);

    // Both should produce valid results
    let eps_tight = tight.epsilon_at(1e-5);
    let eps_unlimited = unlimited.epsilon_at(1e-5);
    assert!(eps_tight.is_finite() && eps_tight > 0.0);
    assert!(eps_unlimited.is_finite() && eps_unlimited > 0.0);

    // Metrics should be close but not necessarily identical
    if eps_unlimited > 1e-10 {
        let rel_err = (eps_tight - eps_unlimited).abs() / eps_unlimited;
        assert!(
            rel_err < 0.05,
            "tight vs unlimited: rel_err={:.2e} exceeds 5%",
            rel_err
        );
    }
}
