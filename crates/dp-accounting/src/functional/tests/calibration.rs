//! Calibration tests
//!
//! Primary validation: calibrate a process, then verify the calibrated process
//! actually satisfies the privacy constraint (round-trip check).
//!
//! Property tests: monotonicity (stricter target -> more noise).

use super::*;
use crate::functional::calibrate::{self, CalibrateConfig};

// =========================================================================
// Round-trip helpers: calibrate -> verify process satisfies constraint
// =========================================================================

fn calibrate_and_verify_epsilon(target_eps: f64, delta: f64) -> calibrate::CalibrateResult {
    calibrate_and_verify_epsilon_with(target_eps, delta, CalibrateConfig::default())
}

fn calibrate_and_verify_epsilon_with(
    target_eps: f64,
    delta: f64,
    config: CalibrateConfig,
) -> calibrate::CalibrateResult {
    let tol = config.tolerance;
    let result = calibrate::calibrate(
        calibrate::target_epsilon_at(target_eps, delta),
        gaussian,
        config,
    )
    .unwrap();

    assert!(result.converged, "calibration did not converge");
    let achieved = gaussian(result.param).unwrap().epsilon_at(delta).unwrap();
    assert!(
        achieved.is_finite(),
        "nm={:.4} gives inf epsilon at delta={:e}",
        result.param,
        delta
    );
    assert!(
        achieved <= target_eps + tol,
        "nm={:.4} gives ε={:.6} > target {:.1} + tol={:e} at δ={:e}",
        result.param,
        achieved,
        target_eps,
        tol,
        delta
    );
    result
}

fn calibrate_and_verify_delta(target_delta: f64, epsilon: f64) -> calibrate::CalibrateResult {
    let config = CalibrateConfig::default();
    let tol = config.tolerance;
    let result = calibrate::calibrate(
        calibrate::target_delta_at(target_delta, epsilon),
        gaussian,
        config,
    )
    .unwrap();

    assert!(result.converged, "calibration did not converge");
    let achieved = gaussian(result.param).unwrap().delta_at(epsilon).unwrap();
    assert!(
        achieved <= target_delta + tol,
        "nm={:.4} gives δ={:.6e} > target {:.1e} + tol={:e} at ε={:.1}",
        result.param,
        achieved,
        target_delta,
        tol,
        epsilon
    );
    result
}

fn calibrate_and_verify_advantage(target_adv: f64) -> calibrate::CalibrateResult {
    let config = CalibrateConfig::default();
    let tol = config.tolerance;
    let result =
        calibrate::calibrate(calibrate::target_advantage(target_adv), gaussian, config).unwrap();

    assert!(result.converged, "calibration did not converge");
    let achieved = gaussian(result.param).unwrap().advantage().unwrap();
    assert!(
        achieved <= target_adv + tol,
        "nm={:.4} gives advantage={:.6} > target {:.2} + tol={:e}",
        result.param,
        achieved,
        target_adv,
        tol
    );
    result
}

fn calibrate_and_verify_beta(target_beta: f64, alpha: f64) -> calibrate::CalibrateResult {
    // Beta peaks around nm≈5, but gaussian() validates [0.1, 1.2].
    // Targets must be achievable within [0.1, 1.2].
    let config = CalibrateConfig::default();
    let tol = config.tolerance;
    let result = calibrate::calibrate(
        calibrate::target_beta_at(target_beta, alpha),
        gaussian,
        config,
    )
    .unwrap();

    assert!(result.converged, "calibration did not converge");
    let achieved = gaussian(result.param).unwrap().beta_at(alpha).unwrap();
    assert!(
        achieved >= target_beta - tol,
        "nm={:.4} gives β={:.6} < target {:.2} - tol={:e} at α={:.2}",
        result.param,
        achieved,
        target_beta,
        tol,
        alpha
    );
    result
}

fn calibrate_and_verify_risk(target_risk: f64, prior: f64) -> calibrate::CalibrateResult {
    // Risk is reversed: higher risk = more private (attacker errs more).
    // gaussian() validates [0.1, 1.2], so targets must be feasible within that range.
    let config = CalibrateConfig::default();
    let tol = config.tolerance;
    let result = calibrate::calibrate(
        calibrate::target_risk_at(target_risk, prior),
        gaussian,
        config,
    )
    .unwrap();

    assert!(result.converged, "calibration did not converge");
    let achieved = gaussian(result.param).unwrap().risk_at(prior).unwrap();
    assert!(
        achieved >= target_risk - tol,
        "nm={:.4} gives risk={:.6} < target {:.2} - tol={:e} at prior={:.2}",
        result.param,
        achieved,
        target_risk,
        tol,
        prior
    );
    result
}

// =========================================================================
// Round-trip tests: epsilon
// =========================================================================

#[test]
fn test_roundtrip_epsilon_4_0_delta_1e5() {
    // ε ≤ 4.0 at δ = 1e-5 needs nm ≈ 1.0, feasible within [0.1, 1.2]
    calibrate_and_verify_epsilon(4.0, 1e-5);
}

#[test]
fn test_roundtrip_epsilon_4_0_delta_1e6() {
    // ε ≤ 4.0 at δ = 1e-6 needs nm ≈ 1.2, feasible within [0.1, 1.2]
    // (nm=1.2 gives ε≈3.98 at δ=1e-6)
    calibrate_and_verify_epsilon(4.0, 1e-6);
}

#[test]
fn test_roundtrip_epsilon_5_0_delta_1e5() {
    calibrate_and_verify_epsilon(5.0, 1e-5);
}

#[test]
fn test_roundtrip_epsilon_8_0_delta_1e6() {
    calibrate_and_verify_epsilon(8.0, 1e-6);
}

// =========================================================================
// Round-trip tests: delta
// =========================================================================

#[test]
fn test_roundtrip_delta_0_1_epsilon_1() {
    // δ ≤ 0.1 at ε = 1.0 is feasible within [0.1, 1.2]
    calibrate_and_verify_delta(0.1, 1.0);
}

#[test]
fn test_roundtrip_delta_0_01_epsilon_3() {
    // δ ≤ 0.01 at ε = 3.0 is feasible within [0.1, 1.2]
    calibrate_and_verify_delta(0.01, 3.0);
}

// =========================================================================
// Round-trip tests: advantage
// =========================================================================

#[test]
fn test_roundtrip_advantage_0_5() {
    calibrate_and_verify_advantage(0.5);
}

#[test]
fn test_roundtrip_advantage_0_35() {
    // adv ≤ 0.35 is feasible within [0.1, 1.2] (nm=1.2 gives adv≈0.32)
    calibrate_and_verify_advantage(0.35);
}

// =========================================================================
// Round-trip tests: beta
// =========================================================================

#[test]
fn test_roundtrip_beta_0_7_alpha_0_05() {
    // β ≥ 0.7 at α = 0.05 is feasible within [0.1, 1.2]
    calibrate_and_verify_beta(0.7, 0.05);
}

#[test]
fn test_roundtrip_beta_0_6_alpha_0_1() {
    // β ≥ 0.6 at α = 0.1 is feasible within [0.1, 1.2]
    calibrate_and_verify_beta(0.6, 0.1);
}

// =========================================================================
// Round-trip tests: risk
// =========================================================================

#[test]
fn test_roundtrip_risk_0_3_prior_0_5() {
    calibrate_and_verify_risk(0.3, 0.5);
}

#[test]
fn test_roundtrip_risk_0_15_prior_0_5() {
    calibrate_and_verify_risk(0.15, 0.5);
}

// =========================================================================
// Property tests: stricter target -> more noise
// =========================================================================

#[test]
fn test_stricter_epsilon_needs_more_noise() {
    // Both targets must be feasible within [0.1, 1.2]
    let loose = calibrate_and_verify_epsilon(8.0, 1e-5);
    let tight = calibrate_and_verify_epsilon(5.0, 1e-5);
    assert!(
        tight.param > loose.param,
        "ε=5.0 (nm={}) should need more noise than ε=8.0 (nm={})",
        tight.param,
        loose.param,
    );
}

#[test]
fn test_smaller_delta_needs_more_noise() {
    // Both targets must be feasible within [0.1, 1.2]
    let loose = calibrate_and_verify_epsilon(5.0, 1e-3);
    let tight = calibrate_and_verify_epsilon(5.0, 1e-5);
    assert!(
        tight.param > loose.param,
        "δ=1e-5 (nm={}) should need more noise than δ=1e-3 (nm={})",
        tight.param,
        loose.param,
    );
}

#[test]
fn test_stricter_advantage_needs_more_noise() {
    // Both targets must be feasible within [0.1, 1.2]
    let loose = calibrate_and_verify_advantage(0.5);
    let tight = calibrate_and_verify_advantage(0.35);
    assert!(
        tight.param > loose.param,
        "adv=0.35 (nm={}) should need more noise than adv=0.5 (nm={})",
        tight.param,
        loose.param,
    );
}

#[test]
fn test_stricter_beta_needs_more_noise() {
    // Higher beta target = more private = more noise needed.
    // Both targets must be feasible within [0.1, 1.2].
    let loose = calibrate_and_verify_beta(0.5, 0.05);
    let tight = calibrate_and_verify_beta(0.6, 0.05);
    assert!(
        tight.param > loose.param,
        "β≥0.6 (nm={}) should need more noise than β≥0.5 (nm={})",
        tight.param,
        loose.param,
    );
}

#[test]
fn test_stricter_risk_needs_more_noise() {
    // Higher risk target = more private = more noise needed (reversed metric)
    let loose = calibrate_and_verify_risk(0.15, 0.5);
    let tight = calibrate_and_verify_risk(0.3, 0.5);
    assert!(
        tight.param > loose.param,
        "risk≥0.3 (nm={}) should need more noise than risk≥0.15 (nm={})",
        tight.param,
        loose.param,
    );
}

// =========================================================================
// Reference tests: pre-computed calibrated noise multipliers
//
// Removed: previous reference values were computed with wider bounds that
// allowed noise multipliers exceeding the [0.1, 1.2] validation range.
// These will be regenerated once all targets are finalized.
// =========================================================================
