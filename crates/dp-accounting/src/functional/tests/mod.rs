//! Shared test helpers and fixtures for functional API integration tests
//!
//! This module provides common utilities used across multiple integration test files,
//! including pre-computed PLDs cached via LazyLock for expensive test cases.

use crate::*;
use std::sync::LazyLock;

// =========================================================================
// Shared helper functions
// =========================================================================

/// Create a Gaussian mechanism with default configuration
pub fn default_gaussian(noise_multiplier: f64) -> Gaussian {
    gaussian(noise_multiplier).expect("invalid noise_multiplier")
}

/// Create a Gaussian PLD with default configuration (convenience wrapper)
pub fn default_pld(noise_multiplier: f64) -> PrivacyLossDistribution {
    gaussian(noise_multiplier)
        .expect("invalid noise_multiplier")
        .pld()
        .expect("PLD construction failed")
}

// =========================================================================
// Cached PLDs for expensive integration tests
// =========================================================================

/// Cached PLD for nm=1.0 (commonly used baseline)
pub static PLD_NM_1_0: LazyLock<PrivacyLossDistribution> = LazyLock::new(|| {
    gaussian(1.0)
        .expect("invalid nm")
        .pld()
        .expect("Failed to create PLD for nm=1.0")
});

/// Cached PLD for nm=0.5 (high privacy)
pub static PLD_NM_0_5: LazyLock<PrivacyLossDistribution> = LazyLock::new(|| {
    gaussian(0.5)
        .expect("invalid nm")
        .pld()
        .expect("Failed to create PLD for nm=0.5")
});

/// Cached PLD for nm=1.2 (low privacy)
pub static PLD_NM_1_2: LazyLock<PrivacyLossDistribution> = LazyLock::new(|| {
    gaussian(1.2)
        .expect("invalid nm")
        .pld()
        .expect("Failed to create PLD for nm=1.2")
});

// =========================================================================
// Principled reference tolerance derived from actual discretization
// =========================================================================

/// Compute the principled tolerance for reference value regression tests.
///
/// The Connect-the-Dots discretization introduces O(disc²) error per step,
/// accumulated linearly across k composition steps. This function computes:
///
///   tolerance = C · disc² · max(k, 1)
///
/// where:
/// - `disc` = the actual grid spacing used by the PLD (after any coarsening)
/// - `k` = number of composition steps (1 for single mechanism)
/// - `C` = 10 (safety factor for floating-point accumulation)
///
/// The caller provides `disc` from the process's actual DiscretizationConfig
/// (or from the PLD's effective discretization after coarsening), and `k`
/// from their knowledge of the test setup.
///
/// For `disc = 1e-4`:
/// - k=1:  tol = 10 · 1e-8 · 1  = 1e-7
/// - k=10: tol = 10 · 1e-8 · 10 = 1e-6
/// - k=50: tol = 10 · 1e-8 · 50 = 5e-6
/// - k=100: tol = 10 · 1e-8 · 100 = 1e-5
pub fn reference_tol(disc: f64, k: usize) -> f64 {
    let c = 10.0;
    c * disc * disc * (k.max(1) as f64)
}

/// Extract the effective discretization from a PLD (the actual grid spacing
/// after any coarsening that occurred during construction or composition).
pub fn pld_disc(pld: &PrivacyLossDistribution) -> f64 {
    pld.pmf_remove.discretization
}

/// Compute the principled tolerance for cross-validation tests that compare
/// two independent PLD implementations (e.g., old API vs new functional API).
///
/// Each PLD has O(disc²·k) error from the true answer, so their difference
/// is bounded by the sum of their individual errors. The safety factor C_xv
/// is larger than for reference tests because:
/// - Different truncation bounds (-32 vs -50) shift the PMF tails differently
/// - Metrics like beta_at and risk_at integrate the PMF, amplifying small
///   tail differences (especially at extreme alpha values like 0.001)
/// - Two independent discretizations can have worst-case constructive error
///
/// Formula: C_xv · (disc1² + disc2²) · max(k, 1)
///
/// For same-disc single-step (disc1 = disc2 = 1e-4, k = 1):
///   tolerance = 200 · 2e-8 · 1 = 4e-6
pub fn cross_validation_tol(disc1: f64, disc2: f64, k: usize) -> f64 {
    let c_xv = 200.0;
    c_xv * (disc1 * disc1 + disc2 * disc2) * (k.max(1) as f64)
}

// =========================================================================
// Test assertion helpers
// =========================================================================

/// Assert that a value is within relative tolerance
#[track_caller]
pub fn assert_relative_eq(actual: f64, expected: f64, epsilon: f64, label: &str) {
    let rel_error = if expected.abs() > 1e-100 {
        ((actual - expected) / expected).abs()
    } else {
        (actual - expected).abs()
    };

    assert!(
        rel_error <= epsilon,
        "{}: relative error {} exceeds tolerance {}\n  actual: {}\n  expected: {}",
        label,
        rel_error,
        epsilon,
        actual,
        expected
    );
}

/// Assert that a value is within absolute tolerance
#[track_caller]
pub fn assert_abs_eq(actual: f64, expected: f64, epsilon: f64, label: &str) {
    let abs_error = (actual - expected).abs();

    assert!(
        abs_error <= epsilon,
        "{}: absolute error {} exceeds tolerance {}\n  actual: {}\n  expected: {}",
        label,
        abs_error,
        epsilon,
        actual,
        expected
    );
}

// =========================================================================
// Submodules (integration test files)
// =========================================================================

#[cfg(test)]
mod smoke;

#[cfg(test)]
mod properties;

#[cfg(test)]
mod composition;

#[cfg(test)]
mod reference;

#[cfg(test)]
mod coarsening;

#[cfg(test)]
mod calibration;

#[cfg(test)]
mod adaclip;

#[cfg(test)]
mod accumulated;

#[cfg(test)]
#[cfg(feature = "serde")]
mod serialisation;
