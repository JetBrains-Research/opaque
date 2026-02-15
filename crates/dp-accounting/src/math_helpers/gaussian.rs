//! Analytic Gaussian mechanism functions
//!
//! Closed-form and numerical routines for the Gaussian mechanism's
//! privacy loss properties, used by both the functional and legacy APIs.

use statrs::distribution::{ContinuousCDF, Normal};

/// Analytic delta for the Gaussian mechanism at a given epsilon.
///
/// Computes the ε-hockey stick divergence:
/// ```text
/// δ(ε) = Φ(0.5Δ̃ − ε/Δ̃) − eᵉ · Φ(0.5Δ̃ − ε/Δ̃ − Δ̃)
/// ```
/// where Δ̃ = sensitivity / sigma is the standardized sensitivity.
///
/// Returns 0.0 when the result would be negative (numerical noise).
pub(crate) fn gaussian_delta_at(delta_tilde: f64, epsilon: f64) -> f64 {
    let standard_normal = Normal::new(0.0, 1.0).unwrap();
    let x_upper = 0.5 * delta_tilde - epsilon / delta_tilde;
    (standard_normal.cdf(x_upper) - epsilon.exp() * standard_normal.cdf(x_upper - delta_tilde))
        .max(0.0)
}

/// Find the epsilon where the Gaussian delta function equals `target`.
///
/// Uses bisection on [`gaussian_delta_at`] to find the tight upper bound
/// on the epsilon range needed for PLD discretization. The analytic
/// approximation `ε_safe = 0.5Δ̃² − Δ̃·Φ⁻¹(target)` provides the
/// initial upper bound for the search.
///
/// # Arguments
///
/// * `delta_tilde` - Standardized sensitivity Δ/σ (= 1/σ for unit sensitivity)
/// * `target` - Target delta value (tail mass threshold)
///
/// # Returns
///
/// The smallest epsilon where `delta(ε) ≤ target`.
pub(crate) fn gaussian_epsilon_for_delta(delta_tilde: f64, target: f64) -> f64 {
    let standard_normal = Normal::new(0.0, 1.0).unwrap();
    let z_tail = standard_normal.inverse_cdf(target);

    // Analytic upper bound (from dropping the second term in delta formula)
    let hi_init = 0.5 * delta_tilde * delta_tilde - delta_tilde * z_tail;

    // Bisect between 0 and the analytic bound
    let mut lo = 0.0_f64;
    let mut hi = hi_init;
    for _ in 0..100 {
        let mid = (lo + hi) / 2.0;
        if gaussian_delta_at(delta_tilde, mid) > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    hi
}

/// Find the epsilon where the Gaussian beta function equals `target`.
///
/// The beta function (reverse hockey-stick divergence) is:
/// ```text
/// β(ε) = δ_reverse(ε) = gaussian_delta_at(Δ̃, −ε)
/// ```
/// This governs the left tail of the PLD and determines accuracy of
/// `beta_at()` and `risk_at()` metrics.
///
/// Uses the same bisection approach as [`gaussian_epsilon_for_delta`],
/// searching for positive `E` where `gaussian_delta_at(Δ̃, −E) = target`,
/// then returning `−E` as the epsilon lower bound.
///
/// # Arguments
///
/// * `delta_tilde` - Standardized sensitivity Δ/σ (= 1/σ for unit sensitivity)
/// * `target` - Target beta value (left tail mass threshold)
///
/// # Returns
///
/// The negative epsilon (left bound) where `β(|ε|) ≤ target`.
pub(crate) fn gaussian_epsilon_for_beta(delta_tilde: f64, target: f64) -> f64 {
    let standard_normal = Normal::new(0.0, 1.0).unwrap();
    let z_tail = standard_normal.inverse_cdf(target);

    // Same analytic upper bound works: beta is symmetric to delta
    let hi_init = 0.5 * delta_tilde * delta_tilde - delta_tilde * z_tail;

    // Bisect: find positive E where gaussian_delta_at(dt, -E) = target
    let mut lo = 0.0_f64;
    let mut hi = hi_init;
    for _ in 0..100 {
        let mid = (lo + hi) / 2.0;
        if gaussian_delta_at(delta_tilde, -mid) > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    -hi // Return as negative epsilon (left bound)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_delta_at_zero_epsilon() {
        // delta(0) = Phi(0.5*dt) - Phi(0.5*dt - dt) = Phi(0.5*dt) - Phi(-0.5*dt)
        // For dt=1: Phi(0.5) - Phi(-0.5) ≈ 0.383
        let d = gaussian_delta_at(1.0, 0.0);
        assert!((d - 0.3829).abs() < 0.001);
    }

    #[test]
    fn test_gaussian_delta_at_large_epsilon() {
        // For large epsilon, delta should be near zero
        let d = gaussian_delta_at(1.0, 10.0);
        assert!(d < 1e-10);
    }

    #[test]
    fn test_gaussian_epsilon_for_delta_monotone() {
        // Smaller target -> larger epsilon
        let dt = 2.0;
        let e1 = gaussian_epsilon_for_delta(dt, 1e-6);
        let e2 = gaussian_epsilon_for_delta(dt, 1e-10);
        assert!(e2 > e1);
    }

    #[test]
    fn test_gaussian_epsilon_for_delta_accuracy() {
        // The returned epsilon should satisfy delta(eps) <= target
        let dt = 10.0; // sigma=0.1
        let target = 0.5 * (-32.0_f64).exp();
        let eps = gaussian_epsilon_for_delta(dt, target);
        let actual_delta = gaussian_delta_at(dt, eps);
        assert!(actual_delta <= target);
        // And delta just above should exceed target
        let actual_delta_below = gaussian_delta_at(dt, eps - 1.0);
        assert!(actual_delta_below > target);
    }

    #[test]
    fn test_gaussian_epsilon_for_beta_is_negative() {
        let dt = 2.0;
        let eps = gaussian_epsilon_for_beta(dt, 1e-6);
        assert!(eps < 0.0, "beta bound should be negative, got {}", eps);
    }

    #[test]
    fn test_gaussian_epsilon_for_beta_accuracy() {
        // eps = gaussian_epsilon_for_beta(dt, target) returns a negative value.
        // It was found by bisecting for positive E where gaussian_delta_at(dt, -E) = target.
        // The returned eps = -E, so to verify: gaussian_delta_at(dt, -eps) <= target.
        let dt = 10.0;
        let target = 1e-6;
        let eps = gaussian_epsilon_for_beta(dt, target);
        let actual_beta = gaussian_delta_at(dt, -eps);
        assert!(
            actual_beta <= target,
            "beta at eps={} is {}, should be <= {}",
            eps,
            actual_beta,
            target
        );
        // And slightly inside should exceed target
        let actual_beta_inside = gaussian_delta_at(dt, -eps - 1.0);
        assert!(actual_beta_inside > target);
    }

    #[test]
    fn test_gaussian_epsilon_for_beta_monotone() {
        // Smaller target -> more negative epsilon (wider left bound)
        let dt = 2.0;
        let e1 = gaussian_epsilon_for_beta(dt, 1e-4);
        let e2 = gaussian_epsilon_for_beta(dt, 1e-8);
        assert!(
            e2 < e1,
            "smaller target should give more negative bound: {} vs {}",
            e2,
            e1
        );
    }
}
