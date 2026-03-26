//! Rectified (clamped) Gaussian mechanism PLD constructor.
//!
//! The rectified Gaussian mechanism samples noise from N(0, σ²) and then
//! **clamps** it to [−R·σ, R·σ]. This produces a mixed distribution with:
//! - Continuous density on the interior (−R·σ, R·σ)
//! - Point masses at the boundaries ±R·σ
//!
//! Because clamping is post-processing of the Gaussian mechanism, the Data
//! Processing Inequality (DPI) guarantees that the rectified Gaussian is at
//! least as private as the standard Gaussian. The exact PLD is strictly
//! tighter for finite radius R.
//!
//! # Parameters
//!
//! - `noise_multiplier` σ/Δ — noise standard deviation divided by sensitivity
//! - `radius` R — support half-width in units of σ (e.g., R=3 → support [−3σ, 3σ])
//!
//! # References
//!
//! - Hu, Zheng, Li (2024). "Privacy Amplification for the Gaussian Mechanism
//!   via Bounded Support." arXiv:2403.05598.
//! - Chen & Hale (2024). "The Bounded Gaussian Mechanism for Differential
//!   Privacy." J. Privacy and Confidentiality, 14(1).

use crate::discretization::{discretize_symmetric_mechanism, DiscretizationConfig, EpsilonBounds};
use crate::error::{PldError, Result};
use crate::pld::PrivacyLossDistribution;
use statrs::distribution::{ContinuousCDF, Normal};

use super::MIN_NOISE_MULTIPLIER;

/// Minimum supported radius (sigma units).
const MIN_RADIUS: f64 = 0.1;

/// Maximum supported radius (sigma units). Beyond this the mechanism is
/// indistinguishable from the standard Gaussian.
const MAX_RADIUS: f64 = 100.0;

/// Compute the PLD for a rectified (clamped) Gaussian mechanism.
///
/// The mechanism adds noise `N(0, σ²)` clamped to `[−R·σ, R·σ]` to a
/// unit-sensitivity query. Under Add/Remove adjacency, the privacy loss
/// distribution accounting for the point masses at the boundaries is
/// strictly tighter than the standard Gaussian PLD.
///
/// # Arguments
///
/// * `noise_multiplier` — σ/Δ ratio, must be >= 0.1
/// * `radius` — support half-width in sigma units, must be in \[0.1, 100\]
/// * `config` — discretization configuration for PLD grid
///
/// # Errors
///
/// Returns `InvalidParameter` if parameters are out of range.
pub fn rectified_gaussian_pld(
    noise_multiplier: f64,
    radius: f64,
    config: &DiscretizationConfig,
) -> Result<PrivacyLossDistribution> {
    if noise_multiplier < MIN_NOISE_MULTIPLIER {
        return Err(PldError::InvalidParameter(format!(
            "noise_multiplier must be >= {}, got {}",
            MIN_NOISE_MULTIPLIER, noise_multiplier
        )));
    }
    if !(MIN_RADIUS..=MAX_RADIUS).contains(&radius) {
        return Err(PldError::InvalidParameter(format!(
            "radius must be in [{}, {}], got {}",
            MIN_RADIUS, MAX_RADIUS, radius
        )));
    }

    let sigma = noise_multiplier;
    let sensitivity = 1.0;
    let bounds = rectified_gaussian_epsilon_bounds(sigma, sensitivity, radius, config);
    let tail_budget = config.tail_mass_truncation / 2.0;

    discretize_symmetric_mechanism(config, bounds, |epsilon| {
        rectified_gaussian_delta_at(sigma, sensitivity, radius, epsilon)
    })
    .map(|pld| pld.with_tail_budgets(tail_budget, tail_budget))
}

/// Hockey-stick divergence δ(ε) for the rectified Gaussian mechanism.
///
/// The rectified Gaussian centered at `μ` with std `σ` and radius `R` has PDF:
///
/// ```text
/// f(x; μ) = φ((x−μ)/σ)/σ   for x ∈ (−Rσ, Rσ)         (interior)
///         = Φ((−Rσ−μ)/σ)     at x = −Rσ  (left point mass)
///         = 1−Φ((Rσ−μ)/σ)    at x = Rσ   (right point mass)
/// ```
///
/// Under Add/Remove adjacency with sensitivity Δ=1, we compare
/// f(x; 0) vs f(x; 1). The hockey-stick divergence is:
///
/// ```text
/// δ(ε) = ∫ max(0, f(x;0) − e^ε · f(x;1)) dx
///       + max(0, p_L(0) − e^ε · p_L(1))     (left point mass)
///       + max(0, p_R(0) − e^ε · p_R(1))     (right point mass)
/// ```
///
/// The interior integral reduces to standard Gaussian CDF evaluations at
/// the crossover point where `f(x;0)/f(x;1) = e^ε`.
fn rectified_gaussian_delta_at(sigma: f64, sensitivity: f64, radius: f64, epsilon: f64) -> f64 {
    let n01 = Normal::new(0.0, 1.0).unwrap();
    let sigma_sq = sigma * sigma;
    let r_abs = radius * sigma; // absolute radius

    // Crossover point: log(f(x;0)/f(x;1)) = ε for the interior density.
    // For Gaussian interior: ε = Δ·(Δ/2 − x)/σ²
    //   ⟹ x_cross = Δ/2 − ε·σ²/Δ
    let x_cross = sensitivity / 2.0 - epsilon * sigma_sq / sensitivity;

    // --- Interior contribution ---
    // δ_interior = ∫_{x_lo}^{x_hi} (f(x;0) − e^ε · f(x;1)) dx
    // where the integrand is positive when x < x_cross (for the Gaussian part)
    // Integration limits are clipped to [−Rσ, Rσ].

    // Interior: integrate f(x;0) − e^ε·f(x;1) over (lower, x_cross) ∩ (−Rσ, Rσ)
    let int_lower = -r_abs;
    let int_upper = x_cross.min(r_abs); // clamp crossover to domain

    let delta_interior = if int_upper > int_lower {
        // ∫_{a}^{b} f(x;0) dx = Φ((b)/σ) − Φ((a)/σ)
        let cdf_0_upper = n01.cdf(int_upper / sigma);
        let cdf_0_lower = n01.cdf(int_lower / sigma);
        let mass_p0 = cdf_0_upper - cdf_0_lower;

        // ∫_{a}^{b} f(x;1) dx = Φ((b−Δ)/σ) − Φ((a−Δ)/σ)
        let cdf_1_upper = n01.cdf((int_upper - sensitivity) / sigma);
        let cdf_1_lower = n01.cdf((int_lower - sensitivity) / sigma);
        let mass_p1 = cdf_1_upper - cdf_1_lower;

        (mass_p0 - epsilon.exp() * mass_p1).max(0.0)
    } else {
        0.0
    };

    // --- Left point mass contribution ---
    // p_L(μ) = Φ((−Rσ − μ)/σ)
    let p_l0 = n01.cdf(-radius); // Φ(−R)
    let p_l1 = n01.cdf(-radius - sensitivity / sigma); // Φ(−R − Δ/σ)
    let delta_left = (p_l0 - epsilon.exp() * p_l1).max(0.0);

    // --- Right point mass contribution ---
    // p_R(μ) = 1 − Φ((Rσ − μ)/σ) = Φ((μ − Rσ)/σ)
    let p_r0 = n01.cdf(-radius); // 1 − Φ(R) = Φ(−R)
    let p_r1 = n01.cdf(sensitivity / sigma - radius); // Φ(Δ/σ − R)
    let delta_right = (p_r0 - epsilon.exp() * p_r1).max(0.0);

    delta_interior + delta_left + delta_right
}

/// Epsilon bounds for the rectified Gaussian mechanism.
///
/// The privacy loss is bounded by the maximum of:
/// - Interior: same as standard Gaussian but domain-limited
/// - Left point mass: log(p_L(0)/p_L(1))
/// - Right point mass: log(p_R(0)/p_R(1))
///
/// The maximum privacy loss occurs either at the domain boundary or at the
/// point mass with the largest density ratio.
fn rectified_gaussian_epsilon_bounds(
    sigma: f64,
    sensitivity: f64,
    radius: f64,
    config: &DiscretizationConfig,
) -> EpsilonBounds {
    let n01 = Normal::new(0.0, 1.0).unwrap();
    let sigma_sq = sigma * sigma;
    let r_abs = radius * sigma;

    // Interior epsilon range (Gaussian privacy loss, clamped to domain)
    // Max interior privacy loss at x = −Rσ (leftmost point):
    //   ε_int_max = Δ·(Δ/2 + Rσ) / σ²
    let eps_int_upper = sensitivity * (sensitivity / 2.0 + r_abs) / sigma_sq;
    // Min interior privacy loss at x = Rσ:
    //   ε_int_min = Δ·(Δ/2 − Rσ) / σ²
    let eps_int_lower = sensitivity * (sensitivity / 2.0 - r_abs) / sigma_sq;

    // Point mass privacy losses:
    // Left: log(Φ(−R) / Φ(−R − Δ/σ))
    let p_l0 = n01.cdf(-radius);
    let p_l1 = n01.cdf(-radius - sensitivity / sigma);
    let eps_left = if p_l1 > 0.0 {
        (p_l0 / p_l1).ln()
    } else {
        f64::INFINITY
    };

    // Right: log(Φ(−R) / Φ(Δ/σ − R))
    // Note: Δ/σ − R > −R always, so p_r1 ≥ p_r0. The privacy loss here
    // is always ≤ 0.  When both underflow to zero the point mass is
    // negligible—use 0.0 rather than ±∞.
    let p_r0 = n01.cdf(-radius);
    let p_r1 = n01.cdf(sensitivity / sigma - radius);
    let eps_right = if p_r0 > 0.0 && p_r1 > 0.0 {
        (p_r0 / p_r1).ln()
    } else if p_r0 > 0.0 {
        // p_r0 > 0 but p_r1 underflowed (shouldn't happen since p_r1 >= p_r0)
        f64::INFINITY
    } else {
        // Both zero: point mass is negligible
        0.0
    };

    // Take the envelope of all components
    let epsilon_upper = eps_int_upper.max(eps_left).max(eps_right);
    let epsilon_lower = eps_int_lower.min(-eps_left).min(-eps_right);

    // Apply tail mass truncation to get finite bounds
    let log_mass = config.log_mass_truncation_bound;
    let half_mass = 0.5 * log_mass.exp();
    let z = n01.inverse_cdf(half_mass);
    let gauss_eps_upper = sensitivity * (0.5 * sensitivity - sigma * z) / sigma_sq;

    // Use the tighter of mechanism-specific and Gaussian-derived bounds
    EpsilonBounds {
        epsilon_lower: epsilon_lower.max(-gauss_eps_upper),
        epsilon_upper: epsilon_upper.min(gauss_eps_upper),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> DiscretizationConfig {
        DiscretizationConfig::default()
    }

    #[test]
    fn test_rectified_rejects_below_min_nm() {
        assert!(rectified_gaussian_pld(0.09, 3.0, &default_config()).is_err());
    }

    #[test]
    fn test_rectified_accepts_high_nm() {
        assert!(rectified_gaussian_pld(5.0, 3.0, &default_config()).is_ok());
    }

    #[test]
    fn test_rectified_rejects_below_min_radius() {
        assert!(rectified_gaussian_pld(0.5, 0.05, &default_config()).is_err());
    }

    #[test]
    fn test_rectified_rejects_above_max_radius() {
        assert!(rectified_gaussian_pld(0.5, 101.0, &default_config()).is_err());
    }

    #[test]
    fn test_rectified_boundary_params() {
        let cfg = default_config();
        assert!(rectified_gaussian_pld(MIN_NOISE_MULTIPLIER, MIN_RADIUS, &cfg).is_ok());
        assert!(rectified_gaussian_pld(1.2, MAX_RADIUS, &cfg).is_ok());
    }

    /// Rectified Gaussian ε ≤ standard Gaussian ε (DPI guarantee).
    #[test]
    fn test_rectified_tighter_than_gaussian() {
        let cfg = default_config();
        for &nm in &[0.25, 0.5, 0.8, 1.0] {
            let eps_gauss = crate::mechanisms::gaussian_pld(nm, &cfg)
                .unwrap()
                .epsilon_at(1e-5);
            for &r in &[1.0, 3.0, 5.0, 10.0] {
                let eps_rect = rectified_gaussian_pld(nm, r, &cfg)
                    .unwrap()
                    .epsilon_at(1e-5);
                assert!(
                    eps_rect <= eps_gauss + 1e-6,
                    "Rectified(σ={}, R={}) ε={:.6} should be ≤ Gaussian ε={:.6}",
                    nm,
                    r,
                    eps_rect,
                    eps_gauss
                );
            }
        }
    }

    /// Larger radius → closer to Gaussian → higher ε (less private).
    ///
    /// Bounded support caps the maximum privacy loss. Smaller radius =
    /// tighter bound on privacy loss = lower ε at small δ.
    #[test]
    fn test_rectified_radius_monotonicity() {
        let cfg = default_config();
        let radii = [1.0, 2.0, 3.0, 5.0, 10.0, 50.0];
        let epsilons: Vec<f64> = radii
            .iter()
            .map(|&r| {
                rectified_gaussian_pld(0.5, r, &cfg)
                    .unwrap()
                    .epsilon_at(1e-5)
            })
            .collect();
        for w in epsilons.windows(2) {
            assert!(
                w[0] <= w[1] + 1e-6,
                "Larger radius should give higher ε (closer to Gaussian): {} vs {}",
                w[0],
                w[1]
            );
        }
    }

    /// At very large radius, rectified ≈ standard Gaussian.
    #[test]
    fn test_rectified_converges_to_gaussian() {
        let cfg = default_config();
        let eps_gauss = crate::mechanisms::gaussian_pld(0.5, &cfg)
            .unwrap()
            .epsilon_at(1e-5);
        let eps_rect = rectified_gaussian_pld(0.5, 50.0, &cfg)
            .unwrap()
            .epsilon_at(1e-5);
        assert!(
            (eps_rect - eps_gauss).abs() < 0.01,
            "At R=50, rectified should ≈ Gaussian: {} vs {}",
            eps_rect,
            eps_gauss
        );
    }

    /// Noise multiplier monotonicity: higher σ → lower ε.
    #[test]
    fn test_rectified_nm_monotonicity() {
        let cfg = default_config();
        let nms = [0.1, 0.25, 0.5, 0.8, 1.0, 1.2];
        let epsilons: Vec<f64> = nms
            .iter()
            .map(|&nm| {
                rectified_gaussian_pld(nm, 3.0, &cfg)
                    .unwrap()
                    .epsilon_at(1e-5)
            })
            .collect();
        for w in epsilons.windows(2) {
            assert!(
                w[0] > w[1] - 1e-6,
                "Higher σ should give lower ε: {} vs {}",
                w[0],
                w[1]
            );
        }
    }

    /// Delta at zero epsilon is meaningful (advantage).
    #[test]
    fn test_rectified_delta_at_zero() {
        let pld = rectified_gaussian_pld(0.5, 3.0, &default_config()).unwrap();
        let delta = pld.delta_at(0.0);
        // Should be positive but less than Gaussian delta at 0
        assert!(delta > 0.0 && delta < 1.0, "delta(0) = {}", delta);
    }
}
