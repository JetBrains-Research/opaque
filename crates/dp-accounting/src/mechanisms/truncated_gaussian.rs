//! Truncated (renormalized) Gaussian mechanism PLD constructor.
//!
//! The truncated Gaussian mechanism samples noise from `N(0, σ²) | [−Rσ, Rσ]`,
//! the standard Gaussian **renormalized** to the interval [−Rσ, Rσ]. Unlike
//! the rectified (clamped) variant, this produces a smooth density with no
//! point masses at the boundaries.
//!
//! The truncated Gaussian provides **strictly tighter** privacy than the
//! rectified Gaussian (which in turn is tighter than the standard Gaussian),
//! because the renormalized density concentrates more mass in the interior.
//!
//! # Parameters
//!
//! - `noise_multiplier` σ/Δ — noise standard deviation divided by sensitivity
//! - `radius` R — support half-width in units of σ (e.g., R=3 → support [−3σ, 3σ])
//!
//! # Math
//!
//! The truncated Gaussian density centered at μ is:
//!
//! ```text
//! f(x; μ) = φ((x−μ)/σ) / (σ · Z(μ))     for x ∈ [−Rσ, Rσ]
//! ```
//!
//! where `Z(μ) = Φ((Rσ−μ)/σ) − Φ((−Rσ−μ)/σ)` is the normalization constant.
//!
//! The privacy loss at point x between distributions centered at 0 and Δ:
//!
//! ```text
//! ℓ(x) = log(f(x;0)/f(x;Δ))
//!       = −Δ·x/σ² + Δ²/(2σ²) + log(Z(Δ)/Z(0))
//! ```
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

use super::{MAX_NOISE_MULTIPLIER, MIN_NOISE_MULTIPLIER};

/// Minimum supported radius (sigma units).
const MIN_RADIUS: f64 = 0.1;

/// Maximum supported radius (sigma units).
const MAX_RADIUS: f64 = 100.0;

/// Compute the PLD for a truncated (renormalized) Gaussian mechanism.
///
/// The mechanism adds noise from `N(0, σ²) | [−Rσ, Rσ]` (truncated normal)
/// to a unit-sensitivity query. The exact PLD accounts for the different
/// normalization constants between centered and shifted distributions,
/// giving strictly tighter bounds than both the standard and rectified
/// Gaussian mechanisms.
///
/// # Arguments
///
/// * `noise_multiplier` — σ/Δ ratio, must be in \[0.1, 1.2\]
/// * `radius` — support half-width in sigma units, must be in \[0.1, 100\]
/// * `config` — discretization configuration for PLD grid
///
/// # Errors
///
/// Returns `InvalidParameter` if parameters are out of range.
pub fn truncated_gaussian_pld(
    noise_multiplier: f64,
    radius: f64,
    config: &DiscretizationConfig,
) -> Result<PrivacyLossDistribution> {
    if !(MIN_NOISE_MULTIPLIER..=MAX_NOISE_MULTIPLIER).contains(&noise_multiplier) {
        return Err(PldError::InvalidParameter(format!(
            "noise_multiplier must be in [{}, {}], got {}",
            MIN_NOISE_MULTIPLIER, MAX_NOISE_MULTIPLIER, noise_multiplier
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
    let bounds = truncated_gaussian_epsilon_bounds(sigma, sensitivity, radius, config);
    let tail_budget = config.tail_mass_truncation / 2.0;

    discretize_symmetric_mechanism(config, bounds, |epsilon| {
        truncated_gaussian_delta_at(sigma, sensitivity, radius, epsilon)
    })
    .map(|pld| pld.with_tail_budgets(tail_budget, tail_budget))
}

/// Hockey-stick divergence δ(ε) for the truncated Gaussian mechanism.
///
/// Computes `δ(ε) = ∫ max(0, f(x;0) − e^ε · f(x;Δ)) dx` over [−Rσ, Rσ].
///
/// The privacy loss `ℓ(x) = log(f(x;0)/f(x;Δ))` is linear in x:
///   `ℓ(x) = −Δ·x/σ² + Δ²/(2σ²) + log(Z₁/Z₀)`
///
/// where Z₀ = Z(0) and Z₁ = Z(Δ). The crossover point (ℓ(x) = ε) is:
///   `x_cross = σ²/Δ · (Δ²/(2σ²) + log(Z₁/Z₀) − ε)`
///
/// For x < x_cross, f(x;0) > e^ε · f(x;Δ), contributing to δ.
fn truncated_gaussian_delta_at(sigma: f64, sensitivity: f64, radius: f64, epsilon: f64) -> f64 {
    let n01 = Normal::new(0.0, 1.0).unwrap();
    let sigma_sq = sigma * sigma;
    let r_abs = radius * sigma; // absolute radius

    // Normalization constants Z(μ) = Φ((Rσ−μ)/σ) − Φ((−Rσ−μ)/σ)
    let z0 = n01.cdf(radius) - n01.cdf(-radius); // Z(0)
    let z1 = n01.cdf(radius - sensitivity / sigma) - n01.cdf(-radius - sensitivity / sigma); // Z(Δ)

    if z0 <= 0.0 || z1 <= 0.0 {
        return 0.0;
    }

    let log_z_ratio = (z1 / z0).ln(); // log(Z₁/Z₀)

    // Privacy loss: ℓ(x) = −Δ·x/σ² + Δ²/(2σ²) + log(Z₁/Z₀)
    // Crossover: ℓ(x_cross) = ε
    //   x_cross = σ²/Δ · (Δ²/(2σ²) + log(Z₁/Z₀) − ε)
    //           = Δ/2 + σ²/Δ · (log(Z₁/Z₀) − ε)
    let x_cross = sensitivity / 2.0 + sigma_sq / sensitivity * (log_z_ratio - epsilon);

    // Integration domain where f(x;0) > e^ε · f(x;Δ): [−Rσ, x_cross] ∩ [−Rσ, Rσ]
    let int_lower = -r_abs;
    let int_upper = x_cross.min(r_abs);

    if int_upper <= int_lower {
        return 0.0;
    }

    // δ = ∫_{int_lower}^{int_upper} f(x;0) dx − e^ε · ∫_{int_lower}^{int_upper} f(x;Δ) dx
    //
    // ∫_{a}^{b} f(x;μ) dx = [Φ((b−μ)/σ) − Φ((a−μ)/σ)] / Z(μ)

    let mass_p0 = (n01.cdf(int_upper / sigma) - n01.cdf(int_lower / sigma)) / z0;
    let mass_p1 = (n01.cdf((int_upper - sensitivity) / sigma)
        - n01.cdf((int_lower - sensitivity) / sigma))
        / z1;

    (mass_p0 - epsilon.exp() * mass_p1).max(0.0)
}

/// Epsilon bounds for the truncated Gaussian mechanism.
///
/// The privacy loss ℓ(x) = −Δx/σ² + Δ²/(2σ²) + log(Z₁/Z₀) is linear in x
/// and monotonically decreasing. So:
///   ε_max = ℓ(−Rσ) = Δ·(Δ/2 + Rσ)/σ² + log(Z₁/Z₀)
///   ε_min = ℓ(Rσ)  = Δ·(Δ/2 − Rσ)/σ² + log(Z₁/Z₀)
fn truncated_gaussian_epsilon_bounds(
    sigma: f64,
    sensitivity: f64,
    radius: f64,
    config: &DiscretizationConfig,
) -> EpsilonBounds {
    let n01 = Normal::new(0.0, 1.0).unwrap();
    let sigma_sq = sigma * sigma;
    let r_abs = radius * sigma;

    // Normalization constants
    let z0 = n01.cdf(radius) - n01.cdf(-radius);
    let z1 = n01.cdf(radius - sensitivity / sigma) - n01.cdf(-radius - sensitivity / sigma);

    let log_z_ratio = if z0 > 0.0 && z1 > 0.0 {
        (z1 / z0).ln()
    } else {
        0.0
    };

    // ε at domain boundaries
    let eps_at_neg_r = sensitivity * (sensitivity / 2.0 + r_abs) / sigma_sq + log_z_ratio;
    let eps_at_pos_r = sensitivity * (sensitivity / 2.0 - r_abs) / sigma_sq + log_z_ratio;

    // Apply tail mass truncation to get reasonable bounds
    let log_mass = config.log_mass_truncation_bound;
    let half_mass = 0.5 * log_mass.exp();
    let z = n01.inverse_cdf(half_mass);
    let gauss_eps_upper = sensitivity * (0.5 * sensitivity - sigma * z) / sigma_sq;

    EpsilonBounds {
        epsilon_lower: eps_at_pos_r.max(-gauss_eps_upper),
        epsilon_upper: eps_at_neg_r.min(gauss_eps_upper),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> DiscretizationConfig {
        DiscretizationConfig::default()
    }

    #[test]
    fn test_truncated_rejects_bad_nm() {
        assert!(truncated_gaussian_pld(0.09, 3.0, &default_config()).is_err());
        assert!(truncated_gaussian_pld(1.21, 3.0, &default_config()).is_err());
    }

    #[test]
    fn test_truncated_rejects_bad_radius() {
        assert!(truncated_gaussian_pld(0.5, 0.05, &default_config()).is_err());
        assert!(truncated_gaussian_pld(0.5, 101.0, &default_config()).is_err());
    }

    #[test]
    fn test_truncated_boundary_params() {
        let cfg = default_config();
        assert!(truncated_gaussian_pld(0.1, 0.1, &cfg).is_ok());
        assert!(truncated_gaussian_pld(1.2, 100.0, &cfg).is_ok());
    }

    /// Truncated Gaussian ε ≤ rectified Gaussian ε (tighter privacy).
    #[test]
    fn test_truncated_tighter_than_rectified() {
        let cfg = default_config();
        for &nm in &[0.25, 0.5, 0.8, 1.0] {
            for &r in &[1.0, 3.0, 5.0] {
                let eps_rect = crate::mechanisms::rectified_gaussian_pld(nm, r, &cfg)
                    .unwrap()
                    .epsilon_at(1e-5);
                let eps_trunc = truncated_gaussian_pld(nm, r, &cfg)
                    .unwrap()
                    .epsilon_at(1e-5);
                assert!(
                    eps_trunc <= eps_rect + 1e-4,
                    "Truncated(σ={}, R={}) ε={:.6} should be ≤ Rectified ε={:.6}",
                    nm,
                    r,
                    eps_trunc,
                    eps_rect
                );
            }
        }
    }

    /// Truncated Gaussian ε ≤ standard Gaussian ε (DPI chain).
    #[test]
    fn test_truncated_tighter_than_gaussian() {
        let cfg = default_config();
        for &nm in &[0.25, 0.5, 0.8, 1.0] {
            let eps_gauss = crate::mechanisms::gaussian_pld(nm, &cfg)
                .unwrap()
                .epsilon_at(1e-5);
            for &r in &[1.0, 3.0, 5.0, 10.0] {
                let eps_trunc = truncated_gaussian_pld(nm, r, &cfg)
                    .unwrap()
                    .epsilon_at(1e-5);
                assert!(
                    eps_trunc <= eps_gauss + 1e-4,
                    "Truncated(σ={}, R={}) ε={:.6} should be ≤ Gaussian ε={:.6}",
                    nm,
                    r,
                    eps_trunc,
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
    fn test_truncated_radius_monotonicity() {
        let cfg = default_config();
        let radii = [1.0, 2.0, 3.0, 5.0, 10.0, 50.0];
        let epsilons: Vec<f64> = radii
            .iter()
            .map(|&r| {
                truncated_gaussian_pld(0.5, r, &cfg)
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

    /// At very large radius, truncated ≈ standard Gaussian.
    #[test]
    fn test_truncated_converges_to_gaussian() {
        let cfg = default_config();
        let eps_gauss = crate::mechanisms::gaussian_pld(0.5, &cfg)
            .unwrap()
            .epsilon_at(1e-5);
        let eps_trunc = truncated_gaussian_pld(0.5, 50.0, &cfg)
            .unwrap()
            .epsilon_at(1e-5);
        assert!(
            (eps_trunc - eps_gauss).abs() < 0.01,
            "At R=50, truncated should ≈ Gaussian: {} vs {}",
            eps_trunc,
            eps_gauss
        );
    }

    /// Noise multiplier monotonicity: higher σ → lower ε.
    #[test]
    fn test_truncated_nm_monotonicity() {
        let cfg = default_config();
        let nms = [0.1, 0.25, 0.5, 0.8, 1.0, 1.2];
        let epsilons: Vec<f64> = nms
            .iter()
            .map(|&nm| {
                truncated_gaussian_pld(nm, 3.0, &cfg)
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

    /// Full ordering: truncated ≤ rectified ≤ Gaussian.
    #[test]
    fn test_full_epsilon_ordering() {
        let cfg = default_config();
        for &nm in &[0.3, 0.5, 0.8] {
            let eps_gauss = crate::mechanisms::gaussian_pld(nm, &cfg)
                .unwrap()
                .epsilon_at(1e-5);
            for &r in &[2.0, 3.0, 5.0] {
                let eps_rect = crate::mechanisms::rectified_gaussian_pld(nm, r, &cfg)
                    .unwrap()
                    .epsilon_at(1e-5);
                let eps_trunc = truncated_gaussian_pld(nm, r, &cfg)
                    .unwrap()
                    .epsilon_at(1e-5);
                assert!(
                    eps_trunc <= eps_rect + 1e-4 && eps_rect <= eps_gauss + 1e-4,
                    "Ordering violated at σ={}, R={}: trunc={:.6} ≤ rect={:.6} ≤ gauss={:.6}",
                    nm,
                    r,
                    eps_trunc,
                    eps_rect,
                    eps_gauss
                );
            }
        }
    }
}
