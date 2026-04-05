//! Truncated (renormalized) Gaussian mechanism PLD constructor.
//!
//! The truncated Gaussian mechanism samples noise from `N(0, σ²) | [−Rσ, Rσ]`,
//! the standard Gaussian **renormalized** to the interval [−Rσ, Rσ].
//!
//! The mechanism outputs values in a fixed domain [−Rσ, Rσ]. The worst-case
//! privacy loss depends on where the distribution centers (determined by the
//! query output) fall within this domain. In DP-SGD, inputs are L2-clipped
//! to norm Δ, so per-coordinate values satisfy |x| ≤ Δ. The PLD constructor
//! optimizes over centers in [−Δ, Δ] to provide a sound upper bound on δ(ε).
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

use super::validate_noise_multiplier;

/// Minimum supported radius (sigma units).
const MIN_RADIUS: f64 = 0.1;

/// Maximum supported radius (sigma units).
const MAX_RADIUS: f64 = 100.0;

/// Number of grid points for worst-case center search.
///
/// The truncated Gaussian mechanism outputs values in a fixed domain
/// [−Rσ, Rσ]. For adjacent inputs differing by Δ, the output distributions
/// are truncated normals centered at μ₀ and μ₁ = μ₀ + Δ within this domain.
///
/// In DP-SGD, inputs are L2-clipped to norm Δ, so per-coordinate values
/// satisfy |x| ≤ Δ. We search μ₀ over [−Δ, Δ] to find the worst case.
const CENTER_SEARCH_POINTS: usize = 200;

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
/// * `noise_multiplier` — σ/Δ ratio, must be > 0 (see [`MIN_NOISE_MULTIPLIER`])
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
    validate_noise_multiplier(noise_multiplier)?;
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

/// Hockey-stick divergence δ(ε) for the truncated Gaussian mechanism,
/// maximized over worst-case distribution centers.
///
/// The mechanism outputs values in the fixed domain [−Rσ, Rσ]. For adjacent
/// inputs differing by Δ, the output distributions are truncated normals
/// centered at μ₀ and μ₁ = μ₀ + Δ. We search μ₀ over [−Δ, Δ] because
/// inputs are L2-clipped: per-coordinate values satisfy |x| ≤ Δ.
fn truncated_gaussian_delta_at(sigma: f64, sensitivity: f64, radius: f64, epsilon: f64) -> f64 {
    // After L2 clipping to norm Δ, per-coordinate inputs are in [−Δ, Δ].
    let search_lo = -sensitivity;
    let search_hi = sensitivity;
    let step = (search_hi - search_lo) / (CENTER_SEARCH_POINTS as f64);

    let mut max_delta = 0.0_f64;
    for i in 0..=CENTER_SEARCH_POINTS {
        let mu0 = search_lo + step * (i as f64);
        let d = truncated_gaussian_delta_at_center(sigma, sensitivity, radius, epsilon, mu0);
        max_delta = max_delta.max(d);
    }
    max_delta
}

/// Hockey-stick divergence δ(ε) for a specific center μ₀.
///
/// Computes `δ(ε) = ∫ max(0, f(x;μ₀) − e^ε · f(x;μ₁)) dx` over [−Rσ, Rσ],
/// where μ₁ = μ₀ + Δ.
///
/// The privacy loss `ℓ(x) = log(f(x;μ₀)/f(x;μ₁))` is linear in x:
///   `ℓ(x) = Δ·(μ₀ + Δ/2 − x)/σ² + log(Z₁/Z₀)`
///
/// where Z₀ = Z(μ₀) and Z₁ = Z(μ₁).
fn truncated_gaussian_delta_at_center(
    sigma: f64,
    sensitivity: f64,
    radius: f64,
    epsilon: f64,
    mu0: f64,
) -> f64 {
    let n01 = Normal::new(0.0, 1.0).unwrap();
    let sigma_sq = sigma * sigma;
    let r_abs = radius * sigma;
    let mu1 = mu0 + sensitivity;

    // Normalization constants Z(μ) = Φ((Rσ−μ)/σ) − Φ((−Rσ−μ)/σ)
    let z0 = n01.cdf((r_abs - mu0) / sigma) - n01.cdf((-r_abs - mu0) / sigma);
    let z1 = n01.cdf((r_abs - mu1) / sigma) - n01.cdf((-r_abs - mu1) / sigma);

    if z0 <= 0.0 || z1 <= 0.0 {
        return 0.0;
    }

    let log_z_ratio = (z1 / z0).ln(); // log(Z₁/Z₀)

    // Crossover: ℓ(x_cross) = ε → x_cross = μ₀ + Δ/2 + σ²/Δ · (log(Z₁/Z₀) − ε)
    let x_cross =
        mu0 + sensitivity / 2.0 + sigma_sq / sensitivity * (log_z_ratio - epsilon);

    // Integration domain where f(x;μ₀) > e^ε · f(x;μ₁): [−Rσ, x_cross] ∩ [−Rσ, Rσ]
    let int_lower = -r_abs;
    let int_upper = x_cross.min(r_abs);

    if int_upper <= int_lower {
        return 0.0;
    }

    // δ = ∫_{lo}^{hi} f(x;μ₀) dx − e^ε · ∫_{lo}^{hi} f(x;μ₁) dx
    // ∫_{a}^{b} f(x;μ) dx = [Φ((b−μ)/σ) − Φ((a−μ)/σ)] / Z(μ)
    let mass_p0 =
        (n01.cdf((int_upper - mu0) / sigma) - n01.cdf((int_lower - mu0) / sigma)) / z0;
    let mass_p1 =
        (n01.cdf((int_upper - mu1) / sigma) - n01.cdf((int_lower - mu1) / sigma)) / z1;

    (mass_p0 - epsilon.exp() * mass_p1).max(0.0)
}

/// Epsilon bounds for the truncated Gaussian mechanism.
///
/// Searches over all possible centers μ₀ to find the widest epsilon range.
/// The privacy loss at domain boundaries for center μ₀ is:
///   ε_max = ℓ(−Rσ) = Δ·(μ₀ + Δ/2 + Rσ)/σ² + log(Z₁/Z₀)
///   ε_min = ℓ(Rσ)  = Δ·(μ₀ + Δ/2 − Rσ)/σ² + log(Z₁/Z₀)
fn truncated_gaussian_epsilon_bounds(
    sigma: f64,
    sensitivity: f64,
    radius: f64,
    config: &DiscretizationConfig,
) -> EpsilonBounds {
    let n01 = Normal::new(0.0, 1.0).unwrap();
    let sigma_sq = sigma * sigma;
    let r_abs = radius * sigma;

    // Search over centers in [−Δ, Δ] (L2-clipped input domain).
    let search_lo = -sensitivity;
    let search_hi = sensitivity;
    let step = (search_hi - search_lo) / (CENTER_SEARCH_POINTS as f64);

    let mut eps_lo = f64::INFINITY;
    let mut eps_hi = f64::NEG_INFINITY;

    for i in 0..=CENTER_SEARCH_POINTS {
        let mu0 = search_lo + step * (i as f64);
        let mu1 = mu0 + sensitivity;

        let z0 = n01.cdf((r_abs - mu0) / sigma) - n01.cdf((-r_abs - mu0) / sigma);
        let z1 = n01.cdf((r_abs - mu1) / sigma) - n01.cdf((-r_abs - mu1) / sigma);

        if z0 <= 0.0 || z1 <= 0.0 {
            continue;
        }

        let log_z_ratio = (z1 / z0).ln();

        let eps_at_neg_r =
            sensitivity * (mu0 + sensitivity / 2.0 + r_abs) / sigma_sq + log_z_ratio;
        let eps_at_pos_r =
            sensitivity * (mu0 + sensitivity / 2.0 - r_abs) / sigma_sq + log_z_ratio;

        eps_hi = eps_hi.max(eps_at_neg_r);
        eps_lo = eps_lo.min(eps_at_pos_r);
    }

    // Safety cap: Gaussian tail-mass truncation bounds
    let log_mass = config.log_mass_truncation_bound;
    let half_mass = 0.5 * log_mass.exp();
    let z = n01.inverse_cdf(half_mass);
    let gauss_eps_upper = sensitivity * (0.5 * sensitivity - sigma * z) / sigma_sq;

    EpsilonBounds {
        epsilon_lower: eps_lo.max(-gauss_eps_upper),
        epsilon_upper: eps_hi.min(gauss_eps_upper),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> DiscretizationConfig {
        DiscretizationConfig::default()
    }

    #[test]
    fn test_truncated_rejects_below_min_nm() {
        assert!(truncated_gaussian_pld(1e-7, 3.0, &default_config()).is_err());
        assert!(truncated_gaussian_pld(0.0, 3.0, &default_config()).is_err());
        assert!(truncated_gaussian_pld(-1.0, 3.0, &default_config()).is_err());
    }

    #[test]
    fn test_truncated_accepts_small_nm() {
        assert!(truncated_gaussian_pld(0.05, 3.0, &default_config()).is_ok());
    }

    #[test]
    fn test_truncated_accepts_high_nm() {
        assert!(truncated_gaussian_pld(5.0, 3.0, &default_config()).is_ok());
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

    /// Truncated Gaussian ε ≤ standard Gaussian ε (bounded support concentrates
    /// mass in the interior, reducing privacy loss).
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
    /// With centers restricted to [−Δ, Δ] (clipped input domain), the
    /// normalization constants are ≈1 at R=50, so the gap is negligible.
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
}
