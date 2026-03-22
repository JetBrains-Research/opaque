//! MIP Gaussian mechanism PLD constructor.
//!
//! Computes the PLD for a Gaussian mechanism with heterogeneous per-example
//! sensitivities (Membership Inference Privacy). The hockey-stick divergence
//! is the weighted average of individual Gaussian hockey-stick divergences:
//!
//! ```text
//! δ_mip(ε) = Σ_i w_i · δ_gaussian(ε; σ, s_i)
//! ```
//!
//! # References
//!
//! - Leemann, Pawelczyk, Kasneci. "Gaussian Membership Inference Privacy."
//!   NeurIPS 2023. <https://arxiv.org/abs/2306.07273>

use crate::discretization::{discretize_symmetric_mechanism, DiscretizationConfig, EpsilonBounds};
use crate::error::{PldError, Result};
use crate::pld::PrivacyLossDistribution;

use super::{MAX_NOISE_MULTIPLIER, MIN_NOISE_MULTIPLIER};

/// Compute the PLD for a MIP Gaussian mechanism with heterogeneous sensitivities.
///
/// Each sensitivity level `s_i` has weight `w_i`. The mechanism output is a
/// mixture: with probability `w_i`, the true sensitivity is `s_i` and Gaussian
/// noise N(0, σ²) is added.
///
/// # Arguments
///
/// * `noise_multiplier` — σ, must be in \[0.1, 1.2\]
/// * `sensitivities` — per-bucket sensitivity values, all > 0
/// * `weights` — per-bucket weights, must sum to 1.0 (within tolerance)
/// * `config` — discretization configuration
///
/// # Errors
///
/// Returns `InvalidParameter` if parameters are out of range or mismatched.
pub fn mip_gaussian_pld(
    noise_multiplier: f64,
    sensitivities: &[f64],
    weights: &[f64],
    config: &DiscretizationConfig,
) -> Result<PrivacyLossDistribution> {
    validate_mip_params(noise_multiplier, sensitivities, weights)?;

    let sigma = noise_multiplier;
    let log_mass = config.log_mass_truncation_bound;
    let tail_budget = config.tail_mass_truncation / 2.0;

    // Epsilon bounds: union across all sensitivity levels.
    let bounds = mip_gaussian_epsilon_bounds(sigma, sensitivities, log_mass);

    discretize_symmetric_mechanism(config, bounds, |epsilon| {
        mip_gaussian_delta_at(sigma, sensitivities, weights, epsilon)
    })
    .map(|pld| pld.with_tail_budgets(tail_budget, tail_budget))
}

/// Weighted hockey-stick divergence for the MIP Gaussian mixture.
///
/// Batched: precomputes `exp(epsilon)` and reuses a single Normal distribution
/// across all K components instead of constructing one per call.
fn mip_gaussian_delta_at(
    sigma: f64,
    sensitivities: &[f64],
    weights: &[f64],
    epsilon: f64,
) -> f64 {
    use statrs::distribution::{ContinuousCDF, Normal};
    let standard_normal = Normal::new(0.0, 1.0).unwrap();
    let exp_eps = epsilon.exp();

    sensitivities
        .iter()
        .zip(weights.iter())
        .map(|(&s, &w)| {
            let delta_tilde = s / sigma;
            let x_upper = 0.5 * delta_tilde - epsilon / delta_tilde;
            let cdf_x = standard_normal.cdf(x_upper);
            let cdf_shifted = standard_normal.cdf(x_upper - delta_tilde);
            w * (cdf_x - exp_eps * cdf_shifted).max(0.0)
        })
        .sum()
}

/// Epsilon bounds for the MIP Gaussian: union of per-component bounds.
fn mip_gaussian_epsilon_bounds(
    sigma: f64,
    sensitivities: &[f64],
    log_mass_truncation_bound: f64,
) -> EpsilonBounds {
    use statrs::distribution::{ContinuousCDF, Normal};

    let standard_normal = Normal::new(0.0, 1.0).unwrap();
    let half_mass = 0.5 * log_mass_truncation_bound.exp();
    let z = standard_normal.inverse_cdf(half_mass);

    let mut epsilon_upper = f64::NEG_INFINITY;

    for &s in sensitivities {
        let eps_upper_i = s * (0.5 * s - sigma * z) / (sigma * sigma);
        if eps_upper_i > epsilon_upper {
            epsilon_upper = eps_upper_i;
        }
    }

    EpsilonBounds {
        epsilon_lower: -epsilon_upper,
        epsilon_upper,
    }
}

/// Validate MIP Gaussian parameters.
fn validate_mip_params(
    noise_multiplier: f64,
    sensitivities: &[f64],
    weights: &[f64],
) -> Result<()> {
    if !(MIN_NOISE_MULTIPLIER..=MAX_NOISE_MULTIPLIER).contains(&noise_multiplier) {
        return Err(PldError::InvalidParameter(format!(
            "noise_multiplier must be in [{}, {}], got {}",
            MIN_NOISE_MULTIPLIER, MAX_NOISE_MULTIPLIER, noise_multiplier
        )));
    }
    if sensitivities.is_empty() {
        return Err(PldError::InvalidParameter(
            "sensitivities must not be empty".to_string(),
        ));
    }
    if sensitivities.len() != weights.len() {
        return Err(PldError::InvalidParameter(format!(
            "sensitivities and weights must have the same length, got {} and {}",
            sensitivities.len(),
            weights.len()
        )));
    }
    for (i, &s) in sensitivities.iter().enumerate() {
        if s <= 0.0 || !s.is_finite() {
            return Err(PldError::InvalidParameter(format!(
                "sensitivities[{}] must be positive and finite, got {}",
                i, s
            )));
        }
    }
    for (i, &w) in weights.iter().enumerate() {
        if w < 0.0 || !w.is_finite() {
            return Err(PldError::InvalidParameter(format!(
                "weights[{}] must be non-negative and finite, got {}",
                i, w
            )));
        }
    }
    let weight_sum: f64 = weights.iter().sum();
    if (weight_sum - 1.0).abs() > 1e-6 {
        return Err(PldError::InvalidParameter(format!(
            "weights must sum to 1.0, got {}",
            weight_sum
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> DiscretizationConfig {
        DiscretizationConfig::default()
    }

    #[test]
    fn test_rejects_bad_noise_multiplier() {
        let cfg = default_config();
        assert!(mip_gaussian_pld(0.05, &[1.0], &[1.0], &cfg).is_err());
        assert!(mip_gaussian_pld(1.5, &[1.0], &[1.0], &cfg).is_err());
    }

    #[test]
    fn test_rejects_empty_sensitivities() {
        let cfg = default_config();
        assert!(mip_gaussian_pld(0.5, &[], &[], &cfg).is_err());
    }

    #[test]
    fn test_rejects_mismatched_lengths() {
        let cfg = default_config();
        assert!(mip_gaussian_pld(0.5, &[1.0, 0.5], &[1.0], &cfg).is_err());
    }

    #[test]
    fn test_rejects_bad_weights_sum() {
        let cfg = default_config();
        assert!(mip_gaussian_pld(0.5, &[1.0, 0.5], &[0.5, 0.3], &cfg).is_err());
    }

    /// Single sensitivity=1 should match standard Gaussian.
    #[test]
    fn test_single_sensitivity_matches_gaussian() {
        let cfg = default_config();
        let mip_pld = mip_gaussian_pld(0.5, &[1.0], &[1.0], &cfg).unwrap();
        let gaussian_pld = crate::mechanisms::gaussian_pld(0.5, &cfg).unwrap();

        let mip_eps = mip_pld.epsilon_at(1e-5);
        let gauss_eps = gaussian_pld.epsilon_at(1e-5);

        assert!(
            (mip_eps - gauss_eps).abs() < 1e-3,
            "MIP single-sensitivity ε={} should match Gaussian ε={}",
            mip_eps,
            gauss_eps
        );
    }

    /// Lower sensitivities should give lower (better) epsilon.
    #[test]
    fn test_lower_sensitivities_improve_epsilon() {
        let cfg = default_config();
        let worst_case = mip_gaussian_pld(0.5, &[1.0], &[1.0], &cfg).unwrap();
        let mixed = mip_gaussian_pld(0.5, &[0.5, 1.0], &[0.8, 0.2], &cfg).unwrap();

        assert!(
            mixed.epsilon_at(1e-5) < worst_case.epsilon_at(1e-5),
            "Mixture with mostly-low sensitivities should have lower epsilon"
        );
    }
}
