//! Poisson-subsampled MIP Gaussian mechanism PLD.
//!
//! Computes the PLD for a Poisson-subsampled Gaussian mechanism with
//! heterogeneous per-example sensitivities. The hockey-stick divergence
//! is the weighted average of per-component Poisson-Gaussian divergences:
//!
//! ```text
//! δ_mip(ε, adj) = Σ_i w_i · δ_poisson(ε, adj; σ, s_i, q)
//! ```
//!
//! # References
//!
//! - Leemann, Pawelczyk, Kasneci. "Gaussian Membership Inference Privacy."
//!   NeurIPS 2023. <https://arxiv.org/abs/2306.07273>

use crate::adjacency::Adjacency;
use crate::discretization::{discretize_asymmetric_mechanism, DiscretizationConfig, EpsilonBounds};
use crate::error::{PldError, Result};
use crate::pld::PrivacyLossDistribution;

use super::poisson::{poisson_gaussian_epsilon_bounds, poisson_gaussian_get_delta};
use super::{validate_noise_multiplier, validate_rate};

/// Compute the PLD for a Poisson-subsampled MIP Gaussian mechanism.
///
/// Each example has sensitivity `sensitivities[i]` with probability `weights[i]`.
/// The result is the mixture PLD under Poisson subsampling with rate `rate`.
///
/// # Arguments
///
/// * `noise_multiplier` — σ, must be in \[0.1, 1.2\]
/// * `rate` — Poisson sampling probability q ∈ (0, 1\]
/// * `sensitivities` — per-bucket sensitivity values, all > 0
/// * `weights` — per-bucket weights, must sum to 1.0
/// * `config` — discretization configuration
///
/// # Errors
///
/// Returns `InvalidParameter` if parameters are out of range.
pub fn poisson_mip_gaussian_pld(
    noise_multiplier: f64,
    rate: f64,
    sensitivities: &[f64],
    weights: &[f64],
    config: &DiscretizationConfig,
) -> Result<PrivacyLossDistribution> {
    validate_noise_multiplier(noise_multiplier)?;
    validate_rate(rate)?;
    validate_mip_params(sensitivities, weights)?;

    let sigma = noise_multiplier;
    let log_mass = config.log_mass_truncation_bound;
    let tail_budget = config.tail_mass_truncation / 2.0;

    // Epsilon bounds: union across all components for both adjacencies.
    let bounds_remove =
        mip_poisson_epsilon_bounds(sigma, sensitivities, rate, Adjacency::Remove, log_mass);
    let bounds_add =
        mip_poisson_epsilon_bounds(sigma, sensitivities, rate, Adjacency::Add, log_mass);

    discretize_asymmetric_mechanism(config, bounds_remove, bounds_add, |epsilon, adj| {
        Ok(mip_poisson_get_delta(
            epsilon,
            adj,
            sigma,
            sensitivities,
            weights,
            rate,
        ))
    })
    .map(|pld| pld.with_tail_budgets(tail_budget, tail_budget))
}

/// Weighted hockey-stick divergence for the Poisson MIP Gaussian mixture.
fn mip_poisson_get_delta(
    epsilon: f64,
    adjacency: Adjacency,
    sigma: f64,
    sensitivities: &[f64],
    weights: &[f64],
    rate: f64,
) -> f64 {
    sensitivities
        .iter()
        .zip(weights.iter())
        .map(|(&s, &w)| w * poisson_gaussian_get_delta(epsilon, adjacency, sigma, s, rate))
        .sum()
}

/// Epsilon bounds for the Poisson MIP Gaussian: union across components.
fn mip_poisson_epsilon_bounds(
    sigma: f64,
    sensitivities: &[f64],
    rate: f64,
    adjacency: Adjacency,
    log_mass: f64,
) -> EpsilonBounds {
    let mut lower = f64::INFINITY;
    let mut upper = f64::NEG_INFINITY;

    for &s in sensitivities {
        let b = poisson_gaussian_epsilon_bounds(sigma, s, rate, adjacency, log_mass);
        if b.epsilon_lower < lower {
            lower = b.epsilon_lower;
        }
        if b.epsilon_upper > upper {
            upper = b.epsilon_upper;
        }
    }

    EpsilonBounds {
        epsilon_lower: lower,
        epsilon_upper: upper,
    }
}

/// Validate sensitivities and weights for MIP parameters.
fn validate_mip_params(sensitivities: &[f64], weights: &[f64]) -> Result<()> {
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
    fn test_rejects_bad_rate() {
        let cfg = default_config();
        assert!(poisson_mip_gaussian_pld(0.5, 0.0, &[1.0], &[1.0], &cfg).is_err());
        assert!(poisson_mip_gaussian_pld(0.5, 1.5, &[1.0], &[1.0], &cfg).is_err());
    }

    #[test]
    fn test_rejects_empty_sensitivities() {
        let cfg = default_config();
        assert!(poisson_mip_gaussian_pld(0.5, 0.01, &[], &[], &cfg).is_err());
    }

    /// Single sensitivity=1 should match standard Poisson Gaussian.
    #[test]
    fn test_single_sensitivity_matches_poisson_gaussian() {
        let cfg = default_config();
        let mip_pld =
            poisson_mip_gaussian_pld(0.5, 0.01, &[1.0], &[1.0], &cfg).unwrap();
        let standard_pld =
            crate::amplification::poisson_gaussian_pld(0.5, 0.01, &cfg).unwrap();

        let mip_eps = mip_pld.epsilon_at(1e-5);
        let std_eps = standard_pld.epsilon_at(1e-5);

        assert!(
            (mip_eps - std_eps).abs() < 1e-3,
            "MIP single-sensitivity ε={} should match standard ε={}",
            mip_eps,
            std_eps
        );
    }

    /// Poisson subsampling should reduce epsilon vs standalone.
    #[test]
    fn test_poisson_amplification_reduces_epsilon() {
        let cfg = default_config();
        let standalone =
            crate::mechanisms::mip_gaussian_pld(0.5, &[0.5, 1.0], &[0.7, 0.3], &cfg).unwrap();
        let subsampled =
            poisson_mip_gaussian_pld(0.5, 0.01, &[0.5, 1.0], &[0.7, 0.3], &cfg).unwrap();

        assert!(
            subsampled.epsilon_at(1e-5) < standalone.epsilon_at(1e-5),
            "Poisson should reduce epsilon"
        );
    }

    /// Lower sensitivities should give lower epsilon.
    #[test]
    fn test_lower_sensitivities_improve_epsilon() {
        let cfg = default_config();
        let worst_case =
            poisson_mip_gaussian_pld(0.5, 0.01, &[1.0], &[1.0], &cfg).unwrap();
        let mixed =
            poisson_mip_gaussian_pld(0.5, 0.01, &[0.3, 1.0], &[0.9, 0.1], &cfg).unwrap();

        assert!(
            mixed.epsilon_at(1e-5) < worst_case.epsilon_at(1e-5),
            "Mixture with mostly low sensitivities should have lower epsilon"
        );
    }
}
