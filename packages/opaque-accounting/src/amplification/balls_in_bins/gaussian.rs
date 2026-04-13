//! Balls-in-Bins amplification for Gaussian mechanism PLD.
//!
//! In the Balls-in-Bins (BnB) sampling scheme, the dataset is randomly
//! partitioned into `num_bins` equally-sized bins each epoch. Each bin is
//! processed once, so every example participates exactly once per epoch.
//!
//! This implementation uses a conservative Poisson per-step approximation:
//! each of the `num_bins` steps is analyzed as a Poisson-subsampled Gaussian
//! with rate `1/num_bins`, and the per-epoch PLD is their composition.
//!
//! References:
//!   - Chua et al. (2025), "Scalable Shuffle Differential Privacy"
//!   - Choquette-Choo et al. (2024), "Privacy Amplification for Matrix Mechanisms"

use crate::discretization::DiscretizationConfig;
use crate::error::{PldError, Result};
use crate::mechanisms::validate_noise_multiplier;
use crate::pld::PrivacyLossDistribution;

use super::super::poisson::poisson_gaussian_pld;

/// Compute the per-epoch PLD for a Balls-in-Bins Gaussian mechanism.
///
/// The dataset is partitioned into `num_bins` bins, each processed with a
/// Gaussian mechanism. This returns the composed PLD for one full epoch
/// (all `num_bins` steps).
///
/// Uses a conservative Poisson approximation: each step is analyzed as
/// Poisson-subsampled Gaussian with rate `1/num_bins`.
///
/// # Arguments
///
/// * `noise_multiplier` — σ/Δ, must be > 0
/// * `num_bins` — number of bins (k ≥ 2)
/// * `config` — discretization configuration
///
/// # Errors
///
/// Returns `InvalidParameter` if parameters are out of range.
pub fn balls_in_bins_gaussian_pld(
    noise_multiplier: f64,
    num_bins: usize,
    config: &DiscretizationConfig,
) -> Result<PrivacyLossDistribution> {
    balls_in_bins_gaussian_pld_epochs(noise_multiplier, num_bins, 1, config)
}

/// Compute the **total** PLD for a multi-epoch Balls-in-Bins Gaussian mechanism.
///
/// Each epoch the dataset is partitioned into `num_bins` bins. Each bin is
/// processed with a Gaussian mechanism. Over `num_epochs` epochs, each
/// example participates `num_epochs` times.
///
/// For Gaussian (independent noise), this equals composing the per-step
/// Poisson PLD `num_bins * num_epochs` times — equivalent to
/// `per_epoch.self_compose(num_epochs)`, but done in a single call.
///
/// # Arguments
///
/// * `noise_multiplier` — σ/Δ, must be > 0
/// * `num_bins` — bins per epoch (k ≥ 2)
/// * `num_epochs` — number of epochs (≥ 1)
/// * `config` — discretization configuration
pub fn balls_in_bins_gaussian_pld_epochs(
    noise_multiplier: f64,
    num_bins: usize,
    num_epochs: usize,
    config: &DiscretizationConfig,
) -> Result<PrivacyLossDistribution> {
    validate_noise_multiplier(noise_multiplier)?;

    if num_bins < 2 {
        return Err(PldError::InvalidParameter(format!(
            "num_bins must be >= 2 for BnB amplification, got {}",
            num_bins
        )));
    }
    if num_epochs == 0 {
        return Err(PldError::InvalidParameter(
            "num_epochs must be >= 1".to_string(),
        ));
    }

    let rate = 1.0 / num_bins as f64;
    let per_step = poisson_gaussian_pld(noise_multiplier, rate, config)?;
    Ok(per_step.self_compose(num_bins * num_epochs))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> DiscretizationConfig {
        DiscretizationConfig::default()
    }

    #[test]
    fn test_bnb_rejects_single_bin() {
        let cfg = default_config();
        assert!(balls_in_bins_gaussian_pld(1.0, 1, &cfg).is_err());
        assert!(balls_in_bins_gaussian_pld(1.0, 0, &cfg).is_err());
    }

    #[test]
    fn test_bnb_rejects_bad_noise() {
        let cfg = default_config();
        assert!(balls_in_bins_gaussian_pld(0.0, 10, &cfg).is_err());
        assert!(balls_in_bins_gaussian_pld(-1.0, 10, &cfg).is_err());
    }

    #[test]
    fn test_bnb_produces_valid_pld() {
        let cfg = default_config();
        let pld = balls_in_bins_gaussian_pld(1.0, 10, &cfg).unwrap();
        let eps = pld.epsilon_at(1e-5);
        assert!(eps > 0.0, "epsilon should be positive");
        assert!(eps.is_finite(), "epsilon should be finite");
    }

    #[test]
    fn test_bnb_more_bins_reduces_epsilon() {
        let cfg = default_config();
        let eps_10 = balls_in_bins_gaussian_pld(1.0, 10, &cfg)
            .unwrap()
            .epsilon_at(1e-5);
        let eps_100 = balls_in_bins_gaussian_pld(1.0, 100, &cfg)
            .unwrap()
            .epsilon_at(1e-5);
        assert!(
            eps_100 < eps_10,
            "more bins should reduce epsilon: {} vs {}",
            eps_100,
            eps_10
        );
    }

    #[test]
    fn test_bnb_matches_poisson_composition() {
        // BnB(k) should equal Poisson(1/k) composed k times
        let cfg = default_config();
        let k = 20;
        let sigma = 0.8;

        let bnb_eps = balls_in_bins_gaussian_pld(sigma, k, &cfg)
            .unwrap()
            .epsilon_at(1e-5);

        let poisson_eps = poisson_gaussian_pld(sigma, 1.0 / k as f64, &cfg)
            .unwrap()
            .self_compose(k)
            .epsilon_at(1e-5);

        assert!(
            (bnb_eps - poisson_eps).abs() < 1e-10,
            "BnB should match Poisson composition: {} vs {}",
            bnb_eps,
            poisson_eps
        );
    }

    #[test]
    fn test_bnb_amplification_vs_gaussian() {
        // BnB epoch should be cheaper than k * bare Gaussian
        let cfg = default_config();
        let k = 10;
        let sigma = 1.0;

        let bnb_eps = balls_in_bins_gaussian_pld(sigma, k, &cfg)
            .unwrap()
            .epsilon_at(1e-5);

        let gauss_eps = crate::mechanisms::gaussian_pld(sigma, &cfg)
            .unwrap()
            .self_compose(k)
            .epsilon_at(1e-5);

        assert!(
            bnb_eps < gauss_eps,
            "BnB should be cheaper than k * Gaussian: {} vs {}",
            bnb_eps,
            gauss_eps
        );
    }
}
