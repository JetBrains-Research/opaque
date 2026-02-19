//! Bounded Gaussian mechanism PLD constructor.
//!
//! The Bounded Gaussian Mechanism (Chen & Hale, 2024) adds truncated Gaussian
//! noise to confine outputs to a bounded domain.  Under **Replace** adjacency
//! (changing one record changes the query answer by at most 2Δ), its PLD
//! is equivalent to the standard Gaussian with noise multiplier halved.
//!
//! The `noise_multiplier` parameter has the same meaning as for `gaussian_pld`:
//! σ/Δ with Δ = 1 (unit sensitivity).  Replace adjacency is handled internally
//! by halving the effective noise multiplier; users should not adjust the value.
//!
//! # References
//!
//! Bo Chen and Matthew Hale, "The Bounded Gaussian Mechanism for Differential
//! Privacy," Journal of Privacy and Confidentiality, 14(1), 2024.
//! <https://arxiv.org/abs/2211.17230>

use crate::discretization::DiscretizationConfig;
use crate::error::{PldError, Result};
use crate::pld::PrivacyLossDistribution;

use super::{MAX_NOISE_MULTIPLIER, MIN_NOISE_MULTIPLIER};

/// Compute the PLD for the Bounded Gaussian mechanism (Replace adjacency).
///
/// The Bounded Gaussian Mechanism adds noise from a truncated normal to keep
/// outputs within a bounded domain.  Under **Replace** adjacency (one record
/// swapped), sensitivity is 2Δ, which is accounted for internally by halving
/// the effective noise multiplier.
///
/// The `noise_multiplier` parameter has the same meaning and the same valid
/// range as for [`gaussian_pld`]: the σ/Δ ratio with Δ = 1.
///
/// # Arguments
///
/// * `noise_multiplier` — σ/Δ ratio, must be in [`MIN_NOISE_MULTIPLIER`, `MAX_NOISE_MULTIPLIER`]
/// * `config` — discretization configuration for PLD grid
///
/// # Errors
///
/// Returns `InvalidParameter` if `noise_multiplier` is outside the allowed range.
pub fn bounded_gaussian_pld(
    noise_multiplier: f64,
    config: &DiscretizationConfig,
) -> Result<PrivacyLossDistribution> {
    if !(MIN_NOISE_MULTIPLIER..=MAX_NOISE_MULTIPLIER).contains(&noise_multiplier) {
        return Err(PldError::InvalidParameter(format!(
            "noise_multiplier must be in [{}, {}], got {}",
            MIN_NOISE_MULTIPLIER, MAX_NOISE_MULTIPLIER, noise_multiplier
        )));
    }
    super::gaussian::gaussian_replace_pld(noise_multiplier, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> DiscretizationConfig {
        DiscretizationConfig::default()
    }

    #[test]
    fn test_bounded_gaussian_rejects_below_min() {
        assert!(bounded_gaussian_pld(0.09, &default_config()).is_err());
    }

    #[test]
    fn test_bounded_gaussian_rejects_above_max() {
        assert!(bounded_gaussian_pld(1.21, &default_config()).is_err());
    }

    #[test]
    fn test_bounded_gaussian_boundary_min() {
        assert!(bounded_gaussian_pld(MIN_NOISE_MULTIPLIER, &default_config()).is_ok());
    }

    #[test]
    fn test_bounded_gaussian_boundary_max() {
        assert!(bounded_gaussian_pld(MAX_NOISE_MULTIPLIER, &default_config()).is_ok());
    }

    /// bounded_gaussian_pld(σ) == gaussian_pld(σ/2) for the same config.
    ///
    /// Both should produce the same epsilon at delta=1e-5 since the
    /// Replace adjacency effectively halves the noise multiplier.
    #[test]
    fn test_equals_gaussian_with_halved_noise_multiplier() {
        use crate::mechanisms::gaussian::gaussian_pld;
        use approx::assert_abs_diff_eq;

        let config = default_config();
        for &nm in &[0.2, 0.5, 0.8, 1.0, 1.2] {
            let bpld = bounded_gaussian_pld(nm, &config).unwrap();
            let gpld = gaussian_pld(nm / 2.0, &config).unwrap();
            let b_eps = bpld.epsilon_at(1e-5);
            let g_eps = gpld.epsilon_at(1e-5);
            assert_abs_diff_eq!(b_eps, g_eps, epsilon = 1e-8);
        }
    }

    /// Monotonicity: higher noise → lower epsilon for fixed delta.
    #[test]
    fn test_bounded_gaussian_epsilon_decreases_with_noise() {
        let sigmas = [0.2, 0.5, 0.8, 1.0, 1.2];
        let cfg = default_config();
        let epsilons: Vec<f64> = sigmas
            .iter()
            .map(|&s| bounded_gaussian_pld(s, &cfg).unwrap().epsilon_at(1e-5))
            .collect();
        for w in epsilons.windows(2) {
            assert!(
                w[0] > w[1],
                "ε should decrease: σ gave ε={}, next gave ε={}",
                w[0],
                w[1]
            );
        }
    }
}
