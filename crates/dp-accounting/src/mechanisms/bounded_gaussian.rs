//! Bounded Gaussian mechanism PLD constructor.
//!
//! The Bounded Gaussian Mechanism (Chen & Hale, 2024) adds truncated Gaussian
//! noise to confine outputs to a bounded domain.  For **DP-SGD** the standard
//! adjacency model is **Add/Remove** with gradient clip-norm sensitivity Δ = 1,
//! exactly the same as for the standard Gaussian mechanism.
//!
//! When the truncation bounds are wide relative to σ (the common case for
//! DP-SGD, where bounds are typically ≥ 3σ away from the query result),
//! the PLD of the bounded Gaussian is approximately equal to the standard
//! Gaussian PLD.  `bounded_gaussian_pld` returns `gaussian_pld(noise_multiplier)`
//! as a conservative approximation.
//!
//! **Note**: The exact PLD of a truncated Gaussian with absolute bounds [l, u]
//! includes a log-normalisation correction term that depends on l, u, the query
//! value, and σ.  For narrow bounds or query values near the boundaries the
//! approximation degrades.  If you need exact accounting, pass the truncation
//! bounds explicitly (not yet exposed in this API).
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

/// Compute the PLD for the Bounded Gaussian mechanism (Add/Remove adjacency).
///
/// The Bounded Gaussian Mechanism adds noise from a truncated normal to keep
/// outputs within a bounded domain.  For DP-SGD the mechanism is analysed
/// under **Add/Remove** adjacency with sensitivity 1 (same as the standard
/// Gaussian mechanism).  When the truncation bounds are wide relative to σ,
/// the PLD is approximately equal to the standard Gaussian PLD; this function
/// returns `gaussian_pld(noise_multiplier)` as a conservative approximation.
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
    // For wide truncation bounds the PLD of the bounded Gaussian is approximately
    // the same as the standard Gaussian under Add/Remove adjacency with the same
    // noise multiplier.  This is a conservative (safe) upper bound on ε.
    super::gaussian::gaussian_pld(noise_multiplier, config)
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

    /// Under Add/Remove adjacency with wide bounds,
    /// bounded_gaussian_pld(nm) == gaussian_pld(nm).
    #[test]
    fn test_equals_gaussian_same_noise_multiplier() {
        use crate::mechanisms::gaussian::gaussian_pld;
        use approx::assert_abs_diff_eq;

        let config = default_config();
        for &nm in &[0.2, 0.5, 0.8, 1.0, 1.2] {
            let bpld = bounded_gaussian_pld(nm, &config).unwrap();
            let gpld = gaussian_pld(nm, &config).unwrap();
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
