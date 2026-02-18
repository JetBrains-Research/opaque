//! Privacy mechanisms: flat functions producing PLDs.
//!
//! Each function takes scalar parameters and a discretization config,
//! and returns a `PrivacyLossDistribution`. No structs, no traits.

use std::collections::BTreeMap;

use crate::discretization::{discretize_symmetric_mechanism, DiscretizationConfig, EpsilonBounds};
use crate::error::{PldError, Result};
use crate::pld::pmf::Pmf;
use crate::pld::PrivacyLossDistribution;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Minimum supported noise multiplier.
///
/// Values below this threshold cause numerical instability in discretization
/// (grid explosion, unreliable epsilon bounds).
pub(crate) const MIN_NOISE_MULTIPLIER: f64 = 0.1;

/// Maximum supported noise multiplier.
///
/// Values above this threshold cause numerical instability
/// (x-to-ε compression artifacts, unreliable beta/risk metrics).
pub(crate) const MAX_NOISE_MULTIPLIER: f64 = 1.2;

// ---------------------------------------------------------------------------
// gaussian_pld
// ---------------------------------------------------------------------------

/// Compute the PLD for a Gaussian mechanism.
///
/// The Gaussian mechanism adds noise N(0, σ²) to a unit-sensitivity query.
/// The noise multiplier σ directly controls the privacy-utility tradeoff.
///
/// # Arguments
///
/// * `noise_multiplier` — σ/Δ ratio, must be in \[0.1, 1.2\]
/// * `config` — discretization configuration for PLD grid
///
/// # Errors
///
/// Returns `InvalidParameter` if `noise_multiplier` is outside \[0.1, 1.2\].
pub fn gaussian_pld(
    noise_multiplier: f64,
    config: &DiscretizationConfig,
) -> Result<PrivacyLossDistribution> {
    if !(MIN_NOISE_MULTIPLIER..=MAX_NOISE_MULTIPLIER).contains(&noise_multiplier) {
        return Err(PldError::InvalidParameter(format!(
            "noise_multiplier must be in [{}, {}], got {}",
            MIN_NOISE_MULTIPLIER, MAX_NOISE_MULTIPLIER, noise_multiplier
        )));
    }

    let bounds = gaussian_epsilon_bounds(noise_multiplier, config.log_mass_truncation_bound);
    let delta_tilde = 1.0 / noise_multiplier;
    let tail_budget = config.tail_mass_truncation / 2.0;

    discretize_symmetric_mechanism(config, bounds, |epsilon| {
        crate::math_helpers::gaussian::gaussian_delta_at(delta_tilde, epsilon)
    })
    .map(|pld| pld.with_tail_budgets(tail_budget, tail_budget))
}

/// Compute the PLD for a Gaussian mechanism under Replace adjacency.
///
/// Under Replace adjacency, changing one record is equivalent to removing one
/// and adding another, doubling the sensitivity. This is equivalent to
/// `gaussian_pld(noise_multiplier / 2, config)` — but bypasses the range
/// check because `noise_multiplier / 2` may fall below `MIN_NOISE_MULTIPLIER`.
pub(crate) fn gaussian_replace_pld(
    noise_multiplier: f64,
    config: &DiscretizationConfig,
) -> Result<PrivacyLossDistribution> {
    let effective_nm = noise_multiplier / 2.0;
    let bounds = gaussian_epsilon_bounds(effective_nm, config.log_mass_truncation_bound);
    let delta_tilde = 1.0 / effective_nm;
    let tail_budget = config.tail_mass_truncation / 2.0;

    discretize_symmetric_mechanism(config, bounds, |epsilon| {
        crate::math_helpers::gaussian::gaussian_delta_at(delta_tilde, epsilon)
    })
    .map(|pld| pld.with_tail_budgets(tail_budget, tail_budget))
}

/// X-space truncation → epsilon bounds for a Gaussian mechanism.
fn gaussian_epsilon_bounds(
    noise_multiplier: f64,
    log_mass_truncation_bound: f64,
) -> EpsilonBounds {
    use statrs::distribution::{ContinuousCDF, Normal};

    let sigma = noise_multiplier;
    let sensitivity = 1.0;

    let standard_normal = Normal::new(0.0, 1.0).unwrap();
    let half_mass = 0.5 * log_mass_truncation_bound.exp();
    let z = standard_normal.inverse_cdf(half_mass);

    let epsilon_upper = sensitivity * (0.5 * sensitivity - sigma * z) / (sigma * sigma);

    EpsilonBounds {
        epsilon_lower: -epsilon_upper,
        epsilon_upper,
    }
}

// ---------------------------------------------------------------------------
// eps_delta_pld
// ---------------------------------------------------------------------------

/// Compute the PLD for a mechanism with a known (ε, δ) guarantee.
///
/// Constructs a worst-case PLD: mass `(1 - δ)` at privacy loss `ε`,
/// infinity mass `δ`.
///
/// # Arguments
///
/// * `epsilon` — privacy loss, must be ≥ 0
/// * `delta` — failure probability, must be in \[0, 1\]
/// * `config` — discretization configuration
///
/// # Errors
///
/// Returns `InvalidParameter` if `epsilon < 0` or `delta ∉ [0, 1]`.
pub fn eps_delta_pld(
    epsilon: f64,
    delta: f64,
    config: &DiscretizationConfig,
) -> Result<PrivacyLossDistribution> {
    if epsilon < 0.0 {
        return Err(PldError::InvalidParameter(format!(
            "epsilon must be non-negative, got {}",
            epsilon
        )));
    }
    if !(0.0..=1.0).contains(&delta) {
        return Err(PldError::InvalidParameter(format!(
            "delta must be in [0, 1], got {}",
            delta
        )));
    }

    let disc = config.discretization;
    let index = (epsilon / disc).round() as i64;
    let mut masses = BTreeMap::new();
    masses.insert(index, 1.0 - delta);

    let pmf = Pmf::from_sparse(disc, masses, delta, true, config.max_grid_size);
    Ok(PrivacyLossDistribution::new_symmetric(pmf))
}

// ---------------------------------------------------------------------------
// identity_pld
// ---------------------------------------------------------------------------

/// Compute the PLD for the identity (zero privacy loss) mechanism.
///
/// This is the neutral element for composition: composing any PLD with
/// the identity PLD yields the same privacy guarantee.
pub fn identity_pld(config: &DiscretizationConfig) -> Result<PrivacyLossDistribution> {
    let mut masses = BTreeMap::new();
    masses.insert(0, 1.0);

    let pmf = Pmf::from_sparse(config.discretization, masses, 0.0, true, config.max_grid_size);
    Ok(PrivacyLossDistribution::new_symmetric(pmf))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> DiscretizationConfig {
        DiscretizationConfig::default()
    }

    // ---- gaussian_pld ----

    #[test]
    fn test_gaussian_rejects_below_min() {
        assert!(gaussian_pld(0.09, &default_config()).is_err());
    }

    #[test]
    fn test_gaussian_rejects_above_max() {
        assert!(gaussian_pld(1.21, &default_config()).is_err());
    }

    #[test]
    fn test_gaussian_boundary_min() {
        assert!(gaussian_pld(MIN_NOISE_MULTIPLIER, &default_config()).is_ok());
    }

    #[test]
    fn test_gaussian_boundary_max() {
        assert!(gaussian_pld(MAX_NOISE_MULTIPLIER, &default_config()).is_ok());
    }

    /// At σ=0.5 the analytical δ(ε=0) ≈ Φ(1) − Φ(−1) ≈ 0.6827.
    /// Discretization error should be < 1e-3 at default grid.
    #[test]
    fn test_gaussian_delta_at_zero_epsilon() {
        let pld = gaussian_pld(0.5, &default_config()).unwrap();
        let delta = pld.delta_at(0.0);
        assert!(
            (delta - 0.6827).abs() < 1e-3,
            "delta_at(0) = {}, expected ≈ 0.6827",
            delta
        );
    }

    /// Monotonicity: higher noise → lower epsilon for fixed delta.
    #[test]
    fn test_gaussian_epsilon_decreases_with_noise() {
        let sigmas = [0.1, 0.25, 0.3, 0.5, 0.8, 1.2];
        let cfg = default_config();
        let epsilons: Vec<f64> = sigmas
            .iter()
            .map(|&s| gaussian_pld(s, &cfg).unwrap().epsilon_at(1e-5))
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

    /// Analytical Gaussian δ(ε) = Φ(Δ/(2σ) − εσ/Δ) − e^ε Φ(−Δ/(2σ) − εσ/Δ)
    /// where Δ=1. Check at several (σ, ε) pairs.
    #[test]
    fn test_gaussian_vs_analytical_delta() {
        use statrs::distribution::{ContinuousCDF, Normal};
        let n = Normal::new(0.0, 1.0).unwrap();

        for &sigma in &[0.25, 0.35, 0.5, 0.65, 0.8, 1.2] {
            let pld = gaussian_pld(sigma, &default_config()).unwrap();
            let dt = 1.0 / sigma; // Δ/σ
            for &eps in &[0.1, 0.5, 1.0, 3.0] {
                let analytical =
                    (n.cdf(dt / 2.0 - eps / dt) - eps.exp() * n.cdf(-dt / 2.0 - eps / dt))
                        .max(0.0);
                let numerical = pld.delta_at(eps);
                let err = (numerical - analytical).abs();
                assert!(
                    err < 1e-3,
                    "σ={}, ε={}: numerical={:.6e}, analytical={:.6e}, err={:.6e}",
                    sigma,
                    eps,
                    numerical,
                    analytical,
                    err
                );
            }
        }
    }

    // ---- eps_delta_pld ----

    #[test]
    fn test_eps_delta_rejects_negative_epsilon() {
        assert!(eps_delta_pld(-0.1, 1e-5, &default_config()).is_err());
    }

    #[test]
    fn test_eps_delta_rejects_invalid_delta() {
        assert!(eps_delta_pld(1.0, 1.1, &default_config()).is_err());
        assert!(eps_delta_pld(1.0, -0.1, &default_config()).is_err());
    }

    #[test]
    fn test_eps_delta_round_trip() {
        let pld = eps_delta_pld(1.0, 1e-5, &default_config()).unwrap();
        let eps = pld.epsilon_at(1e-5);
        assert!(
            (eps - 1.0).abs() < 0.001,
            "epsilon_at(1e-5) = {}, expected ~1.0",
            eps
        );
    }

    #[test]
    fn test_eps_delta_pure_dp() {
        let pld = eps_delta_pld(1.0, 0.0, &default_config()).unwrap();
        assert_eq!(pld.delta_at(1.0), 0.0);
    }

    #[test]
    fn test_eps_delta_zero_is_identity() {
        let pld = eps_delta_pld(0.0, 0.0, &default_config()).unwrap();
        assert_eq!(pld.epsilon_at(1e-5), 0.0);
        assert_eq!(pld.delta_at(0.0), 0.0);
    }

    // ---- identity_pld ----

    #[test]
    fn test_identity_epsilon_zero() {
        let pld = identity_pld(&default_config()).unwrap();
        for &delta in &[1e-10, 1e-5, 0.1, 0.5] {
            assert_eq!(pld.epsilon_at(delta), 0.0, "delta={}", delta);
        }
    }

    #[test]
    fn test_identity_delta_zero() {
        let pld = identity_pld(&default_config()).unwrap();
        for &eps in &[0.0, 0.1, 1.0, 10.0] {
            assert_eq!(pld.delta_at(eps), 0.0, "eps={}", eps);
        }
    }

    #[test]
    fn test_identity_advantage_zero() {
        let pld = identity_pld(&default_config()).unwrap();
        assert_eq!(pld.advantage(), 0.0);
    }
}
