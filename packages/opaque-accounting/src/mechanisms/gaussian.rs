//! Gaussian mechanism PLD constructor.

use crate::discretization::{discretize_symmetric_mechanism, DiscretizationConfig, EpsilonBounds};
use crate::error::{PldError, Result};
use crate::pld::PrivacyLossDistribution;

use super::MIN_NOISE_MULTIPLIER;

/// Compute the PLD for a Gaussian mechanism.
///
/// The Gaussian mechanism adds noise N(0, σ²) to a unit-sensitivity query.
/// The noise multiplier σ directly controls the privacy-utility tradeoff.
///
/// # Arguments
///
/// * `noise_multiplier` — σ/Δ ratio, must be >= 0.1
/// * `config` — discretization configuration for PLD grid
///
/// # Errors
///
/// Returns `InvalidParameter` if `noise_multiplier` < 0.1.
pub fn gaussian_pld(
    noise_multiplier: f64,
    config: &DiscretizationConfig,
) -> Result<PrivacyLossDistribution> {
    if noise_multiplier < MIN_NOISE_MULTIPLIER {
        return Err(PldError::InvalidParameter(format!(
            "noise_multiplier must be >= {}, got {}",
            MIN_NOISE_MULTIPLIER, noise_multiplier
        )));
    }

    let bounds = gaussian_epsilon_bounds(noise_multiplier, config.log_mass_truncation_bound);
    let delta_tilde = 1.0 / noise_multiplier;
    let tail_budget = config.tail_mass_truncation / 2.0;

    discretize_symmetric_mechanism(config, bounds, |epsilon| {
        crate::numerics::gaussian::gaussian_delta_at(delta_tilde, epsilon)
    })
    .map(|pld| pld.with_tail_budgets(tail_budget, tail_budget))
}

/// X-space truncation → epsilon bounds for a Gaussian mechanism.
fn gaussian_epsilon_bounds(noise_multiplier: f64, log_mass_truncation_bound: f64) -> EpsilonBounds {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> DiscretizationConfig {
        DiscretizationConfig::default()
    }

    #[test]
    fn test_gaussian_rejects_below_min() {
        assert!(gaussian_pld(0.09, &default_config()).is_err());
    }

    #[test]
    fn test_gaussian_accepts_high_nm() {
        assert!(gaussian_pld(100.0, &default_config()).is_ok());
    }

    #[test]
    fn test_gaussian_boundary_min() {
        assert!(gaussian_pld(MIN_NOISE_MULTIPLIER, &default_config()).is_ok());
    }

    #[test]
    fn test_gaussian_high_nm_adaclip_bit() {
        // AdaClip bit mechanism: nm = 2 * batch_size * fraction_noise_std
        // e.g. 2 * 100 * 0.05 = 10.0
        let pld = gaussian_pld(10.0, &default_config()).unwrap();
        // Very private — epsilon should be near zero.
        let eps = pld.epsilon_at(1e-10);
        assert!(eps < 0.01, "eps = {}, expected ~0", eps);
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
                    (n.cdf(dt / 2.0 - eps / dt) - eps.exp() * n.cdf(-dt / 2.0 - eps / dt)).max(0.0);
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
}
