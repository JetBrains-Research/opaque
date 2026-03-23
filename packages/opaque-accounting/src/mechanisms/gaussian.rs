//! Gaussian mechanism PLD constructor.

use crate::discretization::{discretize_symmetric_mechanism, DiscretizationConfig, EpsilonBounds};
use crate::error::{PldError, Result};
use crate::pld::PrivacyLossDistribution;

use super::{MAX_NOISE_MULTIPLIER, MIN_NOISE_MULTIPLIER, MAX_COARSENING_FACTOR, MAX_GRID_FRACTION};

/// Compute the PLD for a Gaussian mechanism.
///
/// The Gaussian mechanism adds noise N(0, σ²) to a unit-sensitivity query.
/// The noise multiplier σ directly controls the privacy-utility tradeoff.
///
/// When the PMF grid would require coarsening (effective discretization exceeds
/// the base by more than [`MAX_COARSENING_FACTOR`]), automatically falls back to
/// the CGF-backed path to avoid arithmetic errors from aggressive grid coarsening.
///
/// # Arguments
///
/// * `noise_multiplier` — σ/Δ ratio, must be in \[0.01, 2.5\].
/// * `config` — discretization configuration for PLD grid (ignored for CGF path)
///
/// # Errors
///
/// Returns `InvalidParameter` if `noise_multiplier` is outside \[0.01, 2.5\].
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
    let effective_disc = config.effective_discretization(&bounds);
    let coarsening = effective_disc / config.discretization;

    // Fall back to CGF if:
    // 1. Grid would need coarsening (Dirac-like artifacts), or
    // 2. Grid is large enough that composition would be expensive
    let grid_size = ((bounds.epsilon_upper - bounds.epsilon_lower) / effective_disc).ceil() as usize;
    let grid_too_large = grid_size as f64 > config.max_grid_size as f64 * MAX_GRID_FRACTION;
    if coarsening > MAX_COARSENING_FACTOR || grid_too_large {
        return cgf_gaussian_pld(noise_multiplier);
    }

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

/// Create a CGF-backed PLD for a Gaussian mechanism.
///
/// Unlike `gaussian_pld`, this does not discretize — the privacy loss is
/// represented analytically via its CGF. Composition is O(1) and there
/// is no grid. Privacy metrics use Lugannani-Rice saddle-point approximation.
///
/// # Arguments
///
/// * `noise_multiplier` — σ/Δ ratio, must be in \[0.01, 2.5\]
pub fn cgf_gaussian_pld(noise_multiplier: f64) -> Result<PrivacyLossDistribution> {
    if !(MIN_NOISE_MULTIPLIER..=MAX_NOISE_MULTIPLIER).contains(&noise_multiplier) {
        return Err(PldError::InvalidParameter(format!(
            "noise_multiplier must be in [{}, {}], got {}",
            MIN_NOISE_MULTIPLIER, MAX_NOISE_MULTIPLIER, noise_multiplier
        )));
    }

    let cgf = std::sync::Arc::new(crate::pld::cgf::GaussianCgf::new(noise_multiplier));
    Ok(PrivacyLossDistribution::new_cgf(cgf))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> DiscretizationConfig {
        DiscretizationConfig::default()
    }

    #[test]
    fn test_gaussian_rejects_below_min() {
        assert!(gaussian_pld(0.009, &default_config()).is_err());
    }

    #[test]
    fn test_gaussian_rejects_above_max() {
        assert!(gaussian_pld(2.51, &default_config()).is_err());
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

    /// Small σ values route to CGF (grid would need coarsening) and produce valid results.
    #[test]
    fn test_gaussian_grid_fidelity_routes_to_cgf() {
        for &sigma in &[0.01, 0.03, 0.05, 0.09] {
            let pld = gaussian_pld(sigma, &default_config()).unwrap();
            let eps = pld.epsilon_at(1e-5);
            assert!(eps > 0.0 && eps.is_finite(), "σ={}: ε={}", sigma, eps);
        }
    }

    /// Monotonicity holds across the full σ range (including the CGF/PMF boundary).
    #[test]
    fn test_gaussian_monotonicity_full_range() {
        let cfg = default_config();
        let sigmas = [0.05, 0.09, 0.1, 0.15, 0.25];
        let epsilons: Vec<f64> = sigmas
            .iter()
            .map(|&s| gaussian_pld(s, &cfg).unwrap().epsilon_at(1e-5))
            .collect();
        for w in epsilons.windows(2) {
            assert!(
                w[0] > w[1],
                "ε should decrease with σ: got {} then {}",
                w[0],
                w[1]
            );
        }
    }

    /// With unlimited grid size, even small σ stays PMF (no coarsening).
    #[test]
    fn test_gaussian_unlimited_grid_stays_pmf() {
        let cfg = DiscretizationConfig::default().with_max_grid_size(usize::MAX);
        let pld = gaussian_pld(0.05, &cfg).unwrap();
        // Should be PMF since no coarsening is needed with unlimited grid
        let eps = pld.epsilon_at(1e-5);
        assert!(eps > 0.0 && eps.is_finite(), "ε={}", eps);
    }

    /// With a tiny max_grid_size, even moderate σ triggers CGF fallback.
    #[test]
    fn test_gaussian_small_grid_routes_to_cgf() {
        let cfg = DiscretizationConfig::default().with_max_grid_size(100);
        // σ=0.5 normally uses PMF, but with only 100 grid points,
        // coarsening would be needed → should fall back to CGF
        let pld = gaussian_pld(0.5, &cfg).unwrap();
        let eps = pld.epsilon_at(1e-5);
        assert!(eps > 0.0 && eps.is_finite(), "ε={}", eps);
    }
}
