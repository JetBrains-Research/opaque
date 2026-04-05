//! Poisson-subsampled truncated Gaussian mechanism PLD.
//!
//! Exact PLD for Poisson-subsampled DP-SGD with truncated (renormalized)
//! Gaussian noise. The density is normalized over [−Rσ, Rσ].
//!
//! The worst-case privacy loss depends on where the distribution centers
//! (determined by the query output) fall within the support domain. In DP-SGD,
//! inputs are L2-clipped to norm Δ, so per-coordinate values satisfy |x| ≤ Δ.
//! The PLD constructor optimizes over centers in [−Δ, Δ] to provide a sound bound.

use crate::adjacency::Adjacency;
use crate::discretization::{discretize_asymmetric_mechanism, DiscretizationConfig};
use crate::error::Result;
use crate::pld::PrivacyLossDistribution;
use statrs::distribution::{ContinuousCDF, Normal};

use super::{validate_noise_multiplier, validate_rate};

/// Minimum supported radius (sigma units).
const MIN_RADIUS: f64 = 0.1;
/// Maximum supported radius (sigma units).
const MAX_RADIUS: f64 = 100.0;

/// Number of grid points for worst-case center search.
const CENTER_SEARCH_POINTS: usize = 200;

/// Compute the PLD for a Poisson-subsampled truncated Gaussian mechanism.
///
/// Each record is sampled independently with probability `rate`. The noise
/// mechanism is a truncated (renormalized) Gaussian with support [−R·σ, R·σ].
///
/// The worst-case δ(ε) is computed by searching over distribution centers
/// in [−Δ, Δ], the L2-clipped input domain.
///
/// # Arguments
///
/// * `noise_multiplier` — σ/Δ ratio, must be > 0
/// * `radius` — support half-width in sigma units, in \[0.1, 100\]
/// * `rate` — Poisson sampling probability q ∈ (0, 1)
/// * `config` — discretization configuration
pub fn poisson_truncated_gaussian_pld(
    noise_multiplier: f64,
    radius: f64,
    rate: f64,
    config: &DiscretizationConfig,
) -> Result<PrivacyLossDistribution> {
    validate_noise_multiplier(noise_multiplier)?;
    validate_rate(rate)?;
    validate_radius(radius)?;

    let sigma = noise_multiplier;
    let sensitivity = 1.0;
    let log_mass = config.log_mass_truncation_bound;
    let tail_budget = config.tail_mass_truncation / 2.0;

    // Use Gaussian Poisson epsilon bounds (always a safe envelope).
    let bounds_remove = super::poisson::poisson_gaussian_epsilon_bounds(
        sigma,
        sensitivity,
        rate,
        Adjacency::Remove,
        log_mass,
    );
    let bounds_add = super::poisson::poisson_gaussian_epsilon_bounds(
        sigma,
        sensitivity,
        rate,
        Adjacency::Add,
        log_mass,
    );

    discretize_asymmetric_mechanism(config, bounds_remove, bounds_add, |epsilon, adj| {
        Ok(get_delta_truncated_worst_case(
            epsilon, adj, sigma, sensitivity, radius, rate,
        ))
    })
    .map(|pld| pld.with_tail_budgets(tail_budget, tail_budget))
}

fn validate_radius(radius: f64) -> Result<()> {
    if !(MIN_RADIUS..=MAX_RADIUS).contains(&radius) {
        return Err(crate::error::PldError::InvalidParameter(format!(
            "radius must be in [{}, {}], got {}",
            MIN_RADIUS, MAX_RADIUS, radius
        )));
    }
    Ok(())
}

// ===========================================================================
// Hockey-stick divergence
// ===========================================================================

/// Compute normalization constants Z(μ₀) and Z(μ₁) for truncated Gaussians
/// centered at μ₀ and μ₁ = μ₀ + Δ on the domain [−Rσ, Rσ].
fn normalization_constants(
    sigma: f64,
    sensitivity: f64,
    radius: f64,
    mu0: f64,
) -> (f64, f64) {
    let n01 = Normal::new(0.0, 1.0).unwrap();
    let r_abs = radius * sigma;
    let mu1 = mu0 + sensitivity;
    let z0 = n01.cdf((r_abs - mu0) / sigma) - n01.cdf((-r_abs - mu0) / sigma);
    let z1 = n01.cdf((r_abs - mu1) / sigma) - n01.cdf((-r_abs - mu1) / sigma);
    (z0, z1)
}

/// Hockey-stick δ(ε) for Poisson-subsampled truncated Gaussian,
/// maximized over worst-case distribution centers in [−Δ, Δ].
fn get_delta_truncated_worst_case(
    epsilon: f64,
    adjacency: Adjacency,
    sigma: f64,
    sensitivity: f64,
    radius: f64,
    rate: f64,
) -> f64 {
    // After L2 clipping to norm Δ, per-coordinate inputs are in [−Δ, Δ].
    let search_lo = -sensitivity;
    let search_hi = sensitivity;
    let step = (search_hi - search_lo) / (CENTER_SEARCH_POINTS as f64);

    let mut max_delta = 0.0_f64;
    for i in 0..=CENTER_SEARCH_POINTS {
        let mu0 = search_lo + step * (i as f64);
        let (z0, z1) = normalization_constants(sigma, sensitivity, radius, mu0);
        if z0 <= 0.0 || z1 <= 0.0 {
            continue;
        }
        let d = get_delta_truncated(epsilon, adjacency, sigma, sensitivity, radius, rate, mu0, z0, z1);
        max_delta = max_delta.max(d);
    }
    max_delta
}

/// Hockey-stick δ(ε) for a specific center μ₀.
///
/// # Large-radius delegation
///
/// When Z₀ ≈ Z₁ ≈ 1 (radius large enough that the truncation is
/// negligible), delegates to the numerically-stable log-space
/// implementation in `poisson.rs`.
#[allow(clippy::too_many_arguments)]
fn get_delta_truncated(
    epsilon: f64,
    adjacency: Adjacency,
    sigma: f64,
    sensitivity: f64,
    radius: f64,
    rate: f64,
    mu0: f64,
    z0: f64,
    z1: f64,
) -> f64 {
    // When normalisation constants are effectively 1, delegate to standard
    // Poisson delta (log-space tail arithmetic).
    if (z0 - 1.0).abs() < 1e-10 && (z1 - 1.0).abs() < 1e-10 {
        return crate::amplification::poisson::poisson_gaussian_get_delta(
            epsilon,
            adjacency,
            sigma,
            sensitivity,
            rate,
        );
    }

    match adjacency {
        Adjacency::Remove => {
            get_delta_remove_truncated(epsilon, sigma, sensitivity, radius, rate, mu0, z0, z1)
        }
        Adjacency::Add => {
            get_delta_add_truncated(epsilon, sigma, sensitivity, radius, rate, mu0, z0, z1)
        }
        Adjacency::Replace => {
            let d_rem =
                get_delta_remove_truncated(epsilon, sigma, sensitivity, radius, rate, mu0, z0, z1);
            let d_add =
                get_delta_add_truncated(epsilon, sigma, sensitivity, radius, rate, mu0, z0, z1);
            d_rem.max(d_add)
        }
    }
}

/// REMOVE adjacency: D has the extra record.
///
/// p_D(x)  = q·f₁(x) + (1−q)·f₀(x)
/// p_D'(x) = f₀(x)
///
/// where f_μ(x) = φ((x−μ)/σ) / (σ·Z_μ), with μ₀ and μ₁ = μ₀ + Δ.
#[allow(clippy::too_many_arguments)]
fn get_delta_remove_truncated(
    epsilon: f64,
    sigma: f64,
    sensitivity: f64,
    radius: f64,
    rate: f64,
    mu0: f64,
    z0: f64,
    z1: f64,
) -> f64 {
    let n01 = Normal::new(0.0, 1.0).unwrap();
    let q = rate;
    let r_abs = radius * sigma;
    let mu1 = mu0 + sensitivity;
    let exp_eps = epsilon.exp();

    if exp_eps <= 1.0 - q {
        return (1.0 - exp_eps).max(0.0);
    }

    let a_shifted = q / z1;
    let a_base = (exp_eps - (1.0 - q)) / z0;

    if a_base <= 0.0 {
        return 1.0 - exp_eps;
    }

    // Crossover: a_shifted · φ̃₁(x) = a_base · φ̃₀(x)
    // φ̃₁(x)/φ̃₀(x) = exp(Δ(x − μ₀ − Δ/2)/σ²)
    // exp(Δ(x − μ₀ − Δ/2)/σ²) = a_base / a_shifted
    let ratio = a_base / a_shifted;
    let x_cross = mu0 + sensitivity / 2.0 + sigma * sigma / sensitivity * ratio.ln();

    let int_lo = x_cross.max(-r_abs);
    let int_hi = r_abs;

    if int_hi <= int_lo {
        return 0.0;
    }

    let cdf_base_hi = n01.cdf((int_hi - mu0) / sigma);
    let cdf_base_lo = n01.cdf((int_lo - mu0) / sigma);
    let cdf_shift_hi = n01.cdf((int_hi - mu1) / sigma);
    let cdf_shift_lo = n01.cdf((int_lo - mu1) / sigma);

    (a_shifted * (cdf_shift_hi - cdf_shift_lo) - a_base * (cdf_base_hi - cdf_base_lo)).max(0.0)
}

/// ADD adjacency: D' has the extra record.
///
/// p_D(x)  = f₀(x)
/// p_D'(x) = q·f₁(x) + (1−q)·f₀(x)
#[allow(clippy::too_many_arguments)]
fn get_delta_add_truncated(
    epsilon: f64,
    sigma: f64,
    sensitivity: f64,
    radius: f64,
    rate: f64,
    mu0: f64,
    z0: f64,
    z1: f64,
) -> f64 {
    let n01 = Normal::new(0.0, 1.0).unwrap();
    let q = rate;
    let r_abs = radius * sigma;
    let mu1 = mu0 + sensitivity;
    let exp_eps = epsilon.exp();

    let theoretical_upper = if q < 1.0 {
        -(1.0 - q).ln()
    } else {
        f64::INFINITY
    };
    if epsilon >= theoretical_upper - 1e-10 {
        return 0.0;
    }

    let a_base = (1.0 - exp_eps * (1.0 - q)) / z0;
    let a_shifted = exp_eps * q / z1;

    if a_base <= 0.0 {
        return 0.0;
    }

    // Crossover: a_base · φ̃₀(x) = a_shifted · φ̃₁(x)
    // φ̃₁(x)/φ̃₀(x) = exp(Δ(x − μ₀ − Δ/2)/σ²)
    let ratio = a_base / a_shifted;
    let x_cross = if ratio > 0.0 && ratio.is_finite() {
        mu0 + sensitivity / 2.0 + sigma * sigma / sensitivity * ratio.ln()
    } else if ratio <= 0.0 {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    };

    let int_lo = -r_abs;
    let int_hi = x_cross.min(r_abs);

    if int_hi <= int_lo {
        return 0.0;
    }

    let cdf_base_hi = n01.cdf((int_hi - mu0) / sigma);
    let cdf_base_lo = n01.cdf((int_lo - mu0) / sigma);
    let cdf_shift_hi = n01.cdf((int_hi - mu1) / sigma);
    let cdf_shift_lo = n01.cdf((int_lo - mu1) / sigma);

    (a_base * (cdf_base_hi - cdf_base_lo) - a_shifted * (cdf_shift_hi - cdf_shift_lo)).max(0.0)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> DiscretizationConfig {
        DiscretizationConfig::default()
    }

    #[test]
    fn test_poisson_truncated_rejects_bad_params() {
        let cfg = default_config();
        assert!(poisson_truncated_gaussian_pld(1e-7, 3.0, 0.01, &cfg).is_err()); // bad nm
        assert!(poisson_truncated_gaussian_pld(0.5, 0.05, 0.01, &cfg).is_err()); // bad radius
        assert!(poisson_truncated_gaussian_pld(0.5, 3.0, 0.0, &cfg).is_err()); // bad rate
        assert!(poisson_truncated_gaussian_pld(0.5, 3.0, 1.5, &cfg).is_err()); // bad rate
    }

    #[test]
    fn test_poisson_truncated_valid_params() {
        let cfg = default_config();
        assert!(poisson_truncated_gaussian_pld(0.5, 3.0, 0.01, &cfg).is_ok());
        assert!(poisson_truncated_gaussian_pld(0.5, 3.0, 0.99, &cfg).is_ok());
    }

    /// Poisson subsampling should reduce epsilon compared to base mechanism.
    #[test]
    fn test_poisson_truncated_amplification() {
        let cfg = default_config();
        let eps_base = crate::mechanisms::truncated_gaussian_pld(0.5, 3.0, &cfg)
            .unwrap()
            .epsilon_at(1e-5);
        let eps_sub = poisson_truncated_gaussian_pld(0.5, 3.0, 0.01, &cfg)
            .unwrap()
            .epsilon_at(1e-5);
        assert!(
            eps_sub < eps_base,
            "Poisson should reduce ε: {} vs {}",
            eps_sub,
            eps_base
        );
    }

    /// Poisson-subsampled truncated ε ≤ Poisson Gaussian ε.
    #[test]
    fn test_poisson_truncated_tighter_than_gaussian() {
        let cfg = default_config();
        let eps_gauss = crate::amplification::poisson_gaussian_pld(0.5, 0.01, &cfg)
            .unwrap()
            .epsilon_at(1e-5);
        let eps_trunc = poisson_truncated_gaussian_pld(0.5, 3.0, 0.01, &cfg)
            .unwrap()
            .epsilon_at(1e-5);

        assert!(
            eps_trunc <= eps_gauss + 1e-4,
            "truncated ε={:.6} should ≤ Gaussian ε={:.6}",
            eps_trunc,
            eps_gauss
        );
    }

    /// Rate monotonicity: higher rate → higher epsilon.
    #[test]
    fn test_poisson_truncated_rate_monotonicity() {
        let cfg = default_config();
        let rates = [0.001, 0.01, 0.1, 0.5];
        let epsilons: Vec<f64> = rates
            .iter()
            .map(|&q| {
                poisson_truncated_gaussian_pld(0.5, 3.0, q, &cfg)
                    .unwrap()
                    .epsilon_at(1e-5)
            })
            .collect();
        for w in epsilons.windows(2) {
            assert!(
                w[0] <= w[1] + 1e-6,
                "Higher rate should give higher ε: {} vs {}",
                w[0],
                w[1]
            );
        }
    }

    /// At large radius, should be close to standard Poisson Gaussian.
    /// With centers restricted to [-Δ, Δ] (clipped input domain), the
    /// With centers restricted to [-Δ, Δ] (clipped input domain), the
    /// normalization constants are ≈1 at R=50, so the gap is negligible.
    #[test]
    fn test_poisson_truncated_converges_to_gaussian() {
        let cfg = default_config();
        let nm = 0.5;
        let rate = 0.01;
        let radius = 50.0;

        let pld_gauss = crate::amplification::poisson_gaussian_pld(nm, rate, &cfg).unwrap();
        let pld_trunc = poisson_truncated_gaussian_pld(nm, radius, rate, &cfg).unwrap();

        for delta in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7] {
            let eg = pld_gauss.epsilon_at(delta);
            let et = pld_trunc.epsilon_at(delta);
            assert!(
                (et - eg).abs() < 0.01,
                "At R=50, delta={:.0e}: trunc={:.6} should ≈ gauss={:.6}",
                delta,
                et,
                eg,
            );
        }
    }
}
