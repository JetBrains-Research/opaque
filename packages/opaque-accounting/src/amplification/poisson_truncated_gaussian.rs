//! Poisson-subsampled truncated Gaussian mechanism PLD.
//!
//! Exact PLD for Poisson-subsampled DP-SGD with truncated (renormalized)
//! Gaussian noise. The density is normalized over [−Rσ, Rσ] — no point
//! masses, just tighter tails from the normalization constants Z₀, Z₁.

use crate::adjacency::Adjacency;
use crate::discretization::{discretize_asymmetric_mechanism, DiscretizationConfig, EpsilonBounds};
use crate::error::Result;
use crate::pld::PrivacyLossDistribution;
use statrs::distribution::{ContinuousCDF, Normal};

use super::{validate_noise_multiplier, validate_rate};

/// Minimum supported radius (sigma units).
const MIN_RADIUS: f64 = 0.1;
/// Maximum supported radius (sigma units).
const MAX_RADIUS: f64 = 100.0;

/// Compute the PLD for a Poisson-subsampled truncated Gaussian mechanism.
///
/// Each record is sampled independently with probability `rate`. The noise
/// mechanism is a truncated (renormalized) Gaussian with support [−R·σ, R·σ].
///
/// The truncated Gaussian density is:
///   f_μ(x) = φ((x−μ)/σ) / (σ · Z_μ)  for x ∈ [−Rσ, Rσ]
/// where Z_μ = Φ((Rσ−μ)/σ) − Φ((−Rσ−μ)/σ) is the normalization constant.
///
/// # Arguments
///
/// * `noise_multiplier` — σ/Δ ratio, must be in \[0.1, 1.2\]
/// * `radius` — support half-width in sigma units, in \[0.1, 100\]
/// * `rate` — Poisson sampling probability q ∈ (0, 1\]
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

    // Normalization constants
    let n01 = Normal::new(0.0, 1.0).unwrap();
    let z0 = n01.cdf(radius) - n01.cdf(-radius);
    let z1 = n01.cdf(radius - sensitivity / sigma) - n01.cdf(-radius - sensitivity / sigma);

    let bounds_remove = epsilon_bounds_truncated(
        sigma,
        sensitivity,
        radius,
        rate,
        z0,
        z1,
        Adjacency::Remove,
        log_mass,
    );
    let bounds_add = epsilon_bounds_truncated(
        sigma,
        sensitivity,
        radius,
        rate,
        z0,
        z1,
        Adjacency::Add,
        log_mass,
    );

    discretize_asymmetric_mechanism(config, bounds_remove, bounds_add, |epsilon, adj| {
        Ok(get_delta_truncated(
            epsilon,
            adj,
            sigma,
            sensitivity,
            radius,
            rate,
            z0,
            z1,
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

/// Hockey-stick δ(ε) for Poisson-subsampled truncated Gaussian.
///
/// No point masses — the entire density is continuous on [−Rσ, Rσ].
///
/// # Large-radius delegation
///
/// When Z₀ ≈ Z₁ ≈ 1 (radius large enough that the truncation is
/// negligible), the truncated Gaussian is effectively identical to the
/// unbounded Gaussian.  In this regime we delegate to the numerically-stable
/// log-space implementation in `poisson.rs` to avoid floating-point
/// amplification artifacts in the Connect-the-Dots discretization.
#[allow(clippy::too_many_arguments)]
fn get_delta_truncated(
    epsilon: f64,
    adjacency: Adjacency,
    sigma: f64,
    sensitivity: f64,
    radius: f64,
    rate: f64,
    z0: f64,
    z1: f64,
) -> f64 {
    // When normalisation constants are effectively 1, the truncated density
    // coincides with the unbounded Gaussian density.  Delegate to the standard
    // Poisson delta which uses log-space tail arithmetic.
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
            get_delta_remove_truncated(epsilon, sigma, sensitivity, radius, rate, z0, z1)
        }
        Adjacency::Add => {
            get_delta_add_truncated(epsilon, sigma, sensitivity, radius, rate, z0, z1)
        }
        Adjacency::Replace => {
            let d_rem =
                get_delta_remove_truncated(epsilon, sigma, sensitivity, radius, rate, z0, z1);
            let d_add = get_delta_add_truncated(epsilon, sigma, sensitivity, radius, rate, z0, z1);
            d_rem.max(d_add)
        }
    }
}

/// REMOVE adjacency: D has the extra record.
///
/// p_D(x)  = q·f₁(x) + (1−q)·f₀(x)
/// p_D'(x) = f₀(x)
///
/// where f_μ(x) = φ((x−μ)/σ) / (σ·Z_μ).
///
/// Integrand: q·f₁(x) − (e^ε − (1−q))·f₀(x)
///          = (q/Z₁)·φ̃₁(x) − ((e^ε−(1−q))/Z₀)·φ̃₀(x)
/// where φ̃_μ(x) = φ((x−μ)/σ)/σ is the un-normalized Gaussian density.
fn get_delta_remove_truncated(
    epsilon: f64,
    sigma: f64,
    sensitivity: f64,
    radius: f64,
    rate: f64,
    z0: f64,
    z1: f64,
) -> f64 {
    let n01 = Normal::new(0.0, 1.0).unwrap();
    let q = rate;
    let r_abs = radius * sigma;
    let exp_eps = epsilon.exp();

    if exp_eps <= 1.0 - q {
        return (1.0 - exp_eps).max(0.0);
    }

    let a_shifted = q / z1; // coefficient of un-normalized shifted density
    let a_base = (exp_eps - (1.0 - q)) / z0; // coefficient of un-normalized base density

    if a_base <= 0.0 {
        // Entire integrand is non-negative
        return 1.0 - exp_eps;
    }

    // Crossover: a_shifted · φ̃₁(x) = a_base · φ̃₀(x)
    // exp(Δx/σ² − Δ²/(2σ²)) = a_base / a_shifted
    let ratio = a_base / a_shifted;
    let x_cross = sensitivity / 2.0 + sigma * sigma / sensitivity * ratio.ln();

    // Integrate from max(x_cross, −Rσ) to Rσ
    let int_lo = x_cross.max(-r_abs);
    let int_hi = r_abs;

    if int_hi <= int_lo {
        return 0.0;
    }

    let cdf_base_hi = n01.cdf(int_hi / sigma);
    let cdf_base_lo = n01.cdf(int_lo / sigma);
    let cdf_shift_hi = n01.cdf((int_hi - sensitivity) / sigma);
    let cdf_shift_lo = n01.cdf((int_lo - sensitivity) / sigma);

    // δ = (q/Z₁)·[Φ(...) − Φ(...)] − ((e^ε−(1−q))/Z₀)·[Φ(...) − Φ(...)]
    (a_shifted * (cdf_shift_hi - cdf_shift_lo) - a_base * (cdf_base_hi - cdf_base_lo)).max(0.0)
}

/// ADD adjacency: D' has the extra record.
///
/// p_D(x)  = f₀(x)
/// p_D'(x) = q·f₁(x) + (1−q)·f₀(x)
///
/// Integrand: (1−e^ε·(1−q))·f₀(x) − e^ε·q·f₁(x)
///          = ((1−e^ε·(1−q))/Z₀)·φ̃₀(x) − (e^ε·q/Z₁)·φ̃₁(x)
fn get_delta_add_truncated(
    epsilon: f64,
    sigma: f64,
    sensitivity: f64,
    radius: f64,
    rate: f64,
    z0: f64,
    z1: f64,
) -> f64 {
    let n01 = Normal::new(0.0, 1.0).unwrap();
    let q = rate;
    let r_abs = radius * sigma;
    let exp_eps = epsilon.exp();

    let theoretical_upper = if q < 1.0 {
        -(1.0 - q).ln()
    } else {
        f64::INFINITY
    };
    if epsilon >= theoretical_upper - 1e-10 {
        return 0.0;
    }

    let a_base = (1.0 - exp_eps * (1.0 - q)) / z0; // coefficient of base density
    let a_shifted = exp_eps * q / z1; // coefficient of shifted density [subtracted]

    if a_base <= 0.0 {
        return 0.0;
    }

    // Crossover: a_base · φ̃₀(x) = a_shifted · φ̃₁(x)
    //   φ̃₁/φ̃₀ = exp(Δ(x−Δ/2)/σ²) must equal a_base/a_shifted
    //   x_cross = Δ/2 + σ²/Δ · ln(a_base / a_shifted)
    // Integrand positive for x < x_cross
    let ratio = a_base / a_shifted;
    let x_cross = if ratio > 0.0 && ratio.is_finite() {
        sensitivity / 2.0 + sigma * sigma / sensitivity * ratio.ln()
    } else if ratio <= 0.0 {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY // a_shifted ≈ 0 → integrand positive everywhere
    };

    // Integrand positive for x < x_cross
    let int_lo = -r_abs;
    let int_hi = x_cross.min(r_abs);

    if int_hi <= int_lo {
        return 0.0;
    }

    let cdf_base_hi = n01.cdf(int_hi / sigma);
    let cdf_base_lo = n01.cdf(int_lo / sigma);
    let cdf_shift_hi = n01.cdf((int_hi - sensitivity) / sigma);
    let cdf_shift_lo = n01.cdf((int_lo - sensitivity) / sigma);

    (a_base * (cdf_base_hi - cdf_base_lo) - a_shifted * (cdf_shift_hi - cdf_shift_lo)).max(0.0)
}

// ===========================================================================
// Epsilon bounds
// ===========================================================================

/// Epsilon bounds for Poisson-subsampled truncated Gaussian.
#[allow(clippy::too_many_arguments)]
fn epsilon_bounds_truncated(
    sigma: f64,
    sensitivity: f64,
    radius: f64,
    rate: f64,
    z0: f64,
    z1: f64,
    adjacency: Adjacency,
    log_mass: f64,
) -> EpsilonBounds {
    let n01 = Normal::new(0.0, 1.0).unwrap();
    let q = rate;
    let r_abs = radius * sigma;
    let sigma_sq = sigma * sigma;

    // Gaussian mass truncation range (same approach as standard Poisson)
    let half_mass = 0.5 * log_mass.exp();
    let z = n01.inverse_cdf(half_mass); // negative
    let lower_x_base = sigma * z; // negative
    let upper_x_base = -sigma * z; // positive

    // Base truncated Gaussian privacy loss includes normalization:
    // L_base(x) = Δ·(-Δ/2 - x)/σ² + ln(Z₀/Z₁)
    // Note: using the Poisson convention from poisson.rs where
    // L_raw(x) = Δ·(-Δ/2 - x)/σ², the truncated version adds ln(Z₀/Z₁)
    let log_z_ratio = (z0 / z1).ln(); // ln(Z₀/Z₁)

    // Poisson REMOVE privacy loss for truncated base mechanism
    let l_rem = |x: f64| -> f64 {
        let l_raw = sensitivity * (-0.5 * sensitivity - x) / sigma_sq + log_z_ratio;
        if (q - 1.0).abs() < 1e-15 {
            l_raw
        } else {
            (q * l_raw.exp() + (1.0 - q)).ln()
        }
    };
    let l_add = |x: f64| -> f64 { -l_rem(-x) };

    match adjacency {
        Adjacency::Remove => {
            let lower_x = (lower_x_base - sensitivity).max(-r_abs);
            let upper_x = upper_x_base.min(r_abs);
            EpsilonBounds {
                epsilon_lower: l_rem(upper_x),
                epsilon_upper: l_rem(lower_x),
            }
        }
        Adjacency::Add => {
            let lower_x = lower_x_base.max(-r_abs);
            let upper_x = (upper_x_base + sensitivity).min(r_abs);
            EpsilonBounds {
                epsilon_lower: l_add(upper_x),
                epsilon_upper: l_add(lower_x),
            }
        }
        Adjacency::Replace => {
            let b_rem = epsilon_bounds_truncated(
                sigma,
                sensitivity,
                radius,
                rate,
                z0,
                z1,
                Adjacency::Remove,
                log_mass,
            );
            let b_add = epsilon_bounds_truncated(
                sigma,
                sensitivity,
                radius,
                rate,
                z0,
                z1,
                Adjacency::Add,
                log_mass,
            );
            EpsilonBounds {
                epsilon_lower: b_rem.epsilon_lower.min(b_add.epsilon_lower),
                epsilon_upper: b_rem.epsilon_upper.max(b_add.epsilon_upper),
            }
        }
    }
}

/// Create a CGF-backed PLD for a Poisson-subsampled truncated Gaussian mechanism.
///
/// Uses Gauss-Hermite quadrature with truncated domain.
pub fn cgf_poisson_truncated_gaussian_pld(
    noise_multiplier: f64,
    radius: f64,
    rate: f64,
) -> Result<PrivacyLossDistribution> {
    use std::sync::Arc;
    use crate::pld::cgf::SubsampledTruncatedGaussianCgf;

    validate_noise_multiplier(noise_multiplier)?;
    validate_rate(rate)?;
    if !(MIN_RADIUS..=MAX_RADIUS).contains(&radius) {
        return Err(crate::error::PldError::InvalidParameter(format!(
            "radius must be in [{}, {}], got {}",
            MIN_RADIUS, MAX_RADIUS, radius
        )));
    }

    Ok(PrivacyLossDistribution::new_cgf(Arc::new(
        SubsampledTruncatedGaussianCgf::new(noise_multiplier, radius, rate),
    )))
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
        assert!(poisson_truncated_gaussian_pld(0.005, 3.0, 0.01, &cfg).is_err()); // bad nm
        assert!(poisson_truncated_gaussian_pld(0.5, 0.05, 0.01, &cfg).is_err()); // bad radius
        assert!(poisson_truncated_gaussian_pld(0.5, 3.0, 0.0, &cfg).is_err()); // bad rate
        assert!(poisson_truncated_gaussian_pld(0.5, 3.0, 1.5, &cfg).is_err()); // bad rate
    }

    #[test]
    fn test_poisson_truncated_valid_params() {
        let cfg = default_config();
        assert!(poisson_truncated_gaussian_pld(0.5, 3.0, 0.01, &cfg).is_ok());
        assert!(poisson_truncated_gaussian_pld(0.5, 3.0, 1.0, &cfg).is_ok());
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

    /// Poisson-subsampled truncated ε ≤ Poisson-subsampled rectified ε ≤ Poisson Gaussian ε.
    #[test]
    fn test_poisson_truncated_tighter_than_rectified() {
        let cfg = default_config();
        let eps_gauss = crate::amplification::poisson_gaussian_pld(0.5, 0.01, &cfg)
            .unwrap()
            .epsilon_at(1e-5);
        let eps_rect = crate::amplification::poisson_rectified_gaussian_pld(0.5, 3.0, 0.01, &cfg)
            .unwrap()
            .epsilon_at(1e-5);
        let eps_trunc = poisson_truncated_gaussian_pld(0.5, 3.0, 0.01, &cfg)
            .unwrap()
            .epsilon_at(1e-5);

        assert!(
            eps_trunc <= eps_rect + 1e-4,
            "truncated ε={:.6} should ≤ rectified ε={:.6}",
            eps_trunc,
            eps_rect
        );
        assert!(
            eps_rect <= eps_gauss + 1e-4,
            "rectified ε={:.6} should ≤ Gaussian ε={:.6}",
            eps_rect,
            eps_gauss
        );
    }

    /// Rate monotonicity: higher rate → higher epsilon.
    #[test]
    fn test_poisson_truncated_rate_monotonicity() {
        let cfg = default_config();
        let rates = [0.001, 0.01, 0.1, 0.5, 1.0];
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

    /// At large radius, should converge to standard Poisson Gaussian.
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
