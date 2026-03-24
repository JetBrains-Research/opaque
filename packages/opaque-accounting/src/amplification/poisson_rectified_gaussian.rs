//! Poisson-subsampled rectified Gaussian mechanism PLD.
//!
//! Exact PLD for Poisson-subsampled DP-SGD with rectified (clamped) Gaussian
//! noise. Computes the hockey-stick divergence over the bounded domain
//! [−Rσ, Rσ] including point mass contributions at the boundaries.

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

/// Compute the PLD for a Poisson-subsampled rectified Gaussian mechanism.
///
/// Each record is sampled independently with probability `rate`. The noise
/// mechanism is a rectified (clamped) Gaussian with support [−R·σ, R·σ].
///
/// The exact PLD accounts for both the continuous interior density and the
/// point masses at the domain boundaries.
///
/// # Arguments
///
/// * `noise_multiplier` — σ/Δ ratio, must be in \[0.1, 1.2\]
/// * `radius` — support half-width in sigma units, in \[0.1, 100\]
/// * `rate` — Poisson sampling probability q ∈ (0, 1\]
/// * `config` — discretization configuration
pub fn poisson_rectified_gaussian_pld(
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

    let bounds_remove = epsilon_bounds_rectified(
        sigma,
        sensitivity,
        radius,
        rate,
        Adjacency::Remove,
        log_mass,
    );
    let bounds_add =
        epsilon_bounds_rectified(sigma, sensitivity, radius, rate, Adjacency::Add, log_mass);

    discretize_asymmetric_mechanism(config, bounds_remove, bounds_add, |epsilon, adj| {
        Ok(get_delta_rectified(
            epsilon,
            adj,
            sigma,
            sensitivity,
            radius,
            rate,
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

/// Hockey-stick δ(ε) for Poisson-subsampled rectified Gaussian.
///
/// # Mathematical identity
///
/// For a rectified (clamped) Gaussian, the point masses at ±Rσ exactly
/// compensate for the truncated tails.  In particular, when the crossover
/// x_cutoff lies inside the bounded domain [−Rσ, Rσ], we have:
///
///   ∫_{x_cutoff}^{Rσ} ΔM(x) dx + mass_right = ∫_{x_cutoff}^{∞} ΔM(x) dx
///
/// (where ΔM denotes the hockey-stick integrand).  This is because
/// Φ(hi/σ) − Φ(lo/σ) + Φ(−hi/σ) = 1 − Φ(lo/σ), i.e. the interior CDF
/// plus the clamped point mass equals the unbounded tail.
///
/// Therefore for x_cutoff ∈ [−Rσ, Rσ] the rectified δ(ε) is *identical*
/// to the standard Poisson-subsampled Gaussian δ(ε), and we delegate to the
/// numerically-stable log-space implementation in `poisson.rs`.
///
/// Edge cases (x_cutoff outside the domain) are handled with direct
/// point-mass arithmetic.
fn get_delta_rectified(
    epsilon: f64,
    adjacency: Adjacency,
    sigma: f64,
    sensitivity: f64,
    radius: f64,
    rate: f64,
) -> f64 {
    match adjacency {
        Adjacency::Remove => get_delta_remove_rectified(epsilon, sigma, sensitivity, radius, rate),
        Adjacency::Add => get_delta_add_rectified(epsilon, sigma, sensitivity, radius, rate),
        Adjacency::Replace => {
            let d_rem = get_delta_remove_rectified(epsilon, sigma, sensitivity, radius, rate);
            let d_add = get_delta_add_rectified(epsilon, sigma, sensitivity, radius, rate);
            d_rem.max(d_add)
        }
    }
}

/// Compute x_cutoff for REMOVE/ADD from epsilon.
///
/// Returns None when the cutoff is undefined (early-exit regimes).
fn crossover_x(epsilon: f64, sigma: f64, sensitivity: f64, rate: f64) -> Option<f64> {
    let exp_eps = epsilon.exp();
    let q = rate;
    let ratio = (exp_eps - (1.0 - q)) / q;
    if ratio <= 0.0 {
        return None;
    }
    let l_base = -ratio.ln();
    let sigma_sq = sigma * sigma;
    Some(0.5 * sensitivity - l_base * sigma_sq / sensitivity)
}

/// REMOVE adjacency δ(ε).
///
/// For x_cutoff in [−Rσ, Rσ]: delegates to standard Poisson Gaussian delta
/// (proven identical via the clamping-tail identity).
/// For x_cutoff outside: point-mass-only computation.
fn get_delta_remove_rectified(
    epsilon: f64,
    sigma: f64,
    sensitivity: f64,
    radius: f64,
    rate: f64,
) -> f64 {
    let q = rate;
    let r_abs = radius * sigma;
    let exp_eps = epsilon.exp();

    // Standard early exits (same as poisson.rs)
    let theoretical_lower = (1.0 - q).ln();
    if epsilon <= theoretical_lower {
        return (-epsilon.exp_m1()).max(0.0);
    }

    let x_cutoff = match crossover_x(epsilon, sigma, sensitivity, rate) {
        Some(x) => x,
        None => return (-epsilon.exp_m1()).max(0.0),
    };

    if x_cutoff <= -r_abs {
        // Crossover below left boundary → integrand positive over entire domain
        return (1.0 - exp_eps).max(0.0);
    }

    if x_cutoff >= r_abs {
        // Crossover above right boundary → only the right point mass contributes
        let n01 = Normal::new(0.0, 1.0).unwrap();
        let p_base = n01.cdf(-radius);
        let p_shifted = n01.cdf(sensitivity / sigma - radius);
        let pd = q * p_shifted + (1.0 - q) * p_base;
        return (pd - exp_eps * p_base).max(0.0);
    }

    // Common case: x_cutoff inside domain → rectified = standard (identity)
    crate::amplification::poisson::poisson_gaussian_get_delta(
        epsilon,
        Adjacency::Remove,
        sigma,
        sensitivity,
        rate,
    )
}

/// ADD adjacency δ(ε).
///
/// Same delegation strategy as REMOVE; integrand is positive for x < x_cutoff.
fn get_delta_add_rectified(
    epsilon: f64,
    sigma: f64,
    sensitivity: f64,
    radius: f64,
    rate: f64,
) -> f64 {
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

    // For ADD, x_cutoff from the same formula but using exp(-ε)
    let exp_neg_eps = (-epsilon).exp();
    let ratio = (exp_neg_eps - (1.0 - q)) / q;
    if ratio <= 0.0 {
        return 0.0;
    }
    let l_base = -ratio.ln();
    let sigma_sq = sigma * sigma;
    let x_cutoff = 0.5 * sensitivity - l_base * sigma_sq / sensitivity;

    if x_cutoff >= r_abs {
        // Crossover above right boundary → integrand positive over entire domain
        return (1.0 - exp_eps).max(0.0);
    }

    if x_cutoff <= -r_abs {
        // Crossover below left boundary → only the left point mass contributes
        let n01 = Normal::new(0.0, 1.0).unwrap();
        let p_base = n01.cdf(-radius);
        let p_shifted = n01.cdf(-radius - sensitivity / sigma);
        let pd_prime = q * p_shifted + (1.0 - q) * p_base;
        return (p_base - exp_eps * pd_prime).max(0.0);
    }

    // Common case: x_cutoff inside domain → rectified = standard (identity)
    crate::amplification::poisson::poisson_gaussian_get_delta(
        epsilon,
        Adjacency::Add,
        sigma,
        sensitivity,
        rate,
    )
}

// ===========================================================================
// Epsilon bounds
// ===========================================================================

/// Epsilon bounds for Poisson-subsampled rectified Gaussian.
///
/// Uses the same Gaussian mass truncation approach as `poisson_gaussian_epsilon_bounds`:
/// compute the x-range where the Gaussian density is concentrated, clip to the
/// bounded domain [−Rσ, Rσ], then evaluate the Poisson privacy loss at endpoints.
/// Point mass privacy losses are included only if the point mass probability
/// exceeds the tail mass threshold (otherwise absorbed into tail budget).
fn epsilon_bounds_rectified(
    sigma: f64,
    sensitivity: f64,
    radius: f64,
    rate: f64,
    adjacency: Adjacency,
    log_mass: f64,
) -> EpsilonBounds {
    let n01 = Normal::new(0.0, 1.0).unwrap();
    let q = rate;
    let r_abs = radius * sigma;
    let sigma_sq = sigma * sigma;

    // Gaussian mass truncation range (same as standard Poisson)
    let half_mass = 0.5 * log_mass.exp();
    let z = n01.inverse_cdf(half_mass); // negative
    let lower_x_base = sigma * z; // negative
    let upper_x_base = -sigma * z; // positive

    // Inline Poisson REMOVE privacy loss: L_rem(x) = log(1-q + q·exp(L_raw(x)))
    // where L_raw(x) = Δ·(-Δ/2 - x)/σ² (decreasing in x)
    let l_rem = |x: f64| -> f64 {
        let l_raw = sensitivity * (-0.5 * sensitivity - x) / sigma_sq;
        if (q - 1.0).abs() < 1e-15 {
            l_raw
        } else {
            (q * l_raw.exp() + (1.0 - q)).ln()
        }
    };
    // ADD privacy loss: L_add(x) = -L_rem(-x)
    let l_add = |x: f64| -> f64 { -l_rem(-x) };

    // Point mass probabilities — only include if above tail mass threshold
    let mass_threshold = log_mass.exp(); // e.g. exp(-50) ≈ 2e-22
    let p_l0 = n01.cdf(-radius);
    let p_l1 = n01.cdf(-radius - sensitivity / sigma);
    let p_r0 = n01.cdf(-radius);
    let p_r1 = n01.cdf(sensitivity / sigma - radius);

    // Helper: compute point mass privacy loss for REMOVE if above threshold
    let point_mass_rem = |p0: f64, p1: f64| -> Option<f64> {
        if p0 > mass_threshold {
            Some(((q * p1 + (1.0 - q) * p0) / p0).ln())
        } else {
            None
        }
    };
    // Helper: compute point mass privacy loss for ADD if above threshold
    let point_mass_add = |p0: f64, p1: f64| -> Option<f64> {
        let pd_prime = q * p1 + (1.0 - q) * p0;
        if pd_prime > mass_threshold {
            Some((p0 / pd_prime).ln())
        } else {
            None
        }
    };

    match adjacency {
        Adjacency::Remove => {
            // Interior: clip x-range to [-Rσ, Rσ], evaluate L_rem
            let lower_x = (lower_x_base - sensitivity).max(-r_abs);
            let upper_x = upper_x_base.min(r_abs);
            // L_rem is decreasing → max at lower_x, min at upper_x
            let mut eps_upper = l_rem(lower_x);
            let mut eps_lower = l_rem(upper_x);

            // Extend bounds with point masses (if non-negligible)
            if let Some(eps) = point_mass_rem(p_l0, p_l1) {
                eps_upper = eps_upper.max(eps);
                eps_lower = eps_lower.min(eps);
            }
            if let Some(eps) = point_mass_rem(p_r0, p_r1) {
                eps_upper = eps_upper.max(eps);
                eps_lower = eps_lower.min(eps);
            }

            EpsilonBounds {
                epsilon_lower: eps_lower,
                epsilon_upper: eps_upper,
            }
        }
        Adjacency::Add => {
            // Interior: clip x-range to [-Rσ, Rσ], evaluate L_add
            let lower_x = lower_x_base.max(-r_abs);
            let upper_x = (upper_x_base + sensitivity).min(r_abs);
            // L_add = -L_rem(-x): increasing in poisson.rs convention → max at upper_x
            let mut eps_upper = l_add(lower_x);
            let mut eps_lower = l_add(upper_x);

            // Extend bounds with point masses (if non-negligible)
            if let Some(eps) = point_mass_add(p_l0, p_l1) {
                eps_upper = eps_upper.max(eps);
                eps_lower = eps_lower.min(eps);
            }
            if let Some(eps) = point_mass_add(p_r0, p_r1) {
                eps_upper = eps_upper.max(eps);
                eps_lower = eps_lower.min(eps);
            }

            EpsilonBounds {
                epsilon_lower: eps_lower,
                epsilon_upper: eps_upper,
            }
        }
        Adjacency::Replace => {
            let b_rem = epsilon_bounds_rectified(
                sigma,
                sensitivity,
                radius,
                rate,
                Adjacency::Remove,
                log_mass,
            );
            let b_add = epsilon_bounds_rectified(
                sigma,
                sensitivity,
                radius,
                rate,
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

/// Create a CGF-backed PLD for a Poisson-subsampled rectified Gaussian mechanism.
///
/// The Poisson-subsampled rectified Gaussian's privacy loss is equivalent to the
/// standard Poisson-subsampled Gaussian for most practical parameter ranges
/// (the point masses exactly compensate for the truncated tails).
/// This delegates to `SubsampledGaussianCgf` for efficiency.
pub fn cgf_poisson_rectified_gaussian_pld(
    noise_multiplier: f64,
    radius: f64,
    rate: f64,
) -> Result<PrivacyLossDistribution> {
    use std::sync::Arc;
    use crate::pld::cgf::SubsampledGaussianCgf;

    validate_noise_multiplier(noise_multiplier)?;
    validate_rate(rate)?;
    if !(MIN_RADIUS..=MAX_RADIUS).contains(&radius) {
        return Err(crate::error::PldError::InvalidParameter(format!(
            "radius must be in [{}, {}], got {}",
            MIN_RADIUS, MAX_RADIUS, radius
        )));
    }

    // Delegate to the standard subsampled Gaussian CGF.
    // The rectified Gaussian's privacy loss is equivalent for Poisson subsampling
    // when the crossover point falls within the domain.
    let _ = radius; // Radius doesn't affect the CGF approximation
    Ok(PrivacyLossDistribution::new_cgf(Arc::new(
        SubsampledGaussianCgf::new(noise_multiplier, rate),
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
    fn test_poisson_rectified_rejects_bad_params() {
        let cfg = default_config();
        assert!(poisson_rectified_gaussian_pld(0.005, 3.0, 0.01, &cfg).is_err()); // bad nm
        assert!(poisson_rectified_gaussian_pld(0.5, 0.05, 0.01, &cfg).is_err()); // bad radius
        assert!(poisson_rectified_gaussian_pld(0.5, 3.0, 0.0, &cfg).is_err()); // bad rate
        assert!(poisson_rectified_gaussian_pld(0.5, 3.0, 1.5, &cfg).is_err()); // bad rate
    }

    #[test]
    fn test_poisson_rectified_valid_params() {
        let cfg = default_config();
        assert!(poisson_rectified_gaussian_pld(0.5, 3.0, 0.01, &cfg).is_ok());
        assert!(poisson_rectified_gaussian_pld(0.5, 3.0, 1.0, &cfg).is_ok());
    }

    /// Poisson subsampling should reduce epsilon compared to base mechanism.
    #[test]
    fn test_poisson_rectified_amplification() {
        let cfg = default_config();
        let eps_base = crate::mechanisms::rectified_gaussian_pld(0.5, 3.0, &cfg)
            .unwrap()
            .epsilon_at(1e-5);
        let eps_sub = poisson_rectified_gaussian_pld(0.5, 3.0, 0.01, &cfg)
            .unwrap()
            .epsilon_at(1e-5);
        assert!(
            eps_sub < eps_base,
            "Poisson should reduce ε: {} vs {}",
            eps_sub,
            eps_base
        );
    }

    /// Poisson-subsampled rectified ε ≤ Poisson-subsampled standard Gaussian ε.
    #[test]
    fn test_poisson_rectified_tighter_than_poisson_gaussian() {
        let cfg = default_config();
        let eps_gauss = crate::amplification::poisson_gaussian_pld(0.5, 0.01, &cfg)
            .unwrap()
            .epsilon_at(1e-5);
        let eps_rect = poisson_rectified_gaussian_pld(0.5, 3.0, 0.01, &cfg)
            .unwrap()
            .epsilon_at(1e-5);
        assert!(
            eps_rect <= eps_gauss + 1e-4,
            "Poisson rectified ε={:.6} should ≤ Poisson Gaussian ε={:.6}",
            eps_rect,
            eps_gauss
        );
    }

    /// Rate monotonicity: higher rate → higher epsilon.
    #[test]
    fn test_poisson_rectified_rate_monotonicity() {
        let cfg = default_config();
        let rates = [0.001, 0.01, 0.1, 0.5, 1.0];
        let epsilons: Vec<f64> = rates
            .iter()
            .map(|&q| {
                poisson_rectified_gaussian_pld(0.5, 3.0, q, &cfg)
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
    fn test_poisson_rectified_converges_to_gaussian() {
        let cfg = default_config();
        let nm = 0.5;
        let rate = 0.01;
        let radius = 50.0;

        let pld_gauss = crate::amplification::poisson_gaussian_pld(nm, rate, &cfg).unwrap();
        let pld_rect = poisson_rectified_gaussian_pld(nm, radius, rate, &cfg).unwrap();

        for delta in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7] {
            let eg = pld_gauss.epsilon_at(delta);
            let er = pld_rect.epsilon_at(delta);
            assert!(
                (er - eg).abs() < 0.01,
                "At R=50, delta={:.0e}: rect={:.6} should ≈ gauss={:.6}",
                delta,
                er,
                eg,
            );
        }
    }
}
