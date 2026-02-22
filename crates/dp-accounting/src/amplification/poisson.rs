//! Poisson-subsampled Gaussian mechanism PLD.

use crate::adjacency::Adjacency;
use crate::discretization::{discretize_asymmetric_mechanism, DiscretizationConfig, EpsilonBounds};
use crate::error::Result;
use crate::numerics::logspace::{log_a_times_exp_b_plus_c, log_add};
use crate::numerics::special::{arcsinh_exp, gaussian_log_cdf, log_sinh};
use crate::pld::PrivacyLossDistribution;
use statrs::distribution::{ContinuousCDF, Normal};

use super::{validate_noise_multiplier, validate_rate};

/// Compute the PLD for a Poisson-subsampled Gaussian mechanism.
///
/// Each record is sampled independently with probability `rate`. The result
/// is an asymmetric PLD (different for ADD vs REMOVE adjacency).
///
/// # Arguments
///
/// * `noise_multiplier` — σ/Δ, must be in \[0.1, 1.2\]
/// * `rate` — Poisson sampling probability q ∈ (0, 1\]
/// * `config` — discretization configuration
///
/// # Errors
///
/// Returns `InvalidParameter` if parameters are out of range.
pub fn poisson_gaussian_pld(
    noise_multiplier: f64,
    rate: f64,
    config: &DiscretizationConfig,
) -> Result<PrivacyLossDistribution> {
    validate_noise_multiplier(noise_multiplier)?;
    validate_rate(rate)?;

    let sigma = noise_multiplier;
    let sensitivity = 1.0;
    let log_mass = config.log_mass_truncation_bound;
    let tail_budget = config.tail_mass_truncation / 2.0;

    let bounds_remove =
        poisson_gaussian_epsilon_bounds(sigma, sensitivity, rate, Adjacency::Remove, log_mass);
    let bounds_add =
        poisson_gaussian_epsilon_bounds(sigma, sensitivity, rate, Adjacency::Add, log_mass);

    discretize_asymmetric_mechanism(config, bounds_remove, bounds_add, |epsilon, adj| {
        Ok(poisson_gaussian_get_delta(
            epsilon,
            adj,
            sigma,
            sensitivity,
            rate,
        ))
    })
    .map(|pld| pld.with_tail_budgets(tail_budget, tail_budget))
}

// ===========================================================================
// Poisson math: privacy loss at a point
// ===========================================================================

/// Poisson-transformed privacy loss for REMOVE adjacency.
///
/// `L_rem(x) = log(1−q + q·exp(L_raw(x)))` where `L_raw(x) = Δ·(−0.5Δ − x) / σ²`
fn privacy_loss_remove(x: f64, sigma: f64, sensitivity: f64, rate: f64) -> f64 {
    let sigma_sq = sigma * sigma;
    let l_raw = sensitivity * (-0.5 * sensitivity - x) / sigma_sq;

    if (rate - 1.0).abs() < 1e-15 {
        return l_raw;
    }

    log_a_times_exp_b_plus_c(rate, l_raw, 1.0 - rate)
}

/// Poisson-transformed privacy loss for ADD adjacency.
///
/// By symmetry: `L_add(x) = −L_rem(−x)`
fn privacy_loss_add(x: f64, sigma: f64, sensitivity: f64, rate: f64) -> f64 {
    -privacy_loss_remove(-x, sigma, sensitivity, rate)
}

/// Poisson-transformed privacy loss for REPLACE adjacency.
fn privacy_loss_replace(x: f64, sigma: f64, sensitivity: f64, rate: f64) -> f64 {
    let sigma_sq = sigma * sigma;
    let q = rate;

    let log_ratio_plus = -(2.0 * x * sensitivity + sensitivity * sensitivity) / (2.0 * sigma_sq);
    let log_ratio_minus = (2.0 * x * sensitivity - sensitivity * sensitivity) / (2.0 * sigma_sq);

    let log_num = log_a_times_exp_b_plus_c(q, log_ratio_plus, 1.0 - q);
    let log_den = log_a_times_exp_b_plus_c(q, log_ratio_minus, 1.0 - q);

    log_num - log_den
}

// ===========================================================================
// Poisson math: epsilon bounds
// ===========================================================================

/// X-space truncation → epsilon bounds for Poisson-subsampled Gaussian.
pub(super) fn poisson_gaussian_epsilon_bounds(
    sigma: f64,
    sensitivity: f64,
    rate: f64,
    adjacency: Adjacency,
    log_mass_truncation_bound: f64,
) -> EpsilonBounds {
    let standard_normal = Normal::new(0.0, 1.0).unwrap();
    let half_mass = 0.5 * log_mass_truncation_bound.exp();
    let lower_x_base = sigma * standard_normal.inverse_cdf(half_mass);
    let upper_x_base = -lower_x_base;

    match adjacency {
        Adjacency::Remove => {
            let lower_x = lower_x_base - sensitivity;
            let upper_x = upper_x_base;
            EpsilonBounds {
                epsilon_lower: privacy_loss_remove(upper_x, sigma, sensitivity, rate),
                epsilon_upper: privacy_loss_remove(lower_x, sigma, sensitivity, rate),
            }
        }
        Adjacency::Add => {
            let lower_x = lower_x_base;
            let upper_x = upper_x_base + sensitivity;
            EpsilonBounds {
                epsilon_lower: privacy_loss_add(upper_x, sigma, sensitivity, rate),
                epsilon_upper: privacy_loss_add(lower_x, sigma, sensitivity, rate),
            }
        }
        Adjacency::Replace => {
            let lower_x = lower_x_base - sensitivity;
            let upper_x = upper_x_base + sensitivity;
            EpsilonBounds {
                epsilon_lower: privacy_loss_replace(upper_x, sigma, sensitivity, rate),
                epsilon_upper: privacy_loss_replace(lower_x, sigma, sensitivity, rate),
            }
        }
    }
}

// ===========================================================================
// Poisson math: inverse privacy loss
// ===========================================================================

/// Inverse privacy loss for ADD/REMOVE (Gaussian base).
pub(super) fn inverse_privacy_loss_gaussian(
    privacy_loss: f64,
    sigma: f64,
    sensitivity: f64,
) -> f64 {
    let sigma_sq = sigma * sigma;
    0.5 * sensitivity - privacy_loss * (sigma_sq / sensitivity)
}

/// Inverse privacy loss for REPLACE adjacency (arcsinh formula).
fn inverse_privacy_loss_replace(
    epsilon: f64,
    sigma: f64,
    sensitivity: f64,
    rate: f64,
) -> Result<f64> {
    let sigma_sq = sigma * sigma;

    if epsilon == 0.0 {
        return Ok(0.0);
    }

    if (rate - 1.0).abs() < 1e-15 {
        return Ok(-epsilon * sigma_sq / (2.0 * sensitivity));
    }

    let abs_eps = epsilon.abs();
    let sign_eps = epsilon.signum();

    let ds = sensitivity / sigma;
    let log_alpha = 0.5 * ds * ds + (1.0 - rate).ln() - rate.ln();
    let log_sinh_term = log_alpha + log_sinh(abs_eps / 2.0);
    let asinh_term = arcsinh_exp(log_sinh_term, -sign_eps);

    Ok((sigma_sq / sensitivity) * (asinh_term - epsilon / 2.0))
}

// ===========================================================================
// Poisson math: hockey-stick divergence (get_delta)
// ===========================================================================

/// Hockey-stick divergence for Poisson-subsampled Gaussian.
pub(super) fn poisson_gaussian_get_delta(
    epsilon: f64,
    adjacency: Adjacency,
    sigma: f64,
    sensitivity: f64,
    rate: f64,
) -> f64 {
    let q = rate;

    if (q - 1.0).abs() < 1e-15 {
        return base_gaussian_get_delta(epsilon, sigma, sensitivity, adjacency);
    }

    match adjacency {
        Adjacency::Add => get_delta_add(epsilon, sigma, sensitivity, q),
        Adjacency::Remove => get_delta_remove(epsilon, sigma, sensitivity, q),
        Adjacency::Replace => get_delta_replace(epsilon, sigma, sensitivity, q).unwrap_or(0.0),
    }
}

fn base_gaussian_get_delta(
    epsilon: f64,
    sigma: f64,
    sensitivity: f64,
    adjacency: Adjacency,
) -> f64 {
    let delta_tilde = sensitivity / sigma;
    let standard_normal = Normal::new(0.0, 1.0).unwrap();

    match adjacency {
        Adjacency::Add | Adjacency::Remove => {
            let x_upper = 0.5 * delta_tilde - epsilon / delta_tilde;
            let cdf_x = standard_normal.cdf(x_upper);
            let cdf_shifted = standard_normal.cdf(x_upper - delta_tilde);
            (cdf_x - epsilon.exp() * cdf_shifted).max(0.0)
        }
        Adjacency::Replace => {
            let dt2 = 2.0 * delta_tilde;
            let x_upper = 0.5 * dt2 - epsilon / dt2;
            let cdf_x = standard_normal.cdf(x_upper);
            let cdf_shifted = standard_normal.cdf(x_upper - dt2);
            (cdf_x - epsilon.exp() * cdf_shifted).max(0.0)
        }
    }
}

fn gaussian_cdf(z: f64) -> f64 {
    Normal::new(0.0, 1.0).unwrap().cdf(z)
}

fn get_delta_add(epsilon: f64, sigma: f64, sensitivity: f64, q: f64) -> f64 {
    let theoretical_upper = -(1.0 - q).ln();
    if epsilon >= theoretical_upper - 1e-10 {
        return 0.0;
    }

    let exp_neg_eps = (-epsilon).exp();
    let ratio = (exp_neg_eps - (1.0 - q)) / q;
    if ratio <= 0.0 {
        return 0.0;
    }
    let l_base = -ratio.ln();

    let x_cutoff = inverse_privacy_loss_gaussian(l_base, sigma, sensitivity);
    let mu_upper = gaussian_cdf(x_cutoff / sigma);

    let log_mu_upper = gaussian_log_cdf(x_cutoff / sigma);
    let log_cdf_lower = gaussian_log_cdf((x_cutoff - sensitivity) / sigma);
    let log_1_minus_q = (1.0 - q).ln();
    let log_q = q.ln();
    let log_mu_lower = log_add(log_1_minus_q + log_mu_upper, log_q + log_cdf_lower);

    (mu_upper - (epsilon + log_mu_lower).exp()).max(0.0)
}

fn get_delta_remove(epsilon: f64, sigma: f64, sensitivity: f64, q: f64) -> f64 {
    let theoretical_lower = (1.0 - q).ln();
    if epsilon <= theoretical_lower {
        return (-epsilon.exp_m1()).max(0.0);
    }

    let exp_eps = epsilon.exp();
    let ratio = (exp_eps - (1.0 - q)) / q;
    if ratio <= 0.0 {
        return (-epsilon.exp_m1()).max(0.0);
    }
    let l_base = -ratio.ln();

    let x_cutoff = inverse_privacy_loss_gaussian(l_base, sigma, sensitivity);

    let log_tail_upper = gaussian_log_cdf(-x_cutoff / sigma);
    let log_tail_shifted = gaussian_log_cdf((sensitivity - x_cutoff) / sigma);

    let log_1_minus_q = (1.0 - q).ln();
    let log_q = q.ln();
    let log_mu_upper = log_add(log_1_minus_q + log_tail_upper, log_q + log_tail_shifted);

    (log_mu_upper.exp() - (epsilon + log_tail_upper).exp()).max(0.0)
}

fn get_delta_replace(epsilon: f64, sigma: f64, sensitivity: f64, q: f64) -> Result<f64> {
    let x_cutoff = inverse_privacy_loss_replace(epsilon, sigma, sensitivity, q)?;

    let cdf_center = gaussian_cdf(x_cutoff / sigma);
    let cdf_plus = gaussian_cdf((x_cutoff + sensitivity) / sigma);
    let cdf_minus = gaussian_cdf((x_cutoff - sensitivity) / sigma);

    let mu_upper = q * cdf_plus + (1.0 - q) * cdf_center;
    let mu_lower = q * cdf_minus + (1.0 - q) * cdf_center;

    Ok((mu_upper - epsilon.exp() * mu_lower).max(0.0))
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
    fn test_poisson_rejects_bad_rate() {
        let cfg = default_config();
        assert!(poisson_gaussian_pld(0.5, 0.0, &cfg).is_err());
        assert!(poisson_gaussian_pld(0.5, -0.1, &cfg).is_err());
        assert!(poisson_gaussian_pld(0.5, 1.5, &cfg).is_err());
    }

    #[test]
    fn test_poisson_rejects_bad_noise_multiplier() {
        let cfg = default_config();
        assert!(poisson_gaussian_pld(0.05, 0.01, &cfg).is_err());
        assert!(poisson_gaussian_pld(1.5, 0.01, &cfg).is_err());
    }

    #[test]
    fn test_poisson_amplification_reduces_epsilon() {
        let cfg = default_config();
        let pld_full = crate::mechanisms::gaussian_pld(0.5, &cfg).unwrap();
        let pld_sub = poisson_gaussian_pld(0.5, 0.01, &cfg).unwrap();

        let eps_full = pld_full.epsilon_at(1e-5);
        let eps_sub = pld_sub.epsilon_at(1e-5);

        assert!(
            eps_sub < eps_full,
            "Poisson should reduce epsilon: {} vs {}",
            eps_sub,
            eps_full
        );
    }

    #[test]
    fn test_poisson_rate_monotonicity() {
        let cfg = default_config();
        let rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0];
        let epsilons: Vec<f64> = rates
            .iter()
            .map(|&q| poisson_gaussian_pld(0.5, q, &cfg).unwrap().epsilon_at(1e-5))
            .collect();

        for w in epsilons.windows(2) {
            assert!(
                w[0] <= w[1] + 1e-9,
                "higher rate should give higher epsilon: q changed: {} → {}",
                w[0],
                w[1]
            );
        }
    }

    // ---- Poisson math ----

    #[test]
    fn test_privacy_loss_remove_no_subsampling() {
        let sigma = 1.0;
        let sensitivity = 1.0;
        let x = 0.5;
        let l_raw = sensitivity * (-0.5 * sensitivity - x) / (sigma * sigma);
        let l_sub = privacy_loss_remove(x, sigma, sensitivity, 1.0);
        assert!((l_raw - l_sub).abs() < 1e-12);
    }

    #[test]
    fn test_privacy_loss_add_remove_symmetry() {
        let sigma = 1.0;
        let sensitivity = 1.0;
        let rate = 0.01;
        let x = 0.5;
        let l_add = privacy_loss_add(x, sigma, sensitivity, rate);
        let l_rem = privacy_loss_remove(-x, sigma, sensitivity, rate);
        assert!((l_add + l_rem).abs() < 1e-12);
    }

    #[test]
    fn test_privacy_loss_replace_odd_function() {
        let sigma = 1.0;
        let sensitivity = 1.0;
        let rate = 0.1;
        for x in [0.1, 0.5, 1.0, 2.0] {
            let l_pos = privacy_loss_replace(x, sigma, sensitivity, rate);
            let l_neg = privacy_loss_replace(-x, sigma, sensitivity, rate);
            assert!((l_pos + l_neg).abs() < 1e-12);
        }
    }

    #[test]
    fn test_inverse_privacy_loss_replace_roundtrip() {
        let sigma = 1.0;
        let sensitivity = 1.0;
        let rate = 0.1;

        for eps in [0.01, 0.1, 0.5, 1.0, -0.1, -0.5] {
            let x = inverse_privacy_loss_replace(eps, sigma, sensitivity, rate).unwrap();
            let l = privacy_loss_replace(x, sigma, sensitivity, rate);
            assert!(
                (l - eps).abs() < 1e-8,
                "Round-trip failed: eps={}, x={}, L(x)={}, diff={}",
                eps,
                x,
                l,
                (l - eps).abs()
            );
        }
    }
}
