//! Privacy amplification: flat functions producing PLDs for subsampled/accumulated mechanisms.
//!
//! Each function takes scalar parameters directly (no structs, no traits).

use crate::adjacency::Adjacency;
use crate::discretization::{
    discretize_asymmetric_mechanism, DiscretizationConfig, EpsilonBounds,
};
use crate::error::{PldError, Result};
use crate::math_helpers::logspace::{log_a_times_exp_b_plus_c, log_add, log_sumexp};
use crate::math_helpers::special::{arcsinh_exp, gaussian_log_cdf, log_sinh};
use crate::mechanisms::{MIN_NOISE_MULTIPLIER, MAX_NOISE_MULTIPLIER};
use crate::pld::PrivacyLossDistribution;
use statrs::distribution::{Binomial, ContinuousCDF, DiscreteCDF, Normal};

// ===========================================================================
// Public API
// ===========================================================================

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
            epsilon, adj, sigma, sensitivity, rate,
        ))
    })
    .map(|pld| pld.with_tail_budgets(tail_budget, tail_budget))
}

/// Compute the PLD for a truncated Poisson-subsampled Gaussian mechanism.
///
/// This is the variant actually used in production DP-SGD systems. Unlike
/// standard Poisson (variable batch size), truncated sampling caps the batch
/// at `batch_size_max` for predictable memory/compute.
///
/// Uses the mixture formula from \[Gan25\]:
/// - Component 1 (prob `1 − p_trunc`): standard Poisson PLD
/// - Component 2 (prob `p_trunc`): Poisson with doubled sensitivity at conditional rate
///
/// # Arguments
///
/// * `noise_multiplier` — σ/Δ, must be in \[0.1, 1.2\]
/// * `rate` — Poisson sampling probability q ∈ (0, 1\]
/// * `batch_size_max` — maximum batch size B_max > 0
/// * `dataset_size` — total dataset size n > 0
/// * `config` — discretization configuration
///
/// # Errors
///
/// Returns `InvalidParameter` if parameters are out of range.
pub fn truncated_poisson_gaussian_pld(
    noise_multiplier: f64,
    rate: f64,
    batch_size_max: usize,
    dataset_size: usize,
    config: &DiscretizationConfig,
) -> Result<PrivacyLossDistribution> {
    validate_noise_multiplier(noise_multiplier)?;
    validate_rate(rate)?;
    if batch_size_max == 0 {
        return Err(PldError::InvalidParameter(
            "batch_size_max must be > 0".into(),
        ));
    }
    if dataset_size == 0 {
        return Err(PldError::InvalidParameter("dataset_size must be > 0".into()));
    }

    let sigma = noise_multiplier;
    let sensitivity = 1.0;
    let p_trunc = truncation_probability(dataset_size, rate, batch_size_max);

    // No truncation → fall back to standard Poisson (exact)
    if p_trunc == 0.0 {
        return poisson_gaussian_pld(noise_multiplier, rate, config);
    }

    let q_cond = conditional_sampling_probability(dataset_size, rate, batch_size_max, p_trunc);
    let log_mass = config.log_mass_truncation_bound;
    let tail_budget = config.tail_mass_truncation / 2.0;

    let bounds_remove = truncated_epsilon_bounds(
        sigma,
        sensitivity,
        rate,
        Adjacency::Remove,
        log_mass,
        q_cond,
    );
    let bounds_add = truncated_epsilon_bounds(
        sigma,
        sensitivity,
        rate,
        Adjacency::Add,
        log_mass,
        q_cond,
    );

    discretize_asymmetric_mechanism(config, bounds_remove, bounds_add, |epsilon, adj| {
        Ok(truncated_get_delta(
            epsilon,
            adj,
            sigma,
            sensitivity,
            rate,
            p_trunc,
            q_cond,
        ))
    })
    .map(|pld| pld.with_tail_budgets(tail_budget, tail_budget))
}

/// Compute the PLD for an accumulated Poisson-subsampled Gaussian mechanism.
///
/// Models gradient accumulation: `microbatches` independent Poisson-sampled
/// batches, clipped gradients summed, noise added **once**. A single example
/// appears K ~ Binomial(m, q) times, creating a Mixture of Gaussians.
///
/// This is mathematically different from composing `m` independent Poisson
/// steps: accumulation adds noise once (less total noise, worse privacy) but
/// produces larger effective batch sizes for better ML utility.
///
/// # Arguments
///
/// * `noise_multiplier` — σ/Δ, must be in \[0.1, 1.2\]
/// * `rate` — Poisson sampling probability q ∈ (0, 1\]
/// * `microbatches` — number of microbatches m > 0
/// * `config` — discretization configuration
///
/// # Errors
///
/// Returns `InvalidParameter` if parameters are out of range.
pub fn accumulated_poisson_gaussian_pld(
    noise_multiplier: f64,
    rate: f64,
    microbatches: usize,
    config: &DiscretizationConfig,
) -> Result<PrivacyLossDistribution> {
    validate_noise_multiplier(noise_multiplier)?;
    validate_rate(rate)?;
    if microbatches == 0 {
        return Err(PldError::InvalidParameter(
            "microbatches must be > 0".into(),
        ));
    }

    // m=1 fallback: exact match with standard Poisson
    if microbatches == 1 {
        return poisson_gaussian_pld(noise_multiplier, rate, config);
    }

    let sigma = noise_multiplier;
    let log_mass = config.log_mass_truncation_bound;
    let tail_budget = config.tail_mass_truncation / 2.0;
    let c = MixtureConstants::new(sigma, microbatches, rate);

    let bounds_remove = mixture_gaussian_epsilon_bounds(Adjacency::Remove, &c, log_mass)?;
    let bounds_add = mixture_gaussian_epsilon_bounds(Adjacency::Add, &c, log_mass)?;

    discretize_asymmetric_mechanism(config, bounds_remove, bounds_add, |epsilon, adj| {
        mixture_gaussian_get_delta(epsilon, adj, &c)
    })
    .map(|pld| pld.with_tail_budgets(tail_budget, tail_budget))
}

// ===========================================================================
// Validation helpers
// ===========================================================================

fn validate_noise_multiplier(nm: f64) -> Result<()> {
    if !(MIN_NOISE_MULTIPLIER..=MAX_NOISE_MULTIPLIER).contains(&nm) {
        return Err(PldError::InvalidParameter(format!(
            "noise_multiplier must be in [{}, {}], got {}",
            MIN_NOISE_MULTIPLIER, MAX_NOISE_MULTIPLIER, nm
        )));
    }
    Ok(())
}

fn validate_rate(rate: f64) -> Result<()> {
    if !(rate > 0.0 && rate <= 1.0) {
        return Err(PldError::InvalidParameter(format!(
            "sampling rate must be in (0, 1], got {}",
            rate
        )));
    }
    Ok(())
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
#[allow(dead_code)]
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
fn poisson_gaussian_epsilon_bounds(
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
fn inverse_privacy_loss_gaussian(privacy_loss: f64, sigma: f64, sensitivity: f64) -> f64 {
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
fn poisson_gaussian_get_delta(
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
// Truncated Poisson math
// ===========================================================================

/// Probability that truncation occurs: Pr[Binom(n−1, q) ≥ B_max].
fn truncation_probability(dataset_size: usize, rate: f64, batch_size_max: usize) -> f64 {
    if batch_size_max >= dataset_size {
        return 0.0;
    }

    let binom = Binomial::new(rate, (dataset_size - 1) as u64).unwrap();
    1.0 - binom.cdf((batch_size_max - 1) as u64)
}

/// Conditional sampling probability for the truncated component.
fn conditional_sampling_probability(
    dataset_size: usize,
    rate: f64,
    batch_size_max: usize,
    p_trunc: f64,
) -> f64 {
    if p_trunc == 0.0 {
        return 0.0;
    }

    let n = dataset_size;
    let binom = Binomial::new(rate, n as u64).unwrap();
    let pr_exceed = 1.0 - binom.cdf(batch_size_max as u64);

    pr_exceed * (batch_size_max as f64) / (n as f64) / p_trunc
}

/// Hockey-stick divergence for truncated Poisson (mixture formula from \[Gan25\]).
fn truncated_get_delta(
    epsilon: f64,
    adjacency: Adjacency,
    sigma: f64,
    sensitivity: f64,
    rate: f64,
    p_trunc: f64,
    q_cond: f64,
) -> f64 {
    if p_trunc == 0.0 {
        return poisson_gaussian_get_delta(epsilon, adjacency, sigma, sensitivity, rate);
    }

    // Component 1: standard Poisson, pessimistic bound for ADD/REMOVE
    let delta_comp1 = match adjacency {
        Adjacency::Add | Adjacency::Remove => {
            let d_add =
                poisson_gaussian_get_delta(epsilon, Adjacency::Add, sigma, sensitivity, rate);
            let d_rem =
                poisson_gaussian_get_delta(epsilon, Adjacency::Remove, sigma, sensitivity, rate);
            d_add.max(d_rem)
        }
        Adjacency::Replace => {
            poisson_gaussian_get_delta(epsilon, Adjacency::Replace, sigma, sensitivity, rate)
        }
    };

    // Component 2: doubled sensitivity (σ/2), REPLACE adjacency, conditional rate
    let delta_comp2 = poisson_gaussian_get_delta(
        epsilon,
        Adjacency::Replace,
        sigma / 2.0,
        sensitivity,
        q_cond,
    );

    (1.0 - p_trunc) * delta_comp1 + p_trunc * delta_comp2
}

/// Epsilon bounds for truncated Poisson (union of component bounds).
fn truncated_epsilon_bounds(
    sigma: f64,
    sensitivity: f64,
    rate: f64,
    adjacency: Adjacency,
    log_mass_truncation_bound: f64,
    q_cond: f64,
) -> EpsilonBounds {
    // Component 1: standard Poisson, pessimistic max(ADD, REMOVE) for ADD/REMOVE
    let bounds1 = match adjacency {
        Adjacency::Add | Adjacency::Remove => {
            let b_add = poisson_gaussian_epsilon_bounds(
                sigma,
                sensitivity,
                rate,
                Adjacency::Add,
                log_mass_truncation_bound,
            );
            let b_rem = poisson_gaussian_epsilon_bounds(
                sigma,
                sensitivity,
                rate,
                Adjacency::Remove,
                log_mass_truncation_bound,
            );
            EpsilonBounds {
                epsilon_lower: b_add.epsilon_lower.min(b_rem.epsilon_lower),
                epsilon_upper: b_add.epsilon_upper.max(b_rem.epsilon_upper),
            }
        }
        Adjacency::Replace => poisson_gaussian_epsilon_bounds(
            sigma,
            sensitivity,
            rate,
            Adjacency::Replace,
            log_mass_truncation_bound,
        ),
    };

    // Component 2: Poisson REPLACE, σ/2 (doubled sensitivity), rate q_cond
    let bounds2 = poisson_gaussian_epsilon_bounds(
        sigma / 2.0,
        sensitivity,
        q_cond,
        Adjacency::Replace,
        log_mass_truncation_bound,
    );

    EpsilonBounds {
        epsilon_lower: bounds1.epsilon_lower.min(bounds2.epsilon_lower),
        epsilon_upper: bounds1.epsilon_upper.max(bounds2.epsilon_upper),
    }
}

// ===========================================================================
// Accumulated (Mixture of Gaussians) math
// ===========================================================================

/// Precomputed constants for mixture Gaussian privacy loss computation.
struct MixtureConstants {
    log_probs: Vec<f64>,
    sensitivities: Vec<f64>,
    precomputed_remove: Vec<f64>,
    precomputed_add: Vec<f64>,
    sigma: f64,
    variance: f64,
    sampling_prob: f64,
}

impl MixtureConstants {
    fn new(sigma: f64, m: usize, q: f64) -> Self {
        let variance = sigma * sigma;
        let log_probs = binomial_log_probs(m, q);
        let sensitivities: Vec<f64> = (0..=m).map(|k| k as f64).collect();

        let precomputed_remove: Vec<f64> = (0..=m)
            .map(|k| {
                let s = k as f64;
                log_probs[k] + s * (-0.5 * s) / variance
            })
            .collect();

        let precomputed_add: Vec<f64> = (0..=m)
            .map(|k| {
                let s = k as f64;
                log_probs[k] - s * (0.5 * s) / variance
            })
            .collect();

        let sampling_prob = 1.0 - log_probs[0].exp();

        MixtureConstants {
            log_probs,
            sensitivities,
            precomputed_remove,
            precomputed_add,
            sigma,
            variance,
            sampling_prob,
        }
    }

    fn max_sensitivity(&self) -> f64 {
        *self.sensitivities.last().unwrap()
    }

    fn min_positive_sensitivity(&self) -> f64 {
        1.0
    }
}

/// Stable log Binom(k; m, q) via recurrence.
fn binomial_log_probs(m: usize, q: f64) -> Vec<f64> {
    let mut log_probs = Vec::with_capacity(m + 1);
    let log_1mq = (1.0 - q).ln();
    let log_q_ratio = (q / (1.0 - q)).ln();

    log_probs.push(m as f64 * log_1mq);
    for k in 1..=m {
        let prev = log_probs[k - 1];
        let log_binom_ratio = ((m - k + 1) as f64 / k as f64).ln();
        log_probs.push(prev + log_binom_ratio + log_q_ratio);
    }
    log_probs
}

/// Privacy loss at point x for mixture Gaussian.
fn mixture_privacy_loss(x: f64, adj: Adjacency, c: &MixtureConstants) -> Result<f64> {
    let summands: Vec<f64> = match adj {
        Adjacency::Remove => c
            .sensitivities
            .iter()
            .zip(c.precomputed_remove.iter())
            .map(|(&s, &pre)| pre - s * x / c.variance)
            .collect(),
        Adjacency::Add => c
            .sensitivities
            .iter()
            .zip(c.precomputed_add.iter())
            .map(|(&s, &pre)| pre + s * x / c.variance)
            .collect(),
        Adjacency::Replace => {
            return Err(PldError::InvalidParameter(
                "REPLACE adjacency not supported for mixture Gaussian".into(),
            ));
        }
    };

    let lse = log_sumexp(&summands);
    Ok(match adj {
        Adjacency::Remove => lse,
        Adjacency::Add => -lse,
        Adjacency::Replace => unreachable!(),
    })
}

/// Inverse privacy loss for single Gaussian (closed form).
fn inverse_privacy_loss_single_gaussian(epsilon: f64, sigma: f64, sensitivity: f64) -> f64 {
    let sigma_sq = sigma * sigma;
    0.5 * sensitivity - epsilon * sigma_sq / sensitivity
}

/// Inverse privacy loss for mixture Gaussian (bisection).
fn mixture_inverse_privacy_loss(epsilon: f64, adj: Adjacency, c: &MixtureConstants) -> Result<f64> {
    let sens_min = c.min_positive_sensitivity();
    let sens_max = c.max_sensitivity();

    let candidates = [
        inverse_privacy_loss_single_gaussian(epsilon, c.sigma, sens_min),
        inverse_privacy_loss_single_gaussian(epsilon, c.sigma, sens_max),
    ];

    let mut lo = candidates
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .fold(f64::INFINITY, f64::min);
    let mut hi = candidates
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);

    let margin = (hi - lo).abs().max(1.0) * 0.5;
    lo -= margin;
    hi += margin;

    for _ in 0..20 {
        let pl_lo = mixture_privacy_loss(lo, adj, c)?;
        let pl_hi = mixture_privacy_loss(hi, adj, c)?;
        if pl_lo >= epsilon && pl_hi <= epsilon {
            break;
        }
        if pl_lo < epsilon {
            lo -= (hi - lo).abs().max(1.0);
        }
        if pl_hi > epsilon {
            hi += (hi - lo).abs().max(1.0);
        }
    }

    for _ in 0..100 {
        let mid = (lo + hi) / 2.0;
        if (hi - lo).abs() < 1e-15 * hi.abs().max(1.0) {
            break;
        }
        let pl = mixture_privacy_loss(mid, adj, c)?;
        if pl > epsilon {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    Ok((lo + hi) / 2.0)
}

/// CDF of mu_upper at x.
fn mixture_mu_upper_cdf(x: f64, adj: Adjacency, c: &MixtureConstants) -> Result<f64> {
    let standard_normal = Normal::new(0.0, 1.0).unwrap();

    Ok(match adj {
        Adjacency::Remove => c
            .log_probs
            .iter()
            .zip(c.sensitivities.iter())
            .map(|(&log_p, &s)| log_p.exp() * standard_normal.cdf((x + s) / c.sigma))
            .sum(),
        Adjacency::Add => standard_normal.cdf(x / c.sigma),
        Adjacency::Replace => {
            return Err(PldError::InvalidParameter(
                "REPLACE adjacency not supported for mixture Gaussian".into(),
            ));
        }
    })
}

/// Log CDF of mu_lower at x.
fn mixture_mu_lower_log_cdf(x: f64, adj: Adjacency, c: &MixtureConstants) -> Result<f64> {
    Ok(match adj {
        Adjacency::Remove => gaussian_log_cdf(x / c.sigma),
        Adjacency::Add => {
            let summands: Vec<f64> = c
                .log_probs
                .iter()
                .zip(c.sensitivities.iter())
                .map(|(&log_p, &s)| log_p + gaussian_log_cdf((x - s) / c.sigma))
                .collect();
            log_sumexp(&summands)
        }
        Adjacency::Replace => {
            return Err(PldError::InvalidParameter(
                "REPLACE adjacency not supported for mixture Gaussian".into(),
            ));
        }
    })
}

/// Hockey-stick divergence for mixture Gaussian.
fn mixture_gaussian_get_delta(epsilon: f64, adj: Adjacency, c: &MixtureConstants) -> Result<f64> {
    if c.sampling_prob < 1.0 {
        match adj {
            Adjacency::Add => {
                let upper_bound = -(1.0 - c.sampling_prob).ln();
                if epsilon >= upper_bound - 1e-10 {
                    return Ok(0.0);
                }
            }
            Adjacency::Remove => {
                let lower_bound = (1.0 - c.sampling_prob).ln();
                if epsilon <= lower_bound {
                    return Ok((-epsilon.exp_m1()).max(0.0));
                }
            }
            Adjacency::Replace => {
                return Err(PldError::InvalidParameter(
                    "REPLACE adjacency not supported for mixture Gaussian".into(),
                ));
            }
        }
    }

    let x_cutoff = mixture_inverse_privacy_loss(epsilon, adj, c)?;
    let mu_upper = mixture_mu_upper_cdf(x_cutoff, adj, c)?;
    let log_mu_lower = mixture_mu_lower_log_cdf(x_cutoff, adj, c)?;
    let delta = mu_upper - (epsilon + log_mu_lower).exp();

    Ok(delta.clamp(0.0, 1.0))
}

/// Epsilon bounds for mixture Gaussian via x-space truncation.
fn mixture_gaussian_epsilon_bounds(
    adj: Adjacency,
    c: &MixtureConstants,
    log_mass_truncation_bound: f64,
) -> Result<EpsilonBounds> {
    let standard_normal = Normal::new(0.0, 1.0).unwrap();
    let half_mass = 0.5 * log_mass_truncation_bound.exp();
    let z = standard_normal.inverse_cdf(half_mass);
    let m = c.max_sensitivity();

    let (lower_x, upper_x) = match adj {
        Adjacency::Remove => (c.sigma * z - m, -c.sigma * z),
        Adjacency::Add => (c.sigma * z, -c.sigma * z + m),
        Adjacency::Replace => {
            return Err(PldError::InvalidParameter(
                "REPLACE adjacency not supported for mixture Gaussian".into(),
            ));
        }
    };

    let epsilon_upper = mixture_privacy_loss(lower_x, adj, c)?;
    let epsilon_lower = mixture_privacy_loss(upper_x, adj, c)?;

    Ok(EpsilonBounds {
        epsilon_lower,
        epsilon_upper,
    })
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn default_config() -> DiscretizationConfig {
        DiscretizationConfig::default()
    }

    // ---- poisson_gaussian_pld ----

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
            .map(|&q| {
                poisson_gaussian_pld(0.5, q, &cfg)
                    .unwrap()
                    .epsilon_at(1e-5)
            })
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

    // ---- truncated_poisson_gaussian_pld ----

    #[test]
    fn test_truncated_rejects_zero_batch_size() {
        assert!(truncated_poisson_gaussian_pld(0.5, 0.01, 0, 1000, &default_config()).is_err());
    }

    #[test]
    fn test_truncated_rejects_zero_dataset_size() {
        assert!(truncated_poisson_gaussian_pld(0.5, 0.01, 100, 0, &default_config()).is_err());
    }

    #[test]
    fn test_truncated_no_truncation_matches_poisson() {
        // batch_size_max ≥ dataset_size → no truncation
        let cfg = default_config();
        let pld_poisson = poisson_gaussian_pld(0.5, 0.01, &cfg).unwrap();
        let pld_trunc = truncated_poisson_gaussian_pld(0.5, 0.01, 200, 100, &cfg).unwrap();

        let eps_p = pld_poisson.epsilon_at(1e-5);
        let eps_t = pld_trunc.epsilon_at(1e-5);
        assert!(
            (eps_p - eps_t).abs() < 1e-6,
            "no-truncation should match Poisson: {} vs {}",
            eps_p,
            eps_t
        );
    }

    #[test]
    fn test_truncated_gives_higher_epsilon_than_poisson() {
        // Truncation is a pessimistic bound → more privacy loss
        let cfg = default_config();
        let pld_poisson = poisson_gaussian_pld(0.5, 0.01, &cfg).unwrap();
        let pld_trunc = truncated_poisson_gaussian_pld(0.5, 0.01, 50, 100000, &cfg).unwrap();

        let eps_p = pld_poisson.epsilon_at(1e-5);
        let eps_t = pld_trunc.epsilon_at(1e-5);
        assert!(
            eps_t >= eps_p - 1e-9,
            "truncated should have >= epsilon: {} vs {}",
            eps_t,
            eps_p
        );
    }

    // ---- accumulated_poisson_gaussian_pld ----

    #[test]
    fn test_accumulated_rejects_zero_microbatches() {
        assert!(accumulated_poisson_gaussian_pld(0.5, 0.01, 0, &default_config()).is_err());
    }

    #[test]
    fn test_accumulated_m1_matches_poisson() {
        let cfg = default_config();
        let pld_poisson = poisson_gaussian_pld(0.5, 0.01, &cfg).unwrap();
        let pld_acc = accumulated_poisson_gaussian_pld(0.5, 0.01, 1, &cfg).unwrap();

        let eps_p = pld_poisson.epsilon_at(1e-5);
        let eps_a = pld_acc.epsilon_at(1e-5);
        assert!(
            (eps_p - eps_a).abs() < 1e-6,
            "m=1 should match Poisson: {} vs {}",
            eps_p,
            eps_a
        );
    }

    #[test]
    fn test_accumulated_more_microbatches_higher_epsilon() {
        let cfg = default_config();
        let eps2 = accumulated_poisson_gaussian_pld(0.5, 0.01, 2, &cfg)
            .unwrap()
            .epsilon_at(1e-5);
        let eps4 = accumulated_poisson_gaussian_pld(0.5, 0.01, 4, &cfg)
            .unwrap()
            .epsilon_at(1e-5);
        let eps8 = accumulated_poisson_gaussian_pld(0.5, 0.01, 8, &cfg)
            .unwrap()
            .epsilon_at(1e-5);

        assert!(
            eps2 < eps4 && eps4 < eps8,
            "more microbatches → higher ε: {} < {} < {}",
            eps2,
            eps4,
            eps8
        );
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

    // ---- Truncated math ----

    #[test]
    fn test_truncation_probability_no_truncation() {
        assert_eq!(truncation_probability(100, 0.1, 100), 0.0);
        assert_eq!(truncation_probability(100, 0.1, 200), 0.0);
    }

    #[test]
    fn test_truncation_probability_realistic() {
        let p = truncation_probability(1_000_000, 0.001, 1024);
        assert!(p > 0.0 && p < 1.0);
    }

    #[test]
    fn test_conditional_sampling_probability_zero_truncation() {
        assert_eq!(conditional_sampling_probability(100, 0.1, 200, 0.0), 0.0);
    }

    // ---- Accumulated (Mixture) math ----

    #[test]
    fn test_binomial_log_probs_m1() {
        let log_probs = binomial_log_probs(1, 0.01);
        assert_eq!(log_probs.len(), 2);
        assert_relative_eq!(log_probs[0].exp(), 0.99, epsilon = 1e-12);
        assert_relative_eq!(log_probs[1].exp(), 0.01, epsilon = 1e-12);
    }

    #[test]
    fn test_binomial_log_probs_m4() {
        let log_probs = binomial_log_probs(4, 0.01);
        let total: f64 = log_probs.iter().map(|lp| lp.exp()).sum();
        assert_relative_eq!(total, 1.0, epsilon = 1e-12);
    }

    /// Validate against Google dp_accounting test vectors.
    #[test]
    fn test_mixture_privacy_loss_matches_google() {
        let c = make_test_constants(1.0, &[0.0, 1.0, 2.0], &[0.2, 0.6, 0.2]);

        let pl_add = mixture_privacy_loss(0.5, Adjacency::Add, &c).unwrap();
        assert_relative_eq!(pl_add, 0.1351602748368097, epsilon = 1e-10);

        let pl_rem = mixture_privacy_loss(0.5, Adjacency::Remove, &c).unwrap();
        assert_relative_eq!(pl_rem, -0.8423781325734492, epsilon = 1e-10);
    }

    #[test]
    fn test_mixture_get_delta_matches_google() {
        let c = make_test_constants(1.0, &[0.0, 1.0, 2.0], &[0.2, 0.6, 0.2]);

        let delta_add = mixture_gaussian_get_delta(1.0, Adjacency::Add, &c).unwrap();
        assert_relative_eq!(delta_add, 0.036691263832032806, epsilon = 1e-6);

        let delta_rem = mixture_gaussian_get_delta(1.0, Adjacency::Remove, &c).unwrap();
        assert_relative_eq!(delta_rem, 0.15768284088654105, epsilon = 1e-6);
    }

    /// Helper: construct MixtureConstants from arbitrary sensitivities and probs.
    fn make_test_constants(sigma: f64, sensitivities: &[f64], probs: &[f64]) -> MixtureConstants {
        let variance = sigma * sigma;
        let log_probs: Vec<f64> = probs.iter().map(|&p| p.ln()).collect();

        let precomputed_remove: Vec<f64> = sensitivities
            .iter()
            .zip(log_probs.iter())
            .map(|(&s, &lp)| lp + s * (-0.5 * s) / variance)
            .collect();

        let precomputed_add: Vec<f64> = sensitivities
            .iter()
            .zip(log_probs.iter())
            .map(|(&s, &lp)| lp - s * (0.5 * s) / variance)
            .collect();

        let sampling_prob: f64 = sensitivities
            .iter()
            .zip(probs.iter())
            .filter(|(&s, _)| s > 0.0)
            .map(|(_, &p)| p)
            .sum::<f64>()
            .min(1.0);

        MixtureConstants {
            log_probs,
            sensitivities: sensitivities.to_vec(),
            precomputed_remove,
            precomputed_add,
            sigma,
            variance,
            sampling_prob,
        }
    }
}
