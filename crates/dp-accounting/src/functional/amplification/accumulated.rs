//! Gradient accumulation amplification for privacy mechanisms
//!
//! Implements tight privacy accounting for gradient accumulation in DP-SGD, where
//! m microbatches are Poisson-sampled at rate q, clipped gradients are summed, and
//! noise is added **once**. A single example appears K ~ Binomial(m, q) times,
//! creating a Mixture of Gaussians privacy mechanism.
//!
//! This is mathematically different from `Repeated<Poisson<Gaussian>>` which models
//! m independent noise additions. Accumulation adds noise once (less total noise,
//! worse privacy) but produces larger effective batch sizes for better ML utility.
//!
//! # Algorithm
//!
//! For m microbatches with Poisson rate q:
//! 1. Each example independently included in each microbatch with probability q
//! 2. Example appears K ~ Binomial(m, q) times total
//! 3. Sensitivity is K (random), producing a mixture of Gaussians
//!
//! The privacy loss distribution is computed via the Mixture of Gaussians framework:
//! - For REMOVE: `mu_upper(x) = sum_k Binom(k;m,q) * N(x+k, sigma²)`, `mu_lower(x) = N(x, sigma²)`
//! - For ADD: `mu_upper(x) = N(x, sigma²)`, `mu_lower(x) = sum_k Binom(k;m,q) * N(x-k, sigma²)`
//!
//! # References
//!
//! - Choquette-Choo, Ganesh, Steinke, Thakurta (2023). "Privacy Amplification for
//!   Matrix Mechanisms." <https://arxiv.org/abs/2310.15526>
//! - Google dp_accounting: `MixtureGaussianPrivacyLoss` class

use crate::error::{PldError, Result};
use crate::functional::adjacency::Adjacency;
use crate::functional::discretization::{discretize_asymmetric_mechanism, EpsilonBounds};
use crate::functional::mechanisms::Gaussian;
use crate::functional::pld::PrivacyLossDistribution;
use crate::functional::process::Process;
use crate::math_helpers::logspace::log_sumexp;
use crate::math_helpers::special::gaussian_log_cdf;
use statrs::distribution::{ContinuousCDF, Normal};

use super::poisson::{Poisson, PoissonEvidence, TightGaussianPoissonEvidence};

// ---------------------------------------------------------------------------
// Trait + types
// ---------------------------------------------------------------------------

/// Evidence that gradient accumulation can be applied to a Poisson-subsampled process
///
/// Different evidence types provide different tightness guarantees.
/// `TightGaussianAccumulateEvidence` provides tight bounds for Gaussian mechanisms
/// using the Mixture of Gaussians framework.
pub trait AccumulateEvidence<P, E>: Clone {
    /// Compute the PLD for an accumulated Poisson-subsampled process
    ///
    /// # Arguments
    ///
    /// * `inner` - The Poisson-subsampled base process
    /// * `microbatches` - Number of microbatches m to accumulate
    ///
    /// # Errors
    ///
    /// * `PldError::InvalidParameter` - If parameters are out of range
    /// * `PldError::NumericalError` - If discretization fails
    fn compute_pld(
        &self,
        inner: &Poisson<P, E>,
        microbatches: usize,
    ) -> Result<PrivacyLossDistribution>;
}

/// Accumulated Poisson-subsampled process for gradient accumulation
///
/// Wraps a `Poisson<P, E>` process with gradient accumulation over m microbatches.
/// Each microbatch is independently Poisson-sampled at the inner process's rate.
/// Noise is added once after accumulation, not per microbatch.
///
/// # Type Parameters
///
/// * `P` - The base process type (e.g., `Gaussian`)
/// * `E` - The Poisson evidence type (e.g., `TightGaussianPoissonEvidence`)
/// * `A` - The accumulate evidence type (e.g., `TightGaussianAccumulateEvidence`)
///
/// # Examples
///
/// ```rust,ignore
/// use opaque_dp_accounting::functional::*;
///
/// // Fine-tuning: 50K dataset, physical batch 32, logical batch 256 (m=8)
/// let step = accumulate(poisson(gaussian(1.0), 0.00064), 8);
/// let training = repeat(step, 1000);
/// let epsilon = training.epsilon_at(1e-5)?;
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "P: serde::Serialize, E: serde::Serialize, A: serde::Serialize",
        deserialize = "P: serde::de::DeserializeOwned, E: serde::de::DeserializeOwned, A: serde::de::DeserializeOwned",
    ))
)]
pub struct Accumulated<P, E, A> {
    /// The inner Poisson-subsampled process
    pub inner: Poisson<P, E>,
    /// Number of microbatches to accumulate
    pub microbatches: usize,
    /// Evidence proving accumulation is valid for this process
    pub evidence: A,
}

impl<P: PartialEq, E: PartialEq, A: PartialEq> Eq for Accumulated<P, E, A> {}

impl<P, E, A: AccumulateEvidence<P, E>> Process for Accumulated<P, E, A> {
    fn pld(&self) -> Result<PrivacyLossDistribution> {
        self.evidence.compute_pld(&self.inner, self.microbatches)
    }
}

/// Tight accumulation evidence for Gaussian mechanisms
///
/// Uses the Mixture of Gaussians framework: K ~ Binomial(m, q) appearances
/// create a mixture with sensitivities {0, 1, ..., m} weighted by Binomial
/// probabilities. Computes exact hockey-stick divergence via bisection-based
/// inverse privacy loss.
///
/// # References
///
/// - Choquette-Choo et al. (2023). "Privacy Amplification for Matrix Mechanisms."
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TightGaussianAccumulateEvidence;

/// Trait for Poisson-subsampled processes that support gradient accumulation
///
/// Implementors specify how to convert themselves into a form suitable for
/// accumulation. The associated types determine the evidence used.
///
/// ```rust,ignore
/// accumulate(poisson(gaussian(1.0), 0.00064), 8)
/// accumulate(poisson(adaclip(gaussian(1.0), 50.0), 0.00064), 8)
/// ```
pub trait AccumulateAmplifiable: Sized {
    /// The base process type inside Poisson
    type P: Process;
    /// The Poisson evidence type
    type E;
    /// The accumulate evidence type
    type Evidence: AccumulateEvidence<Self::P, Self::E>;

    /// Convert this process into the Poisson wrapper and accumulate evidence
    fn into_accumulate_parts(self) -> (Poisson<Self::P, Self::E>, Self::Evidence);
}

impl AccumulateAmplifiable for Poisson<Gaussian, TightGaussianPoissonEvidence> {
    type P = Gaussian;
    type E = TightGaussianPoissonEvidence;
    type Evidence = TightGaussianAccumulateEvidence;

    fn into_accumulate_parts(
        self,
    ) -> (
        Poisson<Gaussian, TightGaussianPoissonEvidence>,
        TightGaussianAccumulateEvidence,
    ) {
        (self, TightGaussianAccumulateEvidence)
    }
}
// Note: poisson(adaclip(gaussian(...), sb), q) already returns Poisson<Gaussian, ...>,
// so AdaClip<Gaussian> is covered automatically — no extra impl needed.

// ---------------------------------------------------------------------------
// Math: Binomial log-probabilities
// ---------------------------------------------------------------------------

/// Compute log(Binom(k; m, q)) for k=0..=m using stable recurrence
///
/// Uses: `log_prob[k] = log_prob[k-1] + ln((m-k+1)/k) + ln(q/(1-q))`
/// Starting from: `log_prob[0] = m * ln(1-q)`
fn binomial_log_probs(m: usize, q: f64) -> Vec<f64> {
    let mut log_probs = Vec::with_capacity(m + 1);
    let log_1mq = (1.0 - q).ln();
    let log_q_ratio = (q / (1.0 - q)).ln(); // ln(q/(1-q))

    // k=0: (1-q)^m
    log_probs.push(m as f64 * log_1mq);

    // k=1..=m: recurrence
    for k in 1..=m {
        let prev = log_probs[k - 1];
        let log_binom_ratio = ((m - k + 1) as f64 / k as f64).ln();
        log_probs.push(prev + log_binom_ratio + log_q_ratio);
    }

    log_probs
}

// ---------------------------------------------------------------------------
// Math: precomputed mixture constants
// ---------------------------------------------------------------------------

/// Precomputed constants for mixture Gaussian privacy loss computation
///
/// Caches expensive per-sensitivity calculations to speed up repeated
/// evaluations of `mixture_privacy_loss(x)` during bisection and
/// discretization.
///
/// Mirrors Google's `_precompute_privacy_loss_constants` in
/// `MixtureGaussianPrivacyLoss`.
struct MixtureConstants {
    /// log(Binom(k; m, q)) for k=0..=m
    log_probs: Vec<f64>,
    /// Sensitivities: [0.0, 1.0, 2.0, ..., m as f64]
    sensitivities: Vec<f64>,
    /// Precomputed constants for REMOVE: log_probs[k] + k*(-0.5*k)/sigma²
    precomputed_remove: Vec<f64>,
    /// Precomputed constants for ADD: log_probs[k] - k*(0.5*k)/sigma²
    precomputed_add: Vec<f64>,
    /// Noise standard deviation
    sigma: f64,
    /// sigma²
    variance: f64,
    /// Total probability of positive sensitivity: sum of probs where k > 0
    sampling_prob: f64,
}

impl MixtureConstants {
    fn new(sigma: f64, m: usize, q: f64) -> Self {
        let variance = sigma * sigma;
        let log_probs = binomial_log_probs(m, q);
        let sensitivities: Vec<f64> = (0..=m).map(|k| k as f64).collect();

        // Precompute per-sensitivity privacy loss constants
        // For REMOVE: sens_loss = k * (-0.5 * k) / sigma² (privacy loss at x=0 for sensitivity k)
        // For ADD: sens_loss = k * (0.5 * k) / sigma²
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

        // Total probability of k > 0
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

    /// Maximum sensitivity (= m)
    fn max_sensitivity(&self) -> f64 {
        *self.sensitivities.last().unwrap()
    }

    /// Minimum positive sensitivity (= 1.0, always)
    fn min_positive_sensitivity(&self) -> f64 {
        1.0
    }
}

// ---------------------------------------------------------------------------
// Math: privacy loss at a point
// ---------------------------------------------------------------------------

/// Privacy loss at point x for the mixture Gaussian mechanism
///
/// For REMOVE adjacency:
///   `L(x) = logsumexp_k(precomputed_remove[k] - sensitivities[k] * x / variance)`
///
/// For ADD adjacency:
///   `L(x) = -logsumexp_k(precomputed_add[k] + sensitivities[k] * x / variance)`
///
/// Mirrors Google's `MixtureGaussianPrivacyLoss.privacy_loss()`.
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
                "REPLACE adjacency is not supported for mixture Gaussian".into(),
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

// ---------------------------------------------------------------------------
// Math: inverse privacy loss (bisection)
// ---------------------------------------------------------------------------

/// Inverse privacy loss for a single Gaussian (closed form)
///
/// For REMOVE: `x = 0.5 * sens - eps * sigma² / sens`
fn inverse_privacy_loss_single_gaussian(epsilon: f64, sigma: f64, sensitivity: f64) -> f64 {
    let sigma_sq = sigma * sigma;
    0.5 * sensitivity - epsilon * sigma_sq / sensitivity
}

/// Inverse privacy loss for the mixture Gaussian (bisection)
///
/// Finds x such that `mixture_privacy_loss(x, adj, c) == epsilon`.
/// Privacy loss is strictly monotone decreasing in x, so bisection converges.
///
/// Search bounds: use single-Gaussian inverses at min and max positive sensitivity
/// to bracket. The mixture's true inverse lies between these extremes because
/// the mixture privacy loss is bounded by the component with smallest and
/// largest sensitivity.
///
/// Following Google's `inverse_privacy_losses()` approach.
fn mixture_inverse_privacy_loss(epsilon: f64, adj: Adjacency, c: &MixtureConstants) -> Result<f64> {
    let sens_min = c.min_positive_sensitivity();
    let sens_max = c.max_sensitivity();

    // Single-Gaussian inverse at target epsilon for min/max sensitivity
    // These bracket the mixture inverse because the mixture privacy loss
    // is a logsumexp of per-component losses
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

    // Widen bounds by a safety margin to ensure bracketing
    let margin = (hi - lo).abs().max(1.0) * 0.5;
    lo -= margin;
    hi += margin;

    // Verify bracketing: privacy_loss(lo) should be > epsilon, privacy_loss(hi) < epsilon
    // If not, expand bounds further
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

    // Bisection: privacy loss is monotone decreasing in x
    // So if L(mid) > epsilon, we need larger x → lo = mid
    //    if L(mid) < epsilon, we need smaller x → hi = mid
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

// ---------------------------------------------------------------------------
// Math: mixture CDF functions
// ---------------------------------------------------------------------------

/// CDF of mu_upper at x
///
/// For REMOVE: `mu_upper(x) = sum_k probs[k] * Phi((x + sensitivities[k]) / sigma)`
/// For ADD: `mu_upper(x) = Phi(x / sigma)` (single Gaussian)
fn mixture_mu_upper_cdf(x: f64, adj: Adjacency, c: &MixtureConstants) -> Result<f64> {
    let standard_normal = Normal::new(0.0, 1.0).unwrap();

    Ok(match adj {
        Adjacency::Remove => {
            // Weighted sum of normal CDFs
            c.log_probs
                .iter()
                .zip(c.sensitivities.iter())
                .map(|(&log_p, &s)| log_p.exp() * standard_normal.cdf((x + s) / c.sigma))
                .sum()
        }
        Adjacency::Add => standard_normal.cdf(x / c.sigma),
        Adjacency::Replace => {
            return Err(PldError::InvalidParameter(
                "REPLACE adjacency is not supported for mixture Gaussian".into(),
            ));
        }
    })
}

/// Log CDF of mu_lower at x
///
/// For REMOVE: `mu_lower(x) = Phi(x / sigma)` → returns `log(Phi(x/sigma))`
/// For ADD: `mu_lower(x) = sum_k probs[k] * Phi((x - sensitivities[k]) / sigma)`
///   → returns `logsumexp(log_probs[k] + gaussian_log_cdf((x - sensitivities[k]) / sigma))`
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
                "REPLACE adjacency is not supported for mixture Gaussian".into(),
            ));
        }
    })
}

// ---------------------------------------------------------------------------
// Math: hockey-stick divergence
// ---------------------------------------------------------------------------

/// Hockey-stick divergence for the mixture Gaussian mechanism
///
/// Computes delta(epsilon) for the given adjacency type.
///
/// # Algorithm
///
/// 1. Find x_cutoff where privacy_loss(x_cutoff) = epsilon via bisection
/// 2. delta = mu_upper_cdf(x_cutoff) - exp(epsilon) * mu_lower_cdf(x_cutoff)
/// 3. Handle boundary cases for subsampled mechanisms
fn mixture_gaussian_get_delta(epsilon: f64, adj: Adjacency, c: &MixtureConstants) -> Result<f64> {
    // Boundary cases for subsampled mechanisms (sampling_prob < 1)
    if c.sampling_prob < 1.0 {
        match adj {
            Adjacency::Add => {
                // For ADD: eps >= -log(1-q) → delta = 0
                let upper_bound = -(1.0 - c.sampling_prob).ln();
                if epsilon >= upper_bound - 1e-10 {
                    return Ok(0.0);
                }
            }
            Adjacency::Remove => {
                // For REMOVE: eps <= log(1-q) → delta = -expm1(eps)
                let lower_bound = (1.0 - c.sampling_prob).ln();
                if epsilon <= lower_bound {
                    return Ok((-epsilon.exp_m1()).max(0.0));
                }
            }
            Adjacency::Replace => {
                return Err(PldError::InvalidParameter(
                    "REPLACE adjacency is not supported for mixture Gaussian".into(),
                ));
            }
        }
    }

    // Find x_cutoff via bisection
    let x_cutoff = mixture_inverse_privacy_loss(epsilon, adj, c)?;

    // delta = mu_upper_cdf(x_cutoff) - exp(epsilon + log_mu_lower_cdf(x_cutoff))
    let mu_upper = mixture_mu_upper_cdf(x_cutoff, adj, c)?;
    let log_mu_lower = mixture_mu_lower_log_cdf(x_cutoff, adj, c)?;
    let delta = mu_upper - (epsilon + log_mu_lower).exp();

    Ok(delta.clamp(0.0, 1.0))
}

// ---------------------------------------------------------------------------
// Math: epsilon bounds
// ---------------------------------------------------------------------------

/// Find epsilon_upper: smallest ε where `mixture_delta(ε, adj) ≤ target`.
///
/// Bisects on the mixture Gaussian delta function directly.
/// Initial upper bound from the base Gaussian analytic formula with
/// max_sensitivity, capped at the theoretical Poisson limit for ADD.
fn mixture_epsilon_for_delta(adj: Adjacency, c: &MixtureConstants, target: f64) -> Result<f64> {
    use crate::math_helpers::gaussian::gaussian_epsilon_for_delta;

    // Use base Gaussian bound at max sensitivity as initial overshoot
    let delta_tilde_max = c.max_sensitivity() / c.sigma;
    let mut hi = gaussian_epsilon_for_delta(delta_tilde_max, target);

    // For ADD with subsampling, epsilon is bounded by -log(1-q)
    if adj == Adjacency::Add && c.sampling_prob < 1.0 {
        hi = hi.min(-(1.0 - c.sampling_prob).ln());
    }

    let mut lo = 0.0_f64;
    for _ in 0..100 {
        let mid = (lo + hi) / 2.0;
        if mixture_gaussian_get_delta(mid, adj, c)? > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(hi)
}

/// Find epsilon_lower: most negative ε where `mixture_beta(|ε|, adj) ≤ target`.
///
/// Beta for adjacency A = delta for opposite adjacency at −ε.
fn mixture_epsilon_for_beta(adj: Adjacency, c: &MixtureConstants, target: f64) -> Result<f64> {
    use crate::math_helpers::gaussian::gaussian_epsilon_for_delta;

    let opposite_adj = match adj {
        Adjacency::Add => Adjacency::Remove,
        Adjacency::Remove => Adjacency::Add,
        Adjacency::Replace => {
            return Err(PldError::InvalidParameter(
                "REPLACE adjacency is not supported for mixture Gaussian".into(),
            ));
        }
    };

    let delta_tilde_max = c.max_sensitivity() / c.sigma;
    let hi_init = gaussian_epsilon_for_delta(delta_tilde_max, target);

    let mut lo = 0.0_f64;
    let mut hi = hi_init;
    for _ in 0..100 {
        let mid = (lo + hi) / 2.0;
        if mixture_gaussian_get_delta(-mid, opposite_adj, c)? > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(-hi)
}

/// Compute epsilon bounds for the mixture Gaussian mechanism
///
/// Uses bisection on the mixture delta and beta functions directly,
/// giving tight bounds that match the `min_delta`/`min_beta` semantics.
fn mixture_gaussian_epsilon_bounds(
    adj: Adjacency,
    c: &MixtureConstants,
    min_delta: f64,
    min_beta: f64,
) -> Result<EpsilonBounds> {
    let epsilon_upper = mixture_epsilon_for_delta(adj, c, min_delta)?;
    let epsilon_lower = mixture_epsilon_for_beta(adj, c, min_beta)?;

    Ok(EpsilonBounds {
        epsilon_lower,
        epsilon_upper,
    })
}

// ---------------------------------------------------------------------------
// Evidence implementation
// ---------------------------------------------------------------------------

impl AccumulateEvidence<Gaussian, TightGaussianPoissonEvidence>
    for TightGaussianAccumulateEvidence
{
    fn compute_pld(
        &self,
        inner: &Poisson<Gaussian, TightGaussianPoissonEvidence>,
        microbatches: usize,
    ) -> Result<PrivacyLossDistribution> {
        let sigma = inner.inner.noise_multiplier;
        let rate = inner.rate;
        let config = &inner.inner.config;

        // m=1 fallback: delegate to standard Poisson (exact match)
        if microbatches == 1 {
            return inner.evidence.compute_pld(&inner.inner, rate);
        }

        let min_delta = inner.inner.min_delta;
        let min_beta = inner.inner.min_beta;
        let c = MixtureConstants::new(sigma, microbatches, rate);

        let bounds_remove =
            mixture_gaussian_epsilon_bounds(Adjacency::Remove, &c, min_delta, min_beta)?;
        let bounds_add = mixture_gaussian_epsilon_bounds(Adjacency::Add, &c, min_delta, min_beta)?;

        discretize_asymmetric_mechanism(config, bounds_remove, bounds_add, |epsilon, adj| {
            mixture_gaussian_get_delta(epsilon, adj, &c)
        })
        .map(|pld| pld.with_tail_budgets(min_delta, min_beta))
    }
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

/// Create an accumulated Poisson-subsampled process for gradient accumulation
///
/// Models m microbatches, each Poisson-sampled at the inner process's rate,
/// with noise added once after accumulation.
///
/// Works for any process that implements `AccumulateAmplifiable`:
///
/// ```rust,ignore
/// // Fine-tuning: 50K dataset, physical batch 32, logical batch 256
/// let step = accumulate(poisson(gaussian(1.0)?, 0.00064), 8)?;
///
/// // Also works with AdaClip (auto-converts to equivalent Gaussian)
/// let step = accumulate(poisson(adaclip(gaussian(1.0)?, 50.0), 0.00064), 8)?;
/// ```
///
/// # Errors
///
/// Returns `PldError::InvalidParameter` if `microbatches == 0`.
pub fn accumulate<T: AccumulateAmplifiable>(
    inner: T,
    microbatches: usize,
) -> Result<Accumulated<T::P, T::E, T::Evidence>> {
    if microbatches == 0 {
        return Err(PldError::InvalidParameter(
            "Number of microbatches must be > 0".into(),
        ));
    }
    let (poisson_inner, evidence) = inner.into_accumulate_parts();
    Ok(Accumulated {
        inner: poisson_inner,
        microbatches,
        evidence,
    })
}

/// Create an accumulated process with custom evidence
///
/// For experimental or third-party evidence implementations.
///
/// # Errors
///
/// Returns `PldError::InvalidParameter` if `microbatches == 0`.
pub fn accumulate_with<P, E, A>(
    inner: Poisson<P, E>,
    microbatches: usize,
    evidence: A,
) -> Result<Accumulated<P, E, A>> {
    if microbatches == 0 {
        return Err(PldError::InvalidParameter(
            "Number of microbatches must be > 0".into(),
        ));
    }
    Ok(Accumulated {
        inner,
        microbatches,
        evidence,
    })
}

// ---------------------------------------------------------------------------
// Unit tests for math functions
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_binomial_log_probs_m1() {
        let q = 0.01;
        let log_probs = binomial_log_probs(1, q);
        assert_eq!(log_probs.len(), 2);
        assert_relative_eq!(log_probs[0].exp(), 0.99, epsilon = 1e-12);
        assert_relative_eq!(log_probs[1].exp(), 0.01, epsilon = 1e-12);
    }

    #[test]
    fn test_binomial_log_probs_m4() {
        let q = 0.01;
        let log_probs = binomial_log_probs(4, q);
        assert_eq!(log_probs.len(), 5);
        // Verify probabilities sum to 1
        let total: f64 = log_probs.iter().map(|lp| lp.exp()).sum();
        assert_relative_eq!(total, 1.0, epsilon = 1e-12);
        // P(K=0) ≈ 0.9606
        assert_relative_eq!(log_probs[0].exp(), 0.96059601, epsilon = 1e-6);
        // P(K=1) ≈ 0.0388
        assert_relative_eq!(log_probs[1].exp(), 0.03881196, epsilon = 1e-6);
    }

    #[test]
    fn test_binomial_log_probs_m8_small_q() {
        let q = 0.00064;
        let log_probs = binomial_log_probs(8, q);
        // P(K>=2) should be tiny
        let p_ge2: f64 = log_probs[2..].iter().map(|lp| lp.exp()).sum();
        assert!(p_ge2 < 0.002, "P(K>=2)={} should be < 0.002", p_ge2);
    }

    /// Validate privacy_loss against Google's test cases
    #[test]
    fn test_privacy_loss_matches_google() {
        // sigma=1.0, sensitivities=[0,1,2], probs=[0.2,0.6,0.2]
        // We construct MixtureConstants manually for these arbitrary params
        let c = make_test_constants(1.0, &[0.0, 1.0, 2.0], &[0.2, 0.6, 0.2]);

        // ADD, x=0.5
        let pl = mixture_privacy_loss(0.5, Adjacency::Add, &c).unwrap();
        assert_relative_eq!(pl, 0.1351602748368097, epsilon = 1e-10);

        // REMOVE, x=0.5
        let pl = mixture_privacy_loss(0.5, Adjacency::Remove, &c).unwrap();
        assert_relative_eq!(pl, -0.8423781325734492, epsilon = 1e-10);
    }

    #[test]
    fn test_privacy_loss_matches_google_large_sigma() {
        // sigma=7.0, sensitivities=[0,7,14], probs=[0.2,0.6,0.2]
        let c = make_test_constants(7.0, &[0.0, 7.0, 14.0], &[0.2, 0.6, 0.2]);

        assert_relative_eq!(
            mixture_privacy_loss(0.5, Adjacency::Add, &c).unwrap(),
            0.4746752545839654,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            mixture_privacy_loss(-0.5, Adjacency::Add, &c).unwrap(),
            0.5757291778782041,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            mixture_privacy_loss(0.5, Adjacency::Remove, &c).unwrap(),
            -0.5757291778782041,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            mixture_privacy_loss(-0.5, Adjacency::Remove, &c).unwrap(),
            -0.4746752545839654,
            epsilon = 1e-10
        );
    }

    /// Validate get_delta against Google's test cases
    #[test]
    fn test_get_delta_matches_google() {
        // mixture_gaussian with sensitivities=[0,1,2], probs=[0.2,0.6,0.2], sigma=1.0
        let c = make_test_constants(1.0, &[0.0, 1.0, 2.0], &[0.2, 0.6, 0.2]);

        let delta_add = mixture_gaussian_get_delta(1.0, Adjacency::Add, &c).unwrap();
        assert_relative_eq!(delta_add, 0.036691263832032806, epsilon = 1e-6);

        let delta_rem = mixture_gaussian_get_delta(1.0, Adjacency::Remove, &c).unwrap();
        assert_relative_eq!(delta_rem, 0.15768284088654105, epsilon = 1e-6);
    }

    #[test]
    fn test_get_delta_matches_google_no_zero_sens() {
        // sensitivities=[1,2], probs=[0.2,0.8], sigma=1.0
        let c = make_test_constants(1.0, &[1.0, 2.0], &[0.2, 0.8]);

        let delta_add = mixture_gaussian_get_delta(1.0, Adjacency::Add, &c).unwrap();
        assert_relative_eq!(delta_add, 0.3894964356580768, epsilon = 1e-6);

        let delta_rem = mixture_gaussian_get_delta(1.0, Adjacency::Remove, &c).unwrap();
        assert_relative_eq!(delta_rem, 0.433276675545065, epsilon = 1e-6);
    }

    /// Helper: construct MixtureConstants from arbitrary sensitivities and probs
    /// (not necessarily Binomial — for testing against Google's test vectors)
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
            .sum();

        MixtureConstants {
            log_probs,
            sensitivities: sensitivities.to_vec(),
            precomputed_remove,
            precomputed_add,
            sigma,
            variance,
            sampling_prob: sampling_prob.min(1.0),
        }
    }
}
