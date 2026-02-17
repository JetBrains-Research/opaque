//! Poisson subsampling amplification for privacy mechanisms
//!
//! Implements tight privacy amplification bounds for Poisson-subsampled Gaussian
//! mechanisms using evidence-based extensibility. Each record is sampled independently
//! with probability q, providing privacy amplification.
//!
//! # Privacy Loss Transformation
//!
//! For a base mechanism with privacy loss L(x), Poisson subsampling with rate q gives:
//! - **REMOVE**: `L_rem(x) = log(1-q + q*exp(L_raw(x)))`
//! - **ADD**: `L_add(x) = -L_rem(-x)` (symmetry)
//! - **REPLACE**: `L_rep(x) = log[(q*mu(x+D) + (1-q)*mu(x)) / (q*mu(x-D) + (1-q)*mu(x))]`
//!
//! # References
//!
//! - \[BBG18\]: Balle, Barthe, Gavin. "Privacy amplification by subsampling."
//! - \[LRKS25\]: "Avoiding pitfalls for privacy accounting of subsampled mechanisms."
//! - Doroshenko et al. (2022). "Connect the Dots." PETS 2022.

use crate::error::Result;
use crate::adjacency::Adjacency;
use crate::discretization::{discretize_asymmetric_mechanism, EpsilonBounds};
use crate::mechanisms::Gaussian;
use crate::pld::PrivacyLossDistribution;
use crate::process::Process;
use crate::math_helpers::logspace::{log_a_times_exp_b_plus_c, log_add};
use crate::math_helpers::special::{arcsinh_exp, gaussian_log_cdf, log_sinh};
use statrs::distribution::{ContinuousCDF, Normal};

// ---------------------------------------------------------------------------
// Trait + types
// ---------------------------------------------------------------------------

/// Evidence that Poisson subsampling can be applied to a process of type P
///
/// Different evidence types provide different tightness guarantees.
/// For example, `TightGaussianPoissonEvidence` provides tight bounds for
/// Gaussian mechanisms, while a generic evidence would provide looser but
/// universally applicable bounds.
pub trait PoissonEvidence<P>: Clone {
    /// Compute the PLD for a Poisson-subsampled process
    ///
    /// # Arguments
    ///
    /// * `inner` - The base process
    /// * `rate` - Poisson sampling rate q in (0, 1]
    ///
    /// # Errors
    ///
    /// * `PldError::InvalidParameter` - If parameters are out of range
    /// * `PldError::NumericalError` - If discretization fails
    fn compute_pld(&self, inner: &P, rate: f64) -> Result<PrivacyLossDistribution>;
}

/// Poisson-subsampled process with evidence-based amplification
///
/// Wraps an inner process P with Poisson subsampling at rate q.
/// The evidence type E determines how the amplified PLD is computed.
///
/// # Type Parameters
///
/// * `P` - The inner process type (e.g., `Gaussian`)
/// * `E` - The evidence type (e.g., `TightGaussianPoissonEvidence`)
///
/// # Examples
///
/// ```rust,ignore
/// use opaque_dp_accounting::functional::*;
///
/// let step = poisson(gaussian(1.1), 0.01);
/// let epsilon = step.epsilon_at(1e-5)?;
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "P: serde::Serialize, E: serde::Serialize",
        deserialize = "P: serde::de::DeserializeOwned, E: serde::de::DeserializeOwned",
    ))
)]
pub struct Poisson<P, E> {
    /// The inner (base) process
    pub inner: P,
    /// Poisson sampling rate q in (0, 1]
    pub rate: f64,
    /// Evidence proving amplification is valid for P
    pub evidence: E,
}

impl<P: PartialEq, E: PartialEq> Eq for Poisson<P, E> {}

impl<P, E: PoissonEvidence<P>> Process for Poisson<P, E> {
    fn pld(&self) -> Result<PrivacyLossDistribution> {
        self.evidence.compute_pld(&self.inner, self.rate)
    }
}

/// Tight Poisson amplification evidence for Gaussian mechanisms
///
/// Uses the exact Poisson-subsampled Gaussian privacy loss formulas for all
/// three adjacency types (ADD, REMOVE, REPLACE), providing the tightest
/// possible privacy bounds.
///
/// # References
///
/// - \[LRKS25\]: "Avoiding pitfalls for privacy accounting of subsampled mechanisms."
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TightGaussianPoissonEvidence;

/// Trait for mechanisms that support Poisson subsampling amplification
///
/// Implementors specify how to convert themselves into a form suitable for
/// Poisson amplification. The associated types determine the evidence used
/// and the inner process type stored in the resulting `Poisson` wrapper.
///
/// This enables a single `poisson()` function that works for any mechanism:
///
/// ```rust,ignore
/// poisson(gaussian(1.1), 0.01)
/// poisson(adaclip(gaussian(1.1), 50.0), 0.01)
/// ```
///
/// Future mechanisms (e.g., Laplace) would implement this trait with their
/// own evidence type.
pub trait PoissonAmplifiable: Process + Sized {
    /// The process type stored inside `Poisson<Inner, Evidence>`
    type Inner: Process;
    /// The evidence type proving amplification is valid
    type Evidence: PoissonEvidence<Self::Inner>;

    /// Convert this process into the inner form and its evidence
    fn into_poisson_parts(self) -> (Self::Inner, Self::Evidence);
}

// ---------------------------------------------------------------------------
// Math: privacy loss at a point
// ---------------------------------------------------------------------------

/// Poisson-transformed privacy loss for REMOVE adjacency
///
/// `L_rem(x) = log(1-q + q*exp(L_raw(x)))` where `L_raw(x) = D*(-0.5*D - x) / sigma^2`
fn privacy_loss_remove(x: f64, sigma: f64, sensitivity: f64, rate: f64) -> f64 {
    let sigma_sq = sigma * sigma;
    let l_raw = sensitivity * (-0.5 * sensitivity - x) / sigma_sq;

    if (rate - 1.0).abs() < 1e-15 {
        return l_raw;
    }

    // log(q * exp(l_raw) + (1-q))
    log_a_times_exp_b_plus_c(rate, l_raw, 1.0 - rate)
}

/// Poisson-transformed privacy loss for ADD adjacency
///
/// By symmetry: `L_add(x) = -L_rem(-x)`
fn privacy_loss_add(x: f64, sigma: f64, sensitivity: f64, rate: f64) -> f64 {
    -privacy_loss_remove(-x, sigma, sensitivity, rate)
}

/// Poisson-transformed privacy loss for REPLACE adjacency
///
/// ```text
/// L_replace(x) = log[(q*N(x+D; 0,s^2) + (1-q)*N(x; 0,s^2)) /
///                     (q*N(x-D; 0,s^2) + (1-q)*N(x; 0,s^2))]
/// ```
///
/// Computed in log-space using Gaussian PDF ratios for numerical stability.
#[allow(dead_code)]
pub(crate) fn privacy_loss_replace(x: f64, sigma: f64, sensitivity: f64, rate: f64) -> f64 {
    // N(x; 0, sigma^2) = exp(-x^2 / (2*sigma^2)) / (sigma * sqrt(2*pi))
    // Ratios of PDFs cancel the normalizing constant:
    //   N(x+D)/N(x) = exp(-(2*x*D + D^2) / (2*sigma^2))
    //   N(x-D)/N(x) = exp((2*x*D - D^2) / (2*sigma^2))

    let sigma_sq = sigma * sigma;
    let q = rate;

    // Log-ratios relative to N(x; 0, sigma^2):
    let log_ratio_plus = -(2.0 * x * sensitivity + sensitivity * sensitivity) / (2.0 * sigma_sq);
    let log_ratio_minus = (2.0 * x * sensitivity - sensitivity * sensitivity) / (2.0 * sigma_sq);

    // numerator = q * exp(log_ratio_plus) + (1-q)
    // denominator = q * exp(log_ratio_minus) + (1-q)
    let log_num = log_a_times_exp_b_plus_c(q, log_ratio_plus, 1.0 - q);
    let log_den = log_a_times_exp_b_plus_c(q, log_ratio_minus, 1.0 - q);

    log_num - log_den
}

/// Privacy loss at a point for any adjacency type
#[allow(dead_code)]
fn privacy_loss_at_point(
    x: f64,
    adjacency: Adjacency,
    sigma: f64,
    sensitivity: f64,
    rate: f64,
) -> f64 {
    match adjacency {
        Adjacency::Remove => privacy_loss_remove(x, sigma, sensitivity, rate),
        Adjacency::Add => privacy_loss_add(x, sigma, sensitivity, rate),
        Adjacency::Replace => privacy_loss_replace(x, sigma, sensitivity, rate),
    }
}

// ---------------------------------------------------------------------------
// Math: epsilon bounds (x-space truncation -> epsilon-space)
// ---------------------------------------------------------------------------

/// Compute epsilon bounds for Poisson-subsampled Gaussian using x-space truncation.
///
/// Matches Google dp_accounting's approach: compute x-space truncation points
/// from the Gaussian tail (via `ppf(0.5 * exp(log_mass_truncation_bound))`),
/// then evaluate the Poisson-subsampled privacy loss at those points to get
/// epsilon bounds.
///
/// This ensures our grid size matches Google's exactly, which is critical for
/// accurate beta computation after FFT-based composition.
///
/// For REMOVE adjacency:
/// - `lower_x = sigma * ppf(half_mass) - sensitivity` (shift for mu_upper tail)
/// - `upper_x = -sigma * ppf(half_mass)`
/// - `epsilon_upper = L_remove(lower_x)` (highest privacy loss)
/// - `epsilon_lower = L_remove(upper_x)` ≈ `log(1-q)` (lowest privacy loss)
///
/// For ADD adjacency:
/// - `lower_x = sigma * ppf(half_mass)`
/// - `upper_x = -sigma * ppf(half_mass) + sensitivity` (shift for mu_upper tail)
/// - `epsilon_upper = L_add(lower_x)` ≈ `-log(1-q)` (highest privacy loss)
/// - `epsilon_lower = L_add(upper_x)` = `-epsilon_upper_remove` (lowest privacy loss)
///
/// # References
///
/// Google dp_accounting `GaussianPrivacyLoss.privacy_loss_tail()` and
/// `GaussianPrivacyLoss.connect_dots_bounds()`.
pub(crate) fn poisson_gaussian_epsilon_bounds(
    sigma: f64,
    sensitivity: f64,
    rate: f64,
    adjacency: Adjacency,
    log_mass_truncation_bound: f64,
) -> EpsilonBounds {
    use statrs::distribution::{ContinuousCDF, Normal};

    let standard_normal = Normal::new(0.0, 1.0).unwrap();

    // x-space truncation: find x where Gaussian tail mass = 0.5 * exp(log_mass)
    // ppf returns a very negative z-score for tiny probabilities
    let half_mass = 0.5 * log_mass_truncation_bound.exp();
    let lower_x_base = sigma * standard_normal.inverse_cdf(half_mass);
    let upper_x_base = -lower_x_base; // symmetric

    match adjacency {
        Adjacency::Remove => {
            // Shift lower_x by -sensitivity to cover mu_upper = (1-q)*mu(x) + q*mu(x+D)
            let lower_x = lower_x_base - sensitivity;
            let upper_x = upper_x_base;
            let epsilon_upper = privacy_loss_remove(lower_x, sigma, sensitivity, rate);
            let epsilon_lower = privacy_loss_remove(upper_x, sigma, sensitivity, rate);
            EpsilonBounds {
                epsilon_lower,
                epsilon_upper,
            }
        }
        Adjacency::Add => {
            // Shift upper_x by +sensitivity to cover mu_upper = mu(x)
            let lower_x = lower_x_base;
            let upper_x = upper_x_base + sensitivity;
            let epsilon_upper = privacy_loss_add(lower_x, sigma, sensitivity, rate);
            let epsilon_lower = privacy_loss_add(upper_x, sigma, sensitivity, rate);
            EpsilonBounds {
                epsilon_lower,
                epsilon_upper,
            }
        }
        Adjacency::Replace => {
            // For REPLACE, use both shifts
            let lower_x = lower_x_base - sensitivity;
            let upper_x = upper_x_base + sensitivity;
            let epsilon_upper = privacy_loss_replace(lower_x, sigma, sensitivity, rate);
            let epsilon_lower = privacy_loss_replace(upper_x, sigma, sensitivity, rate);
            EpsilonBounds {
                epsilon_lower,
                epsilon_upper,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Math: inverse privacy loss (for hockey-stick divergence)
// ---------------------------------------------------------------------------

/// Inverse privacy loss for ADD/REMOVE adjacency (Gaussian base)
///
/// For Gaussian: `x = 0.5*D - L * (sigma^2 / D)`
fn inverse_privacy_loss_gaussian(privacy_loss: f64, sigma: f64, sensitivity: f64) -> f64 {
    let sigma_sq = sigma * sigma;
    0.5 * sensitivity - privacy_loss * (sigma_sq / sensitivity)
}

/// Inverse privacy loss for REPLACE adjacency (arcsinh formula)
///
/// ```text
/// x = (sigma^2 / D) * (arcsinh(-sinh(eps/2) * alpha) - eps/2)
/// where alpha = exp(0.5*(D/sigma)^2) * (1-q)/q
/// ```
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
        // q=1: REPLACE without subsampling, sensitivity is effectively 2*D
        // inverse: x = -eps * sigma^2 / (2*D)
        return Ok(-epsilon * sigma_sq / (2.0 * sensitivity));
    }

    let abs_eps = epsilon.abs();
    let sign_eps = epsilon.signum();

    // log(alpha) = 0.5*(D/sigma)^2 + log(1-q) - log(q)
    let ds = sensitivity / sigma;
    let log_alpha = 0.5 * ds * ds + (1.0 - rate).ln() - rate.ln();

    // log(sinh(|eps|/2) * alpha) = log_alpha + log_sinh(|eps|/2)
    let log_sinh_term = log_alpha + log_sinh(abs_eps / 2.0);

    // arcsinh(-sign(eps) * exp(log_sinh_term))
    let asinh_term = arcsinh_exp(log_sinh_term, -sign_eps);

    Ok((sigma_sq / sensitivity) * (asinh_term - epsilon / 2.0))
}

// ---------------------------------------------------------------------------
// Math: hockey-stick divergence (get_delta)
// ---------------------------------------------------------------------------

/// Hockey-stick divergence for Poisson-subsampled Gaussian
///
/// Computes delta(epsilon) for the given adjacency type.
pub(crate) fn poisson_gaussian_get_delta(
    epsilon: f64,
    adjacency: Adjacency,
    sigma: f64,
    sensitivity: f64,
    rate: f64,
) -> f64 {
    let q = rate;

    // For q=1, delegate to base Gaussian delta
    if (q - 1.0).abs() < 1e-15 {
        return base_gaussian_get_delta(epsilon, sigma, sensitivity, adjacency);
    }

    match adjacency {
        Adjacency::Add => get_delta_add(epsilon, sigma, sensitivity, q),
        Adjacency::Remove => get_delta_remove(epsilon, sigma, sensitivity, q),
        Adjacency::Replace => get_delta_replace(epsilon, sigma, sensitivity, q).unwrap_or(0.0),
    }
}

/// Base (unsubsampled) Gaussian hockey-stick divergence
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
            // REPLACE without subsampling: equivalent to sensitivity 2*D
            let dt2 = 2.0 * delta_tilde;
            let x_upper = 0.5 * dt2 - epsilon / dt2;
            let cdf_x = standard_normal.cdf(x_upper);
            let cdf_shifted = standard_normal.cdf(x_upper - dt2);
            (cdf_x - epsilon.exp() * cdf_shifted).max(0.0)
        }
    }
}

/// Hockey-stick divergence for ADD adjacency with Poisson subsampling
fn get_delta_add(epsilon: f64, sigma: f64, sensitivity: f64, q: f64) -> f64 {
    // Theoretical upper bound: eps >= -log(1-q) => delta = 0
    let theoretical_upper = -(1.0 - q).ln();
    if epsilon >= theoretical_upper - 1e-10 {
        return 0.0;
    }

    // Invert Poisson transform to get base privacy loss
    let exp_neg_eps = (-epsilon).exp();
    let ratio = (exp_neg_eps - (1.0 - q)) / q;
    if ratio <= 0.0 {
        return 0.0;
    }
    let l_base = -ratio.ln();

    // x_cutoff from base Gaussian inverse privacy loss (ADD direction)
    let x_cutoff = inverse_privacy_loss_gaussian(l_base, sigma, sensitivity);

    // mu_upper = Phi(x/sigma)
    let mu_upper = gaussian_cdf(x_cutoff / sigma);

    // log_mu_lower = log_add(log(1-q) + logPhi(x/sigma), log(q) + logPhi((x-D)/sigma))
    let log_mu_upper = gaussian_log_cdf(x_cutoff / sigma);
    let log_cdf_lower = gaussian_log_cdf((x_cutoff - sensitivity) / sigma);
    let log_1_minus_q = (1.0 - q).ln();
    let log_q = q.ln();
    let log_mu_lower = log_add(log_1_minus_q + log_mu_upper, log_q + log_cdf_lower);

    (mu_upper - (epsilon + log_mu_lower).exp()).max(0.0)
}

/// Hockey-stick divergence for REMOVE adjacency with Poisson subsampling
fn get_delta_remove(epsilon: f64, sigma: f64, sensitivity: f64, q: f64) -> f64 {
    // Theoretical lower bound: eps <= log(1-q) => delta = -expm1(eps)
    let theoretical_lower = (1.0 - q).ln();
    if epsilon <= theoretical_lower {
        return (-epsilon.exp_m1()).max(0.0);
    }

    // Invert: L_base = -log((exp(eps) - (1-q)) / q)
    let exp_eps = epsilon.exp();
    let ratio = (exp_eps - (1.0 - q)) / q;
    if ratio <= 0.0 {
        return (-epsilon.exp_m1()).max(0.0);
    }
    let l_base = -ratio.ln();

    // Use ADD inverse for x_cutoff
    let x_cutoff = inverse_privacy_loss_gaussian(l_base, sigma, sensitivity);

    // Use tail probabilities for numerical stability
    // log(1 - CDF_N(0,s^2)(x)) = log(Phi(-x/s)) = gaussian_log_cdf(-x/s)
    let log_tail_upper = gaussian_log_cdf(-x_cutoff / sigma);
    let log_tail_shifted = gaussian_log_cdf((sensitivity - x_cutoff) / sigma);

    let log_1_minus_q = (1.0 - q).ln();
    let log_q = q.ln();
    let log_mu_upper = log_add(log_1_minus_q + log_tail_upper, log_q + log_tail_shifted);

    (log_mu_upper.exp() - (epsilon + log_tail_upper).exp()).max(0.0)
}

/// Hockey-stick divergence for REPLACE adjacency with Poisson subsampling
fn get_delta_replace(epsilon: f64, sigma: f64, sensitivity: f64, q: f64) -> Result<f64> {
    let x_cutoff = inverse_privacy_loss_replace(epsilon, sigma, sensitivity, q)?;

    // Mixed CDFs at x*, x*+D, x*-D
    let cdf_center = gaussian_cdf(x_cutoff / sigma);
    let cdf_plus = gaussian_cdf((x_cutoff + sensitivity) / sigma);
    let cdf_minus = gaussian_cdf((x_cutoff - sensitivity) / sigma);

    let mu_upper = q * cdf_plus + (1.0 - q) * cdf_center;
    let mu_lower = q * cdf_minus + (1.0 - q) * cdf_center;

    Ok((mu_upper - epsilon.exp() * mu_lower).max(0.0))
}

/// Standard normal CDF helper
fn gaussian_cdf(z: f64) -> f64 {
    let standard_normal = Normal::new(0.0, 1.0).unwrap();
    standard_normal.cdf(z)
}

// ---------------------------------------------------------------------------
// Evidence implementation
// ---------------------------------------------------------------------------

impl PoissonEvidence<Gaussian> for TightGaussianPoissonEvidence {
    fn compute_pld(&self, inner: &Gaussian, rate: f64) -> Result<PrivacyLossDistribution> {
        let sigma = inner.noise_multiplier;
        let sensitivity = 1.0; // Normalized in functional API
        let config = &inner.config;

        // Use log_mass_truncation_bound for x-space truncation (matching Google dp_accounting)
        let log_mass = config.log_mass_truncation_bound;

        // Compute epsilon bounds for ADD and REMOVE adjacencies
        let bounds_remove = poisson_gaussian_epsilon_bounds(
            sigma,
            sensitivity,
            rate,
            Adjacency::Remove,
            log_mass,
        );
        let bounds_add = poisson_gaussian_epsilon_bounds(
            sigma,
            sensitivity,
            rate,
            Adjacency::Add,
            log_mass,
        );

        discretize_asymmetric_mechanism(config, bounds_remove, bounds_add, |epsilon, adj| {
            Ok(poisson_gaussian_get_delta(
                epsilon,
                adj,
                sigma,
                sensitivity,
                rate,
            ))
        })
        .map(|mut pld| {
            // Chernoff truncation budgets for composition.
            //
            // Tail budgets from config.tail_mass_truncation, split equally
            // between left and right. Matches Google dp_accounting's
            // tail_mass_truncation parameter (default 1e-15).
            let tail_budget = config.tail_mass_truncation / 2.0;
            pld.pmf_remove.right_tail_budget = tail_budget;
            pld.pmf_remove.left_tail_budget = tail_budget;
            if let Some(ref mut pmf_add) = pld.pmf_add {
                pmf_add.right_tail_budget = tail_budget;
                pmf_add.left_tail_budget = tail_budget;
            }
            pld
        })
    }
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

/// Create a Poisson-subsampled mechanism with tight bounds
///
/// Works for any mechanism that implements `PoissonAmplifiable`. The mechanism
/// provides its own evidence type, ensuring the tightest available bounds.
///
/// # Arguments
///
/// * `inner` - The base mechanism (e.g., `gaussian(1.1)` or `adaclip(gaussian(1.1), 50.0)`)
/// * `rate` - Poisson sampling rate q in (0, 1]
///
/// # Panics
///
/// Panics if `rate` is not in (0, 1].
///
/// # Examples
///
/// ```rust,ignore
/// use opaque_dp_accounting::functional::*;
///
/// // Gaussian mechanism
/// let step = poisson(gaussian(1.1), 0.01);
///
/// // AdaClip mechanism
/// let step = poisson(adaclip(gaussian(1.1), 50.0), 0.01);
///
/// // Compose
/// let process = repeat(poisson(gaussian(1.1), 0.01), 1000);
/// let epsilon = process.epsilon_at(1e-5)?;
/// ```
pub fn poisson<P: PoissonAmplifiable>(inner: P, rate: f64) -> Poisson<P::Inner, P::Evidence> {
    assert!(
        rate > 0.0 && rate <= 1.0,
        "Poisson sampling rate must be in (0, 1], got {}",
        rate
    );
    let (inner, evidence) = inner.into_poisson_parts();
    Poisson {
        inner,
        rate,
        evidence,
    }
}

/// Create a Poisson-subsampled process with custom evidence
///
/// Use this for third-party or experimental evidence types. For built-in
/// mechanisms, prefer `poisson()` which automatically selects the tightest
/// available evidence.
///
/// # Arguments
///
/// * `inner` - The base process
/// * `rate` - Poisson sampling rate q in (0, 1]
/// * `evidence` - Evidence proving amplification is valid
///
/// # Panics
///
/// Panics if `rate` is not in (0, 1].
pub fn poisson_with<P, E>(inner: P, rate: f64, evidence: E) -> Poisson<P, E> {
    assert!(
        rate > 0.0 && rate <= 1.0,
        "Poisson sampling rate must be in (0, 1], got {}",
        rate
    );
    Poisson {
        inner,
        rate,
        evidence,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gaussian;

    #[test]
    fn test_poisson_constructor() {
        let pg = poisson(gaussian(1.1).unwrap(), 0.01);
        assert_eq!(pg.inner.noise_multiplier, 1.1);
        assert_eq!(pg.rate, 0.01);
    }

    #[test]
    #[should_panic(expected = "Poisson sampling rate must be in (0, 1]")]
    fn test_poisson_rejects_zero_rate() {
        poisson(gaussian(1.1).unwrap(), 0.0);
    }

    #[test]
    #[should_panic(expected = "Poisson sampling rate must be in (0, 1]")]
    fn test_poisson_rejects_negative_rate() {
        poisson(gaussian(1.1).unwrap(), -0.1);
    }

    #[test]
    #[should_panic(expected = "Poisson sampling rate must be in (0, 1]")]
    fn test_poisson_rejects_rate_above_one() {
        poisson(gaussian(1.1).unwrap(), 1.5);
    }

    #[test]
    fn test_poisson_allows_rate_one() {
        let pg = poisson(gaussian(1.1).unwrap(), 1.0);
        assert_eq!(pg.rate, 1.0);
    }

    #[test]
    fn test_poisson_structural_equality() {
        let a = poisson(gaussian(1.1).unwrap(), 0.01);
        let b = poisson(gaussian(1.1).unwrap(), 0.01);
        assert_eq!(a, b);

        let c = poisson(gaussian(1.1).unwrap(), 0.02);
        assert_ne!(a, c);
    }

    #[test]
    fn test_privacy_loss_remove_no_subsampling() {
        // q=1: should equal raw privacy loss
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
        assert!(
            (l_add + l_rem).abs() < 1e-12,
            "ADD/REMOVE symmetry violated: {} + {} = {}",
            l_add,
            l_rem,
            l_add + l_rem
        );
    }

    #[test]
    fn test_privacy_loss_replace_odd_function() {
        // L_replace(x) = -L_replace(-x) (odd function)
        let sigma = 1.0;
        let sensitivity = 1.0;
        let rate = 0.1;
        for x in [0.1, 0.5, 1.0, 2.0] {
            let l_pos = privacy_loss_replace(x, sigma, sensitivity, rate);
            let l_neg = privacy_loss_replace(-x, sigma, sensitivity, rate);
            assert!(
                (l_pos + l_neg).abs() < 1e-12,
                "REPLACE not odd: L({})={}, L({})={}, sum={}",
                x,
                l_pos,
                -x,
                l_neg,
                l_pos + l_neg
            );
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

    #[test]
    fn test_get_delta_add_theoretical_upper() {
        // For eps >= -log(1-q), delta should be 0
        let q: f64 = 0.1;
        let threshold = -(1.0 - q).ln();
        let delta = get_delta_add(threshold + 0.1, 1.0, 1.0, q);
        assert_eq!(delta, 0.0);
    }

    #[test]
    fn test_get_delta_remove_theoretical_lower() {
        // For eps <= log(1-q), delta = -expm1(eps)
        let q: f64 = 0.1;
        let threshold = (1.0 - q).ln();
        let eps = threshold - 0.1;
        let delta = get_delta_remove(eps, 1.0, 1.0, q);
        let expected = (-eps.exp_m1()).max(0.0);
        assert!(
            (delta - expected).abs() < 1e-10,
            "delta={}, expected={}",
            delta,
            expected
        );
    }

    #[test]
    fn test_get_delta_replace_positive() {
        let delta = get_delta_replace(1.0, 1.0, 1.0, 0.1).unwrap();
        assert!(delta > 0.0 && delta < 1.0, "delta={}", delta);
    }
}
