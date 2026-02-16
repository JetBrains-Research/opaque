//! Truncated Poisson subsampling amplification for privacy mechanisms
//!
//! Implements truncated Poisson sampling, the variant actually used in production
//! DP-SGD systems (Opacus, JAX Privacy, TensorFlow Privacy). Unlike standard
//! Poisson sampling which has variable batch sizes, truncated sampling caps the
//! batch size at B_max for predictable memory usage and compute time.
//!
//! # Algorithm
//!
//! ```text
//! For each iteration:
//!   1. Sample batch B ~ Poisson(q) from dataset
//!   2. If |B| > B_max: randomly subsample B_max examples from B
//!   3. Apply Gaussian mechanism to the (potentially truncated) batch
//! ```
//!
//! # Privacy Analysis
//!
//! From \[Gan25\], the privacy loss distribution is a **mixture** of two components:
//! - Component 1 (prob `1 - p_trunc`): Standard Poisson sampling PLD,
//!   using pessimistic `max(ADD, REMOVE)` for ADD/REMOVE adjacency
//! - Component 2 (prob `p_trunc`): Poisson with doubled sensitivity (σ/2)
//!   at conditional rate q_cond, REPLACE adjacency only
//!
//! The mixture operates at the delta level:
//! ```text
//! δ_truncated(ε) = (1 - p_trunc) · δ_comp1(ε) + p_trunc · δ_comp2(ε)
//! ```
//!
//! # References
//!
//! - \[Gan25\]: "Tighter privacy analysis for truncated poisson sampling" (2025)
//! - \[CGK+24\]: "Scalable DP-SGD: shuffling vs. poisson subsampling" (NeurIPS 2024)

use crate::error::Result;
use crate::functional::adjacency::Adjacency;
use crate::functional::discretization::discretize_asymmetric_mechanism;
use crate::functional::mechanisms::Gaussian;
use crate::functional::pld::PrivacyLossDistribution;
use crate::functional::process::Process;
use statrs::distribution::{Binomial, DiscreteCDF};

use super::poisson::{
    poisson_gaussian_epsilon_bounds, poisson_gaussian_get_delta, PoissonEvidence,
    TightGaussianPoissonEvidence,
};
use crate::functional::discretization::config::EpsilonBounds;

// ---------------------------------------------------------------------------
// Trait + types
// ---------------------------------------------------------------------------

/// Evidence that truncated Poisson subsampling can be applied to a process of type P
///
/// Different evidence types provide different tightness guarantees.
/// For Gaussian mechanisms, `TightGaussianTruncatedPoissonEvidence` provides
/// tight bounds using the mixture formula from \[Gan25\].
pub trait TruncatedPoissonEvidence<P>: Clone {
    /// Compute the PLD for a truncated Poisson-subsampled process
    ///
    /// # Arguments
    ///
    /// * `inner` - The base process
    /// * `rate` - Poisson sampling rate q in (0, 1]
    /// * `batch_size_max` - Maximum batch size B_max
    /// * `dataset_size` - Total dataset size n
    ///
    /// # Errors
    ///
    /// * `PldError::InvalidParameter` - If parameters are out of range
    /// * `PldError::NumericalError` - If discretization fails
    fn compute_pld(
        &self,
        inner: &P,
        rate: f64,
        batch_size_max: usize,
        dataset_size: usize,
    ) -> Result<PrivacyLossDistribution>;
}

/// Truncated Poisson-subsampled process with evidence-based amplification
///
/// Wraps an inner process P with truncated Poisson subsampling. Each record
/// is sampled independently with probability q, and the batch size is capped
/// at B_max. The evidence type E determines how the amplified PLD is computed.
///
/// # Type Parameters
///
/// * `P` - The inner process type (e.g., `Gaussian`)
/// * `E` - The evidence type (e.g., `TightGaussianTruncatedPoissonEvidence`)
///
/// # Examples
///
/// ```rust,ignore
/// use opaque_dp_accounting::functional::*;
///
/// let step = truncated_poisson(gaussian(1.1), 0.01, 1024, 1_000_000);
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
pub struct TruncatedPoisson<P, E> {
    /// The inner (base) process
    pub inner: P,
    /// Poisson sampling rate q in (0, 1]
    pub rate: f64,
    /// Maximum batch size B_max
    pub batch_size_max: usize,
    /// Total dataset size n
    pub dataset_size: usize,
    /// Evidence proving truncated amplification is valid for P
    pub evidence: E,
}

impl<P: PartialEq, E: PartialEq> Eq for TruncatedPoisson<P, E> {}

impl<P, E: TruncatedPoissonEvidence<P>> Process for TruncatedPoisson<P, E> {
    fn pld(&self) -> Result<PrivacyLossDistribution> {
        self.evidence.compute_pld(
            &self.inner,
            self.rate,
            self.batch_size_max,
            self.dataset_size,
        )
    }
}

/// Tight truncated Poisson amplification evidence for Gaussian mechanisms
///
/// Uses the mixture formula from \[Gan25\]:
/// - Component 1 (prob `1 - p_trunc`): Standard Poisson at rate q,
///   pessimistic `max(ADD, REMOVE)` for ADD/REMOVE adjacency
/// - Component 2 (prob `p_trunc`): Poisson at rate q_cond with doubled
///   sensitivity (σ/2), REPLACE adjacency
///
/// Falls back to standard Poisson when `p_trunc == 0`.
///
/// # References
///
/// - \[Gan25\]: "Tighter privacy analysis for truncated poisson sampling" (2025)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TightGaussianTruncatedPoissonEvidence;

/// Trait for mechanisms that support truncated Poisson subsampling amplification
///
/// Implementors specify how to convert themselves into a form suitable for
/// truncated Poisson amplification. The associated types determine the evidence
/// and inner process type.
///
/// ```rust,ignore
/// truncated_poisson(gaussian(1.1), 0.01, 1024, 1_000_000)
/// truncated_poisson(adaclip(gaussian(1.1), 50.0), 0.01, 1024, 1_000_000)
/// ```
pub trait TruncatedPoissonAmplifiable: Process + Sized {
    /// The process type stored inside `TruncatedPoisson<Inner, Evidence>`
    type Inner: Process;
    /// The evidence type proving truncated amplification is valid
    type Evidence: TruncatedPoissonEvidence<Self::Inner>;

    /// Convert this process into the inner form and its evidence
    fn into_truncated_poisson_parts(self) -> (Self::Inner, Self::Evidence);
}

impl TruncatedPoissonAmplifiable for Gaussian {
    type Inner = Gaussian;
    type Evidence = TightGaussianTruncatedPoissonEvidence;

    fn into_truncated_poisson_parts(self) -> (Gaussian, TightGaussianTruncatedPoissonEvidence) {
        (self, TightGaussianTruncatedPoissonEvidence)
    }
}

// ---------------------------------------------------------------------------
// Math: truncation probabilities
// ---------------------------------------------------------------------------

/// Compute truncation probability: Pr\[Binom(n-1, q) >= B_max\]
///
/// This is the probability that truncation occurs under REMOVE adjacency
/// (removing one record from a dataset of size n).
///
/// Returns 0.0 if `B_max >= n` (truncation never happens).
fn truncation_probability(dataset_size: usize, rate: f64, batch_size_max: usize) -> f64 {
    if batch_size_max >= dataset_size {
        return 0.0;
    }

    // Pr[Binom(n-1, q) >= B_max] = 1 - CDF(B_max - 1)
    let binom = Binomial::new(rate, (dataset_size - 1) as u64).unwrap();
    1.0 - binom.cdf((batch_size_max - 1) as u64)
}

/// Compute the conditional sampling probability for the truncated component
///
/// From \[Gan25\]:
/// ```text
/// q_cond = Pr[Binom(n, q) > B_max] · (B_max / n) / p_trunc
/// ```
///
/// This is the effective sampling probability for the sensitive record
/// when truncation occurs.
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
    // Pr[Binom(n, q) > B_max] = 1 - CDF(B_max)
    let binom = Binomial::new(rate, n as u64).unwrap();
    let pr_exceed = 1.0 - binom.cdf(batch_size_max as u64);

    pr_exceed * (batch_size_max as f64) / (n as f64) / p_trunc
}

// ---------------------------------------------------------------------------
// Math: mixture delta
// ---------------------------------------------------------------------------

/// Hockey-stick divergence for truncated Poisson-subsampled Gaussian
///
/// Mixture formula from \[Gan25\]:
/// ```text
/// δ(ε) = (1 - p_trunc) · δ_comp1(ε) + p_trunc · δ_comp2(ε)
/// ```
///
/// - Component 1: Standard Poisson, pessimistic `max(ADD, REMOVE)` for ADD/REMOVE
/// - Component 2: Poisson with doubled sensitivity (σ/2), REPLACE, rate q_cond
pub(crate) fn truncated_get_delta(
    epsilon: f64,
    adjacency: Adjacency,
    sigma: f64,
    sensitivity: f64,
    rate: f64,
    p_trunc: f64,
    q_cond: f64,
) -> f64 {
    // When p_trunc == 0, fall back to standard Poisson (no pessimistic bound needed)
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

    // Mixture
    (1.0 - p_trunc) * delta_comp1 + p_trunc * delta_comp2
}

// ---------------------------------------------------------------------------
// Epsilon bounds for truncated mechanism
// ---------------------------------------------------------------------------

/// Compute epsilon bounds for the truncated Poisson mechanism using x-space
/// truncation on each mixture component.
///
/// The truncated mechanism is a mixture of two Poisson-subsampled Gaussians:
/// - Component 1 (prob 1-p_trunc): Standard Poisson at rate q, pessimistic max(ADD, REMOVE)
/// - Component 2 (prob p_trunc): Poisson REPLACE at rate q_cond, sigma/2
///
/// Each component is a Poisson-subsampled Gaussian, so we compute x-space
/// truncation bounds per component via `poisson_gaussian_epsilon_bounds()`
/// and take the union (widest range).
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

    // Component 2: Poisson REPLACE, sigma/2 (doubled sensitivity), rate q_cond
    let bounds2 = poisson_gaussian_epsilon_bounds(
        sigma / 2.0,
        sensitivity,
        q_cond,
        Adjacency::Replace,
        log_mass_truncation_bound,
    );

    // Union of bounds (widest range covers both components)
    EpsilonBounds {
        epsilon_lower: bounds1.epsilon_lower.min(bounds2.epsilon_lower),
        epsilon_upper: bounds1.epsilon_upper.max(bounds2.epsilon_upper),
    }
}

// ---------------------------------------------------------------------------
// Evidence implementation
// ---------------------------------------------------------------------------

impl TruncatedPoissonEvidence<Gaussian> for TightGaussianTruncatedPoissonEvidence {
    fn compute_pld(
        &self,
        inner: &Gaussian,
        rate: f64,
        batch_size_max: usize,
        dataset_size: usize,
    ) -> Result<PrivacyLossDistribution> {
        let sigma = inner.noise_multiplier;
        let sensitivity = 1.0; // Normalized in functional API
        let config = &inner.config;

        let p_trunc = truncation_probability(dataset_size, rate, batch_size_max);

        // If no truncation, fall back to standard Poisson
        if p_trunc == 0.0 {
            return TightGaussianPoissonEvidence.compute_pld(inner, rate);
        }

        let q_cond = conditional_sampling_probability(dataset_size, rate, batch_size_max, p_trunc);

        // Epsilon bounds: x-space truncation on each mixture component.
        let log_mass = config.log_mass_truncation_bound;
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

        let tail_budget = config.tail_mass_truncation / 2.0;
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
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

/// Create a truncated Poisson-subsampled mechanism with tight bounds
///
/// Works for any mechanism that implements `TruncatedPoissonAmplifiable`.
/// The mechanism provides its own evidence type, ensuring the tightest
/// available bounds.
///
/// # Arguments
///
/// * `inner` - The base mechanism (e.g., `gaussian(1.1)`)
/// * `rate` - Poisson sampling rate q in (0, 1]
/// * `batch_size_max` - Maximum batch size B_max
/// * `dataset_size` - Total dataset size n
///
/// # Panics
///
/// Panics if `rate` is not in (0, 1], `batch_size_max` is 0, or `dataset_size` is 0.
///
/// # Examples
///
/// ```rust,ignore
/// use opaque_dp_accounting::functional::*;
///
/// // Realistic DP-SGD scenario
/// let step = truncated_poisson(gaussian(4.0), 0.001, 1024, 1_000_000);
/// let epsilon = step.epsilon_at(1e-5)?;
///
/// // Composition
/// let process = repeat(truncated_poisson(gaussian(4.0), 0.001, 1024, 1_000_000), 1000);
/// ```
pub fn truncated_poisson<P: TruncatedPoissonAmplifiable>(
    inner: P,
    rate: f64,
    batch_size_max: usize,
    dataset_size: usize,
) -> TruncatedPoisson<P::Inner, P::Evidence> {
    assert!(
        rate > 0.0 && rate <= 1.0,
        "Poisson sampling rate must be in (0, 1], got {}",
        rate
    );
    assert!(
        batch_size_max > 0,
        "Max batch size must be positive, got {}",
        batch_size_max
    );
    assert!(
        dataset_size > 0,
        "Dataset size must be positive, got {}",
        dataset_size
    );
    let (inner, evidence) = inner.into_truncated_poisson_parts();
    TruncatedPoisson {
        inner,
        rate,
        batch_size_max,
        dataset_size,
        evidence,
    }
}

/// Create a truncated Poisson-subsampled process with custom evidence
///
/// Use this for third-party or experimental evidence types. For built-in
/// mechanisms, prefer `truncated_poisson()` which automatically selects
/// the tightest available evidence.
///
/// # Arguments
///
/// * `inner` - The base process
/// * `rate` - Poisson sampling rate q in (0, 1]
/// * `batch_size_max` - Maximum batch size B_max
/// * `dataset_size` - Total dataset size n
/// * `evidence` - Evidence proving truncated amplification is valid
///
/// # Panics
///
/// Panics if `rate` is not in (0, 1], `batch_size_max` is 0, or `dataset_size` is 0.
pub fn truncated_poisson_with<P, E>(
    inner: P,
    rate: f64,
    batch_size_max: usize,
    dataset_size: usize,
    evidence: E,
) -> TruncatedPoisson<P, E> {
    assert!(
        rate > 0.0 && rate <= 1.0,
        "Poisson sampling rate must be in (0, 1], got {}",
        rate
    );
    assert!(
        batch_size_max > 0,
        "Max batch size must be positive, got {}",
        batch_size_max
    );
    assert!(
        dataset_size > 0,
        "Dataset size must be positive, got {}",
        dataset_size
    );
    TruncatedPoisson {
        inner,
        rate,
        batch_size_max,
        dataset_size,
        evidence,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::functional::gaussian;

    #[test]
    fn test_truncated_poisson_constructor() {
        let tp = truncated_poisson(gaussian(1.1).unwrap(), 0.01, 1024, 1_000_000);
        assert_eq!(tp.inner.noise_multiplier, 1.1);
        assert_eq!(tp.rate, 0.01);
        assert_eq!(tp.batch_size_max, 1024);
        assert_eq!(tp.dataset_size, 1_000_000);
    }

    #[test]
    #[should_panic(expected = "Poisson sampling rate must be in (0, 1]")]
    fn test_rejects_zero_rate() {
        truncated_poisson(gaussian(1.1).unwrap(), 0.0, 1024, 1_000_000);
    }

    #[test]
    #[should_panic(expected = "Poisson sampling rate must be in (0, 1]")]
    fn test_rejects_negative_rate() {
        truncated_poisson(gaussian(1.1).unwrap(), -0.1, 1024, 1_000_000);
    }

    #[test]
    #[should_panic(expected = "Max batch size must be positive")]
    fn test_rejects_zero_batch_size() {
        truncated_poisson(gaussian(1.1).unwrap(), 0.01, 0, 1_000_000);
    }

    #[test]
    #[should_panic(expected = "Dataset size must be positive")]
    fn test_rejects_zero_dataset_size() {
        truncated_poisson(gaussian(1.1).unwrap(), 0.01, 1024, 0);
    }

    #[test]
    fn test_structural_equality() {
        let a = truncated_poisson(gaussian(1.1).unwrap(), 0.01, 1024, 1_000_000);
        let b = truncated_poisson(gaussian(1.1).unwrap(), 0.01, 1024, 1_000_000);
        assert_eq!(a, b);

        let c = truncated_poisson(gaussian(1.1).unwrap(), 0.01, 2048, 1_000_000);
        assert_ne!(a, c);

        let d = truncated_poisson(gaussian(1.1).unwrap(), 0.01, 1024, 500_000);
        assert_ne!(a, d);
    }

    #[test]
    fn test_truncation_probability_no_truncation() {
        // b_max >= n -> 0.0
        assert_eq!(truncation_probability(100, 0.1, 100), 0.0);
        assert_eq!(truncation_probability(100, 0.1, 200), 0.0);
    }

    #[test]
    fn test_truncation_probability_realistic() {
        // n=1M, q=0.001, b_max=1024 (expected batch ~999, close to b_max)
        let p = truncation_probability(1_000_000, 0.001, 1024);
        assert!(p > 0.0 && p < 1.0, "p_trunc={}", p);
    }

    #[test]
    fn test_truncation_probability_large_bmax() {
        // b_max >> expected batch -> very small p_trunc
        let p = truncation_probability(1_000_000, 0.001, 10_000);
        assert!(p < 0.001, "p_trunc={}", p);
    }

    #[test]
    fn test_conditional_sampling_probability_valid() {
        let p_trunc = truncation_probability(1_000_000, 0.001, 1024);
        let q_cond = conditional_sampling_probability(1_000_000, 0.001, 1024, p_trunc);
        assert!(
            q_cond > 0.0 && q_cond <= 1.0,
            "q_cond={}, p_trunc={}",
            q_cond,
            p_trunc
        );
    }

    #[test]
    fn test_conditional_sampling_probability_zero_truncation() {
        assert_eq!(conditional_sampling_probability(100, 0.1, 200, 0.0), 0.0);
    }
}
