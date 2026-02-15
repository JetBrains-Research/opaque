//! Basic privacy mechanisms for the functional API
//!
//! This module provides leaf mechanisms that serve as building blocks for
//! more complex privacy analyses. Each mechanism owns its discretization
//! configuration and knows how to compute its own Privacy Loss Distribution (PLD).
//!
//! # Differential Privacy Background
//!
//! A randomized mechanism M provides (ε, δ)-differential privacy if for all
//! neighboring datasets D and D', and all measurable sets S:
//!
//! ```text
//! Pr[M(D) ∈ S] ≤ exp(ε) · Pr[M(D') ∈ S] + δ
//! ```
//!
//! The Privacy Loss Distribution captures the distribution of the privacy loss
//! random variable across all possible outputs, enabling tight composition bounds.
//!
//! # References
//!
//! - Dwork & Roth (2014). "The Algorithmic Foundations of Differential Privacy."
//!   Foundations and Trends in Theoretical Computer Science.
//! - Mironov (2017). "Rényi Differential Privacy." IEEE CSF 2017.
//! - Doroshenko et al. (2022). "Connect the Dots: Tighter Discrete Approximations
//!   of Privacy Loss Distributions." PETS 2022.

use crate::error::{PldError, Result};
use crate::functional::adjacency::Adjacency;
use crate::functional::amplification::{PoissonAmplifiable, TightGaussianPoissonEvidence};
use crate::functional::discretization::{
    discretize_symmetric_mechanism, DiscretizationConfig, EpsilonBounds,
};
use crate::functional::pld::PrivacyLossDistribution;
use crate::functional::process::Process;

/// Minimum supported noise multiplier for the functional API.
///
/// Values below this threshold cause numerical instability in discretization
/// (grid explosion, unreliable epsilon bounds).
pub(crate) const MIN_NOISE_MULTIPLIER: f64 = 0.1;

/// Maximum supported noise multiplier for the functional API.
///
/// Values above this threshold cause numerical instability
/// (x-to-ε compression artifacts, unreliable beta/risk metrics).
pub(crate) const MAX_NOISE_MULTIPLIER: f64 = 1.2;

/// Default minimum delta for PLD right-bound truncation.
///
/// Derived from the largest practical dataset: n = 2^42 ≈ 4.4T records
/// (100T tokens / 30 tokens per record), with δ_min = n^{-1.1} ≈ 6.2e-15.
///
/// This determines accuracy of `delta_at()` and `epsilon_at()` metrics.
pub(crate) const DEFAULT_MIN_DELTA: f64 = 6.2e-15;

/// Default minimum beta for PLD left-bound truncation.
///
/// Set to 1e-6, sufficient for practical `beta_at()` and `risk_at()` queries
/// which typically use thresholds of 1e-4 or larger.
pub(crate) const DEFAULT_MIN_BETA: f64 = 1e-6;

/// Gaussian mechanism with fixed noise multiplier
///
/// The Gaussian mechanism achieves differential privacy by adding noise drawn from
/// a Gaussian (normal) distribution N(0, σ²) to a query result. For a query with
/// sensitivity Δ, the mechanism provides (ε, δ)-DP.
///
/// This implementation uses the normalized form where sensitivity is assumed to be 1.0,
/// so the noise multiplier σ directly determines the privacy-utility tradeoff.
///
/// # Privacy Guarantee
///
/// For the Gaussian mechanism with noise multiplier σ and sensitivity Δ = 1:
/// - Pure DP (δ = 0) is not achievable with finite noise
/// - Approximate DP (ε, δ) is achieved for any ε > 0 and δ > 0
/// - Smaller σ means more noise and stronger privacy (smaller ε for fixed δ)
/// - Larger σ means less noise and weaker privacy (larger ε for fixed δ)
///
/// # Tail bounds
///
/// The PLD is truncated based on `min_delta` (right tail) and `min_beta` (left tail):
/// - `min_delta` controls accuracy of `epsilon_at()` and `delta_at()` queries.
///   Default: ~6.2e-15 (supports datasets up to ~4T records).
/// - `min_beta` controls accuracy of `beta_at()` and `risk_at()` queries.
///   Default: 1e-6 (supports beta queries down to 1e-6).
///
/// Use builder methods to customize: `gaussian(0.5)?.with_min_delta(1e-10).with_min_beta(1e-4)`
///
/// # Parameters
///
/// * `noise_multiplier` - The ratio σ/Δ where σ is the standard deviation
///   and Δ is the sensitivity (assumed to be 1.0 in the functional API)
/// * `config` - Discretization configuration for PLD computation
///
/// # Examples
///
/// ```rust,ignore
/// use opaque_dp_accounting::functional::*;
///
/// // Create Gaussian mechanism with noise multiplier σ = 1.1
/// let g = gaussian(1.1)?;
/// let pld = g.pld()?;
///
/// // Query for privacy parameters
/// let epsilon = pld.epsilon_at(1e-5);  // ε for δ = 10⁻⁵
/// let delta = pld.delta_at(1.0);        // δ for ε = 1.0
///
/// // Custom tail bounds for higher precision
/// let g = gaussian(0.5)?.with_min_delta(1e-20).with_min_beta(1e-8);
/// ```
///
/// # References
///
/// - Dwork & Roth (2014), Section 3.5.3: "The Gaussian Mechanism"
/// - Balle & Wang (2018). "Improving the Gaussian Mechanism for Differential Privacy:
///   Analytical Calibration and Optimal Denoising." ICML 2018.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Gaussian {
    /// Noise multiplier (σ), assuming sensitivity Δ = 1.0
    pub noise_multiplier: f64,
    /// Discretization configuration
    pub config: DiscretizationConfig,
    /// Minimum delta for right-bound truncation (controls epsilon_at/delta_at accuracy)
    pub min_delta: f64,
    /// Minimum beta for left-bound truncation (controls beta_at/risk_at accuracy)
    pub min_beta: f64,
}

impl Eq for Gaussian {}

impl Gaussian {
    /// Internal constructor that skips noise multiplier range validation.
    ///
    /// Used by `pld_replace()` where the equivalent mechanism has `noise_multiplier / 2`,
    /// which may fall below `MIN_NOISE_MULTIPLIER`. Public constructors (`gaussian()`,
    /// `gaussian_with()`) enforce the [0.1, 1.2] range.
    pub(crate) fn new_unchecked(noise_multiplier: f64, config: DiscretizationConfig) -> Self {
        Self {
            noise_multiplier,
            config,
            min_delta: DEFAULT_MIN_DELTA,
            min_beta: DEFAULT_MIN_BETA,
        }
    }

    /// Set a custom minimum delta for right-bound truncation.
    ///
    /// Controls accuracy of `epsilon_at()` and `delta_at()` queries.
    /// Smaller values extend the PLD grid further right, giving higher accuracy
    /// for very small delta queries at the cost of a larger grid.
    pub fn with_min_delta(mut self, min_delta: f64) -> Self {
        self.min_delta = min_delta;
        self
    }

    /// Set a custom minimum beta for left-bound truncation.
    ///
    /// Controls accuracy of `beta_at()` and `risk_at()` queries.
    /// Smaller values extend the PLD grid further left, giving higher accuracy
    /// for very small beta queries at the cost of a larger grid.
    pub fn with_min_beta(mut self, min_beta: f64) -> Self {
        self.min_beta = min_beta;
        self
    }

    /// Compute epsilon bounds for Connect-the-Dots discretization
    ///
    /// For the Gaussian mechanism with noise multiplier σ and sensitivity Δ = 1,
    /// computes the privacy loss range [ε_lower, ε_upper] where discretization
    /// should be performed.
    ///
    /// The right bound (epsilon_upper) is computed by bisection on the analytic
    /// delta function at `min_delta`. The left bound (epsilon_lower) is computed
    /// by bisection on the beta function at `min_beta`.
    fn epsilon_bounds(&self, adjacency: Adjacency) -> Result<EpsilonBounds> {
        use crate::math_helpers::gaussian::{
            gaussian_epsilon_for_beta, gaussian_epsilon_for_delta,
        };

        if adjacency == Adjacency::Replace {
            return Err(crate::error::PldError::InvalidParameter(
                "Use pld_replace() for Replace adjacency".into(),
            ));
        }

        let delta_tilde = 1.0 / self.noise_multiplier;

        // epsilon_upper: bisection on the analytic delta curve.
        // Finds the smallest ε where δ(ε) ≤ min_delta.
        let epsilon_upper = gaussian_epsilon_for_delta(delta_tilde, self.min_delta);

        // epsilon_lower: bisection on the beta (reverse hockey-stick) curve.
        // Finds the most negative ε where β(|ε|) ≤ min_beta.
        let epsilon_lower = gaussian_epsilon_for_beta(delta_tilde, self.min_beta);

        Ok(EpsilonBounds {
            epsilon_lower,
            epsilon_upper,
        })
    }
}

impl Gaussian {
    /// Compute the PLD for Replace adjacency
    ///
    /// Under Replace adjacency, changing one record is equivalent to removing one
    /// and adding another, so sensitivity doubles. This is equivalent to computing
    /// the PLD for a Gaussian mechanism with `noise_multiplier / 2` under
    /// Add/Remove adjacency.
    pub fn pld_replace(&self) -> Result<PrivacyLossDistribution> {
        let equivalent = Gaussian::new_unchecked(self.noise_multiplier / 2.0, self.config.clone())
            .with_min_delta(self.min_delta)
            .with_min_beta(self.min_beta);
        equivalent.pld()
    }
}

impl Process for Gaussian {
    fn pld(&self) -> Result<PrivacyLossDistribution> {
        // The Gaussian delta function is symmetric in ADD/REMOVE, and with
        // delta-space bisection the epsilon bounds are also identical for both.
        // A single symmetric PLD evaluation suffices.
        let bounds = self.epsilon_bounds(Adjacency::Add)?;
        let delta_tilde = 1.0 / self.noise_multiplier;

        discretize_symmetric_mechanism(&self.config, bounds, |epsilon| {
            crate::math_helpers::gaussian::gaussian_delta_at(delta_tilde, epsilon)
        })
        // Tail budgets for Chernoff truncation during composition:
        // - Right budget = min_delta (truncated mass → infinity_mass → delta floor)
        // - Left budget = min_beta (the promised beta accuracy contract)
        .map(|pld| pld.with_tail_budgets(self.min_delta, self.min_beta))
    }
}

impl PoissonAmplifiable for Gaussian {
    type Inner = Gaussian;
    type Evidence = TightGaussianPoissonEvidence;

    fn into_poisson_parts(self) -> (Gaussian, TightGaussianPoissonEvidence) {
        (self, TightGaussianPoissonEvidence)
    }
}

/// Constructor for Gaussian mechanism with default discretization
///
/// Uses default discretization config (discretization=1e-4, log_mass=-32.0).
///
/// # Arguments
///
/// * `noise_multiplier` - The ratio σ/Δ where σ is the standard deviation.
///   Must be in [`MIN_NOISE_MULTIPLIER`, `MAX_NOISE_MULTIPLIER`] = [0.1, 1.2].
///
/// # Errors
///
/// Returns `PldError::InvalidParameter` if `noise_multiplier` is outside [0.1, 1.2].
///
/// # Examples
///
/// ```rust,ignore
/// use opaque_dp_accounting::functional::*;
///
/// let process = gaussian(1.1)?;
/// let epsilon = process.epsilon_at(1e-5)?;
/// ```
pub fn gaussian(noise_multiplier: f64) -> Result<Gaussian> {
    validate_noise_multiplier(noise_multiplier)?;
    Ok(Gaussian {
        noise_multiplier,
        config: DiscretizationConfig::default(),
        min_delta: DEFAULT_MIN_DELTA,
        min_beta: DEFAULT_MIN_BETA,
    })
}

/// Constructor for Gaussian mechanism with custom discretization
///
/// # Arguments
///
/// * `noise_multiplier` - The ratio σ/Δ where σ is the standard deviation.
///   Must be in [`MIN_NOISE_MULTIPLIER`, `MAX_NOISE_MULTIPLIER`] = [0.1, 1.2].
/// * `config` - Custom discretization configuration
///
/// # Errors
///
/// Returns `PldError::InvalidParameter` if `noise_multiplier` is outside [0.1, 1.2].
///
/// # Examples
///
/// ```rust,ignore
/// use opaque_dp_accounting::functional::*;
///
/// let config = DiscretizationConfig::new(0.01, -50.0)?;
/// let process = gaussian_with(1.1, config)?;
/// ```
pub fn gaussian_with(noise_multiplier: f64, config: DiscretizationConfig) -> Result<Gaussian> {
    validate_noise_multiplier(noise_multiplier)?;
    Ok(Gaussian {
        noise_multiplier,
        config,
        min_delta: DEFAULT_MIN_DELTA,
        min_beta: DEFAULT_MIN_BETA,
    })
}

/// Validate that noise_multiplier is within the supported range [0.1, 1.2]
fn validate_noise_multiplier(noise_multiplier: f64) -> Result<()> {
    if !(MIN_NOISE_MULTIPLIER..=MAX_NOISE_MULTIPLIER).contains(&noise_multiplier) {
        return Err(PldError::InvalidParameter(format!(
            "noise_multiplier must be in [{}, {}], got {}. \
             Values outside this range are not supported due to numerical instability.",
            MIN_NOISE_MULTIPLIER, MAX_NOISE_MULTIPLIER, noise_multiplier
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_constructor() {
        let gauss = gaussian(1.1).unwrap();
        assert_eq!(gauss.noise_multiplier, 1.1);
    }

    #[test]
    fn test_gaussian_with_custom_config() {
        let config = DiscretizationConfig::new(0.01, -1e-30).unwrap();
        let gauss = gaussian_with(1.1, config.clone()).unwrap();
        assert_eq!(gauss.noise_multiplier, 1.1);
        assert_eq!(gauss.config.discretization, config.discretization);
    }

    #[test]
    fn test_gaussian_rejects_zero() {
        assert!(gaussian(0.0).is_err());
    }

    #[test]
    fn test_gaussian_rejects_negative() {
        assert!(gaussian(-1.0).is_err());
    }

    #[test]
    fn test_gaussian_rejects_below_min() {
        assert!(gaussian(0.09).is_err());
    }

    #[test]
    fn test_gaussian_rejects_above_max() {
        assert!(gaussian(1.21).is_err());
    }

    #[test]
    fn test_gaussian_boundary_min() {
        assert!(gaussian(MIN_NOISE_MULTIPLIER).is_ok());
    }

    #[test]
    fn test_gaussian_boundary_max() {
        assert!(gaussian(MAX_NOISE_MULTIPLIER).is_ok());
    }

    #[test]
    fn test_gaussian_builder_min_delta() {
        let g = gaussian(0.5).unwrap().with_min_delta(1e-20);
        assert_eq!(g.min_delta, 1e-20);
        assert_eq!(g.min_beta, DEFAULT_MIN_BETA); // unchanged
    }

    #[test]
    fn test_gaussian_builder_min_beta() {
        let g = gaussian(0.5).unwrap().with_min_beta(1e-8);
        assert_eq!(g.min_beta, 1e-8);
        assert_eq!(g.min_delta, DEFAULT_MIN_DELTA); // unchanged
    }

    #[test]
    fn test_gaussian_defaults() {
        let g = gaussian(1.0).unwrap();
        assert_eq!(g.min_delta, DEFAULT_MIN_DELTA);
        assert_eq!(g.min_beta, DEFAULT_MIN_BETA);
    }
}
