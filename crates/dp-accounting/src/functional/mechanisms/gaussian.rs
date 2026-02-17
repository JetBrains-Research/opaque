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
use crate::amplification::{PoissonAmplifiable, TightGaussianPoissonEvidence};
use crate::discretization::{
    discretize_symmetric_mechanism, DiscretizationConfig, EpsilonBounds,
};
use crate::pld::PrivacyLossDistribution;
use crate::process::Process;

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
/// - Smaller σ means less noise and weaker privacy (larger ε for fixed δ)
/// - Larger σ means more noise and stronger privacy (smaller ε for fixed δ)
///
/// # Truncation
///
/// The PLD grid extent is controlled by `config.log_mass_truncation_bound` (default -50),
/// matching Google dp_accounting's x-space truncation approach. Composition truncation
/// is controlled by `config.tail_mass_truncation` (default 1e-15).
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
        }
    }

    /// Compute epsilon bounds using x-space truncation (matching Google dp_accounting).
    ///
    /// Finds x-space truncation points from the Gaussian tail via
    /// `ppf(0.5 * exp(log_mass_truncation_bound))`, then evaluates the privacy loss
    /// function at those points to get epsilon bounds.
    ///
    /// For the non-subsampled Gaussian with sensitivity Δ = 1:
    /// - `lower_x = σ · ppf(half_mass) - 1`  (shift for REMOVE mu_upper)
    /// - `upper_x = -σ · ppf(half_mass)`
    /// - `epsilon_upper = L(lower_x) = (0.5 - σ·ppf(half_mass)) / σ²`
    /// - `epsilon_lower = L(upper_x) = -epsilon_upper`  (symmetric)
    fn epsilon_bounds(&self) -> EpsilonBounds {
        use statrs::distribution::{ContinuousCDF, Normal};

        let sigma = self.noise_multiplier;
        let sensitivity = 1.0;
        let log_mass = self.config.log_mass_truncation_bound;

        let standard_normal = Normal::new(0.0, 1.0).unwrap();
        let half_mass = 0.5 * log_mass.exp();
        let z = standard_normal.inverse_cdf(half_mass); // very negative

        // For the symmetric Gaussian, REMOVE bounds:
        // lower_x = sigma * z - sensitivity
        // epsilon_upper = sensitivity * (-0.5*sensitivity - lower_x) / sigma^2
        //               = sensitivity * (0.5*sensitivity - sigma*z) / sigma^2
        let epsilon_upper =
            sensitivity * (0.5 * sensitivity - sigma * z) / (sigma * sigma);

        // Symmetric: epsilon_lower = -epsilon_upper
        EpsilonBounds {
            epsilon_lower: -epsilon_upper,
            epsilon_upper,
        }
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
        let equivalent = Gaussian::new_unchecked(self.noise_multiplier / 2.0, self.config.clone());
        equivalent.pld()
    }
}

impl Process for Gaussian {
    fn pld(&self) -> Result<PrivacyLossDistribution> {
        let bounds = self.epsilon_bounds();
        let delta_tilde = 1.0 / self.noise_multiplier;

        let tail_budget = self.config.tail_mass_truncation / 2.0;

        discretize_symmetric_mechanism(&self.config, bounds, |epsilon| {
            crate::math_helpers::gaussian::gaussian_delta_at(delta_tilde, epsilon)
        })
        .map(|pld| pld.with_tail_budgets(tail_budget, tail_budget))
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
/// Uses default discretization config (discretization=1e-4, log_mass_truncation_bound=-50).
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
    fn test_gaussian_default_config() {
        let g = gaussian(1.0).unwrap();
        assert_eq!(g.config.log_mass_truncation_bound, -50.0);
        assert_eq!(g.config.tail_mass_truncation, 1e-15);
    }
}
