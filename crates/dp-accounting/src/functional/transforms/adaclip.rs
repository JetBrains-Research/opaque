//! Adaptive clipping mechanism combinator
//!
//! Implements the privacy accounting for DP-FedAvg with adaptive clipping from
//! Andrew et al. (2021). Each round accesses data via two Gaussian mechanisms:
//!
//! 1. **Gradient sum**: clips to norm C, adds N(0, σ_Δ²) noise
//! 2. **Quantile estimation**: binary indicator (clipped or not), sensitivity 0.5,
//!    adds N(0, σ_b²) noise
//!
//! By Theorem 1, these compose into a single equivalent mechanism with effective
//! noise multiplier `z = 1/S` where `S = sqrt((C/σ_Δ)² + 1/(2σ_b)²)`.
//!
//! # Evidence-based design
//!
//! `AdaClip<P, E>` follows the same combinator pattern as `Poisson<P, E>`:
//! the inner mechanism must implement [`AdaClipable`], which provides evidence
//! that the combination is mathematically justified. Mechanisms without proven
//! AdaClip analysis simply don't implement the trait — attempting to wrap them
//! produces a compile error.
//!
//! # Examples
//!
//! ```rust,ignore
//! use opaque_dp_accounting::functional::*;
//!
//! // Standard Gaussian with adaptive clipping
//! let step = adaclip(gaussian(1.1)?, 50.0);
//! let epsilon = step.epsilon_at(1e-5)?;
//!
//! // With Poisson amplification
//! let step = poisson(adaclip(gaussian(1.1)?, 50.0), 0.01);
//! let process = repeat(step, 1000);
//! ```
//!
//! # References
//!
//! - Andrew, Thakkar, McMahan, Ramaswamy (2021). "Differentially Private Learning
//!   with Adaptive Clipping." NeurIPS 2021.

use crate::error::Result;
use crate::functional::amplification::{PoissonAmplifiable, TruncatedPoissonAmplifiable};
use crate::functional::mechanisms::Gaussian;
use crate::functional::pld::PrivacyLossDistribution;
use crate::functional::process::Process;

// ---------------------------------------------------------------------------
// Core traits
// ---------------------------------------------------------------------------

/// Evidence that adaptive clipping is valid for process type `P`.
///
/// The existence of an evidence value proves the combination is mathematically
/// justified. Mechanisms without proven AdaClip analysis simply don't have an
/// evidence type — compile error prevents invalid combinations.
///
/// This mirrors [`PoissonEvidence`](crate::functional::amplification::PoissonEvidence):
/// the evidence system models scarcity of proven combinations, not generic
/// improvement.
pub trait AdaClipEvidence<P>: Clone {
    /// Compute the PLD for the AdaClip-wrapped mechanism.
    ///
    /// The evidence is responsible for computing the effective mechanism
    /// (applying the combined sensitivity transformation) and returning its PLD.
    fn compute_pld(&self, inner: &P, quantile_noise_std: f64) -> Result<PrivacyLossDistribution>;
}

/// A mechanism that can be wrapped by adaptive clipping.
///
/// Mirrors [`PoissonAmplifiable`]: the mechanism declares how to produce
/// evidence for the combination and how to construct the effective inner
/// process for amplification delegation.
///
/// Only mechanisms with proven AdaClip analysis implement this trait:
/// - `Gaussian` ✓ (Andrew et al. 2021, Theorem 1)
/// - `BoundedGaussian` (future — same combined sensitivity argument applies)
/// - `Identity` ✗ (no logical meaning — compile error)
/// - `Laplace` ✗ (no research yet — compile error)
pub trait AdaClipable: Process + Sized + Clone {
    /// Evidence proving the combination is valid
    type Evidence: AdaClipEvidence<Self>;

    /// The noise multiplier of this mechanism (z_Δ = σ_Δ/C)
    fn noise_multiplier(&self) -> f64;

    /// Produce the evidence for this mechanism's AdaClip combination.
    fn adaclip_evidence() -> Self::Evidence;

    /// Create the effective version of this mechanism for AdaClip.
    ///
    /// Computes the effective noise multiplier `z_eff = 1 / S` where
    /// `S = combined_sensitivity(z_Δ, σ_b)`, and returns a copy of self
    /// with the effective noise multiplier applied.
    fn with_effective_adaclip_nm(&self, quantile_noise_std: f64) -> Self;
}

// ---------------------------------------------------------------------------
// Combinator struct
// ---------------------------------------------------------------------------

/// Adaptive clipping mechanism combinator
///
/// Wraps an inner mechanism with adaptive clipping parameters. The inner
/// mechanism is stored in its **original** form; the effective noise multiplier
/// is computed on the fly for PLD construction and amplification delegation.
///
/// # Type Parameters
///
/// * `P` - The inner process type (e.g., `Gaussian`)
/// * `E` - The evidence type (e.g., `GaussianAdaClipEvidence`)
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "P: serde::Serialize, E: serde::Serialize",
        deserialize = "P: serde::de::DeserializeOwned, E: serde::de::DeserializeOwned",
    ))
)]
pub struct AdaClip<P, E> {
    /// The inner mechanism in its original (untransformed) form
    pub inner: P,

    /// Noise standard deviation for the quantile estimation query (`σ_b`)
    ///
    /// Controls how much privacy budget is spent on estimating the clipping
    /// threshold. Larger values mean less privacy cost for the quantile query
    /// but noisier threshold estimates. Typical value: `expected_num_records / 20`.
    pub quantile_noise_std: f64,

    /// Evidence proving the AdaClip combination is valid
    pub evidence: E,
}

impl<P: PartialEq, E: PartialEq> Eq for AdaClip<P, E> {}

impl<P: AdaClipable, E> AdaClip<P, E> {
    /// Combined normalized sensitivity δ̃ from Theorem 1
    ///
    /// ```text
    /// δ̃ = sqrt((C/σ_Δ)² + (0.5/σ_b)²)   (with C = 1 in normalized units)
    ///    = sqrt(1/z_Δ² + 1/(2σ_b)²)
    /// ```
    ///
    /// where `z_Δ` is the gradient noise multiplier and `σ_b` is the quantile
    /// noise std. This is the combined sensitivity-to-noise ratio of the
    /// equivalent single mechanism.
    pub fn combined_sensitivity(&self) -> f64 {
        combined_sensitivity(self.inner.noise_multiplier(), self.quantile_noise_std)
    }

    /// Effective noise multiplier for the equivalent unit-sensitivity mechanism
    ///
    /// By Theorem 1 of Andrew et al. (2021), the effective noise multiplier is
    /// `z = 1 / δ̃ = 1 / combined_sensitivity()`.
    ///
    /// For a Gaussian mechanism with sensitivity 1, the privacy parameter is
    /// `δ̃ = 1 / noise_multiplier`, so the equivalent mechanism must have
    /// `noise_multiplier = 1 / δ̃ = 1 / combined_sensitivity()`.
    pub fn effective_noise_multiplier(&self) -> f64 {
        1.0 / self.combined_sensitivity()
    }
}

// ---------------------------------------------------------------------------
// Process implementation
// ---------------------------------------------------------------------------

impl<P: AdaClipable, E: AdaClipEvidence<P>> Process for AdaClip<P, E> {
    fn pld(&self) -> Result<PrivacyLossDistribution> {
        self.evidence
            .compute_pld(&self.inner, self.quantile_noise_std)
    }
}

// ---------------------------------------------------------------------------
// Amplification delegation
// ---------------------------------------------------------------------------

impl<P, E> PoissonAmplifiable for AdaClip<P, E>
where
    P: AdaClipable + PoissonAmplifiable,
    E: AdaClipEvidence<P>,
{
    type Inner = <P as PoissonAmplifiable>::Inner;
    type Evidence = <P as PoissonAmplifiable>::Evidence;

    fn into_poisson_parts(self) -> (Self::Inner, Self::Evidence) {
        let effective = self
            .inner
            .with_effective_adaclip_nm(self.quantile_noise_std);
        effective.into_poisson_parts()
    }
}

impl<P, E> TruncatedPoissonAmplifiable for AdaClip<P, E>
where
    P: AdaClipable + TruncatedPoissonAmplifiable,
    E: AdaClipEvidence<P>,
{
    type Inner = <P as TruncatedPoissonAmplifiable>::Inner;
    type Evidence = <P as TruncatedPoissonAmplifiable>::Evidence;

    fn into_truncated_poisson_parts(self) -> (Self::Inner, Self::Evidence) {
        let effective = self
            .inner
            .with_effective_adaclip_nm(self.quantile_noise_std);
        effective.into_truncated_poisson_parts()
    }
}

// ---------------------------------------------------------------------------
// Gaussian evidence
// ---------------------------------------------------------------------------

/// Evidence that AdaClip is valid for Gaussian mechanisms.
///
/// By Theorem 1 of Andrew et al. (2021), the adaptive clipping Gaussian
/// mechanism is equivalent to a single Gaussian with effective noise
/// multiplier `z = 1/S` where `S = sqrt(1/z_Δ² + 1/(2σ_b)²)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GaussianAdaClipEvidence;

impl AdaClipEvidence<Gaussian> for GaussianAdaClipEvidence {
    fn compute_pld(
        &self,
        inner: &Gaussian,
        quantile_noise_std: f64,
    ) -> Result<PrivacyLossDistribution> {
        inner.with_effective_adaclip_nm(quantile_noise_std).pld()
    }
}

/// `Gaussian` supports adaptive clipping (Andrew et al. 2021, Theorem 1).
impl AdaClipable for Gaussian {
    type Evidence = GaussianAdaClipEvidence;

    fn noise_multiplier(&self) -> f64 {
        self.noise_multiplier
    }

    fn adaclip_evidence() -> GaussianAdaClipEvidence {
        GaussianAdaClipEvidence
    }

    fn with_effective_adaclip_nm(&self, quantile_noise_std: f64) -> Self {
        let s = combined_sensitivity(self.noise_multiplier, quantile_noise_std);
        Gaussian {
            noise_multiplier: 1.0 / s,
            ..self.clone()
        }
    }
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

/// Wrap a mechanism with adaptive clipping (Andrew et al. 2021).
///
/// Follows the same combinator pattern as [`poisson()`](crate::functional::amplification::poisson):
/// the inner mechanism must implement [`AdaClipable`], which proves the
/// combination is mathematically valid.
///
/// # Arguments
///
/// * `inner` - The base mechanism (must implement `AdaClipable`)
/// * `quantile_noise_std` - Noise std for quantile estimation (`σ_b`), must be positive
///
/// # Panics
///
/// Panics if `quantile_noise_std` is not positive (consistent with `poisson()`
/// which panics on invalid rate).
///
/// # Examples
///
/// ```rust,ignore
/// use opaque_dp_accounting::functional::*;
///
/// // Standard Gaussian
/// let step = adaclip(gaussian(1.1)?, 50.0);
///
/// // With Poisson amplification
/// let step = poisson(adaclip(gaussian(1.1)?, 50.0), 0.01);
///
/// // adaclip(identity()?, 50.0)  // ← compile error: Identity is not AdaClipable
/// ```
pub fn adaclip<P: AdaClipable>(inner: P, quantile_noise_std: f64) -> AdaClip<P, P::Evidence> {
    assert!(
        quantile_noise_std > 0.0,
        "quantile_noise_std must be positive, got {}",
        quantile_noise_std
    );
    AdaClip {
        inner,
        quantile_noise_std,
        evidence: P::adaclip_evidence(),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Combined normalized sensitivity δ̃ from Theorem 1 of Andrew et al. (2021).
///
/// ```text
/// δ̃ = sqrt(1/z_Δ² + 1/(2σ_b)²)
/// ```
///
/// The effective noise multiplier of the equivalent mechanism is `z = 1/δ̃`.
pub fn combined_sensitivity(noise_multiplier: f64, quantile_noise_std: f64) -> f64 {
    let inv_z_sq = 1.0 / (noise_multiplier * noise_multiplier);
    let inv_2sb_sq = 1.0 / (4.0 * quantile_noise_std * quantile_noise_std);
    (inv_z_sq + inv_2sb_sq).sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::functional::discretization::DiscretizationConfig;
    use crate::functional::mechanisms::gaussian::gaussian;

    #[test]
    fn test_adaclip_constructor() {
        let ac = adaclip(gaussian(1.1).unwrap(), 50.0);
        assert_eq!(ac.inner.noise_multiplier, 1.1);
        assert_eq!(ac.quantile_noise_std, 50.0);
    }

    #[test]
    fn test_adaclip_with_custom_config() {
        let config = DiscretizationConfig::new(0.01, -50.0).unwrap();
        let g = Gaussian::new_unchecked(1.1, config.clone());
        let ac = adaclip(g, 50.0);
        assert_eq!(ac.inner.config.discretization, config.discretization);
    }

    #[test]
    #[should_panic(expected = "quantile_noise_std must be positive")]
    fn test_adaclip_panics_zero_sb() {
        let g = gaussian(1.0).unwrap();
        adaclip(g, 0.0);
    }

    #[test]
    #[should_panic(expected = "quantile_noise_std must be positive")]
    fn test_adaclip_panics_negative_sb() {
        let g = gaussian(1.0).unwrap();
        adaclip(g, -1.0);
    }

    #[test]
    fn test_combined_sensitivity_formula() {
        // S = sqrt(1/z^2 + 1/(4*sigma_b^2))
        let ac = adaclip(gaussian(1.0).unwrap(), 50.0);
        let expected = (1.0_f64 + 1.0 / (4.0 * 50.0 * 50.0)).sqrt();
        assert!((ac.combined_sensitivity() - expected).abs() < 1e-15);
    }

    #[test]
    fn test_large_sigma_b_sensitivity_matches_base() {
        // As σ_b → ∞, δ̃ → 1/z_Δ and z_eff → z_Δ (mechanism reduces to base Gaussian)
        let ac = adaclip(gaussian(1.0).unwrap(), 1e10);
        let base_delta_tilde = 1.0 / 1.0;
        let ac_delta_tilde = ac.combined_sensitivity();
        assert!((ac_delta_tilde - base_delta_tilde).abs() < 1e-10);
        assert!((ac.effective_noise_multiplier() - 1.0).abs() < 1e-10);

        let ac2 = adaclip(gaussian(0.5).unwrap(), 1e10);
        let base_delta_tilde2 = 1.0 / 0.5;
        let ac2_delta_tilde = ac2.combined_sensitivity();
        assert!((ac2_delta_tilde - base_delta_tilde2).abs() < 1e-10);
        // z_eff = 1/δ̃ = z_Δ = 0.5 (not z_Δ² = 0.25)
        assert!((ac2.effective_noise_multiplier() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_effective_nm_smaller_than_nm() {
        let ac = adaclip(gaussian(1.0).unwrap(), 50.0);
        assert!(ac.effective_noise_multiplier() < ac.inner.noise_multiplier);
    }

    #[test]
    fn test_structural_equality() {
        let a = adaclip(gaussian(1.0).unwrap(), 50.0);
        let b = adaclip(gaussian(1.0).unwrap(), 50.0);
        assert_eq!(a, b);

        let c = adaclip(gaussian(1.0).unwrap(), 100.0);
        assert_ne!(a, c);
    }

    #[test]
    fn test_stores_original_inner() {
        // Verify that AdaClip stores the original, untransformed inner
        let g = gaussian(1.0).unwrap();
        let ac = adaclip(g.clone(), 50.0);
        assert_eq!(ac.inner.noise_multiplier, g.noise_multiplier);
        // Effective nm should differ from stored nm
        assert_ne!(ac.effective_noise_multiplier(), ac.inner.noise_multiplier);
    }
}
