//! Process with a known (ε, δ) guarantee
//!
//! Useful for incorporating external privacy analyses or pre-computed budgets
//! into a composition pipeline.

use std::collections::BTreeMap;

use crate::error::{PldError, Result};
use crate::functional::discretization::DiscretizationConfig;
use crate::functional::pld::pmf::Pmf;
use crate::functional::pld::PrivacyLossDistribution;
use crate::functional::process::Process;

/// Process with a known (ε, δ) guarantee
///
/// Represents a mechanism whose privacy guarantee is already known analytically,
/// without needing PLD-based accounting. Useful for:
///
/// - Incorporating external privacy analyses into a composition pipeline
/// - Modeling pre-computed privacy budgets
/// - Testing and prototyping
///
/// The PLD is constructed with mass `(1 - δ)` at privacy loss `ε` and
/// infinity mass `δ`, which is the worst-case PLD consistent with (ε, δ)-DP.
///
/// # Examples
///
/// ```rust,ignore
/// use opaque_dp_accounting::functional::*;
///
/// // A mechanism known to be (1.0, 1e-5)-DP
/// let m = eps_delta(1.0, 1e-5);
/// assert!((m.epsilon_at(1e-5)? - 1.0).abs() < 1e-3);
///
/// // Compose with other processes
/// let pipeline = compose(gaussian(1.1), eps_delta(0.5, 1e-6));
/// ```
///
/// # References
///
/// - Dwork & Roth (2014). "The Algorithmic Foundations of Differential Privacy."
///   Section 3.5: Composition Theorems.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EpsDelta {
    /// Privacy loss (ε ≥ 0)
    pub epsilon: f64,
    /// Failure probability (δ ∈ [0, 1])
    pub delta: f64,
    /// Discretization configuration
    pub config: DiscretizationConfig,
}

impl Eq for EpsDelta {}

impl Process for EpsDelta {
    fn pld(&self) -> Result<PrivacyLossDistribution> {
        let disc = self.config.discretization;

        // Place mass (1 - delta) at the grid point closest to epsilon
        // and set infinity_mass = delta
        let index = (self.epsilon / disc).round() as i64;
        let mut masses = BTreeMap::new();
        masses.insert(index, 1.0 - self.delta);

        let pmf = Pmf::from_sparse(
            disc,
            masses,
            self.delta,
            true, // pessimistic: rounds up, conservative for composition
            self.config.max_grid_size,
        );
        Ok(PrivacyLossDistribution::new_symmetric(pmf))
    }
}

/// Create an (ε, δ) process with default discretization
///
/// # Arguments
///
/// * `epsilon` - Privacy loss (must be ≥ 0)
/// * `delta` - Failure probability (must be in [0, 1])
///
/// # Errors
///
/// Returns `PldError::InvalidParameter` if `epsilon < 0` or `delta` is not in [0, 1].
///
/// # Examples
///
/// ```rust,ignore
/// use opaque_dp_accounting::functional::*;
///
/// let m = eps_delta(1.0, 1e-5)?;
/// let pld = m.pld()?;
/// ```
pub fn eps_delta(epsilon: f64, delta: f64) -> Result<EpsDelta> {
    validate_eps_delta_params(epsilon, delta)?;
    Ok(EpsDelta {
        epsilon,
        delta,
        config: DiscretizationConfig::default(),
    })
}

/// Create an (ε, δ) process with custom discretization
///
/// # Errors
///
/// Returns `PldError::InvalidParameter` if `epsilon < 0` or `delta` is not in [0, 1].
pub fn eps_delta_with(epsilon: f64, delta: f64, config: DiscretizationConfig) -> Result<EpsDelta> {
    validate_eps_delta_params(epsilon, delta)?;
    Ok(EpsDelta {
        epsilon,
        delta,
        config,
    })
}

fn validate_eps_delta_params(epsilon: f64, delta: f64) -> Result<()> {
    if epsilon < 0.0 {
        return Err(PldError::InvalidParameter(format!(
            "epsilon must be non-negative, got {}",
            epsilon
        )));
    }
    if !(0.0..=1.0).contains(&delta) {
        return Err(PldError::InvalidParameter(format!(
            "delta must be in [0, 1], got {}",
            delta
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eps_delta_constructor() {
        let m = eps_delta(1.0, 1e-5).unwrap();
        assert_eq!(m.epsilon, 1.0);
        assert_eq!(m.delta, 1e-5);
    }

    #[test]
    fn test_eps_delta_rejects_negative_epsilon() {
        assert!(eps_delta(-0.1, 1e-5).is_err());
    }

    #[test]
    fn test_eps_delta_rejects_delta_above_one() {
        assert!(eps_delta(1.0, 1.1).is_err());
    }

    #[test]
    fn test_eps_delta_rejects_negative_delta() {
        assert!(eps_delta(1.0, -0.1).is_err());
    }

    #[test]
    fn test_eps_delta_epsilon_at() {
        let m = eps_delta(1.0, 1e-5).unwrap();
        let pld = m.pld().unwrap();
        let eps = pld.epsilon_at(1e-5);
        assert!(
            (eps - 1.0).abs() < 0.001,
            "epsilon_at(1e-5) = {}, expected ~1.0",
            eps
        );
    }

    #[test]
    fn test_eps_delta_delta_at() {
        let m = eps_delta(1.0, 1e-5).unwrap();
        let pld = m.pld().unwrap();
        let delta = pld.delta_at(1.0);
        assert!(
            (delta - 1e-5).abs() < 1e-4,
            "delta_at(1.0) = {:.2e}, expected ~1e-5",
            delta
        );
    }

    #[test]
    fn test_eps_delta_pure_dp() {
        let m = eps_delta(1.0, 0.0).unwrap();
        let pld = m.pld().unwrap();
        assert_eq!(pld.delta_at(1.0), 0.0);
    }

    #[test]
    fn test_eps_delta_structural_eq() {
        assert_eq!(eps_delta(1.0, 1e-5).unwrap(), eps_delta(1.0, 1e-5).unwrap());
        assert_ne!(eps_delta(1.0, 1e-5).unwrap(), eps_delta(2.0, 1e-5).unwrap());
        assert_ne!(eps_delta(1.0, 1e-5).unwrap(), eps_delta(1.0, 1e-6).unwrap());
    }

    #[test]
    fn test_eps_delta_zero_is_identity() {
        let m = eps_delta(0.0, 0.0).unwrap();
        let pld = m.pld().unwrap();
        assert_eq!(pld.epsilon_at(1e-5), 0.0);
        assert_eq!(pld.delta_at(0.0), 0.0);
    }
}
