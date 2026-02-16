//! Identity process (zero privacy loss)
//!
//! The neutral element for composition: composing any process P with
//! Identity yields the same privacy guarantee as P alone.

use std::collections::BTreeMap;

use crate::error::Result;
use crate::functional::discretization::DiscretizationConfig;
use crate::functional::pld::pmf::Pmf;
use crate::functional::pld::PrivacyLossDistribution;
use crate::functional::process::Process;

/// Identity process (zero privacy loss)
///
/// Represents perfect privacy — no information about the dataset is leaked.
/// This is the neutral element for composition: composing any process P with
/// Identity yields the same privacy guarantee as P alone.
///
/// # Examples
///
/// ```rust,ignore
/// use opaque_dp_accounting::functional::*;
///
/// let id = identity();
/// assert_eq!(id.epsilon_at(1e-5)?, 0.0);
/// assert_eq!(id.delta_at(0.0)?, 0.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Identity {
    /// Discretization configuration (determines grid spacing for composition)
    pub config: DiscretizationConfig,
}

impl Eq for Identity {}

impl Process for Identity {
    fn pld(&self) -> Result<PrivacyLossDistribution> {
        let mut masses = BTreeMap::new();
        masses.insert(0, 1.0);
        let pmf = Pmf::from_sparse(
            self.config.discretization,
            masses,
            0.0,
            true,
            self.config.max_grid_size,
        );
        Ok(PrivacyLossDistribution::new_symmetric(pmf))
    }
}

/// Create an identity process with default discretization
///
/// # Examples
///
/// ```rust,ignore
/// use opaque_dp_accounting::functional::*;
///
/// let process = compose(gaussian(1.1), identity());
/// // Same privacy as gaussian(1.1) alone
/// ```
pub fn identity() -> Identity {
    Identity {
        config: DiscretizationConfig::default(),
    }
}

/// Create an identity process with custom discretization
pub fn identity_with(config: DiscretizationConfig) -> Identity {
    Identity { config }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_epsilon_is_zero() {
        let id = identity();
        let pld = id.pld().unwrap();
        for &delta in &[1e-10, 1e-5, 0.1, 0.5] {
            assert_eq!(pld.epsilon_at(delta), 0.0, "delta={}", delta);
        }
    }

    #[test]
    fn test_identity_delta_is_zero() {
        let id = identity();
        let pld = id.pld().unwrap();
        for &eps in &[0.0, 0.1, 1.0, 10.0] {
            assert_eq!(pld.delta_at(eps), 0.0, "eps={}", eps);
        }
    }

    #[test]
    fn test_identity_advantage_is_zero() {
        let id = identity();
        assert_eq!(id.advantage().unwrap(), 0.0);
    }

    #[test]
    fn test_identity_structural_eq() {
        assert_eq!(identity(), identity());
    }
}
