//! Identity (zero privacy loss) mechanism PLD constructor.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::discretization::DiscretizationConfig;
use crate::error::Result;
use crate::pld::cgf::IdentityCgf;
use crate::pld::pmf::Pmf;
use crate::pld::PrivacyLossDistribution;

/// Compute the PLD for the identity (zero privacy loss) mechanism.
///
/// This is the neutral element for composition: composing any PLD with
/// the identity PLD yields the same privacy guarantee.
pub fn identity_pld(config: &DiscretizationConfig) -> Result<PrivacyLossDistribution> {
    let mut masses = BTreeMap::new();
    masses.insert(0, 1.0);

    let pmf = Pmf::from_sparse(
        config.discretization,
        masses,
        0.0,
        true,
        config.max_grid_size,
    );
    Ok(PrivacyLossDistribution::new_symmetric(pmf))
}

/// Create a CGF-backed PLD for the identity mechanism.
///
/// Returns a trivial CGF with Λ(t) = 0 for all t.
pub fn cgf_identity_pld() -> PrivacyLossDistribution {
    PrivacyLossDistribution::new_cgf(Arc::new(IdentityCgf))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> DiscretizationConfig {
        DiscretizationConfig::default()
    }

    #[test]
    fn test_identity_epsilon_zero() {
        let pld = identity_pld(&default_config()).unwrap();
        for &delta in &[1e-10, 1e-5, 0.1, 0.5] {
            assert_eq!(pld.epsilon_at(delta), 0.0, "delta={}", delta);
        }
    }

    #[test]
    fn test_identity_delta_zero() {
        let pld = identity_pld(&default_config()).unwrap();
        for &eps in &[0.0, 0.1, 1.0, 10.0] {
            assert_eq!(pld.delta_at(eps), 0.0, "eps={}", eps);
        }
    }

    #[test]
    fn test_identity_advantage_zero() {
        let pld = identity_pld(&default_config()).unwrap();
        assert_eq!(pld.advantage(), 0.0);
    }
}
