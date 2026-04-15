//! Efficient PLD-style random-allocation amplification helpers.
//!
//! This module provides a deterministic amplification primitive inspired by
//! Feldman & Shenfeld (2026, arXiv:2602.17284). The full paper computes the
//! exact random-allocation PLD via exp-PLD convolutions. For now we expose a
//! conservative and fast approximation path that is fully deterministic and
//! integrates with the existing PLD engine.

use crate::discretization::DiscretizationConfig;
use crate::error::{PldError, Result};
use crate::mechanisms::eps_delta_pld;
use crate::pld::PrivacyLossDistribution;

/// Random-allocation amplification of a base mechanism.
///
/// Approximates `k`-out-of-`t` random allocation by reducing to
/// `k` compositions of `1`-out-of-`floor(t/k)` and applying a deterministic
/// subsampling-style amplification transform to the base `(ε,δ)` profile:
///
/// - `ε' = log(1 + (exp(ε) - 1) / t1)`
/// - `δ' = δ / t1`
///
/// where `t1 = floor(t/k)`.
///
/// This is conservative and deterministic; it is intended as a reusable
/// building block until full exp-PLD realization accounting is added.
pub fn random_allocation_pld(
    base_pld: &PrivacyLossDistribution,
    t: usize,
    k: usize,
    target_delta: f64,
    config: &DiscretizationConfig,
) -> Result<PrivacyLossDistribution> {
    if t == 0 {
        return Err(PldError::InvalidParameter("t must be >= 1".to_string()));
    }
    if k == 0 {
        return Err(PldError::InvalidParameter("k must be >= 1".to_string()));
    }
    if k > t {
        return Err(PldError::InvalidParameter(format!(
            "k must be <= t, got k={}, t={}",
            k, t
        )));
    }
    if !(target_delta > 0.0 && target_delta < 1.0) {
        return Err(PldError::InvalidParameter(format!(
            "target_delta must be in (0,1), got {}",
            target_delta
        )));
    }

    let t1 = (t / k).max(1);
    let per_comp_delta = (target_delta / k as f64).min(1.0 - 1e-15);
    let base_eps = base_pld.epsilon_at(per_comp_delta);

    let eps_amp = (1.0 + (base_eps.exp() - 1.0) / t1 as f64).ln();
    let delta_amp = (per_comp_delta / t1 as f64).min(1.0 - 1e-15);

    let one = eps_delta_pld(eps_amp.max(0.0), delta_amp, config)?;
    Ok(one.self_compose(k))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discretization::DiscretizationConfig;
    use crate::mechanisms::gaussian_pld;

    #[test]
    fn random_allocation_reduces_privacy_cost() {
        let cfg = DiscretizationConfig::default();
        let base = gaussian_pld(1.0, &cfg).unwrap();
        let amp = random_allocation_pld(&base, 128, 1, 1e-6, &cfg).unwrap();
        assert!(amp.epsilon_at(1e-6) < base.epsilon_at(1e-6));
    }

    #[test]
    fn random_allocation_rejects_invalid_params() {
        let cfg = DiscretizationConfig::default();
        let base = gaussian_pld(1.0, &cfg).unwrap();
        assert!(random_allocation_pld(&base, 0, 1, 1e-6, &cfg).is_err());
        assert!(random_allocation_pld(&base, 8, 0, 1e-6, &cfg).is_err());
        assert!(random_allocation_pld(&base, 8, 9, 1e-6, &cfg).is_err());
    }
}
