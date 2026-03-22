//! High-level Privacy Loss Distribution with adjacency support.
//!
//! This module provides `PrivacyLossDistribution`, an enum with two
//! representations:
//!
//! - **`Pmf`**: Discretized probability mass function on an ε-grid.
//!   Composition via FFT convolution. Exact up to discretization.
//!
//! - **`Spa`**: Saddle-Point Accountant backed by opaque CGF (Cumulant
//!   Generating Function) handles. No grid — CGFs are evaluated only at
//!   query time. Composition is trivial function addition.
//!
//! Both representations are **mechanism-agnostic**: the PLD never knows
//! which mechanism produced it.

pub mod cgf;
pub(crate) mod metrics;
pub mod pmf;
pub(crate) mod pmf_pld;
pub(crate) mod spa_pld;

pub use cgf::Cgf;
pub use pmf::Pmf;
pub use pmf_pld::PmfPld;
pub use spa_pld::SpaPld;

use std::sync::Arc;

use crate::discretization::DiscretizationConfig;
use crate::error::Result;

/// High-level privacy loss distribution.
///
/// An enum over two representations:
///
/// - **`Pmf`**: Discretized PMF + FFT composition. Created by mechanism
///   constructors like `gaussian_pld()`, `poisson_gaussian_pld()`, etc.
///
/// - **`Spa`**: Saddle-Point Accountant backed by CGFs. Created by
///   SPA constructors like `spa_gaussian_pld()`. Composition is O(1).
///
/// All privacy metrics (`delta_at`, `epsilon_at`, `advantage`, `beta_at`,
/// `risk_at`) dispatch to the appropriate representation. For metrics
/// that require the full PMF (beta, risk), the SPA variant auto-materializes.
///
/// # Examples
///
/// ```rust,ignore
/// use opaque_accounting::mechanisms::gaussian_pld;
///
/// let pld = gaussian_pld(1.1, &config)?;
/// let delta = pld.delta_at(1.0);
/// let epsilon = pld.epsilon_at(1e-5);
/// ```
#[derive(Debug, Clone)]
pub enum PrivacyLossDistribution {
    /// Discretized PMF representation (current, exact up to discretization).
    Pmf(PmfPld),
    /// Saddle-Point Accountant representation (analytical, approximate).
    Spa(SpaPld),
}

impl PrivacyLossDistribution {
    // -- Constructors -------------------------------------------------------

    /// Create a PLD from a symmetric PMF (same for ADD and REMOVE adjacencies).
    pub(crate) fn new_symmetric(pmf: Pmf) -> Self {
        Self::Pmf(PmfPld::new_symmetric(pmf))
    }

    /// Create a PLD from asymmetric PMFs (different for ADD and REMOVE).
    pub(crate) fn new_asymmetric(pmf_remove: Pmf, pmf_add: Pmf) -> Self {
        Self::Pmf(PmfPld::new_asymmetric(pmf_remove, pmf_add))
    }

    /// Create a PLD from a single CGF (Saddle-Point Accountant).
    pub fn new_spa(cgf: Arc<dyn Cgf>) -> Self {
        Self::Spa(SpaPld::new(cgf))
    }

    // -- Properties ----------------------------------------------------------

    /// Check if this PLD is symmetric (same for both adjacency types).
    pub fn is_symmetric(&self) -> bool {
        match self {
            Self::Pmf(p) => p.is_symmetric(),
            Self::Spa(_) => true, // SPA computes worst-case directly
        }
    }

    /// Set Chernoff tail budgets on all contained PMFs.
    /// No-op for the SPA variant.
    pub fn with_tail_budgets(self, right: f64, left: f64) -> Self {
        match self {
            Self::Pmf(p) => Self::Pmf(p.with_tail_budgets(right, left)),
            Self::Spa(_) => self,
        }
    }

    /// Override the max grid size on all contained PMFs.
    /// No-op for the SPA variant.
    pub fn with_max_grid_size(&self, max_grid_size: usize) -> Self {
        match self {
            Self::Pmf(p) => Self::Pmf(p.with_max_grid_size(max_grid_size)),
            Self::Spa(s) => Self::Spa(s.clone()),
        }
    }

    // -- Privacy metrics (dispatch) ------------------------------------------

    /// Smallest δ achieving (ε, δ)-DP.
    pub fn delta_at(&self, epsilon: f64) -> f64 {
        match self {
            Self::Pmf(p) => metrics::delta(p, epsilon).clamp(0.0, 1.0),
            Self::Spa(s) => s.delta_at(epsilon).clamp(0.0, 1.0),
        }
    }

    /// Smallest ε achieving (ε, δ)-DP.
    pub fn epsilon_at(&self, delta: f64) -> f64 {
        match self {
            Self::Pmf(p) => metrics::epsilon(p, delta),
            Self::Spa(s) => s.epsilon_at(delta),
        }
    }

    /// Total-variation advantage (f-DP). 0 = perfect privacy, 1 = none.
    pub fn advantage(&self) -> f64 {
        match self {
            Self::Pmf(p) => metrics::advantage(p).clamp(0.0, 1.0),
            Self::Spa(s) => s.advantage().clamp(0.0, 1.0),
        }
    }

    /// Type-II error β at given Type-I error α.
    ///
    /// For the SPA variant, auto-materializes to PMF first.
    pub fn beta_at(&self, target_alpha: f64) -> f64 {
        match self {
            Self::Pmf(p) => metrics::beta(p, target_alpha).clamp(0.0, 1.0),
            Self::Spa(s) => {
                let pmf = s
                    .to_pmf_pld(&DiscretizationConfig::default())
                    .expect("SPA materialization failed");
                metrics::beta(&pmf, target_alpha).clamp(0.0, 1.0)
            }
        }
    }

    /// Bayes risk under optimal adversary.
    ///
    /// For the SPA variant, auto-materializes to PMF first.
    pub fn risk_at(&self, prior: f64) -> f64 {
        let max_risk = prior.min(1.0 - prior);
        match self {
            Self::Pmf(p) => metrics::bayes_risk(p, prior).clamp(0.0, max_risk),
            Self::Spa(s) => {
                let pmf = s
                    .to_pmf_pld(&DiscretizationConfig::default())
                    .expect("SPA materialization failed");
                metrics::bayes_risk(&pmf, prior).clamp(0.0, max_risk)
            }
        }
    }

    // -- Composition (dispatch) ----------------------------------------------

    /// Compose two PLDs.
    ///
    /// - Pmf + Pmf → Pmf (FFT convolution)
    /// - Spa + Spa → Spa (concatenate component lists)
    /// - Mixed → materialize Spa to Pmf, then FFT
    pub fn compose(&self, other: &Self) -> Result<Self> {
        match (self, other) {
            (Self::Pmf(a), Self::Pmf(b)) => Ok(Self::Pmf(a.compose(b)?)),
            (Self::Spa(a), Self::Spa(b)) => Ok(Self::Spa(a.compose(b))),
            (Self::Spa(s), Self::Pmf(p)) => {
                let materialized = s.to_pmf_pld(&DiscretizationConfig::default())?;
                Ok(Self::Pmf(materialized.compose(p)?))
            }
            (Self::Pmf(p), Self::Spa(s)) => {
                let materialized = s.to_pmf_pld(&DiscretizationConfig::default())?;
                Ok(Self::Pmf(p.compose(&materialized)?))
            }
        }
    }

    /// Self-compose this PLD `count` times.
    ///
    /// - Pmf: FFT power method O(N log N)
    /// - Spa: multiply counts O(k) — effectively free
    pub fn self_compose(&self, count: usize) -> Self {
        match self {
            Self::Pmf(p) => Self::Pmf(p.self_compose(count)),
            Self::Spa(s) => Self::Spa(s.self_compose(count)),
        }
    }

    /// Compose with explicit `max_grid_size` override.
    pub fn compose_with_max_grid_size(&self, other: &Self, max_grid_size: usize) -> Result<Self> {
        match (self, other) {
            (Self::Pmf(a), Self::Pmf(b)) => {
                Ok(Self::Pmf(a.compose_with_max_grid_size(b, max_grid_size)?))
            }
            // For SPA variants, grid size is irrelevant — delegate to compose()
            _ => self.compose(other),
        }
    }

    /// Self-compose with explicit `max_grid_size` override.
    pub fn self_compose_with_max_grid_size(&self, count: usize, max_grid_size: usize) -> Self {
        match self {
            Self::Pmf(p) => Self::Pmf(p.self_compose_with_max_grid_size(count, max_grid_size)),
            Self::Spa(s) => Self::Spa(s.self_compose(count)),
        }
    }
}

// Note: In the functional API, PrivacyLossDistribution is a pure data structure.
// It does not implement the old Mechanism trait, which is specific to the legacy API.

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn create_test_pmf(offset: i64, prob1: f64, prob2: f64) -> Pmf {
        let mut masses = BTreeMap::new();
        masses.insert(offset, prob1);
        masses.insert(offset + 5, prob2);
        Pmf::from_sparse(0.1, masses, 0.0, true, usize::MAX)
    }

    #[test]
    fn test_symmetric_pld_creation() {
        let pmf = create_test_pmf(0, 0.5, 0.5);
        let pld = PrivacyLossDistribution::new_symmetric(pmf);

        assert!(pld.is_symmetric());
    }

    #[test]
    fn test_asymmetric_pld_creation() {
        let pmf_remove = create_test_pmf(0, 0.3, 0.7);
        let pmf_add = create_test_pmf(0, 0.4, 0.6);
        let pld = PrivacyLossDistribution::new_asymmetric(pmf_remove, pmf_add);

        assert!(!pld.is_symmetric());
    }

    #[test]
    fn test_symmetric_same_pmf_both_adjacencies() {
        let pmf = create_test_pmf(0, 0.5, 0.5);
        let pld = PrivacyLossDistribution::new_symmetric(pmf);

        // For symmetric PLD, pmf_add should be None (pmf_remove used for both)
        assert!(pld.is_symmetric());

        // Querying delta should give valid result
        let epsilon = 1.0;
        let delta = pld.delta_at(epsilon);
        assert!(delta >= 0.0 && delta <= 1.0);
    }

    #[test]
    fn test_asymmetric_has_both_pmfs() {
        let pmf_remove = create_test_pmf(0, 0.3, 0.7);
        let pmf_add = create_test_pmf(0, 0.4, 0.6);
        let pld = PrivacyLossDistribution::new_asymmetric(pmf_remove, pmf_add);

        // Asymmetric PLD should not be symmetric
        assert!(!pld.is_symmetric());

        // Verify it produces valid deltas
        let epsilon = 1.0;
        let delta = pld.delta_at(epsilon);
        assert!(delta >= 0.0 && delta <= 1.0);
    }

    #[test]
    fn test_symmetric_delta_query() {
        let pmf = create_test_pmf(0, 0.3, 0.7);
        let pld = PrivacyLossDistribution::new_symmetric(pmf.clone());
        let pld_pmf = PrivacyLossDistribution::new_symmetric(pmf);

        let epsilon = 1.0;
        let delta_pld = pld.delta_at(epsilon);
        let delta_pmf = pld_pmf.delta_at(epsilon);

        // For symmetric PLD, result should match underlying PMF
        assert!((delta_pld - delta_pmf).abs() < 1e-10);
    }

    #[test]
    fn test_asymmetric_delta_query_takes_max() {
        let pmf_remove = create_test_pmf(0, 0.2, 0.8);
        let pmf_add = create_test_pmf(0, 0.8, 0.2);
        let pld = PrivacyLossDistribution::new_asymmetric(pmf_remove.clone(), pmf_add.clone());
        let pld_remove = PrivacyLossDistribution::new_symmetric(pmf_remove);
        let pld_add = PrivacyLossDistribution::new_symmetric(pmf_add);

        let epsilon = 0.5;
        let delta_pld = pld.delta_at(epsilon);
        let delta_remove = pld_remove.delta_at(epsilon);
        let delta_add = pld_add.delta_at(epsilon);

        // PLD should return the maximum
        let expected_max = delta_remove.max(delta_add);
        assert!((delta_pld - expected_max).abs() < 1e-10);
    }

    #[test]
    fn test_symmetric_epsilon_query() {
        let pmf = create_test_pmf(0, 0.3, 0.7);
        let pld = PrivacyLossDistribution::new_symmetric(pmf.clone());
        let pld_pmf = PrivacyLossDistribution::new_symmetric(pmf);

        let delta = 0.1;
        let epsilon_pld = pld.epsilon_at(delta);
        let epsilon_pmf = pld_pmf.epsilon_at(delta);

        // For symmetric PLD, result should match underlying PMF
        assert!((epsilon_pld - epsilon_pmf).abs() < 1e-10);
    }

    #[test]
    fn test_asymmetric_epsilon_query_takes_max() {
        let pmf_remove = create_test_pmf(0, 0.2, 0.8);
        let pmf_add = create_test_pmf(0, 0.8, 0.2);
        let pld = PrivacyLossDistribution::new_asymmetric(pmf_remove.clone(), pmf_add.clone());
        let pld_remove = PrivacyLossDistribution::new_symmetric(pmf_remove);
        let pld_add = PrivacyLossDistribution::new_symmetric(pmf_add);

        let delta = 0.1;
        let epsilon_pld = pld.epsilon_at(delta);
        let epsilon_remove = pld_remove.epsilon_at(delta);
        let epsilon_add = pld_add.epsilon_at(delta);

        // PLD should return the maximum
        let expected_max = epsilon_remove.max(epsilon_add);
        assert!((epsilon_pld - expected_max).abs() < 1e-10);
    }

    #[test]
    fn test_compose_symmetric_symmetric() {
        let pmf1 = create_test_pmf(0, 0.5, 0.5);
        let pmf2 = create_test_pmf(0, 0.6, 0.4);

        let pld1 = PrivacyLossDistribution::new_symmetric(pmf1);
        let pld2 = PrivacyLossDistribution::new_symmetric(pmf2);

        let composed = pld1.compose(&pld2).unwrap();

        // Symmetric + Symmetric = Symmetric
        assert!(composed.is_symmetric());
    }

    #[test]
    fn test_compose_symmetric_asymmetric() {
        let pmf_sym = create_test_pmf(0, 0.5, 0.5);
        let pmf_remove = create_test_pmf(0, 0.3, 0.7);
        let pmf_add = create_test_pmf(0, 0.4, 0.6);

        let pld_sym = PrivacyLossDistribution::new_symmetric(pmf_sym);
        let pld_asym = PrivacyLossDistribution::new_asymmetric(pmf_remove, pmf_add);

        let composed = pld_sym.compose(&pld_asym).unwrap();

        // Symmetric + Asymmetric = Asymmetric
        assert!(!composed.is_symmetric());
    }

    #[test]
    fn test_compose_asymmetric_asymmetric() {
        let pmf_remove1 = create_test_pmf(0, 0.2, 0.8);
        let pmf_add1 = create_test_pmf(0, 0.3, 0.7);
        let pmf_remove2 = create_test_pmf(0, 0.4, 0.6);
        let pmf_add2 = create_test_pmf(0, 0.5, 0.5);

        let pld1 = PrivacyLossDistribution::new_asymmetric(pmf_remove1, pmf_add1);
        let pld2 = PrivacyLossDistribution::new_asymmetric(pmf_remove2, pmf_add2);

        let composed = pld1.compose(&pld2).unwrap();

        // Asymmetric + Asymmetric = Asymmetric
        assert!(!composed.is_symmetric());
    }

    #[test]
    fn test_self_compose_symmetric() {
        let pmf = create_test_pmf(0, 0.3, 0.7);
        let pld = PrivacyLossDistribution::new_symmetric(pmf);

        let composed = pld.self_compose(5);

        // Self-compose preserves symmetry
        assert!(composed.is_symmetric());
    }

    #[test]
    fn test_self_compose_asymmetric() {
        let pmf_remove = create_test_pmf(0, 0.2, 0.8);
        let pmf_add = create_test_pmf(0, 0.3, 0.7);
        let pld = PrivacyLossDistribution::new_asymmetric(pmf_remove, pmf_add);

        let composed = pld.self_compose(3);

        // Self-compose preserves asymmetry
        assert!(!composed.is_symmetric());
    }

    #[test]
    fn test_self_compose_count_one() {
        let pmf = create_test_pmf(0, 0.5, 0.5);
        let pld = PrivacyLossDistribution::new_symmetric(pmf);

        let composed = pld.self_compose(1);

        // Self-compose with count=1 should be equivalent
        let epsilon = 1.0;
        let delta_original = pld.delta_at(epsilon);
        let delta_composed = composed.delta_at(epsilon);

        assert!((delta_original - delta_composed).abs() < 1e-10);
    }

    #[test]
    fn test_compose_commutativity() {
        let pmf1 = create_test_pmf(0, 0.3, 0.7);
        let pmf2 = create_test_pmf(0, 0.4, 0.6);

        let pld1 = PrivacyLossDistribution::new_symmetric(pmf1);
        let pld2 = PrivacyLossDistribution::new_symmetric(pmf2);

        let composed_12 = pld1.compose(&pld2).unwrap();
        let composed_21 = pld2.compose(&pld1).unwrap();

        // Composition should be commutative
        let epsilon = 1.0;
        let delta_12 = composed_12.delta_at(epsilon);
        let delta_21 = composed_21.delta_at(epsilon);

        assert!((delta_12 - delta_21).abs() < 1e-10);
    }

    #[test]
    fn test_bayes_risk_symmetric() {
        let pmf = create_test_pmf(0, 0.3, 0.7);
        let pld = PrivacyLossDistribution::new_symmetric(pmf);

        // Bayes risk with uniform prior
        let risk = pld.risk_at(0.5);

        // Should be in valid range [0, 0.5]
        assert!(risk >= 0.0);
        assert!(risk <= 0.5);

        // Higher privacy loss means lower Bayes risk (worse privacy)
        // With significant privacy loss, risk should be noticeably below 0.5
        assert!(risk < 0.5);
    }

    #[test]
    fn test_bayes_risk_edge_cases() {
        let pmf = create_test_pmf(0, 0.5, 0.5);
        let pld = PrivacyLossDistribution::new_symmetric(pmf);

        // Prior = 0: attacker knows with certainty, risk = 0
        let risk_0 = pld.risk_at(0.0);
        assert!((risk_0 - 0.0).abs() < 1e-10);

        // Prior = 1: attacker knows with certainty, risk = 0
        let risk_1 = pld.risk_at(1.0);
        assert!((risk_1 - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_bayes_risk_symmetry_in_prior() {
        let pmf = create_test_pmf(0, 0.4, 0.6);
        let pld = PrivacyLossDistribution::new_symmetric(pmf);

        // For symmetric PLDs, Bayes risk should be symmetric around prior = 0.5
        let risk_03 = pld.risk_at(0.3);
        let risk_07 = pld.risk_at(0.7);

        // Should be equal (within tolerance)
        assert!((risk_03 - risk_07).abs() < 1e-6);
    }

    #[test]
    fn test_bayes_risk_monotonicity() {
        // As privacy improves (lower epsilon), Bayes risk should increase toward 0.5
        let pmf_low_privacy = create_test_pmf(0, 0.1, 0.9); // High privacy loss
        let pmf_high_privacy = create_test_pmf(0, 0.5, 0.5); // Lower privacy loss

        let pld_low = PrivacyLossDistribution::new_symmetric(pmf_low_privacy);
        let pld_high = PrivacyLossDistribution::new_symmetric(pmf_high_privacy);

        let risk_low = pld_low.risk_at(0.5);
        let risk_high = pld_high.risk_at(0.5);

        // Lower privacy (higher epsilon) means lower Bayes risk
        assert!(risk_low < risk_high);
    }

    // -----------------------------------------------------------------
    // Identity PLD (all mass at loss=0)
    // -----------------------------------------------------------------

    fn make_identity_pld() -> PrivacyLossDistribution {
        let mut masses = BTreeMap::new();
        masses.insert(0, 1.0);
        let pmf = Pmf::from_sparse(1e-4, masses, 0.0, true, usize::MAX);
        PrivacyLossDistribution::new_symmetric(pmf)
    }

    #[test]
    fn test_identity_epsilon_is_zero() {
        let pld = make_identity_pld();
        for &delta in &[1e-10, 1e-5, 0.1, 0.5] {
            assert_eq!(pld.epsilon_at(delta), 0.0, "delta={}", delta);
        }
    }

    #[test]
    fn test_identity_delta_is_zero() {
        let pld = make_identity_pld();
        for &eps in &[0.0, 0.1, 1.0, 10.0] {
            assert_eq!(pld.delta_at(eps), 0.0, "eps={}", eps);
        }
    }

    #[test]
    fn test_identity_advantage_is_zero() {
        let pld = make_identity_pld();
        assert_eq!(pld.advantage(), 0.0);
    }

    #[test]
    fn test_identity_beta_is_one_minus_alpha() {
        let pld = make_identity_pld();
        for &alpha in &[0.0, 0.01, 0.1, 0.5, 0.9, 1.0] {
            let beta = pld.beta_at(alpha);
            assert!(
                (beta - (1.0 - alpha)).abs() < 1e-9,
                "alpha={}, beta={}, expected={}",
                alpha,
                beta,
                1.0 - alpha
            );
        }
    }

    #[test]
    fn test_identity_risk_is_min_prior() {
        let pld = make_identity_pld();
        for &prior in &[0.1, 0.3, 0.5] {
            let risk = pld.risk_at(prior);
            let expected = prior.min(1.0 - prior);
            assert!(
                (risk - expected).abs() < 1e-6,
                "prior={}, risk={}, expected={}",
                prior,
                risk,
                expected
            );
        }
    }

    #[test]
    fn test_identity_is_symmetric() {
        let pld = make_identity_pld();
        assert!(pld.is_symmetric());
    }

    // -----------------------------------------------------------------
    // Advantage
    // -----------------------------------------------------------------

    #[test]
    fn test_advantage_equals_delta_at_zero() {
        // advantage() should equal delta_at(0.0) by definition
        let pmf = create_test_pmf(0, 0.3, 0.7);
        let pld = PrivacyLossDistribution::new_symmetric(pmf);
        assert!((pld.advantage() - pld.delta_at(0.0)).abs() < 1e-10);
    }

    #[test]
    fn test_advantage_asymmetric_takes_max() {
        let pmf_remove = create_test_pmf(0, 0.2, 0.8);
        let pmf_add = create_test_pmf(0, 0.8, 0.2);
        let pld = PrivacyLossDistribution::new_asymmetric(pmf_remove.clone(), pmf_add.clone());

        let adv_remove = PrivacyLossDistribution::new_symmetric(pmf_remove).advantage();
        let adv_add = PrivacyLossDistribution::new_symmetric(pmf_add).advantage();

        assert!((pld.advantage() - adv_remove.max(adv_add)).abs() < 1e-10);
    }

    #[test]
    fn test_advantage_in_valid_range() {
        let pmf = create_test_pmf(0, 0.3, 0.7);
        let pld = PrivacyLossDistribution::new_symmetric(pmf);
        let adv = pld.advantage();
        assert!(adv >= 0.0 && adv <= 1.0, "advantage={}", adv);
    }

    // -----------------------------------------------------------------
    // Beta (trade-off function)
    // -----------------------------------------------------------------

    #[test]
    fn test_beta_at_boundary_alpha_zero() {
        let pmf = create_test_pmf(0, 0.3, 0.7);
        let pld = PrivacyLossDistribution::new_symmetric(pmf);
        // At alpha=0 (no false positives), beta should be 1-infinity_mass
        // For our test PMF with infinity_mass=0, beta(0) = 1.0
        let beta = pld.beta_at(0.0);
        assert!((beta - 1.0).abs() < 1e-9, "beta(0)={}", beta);
    }

    #[test]
    fn test_beta_at_boundary_alpha_one() {
        let pmf = create_test_pmf(0, 0.3, 0.7);
        let pld = PrivacyLossDistribution::new_symmetric(pmf);
        // At alpha=1, beta should be 0
        let beta = pld.beta_at(1.0);
        assert!((beta - 0.0).abs() < 1e-9, "beta(1)={}", beta);
    }

    #[test]
    fn test_beta_decreases_with_alpha() {
        // Trade-off function: as false positive rate increases,
        // false negative rate should decrease (or stay equal)
        let pmf = create_test_pmf(0, 0.3, 0.7);
        let pld = PrivacyLossDistribution::new_symmetric(pmf);

        let alphas = [0.0, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0];
        let mut prev_beta = f64::INFINITY;
        for &alpha in &alphas {
            let beta = pld.beta_at(alpha);
            assert!(
                beta <= prev_beta + 1e-10,
                "beta should decrease: alpha={}, beta={}, prev={}",
                alpha,
                beta,
                prev_beta
            );
            prev_beta = beta;
        }
    }

    // -----------------------------------------------------------------
    // Composition increases privacy cost
    // -----------------------------------------------------------------

    #[test]
    fn test_composition_increases_delta() {
        let pmf = create_test_pmf(0, 0.3, 0.7);
        let pld = PrivacyLossDistribution::new_symmetric(pmf);

        let epsilon = 0.5;
        let delta_1 = pld.delta_at(epsilon);
        let delta_2 = pld.compose(&pld).unwrap().delta_at(epsilon);
        let delta_5 = pld.self_compose(5).delta_at(epsilon);

        assert!(
            delta_2 >= delta_1,
            "delta_2={} < delta_1={}",
            delta_2,
            delta_1
        );
        assert!(
            delta_5 >= delta_2,
            "delta_5={} < delta_2={}",
            delta_5,
            delta_2
        );
    }

    #[test]
    fn test_composition_decreases_beta() {
        // More compositions = weaker privacy = lower beta for same alpha
        let pmf = create_test_pmf(0, 0.3, 0.7);
        let pld = PrivacyLossDistribution::new_symmetric(pmf);

        let alpha = 0.1;
        let beta_1 = pld.beta_at(alpha);
        let beta_5 = pld.self_compose(5).beta_at(alpha);

        assert!(beta_5 <= beta_1, "beta_5={} > beta_1={}", beta_5, beta_1);
    }
}
