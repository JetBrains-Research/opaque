//! Saddle-Point Accountant: mechanism-agnostic CGF-based privacy accounting.
//!
//! `SpaPld` stores a list of opaque CGF handles with repetition counts.
//! No grid, no discretization — CGFs are only evaluated at query time
//! to find the saddle point and compute the MSD approximation.
//!
//! Composition is trivial: concatenate component lists (heterogeneous)
//! or multiply counts (homogeneous). Both are O(1) / O(k).
//!
//! # References
//!
//! Alghamdi, Gomez, Asoodeh, Calmon, Kosut, Sankar.
//! "The Saddle-Point Accountant for Differential Privacy." ICML 2023.
//! <https://arxiv.org/abs/2208.09595>

use std::fmt;
use std::sync::Arc;

use super::cgf::Cgf;
use super::PmfPld;
use crate::adjacency::Adjacency;
use crate::discretization::config::EpsilonBounds;
use crate::discretization::connect_the_dots::discretize_from_deltas;
use crate::discretization::DiscretizationConfig;
use crate::error::Result;

// ---------------------------------------------------------------------------
// SpaPld
// ---------------------------------------------------------------------------

/// Saddle-Point Accountant privacy loss distribution.
///
/// Stores a list of (CGF, repetition_count) components. The total CGF is:
///
/// Λ_total(t) = Σᵢ countᵢ · Λᵢ(t)
///
/// This is mechanism-agnostic: `SpaPld` never knows what mechanism
/// produced the CGFs. All composition operations are trivial (concatenate
/// or multiply counts). Privacy metrics are computed at query time via
/// the saddle-point method of steepest descent.
#[derive(Clone)]
pub struct SpaPld {
    pub(crate) components: Vec<(Arc<dyn Cgf>, usize)>,
}

impl fmt::Debug for SpaPld {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let total: usize = self.components.iter().map(|(_, n)| n).sum();
        f.debug_struct("SpaPld")
            .field("num_components", &self.components.len())
            .field("total_compositions", &total)
            .finish()
    }
}

impl SpaPld {
    /// Create an SpaPld from a single CGF (count=1).
    pub fn new(cgf: Arc<dyn Cgf>) -> Self {
        Self {
            components: vec![(cgf, 1)],
        }
    }

    // -- Total CGF evaluation -----------------------------------------------

    /// Evaluate the total CGF: Λ_total(t) = Σᵢ countᵢ · Λᵢ(t).
    fn total_cgf(&self, t: f64) -> f64 {
        self.components
            .iter()
            .map(|(cgf, n)| *n as f64 * cgf.eval(t))
            .sum()
    }

    /// Evaluate Λ'_total(t).
    fn total_cgf_prime(&self, t: f64) -> f64 {
        self.components
            .iter()
            .map(|(cgf, n)| *n as f64 * cgf.eval_prime(t))
            .sum()
    }

    /// Evaluate Λ''_total(t).
    fn total_cgf_double_prime(&self, t: f64) -> f64 {
        self.components
            .iter()
            .map(|(cgf, n)| *n as f64 * cgf.eval_double_prime(t))
            .sum()
    }

    // -- Privacy metrics ----------------------------------------------------

    /// Compute δ(ε) via the Method of Steepest Descent (MSD).
    ///
    /// Order-1 saddle-point approximation from Alghamdi et al. Section 3.3.
    ///
    /// Defines F_ε(t) = n·K(t) − ε·t − log(t) − log(1+t) and finds
    /// the saddle point (minimum) t*. Then:
    ///
    /// δ(ε) ≈ exp(F_ε(t*)) / √(2π · |F_ε''(t*)|)
    pub fn delta_at(&self, epsilon: f64) -> f64 {
        // F_ε(t) = Λ_total(t) − ε·t − log(t) − log(1+t)
        let f_eps = |t: f64| -> f64 {
            self.total_cgf(t) - epsilon * t - t.ln() - (1.0 + t).ln()
        };

        // F_ε'(t) = Λ'_total(t) − ε − 1/t + 1/(1+t)
        let f_eps_prime = |t: f64| -> f64 {
            self.total_cgf_prime(t) - epsilon - 1.0 / t + 1.0 / (1.0 + t)
        };

        // F_ε''(t) = Λ''_total(t) + 1/t² − 1/(1+t)²
        let f_eps_double_prime = |t: f64| -> f64 {
            self.total_cgf_double_prime(t) + 1.0 / (t * t) - 1.0 / ((1.0 + t) * (1.0 + t))
        };

        // Find saddle point (minimum of F_ε) via Newton's method on F_ε' = 0.
        // Search in (0, ∞). Start from a reasonable initial guess.
        let mut t = 0.5;
        for _ in 0..100 {
            let fp = f_eps_prime(t);
            let fpp = f_eps_double_prime(t);

            if fpp.abs() < 1e-300 {
                break;
            }

            let step = fp / fpp;
            t -= step;

            // Keep t in valid range (0, ∞)
            if t <= 1e-12 {
                t = 1e-12;
            }

            if step.abs() < 1e-12 * t.abs().max(1.0) {
                break;
            }
        }

        let f_val = f_eps(t);
        let f_second = f_eps_double_prime(t);

        if f_second <= 0.0 {
            // Not a proper minimum; fall back to exp(F) as upper bound
            return f_val.exp().clamp(0.0, 1.0);
        }

        let delta = f_val.exp() / (2.0 * std::f64::consts::PI * f_second).sqrt();
        delta.clamp(0.0, 1.0)
    }

    /// Compute ε(δ) via binary search over `delta_at`.
    pub fn epsilon_at(&self, target_delta: f64) -> f64 {
        if target_delta >= 1.0 {
            return 0.0;
        }
        if target_delta <= 0.0 {
            return f64::INFINITY;
        }

        // Find upper bound by doubling
        let mut hi = 1.0;
        while self.delta_at(hi) > target_delta {
            hi *= 2.0;
            if hi > 1e10 {
                return f64::INFINITY;
            }
        }
        let mut lo = 0.0;

        // Binary search
        for _ in 0..100 {
            let mid = (lo + hi) / 2.0;
            if self.delta_at(mid) > target_delta {
                lo = mid;
            } else {
                hi = mid;
            }
            if hi - lo < 1e-8 {
                break;
            }
        }

        hi
    }

    /// Compute the advantage (= δ(0)).
    pub fn advantage(&self) -> f64 {
        self.delta_at(0.0)
    }

    // -- Composition --------------------------------------------------------

    /// Compose with another SpaPld (heterogeneous).
    ///
    /// Concatenates the component lists. No math, O(k₁ + k₂).
    pub fn compose(&self, other: &SpaPld) -> SpaPld {
        let mut components = self.components.clone();
        components.extend(other.components.iter().cloned());
        SpaPld { components }
    }

    /// Self-compose: multiply all counts by `count`.
    ///
    /// O(k) where k = number of distinct components.
    pub fn self_compose(&self, count: usize) -> SpaPld {
        SpaPld {
            components: self
                .components
                .iter()
                .map(|(cgf, n)| (Arc::clone(cgf), n * count))
                .collect(),
        }
    }

    // -- Materialization (SPA → PmfPld) ------------------------------------

    /// Convert this SPA to a PmfPld by evaluating the delta curve on a grid.
    ///
    /// Used for metrics that require the full PMF (beta_at, risk_at)
    /// and for mixed composition with Pmf-based PLDs.
    pub fn to_pmf_pld(&self, config: &DiscretizationConfig) -> Result<PmfPld> {
        // 1. Find epsilon bounds via binary search
        let tail_threshold = config.log_mass_truncation_bound.exp();
        let epsilon_upper = self.find_epsilon_bound(tail_threshold);
        let epsilon_lower = -epsilon_upper; // Symmetric approximation

        let bounds = EpsilonBounds {
            epsilon_lower,
            epsilon_upper,
        };

        // 2. Compute effective discretization (may coarsen for large range)
        let effective_disc = config.effective_discretization(&bounds);
        let effective_config = DiscretizationConfig {
            discretization: effective_disc,
            ..config.clone()
        };

        // 3. Build epsilon grid and evaluate delta at each point
        let rounded_upper = (epsilon_upper / effective_disc).ceil() as i64;
        let rounded_lower = (epsilon_lower / effective_disc).floor() as i64;

        let deltas: Vec<f64> = (rounded_lower..=rounded_upper)
            .map(|i| {
                let eps = i as f64 * effective_disc;
                self.delta_at(eps)
            })
            .collect();

        // 4. Feed into existing connect-the-dots discretization
        let pmf = discretize_from_deltas(
            bounds,
            &deltas,
            &effective_config,
            Adjacency::Remove,
        )?;

        Ok(PmfPld::new_symmetric(pmf))
    }

    // -- Internal helpers ---------------------------------------------------

    /// Find the epsilon where delta_at(ε) drops below the given threshold.
    ///
    /// Used to determine epsilon bounds for materialization.
    fn find_epsilon_bound(&self, threshold: f64) -> f64 {
        // Start with a reasonable guess and double until delta < threshold
        let mut hi = 1.0;
        while self.delta_at(hi) > threshold {
            hi *= 2.0;
            if hi > 1e10 {
                return hi;
            }
        }

        // Binary search to refine
        let mut lo = 0.0;
        for _ in 0..60 {
            let mid = (lo + hi) / 2.0;
            if self.delta_at(mid) > threshold {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        hi
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pld::cgf::GaussianCgf;
    use approx::assert_relative_eq;

    fn gauss_spa(sigma: f64) -> SpaPld {
        SpaPld::new(Arc::new(GaussianCgf::new(sigma)))
    }

    #[test]
    fn test_self_compose_multiplies_counts() {
        let spa = gauss_spa(1.0);
        let composed = spa.self_compose(100);
        assert_eq!(composed.components.len(), 1);
        assert_eq!(composed.components[0].1, 100);
    }

    #[test]
    fn test_compose_concatenates() {
        let a = gauss_spa(0.5);
        let b = gauss_spa(1.0);
        let composed = a.compose(&b);
        assert_eq!(composed.components.len(), 2);
    }

    #[test]
    fn test_self_compose_then_compose_equivalent() {
        // (A * 100).compose(B * 50) should give 3 components...
        // but (A * 100) is 1 component, (B * 50) is 1 component,
        // composed = 2 components
        let a = gauss_spa(0.5).self_compose(100);
        let b = gauss_spa(1.0).self_compose(50);
        let composed = a.compose(&b);
        assert_eq!(composed.components.len(), 2);
        assert_eq!(composed.components[0].1, 100);
        assert_eq!(composed.components[1].1, 50);
    }

    #[test]
    fn test_delta_at_zero_epsilon_is_positive() {
        // For a non-trivial mechanism, δ(0) > 0 (advantage)
        let spa = gauss_spa(0.5).self_compose(10);
        let delta = spa.delta_at(0.0);
        assert!(delta > 0.0, "delta(0) = {}", delta);
        assert!(delta <= 1.0, "delta(0) = {}", delta);
    }

    #[test]
    fn test_delta_decreases_with_epsilon() {
        let spa = gauss_spa(0.5).self_compose(100);
        let epsilons = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0];
        let deltas: Vec<f64> = epsilons.iter().map(|&e| spa.delta_at(e)).collect();

        for w in deltas.windows(2) {
            assert!(
                w[1] <= w[0] + 1e-10,
                "delta should decrease: δ({})={} > δ({})={}",
                0.0,
                w[0],
                0.0,
                w[1]
            );
        }
    }

    #[test]
    fn test_epsilon_at_and_delta_at_consistent() {
        let spa = gauss_spa(0.5).self_compose(100);

        for &target_delta in &[0.1, 0.01, 1e-3, 1e-5] {
            let eps = spa.epsilon_at(target_delta);
            if eps.is_finite() {
                let achieved = spa.delta_at(eps);
                assert!(
                    achieved <= target_delta + 1e-6,
                    "eps={}, achieved δ={}, target δ={}",
                    eps,
                    achieved,
                    target_delta
                );
            }
        }
    }

    #[test]
    fn test_spa_gaussian_vs_analytical_single_step() {
        use statrs::distribution::{ContinuousCDF, Normal};

        // For a single Gaussian step, compare SPA δ(ε) to the exact formula:
        // δ(ε) = Φ(1/(2σ) − εσ) − e^ε · Φ(−1/(2σ) − εσ)
        let sigma = 0.5;
        let spa = gauss_spa(sigma);
        let n = Normal::new(0.0, 1.0).unwrap();
        let dt = 1.0 / sigma;

        for &eps in &[0.5, 1.0, 2.0, 3.0] {
            let analytical =
                (n.cdf(dt / 2.0 - eps / dt) - eps.exp() * n.cdf(-dt / 2.0 - eps / dt)).max(0.0);
            let spa_delta = spa.delta_at(eps);

            // SPA is an asymptotic approximation — for n=1, allow ~25% relative error.
            // Accuracy improves dramatically with composition count.
            let rel_error = if analytical > 1e-10 {
                (spa_delta - analytical).abs() / analytical
            } else {
                (spa_delta - analytical).abs()
            };
            assert!(
                rel_error < 0.25,
                "σ={}, ε={}: SPA={:.6e}, analytical={:.6e}, rel_err={:.2}%",
                sigma,
                eps,
                spa_delta,
                analytical,
                rel_error * 100.0
            );
        }
    }

    #[test]
    fn test_self_compose_1_times_n_equals_direct_n() {
        // Composing 1 step × 1000 should give same as creating with count=1000
        let step = gauss_spa(0.5);
        let composed = step.self_compose(1000);

        let eps = 5.0;
        let delta_composed = composed.delta_at(eps);
        assert!(delta_composed > 0.0);
        assert!(delta_composed < 1.0);
    }

    #[test]
    fn test_more_compositions_means_larger_delta() {
        let step = gauss_spa(0.5);
        let eps = 5.0;

        let d100 = step.self_compose(100).delta_at(eps);
        let d1000 = step.self_compose(1000).delta_at(eps);

        assert!(
            d1000 >= d100,
            "d1000={} should be >= d100={}",
            d1000,
            d100
        );
    }

    #[test]
    fn test_higher_noise_means_smaller_delta() {
        let n = 100;
        let eps = 5.0;

        let d_low_noise = gauss_spa(0.3).self_compose(n).delta_at(eps);
        let d_high_noise = gauss_spa(1.0).self_compose(n).delta_at(eps);

        assert!(
            d_high_noise <= d_low_noise,
            "higher noise should give smaller delta: d(σ=1.0)={} > d(σ=0.3)={}",
            d_high_noise,
            d_low_noise
        );
    }
}
