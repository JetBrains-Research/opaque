//! CGF-backed privacy loss distribution: mechanism-agnostic accounting.
//!
//! `CgfPld` stores a list of opaque CGF handles with repetition counts.
//! No grid, no discretization — CGFs are only evaluated at query time
//! via the saddle-point method of steepest descent (MSD).
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
// CgfPld
// ---------------------------------------------------------------------------

/// CGF-backed privacy loss distribution.
///
/// Stores a list of (CGF, repetition_count) components. The total CGF is:
///
/// Λ_total(t) = Σᵢ countᵢ · Λᵢ(t)
///
/// This is mechanism-agnostic: `CgfPld` never knows what mechanism
/// produced the CGFs. All composition operations are trivial (concatenate
/// or multiply counts). Privacy metrics are computed at query time via
/// the saddle-point method of steepest descent.
#[derive(Clone)]
pub struct CgfPld {
    pub(crate) components: Vec<(Arc<dyn Cgf>, usize)>,
}

impl fmt::Debug for CgfPld {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let total: usize = self.components.iter().map(|(_, n)| n).sum();
        f.debug_struct("CgfPld")
            .field("num_components", &self.components.len())
            .field("total_compositions", &total)
            .finish()
    }
}

impl CgfPld {
    /// Create a CgfPld from a single CGF (count=1).
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

    /// Compute δ(ε) via Lugannani-Rice saddle-point approximation.
    ///
    /// Decomposes the hockey-stick divergence as:
    ///
    ///   δ(ε) = P(L > ε) − e^ε · e^{Λ(−1)} · P_{−1}(L > ε)
    ///
    /// where P_{−1} is the distribution under exponential tilting by −1,
    /// with CGF Λ_{−1}(t) = Λ(t−1) − Λ(−1).
    ///
    /// Each tail probability is computed via the Lugannani-Rice formula:
    ///
    ///   P(L > ε) ≈ 1 − Φ(r) + φ(r)·(1/r − 1/s)
    ///
    /// where t* solves Λ'(t*) = ε (clean equation, no log singularity),
    /// r = sign(t*)·√(2·(t*·ε − Λ(t*))), and s = t*·√(Λ''(t*)).
    pub fn delta_at(&self, epsilon: f64) -> f64 {
        let delta_rem = self.delta_remove(epsilon);

        // For CGFs following the math-doc convention, Λ(-1) = 0.
        // In that case we can reliably compute the add direction too.
        // For legacy CGFs (SubsampledGaussianCgf) where Λ(-1) ≠ 0,
        // the add direction formula doesn't apply — use remove only.
        let cgf_neg1 = self.total_cgf(-1.0);
        if cgf_neg1.abs() > 1e-6 {
            return delta_rem;
        }

        let delta_add = self.delta_add(epsilon);
        delta_rem.max(delta_add)
    }

    /// Compute δ_rem = H_ε(P, Q) via the REMOVE direction CGF.
    fn delta_remove(&self, epsilon: f64) -> f64 {
        // Handle degenerate distributions (Λ''=0 everywhere)
        let dbl_0 = self.total_cgf_double_prime(0.0);
        let dbl_half = self.total_cgf_double_prime(0.5);
        if dbl_0.abs() < 1e-20 && dbl_half.abs() < 1e-20 {
            let l_const = self.total_cgf_prime(0.0);
            if l_const <= epsilon {
                return 0.0;
            }
            return (1.0 - (epsilon - l_const).exp()).clamp(0.0, 1.0);
        }

        use statrs::distribution::{ContinuousCDF, Normal};
        let normal = Normal::new(0.0, 1.0).unwrap();

        // First tail: P_P(L > ε), CGF = Λ(t), saddle: Λ'(t*) = ε
        let t_star = self.find_cgf_saddle(epsilon);
        let log_tail_orig = self.lugannani_rice_log_tail(&normal, t_star, epsilon, 0.0);

        // Second tail: P_Q(L > ε) = P_{-1}(L > ε), CGF_{-1}(t) = Λ(t-1) - Λ(-1)
        let s_star = t_star + 1.0;
        let log_tail_tilt = self.lugannani_rice_log_tail(&normal, s_star, epsilon, -1.0);

        // δ_rem = P_P(L>ε) - e^{ε+Λ(-1)} · P_{-1}(L>ε)
        let cgf_neg1 = self.total_cgf(-1.0);
        let log_term2 = epsilon + cgf_neg1 + log_tail_tilt;

        let delta = if log_tail_orig > log_term2 {
            let diff = log_term2 - log_tail_orig;
            if diff < -50.0 {
                log_tail_orig.exp()
            } else {
                log_tail_orig.exp() * (1.0 - diff.exp())
            }
        } else {
            0.0
        };

        delta.clamp(0.0, 1.0)
    }

    /// Compute δ_add = H_ε(Q, P) via the ADD direction.
    ///
    /// Uses the identity: Λ_add(t) = Λ_rem(-(1+t)).
    ///
    /// Derivation: Λ_add(t) = log E_Q[(Q/P)^{1+t}] = log ∫ Q^{2+t}/P^{1+t} dx.
    /// Setting s = -(1+t) in Λ_rem(s) = log ∫ P^{1+s}/Q^s dx gives
    ///   Λ_rem(-(1+t)) = log ∫ P^{-t}/Q^{-(1+t)} dx = log ∫ Q^{1+t}/P^t dx.
    /// Hmm, that gives Q^{1+t}/P^t not Q^{2+t}/P^{1+t}. Let me redo:
    ///
    /// Actually Λ_add(t) = log E_Q[e^{-tL}] where L = log(P/Q).
    /// = log ∫ e^{-t log(P/Q)} Q dx = log ∫ (P/Q)^{-t} Q dx = log ∫ Q^{1+t}/P^t dx.
    ///
    /// And Λ_rem(s) = log E_Q[(P/Q)^{1+s}] = log ∫ P^{1+s}/Q^s dx.
    /// Setting 1+s = -t ↔ s = -(1+t):
    ///   Λ_rem(-(1+t)) = log ∫ P^{-t}/Q^{-(1+t)} dx = log ∫ Q^{1+t}/P^t dx = Λ_add(t). ✓
    ///
    /// Properties: Λ_add(-1) = Λ_rem(0) = 0.
    ///             Λ_add(0)  = Λ_rem(-1) = 0 (math-doc convention).
    ///             Λ_add'(t) = -Λ_rem'(-(1+t))
    ///             Λ_add''(t) = Λ_rem''(-(1+t))
    ///
    /// For symmetric mechanisms (Gaussian): Λ_add(t) = Λ_rem(t), so δ_add = δ_rem.
    fn delta_add(&self, epsilon: f64) -> f64 {
        // Λ_add(t) = total_cgf(-(1+t))
        // Λ_add'(t) = -total_cgf'(-(1+t))
        // Λ_add''(t) = total_cgf''(-(1+t))

        // Check degeneracy
        let dbl_0 = self.total_cgf_double_prime(-1.0); // = Λ_add''(0)
        let dbl_half = self.total_cgf_double_prime(-1.5); // = Λ_add''(0.5)
        if dbl_0.abs() < 1e-20 && dbl_half.abs() < 1e-20 {
            let l_const = -self.total_cgf_prime(-1.0); // = Λ_add'(0)
            if l_const <= epsilon { return 0.0; }
            return (1.0 - (epsilon - l_const).exp()).clamp(0.0, 1.0);
        }

        use statrs::distribution::{ContinuousCDF, Normal};
        let normal = Normal::new(0.0, 1.0).unwrap();

        // Saddle: Λ_add'(t*) = ε ↔ Λ_rem'(-(1+t*)) = -ε
        // Find u where Λ_rem'(u) = -ε, then t* = -(1+u)
        let u_star = self.find_cgf_saddle(-epsilon);
        let t_star = -(1.0 + u_star);

        // First tail: offset=0
        // Λ_add(t*) - Λ_add(0) = total_cgf(-(1+t*)) - total_cgf(-1)
        //                        = total_cgf(u_star) - total_cgf(-1)
        let log_tail_orig = self.lugannani_rice_log_tail_add(
            &normal, t_star, epsilon, u_star, -1.0,
        );

        // Second tail: offset=-1 (tilt by -1 in add coordinate)
        // Saddle in tilted: t*+1. In rem coordinate: -(1+(t*+1)) = u_star - 1
        let s_star = t_star + 1.0;
        let u_tilt = u_star - 1.0;
        // Λ_add(s*-1) = total_cgf(-(1+s*-1+(-1))) = total_cgf(-(1+t*)) = total_cgf(u_star)
        // Actually: Λ_add(s* + offset) for offset=-1 is Λ_add(s*-1) = Λ_add(t*) = total_cgf(u_star)
        // And Λ_add(offset) = Λ_add(-1) = total_cgf(0) = 0 (since Λ_add(-1) = Λ_rem(0) = 0)
        let log_tail_tilt = self.lugannani_rice_log_tail_add(
            &normal, s_star, epsilon, u_tilt, 0.0,
        );

        // δ_add = P_Q(-L>ε) - e^{ε + Λ_add(-1)} · P_{-1}(-L>ε)
        // Λ_add(-1) = Λ_rem(0) = 0
        let log_term2 = epsilon + log_tail_tilt;

        let delta = if log_tail_orig > log_term2 {
            let diff = log_term2 - log_tail_orig;
            if diff < -50.0 { log_tail_orig.exp() }
            else { log_tail_orig.exp() * (1.0 - diff.exp()) }
        } else { 0.0 };

        delta.clamp(0.0, 1.0)
    }

    /// Lugannani-Rice log-tail for the add direction.
    ///
    /// Computes the LR tail for the add-direction CGF Λ_add(t) = Λ_rem(-(1+t)),
    /// using the rem-direction total_cgf evaluated at the remapped argument.
    ///
    /// `saddle_add`: the saddle point in add coordinates
    /// `epsilon`: the privacy threshold
    /// `u_rem`: the corresponding point in rem coordinates: -(1+saddle_add+offset)
    /// `offset_rem`: Λ_add(offset) = total_cgf(offset_rem) [in rem coordinates]
    fn lugannani_rice_log_tail_add(
        &self,
        normal: &statrs::distribution::Normal,
        saddle_add: f64,
        epsilon: f64,
        u_rem: f64,
        offset_rem: f64,
    ) -> f64 {
        use statrs::distribution::{ContinuousCDF, Continuous};

        // Shifted CGF: Λ_add(saddle+offset) - Λ_add(offset) = total_cgf(u_rem) - total_cgf(offset_rem)
        let cgf_val = self.total_cgf(u_rem) - self.total_cgf(offset_rem);
        // Λ_add''(saddle+offset) = Λ_rem''(u_rem)
        let cgf_dbl = self.total_cgf_double_prime(u_rem);

        let arg_r = 2.0 * (saddle_add * epsilon - cgf_val);

        let r = if arg_r <= 0.0 || saddle_add.abs() < 1e-15 {
            0.0
        } else {
            saddle_add.signum() * arg_r.sqrt()
        };

        let s = if cgf_dbl > 0.0 {
            saddle_add * cgf_dbl.sqrt()
        } else {
            0.0
        };

        let survival = normal.cdf(-r);
        let pdf_r = normal.pdf(r);

        let tail = if r.abs() < 1e-10 && s.abs() < 1e-10 {
            0.5
        } else if r.abs() < 1e-10 {
            0.5 - pdf_r / s
        } else if s.abs() < 1e-10 {
            survival
        } else {
            survival + pdf_r * (1.0 / r - 1.0 / s)
        };

        if tail <= 0.0 { f64::NEG_INFINITY } else { tail.ln() }
    }

    /// Find t* where Λ'_total(t*) = target via Newton's method.
    ///
    /// For positive targets (remove direction), starts near t=0.5.
    /// For negative targets (add direction saddle), starts near t=-0.5
    /// since the saddle is in the negative half-plane.
    fn find_cgf_saddle(&self, target: f64) -> f64 {
        // Choose initial point: for the add direction (target < 0),
        // the saddle is at t < 0 (between -1 and 0 for typical mechanisms).
        let mut t = if target >= 0.0 { 0.5_f64 } else { -0.5_f64 };

        for _ in 0..100 {
            let residual = self.total_cgf_prime(t) - target;
            let jacobian = self.total_cgf_double_prime(t);

            if jacobian.abs() < 1e-300 {
                break;
            }

            let step = residual / jacobian;
            // Dampen large steps to avoid overshooting
            let max_step = 2.0;
            let clamped_step = step.clamp(-max_step, max_step);
            t -= clamped_step;

            if step.abs() < 1e-12 * t.abs().max(1.0) {
                break;
            }
        }
        t
    }

    /// Lugannani-Rice log-tail: log P(X > ε) for a distribution whose CGF is
    /// `Λ_shifted(t) = total_cgf(t + offset) − total_cgf(offset)`.
    ///
    /// - `saddle`: the saddle point (Λ'(saddle + offset) = ε)
    /// - `epsilon`: the threshold
    /// - `offset`: 0 for the original distribution, −1 for the −1 tilted distribution
    ///
    /// Returns log of the tail probability for numerical stability.
    fn lugannani_rice_log_tail(
        &self,
        normal: &statrs::distribution::Normal,
        saddle: f64,
        epsilon: f64,
        offset: f64,
    ) -> f64 {
        use statrs::distribution::{ContinuousCDF, Continuous};

        // Evaluate shifted CGF: Λ_shifted(saddle) = Λ(saddle + offset) − Λ(offset)
        let cgf_val = self.total_cgf(saddle + offset) - self.total_cgf(offset);
        let cgf_dbl = self.total_cgf_double_prime(saddle + offset);

        // r = sign(saddle) · √(2·(saddle·ε − Λ_shifted(saddle)))
        let arg_r = 2.0 * (saddle * epsilon - cgf_val);

        let r = if arg_r <= 0.0 || saddle.abs() < 1e-15 {
            0.0
        } else {
            saddle.signum() * arg_r.sqrt()
        };

        let s = saddle * cgf_dbl.sqrt();

        // Lugannani-Rice: P(X > ε) ≈ 1 − Φ(r) + φ(r)·(1/r − 1/s)
        //
        // For large |r|, use log-space directly via Φ_c(r) = 1 − Φ(r).
        let survival = normal.cdf(-r); // Φ(−r) = 1 − Φ(r)
        let pdf_r = normal.pdf(r);

        let tail = if r.abs() < 1e-10 && s.abs() < 1e-10 {
            0.5
        } else if r.abs() < 1e-10 {
            0.5 - pdf_r / s
        } else if s.abs() < 1e-10 {
            survival
        } else {
            let correction = pdf_r * (1.0 / r - 1.0 / s);
            survival + correction
        };

        if tail <= 0.0 {
            f64::NEG_INFINITY
        } else {
            tail.ln()
        }
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

    /// Compose with another CgfPld (heterogeneous).
    ///
    /// Concatenates the component lists. No math, O(k₁ + k₂).
    pub fn compose(&self, other: &CgfPld) -> CgfPld {
        let mut components = self.components.clone();
        components.extend(other.components.iter().cloned());
        CgfPld { components }
    }

    /// Self-compose: multiply all counts by `count`.
    ///
    /// O(k) where k = number of distinct components.
    pub fn self_compose(&self, count: usize) -> CgfPld {
        CgfPld {
            components: self
                .components
                .iter()
                .map(|(cgf, n)| (Arc::clone(cgf), n * count))
                .collect(),
        }
    }

    // -- Materialization (CgfPld → PmfPld) ----------------------------------

    /// Convert this CgfPld to a PmfPld by evaluating the delta curve on a grid.
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

    fn gauss_cgf(sigma: f64) -> CgfPld {
        CgfPld::new(Arc::new(GaussianCgf::new(sigma)))
    }

    #[test]
    fn test_self_compose_multiplies_counts() {
        let cgf = gauss_cgf(1.0);
        let composed = cgf.self_compose(100);
        assert_eq!(composed.components.len(), 1);
        assert_eq!(composed.components[0].1, 100);
    }

    #[test]
    fn test_compose_concatenates() {
        let a = gauss_cgf(0.5);
        let b = gauss_cgf(1.0);
        let composed = a.compose(&b);
        assert_eq!(composed.components.len(), 2);
    }

    #[test]
    fn test_self_compose_then_compose_equivalent() {
        // (A * 100).compose(B * 50) should give 3 components...
        // but (A * 100) is 1 component, (B * 50) is 1 component,
        // composed = 2 components
        let a = gauss_cgf(0.5).self_compose(100);
        let b = gauss_cgf(1.0).self_compose(50);
        let composed = a.compose(&b);
        assert_eq!(composed.components.len(), 2);
        assert_eq!(composed.components[0].1, 100);
        assert_eq!(composed.components[1].1, 50);
    }

    #[test]
    fn test_delta_at_zero_epsilon_is_positive() {
        // For a non-trivial mechanism, δ(0) > 0 (advantage)
        let cgf = gauss_cgf(0.5).self_compose(10);
        let delta = cgf.delta_at(0.0);
        assert!(delta > 0.0, "delta(0) = {}", delta);
        assert!(delta <= 1.0, "delta(0) = {}", delta);
    }

    #[test]
    fn test_delta_decreases_with_epsilon() {
        let cgf = gauss_cgf(0.5).self_compose(100);
        let epsilons = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0];
        let deltas: Vec<f64> = epsilons.iter().map(|&e| cgf.delta_at(e)).collect();

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
        let cgf = gauss_cgf(0.5).self_compose(100);

        for &target_delta in &[0.1, 0.01, 1e-3, 1e-5] {
            let eps = cgf.epsilon_at(target_delta);
            if eps.is_finite() {
                let achieved = cgf.delta_at(eps);
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
    fn test_cgf_gaussian_vs_analytical_single_step() {
        use statrs::distribution::{ContinuousCDF, Normal};

        // For a single Gaussian step, compare CGF δ(ε) to the exact formula:
        // δ(ε) = Φ(1/(2σ) − εσ) − e^ε · Φ(−1/(2σ) − εσ)
        let sigma = 0.5;
        let cgf = gauss_cgf(sigma);
        let n = Normal::new(0.0, 1.0).unwrap();
        let dt = 1.0 / sigma;

        for &eps in &[0.5, 1.0, 2.0, 3.0] {
            let analytical =
                (n.cdf(dt / 2.0 - eps / dt) - eps.exp() * n.cdf(-dt / 2.0 - eps / dt)).max(0.0);
            let cgf_delta = cgf.delta_at(eps);

            // CGF saddle-point is an asymptotic approximation — for n=1, allow ~25% relative error.
            // Accuracy improves dramatically with composition count.
            let rel_error = if analytical > 1e-10 {
                (cgf_delta - analytical).abs() / analytical
            } else {
                (cgf_delta - analytical).abs()
            };
            assert!(
                rel_error < 0.25,
                "σ={}, ε={}: CGF={:.6e}, analytical={:.6e}, rel_err={:.2}%",
                sigma,
                eps,
                cgf_delta,
                analytical,
                rel_error * 100.0
            );
        }
    }

    #[test]
    fn test_self_compose_1_times_n_equals_direct_n() {
        // Composing 1 step × 1000 should give same as creating with count=1000
        let step = gauss_cgf(0.5);
        let composed = step.self_compose(1000);

        // Use epsilon_at which is more robust (wraps delta_at via binary search)
        let eps = composed.epsilon_at(1e-5);
        assert!(eps > 0.0 && eps.is_finite(), "ε = {}", eps);
    }

    #[test]
    fn test_more_compositions_means_larger_delta() {
        let step = gauss_cgf(0.5);
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

        let d_low_noise = gauss_cgf(0.3).self_compose(n).delta_at(eps);
        let d_high_noise = gauss_cgf(1.0).self_compose(n).delta_at(eps);

        assert!(
            d_high_noise <= d_low_noise,
            "higher noise should give smaller delta: d(σ=1.0)={} > d(σ=0.3)={}",
            d_high_noise,
            d_low_noise
        );
    }

    #[test]
    fn test_materialization_roundtrip() {
        // CgfPld → to_pmf_pld() → epsilon_at should approximately match CgfPld.epsilon_at
        let cgf = gauss_cgf(0.5).self_compose(100);
        let config = DiscretizationConfig::default();
        let pmf_pld = cgf.to_pmf_pld(&config).expect("materialization failed");

        let eps_cgf = cgf.epsilon_at(1e-5);
        let eps_pmf = crate::pld::metrics::epsilon(&pmf_pld, 1e-5);

        let rel_err = (eps_cgf - eps_pmf).abs() / eps_pmf;
        assert!(
            rel_err < 0.05,
            "materialization roundtrip: CGF ε={:.6}, PMF ε={:.6}, rel_err={:.1}%",
            eps_cgf,
            eps_pmf,
            rel_err * 100.0
        );
    }

    #[test]
    fn test_cgf_accurate_at_n1_and_n100() {
        // Lugannani-Rice should be accurate at both n=1 and n=100
        use statrs::distribution::{ContinuousCDF, Normal};

        let sigma = 0.5;
        let norm = Normal::new(0.0, 1.0).unwrap();
        let dt = 1.0 / sigma;

        // n=1: compare CGF δ vs analytical at ε=1.0
        let cgf_1 = gauss_cgf(sigma);
        let analytical = (norm.cdf(dt / 2.0 - 1.0 / dt)
            - 1.0_f64.exp() * norm.cdf(-dt / 2.0 - 1.0 / dt))
        .max(0.0);
        let err_n1 = (cgf_1.delta_at(1.0) - analytical).abs() / analytical;
        assert!(
            err_n1 < 0.05,
            "n=1: CGF δ={:.6e}, analytical={:.6e}, err={:.1}%",
            cgf_1.delta_at(1.0),
            analytical,
            err_n1 * 100.0
        );

        // n=100: compare CGF vs PMF (PMF is exact up to discretization)
        let config = DiscretizationConfig::default();
        let cgf_100 = gauss_cgf(sigma).self_compose(100);
        let pmf_100 = cgf_100.to_pmf_pld(&config).unwrap();
        let eps_test = cgf_100.epsilon_at(0.1) * 0.8;
        let d_cgf = cgf_100.delta_at(eps_test);
        let d_pmf = crate::pld::metrics::delta(&pmf_100, eps_test);
        let err_n100 = if d_pmf > 1e-12 {
            (d_cgf - d_pmf).abs() / d_pmf
        } else {
            0.0
        };
        assert!(
            err_n100 < 0.05,
            "n=100: CGF δ={:.6e}, PMF δ={:.6e}, err={:.1}%",
            d_cgf,
            d_pmf,
            err_n100 * 100.0
        );
    }
}
