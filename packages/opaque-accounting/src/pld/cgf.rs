//! Cumulant Generating Function (CGF) trait and implementations.
//!
//! The CGF Λ(t) = log E[exp(t·L)] of a privacy loss random variable L
//! is the core abstraction for the Saddle-Point Accountant. It is
//! mechanism-agnostic: `CgfPld` stores opaque `Arc<dyn Cgf>` handles
//! and never needs to know which mechanism produced them.
//!
//! Mechanism-specific code lives only in the implementations below
//! (and their constructor functions in `mechanisms/` and `amplification/`).

use std::fmt::Debug;

/// A cumulant generating function Λ(t) = log E[exp(t·L)] of a privacy loss RV.
///
/// Implementations are created by mechanism-specific constructor functions
/// and stored opaquely inside [`super::CgfPld`] via `Arc<dyn Cgf>`.
pub trait Cgf: Debug + Send + Sync {
    /// Evaluate Λ(t).
    fn eval(&self, t: f64) -> f64;

    /// Evaluate Λ'(t) (first derivative).
    fn eval_prime(&self, t: f64) -> f64;

    /// Evaluate Λ''(t) (second derivative).
    fn eval_double_prime(&self, t: f64) -> f64;
}

// ---------------------------------------------------------------------------
// Gaussian mechanism CGF
// ---------------------------------------------------------------------------

/// CGF for the Gaussian mechanism with sensitivity Δ=1 and noise σ.
///
/// Privacy loss: L(x) = (1 − 2x) / (2σ²) where x ~ N(0, σ²).
///
/// - Λ(t)  = t(1 + t) / (2σ²)
/// - Λ'(t) = (1 + 2t) / (2σ²)
/// - Λ''(t) = 1/σ²
#[derive(Debug, Clone)]
pub struct GaussianCgf {
    /// Noise multiplier σ (= noise_std / sensitivity).
    #[allow(dead_code)]
    pub(crate) sigma: f64,
    /// Pre-computed 1 / (2σ²).
    inv_2sigma_sq: f64,
    /// Pre-computed 1 / σ².
    inv_sigma_sq: f64,
}

impl GaussianCgf {
    pub fn new(sigma: f64) -> Self {
        let sigma_sq = sigma * sigma;
        Self {
            sigma,
            inv_2sigma_sq: 1.0 / (2.0 * sigma_sq),
            inv_sigma_sq: 1.0 / sigma_sq,
        }
    }
}

impl Cgf for GaussianCgf {
    #[inline]
    fn eval(&self, t: f64) -> f64 {
        t * (1.0 + t) * self.inv_2sigma_sq
    }

    #[inline]
    fn eval_prime(&self, t: f64) -> f64 {
        (1.0 + 2.0 * t) * self.inv_2sigma_sq
    }

    #[inline]
    fn eval_double_prime(&self, _t: f64) -> f64 {
        self.inv_sigma_sq
    }
}

// ---------------------------------------------------------------------------
// Poisson-subsampled Gaussian mechanism CGF
// ---------------------------------------------------------------------------

/// CGF for the Poisson-subsampled Gaussian mechanism.
///
/// Privacy loss for the subsampled mechanism uses a mixture:
/// with probability q the sample is included (Gaussian privacy loss),
/// with probability 1−q it is not (zero privacy loss).
///
/// Λ_sub(t) = log E_x[ (q · exp(L(x)) + 1−q)^t ]
///
/// where L(x) = (1 − 2x) / (2σ²), x ~ N(0, σ²).
///
/// Evaluated via Gauss-Hermite quadrature; derivatives via central
/// finite differences.
#[derive(Debug, Clone)]
pub struct SubsampledGaussianCgf {
    /// Noise multiplier σ.
    pub(crate) sigma: f64,
    /// Poisson sampling rate q ∈ (0, 1].
    pub(crate) rate: f64,
    /// Pre-computed 1 / (2σ²).
    inv_2sigma_sq: f64,
    /// Gauss-Hermite nodes (standard normal).
    gh_nodes: Vec<f64>,
    /// Gauss-Hermite weights.
    gh_weights: Vec<f64>,
}

/// Number of Gauss-Hermite quadrature nodes.
const GH_ORDER: usize = 30;

/// Step size for finite-difference derivatives.
const FD_STEP: f64 = 1e-7;

impl SubsampledGaussianCgf {
    pub fn new(sigma: f64, rate: f64) -> Self {
        let (nodes, weights) = gauss_hermite_nodes_weights(GH_ORDER);
        Self {
            sigma,
            rate,
            inv_2sigma_sq: 1.0 / (2.0 * sigma * sigma),
            gh_nodes: nodes,
            gh_weights: weights,
        }
    }

    /// Evaluate the CGF Λ(t) via Gauss-Hermite quadrature.
    ///
    /// We compute E_{x~N(0,σ²)}[ (q·exp(L(x)) + 1−q)^t ]
    /// where L(x) = (1 − 2x) / (2σ²).
    ///
    /// Change of variables for Gauss-Hermite: x = σ·√2·z where the
    /// quadrature integrates against exp(-z²). The weights already
    /// include the √π normalization factor.
    fn eval_raw(&self, t: f64) -> f64 {
        let q = self.rate;
        let one_minus_q = 1.0 - q;
        let sqrt2 = std::f64::consts::SQRT_2;

        // Use log-sum-exp for numerical stability.
        // We compute log(Σ wᵢ · fᵢ) where fᵢ can be very large.
        let mut log_vals: Vec<f64> = Vec::with_capacity(self.gh_nodes.len());
        let mut log_weights: Vec<f64> = Vec::with_capacity(self.gh_nodes.len());

        for (&z, &w) in self.gh_nodes.iter().zip(self.gh_weights.iter()) {
            if w <= 0.0 {
                continue;
            }
            // Privacy loss at quadrature point x = σ√2·z
            let loss = self.inv_2sigma_sq - sqrt2 * z / self.sigma;
            // Mixture: q·exp(L) + (1-q)
            let mixture = q * loss.exp() + one_minus_q;
            if mixture <= 0.0 {
                continue;
            }
            // log(w · mixture^t) = log(w) + t·log(mixture)
            log_weights.push(w.ln());
            log_vals.push(t * mixture.ln());
        }

        if log_vals.is_empty() {
            return 0.0;
        }

        // log-sum-exp: log(Σ exp(aᵢ)) where aᵢ = log(wᵢ) + t·log(mixture_i)
        let log_terms: Vec<f64> = log_weights
            .iter()
            .zip(log_vals.iter())
            .map(|(&lw, &lv)| lw + lv)
            .collect();

        let max_log = log_terms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = log_terms.iter().map(|&lt| (lt - max_log).exp()).sum();

        // Result: log(E[f]) = log(Σ wᵢ fᵢ / √π) = max + log(sum_exp) - log(√π)
        max_log + sum_exp.ln() - 0.5 * std::f64::consts::PI.ln()
    }
}

impl Cgf for SubsampledGaussianCgf {
    fn eval(&self, t: f64) -> f64 {
        self.eval_raw(t)
    }

    fn eval_prime(&self, t: f64) -> f64 {
        let h = FD_STEP;
        (self.eval_raw(t + h) - self.eval_raw(t - h)) / (2.0 * h)
    }

    fn eval_double_prime(&self, t: f64) -> f64 {
        let h = FD_STEP;
        (self.eval_raw(t + h) - 2.0 * self.eval_raw(t) + self.eval_raw(t - h)) / (h * h)
    }
}

// ---------------------------------------------------------------------------
// Gauss-Hermite quadrature nodes and weights
// ---------------------------------------------------------------------------

/// Gauss-Hermite quadrature nodes and weights (physicist convention, weight exp(-x²)).
///
/// Returns 20-point nodes and weights. Weights already include the √π factor
/// so that Σ wᵢ f(xᵢ) ≈ ∫ f(x) exp(-x²) dx.
///
/// For expectations under standard normal: E[g(z)] = (1/√π) Σ wᵢ g(xᵢ).
fn gauss_hermite_nodes_weights(_n: usize) -> (Vec<f64>, Vec<f64>) {
    // 20-point Gauss-Hermite quadrature (hardcoded for reliability).
    // Nodes are roots of H₂₀(x); weights from standard tables.
    #[rustfmt::skip]
    let nodes = vec![
        -5.387480890011233, -4.603682449550744, -3.944764040115625,
        -3.347854567383216, -2.788806058428130, -2.254974002089276,
        -1.738537712116586, -1.234076215395323, -0.737473728545394,
        -0.245340708300901,
         0.245340708300901,  0.737473728545394,  1.234076215395323,
         1.738537712116586,  2.254974002089276,  2.788806058428130,
         3.347854567383216,  3.944764040115625,  4.603682449550744,
         5.387480890011233,
    ];
    #[rustfmt::skip]
    let weights = vec![
        2.229393645534e-13, 4.399340992273e-10, 1.086069370769e-07,
        7.802556478532e-06, 2.283386360163e-04, 3.243773342238e-03,
        2.481052088746e-02, 1.090172060200e-01, 2.866755053628e-01,
        4.622436696006e-01,
        4.622436696006e-01, 2.866755053628e-01, 1.090172060200e-01,
        2.481052088746e-02, 3.243773342238e-03, 2.283386360163e-04,
        7.802556478532e-06, 1.086069370769e-07, 4.399340992273e-10,
        2.229393645534e-13,
    ];
    (nodes, weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_gaussian_cgf_at_zero() {
        let cgf = GaussianCgf::new(1.0);
        assert_relative_eq!(cgf.eval(0.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_gaussian_cgf_values() {
        let sigma = 0.5;
        let cgf = GaussianCgf::new(sigma);
        let inv_2s2 = 1.0 / (2.0 * sigma * sigma);

        for &t in &[0.0, 0.5, 1.0, -0.5, 2.0] {
            let expected = t * (1.0 + t) * inv_2s2;
            assert_relative_eq!(cgf.eval(t), expected, epsilon = 1e-12);

            let expected_prime = (1.0 + 2.0 * t) * inv_2s2;
            assert_relative_eq!(cgf.eval_prime(t), expected_prime, epsilon = 1e-12);

            let expected_dbl = 1.0 / (sigma * sigma);
            assert_relative_eq!(cgf.eval_double_prime(t), expected_dbl, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_gaussian_cgf_derivative_vs_finite_diff() {
        let cgf = GaussianCgf::new(0.3);
        let h = 1e-7;
        for &t in &[0.0, 0.5, 1.0, -0.3] {
            let fd_prime = (cgf.eval(t + h) - cgf.eval(t - h)) / (2.0 * h);
            assert_relative_eq!(cgf.eval_prime(t), fd_prime, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_subsampled_gaussian_cgf_at_zero() {
        let cgf = SubsampledGaussianCgf::new(1.0, 0.01);
        // Λ(0) = log E[1] = 0
        assert_relative_eq!(cgf.eval(0.0), 0.0, epsilon = 1e-8);
    }

    #[test]
    fn test_subsampled_gaussian_reduces_to_gaussian_at_rate_one() {
        // At rate=1.0, subsampled should match non-subsampled Gaussian
        let sigma = 0.5;
        let gauss = GaussianCgf::new(sigma);
        let sub = SubsampledGaussianCgf::new(sigma, 1.0);

        for &t in &[0.1, 0.5, 1.0, 1.5] {
            assert_relative_eq!(sub.eval(t), gauss.eval(t), epsilon = 1e-4);
        }
    }

    #[test]
    fn test_gauss_hermite_integrates_constant() {
        // ∫ 1 · exp(-x²) dx = √π, so (1/√π) Σ wᵢ · 1 = 1
        let (_, weights) = gauss_hermite_nodes_weights(20);
        let sum: f64 = weights.iter().sum();
        assert_relative_eq!(sum / std::f64::consts::PI.sqrt(), 1.0, epsilon = 1e-10);
    }
}
