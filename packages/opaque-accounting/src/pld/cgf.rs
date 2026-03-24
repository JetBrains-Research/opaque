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
// Identity mechanism CGF (zero privacy loss)
// ---------------------------------------------------------------------------

/// CGF for the identity mechanism: L = 0 everywhere.
///
/// - Λ(t) = 0
/// - Λ'(t) = 0
/// - Λ''(t) = 0
#[derive(Debug, Clone)]
pub struct IdentityCgf;

impl Cgf for IdentityCgf {
    #[inline]
    fn eval(&self, _t: f64) -> f64 {
        0.0
    }
    #[inline]
    fn eval_prime(&self, _t: f64) -> f64 {
        0.0
    }
    #[inline]
    fn eval_double_prime(&self, _t: f64) -> f64 {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Pure ε-DP mechanism CGF (δ = 0)
// ---------------------------------------------------------------------------

/// CGF for a pure ε-DP mechanism (point mass at privacy loss = ε).
///
/// - Λ(t) = t·ε
/// - Λ'(t) = ε
/// - Λ''(t) = 0
///
/// **Note**: Λ''(t) = 0 means the standard Lugannani-Rice formula has a
/// singularity (ŝ = 0). The CgfPld must handle this degenerate case by
/// computing δ(ε_query) exactly: δ = max(0, 1 − exp(ε_query − ε)).
#[derive(Debug, Clone)]
pub struct PureEpsDpCgf {
    /// The fixed privacy loss ε.
    pub epsilon: f64,
}

impl PureEpsDpCgf {
    pub fn new(epsilon: f64) -> Self {
        Self { epsilon }
    }
}

impl Cgf for PureEpsDpCgf {
    #[inline]
    fn eval(&self, t: f64) -> f64 {
        t * self.epsilon
    }
    #[inline]
    fn eval_prime(&self, _t: f64) -> f64 {
        self.epsilon
    }
    #[inline]
    fn eval_double_prime(&self, _t: f64) -> f64 {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Truncated Gaussian mechanism CGF
// ---------------------------------------------------------------------------

/// CGF for the truncated (renormalized) Gaussian mechanism.
///
/// Privacy loss: L(x) = −Δx/σ² + Δ²/(2σ²) + log(Z₁/Z₀)
/// where x ~ TruncatedNormal(0, σ², [-Rσ, Rσ]) with Z(μ) = Φ((Rσ−μ)/σ) − Φ((−Rσ−μ)/σ)
///
/// Λ(t) = log ∫_{-Rσ}^{Rσ} exp(t·L(x)) · f(x;0) dx
///
/// Evaluated via Gauss-Hermite quadrature with domain clipped to [-R/√2, R/√2].
/// Derivatives via central finite differences.
#[derive(Debug, Clone)]
pub struct TruncatedGaussianCgf {
    sigma: f64,
    radius: f64,
    /// Pre-computed 1 / (2σ²).
    inv_2sigma_sq: f64,
    /// Pre-computed log(Z₁/Z₀) term.
    log_z_ratio: f64,
    /// Pre-computed normalization log(Z₀) for the density.
    log_z0: f64,
    /// Gauss-Hermite nodes.
    gh_nodes: Vec<f64>,
    /// Gauss-Hermite weights.
    gh_weights: Vec<f64>,
}

impl TruncatedGaussianCgf {
    pub fn new(sigma: f64, radius: f64) -> Self {
        let (nodes, weights) = gauss_hermite_nodes_weights(GH_ORDER);
        let sigma_sq = sigma * sigma;
        let sensitivity = 1.0;

        // Z(μ) = Φ((Rσ−μ)/σ) − Φ((−Rσ−μ)/σ)
        let z0 = normal_cdf(radius) - normal_cdf(-radius);
        let z1 = normal_cdf(radius - sensitivity / sigma)
            - normal_cdf(-radius - sensitivity / sigma);
        let log_z_ratio = (z1 / z0).ln();

        Self {
            sigma,
            radius,
            inv_2sigma_sq: 1.0 / (2.0 * sigma_sq),
            log_z_ratio,
            log_z0: z0.ln(),
            gh_nodes: nodes,
            gh_weights: weights,
        }
    }

    fn eval_raw(&self, t: f64) -> f64 {
        let sqrt2 = std::f64::consts::SQRT_2;
        let r_clip = self.radius / sqrt2; // Domain in GH coordinates: z ∈ [-R/√2, R/√2]
        let sensitivity = 1.0;

        // Privacy loss at x: L(x) = -Δx/σ² + Δ²/(2σ²) + log(Z₁/Z₀)
        // The base term that's constant in x:
        let loss_const = sensitivity * sensitivity * self.inv_2sigma_sq + self.log_z_ratio;

        // We want: Λ(t) = log E_{x~TN(0,σ²,[-Rσ,Rσ])}[exp(t·L(x))]
        // = log { (1/Z₀) ∫_{-Rσ}^{Rσ} exp(t·L(x)) · (1/(σ√(2π))) · exp(-x²/(2σ²)) dx }
        // With change of variables x = σ√2·z:
        // = log { (1/Z₀) · (1/√π) ∫ exp(t·L(σ√2·z)) · exp(-z²) dz }
        // where the integral domain is z ∈ [-R/√2, R/√2]

        let mut log_terms: Vec<f64> = Vec::with_capacity(self.gh_nodes.len());

        for (&z, &w) in self.gh_nodes.iter().zip(self.gh_weights.iter()) {
            // Skip nodes outside the truncated domain
            if z < -r_clip || z > r_clip || w <= 0.0 {
                continue;
            }

            let x = self.sigma * sqrt2 * z;
            let loss = -sensitivity * x / (self.sigma * self.sigma) + loss_const;
            // log(w · exp(t·L)) = log(w) + t·L
            log_terms.push(w.ln() + t * loss);
        }

        if log_terms.is_empty() {
            return 0.0;
        }

        // log-sum-exp
        let max_log = log_terms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = log_terms.iter().map(|&lt| (lt - max_log).exp()).sum();

        // Subtract log(Z₀) and log(√π) normalization
        max_log + sum_exp.ln() - 0.5 * std::f64::consts::PI.ln() - self.log_z0
    }
}

impl Cgf for TruncatedGaussianCgf {
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
// Rectified (clamped) Gaussian mechanism CGF
// ---------------------------------------------------------------------------

/// CGF for the rectified (clamped) Gaussian mechanism.
///
/// Mixed distribution: continuous Gaussian interior on (-Rσ, Rσ) plus point
/// masses at the boundaries ±Rσ.
///
/// Λ(t) = log[ I_interior(t) + p_L·exp(t·ε_L) + p_R·exp(t·ε_R) ]
///
/// where I_interior is the GH quadrature integral of the interior density
/// and ε_L, ε_R are the privacy losses at the point masses.
#[derive(Debug, Clone)]
pub struct RectifiedGaussianCgf {
    sigma: f64,
    radius: f64,
    inv_2sigma_sq: f64,
    /// Privacy loss at left boundary: log(p_L(0)/p_L(1))
    eps_left: f64,
    /// Privacy loss at right boundary: log(p_R(0)/p_R(1))
    eps_right: f64,
    /// Left point mass probability under P₀: Φ(-R)
    p_left: f64,
    /// Right point mass probability under P₀: Φ(-R) = 1-Φ(R)
    p_right: f64,
    /// Interior mass probability under P₀: Φ(R) - Φ(-R) = 2Φ(R) - 1
    p_interior: f64,
    /// GH nodes and weights.
    gh_nodes: Vec<f64>,
    gh_weights: Vec<f64>,
}

impl RectifiedGaussianCgf {
    pub fn new(sigma: f64, radius: f64) -> Self {
        let (nodes, weights) = gauss_hermite_nodes_weights(GH_ORDER);
        let sensitivity = 1.0;

        let p_l0 = normal_cdf(-radius); // Φ(-R)
        let p_l1 = normal_cdf(-radius - sensitivity / sigma); // Φ(-R - Δ/σ)
        let eps_left = if p_l1 > 0.0 {
            (p_l0 / p_l1).ln()
        } else {
            f64::INFINITY
        };

        let p_r0 = normal_cdf(-radius); // Φ(-R) = 1 - Φ(R)
        let p_r1 = normal_cdf(sensitivity / sigma - radius); // Φ(Δ/σ - R)
        let eps_right = if p_r0 > 0.0 && p_r1 > 0.0 {
            (p_r0 / p_r1).ln()
        } else {
            0.0
        };

        let p_interior = normal_cdf(radius) - normal_cdf(-radius);

        Self {
            sigma,
            radius,
            inv_2sigma_sq: 1.0 / (2.0 * sigma * sigma),
            eps_left,
            eps_right,
            p_left: p_l0,
            p_right: p_r0,
            p_interior,
            gh_nodes: nodes,
            gh_weights: weights,
        }
    }

    fn eval_raw(&self, t: f64) -> f64 {
        let sqrt2 = std::f64::consts::SQRT_2;
        let r_clip = self.radius / sqrt2;
        let sensitivity = 1.0;

        // Interior: L(x) = Δ(Δ/2 - x)/σ² (same as Gaussian, on bounded domain)
        // E_interior[exp(tL)] = (1/p_interior) ∫_{-Rσ}^{Rσ} exp(tL(x)) · (1/(σ√2π)) · exp(-x²/(2σ²)) dx
        // With x = σ√2·z:
        // = (1/(p_interior·√π)) · ∫ exp(tL(σ√2z)) exp(-z²) dz  [z ∈ [-R/√2, R/√2]]

        let mut log_interior_terms: Vec<f64> = Vec::with_capacity(self.gh_nodes.len());

        for (&z, &w) in self.gh_nodes.iter().zip(self.gh_weights.iter()) {
            if z < -r_clip || z > r_clip || w <= 0.0 {
                continue;
            }
            let x = self.sigma * sqrt2 * z;
            let loss = sensitivity * (sensitivity / 2.0 - x) / (self.sigma * self.sigma);
            log_interior_terms.push(w.ln() + t * loss);
        }

        // Interior contribution (un-normalized by √π but we'll handle later)
        let log_interior_unnorm = if log_interior_terms.is_empty() {
            f64::NEG_INFINITY
        } else {
            let max_log = log_interior_terms
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let sum_exp: f64 = log_interior_terms
                .iter()
                .map(|&lt| (lt - max_log).exp())
                .sum();
            max_log + sum_exp.ln() - 0.5 * std::f64::consts::PI.ln()
        };
        // This is log(∫ exp(tL) φ(x/σ)/σ dx) over [-Rσ, Rσ]
        // The actual interior mass-weighted contribution is: p_interior * E_interior[exp(tL)]
        // But ∫ φ(x/σ)/σ dx over [-Rσ, Rσ] = p_interior, so log_interior_unnorm already
        // includes the p_interior weight.

        // Point mass contributions:
        // p_L · exp(t · ε_L) and p_R · exp(t · ε_R)
        let log_left = if self.p_left > 0.0 && self.eps_left.is_finite() {
            self.p_left.ln() + t * self.eps_left
        } else {
            f64::NEG_INFINITY
        };

        let log_right = if self.p_right > 0.0 && self.eps_right.is_finite() {
            self.p_right.ln() + t * self.eps_right
        } else {
            f64::NEG_INFINITY
        };

        // Λ(t) = log(interior + left + right)
        // Use log-sum-exp over the three terms
        let terms = [log_interior_unnorm, log_left, log_right];
        let max_t = terms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if max_t.is_infinite() && max_t < 0.0 {
            return 0.0; // All terms are -inf
        }
        let sum: f64 = terms.iter().map(|&lt| (lt - max_t).exp()).sum();
        max_t + sum.ln()
    }
}

impl Cgf for RectifiedGaussianCgf {
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
// Mixture CGF (weighted mixture of distributions)
// ---------------------------------------------------------------------------

/// CGF for a mixture distribution: L ~ Σ wᵢ · Pᵢ
///
/// Λ(t) = log Σᵢ wᵢ · exp(Λᵢ(t))
///
/// Stores components as (CGF, log_weight) pairs and evaluates via log-sum-exp.
/// Derivatives via central finite differences.
#[derive(Debug, Clone)]
pub struct MixtureCgf {
    /// Components: (CGF, log_weight).
    components: Vec<(std::sync::Arc<dyn Cgf>, f64)>,
}

impl MixtureCgf {
    /// Create a MixtureCgf from components with linear weights.
    /// Weights must be positive and sum to 1.
    pub fn new(components: Vec<(std::sync::Arc<dyn Cgf>, f64)>) -> Self {
        let log_components: Vec<_> = components
            .into_iter()
            .map(|(cgf, w)| (cgf, w.ln()))
            .collect();
        Self {
            components: log_components,
        }
    }

    /// Create a MixtureCgf from components with log-weights directly.
    pub fn new_log_weights(components: Vec<(std::sync::Arc<dyn Cgf>, f64)>) -> Self {
        Self { components }
    }

    fn eval_raw(&self, t: f64) -> f64 {
        // log Σ exp(log_wᵢ + Λᵢ(t))
        let log_terms: Vec<f64> = self
            .components
            .iter()
            .map(|(cgf, log_w)| log_w + cgf.eval(t))
            .collect();

        let max_log = log_terms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if max_log.is_infinite() && max_log < 0.0 {
            return 0.0;
        }
        let sum: f64 = log_terms.iter().map(|&lt| (lt - max_log).exp()).sum();
        max_log + sum.ln()
    }
}

impl Cgf for MixtureCgf {
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
// Subsampled Truncated Gaussian CGF
// ---------------------------------------------------------------------------

/// CGF for the Poisson-subsampled truncated Gaussian mechanism.
///
/// Λ(t) = log E_{x~TN}[ (q · exp(L_trunc(x)) + 1−q)^t ]
///
/// where L_trunc(x) is the truncated Gaussian privacy loss and x is drawn
/// from TruncatedNormal(0, σ², [-Rσ, Rσ]).
#[derive(Debug, Clone)]
pub struct SubsampledTruncatedGaussianCgf {
    sigma: f64,
    radius: f64,
    rate: f64,
    inv_2sigma_sq: f64,
    log_z_ratio: f64,
    log_z0: f64,
    gh_nodes: Vec<f64>,
    gh_weights: Vec<f64>,
}

impl SubsampledTruncatedGaussianCgf {
    pub fn new(sigma: f64, radius: f64, rate: f64) -> Self {
        let (nodes, weights) = gauss_hermite_nodes_weights(GH_ORDER);
        let sigma_sq = sigma * sigma;
        let sensitivity = 1.0;

        let z0 = normal_cdf(radius) - normal_cdf(-radius);
        let z1 = normal_cdf(radius - sensitivity / sigma)
            - normal_cdf(-radius - sensitivity / sigma);
        let log_z_ratio = (z1 / z0).ln();

        Self {
            sigma,
            radius,
            rate,
            inv_2sigma_sq: 1.0 / (2.0 * sigma_sq),
            log_z_ratio,
            log_z0: z0.ln(),
            gh_nodes: nodes,
            gh_weights: weights,
        }
    }

    fn eval_raw(&self, t: f64) -> f64 {
        let sqrt2 = std::f64::consts::SQRT_2;
        let r_clip = self.radius / sqrt2;
        let q = self.rate;
        let one_minus_q = 1.0 - q;
        let sensitivity = 1.0;
        let loss_const = sensitivity * sensitivity * self.inv_2sigma_sq + self.log_z_ratio;

        let mut log_terms: Vec<f64> = Vec::with_capacity(self.gh_nodes.len());
        let mut log_weights_vec: Vec<f64> = Vec::with_capacity(self.gh_nodes.len());

        for (&z, &w) in self.gh_nodes.iter().zip(self.gh_weights.iter()) {
            if z < -r_clip || z > r_clip || w <= 0.0 {
                continue;
            }
            let x = self.sigma * sqrt2 * z;
            let loss = -sensitivity * x / (self.sigma * self.sigma) + loss_const;
            let mixture = q * loss.exp() + one_minus_q;
            if mixture <= 0.0 {
                continue;
            }
            log_weights_vec.push(w.ln());
            log_terms.push(t * mixture.ln());
        }

        if log_terms.is_empty() {
            return 0.0;
        }

        let combined: Vec<f64> = log_weights_vec
            .iter()
            .zip(log_terms.iter())
            .map(|(&lw, &lv)| lw + lv)
            .collect();

        let max_log = combined.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = combined.iter().map(|&lt| (lt - max_log).exp()).sum();

        // Normalize by Z₀ and √π
        max_log + sum_exp.ln() - 0.5 * std::f64::consts::PI.ln() - self.log_z0
    }
}

impl Cgf for SubsampledTruncatedGaussianCgf {
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

/// Standard normal CDF using statrs.
fn normal_cdf(x: f64) -> f64 {
    use statrs::distribution::{ContinuousCDF, Normal};
    let n = Normal::new(0.0, 1.0).unwrap();
    n.cdf(x)
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
