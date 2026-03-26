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
///
/// The remove-direction CGF is: Λ_rem(t) = log E_Q[(P/Q)^{1+t}].
/// The add-direction CGF is:    Λ_add(t) = log E_Q[(Q/P)^{1+t}].
///
/// Both satisfy Λ(-1) = 0 (normalization).
pub trait Cgf: Debug + Send + Sync {
    /// Evaluate Λ_rem(t) = log E_Q[(P/Q)^{1+t}].
    fn eval(&self, t: f64) -> f64;

    /// Evaluate Λ_rem'(t).
    fn eval_prime(&self, t: f64) -> f64;

    /// Evaluate Λ_rem''(t).
    fn eval_double_prime(&self, t: f64) -> f64;

    /// Evaluate Λ_add(t) = log E_Q[(Q/P)^{1+t}].
    ///
    /// Default: uses the identity Λ_add(t) = Λ_rem(-(1+t)).
    /// Override for numerically stable direct computation when P is a mixture.
    fn eval_add(&self, t: f64) -> f64 {
        self.eval(-(1.0 + t))
    }

    /// Evaluate Λ_add'(t).
    fn eval_add_prime(&self, t: f64) -> f64 {
        let h = 1e-7;
        (self.eval_add(t + h) - self.eval_add(t - h)) / (2.0 * h)
    }

    /// Evaluate Λ_add''(t).
    fn eval_add_double_prime(&self, t: f64) -> f64 {
        let h = 1e-7;
        (self.eval_add(t + h) - 2.0 * self.eval_add(t) + self.eval_add(t - h)) / (h * h)
    }
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
/// Models the remove adjacency where:
///   P(x) = q · N(x; Δ, σ²) + (1−q) · N(x; 0, σ²)  (target may be sampled)
///   Q(x) = N(x; 0, σ²)                               (target absent)
///
/// Following `saddle_point_math.md`:
///   Λ(t) = log E_Q[(P/Q)^{1+t}]
///
/// where P/Q = q · exp((2x−1)/(2σ²)) + (1−q).
///
/// Key property: Λ(−1) = 0 (since (P/Q)^0 = 1).
///
/// At q = 1, reduces to the standard Gaussian CGF: Λ(t) = t(1+t)/(2σ²).
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
    /// Pre-computed √2 / σ.
    sqrt2_over_sigma: f64,
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
            sqrt2_over_sigma: std::f64::consts::SQRT_2 / sigma,
            gh_nodes: nodes,
            gh_weights: weights,
        }
    }

    /// Evaluate Λ(t) = log E_Q[(P/Q)^{1+t}] via Gauss-Hermite quadrature.
    ///
    /// P/Q = q · exp((2x−1)/(2σ²)) + (1−q).
    /// With change of variables x = σ√2·z:
    ///   P/Q = q · exp(√2z/σ − 1/(2σ²)) + (1−q).
    fn eval_raw(&self, t: f64) -> f64 {
        let q = self.rate;
        let one_minus_q = 1.0 - q;
        let exponent = 1.0 + t;

        let mut log_terms: Vec<f64> = Vec::with_capacity(self.gh_nodes.len());

        for (&z, &w) in self.gh_nodes.iter().zip(self.gh_weights.iter()) {
            if w <= 0.0 {
                continue;
            }
            // log(P_1/Q) at x = σ√2·z:  √2z/σ − 1/(2σ²)
            let log_p1_over_q = self.sqrt2_over_sigma * z - self.inv_2sigma_sq;
            // P/Q = q·exp(log_p1_over_q) + (1−q)
            let ratio = q * log_p1_over_q.exp() + one_minus_q;
            if ratio <= 0.0 {
                continue;
            }
            // log(w · ratio^{1+t})
            log_terms.push(w.ln() + exponent * ratio.ln());
        }

        if log_terms.is_empty() {
            return 0.0;
        }

        let max_log = log_terms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = log_terms.iter().map(|&lt| (lt - max_log).exp()).sum();
        max_log + sum_exp.ln() - 0.5 * std::f64::consts::PI.ln()
    }
}

impl SubsampledGaussianCgf {
    /// Direct add-direction CGF: Λ_add(t) = log E_Q[(Q/P)^{1+t}].
    ///
    /// Q/P = 1/(q·exp(√2z/σ − 1/(2σ²)) + (1−q)) is bounded in (0, 1/(1−q)],
    /// making this numerically stable even for small q and large t.
    fn eval_add_raw(&self, t: f64) -> f64 {
        let q = self.rate;
        let one_minus_q = 1.0 - q;
        let exponent = 1.0 + t;

        let mut log_terms: Vec<f64> = Vec::with_capacity(self.gh_nodes.len());

        for (&z, &w) in self.gh_nodes.iter().zip(self.gh_weights.iter()) {
            if w <= 0.0 { continue; }
            let log_p1_over_q = self.sqrt2_over_sigma * z - self.inv_2sigma_sq;
            let log_p_over_q = (q * log_p1_over_q.exp() + one_minus_q).ln();
            // (Q/P)^{1+t} = exp(-(1+t) · log(P/Q))
            log_terms.push(w.ln() - exponent * log_p_over_q);
        }

        if log_terms.is_empty() { return 0.0; }

        let max_log = log_terms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = log_terms.iter().map(|&lt| (lt - max_log).exp()).sum();
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

    // Note: eval_add uses the default (reflected) implementation.
    // The direct eval_add_raw is available but the add direction LR
    // is not accurate enough for SubsampledGaussian at low composition.
    // For m=1 Poisson, use the PMF path for reliable results.
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
        let r_clip = self.radius / sqrt2;
        let sensitivity = 1.0;
        let exponent = 1.0 + t;

        // log(P_trunc/Q_trunc) at x = Δx/σ² − Δ²/(2σ²) − log(Z₁/Z₀)
        // constant part: Δ²/(2σ²) + log(Z₁/Z₀)
        let ratio_const = sensitivity * sensitivity * self.inv_2sigma_sq + self.log_z_ratio;

        // Λ(t) = log E_{Q_trunc}[(P_trunc/Q_trunc)^{1+t}]
        // = log { (1/Z₀) · (1/√π) ∫_{-R/√2}^{R/√2} (P/Q)^{1+t} · exp(-z²) dz }

        let mut log_terms: Vec<f64> = Vec::with_capacity(self.gh_nodes.len());

        for (&z, &w) in self.gh_nodes.iter().zip(self.gh_weights.iter()) {
            if z < -r_clip || z > r_clip || w <= 0.0 {
                continue;
            }

            let x = self.sigma * sqrt2 * z;
            // log(P_trunc/Q_trunc) = Δx/σ² − const
            let log_p_over_q = sensitivity * x / (self.sigma * self.sigma) - ratio_const;
            // log(w · (P/Q)^{1+t}) = log(w) + (1+t)·log(P/Q)
            log_terms.push(w.ln() + exponent * log_p_over_q);
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
        let exponent = 1.0 + t;

        // Λ(t) = log E_Q[(P/Q)^{1+t}] where P = Rect(Δ,σ,R), Q = Rect(0,σ,R).
        //
        // Interior: P(x)/Q(x) = exp(Δ(2x−Δ)/(2σ²)) for x ∈ (−Rσ, Rσ)
        //   = exp((2x−1)/(2σ²)) for Δ=1
        //   = exp(x/σ² − 1/(2σ²))
        //
        // With x = σ√2z: log(P/Q) = √2z/σ − 1/(2σ²)

        let mut log_interior_terms: Vec<f64> = Vec::with_capacity(self.gh_nodes.len());

        for (&z, &w) in self.gh_nodes.iter().zip(self.gh_weights.iter()) {
            if z < -r_clip || z > r_clip || w <= 0.0 {
                continue;
            }
            let x = self.sigma * sqrt2 * z;
            let log_p_over_q = sensitivity * (x - sensitivity / 2.0) / (self.sigma * self.sigma);
            log_interior_terms.push(w.ln() + exponent * log_p_over_q);
        }

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

        // Point mass contributions: (P/Q)^{1+t} at boundaries.
        // eps_left = log(p_L(0)/p_L(Δ)) = log(Q_left/P_left), so log(P_left/Q_left) = -eps_left
        // eps_right = log(p_R(0)/p_R(Δ)) = log(Q_right/P_right), so log(P_right/Q_right) = -eps_right
        //
        // The point mass P/Q ratio at left: P_left/Q_left = exp(-eps_left)
        // p_Q_left · (P_left/Q_left)^{1+t} = p_left · exp(-(1+t) · eps_left)
        let log_left = if self.p_left > 0.0 && self.eps_left.is_finite() {
            self.p_left.ln() - exponent * self.eps_left
        } else {
            f64::NEG_INFINITY
        };

        let log_right = if self.p_right > 0.0 && self.eps_right.is_finite() {
            self.p_right.ln() - exponent * self.eps_right
        } else {
            f64::NEG_INFINITY
        };

        let terms = [log_interior_unnorm, log_left, log_right];
        let max_t = terms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if max_t.is_infinite() && max_t < 0.0 {
            return 0.0;
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
// Mixture-of-Gaussians (MoG) CGF — tight accounting
// ---------------------------------------------------------------------------

/// Exact CGF for the Mixture-of-Gaussians privacy loss.
///
/// Models the REMOVE adjacency where:
///   P(x) = Σ_k  w_k · N(x; k, σ²)   (output when target present, sensitivity k)
///   Q(x) = N(x; 0, σ²)               (output when target absent)
///
/// Following the convention in `saddle_point_math.md`:
///   L = log(P/Q), sampled under P.
///   Λ(t) = log E_P[e^{tL}] = log E_Q[(P/Q)^{1+t}]
///
/// The second form enables Gauss-Hermite quadrature over Q = N(0, σ²).
///
/// Key property: Λ(−1) = log E_Q[(P/Q)^0] = 0 (exact, not approximate).
///
/// For Poisson subsampling (m=1): P/Q = q·exp((2x−1)/(2σ²)) + (1−q).
/// For parallel Poisson (m workers): P/Q = Σ_k Binom(m,q,k)·exp((2kx−k²)/(2σ²)).
///
/// This is **tighter** than `MixtureCgf` over `GaussianCgf` components.
#[derive(Debug, Clone)]
pub struct MogGaussianCgf {
    /// Component log-weights: log(w_k).
    log_weights: Vec<f64>,
    /// Per-component constant in P_k/Q ratio: −k² / (2σ²).
    ratio_const: Vec<f64>,
    /// Per-component linear coefficient: k√2 / σ (for GH node z, with x = σ√2z).
    ratio_linear: Vec<f64>,
    /// Gauss-Hermite nodes.
    gh_nodes: Vec<f64>,
    /// Gauss-Hermite weights.
    gh_weights: Vec<f64>,
}

impl MogGaussianCgf {
    /// Create from (sensitivity, log_weight) pairs.
    ///
    /// Each component k contributes: P_k(x)/Q(x) = exp((2kx − k²)/(2σ²)).
    pub fn new(sigma: f64, sensitivities_and_log_weights: Vec<(f64, f64)>) -> Self {
        let (nodes, weights) = gauss_hermite_nodes_weights(GH_ORDER);
        let inv_2sigma_sq = 1.0 / (2.0 * sigma * sigma);
        let sqrt2_over_sigma = std::f64::consts::SQRT_2 / sigma;

        let mut log_weights = Vec::with_capacity(sensitivities_and_log_weights.len());
        let mut ratio_const = Vec::with_capacity(sensitivities_and_log_weights.len());
        let mut ratio_linear = Vec::with_capacity(sensitivities_and_log_weights.len());

        for &(k, log_w) in &sensitivities_and_log_weights {
            log_weights.push(log_w);
            // P_k/Q at x = σ√2z: exp(k√2z/σ − k²/(2σ²))
            ratio_const.push(-k * k * inv_2sigma_sq);
            ratio_linear.push(k * sqrt2_over_sigma);
        }

        Self { log_weights, ratio_const, ratio_linear, gh_nodes: nodes, gh_weights: weights }
    }

    /// Create from Binomial(m, q) distribution over integer sensitivities 0..=m.
    pub fn from_binomial(sigma: f64, m: usize, q: f64) -> Self {
        let log_probs = binomial_log_probs_internal(m, q);
        let mut components = Vec::with_capacity(m + 1);
        for (k, &log_w) in log_probs.iter().enumerate() {
            if log_w < -300.0 { continue; }
            components.push((k as f64, log_w));
        }
        Self::new(sigma, components)
    }

    fn eval_raw(&self, t: f64) -> f64 {
        let n_comp = self.log_weights.len();
        let exponent = 1.0 + t; // Λ(t) = log E_Q[(P/Q)^{1+t}]

        let mut log_integrand: Vec<f64> = Vec::with_capacity(self.gh_nodes.len());

        for (&z, &w) in self.gh_nodes.iter().zip(self.gh_weights.iter()) {
            if w <= 0.0 { continue; }

            // log(P/Q) at node z via log-sum-exp: log(Σ_k w_k exp(const_k + linear_k·z))
            let mut log_terms: Vec<f64> = Vec::with_capacity(n_comp);
            for j in 0..n_comp {
                log_terms.push(self.log_weights[j] + self.ratio_const[j] + self.ratio_linear[j] * z);
            }

            let max_lt = log_terms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if max_lt == f64::NEG_INFINITY { continue; }
            let sum_exp: f64 = log_terms.iter().map(|&lt| (lt - max_lt).exp()).sum();
            let log_ratio = max_lt + sum_exp.ln(); // log(P(x)/Q(x))

            // log(w · (P/Q)^{1+t}) = log(w) + (1+t) · log(P/Q)
            log_integrand.push(w.ln() + exponent * log_ratio);
        }

        if log_integrand.is_empty() { return 0.0; }

        let max_log = log_integrand.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = log_integrand.iter().map(|&lt| (lt - max_log).exp()).sum();
        max_log + sum.ln() - 0.5 * std::f64::consts::PI.ln()
    }
}

impl MogGaussianCgf {
    /// Direct add-direction CGF: Λ_add(t) = log E_Q[(Q/P)^{1+t}].
    ///
    /// Q/P = 1/(Σ_k p_k exp(k√2z/σ − k²/(2σ²))) is bounded, making
    /// this numerically stable even for small component weights.
    fn eval_add_raw(&self, t: f64) -> f64 {
        let n_comp = self.log_weights.len();
        let exponent = 1.0 + t;

        let mut log_integrand: Vec<f64> = Vec::with_capacity(self.gh_nodes.len());

        for (&z, &w) in self.gh_nodes.iter().zip(self.gh_weights.iter()) {
            if w <= 0.0 { continue; }

            // log(P/Q) at this node via log-sum-exp
            let mut log_terms: Vec<f64> = Vec::with_capacity(n_comp);
            for j in 0..n_comp {
                log_terms.push(self.log_weights[j] + self.ratio_const[j] + self.ratio_linear[j] * z);
            }
            let max_lt = log_terms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if max_lt == f64::NEG_INFINITY { continue; }
            let sum_exp: f64 = log_terms.iter().map(|&lt| (lt - max_lt).exp()).sum();
            let log_p_over_q = max_lt + sum_exp.ln();

            // (Q/P)^{1+t} = exp(-(1+t) · log(P/Q))
            log_integrand.push(w.ln() - exponent * log_p_over_q);
        }

        if log_integrand.is_empty() { return 0.0; }

        let max_log = log_integrand.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = log_integrand.iter().map(|&lt| (lt - max_log).exp()).sum();
        max_log + sum.ln() - 0.5 * std::f64::consts::PI.ln()
    }
}

impl Cgf for MogGaussianCgf {
    fn eval(&self, t: f64) -> f64 { self.eval_raw(t) }
    fn eval_prime(&self, t: f64) -> f64 {
        let h = FD_STEP;
        (self.eval_raw(t + h) - self.eval_raw(t - h)) / (2.0 * h)
    }
    fn eval_double_prime(&self, t: f64) -> f64 {
        let h = FD_STEP;
        (self.eval_raw(t + h) - 2.0 * self.eval_raw(t) + self.eval_raw(t - h)) / (h * h)
    }
    fn eval_add(&self, t: f64) -> f64 {
        self.eval_add_raw(t)
    }
    fn eval_add_prime(&self, t: f64) -> f64 {
        let h = FD_STEP;
        (self.eval_add_raw(t + h) - self.eval_add_raw(t - h)) / (2.0 * h)
    }
    fn eval_add_double_prime(&self, t: f64) -> f64 {
        let h = FD_STEP;
        (self.eval_add_raw(t + h) - 2.0 * self.eval_add_raw(t) + self.eval_add_raw(t - h)) / (h * h)
    }
}

/// Compute Binomial(m, q) log-probabilities via stable recurrence.
fn binomial_log_probs_internal(m: usize, q: f64) -> Vec<f64> {
    let mut log_probs = Vec::with_capacity(m + 1);
    let log_1mq = (1.0 - q).ln();
    let log_q_ratio = (q / (1.0 - q)).ln();
    log_probs.push(m as f64 * log_1mq);
    for k in 1..=m {
        let prev = log_probs[k - 1];
        log_probs.push(prev + ((m - k + 1) as f64 / k as f64).ln() + log_q_ratio);
    }
    log_probs
}

// ---------------------------------------------------------------------------
// Subsampled Truncated Gaussian CGF
// ---------------------------------------------------------------------------

/// CGF for the Poisson-subsampled truncated Gaussian mechanism.
///
/// Following `saddle_point_math.md`:
///   Λ(t) = log E_{Q_trunc}[(P_sub/Q_trunc)^{1+t}]
///
/// where P_sub = q·P_trunc + (1−q)·Q_trunc, and
///   P_trunc/Q_trunc = exp(Δx/σ² − Δ²/(2σ²) − log(Z₁/Z₀))
///
/// (positive coefficient on x — P_trunc shifts output by +Δ).
///
/// Key property: Λ(−1) = 0.
#[derive(Debug, Clone)]
pub struct SubsampledTruncatedGaussianCgf {
    sigma: f64,
    radius: f64,
    rate: f64,
    inv_2sigma_sq: f64,
    /// Pre-computed log(P_trunc/Q_trunc) constant part: Δ²/(2σ²) + log(Z₁/Z₀).
    /// The full log-ratio is: Δx/σ² − log_ratio_const.
    log_ratio_const: f64,
    /// log(Z₀) for normalization of the GH integral over truncated domain.
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
            // log(P_trunc/Q_trunc) = Δx/σ² − Δ²/(2σ²) − log(Z₁/Z₀)
            // constant part = Δ²/(2σ²) + log(Z₁/Z₀)
            log_ratio_const: sensitivity * sensitivity / (2.0 * sigma_sq) + log_z_ratio,
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
        let exponent = 1.0 + t;

        let mut log_terms: Vec<f64> = Vec::with_capacity(self.gh_nodes.len());

        for (&z, &w) in self.gh_nodes.iter().zip(self.gh_weights.iter()) {
            if z < -r_clip || z > r_clip || w <= 0.0 {
                continue;
            }
            let x = self.sigma * sqrt2 * z;
            // log(P_trunc/Q_trunc) = Δx/σ² − const
            let log_p_over_q = sensitivity * x / (self.sigma * self.sigma)
                - self.log_ratio_const;
            // P_sub/Q = q·(P_trunc/Q_trunc) + (1−q)
            let ratio = q * log_p_over_q.exp() + one_minus_q;
            if ratio <= 0.0 {
                continue;
            }
            log_terms.push(w.ln() + exponent * ratio.ln());
        }

        if log_terms.is_empty() {
            return 0.0;
        }

        let max_log = log_terms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = log_terms.iter().map(|&lt| (lt - max_log).exp()).sum();

        // Normalize: divide by Z₀ (truncated domain) and √π (GH normalization)
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

    // --- MogGaussianCgf tests ---

    #[test]
    fn test_mog_cgf_at_zero() {
        // Λ(0) = log E_Q[(P/Q)^1] = log ∫ P = 0
        let cgf = MogGaussianCgf::from_binomial(1.0, 4, 0.01);
        assert_relative_eq!(cgf.eval(0.0), 0.0, epsilon = 1e-8);
    }

    #[test]
    fn test_mog_cgf_at_minus_one_is_zero() {
        // Λ(-1) = log E_Q[(P/Q)^0] = log 1 = 0
        let cgf = MogGaussianCgf::from_binomial(1.0, 4, 0.1);
        assert_relative_eq!(cgf.eval(-1.0), 0.0, epsilon = 1e-8);
    }

    #[test]
    fn test_mog_cgf_single_component_matches_gaussian() {
        // MoG with single component (k=1, weight=1) matches GaussianCgf
        let sigma = 0.5;
        let gauss = GaussianCgf::new(sigma);
        let mog = MogGaussianCgf::new(sigma, vec![(1.0, 0.0)]); // (sensitivity=1, log_weight=0)

        for &t in &[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0] {
            let (m, g) = (mog.eval(t), gauss.eval(t));
            assert!((m - g).abs() / g.abs().max(1.0) < 0.01,
                "Mismatch at t={}: mog={}, gauss={}", t, m, g);
        }
    }

    #[test]
    fn test_mog_cgf_positive_for_positive_t() {
        // Λ(t) ≥ 0 for t ≥ 0 (Jensen's inequality: (P/Q)^{1+t} ≥ E[(P/Q)]^{1+t} = 1)
        let cgf = MogGaussianCgf::from_binomial(1.0, 4, 0.1);
        for &t in &[0.0, 0.5, 1.0, 2.0, 5.0] {
            assert!(cgf.eval(t) >= -1e-8, "Λ({}) = {} should be >= 0", t, cgf.eval(t));
        }
    }
}
