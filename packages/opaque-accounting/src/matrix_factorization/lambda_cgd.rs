//! DP-λCGD sensitivity computation.
//!
//! The DP-λCGD mechanism (Kalinin et al., 2026) uses a lower-triangular
//! Toeplitz strategy matrix C_λ with entries `(C_λ)_{ij} = λ^{i-j}` for
//! `i ≥ j` and 0 otherwise. Its inverse is bidiagonal: 1 on the diagonal,
//! -λ on the subdiagonal.
//!
//! # Sensitivity (Theorem 1, eq 15)
//!
//! For a participation pattern with `k` participations and minimum
//! separation `b`, the squared L2 sensitivity is:
//!
//! ```text
//! sens²_{k,b}(C_λ) = (1 - λ^{2b}) / ((1 - λ²)(1 - λ^b)²) · Σ_{j=0}^{k-1} (1 - λ^{b(j+1)})²
//! ```
//!
//! # References
//!
//! - Kalinin et al. (2026) "DP-λCGD: Leveraging Correlated Gradients for
//!   Improved DP-SGD" <https://arxiv.org/abs/2601.22334>

use crate::error::{PldError, Result};

/// Compute the squared L2 sensitivity of the DP-λCGD strategy matrix.
///
/// Uses the closed-form expression from Theorem 1 (eq 15) of the paper.
///
/// # Arguments
///
/// * `lambda` — Correlation coefficient in [0, 1). λ=0 is DP-SGD.
/// * `n_steps` — Total number of training steps.
/// * `min_sep` — Minimum separation between participations (≥ 1).
/// * `max_participations` — Maximum number of participations per user.
///   If `None`, inferred from `n_steps / min_sep`.
///
/// # Returns
///
/// The squared L2 sensitivity S².
pub fn lambda_cgd_sensitivity_squared(
    lambda: f64,
    n_steps: usize,
    min_sep: usize,
    max_participations: Option<usize>,
) -> Result<f64> {
    if lambda < 0.0 || lambda >= 1.0 {
        return Err(PldError::InvalidParameter(format!(
            "lambda must be in [0, 1), got {}",
            lambda
        )));
    }
    if n_steps == 0 {
        return Err(PldError::InvalidParameter(
            "n_steps must be >= 1".to_string(),
        ));
    }
    if min_sep == 0 {
        return Err(PldError::InvalidParameter(
            "min_sep must be >= 1".to_string(),
        ));
    }

    let b = min_sep;
    // True max participations given the constraints
    let k_inferred = (n_steps + b - 1) / b; // ceiling division
    let k = match max_participations {
        Some(max_k) => max_k.min(k_inferred),
        None => k_inferred,
    };

    if k == 0 {
        return Ok(0.0);
    }

    // λ=0 (DP-SGD): C_λ = I, each column has norm 1
    // Sensitivity = sqrt(k) for k participations
    if lambda == 0.0 {
        return Ok(k as f64);
    }

    // For numerical stability with large exponents, compute λ^b carefully.
    // When b is large enough that λ^b ≈ 0 (e.g. λ=0.9, b=1953),
    // the formula simplifies to k / (1 - λ²).
    let lambda_b = lambda.powi(b as i32);

    // If λ^b is negligibly small, use the simplified formula
    if lambda_b < 1e-15 {
        let lambda2 = lambda * lambda;
        return Ok(k as f64 / (1.0 - lambda2));
    }

    // Full formula: sens² = (1-λ^{2b}) / ((1-λ²)(1-λ^b)²) · Σ (1-λ^{b(j+1)})²
    let lambda2 = lambda * lambda;
    let lambda_2b = lambda_b * lambda_b; // λ^{2b}

    let sum_r = (1.0 - lambda_2b) / (1.0 - lambda2);
    let denom = (1.0 - lambda_b) * (1.0 - lambda_b);

    let mut sum_j = 0.0;
    let mut lambda_bj = lambda_b; // λ^{b·1} initially
    for _j in 0..k {
        let term = 1.0 - lambda_bj;
        sum_j += term * term;
        lambda_bj *= lambda_b; // λ^{b·(j+2)}, etc.
    }

    Ok(sum_r * sum_j / denom)
}

/// Compute the squared L2 sensitivity of the column-normalized DP-λCGD.
///
/// Column normalization: C̃_λ = C_λ · D⁻¹ where D = diag(‖C_λ[:,j]‖).
/// All columns of C̃_λ have unit norm.
///
/// For single participation (k=1), the sensitivity is always 1.0.
/// For multi-participation (k > 1) with min-separation b, computes
/// the Gram matrix from Lemma 8 of the paper.
///
/// # Arguments
///
/// * `lambda` — Correlation coefficient in [0, 1).
/// * `n_steps` — Total number of training steps.
/// * `min_sep` — Minimum separation between participations (≥ 1).
/// * `max_participations` — Maximum number of participations per user.
///   If `None`, inferred from `n_steps / min_sep`.
///
/// # Returns
///
/// The squared L2 sensitivity S² of the column-normalized matrix.
pub fn lambda_cgd_normalized_sensitivity_squared(
    lambda: f64,
    n_steps: usize,
    min_sep: usize,
    max_participations: Option<usize>,
) -> Result<f64> {
    if lambda < 0.0 || lambda >= 1.0 {
        return Err(PldError::InvalidParameter(format!(
            "lambda must be in [0, 1), got {}",
            lambda
        )));
    }
    if n_steps == 0 {
        return Err(PldError::InvalidParameter(
            "n_steps must be >= 1".to_string(),
        ));
    }
    if min_sep == 0 {
        return Err(PldError::InvalidParameter(
            "min_sep must be >= 1".to_string(),
        ));
    }

    let b = min_sep;
    let k_inferred = (n_steps + b - 1) / b;
    let k = match max_participations {
        Some(max_k) => max_k.min(k_inferred),
        None => k_inferred,
    };

    if k == 0 {
        return Ok(0.0);
    }

    // All columns have unit norm → single participation sensitivity = 1
    if k == 1 {
        return Ok(1.0);
    }

    // λ=0: C_0 = I, columns are unit vectors, sens = sqrt(k)
    if lambda == 0.0 {
        return Ok(k as f64);
    }

    let lambda2 = lambda * lambda;
    let lambda_b = lambda.powi(b as i32);

    // Column-norm squared: d²(col_j) = (1 - λ^{2(n - j*b)}) / (1 - λ²)
    // for participation column j*b (0-indexed), j = 0..k-1
    let d_sq: Vec<f64> = (0..k)
        .map(|j| {
            let remaining = n_steps - j * b;
            let lambda2r = lambda2.powi(remaining as i32);
            if lambda2r < 1e-30 {
                1.0 / (1.0 - lambda2)
            } else {
                (1.0 - lambda2r) / (1.0 - lambda2)
            }
        })
        .collect();

    // sens² = k + 2 · Σ_{j<j'} λ^{(j'-j)b} · d(col_{j'}) / d(col_j)
    // where ⟨C̃[:,col_j], C̃[:,col_{j'}]⟩ = λ^{(j'-j)b} · d(col_{j'}) / d(col_j)
    let mut sens_sq = k as f64; // diagonal terms (each column has norm 1)

    for j in 0..k {
        let d_j = d_sq[j].sqrt();
        let mut lambda_power = lambda_b; // λ^b for the first cross-term
        for jp in (j + 1)..k {
            let d_jp = d_sq[jp].sqrt();
            sens_sq += 2.0 * lambda_power * d_jp / d_j;
            lambda_power *= lambda_b;
        }
    }

    Ok(sens_sq)
}

/// Compute the max column norm of the DP-λCGD strategy matrix.
///
/// The column at position 0 has the largest norm, equal to
/// `sqrt((1 - λ^{2n}) / (1 - λ²))`.
///
/// # Arguments
///
/// * `lambda` — Correlation coefficient in [0, 1).
/// * `n_steps` — Total number of steps.
///
/// # Returns
///
/// The max column L2 norm.
pub fn lambda_cgd_max_column_norm(lambda: f64, n_steps: usize) -> Result<f64> {
    let sens_sq = lambda_cgd_sensitivity_squared(lambda, n_steps, n_steps, Some(1))?;
    Ok(sens_sq.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lambda_zero_is_dpsgd() {
        // λ=0 → C_λ = I, sensitivity = sqrt(k)
        let sens_sq = lambda_cgd_sensitivity_squared(0.0, 100, 10, Some(3)).unwrap();
        assert!((sens_sq - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_single_participation() {
        // k=1: sens² = Σ_{r=0}^{n-1} λ^{2r} = (1-λ^{2n})/(1-λ²)
        let lambda = 0.5;
        let n = 10;
        let sens_sq =
            lambda_cgd_sensitivity_squared(lambda, n, n, Some(1)).unwrap();
        let expected: f64 = (0..n).map(|r| lambda.powi(2 * r as i32)).sum();
        assert!(
            (sens_sq - expected).abs() < 1e-10,
            "got {}, expected {}",
            sens_sq,
            expected
        );
    }

    #[test]
    fn test_sensitivity_increases_with_participations() {
        let lambda = 0.9;
        let prev = lambda_cgd_sensitivity_squared(lambda, 100, 10, Some(1)).unwrap();
        for k in 2..=5 {
            let curr = lambda_cgd_sensitivity_squared(lambda, 100, 10, Some(k)).unwrap();
            assert!(curr > prev, "k={}: {} should be > {}", k, curr, prev);
        }
    }

    #[test]
    fn test_rejects_invalid_lambda() {
        assert!(lambda_cgd_sensitivity_squared(-0.1, 10, 1, None).is_err());
        assert!(lambda_cgd_sensitivity_squared(1.0, 10, 1, None).is_err());
        assert!(lambda_cgd_sensitivity_squared(1.5, 10, 1, None).is_err());
    }

    #[test]
    fn test_max_column_norm() {
        let lambda = 0.5;
        let n = 20;
        let norm = lambda_cgd_max_column_norm(lambda, n).unwrap();
        let expected: f64 = ((0..n).map(|r| (lambda * lambda).powi(r as i32)).sum::<f64>()).sqrt();
        assert!((norm - expected).abs() < 1e-10);
    }

    #[test]
    fn test_large_b_simplified_formula() {
        // When λ^b ≈ 0, sens² ≈ k / (1 - λ²)
        let lambda = 0.9;
        let b = 2000; // λ^b ≈ 0
        let k = 5;
        let n = k * b;
        let sens_sq =
            lambda_cgd_sensitivity_squared(lambda, n, b, Some(k)).unwrap();
        let expected = k as f64 / (1.0 - lambda * lambda);
        assert!(
            (sens_sq - expected).abs() / expected < 1e-6,
            "got {}, expected {}",
            sens_sq,
            expected
        );
    }

    // ── Column-normalized sensitivity tests ──────────────────────

    #[test]
    fn test_normalized_single_participation_is_one() {
        // k=1: all columns have unit norm → sensitivity = 1
        for lambda in [0.0, 0.3, 0.5, 0.9, 0.99] {
            let sens_sq = lambda_cgd_normalized_sensitivity_squared(
                lambda, 100, 100, Some(1),
            )
            .unwrap();
            assert!(
                (sens_sq - 1.0).abs() < 1e-10,
                "λ={}: normalized sens² should be 1.0, got {}",
                lambda,
                sens_sq
            );
        }
    }

    #[test]
    fn test_normalized_lambda_zero_is_k() {
        // λ=0: columns are orthogonal unit vectors → sens² = k
        let sens_sq = lambda_cgd_normalized_sensitivity_squared(
            0.0, 100, 10, Some(3),
        )
        .unwrap();
        assert!(
            (sens_sq - 3.0).abs() < 1e-10,
            "got {}, expected 3.0",
            sens_sq
        );
    }

    #[test]
    fn test_normalized_leq_unnormalized() {
        // Normalized sensitivity should be ≤ unnormalized (improvement)
        let lambda = 0.9;
        for k in [1, 2, 3, 5] {
            let unnorm = lambda_cgd_sensitivity_squared(
                lambda, 100, 10, Some(k),
            )
            .unwrap();
            let norm = lambda_cgd_normalized_sensitivity_squared(
                lambda, 100, 10, Some(k),
            )
            .unwrap();
            assert!(
                norm <= unnorm + 1e-10,
                "k={}: normalized {} should be ≤ unnormalized {}",
                k, norm, unnorm
            );
        }
    }

    #[test]
    fn test_normalized_large_b_approx_k() {
        // When λ^b ≈ 0, all column norms ≈ 1/√(1-λ²) → ratios ≈ 1
        // so normalized sens² ≈ k
        let lambda = 0.9;
        let b = 2000;
        let k = 5;
        let n = k * b;
        let sens_sq = lambda_cgd_normalized_sensitivity_squared(
            lambda, n, b, Some(k),
        )
        .unwrap();
        assert!(
            (sens_sq - k as f64).abs() < 0.01,
            "got {}, expected {} (large b limit)",
            sens_sq,
            k
        );
    }

    #[test]
    fn test_normalized_rejects_invalid_params() {
        assert!(lambda_cgd_normalized_sensitivity_squared(-0.1, 10, 1, None).is_err());
        assert!(lambda_cgd_normalized_sensitivity_squared(1.0, 10, 1, None).is_err());
        assert!(lambda_cgd_normalized_sensitivity_squared(0.5, 0, 1, None).is_err());
        assert!(lambda_cgd_normalized_sensitivity_squared(0.5, 10, 0, None).is_err());
    }
}
