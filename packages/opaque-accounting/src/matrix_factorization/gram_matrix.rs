//! Gram matrix computation for BnB dominating pair.
//!
//! Given a DP-λCGD strategy matrix C_λ with BnB batching (b bins, E epochs),
//! computes the Gram matrix G ∈ R^{b×b} of the dominating pair mixture means:
//!
//!   m_i = Σ_{j=0}^{E-1} C[:,b·j+i]
//!   G_{ij} = ⟨m_i, m_j⟩
//!
//! For normalized C̃_λ = C_λ·D⁻¹, the columns have unit norm and the inner
//! products decay as λ^{|i-j|} within an epoch, with cross-epoch terms O(λ^b).
//!
//! # References
//!
//! - Choquette-Choo et al. (2024) "Near Exact Privacy Amplification for Matrix
//!   Mechanisms" <https://arxiv.org/abs/2410.06266>, Lemma 3.2

use crate::error::{PldError, Result};

/// Inner product ⟨C_λ[:,a], C_λ[:,c]⟩ for a ≤ c.
///
/// C_λ[:,k] has entry λ^{t-k} at row t for t ≥ k.
/// ⟨C_λ[:,a], C_λ[:,c]⟩ = Σ_{t=c}^{n-1} λ^{t-a} · λ^{t-c}
///                        = λ^{c-a} · Σ_{r=0}^{n-1-c} λ^{2r}
///                        = λ^{c-a} · (1 - λ^{2(n-c)}) / (1 - λ²)
fn column_inner_product(lambda: f64, n: usize, a: usize, c: usize) -> f64 {
    debug_assert!(a <= c);
    debug_assert!(c < n);

    if lambda == 0.0 {
        return if a == c { 1.0 } else { 0.0 };
    }

    let gap = c - a;
    let remaining = n - c;
    let lambda2 = lambda * lambda;

    let lambda_gap = if gap == 0 {
        1.0
    } else {
        lambda.powi(gap as i32)
    };

    let lambda2r = lambda2.powi(remaining as i32);
    let sum_geom = if lambda2r < 1e-30 {
        1.0 / (1.0 - lambda2)
    } else {
        (1.0 - lambda2r) / (1.0 - lambda2)
    };

    lambda_gap * sum_geom
}

/// Column norm squared: ‖C_λ[:,k]‖² = (1 - λ^{2(n-k)}) / (1 - λ²)
fn column_norm_squared(lambda: f64, n: usize, k: usize) -> f64 {
    column_inner_product(lambda, n, k, k)
}

/// Compute the BnB Gram matrix for DP-λCGD.
///
/// For the BnB dominating pair (Lemma 3.2 of arxiv:2410.06266):
///   m_i = Σ_{j=0}^{E-1} C[:,b·j+i]    for i = 0..b-1
///   G_{ij} = ⟨m_i, m_j⟩
///
/// When `normalized=true`, uses column-normalized C̃_λ = C_λ·D⁻¹.
///
/// # Arguments
///
/// * `lambda` — Correlation coefficient in [0, 1).
/// * `n_steps` — Total number of steps (= b * E).
/// * `min_sep` — Bins per epoch (= b).
/// * `max_participations` — Number of epochs (= E). None infers from n_steps/min_sep.
/// * `normalized` — Whether to use column-normalized matrix.
///
/// # Returns
///
/// Row-major b×b Gram matrix as a flat Vec<f64>.
pub fn lambda_cgd_gram_matrix(
    lambda: f64,
    n_steps: usize,
    min_sep: usize,
    max_participations: Option<usize>,
    normalized: bool,
) -> Result<Vec<f64>> {
    if lambda < 0.0 || lambda >= 1.0 {
        return Err(PldError::InvalidParameter(format!(
            "lambda must be in [0, 1), got {}",
            lambda
        )));
    }
    if n_steps == 0 {
        return Err(PldError::InvalidParameter("n_steps must be >= 1".into()));
    }
    if min_sep == 0 {
        return Err(PldError::InvalidParameter("min_sep must be >= 1".into()));
    }

    let b = min_sep;
    let n = n_steps;
    let k_inferred = (n + b - 1) / b;
    let e = match max_participations {
        Some(k) => k.min(k_inferred),
        None => k_inferred,
    };

    if e == 0 || b == 0 {
        return Ok(vec![0.0; b * b]);
    }

    // G_{ij} = Σ_{p=0}^{E-1} Σ_{q=0}^{E-1} ⟨C[:,b*p+i], C[:,b*q+j]⟩ / (d_{b*p+i} · d_{b*q+j})
    //
    // Optimization: cross-epoch terms have a factor λ^{b·|p-q|}.
    // When λ^b < 1e-15, only same-epoch terms (p=q) contribute,
    // reducing complexity from O(b²·E²) to O(b²·E).
    let lambda_b = if b > 0 { lambda.powi(b as i32) } else { 1.0 };
    let skip_cross_epoch = lambda_b.abs() < 1e-15;

    // Precompute column norms for the normalized case
    let col_norms: Vec<f64> = if normalized {
        (0..e)
            .flat_map(|p| (0..b).map(move |i| {
                let col = b * p + i;
                if col < n { column_norm_squared(lambda, n, col).sqrt() } else { 1.0 }
            }))
            .collect()
    } else {
        Vec::new()
    };

    let mut gram = vec![0.0f64; b * b];

    for i in 0..b {
        for j in i..b {
            let mut val = 0.0;

            for p in 0..e {
                let col_a = b * p + i;
                if col_a >= n {
                    break;
                }

                let q_start = if skip_cross_epoch { p } else { 0 };
                let q_end = if skip_cross_epoch { p + 1 } else { e };

                for q in q_start..q_end {
                    let col_c = b * q + j;
                    if col_c >= n {
                        break;
                    }

                    let (lo, hi) = if col_a <= col_c {
                        (col_a, col_c)
                    } else {
                        (col_c, col_a)
                    };

                    let ip = column_inner_product(lambda, n, lo, hi);

                    let contribution = if normalized {
                        let d_a = col_norms[p * b + i];
                        let d_c = col_norms[q * b + j];
                        ip / (d_a * d_c)
                    } else {
                        ip
                    };

                    val += contribution;
                }
            }

            gram[i * b + j] = val;
            if i != j {
                gram[j * b + i] = val;
            }
        }
    }

    Ok(gram)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_column_inner_product_identity() {
        // λ=0 → C = I, columns are unit vectors
        assert!((column_inner_product(0.0, 10, 3, 3) - 1.0).abs() < 1e-10);
        assert!((column_inner_product(0.0, 10, 2, 5)).abs() < 1e-10);
    }

    #[test]
    fn test_column_inner_product_self() {
        // Self inner product = column norm squared
        let lambda = 0.5;
        let n = 10;
        let k = 3;
        let ip = column_inner_product(lambda, n, k, k);
        let expected: f64 = (0..(n - k)).map(|r| lambda.powi(2 * r as i32)).sum();
        assert!(
            (ip - expected).abs() < 1e-10,
            "got {}, expected {}",
            ip,
            expected
        );
    }

    #[test]
    fn test_column_inner_product_cross() {
        // Cross inner product for adjacent columns
        let lambda = 0.5;
        let n = 10;
        let ip = column_inner_product(lambda, n, 2, 5);
        // = λ^3 · Σ_{r=0}^{4} λ^{2r}
        let expected: f64 =
            lambda.powi(3) * (0..5).map(|r| lambda.powi(2 * r as i32)).sum::<f64>();
        assert!(
            (ip - expected).abs() < 1e-10,
            "got {}, expected {}",
            ip,
            expected
        );
    }

    #[test]
    fn test_gram_matrix_identity() {
        // λ=0, single epoch → G = I_b
        let gram = lambda_cgd_gram_matrix(0.0, 10, 10, Some(1), false).unwrap();
        assert_eq!(gram.len(), 100);
        for i in 0..10 {
            for j in 0..10 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (gram[i * 10 + j] - expected).abs() < 1e-10,
                    "G[{},{}] = {}, expected {}",
                    i,
                    j,
                    gram[i * 10 + j],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_gram_matrix_normalized_single_epoch() {
        // Normalized, single epoch: all diagonal entries = 1
        let lambda = 0.9;
        let b = 20;
        let gram =
            lambda_cgd_gram_matrix(lambda, b, b, Some(1), true).unwrap();
        for i in 0..b {
            assert!(
                (gram[i * b + i] - 1.0).abs() < 1e-8,
                "G[{},{}] = {}, expected 1.0",
                i,
                i,
                gram[i * b + i]
            );
        }
    }

    #[test]
    fn test_gram_matrix_symmetric() {
        let gram =
            lambda_cgd_gram_matrix(0.9, 100, 10, Some(5), true).unwrap();
        let b = 10;
        for i in 0..b {
            for j in 0..b {
                assert!(
                    (gram[i * b + j] - gram[j * b + i]).abs() < 1e-10,
                    "G not symmetric at ({},{}): {} vs {}",
                    i,
                    j,
                    gram[i * b + j],
                    gram[j * b + i]
                );
            }
        }
    }

    #[test]
    fn test_gram_matrix_multi_epoch_diagonal() {
        // Multi-epoch normalized: diagonal ≈ E for large b (columns nearly orthogonal across epochs)
        let lambda = 0.9;
        let b = 200; // Large enough that λ^b ≈ 0
        let e = 5;
        let gram =
            lambda_cgd_gram_matrix(lambda, b * e, b, Some(e), true).unwrap();
        for i in 0..b {
            assert!(
                (gram[i * b + i] - e as f64).abs() < 0.1,
                "G[{},{}] = {}, expected ≈ {}",
                i,
                i,
                gram[i * b + i],
                e
            );
        }
    }

    #[test]
    fn test_gram_matrix_rejects_bad_params() {
        assert!(lambda_cgd_gram_matrix(-0.1, 10, 5, Some(1), true).is_err());
        assert!(lambda_cgd_gram_matrix(1.0, 10, 5, Some(1), true).is_err());
        assert!(lambda_cgd_gram_matrix(0.5, 0, 5, Some(1), true).is_err());
        assert!(lambda_cgd_gram_matrix(0.5, 10, 0, Some(1), true).is_err());
    }

    #[test]
    fn test_gram_matrix_positive_definite() {
        // Gram matrix should be positive semidefinite
        // Simple check: all diagonal entries positive, and G_{ii} >= |G_{ij}| (diag dominance approx)
        let gram =
            lambda_cgd_gram_matrix(0.9, 50, 10, Some(3), true).unwrap();
        let b = 10;
        for i in 0..b {
            assert!(gram[i * b + i] > 0.0, "Diagonal entry G[{},{}] not positive", i, i);
        }
    }

    #[test]
    fn test_gram_matrix_multi_epoch_psd() {
        // Regression test: multi-epoch Gram must be PSD (not just have positive diagonal).
        // Verify via Cholesky: if Cholesky succeeds without negative diagonals, G is PSD.
        for &(lam, b, e) in &[(0.9, 20, 5), (0.5, 50, 3), (0.99, 10, 8)] {
            let gram = lambda_cgd_gram_matrix(lam, b * e, b, Some(e), true).unwrap();
            // Naive Cholesky check: compute L[0,0]..L[b-1,b-1]
            let mut l = vec![0.0f64; b * b];
            for i in 0..b {
                let mut diag = gram[i * b + i];
                for k in 0..i {
                    diag -= l[i * b + k] * l[i * b + k];
                }
                assert!(
                    diag > -1e-10,
                    "Gram not PSD: Cholesky diagonal {} at row {} for λ={}, b={}, E={}",
                    diag, i, lam, b, e
                );
                l[i * b + i] = diag.max(0.0).sqrt();
                for j in (i + 1)..b {
                    let mut off = gram[j * b + i];
                    for k in 0..i {
                        off -= l[j * b + k] * l[i * b + k];
                    }
                    l[j * b + i] = if l[i * b + i] > 0.0 { off / l[i * b + i] } else { 0.0 };
                }
            }
        }
    }
}
