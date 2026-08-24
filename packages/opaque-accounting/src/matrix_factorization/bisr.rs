//! BISR (Banded Inverse Square Root) sensitivity and Gram matrix computation.
//!
//! BISR (Kalinin et al., ICLR 2026) generalises DP-λCGD to arbitrary
//! bandwidth p ≥ 2. The strategy matrix C = A^{1/2} (square root of the
//! workload), and C^{-1} = A^{-1/2} is a lower-triangular Toeplitz matrix
//! truncated to bandwidth p.
//!
//! Given the banded C^{-1} coefficients `[c̃_0, c̃_1, ..., c̃_{p-1}]`,
//! the columns of C satisfy the recurrence:
//!
//! ```text
//! C[j, j] = 1 / c̃_0
//! C[t, j] = -Σ_{k=1}^{min(t-j, p-1)} c̃_k · C[t-k, j] / c̃_0   for t > j
//! ```
//!
//! For p=2 with c̃ = [1, -λ], this gives C[t,j] = λ^{t-j} (standard λCGD).
//!
//! # References
//!
//! - Kalinin, McKenna, Upadhyay, Lampert (2026) "Back to Square Roots:
//!   Banded Inverse Square Root for DP Matrix Factorization"
//!   <https://arxiv.org/abs/2505.12128>

use crate::error::{PldError, Result};

/// Build column 0 of the strategy matrix C from banded C^{-1} coefficients.
///
/// Due to the Toeplitz structure, column j has entries identical to column 0
/// shifted by j positions (and truncated to n-j entries). So we only ever
/// need column 0 of length n.
///
/// Returns a Vec of length `len` containing C[0..len, 0].
pub fn bisr_column_zero_pub(coefficients: &[f64], len: usize) -> Vec<f64> {
    bisr_column_zero(coefficients, len)
}

fn bisr_column_zero(coefficients: &[f64], len: usize) -> Vec<f64> {
    if len == 0 {
        return Vec::new();
    }
    let p = coefficients.len();
    let alpha0 = coefficients[0];
    debug_assert!(alpha0.abs() > 1e-30, "c̃_0 must be non-zero");

    let mut col = vec![0.0f64; len];
    col[0] = 1.0 / alpha0;

    for t in 1..len {
        let mut s = 0.0;
        let k_max = t.min(p - 1);
        for k in 1..=k_max {
            s += coefficients[k] * col[t - k];
        }
        col[t] = -s / alpha0;
    }
    col
}

/// Inner product of two BISR columns with momentum accumulation.
///
/// Column `a` has entries `col0[0..n-a]` (shifted from column 0).
/// Column `c` has entries `col0[0..n-c]` (with c ≥ a).
///
/// With momentum β:
///   m_j^β[t] = Σ_{s=j}^{t} β^{t-s} · C[s, j]
///
/// The inner product is `⟨m_a^β, m_c^β⟩ = Σ_{t=c}^{n-1} m_a[t] · m_c[t]`.
///
/// Due to Toeplitz structure, this depends only on `gap = c - a` and
/// `remaining = n - c`. We exploit this in the Gram matrix computation.
fn bisr_column_inner_product_momentum(
    col0: &[f64],
    momentum: f64,
    n: usize,
    a: usize,
    c: usize,
) -> f64 {
    debug_assert!(a <= c);
    debug_assert!(c < n);

    let remaining = n - c;
    let gap = c - a;

    if momentum == 0.0 {
        // No momentum: direct column inner product
        // ⟨C[:,a], C[:,c]⟩ = Σ_{t=c}^{n-1} col0[t-a] · col0[t-c]
        //                   = Σ_{u=0}^{remaining-1} col0[u+gap] · col0[u]
        let mut dot = 0.0;
        for u in 0..remaining {
            let idx_a = u + gap;
            if idx_a >= col0.len() || u >= col0.len() {
                break;
            }
            dot += col0[idx_a] * col0[u];
        }
        return dot;
    }

    // With momentum: accumulate m_a[t] and m_c[t] incrementally.
    // m_j[t] = β · m_j[t-1] + C[t, j]  for t ≥ j, else 0
    //
    // We need m_a[t] for t = c..n-1 and m_c[t] for t = c..n-1.
    // m_a starts accumulating at t = a (gap steps before m_c).
    // m_c starts accumulating at t = c.

    // First, accumulate m_a from t=a to t=c-1 (gap steps, before the dot starts)
    let mut acc_a = 0.0;
    for t in 0..gap {
        // C[t+a, a] = col0[t]
        if t < col0.len() {
            acc_a = momentum * acc_a + col0[t];
        } else {
            acc_a *= momentum;
        }
    }

    // Now accumulate both and dot from t=c to t=n-1
    let mut acc_c = 0.0;
    let mut dot = 0.0;
    for u in 0..remaining {
        let idx_a = u + gap;
        let ca = if idx_a < col0.len() { col0[idx_a] } else { 0.0 };
        let cc = if u < col0.len() { col0[u] } else { 0.0 };

        acc_a = momentum * acc_a + ca;
        acc_c = momentum * acc_c + cc;

        dot += acc_a * acc_c;
    }

    dot
}

fn validate_coefficients(coefficients: &[f64]) -> Result<()> {
    if coefficients.len() < 2 {
        return Err(PldError::InvalidParameter(format!(
            "BISR bandwidth must be >= 2 (coefficients length), got {}",
            coefficients.len()
        )));
    }
    if coefficients[0].abs() < 1e-30 {
        return Err(PldError::InvalidParameter(
            "c̃_0 (first coefficient) must be non-zero".into(),
        ));
    }
    Ok(())
}

fn validate_common(
    coefficients: &[f64],
    n_steps: usize,
    min_sep: usize,
    momentum: f64,
) -> Result<()> {
    validate_coefficients(coefficients)?;
    if n_steps == 0 {
        return Err(PldError::InvalidParameter("n_steps must be >= 1".into()));
    }
    if min_sep == 0 {
        return Err(PldError::InvalidParameter("min_sep must be >= 1".into()));
    }
    if !(0.0..1.0).contains(&momentum) {
        return Err(PldError::InvalidParameter(format!(
            "momentum must be in [0, 1), got {}",
            momentum
        )));
    }
    Ok(())
}

fn effective_k(n_steps: usize, min_sep: usize, max_participations: Option<usize>) -> usize {
    let k_inferred = n_steps.div_ceil(min_sep);
    match max_participations {
        Some(max_k) => max_k.min(k_inferred),
        None => k_inferred,
    }
}

/// Sensitivity squared for BISR under min-sep participation pattern.
///
/// S² = max_j ‖Σ_{p=0}^{k-1} m_{j+p·b}‖² = Σ_{p,q} ⟨m_{p·b}, m_{q·b}⟩
///
/// Worst case is j=0 (earliest start position, longest columns).
pub fn bisr_sensitivity_squared(
    coefficients: &[f64],
    n_steps: usize,
    min_sep: usize,
    max_participations: Option<usize>,
    momentum: f64,
) -> Result<f64> {
    validate_common(coefficients, n_steps, min_sep, momentum)?;

    let b = min_sep;
    let n = n_steps;
    let k = effective_k(n, b, max_participations);

    if k == 0 {
        return Ok(0.0);
    }

    // Build column 0 (reused for all inner products via Toeplitz structure)
    let col0 = bisr_column_zero(coefficients, n);

    let mut sens_sq = 0.0;
    for p in 0..k {
        let col_p = p * b;
        if col_p >= n {
            break;
        }
        for q in 0..k {
            let col_q = q * b;
            if col_q >= n {
                break;
            }
            let (lo, hi) = if col_p <= col_q {
                (col_p, col_q)
            } else {
                (col_q, col_p)
            };
            sens_sq += bisr_column_inner_product_momentum(&col0, momentum, n, lo, hi);
        }
    }
    Ok(sens_sq)
}

/// Normalized sensitivity squared for BISR (column-normalized).
///
/// All columns are divided by their L2 norm, so single participation (k=1)
/// always gives sensitivity = 1.
pub fn bisr_normalized_sensitivity_squared(
    coefficients: &[f64],
    n_steps: usize,
    min_sep: usize,
    max_participations: Option<usize>,
    momentum: f64,
) -> Result<f64> {
    validate_common(coefficients, n_steps, min_sep, momentum)?;

    let b = min_sep;
    let n = n_steps;
    let k = effective_k(n, b, max_participations);

    if k == 0 {
        return Ok(0.0);
    }
    if k == 1 {
        return Ok(1.0);
    }

    let col0 = bisr_column_zero(coefficients, n);

    // Column norms (with momentum)
    let col_norms: Vec<f64> = (0..k)
        .map(|j| {
            let col = j * b;
            if col >= n {
                return 1.0;
            }
            bisr_column_inner_product_momentum(&col0, momentum, n, col, col).sqrt()
        })
        .collect();

    let mut sens_sq = 0.0;
    for p in 0..k {
        let col_p = p * b;
        if col_p >= n {
            break;
        }
        for q in 0..k {
            let col_q = q * b;
            if col_q >= n {
                break;
            }
            let (lo, hi) = if col_p <= col_q {
                (col_p, col_q)
            } else {
                (col_q, col_p)
            };
            let ip = bisr_column_inner_product_momentum(&col0, momentum, n, lo, hi);
            sens_sq += ip / (col_norms[p] * col_norms[q]);
        }
    }
    Ok(sens_sq)
}

/// BnB Gram matrix for BISR with optional momentum.
///
/// G_{ij} = ⟨m_i, m_j⟩ (or normalized) where m_i aggregates across epochs.
pub fn bisr_gram_matrix(
    coefficients: &[f64],
    n_steps: usize,
    min_sep: usize,
    max_participations: Option<usize>,
    normalized: bool,
    momentum: f64,
) -> Result<Vec<f64>> {
    bisr_prefix_gram_matrix(
        coefficients,
        n_steps,
        n_steps,
        min_sep,
        max_participations,
        normalized,
        momentum,
    )
}

/// Compute a released prefix using column norms from the deployed horizon.
pub(crate) fn bisr_prefix_gram_matrix(
    coefficients: &[f64],
    prefix_steps: usize,
    normalization_steps: usize,
    min_sep: usize,
    max_participations: Option<usize>,
    normalized: bool,
    momentum: f64,
) -> Result<Vec<f64>> {
    validate_common(coefficients, prefix_steps, min_sep, momentum)?;
    if normalization_steps < prefix_steps {
        return Err(PldError::InvalidParameter(format!(
            "normalization_steps ({normalization_steps}) must be >= prefix_steps ({prefix_steps})"
        )));
    }

    let b = min_sep;
    let n = prefix_steps;
    let k_inferred = n.div_ceil(b);
    let e = match max_participations {
        Some(k) => k.min(k_inferred),
        None => k_inferred,
    };

    if e == 0 || b == 0 {
        return Ok(vec![0.0; b * b]);
    }

    let col0 = bisr_column_zero(
        coefficients,
        if normalized {
            normalization_steps
        } else {
            prefix_steps
        },
    );

    // Precompute column norms for normalization
    let col_norms: Vec<f64> = if normalized {
        let mut norms = Vec::with_capacity(e * b);
        for p in 0..e {
            for i in 0..b {
                let col = b * p + i;
                if col < n {
                    norms.push(
                        bisr_column_inner_product_momentum(
                            &col0,
                            momentum,
                            normalization_steps,
                            col,
                            col,
                        )
                        .sqrt(),
                    );
                } else {
                    norms.push(1.0);
                }
            }
        }
        norms
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

                for q in 0..e {
                    let col_c = b * q + j;
                    if col_c >= n {
                        break;
                    }

                    let (lo, hi) = if col_a <= col_c {
                        (col_a, col_c)
                    } else {
                        (col_c, col_a)
                    };

                    let ip = bisr_column_inner_product_momentum(&col0, momentum, n, lo, hi);

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

/// BnB Gram matrix with LR-schedule weighting (numerical rank-1 updates).
///
/// The effective column for bin i at step t uses a ring buffer of p-1
/// recent column values for the banded recurrence.
#[allow(clippy::needless_range_loop)]
pub fn bisr_gram_matrix_lr(
    coefficients: &[f64],
    momentum: f64,
    n_steps: usize,
    min_sep: usize,
    max_participations: Option<usize>,
    normalized: bool,
    lr_weights: &[f64],
) -> Result<Vec<f64>> {
    validate_common(coefficients, n_steps, min_sep, momentum)?;
    if lr_weights.len() != n_steps {
        return Err(PldError::InvalidParameter(format!(
            "lr_weights length ({}) must equal n_steps ({})",
            lr_weights.len(),
            n_steps
        )));
    }

    let p = coefficients.len();
    let b = min_sep;
    let n = n_steps;
    let k_inferred = n.div_ceil(b);
    let e = match max_participations {
        Some(k) => k.min(k_inferred),
        None => k_inferred,
    };

    if e == 0 || b == 0 {
        return Ok(vec![0.0; b * b]);
    }

    let alpha0 = coefficients[0];
    let beta = momentum;

    // For each bin i and epoch ep, maintain:
    // - ring buffer of p-1 recent column values (for the recurrence)
    // - momentum accumulator
    // Total state: b * e * p floats
    let state_size = b * e;
    let mut ring_bufs: Vec<Vec<f64>> = vec![vec![0.0; p - 1]; state_size];
    let mut mom_accs: Vec<f64> = vec![0.0; state_size];

    let mut gram = vec![0.0f64; b * b];
    let mut col_norm_sq = vec![0.0f64; b];

    for t in 0..n {
        let lr_t = lr_weights[t];
        let mut v = vec![0.0f64; b];

        for i in 0..b {
            let mut sum_acc = 0.0;
            for ep in 0..e {
                let col_start = i + ep * b;
                if col_start > t || col_start >= n {
                    continue;
                }
                let idx = i * e + ep;
                let steps_since = t - col_start;

                // Compute the new column entry via recurrence
                let new_val = if steps_since == 0 {
                    1.0 / alpha0
                } else {
                    let mut s = 0.0;
                    let k_max = steps_since.min(p - 1);
                    for k in 1..=k_max {
                        let buf_idx = (k - 1) % (p - 1);
                        s += coefficients[k] * ring_bufs[idx][buf_idx];
                    }
                    -s / alpha0
                };

                // Update ring buffer (shift and insert)
                if p > 2 {
                    // Shift: move entries forward
                    for k in (1..p - 1).rev() {
                        ring_bufs[idx][k] = ring_bufs[idx][k - 1];
                    }
                }
                if p >= 2 {
                    ring_bufs[idx][0] = new_val;
                }

                // Update momentum accumulator
                mom_accs[idx] = beta * mom_accs[idx] + new_val;
                sum_acc += mom_accs[idx];
            }
            v[i] = lr_t * sum_acc;
        }

        // Rank-1 update
        for i in 0..b {
            if v[i] == 0.0 {
                continue;
            }
            col_norm_sq[i] += v[i] * v[i];
            gram[i * b + i] += v[i] * v[i];
            for j in (i + 1)..b {
                gram[i * b + j] += v[i] * v[j];
            }
        }
    }

    // Fill lower triangle
    for i in 0..b {
        for j in (i + 1)..b {
            gram[j * b + i] = gram[i * b + j];
        }
    }

    // Normalization
    if normalized {
        for i in 0..b {
            let d_i = col_norm_sq[i].sqrt();
            if d_i == 0.0 {
                continue;
            }
            for j in 0..b {
                let d_j = col_norm_sq[j].sqrt();
                if d_j == 0.0 {
                    continue;
                }
                gram[i * b + j] /= d_i * d_j;
            }
        }
    }

    Ok(gram)
}

/// BnB Gram matrix for a banded Toeplitz strategy with known forward coefficients.
///
/// For a Toeplitz strategy with coefficients `[c_0, c_1, ..., c_{p-1}]`,
/// column j has entries `C[t,j] = c_{t-j}` for `j ≤ t < j+p`, else 0.
///
/// The inner product of columns a and c (a ≤ c, gap d = c-a) is:
///   ⟨C[:,a], C[:,c]⟩ = Σ_{k=0}^{p-1-d} c_{k+d} · c_k   (if d < p, else 0)
///
/// This is used for BnB accounting of BandMF/BLT mechanisms where the
/// strategy coefficients are known from the Toeplitz optimization.
///
/// Note: the columns are NOT generated via a recurrence (unlike BISR).
/// The strategy coefficients ARE the column entries directly.
pub fn toeplitz_gram_matrix(
    strategy_coef: &[f64],
    n_steps: usize,
    min_sep: usize,
    max_participations: Option<usize>,
    normalized: bool,
) -> Result<Vec<f64>> {
    if strategy_coef.is_empty() {
        return Err(PldError::InvalidParameter(
            "strategy_coef must be non-empty".into(),
        ));
    }
    if n_steps == 0 {
        return Err(PldError::InvalidParameter("n_steps must be >= 1".into()));
    }
    if min_sep == 0 {
        return Err(PldError::InvalidParameter("min_sep must be >= 1".into()));
    }

    let p = strategy_coef.len(); // bandwidth
    let b = min_sep;
    let n = n_steps;
    let k_inferred = n.div_ceil(b);
    let e = match max_participations {
        Some(k) => k.min(k_inferred),
        None => k_inferred,
    };

    if e == 0 || b == 0 {
        return Ok(vec![0.0; b * b]);
    }

    // Precompute the inner product for each gap d = 0..p-1.
    // ip_by_gap[d] = Σ_{k=0}^{p-1-d} c_{k+d} · c_k
    let mut ip_by_gap = vec![0.0f64; p];
    for d in 0..p {
        let s: f64 = strategy_coef[d..p]
            .iter()
            .zip(strategy_coef[..p - d].iter())
            .map(|(a, b)| a * b)
            .sum();
        ip_by_gap[d] = s;
    }

    // Column norm = sqrt(ip_by_gap[0]) for all columns (Toeplitz → same norm).
    // But boundary columns (near end of matrix) have truncated entries.
    // For simplicity and correctness, compute norms per-column accounting for truncation.
    let col_norm = |col: usize| -> f64 {
        let remaining = n - col; // entries available
        let effective_len = remaining.min(p);
        let s: f64 = strategy_coef[..effective_len].iter().map(|c| c * c).sum();
        s.sqrt()
    };

    let col_ip = |a: usize, c: usize| -> f64 {
        // Inner product of columns a and c (a ≤ c)
        let d = c - a;
        if d >= p {
            return 0.0;
        }
        let remaining = n - c; // both columns exist from c to min(c+p, n)
        let effective_len = remaining.min(p - d);
        let mut s = 0.0;
        for k in 0..effective_len {
            s += strategy_coef[k + d] * strategy_coef[k];
        }
        s
    };

    // Build BnB Gram matrix: G[i,j] = Σ_{p,q epochs} ⟨C[:,b*p+i], C[:,b*q+j]⟩ / (norm * norm)
    let mut gram = vec![0.0f64; b * b];

    for i in 0..b {
        for j in i..b {
            let mut val = 0.0;

            for ep_p in 0..e {
                let col_a = b * ep_p + i;
                if col_a >= n {
                    break;
                }

                for ep_q in 0..e {
                    let col_c = b * ep_q + j;
                    if col_c >= n {
                        break;
                    }

                    let (lo, hi) = if col_a <= col_c {
                        (col_a, col_c)
                    } else {
                        (col_c, col_a)
                    };

                    let ip = col_ip(lo, hi);

                    let contribution = if normalized {
                        let d_a = col_norm(col_a);
                        let d_c = col_norm(col_c);
                        if d_a > 0.0 && d_c > 0.0 {
                            ip / (d_a * d_c)
                        } else {
                            0.0
                        }
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

    // BISR inverse coefficients for prefix-sum workload:
    // r̃_0 = 1, r̃_j = ((j - 3/2) / j) · r̃_{j-1}
    // All negative after r̃_0: [1, -1/2, -1/8, -1/16, -5/128, ...]
    const BISR_P2: [f64; 2] = [1.0, -0.5];
    const BISR_P3: [f64; 3] = [1.0, -0.5, -0.125];
    const BISR_P4: [f64; 4] = [1.0, -0.5, -0.125, -0.0625];
    const BISR_P5: [f64; 5] = [1.0, -0.5, -0.125, -0.0625, -0.0390625];

    // λCGD coefficients for comparison
    fn lambda_cgd_coefs(lambda: f64) -> [f64; 2] {
        [1.0, -lambda]
    }

    #[test]
    fn test_column_zero_p2_is_lambda_cgd() {
        // p=2 with [1, -0.5] should give col0[t] = 0.5^t (λCGD with λ=0.5)
        let col = bisr_column_zero(&BISR_P2, 10);
        for t in 0..10 {
            let expected = 0.5_f64.powi(t as i32);
            assert!(
                (col[t] - expected).abs() < 1e-10,
                "t={}: got {}, expected {}",
                t,
                col[t],
                expected
            );
        }
    }

    #[test]
    fn test_column_zero_p2_general_lambda() {
        // [1, -λ] should give col0[t] = λ^t
        let lambda: f64 = 0.9;
        let coefs = lambda_cgd_coefs(lambda);
        let col = bisr_column_zero(&coefs, 20);
        for t in 0..20 {
            let expected = lambda.powi(t as i32);
            assert!(
                (col[t] - expected).abs() < 1e-10,
                "t={}: got {}, expected {}",
                t,
                col[t],
                expected
            );
        }
    }

    #[test]
    fn test_column_zero_p3_entries_decay() {
        // Higher bandwidth columns should still decay
        let col = bisr_column_zero(&BISR_P3, 30);
        assert!((col[0] - 1.0).abs() < 1e-10);
        // Entries should decay towards 0
        let max_entry = col.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
        assert!(max_entry < 10.0, "column entries should be bounded");
        // Late entries should be small
        assert!(col[29].abs() < 0.1, "col[29] = {} should be small", col[29]);
    }

    #[test]
    fn test_sensitivity_p2_matches_lambda_cgd() {
        // BISR p=2 with [1, -λ] should match λCGD sensitivity
        let lambda: f64 = 0.9;
        let coefs = lambda_cgd_coefs(lambda);

        let bisr_sens = bisr_sensitivity_squared(&coefs, 100, 10, Some(3), 0.0).unwrap();

        // Compare with λCGD
        let lcgd_sens = crate::matrix_factorization::lambda_cgd_sensitivity_squared(
            lambda,
            100,
            10,
            Some(3),
            0.0,
        )
        .unwrap();

        assert!(
            (bisr_sens - lcgd_sens).abs() / lcgd_sens < 1e-6,
            "BISR p=2: {}, λCGD: {}",
            bisr_sens,
            lcgd_sens
        );
    }

    #[test]
    fn test_sensitivity_p2_normalized_matches_lambda_cgd() {
        let lambda: f64 = 0.9;
        let coefs = lambda_cgd_coefs(lambda);

        let bisr_sens = bisr_normalized_sensitivity_squared(&coefs, 100, 10, Some(3), 0.0).unwrap();
        let lcgd_sens = crate::matrix_factorization::lambda_cgd_normalized_sensitivity_squared(
            lambda,
            100,
            10,
            Some(3),
            0.0,
        )
        .unwrap();

        assert!(
            (bisr_sens - lcgd_sens).abs() / lcgd_sens < 1e-6,
            "normalized BISR p=2: {}, λCGD: {}",
            bisr_sens,
            lcgd_sens
        );
    }

    #[test]
    fn test_normalized_k1_is_one() {
        // Normalized sensitivity with k=1 should always be 1.0
        for coefs in [&BISR_P2[..], &BISR_P3[..], &BISR_P4[..], &BISR_P5[..]] {
            let sens = bisr_normalized_sensitivity_squared(coefs, 100, 100, Some(1), 0.0).unwrap();
            assert!(
                (sens - 1.0).abs() < 1e-10,
                "p={}: normalized k=1 sens = {}, expected 1.0",
                coefs.len(),
                sens
            );
        }
    }

    #[test]
    fn test_sensitivity_positive_finite() {
        for coefs in [&BISR_P3[..], &BISR_P4[..], &BISR_P5[..]] {
            let sens = bisr_sensitivity_squared(coefs, 200, 20, Some(5), 0.0).unwrap();
            assert!(
                sens > 0.0 && sens.is_finite(),
                "p={}: sens = {}",
                coefs.len(),
                sens
            );
        }
    }

    #[test]
    fn test_sensitivity_brute_force_small() {
        // Brute-force: build C, compute ‖Σ columns at participation steps‖²
        let coefs = &BISR_P3;
        let n: usize = 15;
        let b: usize = 5;
        let k: usize = 3;

        let col0 = bisr_column_zero(coefs, n);

        // S² = Σ_t (Σ_{p=0}^{k-1} C[t, p*b])²
        let mut brute_sq = 0.0;
        for t in 0..n {
            let mut combined = 0.0;
            for p in 0..k {
                let col_start = p * b;
                if col_start > t {
                    continue;
                }
                let entry_idx = t - col_start;
                if entry_idx < col0.len() {
                    combined += col0[entry_idx];
                }
            }
            brute_sq += combined * combined;
        }

        let computed = bisr_sensitivity_squared(coefs, n, b, Some(k), 0.0).unwrap();
        assert!(
            (brute_sq - computed).abs() / brute_sq < 1e-8,
            "brute={}, computed={}",
            brute_sq,
            computed
        );
    }

    #[test]
    fn test_sensitivity_with_momentum() {
        let coefs = &BISR_P3;
        let sens_no_mom = bisr_sensitivity_squared(coefs, 60, 10, Some(3), 0.0).unwrap();
        let sens_with_mom = bisr_sensitivity_squared(coefs, 60, 10, Some(3), 0.9).unwrap();
        assert!(sens_no_mom > 0.0);
        assert!(sens_with_mom > 0.0);
        // Both should be finite
        assert!(sens_no_mom.is_finite());
        assert!(sens_with_mom.is_finite());
    }

    #[test]
    fn test_gram_matrix_p2_matches_lambda_cgd() {
        let lambda: f64 = 0.9;
        let coefs = lambda_cgd_coefs(lambda);
        let b: usize = 10;
        let e: usize = 3;
        let n = b * e;

        let bisr_gram = bisr_gram_matrix(&coefs, n, b, Some(e), true, 0.0).unwrap();
        let lcgd_gram =
            crate::matrix_factorization::lambda_cgd_gram_matrix(lambda, n, b, Some(e), true, 0.0)
                .unwrap();

        for idx in 0..(b * b) {
            assert!(
                (bisr_gram[idx] - lcgd_gram[idx]).abs() / lcgd_gram[idx].abs().max(1e-10) < 1e-4,
                "entry {}: bisr={}, lcgd={}",
                idx,
                bisr_gram[idx],
                lcgd_gram[idx]
            );
        }
    }

    #[test]
    fn test_gram_matrix_symmetric() {
        let gram = bisr_gram_matrix(&BISR_P4, 80, 10, Some(4), true, 0.0).unwrap();
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
    fn test_gram_matrix_psd() {
        // Cholesky PSD check
        for coefs in [&BISR_P3[..], &BISR_P4[..], &BISR_P5[..]] {
            let b = 8;
            let e = 3;
            let gram = bisr_gram_matrix(coefs, b * e, b, Some(e), true, 0.0).unwrap();
            // Naive Cholesky
            let mut l = vec![0.0f64; b * b];
            for i in 0..b {
                let mut diag = gram[i * b + i];
                for k in 0..i {
                    diag -= l[i * b + k] * l[i * b + k];
                }
                assert!(
                    diag > -1e-10,
                    "Gram not PSD at row {} for p={}",
                    i,
                    coefs.len()
                );
                l[i * b + i] = diag.max(0.0).sqrt();
                for j in (i + 1)..b {
                    let mut off = gram[j * b + i];
                    for k in 0..i {
                        off -= l[j * b + k] * l[i * b + k];
                    }
                    l[j * b + i] = if l[i * b + i] > 0.0 {
                        off / l[i * b + i]
                    } else {
                        0.0
                    };
                }
            }
        }
    }

    #[test]
    fn test_gram_lr_uniform_matches_closed_form() {
        let coefs = &BISR_P3;
        let b = 8;
        let e = 3;
        let n = b * e;
        let lr = vec![1.0; n];

        let gram_cf = bisr_gram_matrix(coefs, n, b, Some(e), false, 0.0).unwrap();
        let gram_lr = bisr_gram_matrix_lr(coefs, 0.0, n, b, Some(e), false, &lr).unwrap();

        for idx in 0..(b * b) {
            assert!(
                (gram_cf[idx] - gram_lr[idx]).abs() / gram_cf[idx].abs().max(1e-10) < 1e-4,
                "entry {}: cf={}, lr={}",
                idx,
                gram_cf[idx],
                gram_lr[idx]
            );
        }
    }

    #[test]
    fn test_rejects_bad_params() {
        assert!(bisr_sensitivity_squared(&[1.0], 10, 1, None, 0.0).is_err()); // bandwidth < 2
        assert!(bisr_sensitivity_squared(&BISR_P3, 0, 1, None, 0.0).is_err()); // n=0
        assert!(bisr_sensitivity_squared(&BISR_P3, 10, 0, None, 0.0).is_err()); // min_sep=0
        assert!(bisr_sensitivity_squared(&BISR_P3, 10, 1, None, 1.0).is_err()); // momentum=1
    }
}
