//! Private second-moment sensitivity computation.
//!
//! Implements the sensitivity analysis from Kalinin, Upadhyay, Lampert (2025)
//! "Continual Release Moment Estimation with Differential Privacy"
//! (arXiv:2502.06597, NeurIPS 2025).
//!
//! Private second-moment estimation enables DP-Adam and DP-AdaGrad with MF
//! correlated noise by jointly analyzing the sensitivity of estimating both the
//! first moment (gradients) and second moment (squared gradients). The key
//! result (Theorem 3.2): with optimal λ, the second moment estimation is
//! "free" — the joint sensitivity equals the first-moment-only sensitivity.
//!
//! # Joint Sensitivity (Theorem 3.2 adapted to add/remove DP)
//!
//! The paper's original formula assumes substitute-one adjacency:
//! `s = 2ζ · ‖C₁‖_{1→2}` ("privacy for free").
//!
//! Opaque uses add/remove adjacency, where the joint sensitivity is:
//!
//! ```text
//! s = ζ · ‖C₁‖_{1→2} · √(1 + 1/c_d)
//! ```
//!
//! For d ≥ 2: `s = ζ · ‖C₁‖ · √(3/2)` — the second moment costs
//! ~22% more noise than first-moment-only (√(3/2) ≈ 1.22).
//!
//! # Scaling Parameter λ (Algorithm 1)
//!
//! ```text
//! λ = ‖C₁‖²_{1→2} / (c_d · ζ² · ‖C₂‖²_{1→2})
//! ```
//!
//! where c₁ = 8/(11 + 5√5) for d=1, and c_d = 2 for d ≥ 2.
//!
//! # References
//!
//! - Kalinin, Upadhyay, Lampert (2025) <https://arxiv.org/abs/2502.06597>

use crate::error::{PldError, Result};

/// Dimension-dependent constant c_d from Algorithm 1.
///
/// c₁ = 8/(11 + 5√5) ≈ 0.339 for d=1, c_d = 2 for d ≥ 2.
fn c_d_constant(d: usize) -> Result<f64> {
    if d == 0 {
        return Err(PldError::InvalidParameter(
            "dimension d must be >= 1".into(),
        ));
    }
    if d == 1 {
        Ok(8.0 / (11.0 + 5.0 * 5.0_f64.sqrt()))
    } else {
        Ok(2.0)
    }
}

/// Compute the optimal private second-moment scaling parameter λ.
///
/// Sets λ so that the second moment estimation is "free": the joint
/// sensitivity equals the first-moment-only sensitivity (Theorem 3.2).
///
/// # Arguments
///
/// * `c1_max_col_norm` — ‖C₁‖_{1→2}, max column L2 norm of the first moment strategy.
/// * `c2_max_col_norm` — ‖C₂‖_{1→2}, max column L2 norm of the second moment strategy.
/// * `zeta` — Clipping bound per sample (sensitivity from gradient clipping).
/// * `d` — Parameter dimension (1 for scalar, ≥ 2 for vectors/matrices).
///
/// # Returns
///
/// The optimal scaling parameter λ = ‖C₁‖²_{1→2} / (c_d · ζ² · ‖C₂‖²_{1→2}).
///
/// # Errors
///
/// Returns `InvalidParameter` if inputs are non-positive or d is 0.
pub fn second_moment_lambda(
    c1_max_col_norm: f64,
    c2_max_col_norm: f64,
    zeta: f64,
    d: usize,
) -> Result<f64> {
    if c1_max_col_norm <= 0.0 {
        return Err(PldError::InvalidParameter(format!(
            "c1_max_col_norm must be positive, got {}",
            c1_max_col_norm
        )));
    }
    if c2_max_col_norm <= 0.0 {
        return Err(PldError::InvalidParameter(format!(
            "c2_max_col_norm must be positive, got {}",
            c2_max_col_norm
        )));
    }
    if zeta <= 0.0 {
        return Err(PldError::InvalidParameter(format!(
            "zeta must be positive, got {}",
            zeta
        )));
    }

    let cd = c_d_constant(d)?;
    let denom = cd * zeta * zeta * c2_max_col_norm * c2_max_col_norm;
    Ok(c1_max_col_norm * c1_max_col_norm / denom)
}

/// Compute the joint first+second moment sensitivity under add/remove DP.
///
/// Adapted from Theorem 3.2 (arXiv:2502.06597) to add/remove adjacency:
///
/// ```text
/// s = ζ · ‖C₁‖_{1→2} · √(1 + 1/c_d)
/// ```
///
/// For d ≥ 2: `s = ζ · ‖C₁‖ · √(3/2)` (≈ 1.22× first-moment-only).
///
/// The paper's original formula `s = 2ζ · ‖C₁‖` assumes substitute-one
/// adjacency.  Under add/remove, the contributions from `‖x‖` (first
/// moment) and `‖x‖²` (second moment) are both maximised at `‖x‖ = ζ`
/// without the cross-term cancellation that substitute-one provides.
///
/// # Arguments
///
/// * `c1_max_col_norm` — ‖C₁‖_{1→2}, max column L2 norm of the first moment strategy.
/// * `zeta` — Clipping bound per sample.
/// * `d` — Parameter dimension (≥ 2 for neural networks).
///
/// # Returns
///
/// The joint sensitivity under add/remove DP.
///
/// # Errors
///
/// Returns `InvalidParameter` if inputs are non-positive or d is 0.
pub fn second_moment_joint_sensitivity(c1_max_col_norm: f64, zeta: f64, d: usize) -> Result<f64> {
    if c1_max_col_norm <= 0.0 {
        return Err(PldError::InvalidParameter(format!(
            "c1_max_col_norm must be positive, got {}",
            c1_max_col_norm
        )));
    }
    if zeta <= 0.0 {
        return Err(PldError::InvalidParameter(format!(
            "zeta must be positive, got {}",
            zeta
        )));
    }

    let cd = c_d_constant(d)?;
    Ok(zeta * c1_max_col_norm * (1.0 + 1.0 / cd).sqrt())
}

/// Compute the noise scaling factor for the second moment stream.
///
/// The second moment noise is scaled by λ^{-1/2} relative to the
/// first moment noise. This ensures the combined (first + second)
/// estimation stays within the joint sensitivity budget.
///
/// # Arguments
///
/// * `lambda_second_moment` — The scaling parameter λ (from [`second_moment_lambda`]).
///
/// # Returns
///
/// The scaling factor λ^{-1/2}.
///
/// # Errors
///
/// Returns `InvalidParameter` if λ is non-positive.
pub fn second_moment_noise_scale(lambda_second_moment: f64) -> Result<f64> {
    if lambda_second_moment <= 0.0 {
        return Err(PldError::InvalidParameter(format!(
            "lambda_second_moment must be positive, got {}",
            lambda_second_moment
        )));
    }
    Ok(1.0 / lambda_second_moment.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_c_d_constant() {
        let c1 = c_d_constant(1).unwrap();
        let expected = 8.0 / (11.0 + 5.0 * 5.0_f64.sqrt());
        assert!((c1 - expected).abs() < 1e-10);

        let c2 = c_d_constant(2).unwrap();
        assert!((c2 - 2.0).abs() < 1e-10);

        let c100 = c_d_constant(100).unwrap();
        assert!((c100 - 2.0).abs() < 1e-10);

        assert!(c_d_constant(0).is_err());
    }

    #[test]
    fn test_second_moment_lambda_same_strategy() {
        // When C₁ = C₂ (same strategy for both moments):
        // λ = ‖C‖² / (c_d · ζ² · ‖C‖²) = 1 / (c_d · ζ²)
        let norm = 1.5;
        let zeta = 0.1;
        let lambda = second_moment_lambda(norm, norm, zeta, 2).unwrap();
        let expected = 1.0 / (2.0 * zeta * zeta);
        assert!(
            (lambda - expected).abs() / expected < 1e-10,
            "got {}, expected {}",
            lambda,
            expected
        );
    }

    #[test]
    fn test_second_moment_lambda_different_strategies() {
        let c1_norm = 2.0;
        let c2_norm = 1.0;
        let zeta = 0.5;
        let lambda = second_moment_lambda(c1_norm, c2_norm, zeta, 2).unwrap();
        let expected = (c1_norm * c1_norm) / (2.0 * zeta * zeta * c2_norm * c2_norm);
        assert!(
            (lambda - expected).abs() / expected < 1e-10,
            "got {}, expected {}",
            lambda,
            expected
        );
    }

    #[test]
    fn test_second_moment_lambda_d1() {
        let norm = 1.0;
        let zeta = 1.0;
        let lambda = second_moment_lambda(norm, norm, zeta, 1).unwrap();
        let c1 = 8.0 / (11.0 + 5.0 * 5.0_f64.sqrt());
        let expected = 1.0 / c1;
        assert!(
            (lambda - expected).abs() / expected < 1e-10,
            "got {}, expected {}",
            lambda,
            expected
        );
    }

    #[test]
    fn test_second_moment_lambda_rejects_invalid() {
        assert!(second_moment_lambda(0.0, 1.0, 1.0, 2).is_err());
        assert!(second_moment_lambda(1.0, 0.0, 1.0, 2).is_err());
        assert!(second_moment_lambda(1.0, 1.0, 0.0, 2).is_err());
        assert!(second_moment_lambda(1.0, 1.0, -1.0, 2).is_err());
        assert!(second_moment_lambda(-1.0, 1.0, 1.0, 2).is_err());
    }

    #[test]
    fn test_second_moment_joint_sensitivity_d2() {
        // For d≥2: s = ζ · ‖C₁‖ · √(3/2)
        let c1_norm = 1.5;
        let zeta = 0.1;
        let s = second_moment_joint_sensitivity(c1_norm, zeta, 2).unwrap();
        let expected = zeta * c1_norm * (1.5_f64).sqrt();
        assert!(
            (s - expected).abs() < 1e-10,
            "got {}, expected {}",
            s,
            expected
        );
    }

    #[test]
    fn test_second_moment_joint_sensitivity_d1() {
        // For d=1: s = ζ · ‖C₁‖ · √(1 + 1/c₁)
        let c1_norm = 2.0;
        let zeta = 0.5;
        let c1 = 8.0 / (11.0 + 5.0 * 5.0_f64.sqrt());
        let s = second_moment_joint_sensitivity(c1_norm, zeta, 1).unwrap();
        let expected = zeta * c1_norm * (1.0 + 1.0 / c1).sqrt();
        assert!(
            (s - expected).abs() < 1e-10,
            "got {}, expected {}",
            s,
            expected
        );
    }

    #[test]
    fn test_second_moment_joint_sensitivity_vs_first_only() {
        // Joint sensitivity should be √(3/2) ≈ 1.22× the first-moment-only
        let c1_norm = 3.0;
        let zeta = 0.5;
        let first_only = zeta * c1_norm; // ζ · ‖C₁‖
        let joint = second_moment_joint_sensitivity(c1_norm, zeta, 2).unwrap();
        let ratio = joint / first_only;
        let expected_ratio = (1.5_f64).sqrt(); // √(3/2)
        assert!(
            (ratio - expected_ratio).abs() < 1e-10,
            "ratio={}, expected √(3/2)={}",
            ratio,
            expected_ratio
        );
    }

    #[test]
    fn test_second_moment_joint_sensitivity_rejects_invalid() {
        assert!(second_moment_joint_sensitivity(0.0, 1.0, 2).is_err());
        assert!(second_moment_joint_sensitivity(1.0, 0.0, 2).is_err());
        assert!(second_moment_joint_sensitivity(-1.0, 1.0, 2).is_err());
        assert!(second_moment_joint_sensitivity(1.0, 1.0, 0).is_err());
    }

    #[test]
    fn test_second_moment_noise_scale() {
        let lambda = 4.0;
        let scale = second_moment_noise_scale(lambda).unwrap();
        assert!((scale - 0.5).abs() < 1e-10);

        let scale1 = second_moment_noise_scale(1.0).unwrap();
        assert!((scale1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_second_moment_noise_scale_rejects_invalid() {
        assert!(second_moment_noise_scale(0.0).is_err());
        assert!(second_moment_noise_scale(-1.0).is_err());
    }

    #[test]
    fn test_add_remove_joint_sensitivity() {
        // Under add/remove DP, the joint sensitivity is:
        //   s² = ‖C₁‖²·ζ² + λ·‖C₂‖²·ζ⁴
        // With λ = ‖C₁‖²/(c_d·ζ²·‖C₂‖²):
        //   s² = ‖C₁‖²·ζ²·(1 + 1/c_d)
        //   s  = ζ·‖C₁‖·√(3/2) for d≥2
        //
        // This is √(3/2) ≈ 1.22× the first-moment-only sensitivity.
        // The second moment costs ~22% more noise, not zero.
        let c1_norm = 3.0;
        let c2_norm = 2.0;
        let zeta = 0.5;
        let d = 10;

        let lambda = second_moment_lambda(c1_norm, c2_norm, zeta, d).unwrap();
        let joint_s = second_moment_joint_sensitivity(c1_norm, zeta, d).unwrap();

        // Verify via direct computation
        let s_sq = c1_norm * c1_norm * zeta * zeta
            + lambda * c2_norm * c2_norm * zeta * zeta * zeta * zeta;
        assert!(
            (joint_s * joint_s - s_sq).abs() / s_sq < 1e-10,
            "joint_s²={}, direct={}",
            joint_s * joint_s,
            s_sq
        );

        // Verify ratio to first-moment-only
        let first_only = zeta * c1_norm;
        let ratio = joint_s / first_only;
        assert!(
            (ratio - (1.5_f64).sqrt()).abs() < 1e-10,
            "ratio={}, expected √(3/2)={}",
            ratio,
            (1.5_f64).sqrt()
        );

        // Verify λ^{-1/2} scale factor
        let scale = second_moment_noise_scale(lambda).unwrap();
        let cd: f64 = 2.0;
        let expected_scale = zeta * cd.sqrt() * c2_norm / c1_norm;
        assert!(
            (scale - expected_scale).abs() / expected_scale < 1e-10,
            "got {}, expected {}",
            scale,
            expected_scale
        );
    }
}
