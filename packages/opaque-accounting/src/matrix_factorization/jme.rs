//! Joint Moment Estimation (JME) sensitivity computation.
//!
//! Implements the sensitivity analysis from Kalinin, Upadhyay, Lampert (2025)
//! "Continual Release Moment Estimation with Differential Privacy"
//! (arXiv:2502.06597, NeurIPS 2025).
//!
//! JME enables DP-Adam and DP-AdaGrad with MF correlated noise by jointly
//! analyzing the sensitivity of estimating both the first moment (gradients)
//! and second moment (squared gradients). The key result (Theorem 3.2):
//! with optimal λ, the second moment estimation is "free" — the joint
//! sensitivity equals the first-moment-only sensitivity.
//!
//! # Joint Sensitivity (Theorem 3.2)
//!
//! For noise-shaping matrices C₁ (first moment) and C₂ (second moment),
//! clipping bound ζ, and optimal scaling parameter λ:
//!
//! ```text
//! s_joint = 2ζ · ‖C₁‖_{1→2}
//! ```
//!
//! where ‖C‖_{1→2} = max_j ‖C[:,j]‖₂ is the maximum column L2 norm.
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

/// Compute the optimal JME scaling parameter λ.
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
pub fn jme_lambda(
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

/// Compute the joint sensitivity for JME.
///
/// With optimal λ, the joint sensitivity for both first and second
/// moment estimation is `s = 2ζ · ‖C₁‖_{1→2}` (Theorem 3.2).
///
/// # Arguments
///
/// * `c1_max_col_norm` — ‖C₁‖_{1→2}, max column L2 norm of the first moment strategy.
/// * `zeta` — Clipping bound per sample.
///
/// # Returns
///
/// The joint sensitivity `2ζ · ‖C₁‖_{1→2}`.
///
/// # Errors
///
/// Returns `InvalidParameter` if inputs are non-positive.
pub fn jme_joint_sensitivity(c1_max_col_norm: f64, zeta: f64) -> Result<f64> {
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

    Ok(2.0 * zeta * c1_max_col_norm)
}

/// Compute the noise scaling factor for the second moment stream.
///
/// The second moment noise is scaled by λ^{-1/2} relative to the
/// first moment noise. This ensures the combined (first + second)
/// estimation stays within the joint sensitivity budget.
///
/// # Arguments
///
/// * `lambda_jme` — The JME scaling parameter λ (from [`jme_lambda`]).
///
/// # Returns
///
/// The scaling factor λ^{-1/2}.
///
/// # Errors
///
/// Returns `InvalidParameter` if λ is non-positive.
pub fn jme_second_moment_noise_scale(lambda_jme: f64) -> Result<f64> {
    if lambda_jme <= 0.0 {
        return Err(PldError::InvalidParameter(format!(
            "lambda_jme must be positive, got {}",
            lambda_jme
        )));
    }
    Ok(1.0 / lambda_jme.sqrt())
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
    fn test_jme_lambda_same_strategy() {
        // When C₁ = C₂ (same strategy for both moments):
        // λ = ‖C‖² / (c_d · ζ² · ‖C‖²) = 1 / (c_d · ζ²)
        let norm = 1.5;
        let zeta = 0.1;
        let lambda = jme_lambda(norm, norm, zeta, 2).unwrap();
        let expected = 1.0 / (2.0 * zeta * zeta);
        assert!(
            (lambda - expected).abs() / expected < 1e-10,
            "got {}, expected {}",
            lambda,
            expected
        );
    }

    #[test]
    fn test_jme_lambda_different_strategies() {
        let c1_norm = 2.0;
        let c2_norm = 1.0;
        let zeta = 0.5;
        let lambda = jme_lambda(c1_norm, c2_norm, zeta, 2).unwrap();
        let expected = (c1_norm * c1_norm) / (2.0 * zeta * zeta * c2_norm * c2_norm);
        assert!(
            (lambda - expected).abs() / expected < 1e-10,
            "got {}, expected {}",
            lambda,
            expected
        );
    }

    #[test]
    fn test_jme_lambda_d1() {
        let norm = 1.0;
        let zeta = 1.0;
        let lambda = jme_lambda(norm, norm, zeta, 1).unwrap();
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
    fn test_jme_lambda_rejects_invalid() {
        assert!(jme_lambda(0.0, 1.0, 1.0, 2).is_err());
        assert!(jme_lambda(1.0, 0.0, 1.0, 2).is_err());
        assert!(jme_lambda(1.0, 1.0, 0.0, 2).is_err());
        assert!(jme_lambda(1.0, 1.0, -1.0, 2).is_err());
        assert!(jme_lambda(-1.0, 1.0, 1.0, 2).is_err());
    }

    #[test]
    fn test_jme_joint_sensitivity() {
        let c1_norm = 1.5;
        let zeta = 0.1;
        let s = jme_joint_sensitivity(c1_norm, zeta).unwrap();
        let expected = 2.0 * zeta * c1_norm;
        assert!((s - expected).abs() < 1e-10);
    }

    #[test]
    fn test_jme_joint_sensitivity_rejects_invalid() {
        assert!(jme_joint_sensitivity(0.0, 1.0).is_err());
        assert!(jme_joint_sensitivity(1.0, 0.0).is_err());
        assert!(jme_joint_sensitivity(-1.0, 1.0).is_err());
    }

    #[test]
    fn test_jme_second_moment_noise_scale() {
        let lambda = 4.0;
        let scale = jme_second_moment_noise_scale(lambda).unwrap();
        assert!((scale - 0.5).abs() < 1e-10);

        let scale1 = jme_second_moment_noise_scale(1.0).unwrap();
        assert!((scale1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_jme_second_moment_noise_scale_rejects_invalid() {
        assert!(jme_second_moment_noise_scale(0.0).is_err());
        assert!(jme_second_moment_noise_scale(-1.0).is_err());
    }

    #[test]
    fn test_privacy_for_free_property() {
        // The key JME result: with optimal λ, the joint sensitivity
        // equals 2ζ · ‖C₁‖_{1→2}, which is exactly twice the
        // single-moment sensitivity (ζ · ‖C₁‖_{1→2}).
        //
        // This means the noise for the first moment is the same as if
        // we only estimated the first moment with a substitute-one
        // DP model — the second moment is "free".
        let c1_norm = 3.0;
        let c2_norm = 2.0;
        let zeta = 0.5;
        let d = 10;

        let lambda = jme_lambda(c1_norm, c2_norm, zeta, d).unwrap();
        let joint_s = jme_joint_sensitivity(c1_norm, zeta).unwrap();

        // Verify: s = 2ζ · ‖C₁‖
        let expected_s = 2.0 * zeta * c1_norm;
        assert!((joint_s - expected_s).abs() < 1e-10);

        // Verify: the second moment scale factor is λ^{-1/2}
        let scale = jme_second_moment_noise_scale(lambda).unwrap();
        let cd: f64 = 2.0; // d >= 2
        let expected_scale = zeta * cd.sqrt() * c2_norm / c1_norm;
        assert!(
            (scale - expected_scale).abs() / expected_scale < 1e-10,
            "got {}, expected {}",
            scale,
            expected_scale
        );
    }
}
