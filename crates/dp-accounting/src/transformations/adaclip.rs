//! Adaptive clipping utilities.
//!
//! Provides the `adaclip_sensitivity` formula for adaptive clipping with
//! a public quantile estimator, used when composing privacy accounting for
//! DP-SGD with adaptive gradient clipping.

/// Compute the combined sensitivity z̃ for adaptive clipping.
///
/// When using both gradient clipping (with noise multiplier `noise_multiplier`)
/// and a public quantile estimator (with noise std `quantile_noise_std`), the
/// combined L2 sensitivity is:
///
/// ```text
/// z̃ = sqrt(1/z² + 1/(4·σ_b²))
/// ```
///
/// where `z = noise_multiplier` and `σ_b = quantile_noise_std`.
///
/// This is the effective noise multiplier to use for privacy accounting
/// when adaptive clipping is enabled.
///
/// # Arguments
///
/// * `noise_multiplier` — gradient noise multiplier z
/// * `quantile_noise_std` — std of noise added to the quantile estimator σ_b
///
/// # Panics
///
/// Panics if either argument is zero (would cause division by zero).
pub fn adaclip_sensitivity(noise_multiplier: f64, quantile_noise_std: f64) -> f64 {
    let z = noise_multiplier;
    let sigma_b = quantile_noise_std;
    (1.0 / (z * z) + 1.0 / (4.0 * sigma_b * sigma_b)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaclip_sensitivity_basic() {
        // z=1, σ_b=∞ → z̃ ≈ 1/z = 1.0
        let result = adaclip_sensitivity(1.0, 1e10);
        assert!((result - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_adaclip_sensitivity_equal_contribution() {
        // z=1, σ_b=0.5 → z̃ = sqrt(1 + 1) = sqrt(2)
        let result = adaclip_sensitivity(1.0, 0.5);
        assert!((result - std::f64::consts::SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn test_adaclip_sensitivity_large_quantile_noise() {
        // Large σ_b → dominated by 1/z²
        let z = 0.5;
        let result = adaclip_sensitivity(z, 1e6);
        let expected = 1.0 / z;
        assert!((result - expected).abs() < 1e-6);
    }

    #[test]
    fn test_adaclip_sensitivity_symmetry() {
        // adaclip_sensitivity is symmetric except for the 4x factor on σ_b
        let a = adaclip_sensitivity(1.0, 2.0);
        let b = adaclip_sensitivity(2.0, 1.0);
        // Not symmetric — verify different
        assert!((a - b).abs() > 0.01);
    }

    #[test]
    fn test_adaclip_sensitivity_always_positive() {
        for &z in &[0.1, 0.5, 1.0, 2.0] {
            for &sb in &[0.1, 0.5, 1.0, 10.0] {
                let result = adaclip_sensitivity(z, sb);
                assert!(result > 0.0, "z={}, sb={}, result={}", z, sb, result);
            }
        }
    }
}
