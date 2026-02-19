//! Adaptive clipping helpers.

use pyo3::prelude::*;

/// Compute the combined sensitivity for adaptive clipping.
///
/// When using both gradient clipping (noise multiplier ``z``) and a noisy
/// quantile estimator (std ``σ_b``), the combined L2 sensitivity is:
///
/// ``z̃ = sqrt(1/z² + 1/(4·σ_b²))``
///
/// Args:
///     noise_multiplier (float): Gradient noise multiplier z.
///     quantile_noise_std (float): Std of noise added to the quantile estimator σ_b.
///
/// Returns:
///     float: The combined L2 sensitivity z̃.
#[pyfunction]
#[pyo3(name = "combined_sensitivity")]
pub fn py_combined_sensitivity(noise_multiplier: f64, quantile_noise_std: f64) -> f64 {
    crate::transformations::adaclip::combined_sensitivity(noise_multiplier, quantile_noise_std)
}
