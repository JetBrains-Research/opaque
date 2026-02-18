//! Adaptive clipping helpers.

use pyo3::prelude::*;

/// Compute the combined sensitivity of two adaptive-clipping mechanisms.
///
/// Args:
///     l2_clip_norm (float): L2 clipping norm.
///     quantile_noise (float): Noise multiplier for quantile estimation.
///
/// Returns:
///     float: The combined L2 sensitivity.
#[pyfunction]
#[pyo3(name = "combined_sensitivity")]
pub fn py_combined_sensitivity(l2_clip_norm: f64, quantile_noise: f64) -> f64 {
    crate::adaclip::combined_sensitivity(l2_clip_norm, quantile_noise)
}
