//! Amplification PLD constructors: Poisson, truncated-Poisson, accumulated.

use pyo3::prelude::*;

use super::config::PyDiscretizationConfig;
use super::pld::PyPld;

/// Compute the PLD for a Poisson-subsampled Gaussian mechanism.
///
/// Args:
///     noise_multiplier (float): σ/Δ ratio, in [0.1, 1.2].
///     rate (float): Poisson sampling probability, in (0, 1].
///     config (DiscretizationConfig): Discretization configuration.
///
/// Returns:
///     Pld: The amplified privacy loss distribution.
#[pyfunction]
#[pyo3(name = "poisson_gaussian_pld", signature = (noise_multiplier, rate, config))]
pub fn py_poisson_gaussian_pld(
    noise_multiplier: f64,
    rate: f64,
    config: &PyDiscretizationConfig,
) -> PyResult<PyPld> {
    let pld = crate::amplification::poisson_gaussian_pld(noise_multiplier, rate, &config.inner)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyPld::new(pld))
}

/// Compute the PLD for a truncated Poisson-subsampled Gaussian mechanism.
///
/// This is the actual sampling used in production DP-SGD. Unlike standard
/// Poisson (variable batch), truncated capping at batch_size_max gives
/// predictable memory usage.
///
/// Args:
///     noise_multiplier (float): σ/Δ ratio, in [0.1, 1.2].
///     rate (float): Poisson sampling probability, in (0, 1].
///     batch_size_max (int): Maximum batch size.
///     dataset_size (int): Total dataset size.
///     config (DiscretizationConfig): Discretization configuration.
///
/// Returns:
///     Pld: The amplified privacy loss distribution.
#[pyfunction]
#[pyo3(name = "truncated_poisson_gaussian_pld", signature = (noise_multiplier, rate, batch_size_max, dataset_size, config))]
pub fn py_truncated_poisson_gaussian_pld(
    noise_multiplier: f64,
    rate: f64,
    batch_size_max: usize,
    dataset_size: usize,
    config: &PyDiscretizationConfig,
) -> PyResult<PyPld> {
    let pld = crate::amplification::truncated_poisson_gaussian_pld(
        noise_multiplier,
        rate,
        batch_size_max,
        dataset_size,
        &config.inner,
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyPld::new(pld))
}

/// Compute the PLD for a parallel Poisson-subsampled Gaussian mechanism.
///
/// Models summing multiple independent Poisson samples before adding noise once.
/// Use cases: gradient accumulation (m microbatches) or parallel workers (K workers).
///
/// Args:
///     noise_multiplier (float): σ/Δ ratio, in [0.1, 1.2].
///     rate (float): Poisson sampling probability, in (0, 1].
///     microbatches (int): Number of independent samples, > 0.
///     config (DiscretizationConfig): Discretization configuration.
///
/// Returns:
///     Pld: The amplified privacy loss distribution.
#[pyfunction]
#[pyo3(name = "parallel_poisson_gaussian_pld", signature = (noise_multiplier, rate, microbatches, config))]
pub fn py_parallel_poisson_gaussian_pld(
    noise_multiplier: f64,
    rate: f64,
    microbatches: usize,
    config: &PyDiscretizationConfig,
) -> PyResult<PyPld> {
    let pld = crate::amplification::parallel_poisson_gaussian_pld(
        noise_multiplier,
        rate,
        microbatches,
        &config.inner,
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyPld::new(pld))
}

/// Compute the PLD for a Poisson-subsampled MIP Gaussian mechanism.
///
/// Each example has a per-example sensitivity. The PLD is the weighted mixture
/// of Poisson-subsampled Gaussian PLDs across sensitivity buckets.
///
/// Args:
///     noise_multiplier (float): σ ratio, in [0.1, 1.2].
///     rate (float): Poisson sampling probability, in (0, 1].
///     sensitivities (list[float]): Per-bucket sensitivity values, all > 0.
///     weights (list[float]): Per-bucket weights, must sum to 1.0.
///     config (DiscretizationConfig): Discretization configuration.
///
/// Returns:
///     Pld: The amplified privacy loss distribution.
#[pyfunction]
#[pyo3(name = "poisson_mip_gaussian_pld", signature = (noise_multiplier, rate, sensitivities, weights, config))]
pub fn py_poisson_mip_gaussian_pld(
    noise_multiplier: f64,
    rate: f64,
    sensitivities: Vec<f64>,
    weights: Vec<f64>,
    config: &PyDiscretizationConfig,
) -> PyResult<PyPld> {
    let pld = crate::amplification::poisson_mip_gaussian_pld(
        noise_multiplier,
        rate,
        &sensitivities,
        &weights,
        &config.inner,
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyPld::new(pld))
}

/// Compute the PLD for a Poisson-subsampled rectified Gaussian mechanism.
///
/// The rectified (clamped) Gaussian clips noise to [−R·σ, R·σ], giving
/// tighter privacy bounds than the standard unbounded Gaussian.
///
/// Args:
///     noise_multiplier (float): σ/Δ ratio, in [0.1, 1.2].
///     radius (float): Support half-width in sigma units, in [0.1, 100].
///     rate (float): Poisson sampling probability, in (0, 1].
///     config (DiscretizationConfig): Discretization configuration.
///
/// Returns:
///     Pld: The amplified privacy loss distribution.
#[pyfunction]
#[pyo3(name = "poisson_rectified_gaussian_pld", signature = (noise_multiplier, radius, rate, config))]
pub fn py_poisson_rectified_gaussian_pld(
    noise_multiplier: f64,
    radius: f64,
    rate: f64,
    config: &PyDiscretizationConfig,
) -> PyResult<PyPld> {
    let pld = crate::amplification::poisson_rectified_gaussian_pld(
        noise_multiplier,
        radius,
        rate,
        &config.inner,
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyPld::new(pld))
}

/// Compute the PLD for a Poisson-subsampled truncated Gaussian mechanism.
///
/// The truncated (renormalized) Gaussian restricts noise to [−R·σ, R·σ]
/// with a properly renormalized density, giving even tighter privacy bounds.
///
/// Args:
///     noise_multiplier (float): σ/Δ ratio, in [0.1, 1.2].
///     radius (float): Support half-width in sigma units, in [0.1, 100].
///     rate (float): Poisson sampling probability, in (0, 1].
///     config (DiscretizationConfig): Discretization configuration.
///
/// Returns:
///     Pld: The amplified privacy loss distribution.
#[pyfunction]
#[pyo3(name = "poisson_truncated_gaussian_pld", signature = (noise_multiplier, radius, rate, config))]
pub fn py_poisson_truncated_gaussian_pld(
    noise_multiplier: f64,
    radius: f64,
    rate: f64,
    config: &PyDiscretizationConfig,
) -> PyResult<PyPld> {
    let pld = crate::amplification::poisson_truncated_gaussian_pld(
        noise_multiplier,
        radius,
        rate,
        &config.inner,
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyPld::new(pld))
}
