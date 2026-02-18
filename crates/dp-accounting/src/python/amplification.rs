//! Amplification PLD constructors: Poisson, truncated-Poisson, accumulated.

use pyo3::prelude::*;

use super::config::PyPldConfig;
use super::pld::PyPld;

/// Compute the PLD for a Poisson-subsampled Gaussian mechanism.
///
/// Args:
///     noise_multiplier (float): σ/Δ ratio, in [0.1, 1.2].
///     rate (float): Poisson sampling probability, in (0, 1].
///     config (PldConfig, optional): Discretization configuration.
///
/// Returns:
///     Pld: The amplified privacy loss distribution.
#[pyfunction]
#[pyo3(name = "poisson_gaussian_pld", signature = (noise_multiplier, rate, config=None))]
pub fn py_poisson_gaussian_pld(
    noise_multiplier: f64,
    rate: f64,
    config: Option<&PyPldConfig>,
) -> PyResult<PyPld> {
    let cfg = PyPldConfig::resolve(config);
    let pld = crate::amplification::poisson_gaussian_pld(noise_multiplier, rate, &cfg)
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
///     config (PldConfig, optional): Discretization configuration.
///
/// Returns:
///     Pld: The amplified privacy loss distribution.
#[pyfunction]
#[pyo3(name = "truncated_poisson_gaussian_pld", signature = (noise_multiplier, rate, batch_size_max, dataset_size, config=None))]
pub fn py_truncated_poisson_gaussian_pld(
    noise_multiplier: f64,
    rate: f64,
    batch_size_max: usize,
    dataset_size: usize,
    config: Option<&PyPldConfig>,
) -> PyResult<PyPld> {
    let cfg = PyPldConfig::resolve(config);
    let pld = crate::amplification::truncated_poisson_gaussian_pld(
        noise_multiplier,
        rate,
        batch_size_max,
        dataset_size,
        &cfg,
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyPld::new(pld))
}

/// Compute the PLD for an accumulated Poisson-subsampled Gaussian mechanism.
///
/// Models gradient accumulation: m microbatches, Poisson-sampled, clipped
/// gradients summed, noise added once.
///
/// Args:
///     noise_multiplier (float): σ/Δ ratio, in [0.1, 1.2].
///     rate (float): Poisson sampling probability, in (0, 1].
///     microbatches (int): Number of microbatches, > 0.
///     config (PldConfig, optional): Discretization configuration.
///
/// Returns:
///     Pld: The amplified privacy loss distribution.
#[pyfunction]
#[pyo3(name = "accumulated_poisson_gaussian_pld", signature = (noise_multiplier, rate, microbatches, config=None))]
pub fn py_accumulated_poisson_gaussian_pld(
    noise_multiplier: f64,
    rate: f64,
    microbatches: usize,
    config: Option<&PyPldConfig>,
) -> PyResult<PyPld> {
    let cfg = PyPldConfig::resolve(config);
    let pld = crate::amplification::accumulated_poisson_gaussian_pld(
        noise_multiplier,
        rate,
        microbatches,
        &cfg,
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyPld::new(pld))
}
