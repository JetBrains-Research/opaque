//! Base mechanism PLD constructors: Gaussian, eps-delta, identity, truncated Gaussian.

use pyo3::prelude::*;

use super::config::PyDiscretizationConfig;
use super::pld::PyPld;

/// Compute the PLD for a Gaussian mechanism.
///
/// Args:
///     noise_multiplier (float): σ/Δ ratio, must be > 0.
///     config (DiscretizationConfig): Discretization configuration.
///
/// Returns:
///     Pld: The privacy loss distribution.
#[pyfunction]
#[pyo3(name = "gaussian_pld", signature = (noise_multiplier, config))]
pub fn py_gaussian_pld(noise_multiplier: f64, config: &PyDiscretizationConfig) -> PyResult<PyPld> {
    let pld = crate::mechanisms::gaussian_pld(noise_multiplier, &config.inner)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyPld::new(pld))
}

/// Compute the PLD for an (epsilon, delta)-mechanism.
///
/// Args:
///     epsilon (float): Privacy loss, >= 0.
///     delta (float): Failure probability, in [0, 1].
///     config (DiscretizationConfig): Discretization configuration.
///
/// Returns:
///     Pld: The privacy loss distribution.
#[pyfunction]
#[pyo3(name = "eps_delta_pld", signature = (epsilon, delta, config))]
pub fn py_eps_delta_pld(
    epsilon: f64,
    delta: f64,
    config: &PyDiscretizationConfig,
) -> PyResult<PyPld> {
    let pld = crate::mechanisms::eps_delta_pld(epsilon, delta, &config.inner)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyPld::new(pld))
}

/// Compute the PLD for the identity (zero-privacy-loss) mechanism.
///
/// Args:
///     config (DiscretizationConfig): Discretization configuration.
///
/// Returns:
///     Pld: The identity PLD (neutral element for composition).
#[pyfunction]
#[pyo3(name = "identity_pld", signature = (config))]
pub fn py_identity_pld(config: &PyDiscretizationConfig) -> PyResult<PyPld> {
    let pld = crate::mechanisms::identity_pld(&config.inner)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyPld::new(pld))
}

/// Compute the PLD for a non-private mechanism (ε = ∞, δ = 1).
///
/// Args:
///     config (DiscretizationConfig): Discretization configuration.
///
/// Returns:
///     Pld: A PLD with all mass at +∞ (composition annihilator).
#[pyfunction]
#[pyo3(name = "non_private_pld", signature = (config))]
pub fn py_non_private_pld(config: &PyDiscretizationConfig) -> PyResult<PyPld> {
    let pld = crate::mechanisms::non_private_pld(&config.inner)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyPld::new(pld))
}

/// Compute the PLD for a truncated (renormalized) Gaussian mechanism.
///
/// The truncated Gaussian samples from N(0, σ²) restricted to [−R·σ, R·σ],
/// renormalized to integrate to 1. Gives strictly tighter privacy than both
/// rectified and standard Gaussian.
///
/// Args:
///     noise_multiplier (float): σ/Δ ratio, must be > 0.
///     radius (float): Support half-width in sigma units, in [0.1, 100].
///     config (DiscretizationConfig): Discretization configuration.
///
/// Returns:
///     Pld: The privacy loss distribution.
#[pyfunction]
#[pyo3(name = "truncated_gaussian_pld", signature = (noise_multiplier, radius, config))]
pub fn py_truncated_gaussian_pld(
    noise_multiplier: f64,
    radius: f64,
    config: &PyDiscretizationConfig,
) -> PyResult<PyPld> {
    let pld = crate::mechanisms::truncated_gaussian_pld(noise_multiplier, radius, &config.inner)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyPld::new(pld))
}
