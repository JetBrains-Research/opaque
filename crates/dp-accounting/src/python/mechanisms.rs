//! Base mechanism PLD constructors: Gaussian, eps-delta, identity.

use pyo3::prelude::*;

use super::config::PyDiscretizationConfig;
use super::pld::PyPld;

/// Compute the PLD for a Gaussian mechanism.
///
/// Args:
///     noise_multiplier (float): σ/Δ ratio, in [0.1, 1.2].
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
