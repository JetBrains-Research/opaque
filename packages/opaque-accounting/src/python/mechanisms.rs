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

/// Create a Saddle-Point Accountant PLD for a Gaussian mechanism.
///
/// Unlike gaussian_pld(), this does not discretize — the privacy loss is
/// represented analytically via its CGF. Suitable for small noise multipliers.
///
/// Args:
///     noise_multiplier (float): σ/Δ ratio, in [0.01, 2.5].
///
/// Returns:
///     Pld: The privacy loss distribution (SPA-backed).
#[pyfunction]
#[pyo3(name = "spa_gaussian_pld", signature = (noise_multiplier))]
pub fn py_spa_gaussian_pld(noise_multiplier: f64) -> PyResult<PyPld> {
    let pld = crate::mechanisms::spa_gaussian_pld(noise_multiplier)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyPld::new(pld))
}

/// Compute the PLD for a rectified (clamped) Gaussian mechanism.
///
/// The rectified Gaussian adds noise N(0, σ²) clamped to [−R·σ, R·σ].
/// This is post-processing of the standard Gaussian, giving strictly tighter
/// privacy bounds for finite radius.
///
/// Args:
///     noise_multiplier (float): σ/Δ ratio, in [0.1, 1.2].
///     radius (float): Support half-width in sigma units, in [0.1, 100].
///     config (DiscretizationConfig): Discretization configuration.
///
/// Returns:
///     Pld: The privacy loss distribution.
#[pyfunction]
#[pyo3(name = "rectified_gaussian_pld", signature = (noise_multiplier, radius, config))]
pub fn py_rectified_gaussian_pld(
    noise_multiplier: f64,
    radius: f64,
    config: &PyDiscretizationConfig,
) -> PyResult<PyPld> {
    let pld = crate::mechanisms::rectified_gaussian_pld(noise_multiplier, radius, &config.inner)
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
///     noise_multiplier (float): σ/Δ ratio, in [0.1, 1.2].
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
