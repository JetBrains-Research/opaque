//! Base mechanism PLD constructors: Gaussian, eps-delta, identity.

use pyo3::prelude::*;

use super::config::PyDiscretizationConfig;
use super::pld::PyPld;

/// Compute the PLD for the Bounded Gaussian mechanism (Replace adjacency).
///
/// The Bounded Gaussian Mechanism (Chen & Hale, 2024) adds truncated Gaussian
/// noise to keep outputs in a bounded domain.  Under Replace adjacency,
/// sensitivity is 2Δ, so the PLD equals that of a Gaussian with
/// ``effective_σ = noise_multiplier / 2``.
///
/// Args:
///     noise_multiplier (float): σ/Δ ratio, in [0.1, 2.4].
///     config (DiscretizationConfig, optional): Discretization configuration.
///
/// Returns:
///     Pld: The privacy loss distribution.
#[pyfunction]
#[pyo3(name = "bounded_gaussian_pld", signature = (noise_multiplier, config=None))]
pub fn py_bounded_gaussian_pld(
    noise_multiplier: f64,
    config: Option<&PyDiscretizationConfig>,
) -> PyResult<PyPld> {
    let cfg = PyDiscretizationConfig::resolve(config);
    let pld = crate::mechanisms::bounded_gaussian_pld(noise_multiplier, &cfg)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyPld::new(pld))
}

/// Compute the PLD for a Gaussian mechanism.
///
/// Args:
///     noise_multiplier (float): σ/Δ ratio, in [0.1, 1.2].
///     config (DiscretizationConfig, optional): Discretization configuration.
///
/// Returns:
///     Pld: The privacy loss distribution.
#[pyfunction]
#[pyo3(name = "gaussian_pld", signature = (noise_multiplier, config=None))]
pub fn py_gaussian_pld(
    noise_multiplier: f64,
    config: Option<&PyDiscretizationConfig>,
) -> PyResult<PyPld> {
    let cfg = PyDiscretizationConfig::resolve(config);
    let pld = crate::mechanisms::gaussian_pld(noise_multiplier, &cfg)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyPld::new(pld))
}

/// Compute the PLD for an (epsilon, delta)-mechanism.
///
/// Args:
///     epsilon (float): Privacy loss, >= 0.
///     delta (float): Failure probability, in [0, 1].
///     config (DiscretizationConfig, optional): Discretization configuration.
///
/// Returns:
///     Pld: The privacy loss distribution.
#[pyfunction]
#[pyo3(name = "eps_delta_pld", signature = (epsilon, delta, config=None))]
pub fn py_eps_delta_pld(
    epsilon: f64,
    delta: f64,
    config: Option<&PyDiscretizationConfig>,
) -> PyResult<PyPld> {
    let cfg = PyDiscretizationConfig::resolve(config);
    let pld = crate::mechanisms::eps_delta_pld(epsilon, delta, &cfg)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyPld::new(pld))
}

/// Compute the PLD for the identity (zero-privacy-loss) mechanism.
///
/// Args:
///     config (DiscretizationConfig, optional): Discretization configuration.
///
/// Returns:
///     Pld: The identity PLD (neutral element for composition).
#[pyfunction]
#[pyo3(name = "identity_pld", signature = (config=None))]
pub fn py_identity_pld(config: Option<&PyDiscretizationConfig>) -> PyResult<PyPld> {
    let cfg = PyDiscretizationConfig::resolve(config);
    let pld = crate::mechanisms::identity_pld(&cfg)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyPld::new(pld))
}
