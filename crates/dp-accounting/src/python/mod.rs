//! PyO3 Python bindings for the PLD accounting engine.
//!
//! Exposes flat functions (scalars in → Pld handles out) and the Pld class
//! with privacy metric methods, composition, and self-composition operators.

mod adaclip;
mod amplification;
mod config;
mod mechanisms;
mod pld;

use pyo3::prelude::*;

/// Register all Python-visible types and functions.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Classes
    m.add_class::<pld::PyPld>()?;
    m.add_class::<config::PyDiscretizationConfig>()?;

    // Mechanisms
    m.add_function(wrap_pyfunction!(mechanisms::py_gaussian_pld, m)?)?;
    m.add_function(wrap_pyfunction!(mechanisms::py_eps_delta_pld, m)?)?;
    m.add_function(wrap_pyfunction!(mechanisms::py_identity_pld, m)?)?;

    // Amplification
    m.add_function(wrap_pyfunction!(amplification::py_poisson_gaussian_pld, m)?)?;
    m.add_function(wrap_pyfunction!(
        amplification::py_truncated_poisson_gaussian_pld,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        amplification::py_parallel_poisson_gaussian_pld,
        m
    )?)?;

    // AdaClip
    m.add_function(wrap_pyfunction!(adaclip::py_combined_sensitivity, m)?)?;

    Ok(())
}
