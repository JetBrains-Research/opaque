//! PyO3 Python bindings for the PLD accounting engine.
//!
//! Exposes flat functions (scalars in → Pld handles out) and the Pld class
//! with privacy metric methods, composition, and self-composition operators.

mod adaclip;
mod amplification;
mod config;
mod matrix_factorization;
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
    m.add_function(wrap_pyfunction!(mechanisms::py_rectified_gaussian_pld, m)?)?;
    m.add_function(wrap_pyfunction!(mechanisms::py_truncated_gaussian_pld, m)?)?;

    // SPA mechanisms
    m.add_function(wrap_pyfunction!(mechanisms::py_spa_gaussian_pld, m)?)?;
    m.add_function(wrap_pyfunction!(
        amplification::py_spa_poisson_gaussian_pld,
        m
    )?)?;

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
    m.add_function(wrap_pyfunction!(
        amplification::py_poisson_rectified_gaussian_pld,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        amplification::py_poisson_truncated_gaussian_pld,
        m
    )?)?;

    // Matrix Factorization
    m.add_function(wrap_pyfunction!(
        matrix_factorization::py_mf_gaussian_pld,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        matrix_factorization::py_max_participation_for_linear_fn,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        matrix_factorization::py_minsep_true_max_participations,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        matrix_factorization::py_single_participation_sensitivity,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        matrix_factorization::py_banded_sensitivity,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        matrix_factorization::py_general_sensitivity_upper_bound,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        matrix_factorization::py_fixed_epoch_sensitivity,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        matrix_factorization::py_blt_sensitivity_squared,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        matrix_factorization::py_toeplitz_minsep_sensitivity_squared,
        m
    )?)?;

    // AdaClip
    m.add_function(wrap_pyfunction!(adaclip::py_adaclip_sensitivity, m)?)?;

    Ok(())
}
