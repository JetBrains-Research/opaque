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
    m.add_function(wrap_pyfunction!(mechanisms::py_non_private_pld, m)?)?;

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
        amplification::py_balls_in_bins_gaussian_pld,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        amplification::py_balls_in_bins_gaussian_pld_epochs,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(amplification::py_bnb_mc_pld, m)?)?;

    m.add_function(wrap_pyfunction!(amplification::py_bnb_deterministic_pld, m)?)?;
    m.add_function(wrap_pyfunction!(amplification::py_bnb_deterministic_delta_curve, m)?)?;

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
    m.add_function(wrap_pyfunction!(
        matrix_factorization::py_lambda_cgd_sensitivity_squared,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        matrix_factorization::py_lambda_cgd_normalized_sensitivity_squared,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        matrix_factorization::py_lambda_cgd_max_column_norm,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        matrix_factorization::py_lambda_cgd_gram_matrix,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        matrix_factorization::py_lambda_cgd_gram_matrix_lr,
        m
    )?)?;

    // BISR (Banded Inverse Square Root)
    m.add_function(wrap_pyfunction!(
        matrix_factorization::py_bisr_sensitivity_squared,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        matrix_factorization::py_bisr_normalized_sensitivity_squared,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        matrix_factorization::py_bisr_gram_matrix,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        matrix_factorization::py_bisr_gram_matrix_lr,
        m
    )?)?;

    // Toeplitz Gram matrix (for BnB with BandMF/BLT strategy coefs)
    m.add_function(wrap_pyfunction!(
        matrix_factorization::py_toeplitz_gram_matrix,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        matrix_factorization::py_bisr_strategy_coefficients,
        m
    )?)?;

    // JME (Joint Moment Estimation)
    m.add_function(wrap_pyfunction!(
        matrix_factorization::py_jme_lambda,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        matrix_factorization::py_jme_joint_sensitivity,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        matrix_factorization::py_jme_second_moment_noise_scale,
        m
    )?)?;

    // AdaClip
    m.add_function(wrap_pyfunction!(adaclip::py_adaclip_sensitivity, m)?)?;

    Ok(())
}
