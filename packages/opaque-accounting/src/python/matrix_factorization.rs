//! PyO3 bindings for matrix factorization privacy accounting.

use pyo3::prelude::*;

use super::config::PyDiscretizationConfig;
use super::pld::PyPld;

/// Compute the PLD for a matrix factorization Gaussian mechanism.
///
/// Computes the privacy guarantee for the entire MF training run as a
/// single Gaussian mechanism with effective noise multiplier σ/S.
///
/// The sensitivity S should be pre-computed based on the MF strategy
/// (BandMF, BLT) and participation pattern.
///
/// Args:
///     noise_multiplier (float): Raw noise std σ (before MF). Must be positive.
///     sensitivity (float): L2 sensitivity of the encoder matrix. Must be positive.
///     config (DiscretizationConfig): Discretization configuration.
///
/// Returns:
///     Pld: The privacy loss distribution for the entire MF training run.
///
/// Raises:
///     ValueError: If parameters are out of range.
#[pyfunction]
#[pyo3(name = "mf_gaussian_pld", signature = (noise_multiplier, sensitivity, config))]
pub fn py_mf_gaussian_pld(
    noise_multiplier: f64,
    sensitivity: f64,
    config: &PyDiscretizationConfig,
) -> PyResult<PyPld> {
    let pld =
        crate::matrix_factorization::mf_gaussian_pld(noise_multiplier, sensitivity, &config.inner)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyPld::new(pld))
}

/// Solve max_u <x, u> where u respects min-sep participation.
///
/// Uses dynamic programming (Algorithm 3, VecSens) from
/// Choquette-Choo et al. (2023).
///
/// Args:
///     x (list[float]): Vector of values to optimize over.
///     min_sep (int): Minimum separation between selections (>= 1).
///     max_participations (int | None): Optional upper bound on selections.
///
/// Returns:
///     float: The optimal inner product.
#[pyfunction]
#[pyo3(name = "max_participation_for_linear_fn", signature = (x, min_sep=1, max_participations=None))]
pub fn py_max_participation_for_linear_fn(
    x: Vec<f64>,
    min_sep: usize,
    max_participations: Option<usize>,
) -> f64 {
    crate::matrix_factorization::max_participation_for_linear_fn(&x, min_sep, max_participations)
}

/// Maximum participations under a min-sep constraint.
///
/// Args:
///     n (int): Number of rounds.
///     min_sep (int): Minimum separation between participations.
///     max_participations (int | None): Optional upper bound.
///
/// Returns:
///     int: Effective maximum participations.
#[pyfunction]
#[pyo3(name = "minsep_true_max_participations", signature = (n, min_sep, max_participations=None))]
pub fn py_minsep_true_max_participations(
    n: usize,
    min_sep: usize,
    max_participations: Option<usize>,
) -> usize {
    crate::matrix_factorization::minsep_true_max_participations(n, min_sep, max_participations)
}

/// L2 sensitivity under single participation.
///
/// Args:
///     column_norms (list[float]): L2 norms of encoder matrix columns.
///
/// Returns:
///     float: Maximum column norm (the sensitivity).
///
/// Raises:
///     ValueError: If column_norms is empty or contains non-finite values.
#[pyfunction]
#[pyo3(name = "single_participation_sensitivity", signature = (column_norms,))]
pub fn py_single_participation_sensitivity(column_norms: Vec<f64>) -> PyResult<f64> {
    crate::matrix_factorization::single_participation_sensitivity(&column_norms)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Exact L2 sensitivity for banded Gram matrices under min-sep participation.
///
/// Args:
///     gram_diag (list[float]): Diagonal of Gram matrix X = C^T C.
///     min_sep (int): Minimum separation between participations.
///     max_participations (int | None): Optional upper bound.
///
/// Returns:
///     float: The exact L2 sensitivity.
///
/// Raises:
///     ValueError: If parameters are invalid.
#[pyfunction]
#[pyo3(name = "banded_sensitivity", signature = (gram_diag, min_sep=1, max_participations=None))]
pub fn py_banded_sensitivity(
    gram_diag: Vec<f64>,
    min_sep: usize,
    max_participations: Option<usize>,
) -> PyResult<f64> {
    crate::matrix_factorization::banded_sensitivity(&gram_diag, min_sep, max_participations)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Upper bound on L2 sensitivity for general Gram matrices.
///
/// Args:
///     gram_matrix (list[float]): Flattened row-major Gram matrix X = C^T C.
///     n (int): Matrix dimension.
///     min_sep (int): Minimum separation between participations.
///     max_participations (int | None): Optional upper bound.
///
/// Returns:
///     float: An upper bound on the L2 sensitivity.
///
/// Raises:
///     ValueError: If parameters are invalid.
#[pyfunction]
#[pyo3(name = "general_sensitivity_upper_bound", signature = (gram_matrix, n, min_sep=1, max_participations=None))]
pub fn py_general_sensitivity_upper_bound(
    gram_matrix: Vec<f64>,
    n: usize,
    min_sep: usize,
    max_participations: Option<usize>,
) -> PyResult<f64> {
    crate::matrix_factorization::general_sensitivity_upper_bound(
        &gram_matrix,
        n,
        min_sep,
        max_participations,
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// L2 sensitivity under fixed-epoch participation.
///
/// Args:
///     gram_matrix (list[float]): Flattened row-major Gram matrix X = C^T C.
///     n (int): Matrix dimension (total rounds).
///     epochs (int): Number of epochs (must divide n).
///
/// Returns:
///     float: The L2 sensitivity under fixed-epoch participation.
///
/// Raises:
///     ValueError: If parameters are invalid.
#[pyfunction]
#[pyo3(name = "fixed_epoch_sensitivity", signature = (gram_matrix, n, epochs))]
pub fn py_fixed_epoch_sensitivity(gram_matrix: Vec<f64>, n: usize, epochs: usize) -> PyResult<f64> {
    crate::matrix_factorization::fixed_epoch_sensitivity(&gram_matrix, n, epochs)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Sensitivity squared for a BLT strategy matrix.
///
/// Implements Lemma 5.3 of the BLT paper (https://arxiv.org/abs/2404.16706).
///
/// Args:
///     buf_decay (list[float]): Decay factors for each buffer, each in (0, 1).
///     output_scale (list[float]): Scale factors for each buffer.
///     n (float): Number of iterations (use float('inf') for asymptotic).
///
/// Returns:
///     float: The sensitivity squared.
///
/// Raises:
///     ValueError: If parameters are invalid.
#[pyfunction]
#[pyo3(name = "blt_sensitivity_squared", signature = (buf_decay, output_scale, n))]
pub fn py_blt_sensitivity_squared(
    buf_decay: Vec<f64>,
    output_scale: Vec<f64>,
    n: f64,
) -> PyResult<f64> {
    crate::matrix_factorization::blt_sensitivity_squared(&buf_decay, &output_scale, n)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Sensitivity squared for a Toeplitz matrix under min-sep participation.
///
/// Implements BSR Theorem 2 closed-form for non-negative, non-increasing
/// Toeplitz coefficients.
///
/// Args:
///     strategy_coef (list[float]): Toeplitz coefficients (non-negative, non-increasing).
///     n (int): Matrix dimension (total rounds).
///     min_sep (int): Minimum separation between participations (>= 1).
///     max_participations (int | None): Optional upper bound.
///
/// Returns:
///     float: The sensitivity squared.
///
/// Raises:
///     ValueError: If parameters are invalid.
#[pyfunction]
#[pyo3(name = "toeplitz_minsep_sensitivity_squared", signature = (strategy_coef, n, min_sep=1, max_participations=None))]
pub fn py_toeplitz_minsep_sensitivity_squared(
    strategy_coef: Vec<f64>,
    n: usize,
    min_sep: usize,
    max_participations: Option<usize>,
) -> PyResult<f64> {
    crate::matrix_factorization::toeplitz_minsep_sensitivity_squared(
        &strategy_coef,
        n,
        min_sep,
        max_participations,
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}
