//! Amplification PLD constructors: Poisson, truncated-Poisson, parallel Poisson.

use pyo3::prelude::*;

use super::config::PyDiscretizationConfig;
use super::pld::PyPld;

/// Compute the PLD for a Poisson-subsampled Gaussian mechanism.
///
/// Args:
///     noise_multiplier (float): σ/Δ ratio, must be > 0.
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
///     noise_multiplier (float): σ/Δ ratio, must be > 0.
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
///     noise_multiplier (float): σ/Δ ratio, must be > 0.
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

/// Compute the per-epoch PLD for a Balls-in-Bins Gaussian mechanism.
///
/// The dataset is randomly partitioned into ``num_bins`` equally-sized bins
/// each epoch. Each bin is processed once with a Gaussian mechanism, so every
/// example participates exactly once per epoch.
///
/// Uses a conservative Poisson per-step approximation composed ``num_bins`` times.
///
/// Args:
///     noise_multiplier (float): σ/Δ ratio, must be > 0.
///     num_bins (int): Number of bins (k ≥ 2).
///     config (DiscretizationConfig): Discretization configuration.
///
/// Returns:
///     Pld: The per-epoch privacy loss distribution.
#[pyfunction]
#[pyo3(name = "balls_in_bins_gaussian_pld", signature = (noise_multiplier, num_bins, config))]
pub fn py_balls_in_bins_gaussian_pld(
    noise_multiplier: f64,
    num_bins: usize,
    config: &PyDiscretizationConfig,
) -> PyResult<PyPld> {
    let pld =
        crate::amplification::balls_in_bins_gaussian_pld(noise_multiplier, num_bins, &config.inner)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyPld::new(pld))
}

/// Compute the **total** multi-epoch PLD for a Balls-in-Bins Gaussian mechanism.
///
/// Equivalent to ``balls_in_bins_gaussian_pld(...).self_compose(num_epochs)``
/// but done in a single call for clarity: the result IS the total cost.
///
/// Args:
///     noise_multiplier (float): σ/Δ ratio, must be > 0.
///     num_bins (int): Number of bins per epoch (k ≥ 2).
///     num_epochs (int): Number of training epochs (≥ 1).
///     config (DiscretizationConfig): Discretization configuration.
///
/// Returns:
///     Pld: The total privacy loss distribution for all epochs.
#[pyfunction]
#[pyo3(name = "balls_in_bins_gaussian_pld_epochs", signature = (noise_multiplier, num_bins, num_epochs, config))]
pub fn py_balls_in_bins_gaussian_pld_epochs(
    noise_multiplier: f64,
    num_bins: usize,
    num_epochs: usize,
    config: &PyDiscretizationConfig,
) -> PyResult<PyPld> {
    let pld = crate::amplification::balls_in_bins_gaussian_pld_epochs(
        noise_multiplier,
        num_bins,
        num_epochs,
        &config.inner,
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyPld::new(pld))
}

/// Compute the BnB PLD via Monte Carlo sampling of the dominating pair.
///
/// Uses Algorithm 2 of Choquette-Choo et al. (2024) "Near Exact Privacy
/// Amplification for Matrix Mechanisms" (arxiv:2410.06266).
///
/// The dominating pair is P = (1/b) Σ N(m_i, σ²I), Q = N(0, σ²I), where
/// the Gram matrix G captures the inner products ⟨m_i, m_j⟩.
///
/// Args:
///     gram (list[float]): Flattened row-major b×b Gram matrix.
///     num_bins (int): Number of bins b.
///     sigma (float): Noise multiplier σ, must be > 0.
///     num_samples (int): Number of Monte Carlo samples.
///     seed (int): RNG seed for reproducibility.
///     config (DiscretizationConfig): Discretization configuration.
///
/// Returns:
///     Pld: The privacy loss distribution (asymmetric, remove + add).
///
/// Raises:
///     ValueError: If parameters are invalid.
#[pyfunction]
#[pyo3(name = "bnb_mc_pld", signature = (gram, num_bins, sigma, num_samples, seed, config))]
pub fn py_bnb_mc_pld(
    gram: Vec<f64>,
    num_bins: usize,
    sigma: f64,
    num_samples: usize,
    seed: u64,
    config: &PyDiscretizationConfig,
) -> PyResult<PyPld> {
    let pld =
        crate::amplification::bnb_mc_pld(&gram, num_bins, sigma, num_samples, seed, &config.inner)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyPld::new(pld))
}
