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
///     config (DiscretizationConfig): Discretization configuration (includes num_mc_samples and seed).
///
/// Returns:
///     Pld: The privacy loss distribution (asymmetric, remove + add).
///
/// Raises:
///     ValueError: If parameters are invalid.
#[pyfunction]
#[pyo3(name = "bnb_mc_pld", signature = (gram, num_bins, sigma, config))]
pub fn py_bnb_mc_pld(
    gram: Vec<f64>,
    num_bins: usize,
    sigma: f64,
    config: &PyDiscretizationConfig,
) -> PyResult<PyPld> {
    let pld = crate::amplification::bnb_mc_pld(&gram, num_bins, sigma, &config.inner)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyPld::new(pld))
}

/// Identity-specialized BnB Monte Carlo PLD with optional importance sampling.
///
/// Specialised for the BnB-Identity dominating pair (`C = I` ⇒ Gram is
/// `num_epochs · I_b`, diagonal). Skips the Cholesky step, fixes the shifted
/// bin to index 0 by symmetry, and optionally tilts the proposal on the
/// shifted-bin standard-normal coordinate to concentrate samples in the
/// large-`Y` tail (Importance Sampling).
///
/// Args:
///     num_bins (int): Number of bins ``b ≥ 2``.
///     num_epochs (int): Per-bin participation count ``E ≥ 1``.
///     sigma (float): Raw noise multiplier ``σ > 0``.
///     importance_tilt (float): Tilt parameter ``τ`` for the IS proposal on
///         the shifted-bin coordinate ``z_0`` (under ``P``). ``0`` disables
///         IS (plain identity-specialised MC). Positive values reduce MC
///         variance for typical privacy regimes by oversampling the
///         large-``Y`` tail; the importance weight reweights the estimator
///         back to ``P``.
///     config (DiscretizationConfig): Discretisation configuration
///         (``num_mc_samples``, ``seed``).
///
/// Returns:
///     Pld: The privacy loss distribution (asymmetric, remove + add).
#[pyfunction]
#[pyo3(
    name = "bnb_mc_pld_identity",
    signature = (num_bins, num_epochs, sigma, importance_tilt, config)
)]
pub fn py_bnb_mc_pld_identity(
    num_bins: usize,
    num_epochs: usize,
    sigma: f64,
    importance_tilt: f64,
    config: &PyDiscretizationConfig,
) -> PyResult<PyPld> {
    let pld = crate::amplification::bnb_mc_pld_identity(
        num_bins,
        num_epochs,
        sigma,
        importance_tilt,
        &config.inner,
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyPld::new(pld))
}

/// Monte Carlo PLD for BandMF with warm-start b-min-sep subsampling (Dong & Ganesh, arXiv:2602.09338).
///
/// Args:
///     strategy_coef: First column of the BandMF strategy matrix C (length = bandwidth).
///     n_steps: Total training iterations n.
///     p: Per-iteration Poisson inclusion probability p (Algorithm 2), not the per-example rate p_0.
///     sigma: Raw noise multiplier σ (same as BandMf noise_multiplier).
///     config: Discretization configuration (includes num_mc_samples and seed).
#[pyfunction]
#[pyo3(name = "bandmf_b_min_sep_warm_mc_pld", signature = (strategy_coef, n_steps, p, sigma, config))]
pub fn py_bandmf_b_min_sep_warm_mc_pld(
    strategy_coef: Vec<f64>,
    n_steps: usize,
    p: f64,
    sigma: f64,
    config: &PyDiscretizationConfig,
) -> PyResult<PyPld> {
    let pld = crate::amplification::bandmf_b_min_sep_warm_mc_pld(
        &strategy_coef,
        n_steps,
        p,
        sigma,
        &config.inner,
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyPld::new(pld))
}

#[pyfunction]
#[pyo3(
    name = "register_b_min_sep_transcript_corpus",
    signature = (strategy_coef, n_steps, p, num_samples, seed)
)]
pub fn py_register_b_min_sep_transcript_corpus(
    strategy_coef: Vec<f64>,
    n_steps: usize,
    p: f64,
    num_samples: usize,
    seed: u64,
) -> PyResult<u64> {
    crate::amplification::register_b_min_sep_transcripts(
        &strategy_coef,
        n_steps,
        p,
        num_samples,
        seed,
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(name = "drop_b_min_sep_transcript_corpus", signature = (handle))]
pub fn py_drop_b_min_sep_transcript_corpus(handle: u64) {
    crate::amplification::drop_b_min_sep_transcript_handle(handle);
}

#[pyfunction]
#[pyo3(
    name = "bandmf_b_min_sep_pld_from_transcript_handle",
    signature = (handle, strategy_coef, n_steps, p, sigma, config)
)]
pub fn py_bandmf_b_min_sep_pld_from_transcript_handle(
    handle: u64,
    strategy_coef: Vec<f64>,
    n_steps: usize,
    p: f64,
    sigma: f64,
    config: &PyDiscretizationConfig,
) -> PyResult<PyPld> {
    let pld = crate::amplification::pld_from_transcript_handle(
        handle,
        &strategy_coef,
        n_steps,
        p,
        sigma,
        &config.inner,
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyPld::new(pld))
}
