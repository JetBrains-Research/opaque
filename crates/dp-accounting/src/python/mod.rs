//! PyO3 Python bindings for the functional DP accounting API
//!
//! Exposes the full functional API as Python classes and functions.
//!
//! The bindings use type erasure via `DynProcess` to present a unified
//! `DpProcess` Python class. The trait-object approach stores the concrete
//! Rust type behind `Box<dyn DynProcess>`, keeping full PLD computation
//! capability (needed for composition).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::error::PldError;
use crate::functional::amplification::{accumulate, poisson, truncated_poisson};
use crate::functional::composition::{compose, repeat};
use crate::functional::discretization::DiscretizationConfig;
use crate::functional::mechanisms::eps_delta::{eps_delta, eps_delta_with};
use crate::functional::mechanisms::gaussian::{gaussian, gaussian_with};
use crate::functional::mechanisms::identity::{identity, identity_with};
use crate::functional::pld::PrivacyLossDistribution;
use crate::functional::process::Process;
use crate::functional::transforms::adaclip::adaclip;

// ---------------------------------------------------------------------------
// Error conversion
// ---------------------------------------------------------------------------

fn to_py_err(e: PldError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

// ---------------------------------------------------------------------------
// DynProcess: Type-erased Process trait object
// ---------------------------------------------------------------------------

/// Type-erased wrapper for any `Process` implementation.
///
/// This trait adds `clone_box` and `compute_pld` to allow dynamic dispatch
/// while retaining full PLD computation capability for composition.
trait DynProcess: Send + Sync {
    fn compute_pld(&self) -> Result<PrivacyLossDistribution, PldError>;
    fn clone_box(&self) -> Box<dyn DynProcess>;
}

impl<P: Process + Clone + Send + Sync + 'static> DynProcess for P {
    fn compute_pld(&self) -> Result<PrivacyLossDistribution, PldError> {
        self.pld()
    }
    fn clone_box(&self) -> Box<dyn DynProcess> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn DynProcess> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

// ---------------------------------------------------------------------------
// ProcessWrapper: bridges DynProcess → Process trait
// ---------------------------------------------------------------------------

/// Wrapper that implements `Process` by delegating to a `DynProcess` trait object.
///
/// This allows type-erased processes (from Python) to participate in
/// Rust composition (`repeat`, `compose`) which requires the `Process` trait.
#[derive(Clone)]
struct ProcessWrapper(Box<dyn DynProcess>);

impl PartialEq for ProcessWrapper {
    fn eq(&self, _other: &Self) -> bool {
        // Type-erased processes can't be structurally compared;
        // this is only needed to satisfy Repeated<P: PartialEq> bounds.
        false
    }
}

impl Process for ProcessWrapper {
    fn pld(&self) -> crate::error::Result<PrivacyLossDistribution> {
        self.0.compute_pld()
    }
}

// ---------------------------------------------------------------------------
// Python classes
// ---------------------------------------------------------------------------

/// A differential privacy process that can be queried for privacy guarantees.
///
/// This is the main class returned by all mechanism constructors (gaussian, poisson, etc.).
/// Use methods like ``epsilon_at(delta)`` or ``delta_at(epsilon)`` to query privacy guarantees.
///
/// Examples::
///
///     import opaque_dp_accounting as acc
///
///     # Single Gaussian mechanism
///     proc = acc.py_gaussian(1.1)
///     eps = proc.epsilon_at(1e-5)
///
///     # Poisson-subsampled Gaussian, 1000 steps
///     step = acc.py_poisson(1.1, 0.01)
///     training = acc.py_repeat(step, 1000)
///     eps = training.epsilon_at(1e-5)
#[pyclass(name = "DpProcess")]
#[derive(Clone)]
struct PyDpProcess {
    inner: Box<dyn DynProcess>,
    description: String,
}

#[pymethods]
impl PyDpProcess {
    /// Compute epsilon for a given delta target.
    ///
    /// Args:
    ///     delta: The delta parameter (probability of privacy breach)
    ///
    /// Returns:
    ///     The epsilon value achieving the given delta
    #[pyo3(text_signature = "(self, delta)")]
    fn epsilon_at(&self, delta: f64) -> PyResult<f64> {
        let pld = self.inner.compute_pld().map_err(to_py_err)?;
        Ok(pld.epsilon_at(delta))
    }

    /// Compute delta for a given epsilon target.
    ///
    /// Args:
    ///     epsilon: The epsilon parameter (privacy budget)
    ///
    /// Returns:
    ///     The delta value at the given epsilon
    #[pyo3(text_signature = "(self, epsilon)")]
    fn delta_at(&self, epsilon: f64) -> PyResult<f64> {
        let pld = self.inner.compute_pld().map_err(to_py_err)?;
        Ok(pld.delta_at(epsilon))
    }

    /// Compute the advantage (total variation distance / hockey-stick at eps=0).
    ///
    /// Returns:
    ///     The advantage value
    fn advantage(&self) -> PyResult<f64> {
        let pld = self.inner.compute_pld().map_err(to_py_err)?;
        Ok(pld.advantage())
    }

    /// Compute beta (type II error) at a given alpha (type I error).
    ///
    /// Args:
    ///     alpha: The type I error (false positive rate)
    ///
    /// Returns:
    ///     The beta value (type II error / false negative rate)
    #[pyo3(text_signature = "(self, alpha)")]
    fn beta_at(&self, alpha: f64) -> PyResult<f64> {
        let pld = self.inner.compute_pld().map_err(to_py_err)?;
        Ok(pld.beta_at(alpha))
    }

    /// Compute the Bayes risk at a given prior probability.
    ///
    /// Args:
    ///     prior: Prior probability of the sensitive hypothesis
    ///
    /// Returns:
    ///     The Bayes risk under optimal adversary
    #[pyo3(text_signature = "(self, prior)")]
    fn risk_at(&self, prior: f64) -> PyResult<f64> {
        let pld = self.inner.compute_pld().map_err(to_py_err)?;
        Ok(pld.risk_at(prior))
    }

    fn __repr__(&self) -> String {
        format!("DpProcess({})", self.description)
    }
}

/// Discretization configuration for PLD computation.
///
/// Controls the precision of Privacy Loss Distribution discretization.
///
/// Args:
///     discretization: Grid spacing (default: 1e-4, smaller = more precise)
///     log_mass_truncation_bound: Log of mass to truncate (default: -32.0)
///     pessimistic_estimate: Use pessimistic (upper-bound) estimates (default: True)
///     max_grid_size: Maximum grid size (default: 10_000_000)
#[pyclass(name = "DiscretizationConfig")]
#[derive(Clone)]
struct PyDiscretizationConfig {
    inner: DiscretizationConfig,
}

#[pymethods]
impl PyDiscretizationConfig {
    #[new]
    #[pyo3(signature = (discretization=1e-4, log_mass_truncation_bound=-32.0, pessimistic_estimate=true, max_grid_size=10_000_000))]
    fn new(
        discretization: f64,
        log_mass_truncation_bound: f64,
        pessimistic_estimate: bool,
        max_grid_size: usize,
    ) -> PyResult<Self> {
        let config =
            DiscretizationConfig::with_estimate(discretization, log_mass_truncation_bound, pessimistic_estimate)
                .map_err(to_py_err)?;
        let config = config.with_max_grid_size(max_grid_size);
        Ok(Self { inner: config })
    }

    fn __repr__(&self) -> String {
        format!(
            "DiscretizationConfig(discretization={}, log_mass_truncation_bound={})",
            self.inner.discretization, self.inner.log_mass_truncation_bound
        )
    }
}

// ---------------------------------------------------------------------------
// Helper: build a Gaussian with optional config
// ---------------------------------------------------------------------------

fn make_gaussian(
    noise_multiplier: f64,
    config: Option<&PyDiscretizationConfig>,
) -> Result<crate::functional::mechanisms::Gaussian, PldError> {
    match config {
        Some(c) => gaussian_with(noise_multiplier, c.inner.clone()),
        None => gaussian(noise_multiplier),
    }
}

// ---------------------------------------------------------------------------
// Module functions — mechanisms
// ---------------------------------------------------------------------------

/// Create a Gaussian mechanism with the given noise multiplier.
///
/// Args:
///     noise_multiplier: The ratio sigma/sensitivity (must be > 0)
///     config: Optional discretization config
///
/// Returns:
///     A DpProcess representing the Gaussian mechanism
#[pyfunction]
#[pyo3(signature = (noise_multiplier, config=None))]
fn py_gaussian(
    noise_multiplier: f64,
    config: Option<PyDiscretizationConfig>,
) -> PyResult<PyDpProcess> {
    let g = make_gaussian(noise_multiplier, config.as_ref()).map_err(to_py_err)?;
    Ok(PyDpProcess {
        description: format!("Gaussian(nm={})", noise_multiplier),
        inner: Box::new(g),
    })
}

/// Create an (epsilon, delta)-DP mechanism with specified guarantees.
///
/// Args:
///     epsilon: The epsilon parameter
///     delta: The delta parameter (default: 0.0 for pure DP)
///     config: Optional discretization config
///
/// Returns:
///     A DpProcess representing the (eps, delta)-DP mechanism
#[pyfunction]
#[pyo3(signature = (epsilon, delta=0.0, config=None))]
fn py_eps_delta(
    epsilon: f64,
    delta: f64,
    config: Option<PyDiscretizationConfig>,
) -> PyResult<PyDpProcess> {
    let e = match config {
        Some(c) => eps_delta_with(epsilon, delta, c.inner).map_err(to_py_err)?,
        None => eps_delta(epsilon, delta).map_err(to_py_err)?,
    };
    Ok(PyDpProcess {
        description: format!("EpsDelta(eps={}, delta={})", epsilon, delta),
        inner: Box::new(e),
    })
}

/// Create an identity (perfect privacy) mechanism.
///
/// Returns:
///     A DpProcess with zero privacy loss
#[pyfunction]
#[pyo3(signature = (config=None))]
fn py_identity(config: Option<PyDiscretizationConfig>) -> PyResult<PyDpProcess> {
    let i = match config {
        Some(c) => identity_with(c.inner),
        None => identity(),
    };
    Ok(PyDpProcess {
        description: "Identity".to_string(),
        inner: Box::new(i),
    })
}

// ---------------------------------------------------------------------------
// Module functions — amplification
// ---------------------------------------------------------------------------

/// Apply Poisson subsampling amplification to a Gaussian mechanism.
///
/// Each record is included independently with probability ``rate``,
/// providing privacy amplification.
///
/// Args:
///     noise_multiplier: Gaussian noise multiplier (sigma/sensitivity)
///     rate: Poisson sampling rate q in (0, 1]
///     config: Optional discretization config
///
/// Returns:
///     A DpProcess with amplified privacy guarantees
#[pyfunction]
#[pyo3(signature = (noise_multiplier, rate, config=None))]
fn py_poisson(
    noise_multiplier: f64,
    rate: f64,
    config: Option<PyDiscretizationConfig>,
) -> PyResult<PyDpProcess> {
    let g = make_gaussian(noise_multiplier, config.as_ref()).map_err(to_py_err)?;
    let p = poisson(g, rate);
    Ok(PyDpProcess {
        description: format!("Poisson(Gaussian(nm={}), q={})", noise_multiplier, rate),
        inner: Box::new(p),
    })
}

/// Apply truncated Poisson subsampling (production DP-SGD).
///
/// Like Poisson sampling but caps the batch size at ``batch_size_max``.
/// This models what Opacus, JAX Privacy, and TF Privacy actually use.
///
/// Args:
///     noise_multiplier: Gaussian noise multiplier
///     rate: Poisson sampling rate q in (0, 1]
///     batch_size_max: Maximum batch size B_max
///     dataset_size: Total dataset size n
///     config: Optional discretization config
///
/// Returns:
///     A DpProcess with truncated Poisson amplification
#[pyfunction]
#[pyo3(signature = (noise_multiplier, rate, batch_size_max, dataset_size, config=None))]
fn py_truncated_poisson(
    noise_multiplier: f64,
    rate: f64,
    batch_size_max: usize,
    dataset_size: usize,
    config: Option<PyDiscretizationConfig>,
) -> PyResult<PyDpProcess> {
    let g = make_gaussian(noise_multiplier, config.as_ref()).map_err(to_py_err)?;
    let tp = truncated_poisson(g, rate, batch_size_max, dataset_size);
    Ok(PyDpProcess {
        description: format!(
            "TruncatedPoisson(Gaussian(nm={}), q={}, B_max={}, n={})",
            noise_multiplier, rate, batch_size_max, dataset_size
        ),
        inner: Box::new(tp),
    })
}

/// Apply gradient accumulation to a Poisson-subsampled Gaussian.
///
/// Models ``m`` microbatches with a single noise addition (Mixture of Gaussians).
///
/// Args:
///     noise_multiplier: Gaussian noise multiplier
///     rate: Poisson sampling rate per microbatch
///     microbatches: Number of microbatches m
///     config: Optional discretization config
///
/// Returns:
///     A DpProcess with accumulated privacy guarantees
#[pyfunction]
#[pyo3(signature = (noise_multiplier, rate, microbatches, config=None))]
fn py_accumulate(
    noise_multiplier: f64,
    rate: f64,
    microbatches: usize,
    config: Option<PyDiscretizationConfig>,
) -> PyResult<PyDpProcess> {
    let g = make_gaussian(noise_multiplier, config.as_ref()).map_err(to_py_err)?;
    let p = poisson(g, rate);
    let acc = accumulate(p, microbatches).map_err(to_py_err)?;
    Ok(PyDpProcess {
        description: format!(
            "Accumulate(Poisson(Gaussian(nm={}), q={}), m={})",
            noise_multiplier, rate, microbatches
        ),
        inner: Box::new(acc),
    })
}

// ---------------------------------------------------------------------------
// Module functions — transforms
// ---------------------------------------------------------------------------

/// Wrap a Gaussian mechanism with adaptive clipping (Andrew et al. 2021).
///
/// Args:
///     noise_multiplier: Gaussian noise multiplier
///     quantile_noise_std: Noise std for quantile estimation
///     config: Optional discretization config
///
/// Returns:
///     A DpProcess representing the AdaClip mechanism
#[pyfunction]
#[pyo3(signature = (noise_multiplier, quantile_noise_std, config=None))]
fn py_adaclip(
    noise_multiplier: f64,
    quantile_noise_std: f64,
    config: Option<PyDiscretizationConfig>,
) -> PyResult<PyDpProcess> {
    let g = make_gaussian(noise_multiplier, config.as_ref()).map_err(to_py_err)?;
    let ac = adaclip(g, quantile_noise_std);
    Ok(PyDpProcess {
        description: format!(
            "AdaClip(Gaussian(nm={}), sb={})",
            noise_multiplier, quantile_noise_std
        ),
        inner: Box::new(ac),
    })
}

/// Apply Poisson subsampling to an AdaClip-wrapped Gaussian mechanism.
///
/// Args:
///     noise_multiplier: Gaussian noise multiplier
///     quantile_noise_std: AdaClip quantile noise std
///     rate: Poisson sampling rate
///     config: Optional discretization config
///
/// Returns:
///     A DpProcess with Poisson-amplified AdaClip
#[pyfunction]
#[pyo3(signature = (noise_multiplier, quantile_noise_std, rate, config=None))]
fn py_poisson_adaclip(
    noise_multiplier: f64,
    quantile_noise_std: f64,
    rate: f64,
    config: Option<PyDiscretizationConfig>,
) -> PyResult<PyDpProcess> {
    let g = make_gaussian(noise_multiplier, config.as_ref()).map_err(to_py_err)?;
    let ac = adaclip(g, quantile_noise_std);
    let p = poisson(ac, rate);
    Ok(PyDpProcess {
        description: format!(
            "Poisson(AdaClip(Gaussian(nm={}), sb={}), q={})",
            noise_multiplier, quantile_noise_std, rate
        ),
        inner: Box::new(p),
    })
}

// ---------------------------------------------------------------------------
// Module functions — composition
// ---------------------------------------------------------------------------

/// Compose a process with itself ``count`` times (homogeneous composition).
///
/// This uses FFT-based self-convolution for efficient computation.
///
/// Args:
///     process: A DpProcess to repeat
///     count: Number of repetitions k
///
/// Returns:
///     A DpProcess representing k-fold composition
#[pyfunction]
#[pyo3(text_signature = "(process, count)")]
fn py_repeat(process: PyDpProcess, count: usize) -> PyResult<PyDpProcess> {
    let wrapper = ProcessWrapper(process.inner.clone());
    let r = repeat(wrapper, count).map_err(to_py_err)?;
    Ok(PyDpProcess {
        description: format!("Repeat({}, k={})", process.description, count),
        inner: Box::new(r),
    })
}

/// Compose two heterogeneous processes.
///
/// Args:
///     left: First DpProcess
///     right: Second DpProcess
///
/// Returns:
///     A DpProcess representing the composition of both
#[pyfunction]
#[pyo3(text_signature = "(left, right)")]
fn py_compose(left: PyDpProcess, right: PyDpProcess) -> PyResult<PyDpProcess> {
    let lw = ProcessWrapper(left.inner.clone());
    let rw = ProcessWrapper(right.inner.clone());
    let c = compose(lw, rw);
    Ok(PyDpProcess {
        description: format!("Compose({}, {})", left.description, right.description),
        inner: Box::new(c),
    })
}

// ---------------------------------------------------------------------------
// Module functions — calibration & convenience
// ---------------------------------------------------------------------------

/// Calibrate the noise multiplier for a DP-SGD training run.
///
/// Uses bisection search to find the smallest noise multiplier that
/// achieves the target privacy guarantee.
///
/// Args:
///     target_epsilon: Target epsilon value
///     target_delta: Target delta value
///     sample_rate: Poisson sampling rate q (= batch_size / dataset_size)
///     num_steps: Number of training steps (composition count)
///     param_min: Minimum noise multiplier to search (default: 0.1)
///     param_max: Maximum noise multiplier to search (default: 100.0)
///     tolerance: Convergence tolerance (default: 1e-6)
///     max_iterations: Maximum bisection iterations (default: 100)
///
/// Returns:
///     The calibrated noise multiplier
#[pyfunction]
#[pyo3(signature = (target_epsilon, target_delta, sample_rate, num_steps, param_min=0.1, param_max=100.0, tolerance=1e-6, max_iterations=100))]
fn py_calibrate_noise(
    target_epsilon: f64,
    target_delta: f64,
    sample_rate: f64,
    num_steps: usize,
    param_min: f64,
    param_max: f64,
    tolerance: f64,
    max_iterations: usize,
) -> PyResult<f64> {
    let mut lo = param_min;
    let mut hi = param_max;

    for _ in 0..max_iterations {
        let mid = (lo + hi) / 2.0;
        if (hi - lo) / 2.0 < tolerance {
            return Ok(mid);
        }

        let g = gaussian(mid).map_err(to_py_err)?;
        let step = poisson(g, sample_rate);
        let training = repeat(step, num_steps).map_err(to_py_err)?;
        let eps = training.epsilon_at(target_delta).map_err(to_py_err)?;

        if eps > target_epsilon {
            lo = mid; // Need more noise
        } else {
            hi = mid; // Can use less noise
        }
    }

    Ok((lo + hi) / 2.0)
}

/// Compute epsilon for a DP-SGD training run (convenience function).
///
/// This is the most common use case: Poisson-subsampled Gaussian, composed
/// over ``num_steps`` training steps.
///
/// Args:
///     noise_multiplier: Gaussian noise multiplier (sigma / sensitivity)
///     sample_rate: Poisson sampling rate q (= batch_size / dataset_size)
///     num_steps: Number of training steps
///     delta: Target delta value
///
/// Returns:
///     The epsilon value
#[pyfunction]
#[pyo3(signature = (noise_multiplier, sample_rate, num_steps, delta))]
fn py_compute_epsilon(
    noise_multiplier: f64,
    sample_rate: f64,
    num_steps: usize,
    delta: f64,
) -> PyResult<f64> {
    let g = gaussian(noise_multiplier).map_err(to_py_err)?;
    let step = poisson(g, sample_rate);
    let training = repeat(step, num_steps).map_err(to_py_err)?;
    training.epsilon_at(delta).map_err(to_py_err)
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Classes
    m.add_class::<PyDpProcess>()?;
    m.add_class::<PyDiscretizationConfig>()?;

    // Mechanisms
    m.add_function(wrap_pyfunction!(py_gaussian, m)?)?;
    m.add_function(wrap_pyfunction!(py_eps_delta, m)?)?;
    m.add_function(wrap_pyfunction!(py_identity, m)?)?;

    // Amplification
    m.add_function(wrap_pyfunction!(py_poisson, m)?)?;
    m.add_function(wrap_pyfunction!(py_truncated_poisson, m)?)?;
    m.add_function(wrap_pyfunction!(py_accumulate, m)?)?;

    // Transforms
    m.add_function(wrap_pyfunction!(py_adaclip, m)?)?;
    m.add_function(wrap_pyfunction!(py_poisson_adaclip, m)?)?;

    // Composition
    m.add_function(wrap_pyfunction!(py_repeat, m)?)?;
    m.add_function(wrap_pyfunction!(py_compose, m)?)?;

    // Calibration & convenience
    m.add_function(wrap_pyfunction!(py_calibrate_noise, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_epsilon, m)?)?;

    Ok(())
}
