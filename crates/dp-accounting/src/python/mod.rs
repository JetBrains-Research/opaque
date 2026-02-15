//! PyO3 Python bindings for the functional DP accounting API.
//!
//! Exposes a Pythonic API where functions have natural names (`gaussian`,
//! not `py_gaussian`), processes support operators (`step * 1000`,
//! `a | b`), and rich introspection is available for debugging.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::time::Instant;

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
// DynProcess: type-erased Process trait object
// ---------------------------------------------------------------------------

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

#[derive(Clone)]
struct ProcessWrapper(Box<dyn DynProcess>);

impl PartialEq for ProcessWrapper {
    fn eq(&self, _other: &Self) -> bool {
        false
    }
}

impl Process for ProcessWrapper {
    fn pld(&self) -> crate::error::Result<PrivacyLossDistribution> {
        self.0.compute_pld()
    }
}

// ---------------------------------------------------------------------------
// PyDpProcess
// ---------------------------------------------------------------------------

/// A differential privacy process that can be queried for privacy guarantees.
///
/// Construct with module-level functions like :func:`gaussian`,
/// :func:`poisson`, etc.  Compose with :func:`repeat` / :func:`compose`,
/// or use the operator shorthands::
///
///     import opaque_dp_accounting as dp
///
///     step = dp.poisson(1.1, 0.01)
///     training = step * 1000          # same as dp.repeat(step, 1000)
///     eps = training.epsilon_at(1e-5)
///
///     # heterogeneous composition
///     combined = step | dp.gaussian(0.8)
///
///     # debugging
///     print(training)                 # human-readable summary
///     training.describe()             # dict of parameters
///     training.pld_info()             # PLD grid diagnostics
///     training.summary(delta=1e-5)    # formatted privacy summary
#[pyclass(name = "DpProcess")]
#[derive(Clone)]
struct PyDpProcess {
    inner: Box<dyn DynProcess>,
    /// Machine-readable tag, e.g. "Gaussian(noise_multiplier=1.1)"
    label: String,
    /// Preserved kwargs from the constructor for introspection.
    params: Vec<(String, ParamValue)>,
}

/// Parameter values that can be stored for introspection.
#[derive(Clone, Debug)]
enum ParamValue {
    Float(f64),
    Int(usize),
    Str(String),
}

impl IntoPy<PyObject> for ParamValue {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            ParamValue::Float(f) => f.into_py(py),
            ParamValue::Int(i) => i.into_py(py),
            ParamValue::Str(s) => s.into_py(py),
        }
    }
}

impl PyDpProcess {
    fn new(inner: Box<dyn DynProcess>, label: String, params: Vec<(&str, ParamValue)>) -> Self {
        Self {
            inner,
            label,
            params: params.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        }
    }
}

#[pymethods]
impl PyDpProcess {
    // ---- metrics ----------------------------------------------------------

    /// Compute epsilon for a given delta.
    ///
    /// Args:
    ///     delta: Probability of privacy breach.
    ///
    /// Returns:
    ///     float: The smallest epsilon such that the mechanism is (epsilon, delta)-DP.
    #[pyo3(text_signature = "(self, delta)")]
    fn epsilon_at(&self, delta: f64) -> PyResult<f64> {
        let pld = self.inner.compute_pld().map_err(to_py_err)?;
        Ok(pld.epsilon_at(delta))
    }

    /// Compute delta for a given epsilon.
    ///
    /// Args:
    ///     epsilon: Privacy budget.
    ///
    /// Returns:
    ///     float: The delta value at the given epsilon.
    #[pyo3(text_signature = "(self, epsilon)")]
    fn delta_at(&self, epsilon: f64) -> PyResult<f64> {
        let pld = self.inner.compute_pld().map_err(to_py_err)?;
        Ok(pld.delta_at(epsilon))
    }

    /// Total-variation advantage (= delta at epsilon=0).
    fn advantage(&self) -> PyResult<f64> {
        let pld = self.inner.compute_pld().map_err(to_py_err)?;
        Ok(pld.advantage())
    }

    /// Type-II error (beta) at a given type-I error (alpha).
    #[pyo3(text_signature = "(self, alpha)")]
    fn beta_at(&self, alpha: f64) -> PyResult<f64> {
        let pld = self.inner.compute_pld().map_err(to_py_err)?;
        Ok(pld.beta_at(alpha))
    }

    /// Bayes risk under an optimal adversary.
    #[pyo3(text_signature = "(self, prior)")]
    fn risk_at(&self, prior: f64) -> PyResult<f64> {
        let pld = self.inner.compute_pld().map_err(to_py_err)?;
        Ok(pld.risk_at(prior))
    }

    // ---- operators --------------------------------------------------------

    /// ``process * k`` is shorthand for ``repeat(process, k)``.
    fn __mul__(&self, count: usize) -> PyResult<PyDpProcess> {
        let wrapper = ProcessWrapper(self.inner.clone());
        let r = repeat(wrapper, count).map_err(to_py_err)?;
        Ok(PyDpProcess::new(
            Box::new(r),
            format!("Repeat({}, k={})", self.label, count),
            vec![
                ("inner", ParamValue::Str(self.label.clone())),
                ("count", ParamValue::Int(count)),
            ],
        ))
    }

    /// ``k * process`` also works (reflected multiply).
    fn __rmul__(&self, count: usize) -> PyResult<PyDpProcess> {
        self.__mul__(count)
    }

    /// ``a | b`` is shorthand for ``compose(a, b)``.
    fn __or__(&self, other: &PyDpProcess) -> PyResult<PyDpProcess> {
        let lw = ProcessWrapper(self.inner.clone());
        let rw = ProcessWrapper(other.inner.clone());
        let c = compose(lw, rw);
        Ok(PyDpProcess::new(
            Box::new(c),
            format!("Compose({}, {})", self.label, other.label),
            vec![
                ("left", ParamValue::Str(self.label.clone())),
                ("right", ParamValue::Str(other.label.clone())),
            ],
        ))
    }

    // ---- introspection / debugging ----------------------------------------

    /// Return constructor parameters as a dict.
    ///
    /// >>> dp.poisson(1.1, 0.01).describe()
    /// {'type': 'Poisson', 'noise_multiplier': 1.1, 'sample_rate': 0.01}
    fn describe(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("type", &self.label)?;
        for (k, v) in &self.params {
            dict.set_item(k, v.clone().into_py(py))?;
        }
        Ok(dict.into())
    }

    /// Compute the PLD and return diagnostic info about the internal grid.
    ///
    /// This is useful for understanding numerical precision and debugging
    /// unexpected results.
    ///
    /// Returns:
    ///     dict with keys: ``grid_size``, ``discretization``, ``lower_index``,
    ///     ``upper_index``, ``infinity_mass``, ``neg_infinity_mass``,
    ///     ``pessimistic``, ``total_mass``, ``elapsed_ms``.
    fn pld_info(&self, py: Python<'_>) -> PyResult<PyObject> {
        let start = Instant::now();
        let pld = self.inner.compute_pld().map_err(to_py_err)?;
        let elapsed = start.elapsed();

        let pmf = &pld.pmf_remove;
        let dict = PyDict::new(py);
        dict.set_item("grid_size", pmf.probs.len())?;
        dict.set_item("discretization", pmf.discretization)?;
        dict.set_item("lower_index", pmf.lower_loss_index)?;
        dict.set_item(
            "upper_index",
            pmf.lower_loss_index + pmf.probs.len() as i64 - 1,
        )?;
        dict.set_item("infinity_mass", pmf.infinity_mass)?;
        dict.set_item("neg_infinity_mass", pmf.negative_infinity_mass)?;
        dict.set_item("pessimistic", pmf.pessimistic_estimate)?;
        let total: f64 =
            pmf.probs.iter().sum::<f64>() + pmf.infinity_mass + pmf.negative_infinity_mass;
        dict.set_item("total_mass", total)?;
        dict.set_item("is_symmetric", pld.pmf_add.is_none())?;
        dict.set_item("elapsed_ms", elapsed.as_secs_f64() * 1000.0)?;
        Ok(dict.into())
    }

    /// Print a human-readable privacy summary.
    ///
    /// Args:
    ///     delta: Delta for epsilon computation (default 1e-5).
    ///     epsilon: Epsilon for delta computation (default 1.0).
    ///     alpha: Type-I error for beta computation (default 0.05).
    ///     prior: Prior for risk computation (default 0.5).
    ///
    /// Returns:
    ///     str: Formatted multi-line summary.
    #[pyo3(signature = (delta=1e-5, epsilon=1.0, alpha=0.05, prior=0.5))]
    fn summary(&self, delta: f64, epsilon: f64, alpha: f64, prior: f64) -> PyResult<String> {
        let start = Instant::now();
        let pld = self.inner.compute_pld().map_err(to_py_err)?;
        let pld_ms = start.elapsed().as_secs_f64() * 1000.0;

        let eps_val = pld.epsilon_at(delta);
        let delta_val = pld.delta_at(epsilon);
        let adv = pld.advantage();
        let beta_val = pld.beta_at(alpha);
        let risk_val = pld.risk_at(prior);

        let pmf = &pld.pmf_remove;
        let grid = pmf.probs.len();

        Ok(format!(
            concat!(
                "--- {label} ---\n",
                "epsilon(delta={delta:.0e})  = {eps:.6}\n",
                "delta(epsilon={epsilon})      = {dval:.6e}\n",
                "advantage                 = {adv:.6e}\n",
                "beta(alpha={alpha})        = {beta:.6}\n",
                "risk(prior={prior})        = {risk:.6}\n",
                "---\n",
                "PLD grid: {grid} bins, disc={disc}, inf_mass={inf:.2e}\n",
                "PLD computed in {ms:.1} ms\n",
            ),
            label = self.label,
            delta = delta,
            eps = eps_val,
            epsilon = epsilon,
            dval = delta_val,
            adv = adv,
            alpha = alpha,
            beta = beta_val,
            prior = prior,
            risk = risk_val,
            grid = grid,
            disc = pmf.discretization,
            inf = pmf.infinity_mass,
            ms = pld_ms,
        ))
    }

    // ---- dunder -----------------------------------------------------------

    fn __repr__(&self) -> String {
        format!("DpProcess({})", self.label)
    }

    fn __str__(&self) -> String {
        // Quick summary: try to compute epsilon cheaply.
        // If PLD computation is too heavy we fall back to label only.
        match self.inner.compute_pld() {
            Ok(pld) => {
                let eps = pld.epsilon_at(1e-5);
                format!("{} | eps(delta=1e-5)={:.6}", self.label, eps)
            }
            Err(_) => self.label.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// PyDiscretizationConfig
// ---------------------------------------------------------------------------

/// Configuration controlling PLD discretization precision.
///
/// Args:
///     discretization (float): Grid spacing (default 1e-4).
///     log_mass_truncation_bound (float): log2 threshold for tail truncation (default -32).
///     pessimistic_estimate (bool): Use pessimistic (upper-bound) rounding (default True).
///     max_grid_size (int): Maximum grid bins before coarsening (default 10M).
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
        let config = DiscretizationConfig::with_estimate(
            discretization,
            log_mass_truncation_bound,
            pessimistic_estimate,
        )
        .map_err(to_py_err)?;
        let config = config.with_max_grid_size(max_grid_size);
        Ok(Self { inner: config })
    }

    #[getter]
    fn discretization(&self) -> f64 {
        self.inner.discretization
    }
    #[getter]
    fn log_mass_truncation_bound(&self) -> f64 {
        self.inner.log_mass_truncation_bound
    }
    #[getter]
    fn pessimistic_estimate(&self) -> bool {
        self.inner.pessimistic_estimate
    }
    #[getter]
    fn max_grid_size(&self) -> usize {
        self.inner.max_grid_size
    }

    fn __repr__(&self) -> String {
        format!(
            "DiscretizationConfig(discretization={}, log_mass_truncation_bound={}, pessimistic_estimate={}, max_grid_size={})",
            self.inner.discretization,
            self.inner.log_mass_truncation_bound,
            self.inner.pessimistic_estimate,
            self.inner.max_grid_size,
        )
    }

    fn __eq__(&self, other: &PyDiscretizationConfig) -> bool {
        self.inner.discretization == other.inner.discretization
            && self.inner.log_mass_truncation_bound == other.inner.log_mass_truncation_bound
            && self.inner.pessimistic_estimate == other.inner.pessimistic_estimate
            && self.inner.max_grid_size == other.inner.max_grid_size
    }
}

// ---------------------------------------------------------------------------
// Helper
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
// Module-level functions — mechanisms
// ---------------------------------------------------------------------------

/// Create a Gaussian mechanism.
///
/// Args:
///     noise_multiplier (float): sigma / sensitivity. Must be in [0.1, 1.2].
///     config (DiscretizationConfig, optional): Custom discretization.
///
/// Returns:
///     DpProcess
///
/// Example::
///
///     proc = dp.gaussian(1.1)
///     proc.epsilon_at(1e-5)  # ~3.73
#[pyfunction]
#[pyo3(name = "gaussian", signature = (noise_multiplier, config=None))]
fn py_gaussian(
    noise_multiplier: f64,
    config: Option<PyDiscretizationConfig>,
) -> PyResult<PyDpProcess> {
    let g = make_gaussian(noise_multiplier, config.as_ref()).map_err(to_py_err)?;
    Ok(PyDpProcess::new(
        Box::new(g),
        format!("Gaussian(noise_multiplier={})", noise_multiplier),
        vec![("noise_multiplier", ParamValue::Float(noise_multiplier))],
    ))
}

/// Create an (epsilon, delta)-DP mechanism.
///
/// Args:
///     epsilon (float): Privacy parameter.
///     delta (float): Failure probability (default 0).
///     config (DiscretizationConfig, optional): Custom discretization.
///
/// Returns:
///     DpProcess
#[pyfunction]
#[pyo3(name = "eps_delta", signature = (epsilon, delta=0.0, config=None))]
fn py_eps_delta(
    epsilon: f64,
    delta: f64,
    config: Option<PyDiscretizationConfig>,
) -> PyResult<PyDpProcess> {
    let e = match config {
        Some(c) => eps_delta_with(epsilon, delta, c.inner).map_err(to_py_err)?,
        None => eps_delta(epsilon, delta).map_err(to_py_err)?,
    };
    Ok(PyDpProcess::new(
        Box::new(e),
        format!("EpsDelta(epsilon={}, delta={})", epsilon, delta),
        vec![
            ("epsilon", ParamValue::Float(epsilon)),
            ("delta", ParamValue::Float(delta)),
        ],
    ))
}

/// Create an identity (zero privacy loss) mechanism.
///
/// Returns:
///     DpProcess
#[pyfunction]
#[pyo3(name = "identity", signature = (config=None))]
fn py_identity(config: Option<PyDiscretizationConfig>) -> PyResult<PyDpProcess> {
    let i = match config {
        Some(c) => identity_with(c.inner),
        None => identity(),
    };
    Ok(PyDpProcess::new(
        Box::new(i),
        "Identity()".to_string(),
        vec![],
    ))
}

// ---------------------------------------------------------------------------
// Module-level functions — amplification
// ---------------------------------------------------------------------------

/// Poisson-subsampled Gaussian mechanism.
///
/// Each record is included independently with probability *sample_rate*,
/// providing privacy amplification by subsampling.
///
/// Args:
///     noise_multiplier (float): Gaussian sigma / sensitivity.
///     sample_rate (float): Poisson sampling probability q in (0, 1].
///     config (DiscretizationConfig, optional): Custom discretization.
///
/// Returns:
///     DpProcess
///
/// Example::
///
///     step = dp.poisson(1.1, 0.01)
///     training = step * 1000
///     training.epsilon_at(1e-5)
#[pyfunction]
#[pyo3(name = "poisson", signature = (noise_multiplier, sample_rate, config=None))]
fn py_poisson(
    noise_multiplier: f64,
    sample_rate: f64,
    config: Option<PyDiscretizationConfig>,
) -> PyResult<PyDpProcess> {
    let g = make_gaussian(noise_multiplier, config.as_ref()).map_err(to_py_err)?;
    let p = poisson(g, sample_rate);
    Ok(PyDpProcess::new(
        Box::new(p),
        format!(
            "Poisson(noise_multiplier={}, sample_rate={})",
            noise_multiplier, sample_rate
        ),
        vec![
            ("noise_multiplier", ParamValue::Float(noise_multiplier)),
            ("sample_rate", ParamValue::Float(sample_rate)),
        ],
    ))
}

/// Truncated-Poisson-subsampled Gaussian (production DP-SGD).
///
/// Like Poisson sampling but caps the batch at *batch_size_cap*.
/// This matches what Opacus / JAX Privacy / TF Privacy actually do.
///
/// Args:
///     noise_multiplier (float): Gaussian sigma / sensitivity.
///     sample_rate (float): Expected sampling rate q.
///     batch_size_cap (int): Maximum batch size B_max.
///     dataset_size (int): Total dataset size n.
///     config (DiscretizationConfig, optional): Custom discretization.
///
/// Returns:
///     DpProcess
#[pyfunction]
#[pyo3(name = "truncated_poisson", signature = (noise_multiplier, sample_rate, batch_size_cap, dataset_size, config=None))]
fn py_truncated_poisson(
    noise_multiplier: f64,
    sample_rate: f64,
    batch_size_cap: usize,
    dataset_size: usize,
    config: Option<PyDiscretizationConfig>,
) -> PyResult<PyDpProcess> {
    let g = make_gaussian(noise_multiplier, config.as_ref()).map_err(to_py_err)?;
    let tp = truncated_poisson(g, sample_rate, batch_size_cap, dataset_size);
    Ok(PyDpProcess::new(
        Box::new(tp),
        format!(
            "TruncatedPoisson(noise_multiplier={}, sample_rate={}, batch_size_cap={}, dataset_size={})",
            noise_multiplier, sample_rate, batch_size_cap, dataset_size
        ),
        vec![
            ("noise_multiplier", ParamValue::Float(noise_multiplier)),
            ("sample_rate", ParamValue::Float(sample_rate)),
            ("batch_size_cap", ParamValue::Int(batch_size_cap)),
            ("dataset_size", ParamValue::Int(dataset_size)),
        ],
    ))
}

/// Gradient-accumulated Poisson-subsampled Gaussian.
///
/// Models *microbatches* micro-batches accumulated before a single noise
/// addition (Mixture-of-Gaussians framework).
///
/// Args:
///     noise_multiplier (float): Gaussian sigma / sensitivity.
///     sample_rate (float): Per-microbatch Poisson rate.
///     microbatches (int): Number of micro-batches m.
///     config (DiscretizationConfig, optional): Custom discretization.
///
/// Returns:
///     DpProcess
#[pyfunction]
#[pyo3(name = "accumulate", signature = (noise_multiplier, sample_rate, microbatches, config=None))]
fn py_accumulate(
    noise_multiplier: f64,
    sample_rate: f64,
    microbatches: usize,
    config: Option<PyDiscretizationConfig>,
) -> PyResult<PyDpProcess> {
    let g = make_gaussian(noise_multiplier, config.as_ref()).map_err(to_py_err)?;
    let p = poisson(g, sample_rate);
    let acc = accumulate(p, microbatches).map_err(to_py_err)?;
    Ok(PyDpProcess::new(
        Box::new(acc),
        format!(
            "Accumulate(noise_multiplier={}, sample_rate={}, microbatches={})",
            noise_multiplier, sample_rate, microbatches
        ),
        vec![
            ("noise_multiplier", ParamValue::Float(noise_multiplier)),
            ("sample_rate", ParamValue::Float(sample_rate)),
            ("microbatches", ParamValue::Int(microbatches)),
        ],
    ))
}

// ---------------------------------------------------------------------------
// Module-level functions — transforms
// ---------------------------------------------------------------------------

/// Gaussian mechanism with adaptive clipping (Andrew et al. 2021).
///
/// Args:
///     noise_multiplier (float): Gradient noise multiplier.
///     quantile_noise_std (float): Noise std for quantile estimation.
///     config (DiscretizationConfig, optional): Custom discretization.
///
/// Returns:
///     DpProcess
#[pyfunction]
#[pyo3(name = "adaclip", signature = (noise_multiplier, quantile_noise_std, config=None))]
fn py_adaclip(
    noise_multiplier: f64,
    quantile_noise_std: f64,
    config: Option<PyDiscretizationConfig>,
) -> PyResult<PyDpProcess> {
    let g = make_gaussian(noise_multiplier, config.as_ref()).map_err(to_py_err)?;
    let ac = adaclip(g, quantile_noise_std);
    Ok(PyDpProcess::new(
        Box::new(ac),
        format!(
            "AdaClip(noise_multiplier={}, quantile_noise_std={})",
            noise_multiplier, quantile_noise_std
        ),
        vec![
            ("noise_multiplier", ParamValue::Float(noise_multiplier)),
            ("quantile_noise_std", ParamValue::Float(quantile_noise_std)),
        ],
    ))
}

/// Poisson-subsampled AdaClip Gaussian.
///
/// Convenience wrapper combining :func:`adaclip` and :func:`poisson`.
///
/// Args:
///     noise_multiplier (float): Gradient noise multiplier.
///     quantile_noise_std (float): AdaClip quantile noise std.
///     sample_rate (float): Poisson sampling rate.
///     config (DiscretizationConfig, optional): Custom discretization.
///
/// Returns:
///     DpProcess
#[pyfunction]
#[pyo3(name = "poisson_adaclip", signature = (noise_multiplier, quantile_noise_std, sample_rate, config=None))]
fn py_poisson_adaclip(
    noise_multiplier: f64,
    quantile_noise_std: f64,
    sample_rate: f64,
    config: Option<PyDiscretizationConfig>,
) -> PyResult<PyDpProcess> {
    let g = make_gaussian(noise_multiplier, config.as_ref()).map_err(to_py_err)?;
    let ac = adaclip(g, quantile_noise_std);
    let p = poisson(ac, sample_rate);
    Ok(PyDpProcess::new(
        Box::new(p),
        format!(
            "PoissonAdaClip(noise_multiplier={}, quantile_noise_std={}, sample_rate={})",
            noise_multiplier, quantile_noise_std, sample_rate
        ),
        vec![
            ("noise_multiplier", ParamValue::Float(noise_multiplier)),
            ("quantile_noise_std", ParamValue::Float(quantile_noise_std)),
            ("sample_rate", ParamValue::Float(sample_rate)),
        ],
    ))
}

// ---------------------------------------------------------------------------
// Module-level functions — composition
// ---------------------------------------------------------------------------

/// Homogeneous k-fold composition (repeat a process *count* times).
///
/// Equivalent to ``process * count``.
///
/// Args:
///     process (DpProcess): The process to repeat.
///     count (int): Number of repetitions.
///
/// Returns:
///     DpProcess
#[pyfunction]
#[pyo3(name = "repeat", text_signature = "(process, count)")]
fn py_repeat(process: &PyDpProcess, count: usize) -> PyResult<PyDpProcess> {
    process.__mul__(count)
}

/// Heterogeneous composition of two processes.
///
/// Equivalent to ``left | right``.
///
/// Args:
///     left (DpProcess): First process.
///     right (DpProcess): Second process.
///
/// Returns:
///     DpProcess
#[pyfunction]
#[pyo3(name = "compose", text_signature = "(left, right)")]
fn py_compose(left: &PyDpProcess, right: &PyDpProcess) -> PyResult<PyDpProcess> {
    left.__or__(right)
}

// ---------------------------------------------------------------------------
// Module-level functions — convenience / calibration
// ---------------------------------------------------------------------------

/// Compute epsilon for a standard DP-SGD run.
///
/// Shorthand for ``(poisson(noise_multiplier, sample_rate) * num_steps).epsilon_at(delta)``.
///
/// Args:
///     noise_multiplier (float): Gaussian sigma / sensitivity.
///     sample_rate (float): batch_size / dataset_size.
///     num_steps (int): Number of training steps.
///     delta (float): Target delta.
///
/// Returns:
///     float: The epsilon value.
///
/// Example::
///
///     eps = dp.compute_epsilon(1.1, 0.01, 1000, delta=1e-5)
#[pyfunction]
#[pyo3(name = "compute_epsilon", signature = (noise_multiplier, sample_rate, num_steps, delta))]
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

/// Calibrate the noise multiplier for a target epsilon.
///
/// Uses bisection search over the noise multiplier.
///
/// Args:
///     target_epsilon (float): Desired epsilon.
///     target_delta (float): Delta for the (epsilon, delta) guarantee.
///     sample_rate (float): Poisson sampling rate.
///     num_steps (int): Number of composition steps.
///     param_min (float): Lower bound on noise multiplier (default 0.1).
///     param_max (float): Upper bound on noise multiplier (default 1.2).
///     tolerance (float): Bisection tolerance (default 1e-6).
///     max_iterations (int): Maximum iterations (default 100).
///
/// Returns:
///     float: Calibrated noise multiplier.
#[pyfunction]
#[pyo3(name = "calibrate_noise", signature = (target_epsilon, target_delta, sample_rate, num_steps, param_min=0.1, param_max=1.2, tolerance=1e-6, max_iterations=100))]
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
            lo = mid;
        } else {
            hi = mid;
        }
    }

    Ok((lo + hi) / 2.0)
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

    // Convenience
    m.add_function(wrap_pyfunction!(py_compute_epsilon, m)?)?;
    m.add_function(wrap_pyfunction!(py_calibrate_noise, m)?)?;

    Ok(())
}
