//! PyO3 Python bindings for the functional DP accounting API.
//!
//! This module exposes the Rust-based PLD accounting engine to Python as
//! `opaque_dp_accounting`. The API is designed to feel native to Python:
//!
//! - **Natural names**: `gaussian`, `poisson`, `compose` (no `py_` prefix)
//! - **Operator overloads**: `step * 1000` for repetition, `a | b` for composition
//! - **Rich introspection**: `describe()`, `pld_info()`, `summary()` for debugging
//! - **Type erasure**: All mechanism types are presented as a single `DpProcess` class
//!
//! # Design
//!
//! Rust mechanisms are heterogeneous types (`Gaussian`, `Poisson<Gaussian>`, etc.).
//! Python sees a single `DpProcess` class via the [`DynProcess`] trait-object pattern:
//! each mechanism is boxed behind `Box<dyn DynProcess>`, and [`ProcessWrapper`] bridges
//! this back to the `Process` trait so Rust composition still works.

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
/// ``DpProcess`` is the central class in ``opaque_dp_accounting``.  Every
/// mechanism constructor (``gaussian``, ``poisson``, ``adaclip``, ...) returns
/// a ``DpProcess``, and composition operators produce new ``DpProcess`` instances.
///
/// **Constructors** (module-level functions):
///
/// - :func:`gaussian` -- Gaussian mechanism
/// - :func:`poisson` -- Poisson-subsampled Gaussian
/// - :func:`truncated_poisson` -- production DP-SGD with capped batch size
/// - :func:`accumulate` -- gradient accumulation (microbatching)
/// - :func:`eps_delta` -- fixed (epsilon, delta) guarantee
/// - :func:`identity` -- zero privacy loss
/// - :func:`adaclip` -- adaptive clipping (Andrew et al. 2021)
/// - :func:`poisson_adaclip` -- Poisson + AdaClip combined
///
/// **Composition**:
///
/// - ``step * 1000`` or ``dp.repeat(step, 1000)`` -- homogeneous k-fold
/// - ``a | b`` or ``dp.compose(a, b)`` -- heterogeneous two-process
///
/// **Privacy metrics** (all derived from the same PLD):
///
/// - :meth:`epsilon_at` -- (epsilon, delta)-DP
/// - :meth:`delta_at` -- (epsilon, delta)-DP (inverse direction)
/// - :meth:`advantage` -- f-DP total-variation advantage
/// - :meth:`beta_at` -- Type-II error at given Type-I error
/// - :meth:`risk_at` -- Bayes risk
///
/// **Debugging**:
///
/// - ``print(proc)`` -- one-line summary with epsilon
/// - :meth:`describe` -- constructor parameters as dict
/// - :meth:`pld_info` -- PLD grid diagnostics with timing
/// - :meth:`summary` -- formatted multi-line privacy report
///
/// Example::
///
///     import opaque_dp_accounting as dp
///
///     step = dp.poisson(1.1, 0.01)
///     training = step * 1000
///     print(training.epsilon_at(1e-5))    # ~3.73
///     print(training.summary())            # full report
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

    /// Compute the smallest epsilon such that the mechanism satisfies (epsilon, delta)-DP.
    ///
    /// This solves: find min epsilon s.t. ``P[M(D) in S] <= exp(epsilon) * P[M(D') in S] + delta``
    /// for all neighboring datasets D, D' and all output sets S.
    ///
    /// Args:
    ///     delta (float): Failure probability (typically 1e-5 to 1e-7).
    ///         Must be in [0, 1). Smaller delta = stricter guarantee.
    ///
    /// Returns:
    ///     float: The smallest epsilon achieving (epsilon, delta)-DP.
    ///
    /// Example::
    ///
    ///     proc = dp.gaussian(1.1)
    ///     eps = proc.epsilon_at(1e-5)  # ~3.73
    #[pyo3(text_signature = "(self, delta)")]
    fn epsilon_at(&self, delta: f64) -> PyResult<f64> {
        let pld = self.inner.compute_pld().map_err(to_py_err)?;
        Ok(pld.epsilon_at(delta))
    }

    /// Compute the smallest delta such that the mechanism satisfies (epsilon, delta)-DP.
    ///
    /// This is the inverse of :meth:`epsilon_at`: given an epsilon budget,
    /// find the failure probability delta.
    ///
    /// Args:
    ///     epsilon (float): Privacy budget. Must be >= 0.
    ///         epsilon=0 gives delta = advantage (worst-case distinguishing probability).
    ///
    /// Returns:
    ///     float: The delta value at the given epsilon.
    ///
    /// Example::
    ///
    ///     proc = dp.gaussian(1.1)
    ///     delta = proc.delta_at(1.0)  # delta when epsilon=1
    #[pyo3(text_signature = "(self, epsilon)")]
    fn delta_at(&self, epsilon: f64) -> PyResult<f64> {
        let pld = self.inner.compute_pld().map_err(to_py_err)?;
        Ok(pld.delta_at(epsilon))
    }

    /// Total-variation advantage: max probability of distinguishing neighboring datasets.
    ///
    /// Equivalent to ``delta_at(0.0)`` — the hockey-stick divergence at epsilon=0.
    /// This is the f-DP advantage metric from Dong et al. (2019).
    ///
    /// Returns:
    ///     float: Advantage in [0, 1]. Lower = more private.
    ///
    /// Example::
    ///
    ///     proc = dp.gaussian(1.0)
    ///     adv = proc.advantage()  # ~0.31
    fn advantage(&self) -> PyResult<f64> {
        let pld = self.inner.compute_pld().map_err(to_py_err)?;
        Ok(pld.advantage())
    }

    /// Type-II error (beta) at a given Type-I error (alpha).
    ///
    /// In the hypothesis testing interpretation of DP, an adversary tries
    /// to distinguish D from D'. Alpha is the false-positive rate and beta
    /// is the false-negative rate. Higher beta = harder to detect = more private.
    ///
    /// Args:
    ///     alpha (float): Type-I error rate (false positive). Must be in [0, 1].
    ///
    /// Returns:
    ///     float: Type-II error rate (beta) in [0, 1].
    ///
    /// Example::
    ///
    ///     proc = dp.gaussian(1.0)
    ///     beta = proc.beta_at(0.05)  # Type-II error at alpha=0.05
    #[pyo3(text_signature = "(self, alpha)")]
    fn beta_at(&self, alpha: f64) -> PyResult<f64> {
        let pld = self.inner.compute_pld().map_err(to_py_err)?;
        Ok(pld.beta_at(alpha))
    }

    /// Bayes risk under an optimal adversary with a given prior.
    ///
    /// The risk is the minimum expected loss of any decision rule trying to
    /// distinguish D from D', weighted by the prior probability.
    /// ``risk = prior * beta + (1 - prior) * alpha`` at the optimal threshold.
    ///
    /// Args:
    ///     prior (float): Prior probability that the data came from D (vs D').
    ///         Typically 0.5 for a balanced prior.
    ///
    /// Returns:
    ///     float: Bayes risk in [0, 0.5]. Higher = more private.
    ///
    /// Example::
    ///
    ///     proc = dp.gaussian(1.0)
    ///     risk = proc.risk_at(0.5)  # risk under uniform prior
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
/// The PLD is represented as a discrete probability mass function (PMF)
/// on a regular grid.  These parameters control the grid resolution,
/// tail truncation, and rounding direction.
///
/// **Defaults are chosen for high accuracy** (discretization=1e-4 gives
/// ~1e-8 error per composition step).  Coarser grids are faster but
/// less precise; finer grids are more precise but use more memory.
///
/// Args:
///     discretization (float): Grid spacing for the PLD PMF (default 1e-4).
///         Smaller = more precise, larger grid.  Error scales as O(disc^2).
///     log_mass_truncation_bound (float): Tails with probability below
///         exp(bound) are truncated (default -50, matching Google dp_accounting).
///     pessimistic_estimate (bool): If True (default), round probabilities
///         to produce an **upper bound** on privacy loss.  Set to False for
///         an optimistic (lower-bound) estimate -- useful for debugging
///         but not safe for privacy guarantees.
///     max_grid_size (int): If the grid exceeds this many bins, the
///         discretization is automatically coarsened (default 10,000,000).
///
/// Example::
///
///     # Faster but less precise
///     cfg = dp.DiscretizationConfig(discretization=1e-3)
///
///     # Maximum precision
///     cfg = dp.DiscretizationConfig(
///         discretization=1e-5,
///         log_mass_truncation_bound=-50.0,
///     )
///
///     # Use with any mechanism
///     proc = dp.gaussian(1.1, config=cfg)
#[pyclass(name = "DiscretizationConfig")]
#[derive(Clone)]
struct PyDiscretizationConfig {
    inner: DiscretizationConfig,
}

#[pymethods]
impl PyDiscretizationConfig {
    #[new]
    #[pyo3(signature = (discretization=1e-4, log_mass_truncation_bound=-50.0, pessimistic_estimate=true, max_grid_size=10_000_000))]
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

/// Create a Gaussian mechanism with sensitivity 1.
///
/// The Gaussian mechanism adds N(0, noise_multiplier^2) noise to a
/// function with L2 sensitivity 1.  This is the building block for
/// DP-SGD: after clipping gradients to norm C, the effective noise
/// standard deviation is ``noise_multiplier * C``.
///
/// Args:
///     noise_multiplier (float): Ratio of noise std to sensitivity (sigma / Delta).
///         Typical range for DP-SGD is [0.5, 2.0].  Higher = more private.
///     config (DiscretizationConfig, optional): Override default PLD precision.
///
/// Returns:
///     DpProcess: A process representing a single Gaussian mechanism application.
///
/// Raises:
///     ValueError: If noise_multiplier is out of the supported range.
///
/// Example::
///
///     proc = dp.gaussian(1.1)
///     proc.epsilon_at(1e-5)  # ~3.73
///
///     # With custom precision
///     cfg = dp.DiscretizationConfig(discretization=1e-3)
///     proc = dp.gaussian(1.1, config=cfg)
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

/// Create a mechanism with a fixed (epsilon, delta)-DP guarantee.
///
/// This represents a mechanism whose privacy loss is known analytically
/// (e.g., randomized response, Laplace mechanism).  The resulting PLD is
/// a two-point distribution capturing the worst-case privacy loss.
///
/// Useful for composing non-Gaussian mechanisms with Gaussian ones.
///
/// Args:
///     epsilon (float): Privacy parameter (must be >= 0).
///     delta (float): Failure probability (default 0, must be in [0, 1)).
///     config (DiscretizationConfig, optional): Override default PLD precision.
///
/// Returns:
///     DpProcess: A process with the given (epsilon, delta) guarantee.
///
/// Example::
///
///     # Pure epsilon-DP mechanism
///     proc = dp.eps_delta(1.0)
///
///     # Approximate DP
///     proc = dp.eps_delta(1.0, delta=1e-5)
///
///     # Compose with a Gaussian
///     combined = dp.gaussian(1.1) | dp.eps_delta(0.5)
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

/// Create an identity mechanism with zero privacy loss.
///
/// The identity process represents a computation that reveals no
/// information about the dataset (e.g., returning a constant).
/// Its PLD is a point mass at privacy loss = 0.  Composing with
/// identity has no effect: ``proc | dp.identity() == proc``.
///
/// Args:
///     config (DiscretizationConfig, optional): Override default PLD precision.
///
/// Returns:
///     DpProcess: A process with epsilon=0, delta=0 for all queries.
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

/// Poisson-subsampled Gaussian mechanism (standard DP-SGD step).
///
/// Each record is included independently with probability ``sample_rate``,
/// providing **privacy amplification by subsampling**.  This is the standard
/// model for DP-SGD: ``sample_rate = batch_size / dataset_size``.
///
/// The privacy amplification can be substantial.  For example, with
/// ``sample_rate=0.01`` the effective epsilon can be 50-100x smaller than
/// the un-subsampled Gaussian.
///
/// Args:
///     noise_multiplier (float): Gaussian noise std / sensitivity.
///     sample_rate (float): Poisson sampling probability q = batch_size / dataset_size.
///         Must be in (0, 1].
///     config (DiscretizationConfig, optional): Override default PLD precision.
///
/// Returns:
///     DpProcess: A single Poisson-subsampled Gaussian step.
///
/// Example::
///
///     # Standard DP-SGD: 1000 steps
///     step = dp.poisson(1.1, 0.01)
///     training = step * 1000
///     eps = training.epsilon_at(1e-5)
///
///     # One-liner equivalent
///     eps = dp.compute_epsilon(1.1, 0.01, 1000, 1e-5)
///
/// See Also:
///     :func:`truncated_poisson` for capped batch sizes (tighter bounds).
///     :func:`compute_epsilon` for a one-liner.
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
/// Like ``poisson()`` but caps the batch at ``batch_size_cap``.  This models
/// what production DP-SGD frameworks (Opacus, JAX Privacy, TF Privacy)
/// actually do: sample a random batch, but truncate if it exceeds a maximum.
///
/// The truncated Poisson analysis provides **tighter privacy bounds** than
/// the standard (worst-case) Poisson analysis -- up to 20% improvement in
/// epsilon for the same noise level.
///
/// Args:
///     noise_multiplier (float): Gaussian noise std / sensitivity.
///     sample_rate (float): Expected sampling rate q = batch_size / dataset_size.
///     batch_size_cap (int): Maximum batch size B_max.
///     dataset_size (int): Total dataset size n.
///     config (DiscretizationConfig, optional): Override default PLD precision.
///
/// Returns:
///     DpProcess: A single truncated-Poisson step.
///
/// Example::
///
///     step = dp.truncated_poisson(1.1, 0.01, batch_size_cap=100, dataset_size=10000)
///     training = step * 1000
///     eps = training.epsilon_at(1e-5)
///
/// See Also:
///     :func:`poisson` for standard (non-truncated) analysis.
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
/// Models ``microbatches`` micro-batches accumulated before a single noise
/// addition step.  This uses the Mixture-of-Gaussians framework to account
/// for the fact that gradient accumulation processes multiple micro-batches
/// per noise injection, which is common when GPU memory is limited.
///
/// The privacy analysis is exact: it computes a mixture PLD over the
/// possible numbers of records contributed by the micro-batches.
///
/// Args:
///     noise_multiplier (float): Gaussian noise std / sensitivity.
///     sample_rate (float): Per-microbatch Poisson sampling rate.
///     microbatches (int): Number of micro-batches m accumulated per step.
///     config (DiscretizationConfig, optional): Override default PLD precision.
///
/// Returns:
///     DpProcess: A single accumulated step.
///
/// Example::
///
///     # 4 microbatches accumulated per noise step
///     step = dp.accumulate(1.1, sample_rate=0.01, microbatches=4)
///     training = step * 500
///     eps = training.epsilon_at(1e-5)
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
/// Adaptive clipping adjusts the clipping threshold based on the
/// empirical distribution of gradient norms.  The quantile estimation
/// itself uses a noisy mechanism, adding extra privacy cost.
///
/// The total privacy cost is the composition of the base Gaussian
/// mechanism and the quantile-estimation mechanism (with noise std
/// ``quantile_noise_std``).
///
/// Args:
///     noise_multiplier (float): Gradient noise multiplier for the main mechanism.
///     quantile_noise_std (float): Noise std for the quantile estimation.
///         Larger values = more private quantile estimation, less accurate clipping.
///     config (DiscretizationConfig, optional): Override default PLD precision.
///
/// Returns:
///     DpProcess: A single AdaClip step.
///
/// Example::
///
///     step = dp.adaclip(1.1, quantile_noise_std=50.0)
///     eps = step.epsilon_at(1e-5)
///
/// See Also:
///     :func:`poisson_adaclip` for the subsampled variant.
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
/// Convenience wrapper combining adaptive clipping with Poisson subsampling.
/// Equivalent to ``dp.poisson(dp.adaclip(nm, sigma_b), sample_rate)`` but
/// constructed in a single call.
///
/// Args:
///     noise_multiplier (float): Gradient noise multiplier.
///     quantile_noise_std (float): AdaClip quantile noise std.
///     sample_rate (float): Poisson sampling rate q = batch_size / dataset_size.
///     config (DiscretizationConfig, optional): Override default PLD precision.
///
/// Returns:
///     DpProcess: A single Poisson-subsampled AdaClip step.
///
/// Example::
///
///     step = dp.poisson_adaclip(1.1, quantile_noise_std=50.0, sample_rate=0.01)
///     training = step * 1000
///     eps = training.epsilon_at(1e-5)
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

/// Compute epsilon for a standard DP-SGD training run (one-liner).
///
/// This is a convenience function equivalent to::
///
///     (dp.poisson(noise_multiplier, sample_rate) * num_steps).epsilon_at(delta)
///
/// Use this for quick privacy analysis without constructing intermediate
/// process objects.  For more control (custom config, different mechanisms),
/// build the process explicitly.
///
/// Args:
///     noise_multiplier (float): Gaussian noise std / sensitivity.
///     sample_rate (float): batch_size / dataset_size (Poisson sampling rate).
///     num_steps (int): Number of DP-SGD training steps.
///     delta (float): Target delta for (epsilon, delta)-DP.
///
/// Returns:
///     float: The epsilon value for the full training run.
///
/// Raises:
///     ValueError: If parameters are out of range.
///
/// Example::
///
///     # Quick privacy check for a training run
///     eps = dp.compute_epsilon(1.1, 0.01, 1000, delta=1e-5)
///     print(f"Training gives epsilon={eps:.2f}")  # ~3.73
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

/// Calibrate the noise multiplier to achieve a target (epsilon, delta) guarantee.
///
/// Uses bisection search: higher noise_multiplier = lower epsilon.  The search
/// finds the **smallest** noise_multiplier such that the composed epsilon is
/// at most ``target_epsilon``.
///
/// This is the inverse of :func:`compute_epsilon`: given a privacy target,
/// find the noise level needed to achieve it.
///
/// Args:
///     target_epsilon (float): Desired maximum epsilon.
///     target_delta (float): Delta for the (epsilon, delta) guarantee.
///     sample_rate (float): Poisson sampling rate q = batch_size / dataset_size.
///     num_steps (int): Number of DP-SGD training steps.
///     param_min (float): Lower bound on noise multiplier search (default 0.1).
///     param_max (float): Upper bound on noise multiplier search (default 1.2).
///     tolerance (float): Bisection convergence tolerance (default 1e-6).
///     max_iterations (int): Maximum bisection iterations (default 100).
///
/// Returns:
///     float: Calibrated noise multiplier.
///
/// Example::
///
///     nm = dp.calibrate_noise(
///         target_epsilon=8.0,
///         target_delta=1e-5,
///         sample_rate=0.01,
///         num_steps=1000,
///     )
///     # Verify
///     actual_eps = dp.compute_epsilon(nm, 0.01, 1000, 1e-5)
///     assert abs(actual_eps - 8.0) < 0.1
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
