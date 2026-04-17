//! PyPld: opaque handle wrapping PrivacyLossDistribution.

use pyo3::prelude::*;

use crate::error::PldError;
use crate::pld::PrivacyLossDistribution;

use super::config::PyDiscretizationConfig;

fn to_py_err(e: PldError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

/// An opaque Privacy Loss Distribution (PLD).
///
/// Created by mechanism/amplification functions. Supports privacy metric
/// queries, composition, and self-composition (repetition).
///
/// Example::
///
///     import opaque_accounting as dp
///     pld = dp.gaussian_pld(1.1)
///     print(pld.epsilon_at(1e-5))
///     composed = pld.self_compose(1000)
///     print(composed.epsilon_at(1e-5))
#[pyclass(name = "Pld")]
#[derive(Clone)]
pub struct PyPld {
    pub(super) inner: PrivacyLossDistribution,
}

impl PyPld {
    pub(super) fn new(inner: PrivacyLossDistribution) -> Self {
        Self { inner }
    }

    pub(super) fn inner(&self) -> &PrivacyLossDistribution {
        &self.inner
    }
}

#[pymethods]
impl PyPld {
    /// Smallest epsilon achieving (epsilon, delta)-DP.
    ///
    /// Args:
    ///     delta (float): Failure probability (typically 1e-5 to 1e-7).
    ///
    /// Returns:
    ///     float: Epsilon value.
    #[pyo3(text_signature = "(self, delta)")]
    fn epsilon_at(&self, delta: f64) -> f64 {
        self.inner.epsilon_at(delta)
    }

    /// Smallest delta achieving (epsilon, delta)-DP.
    ///
    /// Args:
    ///     epsilon (float): Privacy budget. Must be >= 0.
    ///
    /// Returns:
    ///     float: Delta value.
    #[pyo3(text_signature = "(self, epsilon)")]
    fn delta_at(&self, epsilon: f64) -> f64 {
        self.inner.delta_at(epsilon)
    }

    /// Total-variation advantage.
    ///
    /// Returns:
    ///     float: Advantage in [0, 1]. Lower is more private.
    fn advantage(&self) -> f64 {
        self.inner.advantage()
    }

    /// Type-II error (beta) at a given Type-I error (alpha).
    ///
    /// Args:
    ///     alpha (float): Type-I error rate in [0, 1].
    ///
    /// Returns:
    ///     float: Beta value in [0, 1].
    #[pyo3(text_signature = "(self, alpha)")]
    fn beta_at(&self, alpha: f64) -> f64 {
        self.inner.beta_at(alpha)
    }

    /// Bayes risk under an optimal adversary.
    ///
    /// Args:
    ///     prior (float): Prior probability (typically 0.5).
    ///
    /// Returns:
    ///     float: Risk value in [0, 0.5].
    #[pyo3(text_signature = "(self, prior)")]
    fn risk_at(&self, prior: f64) -> f64 {
        self.inner.risk_at(prior)
    }

    /// Compose this PLD with another PLD.
    ///
    /// Returns a new PLD representing the composition (running both
    /// mechanisms sequentially).
    ///
    /// Args:
    ///     other (Pld): The other PLD to compose with.
    ///
    /// Returns:
    ///     Pld: Composed PLD.
    fn compose(&self, other: &PyPld) -> PyResult<PyPld> {
        let pld = self.inner.compose(&other.inner).map_err(to_py_err)?;
        Ok(PyPld { inner: pld })
    }

    /// Compose this PLD with itself `count` times (homogeneous repetition).
    ///
    /// Equivalent to running the same mechanism `count` times.
    ///
    /// Args:
    ///     count (int): Repetition count. Must be > 0.
    ///
    /// Returns:
    ///     Pld: Self-composed PLD.
    #[pyo3(text_signature = "(self, count)")]
    fn self_compose(&self, count: usize) -> PyPld {
        PyPld {
            inner: self.inner.self_compose(count),
        }
    }

    /// ``pld * k`` is shorthand for ``pld.self_compose(k)``.
    fn __mul__(&self, count: usize) -> PyPld {
        self.self_compose(count)
    }

    /// ``k * pld`` also works.
    fn __rmul__(&self, count: usize) -> PyPld {
        self.self_compose(count)
    }

    /// ``a | b`` is shorthand for ``a.compose(b)``.
    fn __or__(&self, other: &PyPld) -> PyResult<PyPld> {
        self.compose(other)
    }

    fn __repr__(&self) -> String {
        let grid = self.inner.pmf_remove.probs.len();
        let sym = if self.inner.pmf_add.is_none() {
            "symmetric"
        } else {
            "asymmetric"
        };
        format!("Pld({}, {} bins)", sym, grid)
    }

    fn __str__(&self) -> String {
        let eps = self.inner.epsilon_at(1e-5);
        format!("Pld(eps(1e-5)={:.6})", eps)
    }
}
