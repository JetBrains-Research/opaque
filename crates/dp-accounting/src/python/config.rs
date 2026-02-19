//! PyDiscretizationConfig: discretization configuration for PLD computation.

use pyo3::prelude::*;

use crate::discretization::DiscretizationConfig;
use crate::error::PldError;

fn to_py_err(e: PldError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

/// Discretization configuration for PLD computation.
///
/// Args:
///     discretization (float): Grid spacing (default 1e-4). Smaller = tighter.
///     log_mass_truncation_bound (float): Log tail mass (default -50).
///
/// Example::
///
///     config = dp.DiscretizationConfig(discretization=0.001)
///     pld = dp.gaussian_pld(1.1, config=config)
#[pyclass(name = "DiscretizationConfig")]
#[derive(Clone)]
pub struct PyDiscretizationConfig {
    pub(super) inner: DiscretizationConfig,
}

impl PyDiscretizationConfig {
    pub(super) fn resolve(config: Option<&PyDiscretizationConfig>) -> DiscretizationConfig {
        match config {
            Some(c) => c.inner.clone(),
            None => DiscretizationConfig::default(),
        }
    }
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
        let inner = DiscretizationConfig::with_estimate(
            discretization,
            log_mass_truncation_bound,
            pessimistic_estimate,
        )
        .map_err(to_py_err)?
        .with_max_grid_size(max_grid_size);
        Ok(Self { inner })
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
            self.inner.discretization, self.inner.log_mass_truncation_bound,
            self.inner.pessimistic_estimate, self.inner.max_grid_size,
        )
    }

    fn __eq__(&self, other: &PyDiscretizationConfig) -> bool {
        self.inner.discretization == other.inner.discretization
            && self.inner.log_mass_truncation_bound == other.inner.log_mass_truncation_bound
            && self.inner.pessimistic_estimate == other.inner.pessimistic_estimate
            && self.inner.max_grid_size == other.inner.max_grid_size
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.inner.discretization.to_bits().hash(&mut hasher);
        self.inner.log_mass_truncation_bound.to_bits().hash(&mut hasher);
        self.inner.pessimistic_estimate.hash(&mut hasher);
        self.inner.max_grid_size.hash(&mut hasher);
        hasher.finish()
    }
}
