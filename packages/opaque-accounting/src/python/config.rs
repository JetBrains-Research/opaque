//! PyDiscretizationConfig: discretization configuration for PLD computation.

use pyo3::prelude::*;

use crate::discretization::DiscretizationConfig;

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

#[pymethods]
impl PyDiscretizationConfig {
    #[new]
    #[pyo3(signature = (discretization=1e-4, log_mass_truncation_bound=-50.0, max_grid_size=10_000_000, tail_mass_truncation=1e-15, max_conv_grid=32_768, num_mc_samples=100_000, seed=42))]
    fn new(
        discretization: f64,
        log_mass_truncation_bound: f64,
        max_grid_size: usize,
        tail_mass_truncation: f64,
        max_conv_grid: usize,
        num_mc_samples: usize,
        seed: u64,
    ) -> PyResult<Self> {
        let mut inner = DiscretizationConfig::new(discretization, log_mass_truncation_bound)?
            .with_max_grid_size(max_grid_size)
            .with_max_conv_grid(max_conv_grid);
        inner.tail_mass_truncation = tail_mass_truncation;
        inner.num_mc_samples = num_mc_samples;
        inner.seed = seed;
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
    fn max_grid_size(&self) -> usize {
        self.inner.max_grid_size
    }

    #[getter]
    fn max_conv_grid(&self) -> usize {
        self.inner.max_conv_grid
    }

    #[getter]
    fn tail_mass_truncation(&self) -> f64 {
        self.inner.tail_mass_truncation
    }

    #[getter]
    fn num_mc_samples(&self) -> usize {
        self.inner.num_mc_samples
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }

    fn __repr__(&self) -> String {
        format!(
            "DiscretizationConfig(discretization={}, log_mass_truncation_bound={}, max_grid_size={}, tail_mass_truncation={}, max_conv_grid={}, num_mc_samples={}, seed={})",
            self.inner.discretization, self.inner.log_mass_truncation_bound,
            self.inner.max_grid_size,
            self.inner.tail_mass_truncation,
            self.inner.max_conv_grid,
            self.inner.num_mc_samples, self.inner.seed,
        )
    }

    fn __eq__(&self, other: &PyDiscretizationConfig) -> bool {
        self.inner.discretization == other.inner.discretization
            && self.inner.log_mass_truncation_bound == other.inner.log_mass_truncation_bound
            && self.inner.max_grid_size == other.inner.max_grid_size
            && self.inner.tail_mass_truncation == other.inner.tail_mass_truncation
            && self.inner.max_conv_grid == other.inner.max_conv_grid
            && self.inner.num_mc_samples == other.inner.num_mc_samples
            && self.inner.seed == other.inner.seed
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.inner.discretization.to_bits().hash(&mut hasher);
        self.inner
            .log_mass_truncation_bound
            .to_bits()
            .hash(&mut hasher);
        self.inner.max_grid_size.hash(&mut hasher);
        self.inner.tail_mass_truncation.to_bits().hash(&mut hasher);
        self.inner.max_conv_grid.hash(&mut hasher);
        self.inner.num_mc_samples.hash(&mut hasher);
        self.inner.seed.hash(&mut hasher);
        hasher.finish()
    }
}
