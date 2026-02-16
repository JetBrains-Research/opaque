//! # opaque-dp-accounting
//!
//! Privacy Loss Distribution (PLD) accounting for differential privacy.
//!
//! This crate provides privacy accounting using the Privacy Loss Distribution
//! framework with Connect-the-Dots discretization. It is a functional,
//! composable API for building and analyzing differential privacy mechanisms.
//!
//! ## Quick Start (Functional API)
//!
//! ```rust
//! use opaque_dp_accounting::functional::*;
//!
//! // Standard DP-SGD: Poisson-subsampled Gaussian, 1000 rounds
//! let step = poisson(gaussian(1.1).unwrap(), 0.01);
//! let training = repeat(step, 1000).unwrap();
//! let epsilon = training.epsilon_at(1e-5).unwrap();
//! ```
//!
//! ## Module Overview
//!
//! - [`functional`]: Composable functional API for privacy accounting
//! - [`error`]: Error types
//! - [`math_helpers`]: Numerical primitives (FFT, log-space arithmetic)

pub mod error;
pub mod functional;
pub mod math_helpers;

pub use error::{PldError, Result};

#[cfg(feature = "python-extension")]
mod python;

#[cfg(feature = "python-extension")]
use pyo3::prelude::*;

#[cfg(feature = "python-extension")]
#[pymodule]
fn opaque_dp_accounting(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register(m)?;
    Ok(())
}
