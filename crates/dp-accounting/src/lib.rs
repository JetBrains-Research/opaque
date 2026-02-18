//! # opaque-accounting
//!
//! Privacy Loss Distribution (PLD) accounting for differential privacy.
//!
//! This crate provides privacy accounting using the Privacy Loss Distribution
//! framework with Connect-the-Dots discretization.
//!
//! ## Architecture
//!
//! Rust is a **PLD computation engine**: flat functions that take scalar
//! parameters and return opaque PLD handles. Python owns composition,
//! repetition, caching, and calibration.
//!
//! ## Rust API
//!
//! ```rust
//! use opaque_accounting::mechanisms::gaussian_pld;
//! use opaque_accounting::amplification::poisson_gaussian_pld;
//! use opaque_accounting::DiscretizationConfig;
//!
//! let config = DiscretizationConfig::default();
//! let pld = gaussian_pld(1.1, &config).unwrap();
//! let epsilon = pld.epsilon_at(1e-5);
//! ```
//!
//! ## Module Overview
//!
//! - [`mechanisms`]: Gaussian, (ε,δ), identity PLD constructors
//! - [`amplification`]: Poisson, truncated Poisson, accumulated PLDs
//! - [`adaclip`]: Adaptive clipping sensitivity formula
//! - [`pld`]: The `PrivacyLossDistribution` type and metrics
//! - [`discretization`]: Connect-the-Dots discretization
//! - [`error`]: Error types
//! - [`math_helpers`]: Numerical primitives

pub mod error;
pub mod math_helpers;

// --- Core infrastructure (kept from functional/) ---
#[path = "functional/adjacency.rs"]
pub(crate) mod adjacency;
#[path = "functional/discretization/mod.rs"]
pub(crate) mod discretization;
#[path = "functional/pld/mod.rs"]
pub mod pld;

// --- New flat-function modules ---
pub mod adaclip;
pub mod amplification;
pub mod mechanisms;

// --- Public re-exports ---
pub use discretization::DiscretizationConfig;
pub use error::{PldError, Result};
pub use pld::PrivacyLossDistribution;
pub use adjacency::Adjacency;

#[cfg(feature = "python-extension")]
mod python;

#[cfg(feature = "python-extension")]
use pyo3::prelude::*;

#[cfg(feature = "python-extension")]
#[pymodule]
fn opaque_accounting(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register(m)?;
    Ok(())
}
