//! # opaque-accounting
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
//! use opaque_accounting::*;
//!
//! // Standard DP-SGD: Poisson-subsampled Gaussian, 1000 rounds
//! let step = poisson(gaussian(1.1).unwrap(), 0.01);
//! let training = repeat(step, 1000).unwrap();
//! let epsilon = training.epsilon_at(1e-5).unwrap();
//! ```
//!
//! ## Module Overview
//!
//! - Composable functional API for privacy accounting
//! - [`error`]: Error types
//! - [`math_helpers`]: Numerical primitives (FFT, log-space arithmetic)

pub mod error;
pub mod math_helpers;

#[path = "functional/adjacency.rs"]
pub mod adjacency;
#[path = "functional/amplification/mod.rs"]
pub mod amplification;
#[path = "functional/cached.rs"]
pub mod cached;
#[path = "functional/calibrate.rs"]
pub mod calibrate;
#[path = "functional/composition.rs"]
pub mod composition;
#[path = "functional/discretization/mod.rs"]
pub mod discretization;
#[path = "functional/mechanisms/mod.rs"]
pub mod mechanisms;
#[path = "functional/pld/mod.rs"]
pub mod pld;
#[path = "functional/process.rs"]
pub mod process;
#[path = "functional/transforms/mod.rs"]
pub mod transforms;

pub use adjacency::Adjacency;
pub use amplification::{
    accumulate, accumulate_with, poisson, poisson_with, truncated_poisson, truncated_poisson_with,
    AccumulateAmplifiable, AccumulateEvidence, Accumulated, Poisson, PoissonAmplifiable,
    PoissonEvidence, TightGaussianAccumulateEvidence, TightGaussianPoissonEvidence,
    TightGaussianTruncatedPoissonEvidence, TruncatedPoisson, TruncatedPoissonAmplifiable,
    TruncatedPoissonEvidence,
};
pub use cached::{cached, Cached};
pub use calibrate::{
    calibrate, target_advantage, target_beta_at, target_delta_at, target_epsilon_at,
    target_risk_at, CalibrateConfig, CalibrateResult, Target,
};
pub use composition::{compose, repeat, repeat_flat, Composed, Repeated};
pub use discretization::DiscretizationConfig;
pub use mechanisms::{
    eps_delta, eps_delta_with, gaussian, gaussian_with, identity, identity_with, EpsDelta,
    Gaussian, Identity,
};
pub use pld::PrivacyLossDistribution;
pub use process::Process;
pub use transforms::{adaclip, AdaClip, AdaClipEvidence, AdaClipable, GaussianAdaClipEvidence};

pub use error::{PldError, Result};

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
