//! Functional API for differential privacy accounting
//!
//! This module provides a composable, immutable API for privacy analysis
//! based on the Process trait and evidence-based extensibility.
//!
//! ## Core Concepts
//!
//! - **Process**: Anything that can be evaluated for privacy metrics
//! - **Evidence**: Type-level proofs for specialized amplification techniques
//! - **Composition**: Universal via PLD convolution
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use opaque_dp_accounting::functional::*;
//!
//! // Basic DP-SGD (noise_multiplier must be in [0.1, 1.2])
//! let process = repeat(poisson(gaussian(1.1)?, 0.01), 1000)?;
//! let epsilon = process.epsilon_at(1e-5)?;
//! ```

pub mod adjacency;
pub mod amplification;
pub mod cached;
pub mod calibrate;
pub mod composition;
pub mod discretization;
pub mod mechanisms;
pub mod pld;
pub mod process;
pub mod transforms;

#[cfg(test)]
pub mod tests;

// Re-export key types
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
