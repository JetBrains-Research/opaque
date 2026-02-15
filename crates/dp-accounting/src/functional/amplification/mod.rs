//! Privacy amplification by subsampling
//!
//! This module provides Poisson subsampling amplification for privacy mechanisms.
//! Poisson subsampling includes each record independently with probability q,
//! providing privacy amplification through reduced data exposure.
//!
//! # References
//!
//! - \[BBG18\]: Balle, Barthe, Gavin. "Privacy amplification by subsampling."
//! - \[LRKS25\]: "Avoiding pitfalls for privacy accounting of subsampled mechanisms."

pub mod accumulated;
pub mod poisson;
pub mod truncated;

pub use accumulated::{
    accumulate, accumulate_with, AccumulateAmplifiable, AccumulateEvidence, Accumulated,
    TightGaussianAccumulateEvidence,
};
pub use poisson::{
    poisson, poisson_with, Poisson, PoissonAmplifiable, PoissonEvidence,
    TightGaussianPoissonEvidence,
};
pub use truncated::{
    truncated_poisson, truncated_poisson_with, TightGaussianTruncatedPoissonEvidence,
    TruncatedPoisson, TruncatedPoissonAmplifiable, TruncatedPoissonEvidence,
};
