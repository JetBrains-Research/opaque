//! Matrix factorization (MF) privacy accounting.
//!
//! This module provides privacy accounting for matrix factorization DP
//! mechanisms, including BandMF and BLT strategies. Unlike standard
//! DP-SGD which uses i.i.d. Gaussian noise at each step, MF mechanisms
//! inject *correlated* noise across all training steps via a matrix
//! factorization A = C⁻¹.
//!
//! # Privacy Analysis
//!
//! The privacy of the entire MF training run reduces to a single Gaussian
//! mechanism with effective noise multiplier σ/S, where:
//! - σ is the raw noise standard deviation
//! - S is the L2 sensitivity of the encoder matrix C under the given
//!   participation pattern
//!
//! The sensitivity S captures the correlation structure and depends on
//! both the MF strategy and the participation pattern:
//!
//! | Strategy | Participation | Sensitivity function |
//! |----------|--------------|---------------------|
//! | BandMF   | Single       | [`single_participation_sensitivity`] |
//! | BandMF   | Min-sep      | [`banded_sensitivity`] |
//! | BLT      | Min-sep      | [`banded_sensitivity`] or [`general_sensitivity_upper_bound`] |
//! | Any      | General      | [`general_sensitivity_upper_bound`] |
//!
//! # References
//!
//! - BandMF: Choquette-Choo et al. (2023) <https://arxiv.org/abs/2306.08153>
//! - BLT: Choquette-Choo et al. (2024) <https://arxiv.org/abs/2404.16706>

mod mf_gaussian;
pub mod sensitivity;

pub use mf_gaussian::mf_gaussian_pld;
pub use sensitivity::{
    banded_sensitivity, blt_sensitivity_squared, fixed_epoch_sensitivity,
    general_sensitivity_upper_bound, max_participation_for_linear_fn,
    minsep_true_max_participations, single_participation_sensitivity,
    toeplitz_minsep_sensitivity_squared,
};
