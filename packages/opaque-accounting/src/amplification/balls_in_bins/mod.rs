//! Balls-in-Bins privacy amplification.
//!
//! Two algorithms for the same BnB sampling scheme (dataset partitioned into
//! `num_bins` bins each epoch, each bin processed once):
//!
//! - [`gaussian`] — Conservative Poisson per-step approximation for independent
//!   (Gaussian) noise.  Closed-form, fast.
//! - [`monte_carlo`] — Monte Carlo sampling of the dominating pair for
//!   correlated-noise matrix mechanisms (DP-λCGD, BISR, BLT).  Uses banded
//!   Cholesky on the Gram matrix.

mod gaussian;
pub mod deterministic;
pub mod monte_carlo;

pub use gaussian::{balls_in_bins_gaussian_pld, balls_in_bins_gaussian_pld_epochs};
pub use deterministic::{
    bnb_deterministic_delta_curve, bnb_deterministic_pld, DeterministicOptions,
};
pub use monte_carlo::bnb_mc_pld;
