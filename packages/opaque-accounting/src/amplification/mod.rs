//! Privacy amplification: subsampled and accumulated mechanism PLDs.
//!
//! Each public function takes scalar parameters directly (no structs, no traits).
//!
//! - [`poisson_gaussian_pld`] — Poisson-subsampled Gaussian
//! - [`truncated_poisson_gaussian_pld`] — Truncated Poisson-subsampled Gaussian
//! - [`parallel_poisson_gaussian_pld`] — Parallel Poisson Gaussian (gradient accumulation or parallel workers)
//! - [`balls_in_bins_gaussian_pld`] — Balls-in-Bins Gaussian (epoch-level composition)
//! - [`bnb_mc_pld`] — Balls-in-Bins Monte Carlo for matrix mechanisms
//! - [`bnb_deterministic_pld`] — Balls-in-Bins deterministic Rényi bound for matrix mechanisms
//! - [`random_allocation_pld`] — Random-allocation amplification from a base PLD (conservative)

pub mod balls_in_bins;
mod parallel_poisson;
pub(crate) mod poisson;
mod random_allocation;
mod truncated_poisson;

pub use balls_in_bins::bnb_deterministic_pld;
pub use balls_in_bins::{
    balls_in_bins_gaussian_pld, balls_in_bins_gaussian_pld_epochs, bnb_mc_pld,
};
pub use parallel_poisson::parallel_poisson_gaussian_pld;
pub use poisson::poisson_gaussian_pld;
pub use random_allocation::random_allocation_pld;
pub use truncated_poisson::truncated_poisson_gaussian_pld;

use crate::error::{PldError, Result};
use crate::mechanisms::validate_noise_multiplier;

fn validate_rate(rate: f64) -> Result<()> {
    if !(rate > 0.0 && rate < 1.0) {
        return Err(PldError::InvalidParameter(format!(
            "sampling rate must be in (0, 1), got {}",
            rate
        )));
    }
    Ok(())
}
