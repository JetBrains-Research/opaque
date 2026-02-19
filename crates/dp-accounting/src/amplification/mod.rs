//! Privacy amplification: subsampled and accumulated mechanism PLDs.
//!
//! Each public function takes scalar parameters directly (no structs, no traits).
//!
//! - [`poisson_gaussian_pld`] — Poisson-subsampled Gaussian
//! - [`truncated_poisson_gaussian_pld`] — Truncated Poisson-subsampled Gaussian
//! - [`accumulated_poisson_gaussian_pld`] — Gradient-accumulated Poisson Gaussian

mod accumulated;
mod poisson;
mod truncated_poisson;

pub use accumulated::accumulated_poisson_gaussian_pld;
pub use poisson::poisson_gaussian_pld;
pub use truncated_poisson::truncated_poisson_gaussian_pld;

use crate::error::{PldError, Result};
use crate::mechanisms::{MAX_NOISE_MULTIPLIER, MIN_NOISE_MULTIPLIER};

// ===========================================================================
// Shared validation helpers (visible to child modules via `super::`)
// ===========================================================================

fn validate_noise_multiplier(nm: f64) -> Result<()> {
    if !(MIN_NOISE_MULTIPLIER..=MAX_NOISE_MULTIPLIER).contains(&nm) {
        return Err(PldError::InvalidParameter(format!(
            "noise_multiplier must be in [{}, {}], got {}",
            MIN_NOISE_MULTIPLIER, MAX_NOISE_MULTIPLIER, nm
        )));
    }
    Ok(())
}

fn validate_rate(rate: f64) -> Result<()> {
    if !(rate > 0.0 && rate <= 1.0) {
        return Err(PldError::InvalidParameter(format!(
            "sampling rate must be in (0, 1], got {}",
            rate
        )));
    }
    Ok(())
}
