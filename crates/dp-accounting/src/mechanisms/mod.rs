//! Privacy mechanisms: flat functions producing PLDs.
//!
//! Each function takes scalar parameters and a discretization config,
//! and returns a `PrivacyLossDistribution`. No structs, no traits.
//!
//! - [`gaussian`]: Gaussian mechanism (base noise for DP-SGD)
//! - [`rectified_gaussian`]: Rectified (clamped) Gaussian mechanism
//! - [`truncated_gaussian`]: Truncated (renormalized) Gaussian mechanism
//! - [`eps_delta`]: Fixed (ε, δ)-DP mechanism
//! - [`identity`]: Identity (zero privacy loss) mechanism

mod eps_delta;
mod gaussian;
mod identity;
mod rectified_gaussian;
mod truncated_gaussian;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Minimum supported noise multiplier.
///
/// Values below this threshold cause numerical instability in discretization
/// (grid explosion, unreliable epsilon bounds).
pub(crate) const MIN_NOISE_MULTIPLIER: f64 = 0.1;

/// Maximum supported noise multiplier.
///
/// Values above this threshold cause numerical instability
/// (x-to-ε compression artifacts, unreliable beta/risk metrics).
pub(crate) const MAX_NOISE_MULTIPLIER: f64 = 1.2;

// ---------------------------------------------------------------------------
// Re-exports
// ---------------------------------------------------------------------------

pub use eps_delta::eps_delta_pld;
pub use gaussian::gaussian_pld;
pub use identity::identity_pld;
pub use rectified_gaussian::rectified_gaussian_pld;
pub use truncated_gaussian::truncated_gaussian_pld;
