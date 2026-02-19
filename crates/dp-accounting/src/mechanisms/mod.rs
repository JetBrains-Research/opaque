//! Privacy mechanisms: flat functions producing PLDs.
//!
//! Each function takes scalar parameters and a discretization config,
//! and returns a `PrivacyLossDistribution`. No structs, no traits.
//!
//! - [`gaussian`]: Gaussian mechanism (base noise for DP-SGD)
//! - [`bounded_gaussian`]: Bounded Gaussian mechanism (Replace adjacency)
//! - [`eps_delta`]: Fixed (ε, δ)-DP mechanism
//! - [`identity`]: Identity (zero privacy loss) mechanism

mod bounded_gaussian;
mod eps_delta;
mod gaussian;
mod identity;

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

pub use bounded_gaussian::bounded_gaussian_pld;
pub use eps_delta::eps_delta_pld;
pub use gaussian::gaussian_pld;
#[allow(unused_imports)] // used by bounded-Gaussian adjacency
pub(crate) use gaussian::gaussian_replace_pld;
pub use identity::identity_pld;
