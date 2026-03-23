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
/// Values below this threshold cause numerical instability in PLD discretization
/// (grid explosion). The CGF-backed path handles small noise
/// multipliers via analytical CGF evaluation without discretization.
pub(crate) const MIN_NOISE_MULTIPLIER: f64 = 0.01;

/// Maximum supported noise multiplier.
///
/// Values above this threshold provide near-perfect privacy.
/// For σ > 2.5, use `identity()` instead.
pub(crate) const MAX_NOISE_MULTIPLIER: f64 = 2.5;

/// Maximum acceptable grid coarsening factor before switching to CGF.
///
/// When the epsilon grid at base discretization exceeds `max_grid_size`,
/// the effective discretization is coarsened by a power-of-2 factor.
/// If this factor exceeds this threshold, the CGF path is used instead
/// to avoid arithmetic errors from aggressive grid coarsening (the PMF
/// becomes Dirac-like with mass concentrated in too few grid cells).
pub(crate) const MAX_COARSENING_FACTOR: f64 = 1.0;

/// Maximum fraction of `max_grid_size` to use for a single-step PMF.
///
/// Even without coarsening, a large PMF grid makes self_compose()
/// expensive (FFT convolution grows the grid linearly with composition
/// count). When the grid exceeds this fraction of `max_grid_size`, we
/// prefer CGF for better composition performance.
pub(crate) const MAX_GRID_FRACTION: f64 = 0.5;

// ---------------------------------------------------------------------------
// Re-exports
// ---------------------------------------------------------------------------

pub use eps_delta::eps_delta_pld;
pub use gaussian::gaussian_pld;
pub use gaussian::cgf_gaussian_pld;
pub use identity::identity_pld;
pub use rectified_gaussian::rectified_gaussian_pld;
pub use truncated_gaussian::truncated_gaussian_pld;
