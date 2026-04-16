//! Discretization configuration and epsilon bounds
//!
//! Controls the accuracy-performance tradeoff when approximating continuous privacy
//! loss distributions with discrete probability mass functions.

use crate::error::{PldError, Result};

/// Configuration for Connect-the-Dots PLD discretization
///
/// Controls the accuracy-performance tradeoff when approximating continuous privacy
/// loss distributions with discrete probability mass functions.
///
/// # Defaults
///
/// The default configuration (`DiscretizationConfig::default()`) uses:
/// discretization = 1e-4, log_mass_truncation_bound = -50.0,
/// pessimistic_estimate = true.
///
/// The truncation bound -50 matches Google dp_accounting's default, ensuring
/// Poisson-subsampled mechanisms produce identical epsilon bounds and grid
/// sizes for accurate beta computation after FFT-based composition.
///
/// # References
///
/// See Doroshenko et al. (2022) for algorithm details and truncation analysis.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DiscretizationConfig {
    /// Grid spacing Δε between consecutive epsilon values
    ///
    /// Smaller values give more accurate approximations but require more computation.
    /// Typical range: 1e-5 (very accurate) to 1e-3 (coarse). Default: 1e-4.
    ///
    /// The grid will have approximately (ε_upper - ε_lower) / discretization points.
    /// If this exceeds `max_grid_size`, the effective discretization is automatically
    /// coarsened to `discretization * 2^k` for the smallest k that fits.
    pub discretization: f64,

    /// Log of probability mass to truncate from distribution tails
    ///
    /// The PLD is computed only for outcomes with tail probability ≥ exp(log_mass_truncation_bound).
    /// More negative values give higher accuracy but larger grids.
    ///
    /// Example: -50 means truncate mass < e^{-50} ≈ 1.9×10^{-22}. Default: -50.0.
    pub log_mass_truncation_bound: f64,

    /// Whether to use pessimistic (conservative) estimation
    ///
    /// - `true` (default): Adds truncated tail mass to boundary elements, providing
    ///   a conservative upper bound on privacy loss. Use for privacy guarantees.
    /// - `false`: Discards truncated mass, providing a tighter but potentially
    ///   underestimating bound. Use only for analysis, not guarantees.
    pub pessimistic_estimate: bool,

    /// Maximum number of grid points allowed before adaptive coarsening kicks in
    ///
    /// When the epsilon range would produce more than `max_grid_size` points at the
    /// base `discretization`, the effective discretization is automatically coarsened
    /// to `discretization * 2^k` (smallest power-of-2 multiplier that fits).
    ///
    /// Default: 10,000,000. Set to `usize::MAX` to disable adaptive coarsening.
    pub max_grid_size: usize,

    /// Total tail mass budget for Chernoff truncation during composition.
    ///
    /// During `self_compose()`, the composed PLD is truncated using Chernoff bounds
    /// with this total budget split equally between left and right tails.
    /// Smaller values preserve more tail precision at the cost of larger composed grids.
    ///
    /// Default: 1e-15, matching Google dp_accounting's `tail_mass_truncation`.
    pub tail_mass_truncation: f64,

    /// Number of Monte Carlo samples for MC-based PLD computation.
    ///
    /// Used by BnB Monte Carlo and b-min-sep accounting.
    /// Ignored by analytic PLD paths (Gaussian, Poisson).
    /// Default: 100,000.
    pub num_mc_samples: usize,

    /// RNG seed for reproducibility of Monte Carlo PLD computation.
    ///
    /// Default: 42.
    pub seed: u64,
}

impl DiscretizationConfig {
    /// Create a new discretization configuration with pessimistic estimation (default)
    ///
    /// # Errors
    ///
    /// * `PldError::InvalidParameter` - If discretization ≤ 0 or log_mass_truncation_bound ≥ 0
    pub fn new(discretization: f64, log_mass_truncation_bound: f64) -> Result<Self> {
        Self::with_estimate(discretization, log_mass_truncation_bound, true)
    }

    /// Create a new discretization configuration with specified estimation mode
    ///
    /// # Errors
    ///
    /// * `PldError::InvalidParameter` - If discretization ≤ 0 or log_mass_truncation_bound ≥ 0
    pub fn with_estimate(
        discretization: f64,
        log_mass_truncation_bound: f64,
        pessimistic_estimate: bool,
    ) -> Result<Self> {
        if discretization <= 0.0 {
            return Err(PldError::InvalidParameter(format!(
                "Discretization must be positive, got {}",
                discretization
            )));
        }
        if log_mass_truncation_bound >= 0.0 {
            return Err(PldError::InvalidParameter(format!(
                "Log mass truncation bound must be negative, got {}",
                log_mass_truncation_bound
            )));
        }

        Ok(Self {
            discretization,
            log_mass_truncation_bound,
            pessimistic_estimate,
            max_grid_size: 10_000_000,
            tail_mass_truncation: 1e-15,
            num_mc_samples: 100_000,
            seed: 42,
        })
    }

    /// Builder method to override the maximum grid size
    pub fn with_max_grid_size(mut self, max_grid_size: usize) -> Self {
        self.max_grid_size = max_grid_size;
        self
    }

    /// Compute the effective discretization adapted to the given epsilon bounds.
    ///
    /// If the grid at `self.discretization` would exceed `max_grid_size`, coarsen
    /// to `self.discretization * 2^k` for the smallest k that fits.
    pub(crate) fn effective_discretization(&self, bounds: &EpsilonBounds) -> f64 {
        let range = bounds.epsilon_upper - bounds.epsilon_lower;
        let grid_at_base = (range / self.discretization).ceil() as usize;
        if grid_at_base <= self.max_grid_size {
            return self.discretization;
        }
        let raw_ratio = (grid_at_base as f64 / self.max_grid_size as f64).ceil();
        let k = (raw_ratio.log2().ceil() as u32).max(1);
        self.discretization * 2_f64.powi(k as i32)
    }
}

impl Default for DiscretizationConfig {
    fn default() -> Self {
        Self {
            discretization: 1e-4,
            log_mass_truncation_bound: -50.0,
            pessimistic_estimate: true,
            max_grid_size: 10_000_000,
            tail_mass_truncation: 1e-15,
            num_mc_samples: 100_000,
            seed: 42,
        }
    }
}

/// Privacy loss bounds for PLD discretization
///
/// Specifies the range [ε_lower, ε_upper] over which the Privacy Loss Distribution
/// will be discretized.
#[derive(Debug, Clone, Copy)]
pub(crate) struct EpsilonBounds {
    /// Lower bound of the privacy loss range (minimum epsilon)
    pub epsilon_lower: f64,

    /// Upper bound of the privacy loss range (maximum epsilon)
    pub epsilon_upper: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discretization_config_valid() {
        let config = DiscretizationConfig::new(0.01, -50.0).unwrap();
        assert_eq!(config.discretization, 0.01);
        assert_eq!(config.log_mass_truncation_bound, -50.0);
    }

    #[test]
    fn test_discretization_config_default() {
        let config = DiscretizationConfig::default();
        assert_eq!(config.discretization, 1e-4);
        assert_eq!(config.log_mass_truncation_bound, -50.0);
        assert_eq!(config.pessimistic_estimate, true);
    }

    #[test]
    fn test_discretization_config_invalid_discretization() {
        assert!(DiscretizationConfig::new(0.0, -50.0).is_err());
        assert!(DiscretizationConfig::new(-0.01, -50.0).is_err());
    }

    #[test]
    fn test_discretization_config_invalid_log_mass() {
        assert!(DiscretizationConfig::new(0.01, 0.0).is_err());
        assert!(DiscretizationConfig::new(0.01, 1.0).is_err());
    }

    #[test]
    fn test_discretization_config_with_estimate() {
        let pessimistic = DiscretizationConfig::with_estimate(0.01, -30.0, true).unwrap();
        assert!(pessimistic.pessimistic_estimate);

        let optimistic = DiscretizationConfig::with_estimate(0.01, -30.0, false).unwrap();
        assert!(!optimistic.pessimistic_estimate);
    }

    #[test]
    fn test_effective_disc_no_coarsening_when_grid_fits() {
        let config = DiscretizationConfig::default();
        let bounds = EpsilonBounds {
            epsilon_lower: -50.0,
            epsilon_upper: 50.0,
        };
        let eff = config.effective_discretization(&bounds);
        assert!((eff - 1e-4).abs() < 1e-15);
    }

    #[test]
    fn test_effective_disc_coarsens_when_grid_exceeds_max() {
        let config = DiscretizationConfig::default();
        let bounds = EpsilonBounds {
            epsilon_lower: -1600.0,
            epsilon_upper: 1200.0,
        };
        let eff = config.effective_discretization(&bounds);
        assert!(eff > 1e-4);
        let grid = ((bounds.epsilon_upper - bounds.epsilon_lower) / eff).ceil() as usize;
        assert!(grid <= 10_000_000);
    }

    #[test]
    fn test_effective_disc_is_power_of_2_multiple() {
        let config = DiscretizationConfig::default();
        let test_ranges = [
            (-1600.0, 1200.0),
            (-650.0, 550.0),
            (-281.0, 256.0),
            (-45.0, 41.0),
        ];
        for (lo, hi) in test_ranges {
            let bounds = EpsilonBounds {
                epsilon_lower: lo,
                epsilon_upper: hi,
            };
            let eff = config.effective_discretization(&bounds);
            let ratio = eff / config.discretization;
            let rounded = ratio.round();
            assert!((ratio - rounded).abs() < 1e-9);
            assert!((rounded as usize).is_power_of_two());
        }
    }

    #[test]
    fn test_effective_disc_unlimited_grid_never_coarsens() {
        let config = DiscretizationConfig::default().with_max_grid_size(usize::MAX);
        let bounds = EpsilonBounds {
            epsilon_lower: -20000.0,
            epsilon_upper: 10000.0,
        };
        let eff = config.effective_discretization(&bounds);
        assert!((eff - 1e-4).abs() < 1e-15);
    }

    #[test]
    fn test_effective_disc_smallest_sufficient_power_of_2() {
        let config = DiscretizationConfig::default();
        let bounds = EpsilonBounds {
            epsilon_lower: -1600.0,
            epsilon_upper: 1200.0,
        };
        let eff = config.effective_discretization(&bounds);
        let factor = (eff / config.discretization).round() as usize;
        assert_eq!(factor, 4);
    }
}
