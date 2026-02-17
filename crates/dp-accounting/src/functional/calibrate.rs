//! Calibration for the functional API
//!
//! Finds optimal parameters for privacy processes via binary search.
//!
//! # Overview
//!
//! Calibration solves the inverse problem: given a privacy target (e.g., ε ≤ 1.0
//! at δ = 10⁻⁵), find the minimum noise multiplier that satisfies the constraint.
//!
//! The `calibrate` function takes a closure that builds a `Process` from a parameter
//! value, and binary-searches over the parameter space.
//!
//! # Examples
//!
//! ```rust,ignore
//! use opaque_dp_accounting::functional::*;
//! use opaque_dp_accounting::functional::calibrate::*;
//!
//! // Find minimum noise for ε ≤ 1.0 at δ = 1e-5
//! let result = calibrate(
//!     target_epsilon_at(1.0, 1e-5),
//!     |noise| gaussian(noise),
//!     CalibrateConfig::default(),
//! ).unwrap();
//!
//! println!("Optimal noise: {:.3}", result.param);
//! println!("Achieved epsilon: {:.6}", result.achieved);
//! ```

use crate::error::{PldError, Result};
use crate::process::Process;

/// Type alias for the metric evaluation closure used in calibration
type MetricEvaluator<P> = Box<dyn Fn(&P) -> Result<f64>>;

/// Calibration target: which privacy metric to constrain
///
/// Defines the privacy constraint that calibration should satisfy.
/// All metrics use the convention that increasing the noise parameter improves
/// privacy, but the *direction* of the metric differs:
///
/// - Epsilon, Delta, Advantage: **lower** = more private (satisfied when metric ≤ bound)
/// - Beta, Risk: **higher** = more private (satisfied when metric ≥ bound), because higher
///   error rates mean the attacker's hypothesis test is weaker
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Target {
    /// Target: ε ≤ bound for given δ
    Epsilon {
        /// Maximum acceptable epsilon
        bound: f64,
        /// Delta value (failure probability) to evaluate at
        delta: f64,
    },

    /// Target: δ ≤ bound for given ε
    Delta {
        /// Maximum acceptable delta
        bound: f64,
        /// Epsilon value to evaluate at
        epsilon: f64,
    },

    /// Target: advantage ≤ bound
    ///
    /// Advantage is δ(ε=0), the maximum discriminative power without any
    /// epsilon budget. Lower advantage = more private.
    Advantage {
        /// Maximum acceptable advantage
        bound: f64,
    },

    /// Target: β(α) ≥ bound
    ///
    /// Beta is the false negative rate in the optimal hypothesis test at
    /// false positive rate α. Higher β = weaker test = more private.
    /// Note the reversed direction: satisfied when metric **≥** bound.
    ///
    /// # Warning: Non-Monotonic Metric
    ///
    /// Beta is **not monotonic** in noise multiplier. It increases with noise up to
    /// a peak (around nm ≈ 5 for Gaussian), then decreases toward 0.5 as noise → ∞
    /// (the mechanism becomes a coin flip). Binary search only works on the ascending
    /// side, so you should set `param_max` to stay in the monotonic region.
    Beta {
        /// Minimum acceptable beta (false negative rate)
        bound: f64,
        /// Alpha (false positive rate) to evaluate at
        alpha: f64,
    },

    /// Target: risk(prior) ≥ bound
    ///
    /// Bayes risk is the minimum achievable error rate in a binary hypothesis
    /// test with the given prior. Higher risk = attacker makes more errors = more private.
    /// Note the reversed direction: satisfied when metric **≥** bound.
    ///
    /// # Warning: Non-Monotonic Metric
    ///
    /// Risk is **not monotonic** in noise multiplier. It increases with noise up to
    /// a peak (around nm ≈ 10 for Gaussian with prior=0.5), then decreases toward 0.5
    /// as noise → ∞. Binary search only works on the ascending side, so you should
    /// set `param_max` to stay in the monotonic region (e.g., ≤ 5).
    Risk {
        /// Minimum acceptable Bayes risk
        bound: f64,
        /// Prior probability of the "true" dataset
        prior: f64,
    },
}

impl Target {
    /// Returns the bound value for this target
    pub fn bound(&self) -> f64 {
        match self {
            Target::Epsilon { bound, .. }
            | Target::Delta { bound, .. }
            | Target::Advantage { bound }
            | Target::Beta { bound, .. }
            | Target::Risk { bound, .. } => *bound,
        }
    }

    /// Returns true if this target uses reversed semantics (higher = better privacy)
    pub fn is_reversed(&self) -> bool {
        matches!(self, Target::Beta { .. } | Target::Risk { .. })
    }
}

/// Convenience constructor for epsilon target
///
/// # Arguments
///
/// * `epsilon` - Maximum acceptable epsilon
/// * `delta` - Delta value to evaluate at
pub fn target_epsilon_at(epsilon: f64, delta: f64) -> Target {
    Target::Epsilon {
        bound: epsilon,
        delta,
    }
}

/// Convenience constructor for delta target
///
/// # Arguments
///
/// * `delta` - Maximum acceptable delta
/// * `epsilon` - Epsilon value to evaluate at
pub fn target_delta_at(delta: f64, epsilon: f64) -> Target {
    Target::Delta {
        bound: delta,
        epsilon,
    }
}

/// Convenience constructor for advantage target
///
/// # Arguments
///
/// * `advantage` - Maximum acceptable advantage
pub fn target_advantage(advantage: f64) -> Target {
    Target::Advantage { bound: advantage }
}

/// Convenience constructor for beta target
///
/// # Arguments
///
/// * `beta` - Minimum acceptable beta (false negative rate)
/// * `alpha` - False positive rate to evaluate at
pub fn target_beta_at(beta: f64, alpha: f64) -> Target {
    Target::Beta { bound: beta, alpha }
}

/// Convenience constructor for risk target
///
/// # Arguments
///
/// * `risk` - Minimum acceptable Bayes risk
/// * `prior` - Prior probability
pub fn target_risk_at(risk: f64, prior: f64) -> Target {
    Target::Risk { bound: risk, prior }
}

/// Configuration for calibration search
///
/// Controls the binary search algorithm parameters.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CalibrateConfig {
    /// Minimum parameter value to search
    pub param_min: f64,
    /// Maximum parameter value to search
    pub param_max: f64,
    /// Convergence tolerance on parameter interval width
    pub tolerance: f64,
    /// Maximum number of binary search iterations
    pub max_iterations: usize,
}

impl Default for CalibrateConfig {
    fn default() -> Self {
        Self {
            param_min: 0.1,
            param_max: 1.2,
            tolerance: 1e-6,
            max_iterations: 100,
        }
    }
}

impl CalibrateConfig {
    /// Set search bounds
    pub fn with_bounds(mut self, min: f64, max: f64) -> Self {
        self.param_min = min;
        self.param_max = max;
        self
    }

    /// Set convergence tolerance
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Set maximum iterations
    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }
}

/// Result of calibration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CalibrateResult {
    /// Optimal parameter value found
    pub param: f64,
    /// Achieved metric value at optimal parameter
    pub achieved: f64,
    /// Did the search converge?
    pub converged: bool,
    /// Number of process evaluations performed
    pub evaluations: usize,
}

/// Calibrate a parameter for a privacy process
///
/// Finds the minimum parameter value θ such that the process built by `build(θ)`
/// satisfies the privacy `target`. Assumes that increasing the parameter improves
/// privacy (i.e., the metric decreases monotonically with parameter).
///
/// This is the standard assumption for noise multiplier calibration: more noise
/// → better privacy → smaller ε/δ.
///
/// # Algorithm
///
/// Binary search over [`config.param_min`, `config.param_max`]:
/// 1. Evaluate privacy metric at midpoint
/// 2. If metric ≤ target: search lower half (try less noise)
/// 3. If metric > target: search upper half (need more noise)
/// 4. Converge when interval width < `config.tolerance`
/// 5. Return conservative bound (upper end of final interval)
///
/// # Arguments
///
/// * `target` - Privacy constraint to satisfy
/// * `build` - Closure that builds a `Process` from a parameter value
/// * `config` - Search configuration (use `CalibrateConfig::default()` for [0.1, 1.2])
///
/// # Returns
///
/// The optimal parameter and achieved privacy metric.
///
/// # Errors
///
/// Returns error if:
/// - Configuration is invalid (min ≥ max, tolerance ≤ 0)
/// - Process evaluation fails at any point
/// - Search does not converge within `max_iterations`
///
/// # Examples
///
/// ```rust,ignore
/// use opaque_dp_accounting::functional::*;
/// use opaque_dp_accounting::functional::calibrate::*;
///
/// // Find noise for ε ≤ 8.0 at δ = 1e-6
/// let result = calibrate(
///     target_epsilon_at(8.0, 1e-6),
///     |noise| gaussian(noise),
///     CalibrateConfig::default(),
/// ).unwrap();
/// assert!(result.converged);
/// ```
pub fn calibrate<P, F>(target: Target, build: F, config: CalibrateConfig) -> Result<CalibrateResult>
where
    P: Process,
    F: Fn(f64) -> Result<P>,
{
    // Validate config
    if config.param_min >= config.param_max {
        return Err(PldError::CalibrationInvalidConfig(
            "param_min must be less than param_max".into(),
        ));
    }
    if config.tolerance <= 0.0 {
        return Err(PldError::CalibrationInvalidConfig(
            "tolerance must be positive".into(),
        ));
    }
    if config.max_iterations == 0 {
        return Err(PldError::CalibrationInvalidConfig(
            "max_iterations must be positive".into(),
        ));
    }

    // Evaluate the privacy metric for the current process.
    //
    // For epsilon targets: epsilon_at() may return inf when the PLD grid is too
    // narrow. This happens for large noise multipliers where the mechanism is
    // very private. When inf is returned, the constraint IS satisfied.
    let evaluate: MetricEvaluator<P> = match target {
        Target::Epsilon { delta, .. } => Box::new(move |p: &P| p.epsilon_at(delta)),
        Target::Delta { epsilon, .. } => Box::new(move |p: &P| p.delta_at(epsilon)),
        Target::Advantage { .. } => Box::new(|p: &P| p.advantage()),
        Target::Beta { alpha, .. } => Box::new(move |p: &P| p.beta_at(alpha)),
        Target::Risk { prior, .. } => Box::new(move |p: &P| p.risk_at(prior)),
    };

    let target_value = target.bound();
    let is_reversed = target.is_reversed();

    // Determine if the constraint is satisfied.
    //
    // Most metrics: lower = more private → satisfied when metric ≤ bound.
    // Beta/Risk: higher = more private → satisfied when metric ≥ bound.
    //
    // No special-casing for infinity: only `epsilon_at` can return inf (when
    // the PLD grid is too narrow), and `inf <= bound` is false, which correctly
    // drives binary search toward more noise.
    let is_satisfied = |metric: f64| -> bool {
        if is_reversed {
            metric >= target_value
        } else {
            metric <= target_value
        }
    };

    let mut low = config.param_min;
    let mut high = config.param_max;
    let mut evaluations = 0;

    // Binary search: increasing parameter → decreasing metric (better privacy)
    while evaluations < config.max_iterations {
        // Check convergence
        if (high - low).abs() < config.tolerance {
            // Return conservative bound (upper end)
            let process = build(high)?;
            let achieved = evaluate(&process)?;
            evaluations += 1;

            // If the achieved metric is infinite or NaN, the grid couldn't resolve
            // the answer even at the converged parameter — calibration failed.
            if achieved.is_infinite() || achieved.is_nan() {
                return Err(PldError::CalibrationConvergenceFailed {
                    iterations: evaluations,
                    last_param: high,
                });
            }

            // Verify the constraint is actually satisfied at the final parameter.
            // If not, the target is infeasible within the search bounds.
            if !is_satisfied(achieved) {
                return Err(PldError::CalibrationConvergenceFailed {
                    iterations: evaluations,
                    last_param: high,
                });
            }

            return Ok(CalibrateResult {
                param: high,
                achieved,
                converged: true,
                evaluations,
            });
        }

        let mid = (low + high) / 2.0;
        let process = build(mid)?;
        let metric = evaluate(&process)?;
        evaluations += 1;

        if is_satisfied(metric) {
            // Constraint satisfied — try smaller parameter (less noise)
            high = mid;
        } else {
            // Constraint not satisfied — need larger parameter (more noise)
            low = mid;
        }
    }

    // Max iterations reached without convergence
    Err(PldError::CalibrationConvergenceFailed {
        iterations: evaluations,
        last_param: high,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gaussian;

    #[test]
    fn test_epsilon_target_constructor() {
        let target = target_epsilon_at(1.0, 1e-5);
        match target {
            Target::Epsilon { bound, delta } => {
                assert_eq!(bound, 1.0);
                assert_eq!(delta, 1e-5);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_delta_target_constructor() {
        let target = target_delta_at(1e-5, 1.0);
        match target {
            Target::Delta { bound, epsilon } => {
                assert_eq!(bound, 1e-5);
                assert_eq!(epsilon, 1.0);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_advantage_target_constructor() {
        let target = target_advantage(0.1);
        match target {
            Target::Advantage { bound } => assert_eq!(bound, 0.1),
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_beta_target_constructor() {
        let target = target_beta_at(0.9, 0.05);
        match target {
            Target::Beta { bound, alpha } => {
                assert_eq!(bound, 0.9);
                assert_eq!(alpha, 0.05);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_risk_target_constructor() {
        let target = target_risk_at(0.3, 0.5);
        match target {
            Target::Risk { bound, prior } => {
                assert_eq!(bound, 0.3);
                assert_eq!(prior, 0.5);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_target_bound() {
        assert_eq!(target_epsilon_at(1.0, 1e-5).bound(), 1.0);
        assert_eq!(target_delta_at(1e-5, 1.0).bound(), 1e-5);
        assert_eq!(target_advantage(0.5).bound(), 0.5);
        assert_eq!(target_beta_at(0.9, 0.05).bound(), 0.9);
        assert_eq!(target_risk_at(0.3, 0.5).bound(), 0.3);
    }

    #[test]
    fn test_target_is_reversed() {
        assert!(!target_epsilon_at(1.0, 1e-5).is_reversed());
        assert!(!target_delta_at(1e-5, 1.0).is_reversed());
        assert!(!target_advantage(0.5).is_reversed());
        assert!(target_beta_at(0.9, 0.05).is_reversed());
        assert!(target_risk_at(0.3, 0.5).is_reversed());
    }

    #[test]
    fn test_calibrate_config_default() {
        let config = CalibrateConfig::default();
        assert_eq!(config.param_min, 0.1);
        assert_eq!(config.param_max, 1.2);
        assert_eq!(config.tolerance, 1e-6);
        assert_eq!(config.max_iterations, 100);
    }

    #[test]
    fn test_calibrate_config_builder() {
        let config = CalibrateConfig::default()
            .with_bounds(0.1, 50.0)
            .with_tolerance(1e-4)
            .with_max_iterations(200);
        assert_eq!(config.param_min, 0.1);
        assert_eq!(config.param_max, 50.0);
        assert_eq!(config.tolerance, 1e-4);
        assert_eq!(config.max_iterations, 200);
    }

    #[test]
    fn test_calibrate_invalid_bounds() {
        let result = calibrate(
            target_epsilon_at(1.0, 1e-5),
            gaussian,
            CalibrateConfig::default().with_bounds(10.0, 1.0),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_calibrate_invalid_tolerance() {
        let result = calibrate(
            target_epsilon_at(1.0, 1e-5),
            gaussian,
            CalibrateConfig::default().with_tolerance(-1.0),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_calibrate_invalid_max_iterations() {
        let result = calibrate(
            target_epsilon_at(1.0, 1e-5),
            gaussian,
            CalibrateConfig::default().with_max_iterations(0),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_calibrate_infeasible_target() {
        // Target ε ≤ 0.001 at δ = 1e-10 is infeasible within [0.1, 1.2]
        let result = calibrate(
            target_epsilon_at(0.001, 1e-10),
            gaussian,
            CalibrateConfig::default(),
        );
        assert!(result.is_err(), "infeasible target should fail");
    }

    #[test]
    fn test_epsilon_at_no_inf_for_realistic_params() {
        let nms = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.2];
        let deltas = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10];
        for &nm in &nms {
            for &delta in &deltas {
                let eps = gaussian(nm).unwrap().epsilon_at(delta).unwrap();
                assert!(
                    eps.is_finite(),
                    "epsilon_at({:e}) returned inf for nm={}",
                    delta,
                    nm
                );
            }
        }
    }

    #[test]
    fn test_calibrate_gaussian_epsilon() {
        // ε ≤ 5.0 at δ = 1e-5 needs nm ≈ 0.89, feasible within [0.1, 1.2]
        let result = calibrate(
            target_epsilon_at(5.0, 1e-5),
            gaussian,
            CalibrateConfig::default(),
        )
        .unwrap();

        assert!(result.converged);
        assert!(result.param > 0.0);
        let verified_eps = gaussian(result.param).unwrap().epsilon_at(1e-5).unwrap();
        assert!(
            verified_eps.is_finite(),
            "calibrated nm={} gives inf epsilon — grid too narrow",
            result.param
        );
        assert!(
            verified_eps <= 5.0 + 1e-3,
            "calibrated nm={} gives epsilon={} > target 5.0",
            result.param,
            verified_eps
        );
        assert!(result.evaluations > 0);
    }

    #[test]
    fn test_calibrate_gaussian_delta() {
        // δ ≤ 0.1 at ε = 1.0 is feasible within [0.1, 1.2]
        let config = CalibrateConfig::default();
        let tol = config.tolerance;
        let result = calibrate(target_delta_at(0.1, 1.0), gaussian, config).unwrap();

        assert!(result.converged);
        assert!(result.param > 0.0);
        let verified = gaussian(result.param).unwrap().delta_at(1.0).unwrap();
        assert!(
            verified <= 0.1 + tol,
            "nm={} gives delta={} > target 0.1",
            result.param,
            verified
        );
    }

    #[test]
    fn test_calibrate_gaussian_advantage() {
        let config = CalibrateConfig::default();
        let tol = config.tolerance;
        let result = calibrate(target_advantage(0.5), gaussian, config).unwrap();

        assert!(result.converged);
        let verified = gaussian(result.param).unwrap().advantage().unwrap();
        assert!(
            verified <= 0.5 + tol,
            "nm={} gives advantage={} > target 0.5",
            result.param,
            verified
        );
    }

    #[test]
    fn test_calibrate_gaussian_beta() {
        // Beta is reversed: higher β = more private.
        // Within [0.1, 1.2], β is monotonically increasing (ascending side).
        // β ≥ 0.7 at α = 0.05 is feasible within [0.1, 1.2].
        let config = CalibrateConfig::default();
        let tol = config.tolerance;
        let result = calibrate(target_beta_at(0.7, 0.05), gaussian, config).unwrap();

        assert!(result.converged);
        let verified = gaussian(result.param).unwrap().beta_at(0.05).unwrap();
        assert!(
            verified >= 0.7 - tol,
            "nm={} gives beta={} < target 0.7",
            result.param,
            verified
        );
    }

    #[test]
    fn test_calibrate_gaussian_risk() {
        // Risk is reversed: higher risk = more private (attacker errs more).
        // Within [0.1, 1.2], risk is monotonically increasing (ascending side).
        let config = CalibrateConfig::default();
        let tol = config.tolerance;
        let result = calibrate(target_risk_at(0.3, 0.5), gaussian, config).unwrap();

        assert!(result.converged);
        let verified = gaussian(result.param).unwrap().risk_at(0.5).unwrap();
        assert!(
            verified >= 0.3 - tol,
            "nm={} gives risk={} < target 0.3",
            result.param,
            verified
        );
    }

    #[test]
    fn test_calibrate_result_fields() {
        let result = calibrate(
            target_epsilon_at(5.0, 1e-5),
            gaussian,
            CalibrateConfig::default(),
        )
        .unwrap();

        assert!(result.converged);
        assert!(result.param > 0.0);
        assert!(result.achieved >= 0.0);
        assert!(result.evaluations > 0);
        assert!(result.evaluations <= 100);
    }
}
