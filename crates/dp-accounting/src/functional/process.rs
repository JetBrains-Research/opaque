//! Process trait: the core abstraction for the functional API
//!
//! A `Process` represents any computation that can be analyzed for differential privacy.
//! The trait requires only one method: `pld()`, which computes the Privacy Loss Distribution.
//! All privacy queries (epsilon_at, delta_at) have default implementations.
//!
//! Each process owns its discretization configuration, eliminating the need to thread
//! config parameters through the API.

use super::pld::PrivacyLossDistribution;
use crate::error::Result;

/// Core trait for privacy processes
///
/// A `Process` is anything that can be evaluated for privacy metrics. This includes:
/// - Leaf mechanisms (Gaussian, Laplace, etc.)
/// - Composed processes (sequential, parallel)
/// - Amplified processes (Poisson subsampling, shuffling)
/// - Repeated processes (homogeneous DP-SGD)
///
/// # Required Method
///
/// Implementors must provide `pld()`, which computes the Privacy Loss Distribution
/// using the discretization configuration stored within the process.
///
/// # Default Methods
///
/// The trait provides default implementations for privacy queries:
/// - `epsilon_at(delta)`: Find epsilon for a given delta
/// - `delta_at(epsilon)`: Find delta for a given epsilon
/// - `advantage()`: Compute the advantage (TV privacy)
/// - `beta_at(alpha)`: Compute the trade-off function β(α)
/// - `risk_at(prior)`: Compute the Bayes risk for a given prior
///
/// # Examples
///
/// ```rust,ignore
/// use opaque_dp_accounting::functional::*;
///
/// let process = gaussian(1.1);
/// let pld = process.pld()?;
/// let epsilon = pld.epsilon_at(1e-5);
/// ```
pub trait Process {
    /// Compute the Privacy Loss Distribution for this process
    ///
    /// Uses the discretization configuration stored within the process.
    ///
    /// # Returns
    ///
    /// A `PrivacyLossDistribution` representing the privacy guarantee of this process.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Numerical computation fails
    /// - Process parameters are out of valid range
    fn pld(&self) -> Result<PrivacyLossDistribution>;

    /// Get epsilon for a given delta (default implementation)
    ///
    /// Computes the PLD and queries it for epsilon. Override this method if you have
    /// a more efficient implementation that avoids PLD computation.
    ///
    /// # Arguments
    ///
    /// * `delta` - The target delta value
    ///
    /// # Returns
    ///
    /// The epsilon value such that the process is (epsilon, delta)-DP.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let process = gaussian(1.1);
    /// let epsilon = process.epsilon_at(1e-5)?;
    /// ```
    fn epsilon_at(&self, delta: f64) -> Result<f64> {
        let pld = self.pld()?;
        Ok(pld.epsilon_at(delta))
    }

    /// Get delta for a given epsilon (default implementation)
    ///
    /// Computes the PLD and queries it for delta. Override this method if you have
    /// a more efficient implementation that avoids PLD computation.
    ///
    /// # Arguments
    ///
    /// * `epsilon` - The target epsilon value
    ///
    /// # Returns
    ///
    /// The delta value such that the process is (epsilon, delta)-DP.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let process = gaussian(1.1);
    /// let delta = process.delta_at(1.0)?;
    /// ```
    fn delta_at(&self, epsilon: f64) -> Result<f64> {
        let pld = self.pld()?;
        Ok(pld.delta_at(epsilon))
    }

    /// Compute the advantage (TV privacy) (default implementation)
    ///
    /// The advantage represents the maximum discriminative power of any attacker.
    ///
    /// # Returns
    ///
    /// The advantage value in [0, 1].
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let process = gaussian(1.1);
    /// let advantage = process.advantage()?;
    /// ```
    fn advantage(&self) -> Result<f64> {
        let pld = self.pld()?;
        Ok(pld.advantage())
    }

    /// Compute the trade-off function β(α) (default implementation)
    ///
    /// Maps false positive rate (α) to false negative rate (β) for the
    /// optimal hypothesis test distinguishing neighboring datasets.
    ///
    /// # Arguments
    ///
    /// * `alpha` - False positive rate (Type I error rate) in [0, 1]
    ///
    /// # Returns
    ///
    /// The false negative rate β in [0, 1].
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let process = gaussian(1.1);
    /// let beta = process.beta_at(0.01)?;
    /// ```
    fn beta_at(&self, alpha: f64) -> Result<f64> {
        let pld = self.pld()?;
        Ok(pld.beta_at(alpha))
    }

    /// Compute the Bayes risk for a given prior (default implementation)
    ///
    /// Bayes risk measures the maximum accuracy of an attack against privacy
    /// of a single record under a binary prior.
    ///
    /// # Arguments
    ///
    /// * `prior` - Prior probability in [0, 1]
    ///
    /// # Returns
    ///
    /// The Bayes risk value in [0, 0.5].
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let process = gaussian(1.1);
    /// let risk = process.risk_at(0.5)?;
    /// ```
    fn risk_at(&self, prior: f64) -> Result<f64> {
        let pld = self.pld()?;
        Ok(pld.risk_at(prior))
    }
}
