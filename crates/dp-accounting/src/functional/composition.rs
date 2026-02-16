//! Composition types for the functional API
//!
//! Provides `Repeated<P>` (homogeneous composition) and `Composed<L, R>`
//! (heterogeneous composition) wrappers that implement the `Process` trait.
//!
//! # Optimization
//!
//! At `pld()` evaluation time, `Composed` detects when both sides share the
//! same underlying process and merges them into a single `self_compose()` call,
//! which is significantly cheaper than separate PLD convolution.
//!
//! Merge cases handled (stable):
//! - `compose(a, a)` → `a.pld().self_compose(2)`
//! - `compose(repeat(a, n), a)` → `a.pld().self_compose(n + 1)`
//! - `compose(a, repeat(a, n))` → `a.pld().self_compose(n + 1)`
//! - `compose(repeat(a, n), repeat(a, n))` → effectively `self_compose(2n)`
//!
//! Additional case with `nightly` feature:
//! - `compose(repeat(a, n), repeat(a, m))` → `a.pld().self_compose(n + m)`
//!
//! Nested `repeat` is flattened at construction time via [`repeat_flat`]:
//! - `repeat_flat(repeat(a, n), m)` → `Repeated { inner: a, count: n * m }`
//!
//! # References
//!
//! - Kairouz, Oh & Viswanath (2015). "The Composition Theorem for Differential
//!   Privacy." ICML 2015.

use std::any::Any;

use crate::error::{PldError, Result};
use crate::functional::pld::PrivacyLossDistribution;
use crate::functional::process::Process;

// =========================================================================
// Internal trait: Flatten (on Repeated only, never on leaf types)
// =========================================================================

/// Construction-time flattening for nested `Repeated` wrappers.
///
/// Implemented only by `Repeated<P>`. Enables [`repeat_flat`] to unwrap
/// nested `Repeated` and multiply counts instead of creating
/// `Repeated<Repeated<P>>`.
///
/// Leaf types (Gaussian etc.) never implement this trait.
pub(crate) trait Flatten: Process + Sized {
    type Leaf: Process;

    /// Unwrap into the leaf process and total accumulated count.
    fn into_leaf_and_count(self) -> (Self::Leaf, usize);
}

impl<P: Process> Flatten for Repeated<P> {
    type Leaf = P;

    fn into_leaf_and_count(self) -> (P, usize) {
        (self.inner, self.count)
    }
}

// =========================================================================
// Internal trait: LeafExtract (nightly only, for case 4 optimization)
// =========================================================================

/// Type-erased leaf extraction for merge optimization.
///
/// Has a blanket impl for all `Process + PartialEq + 'static` types that
/// returns (self.pld(), 1, self_as_any). The specialized override for
/// `Repeated<P>` returns (inner.pld(), count, inner_as_any).
///
/// Requires `#![feature(specialization)]` — gated behind the `nightly` feature.
#[cfg(feature = "nightly")]
pub(crate) trait LeafExtract: Process + 'static {
    /// Compute the leaf PLD (before any self_compose)
    fn leaf_pld(&self) -> Result<PrivacyLossDistribution>;

    /// The accumulated repeat count (1 for non-Repeated types)
    fn leaf_count(&self) -> usize;

    /// The leaf process as `&dyn Any` for equality checking
    fn leaf_any(&self) -> &dyn Any;

    /// Check if this type's leaf equals another leaf (via downcast + PartialEq)
    fn leaf_eq(&self, other: &dyn Any) -> bool;
}

#[cfg(feature = "nightly")]
impl<P: Process + PartialEq + 'static> LeafExtract for P {
    default fn leaf_pld(&self) -> Result<PrivacyLossDistribution> {
        self.pld()
    }

    default fn leaf_count(&self) -> usize {
        1
    }

    default fn leaf_any(&self) -> &dyn Any {
        self
    }

    default fn leaf_eq(&self, other: &dyn Any) -> bool {
        other.downcast_ref::<P>().map_or(false, |o| self == o)
    }
}

#[cfg(feature = "nightly")]
impl<P: Process + PartialEq + 'static> LeafExtract for Repeated<P> {
    fn leaf_pld(&self) -> Result<PrivacyLossDistribution> {
        self.inner.pld()
    }

    fn leaf_count(&self) -> usize {
        self.count
    }

    fn leaf_any(&self) -> &dyn Any {
        &self.inner
    }

    fn leaf_eq(&self, other: &dyn Any) -> bool {
        other
            .downcast_ref::<P>()
            .map_or(false, |o| self.inner == *o)
    }
}

// =========================================================================
// Repeated<P>
// =========================================================================

/// Homogeneous composition: apply the same process `count` times
///
/// Evaluates by computing the inner process's PLD once, then using
/// `self_compose(count)` for efficient FFT-based exponentiation.
///
/// Nested `Repeated` is flattened via [`repeat_flat`] at construction:
/// `repeat_flat(repeat(a, n), m)` produces `Repeated { inner: a, count: n * m }`.
///
/// # Examples
///
/// ```rust,ignore
/// use opaque_dp_accounting::functional::*;
///
/// // 1000 steps of Gaussian mechanism with σ = 1.1
/// let process = repeat(gaussian(1.1), 1000);
/// let epsilon = process.epsilon_at(1e-5)?;
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "P: serde::Serialize",
        deserialize = "P: serde::de::DeserializeOwned",
    ))
)]
pub struct Repeated<P> {
    /// The inner process to repeat
    pub(crate) inner: P,
    /// Number of repetitions (includes flattened nested counts)
    pub(crate) count: usize,
}

impl<P: Process> Process for Repeated<P> {
    fn pld(&self) -> Result<PrivacyLossDistribution> {
        let pld = self.inner.pld()?;
        Ok(pld.self_compose(self.count))
    }
}

// =========================================================================
// Composed<L, R>
// =========================================================================

/// Heterogeneous composition: apply two processes sequentially
///
/// At evaluation time, attempts to merge both sides into a single
/// `self_compose()` call when they share the same underlying process.
/// Falls back to standard PLD convolution when processes differ.
///
/// # Examples
///
/// ```rust,ignore
/// use opaque_dp_accounting::functional::*;
///
/// // Two different noise levels composed
/// let process = compose(gaussian(0.5), gaussian(1.0));
/// let epsilon = process.epsilon_at(1e-5)?;
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "L: serde::Serialize, R: serde::Serialize",
        deserialize = "L: serde::de::DeserializeOwned, R: serde::de::DeserializeOwned",
    ))
)]
pub struct Composed<L, R> {
    /// Left (first) process
    pub(crate) left: L,
    /// Right (second) process
    pub(crate) right: R,
}

impl<L, R> Composed<L, R>
where
    L: Process + PartialEq + 'static,
    R: Process + PartialEq + 'static,
{
    /// Attempt to merge both sides into a single `self_compose()` call.
    ///
    /// Uses `Any` downcast to detect when left and right share the same
    /// underlying process. Cases 1-3 work on stable Rust. Case 4 requires
    /// the `nightly` feature for specialization.
    fn try_merge(&self) -> Result<Option<PrivacyLossDistribution>> {
        // Case 1: Same concrete type and equal value → self_compose(2)
        // Handles: compose(g, g), compose(repeat(g,n), repeat(g,n))
        if let Some(right_as_l) = (&self.right as &dyn Any).downcast_ref::<L>() {
            if self.left == *right_as_l {
                return Ok(Some(self.left.pld()?.self_compose(2)));
            }
        }

        // Case 2: Left is Repeated<R> → check if inner matches right
        // Handles: compose(repeat(g, n), g)
        if let Some(left_rep) = (&self.left as &dyn Any).downcast_ref::<Repeated<R>>() {
            if left_rep.inner == self.right {
                return Ok(Some(self.right.pld()?.self_compose(left_rep.count + 1)));
            }
        }

        // Case 3: Right is Repeated<L> → check if inner matches left
        // Handles: compose(g, repeat(g, n))
        if let Some(right_rep) = (&self.right as &dyn Any).downcast_ref::<Repeated<L>>() {
            if right_rep.inner == self.left {
                return Ok(Some(self.left.pld()?.self_compose(right_rep.count + 1)));
            }
        }

        Ok(None)
    }
}

/// Stable Process impl — uses cases 1-3 from `try_merge`.
#[cfg(not(feature = "nightly"))]
impl<L, R> Process for Composed<L, R>
where
    L: Process + PartialEq + 'static,
    R: Process + PartialEq + 'static,
{
    fn pld(&self) -> Result<PrivacyLossDistribution> {
        if let Some(pld) = self.try_merge()? {
            return Ok(pld);
        }
        self.left.pld()?.compose(&self.right.pld()?)
    }
}

/// Nightly Process impl — uses cases 1-3 plus case 4 via `LeafExtract`.
///
/// Case 4 handles `compose(repeat(a, n), repeat(a, m))` with different counts
/// by extracting the leaf process and summing counts via specialization.
#[cfg(feature = "nightly")]
impl<L, R> Process for Composed<L, R>
where
    L: LeafExtract + PartialEq,
    R: LeafExtract + PartialEq,
{
    fn pld(&self) -> Result<PrivacyLossDistribution> {
        // Try cases 1-3 first (cheaper: no trait dispatch)
        if let Some(pld) = self.try_merge()? {
            return Ok(pld);
        }

        // Case 4: extract leaf + count from both sides via LeafExtract.
        // For bare processes: leaf=self, count=1 (blanket impl).
        // For Repeated<P>: leaf=inner, count=self.count (specialized impl).
        if self.left.leaf_eq(self.right.leaf_any()) {
            let total = self.left.leaf_count() + self.right.leaf_count();
            return Ok(self.left.leaf_pld()?.self_compose(total));
        }

        self.left.pld()?.compose(&self.right.pld()?)
    }
}

// =========================================================================
// Constructors
// =========================================================================

/// Create a repeated (homogeneous) composition
///
/// Applies the same process `count` times. At evaluation time, computes
/// the PLD once and uses efficient FFT-based `self_compose()`.
///
/// For nested repeats, use [`repeat_flat`] to flatten the nesting and
/// multiply counts: `repeat_flat(repeat(a, n), m)` → count `n * m`.
///
/// # Arguments
///
/// * `process` - The process to repeat
/// * `count` - Number of repetitions (must be ≥ 1)
///
/// # Errors
///
/// Returns `PldError::InvalidParameter` if `count` is 0.
///
/// # Examples
///
/// ```rust,ignore
/// use opaque_dp_accounting::functional::*;
///
/// let process = repeat(gaussian(1.1)?, 1000)?;
/// let epsilon = process.epsilon_at(1e-5)?;
/// ```
pub fn repeat<P: Process>(process: P, count: usize) -> Result<Repeated<P>> {
    if count < 1 {
        return Err(PldError::InvalidParameter(format!(
            "repeat count must be at least 1, got {}",
            count
        )));
    }
    Ok(Repeated {
        inner: process,
        count,
    })
}

/// Create a repeated composition, flattening nested `Repeated` wrappers.
///
/// `repeat_flat(repeat(a, n)?, m)?` produces `Repeated { inner: a, count: n * m }`
/// instead of `Repeated<Repeated<P>>`. This is important because flattened
/// `Repeated` values enable the merge optimization in `Composed`.
///
/// # Arguments
///
/// * `process` - A `Repeated` process to flatten and re-repeat
/// * `count` - Additional repetition count (must be ≥ 1)
///
/// # Errors
///
/// Returns `PldError::InvalidParameter` if `count` is 0.
///
/// # Examples
///
/// ```rust,ignore
/// use opaque_dp_accounting::functional::*;
///
/// let inner = repeat(gaussian(1.1)?, 100)?;
/// let process = repeat_flat(inner, 10)?;
/// // Equivalent to repeat(gaussian(1.1)?, 1000)?
/// ```
pub fn repeat_flat<P: Process>(process: Repeated<P>, count: usize) -> Result<Repeated<P>> {
    if count < 1 {
        return Err(PldError::InvalidParameter(format!(
            "repeat count must be at least 1, got {}",
            count
        )));
    }
    let (leaf, inner_count) = process.into_leaf_and_count();
    Ok(Repeated {
        inner: leaf,
        count: inner_count * count,
    })
}

/// Create a composed (heterogeneous) composition
///
/// Applies two processes sequentially. At evaluation time, attempts to
/// merge into `self_compose()` when both processes are the same.
///
/// # Arguments
///
/// * `left` - First process
/// * `right` - Second process
///
/// # Examples
///
/// ```rust,ignore
/// use opaque_dp_accounting::functional::*;
///
/// // Compose same mechanism (optimized to self_compose(2))
/// let g = gaussian(1.1);
/// let process = compose(g.clone(), g);
///
/// // Compose different mechanisms (standard PLD convolution)
/// let process = compose(gaussian(0.5), gaussian(1.0));
/// ```
pub fn compose<L: Process, R: Process>(left: L, right: R) -> Composed<L, R> {
    Composed { left, right }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::functional::mechanisms::gaussian;

    #[test]
    fn test_repeat_basic() {
        let g = gaussian(1.0).unwrap();
        let repeated = repeat(g.clone(), 5).unwrap();
        assert_eq!(repeated.inner, g);
        assert_eq!(repeated.count, 5);
    }

    #[test]
    fn test_repeat_rejects_zero() {
        assert!(repeat(gaussian(1.0).unwrap(), 0).is_err());
    }

    #[test]
    fn test_repeat_flat_flattens() {
        let g = gaussian(1.0).unwrap();
        let inner = repeat(g.clone(), 5).unwrap();
        let outer = repeat_flat(inner, 10).unwrap();
        assert_eq!(outer.inner, g);
        assert_eq!(outer.count, 50);
    }

    #[test]
    fn test_compose_basic() {
        let g1 = gaussian(0.5).unwrap();
        let g2 = gaussian(1.0).unwrap();
        let composed = compose(g1.clone(), g2.clone());
        assert_eq!(composed.left, g1);
        assert_eq!(composed.right, g2);
    }

    #[test]
    fn test_repeated_eq() {
        let r1 = repeat(gaussian(1.0).unwrap(), 5).unwrap();
        let r2 = repeat(gaussian(1.0).unwrap(), 5).unwrap();
        let r3 = repeat(gaussian(1.0).unwrap(), 10).unwrap();
        assert_eq!(r1, r2);
        assert_ne!(r1, r3);
    }

    #[test]
    fn test_composed_eq() {
        let c1 = compose(gaussian(0.5).unwrap(), gaussian(1.0).unwrap());
        let c2 = compose(gaussian(0.5).unwrap(), gaussian(1.0).unwrap());
        let c3 = compose(gaussian(1.0).unwrap(), gaussian(0.5).unwrap());
        assert_eq!(c1, c2);
        assert_ne!(c1, c3);
    }
}
