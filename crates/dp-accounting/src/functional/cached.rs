//! Opt-in PLD caching wrapper
//!
//! [`Cached<P>`] computes the inner process's PLD on first `pld()` call and
//! caches the result. Clones share the same cache via `Arc<OnceLock<...>>`.
//!
//! This is useful when the same process is composed many times (e.g., an
//! accounting loop that repeatedly composes the same step), avoiding redundant
//! PLD recomputation.
//!
//! # Examples
//!
//! ```rust,ignore
//! use opaque_dp_accounting::functional::*;
//!
//! let step = cached(gaussian(1.1));
//!
//! // First call computes the PLD, subsequent calls return the cached result
//! let eps1 = step.epsilon_at(1e-5)?;
//! let eps2 = step.epsilon_at(1e-5)?; // cached — no recomputation
//!
//! // Clones share the cache
//! let step2 = step.clone();
//! let eps3 = step2.epsilon_at(1e-5)?; // still cached
//! ```

use std::fmt;
use std::sync::{Arc, OnceLock};

use crate::error::Result;
use crate::functional::pld::PrivacyLossDistribution;
use crate::functional::process::Process;

/// Opt-in PLD caching wrapper
///
/// Computes the inner process's PLD on first `pld()` call and caches the
/// result (including errors). Clones share the same cache via `Arc<OnceLock<...>>`.
///
/// # Caching semantics
///
/// - The cache stores `Result<PrivacyLossDistribution>`, so errors are also cached.
///   This is correct because the same parameters always produce the same error.
/// - `PartialEq` and `Eq` delegate to the inner process only — cache state is
///   not part of the semantic identity.
pub struct Cached<P> {
    inner: P,
    pld_cache: Arc<OnceLock<Result<PrivacyLossDistribution>>>,
}

impl<P: Process> Process for Cached<P> {
    fn pld(&self) -> Result<PrivacyLossDistribution> {
        self.pld_cache.get_or_init(|| self.inner.pld()).clone()
    }
}

impl<P: Clone> Clone for Cached<P> {
    fn clone(&self) -> Self {
        Cached {
            inner: self.inner.clone(),
            pld_cache: Arc::clone(&self.pld_cache),
        }
    }
}

impl<P: fmt::Debug> fmt::Debug for Cached<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Cached")
            .field("inner", &self.inner)
            .field("cached", &self.pld_cache.get().is_some())
            .finish()
    }
}

impl<P: PartialEq> PartialEq for Cached<P> {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<P: Eq> Eq for Cached<P> {}

// ---------------------------------------------------------------------------
// Serde support (feature-gated)
// ---------------------------------------------------------------------------

/// Serializes only the inner process — the PLD cache is transient and not
/// included. Deserializing produces a `Cached<P>` with an empty cache.
#[cfg(feature = "serde")]
impl<P: serde::Serialize> serde::Serialize for Cached<P> {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        self.inner.serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, P: serde::Deserialize<'de>> serde::Deserialize<'de> for Cached<P> {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let inner = P::deserialize(deserializer)?;
        Ok(Cached {
            inner,
            pld_cache: Arc::new(OnceLock::new()),
        })
    }
}

/// Wrap a process in an opt-in PLD cache
///
/// The returned `Cached<P>` computes the PLD on first access and caches
/// the result. Clones share the same cache via `Arc`.
///
/// # Arguments
///
/// * `process` - The process to cache
///
/// # Examples
///
/// ```rust,ignore
/// use opaque_dp_accounting::functional::*;
///
/// let step = cached(gaussian(1.1));
/// let eps = step.epsilon_at(1e-5)?; // computes PLD
/// let eps2 = step.epsilon_at(1e-5)?; // returns cached result
/// ```
pub fn cached<P: Process>(process: P) -> Cached<P> {
    Cached {
        inner: process,
        pld_cache: Arc::new(OnceLock::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::functional::mechanisms::gaussian;

    #[test]
    fn test_cache_hit() {
        let c = cached(gaussian(1.0).unwrap());

        // First call populates cache
        let eps1 = c.epsilon_at(1e-5).unwrap();

        // Second call returns same result from cache
        let eps2 = c.epsilon_at(1e-5).unwrap();

        assert_eq!(eps1, eps2, "cached results should be identical");
        assert!(c.pld_cache.get().is_some(), "cache should be populated");
    }

    #[test]
    fn test_clone_shares_cache() {
        let c = cached(gaussian(1.0).unwrap());

        // Populate cache via original
        let _ = c.pld().unwrap();
        assert!(c.pld_cache.get().is_some());

        // Clone shares the same Arc
        let c2 = c.clone();
        assert!(c2.pld_cache.get().is_some(), "clone should see cached PLD");
        assert!(
            Arc::ptr_eq(&c.pld_cache, &c2.pld_cache),
            "clone should share the same Arc"
        );
    }

    #[test]
    fn test_partial_eq_ignores_cache() {
        let a = cached(gaussian(1.0).unwrap());
        let b = cached(gaussian(1.0).unwrap());

        // Both equal despite neither being cached
        assert_eq!(a, b);

        // Populate a's cache
        let _ = a.pld().unwrap();

        // Still equal even though a is cached and b is not
        assert_eq!(a, b);
    }

    #[test]
    fn test_debug_shows_cache_state() {
        let c = cached(gaussian(1.0).unwrap());

        let debug_before = format!("{:?}", c);
        assert!(
            debug_before.contains("cached: false"),
            "should show uncached: {}",
            debug_before
        );

        let _ = c.pld().unwrap();

        let debug_after = format!("{:?}", c);
        assert!(
            debug_after.contains("cached: true"),
            "should show cached: {}",
            debug_after
        );
    }
}
