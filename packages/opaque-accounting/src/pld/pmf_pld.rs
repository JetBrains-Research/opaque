//! PMF-based Privacy Loss Distribution.
//!
//! `PmfPld` wraps one or two `Pmf` objects (for symmetric and asymmetric
//! mechanisms) and provides composition via FFT convolution.
//!
//! This is the original `PrivacyLossDistribution` struct, extracted to
//! live alongside `CgfPld` under the `PrivacyLossDistribution` enum.

use super::pmf::Pmf;
use crate::error::Result;

/// PMF-based privacy loss distribution with adjacency support.
///
/// Wraps one or two `Pmf` objects to support differential privacy mechanisms
/// that may have different privacy loss distributions depending on the adjacency
/// type (whether a dataset has one more or one fewer element).
#[derive(Debug, Clone)]
pub struct PmfPld {
    /// PLD for REMOVE adjacency (D has fewer elements than D')
    pub(crate) pmf_remove: Pmf,

    /// PLD for ADD adjacency (D has more elements than D')
    ///
    /// If `None`, this is a symmetric mechanism and `pmf_remove` is used for both.
    pub(crate) pmf_add: Option<Pmf>,
}

impl PmfPld {
    /// Create a symmetric PmfPld (same PLD for both adjacencies).
    pub(crate) fn new_symmetric(pmf: Pmf) -> Self {
        Self {
            pmf_remove: pmf,
            pmf_add: None,
        }
    }

    /// Create an asymmetric PmfPld (different PLDs for ADD and REMOVE).
    pub(crate) fn new_asymmetric(pmf_remove: Pmf, pmf_add: Pmf) -> Self {
        Self {
            pmf_remove,
            pmf_add: Some(pmf_add),
        }
    }

    /// Check if this PLD is symmetric.
    pub fn is_symmetric(&self) -> bool {
        self.pmf_add.is_none()
    }

    /// Set Chernoff tail budgets on all contained PMFs.
    pub fn with_tail_budgets(mut self, right: f64, left: f64) -> Self {
        self.pmf_remove.right_tail_budget = right;
        self.pmf_remove.left_tail_budget = left;
        if let Some(ref mut pmf_add) = self.pmf_add {
            pmf_add.right_tail_budget = right;
            pmf_add.left_tail_budget = left;
        }
        self
    }

    /// Override the max grid size on all contained PMFs.
    pub fn with_max_grid_size(&self, max_grid_size: usize) -> Self {
        Self {
            pmf_remove: self.pmf_remove.with_max_grid_size(max_grid_size),
            pmf_add: self
                .pmf_add
                .as_ref()
                .map(|p| p.with_max_grid_size(max_grid_size)),
        }
    }

    // -- Composition --------------------------------------------------------

    /// Compose two PmfPlds via FFT convolution.
    pub fn compose(&self, other: &Self) -> Result<Self> {
        let pmf_remove = self
            .pmf_remove
            .clone()
            .compose(other.pmf_remove.clone(), 0.0)?;

        let pmf_add = match (&self.pmf_add, &other.pmf_add) {
            (None, None) => None,
            (None, Some(other_add)) => {
                let composed = self.pmf_remove.clone().compose(other_add.clone(), 0.0)?;
                Some(composed)
            }
            (Some(self_add), None) => {
                let composed = self_add.clone().compose(other.pmf_remove.clone(), 0.0)?;
                Some(composed)
            }
            (Some(self_add), Some(other_add)) => {
                let composed = self_add.clone().compose(other_add.clone(), 0.0)?;
                Some(composed)
            }
        };

        Ok(Self {
            pmf_remove,
            pmf_add,
        })
    }

    /// Self-compose via FFT power method.
    pub fn self_compose(&self, count: usize) -> Self {
        let pmf_remove = self.pmf_remove.clone().self_compose(count);
        let pmf_add = self
            .pmf_add
            .as_ref()
            .map(|pmf| pmf.clone().self_compose(count));
        Self {
            pmf_remove,
            pmf_add,
        }
    }

    /// Compose with explicit max_grid_size override.
    pub fn compose_with_max_grid_size(&self, other: &Self, max_grid_size: usize) -> Result<Self> {
        let lhs = self.with_max_grid_size(max_grid_size);
        let rhs = other.with_max_grid_size(max_grid_size);
        lhs.compose(&rhs)
    }

    /// Self-compose with explicit max_grid_size override.
    pub fn self_compose_with_max_grid_size(&self, count: usize, max_grid_size: usize) -> Self {
        let pmf_remove = self
            .pmf_remove
            .clone()
            .self_compose_with_max_grid_size(count, max_grid_size);
        let pmf_add = self.pmf_add.as_ref().map(|pmf| {
            pmf.clone()
                .self_compose_with_max_grid_size(count, max_grid_size)
        });
        Self {
            pmf_remove,
            pmf_add,
        }
    }
}
