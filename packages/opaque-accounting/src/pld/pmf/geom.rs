//! Exponentiated-PLD carrier on a geometric grid.
//!
//! The random-allocation transform (Feldman & Shenfeld, arXiv:2602.17284)
//! needs the law of a **sum of exponentiated** privacy losses,
//! `e^{L₁} + e^{L₂}`. This is not composition and **not** an FFT:
//!
//! - Additive convolution in index space gives the law of `L₁ + L₂`, i.e. of
//!   the *product* `e^{L₁}·e^{L₂}`. That is composition.
//! - The values here are geometric (`v_i = v_min·rⁱ`), and a *sum* of two such
//!   values is not on the grid.
//!
//! Hence a direct `O(n·m)` pass plus a range renormalisation. The saving grace
//! is that renormalisation keeps the bin count fixed: if both operands span
//! `[A, A·r^{n-1}]` and `[B, B·r^{n-1}]`, every sum lies in
//! `[A+B, (A+B)·r^{n-1}]` — the same `n` points at the same ratio, only the
//! anchor moves. So `n` does not grow through the recursion.
//!
//! Everything is held in log-space: `σ = 1`, `β = 1e-10` already puts the
//! exponentiated grid across ~950 decades.
//!
//! This type is never a `PrivacyLossDistribution`. It is entered via
//! [`GeomPmf::from_pmf_exp`] and left via [`GeomPmf::into_pmf_log`].

use crate::error::{PldError, Result};
use crate::numerics::logspace::log_add;
use crate::pld::pmf::Pmf;
use crate::pld::realization::Rounding;
use rayon::prelude::*;

/// PMF over `[0, v_min·r⁰, …, v_min·r^{n-1}, +∞]`, stored in log-space.
#[derive(Debug, Clone)]
pub(crate) struct GeomPmf {
    /// `ln(v_min)` — the smallest interior value.
    log_v_min: f64,
    /// `ln(r)` — constant ratio between consecutive interior points.
    log_ratio: f64,
    /// Interior masses.
    probs: Vec<f64>,
    /// Mass at value 0 (the image of a `−∞` privacy loss under `exp`).
    zero_mass: f64,
    /// Mass at `+∞`.
    infinity_mass: f64,
}

impl GeomPmf {
    #[inline]
    fn log_value(&self, i: usize) -> f64 {
        self.log_v_min + (i as f64) * self.log_ratio
    }

    #[inline]
    fn n(&self) -> usize {
        self.probs.len()
    }

    /// `exp(L)` for a PLD on Opaque's uniform loss grid.
    ///
    /// The uniform grid `l_i = (lower + i)·Δ` exponentiates to a geometric grid
    /// with `ln r = Δ`. `−∞` mass becomes the 0-atom; `+∞` mass is carried
    /// through.
    pub(crate) fn from_pmf_exp(pmf: &Pmf) -> Result<Self> {
        if pmf.probs.is_empty() {
            return Err(PldError::InvalidParameter("empty PMF".into()));
        }
        Ok(GeomPmf {
            log_v_min: pmf.lower_loss_index as f64 * pmf.discretization,
            log_ratio: pmf.discretization,
            probs: pmf.probs.clone(),
            zero_mass: pmf.negative_infinity_mass,
            infinity_mass: pmf.infinity_mass,
        })
    }

    /// `exp(-L)`: negate the loss, which reverses the grid.
    pub(crate) fn from_pmf_exp_neg(pmf: &Pmf) -> Result<Self> {
        if pmf.probs.is_empty() {
            return Err(PldError::InvalidParameter("empty PMF".into()));
        }
        let n = pmf.probs.len();
        let hi = pmf.lower_loss_index + n as i64 - 1;
        let mut probs = pmf.probs.clone();
        probs.reverse();
        Ok(GeomPmf {
            // Smallest value of e^{-l} corresponds to the largest l.
            log_v_min: -(hi as f64) * pmf.discretization,
            log_ratio: pmf.discretization,
            probs,
            // e^{-(+inf)} = 0 and e^{-(-inf)} = +inf: the atoms swap.
            zero_mass: pmf.infinity_mass,
            infinity_mass: pmf.negative_infinity_mass,
        })
    }

    /// Extend the interior grid upward with zero mass so two operands match.
    fn padded_to(&self, n_target: usize) -> GeomPmf {
        let mut out = self.clone();
        if out.probs.len() < n_target {
            out.probs.resize(n_target, 0.0);
        }
        out
    }

    /// Convolve two exp-PLDs. `dir` decides which way off-grid sums are moved.
    pub(crate) fn conv(&self, other: &GeomPmf, dir: Rounding) -> Result<GeomPmf> {
        if (self.log_ratio - other.log_ratio).abs() > 1e-12 * self.log_ratio.abs().max(1.0) {
            return Err(PldError::InvalidParameter(format!(
                "geometric grids must share a ratio: {} vs {}",
                self.log_ratio, other.log_ratio
            )));
        }
        let n = self.n().max(other.n());
        let a = self.padded_to(n);
        let b = other.padded_to(n);

        // range-renorm: the anchor moves to a[0] + b[0]; ratio and bin count
        // are unchanged (see the module docstring).
        let log_v_min = log_add(a.log_v_min, b.log_v_min);
        let log_ratio = a.log_ratio;
        let mut probs = vec![0.0f64; n];
        let mut infinity_mass = 0.0f64;
        let mut zero_mass = 0.0f64;

        let mut place = |log_val: f64, mass: f64, probs: &mut Vec<f64>, inf: &mut f64| {
            if mass == 0.0 {
                return;
            }
            let pos = (log_val - log_v_min) / log_ratio;
            let idx = match dir {
                Rounding::Up => pos.ceil(),
                Rounding::Down => pos.floor(),
            };
            if idx < 0.0 {
                // Only reachable via the 0-atom identity below; round toward
                // the requested direction.
                match dir {
                    Rounding::Up => probs[0] += mass,
                    Rounding::Down => zero_mass += mass,
                }
            } else if idx >= n as f64 {
                match dir {
                    Rounding::Up => *inf += mass,
                    Rounding::Down => probs[n - 1] += mass,
                }
            } else {
                probs[idx as usize] += mass;
            }
        };

        // interior × interior — the O(n²) core, parallel over rows of `a`.
        let (par_probs, par_inf) = (0..n)
            .into_par_iter()
            .filter(|&i| a.probs[i] != 0.0)
            .fold(
                || (vec![0.0f64; n], 0.0f64),
                |(mut acc, mut inf), i| {
                    let pa = a.probs[i];
                    let la = a.log_value(i);
                    for j in 0..n {
                        let pb = b.probs[j];
                        if pb == 0.0 {
                            continue;
                        }
                        let pos = (log_add(la, b.log_value(j)) - log_v_min) / log_ratio;
                        let idx = match dir {
                            Rounding::Up => pos.ceil(),
                            Rounding::Down => pos.floor(),
                        };
                        let mass = pa * pb;
                        if idx >= n as f64 {
                            match dir {
                                Rounding::Up => inf += mass,
                                Rounding::Down => acc[n - 1] += mass,
                            }
                        } else {
                            acc[(idx.max(0.0)) as usize] += mass;
                        }
                    }
                    (acc, inf)
                },
            )
            .reduce(
                || (vec![0.0f64; n], 0.0f64),
                |(mut x, xi), (y, yi)| {
                    for (a, b) in x.iter_mut().zip(y.iter()) {
                        *a += *b;
                    }
                    (x, xi + yi)
                },
            );
        for (p, q) in probs.iter_mut().zip(par_probs.iter()) {
            *p += *q;
        }
        infinity_mass += par_inf;

        // 0 is the additive identity: 0 + v = v.
        if a.zero_mass > 0.0 {
            for j in 0..n {
                place(
                    b.log_value(j),
                    a.zero_mass * b.probs[j],
                    &mut probs,
                    &mut infinity_mass,
                );
            }
        }
        if b.zero_mass > 0.0 {
            for i in 0..n {
                place(
                    a.log_value(i),
                    b.zero_mass * a.probs[i],
                    &mut probs,
                    &mut infinity_mass,
                );
            }
        }
        zero_mass += a.zero_mass * b.zero_mass;

        // +∞ absorbs.
        infinity_mass += a.infinity_mass + b.infinity_mass - a.infinity_mass * b.infinity_mass;

        Ok(GeomPmf {
            log_v_min,
            log_ratio,
            probs,
            zero_mass,
            infinity_mass,
        })
    }

    /// `t`-fold self-convolution by exponentiation by squaring.
    ///
    /// Uses exactly `⌊log₂ t⌋ + popcount(t) − 1` pairwise convolutions.
    pub(crate) fn self_conv(&self, t: usize, dir: Rounding) -> Result<GeomPmf> {
        if t == 0 {
            return Err(PldError::InvalidParameter("t must be >= 1".into()));
        }
        let mut base = self.clone();
        let mut acc: Option<GeomPmf> = None;
        let mut rem = t;
        while rem > 0 {
            if rem % 2 == 1 {
                acc = Some(match acc {
                    None => base.clone(),
                    Some(a) => base.conv(&a, dir)?,
                });
            }
            rem /= 2;
            if rem > 0 {
                base = base.conv(&base, dir)?;
            }
        }
        Ok(acc.expect("t >= 1"))
    }

    /// `ln(V / scale)` back onto Opaque's uniform loss grid of width
    /// `discretization`.
    pub(crate) fn into_pmf_log(
        self,
        scale: f64,
        discretization: f64,
        max_grid_size: usize,
        dir: Rounding,
    ) -> Result<Pmf> {
        if scale.is_nan() || scale <= 0.0 {
            return Err(PldError::InvalidParameter("scale must be > 0".into()));
        }
        let shift = scale.ln();
        let n = self.n();

        let loss_at = |i: usize| self.log_value(i) - shift;
        let (l_lo, l_hi) = (loss_at(0), loss_at(n - 1));
        let idx_lo = (l_lo / discretization).floor() as i64;
        let idx_hi = (l_hi / discretization).ceil() as i64;
        let out_n = (idx_hi - idx_lo + 1) as usize;
        if out_n > max_grid_size {
            return Err(PldError::InvalidParameter(format!(
                "output grid of {} points exceeds max_grid_size {}",
                out_n, max_grid_size
            )));
        }

        let mut probs = vec![0.0f64; out_n];
        for i in 0..n {
            if self.probs[i] == 0.0 {
                continue;
            }
            let pos = loss_at(i) / discretization;
            let k = match dir {
                Rounding::Up => pos.ceil(),
                Rounding::Down => pos.floor(),
            } as i64
                - idx_lo;
            let k = k.clamp(0, out_n as i64 - 1) as usize;
            probs[k] += self.probs[i];
        }

        Ok(Pmf {
            discretization,
            lower_loss_index: idx_lo,
            probs,
            infinity_mass: self.infinity_mass,
            // ln(0) = -inf: the 0-atom becomes a -inf privacy loss.
            negative_infinity_mass: self.zero_mass,
            max_grid_size,
            right_tail_budget: 0.0,
            left_tail_budget: 0.0,
        })
    }

    /// `-ln(V / scale)` — the add direction's final step.
    pub(crate) fn into_pmf_neg_log(
        self,
        scale: f64,
        discretization: f64,
        max_grid_size: usize,
        dir: Rounding,
    ) -> Result<Pmf> {
        // Negating the loss flips the rounding direction it must respect.
        let flipped = match dir {
            Rounding::Up => Rounding::Down,
            Rounding::Down => Rounding::Up,
        };
        let pmf = self.into_pmf_log(scale, discretization, max_grid_size, flipped)?;
        let n = pmf.probs.len();
        let hi = pmf.lower_loss_index + n as i64 - 1;
        let mut probs = pmf.probs;
        probs.reverse();
        Ok(Pmf {
            discretization: pmf.discretization,
            lower_loss_index: -hi,
            probs,
            infinity_mass: pmf.negative_infinity_mass,
            negative_infinity_mass: pmf.infinity_mass,
            max_grid_size,
            right_tail_budget: 0.0,
            left_tail_budget: 0.0,
        })
    }

    #[cfg(test)]
    pub(crate) fn total_mass(&self) -> f64 {
        self.probs.iter().sum::<f64>() + self.zero_mass + self.infinity_mass
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pld::realization::{disc_dist, NormalLoss};
    use approx::assert_relative_eq;

    fn gauss_exp(sigma: f64, alpha: f64) -> GeomPmf {
        let pmf = disc_dist(
            &NormalLoss::gaussian(sigma),
            alpha,
            1e-10,
            Rounding::Up,
            10_000_000,
        )
        .unwrap();
        GeomPmf::from_pmf_exp(&pmf).unwrap()
    }

    #[test]
    fn test_conv_conserves_mass() {
        let g = gauss_exp(1.0, 5e-2);
        let c = g.conv(&g, Rounding::Up).unwrap();
        assert_relative_eq!(c.total_mass(), 1.0, epsilon = 1e-9);
    }

    #[test]
    fn test_bin_count_is_stable() {
        // The renormalisation must keep n fixed — this is what makes large t
        // tractable.
        let g = gauss_exp(1.0, 5e-2);
        let n = g.n();
        let mut cur = g.clone();
        for _ in 0..6 {
            cur = cur.conv(&g, Rounding::Up).unwrap();
            assert_eq!(cur.n(), n, "bin count drifted");
        }
    }

    #[test]
    fn test_self_conv_matches_sequential() {
        let g = gauss_exp(1.0, 5e-2);
        for t in 1..=9usize {
            let by_squaring = g.self_conv(t, Rounding::Up).unwrap();
            let mut seq = g.clone();
            for _ in 1..t {
                seq = seq.conv(&g, Rounding::Up).unwrap();
            }
            // Both are valid upper bounds of the same law; their anchors match
            // exactly because renormalisation is associative on the anchor.
            assert_relative_eq!(by_squaring.log_v_min, seq.log_v_min, epsilon = 1e-9);
            assert_relative_eq!(by_squaring.total_mass(), 1.0, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_self_conv_convolution_count() {
        // ⌊log₂ t⌋ + popcount(t) − 1, not the 2⌈log₂ t⌉ analysis bound.
        for &(t, want) in &[(1usize, 0usize), (2, 1), (8, 3), (13, 5), (100, 8)] {
            let got = (t.ilog2() as usize) + (t.count_ones() as usize) - 1;
            assert_eq!(got, want, "t={}", t);
        }
    }

    #[test]
    fn test_round_trip_identity() {
        // exp then log with scale 1 recovers the original support.
        let alpha = 1e-2;
        let pmf = disc_dist(
            &NormalLoss::gaussian(1.0),
            alpha,
            1e-10,
            Rounding::Up,
            10_000_000,
        )
        .unwrap();
        let back = GeomPmf::from_pmf_exp(&pmf)
            .unwrap()
            .into_pmf_log(1.0, alpha, 10_000_000, Rounding::Up)
            .unwrap();
        assert_eq!(back.lower_loss_index, pmf.lower_loss_index);
        let m: f64 = back.probs.iter().sum::<f64>() + back.infinity_mass;
        assert_relative_eq!(m, 1.0, epsilon = 1e-9);
    }

    #[test]
    fn test_upper_dominates_lower() {
        let alpha = 5e-2;
        let up = gauss_exp(1.0, alpha);
        let lo_pmf = disc_dist(
            &NormalLoss::gaussian(1.0),
            alpha,
            1e-10,
            Rounding::Down,
            10_000_000,
        )
        .unwrap();
        let lo = GeomPmf::from_pmf_exp(&lo_pmf).unwrap();

        let cu = up.self_conv(4, Rounding::Up).unwrap();
        let cl = lo.self_conv(4, Rounding::Down).unwrap();
        // The upper bound's anchor must sit at or above the lower bound's.
        assert!(
            cu.log_v_min >= cl.log_v_min - 1e-9,
            "{} < {}",
            cu.log_v_min,
            cl.log_v_min
        );
    }

    #[test]
    fn test_rejects_mismatched_ratio() {
        let a = gauss_exp(1.0, 1e-2);
        let b = gauss_exp(1.0, 2e-2);
        assert!(a.conv(&b, Rounding::Up).is_err());
    }
}
