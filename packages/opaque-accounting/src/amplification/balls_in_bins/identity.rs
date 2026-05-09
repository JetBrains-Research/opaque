//! Identity-specialized BnB Monte Carlo accountant with optional importance sampling.
//!
//! For BnB-Identity (`C = I`), the Lemma 3.2 dominating pair (Choquette-Choo et al.
//! 2024) has orthogonal mode vectors `m_i` with `‖m_i‖² = num_epochs`, so the
//! Gram matrix reduces to `num_epochs · I_b` (diagonal).  After dimension
//! reduction, the mechanism is the **shuffled Gaussian** in `b`-dim space:
//!
//! ```text
//!   Q = N(0, σ_eff² I_b),                σ_eff = σ / √E
//!   P = (1/b) Σ_{i=1}^b N(e_i, σ_eff² I_b)
//! ```
//!
//! By permutation symmetry of `P` over bin index, sampling under `P` is
//! statistically identical to fixing the shifted bin to index 0 and sampling
//! the remaining bins from `Q` (no MC variance is "wasted" on randomising the
//! shifted bin).
//!
//! With `α = 1/(2 σ_eff²) = E/(2 σ²)` and `β = 1/σ_eff = √E/σ`, define the
//! per-bin log-LR terms (in standard-normal `z`-coordinates):
//!
//! ```text
//!   t_0 = α + β · z_0     (shifted bin, mean +α)
//!   t_j = -α + β · z_j    (other bins,  mean -α)     j > 0
//!   Y   = log((1/b) Σ_b exp(t_b))
//! ```
//!
//! All `z_j ~ N(0, 1)` iid.  Cholesky is trivial (diagonal Gram), so the per-
//! sample cost is `O(b)` with no matrix algebra.
//!
//! ## Importance sampling (`importance_tilt = τ > 0`)
//!
//! The hockey-stick estimator `max(1 − exp(ε − Y), 0)` is non-zero only on the
//! large-`Y` tail of `P`.  Tilting `z_0`'s proposal to `N(τ, 1)` puts more
//! samples there and we reweight by the likelihood ratio
//! `w = N(0,1)(z_0) / N(τ,1)(z_0) = exp(−τ z_0 + τ²/2)`.  `τ = 0` reduces to
//! plain identity-specialized MC.
//!
//! Tilt only applies to the "remove" direction (where the shifted bin matters).
//! The "add" direction samples from `Q` (no shift) where all bins are symmetric,
//! and there is no obvious tilt that helps without case analysis.

use crate::discretization::DiscretizationConfig;
use crate::error::{PldError, Result};
use crate::pld::pmf::Pmf;
use crate::pld::PrivacyLossDistribution;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use rayon::prelude::*;

use super::monte_carlo::samples_to_pmf;

/// Sample one (Y, weight) pair under `P` with importance tilt on `z_0`.
///
/// Returns `(Y, w)` where `Y = log((1/k) Σ exp(t_b))` evaluated on the sampled
/// `z` and `w` is the IS reweight (`1.0` when `tilt == 0`).
fn sample_remove_identity(
    k: usize,
    alpha: f64,
    beta: f64,
    tilt: f64,
    rng: &mut impl Rng,
    t_buf: &mut [f64],
) -> (f64, f64) {
    debug_assert_eq!(t_buf.len(), k);

    let z0_centered: f64 = rng.sample::<f64, _>(StandardNormal);
    let z0 = z0_centered + tilt;
    let weight = if tilt == 0.0 {
        1.0
    } else {
        // N(0,1)(z) / N(τ,1)(z) = exp(−τ·z + τ²/2)
        (-tilt * z0 + 0.5 * tilt * tilt).exp()
    };

    let t_0 = alpha + beta * z0;
    t_buf[0] = t_0;
    let mut max_val = t_0;

    for slot in t_buf.iter_mut().skip(1) {
        let zj: f64 = rng.sample::<f64, _>(StandardNormal);
        let tj = -alpha + beta * zj;
        *slot = tj;
        if tj > max_val {
            max_val = tj;
        }
    }

    let mut sum_exp = 0.0;
    for &t in t_buf.iter() {
        sum_exp += (t - max_val).exp();
    }
    let y = max_val + sum_exp.ln() - (k as f64).ln();
    (y, weight)
}

/// Sample one `Y_add = -log L(y)` under `Q` (no shift, all bins symmetric).
fn sample_add_identity(
    k: usize,
    alpha: f64,
    beta: f64,
    rng: &mut impl Rng,
    t_buf: &mut [f64],
) -> f64 {
    debug_assert_eq!(t_buf.len(), k);

    let mut max_val = f64::NEG_INFINITY;
    for slot in t_buf.iter_mut() {
        let zj: f64 = rng.sample::<f64, _>(StandardNormal);
        let tj = -alpha + beta * zj;
        *slot = tj;
        if tj > max_val {
            max_val = tj;
        }
    }
    let mut sum_exp = 0.0;
    for &t in t_buf.iter() {
        sum_exp += (t - max_val).exp();
    }
    let log_l = max_val + sum_exp.ln() - (k as f64).ln();
    -log_l
}

/// Build a discrete PMF from weighted MC samples (unbiased histogram).
///
/// Each sample contributes `w_m / n` mass to its bucket, so the expected
/// total mass is 1 (provided weights have unit mean under the proposal).
fn weighted_samples_to_pmf(
    samples: &[(f64, f64)],
    discretization: f64,
    pessimistic_estimate: bool,
    max_grid_size: usize,
) -> Pmf {
    if samples.is_empty() {
        return Pmf::new(
            discretization,
            0,
            vec![1.0],
            0.0,
            pessimistic_estimate,
            max_grid_size,
        );
    }

    let n = samples.len() as f64;
    let inv_n = 1.0 / n;

    // Grid bounds from finite samples (use a wide-enough envelope).
    let (min_sample, max_sample) = samples
        .iter()
        .filter(|(y, _)| y.is_finite())
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), (y, _)| {
            (lo.min(*y), hi.max(*y))
        });

    if !min_sample.is_finite() || !max_sample.is_finite() {
        // All samples non-finite (edge case).
        let inf_mass: f64 = samples
            .iter()
            .filter(|(y, _)| *y > 0.0)
            .map(|(_, w)| *w)
            .sum::<f64>()
            * inv_n;
        return Pmf::new(
            discretization,
            0,
            vec![0.0],
            inf_mass,
            pessimistic_estimate,
            max_grid_size,
        );
    }

    let grid_lo = (min_sample / discretization).floor() as i64 - 1;
    let grid_hi = (max_sample / discretization).ceil() as i64 + 1;
    let num_buckets = (grid_hi - grid_lo + 1) as usize;

    let effective_grid_size = num_buckets.min(max_grid_size);
    let effective_disc = if num_buckets > max_grid_size {
        (max_sample - min_sample) / (max_grid_size as f64 - 2.0)
    } else {
        discretization
    };
    let effective_lo = if num_buckets > max_grid_size {
        (min_sample / effective_disc).floor() as i64 - 1
    } else {
        grid_lo
    };

    let mut probs = vec![0.0f64; effective_grid_size];
    let mut infinity_mass = 0.0f64;

    for &(y, w) in samples {
        let p = w * inv_n;
        if !y.is_finite() {
            if y > 0.0 {
                infinity_mass += p;
            }
            continue;
        }
        let bucket_idx = if pessimistic_estimate {
            (y / effective_disc).ceil() as i64 - effective_lo
        } else {
            (y / effective_disc).round() as i64 - effective_lo
        };
        if bucket_idx < 0 {
            probs[0] += p;
        } else if bucket_idx >= effective_grid_size as i64 {
            infinity_mass += p;
        } else {
            probs[bucket_idx as usize] += p;
        }
    }

    Pmf::new(
        effective_disc,
        effective_lo,
        probs,
        infinity_mass,
        pessimistic_estimate,
        max_grid_size,
    )
}

/// Compute the BnB-Identity PLD via importance-sampled Monte Carlo.
///
/// # Arguments
///
/// * `num_bins` — Number of bins `b ≥ 2`.
/// * `num_epochs` — Per-bin participation count `E ≥ 1`.
/// * `sigma` — Raw noise multiplier `σ > 0`.
/// * `importance_tilt` — Importance-sampling shift `τ` applied to `z_0` on the
///   "remove" direction.  `0.0` disables IS (plain identity-specialized MC);
///   positive values bias proposals toward large `Y` for tighter tail
///   estimation in typical privacy regimes.
/// * `config` — Discretisation configuration (carries `num_mc_samples`,
///   `seed`, etc.).
pub fn bnb_mc_pld_identity(
    num_bins: usize,
    num_epochs: usize,
    sigma: f64,
    importance_tilt: f64,
    config: &DiscretizationConfig,
) -> Result<PrivacyLossDistribution> {
    if num_bins < 2 {
        return Err(PldError::InvalidParameter(format!(
            "num_bins must be >= 2, got {}",
            num_bins
        )));
    }
    if num_epochs < 1 {
        return Err(PldError::InvalidParameter(format!(
            "num_epochs must be >= 1, got {}",
            num_epochs
        )));
    }
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(PldError::InvalidParameter(format!(
            "sigma must be a positive finite number, got {}",
            sigma
        )));
    }
    if !importance_tilt.is_finite() {
        return Err(PldError::InvalidParameter(format!(
            "importance_tilt must be finite, got {}",
            importance_tilt
        )));
    }
    if config.num_mc_samples == 0 {
        return Err(PldError::InvalidParameter("num_samples must be > 0".into()));
    }

    let k = num_bins;
    let e_f = num_epochs as f64;
    let beta = e_f.sqrt() / sigma;
    let alpha = 0.5 * beta * beta;

    let num_samples = config.num_mc_samples;
    let seed = config.seed;
    let n_threads = rayon::current_num_threads().max(1);
    let samples_per_thread = num_samples / n_threads;
    let remainder = num_samples - samples_per_thread * n_threads;

    // "Remove" direction with optional IS tilt on z_0.
    let remove_samples: Vec<(f64, f64)> = (0..n_threads)
        .into_par_iter()
        .flat_map(|tid| {
            let n = if tid == 0 {
                samples_per_thread + remainder
            } else {
                samples_per_thread
            };
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(tid as u64));
            let mut t_buf = vec![0.0f64; k];
            (0..n)
                .map(|_| {
                    sample_remove_identity(k, alpha, beta, importance_tilt, &mut rng, &mut t_buf)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    // "Add" direction: sample under Q, no IS.
    let add_samples: Vec<f64> = (0..n_threads)
        .into_par_iter()
        .flat_map(|tid| {
            let n = if tid == 0 {
                samples_per_thread + remainder
            } else {
                samples_per_thread
            };
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(1000 + tid as u64));
            let mut t_buf = vec![0.0f64; k];
            (0..n)
                .map(|_| sample_add_identity(k, alpha, beta, &mut rng, &mut t_buf))
                .collect::<Vec<_>>()
        })
        .collect();

    let disc = config.discretization;
    let pessimistic = config.pessimistic_estimate;
    let max_grid = config.max_grid_size;

    let pmf_remove = weighted_samples_to_pmf(&remove_samples, disc, pessimistic, max_grid);
    let pmf_add = samples_to_pmf(&add_samples, disc, pessimistic, max_grid);

    Ok(PrivacyLossDistribution::new_asymmetric(pmf_remove, pmf_add))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(num_samples: usize, seed: u64) -> DiscretizationConfig {
        let mut cfg = DiscretizationConfig::default();
        cfg.num_mc_samples = num_samples;
        cfg.seed = seed;
        cfg
    }

    #[test]
    fn test_rejects_bad_params() {
        let cfg = config(1000, 0);
        assert!(bnb_mc_pld_identity(1, 4, 1.0, 0.0, &cfg).is_err()); // num_bins
        assert!(bnb_mc_pld_identity(10, 0, 1.0, 0.0, &cfg).is_err()); // num_epochs
        assert!(bnb_mc_pld_identity(10, 4, 0.0, 0.0, &cfg).is_err()); // sigma
        assert!(bnb_mc_pld_identity(10, 4, -1.0, 0.0, &cfg).is_err()); // sigma
        assert!(bnb_mc_pld_identity(10, 4, 1.0, f64::NAN, &cfg).is_err()); // tilt
        let zero = config(0, 0);
        assert!(bnb_mc_pld_identity(10, 4, 1.0, 0.0, &zero).is_err()); // samples
    }

    #[test]
    fn test_zero_tilt_matches_generic_bnb_mc() {
        // Identity-specialized MC with τ=0 should agree (within MC noise) with
        // the generic bnb_mc_pld at G = E·I_b for the same RNG seed regime.
        let k = 16usize;
        let e = 4usize;
        let sigma = 1.5_f64;
        let cfg = config(20_000, 42);

        let id_pld = bnb_mc_pld_identity(k, e, sigma, 0.0, &cfg).unwrap();

        // Reference: generic bnb_mc_pld with diagonal Gram.
        let gram: Vec<f64> = (0..k * k)
            .map(|idx| if idx / k == idx % k { e as f64 } else { 0.0 })
            .collect();
        let ref_pld = crate::amplification::bnb_mc_pld(&gram, k, sigma, &cfg).unwrap();

        let eps_id = id_pld.epsilon_at(1e-5);
        let eps_ref = ref_pld.epsilon_at(1e-5);
        // Both Monte Carlo with independent random streams: agreement within
        // ~25% at 20k samples in this regime is plenty to confirm we're
        // computing the same dominating-pair PLD.
        assert!(
            (eps_id - eps_ref).abs() < 0.25 * eps_ref.abs() + 0.5,
            "identity-specialised MC should match generic at same params: id={}, ref={}",
            eps_id,
            eps_ref
        );
    }

    #[test]
    fn test_more_noise_lower_epsilon() {
        let cfg = config(20_000, 7);
        let eps_lo = bnb_mc_pld_identity(16, 4, 0.5, 0.0, &cfg)
            .unwrap()
            .epsilon_at(1e-5);
        let eps_hi = bnb_mc_pld_identity(16, 4, 2.0, 0.0, &cfg)
            .unwrap()
            .epsilon_at(1e-5);
        assert!(
            eps_hi < eps_lo,
            "more noise should lower ε: lo={}, hi={}",
            eps_lo,
            eps_hi
        );
    }

    #[test]
    fn test_importance_tilt_does_not_blow_up() {
        // With moderate IS tilt, ε should remain finite and close to the τ=0 value.
        let cfg = config(20_000, 11);
        let eps_no_tilt = bnb_mc_pld_identity(16, 4, 1.5, 0.0, &cfg)
            .unwrap()
            .epsilon_at(1e-5);
        let eps_tilt = bnb_mc_pld_identity(16, 4, 1.5, 1.0, &cfg)
            .unwrap()
            .epsilon_at(1e-5);
        assert!(eps_no_tilt.is_finite());
        assert!(eps_tilt.is_finite());
        // The tilted estimator should agree at the same sample count up to MC noise.
        assert!(
            (eps_no_tilt - eps_tilt).abs() < 0.15 * eps_no_tilt.abs() + 0.1,
            "moderate IS tilt should give consistent ε: τ=0 → {}, τ=1 → {}",
            eps_no_tilt,
            eps_tilt
        );
    }
}
