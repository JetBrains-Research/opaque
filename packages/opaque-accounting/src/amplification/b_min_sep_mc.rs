//! Monte Carlo privacy accounting for BandMF with warm-start b-min-sep subsampling.
//!
//! Implements the likelihood-ratio dynamic program from Dong & Ganesh (2026),
//! "Privacy Amplification for BandMF via b-Min-Sep Subsampling" (arXiv:2602.09338),
//! Section 5, Equation (2), and the warm-start correction after Theorem 5.1:
//!
//! `P(y)/Q(y) = f_1(y) + p * sum_{i=2}^{b} f_i(y) / (1 + (b-1)p)`
//!
//! where `f_i` is the backward recursion on iteration index (1-based in the paper).
//! Privacy loss samples: `L_remove = ln(P/Q)`, `L_add = -ln(P/Q)` with `y ~ Q`,
//! matching the asymmetric Monte Carlo pattern used by [`super::balls_in_bins::monte_carlo::bnb_mc_pld`].

use crate::amplification::balls_in_bins::monte_carlo::samples_to_pmf;
use crate::discretization::DiscretizationConfig;
use crate::error::{PldError, Result};
use crate::pld::PrivacyLossDistribution;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use rayon::prelude::*;

/// Log-density ratio log N(μ, σ²I)(y) - log N(0, σ²I)(y) for diagonal-covariance isotropic Gaussian.
#[inline]
fn log_gaussian_ratio_block(mu: &[f64], y: &[f64], sigma2: f64) -> f64 {
    let mut dot = 0.0;
    let mut norm_sq = 0.0;
    for k in 0..mu.len().min(y.len()) {
        dot += mu[k] * y[k];
        norm_sq += mu[k] * mu[k];
    }
    (dot - 0.5 * norm_sq) / sigma2
}

/// Column `col` of lower-triangular Toeplitz with first column `coef` (length `bands`), padded to length `bands`.
fn column_mu(coef: &[f64], n: usize, col: usize, bands: usize) -> Vec<f64> {
    let mut mu = vec![0.0f64; bands];
    for k in 0..bands {
        if col + k < n && k < coef.len() {
            mu[k] = coef[k];
        }
    }
    mu
}

/// `P(y)/Q(y)` for warm-start b-min-sep (paper notation after Theorem 5.1).
fn likelihood_ratio_warm(
    y: &[f64],
    coef: &[f64],
    n: usize,
    bands: usize,
    p: f64,
    sigma2: f64,
) -> f64 {
    if bands < 1 || coef.is_empty() {
        return 1.0;
    }
    let mut f = vec![0.0f64; n + 1];
    f[n] = 1.0;
    for i in (0..n).rev() {
        let mu = column_mu(coef, n, i, bands);
        let hi = (i + bands).min(n);
        let yblk = &y[i..hi];
        let log_block = log_gaussian_ratio_block(&mu[..yblk.len()], yblk, sigma2);
        let block_ratio = log_block.exp();
        let f_skip = if i + bands <= n { f[i + bands] } else { 1.0 };
        f[i] = (1.0 - p) * f[i + 1] + p * block_ratio * f_skip;
    }
    let denom = 1.0 + (bands.saturating_sub(1) as f64) * p;
    let mut sum_f = f[0];
    if bands > 1 {
        let tail: f64 = (1..bands).map(|j| f[j]).sum::<f64>();
        sum_f += p * tail / denom;
    }
    sum_f
}

fn sample_y_under_q(n: usize, sigma: f64, rng: &mut impl Rng, buf: &mut [f64]) {
    for v in buf.iter_mut().take(n) {
        *v = rng.sample::<f64, _>(StandardNormal) * sigma;
    }
}

/// Sample `y = Cx + z` under `P` (single distinguished example, warm-start b-min-sep on participation).
fn sample_y_under_p(
    coef: &[f64],
    n: usize,
    bands: usize,
    p: f64,
    sigma: f64,
    rng: &mut impl Rng,
    x_buf: &mut [f64],
    z_buf: &mut [f64],
    y_buf: &mut [f64],
) {
    x_buf.fill(0.0);
    // Warm-start initial state for the distinguished example (Algorithm 2, lines 1–2).
    let mut barred_remaining: usize = if bands <= 1 {
        0
    } else {
        let denom = 1.0 + (bands - 1) as f64 * p;
        if rng.gen::<f64>() * denom < 1.0 {
            0
        } else {
            1 + rng.gen_range(0..bands - 1)
        }
    };

    for i in 0..n {
        let eligible = barred_remaining == 0;
        let participate = eligible && rng.gen::<f64>() < p;
        x_buf[i] = if participate { 1.0 } else { 0.0 };
        if participate {
            // Skip the next (b-1) iterations (Algorithm 1).
            barred_remaining = bands.saturating_sub(1);
        } else if barred_remaining > 0 {
            barred_remaining -= 1;
        }
    }

    for v in z_buf.iter_mut().take(n) {
        *v = rng.sample::<f64, _>(StandardNormal) * sigma;
    }

    // y = C x + z, C lower-triangular Toeplitz(first column = coef padded).
    for i in 0..n {
        let mut acc = z_buf[i];
        let j0 = i.saturating_sub(coef.len().saturating_sub(1));
        for j in j0..=i {
            let k = i - j;
            if k < coef.len() {
                acc += coef[k] * x_buf[j];
            }
        }
        y_buf[i] = acc;
    }
}

/// Monte Carlo PLD for BandMF + warm-start b-min-sep subsampling (single-example adjacent analysis).
///
/// * `strategy_coef` — first column of strategy matrix `C` (length = bandwidth).
/// * `n_steps` — number of iterations `n`.
/// * `p` — per-iteration inclusion probability `p` in Algorithm 2 (not `p_0`).
/// * `sigma` — raw noise multiplier σ (same units as [`super::bnb_mc_pld`]).
pub fn bandmf_b_min_sep_warm_mc_pld(
    strategy_coef: &[f64],
    n_steps: usize,
    p: f64,
    sigma: f64,
    num_samples: usize,
    seed: u64,
    config: &DiscretizationConfig,
) -> Result<PrivacyLossDistribution> {
    if strategy_coef.is_empty() {
        return Err(PldError::InvalidParameter(
            "strategy_coef must be non-empty".into(),
        ));
    }
    if n_steps == 0 {
        return Err(PldError::InvalidParameter(
            "n_steps must be > 0".into(),
        ));
    }
    if !(p > 0.0 && p <= 1.0) {
        return Err(PldError::InvalidParameter(format!(
            "p must be in (0, 1], got {}",
            p
        )));
    }
    if sigma <= 0.0 {
        return Err(PldError::InvalidParameter(format!(
            "sigma must be > 0, got {}",
            sigma
        )));
    }
    if num_samples == 0 {
        return Err(PldError::InvalidParameter("num_samples must be > 0".into()));
    }

    let bands = strategy_coef.len();
    let sigma2 = sigma * sigma;
    let disc = config.discretization;
    let pessimistic = config.pessimistic_estimate;

    let n_threads = rayon::current_num_threads().max(1);
    let samples_per_thread = num_samples / n_threads;
    let remainder = num_samples - samples_per_thread * n_threads;

    let coef = strategy_coef.to_vec();

    let remove_samples: Vec<f64> = (0..n_threads)
        .into_par_iter()
        .flat_map(|tid| {
            let n_samp = if tid == 0 {
                samples_per_thread + remainder
            } else {
                samples_per_thread
            };
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(tid as u64));
            let mut xb = vec![0.0f64; n_steps];
            let mut zb = vec![0.0f64; n_steps];
            let mut yb = vec![0.0f64; n_steps];
            (0..n_samp)
                .map(|_| {
                    sample_y_under_p(
                        &coef,
                        n_steps,
                        bands,
                        p,
                        sigma,
                        &mut rng,
                        &mut xb,
                        &mut zb,
                        &mut yb,
                    );
                    let r = likelihood_ratio_warm(&yb, &coef, n_steps, bands, p, sigma2);
                    r.ln()
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let add_samples: Vec<f64> = (0..n_threads)
        .into_par_iter()
        .flat_map(|tid| {
            let n_samp = if tid == 0 {
                samples_per_thread + remainder
            } else {
                samples_per_thread
            };
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(1000 + tid as u64));
            let mut yb = vec![0.0f64; n_steps];
            (0..n_samp)
                .map(|_| {
                    sample_y_under_q(n_steps, sigma, &mut rng, &mut yb);
                    let r = likelihood_ratio_warm(&yb, &coef, n_steps, bands, p, sigma2);
                    -(r.ln())
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let pmf_remove = samples_to_pmf(&remove_samples, disc, pessimistic, config.max_grid_size);
    let pmf_add = samples_to_pmf(&add_samples, disc, pessimistic, config.max_grid_size);

    Ok(PrivacyLossDistribution::new_asymmetric(
        pmf_remove,
        pmf_add,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discretization::DiscretizationConfig;

    fn default_config() -> DiscretizationConfig {
        DiscretizationConfig::default()
    }

    #[test]
    fn likelihood_ratio_positive_finite() {
        let coef = vec![1.0, 0.0];
        let n = 20;
        let y = vec![0.1; n];
        let r = likelihood_ratio_warm(&y, &coef, n, 2, 0.05, 1.0);
        assert!(r.is_finite() && r > 0.0);
    }

    #[test]
    fn mc_pld_smoke() {
        let coef = vec![0.7_f64.sqrt(), 0.3_f64.sqrt()];
        let cfg = default_config();
        let pld = bandmf_b_min_sep_warm_mc_pld(&coef, 50, 0.05, 1.0, 5000, 42, &cfg).unwrap();
        let eps = pld.epsilon_at(1e-3);
        assert!(eps > 0.0 && eps.is_finite());
    }
}
