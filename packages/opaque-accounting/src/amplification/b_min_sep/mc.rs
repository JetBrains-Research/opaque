//! Monte Carlo privacy accounting for BandMF with warm-start b-min-sep subsampling.
//!
//! Implements the likelihood-ratio dynamic program from Dong & Ganesh (2026),
//! "Privacy Amplification for BandMF via b-Min-Sep Subsampling" (arXiv:2602.09338),
//! Section 5, Equation (1), and the warm-start corollary after Theorem 5.1:
//!
//! `P(y)/Q(y) = (f_1(y) + p * sum_{i=2}^{b} f_i(y)) / (1 + (b-1)p)`
//!
//! where `f_i` is the backward recursion on iteration index (1-based in the paper).
//! Remove samples use `y ~ P` and `ln(P/Q)`; add samples use `y ~ Q` and `ln(Q/P)`.

use crate::amplification::balls_in_bins::monte_carlo::samples_to_pmf;
use crate::discretization::DiscretizationConfig;
use crate::error::{PldError, Result};
use crate::numerics::logspace::{log_add, log_sumexp};
use crate::pld::PrivacyLossDistribution;

use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};
use rand_distr::StandardNormal;
use rayon::prelude::*;

/// `log N(μ, σ²I)(y) - log N(0, σ²I)(y)`.
#[inline]
fn log_gaussian_ratio_block(mu: &[f64], y: &[f64], sigma2: f64) -> f64 {
    let mut dot = 0.0;
    let mut norm_sq = 0.0;
    for (&mean, &value) in mu.iter().zip(y) {
        dot += mean * value;
        norm_sq += mean * mean;
    }
    (dot - 0.5 * norm_sq) / sigma2
}

/// Evaluates warm-start b-min-sep `ln(P(y)/Q(y))` with reusable storage.
struct WarmLogLikelihoodRatio<'a> {
    coef: &'a [f64],
    sigma2: f64,
    log_p: f64,
    log_one_minus_p: f64,
    log_denominator: f64,
    log_f: Vec<f64>,
}

impl<'a> WarmLogLikelihoodRatio<'a> {
    fn new(coef: &'a [f64], p: f64, sigma2: f64) -> Self {
        Self {
            coef,
            sigma2,
            log_p: p.ln(),
            log_one_minus_p: (-p).ln_1p(),
            log_denominator: ((coef.len().saturating_sub(1)) as f64 * p).ln_1p(),
            log_f: Vec::new(),
        }
    }

    fn evaluate(&mut self, y: &[f64]) -> f64 {
        let n = y.len();
        let bands = self.coef.len();
        if bands == 0 {
            return 0.0;
        }

        self.log_f.resize(n + bands, 0.0);
        // Boundary: log f_i = 0 for i >= n.
        self.log_f[n..].fill(0.0);

        for i in (0..n).rev() {
            let block_len = bands.min(n - i);
            let log_block = log_gaussian_ratio_block(
                &self.coef[..block_len],
                &y[i..i + block_len],
                self.sigma2,
            );
            self.log_f[i] = log_add(
                self.log_one_minus_p + self.log_f[i + 1],
                self.log_p + log_block + self.log_f[i + bands],
            );
        }

        let log_numerator = log_add(
            self.log_f[0],
            self.log_p + log_sumexp(&self.log_f[1..bands]),
        );
        log_numerator - self.log_denominator
    }
}

fn sample_y_under_q(n: usize, sigma: f64, rng: &mut impl Rng, buf: &mut [f64]) {
    for v in buf.iter_mut().take(n) {
        *v = rng.sample::<f64, _>(StandardNormal) * sigma;
    }
}

/// Draw participation `x` under warm-start b-min-sep at fixed `p` (noise independent).
fn sample_x_under_p(n: usize, bands: usize, p: f64, rng: &mut impl Rng, x_buf: &mut [f64]) {
    x_buf.fill(0.0);
    let mut barred_remaining: usize = if bands <= 1 {
        0
    } else {
        let denom = 1.0 + (bands - 1) as f64 * p;
        if rng.random::<f64>() * denom < 1.0 {
            0
        } else {
            1 + rng.random_range(0..bands - 1)
        }
    };

    for xb in x_buf.iter_mut().take(n) {
        let eligible = barred_remaining == 0;
        let participate = eligible && rng.random::<f64>() < p;
        *xb = if participate { 1.0 } else { 0.0 };
        if participate {
            barred_remaining = bands.saturating_sub(1);
        } else {
            barred_remaining = barred_remaining.saturating_sub(1);
        }
    }
}

/// `y = Cx + σ ζ` with standard normal `ζ` (column `i` of `C` applied to `x`).
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn y_from_x_and_zeta(
    coef: &[f64],
    n: usize,
    x: &[f64],
    zeta: &[f64],
    sigma: f64,
    y_out: &mut [f64],
) {
    for i in 0..n {
        let mut acc = sigma * zeta[i];
        let j0 = i.saturating_sub(coef.len().saturating_sub(1));
        for j in j0..=i {
            let k = i - j;
            if k < coef.len() {
                acc += coef[k] * x[j];
            }
        }
        y_out[i] = acc;
    }
}

/// Standard-normal draws for the Q branch (`y = σ η`).
fn sample_eta_under_q(n: usize, rng: &mut impl Rng, buf: &mut [f64]) {
    for v in buf.iter_mut().take(n) {
        *v = rng.sample::<f64, _>(StandardNormal);
    }
}

/// Prepare Monte Carlo transcripts for reuse across many `σ` values (calibration).
///
/// Returns flattened row-major arrays of length `num_samples * n_steps`:
/// - `remove_x`, `remove_zeta` for the P-branch (`y = Cx + σ ζ`)
/// - `add_eta` for the Q-branch (`y = σ η`)
///
/// Sample order matches [`bandmf_b_min_sep_warm_mc_pld`]: thread 0 chunk, then
/// thread 1, … with the same per-thread `StdRng` seeds (`seed+tid` and `1000+tid`).
pub fn bandmf_b_min_sep_prepare_transcripts(
    strategy_coef: &[f64],
    n_steps: usize,
    p: f64,
    num_samples: usize,
    seed: u64,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    if strategy_coef.is_empty() {
        return Err(PldError::InvalidParameter(
            "strategy_coef must be non-empty".into(),
        ));
    }
    if n_steps == 0 {
        return Err(PldError::InvalidParameter("n_steps must be > 0".into()));
    }
    if !(p > 0.0 && p <= 1.0) {
        return Err(PldError::InvalidParameter(format!(
            "p must be in (0, 1], got {}",
            p
        )));
    }
    if num_samples == 0 {
        return Err(PldError::InvalidParameter("num_samples must be > 0".into()));
    }

    let bands = strategy_coef.len();
    let n_threads = rayon::current_num_threads().max(1);
    let samples_per_thread = num_samples / n_threads;
    let remainder = num_samples - samples_per_thread * n_threads;

    let mut remove_x = vec![0.0f64; num_samples * n_steps];
    let mut remove_zeta = vec![0.0f64; num_samples * n_steps];
    let mut idx = 0usize;
    for tid in 0..n_threads {
        let n_samp = if tid == 0 {
            samples_per_thread + remainder
        } else {
            samples_per_thread
        };
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(tid as u64));
        let mut xb = vec![0.0f64; n_steps];
        let mut zb = vec![0.0f64; n_steps];
        for _ in 0..n_samp {
            sample_x_under_p(n_steps, bands, p, &mut rng, &mut xb);
            for v in zb.iter_mut() {
                *v = rng.sample::<f64, _>(StandardNormal);
            }
            remove_x[idx * n_steps..(idx + 1) * n_steps].copy_from_slice(&xb);
            remove_zeta[idx * n_steps..(idx + 1) * n_steps].copy_from_slice(&zb);
            idx += 1;
        }
    }
    debug_assert_eq!(idx, num_samples);

    let mut add_eta = vec![0.0f64; num_samples * n_steps];
    idx = 0;
    for tid in 0..n_threads {
        let n_samp = if tid == 0 {
            samples_per_thread + remainder
        } else {
            samples_per_thread
        };
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(1000 + tid as u64));
        let mut eb = vec![0.0f64; n_steps];
        for _ in 0..n_samp {
            sample_eta_under_q(n_steps, &mut rng, &mut eb);
            add_eta[idx * n_steps..(idx + 1) * n_steps].copy_from_slice(&eb);
            idx += 1;
        }
    }
    debug_assert_eq!(idx, num_samples);

    Ok((remove_x, remove_zeta, add_eta))
}

/// Build PLD from precomputed transcripts at noise multiplier `sigma`.
#[allow(clippy::too_many_arguments)]
pub fn bandmf_b_min_sep_pld_from_transcripts(
    remove_x: &[f64],
    remove_zeta: &[f64],
    add_eta: &[f64],
    strategy_coef: &[f64],
    n_steps: usize,
    p: f64,
    sigma: f64,
    config: &DiscretizationConfig,
) -> Result<PrivacyLossDistribution> {
    let num_samples = add_eta.len() / n_steps;
    if num_samples == 0 || add_eta.len() != num_samples * n_steps {
        return Err(PldError::InvalidParameter(
            "add_eta length must be positive multiple of n_steps".into(),
        ));
    }
    if remove_x.len() != add_eta.len() || remove_zeta.len() != add_eta.len() {
        return Err(PldError::InvalidParameter(
            "remove_x/remove_zeta/add_eta length mismatch".into(),
        ));
    }
    if sigma <= 0.0 {
        return Err(PldError::InvalidParameter(format!(
            "sigma must be > 0, got {}",
            sigma
        )));
    }
    let sigma2 = sigma * sigma;
    let coef = strategy_coef.to_vec();
    let n_threads = rayon::current_num_threads().max(1);
    let samples_per_thread = num_samples / n_threads;
    let remainder = num_samples - samples_per_thread * n_threads;

    let mut remove_samples: Vec<f64> = (0..n_threads)
        .into_par_iter()
        .flat_map(|tid| {
            let n_samp = if tid == 0 {
                samples_per_thread + remainder
            } else {
                samples_per_thread
            };
            let base = if tid == 0 {
                0
            } else {
                remainder + tid * samples_per_thread
            };
            let mut yb = vec![0.0f64; n_steps];
            let mut evaluator = WarmLogLikelihoodRatio::new(&coef, p, sigma2);
            (0..n_samp)
                .map(|j| {
                    let s = base + j;
                    let x = &remove_x[s * n_steps..(s + 1) * n_steps];
                    let z = &remove_zeta[s * n_steps..(s + 1) * n_steps];
                    y_from_x_and_zeta(&coef, n_steps, x, z, sigma, &mut yb);
                    evaluator.evaluate(&yb)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let mut add_samples: Vec<f64> = (0..n_threads)
        .into_par_iter()
        .flat_map(|tid| {
            let n_samp = if tid == 0 {
                samples_per_thread + remainder
            } else {
                samples_per_thread
            };
            let base = if tid == 0 {
                0
            } else {
                remainder + tid * samples_per_thread
            };
            let mut yb = vec![0.0f64; n_steps];
            let mut evaluator = WarmLogLikelihoodRatio::new(&coef, p, sigma2);
            (0..n_samp)
                .map(|j| {
                    let s = base + j;
                    let eta = &add_eta[s * n_steps..(s + 1) * n_steps];
                    for i in 0..n_steps {
                        yb[i] = sigma * eta[i];
                    }
                    -evaluator.evaluate(&yb)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let (pmf_remove, remove_resolution) = samples_to_pmf(&mut remove_samples, config, 2)?;
    let (pmf_add, add_resolution) = samples_to_pmf(&mut add_samples, config, 2)?;

    Ok(
        PrivacyLossDistribution::new_asymmetric(pmf_remove, pmf_add).with_monte_carlo_guarantee(
            config.mc_failure_probability,
            remove_resolution.max(add_resolution),
        ),
    )
}

/// Monte Carlo PLD for BandMF + warm-start b-min-sep subsampling (single-example adjacent analysis).
///
/// * `strategy_coef` — first column of strategy matrix `C` (length = bandwidth).
/// * `n_steps` — number of iterations `n`.
/// * `p` — per-iteration inclusion probability `p` in Algorithm 2 (not `p_0`).
/// * `sigma` — raw noise multiplier σ (same units as [`crate::amplification::bnb_mc_pld`]).
/// * `config` — discretization and Monte Carlo confidence configuration.
pub fn bandmf_b_min_sep_warm_mc_pld(
    strategy_coef: &[f64],
    n_steps: usize,
    p: f64,
    sigma: f64,
    config: &DiscretizationConfig,
) -> Result<PrivacyLossDistribution> {
    let num_samples = config.resolved_num_mc_samples(2)?;
    let seed = config.seed;

    if strategy_coef.is_empty() {
        return Err(PldError::InvalidParameter(
            "strategy_coef must be non-empty".into(),
        ));
    }
    if n_steps == 0 {
        return Err(PldError::InvalidParameter("n_steps must be > 0".into()));
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
    let bands = strategy_coef.len();
    let sigma2 = sigma * sigma;
    let n_threads = rayon::current_num_threads().max(1);
    let samples_per_thread = num_samples / n_threads;
    let remainder = num_samples - samples_per_thread * n_threads;

    let coef = strategy_coef.to_vec();

    let mut remove_samples: Vec<f64> = (0..n_threads)
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
            let mut evaluator = WarmLogLikelihoodRatio::new(&coef, p, sigma2);
            (0..n_samp)
                .map(|_| {
                    sample_x_under_p(n_steps, bands, p, &mut rng, &mut xb);
                    for v in zb.iter_mut() {
                        *v = rng.sample::<f64, _>(StandardNormal);
                    }
                    y_from_x_and_zeta(&coef, n_steps, &xb, &zb, sigma, &mut yb);
                    evaluator.evaluate(&yb)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let mut add_samples: Vec<f64> = (0..n_threads)
        .into_par_iter()
        .flat_map(|tid| {
            let n_samp = if tid == 0 {
                samples_per_thread + remainder
            } else {
                samples_per_thread
            };
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(1000 + tid as u64));
            let mut yb = vec![0.0f64; n_steps];
            let mut evaluator = WarmLogLikelihoodRatio::new(&coef, p, sigma2);
            (0..n_samp)
                .map(|_| {
                    sample_y_under_q(n_steps, sigma, &mut rng, &mut yb);
                    -evaluator.evaluate(&yb)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let (pmf_remove, remove_resolution) = samples_to_pmf(&mut remove_samples, config, 2)?;
    let (pmf_add, add_resolution) = samples_to_pmf(&mut add_samples, config, 2)?;

    Ok(
        PrivacyLossDistribution::new_asymmetric(pmf_remove, pmf_add).with_monte_carlo_guarantee(
            config.mc_failure_probability,
            remove_resolution.max(add_resolution),
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discretization::DiscretizationConfig;
    use approx::assert_abs_diff_eq;

    fn warm_path_probability(mask: u64, n: usize, bands: usize, p: f64) -> f64 {
        let denominator = 1.0 + (bands - 1) as f64 * p;
        (0..bands)
            .map(|initial_state| {
                let mut probability = if initial_state == 0 { 1.0 } else { p } / denominator;
                let mut barred_remaining = initial_state;

                for i in 0..n {
                    let participates = mask & (1 << i) != 0;
                    if barred_remaining > 0 {
                        if participates {
                            return 0.0;
                        }
                        barred_remaining -= 1;
                    } else if participates {
                        probability *= p;
                        barred_remaining = bands - 1;
                    } else {
                        probability *= 1.0 - p;
                    }
                }
                probability
            })
            .sum()
    }

    fn exact_warm_ratio(y: &[f64], coef: &[f64], p: f64, sigma2: f64) -> f64 {
        let n = y.len();
        assert!(n < u64::BITS as usize);

        (0..1_u64 << n)
            .map(|mask| {
                let probability = warm_path_probability(mask, n, coef.len(), p);
                if probability == 0.0 {
                    return 0.0;
                }
                let mut mean = vec![0.0; n];
                for col in 0..n {
                    if mask & (1 << col) != 0 {
                        for (offset, value) in coef.iter().take(n - col).enumerate() {
                            mean[col + offset] += *value;
                        }
                    }
                }
                let log_ratio: f64 = mean
                    .iter()
                    .zip(y)
                    .map(|(mu, value)| mu * value - 0.5 * mu * mu)
                    .sum::<f64>()
                    / sigma2;
                probability * log_ratio.exp()
            })
            .sum()
    }

    fn default_config() -> DiscretizationConfig {
        DiscretizationConfig {
            mc_resolution: 5e-3,
            mc_failure_probability: 1e-2,
            ..DiscretizationConfig::default()
        }
    }

    #[test]
    fn privacy_loss_matches_jax_reference() {
        let loss = WarmLogLikelihoodRatio::new(&[1.0, 0.5], 0.5, 1.0).evaluate(&[1.0, 1.0, 1.0]);
        assert_abs_diff_eq!(loss, 0.832_939_838_080_925_2, epsilon = 1e-14);
    }

    #[test]
    fn privacy_loss_matches_exact_mixture() {
        let cases = [
            (
                vec![0.8, 0.3],
                vec![-0.7, 0.1, 0.9, -0.2, 0.4, 0.6, -0.5, 0.3],
                0.17,
                1.44,
            ),
            (
                vec![0.7, 0.4, 0.2],
                vec![0.2, -0.4, 0.8, 0.1, -0.6, 0.5, 0.9, -0.3, 0.7],
                0.5,
                0.81,
            ),
            (
                vec![0.6, 0.5, 0.4, 0.3],
                vec![0.2, 0.4, -0.1, 0.8],
                1.0,
                1.0,
            ),
            (vec![1.0], vec![0.3, -0.2, 0.7, 0.1, -0.4], 0.999, 1.21),
        ];

        for (coef, y, p, sigma2) in cases {
            let expected = exact_warm_ratio(&y, &coef, p, sigma2).ln();
            let actual = WarmLogLikelihoodRatio::new(&coef, p, sigma2).evaluate(&y);
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-12);
        }
    }

    #[test]
    fn zero_signal_has_zero_privacy_loss() {
        for bands in [2, 4, 8] {
            let coef = vec![0.0; bands];
            let y = vec![0.3; 12];
            assert_abs_diff_eq!(
                WarmLogLikelihoodRatio::new(&coef, 0.2, 4.0).evaluate(&y),
                0.0,
                epsilon = 1e-12
            );
        }
    }

    #[test]
    fn bandwidth_one_reduces_to_poisson_subsampling() {
        let y = [0.3, -0.2, 0.7, 0.1, -0.4];
        let coef = [1.1];
        let p: f64 = 0.23;
        let sigma2 = 1.21;
        let expected = y
            .iter()
            .map(|value| {
                let log_gaussian_ratio = (coef[0] * value - 0.5 * coef[0] * coef[0]) / sigma2;
                log_add((-p).ln_1p(), p.ln() + log_gaussian_ratio)
            })
            .sum::<f64>();

        assert_abs_diff_eq!(
            WarmLogLikelihoodRatio::new(&coef, p, sigma2).evaluate(&y),
            expected,
            epsilon = 1e-14
        );
    }

    #[test]
    fn privacy_loss_remains_finite_when_ratio_overflows() {
        let n = 1_000;
        let y = vec![1.0; n];
        let loss = WarmLogLikelihoodRatio::new(&[1.0, 0.0], 1.0, 0.25).evaluate(&y);
        assert_abs_diff_eq!(loss, 1_000.0, epsilon = 1e-10);
    }

    #[test]
    fn likelihood_ratio_normalizes_under_q() {
        let n = 10;
        let p = 0.2;
        let sigma = 2.0;
        let mut rng = StdRng::seed_from_u64(779);
        let mut y = vec![0.0; n];

        for bands in [2, 4, 8] {
            let coef = vec![1.0 / (bands as f64).sqrt(); bands];
            let mut evaluator = WarmLogLikelihoodRatio::new(&coef, p, sigma * sigma);
            let mean = (0..20_000)
                .map(|_| {
                    sample_y_under_q(n, sigma, &mut rng, &mut y);
                    evaluator.evaluate(&y).exp()
                })
                .sum::<f64>()
                / 20_000.0;
            assert_abs_diff_eq!(mean, 1.0, epsilon = 0.02);
        }
    }

    #[test]
    fn mc_pld_smoke() {
        let coef = vec![0.7_f64.sqrt(), 0.3_f64.sqrt()];
        let cfg = default_config();
        let pld = bandmf_b_min_sep_warm_mc_pld(&coef, 50, 0.05, 1.0, &cfg).unwrap();
        let eps = pld.epsilon_at(1e-2);
        assert!(eps > 0.0 && eps.is_finite());
    }

    #[test]
    fn transcripts_match_one_shot_epsilon() {
        let coef = vec![0.8_f64.sqrt(), 0.2_f64.sqrt(), 0.0];
        let mut cfg = default_config();
        cfg.seed = 99;
        let sigma = 1.15;
        let n = 40;
        let p = 0.06;
        let s = cfg.resolved_num_mc_samples(2).unwrap();
        let (rx, rz, ae) = bandmf_b_min_sep_prepare_transcripts(&coef, n, p, s, cfg.seed).unwrap();
        let pld_t =
            bandmf_b_min_sep_pld_from_transcripts(&rx, &rz, &ae, &coef, n, p, sigma, &cfg).unwrap();
        let pld_1 = bandmf_b_min_sep_warm_mc_pld(&coef, n, p, sigma, &cfg).unwrap();
        let d = 1e-2;
        let e1 = pld_1.epsilon_at(d);
        let e2 = pld_t.epsilon_at(d);
        assert!(
            (e1 - e2).abs() < 0.05,
            "epsilon mismatch: one_shot={e1} transcripts={e2}"
        );
    }
}
