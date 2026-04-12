//! Monte Carlo BnB accountant for matrix mechanisms.
//!
//! Implements Algorithm 2 of Choquette-Choo et al. (2024) "Near Exact Privacy
//! Amplification for Matrix Mechanisms" (arxiv:2410.06266).
//!
//! Given a Gram matrix G of the BnB dominating pair mixture means, samples the
//! privacy loss distribution via Monte Carlo and returns a discretized PLD.
//!
//! The dominating pair is:
//!   P = (1/b) Σ_{i=1}^{b} N(m_i, σ²I)
//!   Q = N(0, σ²I)
//!
//! After Gram matrix reduction, the privacy loss depends on G only:
//!   Y = log((1/b) Σ_k exp((2u_k - G_kk) / (2σ²)))
//! where u ~ N(G_i, σ²G) with i ~ Uniform([b]).
//!
//! # Performance
//!
//! For DP-λCGD, the Gram matrix has near-AR(1) structure (entries decay as
//! λ^{|i-j|}). The Cholesky factor inherits this bandedness, so we use a
//! **banded Cholesky** with automatic bandwidth detection. This reduces the
//! per-sample cost from O(b²) to O(b·p) where p is the effective bandwidth.
//!
//! # References
//!
//! - Choquette-Choo et al. (2024), "Near Exact Privacy Amplification for Matrix
//!   Mechanisms" <https://arxiv.org/abs/2410.06266>

use crate::discretization::DiscretizationConfig;
use crate::error::{PldError, Result};
use crate::pld::pmf::Pmf;
use crate::pld::PrivacyLossDistribution;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use rayon::prelude::*;

/// Banded Cholesky decomposition.
///
/// Computes L such that G ≈ L·Lᵀ, where L is lower-triangular with
/// bandwidth `bw` (L[i,j] = 0 for j < i - bw).
///
/// The bandwidth is auto-detected: entries of L smaller than `threshold`
/// times the diagonal are set to zero.
struct BandedCholesky {
    /// Cholesky entries stored as (b × (bw+1)) in row-major.
    /// data[i * stride + (j - (i-bw).max(0))] = L[i, j]
    data: Vec<f64>,
    b: usize,
    bw: usize, // bandwidth (number of sub-diagonals kept)
    stride: usize,
}

impl BandedCholesky {
    /// Compute banded Cholesky with estimated bandwidth.
    ///
    /// Estimates bandwidth from the Gram matrix structure (smallest p where
    /// off-diagonal entries drop below threshold), then computes only the
    /// banded part of the Cholesky. O(b·bw²) instead of O(b³).
    fn compute(gram: &[f64], b: usize, threshold: f64) -> Result<Self> {
        // Estimate bandwidth from the Gram matrix: find the smallest p such
        // that max_{|i-j|>p} |G_{ij}| < threshold * max(G_{ii}).
        let max_diag = (0..b).map(|i| gram[i * b + i]).fold(0.0f64, f64::max);
        let abs_thresh = threshold * max_diag;

        let mut est_bw: usize = 1;
        for d in 1..b {
            let mut any_above = false;
            // Check a few entries at distance d
            for i in (0..b.saturating_sub(d)).step_by((b / 20).max(1)) {
                if gram[i * b + (i + d)].abs() > abs_thresh {
                    any_above = true;
                    break;
                }
            }
            if any_above {
                est_bw = d;
            } else {
                break;
            }
        }
        // Safety margin
        let bw = (est_bw * 2 + 10).min(b - 1);
        let stride = bw + 1;

        // Compute banded Cholesky directly: only L[i,j] for j >= i - bw
        let mut data = vec![0.0f64; b * stride];

        for i in 0..b {
            let j_lo = i.saturating_sub(bw);

            for j in j_lo..=i {
                let band_j = j - j_lo;

                let k_lo = i.saturating_sub(bw).max(j.saturating_sub(bw));
                let mut sum = 0.0;
                for k in k_lo..j {
                    let ik_band = k.saturating_sub(i.saturating_sub(bw));
                    let jk_band = k.saturating_sub(j.saturating_sub(bw));
                    if ik_band < stride && jk_band < stride {
                        sum += data[i * stride + ik_band] * data[j * stride + jk_band];
                    }
                }

                if i == j {
                    let diag = gram[i * b + i] - sum;
                    if diag <= 1e-15 {
                        // Regularize slightly for numerical stability
                        data[i * stride + band_j] = (diag.max(1e-30)).sqrt();
                    } else {
                        data[i * stride + band_j] = diag.sqrt();
                    }
                } else {
                    let diag_j_band = bw.min(j); // band index of diagonal of row j
                    let l_jj = data[j * stride + diag_j_band];
                    if l_jj > 0.0 {
                        data[i * stride + band_j] = (gram[i * b + j] - sum) / l_jj;
                    }
                }
            }
        }

        Ok(BandedCholesky {
            data,
            b,
            bw,
            stride,
        })
    }

    /// Compute u = mean + σ * L * z using banded structure.
    /// O(b * bw) instead of O(b²).
    fn sample_gaussian(&self, mean: &[f64], sigma: f64, z: &[f64], out: &mut [f64]) {
        for k in 0..self.b {
            let j_lo = k.saturating_sub(self.bw);
            let mut lz = 0.0;
            for j in j_lo..=k {
                let band_col = j - j_lo;
                lz += self.data[k * self.stride + band_col] * z[j];
            }
            out[k] = mean[k] + sigma * lz;
        }
    }
}

/// Sample one privacy loss value from the BnB dominating pair.
///
/// For the "remove" direction: X ~ P, Y = log(P(X)/Q(X))
fn sample_privacy_loss_remove(
    gram: &[f64],
    chol: &BandedCholesky,
    b: usize,
    sigma: f64,
    inv_2sig2: f64,
    diag_terms: &[f64],
    z_buf: &mut Vec<f64>,
    u_buf: &mut Vec<f64>,
    rng: &mut impl Rng,
) -> f64 {
    // Step 1: Sample bin i ~ Uniform([b])
    let i: usize = rng.gen_range(0..b);

    // Step 2: Sample z ~ N(0, I_b)
    for v in z_buf.iter_mut() {
        *v = rng.sample::<f64, _>(StandardNormal);
    }

    // Step 3: Compute u = G[i,:] + σ * L * z (banded, O(b*bw))
    let mean = &gram[i * b..i * b + b];
    chol.sample_gaussian(mean, sigma, z_buf, u_buf);

    // Step 4: Y = log((1/b) Σ_k exp((2u_k - G_kk) / (2σ²)))
    //        = log_sum_exp(2*u_k*inv_2sig2 + diag_terms) - log(b)
    let mut max_val = f64::NEG_INFINITY;
    for k in 0..b {
        let t = u_buf[k] * inv_2sig2 * 2.0 + diag_terms[k];
        u_buf[k] = t; // reuse buffer for terms
        if t > max_val {
            max_val = t;
        }
    }
    let mut sum_exp = 0.0f64;
    for k in 0..b {
        sum_exp += (u_buf[k] - max_val).exp();
    }
    max_val + sum_exp.ln() - (b as f64).ln()
}

/// Sample one privacy loss value for the "add" direction.
fn sample_privacy_loss_add(
    gram: &[f64],
    chol: &BandedCholesky,
    b: usize,
    sigma: f64,
    inv_2sig2: f64,
    diag_terms: &[f64],
    z_buf: &mut Vec<f64>,
    u_buf: &mut Vec<f64>,
    rng: &mut impl Rng,
) -> f64 {
    // Sample z and compute u = 0 + σ * L * z (mean is 0 under Q)
    for v in z_buf.iter_mut() {
        *v = rng.sample::<f64, _>(StandardNormal);
    }
    let zeros = vec![0.0; b];
    chol.sample_gaussian(&zeros, sigma, z_buf, u_buf);

    // Y_add = -log(P(X)/Q(X))
    let mut max_val = f64::NEG_INFINITY;
    for k in 0..b {
        let t = u_buf[k] * inv_2sig2 * 2.0 + diag_terms[k];
        u_buf[k] = t;
        if t > max_val {
            max_val = t;
        }
    }
    let mut sum_exp = 0.0f64;
    for k in 0..b {
        sum_exp += (u_buf[k] - max_val).exp();
    }
    -(max_val + sum_exp.ln() - (b as f64).ln())
}

/// Compute the BnB PLD via Monte Carlo sampling.
///
/// Samples privacy loss values from the dominating pair, builds a histogram
/// on the PLD discretization grid, and returns a standard
/// `PrivacyLossDistribution` (asymmetric: separate remove/add PLDs).
///
/// # Arguments
///
/// * `gram` — b×b Gram matrix (row-major, symmetric positive definite)
/// * `num_bins` — Number of bins b
/// * `sigma` — Noise multiplier
/// * `num_samples` — Number of MC samples (e.g., 100_000)
/// * `seed` — RNG seed for reproducibility
/// * `config` — Discretization configuration
pub fn bnb_mc_pld(
    gram: &[f64],
    num_bins: usize,
    sigma: f64,
    num_samples: usize,
    seed: u64,
    config: &DiscretizationConfig,
) -> Result<PrivacyLossDistribution> {
    let b = num_bins;

    if gram.len() != b * b {
        return Err(PldError::InvalidParameter(format!(
            "Gram matrix size {} doesn't match num_bins²={}",
            gram.len(),
            b * b
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

    // Banded Cholesky: auto-detects bandwidth from Gram structure.
    // For DP-λCGD with λ=0.9, b=1953: bandwidth ≈ 2-5 (nearly bidiagonal).
    let chol = BandedCholesky::compute(gram, b, 1e-6)?;

    let disc = config.discretization;
    let pessimistic = config.pessimistic_estimate;
    let sigma2 = sigma * sigma;
    let inv_2sig2 = 1.0 / (2.0 * sigma2);

    // Precompute -G_kk / (2σ²) for the log-sum-exp
    let diag_terms: Vec<f64> = (0..b).map(|k| -gram[k * b + k] * inv_2sig2).collect();

    // Parallel MC sampling
    let n_threads = rayon::current_num_threads().max(1);
    let samples_per_thread = num_samples / n_threads;
    let remainder = num_samples - samples_per_thread * n_threads;

    // Sample "remove" direction
    let remove_samples: Vec<f64> = (0..n_threads)
        .into_par_iter()
        .flat_map(|tid| {
            let n = if tid == 0 {
                samples_per_thread + remainder
            } else {
                samples_per_thread
            };
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(tid as u64));
            let mut z_buf = vec![0.0f64; b];
            let mut u_buf = vec![0.0f64; b];
            (0..n)
                .map(|_| {
                    sample_privacy_loss_remove(
                        gram,
                        &chol,
                        b,
                        sigma,
                        inv_2sig2,
                        &diag_terms,
                        &mut z_buf,
                        &mut u_buf,
                        &mut rng,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect();

    // Sample "add" direction
    let add_samples: Vec<f64> = (0..n_threads)
        .into_par_iter()
        .flat_map(|tid| {
            let n = if tid == 0 {
                samples_per_thread + remainder
            } else {
                samples_per_thread
            };
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(1000 + tid as u64));
            let mut z_buf = vec![0.0f64; b];
            let mut u_buf = vec![0.0f64; b];
            (0..n)
                .map(|_| {
                    sample_privacy_loss_add(
                        gram,
                        &chol,
                        b,
                        sigma,
                        inv_2sig2,
                        &diag_terms,
                        &mut z_buf,
                        &mut u_buf,
                        &mut rng,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect();

    // Build PMFs from samples
    let pmf_remove = samples_to_pmf(&remove_samples, disc, pessimistic, config.max_grid_size);
    let pmf_add = samples_to_pmf(&add_samples, disc, pessimistic, config.max_grid_size);

    Ok(PrivacyLossDistribution::new_asymmetric(pmf_remove, pmf_add))
}

/// Convert MC samples into a discrete PMF on the PLD grid.
fn samples_to_pmf(
    samples: &[f64],
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

    let min_sample = samples.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_sample = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

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

    for &y in samples {
        if !y.is_finite() {
            if y > 0.0 {
                infinity_mass += 1.0 / n;
            }
            continue;
        }

        let bucket_idx = if pessimistic_estimate {
            (y / effective_disc).ceil() as i64 - effective_lo
        } else {
            (y / effective_disc).round() as i64 - effective_lo
        };

        if bucket_idx < 0 {
            probs[0] += 1.0 / n;
        } else if bucket_idx >= effective_grid_size as i64 {
            infinity_mass += 1.0 / n;
        } else {
            probs[bucket_idx as usize] += 1.0 / n;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> DiscretizationConfig {
        DiscretizationConfig::default()
    }

    #[test]
    fn test_banded_cholesky_identity() {
        let gram = vec![1.0, 0.0, 0.0, 1.0];
        let chol = BandedCholesky::compute(&gram, 2, 1e-6).unwrap();
        // L should be identity → L*z = z
        let mut out = vec![0.0; 2];
        chol.sample_gaussian(&[0.0, 0.0], 1.0, &[1.0, 2.0], &mut out);
        assert!((out[0] - 1.0).abs() < 1e-8, "out[0]={}", out[0]);
        assert!((out[1] - 2.0).abs() < 1e-8, "out[1]={}", out[1]);
    }

    #[test]
    fn test_banded_cholesky_2x2() {
        let gram = vec![4.0, 2.0, 2.0, 3.0];
        let chol = BandedCholesky::compute(&gram, 2, 1e-6).unwrap();
        // L = [[2, 0], [1, sqrt(2)]]
        // L * [1, 0] = [2, 1]
        let mut out = vec![0.0; 2];
        chol.sample_gaussian(&[0.0, 0.0], 1.0, &[1.0, 0.0], &mut out);
        assert!((out[0] - 2.0).abs() < 1e-10);
        assert!((out[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bnb_mc_pld_dpsgd_small() {
        let b = 10;
        let mut gram = vec![0.0; b * b];
        for i in 0..b {
            gram[i * b + i] = 1.0;
        }
        let config = default_config();
        let pld = bnb_mc_pld(&gram, b, 1.0, 100_000, 42, &config).unwrap();
        let eps = pld.epsilon_at(1e-5);
        assert!(eps > 0.0 && eps.is_finite(), "eps = {}", eps);
    }

    #[test]
    fn test_bnb_mc_pld_more_noise_less_epsilon() {
        let b = 5;
        let mut gram = vec![0.0; b * b];
        for i in 0..b {
            gram[i * b + i] = 1.0;
        }
        let config = default_config();

        let eps_low = bnb_mc_pld(&gram, b, 0.5, 100_000, 42, &config)
            .unwrap()
            .epsilon_at(1e-5);
        let eps_high = bnb_mc_pld(&gram, b, 2.0, 100_000, 42, &config)
            .unwrap()
            .epsilon_at(1e-5);
        assert!(
            eps_high < eps_low,
            "More noise: {} should be < {}",
            eps_high,
            eps_low
        );
    }

    #[test]
    fn test_bnb_mc_pld_rejects_bad_params() {
        let config = default_config();
        assert!(bnb_mc_pld(&[1.0], 2, 1.0, 1000, 42, &config).is_err());
        assert!(bnb_mc_pld(&[1.0, 0.0, 0.0, 1.0], 2, 0.0, 1000, 42, &config).is_err());
        assert!(bnb_mc_pld(&[1.0, 0.0, 0.0, 1.0], 2, 1.0, 0, 42, &config).is_err());
    }

    #[test]
    fn test_bnb_mc_pld_lambda_cgd() {
        let b = 5;
        let lambda: f64 = 0.5;
        let e = 2;
        let mut gram = vec![0.0; b * b];
        for i in 0..b {
            for j in 0..b {
                let gap = (i as i32 - j as i32).unsigned_abs() as i32;
                gram[i * b + j] = e as f64 * lambda.powi(gap);
            }
        }
        let config = default_config();
        let pld = bnb_mc_pld(&gram, b, 1.0, 100_000, 42, &config).unwrap();
        let eps = pld.epsilon_at(1e-5);
        assert!(eps > 0.0 && eps.is_finite());
    }
}
