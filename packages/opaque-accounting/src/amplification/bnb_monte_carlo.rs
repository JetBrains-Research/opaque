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

/// Cholesky decomposition of a symmetric positive-definite matrix.
///
/// Returns the lower-triangular factor L such that G = L·Lᵀ.
/// Input: row-major b×b matrix. Output: row-major b×b lower-triangular L.
fn cholesky(gram: &[f64], b: usize) -> Result<Vec<f64>> {
    let mut l = vec![0.0f64; b * b];

    for i in 0..b {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[i * b + k] * l[j * b + k];
            }

            if i == j {
                let diag = gram[i * b + i] - sum;
                if diag <= 0.0 {
                    return Err(PldError::InvalidParameter(format!(
                        "Gram matrix is not positive definite (diag={} at index {})",
                        diag, i
                    )));
                }
                l[i * b + j] = diag.sqrt();
            } else {
                l[i * b + j] = (gram[i * b + j] - sum) / l[j * b + j];
            }
        }
    }

    Ok(l)
}

/// Sample one privacy loss value from the BnB dominating pair.
///
/// For the "remove" direction: X ~ P, Y = log(P(X)/Q(X))
fn sample_privacy_loss_remove(
    gram: &[f64],
    chol: &[f64],
    b: usize,
    sigma: f64,
    rng: &mut impl Rng,
) -> f64 {
    let sigma2 = sigma * sigma;

    // Step 1: Sample bin i ~ Uniform([b])
    let i: usize = rng.gen_range(0..b);

    // Step 2: Sample z ~ N(0, I_b)
    let z: Vec<f64> = (0..b).map(|_| rng.sample::<f64, _>(StandardNormal)).collect();

    // Step 3: Compute u = G[i,:] + σ * L * z
    let mut u = vec![0.0f64; b];
    for k in 0..b {
        // u[k] = G[i,k] + σ * Σ_j L[k,j] * z[j]
        let mut lz = 0.0;
        for j in 0..=k {
            lz += chol[k * b + j] * z[j];
        }
        u[k] = gram[i * b + k] + sigma * lz;
    }

    // Step 4: Compute Y = log((1/b) Σ_k exp((2u_k - G_kk) / (2σ²)))
    //        = log_sum_exp(terms) - log(b)
    let terms: Vec<f64> = (0..b)
        .map(|k| (2.0 * u[k] - gram[k * b + k]) / (2.0 * sigma2))
        .collect();

    log_sum_exp(&terms) - (b as f64).ln()
}

/// Sample one privacy loss value for the "add" direction.
///
/// For the "add" direction: X ~ Q, Y = log(Q(X)/P(X))
fn sample_privacy_loss_add(
    gram: &[f64],
    chol: &[f64],
    b: usize,
    sigma: f64,
    rng: &mut impl Rng,
) -> f64 {
    let sigma2 = sigma * sigma;

    // X ~ Q = N(0, σ²I) — project onto m_k's
    // u_k = ⟨X, m_k⟩ ~ N(0, σ² G_kk)
    // But we need the joint: u ~ N(0, σ²G)
    let z: Vec<f64> = (0..b).map(|_| rng.sample::<f64, _>(StandardNormal)).collect();

    let mut u = vec![0.0f64; b];
    for k in 0..b {
        let mut lz = 0.0;
        for j in 0..=k {
            lz += chol[k * b + j] * z[j];
        }
        u[k] = sigma * lz; // mean is 0 under Q
    }

    // Y_add = log(Q(X)/P(X)) = -log(P(X)/Q(X))
    // log(P(X)/Q(X)) = log((1/b) Σ_k exp((2u_k - G_kk) / (2σ²)))
    let terms: Vec<f64> = (0..b)
        .map(|k| (2.0 * u[k] - gram[k * b + k]) / (2.0 * sigma2))
        .collect();

    -(log_sum_exp(&terms) - (b as f64).ln())
}

/// Numerically stable log-sum-exp.
fn log_sum_exp(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_val.is_infinite() {
        return max_val;
    }
    let sum_exp: f64 = values.iter().map(|&v| (v - max_val).exp()).sum();
    max_val + sum_exp.ln()
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
/// * `num_samples` — Number of MC samples (e.g., 1_000_000)
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
        return Err(PldError::InvalidParameter(
            "num_samples must be > 0".into(),
        ));
    }

    let chol = cholesky(gram, b)?;
    let disc = config.discretization;
    let pessimistic = config.pessimistic_estimate;

    // Determine PLD range from the Gram matrix.
    // Max privacy loss ≈ max(G_ii) / (2σ²) + some slack for noise.
    let max_diag = (0..b)
        .map(|i| gram[i * b + i])
        .fold(0.0f64, f64::max);
    let max_loss_estimate = max_diag / (2.0 * sigma * sigma)
        + 4.0 * (max_diag / (sigma * sigma)).sqrt(); // ~4σ tail

    // Use parallel sampling with rayon for large sample counts
    let chunk_size = num_samples.max(1);
    let n_chunks = 4; // Use 4 parallel threads
    let samples_per_chunk = chunk_size / n_chunks;
    let remainder = chunk_size - samples_per_chunk * n_chunks;

    // Sample "remove" direction: X ~ P, Y = log(P(X)/Q(X))
    let remove_samples: Vec<f64> = (0..n_chunks)
        .into_par_iter()
        .flat_map(|chunk_id| {
            let n = if chunk_id == 0 {
                samples_per_chunk + remainder
            } else {
                samples_per_chunk
            };
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(chunk_id as u64));
            (0..n)
                .map(|_| sample_privacy_loss_remove(gram, &chol, b, sigma, &mut rng))
                .collect::<Vec<_>>()
        })
        .collect();

    // Sample "add" direction: X ~ Q, Y = log(Q(X)/P(X))
    let add_samples: Vec<f64> = (0..n_chunks)
        .into_par_iter()
        .flat_map(|chunk_id| {
            let n = if chunk_id == 0 {
                samples_per_chunk + remainder
            } else {
                samples_per_chunk
            };
            let mut rng =
                StdRng::seed_from_u64(seed.wrapping_add(100 + chunk_id as u64));
            (0..n)
                .map(|_| sample_privacy_loss_add(gram, &chol, b, sigma, &mut rng))
                .collect::<Vec<_>>()
        })
        .collect();

    // Build PMFs from samples
    let pmf_remove = samples_to_pmf(
        &remove_samples,
        disc,
        pessimistic,
        max_loss_estimate,
        config.max_grid_size,
    );
    let pmf_add = samples_to_pmf(
        &add_samples,
        disc,
        pessimistic,
        max_loss_estimate,
        config.max_grid_size,
    );

    Ok(PrivacyLossDistribution::new_asymmetric(pmf_remove, pmf_add))
}

/// Convert MC samples into a discrete PMF on the PLD grid.
fn samples_to_pmf(
    samples: &[f64],
    discretization: f64,
    pessimistic_estimate: bool,
    _max_loss_estimate: f64,
    max_grid_size: usize,
) -> Pmf {
    if samples.is_empty() {
        return Pmf::new(discretization, 0, vec![1.0], 0.0, pessimistic_estimate, max_grid_size);
    }

    let n = samples.len() as f64;

    // Determine grid range from samples
    let min_sample = samples
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let max_sample = samples
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    // Add some margin
    let grid_lo = (min_sample / discretization).floor() as i64 - 1;
    let grid_hi = (max_sample / discretization).ceil() as i64 + 1;

    let num_buckets = (grid_hi - grid_lo + 1) as usize;

    // Safety: limit grid size
    let effective_grid_size = num_buckets.min(max_grid_size);
    let effective_disc = if num_buckets > max_grid_size {
        // Coarsen discretization to fit
        (max_sample - min_sample) / (max_grid_size as f64 - 2.0)
    } else {
        discretization
    };

    let effective_lo = if num_buckets > max_grid_size {
        (min_sample / effective_disc).floor() as i64 - 1
    } else {
        grid_lo
    };
    let _effective_hi = effective_lo + effective_grid_size as i64 - 1;

    let mut probs = vec![0.0f64; effective_grid_size];
    let mut infinity_mass = 0.0f64;

    for &y in samples {
        if !y.is_finite() {
            if y > 0.0 {
                infinity_mass += 1.0 / n;
            }
            // Negative infinity contributes to the leftmost bucket
            continue;
        }

        // Map to grid index
        let bucket_idx = if pessimistic_estimate {
            // Pessimistic: round UP (more probability at higher privacy loss)
            (y / effective_disc).ceil() as i64 - effective_lo
        } else {
            // Optimistic: round to nearest
            (y / effective_disc).round() as i64 - effective_lo
        };

        if bucket_idx < 0 {
            // Below grid — add to first bucket
            probs[0] += 1.0 / n;
        } else if bucket_idx >= effective_grid_size as i64 {
            // Above grid — add to infinity mass
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
    fn test_cholesky_identity() {
        let gram = vec![1.0, 0.0, 0.0, 1.0];
        let l = cholesky(&gram, 2).unwrap();
        assert!((l[0] - 1.0).abs() < 1e-10);
        assert!((l[1]).abs() < 1e-10);
        assert!((l[2]).abs() < 1e-10);
        assert!((l[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cholesky_2x2() {
        // G = [[4, 2], [2, 3]]
        // L = [[2, 0], [1, sqrt(2)]]
        let gram = vec![4.0, 2.0, 2.0, 3.0];
        let l = cholesky(&gram, 2).unwrap();
        assert!((l[0] - 2.0).abs() < 1e-10);
        assert!((l[1]).abs() < 1e-10);
        assert!((l[2] - 1.0).abs() < 1e-10);
        assert!((l[3] - 2.0f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_log_sum_exp() {
        let vals = vec![1.0, 2.0, 3.0];
        let result = log_sum_exp(&vals);
        let expected = (1.0f64.exp() + 2.0f64.exp() + 3.0f64.exp()).ln();
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_log_sum_exp_large() {
        // Test numerical stability with large values
        let vals = vec![1000.0, 1001.0, 999.0];
        let result = log_sum_exp(&vals);
        assert!(result.is_finite());
        assert!(result > 1000.0);
    }

    #[test]
    fn test_bnb_mc_pld_dpsgd_small() {
        // For C=I (DP-SGD), single epoch: G = I_b
        // BnB should provide amplification
        let b = 10;
        let mut gram = vec![0.0; b * b];
        for i in 0..b {
            gram[i * b + i] = 1.0;
        }
        let sigma = 1.0;
        let config = default_config();

        let pld = bnb_mc_pld(&gram, b, sigma, 100_000, 42, &config).unwrap();
        let eps = pld.epsilon_at(1e-5);
        assert!(eps > 0.0, "epsilon should be positive");
        assert!(eps.is_finite(), "epsilon should be finite");
    }

    #[test]
    fn test_bnb_mc_pld_more_noise_less_epsilon() {
        let b = 5;
        let mut gram = vec![0.0; b * b];
        for i in 0..b {
            gram[i * b + i] = 1.0;
        }
        let config = default_config();

        let eps_low_noise = bnb_mc_pld(&gram, b, 0.5, 100_000, 42, &config)
            .unwrap()
            .epsilon_at(1e-5);
        let eps_high_noise = bnb_mc_pld(&gram, b, 2.0, 100_000, 42, &config)
            .unwrap()
            .epsilon_at(1e-5);

        assert!(
            eps_high_noise < eps_low_noise,
            "More noise should give lower epsilon: {} vs {}",
            eps_high_noise,
            eps_low_noise
        );
    }

    #[test]
    fn test_bnb_mc_pld_rejects_bad_params() {
        let config = default_config();
        assert!(bnb_mc_pld(&[1.0], 2, 1.0, 1000, 42, &config).is_err()); // wrong gram size
        assert!(bnb_mc_pld(&[1.0, 0.0, 0.0, 1.0], 2, 0.0, 1000, 42, &config).is_err()); // sigma=0
        assert!(bnb_mc_pld(&[1.0, 0.0, 0.0, 1.0], 2, 1.0, 0, 42, &config).is_err()); // 0 samples
    }

    #[test]
    fn test_bnb_mc_pld_lambda_cgd() {
        // Test with a non-trivial Gram matrix (AR(1) structure)
        let b = 5;
        let lambda: f64 = 0.5;
        let e = 2; // 2 epochs
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
