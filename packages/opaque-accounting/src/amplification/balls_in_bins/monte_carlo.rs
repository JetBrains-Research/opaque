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
//! For DP-λCGD the Gram is *cyclically* banded — `G_{ij} ≈ E·λ^d +
//! (E-1)·λ^{b-d}` for `d = |i-j|`, so entries decay away from the diagonal and
//! then rise again at the corner. We therefore use a **cyclically banded
//! Cholesky** (`super::cyclic_cholesky`), which keeps the per-sample cost at
//! O(b·p) while retaining the wrap that a linear band would discard.
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

use super::cyclic_cholesky::CyclicBandedCholesky;

/// Number of samples assigned to one deterministic Monte Carlo stream.
///
/// The shard count must depend only on `num_mc_samples`, not on the Rayon
/// worker count: users expect a fixed seed to select the same samples on every
/// machine.
const SAMPLES_PER_SHARD: usize = 1024;

/// Sample one privacy loss value from the BnB dominating pair.
///
/// For the "remove" direction: X ~ P, Y = log(P(X)/Q(X))
#[allow(clippy::too_many_arguments, clippy::ptr_arg)]
fn sample_privacy_loss_remove(
    gram: &[f64],
    chol: &CyclicBandedCholesky,
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
    for val in &u_buf[..b] {
        sum_exp += (val - max_val).exp();
    }
    max_val + sum_exp.ln() - (b as f64).ln()
}

/// Sample one privacy loss value for the "add" direction.
#[allow(clippy::too_many_arguments, clippy::ptr_arg)]
fn sample_privacy_loss_add(
    _gram: &[f64],
    chol: &CyclicBandedCholesky,
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
    for val in &u_buf[..b] {
        sum_exp += (val - max_val).exp();
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
/// * `config` — Discretization configuration (includes `num_mc_samples` and `seed`)
pub fn bnb_mc_pld(
    gram: &[f64],
    num_bins: usize,
    sigma: f64,
    config: &DiscretizationConfig,
) -> Result<PrivacyLossDistribution> {
    let b = num_bins;
    let num_samples = config.num_mc_samples;
    let seed = config.seed;

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

    // Cyclically banded Cholesky: the Gram wraps, so a linear band would
    // discard the corner (up to ~68% of the diagonal for λ-CGD).
    let chol = CyclicBandedCholesky::compute(gram, b, 1e-6)?;

    // The factorisation must actually reproduce the Gram it was handed —
    // otherwise the sampler draws u ~ N(m_i, σ²·LLᵀ) from the wrong
    // covariance, which is not conservative in either direction.
    let max_diag = (0..b).map(|i| gram[i * b + i]).fold(0.0f64, f64::max);
    let residual = chol.max_residual(gram);
    if residual > 1e-8 * max_diag.max(1.0) {
        return Err(PldError::NumericalError(format!(
            "Cholesky does not reproduce the Gram matrix: max|G - LLᵀ| = {} \
             (tolerance {}, b={}). The sampled covariance would not be G.",
            residual,
            1e-8 * max_diag.max(1.0),
            b
        )));
    }

    let disc = config.discretization;
    let sigma2 = sigma * sigma;
    let inv_2sig2 = 1.0 / (2.0 * sigma2);

    // Precompute -G_kk / (2σ²) for the log-sum-exp
    let diag_terms: Vec<f64> = (0..b).map(|k| -gram[k * b + k] * inv_2sig2).collect();

    // Partition samples into fixed, seed-indexed shards. Rayon may execute
    // these shards on any number of workers without changing the streams or
    // their output order.
    let num_shards =
        num_samples / SAMPLES_PER_SHARD + usize::from(num_samples % SAMPLES_PER_SHARD != 0);

    // Sample "remove" direction
    let remove_samples: Vec<f64> = (0..num_shards)
        .into_par_iter()
        .map(|shard| {
            let start = shard * SAMPLES_PER_SHARD;
            let end = (start + SAMPLES_PER_SHARD).min(num_samples);
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(shard as u64));
            let mut z_buf = vec![0.0f64; b];
            let mut u_buf = vec![0.0f64; b];
            (start..end)
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
        .collect::<Vec<_>>()
        .into_iter()
        .flatten()
        .collect();

    // Sample "add" direction
    let add_samples: Vec<f64> = (0..num_shards)
        .into_par_iter()
        .map(|shard| {
            let start = shard * SAMPLES_PER_SHARD;
            let end = (start + SAMPLES_PER_SHARD).min(num_samples);
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(1000 + shard as u64));
            let mut z_buf = vec![0.0f64; b];
            let mut u_buf = vec![0.0f64; b];
            (start..end)
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
        .collect::<Vec<_>>()
        .into_iter()
        .flatten()
        .collect();

    // Build PMFs from samples
    let pmf_remove = samples_to_pmf(&remove_samples, disc, config.max_grid_size)?;
    let pmf_add = samples_to_pmf(&add_samples, disc, config.max_grid_size)?;

    Ok(PrivacyLossDistribution::new_asymmetric(pmf_remove, pmf_add))
}

/// Convert MC samples into a discrete PMF on the PLD grid.
///
/// # Errors
///
/// Returns `NumericalError` on any non-finite sample. The privacy loss here is
/// `log((1/b) Σ exp(·))` of a strictly positive, finite sum, so it is finite
/// whenever the sampler is well-conditioned — a `NaN` or `±inf` means the
/// Cholesky factor blew up, not that the true privacy loss is infinite.
/// Previously `NaN` (which is neither `is_finite()` nor `> 0.0`) was dropped
/// outright and `-inf` vanished, so the PMF silently lost mass and `Pmf::new`
/// does no renormalisation.
pub(crate) fn samples_to_pmf(
    samples: &[f64],
    discretization: f64,
    max_grid_size: usize,
) -> Result<Pmf> {
    if samples.is_empty() {
        return Ok(Pmf::new(discretization, 0, vec![1.0], 0.0, max_grid_size));
    }

    if let Some((idx, bad)) = samples
        .iter()
        .enumerate()
        .find(|(_, y)| !y.is_finite())
        .map(|(i, y)| (i, *y))
    {
        return Err(PldError::NumericalError(format!(
            "non-finite privacy loss sample {} at index {} of {}; the sampler \
             is ill-conditioned (check the Cholesky factor and the Gram matrix)",
            bad,
            idx,
            samples.len()
        )));
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
        let bucket_idx = (y / effective_disc).ceil() as i64 - effective_lo;

        if bucket_idx < 0 {
            probs[0] += 1.0 / n;
        } else if bucket_idx >= effective_grid_size as i64 {
            infinity_mass += 1.0 / n;
        } else {
            probs[bucket_idx as usize] += 1.0 / n;
        }
    }

    // `Pmf::new` performs no renormalisation, so the histogram has to account
    // for every sample itself.
    let total: f64 = probs.iter().sum::<f64>() + infinity_mass;
    debug_assert!(
        (total - 1.0).abs() < 1e-9,
        "samples_to_pmf lost mass: total = {}",
        total
    );

    Ok(Pmf::new(
        effective_disc,
        effective_lo,
        probs,
        infinity_mass,
        max_grid_size,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> DiscretizationConfig {
        DiscretizationConfig::default()
    }

    /// The window that used to produce |L| ~ 1e219 and `inf`.
    ///
    /// At λ=0.9, E=4, momentum=0 and b in [278, 295] the *linearly* banded
    /// factorisation of the (now corner-retaining) Gram is not PSD. The old
    /// code regularised the pivot to sqrt(1e-30)=1e-15 and then divided by it;
    /// the resulting non-finite losses were discarded silently by
    /// `samples_to_pmf`, so the PMF quietly lost mass.
    ///
    /// The cyclic factorisation is PSD here, so this must now simply work —
    /// and in no case may it return a non-finite ε.
    #[test]
    fn test_no_blowup_in_former_non_psd_window() {
        use crate::matrix_factorization::lambda_cgd_gram_matrix;

        let mut cfg = default_config();
        cfg.num_mc_samples = 2000;
        for &b in &[278usize, 280, 282, 285, 290, 295] {
            let e = 4;
            let gram = lambda_cgd_gram_matrix(0.9, b * e, b, Some(e), true, 0.0).unwrap();
            let pld = bnb_mc_pld(&gram, b, 1.5, &cfg)
                .unwrap_or_else(|err| panic!("b={} failed: {}", b, err));
            let eps = pld.epsilon_at(1e-5);
            assert!(eps.is_finite() && eps > 0.0, "b={}: eps={}", b, eps);
        }
    }

    /// Non-finite samples must error, not vanish.
    #[test]
    fn test_samples_to_pmf_rejects_non_finite() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let samples = vec![0.1, 0.2, bad, 0.3];
            let err = samples_to_pmf(&samples, 1e-4, 10_000);
            assert!(err.is_err(), "{} must be rejected", bad);
        }
    }

    /// Every finite sample lands somewhere: the histogram conserves mass.
    #[test]
    fn test_samples_to_pmf_conserves_mass() {
        let samples: Vec<f64> = (0..1000).map(|i| (i as f64 - 500.0) * 0.01).collect();
        let pmf = samples_to_pmf(&samples, 1e-3, 10_000).unwrap();
        let total: f64 = pmf.probs.iter().sum::<f64>() + pmf.infinity_mass;
        assert!((total - 1.0).abs() < 1e-9, "total mass = {}", total);
    }

    #[test]
    fn test_cyclic_cholesky_identity() {
        let gram = vec![1.0, 0.0, 0.0, 1.0];
        let chol = CyclicBandedCholesky::compute(&gram, 2, 1e-6).unwrap();
        // L should be identity → L*z = z
        let mut out = vec![0.0; 2];
        chol.sample_gaussian(&[0.0, 0.0], 1.0, &[1.0, 2.0], &mut out);
        assert!((out[0] - 1.0).abs() < 1e-8, "out[0]={}", out[0]);
        assert!((out[1] - 2.0).abs() < 1e-8, "out[1]={}", out[1]);
    }

    #[test]
    fn test_cyclic_cholesky_2x2() {
        let gram = vec![4.0, 2.0, 2.0, 3.0];
        let chol = CyclicBandedCholesky::compute(&gram, 2, 1e-6).unwrap();
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
        let pld = bnb_mc_pld(&gram, b, 1.0, &config).unwrap();
        let eps = pld.epsilon_at(1e-5);
        assert!(eps > 0.0 && eps.is_finite(), "eps = {}", eps);
    }

    #[test]
    fn test_bnb_mc_pld_is_identical_across_thread_counts() {
        let b = 8;
        let mut gram = vec![0.0; b * b];
        for i in 0..b {
            for j in 0..b {
                let distance = i.abs_diff(j).min(b - i.abs_diff(j));
                gram[i * b + j] = 0.2f64.powi(distance as i32);
            }
        }
        let mut config = default_config();
        config.discretization = 1e-3;
        config.num_mc_samples = 4096;
        config.seed = 173;

        let single_thread = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| bnb_mc_pld(&gram, b, 1.3, &config).unwrap());
        let many_threads = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap()
            .install(|| bnb_mc_pld(&gram, b, 1.3, &config).unwrap());

        assert_eq!(
            single_thread.pmf_remove.discretization,
            many_threads.pmf_remove.discretization
        );
        assert_eq!(
            single_thread.pmf_remove.lower_loss_index,
            many_threads.pmf_remove.lower_loss_index
        );
        assert_eq!(
            single_thread.pmf_remove.probs,
            many_threads.pmf_remove.probs
        );
        assert_eq!(
            single_thread.pmf_remove.infinity_mass,
            many_threads.pmf_remove.infinity_mass
        );

        let single_add = single_thread.pmf_add.as_ref().unwrap();
        let many_add = many_threads.pmf_add.as_ref().unwrap();
        assert_eq!(single_add.discretization, many_add.discretization);
        assert_eq!(single_add.lower_loss_index, many_add.lower_loss_index);
        assert_eq!(single_add.probs, many_add.probs);
        assert_eq!(single_add.infinity_mass, many_add.infinity_mass);
    }

    #[test]
    fn test_bnb_mc_pld_more_noise_less_epsilon() {
        let b = 5;
        let mut gram = vec![0.0; b * b];
        for i in 0..b {
            gram[i * b + i] = 1.0;
        }
        let config = default_config();

        let eps_low = bnb_mc_pld(&gram, b, 0.5, &config).unwrap().epsilon_at(1e-5);
        let eps_high = bnb_mc_pld(&gram, b, 2.0, &config).unwrap().epsilon_at(1e-5);
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
        assert!(bnb_mc_pld(&[1.0], 2, 1.0, &config).is_err());
        assert!(bnb_mc_pld(&[1.0, 0.0, 0.0, 1.0], 2, 0.0, &config).is_err());
        let mut zero_mc = default_config();
        zero_mc.num_mc_samples = 0;
        assert!(bnb_mc_pld(&[1.0, 0.0, 0.0, 1.0], 2, 1.0, &zero_mc).is_err());
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
        let pld = bnb_mc_pld(&gram, b, 1.0, &config).unwrap();
        let eps = pld.epsilon_at(1e-5);
        assert!(eps > 0.0 && eps.is_finite());
    }
}
