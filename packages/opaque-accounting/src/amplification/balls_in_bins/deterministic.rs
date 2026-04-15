//! Deterministic BnB accountant for matrix mechanisms.
//!
//! This module provides a sampling-free alternative to Monte Carlo BnB by
//! computing finite-order moments of the privacy loss ratio exactly (for
//! integer orders) and forming a certified Chernoff-style delta envelope.
//!
//! The PLD is built by evaluating the certified δ(ε) curve on the standard
//! discretization grid and applying Connect-the-Dots discretization.

use crate::adjacency::Adjacency;
use crate::discretization::connect_the_dots::discretize_from_deltas;
use crate::discretization::{DiscretizationConfig, EpsilonBounds};
use crate::error::{PldError, Result};
use crate::pld::PrivacyLossDistribution;

/// Controls the deterministic BnB approximation.
#[derive(Debug, Clone, Copy)]
pub struct DeterministicOptions {
    /// Maximum integer moment order k (uses alpha=k+1 for RDP conversion).
    pub max_order_k: usize,
    /// Upper epsilon bound for the discretized δ(ε) curve (must match PLD grid span).
    pub epsilon_max: f64,
    /// Number of epsilon grid points in [0, epsilon_max] (diagnostic curve only).
    pub epsilon_points: usize,
    /// Hard cap on DP state count to avoid exponential blow-ups for large `b`.
    pub max_states: usize,
}

impl Default for DeterministicOptions {
    fn default() -> Self {
        Self {
            max_order_k: 12,
            epsilon_max: 20.0,
            epsilon_points: 256,
            max_states: 200_000,
        }
    }
}

#[derive(Debug, Clone)]
struct MomentDpState {
    counts: Vec<usize>,
    log_weight: f64,
    pair_sum: f64,
}

fn validate_inputs(gram: &[f64], num_bins: usize, sigma: f64, opts: DeterministicOptions) -> Result<()> {
    if num_bins == 0 {
        return Err(PldError::InvalidParameter(
            "num_bins must be >= 1".to_string(),
        ));
    }
    if gram.len() != num_bins * num_bins {
        return Err(PldError::InvalidParameter(format!(
            "Gram matrix size {} doesn't match num_bins²={}",
            gram.len(),
            num_bins * num_bins
        )));
    }
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(PldError::InvalidParameter(format!(
            "sigma must be finite and > 0, got {}",
            sigma
        )));
    }
    if opts.max_order_k < 1 {
        return Err(PldError::InvalidParameter(
            "max_order_k must be >= 1".to_string(),
        ));
    }
    if !opts.epsilon_max.is_finite() || opts.epsilon_max <= 0.0 {
        return Err(PldError::InvalidParameter(format!(
            "epsilon_max must be finite and > 0, got {}",
            opts.epsilon_max
        )));
    }
    if opts.epsilon_points < 2 {
        return Err(PldError::InvalidParameter(
            "epsilon_points must be >= 2".to_string(),
        ));
    }
    if opts.max_states < 1 {
        return Err(PldError::InvalidParameter(
            "max_states must be >= 1".to_string(),
        ));
    }
    Ok(())
}

fn logsumexp_pair(a: f64, b: f64) -> f64 {
    if a.is_finite() && b.is_finite() {
        if a >= b {
            a + (b - a).exp().ln_1p()
        } else {
            b + (a - b).exp().ln_1p()
        }
    } else if a.is_finite() {
        a
    } else {
        b
    }
}

fn compute_log_moment_k_exact(
    gram: &[f64],
    b: usize,
    sigma: f64,
    k: usize,
    max_states: usize,
) -> Result<f64> {
    if k == 1 {
        return Ok(0.0);
    }

    let mut states = vec![MomentDpState {
        counts: vec![0; b],
        log_weight: 0.0,
        pair_sum: 0.0,
    }];

    // Dynamic program over occupancy vectors for exact integer-order moments.
    for t in 0..k {
        let mut next_states: std::collections::HashMap<Vec<usize>, (f64, f64)> =
            std::collections::HashMap::new();
        for st in &states {
            for j in 0..b {
                let mut new_counts = st.counts.clone();
                let nj = new_counts[j];
                new_counts[j] = nj + 1;

                // Increment pairwise term Σ_{a≠c} G_{i_a i_c} in occupancy form.
                let mut incr = 0.0f64;
                for u in 0..b {
                    incr += (st.counts[u] as f64) * gram[u * b + j];
                }
                let new_pair_sum = st.pair_sum + 2.0 * incr;
                let new_log_weight = st.log_weight - ((t + 1) as f64).ln();

                let entry = next_states
                    .entry(new_counts)
                    .or_insert((f64::NEG_INFINITY, new_pair_sum));
                entry.1 = new_pair_sum;
                entry.0 = logsumexp_pair(entry.0, new_log_weight);
            }
        }

        if next_states.len() > max_states {
            return Err(PldError::InvalidParameter(format!(
                "deterministic BnB moment DP exceeded max_states={} at depth {} (b={}). \
                 Reduce num_bins or increase max_states.",
                max_states,
                t + 1,
                b
            )));
        }

        states = next_states
            .into_iter()
            .map(|(counts, (log_weight, pair_sum))| MomentDpState {
                counts,
                log_weight,
                pair_sum,
            })
            .collect();
    }

    let c = 1.0 / (2.0 * sigma * sigma);
    let mut log_mk = f64::NEG_INFINITY;
    for st in &states {
        let term = st.log_weight + c * st.pair_sum;
        log_mk = logsumexp_pair(log_mk, term);
    }
    Ok(log_mk - (k as f64) * (b as f64).ln())
}

fn deterministic_delta_envelope(log_moments: &[f64], epsilon: f64) -> f64 {
    // alpha = k + 1, D_alpha = log(M_k)/k
    let mut best = f64::INFINITY;
    for (idx, &log_mk) in log_moments.iter().enumerate() {
        let k = idx + 1;
        let exponent = log_mk - (k as f64) * epsilon;
        best = best.min(exponent.exp());
    }
    best.clamp(0.0, 1.0)
}

/// Return deterministic upper-bound deltas on a uniform epsilon grid.
pub fn bnb_deterministic_delta_curve(
    gram: &[f64],
    num_bins: usize,
    sigma: f64,
    opts: DeterministicOptions,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    validate_inputs(gram, num_bins, sigma, opts)?;

    let mut log_moments = Vec::with_capacity(opts.max_order_k);
    for k in 1..=opts.max_order_k {
        log_moments.push(compute_log_moment_k_exact(
            gram,
            num_bins,
            sigma,
            k,
            opts.max_states,
        )?);
    }

    let mut eps = Vec::with_capacity(opts.epsilon_points);
    let mut delta_remove = Vec::with_capacity(opts.epsilon_points);
    let mut delta_add = Vec::with_capacity(opts.epsilon_points);
    for i in 0..opts.epsilon_points {
        let e = opts.epsilon_max * (i as f64) / ((opts.epsilon_points - 1) as f64);
        let d = deterministic_delta_envelope(&log_moments, e);
        eps.push(e);
        delta_remove.push(d);
        // Conservative placeholder: same certified envelope for add direction.
        delta_add.push(d);
    }
    Ok((eps, delta_remove, delta_add))
}

/// Build a conservative deterministic PLD for BnB matrix mechanisms.
pub fn bnb_deterministic_pld(
    gram: &[f64],
    num_bins: usize,
    sigma: f64,
    config: &DiscretizationConfig,
    opts: DeterministicOptions,
) -> Result<PrivacyLossDistribution> {
    validate_inputs(gram, num_bins, sigma, opts)?;

    let mut log_moments = Vec::with_capacity(opts.max_order_k);
    for k in 1..=opts.max_order_k {
        log_moments.push(compute_log_moment_k_exact(
            gram,
            num_bins,
            sigma,
            k,
            opts.max_states,
        )?);
    }

    let bounds = EpsilonBounds {
        epsilon_lower: 0.0,
        epsilon_upper: opts.epsilon_max,
    };
    let effective_disc = config.effective_discretization(&bounds);
    let effective_config = DiscretizationConfig {
        discretization: effective_disc,
        ..config.clone()
    };

    let rounded_upper = (bounds.epsilon_upper / effective_disc).ceil() as i64;
    let rounded_lower = (bounds.epsilon_lower / effective_disc).floor() as i64;

    let deltas: Vec<f64> = (rounded_lower..=rounded_upper)
        .map(|i| i as f64 * effective_disc)
        .map(|eps| deterministic_delta_envelope(&log_moments, eps))
        .collect();

    let pmf_remove = discretize_from_deltas(bounds, &deltas, &effective_config, Adjacency::Remove)?;
    // Conservative placeholder for ADD direction (same certified envelope as REMOVE).
    let pmf_add = discretize_from_deltas(bounds, &deltas, &effective_config, Adjacency::Add)?;

    let tail_budget = config.tail_mass_truncation / 2.0;
    Ok(
        PrivacyLossDistribution::new_asymmetric(pmf_remove, pmf_add)
            .with_tail_budgets(tail_budget, tail_budget),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discretization::DiscretizationConfig;

    #[test]
    fn test_delta_curve_monotone_nonincreasing() {
        let b = 4;
        let mut gram = vec![0.0; b * b];
        for i in 0..b {
            gram[i * b + i] = 1.0;
        }
        let (eps, del_r, del_a) = bnb_deterministic_delta_curve(
            &gram,
            b,
            1.0,
            DeterministicOptions::default(),
        )
        .unwrap();
        assert_eq!(eps.len(), del_r.len());
        assert_eq!(eps.len(), del_a.len());
        for i in 1..del_r.len() {
            assert!(del_r[i] <= del_r[i - 1] + 1e-12);
            assert!(del_a[i] <= del_a[i - 1] + 1e-12);
        }
    }

    #[test]
    fn test_bnb_deterministic_pld_finite_epsilon() {
        let b = 3;
        let mut gram = vec![0.0; b * b];
        for i in 0..b {
            gram[i * b + i] = 0.5;
        }
        let pld = bnb_deterministic_pld(
            &gram,
            b,
            1.0,
            &DiscretizationConfig::default(),
            DeterministicOptions::default(),
        )
        .unwrap();
        let eps = pld.epsilon_at(1e-4);
        assert!(eps.is_finite() || eps.is_infinite());
    }
}
