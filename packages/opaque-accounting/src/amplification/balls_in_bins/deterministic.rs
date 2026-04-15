//! Deterministic BnB accountant for matrix mechanisms.
//!
//! Sampling-free upper bound based on Schuchardt & Kalinin (2026):
//! - Rényi accountant for remove direction (Algorithm 1, DP on cyclic-banded Gram)
//! - Closed-form add-direction bound (Algorithm 2)
//! - Conversion to a conservative `(ε, δ)` mechanism using Theorem 3.1.

use std::collections::HashMap;

use crate::discretization::DiscretizationConfig;
use crate::error::{PldError, Result};
use crate::mechanisms::eps_delta_pld;
use crate::pld::PrivacyLossDistribution;

fn log_sum_exp(a: f64, b: f64) -> f64 {
    if !a.is_finite() {
        return b;
    }
    if !b.is_finite() {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

fn cyclic_dist(i: usize, j: usize, b: usize) -> usize {
    let d = i.abs_diff(j);
    d.min(b - d)
}

fn key_from_state(m: usize, r: &[u16]) -> (usize, Vec<u16>) {
    (m, r.to_vec())
}

fn log_factorials(alpha: usize) -> Vec<f64> {
    let mut out = vec![0.0; alpha + 1];
    for i in 2..=alpha {
        out[i] = out[i - 1] + (i as f64).ln();
    }
    out
}

fn enumerate_prefixes(
    len: usize,
    idx: usize,
    rem: usize,
    cur: &mut [u16],
    out: &mut Vec<Vec<u16>>,
) {
    if idx == len {
        out.push(cur.to_vec());
        return;
    }
    for v in 0..=rem {
        cur[idx] = v as u16;
        enumerate_prefixes(len, idx + 1, rem - v, cur, out);
    }
}

fn renyi_remove_upper_bound(
    gram: &[f64],
    b: usize,
    sigma: f64,
    alpha: usize,
    bandwidth: usize,
) -> Result<f64> {
    let sigma2 = sigma * sigma;
    let p = bandwidth.max(1).min(b.saturating_sub(1).max(1));
    let pref_len = p.saturating_sub(1);

    let mut tau = 0.0f64;
    for i in 0..b {
        for j in 0..b {
            if cyclic_dist(i, j, b) >= p {
                tau = tau.max(gram[i * b + j]);
            }
        }
    }

    let mut gp = vec![0.0f64; b * b];
    for i in 0..b {
        for j in 0..b {
            if cyclic_dist(i, j, b) < p {
                gp[i * b + j] = (gram[i * b + j] - tau).max(0.0);
            }
        }
    }

    let log_fact = log_factorials(alpha);
    let mut prefixes = Vec::<Vec<u16>>::new();
    if pref_len == 0 {
        prefixes.push(Vec::new());
    } else {
        let mut cur = vec![0u16; pref_len];
        enumerate_prefixes(pref_len, 0, alpha, &mut cur, &mut prefixes);
    }

    let mut log_s = f64::NEG_INFINITY;

    for l in prefixes {
        let m_l: usize = l.iter().map(|&x| x as usize).sum();
        if m_l > alpha {
            continue;
        }

        let mut log_w_l = 0.0f64;
        for i in 0..pref_len {
            for j in (i + 1)..pref_len {
                log_w_l += gp[i * b + j] * (l[i] as f64) * (l[j] as f64) / sigma2;
            }
            let li = l[i] as usize;
            log_w_l += gp[i * b + i] * (li * li.saturating_sub(1)) as f64 / (2.0 * sigma2);
            log_w_l -= log_fact[li];
        }

        let mut states: HashMap<(usize, Vec<u16>), f64> = HashMap::new();
        states.insert(key_from_state(m_l, &l), log_w_l);

        for k in pref_len..b {
            let mut new_states: HashMap<(usize, Vec<u16>), f64> = HashMap::new();
            for ((m, r), log_w) in &states {
                let l_cur = r.len();
                let mut s1 = 0.0f64;
                for (i, &ri) in r.iter().enumerate() {
                    let col = k - l_cur + i;
                    s1 += 2.0 * gp[k * b + col] * (ri as f64);
                }

                for t in 0..=(alpha - *m) {
                    let delta = (gp[k * b + k] * (t * t.saturating_sub(1)) as f64 + s1 * t as f64)
                        / (2.0 * sigma2)
                        - log_fact[t];
                    let m2 = *m + t;
                    let mut r2 = if p == 1 {
                        Vec::new()
                    } else if l_cur < pref_len {
                        let mut vv = r.clone();
                        vv.push(t as u16);
                        vv
                    } else {
                        let mut vv = if r.is_empty() {
                            Vec::new()
                        } else {
                            r[1..].to_vec()
                        };
                        vv.push(t as u16);
                        vv
                    };
                    if p == 1 {
                        r2.clear();
                    }
                    let key = (m2, r2);
                    let v = *log_w + delta;
                    let entry = new_states.entry(key).or_insert(f64::NEG_INFINITY);
                    *entry = log_sum_exp(*entry, v);
                }
            }
            states = new_states;
        }

        let mut log_s_l = f64::NEG_INFINITY;
        for ((m, r), log_w) in &states {
            if *m != alpha {
                continue;
            }
            let mut close_term = 0.0f64;
            if pref_len > 0 && !r.is_empty() {
                for i in 0..pref_len {
                    for j in 0..pref_len {
                        let rr = r[pref_len - 1 - j] as f64;
                        close_term += gp[i * b + (b - 1 - j)] * (l[i] as f64) * rr / sigma2;
                    }
                }
            }
            log_s_l = log_sum_exp(log_s_l, *log_w + close_term);
        }
        log_s = log_sum_exp(log_s, log_s_l);
    }

    if !log_s.is_finite() {
        return Err(PldError::NumericalError(
            "deterministic BnB remove-direction DP failed".to_string(),
        ));
    }

    let rho = (log_s + log_factorials(alpha)[alpha] - (alpha as f64) * (b as f64).ln())
        / ((alpha - 1) as f64)
        + tau * (alpha as f64) / (2.0 * sigma2);
    Ok(rho.max(0.0))
}

fn renyi_add_upper_bound(gram: &[f64], b: usize, sigma: f64, alpha: usize) -> f64 {
    let sigma2 = sigma * sigma;
    let mut diag_sum = 0.0f64;
    let mut total_sum = 0.0f64;
    for i in 0..b {
        diag_sum += gram[i * b + i];
        for j in 0..b {
            total_sum += gram[i * b + j];
        }
    }
    diag_sum / (2.0 * b as f64 * sigma2)
        + ((alpha - 1) as f64) * total_sum / (2.0 * (b * b) as f64 * sigma2)
}

fn epsilon_from_renyi_target_delta(rho: f64, alpha: usize, target_delta: f64) -> f64 {
    let a = alpha as f64;
    let rhs = target_delta.ln() - a * (1.0 - 1.0 / a).ln() + (a - 1.0).ln();
    rho - rhs / (a - 1.0)
}

/// Deterministic BnB accountant for matrix mechanisms.
///
/// Returns a conservative `(ε,δ)` PLD using Rényi upper bounds.
#[allow(clippy::too_many_arguments)]
pub fn bnb_deterministic_pld(
    gram: &[f64],
    num_bins: usize,
    sigma: f64,
    alpha_max: usize,
    bandwidth: usize,
    target_delta: f64,
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
    if alpha_max < 2 {
        return Err(PldError::InvalidParameter(format!(
            "alpha_max must be >= 2, got {}",
            alpha_max
        )));
    }
    if !(target_delta > 0.0 && target_delta < 1.0) {
        return Err(PldError::InvalidParameter(format!(
            "target_delta must be in (0,1), got {}",
            target_delta
        )));
    }

    let mut best_eps = f64::INFINITY;
    for alpha in 2..=alpha_max {
        let rho_remove = renyi_remove_upper_bound(gram, b, sigma, alpha, bandwidth)?;
        let rho_add = renyi_add_upper_bound(gram, b, sigma, alpha);
        let rho = rho_remove.max(rho_add);
        let eps = epsilon_from_renyi_target_delta(rho, alpha, target_delta);
        if eps.is_finite() {
            best_eps = best_eps.min(eps);
        }
    }

    if !best_eps.is_finite() {
        return Err(PldError::NumericalError(
            "failed to produce finite epsilon in deterministic BnB".to_string(),
        ));
    }
    eps_delta_pld(best_eps.max(0.0), target_delta, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> DiscretizationConfig {
        DiscretizationConfig::default()
    }

    #[test]
    fn deterministic_bnb_is_reproducible() {
        let b = 8usize;
        let mut gram = vec![0.0; b * b];
        for i in 0..b {
            for j in 0..b {
                let d = i.abs_diff(j) as i32;
                gram[i * b + j] = 0.7f64.powi(d);
            }
        }
        let pld1 = bnb_deterministic_pld(&gram, b, 1.0, 12, 3, 1e-8, &cfg()).unwrap();
        let pld2 = bnb_deterministic_pld(&gram, b, 1.0, 12, 3, 1e-8, &cfg()).unwrap();
        let e1 = pld1.epsilon_at(1e-5);
        let e2 = pld2.epsilon_at(1e-5);
        assert!((e1 - e2).abs() < 1e-12);
    }

    #[test]
    fn deterministic_bnb_rejects_bad_params() {
        let c = cfg();
        assert!(bnb_deterministic_pld(&[1.0], 2, 1.0, 8, 2, 1e-8, &c).is_err());
        assert!(bnb_deterministic_pld(&[1.0, 0.0, 0.0, 1.0], 2, 0.0, 8, 2, 1e-8, &c).is_err());
        assert!(bnb_deterministic_pld(&[1.0, 0.0, 0.0, 1.0], 2, 1.0, 1, 2, 1e-8, &c).is_err());
    }

    /// Run with: `cargo test bench_deterministic_vs_mc_timing -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_deterministic_vs_mc_timing() {
        use std::time::Instant;

        use super::super::monte_carlo::bnb_mc_pld;

        let b = 32usize;
        let mut gram = vec![0.0f64; b * b];
        for i in 0..b {
            for j in 0..b {
                let d = i.abs_diff(j) as i32;
                gram[i * b + j] = 0.85f64.powi(d);
            }
        }
        let c = cfg();
        let t0 = Instant::now();
        let det = bnb_deterministic_pld(&gram, b, 1.0, 16, 4, 1e-8, &c).unwrap();
        let det_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t1 = Instant::now();
        let mc = bnb_mc_pld(&gram, b, 1.0, 50_000, 0, &c).unwrap();
        let mc_ms = t1.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "bench b={}: deterministic {:.2} ms (eps@1e-5={:.6}), mc {:.2} ms (eps@1e-5={:.6})",
            b,
            det_ms,
            det.epsilon_at(1e-5),
            mc_ms,
            mc.epsilon_at(1e-5)
        );
    }
}
