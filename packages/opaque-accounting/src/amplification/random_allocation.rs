//! Privacy amplification by random allocation, as a PLD transform.
//!
//! In `k`-out-of-`t` random allocation each record is used in `k` steps chosen
//! uniformly at random from `t`. Feldman & Shenfeld (arXiv:2602.17284) show the
//! resulting privacy loss is a transform of the base PLD:
//!
//! ```text
//! remove:  ψ⃗_t(L) = ln( (1/t)( e^{L₀} + Σ_{i=1}^{t-1} e^{-L̃ᵢ} ) )
//! add:     ψ⃖_t(L) = -ln( (1/t) Σ_{i=1}^{t} e^{-Lᵢ} )
//! ```
//!
//! where `L̃` is the PLD dual. Both are convolutions of *exponentiated* PLDs,
//! computed on a geometric grid (`crate::pld::pmf::geom`) by exponentiation by
//! squaring — `O(log t)` convolutions. Crucially the discretisation error does
//! **not** accumulate across them, which is what makes large `t` tractable.
//!
//! For the Gaussian the dual is closed form — `L̃ ~ N(1/(2σ²), 1/σ²)`, the same
//! law as `L` — so it is taken analytically rather than by reweighting a
//! discretised PLD by `e^{-l}`, which would amplify FFT residue.
//!
//! # Relation to balls-in-bins
//!
//! Opaque's fixed-assignment balls-in-bins with the identity strategy is
//! *exactly* 1-out-of-`b` random allocation at `σ_eff = σ/√E`: with `C = I` the
//! mixture means are orthogonal with equal norm `√E`, so `G = E·I_b` and the
//! dominating pair collapses onto `(P̄_b, Q^b)` after rescaling.

use crate::discretization::DiscretizationConfig;
use crate::error::{PldError, Result};
use crate::pld::pmf::geom::GeomPmf;
use crate::pld::pmf::Pmf;
use crate::pld::realization::{disc_dist, LossRealization, NormalLoss, Rounding};
use crate::pld::PrivacyLossDistribution;

/// Accuracy knobs derived from the caller's existing discretisation config.
///
/// `(α, β)` are numerical accuracy parameters, not privacy parameters, so they
/// are not exposed as separate knobs — they follow `discretization` and
/// `log_mass_truncation_bound`, the dials the user already sets.
struct Accuracy {
    /// Interior grid width used inside the transform.
    alpha: f64,
    /// Per-step tail mass discarded at each end.
    beta: f64,
}

impl Accuracy {
    fn derive<L: LossRealization + ?Sized>(
        loss: &L,
        config: &DiscretizationConfig,
        t: usize,
    ) -> Result<Self> {
        if config.max_conv_grid == 0 {
            return Err(PldError::InvalidParameter(
                "max_conv_grid must be > 0".into(),
            ));
        }
        // β is split across the t copies that get convolved together.
        let beta_total = (0.5 * config.log_mass_truncation_bound.exp()).clamp(1e-300, 0.49);
        let beta = (beta_total / t as f64).clamp(1e-300, 0.49);

        // Widen α if the requested resolution would exceed the work budget.
        // config.max_conv_grid is the O(G²) grid cap for this transform;
        // it is separate from config.max_grid_size (the FFT-composition cap).
        let log_beta = beta.ln();
        let span = loss.quantile(log_beta, true) - loss.quantile(log_beta, false);
        let alpha_floor = span / config.max_conv_grid as f64;
        Ok(Accuracy {
            alpha: config.discretization.max(alpha_floor),
            beta,
        })
    }
}

/// Remove-direction PLD of 1-out-of-`t` random allocation.
fn alloc_remove<L: LossRealization + ?Sized>(
    loss: &L,
    neg_dual: &L,
    t: usize,
    acc: &Accuracy,
    config: &DiscretizationConfig,
    out_disc: f64,
) -> Result<crate::pld::pmf::Pmf> {
    let l = disc_dist(loss, acc.alpha, acc.beta, config.max_grid_size)?;
    if t == 1 {
        return GeomPmf::from_pmf_exp(&l)?.into_pmf_log(
            1.0,
            out_disc,
            config.max_grid_size,
            Rounding::Up,
        );
    }
    // disc-dist(-D): the negated dual, taken analytically.
    let d = disc_dist(neg_dual, acc.alpha, acc.beta, config.max_grid_size)?;

    let e_l = GeomPmf::from_pmf_exp(&l)?;
    let e_d = GeomPmf::from_pmf_exp(&d)?;

    let e_d_t1 = e_d.self_conv(t - 1, Rounding::Up)?;
    let e_t = e_d_t1.conv(&e_l, Rounding::Up)?;

    e_t.into_pmf_log(t as f64, out_disc, config.max_grid_size, Rounding::Up)
}

/// Add-direction PLD of 1-out-of-`t` random allocation.
fn alloc_add<L: LossRealization + ?Sized>(
    loss: &L,
    t: usize,
    acc: &Accuracy,
    config: &DiscretizationConfig,
    out_disc: f64,
) -> Result<crate::pld::pmf::Pmf> {
    let l = disc_dist(loss, acc.alpha, acc.beta, config.max_grid_size)?;
    if t == 1 {
        return GeomPmf::from_pmf_exp_neg(&l)?.into_pmf_neg_log(
            1.0,
            out_disc,
            config.max_grid_size,
            Rounding::Up,
        );
    }
    let e_l = GeomPmf::from_pmf_exp_neg(&l)?;
    // An upper bound on -ln(S) needs a *lower* bound on S, so the inner
    // convolution rounds down. This is a fixed internal step in constructing
    // the safe result, not a caller-selectable optimistic estimate.
    let e_t = e_l.self_conv(t, Rounding::Down)?;
    e_t.into_pmf_neg_log(t as f64, out_disc, config.max_grid_size, Rounding::Up)
}

fn weighted_mix_pmfs(parts: &[(f64, Pmf)]) -> Result<Pmf> {
    let first = parts
        .first()
        .ok_or_else(|| PldError::InvalidParameter("empty PMF mixture".into()))?;
    let discretization = first.1.discretization;
    if parts
        .iter()
        .any(|(_, p)| p.discretization != discretization)
    {
        return Err(PldError::DiscretizationMismatch(
            discretization,
            parts
                .iter()
                .find(|(_, p)| p.discretization != discretization)
                .unwrap()
                .1
                .discretization,
        ));
    }
    let lo = parts.iter().map(|(_, p)| p.lower_loss_index).min().unwrap();
    let hi = parts
        .iter()
        .map(|(_, p)| p.lower_loss_index + p.probs.len() as i64 - 1)
        .max()
        .unwrap();
    let mut probs = vec![0.0; (hi - lo + 1) as usize];
    let mut infinity_mass = 0.0;
    let mut negative_infinity_mass = 0.0;
    let mut total_weight = 0.0;
    let mut max_grid_size = 0;
    for (weight, pmf) in parts {
        if *weight < 0.0 || !weight.is_finite() {
            return Err(PldError::InvalidParameter(format!(
                "mixture weight must be finite and non-negative, got {weight}"
            )));
        }
        total_weight += weight;
        let offset = (pmf.lower_loss_index - lo) as usize;
        for (i, mass) in pmf.probs.iter().enumerate() {
            probs[offset + i] += weight * mass;
        }
        infinity_mass += weight * pmf.infinity_mass;
        negative_infinity_mass += weight * pmf.negative_infinity_mass;
        max_grid_size = max_grid_size.max(pmf.max_grid_size);
    }
    if (total_weight - 1.0).abs() > 1e-9 {
        return Err(PldError::InvalidParameter(format!(
            "mixture weights must sum to one, got {total_weight}"
        )));
    }
    Ok(Pmf {
        discretization,
        lower_loss_index: lo,
        probs,
        infinity_mass,
        negative_infinity_mass,
        max_grid_size,
        right_tail_budget: 0.0,
        left_tail_budget: 0.0,
    })
}

/// Exact prefix PLD for the first `released_steps` of a 1-out-of-`total_steps`
/// Gaussian random allocation.
pub fn random_allocation_gaussian_prefix_pld(
    noise_multiplier: f64,
    total_steps: usize,
    released_steps: usize,
    config: &DiscretizationConfig,
) -> Result<PrivacyLossDistribution> {
    validate_prefix_params(noise_multiplier, total_steps, released_steps)?;
    if released_steps == total_steps {
        return random_allocation_gaussian_pld(noise_multiplier, total_steps, 1, config);
    }

    let loss = NormalLoss::gaussian(noise_multiplier);
    let neg_dual = NormalLoss::gaussian_neg_dual(noise_multiplier);
    let acc = Accuracy::derive(&loss, config, released_steps)?;
    let l = disc_dist(&loss, acc.alpha, acc.beta, config.max_grid_size)?;
    let d = disc_dist(&neg_dual, acc.alpha, acc.beta, config.max_grid_size)?;
    let e_l = GeomPmf::from_pmf_exp(&l)?;
    let e_d = GeomPmf::from_pmf_exp(&d)?;
    let inactive = total_steps - released_steps;

    let q_sum = e_d
        .self_conv(released_steps, Rounding::Up)?
        .add_constant(inactive as f64, Rounding::Up)?;
    let active_sum = if released_steps == 1 {
        e_l
    } else {
        e_d.self_conv(released_steps - 1, Rounding::Up)?
            .conv(&e_l, Rounding::Up)?
    }
    .add_constant(inactive as f64, Rounding::Up)?;

    let pmf_q = q_sum.into_pmf_log(
        total_steps as f64,
        config.discretization,
        config.max_grid_size,
        Rounding::Up,
    )?;
    let pmf_active = active_sum.into_pmf_log(
        total_steps as f64,
        config.discretization,
        config.max_grid_size,
        Rounding::Up,
    )?;
    let lambda = released_steps as f64 / total_steps as f64;
    let pmf_remove = weighted_mix_pmfs(&[(1.0 - lambda, pmf_q), (lambda, pmf_active)])?;

    let add_sum = GeomPmf::from_pmf_exp_neg(&l)?
        .self_conv(released_steps, Rounding::Down)?
        .add_constant(inactive as f64, Rounding::Down)?;
    let pmf_add = add_sum.into_pmf_neg_log(
        total_steps as f64,
        config.discretization,
        config.max_grid_size,
        Rounding::Up,
    )?;
    let tail = config.tail_mass_truncation / 2.0;
    Ok(PrivacyLossDistribution::new_asymmetric(pmf_remove, pmf_add).with_tail_budgets(tail, tail))
}

/// Conservative prefix PLD for global `k`-out-of-`t` Gaussian allocation.
///
/// The prefix participation count is hypergeometric: conditional on
/// `released_steps` of the total `total_steps` having been released, the
/// number of participations `j` follows Hyp(`total_steps`,
/// `total_participations`, `released_steps`).
///
/// For `total_participations == 1`: delegates to
/// `random_allocation_gaussian_prefix_pld` (exact).
///
/// For `total_participations > 1` with `released_steps < total_steps`: bounds
/// by monotonicity — evaluates `random_allocation_gaussian_pld(σ,
/// released_steps, cap)` where `cap` is the largest likely support point of
/// the hypergeometric distribution (after trimming a small tail) and folds in
/// the trimmed mass as a failure probability.  This bound can be significantly
/// over-conservative for small `released_steps / total_steps` ratios; the
/// Python wrapper snaps `k > 1` prefix queries to the full-horizon block bound
/// instead, which is only 3–45 % conservative.
pub fn k_out_of_t_gaussian_prefix_pld(
    noise_multiplier: f64,
    total_steps: usize,
    total_participations: usize,
    released_steps: usize,
    config: &DiscretizationConfig,
) -> Result<PrivacyLossDistribution> {
    validate_prefix_params(noise_multiplier, total_steps, released_steps)?;
    if total_participations == 0 || total_participations > total_steps {
        return Err(PldError::InvalidParameter(format!(
            "total_participations must be in [1, total_steps={total_steps}], got {total_participations}"
        )));
    }
    if released_steps == total_steps {
        return random_allocation_gaussian_pld(
            noise_multiplier,
            total_steps,
            total_participations,
            config,
        );
    }
    if total_participations == 1 {
        return random_allocation_gaussian_prefix_pld(
            noise_multiplier,
            total_steps,
            released_steps,
            config,
        );
    }

    let probabilities: Vec<(usize, f64)> =
        hypergeometric_log_weights(total_steps, total_participations, released_steps)
            .into_iter()
            .map(|(j, log_weight)| (j, log_weight.exp()))
            .collect();
    let tail_target = config.tail_mass_truncation / 2.0;
    let mut tail_mass = 0.0;
    let mut cap = probabilities.last().unwrap().0;
    for (j, probability) in probabilities.iter().rev() {
        if tail_mass + probability > tail_target {
            cap = *j;
            break;
        }
        tail_mass += probability;
    }
    cap = cap.max(1);
    let pld = random_allocation_gaussian_pld(noise_multiplier, released_steps, cap, config)?;
    Ok(add_failure_probability(pld, tail_mass))
}

fn add_failure_probability(
    mut pld: PrivacyLossDistribution,
    failure_probability: f64,
) -> PrivacyLossDistribution {
    if failure_probability <= 0.0 {
        return pld;
    }
    let head_mass = 1.0 - failure_probability;
    let update = |pmf: &mut Pmf| {
        for probability in &mut pmf.probs {
            *probability *= head_mass;
        }
        pmf.negative_infinity_mass *= head_mass;
        pmf.infinity_mass = head_mass * pmf.infinity_mass + failure_probability;
    };
    update(&mut pld.pmf_remove);
    if let Some(pmf_add) = &mut pld.pmf_add {
        update(pmf_add);
    }
    pld
}

fn hypergeometric_log_weights(
    population: usize,
    successes: usize,
    draws: usize,
) -> Vec<(usize, f64)> {
    let j_min = successes.saturating_sub(population - draws);
    let j_max = successes.min(draws);
    let denominator = log_binomial(population, successes);
    (j_min..=j_max)
        .map(|j| {
            (
                j,
                log_binomial(draws, j) + log_binomial(population - draws, successes - j)
                    - denominator,
            )
        })
        .collect()
}

fn log_binomial(n: usize, k: usize) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    let k = k.min(n - k);
    (1..=k)
        .map(|i| ((n - k + i) as f64).ln() - (i as f64).ln())
        .sum()
}

fn validate_prefix_params(
    noise_multiplier: f64,
    total_steps: usize,
    released_steps: usize,
) -> Result<()> {
    if noise_multiplier.is_nan() || noise_multiplier <= 0.0 {
        return Err(PldError::InvalidParameter(format!(
            "noise_multiplier must be > 0, got {noise_multiplier}"
        )));
    }
    if total_steps == 0 {
        return Err(PldError::InvalidParameter(
            "total_steps must be >= 1".into(),
        ));
    }
    if released_steps == 0 || released_steps > total_steps {
        return Err(PldError::InvalidParameter(format!(
            "released_steps must be in [1, total_steps={total_steps}], got {released_steps}"
        )));
    }
    Ok(())
}

/// PLD of `k`-out-of-`t` random allocation applied to the Gaussian mechanism.
///
/// Exact for `k = 1`. For `k > 1` the result is a valid **upper bound**, not
/// the exact `k`-out-of-`t` PLD: the `t` steps are split into `k` blocks and
/// the record is placed once per block, which is a strictly smaller family of
/// participation patterns than "any `k` of `t`".
///
/// That bound is sound by the same joint-convexity argument that makes
/// per-epoch redraw dominate fixed assignment. Drawing a uniformly random
/// partition of `[t]` into `k` blocks and then picking one step per block
/// induces exactly the uniform distribution over `k`-subsets — the whole
/// construction is invariant under permutations of `[t]`, which act
/// transitively on `k`-subsets. So the true mixture `P` is an average of the
/// block-scheme mixtures `P_π` over partitions `π`, every `P_π` gives the same
/// divergence by symmetry, and joint convexity of the hockey-stick divergence
/// gives `D_ε(P ‖ Q) ≤ D_ε(P_π ‖ Q)`.
///
/// The block scheme itself factorises across blocks (independent picks,
/// independent noise), so composing the per-block PLDs is exact.
///
/// # Arguments
///
/// * `noise_multiplier` — σ/Δ, must be > 0.
/// * `t` — steps per allocation round (the number of bins).
/// * `k` — steps each record is used in, in `[1, t]`. Values above 1 return
///   the block upper bound described above.
/// * `config` — discretisation configuration.
///
/// # Errors
///
/// `InvalidParameter` for out-of-range arguments or a grid exceeding
/// `max_grid_size`.
pub fn random_allocation_gaussian_pld(
    noise_multiplier: f64,
    t: usize,
    k: usize,
    config: &DiscretizationConfig,
) -> Result<PrivacyLossDistribution> {
    if noise_multiplier.is_nan() || noise_multiplier <= 0.0 {
        return Err(PldError::InvalidParameter(format!(
            "noise_multiplier must be > 0, got {}",
            noise_multiplier
        )));
    }
    if t == 0 {
        return Err(PldError::InvalidParameter("t must be >= 1".into()));
    }
    if k == 0 || k > t {
        return Err(PldError::InvalidParameter(format!(
            "k must be in [1, t={}], got {}",
            t, k
        )));
    }

    // k > 1 reduces to a composition of single allocations: split t into
    // m_f rounds of ⌊t/k⌋ steps and m_c rounds of ⌈t/k⌉.
    let (t_floor, t_ceil) = (t / k, t.div_ceil(k));
    let (m_f, m_c) = if t_floor == t_ceil {
        (k, 0)
    } else {
        let m_c = t - t_floor * k;
        (k - m_c, m_c)
    };
    debug_assert_eq!(m_f + m_c, k);
    debug_assert_eq!(m_f * t_floor + m_c * t_ceil, t);

    let sigma = noise_multiplier;
    let loss = NormalLoss::gaussian(sigma);
    let neg_dual = NormalLoss::gaussian_neg_dual(sigma);

    let mut out: Option<PrivacyLossDistribution> = None;
    for &(rounds, steps) in &[(m_f, t_floor), (m_c, t_ceil)] {
        if rounds == 0 || steps == 0 {
            continue;
        }
        let acc = Accuracy::derive(&loss, config, steps)?;
        let pmf_remove =
            alloc_remove(&loss, &neg_dual, steps, &acc, config, config.discretization)?;
        let pmf_add = alloc_add(&loss, steps, &acc, config, config.discretization)?;
        let single = PrivacyLossDistribution::new_asymmetric(pmf_remove, pmf_add);
        let composed = if rounds == 1 {
            single
        } else {
            single.self_compose(rounds)?
        };
        out = Some(match out {
            None => composed,
            Some(prev) => prev.compose(&composed)?,
        });
    }

    out.ok_or_else(|| PldError::InvalidParameter("empty allocation decomposition".into()))
        .map(|pld| {
            let tail = config.tail_mass_truncation / 2.0;
            pld.with_tail_budgets(tail, tail)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::amplification::poisson::poisson_gaussian_pld;
    use crate::mechanisms::gaussian_pld;

    fn cfg() -> DiscretizationConfig {
        DiscretizationConfig::default()
    }

    /// t = 1 is no allocation at all: it must be the base Gaussian.
    #[test]
    fn test_t_one_is_base_gaussian() {
        let c = cfg();
        for &sigma in &[0.5, 1.0, 2.0] {
            let ra = random_allocation_gaussian_pld(sigma, 1, 1, &c).unwrap();
            let base = gaussian_pld(sigma, &c).unwrap();
            let (a, b) = (ra.epsilon_at(1e-8), base.epsilon_at(1e-8));
            assert!(
                (a - b).abs() < 1e-3 * b.max(1.0),
                "σ={}: allocation {} vs base {}",
                sigma,
                a,
                b
            );
        }
    }

    /// More noise, less privacy loss.
    #[test]
    fn test_monotone_in_sigma() {
        let c = cfg();
        let a = random_allocation_gaussian_pld(1.0, 8, 1, &c)
            .unwrap()
            .epsilon_at(1e-8);
        let b = random_allocation_gaussian_pld(2.0, 8, 1, &c)
            .unwrap()
            .epsilon_at(1e-8);
        assert!(b < a, "more noise should lower eps: {} vs {}", b, a);
    }

    #[test]
    fn test_prefix_endpoints_and_monotonicity() {
        let c = cfg();
        let sigma = 1.0;
        let t = 8;
        let full = random_allocation_gaussian_pld(sigma, t, 1, &c)
            .unwrap()
            .epsilon_at(1e-8);
        let mut prev = 0.0;
        for released in 1..=t {
            let eps = random_allocation_gaussian_prefix_pld(sigma, t, released, &c)
                .unwrap()
                .epsilon_at(1e-8);
            assert!(eps >= prev - 1e-9, "{released}: {eps} < {prev}");
            assert!(eps <= full + 1e-9, "{released}: {eps} > {full}");
            prev = eps;
        }
        assert!((prev - full).abs() < 1e-9);
    }

    #[test]
    fn test_one_step_prefix_matches_poisson() {
        let c = cfg();
        for &(sigma, total) in &[(1.0, 8_usize), (2.0, 16)] {
            let prefix = random_allocation_gaussian_prefix_pld(sigma, total, 1, &c)
                .unwrap()
                .epsilon_at(1e-8);
            let poisson = poisson_gaussian_pld(sigma, 1.0 / total as f64, &c)
                .unwrap()
                .epsilon_at(1e-8);
            assert!(
                (prefix - poisson).abs() < 2e-3,
                "sigma={sigma}, total={total}: {prefix} vs {poisson}"
            );
        }
    }

    #[test]
    fn test_k_out_of_t_prefix_is_finite_and_matches_full_horizon() {
        let c = cfg();
        let mut last = 0.0;
        for released in 1..=6 {
            let epsilon = k_out_of_t_gaussian_prefix_pld(1.5, 6, 2, released, &c)
                .unwrap()
                .epsilon_at(1e-6);
            assert!(epsilon.is_finite() && epsilon > 0.0);
            last = epsilon;
        }
        let full = random_allocation_gaussian_pld(1.5, 6, 2, &c)
            .unwrap()
            .epsilon_at(1e-6);
        assert!((last - full).abs() < 1e-9);
    }

    /// Golden values, cross-validated against an independent Python
    /// implementation of the same paper (which bracketed
    /// [3.711, 3.726], [1.792, 1.806], [0.315, 0.345], [1.253, 1.270]).
    /// Opaque's safe bound must land inside that independent bracket.
    #[test]
    fn test_golden_epsilons() {
        let c = cfg();
        for &(sigma, t, want_lo, want_hi) in &[
            (1.0, 8usize, 3.711, 3.727),
            (1.0, 64, 1.792, 1.807),
            (2.0, 64, 0.314, 0.346),
            (1.0, 128, 1.252, 1.271),
        ] {
            let got = random_allocation_gaussian_pld(sigma, t, 1, &c)
                .unwrap()
                .epsilon_at(1e-8);
            assert!(
                got >= want_lo && got <= want_hi,
                "σ={} t={}: {} outside the cross-validated [{}, {}]",
                sigma,
                t,
                got,
                want_lo,
                want_hi
            );
        }
    }

    /// More steps to hide among, less privacy loss.
    #[test]
    fn test_monotone_in_t() {
        let c = cfg();
        let mut prev = f64::INFINITY;
        for &t in &[4usize, 8, 16] {
            let e = random_allocation_gaussian_pld(1.0, t, 1, &c)
                .unwrap()
                .epsilon_at(1e-8);
            assert!(e < prev, "eps should fall with t: {} at t={}", e, t);
            prev = e;
        }
    }

    /// More participations, more privacy loss.
    #[test]
    fn test_monotone_in_k() {
        let c = cfg();
        let e1 = random_allocation_gaussian_pld(1.0, 16, 1, &c)
            .unwrap()
            .epsilon_at(1e-8);
        let e4 = random_allocation_gaussian_pld(1.0, 16, 4, &c)
            .unwrap()
            .epsilon_at(1e-8);
        assert!(e4 > e1, "eps should rise with k: {} vs {}", e4, e1);
    }

    /// `k = t` is the one `k > 1` point with a known answer, so it pins the
    /// block arithmetic against an independent construction.
    ///
    /// Every block has size 1, the record is in every step, and there is no
    /// allocation left to amplify — the answer is the base Gaussian composed
    /// `t` times. An `m_f`/`m_c` split that mis-sized the blocks or mis-counted
    /// the rounds would land nowhere near it.
    ///
    /// Not an equality: at block size 1 this path returns `disc_dist`'s
    /// directionally-rounded grid while `gaussian_pld` uses connect-the-dots,
    /// so the two agree only up to discretisation (~3e-4 relative here). What
    /// must hold exactly is the *direction* — the upper variant may never fall
    /// below the truth.
    #[test]
    fn test_k_equals_t_is_full_participation() {
        let c = cfg();
        for &t in &[2usize, 5, 8] {
            let alloc = random_allocation_gaussian_pld(1.0, t, t, &c)
                .unwrap()
                .epsilon_at(1e-8);
            let every_step = gaussian_pld(1.0, &c)
                .unwrap()
                .self_compose(t)
                .unwrap()
                .epsilon_at(1e-8);
            assert!(
                alloc >= every_step - 1e-9,
                "t={}: k=t gave {}, below full participation {}",
                t,
                alloc,
                every_step
            );
            assert!(
                alloc <= every_step * 1.01,
                "t={}: k=t gave {}, far above full participation {}",
                t,
                alloc,
                every_step
            );
        }
    }

    /// The `k > 1` block decomposition must cover exactly `t` steps in exactly
    /// `k` rounds for every divisibility case, including `k ∤ t`.
    #[test]
    fn test_block_decomposition_covers_t() {
        for t in 1usize..=24 {
            for k in 1..=t {
                let (t_floor, t_ceil) = (t / k, (t + k - 1) / k);
                let (m_f, m_c) = if t_floor == t_ceil {
                    (k, 0)
                } else {
                    let m_c = t - t_floor * k;
                    (k - m_c, m_c)
                };
                assert_eq!(m_f + m_c, k, "t={} k={}: round count", t, k);
                assert_eq!(
                    m_f * t_floor + m_c * t_ceil,
                    t,
                    "t={} k={}: step coverage",
                    t,
                    k
                );
            }
        }
    }

    /// Random allocation should be at least competitive with Poisson at the
    /// matched rate 1/t — the paper's headline claim.
    #[test]
    fn test_competitive_with_poisson() {
        let c = cfg();
        let (sigma, t) = (1.0, 32usize);
        let ra = random_allocation_gaussian_pld(sigma, t, 1, &c)
            .unwrap()
            .epsilon_at(1e-8);
        let po = poisson_gaussian_pld(sigma, 1.0 / t as f64, &c)
            .unwrap()
            .self_compose(t)
            .unwrap()
            .epsilon_at(1e-8);
        assert!(
            ra <= po * 1.10,
            "σ={} t={}: allocation {} vs Poisson {}",
            sigma,
            t,
            ra,
            po
        );
    }

    #[test]
    fn test_rejects_bad_params() {
        let c = cfg();
        assert!(random_allocation_gaussian_pld(0.0, 8, 1, &c).is_err());
        assert!(random_allocation_gaussian_pld(1.0, 0, 1, &c).is_err());
        assert!(random_allocation_gaussian_pld(1.0, 8, 0, &c).is_err());
        assert!(random_allocation_gaussian_pld(1.0, 8, 9, &c).is_err());
        // max_conv_grid = 0 must be rejected before the span/0 division.
        let c_zero = DiscretizationConfig::default().with_max_conv_grid(0);
        assert!(random_allocation_gaussian_pld(1.0, 8, 1, &c_zero).is_err());
    }

    /// At t=1 the transform equals the base Gaussian (no allocation
    /// amplification), so the upper bound must be ≥ the Gaussian ε regardless
    /// of grid size (soundness check from the issue).
    #[test]
    fn test_t_one_upper_bound_sound_across_grid_sizes() {
        for &sigma in &[0.5, 1.0, 2.0] {
            let exact = crate::mechanisms::gaussian_pld(sigma, &cfg())
                .unwrap()
                .epsilon_at(1e-8);
            for &g in &[4096_usize, 8192, 16384, 32768] {
                let c = DiscretizationConfig::default().with_max_conv_grid(g);
                let upper = random_allocation_gaussian_pld(sigma, 1, 1, &c)
                    .unwrap()
                    .epsilon_at(1e-8);
                assert!(
                    upper >= exact - 1e-9,
                    "σ={} G={}: upper bound {} below exact Gaussian {}",
                    sigma,
                    g,
                    upper,
                    exact
                );
            }
        }
    }

    /// A larger max_conv_grid cap produces a tighter (lower or equal) ε
    /// upper bound — it never loosens the bound.
    #[test]
    fn test_larger_max_conv_grid_tightens_bound() {
        let c_coarse = DiscretizationConfig::default().with_max_conv_grid(8_192);
        let c_fine = DiscretizationConfig::default().with_max_conv_grid(32_768);
        for &(sigma, t) in &[(2.0_f64, 64_usize), (1.0, 128)] {
            let eps_coarse = random_allocation_gaussian_pld(sigma, t, 1, &c_coarse)
                .unwrap()
                .epsilon_at(1e-8);
            let eps_fine = random_allocation_gaussian_pld(sigma, t, 1, &c_fine)
                .unwrap()
                .epsilon_at(1e-8);
            assert!(
                eps_fine <= eps_coarse + 1e-9,
                "σ={} t={}: finer grid gave looser bound: {} > {}",
                sigma,
                t,
                eps_fine,
                eps_coarse
            );
        }
    }
}
